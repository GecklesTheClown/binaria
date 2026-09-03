import warnings

import pytest
import torch

from binaria._core import Core
from binaria._optim import _ConvergenceMonitor, fit
from binaria.estimator import SiGMoiD
from binaria.selection import SiGMoiDSelector


def _binary(n_samples: int = 12, n_features: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 2, (n_samples, n_features), dtype=torch.float64)


def test_sgd_is_exactly_one_plain_gradient_ascent_step() -> None:
    data = _binary()
    torch.manual_seed(1)
    core = Core(n_samples=12, n_features=8, n_components=3)
    old_beta = core.beta.detach().clone()
    old_energy = core.energy.detach().clone()
    grad_beta, grad_energy = core.analytic_gradients(data)
    lr = 1e-4

    fit(core, data, optimizer="sgd", lr=lr, max_iter=1, tol=0.0)

    assert torch.equal(core.beta, old_beta + lr * grad_beta)
    assert torch.equal(core.energy, old_energy + lr * grad_energy)


def test_patience_requires_consecutive_small_objective_changes() -> None:
    core = Core(4, 3, 2)
    data = torch.zeros((4, 3), dtype=torch.float64)

    result = fit(core, data, alpha=0.1, lr=0.0, max_iter=10, tol=1e-6, patience=3)

    # Iteration 1 establishes the baseline, then three unchanged objectives
    # satisfy the tolerance on iterations 2, 3, and 4.
    assert result.converged is True
    assert result.n_iter == 4


def test_patience_one_reproduces_the_previous_one_hit_rule() -> None:
    core = Core(4, 3, 2)
    data = torch.zeros((4, 3), dtype=torch.float64)

    result = fit(core, data, alpha=0.1, lr=0.0, max_iter=10, tol=1e-6, patience=1)

    assert result.converged is True
    assert result.n_iter == 2


def test_nonqualifying_change_resets_the_patience_streak() -> None:
    monitor = _ConvergenceMonitor(tol=0.01, patience=2)

    assert monitor.update(100.0) is False
    assert monitor.update(99.5) is False
    assert monitor.update(90.0) is False
    assert monitor.update(89.5) is False
    assert monitor.update(89.0) is True


@pytest.mark.parametrize("gradient", ["analytic", "autograd"])
@pytest.mark.parametrize("alpha", [0.0, 0.2])
def test_energy_gradient_rule_uses_the_objective_gradient(gradient: str, alpha: float) -> None:
    data = _binary(4, 3)
    torch.manual_seed(1)
    core = Core(4, 3, 2)
    _, grad_energy = core.analytic_gradients(data)
    objective_grad_energy = grad_energy - 2.0 * alpha * core.energy.detach()
    statistic = objective_grad_energy.norm().item() / core.energy.norm().item()

    result = fit(
        core,
        data,
        alpha=alpha,
        gradient=gradient,  # type: ignore[arg-type]
        lr=0.0,
        max_iter=1,
        tol=statistic * 1.01,
        patience=1,
        stopping_rule="energy_gradient",
    )

    assert result.converged is True
    assert result.n_iter == 1


def test_energy_gradient_rule_respects_patience() -> None:
    result = fit(
        Core(4, 3, 2),
        _binary(4, 3),
        alpha=0.0,
        lr=0.0,
        max_iter=5,
        tol=1e9,
        patience=3,
        stopping_rule="energy_gradient",
    )

    assert result.converged is True
    assert result.n_iter == 3


@pytest.mark.parametrize("alpha", [0.0, 0.2])
def test_sgd_analytic_and_autograd_paths_are_bit_identical(alpha: float) -> None:
    data = _binary()
    fitted = []
    for gradient in ("analytic", "autograd"):
        torch.manual_seed(1)
        core = Core(n_samples=12, n_features=8, n_components=3)
        fit(
            core,
            data,
            optimizer="sgd",
            gradient=gradient,  # type: ignore[arg-type]
            alpha=alpha,
            lr=1e-4,
            max_iter=20,
            tol=0.0,
        )
        fitted.append((core.beta.detach().clone(), core.energy.detach().clone()))

    assert torch.equal(fitted[0][0], fitted[1][0])
    assert torch.equal(fitted[0][1], fitted[1][1])


def test_default_optimizer_remains_bit_identical_to_explicit_adam() -> None:
    assert SiGMoiD(n_components=2).optimizer == "adam"
    assert SiGMoiDSelector([2]).optimizer == "adam"
    assert SiGMoiDSelector([2]).patience == 100
    assert SiGMoiD(n_components=2).stopping_rule == "objective"
    assert SiGMoiDSelector([2]).stopping_rule == "objective"
    data = _binary()
    fitted = []
    for kwargs in ({}, {"optimizer": "adam"}):
        torch.manual_seed(1)
        core = Core(n_samples=12, n_features=8, n_components=3)
        fit(core, data, lr=0.01, max_iter=20, tol=0.0, **kwargs)  # type: ignore[arg-type]
        fitted.append((core.beta.detach().clone(), core.energy.detach().clone()))

    assert torch.equal(fitted[0][0], fitted[1][0])
    assert torch.equal(fitted[0][1], fitted[1][1])


def test_unknown_optimizer_is_rejected_during_fit_not_construction() -> None:
    model = SiGMoiD(n_components=2, optimizer="rmsprop")  # type: ignore[arg-type]
    assert model.optimizer == "rmsprop"
    with pytest.raises(ValueError, match="optimizer"):
        model.fit(_binary())


def test_sgd_optimizer_applies_to_transform_and_score(monkeypatch: pytest.MonkeyPatch) -> None:
    import binaria.estimator as estimator

    names: list[str] = []
    original = estimator.make_optimizer

    def recording(parameters, *, name, lr):  # type: ignore[no-untyped-def]
        names.append(name)
        return original(parameters, name=name, lr=lr)

    monkeypatch.setattr(estimator, "make_optimizer", recording)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = SiGMoiD(
            n_components=2,
            optimizer="sgd",
            learning_rate=1e-4,
            max_iter=2,
            random_state=0,
        ).fit(_binary())
        model.transform(_binary(4, 8))
        model.score(_binary(4, 8))

    assert names == ["sgd", "sgd"]


def test_selector_runs_its_fits_with_sgd() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector = SiGMoiDSelector(
            [2],
            alpha_range=(0.3,),
            n_repeats=2,
            max_iter=2,
            optimizer="sgd",
            learning_rate=1e-4,
            random_state=0,
        ).fit(_binary())

    assert selector.optimizer == "sgd"
    assert selector.cv_results_["score"].shape == (2,)


def test_selector_propagates_the_energy_gradient_stopping_rule() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector = SiGMoiDSelector(
            [2],
            alpha_range=(0.0,),
            n_repeats=2,
            max_iter=2,
            tol=1e9,
            patience=1,
            stopping_rule="energy_gradient",
            random_state=0,
        ).fit(_binary())

    assert bool(selector.cv_results_["converged"].all())
