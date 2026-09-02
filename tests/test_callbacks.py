from pathlib import Path

import numpy as np
import pytest
import torch

from binaria import History, IterationState, SiGMoiD
from binaria._core import Core
from binaria._optim import GradientPath, fit
from binaria.callbacks import Checkpoint, load_checkpoint


def _binary() -> np.ndarray:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 2, (12, 8), generator=generator, dtype=torch.float64).numpy()


def test_final_history_likelihood_describes_the_fitted_model() -> None:
    model = SiGMoiD(
        n_components=2,
        optimizer="sgd",
        learning_rate=1e-3,
        max_iter=3,
        tol=0.0,
        random_state=0,
    ).fit(_binary())

    assert model.history_.iteration[-1] == model.n_iter_
    assert model.history_.log_likelihood[-1] == model.log_likelihood_


def test_callback_state_and_core_describe_the_same_completed_iteration() -> None:
    data = _binary()
    states: list[IterationState] = []

    class Recorder:
        def on_iteration(self, *, state, core, optimizer) -> None:  # type: ignore[no-untyped-def]
            del optimizer
            tensor = torch.as_tensor(data, dtype=core.beta.dtype, device=core.beta.device)
            assert state.log_likelihood == core.log_likelihood(tensor).item()
            assert state.objective == state.log_likelihood - 0.2 * state.l2_penalty
            states.append(state)

    recorder = Recorder()
    SiGMoiD(
        n_components=2,
        optimizer="sgd",
        learning_rate=1e-3,
        max_iter=3,
        tol=0.0,
        alpha=0.2,
        random_state=0,
        callbacks=[recorder],
    ).fit(data)

    assert [state.iteration for state in states] == [1, 2, 3]
    assert states[-1].is_final is True
    assert states[-1].elapsed_time >= 0.0


def test_only_the_iteration_that_exhausts_patience_is_marked_final() -> None:
    data = torch.as_tensor(_binary())
    states: list[IterationState] = []

    class Recorder:
        def on_iteration(self, *, state, core, optimizer) -> None:  # type: ignore[no-untyped-def]
            del core, optimizer
            states.append(state)

    fit(
        Core(12, 8, 2),
        data,
        alpha=0.1,
        lr=0.0,
        max_iter=10,
        tol=1e-6,
        patience=3,
        callbacks=[Recorder()],
    )

    assert [state.iteration for state in states] == [1, 2, 3, 4]
    assert [state.is_final for state in states] == [False, False, False, True]


def test_diagnostic_history_records_the_penalized_objective() -> None:
    alpha = 0.2
    model = SiGMoiD(
        n_components=2,
        optimizer="sgd",
        learning_rate=1e-3,
        max_iter=3,
        tol=0.0,
        alpha=alpha,
        random_state=0,
        callbacks=[History(diagnostics=True)],
    ).fit(_binary())

    penalty = np.square(model.embedding_).sum() + np.square(model.components_).sum()
    assert model.history_.l2_penalty[-1] == pytest.approx(penalty)
    assert model.history_.objective[-1] == pytest.approx(model.log_likelihood_ - alpha * penalty)


def test_diagnostic_history_records_logit_factor_and_saturation_metrics() -> None:
    thresholds = (1e-2, 1e-6)
    model = SiGMoiD(
        n_components=2,
        optimizer="sgd",
        learning_rate=1e-3,
        max_iter=3,
        tol=0.0,
        random_state=0,
        callbacks=[History(diagnostics=True, saturation_thresholds=thresholds)],
    ).fit(_binary())

    history = model.history_
    logits = -(model.embedding_ @ model.components_)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    beta_norm = np.linalg.norm(model.embedding_)
    energy_norm = np.linalg.norm(model.components_)

    assert history.max_abs_logit[-1] == pytest.approx(np.abs(logits).max())
    assert history.beta_norm[-1] == pytest.approx(beta_norm)
    assert history.energy_norm[-1] == pytest.approx(energy_norm)
    assert history.factor_norm[-1] == pytest.approx(np.hypot(beta_norm, energy_norm))
    for threshold in thresholds:
        expected = np.mean((probabilities < threshold) | (probabilities > 1.0 - threshold))
        assert history.saturation_fraction[threshold][-1] == pytest.approx(expected)


def test_history_stride_includes_final_iteration_and_exports_equal_length_columns() -> None:
    model = SiGMoiD(
        n_components=2,
        optimizer="sgd",
        learning_rate=1e-3,
        max_iter=5,
        tol=0.0,
        random_state=0,
        callbacks=[
            History(
                every=2,
                diagnostics=True,
                record_gradient_norm=True,
                saturation_thresholds=(1e-2,),
            )
        ],
    ).fit(_binary())

    history = model.history_
    assert history.iteration == [2, 4, 5]
    columns = history.as_dict()
    assert set(columns) == {
        "iteration",
        "log_likelihood",
        "elapsed_time",
        "gradient_norm",
        "objective",
        "l2_penalty",
        "max_abs_logit",
        "beta_norm",
        "energy_norm",
        "factor_norm",
        "saturation_fraction_1e-02",
    }
    assert {len(column) for column in columns.values()} == {3}


@pytest.mark.parametrize("every", [0, -1])
def test_history_rejects_nonpositive_stride(every: int) -> None:
    with pytest.raises(ValueError, match="every"):
        History(every=every)


@pytest.mark.parametrize(
    "thresholds",
    [(0.0,), (0.5,), (-1e-3,), (float("nan"),), (1e-2, 1e-2)],
)
def test_history_rejects_invalid_saturation_thresholds(
    thresholds: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="saturation_thresholds"):
        History(saturation_thresholds=thresholds)


@pytest.mark.parametrize(
    "callback", [lambda: History(every=0), lambda: Checkpoint(Path("x"), every=0)]
)
def test_callbacks_reject_invalid_strides(callback) -> None:
    with pytest.raises(ValueError, match="positive int"):
        callback()


@pytest.mark.parametrize("gradient", ["analytic", "autograd"])
def test_history_records_the_post_step_model(gradient: GradientPath) -> None:
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    core = Core(n_samples=2, n_features=2, n_components=1)
    history = History()

    fit(core, data, max_iter=1, lr=0.1, gradient=gradient, callbacks=[history])

    assert history.log_likelihood == [core.log_likelihood(data).item()]


def test_history_stride_always_includes_the_final_iteration() -> None:
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    history = History(every=2, record_gradient_norm=True)

    fit(Core(2, 2, 1), data, max_iter=3, tol=0.0, callbacks=[history])

    assert history.iteration == [2, 3]
    assert len(history.gradient_norm) == 2


def test_checkpoint_round_trips_model_and_optimizer_state(tmp_path: Path) -> None:
    data = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    core = Core(2, 2, 1)
    path = tmp_path / "nested" / "fit.pt"

    fit(core, data, max_iter=2, tol=0.0, callbacks=[Checkpoint(path, every=1)])

    restored = Core(2, 2, 1)
    optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3, maximize=True)
    iteration = load_checkpoint(path, restored, optimizer)

    assert iteration == 2
    assert torch.equal(restored.beta, core.beta)
    assert torch.equal(restored.energy, core.energy)
    assert optimizer.state
    assert not path.with_name(path.name + ".tmp").exists()
