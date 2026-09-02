"""
The third oracle: does this behave like the *original* implementation?

``_reference.py`` reproduces the procedure recovered in the release plan's
A.7 -- float64, NumPy Uniform[0, 0.1) init, simultaneous (Jacobi) plain
gradient ascent, and the ``||grad E||_F / ||E||_F`` stopping rule. It is
not a default and not recommended to users; it exists so the suite can ask
whether the modern path agrees with the published one.

That makes it a strange thing to test, because several of its properties
are *deliberately worse* than the production path. The stopping rule is
known-flawed; ``||E||`` grows during training, so the ratio can fall
because the denominator grew rather than because the gradient shrank.
These tests pin the reproduction, not the quality -- a change that
"improves" any of it has broken the oracle.
"""

import numpy as np
import pytest
import torch

from binaria._core import Core
from binaria._reference import fit_reference, reference_init


def _binary(n_samples: int = 30, n_features: int = 18, true_k: int = 3, seed: int = 0):
    torch.manual_seed(seed)
    beta = torch.randn(n_samples, true_k, dtype=torch.float64)
    energy = torch.randn(true_k, n_features, dtype=torch.float64)
    return torch.bernoulli(torch.sigmoid(-(beta @ energy)))


# --- initialisation -------------------------------------------------------


def test_init_shapes_and_dtype() -> None:
    beta, energy = reference_init(30, 18, 4, seed=0)
    assert beta.shape == (30, 4)
    assert energy.shape == (4, 18)
    # A.8 requires float64 exactly; NumPy's uniform gives it by default, so
    # this catches an accidental cast rather than a missing one.
    assert beta.dtype == torch.float64
    assert energy.dtype == torch.float64


def test_init_is_uniform_on_the_original_interval() -> None:
    # Uniform[0, 0.1), not standard normal. Both bounds matter: the
    # original starts small and strictly positive.
    beta, energy = reference_init(200, 200, 5, seed=0)
    for tensor in (beta, energy):
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) < 0.1
        assert float(tensor.mean()) == pytest.approx(0.05, abs=0.005)


def test_init_comes_from_numpy_not_torch() -> None:
    # A.8: the arrays must come from np.random.RandomState. Pinning the
    # exact stream is what makes "same seed, same numbers as the original"
    # checkable -- matching torch's RNG to NumPy's MT19937 is not possible,
    # so the generator identity is the specification.
    expected_rng = np.random.RandomState(7)
    expected_beta = expected_rng.uniform(0.0, 0.1, size=(6, 2))
    expected_energy = expected_rng.uniform(0.0, 0.1, size=(2, 5))

    beta, energy = reference_init(6, 5, 2, seed=7)
    assert np.array_equal(beta.numpy(), expected_beta)
    assert np.array_equal(energy.numpy(), expected_energy)


def test_init_draw_order_is_beta_then_energy() -> None:
    # Swapping the draw order desynchronises the stream against the
    # original, which would make every downstream comparison wrong while
    # looking statistically fine.
    beta, _ = reference_init(4, 4, 2, seed=11)
    first_draws = np.random.RandomState(11).uniform(0.0, 0.1, size=(4, 2))
    assert np.array_equal(beta.numpy(), first_draws)


def test_init_is_reproducible_and_seed_sensitive() -> None:
    a = reference_init(8, 6, 2, seed=3)
    b = reference_init(8, 6, 2, seed=3)
    c = reference_init(8, 6, 2, seed=4)
    assert torch.equal(a[0], b[0])
    assert not torch.equal(a[0], c[0])


def test_init_does_not_disturb_global_numpy_state() -> None:
    # A local RandomState, not the global one: seeding globally would
    # perturb whatever else the caller is running.
    np.random.seed(1234)
    before = np.random.rand()
    np.random.seed(1234)
    reference_init(5, 5, 2, seed=None)
    assert np.random.rand() == before


# --- the fit --------------------------------------------------------------


def test_fit_rejects_non_float64() -> None:
    # float32 here would measure dtype drift rather than compare
    # procedures, so it is refused rather than silently promoted.
    core = Core(n_samples=10, n_features=8, n_components=2, dtype=torch.float32)
    with pytest.raises(ValueError, match="float64"):
        fit_reference(core, _binary(10, 8).float(), max_iter=10)


def test_fit_increases_the_log_likelihood() -> None:
    # Gradient *ascent* on the log-likelihood. If a sign were flipped this
    # would march the wrong way, and nothing else in this file would catch
    # it.
    data = _binary()
    beta, energy = reference_init(30, 18, 3, seed=0)
    core = Core(n_samples=30, n_features=18, n_components=3, beta=beta, energy=energy)

    with torch.no_grad():
        before = core.log_likelihood(data).item()
    fit_reference(core, data, lr=1e-3, max_iter=500, check_every=10_000)
    with torch.no_grad():
        after = core.log_likelihood(data).item()

    assert after > before


def test_fit_updates_are_simultaneous_not_sequential() -> None:
    # The Jacobi property, and the one most likely to be broken by a
    # well-meaning refactor: both gradients must come from the SAME
    # pre-update snapshot. A sequential (Gauss-Seidel) update would use the
    # already-moved beta when computing energy's gradient, and diverges
    # from the original from iteration one.
    data = _binary(12, 10, 2)
    beta, energy = reference_init(12, 10, 2, seed=5)
    core = Core(n_samples=12, n_features=10, n_components=2, beta=beta, energy=energy)

    grad_beta, grad_energy = core.analytic_gradients(data)
    expected_beta = core.beta.detach() + 1e-3 * grad_beta
    expected_energy = core.energy.detach() + 1e-3 * grad_energy

    fit_reference(core, data, lr=1e-3, max_iter=1, check_every=10_000)

    assert torch.allclose(core.beta.detach(), expected_beta, atol=0, rtol=0)
    assert torch.allclose(core.energy.detach(), expected_energy, atol=0, rtol=0)


def test_fit_reports_the_iteration_count_and_stops_at_max_iter() -> None:
    data = _binary(12, 10, 2)
    beta, energy = reference_init(12, 10, 2, seed=1)
    core = Core(n_samples=12, n_features=10, n_components=2, beta=beta, energy=energy)
    # check_every larger than max_iter: the stopping rule never fires, so
    # this isolates the budget behaviour.
    result = fit_reference(core, data, lr=1e-4, max_iter=25, check_every=10_000)
    assert result.n_iter == 25
    assert result.converged is False


def test_fit_can_converge_by_the_original_stopping_rule() -> None:
    # ||grad E|| / ||E|| < tol, checked periodically on E only. A loose tol
    # so this terminates quickly; the point is that the path exists and
    # sets converged=True, not that the rule is good. It is not -- see the
    # module docstring.
    data = _binary(20, 14, 2)
    beta, energy = reference_init(20, 14, 2, seed=2)
    core = Core(n_samples=20, n_features=14, n_components=2, beta=beta, energy=energy)
    result = fit_reference(core, data, lr=1e-3, max_iter=5000, tol=1e9, check_every=10)

    assert result.converged is True
    assert result.n_iter == 10  # fires at the first check


def test_fit_keeps_parameters_registered() -> None:
    # The in-place += matters: rebinding core.beta would replace the
    # registered nn.Parameter with a plain Tensor and quietly break
    # core.parameters(), state_dict() and therefore save/load.
    data = _binary(10, 8, 2)
    beta, energy = reference_init(10, 8, 2, seed=0)
    core = Core(n_samples=10, n_features=8, n_components=2, beta=beta, energy=energy)
    fit_reference(core, data, lr=1e-4, max_iter=5, check_every=10_000)

    assert isinstance(core.beta, torch.nn.Parameter)
    assert isinstance(core.energy, torch.nn.Parameter)
    assert set(dict(core.named_parameters())) == {"beta", "energy"}
