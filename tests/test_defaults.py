"""
The shipped defaults, and why they are what they are.

These pin behaviour a user gets without passing anything, which is the
configuration most people will actually run and the one least likely to be
covered by tests that set every parameter explicitly.

Two defaults are entangled and must move together. ``alpha`` is nonzero
because at ``alpha=0`` the objective has no maximizer -- it rises without
bound along any direction that signs entries correctly -- so there is
nothing for a stopping rule to detect and ``converged_`` can never be
True. ``learning_rate`` is 0.1 because that reaches the optimum roughly
twenty times faster than the old 1e-3, which is only a good idea once
there *is* an optimum: on separable data a faster rate at ``alpha=0``
merely drives the sigmoid into saturation sooner.

Measured in ``benchmarks/learning_rate_sweep.py``; the numbers quoted in
``SiGMoiD``'s docstring come from there.
"""

import warnings

import numpy as np
import torch

from binaria import SiGMoiD


def _binary(n_samples: int = 200, n_features: int = 100, true_k: int = 3, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    beta = torch.randn(n_samples, true_k, generator=generator, dtype=torch.float64)
    energy = torch.randn(true_k, n_features, generator=generator, dtype=torch.float64)
    return torch.bernoulli(torch.sigmoid(-(beta @ energy)), generator=generator).numpy()


def _fit(**kwargs: object) -> SiGMoiD:
    params: dict[str, object] = {"n_components": 3, "random_state": 0}
    params.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return SiGMoiD(**params).fit(_binary())  # type: ignore[arg-type]


def test_the_default_configuration_converges() -> None:
    # The load-bearing one. Before alpha defaulted nonzero this could not
    # pass at any budget: the convergence check is gated on alpha > 0
    # precisely because an unpenalized fit has nothing to converge to.
    model = _fit()
    assert model.converged_ is True
    assert model.n_iter_ < model.max_iter
    assert model.n_iter_ >= model.patience + 1


def test_the_default_patience_is_conservative() -> None:
    assert SiGMoiD(n_components=3).patience == 100


def test_the_default_alpha_is_small_but_nonzero() -> None:
    # Nonzero so an optimum exists; small so it barely moves the fit.
    # If this is ever set back to 0.0, converged_ becomes permanently
    # False and every downstream convergence check goes silent rather
    # than failing, so it is asserted rather than left to a docstring.
    assert SiGMoiD(n_components=3).alpha > 0.0
    assert SiGMoiD(n_components=3).alpha <= 0.1


def test_the_default_alpha_costs_almost_nothing_in_fit() -> None:
    # The justification for the specific value: the penalty buys existence
    # of an optimum without meaningfully worse likelihood than the best an
    # unpenalized fit reaches.
    penalized = _fit()
    unpenalized = _fit(alpha=0.0)
    gap = abs(penalized.log_likelihood_ - unpenalized.log_likelihood_)
    assert gap / abs(unpenalized.log_likelihood_) < 0.01


def test_the_default_rate_reaches_a_better_optimum_far_sooner() -> None:
    # Paired: identical data, seed and budget, only the rate differs. The
    # old default needed ~20x the iterations and still landed lower, which
    # is the whole case for the change.
    fast = _fit()
    slow = _fit(learning_rate=1e-3)
    assert fast.log_likelihood_ > slow.log_likelihood_
    assert fast.n_iter_ * 5 < slow.n_iter_


def test_unpenalized_fitting_is_still_available() -> None:
    # alpha=0.0 reproduces the original procedure and must keep working.
    # No claim is made here about convergence: the relative-change rule
    # fires at alpha=0 too, so converged_ is True without there being an
    # optimum to have reached. That is exactly why the default is nonzero,
    # and why this test asserts only that the opt-out still runs.
    model = _fit(alpha=0.0, max_iter=200)
    assert np.isfinite(model.components_).all()
    assert np.isfinite(model.embedding_).all()
    assert model.log_likelihood_ < 0.0


def test_the_penalty_keeps_logits_smaller_than_no_penalty_at_all() -> None:
    # The mechanism behind the default, asserted on the gauge-invariant
    # quantity: beta and E are only defined up to GL(K), but their product
    # is not, so the logit scale is the thing that can be compared.
    penalized = _fit(max_iter=3000)
    unpenalized = _fit(alpha=0.0, max_iter=3000)
    with torch.no_grad():
        assert penalized._core().abs().max() < unpenalized._core().abs().max()
