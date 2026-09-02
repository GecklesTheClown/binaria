from itertools import pairwise

import pytest
import torch

from binaria._core import Core
from binaria._optim import fit
from binaria.selection import make_block_mask


def _synthetic_binary(n_samples: int, n_features: int, true_k: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    beta = torch.randn(n_samples, true_k, dtype=torch.float64)
    energy = torch.randn(true_k, n_features, dtype=torch.float64)
    return torch.bernoulli(torch.sigmoid(-(beta @ energy)))


def test_l2_penalty_is_sum_of_squared_frobenius_norms() -> None:
    torch.manual_seed(0)
    core = Core(n_samples=6, n_features=5, n_components=3)
    expected = core.beta.pow(2).sum() + core.energy.pow(2).sum()
    assert torch.allclose(core.l2_penalty(), expected)


def test_l2_penalty_equals_twice_nuclear_norm_at_balanced_factorization() -> None:
    # The identity that makes this principled rather than ad-hoc:
    # min over factorizations of (||b||^2 + ||E||^2) == 2 * ||b @ E||_*.
    # Checked at the balanced (SVD) factorization, which is the minimiser.
    torch.manual_seed(0)
    matrix = torch.randn(20, 6, dtype=torch.float64) @ torch.randn(6, 15, dtype=torch.float64)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    rank = 6
    beta = u[:, :rank] @ torch.diag(s[:rank].sqrt())
    energy = torch.diag(s[:rank].sqrt()) @ vh[:rank]

    core = Core(
        n_samples=20, n_features=15, n_components=rank, beta=beta.clone(), energy=energy.clone()
    )
    assert torch.allclose(core.l2_penalty(), 2.0 * s[:rank].sum())
    # ...and this factorization really is balanced
    assert torch.allclose(beta.T @ beta, energy @ energy.T, atol=1e-10)


def test_gauge_transform_preserves_predictions_but_not_the_penalty() -> None:
    # Why the penalty partially fixes the gauge: the GL(K) orbit leaves the
    # product (and so every prediction) identical while changing the penalty,
    # so minimising it selects a canonical representative.
    torch.manual_seed(0)
    core = Core(n_samples=12, n_features=8, n_components=3)
    gauge = torch.randn(3, 3, dtype=torch.float64)

    transformed = Core(
        n_samples=12,
        n_features=8,
        n_components=3,
        beta=(core.beta.detach() @ gauge).clone(),
        energy=(torch.linalg.inv(gauge) @ core.energy.detach()).clone(),
    )

    assert torch.allclose(core(), transformed(), atol=1e-8)
    assert not torch.isclose(core.l2_penalty(), transformed.l2_penalty(), rtol=1e-3)


@pytest.mark.parametrize(("n_samples", "n_features", "n_components"), [(6, 5, 3), (12, 4, 2)])
def test_penalized_analytic_gradient_matches_autograd(
    n_samples: int, n_features: int, n_components: int
) -> None:
    # The penalty is applied in _optim.fit(), not inside analytic_gradients().
    # This checks the two paths still agree once it is applied -- i.e. that
    # `-2 * alpha * param` really is the gradient of `-alpha * ||param||^2`.
    alpha = 0.05
    torch.manual_seed(0)
    core = Core(n_samples=n_samples, n_features=n_features, n_components=n_components)
    data = torch.randint(0, 2, (n_samples, n_features)).double()

    core.zero_grad()
    (core.log_likelihood(data) - alpha * core.l2_penalty()).backward()
    assert core.beta.grad is not None
    assert core.energy.grad is not None
    autograd_beta = core.beta.grad.clone()
    autograd_energy = core.energy.grad.clone()

    plain_beta, plain_energy = core.analytic_gradients(data)
    analytic_beta = plain_beta - 2.0 * alpha * core.beta.detach()
    analytic_energy = plain_energy - 2.0 * alpha * core.energy.detach()

    assert torch.allclose(autograd_beta, analytic_beta, atol=1e-12)
    assert torch.allclose(autograd_energy, analytic_energy, atol=1e-12)


def test_alpha_zero_is_bit_identical_to_no_regularization() -> None:
    # alpha must be a strictly additive feature: the default path cannot
    # change numerically just because the parameter now exists.
    data = _synthetic_binary(30, 20, true_k=3)
    results = []
    for alpha in (0.0, None):
        torch.manual_seed(42)
        core = Core(n_samples=30, n_features=20, n_components=3)
        kwargs = {} if alpha is None else {"alpha": alpha}
        fit(core, data, max_iter=100, lr=0.01, tol=0.0, **kwargs)  # type: ignore[arg-type]
        results.append(core.beta.detach().clone())
    assert torch.equal(results[0], results[1])


@pytest.mark.parametrize("gradient", ["analytic", "autograd"])
def test_alpha_shrinks_parameters(gradient: str) -> None:
    data = _synthetic_binary(40, 25, true_k=3)
    norms = []
    for alpha in (0.0, 0.01, 0.1, 1.0):
        torch.manual_seed(42)
        core = Core(n_samples=40, n_features=25, n_components=6)
        fit(core, data, max_iter=400, lr=0.05, tol=0.0, alpha=alpha, gradient=gradient)  # type: ignore[arg-type]
        with torch.no_grad():
            norms.append(core.l2_penalty().item())
    # STRICTLY decreasing in alpha. Not `norms == sorted(norms,
    # reverse=True)`: that predicate is true of a *constant* list, so if
    # alpha stopped having any effect on this path the test would pass
    # while asserting nothing. Verified by mutation -- deleting the penalty
    # term from the analytic gradient left this test green.
    assert all(a > b for a, b in pairwise(norms)), norms


def test_alpha_bounds_logit_magnitude() -> None:
    # The actual failure being fixed: unregularized fitting drives logits to
    # magnitudes where sigmoid saturates completely (|logit| > 35 already
    # pins a probability to within 1e-15 of 0 or 1), which then extrapolates
    # as confidently-wrong out-of-sample predictions.
    data = _synthetic_binary(80, 40, true_k=4)
    train_mask, _ = make_block_mask(80, 40, seed=0)

    magnitudes = []
    for alpha in (0.0, 1.0):
        torch.manual_seed(42)
        core = Core(n_samples=80, n_features=40, n_components=8)
        fit(core, data, mask=train_mask, max_iter=800, lr=0.05, tol=0.0, alpha=alpha)
        with torch.no_grad():
            magnitudes.append(core().abs().max().item())

    unregularized, regularized = magnitudes
    assert unregularized > 100.0
    assert regularized < 20.0


def test_unregularized_fit_never_claims_a_finite_optimum() -> None:
    torch.manual_seed(0)
    data = torch.randint(0, 2, (8, 6), dtype=torch.float64)
    result = fit(Core(8, 6, 2), data, alpha=0.0, max_iter=5, tol=1.0)
    assert result.converged is False
    assert result.n_iter == 5


def test_convergence_watches_the_objective_not_the_log_likelihood() -> None:
    # Regression test for a real bug class, not a hypothetical one.
    #
    # Under a penalty the optimizer maximises `LL - alpha * penalty`. Starting
    # from large parameters it correctly trades log-likelihood away to buy a
    # bigger reduction in the penalty, so LL *falls* for hundreds of steps
    # while the objective improves. A convergence rule watching LL would be
    # testing a quantity that is not being optimized.
    #
    # Note this only misbehaves from a large-parameter start -- from a small
    # init LL happens to stay monotone, so the bug passes the obvious test
    # and only shows up on warm starts and resumed checkpoints. Hence pinning
    # the awkward case explicitly.
    data = _synthetic_binary(80, 40, true_k=4)
    train_mask, _ = make_block_mask(80, 40, seed=0)
    alpha = 0.05

    torch.manual_seed(42)
    core = Core(n_samples=80, n_features=40, n_components=8)
    fit(core, data, mask=train_mask, max_iter=1500, lr=0.05, tol=0.0)  # unregularized blow-up
    with torch.no_grad():
        assert core.l2_penalty().item() > 1000.0, "warm-up should leave parameters large"

    optimizer = torch.optim.Adam(core.parameters(), lr=0.05, maximize=True)
    previous_ll = previous_objective = None
    ll_decreases = objective_decreases = 0
    for _ in range(200):
        grad_beta, grad_energy = core.analytic_gradients(data, mask=train_mask)
        with torch.no_grad():
            core.beta.grad = grad_beta - 2.0 * alpha * core.beta
            core.energy.grad = grad_energy - 2.0 * alpha * core.energy
        optimizer.step()
        with torch.no_grad():
            log_likelihood = core.log_likelihood(data, mask=train_mask).item()
            objective = log_likelihood - alpha * core.l2_penalty().item()
        if previous_ll is not None:
            if log_likelihood < previous_ll:
                ll_decreases += 1
            assert previous_objective is not None
            if objective < previous_objective:
                objective_decreases += 1
        previous_ll, previous_objective = log_likelihood, objective

    # The objective -- the thing actually being maximised -- improves nearly
    # everywhere, while the log-likelihood falls nearly everywhere.
    assert objective_decreases < 10
    assert ll_decreases > 150


@pytest.mark.parametrize("alpha", [0.05, 0.5, 2.0])
def test_both_gradient_paths_agree_through_the_whole_fit(alpha: float) -> None:
    # The gradient-level test above checks the penalty derivative at a
    # single point, with the correction applied by hand. This checks the
    # thing that actually runs: fit() applies `-2 * alpha * param` on the
    # analytic path inside its own loop, so the two paths can only agree
    # end to end if that is present and correct.
    #
    # Worth having as its own test because the coverage was previously
    # incidental -- deleting the penalty term from the analytic path was
    # caught only by two behavioural tests elsewhere, and could plausibly
    # have been missed entirely by a slightly different test suite. The
    # analytic path is the default, so a silent divergence here would be
    # the one nobody notices.
    data = _synthetic_binary(40, 25, true_k=3)

    results = []
    for path in ("analytic", "autograd"):
        torch.manual_seed(42)
        core = Core(n_samples=40, n_features=25, n_components=5)
        fit(core, data, max_iter=300, lr=0.05, tol=0.0, alpha=alpha, gradient=path)  # type: ignore[arg-type]
        results.append(core.beta.detach().clone())

    # Bit-identical: same seed, same arithmetic, same order of operations.
    assert torch.equal(results[0], results[1])
