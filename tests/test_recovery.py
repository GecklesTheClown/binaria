"""
Does selection actually find the true k?

Every other selection test checks the *machinery* -- that the t-test is
applied correctly, that BH corrects, that elimination drops the right
cells. None of them check the thing the machinery exists for. These do.

Assertions are on the *selected k*, never on beta or E -- the GL(K) gauge
freedom makes those meaningless to compare.

Three kinds of claim live here, and conflating them is how a test suite
starts lying. Selection is *stochastic* -- random Bernoulli draw, random
partitions, random initialisations -- so "selects 4" is never something
that can be asserted with certainty. It can only be pinned at a seed.

**Deterministic properties.** Same seed gives the same answer; held-out
score rises monotonically with k below the convergence floor; the
generator hits its requested density; ``identifiable_rank`` computes what
it says. These are true regardless of seed and are asserted outright.

**Paired comparisons.** Penalised versus unpenalised *on identical data*,
or more data versus less. The comparison cancels most of the noise, so
these survive a seed change in a way absolute claims do not. Substantive
conclusions belong in this form wherever possible.

**Regression pins.** A fixed seed produced a particular answer, recorded
so that a change is noticed. A red pin means *investigate*, not
necessarily *bug* -- three times today a failing recovery assertion turned
out to be the test's premise being wrong rather than the selector: a
degenerate generator component, an iteration budget below the convergence
floor, and an absolute assertion at an unverified seed.

The *rate* -- how often recovery succeeds across seeds and regimes -- is
not asserted anywhere here. It is measured in
``benchmarks/recovery_sweep.py`` and reported as a documented table,
because a rate assertion carries its own false-failure probability: at 90%
per-seed recovery, ``assert hits >= 9/10`` goes red about a quarter of the
time, and a test that is right about the method while failing a quarter of
the time gets deleted.

Two disciplines this file exists to enforce, both learned by getting them
wrong first:

**Verify the estimand before blaming the estimator.** Nominal k is a
property of the generator; what selection can recover is the rank at which
held-out likelihood stops improving. An earlier generator reserved a
constant component, whose contribution is a rank-1 matrix with a single
distinct value -- absorbed trivially by a (k-1)-component model. Selection
returned k-1 for nominal k = 3, 4 and 5 and was recorded as four failures
when it had been right every time. ``identifiable_rank`` now measures the
estimand directly, and ``test_generator_makes_true_k_identifiable`` makes
it a precondition rather than an assumption.

**Check convergence before reading the answer.** Below the iteration floor
the held-out score rises monotonically with k, so the ranking returns the
largest candidate in the grid and the sweep is measuring the optimizer
rather than the criterion. The floor is problem-dependent, and it moves
with the learning rate as well as with the data: at 400x200 the pins below
converge fully in 1500 iterations at the default rate of 0.1 and not at
all at 1e-3. Pinned by
``test_under_convergence_selects_the_largest_k_in_the_grid``.
"""

import warnings

import numpy as np
import pytest
import torch

from binaria.selection import SiGMoiDSelector, make_block_mask


def _true_logits(
    n_samples: int, n_features: int, true_k: int, density: float, seed: int
) -> torch.Tensor:
    """
    Rank-`true_k` logits whose Bernoulli draws average `density` ones.

    SiGMoiD has no intercept -- ``pi = sigmoid(-beta @ E)`` exactly -- so
    density cannot be dialled in with a bias term without generating data
    the model is unable to represent, at which point a failure to recover
    k would be measuring misspecification rather than selection.

    An earlier version reserved one component as a constant
    (``beta[:, 0] = 1``, ``energy[0, :] = c``). That was wrong in a way
    worth recording: the reserved component contributes a matrix of rank
    one with a *single distinct value*, so a (k-1)-component model absorbs
    it and the identifiable rank sits below the nominal one. Selection
    duly returned k-1 and looked broken when it was correct.

    Instead both factors carry a nonzero mean, which shifts every logit
    without spending a dimension: ``E[beta @ E] = k * mean_b * mean_e``.
    Every component stays generic, so the logits are full rank k.
    """
    generator = torch.Generator().manual_seed(seed)
    beta = torch.randn(n_samples, true_k, dtype=torch.float64, generator=generator) + 1.0
    base = torch.randn(true_k, n_features, dtype=torch.float64, generator=generator)

    def logits_at(shift: float) -> torch.Tensor:
        return -(beta @ (base + shift))

    low, high = -8.0, 8.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if float(torch.sigmoid(logits_at(mid)).mean()) > density:
            low = mid
        else:
            high = mid
    return logits_at(0.5 * (low + high))


def identifiable_rank(logits: torch.Tensor, seed: int = 0, tol: float = 0.002) -> int:
    """
    How many components are actually *detectable* in data from `logits`.

    Nominal k is a property of the generator; what any estimator can
    recover is the rank at which held-out likelihood stops improving. The
    two come apart whenever a component's contribution is small against
    Bernoulli sampling noise -- a generator can be exactly rank 6 and
    carry only 4 detectable components at 80x40.

    Computed without any fitting: truncate the *true* logits to rank r by
    SVD and score the truncation on a held-out block. That is an upper
    bound on what an estimator could achieve, so if it peaks below nominal
    k then nominal k is not the estimand and no "recovery rate" measured
    against it means anything.

    Returns the smallest r within `tol` nats/entry of the best score.
    Smallest, not argmax: beyond rank k the truncation is *exact*, so
    ranks k, k+1, ... tie to within floating point and argmax would pick
    among them arbitrarily.
    """
    n_samples, n_features = logits.shape
    torch.manual_seed(seed)
    data = torch.bernoulli(torch.sigmoid(logits))
    _, test_mask = make_block_mask(n_samples, n_features, seed=seed)
    n_held = int(test_mask.sum())

    u, s, vh = torch.linalg.svd(logits, full_matrices=False)
    scores = []
    for rank in range(1, len(s) + 1):
        approx = (u[:, :rank] * s[:rank]) @ vh[:rank]
        per_entry = -torch.nn.functional.binary_cross_entropy_with_logits(
            approx, data, reduction="none"
        )
        scores.append(float((test_mask * per_entry).sum() / n_held))

    best = max(scores)
    return next(r for r, value in enumerate(scores, start=1) if value >= best - tol)


def _rank_k_binary(
    n_samples: int, n_features: int, true_k: int, density: float = 0.5, seed: int = 0
) -> torch.Tensor:
    logits = _true_logits(n_samples, n_features, true_k, density, seed)
    generator = torch.Generator().manual_seed(seed)
    return torch.bernoulli(torch.sigmoid(logits), generator=generator)


def _select(data: torch.Tensor, **kwargs: object) -> SiGMoiDSelector:
    defaults: dict[str, object] = {
        "criterion": "held_out_ll",
        "alpha_range": (0.0, 0.3, 1.0),
        "n_repeats": 3,
        "max_iter": 1500,
        "random_state": 0,
    }
    defaults.update(kwargs)
    selector = SiGMoiDSelector([2, 3, 4, 6, 8], **defaults)  # type: ignore[arg-type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(data.numpy())
    return selector


def test_generator_produces_the_requested_density() -> None:
    # If this drifts, every sparsity result below is measuring something
    # other than what it says.
    for density in (0.1, 0.3, 0.5):
        data = _rank_k_binary(200, 100, true_k=4, density=density, seed=0)
        assert float(data.mean()) == pytest.approx(density, abs=0.03)


@pytest.mark.parametrize("true_k", [2, 4, 6])
@pytest.mark.parametrize(("n_samples", "n_features"), [(80, 40), (400, 200)])
def test_generator_makes_true_k_identifiable(true_k: int, n_samples: int, n_features: int) -> None:
    # The precondition every recovery test below silently depends on. If
    # the generator produces a component that cannot be detected even by
    # an oracle with the true parameters, then "recovery of true k" is not
    # a well-posed question and a failure downstream would be measuring
    # the generator rather than the selector.
    #
    # This is exactly what went wrong with the first version, which
    # reserved a constant component: at 80x40 with nominal k=4 the
    # identifiable rank was 3, and selection was marked down for returning
    # the right answer.
    logits = _true_logits(n_samples, n_features, true_k, density=0.5, seed=7)
    assert torch.linalg.matrix_rank(logits).item() == true_k
    assert identifiable_rank(logits, seed=7) == true_k


def test_generated_data_is_binary_and_the_right_shape() -> None:
    data = _rank_k_binary(60, 30, true_k=3)
    assert data.shape == (60, 30)
    assert set(torch.unique(data).tolist()) <= {0.0, 1.0}


@pytest.mark.slow
def test_recovery_is_reproducible_from_the_seed() -> None:
    # Selection involves random partitions and random inits. If two runs at
    # the same random_state disagree, no recovery rate reported anywhere is
    # meaningful.
    data = _rank_k_binary(200, 100, true_k=4, seed=9)
    first, second = _select(data), _select(data)
    assert first.best_n_components_ == second.best_n_components_
    assert first.best_alpha_ == second.best_alpha_
    assert first.best_score_ == second.best_score_


@pytest.mark.slow
def test_under_convergence_selects_the_largest_k_in_the_grid() -> None:
    # The trap that made three of these tests look broken before it was
    # diagnosed, pinned so it stays visible.
    #
    # Below the iteration floor the held-out score is monotone INCREASING
    # in k, so selection returns whatever the largest candidate happens to
    # be -- a bigger model makes faster progress per iteration and has not
    # yet had time to overfit, so more capacity always looks better. The
    # answer then appears stable across nearby budgets because it is
    # pinned to the top of the grid rather than to the data.
    #
    # The floor is problem-dependent and moves a long way, and it moves
    # with the learning rate too: at the default 0.1 this problem passes
    # through the monotone regime somewhere around 10 iterations, which is
    # too narrow a window to pin. learning_rate is therefore fixed at the
    # slow 1e-3 that makes the regime wide and reproducible. The pathology
    # is a property of being below the floor, not of any particular rate --
    # pinning the rate keeps this test from re-deriving itself every time
    # the default moves.
    data = _rank_k_binary(120, 60, true_k=4, seed=13)
    selector = _select(data, max_iter=40, alpha_range=(0.0,), n_repeats=2, learning_rate=1e-3)

    assert selector.cv_results_["converged"].mean() == 0.0
    scores = selector.summary_["median_score"]
    components = selector.summary_["n_components"]

    # The pathology lives in the RANKING, which is what the criterion
    # actually reports. Whether it becomes best_n_components_ depends on
    # the stopping rule: with few repeats nothing separates, parsimony
    # fires, and the answer flips to the *smallest* k instead. Both
    # outcomes are wrong and neither is about the data -- which is why the
    # assertion is on the ranking rather than on the final pick.
    assert list(scores) == sorted(scores), (
        "under-convergence should make held-out score rise monotonically with k"
    )
    assert components[int(np.argmax(scores))] == max(components)
    assert selector.selection_rule_ in {"separation", "parsimony"}


@pytest.mark.slow
def test_convergence_fraction_is_recorded_so_the_trap_is_detectable() -> None:
    # The escape hatch for the above: a sweep reporting 0% converged is
    # measuring the optimizer rather than the criterion, and cv_results_
    # has to make that checkable without rerunning anything.
    data = _rank_k_binary(120, 60, true_k=3, seed=17)
    starved = _select(data, max_iter=40, alpha_range=(0.0,), n_repeats=2)
    assert "converged" in starved.cv_results_
    assert starved.cv_results_["converged"].mean() == 0.0

    # And with a real budget and regularization, some fits do converge.
    trained = _select(data, max_iter=3000, alpha_range=(0.3, 1.0), n_repeats=2)
    assert trained.cv_results_["converged"].mean() > 0.0


# --- paired comparisons ---------------------------------------------------
#
# The substantive claims live here rather than in absolute assertions.
# Comparing two runs on *identical data* cancels most of the noise that
# makes "selects 4" a coin flip with a fixed seed.


@pytest.mark.slow
def test_a_larger_iteration_budget_converges_the_penalized_cells() -> None:
    # Paired: same matrix, same alpha grid, same seeds, only the budget
    # differs. Unpenalized fits deliberately never report convergence: the
    # relative-change rule can fire while their factors continue growing
    # toward a likelihood supremum at infinity. With enough budget, every
    # penalized cell in this sweep does converge.
    data = _rank_k_binary(80, 40, true_k=4, seed=3)
    starved = _select(data, alpha_range=(0.0, 0.3, 1.0), max_iter=20)
    trained = _select(data, alpha_range=(0.0, 0.3, 1.0), max_iter=6000)

    assert starved.cv_results_["converged"].mean() == 0.0
    assert trained.cv_results_["converged"].mean() == pytest.approx(2.0 / 3.0)


@pytest.mark.slow
def test_neither_budget_reaches_the_identifiable_rank_at_this_size() -> None:
    # Recorded because it is the honest limit, not a defect to be fixed by
    # tuning: an oracle holding the true parameters identifies rank 4 at
    # 80x40, and the estimator does not get there even at 6000 iterations
    # with every penalized fit converged and alpha swept to 1.0. The gap is
    # estimation error -- 80*4 + 4*40 = 480 parameters from 2400 visible
    # entries -- not a failure of the criterion.
    logits = _true_logits(80, 40, true_k=4, density=0.5, seed=3)
    assert identifiable_rank(logits, seed=3) == 4

    data = _rank_k_binary(80, 40, true_k=4, seed=3)
    trained = _select(data, alpha_range=(0.0, 0.3, 1.0), max_iter=6000)
    assert trained.best_n_components_ < 4
    # ...and it says so, rather than asserting a winner it has not earned.
    assert trained.selection_rule_ == "parsimony"
    assert trained.resolved_ is False


# --- regression pins ------------------------------------------------------
#
# A fixed seed produced a particular answer. Recorded so a change is
# noticed; a red pin means INVESTIGATE, not BUG.
#
# Every penalized cell below converges and each sweep resolves by
# *separation*, which is asserted alongside the pinned k. The unpenalized
# third of the grid deliberately remains marked unconverged because its
# objective need not attain a finite maximum. These therefore record real
# penalized recovery, not the tie-break path.


@pytest.mark.slow
@pytest.mark.parametrize(
    ("shape", "true_k", "density", "seed", "expected"),
    [
        ((400, 200), 4, 0.5, 7, 4),
        ((400, 200), 4, 0.5, 11, 4),
        ((400, 200), 4, 0.3, 11, 4),
    ],
)
def test_regression_pin_selected_k(
    shape: tuple[int, int], true_k: int, density: float, seed: int, expected: int
) -> None:
    data = _rank_k_binary(*shape, true_k=true_k, density=density, seed=seed)
    selector = _select(data, alpha_range=(0.0, 0.3, 1.0), max_iter=1500)

    assert selector.best_n_components_ == expected
    # The regime this pin was taken in. If either of these changes, the
    # pinned value above is measuring something different and must be
    # re-derived rather than adjusted.
    assert selector.cv_results_["converged"].mean() == pytest.approx(2.0 / 3.0)
    assert selector.selection_rule_ == "separation"


def test_true_k_must_not_sit_at_either_end_of_the_candidate_grid() -> None:
    # A design rule for every recovery test, learned from a pin that was
    # passing for no reason. The bottom of the grid is reachable by
    # parsimony's tie-break and the top by under-convergence, so a test
    # whose true_k sits at either end can pass without selection working
    # at all. The grid used throughout this file is [2, 3, 4, 6, 8], so
    # usable true_k values are 3, 4 and 6.
    grid = [2, 3, 4, 6, 8]
    for true_k in (3, 4, 6):
        assert min(grid) < true_k < max(grid)
