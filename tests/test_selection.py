import json
import warnings
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from binaria._core import Core
from binaria._optim import fit
from binaria.selection import (
    _AIC,
    _BIC,
    _DATA_CACHE,
    _HELD_OUT_LL,
    DEFAULT_ALPHA_RANGE,
    SiGMoiDSelector,
    _best_index,
    _resolve_data,
    aic,
    bic,
    make_block_mask,
    param_count,
)


def _synthetic_binary(n_samples: int, n_features: int, true_k: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    beta = torch.randn(n_samples, true_k, dtype=torch.float64)
    energy = torch.randn(true_k, n_features, dtype=torch.float64)
    return torch.bernoulli(torch.sigmoid(-(beta @ energy)))


# --- masking -------------------------------------------------------------


@pytest.mark.parametrize(("n_samples", "n_features"), [(8, 6), (5, 7), (2, 2), (40, 11)])
def test_block_mask_is_a_complementary_block(n_samples: int, n_features: int) -> None:
    train_mask, test_mask = make_block_mask(n_samples, n_features, seed=0)

    ones = torch.ones(n_samples, n_features, dtype=torch.float64)
    assert torch.equal(train_mask + test_mask, ones)

    # The held-out set must be an actual (rows x cols) block, not scattered
    # entries: |held rows| * |held cols| must equal the number of held
    # entries exactly. Scattered dropout would fail this.
    n_held_rows = int((test_mask.sum(dim=1) > 0).sum())
    n_held_cols = int((test_mask.sum(dim=0) > 0).sum())
    assert n_held_rows * n_held_cols == int(test_mask.sum())

    # Every row and column must keep visible entries -- this is what lets
    # beta_s be fit for every sample without out-of-sample inference.
    assert bool((train_mask.sum(dim=1) > 0).all())
    assert bool((train_mask.sum(dim=0) > 0).all())


def test_block_mask_is_deterministic_per_seed() -> None:
    a, _ = make_block_mask(10, 8, seed=7)
    b, _ = make_block_mask(10, 8, seed=7)
    c, _ = make_block_mask(10, 8, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize(("n_samples", "n_features"), [(1, 5), (5, 1), (0, 3)])
def test_block_mask_rejects_degenerate_shapes(n_samples: int, n_features: int) -> None:
    with pytest.raises(ValueError, match="at least 2 samples and 2 features"):
        make_block_mask(n_samples, n_features)


@pytest.mark.parametrize(("n_samples", "n_features", "n_components"), [(6, 5, 3), (12, 4, 2)])
def test_masked_analytic_gradient_matches_autograd(
    n_samples: int, n_features: int, n_components: int
) -> None:
    # The masked gradient is the same Eq 3 formula restricted to visible
    # entries. If masking were applied in the wrong place (e.g. after the
    # matmul instead of to the residual), this would diverge.
    torch.manual_seed(0)
    core = Core(n_samples=n_samples, n_features=n_features, n_components=n_components)
    data = torch.randint(0, 2, (n_samples, n_features)).double()
    mask = (torch.rand(n_samples, n_features) > 0.3).double()

    core.zero_grad()
    core.log_likelihood(data, mask=mask).backward()
    assert core.beta.grad is not None
    assert core.energy.grad is not None
    autograd_beta = core.beta.grad.clone()
    autograd_energy = core.energy.grad.clone()

    analytic_beta, analytic_energy = core.analytic_gradients(data, mask=mask)

    assert torch.allclose(autograd_beta, analytic_beta, atol=1e-12)
    assert torch.allclose(autograd_energy, analytic_energy, atol=1e-12)


def test_mask_of_all_ones_matches_unmasked() -> None:
    # mask=None must take the original path, not a numerically-different one.
    torch.manual_seed(0)
    core = Core(n_samples=6, n_features=5, n_components=3)
    data = torch.randint(0, 2, (6, 5)).double()
    ones = torch.ones(6, 5, dtype=torch.float64)

    assert core.log_likelihood(data).item() == core.log_likelihood(data, mask=ones).item()

    unmasked = core.analytic_gradients(data)
    all_visible = core.analytic_gradients(data, mask=ones)
    assert torch.equal(unmasked[0], all_visible[0])
    assert torch.equal(unmasked[1], all_visible[1])


def test_worker_data_cache_keeps_dtypes_separate(tmp_path: Path) -> None:
    path = tmp_path / "data.pt"
    torch.save(torch.ones(2, 2), path)
    _DATA_CACHE.clear()
    try:
        float32 = _resolve_data(path, dtype=torch.float32, device=None)
        float64 = _resolve_data(path, dtype=torch.float64, device=None)
    finally:
        _DATA_CACHE.clear()
    assert float32.dtype == torch.float32
    assert float64.dtype == torch.float64


def test_masked_fit_ignores_held_out_entries() -> None:
    # Corrupting only the held-out block must not change a masked fit at
    # all. This is the strongest available check that the mask really is
    # excluding those entries from the objective, rather than merely
    # down-weighting them somewhere.
    data = _synthetic_binary(30, 20, true_k=3)
    train_mask, test_mask = make_block_mask(30, 20, seed=0)
    corrupted = torch.where(test_mask.bool(), 1.0 - data, data)

    results = []
    for matrix in (data, corrupted):
        torch.manual_seed(42)
        core = Core(n_samples=30, n_features=20, n_components=3)
        fit(core, matrix, mask=train_mask, max_iter=50, lr=0.01, tol=0.0)
        results.append(core.beta.detach().clone())

    assert torch.equal(results[0], results[1])


# --- criteria ------------------------------------------------------------


def test_param_count() -> None:
    assert param_count(100, 5, 20) == 100 * 5 + 5 * 20


def test_aic_and_bic_penalise_complexity() -> None:
    # At equal log-likelihood, more parameters must score worse (higher,
    # since both are minimized).
    assert aic(-500.0, 600) > aic(-500.0, 300)
    assert bic(-500.0, 600, 100) > bic(-500.0, 300, 100)
    # And at equal parameter count, better fit must score better.
    assert aic(-400.0, 600) < aic(-500.0, 600)


# --- direction of goodness ----------------------------------------------


def test_best_index_selects_opposite_ends_for_monotone_scores() -> None:
    # The trap this whole design exists to prevent: a hardcoded argmin
    # would silently pick the *worst* model under a maximized criterion.
    # With monotone scores the two directions must land on opposite ends.
    monotone = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _best_index(monotone, greater_is_better=False) == 0
    assert _best_index(monotone, greater_is_better=True) == len(monotone) - 1


def test_criterion_directions_are_declared_correctly() -> None:
    assert _AIC.greater_is_better is False
    assert _BIC.greater_is_better is False
    assert _HELD_OUT_LL.greater_is_better is True
    assert _AIC.requires_masking is False
    assert _BIC.requires_masking is False
    assert _HELD_OUT_LL.requires_masking is True


def test_selector_rejects_criterion_objects_instead_of_mis_scoring_them() -> None:
    criterion = SimpleNamespace(name="custom", greater_is_better=True, requires_masking=False)
    selector = SiGMoiDSelector([2, 3], criterion=criterion)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expected one of"):
        selector.fit(_synthetic_binary(20, 12, true_k=2).numpy())


@pytest.mark.parametrize("criterion", ["aic", "bic", "held_out_ll"])
def test_selector_best_matches_its_criterion_direction(criterion: str) -> None:
    # Pins direction per criterion against the actual selected value,
    # rather than trusting that the flag is consulted correctly.
    # Uses the sign test at its floor: with 5 repeats a unanimous result
    # sits right at the significance boundary, so the run is very unlikely
    # to separate and reports the ranking leader -- which is what the
    # direction is being checked against.
    data = _synthetic_binary(60, 30, true_k=3).numpy()
    selector = SiGMoiDSelector(
        [2, 3, 5],
        alpha_range=(0.0,),
        criterion=criterion,
        test="sign",
        n_repeats=5,
        max_iter=60,
        learning_rate=0.01,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(data)

    # Asserted on the RANKING leader, not on best_n_components_: when the
    # test cannot separate, parsimony legitimately returns a smaller k than
    # the ranking favours, which would mask a criterion scored backwards.
    medians = selector.summary_["median_score"]
    components = selector.summary_["n_components"]
    expected = components[
        int(np.argmax(medians) if criterion == "held_out_ll" else np.argmin(medians))
    ]
    _, _, leader, _ = selector._separation(selector.scores_, selector._resolve_criterion())
    assert leader[0] == expected


# --- selector ------------------------------------------------------------


def test_selector_records_schema_and_seeds() -> None:
    data = _synthetic_binary(40, 20, true_k=2).numpy()
    selector = SiGMoiDSelector(
        [2, 3],
        alpha_range=(0.0,),
        criterion="held_out_ll",
        n_repeats=2,
        max_iter=40,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(data)

    assert len(selector.cv_results_["score"]) == 4
    # held_out_ll uses a partition and has no parameter count; the reverse
    # holds for AIC/BIC. Recording both seeds is what makes "repeat"
    # unambiguous between the two.
    assert all(seed is not None for seed in selector.cv_results_["partition_seed"])
    assert all(count is None for count in selector.cv_results_["n_params"])
    assert bool(selector.cv_results_["greater_is_better"][0]) is True
    assert bool((selector.cv_results_["fit_time"] > 0.0).all())
    assert bool((selector.summary_["total_fit_time"] > 0.0).all())


def test_audit_mode_records_and_saves_the_complete_decision(tmp_path: Path) -> None:
    data = _synthetic_binary(30, 16, true_k=2).numpy()
    selector = SiGMoiDSelector(
        [2, 3],
        alpha_range=(0.3,),
        n_repeats=2,
        max_iter=20,
        random_state=0,
        audit=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(data)

    trail = selector.audit_trail_
    assert len(trail["settings"]["data_digest"]) == 64
    assert trail["settings"]["patience"] == selector.patience
    assert len(trail["fits"]) == 4
    assert len(trail["summary"]) == 2
    assert trail["decision"]["selected"]["n_components"] == selector.best_n_components_
    assert trail["decision"]["selection_rule"] == selector.selection_rule_
    assert trail["decision"]["comparisons"]
    assert "adjusted_pvalue" in trail["decision"]["comparisons"][0]
    assert trail["convergence"]["total_fit_time"] > 0.0

    path = tmp_path / "nested" / "selection-audit.json"
    selector.save_audit(path)
    with path.open(encoding="utf-8") as stream:
        assert json.load(stream) == trail
    assert not path.with_name(path.name + ".tmp").exists()


def test_audit_export_requires_a_run_in_audit_mode(tmp_path: Path) -> None:
    selector = SiGMoiDSelector([2], alpha_range=(0.0,), n_repeats=2)
    with pytest.raises(ValueError, match="audit=True"):
        selector.save_audit(tmp_path / "audit.json")


def test_selector_aic_has_no_partition_but_has_params() -> None:
    data = _synthetic_binary(40, 20, true_k=2).numpy()
    selector = SiGMoiDSelector(
        [2, 3], alpha_range=(0.0,), criterion="aic", n_repeats=2, max_iter=40, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(data)

    assert all(seed is None for seed in selector.cv_results_["partition_seed"])
    assert selector.cv_results_["n_params"][0] == param_count(40, 2, 20)


def test_selector_warns_about_unconverged_fits_but_keeps_them() -> None:
    data = _synthetic_binary(40, 20, true_k=2).numpy()
    selector = SiGMoiDSelector(
        [2, 3],
        alpha_range=(0.0,),
        criterion="aic",
        n_repeats=2,
        max_iter=3,
        tol=1e-12,
        random_state=0,
    )
    with pytest.warns(UserWarning, match="did not converge"):
        selector.fit(data)

    # Recorded, not dropped -- silently discarding them would hide exactly
    # the failure mode the warning exists to surface.
    assert len(selector.cv_results_["score"]) == 2 * 2  # 2 k x 2 repeats
    assert not any(selector.cv_results_["converged"])
    assert selector.summary_["n_converged"].tolist() == [0, 0]


def test_selector_rejects_unknown_criterion() -> None:
    selector = SiGMoiDSelector([2], n_repeats=5, alpha_range=(0.0,), criterion="not_a_criterion")
    with pytest.raises(ValueError, match="Unknown criterion"):
        selector.fit(_synthetic_binary(20, 10, true_k=2).numpy())


def test_selector_recovers_true_rank_in_a_well_sampled_regime() -> None:
    # Adequately sampled (~25 observations per parameter at the true k).
    # In an under-sampled regime this does NOT hold -- unregularized
    # separation dominates and the selected k becomes an artifact of the
    # training budget instead (see the release plan's 6.1.2).
    data = _synthetic_binary(400, 200, true_k=4).numpy()
    selector = SiGMoiDSelector(
        [2, 3, 4, 6, 8],
        alpha_range=(0.0,),
        criterion="held_out_ll",
        n_repeats=2,
        max_iter=300,
        learning_rate=0.01,
        tol=0.0,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(data)

    assert selector.best_n_components_ == 4


def test_selector_sweeps_joint_k_alpha_grid() -> None:
    data = _synthetic_binary(40, 20, true_k=2).numpy()
    selector = SiGMoiDSelector(
        [2, 3],
        alpha_range=(0.0, 0.1),
        criterion="held_out_ll",
        n_repeats=2,
        max_iter=40,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(data)

    # 2 k x 2 alpha x 2 repeats
    assert len(selector.cv_results_["score"]) == 8
    assert sorted(set(selector.cv_results_["alpha"].tolist())) == [0.0, 0.1]
    assert len(selector.summary_["n_components"]) == 4  # one row per cell
    assert selector.best_alpha_ in (0.0, 0.1)
    assert selector.best_n_components_ in (2, 3)


def test_selector_warns_when_alpha_combined_with_information_criterion() -> None:
    # param_count assumes unpenalized MLE, so a penalized AIC is neither
    # standard AIC nor a corrected one -- warn rather than return a
    # confidently-wrong number.
    data = _synthetic_binary(30, 15, true_k=2).numpy()
    selector = SiGMoiDSelector(
        [2], alpha_range=(0.5,), criterion="aic", n_repeats=2, max_iter=20, random_state=0
    )
    with pytest.warns(UserWarning, match="assumes unpenalized maximum"):
        selector.fit(data)


def test_selector_does_not_warn_for_alpha_with_held_out_ll() -> None:
    # held_out_ll needs no parameter count, so regularization composes fine.
    data = _synthetic_binary(30, 15, true_k=2).numpy()
    selector = SiGMoiDSelector(
        [2], alpha_range=(0.5,), criterion="held_out_ll", n_repeats=2, max_iter=20, random_state=0
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selector.fit(data)
    # An unconverged-fit warning is legitimate here and unrelated; assert
    # only that the parameter-count warning specifically did not fire.
    assert not any("assumes unpenalized maximum" in str(w.message) for w in caught)


def test_alpha_rescues_selection_in_the_under_sampled_regime() -> None:
    # At 80x40 with true k=4 there are only ~5 observations per parameter,
    # and unpenalized fitting collapses onto the smallest k -- a model worse
    # than predicting the global rate. Regularization pulls selection back
    # into the neighbourhood of the true rank.
    #
    # The exact value IS asserted here, and that is a recent change. Before
    # partitions were paired this selected k=6 at 1, 3 and 5 repeats and
    # only reached k=4 at 10 -- so pinning it would have fitted the test to
    # a repeat count. With common random numbers the true k is selected
    # from a single repeat and holds at 1, 2, 3, 5 and 10, which makes the
    # strict assertion meaningful rather than lucky.
    data = _synthetic_binary(80, 40, true_k=4).numpy()

    selected = {}
    for alpha in (0.0, 1.0):
        selector = SiGMoiDSelector(
            [1, 2, 3, 4, 6, 8],
            alpha_range=(alpha,),
            criterion="held_out_ll",
            n_repeats=2,
            max_iter=1500,
            learning_rate=0.02,
            tol=0.0,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            selector.fit(data)
        selected[alpha] = selector.best_n_components_

    assert selected[0.0] < 4, f"unpenalized should severely underselect here, got k={selected[0.0]}"
    assert selected[1.0] >= 4, (
        f"regularized should reach at least the true rank, got k={selected[1.0]}"
    )
    assert selected[1.0] > selected[0.0]


# --- paired partitions (common random numbers) ---------------------------


def test_partitions_are_shared_across_k_and_alpha_within_a_repeat() -> None:
    # The pairing invariant. Every (k, alpha) cell at a given repeat index
    # must see the SAME partition, so that comparing two configurations is
    # not polluted by the accidental difficulty difference between two
    # independently drawn train/test splits.
    data = _synthetic_binary(40, 20, true_k=2).numpy()
    selector = SiGMoiDSelector(
        [2, 3, 4],
        alpha_range=(0.0, 0.5),
        criterion="held_out_ll",
        n_repeats=3,
        max_iter=20,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(data)

    repeats = selector.cv_results_["repeat"]
    partitions = selector.cv_results_["partition_seed"]
    for repeat in set(repeats.tolist()):
        seeds = {p for p, r in zip(partitions, repeats, strict=True) if r == repeat}
        assert len(seeds) == 1, f"repeat {repeat} used {len(seeds)} partitions, expected 1"

    # ...and different repeats must still use different partitions, or the
    # repeats would be measuring nothing.
    assert len(set(partitions.tolist())) == 3


def test_init_seeds_pair_across_alpha_but_differ_across_k() -> None:
    # Initializations are shared across alpha (pairing that axis too) but
    # must differ across k: the parameter shapes differ, so "the same
    # initialization" is not a meaningful notion between two k values.
    data = _synthetic_binary(40, 20, true_k=2).numpy()
    selector = SiGMoiDSelector(
        [2, 3],
        alpha_range=(0.0, 0.5),
        criterion="held_out_ll",
        n_repeats=2,
        max_iter=20,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(data)

    ks = selector.cv_results_["n_components"]
    repeats = selector.cv_results_["repeat"]
    inits = selector.cv_results_["init_seed"]

    for k in (2, 3):
        for repeat in (0, 1):
            seeds = {
                i for i, kk, r in zip(inits, ks, repeats, strict=True) if kk == k and r == repeat
            }
            assert len(seeds) == 1, "init should be shared across alpha"

    k2 = {i for i, kk in zip(inits, ks, strict=True) if kk == 2}
    k3 = {i for i, kk in zip(inits, ks, strict=True) if kk == 3}
    assert not (k2 & k3), "init seeds should differ across k"


def test_separation_detects_a_clean_winner_and_a_genuine_tie() -> None:
    # Tests the decision logic directly on constructed scores rather than
    # hoping a real fit happens to tie. A previous version of this test used
    # a duplicated candidate grid to force a tie; it did not force one (the
    # alpha axis still separated) and instead surfaced a real bug -- see
    # the deduplication in fit().
    selector = SiGMoiDSelector(
        [2, 3], n_repeats=5, alpha_range=(0.0,), criterion="held_out_ll", fdr=0.05
    )

    clean = {(2, 0.0): [1.0] * 6, (3, 0.0): [0.0] * 6}
    resolved, tied, leader, comparisons = selector._separation(clean, _HELD_OUT_LL)
    assert resolved is True
    assert leader == (2, 0.0)
    assert tied == [(2, 0.0)]
    assert comparisons[0]["adjusted_pvalue"] == 0.0

    # Alternating wins: 3 of 6, indistinguishable from a coin flip.
    tie = {(2, 0.0): [1.0, 0.0] * 3, (3, 0.0): [0.0, 1.0] * 3}
    resolved, tied, _, _ = selector._separation(tie, _HELD_OUT_LL)
    assert resolved is False
    assert len(tied) == 2


def test_separation_respects_criterion_direction() -> None:
    # For a minimized criterion the leader is the LOWEST scorer, and wins
    # are counted the other way round.
    selector = SiGMoiDSelector([2, 3], n_repeats=5, alpha_range=(0.0,), criterion="aic", fdr=0.05)
    scores = {(2, 0.0): [100.0] * 6, (3, 0.0): [200.0] * 6}
    resolved, _, leader, _ = selector._separation(scores, _AIC)
    assert resolved is True
    assert leader == (2, 0.0), "aic is minimized, so the lower score should lead"


def test_parsimony_prefers_smallest_k_then_largest_alpha() -> None:
    # Exact ties across a (k, alpha) grid, so nothing can separate and the
    # tie-break rule is exercised deterministically.
    selector = SiGMoiDSelector(
        [2, 4],
        alpha_range=(0.0, 1.0),
        criterion="held_out_ll",
        n_repeats=5,
        max_iter=5,
        random_state=0,
    )
    identical = {cell: [0.5] * 5 for cell in [(2, 0.0), (2, 1.0), (4, 0.0), (4, 1.0)]}
    resolved, tied, _, _ = selector._separation(identical, _HELD_OUT_LL)
    assert resolved is False
    assert len(tied) == 4

    chosen = min(tied, key=lambda cell: (cell[0], -cell[1]))
    assert chosen == (2, 1.0), "smallest k, then largest alpha"


def test_rejects_a_bad_n_repeats_value() -> None:
    selector = SiGMoiDSelector([2], alpha_range=(0.0,), n_repeats="sometimes")
    with pytest.raises(ValueError, match="must be an int"):
        selector.fit(_synthetic_binary(30, 15, true_k=2).numpy())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_components_range": []},
        {"n_components_range": [0]},
        {"alpha_range": []},
        {"alpha_range": [-1.0]},
        {"max_iter": 0},
        {"tol": -1.0},
        {"patience": 0},
        {"patience": True},
        {"patience": 1.5},
        {"learning_rate": 0.0},
        {"gradient": "wrong"},
        {"dtype": torch.int64},
        {"random_state": -1},
        {"fdr": 0.0},
    ],
)
def test_selector_rejects_invalid_hyperparameters(kwargs: dict[str, object]) -> None:
    params: dict[str, object] = {
        "n_components_range": [2],
        "alpha_range": [0.0],
        "n_repeats": 2,
        "max_iter": 1,
    }
    params.update(kwargs)
    selector = SiGMoiDSelector(**params)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        selector.fit(_synthetic_binary(20, 10, true_k=2).numpy())


def test_selector_rejects_an_unknown_test() -> None:
    selector = SiGMoiDSelector(
        [2, 3], alpha_range=(0.0,), test="bogus", n_repeats=2, max_iter=10, random_state=0
    )
    with pytest.raises(ValueError, match="test must be"):
        selector.fit(_synthetic_binary(20, 12, true_k=2).numpy())


def test_default_alpha_range_resolves_by_criterion() -> None:
    # None means "choose for me", and the choice cannot be one constant.
    # Held-out LL gets the measured grid; AIC/BIC must get the unpenalized
    # fit only, because param_count assumes unpenalized maximum likelihood
    # -- a penalized grid would make the DEFAULT path for those criteria
    # one the docs describe as invalid, and would warn on every run.
    selector = SiGMoiDSelector([2, 3], n_repeats=5)
    assert selector.alpha_range is None
    assert selector._resolve_alpha_range(_HELD_OUT_LL) == DEFAULT_ALPHA_RANGE
    assert selector._resolve_alpha_range(_AIC) == (0.0,)
    assert selector._resolve_alpha_range(_BIC) == (0.0,)

    # An explicit value is honoured for every criterion, including the
    # combination that warns.
    explicit = SiGMoiDSelector([2, 3], n_repeats=5, alpha_range=(0.0, 0.5))
    for criterion in (_HELD_OUT_LL, _AIC, _BIC):
        assert explicit._resolve_alpha_range(criterion) == (0.0, 0.5)


def test_default_alpha_range_brackets_the_measured_useful_window() -> None:
    # Both bounds are empirical, so pin them. Below ~0.03 the penalty is
    # indistinguishable from alpha=0; above ~3 the fit collapses to
    # predicting 0.5 everywhere (-ln 2 per entry, max|logit| = 0) and the
    # selected k becomes tie-breaking noise.
    assert DEFAULT_ALPHA_RANGE[0] == 0.0
    positive = DEFAULT_ALPHA_RANGE[1:]
    assert min(positive) == pytest.approx(0.03)
    assert max(positive) == pytest.approx(3.0)
    # Roughly sqrt(10) spacing: consecutive ratios all within [2.5, 4].
    ratios = [b / a for a, b in pairwise(positive)]
    assert all(2.5 <= r <= 4.0 for r in ratios), ratios


def test_default_alpha_range_does_not_warn_for_information_criteria() -> None:
    # Regression guard for the interaction the sentinel exists to prevent:
    # a penalized default would emit the incomparability warning on every
    # AIC/BIC run, training users to ignore a warning that matters.
    data = _synthetic_binary(30, 16, true_k=2).numpy()
    selector = SiGMoiDSelector([2, 3], criterion="aic", n_repeats=2, max_iter=20, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="did not converge"):
            selector.fit(data)  # the ONLY warning raised is the convergence one
