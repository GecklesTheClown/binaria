"""
Surviving interruption.

A sweep on real data is an hour or more, and a queue can kill it at
minute fifty-nine. The requirement is not merely that a resumed run
finishes -- it is that it produces *the same answer* as an uninterrupted
one. Anything weaker means the result depends on whether the job happened
to be preempted.

That property is achievable here for a specific reason: a fit is a pure
function of ``(data, k, alpha, init_seed, partition_seed)``, and all of
those are pinned before the loop starts. Resuming reuses finished fits
rather than re-deriving them, so the only way it can go wrong is if a
setting changed underneath -- which is what the fingerprint check exists
to catch.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from binaria import SiGMoiDSelector


def _binary(n_samples: int = 40, n_features: int = 24, true_k: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    beta = torch.randn(n_samples, true_k, dtype=torch.float64)
    energy = torch.randn(true_k, n_features, dtype=torch.float64)
    return torch.bernoulli(torch.sigmoid(-(beta @ energy))).numpy()


def _run(data, **kwargs: object) -> SiGMoiDSelector:
    defaults: dict[str, object] = {
        "alpha_range": (0.0, 0.5),
        "n_repeats": 3,
        "max_iter": 40,
        "random_state": 0,
    }
    defaults.update(kwargs)
    selector = SiGMoiDSelector([2, 3, 4], **defaults)  # type: ignore[arg-type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return selector.fit(data)


def test_checkpoint_is_not_written_unless_asked(tmp_path: Path) -> None:
    _run(_binary())
    assert list(tmp_path.iterdir()) == []


def test_a_resumed_sweep_is_bit_identical_to_an_uninterrupted_one(tmp_path: Path) -> None:
    # The whole point. Not "close" -- identical, in every column.
    data = _binary()
    path = tmp_path / "sweep.pt"

    uninterrupted = _run(data)
    first = _run(data, checkpoint=path)  # writes the checkpoint
    resumed = _run(data, checkpoint=path)  # reuses every fit in it

    for column in uninterrupted.cv_results_:
        expected = uninterrupted.cv_results_[column]
        if column == "fit_time":
            assert np.array_equal(first.cv_results_[column], resumed.cv_results_[column])
            assert bool((resumed.cv_results_[column] > 0.0).all())
            continue
        for other in (first.cv_results_[column], resumed.cv_results_[column]):
            if expected.dtype.kind == "f":
                assert np.array_equal(expected, other), column
            else:
                assert list(expected) == list(other), column

    assert resumed.best_n_components_ == uninterrupted.best_n_components_
    assert resumed.best_alpha_ == uninterrupted.best_alpha_
    assert resumed.best_score_ == uninterrupted.best_score_
    assert resumed.selection_rule_ == uninterrupted.selection_rule_


def test_resuming_actually_skips_the_finished_fits(tmp_path: Path) -> None:
    # Guards against a checkpoint that is written, read, and then quietly
    # ignored -- which would pass the equality test above while delivering
    # none of the benefit.
    data = _binary()
    path = tmp_path / "sweep.pt"
    _run(data, checkpoint=path)

    state = torch.load(path, weights_only=False)
    assert len(state["rows"]) == 3 * 2 * 3  # k grid x alphas x repeats

    calls = 0
    import binaria.selection as selection

    original = selection._execute_fit_job

    def counting(job: object) -> object:
        nonlocal calls
        calls += 1
        return original(job)  # type: ignore[arg-type]

    selection._execute_fit_job = counting  # type: ignore[assignment]
    try:
        _run(data, checkpoint=path)
    finally:
        selection._execute_fit_job = original  # type: ignore[assignment]

    assert calls == 0, "a fully checkpointed sweep should refit nothing"


def test_a_partial_checkpoint_completes_the_rest(tmp_path: Path) -> None:
    # The realistic case: killed partway, so some fits exist and some do
    # not. Simulated by truncating a complete checkpoint.
    data = _binary()
    path = tmp_path / "sweep.pt"
    _run(data, checkpoint=path)

    state = torch.load(path, weights_only=False)
    keep = state["rows"][: len(state["rows"]) // 2]
    torch.save({"fingerprint": state["fingerprint"], "rows": keep}, path)

    resumed = _run(data, checkpoint=path)
    uninterrupted = _run(data)
    assert np.array_equal(resumed.cv_results_["score"], uninterrupted.cv_results_["score"])


@pytest.mark.parametrize(
    "changed",
    [
        {"random_state": 1},
        {"max_iter": 80},
        {"patience": 5},
        {"alpha_range": (0.0, 0.25)},
        {"learning_rate": 0.02},
        {"optimizer": "sgd", "learning_rate": 1e-4},
        {"fdr": 0.1},
    ],
)
def test_resuming_across_a_changed_setting_refuses_loudly(changed: dict, tmp_path: Path) -> None:
    # Silently resuming here would splice two different experiments into
    # one cv_results_. Refusing is also better than ignoring the
    # checkpoint and refitting, which discards hours of compute just as
    # silently.
    data = _binary()
    path = tmp_path / "sweep.pt"
    _run(data, checkpoint=path)

    with pytest.raises(ValueError, match="different sweep"):
        _run(data, checkpoint=path, **changed)


def test_resuming_against_differently_shaped_data_refuses(tmp_path: Path) -> None:
    path = tmp_path / "sweep.pt"
    _run(_binary(40, 24), checkpoint=path)
    with pytest.raises(ValueError, match="different sweep"):
        _run(_binary(30, 24), checkpoint=path)


def test_resuming_with_a_different_statistical_test_refuses(tmp_path: Path) -> None:
    data = _binary()
    path = tmp_path / "sweep.pt"
    _run(data, checkpoint=path, n_repeats=5, test="t")
    with pytest.raises(ValueError, match="different sweep"):
        _run(data, checkpoint=path, n_repeats=5, test="sign")


def test_resuming_against_different_data_of_the_same_shape_refuses(tmp_path: Path) -> None:
    path = tmp_path / "sweep.pt"
    _run(_binary(seed=0), checkpoint=path)
    with pytest.raises(ValueError, match="different sweep"):
        _run(_binary(seed=1), checkpoint=path)


def test_checkpoint_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    # Written through .tmp then os.replace, so a kill mid-write cannot
    # destroy a good checkpoint -- which would be a particularly unkind
    # failure for the feature whose entire purpose is surviving kills.
    path = tmp_path / "sweep.pt"
    _run(_binary(), checkpoint=path)
    assert [p.name for p in tmp_path.iterdir()] == ["sweep.pt"]


def test_checkpoint_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "today" / "sweep.pt"
    _run(_binary(), checkpoint=path)
    assert path.exists()
