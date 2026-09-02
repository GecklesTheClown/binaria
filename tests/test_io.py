"""
Persistence: does a fit survive a round trip to disk?

This matters more than it looks. A selection sweep on real data is an
hour of compute, and ``save``/``load`` is the only thing standing between
that and having to run it again. The round trip must be *exact*, not
approximate -- a "nearly identical" reload silently changes every
downstream number.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from binaria import SiGMoiD, load, save


def _binary(n_samples: int = 40, n_features: int = 20, true_k: int = 3, seed: int = 0):
    torch.manual_seed(seed)
    beta = torch.randn(n_samples, true_k, dtype=torch.float64)
    energy = torch.randn(true_k, n_features, dtype=torch.float64)
    return torch.bernoulli(torch.sigmoid(-(beta @ energy))).numpy()


def _fitted(**kwargs: object) -> SiGMoiD:
    defaults: dict[str, object] = {"n_components": 3, "max_iter": 60, "random_state": 0}
    defaults.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return SiGMoiD(**defaults).fit(_binary())  # type: ignore[arg-type]


def test_round_trip_reproduces_predictions_exactly(tmp_path: Path) -> None:
    # The load-bearing assertion. Predictions are the gauge-invariant
    # quantity -- beta and E are only defined up to GL(K) -- so this is
    # what "the same model" actually means.
    model = _fitted()
    data = _binary()
    path = tmp_path / "model.pt"
    save(model, path)
    reloaded = load(path)

    assert np.array_equal(model.transform(data), reloaded.transform(data))
    assert model.score(data) == reloaded.score(data)


def test_round_trip_preserves_parameters_bitwise(tmp_path: Path) -> None:
    # The tensors are saved, not re-derived, so there is no reconstruction
    # error to tolerate. Anything less than bitwise equality means a
    # dtype or device conversion crept into the path.
    model = _fitted()
    path = tmp_path / "model.pt"
    save(model, path)
    reloaded = load(path)

    assert np.array_equal(model.components_, reloaded.components_)
    assert np.array_equal(model.embedding_, reloaded.embedding_)


def test_round_trip_preserves_fit_metadata(tmp_path: Path) -> None:
    model = _fitted(alpha=0.5, optimizer="sgd", learning_rate=1e-4, patience=7)
    path = tmp_path / "model.pt"
    save(model, path)
    reloaded = load(path)

    assert reloaded.n_iter_ == model.n_iter_
    assert reloaded.fit_time_ == model.fit_time_
    assert reloaded.converged_ == model.converged_
    assert reloaded.log_likelihood_ == model.log_likelihood_
    assert reloaded.history_ == model.history_
    # Hyperparameters travel too, so a reloaded estimator can be re-fit or
    # cloned into a new sweep without the caller re-specifying them.
    assert reloaded.get_params() == model.get_params()
    assert reloaded.optimizer == "sgd"
    assert reloaded.patience == 7


def test_loading_a_legacy_save_preserves_the_old_one_hit_rule(tmp_path: Path) -> None:
    model = _fitted()
    path = tmp_path / "legacy.pt"
    save(model, path)
    state = torch.load(path, weights_only=False)
    del state["params"]["patience"]
    torch.save(state, path)

    reloaded = load(path)

    assert reloaded.patience == 1


def test_saving_an_unfitted_estimator_raises(tmp_path: Path) -> None:
    with pytest.raises(Exception, match=r"(?i)fit"):
        save(SiGMoiD(n_components=2), tmp_path / "nope.pt")


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "model.pt"
    save(_fitted(), path)
    assert path.exists()


def test_save_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    # The write is staged through a .tmp then os.replace'd, so a process
    # killed mid-write cannot corrupt a previously-good save. The staging
    # file must not survive a successful write.
    path = tmp_path / "model.pt"
    save(_fitted(), path)
    assert [p.name for p in tmp_path.iterdir()] == ["model.pt"]


def test_overwriting_an_existing_save_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    save(_fitted(n_components=2), path)
    save(_fitted(n_components=4), path)

    reloaded = load(path)
    assert reloaded.components_.shape[0] == 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_round_trip_preserves_dtype(dtype: torch.dtype, tmp_path: Path) -> None:
    # float32 is offered as an option, so it has to survive persistence --
    # a reload that silently promoted to float64 would change every
    # subsequent number while looking like it worked.
    model = _fitted(dtype=dtype)
    path = tmp_path / "model.pt"
    save(model, path)
    reloaded = load(path)
    assert reloaded._core.beta.dtype == dtype
    assert reloaded._core.energy.dtype == dtype


def test_load_can_map_saved_tensors_to_cpu(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    save(_fitted(), path)
    reloaded = load(path, map_location="cpu")
    assert reloaded._core.beta.device.type == "cpu"
    assert reloaded._core.energy.device.type == "cpu"
