import importlib.metadata
import os
from dataclasses import asdict
from pathlib import Path

import torch

from binaria._core import Core
from binaria.callbacks import History
from binaria.estimator import SiGMoiD
from binaria.validation import to_output


def save(estimator: SiGMoiD, path: str | Path) -> None:
    """
    Save a fitted SiGMoiD estimator to disk.

    Round-trips bit-identically via `load`: the exact `beta`/`energy`
    tensors are saved, not re-derived, so there's no reconstruction
    precision to lose.

    Parameters
    ----------
    estimator : SiGMoiD
        A fitted estimator (i.e. `fit` has been called).
    path : str or Path
        Destination file.
    """
    estimator._check_is_fitted()

    state = {
        "metadata": {
            "package_version": importlib.metadata.version("binaria"),
            "dtype": str(estimator._core.beta.dtype),
            "n_components": estimator.n_components,
            "random_state": estimator.random_state,
        },
        "params": estimator.get_params(),
        "model_state_dict": estimator._core.state_dict(),
        "n_iter_": estimator.n_iter_,
        "fit_time_": estimator.fit_time_,
        "converged_": estimator.converged_,
        "log_likelihood_": estimator.log_likelihood_,
        "history_": asdict(estimator.history_),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same atomic-write pattern as callbacks.Checkpoint, for the same
    # reason: a process killed mid-write must not corrupt a previously-good
    # save file.
    tmp_path = path.parent / (path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load(path: str | Path, *, map_location: str | torch.device | None = "cpu") -> SiGMoiD:
    """
    Load a SiGMoiD estimator previously saved with `save`.

    Parameters
    ----------
    path : str or Path
    map_location : str, torch.device, or None, default="cpu"
        Device for restored tensors. The CPU default makes models saved on
        CUDA portable to machines without a GPU. Pass ``None`` to preserve
        the saved tensor devices.

    Returns
    -------
    estimator : SiGMoiD
        A fitted estimator, equivalent to the one that was saved.
    """
    # weights_only=False: this reads back a file this package wrote itself,
    # not an untrusted one -- same reasoning as callbacks.load_checkpoint.
    state = torch.load(path, weights_only=False, map_location=map_location)

    params = dict(state["params"])
    # Before patience existed, one qualifying relative change stopped a fit.
    # Preserve that behaviour when an old save is re-fit or used by score().
    params.setdefault("patience", 1)
    estimator = SiGMoiD(**params)

    beta = state["model_state_dict"]["beta"]
    energy = state["model_state_dict"]["energy"]
    core = Core(
        n_samples=beta.shape[0],
        n_features=energy.shape[1],
        n_components=beta.shape[1],
        dtype=beta.dtype,
        beta=beta.clone(),
        energy=energy.clone(),
    )

    estimator._core = core
    estimator.components_ = to_output(core.energy, output="numpy")
    estimator.embedding_ = to_output(core.beta, output="numpy")
    estimator.n_iter_ = state["n_iter_"]
    estimator.fit_time_ = state["fit_time_"]
    estimator.converged_ = state["converged_"]
    estimator.log_likelihood_ = state["log_likelihood_"]
    estimator.history_ = History(**state.get("history_", {}))
    return estimator
