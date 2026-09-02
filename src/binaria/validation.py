from typing import Literal

import numpy as np
import torch


def _is_pandas_dataframe(x: object) -> bool:
    # Lazy import: pandas stays optional. If it's not installed, nothing the
    # caller passed could be a DataFrame anyway, so returning False is
    # correct, not a fallback.
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(x, pd.DataFrame)


def _is_scipy_sparse(x: object) -> bool:
    try:
        import scipy.sparse as sp
    except ImportError:
        return False
    return bool(sp.issparse(x))


def validate_binary_matrix(
    x: object,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    # Coercion order: torch tensors pass through as-is; pandas and
    # scipy.sparse get their own branch (checked before the numpy fallback,
    # since np.asarray on a sparse matrix silently produces a 0-d object
    # array instead of raising); everything else goes through np.asarray,
    # which also accepts plain nested lists.
    if isinstance(x, torch.Tensor):
        tensor = x
    elif _is_pandas_dataframe(x):
        tensor = torch.from_numpy(x.to_numpy())  # type: ignore[attr-defined]
    elif _is_scipy_sparse(x):
        # Accepted and densified: the model is dense in beta@energy
        # regardless, so exploiting sparsity is a research project, not a
        # v1 feature.
        tensor = torch.from_numpy(x.toarray())  # type: ignore[attr-defined]
    else:
        tensor = torch.from_numpy(np.asarray(x))

    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a 2D (samples, features) matrix, got shape {tuple(tensor.shape)}"
        )

    # Matches sklearn's own check_array wording exactly (confirmed via
    # check_estimator's expected pattern for this message), not an
    # independently-invented phrasing -- picks whichever dimension is 0 for
    # the message, matching how sklearn itself would report it.
    if tensor.shape[0] == 0:
        raise ValueError(
            f"Found array with {tensor.shape[0]} sample(s) (shape={tuple(tensor.shape)}) "
            "while a minimum of 1 is required."
        )
    if tensor.shape[1] == 0:
        raise ValueError(
            f"Found array with {tensor.shape[1]} feature(s) (shape={tuple(tensor.shape)}) "
            "while a minimum of 1 is required."
        )

    tensor = tensor.to(dtype=dtype)

    # Checked explicitly, with their own messages, rather than left to the
    # binary-value check below to reject implicitly: NaN/inf are a
    # different failure mode from "wrong but finite value", worth a
    # message that says so directly.
    if torch.isnan(tensor).any():
        raise ValueError("Input contains NaN.")
    if torch.isinf(tensor).any():
        raise ValueError("Input contains infinity.")
    # Checked ahead of the general binary-value check, with sklearn's own
    # conventional phrasing ("Negative values in data"): declaring the
    # positive_only tag means sklearn's own tooling specifically probes for
    # this exact message on negative input, not just any rejection.
    if bool((tensor < 0).any()):
        raise ValueError(
            "Negative values in data are not allowed. X must be a binary (0/1) matrix."
        )

    unique_values = torch.unique(tensor)
    if not torch.all((unique_values == 0) | (unique_values == 1)):
        raise ValueError(
            f"Expected a binary (0/1) matrix, got values outside {{0, 1}}: {unique_values.tolist()}"
        )

    if device is not None:
        tensor = tensor.to(device=device)

    return tensor


def to_output(
    tensor: torch.Tensor, *, output: Literal["numpy", "torch"] = "numpy"
) -> np.ndarray | torch.Tensor:
    # output="numpy" forces .cpu() -- mandatory, numpy can't represent GPU
    # memory. output="torch" leaves the tensor on whatever device it's
    # already on: a CUDA tensor only exists here at all if the caller
    # explicitly set device="cuda" further up, so it was asked for, not
    # returned unasked.
    if output == "numpy":
        return tensor.detach().cpu().numpy()
    if output == "torch":
        return tensor.detach()
    raise ValueError(f"Unknown output format: {output!r}")
