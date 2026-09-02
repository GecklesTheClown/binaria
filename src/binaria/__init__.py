"""
Binaria — a Python/PyTorch implementation of SiGMoiD for binary data.

SiGMoiD (Zhao, Plata & Dixit, 2021, PLOS Comp Bio,
doi:10.1371/journal.pcbi.1009275) factorises a binary matrix into a
per-sample latent factor and a set of shared features through a logistic
link: ``pi = sigmoid(-(beta @ energy))``.

Quick start
-----------
::

    from binaria import SiGMoiD, SiGMoiDSelector

    model = SiGMoiD(n_components=4, alpha=0.3).fit(x)
    model.components_     # (k, n_features) shared features
    model.embedding_      # (n_samples, k) per-sample factor

    # Choosing k
    selector = SiGMoiDSelector([2, 3, 4, 6, 8]).fit(x)
    selector.best_n_components_
    selector.selection_rule_   # "separation" | "parsimony"

The numerical defaults and selection behavior have not yet been evaluated
experimentally. Inspect ``selection_rule_`` and the convergence values in
``cv_results_`` when interpreting a selected k.

What is exported
----------------
The names below are the supported surface. Everything else, including
every ``_``-prefixed module, is internal and may change without notice.
"""

from binaria.callbacks import Callback, Checkpoint, History, IterationState
from binaria.canonical import CanonicalFactors, canonicalize, subspace_distance
from binaria.estimator import SiGMoiD
from binaria.executors import Executor, MultiGPUExecutor, SerialExecutor
from binaria.io import load, save
from binaria.selection import (
    DEFAULT_ALPHA_RANGE,
    SiGMoiDSelector,
    aic,
    bic,
    make_block_mask,
    param_count,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_ALPHA_RANGE",
    "Callback",
    "CanonicalFactors",
    "Checkpoint",
    "Executor",
    "History",
    "IterationState",
    "MultiGPUExecutor",
    "SerialExecutor",
    "SiGMoiD",
    "SiGMoiDSelector",
    "__version__",
    "aic",
    "bic",
    "canonicalize",
    "load",
    "make_block_mask",
    "param_count",
    "save",
    "subspace_distance",
]
