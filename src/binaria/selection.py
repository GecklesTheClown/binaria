"""
Model selection: choosing the number of latent components (k).

Two families of criterion live here, and they are not interchangeable.
AIC/BIC are pure post-hoc functions of one full-data fit. Held-out
log-likelihood is not -- it requires a *masked* fit plus a second evaluation
on the complement, i.e. it changes how the fit runs, not just how the result
is scored. The public interface therefore accepts only the three implemented
criterion names; the fit strategy remains an internal detail.

"""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.stats import binomtest, false_discovery_control, ttest_1samp
from sklearn.base import BaseEstimator

from binaria._core import Core
from binaria._optim import GradientPath, OptimizerName, StoppingRule
from binaria._optim import fit as _fit_core
from binaria.executors import Executor, SerialExecutor, current_device
from binaria.validation import validate_binary_matrix


def make_block_mask(
    n_samples: int,
    n_features: int,
    *,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a 2x2 block train/test partition of a binary matrix.

    Rows are split into 2 groups and columns into 2 groups, forming a 2x2
    grid of blocks; one block is held out as the test set and the other
    three are visible for fitting.

    A held-out block excludes entries in both directions at once. Its rows had their
    ``beta_s`` fit only from the other column group, and its columns had
    their ``energy_i`` fit only from the other row group.

    Entry-level (rather than whole-row) holdout is what makes this work at
    all: every sample keeps visible entries, so ``beta_s`` is still fit
    normally for every row as part of the joint optimization, and no
    out-of-sample ``transform``-style inference is needed.

    Parameters
    ----------
    n_samples, n_features : int
        Shape of the matrix to partition. Both must be >= 2 -- a 2x2 block
        partition is undefined otherwise.
    seed : int or None, default=None
        Seed for the row/column group assignment. The same seed always
        produces the same partition.
    dtype : torch.dtype, default=torch.float64
        Should match the dtype of the data being masked.
    device : str, torch.device, or None, default=None

    Returns
    -------
    train_mask, test_mask : Tensor of shape (n_samples, n_features)
        Complementary 0/1 masks; ``train_mask + test_mask`` is all ones.
        With even splits the test block is roughly a quarter of the matrix.
    """
    if n_samples < 2 or n_features < 2:
        raise ValueError(
            f"A 2x2 block partition needs at least 2 samples and 2 features, "
            f"got {n_samples} and {n_features}"
        )

    rng = np.random.default_rng(seed)
    # Always holding out the *first* block is not a loss of generality:
    # the row/column group assignment is already a fresh random
    # permutation, so block (0, 0) is a uniformly random block. Choosing
    # among the four afterwards would just be re-randomising the same
    # thing.
    held_rows = torch.as_tensor(rng.permutation(n_samples)[: n_samples // 2].copy())
    held_cols = torch.as_tensor(rng.permutation(n_features)[: n_features // 2].copy())

    test_mask = torch.zeros(n_samples, n_features, dtype=dtype, device=device)
    test_mask[held_rows[:, None], held_cols[None, :]] = 1.0
    train_mask = 1.0 - test_mask
    return train_mask, test_mask


def param_count(n_samples: int, n_components: int, n_features: int) -> int:
    """
    Number of free parameters: ``s*k + k*i``.

    This is the count used in the paper: every entry of ``beta`` (s x k)
    plus every entry of ``energy`` (k x i). It does not adjust for the
    ``GL(K)`` gauge freedom under which multiple factor pairs produce the
    same predictions.
    """
    return n_samples * n_components + n_components * n_features


def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike information criterion, ``2p - 2*LL``. Lower is better."""
    return 2.0 * n_params - 2.0 * log_likelihood


def bic(log_likelihood: float, n_params: int, n_observations: int) -> float:
    """
    Bayesian information criterion, ``p*ln(n) - 2*LL``. Lower is better.

    ``n_observations`` is the number of *samples* (matrix rows), a
    deliberate choice rather than a default: for a matrix factorization
    there is a live argument for using the total number of binary entries
    (``s*i``) instead, which would penalise complexity far more heavily.
    Callers pass this explicitly so the choice stays visible.
    """
    return n_params * math.log(n_observations) - 2.0 * log_likelihood


_CriterionName = Literal["held_out_ll", "aic", "bic"]


@dataclass(frozen=True)
class _Criterion:
    """
    A model-selection criterion, carrying its own direction and fit needs.

    ``greater_is_better`` is an attribute of the criterion rather than a
    lookup keyed by name elsewhere: a separate registry can be forgotten
    when a criterion is added, and a criterion scored in the wrong
    direction fails silently by selecting the *worst* model.

    ``requires_masking`` exists because AIC/BIC and held-out log-likelihood
    are not the same kind of object. AIC/BIC are pure post-hoc functions of
    one full-data fit. Held-out log-likelihood needs a *masked* fit plus a
    second evaluation on the complement -- it changes how the fit runs, not
    just how the result is scored.
    """

    name: str
    greater_is_better: bool
    requires_masking: bool


_AIC = _Criterion(name="aic", greater_is_better=False, requires_masking=False)
_BIC = _Criterion(name="bic", greater_is_better=False, requires_masking=False)
_HELD_OUT_LL = _Criterion(name="held_out_ll", greater_is_better=True, requires_masking=True)

_CRITERIA = {c.name: c for c in (_AIC, _BIC, _HELD_OUT_LL)}

# Provisional grid spanning the unpenalized and penalized objectives.
DEFAULT_ALPHA_RANGE = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)

_RESULT_COLUMNS = (
    "n_components",
    "alpha",
    "criterion",
    "repeat",
    "init_seed",
    "partition_seed",
    "score",
    "greater_is_better",
    "log_likelihood",
    "n_params",
    "converged",
    "n_iter",
    "fit_time",
)


def _best_index(scores: Sequence[float], *, greater_is_better: bool) -> int:
    # The one place direction is applied. Callers go through here rather
    # than re-deriving argmin/argmax, so a criterion is not ranked
    # backwards at one call site while being right at another.
    values = np.asarray(scores, dtype=float)
    return int(np.argmax(values) if greater_is_better else np.argmin(values))


@dataclass(frozen=True)
class _FitJob:
    """
    Everything one fit needs, in a form that survives pickling.

    ``data`` is deliberately loose. In-process it is the tensor itself,
    which costs nothing to pass. Across processes it is a *path*: the
    matrix is written once and each worker loads and caches it, instead of
    being pickled and shipped with all several hundred jobs.

    The estimator is not carried. Its relevant settings are copied in as
    plain fields, so a worker never has to reconstruct a
    ``SiGMoiDSelector`` and the job stays cheap to serialise.
    """

    data: object
    criterion: _Criterion
    n_components: int
    alpha: float
    repeat: int
    init_seed: int
    partition_seed: int | None
    device: torch.device | None
    dtype: torch.dtype
    max_iter: int
    tol: float
    patience: int
    stopping_rule: StoppingRule
    learning_rate: float
    optimizer: OptimizerName
    gradient: GradientPath


# Worker-local, keyed by (path, device). Workers are long-lived and take
# many jobs, so without this the matrix would be re-read from disk for
# every fit.
_DATA_CACHE: dict[tuple[str, str, str], torch.Tensor] = {}


def _resolve_data(
    handle: object, *, dtype: torch.dtype, device: torch.device | None
) -> torch.Tensor:
    if isinstance(handle, torch.Tensor):
        return handle if device is None else handle.to(device)
    key = (str(handle), str(device), str(dtype))
    if key not in _DATA_CACHE:
        # weights_only=True: this file was written by this package moments
        # ago, but the flag costs nothing and keeps the habit right.
        loaded = torch.load(str(handle), weights_only=True)
        _DATA_CACHE[key] = loaded.to(dtype=dtype, device=device)
    return _DATA_CACHE[key]


def _data_digest(data: torch.Tensor) -> str:
    values = data.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _job_key(job: _FitJob) -> tuple[int, float, int]:
    """Identity of a fit: which cell, which repeat. Nothing else varies."""
    return (job.n_components, job.alpha, job.repeat)


def _write_checkpoint(
    path: Path,
    fingerprint: dict[str, object],
    completed: dict[tuple[int, float, int], dict[str, object]],
) -> None:
    """
    Persist finished fits so an interrupted sweep can be resumed.

    Written through a temporary file and then ``os.replace``, for the same
    reason ``io.save`` does: a process killed mid-write must not leave a
    corrupt checkpoint where a good one used to be. That failure would be
    especially cruel here, since the whole point is surviving being killed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.name + ".tmp")
    torch.save(
        {"fingerprint": fingerprint, "rows": list(completed.items())},
        tmp_path,
    )
    os.replace(tmp_path, path)


def _load_checkpoint(
    path: Path, fingerprint: dict[str, object]
) -> dict[tuple[int, float, int], dict[str, object]]:
    """
    Read a checkpoint, refusing it if the sweep it belongs to has changed.

    The fingerprint check is the load-bearing part. Resuming is only sound
    because a fit is a pure function of its inputs, so *every* input that
    could change a result has to match: the grid, the criterion, the
    seeding, the optimizer settings, and the data's shape and dtype.
    Silently resuming across a changed setting would splice two different
    experiments together and produce a ``cv_results_`` that never existed.

    Refusing loudly is the right failure. The alternative -- ignoring a
    stale checkpoint and refitting -- discards hours of compute just as
    silently.
    """
    # weights_only=False: this reads a file this package wrote itself.
    state = torch.load(path, weights_only=False)
    stored = state["fingerprint"]
    if stored != fingerprint:
        differing = sorted(
            key for key in set(stored) | set(fingerprint) if stored.get(key) != fingerprint.get(key)
        )
        raise ValueError(
            f"Checkpoint at {path} was written for a different sweep; "
            f"these settings differ: {differing}. Resuming would splice two "
            f"experiments together. Delete the checkpoint to start over, or "
            f"restore the original settings to continue."
        )
    return dict(state["rows"])


def _execute_fit_job(job: _FitJob) -> dict[str, object]:
    """
    Run one fit and return its ``cv_results_`` row.

    Module-level and self-contained because it has to survive being
    pickled and sent to a worker process: a bound method would drag the
    whole estimator across, and a closure could not be pickled at all.
    Everything the fit needs travels in the job.

    The device is *asked for* rather than dictated. In a worker,
    ``current_device()`` returns the card that worker pinned; in-process
    it returns ``None`` and the selector's own ``device`` applies. That
    keeps one code path for both, so the serial and parallel executors
    cannot drift apart -- which is what makes their equivalence testable
    rather than merely hoped for.
    """
    pinned = current_device()
    device = torch.device(pinned) if pinned is not None else job.device
    data = _resolve_data(job.data, dtype=job.dtype, device=device)
    fit_device = data.device
    criterion = job.criterion
    n_components, alpha = job.n_components, job.alpha

    n_samples, n_features = data.shape
    train_mask: torch.Tensor | None = None
    test_mask: torch.Tensor | None = None
    if criterion.requires_masking:
        assert job.partition_seed is not None
        train_mask, test_mask = make_block_mask(
            n_samples,
            n_features,
            seed=job.partition_seed,
            dtype=job.dtype,
            device=fit_device,
        )

    generator = torch.Generator(device=fit_device).manual_seed(job.init_seed)
    core = Core(
        n_samples=n_samples,
        n_features=n_features,
        n_components=n_components,
        dtype=job.dtype,
        device=fit_device,
        generator=generator,
    )

    result = _fit_core(
        core,
        data,
        mask=train_mask,
        max_iter=job.max_iter,
        tol=job.tol,
        patience=job.patience,
        stopping_rule=job.stopping_rule,
        lr=job.learning_rate,
        optimizer=job.optimizer,
        gradient=job.gradient,
        alpha=alpha,
    )

    with torch.no_grad():
        if criterion.requires_masking:
            # Fit LL is on the training mask; the score is the
            # complement. make_block_mask holds out exactly
            # (S//2)*(N//2) entries for a given shape, so this sum is
            # comparable across repeats without normalising.
            log_likelihood = core.log_likelihood(data, mask=train_mask).item()
            score = core.log_likelihood(data, mask=test_mask).item()
            n_params: int | None = None
        else:
            log_likelihood = core.log_likelihood(data).item()
            n_params = param_count(n_samples, n_components, n_features)
            score = (
                aic(log_likelihood, n_params)
                if criterion.name == "aic"
                else bic(log_likelihood, n_params, n_samples)
            )

    return {
        "n_components": n_components,
        "alpha": alpha,
        "criterion": criterion.name,
        "repeat": job.repeat,
        "init_seed": job.init_seed,
        "partition_seed": job.partition_seed,
        "score": score,
        "greater_is_better": criterion.greater_is_better,
        "log_likelihood": log_likelihood,
        "n_params": n_params,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "fit_time": result.fit_time,
    }


class SiGMoiDSelector(BaseEstimator):  # type: ignore[misc]
    """
    Select the number of latent components (k) for SiGMoiD.

    Fits a grid of candidate ``k`` values, repeats each cell, and ranks the
    cells by the chosen criterion. Per-fit results are stored in
    ``cv_results_`` and aggregates in ``summary_``.

    Parameters
    ----------
    n_components_range : sequence of int
        Candidate values of k to evaluate.
    criterion : {"held_out_ll", "aic", "bic"}, default="held_out_ll"
        Criterion used to rank candidates. Held-out log-likelihood is the
        default. AIC and BIC use ``param_count``.
    alpha_range : sequence of float or None, default=None
        L2 regularization strengths to sweep jointly with ``k``.

        ``None`` resolves to ``DEFAULT_ALPHA_RANGE`` for held-out
        likelihood and ``(0.0,)`` for AIC or BIC. An explicit sequence is
        always used as given. The default grid is provisional and has not
        been established empirically.
    n_repeats : int, default=20
        Repeats per cell. The comparison is performed once, after all
        repeats finish.

        Must be at least the floor for the chosen ``test``: 2 for ``"t"``
        (enough for a variance estimate) and, for ``"sign"``, enough that
        ``0.5**n_repeats`` clears ``fdr``. Below that no comparison could
        reach significance regardless of the data, so ``fit`` raises rather
        than running a sweep that cannot conclude anything.

        For ``held_out_ll`` each repeat draws a fresh 2x2 partition (shared
        across all cells) *and* fresh initializations; for AIC/BIC there is
        no partition, so repeats vary only the initialization.
    test : {"t", "sign"}, default="t"
        How the leader is compared with its rivals. Both use paired scores
        from shared partitions.

        - ``"t"``: one-sided paired t-test, Benjamini-Hochberg corrected
          across rivals. Assumes the paired differences are roughly normal.
        - ``"sign"``: one-sided sign test.
    fdr : float, default=0.05
        Target false discovery rate for the Benjamini-Hochberg correction
        applied across the comparison set.
    max_iter : int, default=6000
        Maximum optimizer iterations for each fit. Check
        ``cv_results_["converged"]`` when interpreting the result.
    tol, patience, stopping_rule, learning_rate, optimizer, gradient, dtype, device
        Passed through to each individual fit.

        ``optimizer`` accepts ``"adam"`` (the default) or ``"sgd"``.
        SGD is full-batch plain gradient ascent with zero momentum, not
        minibatch training. Set its ``learning_rate`` explicitly; the 1e-4
        shown in the documentation is illustrative rather than a validated
        or universally safe recommendation.
    executor : Executor or None, default=None
        How to run the fits. ``None`` runs them in this process, one at a
        time. Pass ``MultiGPUExecutor()`` to spread them across devices.
    checkpoint : str, Path or None, default=None
        Where to record finished fits so an interrupted sweep can resume.
        ``None`` writes nothing.

        Resuming reuses completed fits. The checkpoint includes a
        fingerprint of the grid, criterion, seeds, optimizer settings, and
        data metadata; a mismatch raises instead of resuming.

        Written through a temporary file and ``os.replace``, so a process
        killed mid-write cannot destroy a good checkpoint.
    audit : bool, default=False
        Build a self-contained, JSON-serializable decision record in
        ``audit_trail_``. It includes resolved settings, a data digest,
        every fit and seed, aggregate scores, adjusted pairwise p-values,
        convergence outcomes, and the final tie-break or separation rule.

    Attributes
    ----------
    cv_results_ : dict of arrays
        One row per individual fit. Convert with ``pd.DataFrame(...)``.
        ``score`` is the raw criterion value, never negated.
        ``greater_is_better`` records the ranking direction.
        ``fit_time`` is optimizer wall time in seconds.
    summary_ : dict of arrays
        One row per ``(n_components, alpha)``: ``n_fits``,
        ``n_converged``, and ``best_score``/``median_score``/
        ``mean_score``/``std_score``, plus mean and total fit time.
    audit_trail_ : dict
        Present when ``audit=True``. Pass it to ``save_audit`` for an
        atomic JSON record that can be inspected without this Python object.
    best_alpha_ : float
        The regularization strength in the winning cell.
    resolved_ : bool
        Whether the leading cell was statistically separated from every
        other under the configured test. False means the reported
        ``best_n_components_`` came from a tie-break.
    tied_n_components_ : ndarray
        Every k that was not distinguished from the leader by the
        configured test.
    selection_rule_ : {"separation", "parsimony"}
        Which mechanism produced the answer. ``"separation"`` means the
        leader passed the configured comparisons. ``"parsimony"`` means
        the simplest tied cell was taken (smallest k, then largest alpha).
    n_repeats_ : int
        Repeats actually run per cell.
    scores_ : dict
        Per-cell score lists, keyed by ``(n_components, alpha)`` and
        aligned by repeat -- the paired structure the comparison relies on.
    best_n_components_ : int
        Component count in the cell with the best median score across
        repeats, after applying the configured tie-break.
    best_score_ : float

    Notes
    -----
    The selector's recovery, accuracy, stability, and performance have not
    yet been evaluated experimentally.

    Inherits ``BaseEstimator`` for ``get_params``/``set_params``/``clone``,
    but is not claimed to pass sklearn's ``check_estimator`` -- it is a
    sweep object, not a plain transformer. Note also that AIC/BIC are
    *minimized* while held-out log-likelihood is *maximized*, so this class
    must not be wired into sklearn tooling (``GridSearchCV`` and friends)
    that assumes one uniform higher-is-better direction.
    """

    def __init__(
        self,
        n_components_range: Sequence[int],
        *,
        criterion: _CriterionName = "held_out_ll",
        alpha_range: Sequence[float] | None = None,
        n_repeats: int = 20,
        test: str = "t",
        fdr: float = 0.05,
        max_iter: int = 6000,
        tol: float = 1e-6,
        patience: int = 100,
        stopping_rule: StoppingRule = "objective",
        learning_rate: float = 0.1,
        optimizer: OptimizerName = "adam",
        gradient: GradientPath = "analytic",
        random_state: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device | None = None,
        executor: Executor | None = None,
        checkpoint: str | Path | None = None,
        audit: bool = False,
    ) -> None:
        # No work here -- values stored unmodified, all validation in fit().
        self.n_components_range = n_components_range
        self.criterion = criterion
        self.alpha_range = alpha_range
        self.n_repeats = n_repeats
        self.test = test
        self.fdr = fdr
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.stopping_rule = stopping_rule
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.gradient = gradient
        self.random_state = random_state
        self.dtype = dtype
        self.device = device
        self.executor = executor
        self.checkpoint = checkpoint
        self.audit = audit

    def _resolve_alpha_range(self, criterion: _Criterion) -> Sequence[float]:
        """
        The alpha grid this run will sweep.

        ``None`` resolves to ``DEFAULT_ALPHA_RANGE`` for held-out
        likelihood and ``(0.0,)`` for AIC or BIC. The information criteria
        default to the unpenalized objective because ``param_count`` does
        not account for shrinkage.

        An explicit ``alpha_range`` is always honoured as given, including
        a penalized grid with AIC (which still warns).
        """
        if self.alpha_range is not None:
            return self.alpha_range
        return DEFAULT_ALPHA_RANGE if criterion.requires_masking else (0.0,)

    def _resolve_criterion(self) -> _Criterion:
        if not isinstance(self.criterion, str) or self.criterion not in _CRITERIA:
            raise ValueError(
                f"Unknown criterion {self.criterion!r}; expected one of {sorted(_CRITERIA)}"
            )
        return _CRITERIA[self.criterion]

    def _warn_if_penalized_information_criterion(self, criterion: _Criterion) -> None:
        # AIC/BIC and alpha > 0 do not compose. param_count() counts free
        # parameters under *unpenalized* maximum likelihood; under shrinkage
        # the effective degrees of freedom is strictly smaller, so the
        # resulting number is neither standard AIC/BIC nor a corrected
        # version of one. AIC/BIC exist here as legacy reproduction support
        # for a paper that used no regularization, so the two are not
        # expected to be combined -- warn rather than return a
        # confidently-wrong value.
        if criterion.requires_masking:
            return
        penalized = [a for a in self._resolve_alpha_range(criterion) if a]
        if penalized:
            warnings.warn(
                f"criterion={criterion.name!r} was combined with a nonzero alpha "
                f"({penalized}). param_count() assumes unpenalized maximum "
                "likelihood; under L2 shrinkage the effective degrees of freedom "
                "is smaller, so these AIC/BIC values are not comparable to "
                "unpenalized ones (or to published values). Use alpha=0.0 for "
                "information criteria, or criterion='held_out_ll', which needs "
                "no parameter count at all.",
                UserWarning,
                stacklevel=3,
            )

    def _separation(
        self,
        scores: dict[tuple[int, float], list[float]],
        criterion: _Criterion,
    ) -> tuple[
        bool,
        list[tuple[int, float]],
        tuple[int, float],
        list[dict[str, object]],
    ]:
        """
        Is the leading cell distinguishable from the others?

        Returns ``(resolved, survivors, leader, comparisons)``.

        Partitions are shared across cells (common random numbers), so each
        repeat is a head-to-head match on identical data and the comparison
        reduces to a one-sample test on the paired advantage of the leader.

        - ``"t"`` (default): a one-sided paired t-test, which assumes the
          paired differences are roughly normal.
        - ``"sign"``: counts how many repeats the leader won.

        p-values are Benjamini-Hochberg corrected across the comparison set,
        controlling the false discovery rate rather than the family-wise
        error rate.

        The leader is chosen from the same scores used in these tests, so
        the returned p-values do not account for that selection step.
        """
        cells = list(scores)
        medians = [float(np.median(scores[cell])) for cell in cells]
        leader_index = _best_index(medians, greater_is_better=criterion.greater_is_better)
        leader = cells[leader_index]
        leader_scores = scores[leader]
        n_repeats = len(leader_scores)

        pvalues: list[float] = []
        challengers: list[tuple[int, float]] = []
        comparisons: list[dict[str, object]] = []
        for index, cell in enumerate(cells):
            if index == leader_index:
                continue
            # "Advantage" is always oriented so that positive favours the
            # leader, whichever direction the criterion runs.
            advantage = np.asarray(
                [
                    (a - b) if criterion.greater_is_better else (b - a)
                    for a, b in zip(leader_scores, scores[cell], strict=True)
                ]
            )
            if self.test == "sign":
                wins = int((advantage > 0.0).sum())
                pvalue = float(binomtest(wins, n_repeats, 0.5, alternative="greater").pvalue)
                statistic: float | int | None = wins
            elif self.test == "t":
                if float(advantage.std(ddof=1)) == 0.0:
                    # Identical every repeat: a positive mean is certain
                    # rather than undefined, a non-positive one is no
                    # evidence at all. Without this the t-test returns nan,
                    # which propagates into the BH step and raises.
                    pvalue = 0.0 if float(advantage.mean()) > 0.0 else 1.0
                    statistic = None
                else:
                    result = ttest_1samp(advantage, 0.0, alternative="greater")
                    statistic = float(result.statistic)
                    pvalue = float(result.pvalue)
                    if not np.isfinite(pvalue):
                        pvalue = 1.0
            else:
                raise ValueError(f"test must be 't' or 'sign', got {self.test!r}")
            pvalues.append(pvalue)
            challengers.append(cell)
            comparisons.append(
                {
                    "challenger": {"n_components": int(cell[0]), "alpha": float(cell[1])},
                    "mean_advantage": float(advantage.mean()),
                    "median_advantage": float(np.median(advantage)),
                    "wins": int((advantage > 0.0).sum()),
                    "ties": int((advantage == 0.0).sum()),
                    "statistic": statistic,
                    "raw_pvalue": pvalue,
                }
            )

        if not pvalues:
            return True, [leader], leader, comparisons

        adjusted = false_discovery_control(np.asarray(pvalues), method="bh")
        survivors = [leader]
        for cell, value, comparison in zip(challengers, adjusted, comparisons, strict=True):
            comparison["adjusted_pvalue"] = float(value)
            comparison["leader_not_significantly_better"] = bool(value > self.fdr)
            if value > self.fdr:
                survivors.append(cell)
        return len(survivors) == 1, survivors, leader, comparisons

    def fit(self, x: object, y: object = None) -> "SiGMoiDSelector":
        """
        Run the sweep.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Binary (0/1) data.
        y : ignored

        Returns
        -------
        self : SiGMoiDSelector
        """
        del y
        self.__dict__.pop("audit_trail_", None)
        criterion = self._resolve_criterion()
        components = list(dict.fromkeys(self.n_components_range))
        alphas = list(dict.fromkeys(self._resolve_alpha_range(criterion)))
        self._validate_hyperparameters(components, alphas)
        n_repeats = self._validated_n_repeats()
        self._warn_if_penalized_information_criterion(criterion)

        device = torch.device(self.device) if self.device is not None else None
        data = validate_binary_matrix(x, dtype=self.dtype, device=device)

        # Duplicates would collapse in the score dictionary and break pairing.
        cells = [(index, k, a) for index, k in enumerate(components) for a in alphas]

        seed_rng = np.random.default_rng(self.random_state)
        # Common random numbers. Partition seeds are drawn once per REPEAT
        # and shared by every (k, alpha) cell, so any two configurations are
        # always compared on identical train/test splits.
        #
        # This matters for selection, not just tidiness: a score decomposes
        # as q(k) + e(partition) + noise, and e(partition) -- how easy the
        # held-out block happens to be -- is large. Drawing a fresh
        # partition per cell leaves that term uncancelled in every
        # comparison, so the ranking mixes model quality with the accidental
        # difficulty of two unrelated splits. Sharing it cancels e exactly:
        # an easy split lifts every k together and changes no ordering.
        #
        # Init seeds are shared across alpha (pairing that axis too) but
        # must vary across k, since the parameter shapes differ and "the
        # same initialization" is not a meaningful notion between them.
        partition_seeds = seed_rng.integers(0, 2**31 - 1, size=n_repeats)
        init_seeds = seed_rng.integers(0, 2**31 - 1, size=(len(components), n_repeats))

        rows: dict[str, list[object]] = {name: [] for name in _RESULT_COLUMNS}
        unconverged: list[tuple[int, int]] = []
        scores: dict[tuple[int, float], list[float]] = {(k, a): [] for _, k, a in cells}

        executor: Executor = self.executor if self.executor is not None else SerialExecutor()

        # In-process, jobs carry the tensor directly. Across processes they
        # carry a path instead: pickling the matrix into every one of
        # several hundred jobs would dominate the transfer cost, whereas a
        # worker loads it once and caches it. Written to CPU because a
        # worker pins its own device and may not be the one that saved it.
        staging: tempfile.TemporaryDirectory[str] | None = None
        data_handle: object = data
        if not isinstance(executor, SerialExecutor):
            staging = tempfile.TemporaryDirectory(prefix="binaria-sweep-")
            data_handle = os.path.join(staging.name, "data.pt")
            torch.save(data.cpu(), data_handle)

        # Everything that could change a fit's result. Deliberately
        # exhaustive rather than minimal: a missing entry here means a
        # changed setting resumes silently and corrupts the sweep, while a
        # spurious one only costs a refit.
        fingerprint: dict[str, object] = {
            "result_schema_version": 4,
            "package_version": importlib.metadata.version("binaria"),
            "components": tuple(components),
            "alphas": tuple(alphas),
            "criterion": criterion.name,
            "test": self.test,
            "fdr": float(self.fdr),
            "n_repeats": n_repeats,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "patience": self.patience,
            "stopping_rule": self.stopping_rule,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "gradient": self.gradient,
            "dtype": str(self.dtype),
            "device": str(data.device),
            "shape": tuple(data.shape),
            "data_digest": _data_digest(data),
        }
        checkpoint_path = Path(self.checkpoint) if self.checkpoint is not None else None
        completed: dict[tuple[int, float, int], dict[str, object]] = {}
        if checkpoint_path is not None and checkpoint_path.exists():
            completed = _load_checkpoint(checkpoint_path, fingerprint)

        # Repeats are the OUTER loop, so after each pass every cell has the
        # same number of observations on the same partitions. Adapting per
        # cell would stop them sharing partitions, destroying the pairing
        # above. It also makes every pass boundary a balanced snapshot, and
        # therefore a natural checkpoint.
        try:
            for repeat in range(n_repeats):
                wave = [
                    _FitJob(
                        data=data_handle,
                        criterion=criterion,
                        n_components=n_components,
                        alpha=alpha,
                        repeat=repeat,
                        init_seed=int(init_seeds[k_index, repeat]),
                        partition_seed=(
                            int(partition_seeds[repeat]) if criterion.requires_masking else None
                        ),
                        device=device,
                        dtype=self.dtype,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        patience=self.patience,
                        stopping_rule=self.stopping_rule,
                        learning_rate=self.learning_rate,
                        optimizer=self.optimizer,
                        gradient=self.gradient,
                    )
                    for k_index, n_components, alpha in cells
                ]
                # Anything already on disk from an interrupted run is reused
                # rather than refitted. Safe because a fit is a pure function
                # of (data, k, alpha, init_seed, partition_seed) and all of
                # those are pinned before the loop starts -- which is why the
                # fingerprint has to be strict about every input.
                pending = [job for job in wave if _job_key(job) not in completed]
                fresh = dict(
                    zip(
                        (_job_key(job) for job in pending),
                        executor.map(_execute_fit_job, pending),
                        strict=True,
                    )
                )
                completed.update(fresh)
                if checkpoint_path is not None and fresh:
                    _write_checkpoint(checkpoint_path, fingerprint, completed)

                # Rebuilt in job order rather than completion order, so a
                # resumed or parallel sweep produces the same cv_results_
                # ordering as a serial uninterrupted one.
                for job, values in zip(
                    wave, (completed[_job_key(job)] for job in wave), strict=True
                ):
                    for name in _RESULT_COLUMNS:
                        rows[name].append(values[name])
                    scores[(job.n_components, job.alpha)].append(float(values["score"]))  # type: ignore[arg-type]
                    if not values["converged"]:
                        unconverged.append((job.n_components, repeat))
        finally:
            if staging is not None:
                staging.cleanup()

        if unconverged:
            # Recorded, not dropped: ranking unconverged fits silently is
            # how you get a confident, wrong k.
            warnings.warn(
                f"{len(unconverged)} of {len(rows['score'])} fits did not converge "
                f"within max_iter={self.max_iter} "
                f"(k, repeat): {unconverged[:10]}"
                f"{' ...' if len(unconverged) > 10 else ''}. "
                "Their results are still recorded in cv_results_ with "
                "converged=False, but ranking unconverged fits is unreliable.",
                UserWarning,
                stacklevel=2,
            )

        self.cv_results_ = {name: np.asarray(values) for name, values in rows.items()}
        self.summary_ = self._summarise(criterion)
        self.n_repeats_ = n_repeats
        self.scores_ = scores

        resolved, tied, leader, comparisons = self._separation(scores, criterion)
        if resolved:
            best_cell = leader
            self.resolved_ = True
            self.selection_rule_ = "separation"
        else:
            self.resolved_ = False
            # Parsimony: among cells that could not be told apart, prefer
            # the simplest -- smallest k first, then largest alpha (more
            # shrinkage is a lower effective complexity). Surfaced via
            # selection_rule_ and tied_n_components_ rather than folded
            # silently into best_n_components_, because "these were
            # indistinguishable and we broke the tie" is a materially
            # different claim from "this one won".
            best_cell = min(tied, key=lambda cell: (cell[0], -cell[1]))
            self.selection_rule_ = "parsimony"

        self.tied_n_components_ = np.asarray(sorted({k for k, _ in tied}))
        self.best_n_components_, self.best_alpha_ = int(best_cell[0]), float(best_cell[1])
        self.best_score_ = float(np.median(scores[best_cell]))
        if self.audit:
            self.audit_trail_ = self._build_audit_trail(
                fingerprint=fingerprint,
                criterion=criterion,
                rows=rows,
                comparisons=comparisons,
                leader=leader,
                tied=tied,
                best_cell=best_cell,
            )
        return self

    def _build_audit_trail(
        self,
        *,
        fingerprint: dict[str, object],
        criterion: _Criterion,
        rows: dict[str, list[object]],
        comparisons: list[dict[str, object]],
        leader: tuple[int, float],
        tied: list[tuple[int, float]],
        best_cell: tuple[int, float],
    ) -> dict[str, object]:
        fit_records = [
            dict(zip(_RESULT_COLUMNS, values, strict=True))
            for values in zip(*(rows[name] for name in _RESULT_COLUMNS), strict=True)
        ]
        summary_names = tuple(self.summary_)
        summary_records = [
            {
                name: (value.item() if isinstance(value, np.generic) else value)
                for name, value in zip(summary_names, values, strict=True)
            }
            for values in zip(*(self.summary_[name] for name in summary_names), strict=True)
        ]
        n_converged = int(np.asarray(self.cv_results_["converged"], dtype=bool).sum())
        n_fits = len(fit_records)
        total_fit_time = float(np.asarray(self.cv_results_["fit_time"], dtype=float).sum())
        json_fingerprint = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in fingerprint.items()
        }
        return {
            "schema_version": 1,
            "package_version": importlib.metadata.version("binaria"),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": importlib.metadata.version("scipy"),
                "torch": torch.__version__,
            },
            "criterion": {
                "name": criterion.name,
                "greater_is_better": criterion.greater_is_better,
                "requires_masking": criterion.requires_masking,
            },
            "settings": {
                **json_fingerprint,
                "executor": (
                    "SerialExecutor" if self.executor is None else type(self.executor).__name__
                ),
            },
            "convergence": {
                "n_fits": n_fits,
                "n_converged": n_converged,
                "fraction": n_converged / n_fits,
                "total_fit_time": total_fit_time,
            },
            "fits": fit_records,
            "summary": summary_records,
            "decision": {
                "ranking_basis": "median_score",
                "leader": {
                    "n_components": int(leader[0]),
                    "alpha": float(leader[1]),
                },
                "comparisons": comparisons,
                "resolved": self.resolved_,
                "tied_cells": [
                    {"n_components": int(k), "alpha": float(alpha)} for k, alpha in tied
                ],
                "selection_rule": self.selection_rule_,
                "selected": {
                    "n_components": int(best_cell[0]),
                    "alpha": float(best_cell[1]),
                    "score": self.best_score_,
                },
            },
        }

    def save_audit(self, path: str | Path) -> None:
        """Write ``audit_trail_`` as an atomic, human-readable JSON file."""
        if not hasattr(self, "audit_trail_"):
            raise ValueError("No audit trail is available; fit with audit=True first")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.parent / (destination.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as stream:
            json.dump(self.audit_trail_, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(tmp_path, destination)

    def _validate_hyperparameters(
        self, components: Sequence[object], alphas: Sequence[object]
    ) -> None:
        if not components:
            raise ValueError("n_components_range must contain at least one value")
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) or value < 1
            for value in components
        ):
            raise ValueError(f"n_components_range must contain positive integers, got {components}")
        if not alphas:
            raise ValueError("alpha_range must contain at least one value")
        if any(
            not isinstance(value, Real) or not math.isfinite(value) or value < 0 for value in alphas
        ):
            raise ValueError(f"alpha_range must contain finite non-negative numbers, got {alphas}")
        if (
            not isinstance(self.max_iter, Integral)
            or isinstance(self.max_iter, bool)
            or self.max_iter < 1
        ):
            raise ValueError(f"max_iter must be a positive int, got {self.max_iter!r}")
        if not isinstance(self.tol, Real) or not math.isfinite(self.tol) or self.tol < 0:
            raise ValueError(f"tol must be a finite non-negative number, got {self.tol!r}")
        if (
            not isinstance(self.patience, Integral)
            or isinstance(self.patience, bool)
            or self.patience < 1
        ):
            raise ValueError(f"patience must be a positive int, got {self.patience!r}")
        if self.stopping_rule not in ("objective", "energy_gradient"):
            raise ValueError(
                "stopping_rule must be 'objective' or 'energy_gradient', "
                f"got {self.stopping_rule!r}"
            )
        if (
            not isinstance(self.learning_rate, Real)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                f"learning_rate must be a finite positive number, got {self.learning_rate!r}"
            )
        if self.gradient not in ("analytic", "autograd"):
            raise ValueError(f"gradient must be 'analytic' or 'autograd', got {self.gradient!r}")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {self.dtype!r}")
        if self.random_state is not None and (
            not isinstance(self.random_state, Integral)
            or isinstance(self.random_state, bool)
            or self.random_state < 0
        ):
            raise ValueError(
                f"random_state must be a non-negative int or None, got {self.random_state!r}"
            )
        if self.test not in ("t", "sign"):
            raise ValueError(f"test must be 't' or 'sign', got {self.test!r}")
        if not isinstance(self.fdr, Real) or not math.isfinite(self.fdr) or not 0 < self.fdr < 1:
            raise ValueError(f"fdr must be between 0 and 1, got {self.fdr!r}")
        if not isinstance(self.audit, bool):
            raise ValueError(f"audit must be a bool, got {self.audit!r}")

    def _validated_n_repeats(self) -> int:
        """
        Repeats to run, rejected up front if the test could never resolve.

        The floor differs by test and is checked before any fitting, so a
        budget too small to establish anything fails in milliseconds rather
        than after the sweep. A sign test with R repeats has a p-value floor
        of ``0.5**R`` even on a unanimous result, so it needs enough repeats
        for that to clear ``fdr``; a t-test only needs a variance estimate.
        """
        if not isinstance(self.n_repeats, int) or isinstance(self.n_repeats, bool):
            raise ValueError(f"n_repeats must be an int, got {self.n_repeats!r}")
        floor = 2
        if self.test == "sign":
            floor = math.ceil(math.log(self.fdr) / math.log(0.5))
        if self.n_repeats < floor:
            raise ValueError(
                f"n_repeats={self.n_repeats} is below the floor of {floor} for "
                f"test={self.test!r} at fdr={self.fdr}: no comparison could reach "
                f"significance regardless of the data."
            )
        return int(self.n_repeats)

    def _summarise(self, criterion: _Criterion) -> dict[str, np.ndarray]:
        # One row per (n_components, alpha) cell -- the grid is flattened
        # rather than kept 2D so the result still converts to a DataFrame in
        # one call, and so _best_index works over it unchanged.
        scores = np.asarray(self.cv_results_["score"], dtype=float)
        ks = np.asarray(self.cv_results_["n_components"])
        alphas = np.asarray(self.cv_results_["alpha"], dtype=float)
        converged = np.asarray(self.cv_results_["converged"], dtype=bool)
        fit_times = np.asarray(self.cv_results_["fit_time"], dtype=float)

        out: dict[str, list[object]] = {
            name: []
            for name in (
                "n_components",
                "alpha",
                "criterion",
                "n_fits",
                "n_converged",
                "best_score",
                "median_score",
                "mean_score",
                "std_score",
                "mean_fit_time",
                "total_fit_time",
            )
        }
        cells = dict.fromkeys(zip(ks.tolist(), alphas.tolist(), strict=True))
        for n_components, alpha in cells:
            sel = (ks == n_components) & (alphas == alpha)
            group = scores[sel]
            out["n_components"].append(n_components)
            out["alpha"].append(alpha)
            out["criterion"].append(criterion.name)
            out["n_fits"].append(int(sel.sum()))
            out["n_converged"].append(int(converged[sel].sum()))
            out["best_score"].append(
                float(group.max() if criterion.greater_is_better else group.min())
            )
            out["median_score"].append(float(np.median(group)))
            out["mean_score"].append(float(group.mean()))
            out["std_score"].append(float(group.std()))
            out["mean_fit_time"].append(float(fit_times[sel].mean()))
            out["total_fit_time"].append(float(fit_times[sel].sum()))
        return {name: np.asarray(values) for name, values in out.items()}
