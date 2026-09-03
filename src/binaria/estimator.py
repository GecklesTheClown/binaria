"""Public scikit-learn-style estimator."""

import copy
import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from binaria._core import Core
from binaria._optim import (
    GradientPath,
    OptimizerName,
    StoppingRule,
    _ConvergenceMonitor,
    make_optimizer,
)
from binaria._optim import fit as _fit_core
from binaria.callbacks import Callback, History
from binaria.validation import to_output, validate_binary_matrix


# The ignore comment on the next line is confirmed needed, not preemptive:
# sklearn.* is ignore_missing_imports'd (no stubs upstream), which makes
# mypy treat BaseEstimator/TransformerMixin as `Any` -- and mypy
# specifically flags subclassing from Any ("Class cannot subclass X (has
# type Any)"), independent of the missing-stubs issue itself.
class SiGMoiD(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """
    SiGMoiD binary matrix factorization (Zhao, Plata & Dixit, 2021).

    Decomposes a binary (samples, features) matrix into a sample-specific
    latent factor ``embedding_`` (S x K) and shared feature loadings
    ``components_`` (K x N) via a logistic link: the fitted probability of
    entry (s, i) being 1 is ``sigmoid(-(embedding_ @ components_))[s, i]``.

    Parameters
    ----------
    n_components : int
        Number of latent factors (K).
    max_iter : int, default=6000
        Maximum optimizer iterations. Check ``converged_`` after fitting;
        the current default has not been established empirically.
    tol : float, default=1e-6
        Threshold applied to the statistic selected by ``stopping_rule``.
        See ``patience`` and ``converged_``.
    patience : int, default=100
        Number of consecutive qualifying checks required before the fit is
        considered converged. A check qualifies when the statistic selected
        by ``stopping_rule`` is below ``tol``.
    stopping_rule : {"objective", "energy_gradient"}, default="objective"
        Statistic used to determine convergence during the joint fit.
        ``"objective"`` monitors relative penalized-objective change.
        ``"energy_gradient"`` monitors
        ``||gradient_E objective||_F / ||E||_F``. With ``alpha=0``, setting
        ``tol=1e-3`` and ``patience=1`` tests the stopping threshold used by
        the original authors. Beta-only fits in ``transform`` and ``score``
        continue to use objective change because their energy matrix is
        frozen.
    learning_rate : float, default=0.1
        Optimizer learning rate. The default is provisional and intended
        for Adam; choose an explicit value when using SGD.
    optimizer : {"adam", "sgd"}, default="adam"
        PyTorch optimizer used for every optimization path, including the
        beta-only fits performed by ``transform`` and ``score``. ``"sgd"``
        is full-batch plain gradient ascent with zero momentum; it is not
        minibatch training. Because the likelihood is summed over the
        matrix, SGD gradient magnitudes depend on matrix shape. Set an
        explicit ``learning_rate`` and tune it for the data. The 1e-4 used
        in the documentation is illustrative, not a validated or universally
        safe recommendation.
    gradient : {"analytic", "autograd"}, default="analytic"
        Which gradient path to use during fitting. Both are mathematically
        equivalent; "analytic" avoids per-iteration autograd graph
        construction.
    alpha : float, default=0.03
        L2 regularization strength. The fitted objective becomes
        ``log_likelihood - alpha * (||embedding_||^2 + ||components_||^2)``.

        Pass ``alpha=0.0`` for the unpenalized objective. For separable
        data, an unpenalized Bernoulli likelihood may approach its
        supremum without attaining a finite maximizer. A positive value
        makes the objective coercive, so it attains a finite maximizer.
        The current default has not been established empirically.

        ``log_likelihood_`` remains the *unpenalized* log-likelihood
        regardless of ``alpha``.
    random_state : int or None, default=None
        Seed used to initialize parameters. Fitting does not modify
        PyTorch's global random state.
    device : str, torch.device, or None, default=None
        Device to fit on. None means whatever ``torch.Tensor`` defaults to
        (CPU unless the input is already on another device). Never
        implicitly moved to CUDA.
    dtype : torch.dtype, default=torch.float64
        Floating point precision for the fit.
    callbacks : sequence of Callback or None, default=None
        Objects satisfying the ``Callback`` protocol (``History``,
        ``Checkpoint``, or a third-party logger). Reset at the start of
        every ``fit()`` call via a deep copy, so constructor-supplied
        callback instances are never mutated -- required for sklearn's
        ``clone()`` contract, which assumes constructor parameters are
        inert. Each callback receives an ``IterationState`` describing the
        completed, post-update iteration, plus the matching core and optimizer.
    history_every : int, default=1
        Stride for the automatically-attached ``History`` callback, used
        when ``callbacks`` doesn't already include a ``History`` instance.
        Ignored if it does -- an explicit ``History(every=...)`` in
        ``callbacks`` takes precedence.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Fitted shared feature loadings (E). Always a NumPy array,
        regardless of what dtype/device fitting used.
    embedding_ : ndarray of shape (n_samples, n_components)
        Fitted per-sample latent factors (beta) for the data passed to
        ``fit``.
    n_iter_ : int
        Number of iterations actually run.
    fit_time_ : float
        Wall-clock seconds spent in the optimizer.
    converged_ : bool
        Whether the selected stopping statistic stayed below ``tol`` for
        ``patience`` consecutive iterations before ``max_iter`` was reached.
    log_likelihood_ : float
        Log-likelihood (Eq 2) of the training data under the fitted
        parameters. Higher is better.
    history_ : History
        The ``History`` callback used during this fit (either the one
        found in ``callbacks``, or an automatically-attached one). Pass
        ``History(diagnostics=True)`` in ``callbacks`` to record penalized
        objective, factor and logit norms, and saturation fractions.

    Notes
    -----
    The numerical defaults are provisional and may change. No empirical
    accuracy, recovery, stability, or performance claims are made.
    """

    def __init__(
        self,
        n_components: int,
        *,
        max_iter: int = 6000,
        tol: float = 1e-6,
        patience: int = 100,
        stopping_rule: StoppingRule = "objective",
        learning_rate: float = 0.1,
        optimizer: OptimizerName = "adam",
        gradient: GradientPath = "analytic",
        alpha: float = 0.03,
        random_state: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        callbacks: Sequence[Callback] | None = None,
        history_every: int = 1,
    ) -> None:
        # No work here -- every value stored unmodified, nothing computed
        # or validated. sklearn requirement: get_params()/clone() rely on
        # constructor parameters being stored under their own names,
        # untouched. All validation happens in fit().
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.stopping_rule = stopping_rule
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.gradient = gradient
        self.alpha = alpha
        self.random_state = random_state
        self.device = device
        self.dtype = dtype
        self.callbacks = callbacks
        self.history_every = history_every

    def __sklearn_tags__(self):  # type: ignore[no-untyped-def]
        # Declares what validate_binary_matrix already enforces at runtime,
        # so sklearn's generic test suite (and downstream tooling) knows
        # this estimator's actual input domain rather than discovering it
        # via a ValueError. positive_only is the closest available tag --
        # sklearn has no "strictly 0/1" tag, only "non-negative", so this
        # is necessary but not sufficient to fully describe the real
        # constraint (see check_estimator's expected_failed_checks in
        # tests/test_estimator_api.py for what that gap means in practice).
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        # NOT declaring sparse=True, even though validate_binary_matrix
        # does accept and densify scipy.sparse input: sklearn's own
        # consistency check for this tag specifically wants evidence of
        # its own check_array/validate_data(accept_sparse=...) machinery,
        # not just that sparse input happens to work end-to-end through an
        # independent implementation -- confirmed via check_estimator_
        # sparse_tag's actual failure message. Declaring it anyway would
        # claim a guarantee sklearn's own tooling can't verify.
        return tags

    def _check_is_fitted(self) -> None:
        # sklearn's own check, not a hand-rolled RuntimeError: raises the
        # conventional NotFittedError, which check_estimator and downstream
        # tooling (Pipeline, etc.) specifically expect.
        check_is_fitted(self, attributes=["_core"])

    def _prepare_callbacks(self) -> tuple[list[Callback], History]:
        # Deep copy, not the stored objects directly: constructor
        # parameters must stay inert (sklearn's clone() contract), and
        # History/Checkpoint are stateful -- reusing the same instance
        # across repeated fit() calls would accumulate the previous run's
        # history instead of starting fresh.
        callbacks = [copy.deepcopy(cb) for cb in (self.callbacks or [])]
        history = next((cb for cb in callbacks if isinstance(cb, History)), None)
        if history is None:
            # history_ is populated by default so the common case needs no
            # configuration -- only constructed here if the caller didn't
            # already supply their own History (whose own `every` then
            # takes precedence over history_every).
            history = History(every=self.history_every)
            callbacks.append(history)
        return callbacks, history

    def _validate_hyperparameters(self) -> None:
        if (
            not isinstance(self.n_components, Integral)
            or isinstance(self.n_components, bool)
            or self.n_components < 1
        ):
            raise ValueError(f"n_components must be a positive int, got {self.n_components!r}")
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
        if not isinstance(self.alpha, Real) or not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError(f"alpha must be a finite non-negative number, got {self.alpha!r}")
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
        if (
            not isinstance(self.history_every, Integral)
            or isinstance(self.history_every, bool)
            or self.history_every < 1
        ):
            raise ValueError(f"history_every must be a positive int, got {self.history_every!r}")

    def fit(self, x: object, y: object = None) -> "SiGMoiD":
        """
        Fit the model to a binary matrix.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Binary (0/1) data. Accepts NumPy arrays, PyTorch tensors,
            pandas DataFrames, or scipy.sparse matrices (densified).
        y : ignored
            Present for sklearn API compatibility; this is an unsupervised
            estimator.

        Returns
        -------
        self : SiGMoiD
        """
        del y
        self._validate_hyperparameters()
        device = torch.device(self.device) if self.device is not None else None
        data = validate_binary_matrix(x, dtype=self.dtype, device=device)
        n_samples, n_features = data.shape

        generator = None
        if self.random_state is not None:
            generator = torch.Generator(device=data.device).manual_seed(self.random_state)

        core = Core(
            n_samples=n_samples,
            n_features=n_features,
            n_components=self.n_components,
            dtype=self.dtype,
            device=data.device,
            generator=generator,
        )

        callbacks, history = self._prepare_callbacks()

        result = _fit_core(
            core,
            data,
            max_iter=self.max_iter,
            tol=self.tol,
            patience=self.patience,
            stopping_rule=self.stopping_rule,
            lr=self.learning_rate,
            optimizer=self.optimizer,
            gradient=self.gradient,
            alpha=self.alpha,
            callbacks=callbacks,
        )

        self._core = core
        self.components_ = to_output(core.energy, output="numpy")
        self.embedding_ = to_output(core.beta, output="numpy")
        self.n_iter_ = result.n_iter
        self.fit_time_ = result.fit_time
        self.converged_ = result.converged
        self.log_likelihood_ = core.log_likelihood(data).item()
        self.history_ = history
        return self

    def _fit_beta_only(self, data: torch.Tensor) -> torch.Tensor:
        # Shared by transform() and score(): fit a fresh beta for `data`
        # under FROZEN components_ (energy). Deliberately not _optim.fit()
        # -- its analytic gradient path assigns .grad directly to both
        # beta AND energy, bypassing requires_grad entirely, which would
        # silently update the "frozen" energy. Autograd-only, optimizing
        # over [beta] alone, so energy genuinely cannot move here.
        #
        # Reuse the outer fit's optimizer budget because there is no
        # separate transform/score budget. score(x) refits beta from zero
        # with energy fixed, so it need not equal log_likelihood_ from the
        # joint fit.
        n_samples = data.shape[0]
        frozen_energy = self._core.energy.detach().clone()
        new_core = Core(
            n_samples=n_samples,
            n_features=data.shape[1],
            n_components=self.n_components,
            dtype=self.dtype,
            beta=torch.zeros(n_samples, self.n_components, dtype=self.dtype, device=data.device),
            energy=frozen_energy,
        )
        new_core.energy.requires_grad_(False)

        optimizer = make_optimizer([new_core.beta], name=self.optimizer, lr=self.learning_rate)
        convergence = _ConvergenceMonitor(
            tol=self.tol,
            patience=self.patience,
            enabled=self.alpha > 0.0,
        )
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            # Penalized on the same terms as fit(), so a transform against a
            # regularized model isn't itself unregularized -- otherwise beta
            # for new samples could blow up against a components_ matrix that
            # was fit with shrinkage. Only beta moves here, so energy's
            # contribution to the penalty is a constant and drops out of the
            # gradient; it is still included in the value so the convergence
            # check matches the quantity being maximized (see _optim.fit).
            objective = new_core.log_likelihood(data) - self.alpha * new_core.l2_penalty()
            objective.backward()  # type: ignore[no-untyped-call]  # torch stubs don't type backward
            optimizer.step()

            # Test the parameters that will actually be returned, not the
            # stale objective tensor that produced this iteration's update.
            with torch.no_grad():
                current_objective = (
                    new_core.log_likelihood(data) - self.alpha * new_core.l2_penalty()
                ).item()
            if convergence.update(current_objective):
                break

        return new_core.beta.detach()

    def transform(
        self, x: object, *, output: Literal["numpy", "torch"] = "numpy"
    ) -> np.ndarray | torch.Tensor:
        """
        Infer latent factors for new samples, holding ``components_`` fixed.

        This is a partial refit: only the per-sample latent factor (beta)
        is optimized for `x`; the shared feature loadings learned during
        ``fit`` are not updated.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Must have the same number of features as the data ``fit`` was
            called with.
        output : {"numpy", "torch"}, default="numpy"
            Return a NumPy array or a PyTorch tensor. A CUDA tensor is
            only returned if this estimator was fit with ``device`` set to
            a CUDA device -- never implicitly.

        Returns
        -------
        embedding : ndarray or Tensor of shape (n_samples, n_components)
        """
        self._check_is_fitted()
        data = validate_binary_matrix(x, dtype=self.dtype, device=self._core.energy.device)
        if data.shape[1] != self._core.energy.shape[1]:
            raise ValueError(
                f"x has {data.shape[1]} features, but this {type(self).__name__} "
                f"was fit with {self._core.energy.shape[1]}"
            )
        beta = self._fit_beta_only(data)
        return to_output(beta, output=output)

    def fit_transform(
        self, x: object, y: object = None, *, output: Literal["numpy", "torch"] = "numpy"
    ) -> np.ndarray | torch.Tensor:
        """
        Fit the model to `x` and return its embedding.

        Cheaper than ``fit(x).transform(x)``: the embedding from `fit` is
        reused directly rather than partial-refit from scratch.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
        y : ignored
        output : {"numpy", "torch"}, default="numpy"

        Returns
        -------
        embedding : ndarray or Tensor of shape (n_samples, n_components)
        """
        self.fit(x, y)
        return to_output(self._core.beta, output=output)

    def score(self, x: object, y: object = None) -> float:
        """
        Log-likelihood of `x` under the fitted model. Higher is better.

        Per sklearn convention for unsupervised estimators, this is a
        score to *maximize*, not a loss. `x` need not be the training
        data: a fresh beta is fit for it under frozen ``components_`` (the
        same partial refit ``transform`` does) before scoring, so this is
        also what a held-out/cross-validation split should call.

        Calling this on the exact training data will *not* closely match
        ``log_likelihood_`` unless ``fit`` actually reached
        ``converged_ = True``. The refit uses the same optimizer and
        ``max_iter``/``tol``/``patience`` budget, but always monitors objective
        change because its energy matrix is frozen.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        log_likelihood : float
        """
        del y
        self._check_is_fitted()
        data = validate_binary_matrix(x, dtype=self.dtype, device=self._core.energy.device)
        if data.shape[1] != self._core.energy.shape[1]:
            raise ValueError(
                f"x has {data.shape[1]} features, but this {type(self).__name__} "
                f"was fit with {self._core.energy.shape[1]}"
            )
        beta = self._fit_beta_only(data)
        scoring_core = Core(
            n_samples=data.shape[0],
            n_features=data.shape[1],
            n_components=self.n_components,
            dtype=self.dtype,
            beta=beta,
            energy=self._core.energy.detach(),
        )
        with torch.no_grad():
            return scoring_core.log_likelihood(data).item()

    def sample(
        self,
        n_samples: int = 1,
        *,
        random_state: int | None = None,
        output: Literal["numpy", "torch"] = "numpy",
    ) -> np.ndarray | torch.Tensor:
        """
        Draw samples from the fitted generative model (Eq 4/5).

        Picks `n_samples` latent factors at random (with replacement) from
        the fitted training embedding, evaluates the resulting
        probabilities, and draws Bernoulli variables.

        **This is a bootstrap over fitted latents, not a density model,**
        and the distinction matters more than it first appears. Every
        sample returned uses a `beta_s` that some training sample already
        had, so the generator cannot interpolate between two training rows
        or extrapolate beyond them -- it reproduces the *empirical* latent
        distribution and nothing else. Drawing 10,000 samples from a
        50-sample fit gives 10,000 draws over 50 distinct latent states.

        That is faithful to the paper rather than a shortcut: Eq 4/5
        specifies picking a random latent `beta_s`, and the population of
        `beta_s` across samples *is* the super-statistical distribution,
        represented empirically. But anyone reaching for `sample()`
        expecting a fitted generative density should know it is not one.
        Sampling new latent states would mean fitting a distribution to the
        rows of `beta` -- an addition beyond the published method, not an
        implementation of it.

        Parameters
        ----------
        n_samples : int, default=1
        random_state : int or None, default=None
            Seeds a local generator for this call only -- unlike `fit`,
            does not touch global RNG state.
        output : {"numpy", "torch"}, default="numpy"

        Returns
        -------
        samples : ndarray or Tensor of shape (n_samples, n_features)
        """
        self._check_is_fitted()
        device = self._core.beta.device
        generator = None
        if random_state is not None:
            generator = torch.Generator(device=device).manual_seed(random_state)

        n_fitted_samples = self._core.beta.shape[0]
        indices = torch.randint(
            0, n_fitted_samples, (n_samples,), generator=generator, device=device
        )
        with torch.no_grad():
            sampled_beta = self._core.beta[indices]
            logits = -(sampled_beta @ self._core.energy)
            probabilities = torch.sigmoid(logits)
            samples = torch.bernoulli(probabilities, generator=generator)
        return to_output(samples, output=output)
