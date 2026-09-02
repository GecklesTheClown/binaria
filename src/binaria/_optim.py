import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from binaria._core import Core
from binaria.callbacks import Callback, IterationState

GradientPath = Literal["analytic", "autograd"]
OptimizerName = Literal["adam", "sgd"]


@dataclass
class FitResult:
    n_iter: int
    converged: bool
    fit_time: float


@dataclass
class _ConvergenceMonitor:
    """Track consecutive small relative changes in an objective."""

    tol: float
    patience: int
    enabled: bool = True
    previous_objective: float | None = None
    streak: int = 0

    def update(self, objective: float) -> bool:
        previous = self.previous_objective
        self.previous_objective = objective
        if not self.enabled or previous is None:
            return False

        relative_change = abs(objective - previous) / (abs(previous) + 1e-12)
        if relative_change < self.tol:
            self.streak += 1
        else:
            self.streak = 0
        return self.streak >= self.patience


def make_optimizer(
    parameters: Iterable[torch.Tensor], *, name: OptimizerName, lr: float
) -> torch.optim.Optimizer:
    """Construct an ascent optimizer for Binaria's maximization objective."""
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, maximize=True)
    if name == "sgd":
        # Deliberately plain full-batch gradient ascent. L2 regularization
        # belongs to `alpha`; PyTorch weight_decay would duplicate it with
        # different scaling, and momentum would no longer reproduce the
        # published update rule.
        return torch.optim.SGD(parameters, lr=lr, momentum=0, maximize=True)
    raise ValueError(f"Unknown optimizer {name!r}; expected 'adam' or 'sgd'")


def fit(
    core: Core,
    data: torch.Tensor,
    *,
    max_iter: int = 6000,
    tol: float = 1e-6,
    patience: int = 100,
    lr: float = 0.1,
    optimizer: OptimizerName = "adam",
    gradient: GradientPath = "analytic",
    lr_decay: float | None = None,
    callbacks: Sequence[Callback] = (),
    mask: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> FitResult:
    # maximize=True: ascend log_likelihood() directly, no manual negation --
    # see _core.py's module docstring for the sign convention this relies on.
    torch_optimizer = make_optimizer(core.parameters(), name=optimizer, lr=lr)

    # A pre-built scheduler instance isn't a usable parameter here: an
    # LRScheduler must be constructed bound to a specific optimizer
    # (ExponentialLR(optimizer, gamma=...)), and `optimizer` only exists
    # inside this function -- a caller could never construct a matching
    # one beforehand. lr_decay=None (default) disables decay: plain
    # fixed-lr Adam rarely triggers converged_=True within max_iter, but
    # makes more progress within a fixed budget than decay does, since
    # decaying from iteration 1 shrinks the effective learning rate too
    # far too early. Decay is therefore opt-in, not default.
    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(torch_optimizer, gamma=lr_decay)
        if lr_decay is not None
        else None
    )

    convergence = _ConvergenceMonitor(
        tol=tol,
        patience=patience,
        enabled=alpha > 0.0,
    )
    converged = False
    n_iter = 0
    started_at = time.perf_counter()

    # Convergence follows the masked, penalized objective actually optimized.
    # Callbacks still receive the unpenalized likelihood used by scoring.
    for n_iter in range(1, max_iter + 1):
        if gradient == "analytic":
            grad_beta, grad_energy = core.analytic_gradients(data, mask=mask)
            if alpha:
                grad_beta = grad_beta - 2.0 * alpha * core.beta.detach()
                grad_energy = grad_energy - 2.0 * alpha * core.energy.detach()
            core.beta.grad = grad_beta
            core.energy.grad = grad_energy
        elif gradient == "autograd":
            torch_optimizer.zero_grad()
            penalty_tensor = core.l2_penalty()
            objective = core.log_likelihood(data, mask=mask) - alpha * penalty_tensor
            objective.backward()  # type: ignore[no-untyped-call]  # torch stubs don't type backward
        else:
            raise ValueError(f"Unknown gradient path: {gradient!r}")

        torch_optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Measure the parameters that callbacks and checkpoints receive,
        # after the update rather than the stale pre-step state.
        with torch.no_grad():
            current_ll = core.log_likelihood(data, mask=mask).item()
            penalty = core.l2_penalty().item() if alpha else 0.0
        current_objective = current_ll - alpha * penalty

        just_converged = convergence.update(current_objective)
        is_final = just_converged or n_iter == max_iter
        state = IterationState(
            iteration=n_iter,
            log_likelihood=current_ll,
            objective=current_objective,
            l2_penalty=penalty,
            elapsed_time=time.perf_counter() - started_at,
            is_final=is_final,
        )
        for callback in callbacks:
            callback.on_iteration(
                state=state,
                core=core,
                optimizer=torch_optimizer,
            )

        if just_converged:
            converged = True
            break

    return FitResult(n_iter=n_iter, converged=converged, fit_time=time.perf_counter() - started_at)
