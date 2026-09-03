import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from binaria._core import Core
from binaria.callbacks import Callback, IterationState

GradientPath = Literal["analytic", "autograd"]
OptimizerName = Literal["adam", "sgd"]
StoppingRule = Literal["objective", "energy_gradient"]


@dataclass
class FitResult:
    n_iter: int
    converged: bool
    fit_time: float


@dataclass
class _ConvergenceMonitor:
    """Apply patience to scalar convergence statistics."""

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
        return self.update_statistic(relative_change)

    def update_statistic(self, statistic: float) -> bool:
        """Count one check whose scalar statistic is compared with ``tol``."""
        if not self.enabled:
            return False
        if statistic < self.tol:
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
    stopping_rule: StoppingRule = "objective",
    lr: float = 0.1,
    optimizer: OptimizerName = "adam",
    gradient: GradientPath = "analytic",
    lr_decay: float | None = None,
    callbacks: Sequence[Callback] = (),
    mask: torch.Tensor | None = None,
    alpha: float = 0.0,
) -> FitResult:
    if stopping_rule not in ("objective", "energy_gradient"):
        raise ValueError(f"Unknown stopping rule: {stopping_rule!r}")

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
        enabled=alpha > 0.0 if stopping_rule == "objective" else True,
    )
    converged = False
    n_iter = 0
    started_at = time.perf_counter()

    # Both stopping rules describe the masked, penalized objective actually
    # optimized. Callbacks still receive the unpenalized likelihood for scoring.
    for n_iter in range(1, max_iter + 1):
        objective_grad_energy: torch.Tensor | None = None
        if gradient == "analytic":
            grad_beta, grad_energy = core.analytic_gradients(data, mask=mask)
            if alpha:
                grad_beta = grad_beta - 2.0 * alpha * core.beta.detach()
                grad_energy = grad_energy - 2.0 * alpha * core.energy.detach()
            if stopping_rule == "energy_gradient":
                objective_grad_energy = grad_energy.detach()
            core.beta.grad = grad_beta
            core.energy.grad = grad_energy
        elif gradient == "autograd":
            torch_optimizer.zero_grad()
            penalty_tensor = core.l2_penalty()
            objective = core.log_likelihood(data, mask=mask) - alpha * penalty_tensor
            objective.backward()  # type: ignore[no-untyped-call]  # torch stubs don't type backward
            if stopping_rule == "energy_gradient":
                assert core.energy.grad is not None
                objective_grad_energy = core.energy.grad.detach()
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

        if stopping_rule == "objective":
            just_converged = convergence.update(current_objective)
        elif stopping_rule == "energy_gradient":
            assert objective_grad_energy is not None
            energy_norm = torch.linalg.vector_norm(core.energy).item()
            statistic = (
                torch.linalg.vector_norm(objective_grad_energy).item() / energy_norm
                if energy_norm > 0.0
                else float("inf")
            )
            just_converged = convergence.update_statistic(statistic)
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
