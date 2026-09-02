import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from binaria._core import Core


@dataclass(frozen=True)
class IterationState:
    """Scalar fit state after one completed optimizer step."""

    iteration: int
    log_likelihood: float
    objective: float
    l2_penalty: float
    elapsed_time: float
    is_final: bool = False


class Callback(Protocol):
    # Structural typing lets third-party loggers implement this directly.
    def on_iteration(
        self,
        *,
        state: IterationState,
        core: Core,
        optimizer: torch.optim.Optimizer,
    ) -> None: ...


@dataclass
class History:
    """Record scalar optimization history at a configurable stride.

    ``diagnostics=False`` keeps the default path inexpensive: only values
    already computed by the optimizer are stored. Opting in records factor,
    logit, and probability-saturation summaries without retaining full
    parameter or probability tensors.

    ``gradient_norm`` describes the gradient used for that iteration's
    update. All other recorded values describe the post-update model exposed
    through ``core`` and, on the final iteration, by the fitted estimator.
    """

    every: int = 1
    record_gradient_norm: bool = False
    diagnostics: bool = False
    saturation_thresholds: tuple[float, ...] = (1e-2, 1e-3, 1e-6)
    iteration: list[int] = field(default_factory=list)
    log_likelihood: list[float] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    elapsed_time: list[float] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    l2_penalty: list[float] = field(default_factory=list)
    max_abs_logit: list[float] = field(default_factory=list)
    beta_norm: list[float] = field(default_factory=list)
    energy_norm: list[float] = field(default_factory=list)
    factor_norm: list[float] = field(default_factory=list)
    saturation_fraction: dict[float, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.every, int) or isinstance(self.every, bool) or self.every < 1:
            raise ValueError(f"every must be a positive int, got {self.every!r}")
        thresholds = tuple(self.saturation_thresholds)
        if any(not math.isfinite(value) or not 0.0 < value < 0.5 for value in thresholds) or len(
            set(thresholds)
        ) != len(thresholds):
            raise ValueError(
                "saturation_thresholds must contain unique finite values between 0 and 0.5"
            )
        self.saturation_thresholds = thresholds
        self.saturation_fraction = {threshold: [] for threshold in thresholds}

    def on_iteration(
        self,
        *,
        state: IterationState,
        core: Core,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        del optimizer  # part of the Callback protocol, unused here
        if not (state.is_final or state.iteration % self.every == 0):
            return
        self.iteration.append(state.iteration)
        self.log_likelihood.append(state.log_likelihood)
        self.elapsed_time.append(state.elapsed_time)
        if self.diagnostics:
            self.objective.append(state.objective)
            self.l2_penalty.append(state.l2_penalty)
            with torch.no_grad():
                logits = core()
                beta_norm = torch.linalg.vector_norm(core.beta).item()
                energy_norm = torch.linalg.vector_norm(core.energy).item()
                self.max_abs_logit.append(logits.abs().max().item())
                self.beta_norm.append(beta_norm)
                self.energy_norm.append(energy_norm)
                self.factor_norm.append(math.sqrt(state.l2_penalty))
                probabilities = torch.sigmoid(logits)
                for threshold in self.saturation_thresholds:
                    saturated = (probabilities < threshold) | (probabilities > 1.0 - threshold)
                    fraction = saturated.count_nonzero().item() / saturated.numel()
                    self.saturation_fraction[threshold].append(fraction)
        if self.record_gradient_norm:
            assert core.beta.grad is not None
            assert core.energy.grad is not None
            grad_norm_sq = core.beta.grad.pow(2).sum() + core.energy.grad.pow(2).sum()
            self.gradient_norm.append(torch.sqrt(grad_norm_sq).item())

    def as_dict(self) -> dict[str, list[int] | list[float]]:
        """Return equal-length columns suitable for pandas or plotting."""
        columns: dict[str, list[int] | list[float]] = {
            "iteration": list(self.iteration),
            "log_likelihood": list(self.log_likelihood),
            "elapsed_time": list(self.elapsed_time),
        }
        if self.record_gradient_norm:
            columns["gradient_norm"] = list(self.gradient_norm)
        if self.diagnostics:
            columns.update(
                {
                    "objective": list(self.objective),
                    "l2_penalty": list(self.l2_penalty),
                    "max_abs_logit": list(self.max_abs_logit),
                    "beta_norm": list(self.beta_norm),
                    "energy_norm": list(self.energy_norm),
                    "factor_norm": list(self.factor_norm),
                }
            )
            for threshold in self.saturation_thresholds:
                columns[f"saturation_fraction_{_format_threshold(threshold)}"] = list(
                    self.saturation_fraction[threshold]
                )
        return columns


def _format_threshold(value: float) -> str:
    """Format a threshold compactly without conflating distinct values."""
    mantissa, exponent = f"{value:.15e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent):+03d}"


@dataclass
class Checkpoint:
    path: Path
    every: int = 1000

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not isinstance(self.every, int) or isinstance(self.every, bool) or self.every < 1:
            raise ValueError(f"every must be a positive int, got {self.every!r}")

    def on_iteration(
        self,
        *,
        state: IterationState,
        core: Core,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if not (state.is_final or state.iteration % self.every == 0):
            return
        checkpoint_state = {
            "iteration": state.iteration,
            "log_likelihood": state.log_likelihood,
            # Parameters alone would mean "resume" silently discards the
            # optimizer's state (such as Adam moments) and may produce a
            # different run from here on -- both must be saved together.
            "model_state_dict": core.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Same-directory replacement keeps the write atomic.
        tmp_path = self.path.parent / (self.path.name + ".tmp")
        torch.save(checkpoint_state, tmp_path)
        os.replace(tmp_path, self.path)


def load_checkpoint(path: Path, core: Core, optimizer: torch.optim.Optimizer) -> int:
    state = torch.load(path, weights_only=False)
    core.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    torch.set_rng_state(state["rng_state"])
    return int(state["iteration"])
