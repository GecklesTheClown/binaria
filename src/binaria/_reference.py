"""
Reproduces the procedure of the original implementation:
float64, NumPy Uniform[0, 0.1) initialisation, simultaneous (Jacobi) plain
gradient ascent, and the ||grad E||_F / ||E||_F stopping rule. This is a
test oracle, not the recommended user path.
"""

from time import perf_counter

import numpy as np
import torch

from binaria._core import Core
from binaria._optim import FitResult


def reference_init(
    n_samples: int,
    n_features: int,
    n_components: int,
    *,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # NumPy's RNG and beta-before-energy draw order are reference invariants.
    rng = np.random.RandomState(seed)
    beta = rng.uniform(0.0, 0.1, size=(n_samples, n_components))
    energy = rng.uniform(0.0, 0.1, size=(n_components, n_features))
    return torch.from_numpy(beta).clone(), torch.from_numpy(energy).clone()


def fit_reference(
    core: Core,
    data: torch.Tensor,
    *,
    lr: float = 1e-4,
    max_iter: int = 400_000,
    tol: float = 1e-3,
    check_every: int = 1000,
) -> FitResult:
    started_at = perf_counter()
    if core.beta.dtype != torch.float64 or core.energy.dtype != torch.float64:
        # float64 must match exactly, or this measures dtype drift
        # instead of comparing procedures.
        raise ValueError("fit_reference requires float64 beta/energy, matching the original")

    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        # Compute both gradients before either simultaneous update.
        grad_beta, grad_energy = core.analytic_gradients(data)
        with torch.no_grad():
            core.beta += lr * grad_beta  # type: ignore[misc]
            core.energy += lr * grad_energy  # type: ignore[misc]

        if n_iter % check_every == 0:
            # Preserve the reference rule even though a growing denominator
            # can make it declare convergence too early.
            grad_energy_norm = grad_energy.norm()
            energy_norm = core.energy.norm()
            if (grad_energy_norm / energy_norm) < tol:
                converged = True
                break

    return FitResult(n_iter=n_iter, converged=converged, fit_time=perf_counter() - started_at)
