"""Compare candidate learning rates on a synthetic configuration grid.

Run it, do not import it.

    uv run python benchmarks/learning_rate_sweep.py            # trimmed grid
    uv run python benchmarks/learning_rate_sweep.py --full     # the whole thing

For each configuration, the script reports how far each rate falls short
of the best objective reached within the same budget. This is an
exploratory benchmark; the repository does not currently report its output
or use it as empirical validation of the defaults.
"""

import argparse
import itertools
import time
import warnings

import torch

from binaria._core import Core

LR_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
TOL = 1e-6


def rank_k_binary(
    n_samples: int, n_features: int, k: int, density: float | None, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    beta = torch.randn(n_samples, k, generator=generator, dtype=torch.float64)
    energy = torch.randn(k, n_features, generator=generator, dtype=torch.float64)
    logits = -(beta @ energy)
    if density is not None:
        logits = logits - torch.quantile(logits.flatten(), 1 - density)
    return torch.bernoulli(torch.sigmoid(logits), generator=generator)


def fit(
    data: torch.Tensor, k: int, alpha: float, lr: float, budget: int, device: str
) -> tuple[float, bool, int]:
    """Best objective reached, whether the tol rule fired, and where."""
    core = Core(*data.shape, k, dtype=torch.float64, device=device)
    optimizer = torch.optim.Adam(core.parameters(), lr=lr, maximize=True)
    previous, converged, stopped_at = None, False, budget
    best = float("-inf")
    for iteration in range(1, budget + 1):
        grad_beta, grad_energy = core.analytic_gradients(data)
        grad_beta = grad_beta - 2.0 * alpha * core.beta.detach()
        grad_energy = grad_energy - 2.0 * alpha * core.energy.detach()
        core.beta.grad, core.energy.grad = grad_beta, grad_energy
        optimizer.step()
        with torch.no_grad():
            objective = core.log_likelihood(data).item() - alpha * core.l2_penalty().item()
        if objective == objective:  # NaN-safe
            best = max(best, objective)
        if previous is not None and not converged:
            if abs(objective - previous) / (abs(previous) + 1e-12) < TOL:
                converged, stopped_at = True, iteration
        previous = objective
    return best, converged, stopped_at


def saturation_report(device: str, budget: int) -> None:
    """The alpha=0 interaction, on data that is guaranteed separable."""
    size = 200
    axis = torch.arange(size, dtype=torch.float64)
    grid = axis[:, None] / size + axis[None, :] / size
    data = (grid < 1.0).double().to(device)  # staircase: sign-rank 2

    print(f"\nalpha=0 on separable data ({size}x{size} staircase, k=2, {budget} iters)")
    print(f"{'lr':>8} {'max|logit|':>12} {'% saturated':>12}")
    for lr in LR_GRID:
        core = Core(size, size, 2, dtype=torch.float64, device=device)
        optimizer = torch.optim.Adam(core.parameters(), lr=lr, maximize=True)
        for _ in range(budget):
            core.beta.grad, core.energy.grad = core.analytic_gradients(data)
            optimizer.step()
        with torch.no_grad():
            logits = core()
            probabilities = torch.sigmoid(logits)
            saturated = (probabilities == probabilities.round()).double().mean().item()
        print(f"{lr:8.0e} {logits.abs().max().item():12.1f} {saturated:12.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="the whole grid")
    parser.add_argument("--budget", type=int, default=6000, help="max_iter to compare at")
    parser.add_argument("--saturation", action="store_true", help="alpha=0 report only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    if args.saturation:
        saturation_report(args.device, args.budget)
        return

    shapes = [(200, 100), (1000, 500)] if not args.full else [(200, 100), (500, 250), (1000, 500)]
    densities = [None, 0.05] if not args.full else [None, 0.2, 0.05]
    ks = [3, 6]
    alphas = [0.03, 0.3]
    seeds = [0, 1]

    configs = list(itertools.product(shapes, densities, ks, alphas, seeds))
    shortfalls: dict[float, list[float]] = {lr: [] for lr in LR_GRID}
    outcomes: dict[float, list[tuple[bool, int]]] = {lr: [] for lr in LR_GRID}

    started = time.perf_counter()
    for (rows, cols), density, k, alpha, seed in configs:
        data = rank_k_binary(rows, cols, k, density, seed).to(args.device)
        results = {lr: fit(data, k, alpha, lr, args.budget, args.device) for lr in LR_GRID}
        top = max(objective for objective, _, _ in results.values())
        for lr, (objective, converged, stopped_at) in results.items():
            shortfalls[lr].append((top - objective) / abs(top))
            outcomes[lr].append((converged, stopped_at))

    print(
        f"{len(configs)} configurations at max_iter={args.budget}, tol={TOL:g}, "
        f"device={args.device}\nshortfall is measured against the best objective "
        f"any rate reached on that configuration\n"
    )
    print(
        f"{'lr':>8} {'converged':>11} {'median stop':>12} {'mean short':>12} "
        f"{'worst short':>12} {'>1% short':>10}"
    )
    for lr in LR_GRID:
        short = shortfalls[lr]
        stops = sorted(stop for converged, stop in outcomes[lr] if converged)
        hits = sum(converged for converged, _ in outcomes[lr])
        median = stops[len(stops) // 2] if stops else None
        print(
            f"{lr:8.0e} {f'{hits}/{len(short)}':>11} {median if median else '-':>12} "
            f"{sum(short) / len(short):12.5%} {max(short):12.5%} "
            f"{sum(value > 0.01 for value in short):10d}"
        )
    print(f"\nin {(time.perf_counter() - started) / 60:.1f} min")
    saturation_report(args.device, args.budget)


if __name__ == "__main__":
    main()
