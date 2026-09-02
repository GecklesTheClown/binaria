"""Explore selector recovery on a synthetic configuration grid.

Run it directly; do not import it.

    uv run python benchmarks/recovery_sweep.py            # trimmed grid
    uv run python benchmarks/recovery_sweep.py --full     # the whole thing

The script prints recovery against both nominal and estimated identifiable
rank, together with the convergence fraction. Its output has not yet been
used to establish public recovery or convergence claims.
"""

import argparse
import time
import warnings

import numpy as np
import torch

from binaria.selection import SiGMoiDSelector, make_block_mask

K_GRID = [2, 3, 4, 6, 8]


def true_logits(
    n_samples: int, n_features: int, true_k: int, density: float, seed: int
) -> torch.Tensor:
    """
    Rank-`true_k` logits whose Bernoulli draws average `density` ones.

    SiGMoiD has no intercept, so density is set by giving both factors a
    nonzero mean -- ``E[beta @ E] = k * mean_b * mean_e`` -- rather than by
    adding a bias the model cannot represent, and rather than by reserving
    a component (which would be degenerate: a constant contributes a rank-1
    matrix with one distinct value, which a k-1 model simply absorbs).
    """
    generator = torch.Generator().manual_seed(seed)
    beta = torch.randn(n_samples, true_k, dtype=torch.float64, generator=generator) + 1.0
    base = torch.randn(true_k, n_features, dtype=torch.float64, generator=generator)

    def logits_at(shift: float) -> torch.Tensor:
        return -(beta @ (base + shift))

    low, high = -8.0, 8.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if float(torch.sigmoid(logits_at(mid)).mean()) > density:
            low = mid
        else:
            high = mid
    return logits_at(0.5 * (low + high))


def identifiable_rank(logits: torch.Tensor, seed: int = 0, tol: float = 0.002) -> int:
    """
    Components detectable by an ORACLE holding the true parameters.

    No fitting: truncate the true logits to rank r by SVD and score the
    truncation on a held-out block. An upper bound on what any estimator
    could achieve. Smallest r within `tol` of the best, not argmax --
    beyond rank k the truncation is exact, so later ranks tie to within
    floating point.
    """
    n_samples, n_features = logits.shape
    torch.manual_seed(seed)
    data = torch.bernoulli(torch.sigmoid(logits))
    _, test_mask = make_block_mask(n_samples, n_features, seed=seed)
    n_held = int(test_mask.sum())

    u, s, vh = torch.linalg.svd(logits, full_matrices=False)
    scores = []
    for rank in range(1, len(s) + 1):
        approx = (u[:, :rank] * s[:rank]) @ vh[:rank]
        per_entry = -torch.nn.functional.binary_cross_entropy_with_logits(
            approx, data, reduction="none"
        )
        scores.append(float((test_mask * per_entry).sum() / n_held))

    best = max(scores)
    return next(r for r, value in enumerate(scores, start=1) if value >= best - tol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="the full grid (hours)")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=6000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shapes = [(80, 40), (200, 100), (400, 200)] if args.full else [(80, 40), (200, 100)]
    true_ks = [2, 4, 6]
    densities = [0.1, 0.3, 0.5]
    alpha_range = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0) if args.full else (0.0, 0.3, 1.0)

    print(f"device={device}  shapes={shapes}  nominal k={true_ks}  density={densities}")
    print(f"alpha_range={alpha_range}  trials={args.trials}  repeats={args.repeats}")
    print(f"candidate grid={K_GRID}  max_iter={args.max_iter}\n")
    header = (
        f"{'shape':>9} {'nom k':>6} {'ident':>6} {'dens':>5} "
        f"{'hit ident':>10} {'hit nom':>8} {'conv':>5} {'picked':>16}"
    )
    print(header)
    print("-" * len(header))

    hit_identifiable = hit_nominal = total = 0
    started = time.perf_counter()

    for shape in shapes:
        for nominal in true_ks:
            for density in densities:
                picked: list[int] = []
                identifiables: list[int] = []
                converged: list[float] = []
                for trial in range(args.trials):
                    seed = 1000 * trial + nominal
                    logits = true_logits(*shape, nominal, density, seed)
                    identifiables.append(identifiable_rank(logits, seed=seed))
                    generator = torch.Generator().manual_seed(seed)
                    data = torch.bernoulli(torch.sigmoid(logits), generator=generator)

                    selector = SiGMoiDSelector(
                        K_GRID,
                        criterion="held_out_ll",
                        alpha_range=alpha_range,
                        n_repeats=args.repeats,
                        max_iter=args.max_iter,
                        device=device,
                        random_state=trial,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        selector.fit(data.numpy())
                    picked.append(selector.best_n_components_)
                    converged.append(float(selector.cv_results_["converged"].mean()))

                ident = int(np.median(identifiables))
                hits_i = sum(k == i for k, i in zip(picked, identifiables, strict=True))
                hits_n = sum(k == nominal for k in picked)
                hit_identifiable += hits_i
                hit_nominal += hits_n
                total += len(picked)
                counts = {k: picked.count(k) for k in sorted(set(picked))}
                print(
                    f"{f'{shape[0]}x{shape[1]}':>9} {nominal:>6} {ident:>6} {density:>5.1f} "
                    f"{f'{hits_i}/{len(picked)}':>10} {f'{hits_n}/{len(picked)}':>8} "
                    f"{np.mean(converged):>4.0%} {counts!s:>16}"
                )

    elapsed = time.perf_counter() - started
    print(
        f"\nrecovery vs identifiable rank {hit_identifiable}/{total} "
        f"({hit_identifiable / total:.0%});  vs nominal k {hit_nominal}/{total} "
        f"({hit_nominal / total:.0%})  in {elapsed / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
