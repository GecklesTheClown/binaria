"""Select k for one binary matrix across the visible CUDA devices.

    uv run python examples/multi_gpu_sweep.py data.npy --out results/

The main guard is required by spawned CUDA workers. ``MultiGPUExecutor``
dispatches fits, and ``checkpoint`` allows completed fits to be reused
after an interruption. The script prints convergence diagnostics with the
selected cell; model-selection behavior has not yet been validated
experimentally.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from binaria import SiGMoiD, SiGMoiDSelector, save
from binaria.executors import MultiGPUExecutor, available_devices


def load_matrix(path: Path) -> np.ndarray:
    """Load an (n_samples x n_features) binary matrix and check it is one."""
    matrix = np.load(path) if path.suffix == ".npy" else np.loadtxt(path, delimiter=",")
    matrix = np.asarray(matrix, dtype=np.float64)
    unique = np.unique(matrix)
    if not np.isin(unique, [0.0, 1.0]).all():
        raise ValueError(f"expected a 0/1 matrix, found values {unique[:5]}")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help=".npy or .csv, samples x features")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=6000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = load_matrix(args.data)
    devices = available_devices()

    print(f"data      : {data.shape[0]} x {data.shape[1]}, density {data.mean():.3f}")
    print(f"devices   : {devices}")
    print(f"grid      : k={args.k}, alpha=default (0, .03, .1, .3, 1, 3)")
    print(f"fits      : {len(args.k) * 6 * args.repeats}")

    # The pool manages one worker per visible device by default.
    with MultiGPUExecutor() as pool:
        selector = SiGMoiDSelector(
            args.k,
            criterion="held_out_ll",
            n_repeats=args.repeats,
            max_iter=args.max_iter,
            executor=pool,
            # Re-running with matching settings reuses completed fits.
            checkpoint=args.out / "sweep.ckpt",
            random_state=0,
        )
        selector.fit(data)

    # Report the choice together with its diagnostics.
    converged = float(selector.cv_results_["converged"].mean())
    print(f"\nselected k : {selector.best_n_components_}")
    print(f"alpha      : {selector.best_alpha_}")
    print(f"rule       : {selector.selection_rule_}")
    print(f"converged  : {converged:.0%}")
    print(f"tied k     : {selector.tied_n_components_.tolist()}")

    if converged < 0.10:
        print(
            "\nWARNING: fewer than 10% of fits met the stopping criterion.\n"
            "Treat the selected cell as provisional and review the optimizer\n"
            f"settings, including max_iter={args.max_iter}, before interpreting it."
        )
    elif selector.selection_rule_ == "parsimony":
        print(
            "\nNOTE: k was not separated statistically -- the tied candidates above\n"
            "were not distinguished, and the smallest was returned. Interpret k\n"
            "over the tied set rather than as a unique choice."
        )
    elif selector.best_n_components_ in (min(args.k), max(args.k)):
        print(
            "\nNOTE: the winner sits at an edge of the candidate grid, so the true\n"
            "value may lie outside it. Widen --k in that direction and re-run."
        )

    # The full per-fit record, for plotting or a second opinion.
    np.savez(
        args.out / "cv_results.npz",
        **{k: np.asarray(v) for k, v in selector.cv_results_.items()},
    )
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "shape": list(data.shape),
                "density": float(data.mean()),
                "best_n_components": int(selector.best_n_components_),
                "best_alpha": float(selector.best_alpha_),
                "selection_rule": selector.selection_rule_,
                "resolved": bool(selector.resolved_),
                "tied_n_components": selector.tied_n_components_.tolist(),
                "converged_fraction": converged,
                "n_repeats": int(selector.n_repeats_),
            },
            indent=2,
        )
    )

    # Refit at the chosen (k, alpha) on the full matrix -- the sweep's fits
    # are all on 3/4 of the entries, so none of them is the model you want
    # to keep.
    print("\nrefitting on the full matrix at the selected settings...")
    final = SiGMoiD(
        n_components=selector.best_n_components_,
        alpha=selector.best_alpha_,
        max_iter=args.max_iter,
        device=devices[0],
        random_state=0,
    ).fit(data)
    save(final, args.out / "model.pt")
    print(f"saved -> {args.out / 'model.pt'}  (converged={final.converged_})")


# REQUIRED. Workers are started with `spawn` because CUDA and fork do not
# coexist, which means every worker re-executes this file. Without this
# guard each one would launch its own sweep.
if __name__ == "__main__":
    main()
