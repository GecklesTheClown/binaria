# Binaria

Binaria is an independent Python/PyTorch implementation of the published
**SiGMoiD** equations (Zhao, Plata & Dixit, 2021):

```text
π = σ(−βE)
```

It provides a scikit-learn-style estimator for binary matrix factorization
and utilities for selecting the number of components.

> [!CAUTION]
> Binaria is pre-release software. Its numerical defaults and model-selection
> behavior have not yet been evaluated experimentally. Treat fitted models and
> selected component counts as provisional, and inspect the reported
> convergence diagnostics.

## Install

Binaria is not yet on PyPI. From a checkout:

```bash
uv sync
```

To add Binaria to another uv project:

```bash
uv add git+https://github.com/GecklesTheClown/binaria.git
```

Python 3.11 or newer is required. CUDA devices are supported but optional.

## Quick start

```python
import numpy as np
from binaria import SiGMoiD, SiGMoiDSelector

x = np.load("data.npy")  # (n_samples, n_features), entries in {0, 1}

model = SiGMoiD(n_components=4).fit(x)
model.components_  # (n_components, n_features)
model.embedding_  # (n_samples, n_components)

selector = SiGMoiDSelector([2, 3, 4, 6, 8], audit=True).fit(x)
selector.best_n_components_
selector.selection_rule_
selector.cv_results_["converged"].mean()
selector.save_audit("results/selection-audit.json")
```

Adam is the default optimizer with `learning_rate=0.1`. Full-batch SGD with
zero momentum is also available; set its learning rate explicitly because the
default rate is intended for Adam:

```python
sgd_model = SiGMoiD(
    n_components=4,
    optimizer="sgd",
    learning_rate=1e-4,
).fit(x)
```

This is plain gradient ascent over the entire matrix, not minibatch training.
The optimizer setting also applies to `transform()`, `score()`, and selector
fits. The shown `1e-4` is illustrative, not a validated or universally safe
recommendation; stable rates depend on matrix shape.

For optimization diagnostics, attach an opt-in history callback:

```python
from binaria import History, SiGMoiD

model = SiGMoiD(
    n_components=4,
    alpha=0.1,
    callbacks=[History(every=10, diagnostics=True)],
).fit(x)

history = model.history_.as_dict()
```

This records the training likelihood, penalized objective, factor norms,
maximum absolute logit, and probability-saturation fractions over time. See
[Optimization diagnostics](docs/diagnostics.md) for definitions and plotting.
By default, convergence requires 100 consecutive iterations with relative
penalized-objective change below `tol=1e-6`.

`best_n_components_` is always populated. Read it together with
`selection_rule_`, `tied_n_components_`, and the convergence values in
`cv_results_`. See [Interpreting model selection](docs/selecting-k.md).

## Parallel sweeps

```python
from binaria import SiGMoiDSelector
from binaria.executors import MultiGPUExecutor

with MultiGPUExecutor() as pool:
    selector = SiGMoiDSelector(
        [2, 3, 4, 6, 8],
        executor=pool,
        checkpoint="results/sweep.ckpt",
    ).fit(x)
```

`MultiGPUExecutor` uses the visible CUDA devices. A checkpoint lets an
interrupted sweep reuse completed fits. See
[`examples/multi_gpu_sweep.py`](examples/multi_gpu_sweep.py) for a complete
script.

## Documentation

NumPy input uses the CPU unless `device` is set. A PyTorch tensor stays on
its existing device.

The documentation is in [`docs/`](docs/index.md) and can be built with:

```bash
uv run mkdocs build --strict
```

## Citing

Please cite the [original SiGMoiD paper](https://doi.org/10.1371/journal.pcbi.1009275)
and the software metadata in [`CITATION.cff`](CITATION.cff).

## License

Binaria is distributed under the [BSD 3-Clause License](LICENSE).
