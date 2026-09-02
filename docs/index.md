# Binaria documentation

Binaria is a pre-release Python/PyTorch implementation of the published
SiGMoiD equations for binary matrix factorization.

The numerical defaults and model-selection behavior have not yet been
evaluated experimentally. Results should be treated as provisional and read
alongside the reported convergence diagnostics.

## Optimizers

Adam remains the default with `learning_rate=0.1`. Plain gradient ascent is
available through PyTorch's full-batch, zero-momentum SGD implementation:

```python
from binaria import SiGMoiD

model = SiGMoiD(
    n_components=4,
    optimizer="sgd",
    learning_rate=1e-4,
).fit(x)
```

This is not minibatch training. The optimizer choice also applies to
`transform()`, `score()`, and selector fits. The shown rate is illustrative,
not a validated or universally safe recommendation; stable rates depend on
matrix shape.

- The [project README](https://github.com/GecklesTheClown/binaria#readme)
  contains installation instructions and a minimal example.
- [Optimization diagnostics](diagnostics.md) shows how to record and plot
  objective, logit, factor-norm, and probability-saturation trajectories.
- [Interpreting model selection](selecting-k.md) describes the selector's
  outputs, regularization options, and current limitations.
