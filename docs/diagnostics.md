# Optimization diagnostics

The default `model.history_` stores iteration, training log-likelihood, and
elapsed time. More expensive summaries are opt-in because computing the full
logit and probability matrices adds work and may matter for large GPU fits.

```python
from binaria import History, SiGMoiD

model = SiGMoiD(
    n_components=4,
    optimizer="sgd",
    learning_rate=1e-4,
    max_iter=20_000,
    alpha=0.1,
    callbacks=[
        History(
            every=10,
            diagnostics=True,
            saturation_thresholds=(1e-2, 1e-3, 1e-6),
        )
    ],
).fit(x)

history = model.history_
history.iteration
history.objective
history.max_abs_logit
history.saturation_fraction[1e-6]
```

The final entry is always recorded, even when it does not fall on the
configured stride. Except for `gradient_norm`, each entry describes the model
*after* that iteration's update. Consequently,
`history.log_likelihood[-1] == model.log_likelihood_`.

## Convergence

After each update, Binaria computes the relative change in the penalized
objective:

\[
r_t =
\frac{\left|\mathcal L_t-\mathcal L_{t-1}\right|}
     {\left|\mathcal L_{t-1}\right|+10^{-12}}.
\]

A fit reports `converged_=True` only after `r_t < tol` for `patience`
consecutive iterations. The defaults are `tol=1e-6` and `patience=100`; a
non-qualifying iteration resets the streak. Iteration 1 establishes the
baseline, so the earliest possible convergence with the defaults is iteration
101. Reaching `max_iter` is not convergence, and unpenalized fits (`alpha=0`)
never claim a finite optimum through this rule.

## Recorded quantities

With `diagnostics=True`, `History` records:

- `log_likelihood`: unpenalized training log-likelihood.
- `objective`: `log_likelihood - alpha * l2_penalty`, the quantity optimized
  and monitored for convergence.
- `l2_penalty`: `||beta||_F^2 + ||energy||_F^2`, before multiplying by
  `alpha`.
- `max_abs_logit`: maximum of `|-(beta @ energy)|`.
- `beta_norm`, `energy_norm`, and `factor_norm`: the two Frobenius norms and
  their joint norm.
- `saturation_fraction[t]`: fraction of fitted probabilities below `t` or
  above `1 - t` for each requested threshold.
- `elapsed_time`: seconds since optimization started.

Set `record_gradient_norm=True` to also record the joint norm of the gradient
used for each update. This is separate from `diagnostics` and is based on the
gradient that produced the post-update state.

The factor norms are parameterization-dependent: transformations of the
factors can leave predictions unchanged. Interpret them together with the
logits, probabilities, and likelihood rather than as independently
identifiable scientific quantities.

## Plotting and tables

`as_dict()` returns copies of the active, equal-length columns. Saturation
thresholds are flattened into stable column names, so the result can be passed
directly to pandas:

```python
import matplotlib.pyplot as plt
import pandas as pd

frame = pd.DataFrame(model.history_.as_dict())

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
frame.plot(x="iteration", y="log_likelihood", ax=axes[0, 0])
frame.plot(x="iteration", y="objective", ax=axes[0, 1])
frame.plot(x="iteration", y="max_abs_logit", ax=axes[0, 2])
frame.plot(x="iteration", y=["beta_norm", "energy_norm"], ax=axes[1, 0])
frame.plot(x="iteration", y="saturation_fraction_1e-06", ax=axes[1, 1])
axes[1, 2].axis("off")
fig.tight_layout()
```

The saturation values are fractions in `[0, 1]`; multiply by 100 when labeling
an axis as a percentage.

Held-out log-likelihood is deliberately not computed by `History`: it requires
a separately defined split and potentially an inference procedure, rather than
only the current training state. Use model-selection or held-out evaluation
code to record it alongside these training diagnostics.

## Custom callbacks

Callbacks use the public `IterationState` value object:

```python
class Logger:
    def on_iteration(self, *, state, core, optimizer):
        if state.is_final or state.iteration % 100 == 0:
            print(state.iteration, state.objective, state.elapsed_time)
```

`state` and `core` refer to the same completed optimizer step. The state also
contains `log_likelihood`, the unscaled `l2_penalty`, and `is_final`.
