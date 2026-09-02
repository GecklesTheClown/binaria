# Interpreting model selection

`SiGMoiDSelector` evaluates candidate values of `k` and, optionally, L2
regularization strengths. The selector is implemented, but its ability to
recover an appropriate latent dimension has not yet been evaluated
experimentally.

## Inspect the result

```python
selector = SiGMoiDSelector([2, 3, 4, 6, 8]).fit(x)

selector.best_n_components_
selector.best_alpha_
selector.selection_rule_
selector.tied_n_components_
selector.cv_results_["converged"]
```

`best_n_components_` and `best_alpha_` are populated even when the configured
test does not separate one candidate from every rival. The accompanying fields
describe how the choice was made:

| Field | Meaning |
| --- | --- |
| `selection_rule_ == "separation"` | The leading cell passed the configured pairwise test against its rivals. |
| `selection_rule_ == "parsimony"` | The test did not identify a unique leader, so the selector used its simplicity tie-break. |
| `tied_n_components_` | Component counts not distinguished from the leading cell by the configured test. |
| `cv_results_["converged"]` | Whether relative penalized-objective change stayed below `tol` for `patience` consecutive iterations before `max_iter`. |

These are descriptions of the implemented procedure, not evidence that the
selected value is correct for a particular dataset. Inspect `cv_results_` and
`summary_`, and be cautious when fits have not converged.

## Regularization

For held-out likelihood, the default alpha grid is
`(0.0, 0.03, 0.1, 0.3, 1.0, 3.0)`. Pass an explicit `alpha_range` to change it,
or `alpha_range=(0.0,)` to evaluate only the unpenalized objective.

For separable data, an unpenalized Bernoulli likelihood may approach its
supremum as the factor magnitudes grow rather than attain a finite maximizer.
A positive L2 penalty makes the objective coercive, so it attains a finite
maximizer. This mathematical property does not establish which alpha is
appropriate for real data; the current default grid is provisional.

## Repeats and tests

The selector runs `n_repeats` fits per cell and performs its comparison once,
after all repeats finish. Partitions are shared across candidates within a
repeat so the comparisons are paired.

The `"t"` test requires at least two repeats. For the `"sign"` test,
`n_repeats` must be large enough that `0.5**n_repeats <= fdr`; otherwise
`fit` raises `ValueError`. More repeats may improve the ability to distinguish
small score differences, at additional compute cost.

## Current limitations

- No recovery, accuracy, stability, or performance results are currently
  reported.
- Optimizer convergence does not by itself validate the selected component
  count.
- The leading candidate is chosen from the same scores used in the subsequent
  comparisons, so the reported separation may be optimistic.
- A selection at either edge of the candidate grid leaves behavior outside the
  grid unknown.

## Keep a decision paper trail

Enable audit mode when the selected value must be reconstructable:

```python
selector = SiGMoiDSelector(
    [2, 3, 4, 6, 8],
    audit=True,
).fit(x)
selector.save_audit("selection-audit.json")
```

The JSON contains the resolved grid and settings, a digest of the input
matrix, every fit's seeds and scores, convergence, aggregate rankings, raw
and adjusted pairwise p-values, per-fit and aggregate training times, tied
candidates, and the exact rule that produced the selected `(k, alpha)` pair.
It contains no copy of the input matrix itself.
