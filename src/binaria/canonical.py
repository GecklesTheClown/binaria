"""
Putting a fitted factorization into a canonical form, for comparison.

``beta`` and ``energy`` are not identified individually: predictions
depend only on their product, so ``beta -> beta M``, ``energy -> M^-1
energy`` is the same model for any invertible ``M``. An L2 penalty removes
most of that by forcing the balanced factorization
(``beta.T @ beta == energy @ energy.T``), but balance survives any
orthogonal ``Q``, leaving a rotational freedom. Two runs of one fit can
therefore return visibly different factors that are the same model.

This module picks one representative of that orbit, deterministically, so
two fits can be laid side by side.

A post-hoc view: the transformation is a gauge transformation, so
predictions, samples and log-likelihoods are unchanged, and distance-based
clustering on the embedding was rotation-invariant already. It buys
reproducibility, not interpretability -- "component 2 means X" is still
not a statement about the data, since the SVD convention used here is one
choice among several with equal standing, as with PCA loadings.

Rotation is removed outright, and sign by convention. Ties are *not*
removable: where two singular values coincide the plane they span is
genuinely rotation-invariant, so no convention can pick a basis inside it.
Near-ties are the practical case, since those directions are
ill-conditioned; they are reported through ``relative_gaps`` and ``tied``.
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["CanonicalFactors", "canonicalize"]


@dataclass(frozen=True)
class CanonicalFactors:
    """
    A fitted factorization in canonical form, with stability diagnostics.

    Attributes
    ----------
    beta : ndarray of shape (n_samples, k)
        ``U @ sqrt(S)``. The balanced, sign-fixed left factor.
    energy : ndarray of shape (k, n_features)
        ``sqrt(S) @ Vt``.
    singular_values : ndarray of shape (k,)
        Singular values of the product, descending. The magnitude of each
        component's contribution to the natural parameters.
    relative_gaps : ndarray of shape (k,)
        ``(s[i] - s[i+1]) / s[i]``, with the value past the last component
        taken as zero. This is the quantity that governs how stable each
        direction is: perturbation theory ties the sensitivity of a
        singular vector to its separation from its neighbour, so a small
        gap means the direction moves a lot under a small change in data.
    tied : ndarray of shape (k,) of bool
        Components sitting in a near-tied block, i.e. whose gap to a
        neighbour is below the tolerance. Individual components flagged
        here are **not** stable across refits; only the span of the block
        is. Compare those jointly or not at all.
    """

    beta: np.ndarray
    energy: np.ndarray
    singular_values: np.ndarray
    relative_gaps: np.ndarray
    tied: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.beta.shape[1])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_components={self.n_components}, "
            f"n_tied={int(self.tied.sum())})"
        )


def canonicalize(
    beta: np.ndarray, energy: np.ndarray, *, tie_tolerance: float = 0.01
) -> CanonicalFactors:
    """
    Rewrite ``(beta, energy)`` in a form two runs can agree on.

    Parameters
    ----------
    beta : array of shape (n_samples, k)
        Typically ``model.embedding_``.
    energy : array of shape (k, n_features)
        Typically ``model.components_``.
    tie_tolerance : float, default=0.01
        Relative-gap threshold below which neighbouring components are
        treated as tied. 1% is a starting point, not a derived constant --
        how large a gap you need depends on how much your data would move
        under resampling, so widen it if refits disagree.

    Returns
    -------
    CanonicalFactors

    Notes
    -----
    The product is preserved to floating-point accuracy, so predictions,
    samples and likelihoods are unchanged to that accuracy. Safe to apply
    at any point, including to a model loaded from disk.

    Uses the SVD of ``beta @ energy`` rather than of either factor alone:
    the gauge freedom acts on the pair, and only their product is
    invariant, so the product is the only thing that can define a canonical
    form.
    """
    beta = np.asarray(beta, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    if beta.ndim != 2 or energy.ndim != 2:
        raise ValueError(f"expected 2-D factors, got {beta.ndim}-D and {energy.ndim}-D")
    if beta.shape[1] != energy.shape[0]:
        raise ValueError(
            f"inner dimensions must match: beta is {beta.shape}, energy is {energy.shape}"
        )

    n_components = beta.shape[1]
    left, values, right = np.linalg.svd(beta @ energy, full_matrices=False)
    left = left[:, :n_components]
    values = values[:n_components]
    right = right[:n_components]

    # Sign convention: force the largest-magnitude entry of each left
    # vector positive, flipping the matching right vector to compensate so
    # the product is untouched. Same rule sklearn's PCA uses.
    dominant = np.argmax(np.abs(left), axis=0)
    signs = np.sign(left[dominant, np.arange(n_components)])
    signs[signs == 0] = 1.0  # an all-zero column has no preferred sign
    left = left * signs
    right = right * signs[:, None]

    scale = np.sqrt(values)
    canonical_beta = left * scale
    canonical_energy = scale[:, None] * right

    # Gap to the next singular value, relative to this one. The value past
    # the final component is zero -- the product has rank k exactly -- so
    # the last gap is 1 unless that component has itself collapsed.
    following = np.concatenate([values[1:], [0.0]])
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_gaps = np.where(values > 0, (values - following) / values, 0.0)

    # A small gap makes BOTH neighbours unstable, not just the earlier one,
    # so the flag propagates in both directions.
    tied = relative_gaps < tie_tolerance
    tied[1:] |= relative_gaps[:-1] < tie_tolerance

    return CanonicalFactors(
        beta=canonical_beta,
        energy=canonical_energy,
        singular_values=values,
        relative_gaps=relative_gaps,
        tied=tied,
    )


def subspace_distance(left: np.ndarray, right: np.ndarray) -> float:
    """
    How far apart two embeddings are, ignoring the rotation between them.

    The quantity to compare when asking whether two fits found the same
    structure. Comparing ``beta`` entrywise answers a different and much
    less interesting question, since any orthogonal rotation gives the same
    model -- and where components are tied, even canonical form will not
    make them line up.

    Returns the largest principal angle between the column spaces, in
    radians: 0 for identical subspaces, ``pi/2`` for orthogonal ones.

    Accurate to about ``sqrt(eps)``, i.e. ~1e-8, near zero. That is
    inherent to recovering a small angle through ``arccos``: if the cosine
    is ``1 - d``, the angle is roughly ``sqrt(2d)``, so rounding at 1e-16
    in the cosine surfaces as 1e-8 in the angle. Identical subspaces
    therefore report ~1e-8 rather than exactly 0 -- compare against a
    tolerance, not against zero.
    """
    left_basis = np.linalg.qr(np.asarray(left, dtype=np.float64))[0]
    right_basis = np.linalg.qr(np.asarray(right, dtype=np.float64))[0]
    # Singular values of the basis overlap are the cosines of the principal
    # angles; clipping guards the arccos against 1 + 1e-16.
    cosines = np.linalg.svd(left_basis.T @ right_basis, compute_uv=False)
    return float(math.acos(min(1.0, max(-1.0, float(cosines.min())))))
