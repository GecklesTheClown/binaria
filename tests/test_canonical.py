"""
Canonical form: can two fits be compared?

The claim being tested is narrow and worth stating precisely. An L2
penalty leaves a rotational freedom -- ``beta -> beta Q``,
``energy -> Q.T energy`` for orthogonal ``Q`` -- so two runs of the same
fit can return different-looking factors that are the same model.
``canonicalize`` picks one representative, deterministically.

The load-bearing test is that two *different* gauge transformations of one
model canonicalize to the same thing. That is also the first empirical
confirmation in this suite that the residual freedom really is O(K) and
not something larger.
"""

import numpy as np
import pytest
import torch
from scipy.stats import ortho_group

from binaria import SiGMoiD
from binaria.canonical import canonicalize, subspace_distance


def _factors(n_samples: int = 30, n_features: int = 18, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, k)), rng.normal(size=(k, n_features))


def _rotate(beta, energy, seed=0):
    """An orthogonal gauge transformation: same model, different factors."""
    q = ortho_group.rvs(beta.shape[1], random_state=seed)
    return beta @ q, q.T @ energy


def _general_gauge(beta, energy, seed=0):
    """A full GL(K) transformation -- not orthogonal, so unbalanced too."""
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(beta.shape[1], beta.shape[1]))
    while abs(np.linalg.det(m)) < 0.1:
        m = rng.normal(size=(beta.shape[1], beta.shape[1]))
    return beta @ m, np.linalg.inv(m) @ energy


# --- the invariant that makes this safe to apply -------------------------


def test_the_product_is_unchanged() -> None:
    # A gauge transformation, so every probability, every sample and every
    # likelihood is untouched. This is what makes it safe to apply to
    # anything at any time.
    beta, energy = _factors()
    result = canonicalize(beta, energy)
    assert np.allclose(result.beta @ result.energy, beta @ energy, atol=1e-10)


def test_a_fitted_model_keeps_its_predictions() -> None:
    torch.manual_seed(0)
    b = torch.randn(40, 3, dtype=torch.float64)
    e = torch.randn(3, 20, dtype=torch.float64)
    data = torch.bernoulli(torch.sigmoid(-(b @ e))).numpy()

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = SiGMoiD(n_components=3, alpha=0.3, max_iter=200, random_state=0).fit(data)

    result = canonicalize(model.embedding_, model.components_)
    before = model.embedding_ @ model.components_
    assert np.allclose(result.beta @ result.energy, before, atol=1e-8)


# --- the point of the exercise -------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotated_factors_canonicalize_to_the_same_thing(seed: int) -> None:
    # THE test. Two orthogonal gauge transformations of one model produce
    # visibly different factors; canonical form must collapse them onto
    # each other. This is the empirical confirmation that the freedom left
    # by an L2 penalty is exactly O(K).
    beta, energy = _factors(seed=seed)
    left = canonicalize(*_rotate(beta, energy, seed=seed))
    right = canonicalize(*_rotate(beta, energy, seed=seed + 100))

    assert np.allclose(left.beta, right.beta, atol=1e-8)
    assert np.allclose(left.energy, right.energy, atol=1e-8)
    assert np.allclose(left.singular_values, right.singular_values, atol=1e-8)


def test_a_full_gl_transformation_also_collapses() -> None:
    # Stronger: even an unbalanced, non-orthogonal gauge transformation --
    # the kind an unpenalized fit can wander into -- lands in the same
    # place, because the canonical form is defined from the product alone.
    beta, energy = _factors(seed=5)
    original = canonicalize(beta, energy)
    transformed = canonicalize(*_general_gauge(beta, energy, seed=5))

    assert np.allclose(original.beta, transformed.beta, atol=1e-8)
    assert np.allclose(original.energy, transformed.energy, atol=1e-8)


def test_canonicalizing_twice_changes_nothing() -> None:
    beta, energy = _factors(seed=3)
    once = canonicalize(beta, energy)
    twice = canonicalize(once.beta, once.energy)
    assert np.allclose(once.beta, twice.beta, atol=1e-10)
    assert np.allclose(once.energy, twice.energy, atol=1e-10)


def test_sign_convention_is_deterministic_and_applied() -> None:
    # Flipping a component's sign is a gauge transformation too, so the
    # convention has to survive it. The rule is that the
    # largest-magnitude entry of each left factor column is positive.
    beta, energy = _factors(seed=7)
    flip = np.diag([1.0, -1.0, 1.0])
    flipped = canonicalize(beta @ flip, flip @ energy)
    original = canonicalize(beta, energy)

    assert np.allclose(original.beta, flipped.beta, atol=1e-10)
    dominant = np.argmax(np.abs(original.beta), axis=0)
    assert (original.beta[dominant, np.arange(original.n_components)] > 0).all()


# --- balance and diagnostics ---------------------------------------------


def test_canonical_form_is_balanced() -> None:
    # beta.T @ beta == energy @ energy.T, which is where the L2 penalty
    # was pushing anyway -- so canonicalizing an unpenalized fit also
    # rescues the arbitrary split of magnitude between the two factors.
    beta, energy = _factors(seed=11)
    result = canonicalize(*_general_gauge(beta, energy, seed=11))
    assert np.allclose(result.beta.T @ result.beta, result.energy @ result.energy.T, atol=1e-8)


def test_singular_values_describe_the_product() -> None:
    beta, energy = _factors(seed=13)
    result = canonicalize(beta, energy)
    expected = np.linalg.svd(beta @ energy, compute_uv=False)[: result.n_components]
    assert np.allclose(result.singular_values, expected, atol=1e-10)
    assert (np.diff(result.singular_values) <= 1e-12).all(), "must be descending"


def test_near_tied_components_are_flagged_and_well_separated_ones_are_not() -> None:
    # Constructed with two nearly equal singular values, so the plane they
    # span is effectively rotation-invariant and neither direction alone
    # is reproducible.
    rng = np.random.default_rng(0)
    left = np.linalg.qr(rng.normal(size=(40, 3)))[0]
    right = np.linalg.qr(rng.normal(size=(20, 3)))[0].T
    values = np.array([10.0, 3.0, 2.999])  # last two are tied

    result = canonicalize(left * values, right)

    assert not result.tied[0], "a well-separated component should not be flagged"
    assert result.tied[1] and result.tied[2], "both members of a tied block must be flagged"


def test_relative_gaps_are_in_range_and_ordered_sensibly() -> None:
    beta, energy = _factors(seed=17)
    result = canonicalize(beta, energy)
    assert ((result.relative_gaps >= 0.0) & (result.relative_gaps <= 1.0)).all()
    # The product has rank k exactly, so the final component is separated
    # from the null space by its whole magnitude.
    assert result.relative_gaps[-1] == pytest.approx(1.0)


# --- comparing two fits ---------------------------------------------------


def test_subspace_distance_is_zero_for_a_rotation_and_large_for_noise() -> None:
    # The quantity to compare when asking whether two fits found the same
    # structure -- entrywise comparison of beta answers a different and
    # much less interesting question.
    beta, energy = _factors(seed=19)
    rotated, _ = _rotate(beta, energy, seed=19)
    # 1e-6, not 0: recovering a small angle through arccos is only accurate
    # to about sqrt(eps), so identical subspaces report ~1e-8. Documented
    # on the function.
    assert subspace_distance(beta, rotated) < 1e-6

    unrelated, _ = _factors(seed=20)
    assert subspace_distance(beta, unrelated) > 0.1


# --- input validation -----------------------------------------------------


def test_mismatched_factors_raise() -> None:
    with pytest.raises(ValueError, match="inner dimensions"):
        canonicalize(np.zeros((10, 3)), np.zeros((4, 8)))
    with pytest.raises(ValueError, match="2-D"):
        canonicalize(np.zeros(10), np.zeros((3, 8)))
