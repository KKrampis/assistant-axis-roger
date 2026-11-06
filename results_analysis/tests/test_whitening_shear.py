"""Tests for results_analysis.canonical_angles.whitening.fit_shear and
the spec parser.  Whitening (raw, soft_K, lw, oas) tests are added here
opportunistically alongside the shear tests."""
from __future__ import annotations

import numpy as np
import pytest

from results_analysis.canonical_angles.whitening import (
    WhiteningBasis,
    fit_shear,
    fit_whitening,
    parse_whitening_spec,
)


@pytest.fixture(autouse=True)
def deterministic():
    np.random.seed(0)


# ---------------------------------------------------------------------------
# fit_shear: structural tests
# ---------------------------------------------------------------------------

def test_fit_shear_L0_is_identity():
    # Project canonical shape: D=5120, n=30, columns-as-vectors.
    A_g = np.random.randn(5120, 30)
    A_n = np.random.randn(5120, 30)
    basis = fit_shear(A_g, A_n, L=0)
    assert basis.method == "raw"
    X = np.random.randn(50, 5120)
    np.testing.assert_array_equal(basis.apply(X), X)


def test_fit_shear_returns_basis_with_correct_shape():
    # Project canonical: (D=200, n=20) columns-as-vectors.
    A_g = np.random.randn(200, 20)
    A_n = np.random.randn(200, 20)
    basis = fit_shear(A_g, A_n, L=5)
    assert basis.method == "soft_shear"
    assert basis.L == 5
    assert basis.shear_basis.shape == (200, 10)        # (D, 2L)
    assert basis.shear_factors.shape == (10,)          # (2L,)
    assert basis.canonical_angles_rad.shape == (20,)   # min(n_g, n_n)


def test_fit_shear_apply_preserves_shape():
    # D=200 ambient, n=8 vectors per subspace.
    A_g = np.random.randn(200, 8)
    A_n = np.random.randn(200, 8)
    basis = fit_shear(A_g, A_n, L=3)
    X = np.random.randn(7, 200)
    out = basis.apply(X)
    assert out.shape == X.shape
    # 1-D input
    v = np.random.randn(200)
    out_v = basis.apply(v)
    assert out_v.shape == (200,)


def test_fit_shear_validates_L():
    A_g = np.random.randn(200, 5)
    A_n = np.random.randn(200, 5)
    with pytest.raises(ValueError, match="L must be >= 0"):
        fit_shear(A_g, A_n, L=-1)
    with pytest.raises(ValueError, match="exceeds min"):
        fit_shear(A_g, A_n, L=10)


def test_fit_shear_validates_vectors_as():
    A_g = np.random.randn(200, 8)
    A_n = np.random.randn(200, 8)
    with pytest.raises(ValueError, match="vectors_as must be"):
        fit_shear(A_g, A_n, L=2, vectors_as="bogus")


def test_fit_shear_rows_and_cols_orientations_agree():
    """vectors_as='rows' and vectors_as='cols' on transposed inputs give
    identical bases."""
    np.random.seed(42)
    A_g_cols = np.random.randn(200, 10)            # (D, n)
    A_n_cols = np.random.randn(200, 10)
    A_g_rows = A_g_cols.T                          # (n, D)
    A_n_rows = A_n_cols.T

    basis_cols = fit_shear(A_g_cols, A_n_cols, L=4, vectors_as="cols")
    basis_rows = fit_shear(A_g_rows, A_n_rows, L=4, vectors_as="rows")

    X = np.random.randn(20, 200)
    np.testing.assert_allclose(basis_cols.apply(X), basis_rows.apply(X),
                                atol=1e-10, rtol=1e-8)


def test_fit_shear_volume_preserving():
    """The shear is volume-preserving in each sheared (e+, e-) plane:
    sqrt(tan(theta/2)) * sqrt(1/tan(theta/2)) = 1.
    """
    np.random.seed(1)
    # D >> n so canonical angles are well above 0 (random subspaces in
    # high-D are nearly orthogonal).
    A_g = np.random.randn(500, 8)
    A_n = np.random.randn(500, 8)
    L = 3
    basis = fit_shear(A_g, A_n, L=L)
    factors = basis.shear_factors   # (2L,)
    # multipliers[i] = sqrt(tan(theta_i / 2))         for squash directions
    # multipliers[L+i] = sqrt(1 / tan(theta_i / 2))   for stretch directions
    # Their product is always 1 (volume preservation).
    multipliers = 1.0 + factors
    for i in range(L):
        product = multipliers[i] * multipliers[L + i]
        assert product == pytest.approx(1.0, abs=1e-10), \
            f"pair {i}: squash * stretch = {product}, expected 1.0"


def test_fit_shear_reduces_canonical_angle():
    """After applying the shear to the source subspaces, the top-L
    most-aligned canonical pairs become exactly orthogonal (theta = pi/2,
    cos = 0).  Since numpy's SVD returns sigmas in descending order
    (sigma=cos(theta), so smaller theta = larger sigma), the L pairs that
    just got driven to theta=pi/2 land at the END of sigma_after.
    """
    np.random.seed(2)
    A_g_cols = np.random.randn(500, 6)             # (D, n) cols
    A_n_cols = np.random.randn(500, 6)
    basis = fit_shear(A_g_cols, A_n_cols, L=3)

    # Apply expects (n, D) entity matrices; transpose to that convention.
    A_g_sheared = basis.apply(A_g_cols.T)          # (n, D)
    A_n_sheared = basis.apply(A_n_cols.T)
    Qa, _ = np.linalg.qr(A_g_sheared.T)            # back to (D, n) for QR
    Qb, _ = np.linalg.qr(A_n_sheared.T)
    sigma_after = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    # The 3 sheared pairs are now at theta=pi/2 -> sigma=0; they sit at
    # the *end* of the descending-sorted sigma list.
    np.testing.assert_allclose(sigma_after[-3:], 0.0, atol=1e-10)
    # The 3 untouched pairs (originally larger theta) keep their angles.
    sigma_before = np.linalg.svd(
        np.linalg.qr(A_g_cols)[0].T @ np.linalg.qr(A_n_cols)[0],
        compute_uv=False,
    )
    # The original pairs at indices 3..6 (sorted descending) were untouched:
    np.testing.assert_allclose(sigma_after[:3], sigma_before[3:6], atol=1e-10)


# ---------------------------------------------------------------------------
# parse_whitening_spec
# ---------------------------------------------------------------------------

def test_parse_whitening_spec_existing_regimes():
    assert parse_whitening_spec("raw") == ("raw", None)
    assert parse_whitening_spec("soft_K=4") == ("soft_K", 4)
    assert parse_whitening_spec("soft_K_4") == ("soft_K", 4)
    assert parse_whitening_spec("lw") == ("lw", None)
    assert parse_whitening_spec("oas") == ("oas", None)


def test_parse_whitening_spec_shear():
    assert parse_whitening_spec("soft_shear=5") == ("soft_shear", 5)
    assert parse_whitening_spec("soft_shear_5") == ("soft_shear", 5)
    assert parse_whitening_spec("shear=5") == ("soft_shear", 5)
    assert parse_whitening_spec("shear_2") == ("soft_shear", 2)


def test_parse_whitening_spec_invalid():
    with pytest.raises(ValueError):
        parse_whitening_spec("nonsense")
    with pytest.raises(ValueError):
        parse_whitening_spec("soft_K")  # missing value
    with pytest.raises(ValueError):
        parse_whitening_spec("shear")   # missing value


# ---------------------------------------------------------------------------
# Integration: shear and soft_K compose
# ---------------------------------------------------------------------------

def test_shear_and_soft_K_compose_apply():
    np.random.seed(3)
    D = 200
    pool = np.random.randn(500, D)         # (n_pool, D) -- soft-K pool
    A_g = np.random.randn(D, 10)           # (D, n) -- shear subspaces
    A_n = np.random.randn(D, 10)
    shear = fit_shear(A_g, A_n, L=3)
    soft_k = fit_whitening("soft_K", pool, K=4)

    X = np.random.randn(50, D)
    # Compose: shear then whiten
    Y = soft_k.apply(shear.apply(X))
    assert Y.shape == X.shape
    # Verify the two regimes are different transforms (sanity)
    assert not np.allclose(shear.apply(X), soft_k.apply(X))


# ---------------------------------------------------------------------------
# WhiteningBasis backward compat: existing fields untouched
# ---------------------------------------------------------------------------

def test_whitening_basis_has_new_fields_with_safe_defaults():
    """Adding shear fields didn't break existing soft_K / raw / lw bases."""
    pool = np.random.randn(100, 50)
    raw = fit_whitening("raw", pool)
    assert raw.method == "raw"
    assert raw.shear_basis is None
    assert raw.shear_factors is None
    assert raw.L is None

    sk = fit_whitening("soft_K", pool, K=2)
    assert sk.method == "soft_K"
    assert sk.shear_basis is None
    assert sk.shear_factors is None
