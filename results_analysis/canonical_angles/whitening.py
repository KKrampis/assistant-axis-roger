"""PCA-style whitening (and CA-style shearing) regimes for analysis tools.

Five regimes are supported:

- ``raw``        identity (no whitening at all)
- ``soft_K``     rescale the top-K right-singular components of a centered pool
                 so that their effective standard deviation matches sigma_{K+1};
                 components K+1..D are left untouched.  This is the historical
                 "soft-K" PCA whitening used elsewhere in the project (and by
                 ``axis_judge_correlation.py``).
- ``lw``         Ledoit-Wolf shrinkage covariance, then ``cov^{-1/2}`` applied
                 as the whitening matrix.
- ``oas``        Oracle Approximating Shrinkage; same structure as ``lw``.
- ``soft_shear`` truncated soft-shear that orthogonalizes the top-L
                 canonical-angle pairs of two subspaces (typically the
                 goal vs no-goal residual subspaces).  Volume-preserving:
                 each top-L plane is squashed in one direction and stretched
                 in the orthogonal direction by ``sqrt(tan(theta_i / 2))``,
                 rotating ``e+`` and ``e-`` into orthogonality.  Lower
                 (K+1..n) canonical pairs are left untouched.

A fitted basis is a :class:`WhiteningBasis` and is applied to a vector or
``(n, D)`` matrix via ``basis.apply(X)``.  The four whitening regimes and
the shear regime share the apply API, so callers can compose them freely::

    M_done = (fit_whitening("soft_K", pool, K=2)
              .apply(fit_shear(A_goal, A_nogoal, L=5).apply(M)))

Notes:

- The pool passed to :func:`fit_whitening` is mean-centered internally before
  the SVD / covariance is computed.  The fitted transform is then applied to
  vectors that may be expressed in any frame (the linear scaling is
  translation-equivariant for soft-K; for ``lw``/``oas`` you should center
  consistently or the cov^{-1/2} mapping mixes mean information into the
  output).
- Soft-K with K >= rank(pool) is a no-op (returns the identity transform).
- Soft-shear with L=0 is a no-op (returns the identity transform).
- The shear is fit on the mean-anchored subspaces as supplied (no centering
  applied to ``A_goal`` / ``A_nogoal`` -- caller controls the frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WhiteningMethod = str  # 'raw' | 'soft_K' | 'lw' | 'oas' | 'soft_shear'


# ----------------------------------------------------------------------
# Project-wide defaults (Apr 2026)
# ----------------------------------------------------------------------
# These are the "if you have to pick one number" defaults for analyses
# going forward.  Selection history:
#
# May 2026 (initial): based on the LKM grid sweep at three (slot, layer)
# configurations -- (3, 25), (0, 26), (0, 49) -- on the 33 desc+inst +
# 12 response = 45-axis set, with the inst-tiebreak desc/inst weighting.
# At that time L=2 was the single most-defensible primary default; the
# optimum varied by (slot, layer): slot 3 preferred L=2 with K=0; slot 0
# preferred a small shear (L=0..1) with K=2.  See
# ``/tmp/lkm_grid_3variants_results.json`` for the underlying numbers.
#
# May 11 2026 (bumped to L=3): re-derived from the new 35-axis di-cohort
# shear_l_sweep at slots 3 / 6 / 7 with the canonical-mean-blend overlay
# (`results_analysis.shear_l_sweep` + `judge_score_combine.cohort_mean_curves`,
# 0.80·rs + 0.20·di per axis, 5x cross-axis weight on primary axes).
# Per-axis paired t-tests over the 35 axes show:
#
#   * slot 3: L=1..L=5 statistically all-tied (broad plateau).
#   * slot 6: L=1 ≈ L=3 (paired-t p=0.29), L=2 dip is real (p=0.019),
#             L=3 -> L=4 is a highly-significant cliff (p=0.0006).
#   * slot 7: L=1 ≈ L=2 ≈ L=3 (all paired-t p > 0.6); L=3 -> L=4 cliff
#             again (p=0.0004).
#
# L=3 is the cohort-defensible single default across all three slots:
# at-or-above optimum everywhere, never significantly worse than L=1 in
# any cohort, and the goal/no-goal subspace cleanliness tie-breaker
# (more aligned pairs orthogonalised = cleaner subspace separation)
# prefers higher L when ρ is tied.  All three slots agree the
# universally-harmful transition is L=3 -> L=4, so L=3 is the highest
# "safe" value with no significant harm vs L=1.
DEFAULT_SOFT_SHEAR_L: int = 3
"""Default truncation level for pooled soft-shear (combined r+t goal/no-goal).

Used as the soft-shear default when scripts pick a single L without
sweeping.  At-or-above optimum at slots 3, 6, 7 in the May 11 2026
shear_l_sweep paired-t analysis; the L=3 -> L=4 transition is the
universally-harmful cliff across all tested slots.  Was ``2`` until
May 11 2026; see selection-history comment block above for the
derivation.
"""

DEFAULT_SOFT_K: int = 2
"""Default soft-K whitening level when not using soft-shear.

Used for backward-compatible script defaults that previously hardcoded
K=3.  K=2 wins on the no-shear baseline at slot 0 (both layers tested);
at slot 3 it's slightly behind K=0 (because soft-shear alone wins there)
but better than K=3.  Picking 2 over 3 for a small but consistent
improvement (~0.005 ρ on the 45-axis evaluation, monotonic across slots).
Was 3 historically (peak from the older K-sweep parabola fit done
without shear in the toolbox).  One script keeps 3 on purpose:
:data:`OPTIMAL_AXIS_SOFT_K`.
"""

OPTIMAL_AXIS_SOFT_K: int = 3
"""Default soft-K for ``results_analysis/optimal_axis_for_judge.py``
(its ``DEFAULT_WHITENING`` is ``f"soft_K={OPTIMAL_AXIS_SOFT_K}"``).

Deliberately not :data:`DEFAULT_SOFT_K`: every recorded run of that script
(the five ``roger/optimal_axis/truthful_vs_deceitful_*_softK3_*`` outputs of
2026-05-08 to 05-11) used soft_K=3, its default since 2026-05-01, and the
script's defaults are kept able to reproduce them (walk-through D6, Roger,
2026-10-12: align the script to what its recorded runs used).  Unifying the
project on 2 is a separate decision; a run at 2 writes a ``softK2`` output
directory, so it would not overwrite the records.
"""

DEFAULT_WHITENING_SPEC: str = "soft_shear=3"
"""Recommended primary whitening regime for new scripts.

Pass this as the ``--whitening`` argument default in scripts that support
the soft-shear regime via :func:`fit_shear` and the goal/no-goal
subspaces from :func:`canonical_angles.data.build_goal_nogoal_subspaces`.
Scripts that only support :func:`fit_whitening` should default to
``soft_K=2``; see :data:`DEFAULT_SOFT_K`.

Was ``"soft_shear=2"`` until May 11 2026; bumped to track
:data:`DEFAULT_SOFT_SHEAR_L`'s same-day revision.  See the selection-
history comment block above the constants for derivation.
"""


@dataclass
class WhiteningBasis:
    """Fitted whitening (or shearing) transform.

    Most callers should treat this as opaque and only call :meth:`apply`.
    The fields are exposed for diagnostics and caching keys.
    """

    method: WhiteningMethod
    K: int | None = None
    L: int | None = None
    # soft_K state: top-K right singular vectors and per-PC scaling factors.
    Vt: np.ndarray | None = None       # (K, D)
    scales: np.ndarray | None = None   # (K,)  effective sigma_target / sigma_k
    # lw / oas state: cov^{-1/2} as a (D, D) matrix.
    cov_inv_sqrt: np.ndarray | None = None
    # soft_shear state: the (D, 2L) basis ``[e+_1..L, e-_1..L]`` and the
    # (2L,) per-direction multiplicative factors.  Apply via
    # ``X + (X @ E) * factors @ E.T`` (volume-preserving).
    shear_basis: np.ndarray | None = None       # (D, 2L)
    shear_factors: np.ndarray | None = None     # (2L,)
    # Diagnostic: the canonical angles (rad) of the input pair, length min(n_g, n_n).
    canonical_angles_rad: np.ndarray | None = None

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted transform to a vector or (n, D) matrix.

        Returns a new array of the same shape; ``X`` is not modified.
        """
        if self.method == "raw":
            return X
        if X.ndim == 1:
            return self.apply(X[None, :])[0]

        if self.method == "soft_K":
            # X_w = X + sum_k (scale_k - 1) * (X . PC_k) * PC_k
            # i.e. scale only the top-K PC components, leave the residual fixed.
            assert self.Vt is not None and self.scales is not None
            coefs = X @ self.Vt.T                              # (n, K)
            adjustments = (coefs * (self.scales - 1.0)) @ self.Vt  # (n, D)
            return X + adjustments

        if self.method in ("lw", "oas"):
            assert self.cov_inv_sqrt is not None
            # Apply on the right: equivalent to multiplying each row by cov^{-1/2}.
            return X @ self.cov_inv_sqrt.T

        if self.method == "soft_shear":
            assert self.shear_basis is not None and self.shear_factors is not None
            # X' = X + (X @ E .* factors) @ E.T  -- volume-preserving rotation
            # confined to the L sheared CA pairs.
            return X + (X @ self.shear_basis * self.shear_factors[None, :]) \
                       @ self.shear_basis.T

        raise ValueError(f"Unknown whitening method {self.method!r}")


def fit_whitening(method: WhiteningMethod,
                  pool: np.ndarray,
                  K: int | None = None) -> WhiteningBasis:
    """Fit a whitening basis on a pool of shape ``(n_samples, hidden)``.

    Parameters
    ----------
    method : 'raw' | 'soft_K' | 'lw' | 'oas'
    pool   : (n, D) numpy array of pool vectors (will be mean-centered before fit)
    K      : required when ``method == 'soft_K'``; number of top PCs to rescale

    Returns
    -------
    WhiteningBasis
        Use ``basis.apply(X)`` to whiten new vectors / matrices.
    """
    if method == "raw":
        return WhiteningBasis(method="raw")

    if method == "soft_K":
        if K is None or K < 0:
            raise ValueError(f"soft_K requires K >= 0, got {K!r}")
        if pool.ndim != 2:
            raise ValueError(f"pool must be 2-D, got shape {pool.shape}")
        centered = pool - pool.mean(axis=0, keepdims=True)
        # SVD of centered: centered = U @ diag(S) @ Vt.  Right singular vectors
        # are the principal components in row-vector convention (i.e. PC_k
        # is the k-th row of Vt).
        _U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        if K >= len(S):
            # Nothing to rescale: pool has rank <= K, so soft_K is identity.
            return WhiteningBasis(method="raw", K=K)
        target_sigma = float(S[K])  # the (K+1)-th singular value (0-indexed: S[K])
        scales = target_sigma / np.maximum(S[:K], 1e-12)
        return WhiteningBasis(
            method="soft_K", K=K,
            Vt=Vt[:K].astype(np.float64),
            scales=scales.astype(np.float64),
        )

    if method in ("lw", "oas"):
        from sklearn.covariance import LedoitWolf, OAS
        est = LedoitWolf() if method == "lw" else OAS()
        est.fit(pool)
        cov = est.covariance_
        # Symmetric matrix; use eigh for numerical stability over generic eig.
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        cov_inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
        return WhiteningBasis(method=method, cov_inv_sqrt=cov_inv_sqrt.astype(np.float64))

    raise ValueError(f"Unknown whitening method {method!r}")


def fit_shear(A_goal: np.ndarray,
              A_nogoal: np.ndarray,
              L: int,
              vectors_as: str = "cols") -> WhiteningBasis:
    """Fit a soft-shear transform that orthogonalizes the top-L canonical-angle
    pairs of two subspaces.

    Geometrically: the two subspaces have a sequence of canonical angles
    ``theta_0 <= theta_1 <= ...`` that measure how aligned they are.  The
    smallest angles are the *most* aligned pairs (closest to zero).  The
    soft-shear takes the top-L most-aligned pairs (smallest theta) and
    rotates them to be exactly orthogonal, leaving the lower pairs
    untouched.  The transformation is volume-preserving in each plane:
    ``e+`` is squashed by ``sqrt(tan(theta_i / 2))`` and ``e-`` is stretched
    by the reciprocal.

    Parameters
    ----------
    A_goal, A_nogoal : numpy arrays
        Two subspaces in D-dimensional space.  Orientation set by
        ``vectors_as``.
    L : int
        Number of top canonical-angle pairs to orthogonalize.  ``L=0``
        returns the identity transform.  Must be ``<= min(n_g, n_n)``.
    vectors_as : str
        - ``'cols'`` (default): inputs are ``(D, n_*)``, columns are vectors.
          This matches the project convention for subspaces stored on disk
          via :func:`build_goal_nogoal_subspaces` and the historical
          ``make_shear`` ad-hoc copies.
        - ``'rows'``: inputs are ``(n_*, D)``, rows are vectors (sklearn
          convention; matches the entity matrix layout that ``.apply()``
          consumes).

    Returns
    -------
    WhiteningBasis
        ``method='soft_shear'`` with the (D, 2L) basis and (2L,) factors.
        Use ``basis.apply(X)`` to shear new vectors / matrices, where ``X``
        is **rows-as-vectors** (n, D) or a single (D,) vector -- this matches
        the apply convention of ``fit_whitening`` regardless of which
        ``vectors_as`` was chosen at fit time.

    Raises
    ------
    ValueError
        If ``L < 0`` or ``L > min(n_g, n_n)``, or if shapes are inconsistent.

    Notes
    -----
    The standard project usage is::

        from results_analysis.canonical_angles.data import build_goal_nogoal_subspaces
        from results_analysis.canonical_angles.whitening import fit_shear
        A_g, A_n = build_goal_nogoal_subspaces(data_dir, slot=3, layer=25,
                                                kind='combined')   # (D, 60) cols
        shear = fit_shear(A_g, A_n, L=5)                            # default 'cols'
        M_done = shear.apply(M)                                     # M is (n, D)
    """
    if vectors_as not in ("cols", "rows"):
        raise ValueError(f"vectors_as must be 'cols' or 'rows', got {vectors_as!r}")

    A_g = np.asarray(A_goal, dtype=np.float64)
    A_n = np.asarray(A_nogoal, dtype=np.float64)
    if A_g.ndim != 2 or A_n.ndim != 2:
        raise ValueError(f"A_goal/A_nogoal must be 2-D, got shapes "
                         f"{A_g.shape}, {A_n.shape}")

    # Normalise to columns-as-vectors for the math below.
    if vectors_as == "rows":
        A_g, A_n = A_g.T, A_n.T

    if A_g.shape[0] != A_n.shape[0]:
        raise ValueError(
            f"A_goal and A_nogoal must share the ambient dimension D; got "
            f"D_goal={A_g.shape[0]}, D_nogoal={A_n.shape[0]} "
            f"(with vectors_as={vectors_as!r}). "
            f"If the inputs are flipped, try vectors_as='rows'."
        )
    D = A_g.shape[0]
    n_g = A_g.shape[1]
    n_n = A_n.shape[1]
    n_min = min(n_g, n_n)

    if L < 0:
        raise ValueError(f"L must be >= 0, got {L}")
    if L > n_min:
        raise ValueError(f"L={L} exceeds min(n_goal, n_nogoal)={n_min}")
    if L == 0:
        return WhiteningBasis(method="raw", L=0)

    # Orthonormal bases for each subspace.
    Qa, _ = np.linalg.qr(A_g)
    Qb, _ = np.linalg.qr(A_n)
    # Cross-correlation: singular values are cos(theta_i) of the canonical
    # angles in ascending angle order (== descending cosine order, which is
    # what numpy's SVD returns).
    U, sigma, Vt = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    thetas = np.arccos(sigma)  # length n_min, ascending in theta

    # Truncate to top-L most-aligned pairs (smallest theta -> largest sigma -> first).
    half = thetas[:L] / 2.0
    UL = U[:, :L]
    VtL = Vt[:L, :]
    # Canonical pair directions in the original D-dim space.
    e_plus = ((Qa @ UL) + (Qb @ VtL.T)) / (2.0 * np.cos(half))[None, :]   # (D, L)
    e_minus = ((Qa @ UL) - (Qb @ VtL.T)) / (2.0 * np.sin(half))[None, :]  # (D, L)
    E = np.concatenate([e_plus, e_minus], axis=1)  # (D, 2L)
    factors = np.concatenate([
        np.sqrt(np.tan(half)) - 1.0,        # squash e+
        np.sqrt(1.0 / np.tan(half)) - 1.0,  # stretch e-
    ])  # (2L,)

    return WhiteningBasis(
        method="soft_shear",
        L=L,
        shear_basis=E,
        shear_factors=factors,
        canonical_angles_rad=thetas,
    )


def parse_whitening_spec(s: str) -> tuple[str, int | None]:
    """Parse a CLI string into ``(method, K_or_L)``.

    Recognised forms:

    - ``raw``                       -> ('raw', None)
    - ``soft_K=N`` / ``soft_K_N``   -> ('soft_K', N)
    - ``lw`` / ``oas``              -> (s, None)
    - ``soft_shear=L`` / ``shear=L`` -> ('soft_shear', L)

    Used by wrappers to convert their --whitening flag to a method+K pair.
    For the shear regime, the second element is the L truncation level.
    Note that fitting a shear additionally requires two subspaces, which
    are not encoded in the spec string -- callers must supply those
    separately when invoking :func:`fit_shear`.
    """
    s = s.strip()
    if s == "raw":
        return "raw", None
    if s in ("lw", "oas"):
        return s, None

    def _parse_with_value(s: str, prefixes: tuple[str, ...], canonical: str
                          ) -> tuple[str, int]:
        """Return (method, n) for ``<prefix><sep?><n>``; sep is '=' or '_' or empty."""
        for prefix in prefixes:
            if s.startswith(prefix):
                tail = s[len(prefix):]
                # Strip leading '=' or '_' if present.
                if tail.startswith(("=", "_")):
                    tail = tail[1:]
                if not tail:
                    raise ValueError(
                        f"{canonical} requires a value, e.g. '{canonical}=4'")
                try:
                    return canonical, int(tail)
                except ValueError as e:
                    raise ValueError(
                        f"Cannot parse {canonical} value from {s!r}: "
                        f"{tail!r} is not an int") from e
        raise AssertionError(f"caller bug: {s!r} doesn't match any of {prefixes}")

    if s.startswith("soft_K"):
        return _parse_with_value(s, ("soft_K",), "soft_K")
    if s.startswith(("soft_shear", "shear")):
        return _parse_with_value(s, ("soft_shear", "shear"), "soft_shear")
    raise ValueError(f"Cannot parse whitening spec {s!r}; "
                     f"expected one of: 'raw', 'soft_K=N', 'lw', 'oas', "
                     f"'soft_shear=L'")
