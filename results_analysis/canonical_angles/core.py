"""Canonical-angles computation orchestration.

The user-facing entry point is :func:`compute_ca_grid`, which takes a list of
:class:`CASpec` configurations and returns one canonical-angles array per
spec.  A shared :class:`ComputeEnv` caches expensive intermediate results
(loaded entity vectors, fitted whitening bases, computed origins) so that
sweeps over many specs that share these intermediates run at near-O(1)
amortised cost.

Concept layout
--------------

A :class:`CASpec` describes a single CA computation:

1. Two subspaces (``subspace_a``, ``subspace_b``) given as lists of
   ``(etype, name)`` entries.  ``data.py`` knows how to resolve these.
2. A ``slot_indices`` tuple plus ``slot_mode`` (``'single'`` / ``'avg'`` /
   ``'concat'``) selecting which slot(s) to use and how to combine them.
3. A transformer ``layer`` index.
4. An ``OriginSpec`` describing what to subtract from each subspace before
   the SVD.  The ``'full_combo_mean'`` mode requires a ``combos_kind``
   (``'r'``, ``'t'``, or ``'combined'``) so we know which combinations to
   average over.
5. An optional :class:`WhiteningSpec` describing PCA whitening to apply to
   each (centered) entity vector before computing its SVD basis.

The canonical angles between the two resulting bases are returned in
ascending order, in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable

import numpy as np

from . import data as ca_data
from .whitening import (
    WhiteningBasis,
    WhiteningMethod,
    fit_whitening,
)


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WhiteningSpec:
    """Description of a whitening regime + its pool.

    The pool is described as a tuple of ``(etype, name)`` entries; using
    a tuple (not list) makes the spec hashable for cache lookups.
    """
    method: WhiteningMethod                       # 'raw' | 'soft_K' | 'lw' | 'oas'
    K: int | None = None
    pool: tuple[tuple[str, str], ...] = ()        # entries to fit the basis on


@dataclass(frozen=True)
class AggSpec:
    """Aggregation: how to construct the goal/nogoal vectors.

    Default is ``mode='combo_residual_theat_shifted'`` with
    ``include_default=False``.  Plain ``combo_residual`` marginals carry a
    constant offset along a "performative-persona" axis ``v_theat`` (see
    ``compute_theatricality_axis`` in
    :mod:`results_analysis.compute_combo_marginals`) inherited from the
    additive structure of role+trait combinations; the
    ``_theat_shifted`` variant cancels that offset on the fly so the four
    residual subspaces sit at the additive origin where
    ``combo ≈ role + trait``.  See ``canonical_angles/README.md`` for the
    full derivation.

    Use plain ``mode='combo_residual'`` for slot-0 (body-mean) analyses
    where the shift hurts CA1, or for "before" comparisons.  The legacy
    ``'combo_centroid'`` mode is deprecated; see :mod:`.data`.
    """
    mode: str = "combo_residual_theat_shifted"    # 'standalone' | 'combo_residual' |
                                                  # 'combo_residual_theat_shifted' (default) |
                                                  # 'combo_centroid' (deprecated)
    include_default: bool = False


@dataclass(frozen=True)
class OriginSpec:
    """What to subtract from each subspace before the SVD basis is taken.

    Default is ``kind='none'`` -- the natural choice for residual marginals,
    which already live in a delta-space with a semantic zero.  Centering
    via ``self_mean`` etc. removes one dimension from the subspace.
    """
    kind: str = "none"                            # 'none' | 'default' | 'self_mean'
                                                  # | 'pool_mean' | 'pool_mean_per_side'
                                                  # | 'full_combo_mean' | 'paired'
    # 'full_combo_mean' needs to know which combinations to average over.
    combos_kind: str | None = None                # 'r' | 't' | 'combined'  (required for full_combo_mean)
    # 'pool_mean' uses the same scope/leave_out as the WhiteningSpec by default;
    # callers may want to override these for 'pool_mean' specifically.
    pool_scope: str | None = None                 # 'roles' | 'traits' | 'roles+traits'


@dataclass(frozen=True)
class CASpec:
    """A single canonical-angles computation specification.

    Hashable: the lists/tuples are themselves frozen tuples and the dataclass
    is frozen.  This lets us deduplicate specs and use them as cache keys.
    """
    subspace_a: tuple[tuple[str, str], ...]
    subspace_b: tuple[tuple[str, str], ...]
    slot_indices: tuple[int, ...]
    slot_mode: str                                # 'single' | 'avg' | 'concat'
    layer: int
    aggregation: AggSpec
    origin: OriginSpec
    whitening: WhiteningSpec
    # Free-form label for the result dict key.  If empty, compute_ca_grid
    # auto-generates one from the kind/slot/layer/whitening fields where
    # possible (best-effort, only useful for debugging).
    label: str = ""

    @property
    def slot_label(self) -> str:
        """Human-readable label for the slot composition."""
        if self.slot_mode == "single":
            return f"{self.slot_indices[0]}"
        if self.slot_mode in ("avg", "average"):
            return f"avg({','.join(str(i) for i in self.slot_indices)})"
        if self.slot_mode == "concat":
            return f"concat({','.join(str(i) for i in self.slot_indices)})"
        return f"{self.slot_mode}({self.slot_indices})"


# ---------------------------------------------------------------------------
# Cached compute environment
# ---------------------------------------------------------------------------

@dataclass
class ComputeEnv:
    """Cache holder for compute_ca / compute_ca_grid.

    Holds three caches:

    1. ``_entity_cache``: fully-loaded ``(n_slots, n_layers, hidden)`` arrays
       per ``(etype, name)``, so each .pt file is read at most once per env.
    2. ``_whitener_cache``: fitted :class:`WhiteningBasis` per
       ``(method, K, pool_signature, slot_label, layer)``, so a sweep of
       many specs sharing the same whitening pool only does the SVD once.
    3. ``_origin_cache``: computed origin vectors per ``(origin_signature,
       slot_label, layer)``, so for example the
       ``full_combo_mean`` for kind='r' is computed once per (slot, layer)
       and reused across all specs sharing that origin.
    """
    data_dir: Path
    _entity_cache: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    _weight_cache: dict[tuple[str, str], int] = field(default_factory=dict)
    _whitener_cache: dict[tuple, WhiteningBasis] = field(default_factory=dict)
    _origin_cache: dict[tuple, np.ndarray] = field(default_factory=dict)

    def get_weight(self, etype: str, name: str) -> int:
        """Effective sample size for an entry; see :func:`data.load_entity_weight`.

        Cached per ``(etype, name)``.  Used by the weighted ``self_mean``
        origin so that under-sampled marginals contribute less.
        """
        key = (etype, name)
        cached = self._weight_cache.get(key)
        if cached is None:
            cached = ca_data.load_entity_weight(self.data_dir, etype, name)
            self._weight_cache[key] = cached
        return cached

    def load_subspace_weights(self, entries: tuple[tuple[str, str], ...]
                              ) -> np.ndarray:
        """Return a (N,) float array of per-entry weights for use in weighted
        means.  Defaults to 1 for non-marginal etypes."""
        return np.array([self.get_weight(e, n) for e, n in entries],
                        dtype=np.float64)

    def get_entity(self, etype: str, name: str) -> np.ndarray:
        """Load a (n_slots, n_layers, hidden) array, caching by (etype, name).

        Recognised etypes:

        - ``traits``, ``roles``, ``combinations``: direct on-disk read.
        - ``r_goal``/``r_nogoal``/``t_goal``/``t_nogoal``: on-disk per-axis
          residual marginal (produced by ``compute_combo_marginals.py`` in
          its default mode).  Falls back to on-the-fly residual computation
          if the file is missing.
        - ``*_theat_shifted`` (e.g. ``r_goal_theat_shifted``): residual
          marginal as above, plus the per-(slot, layer) theatricality shift
          from ``combinations/vectors/theatricality_axis.pt`` -- relocates
          the residual to the additive model's true origin.
        - ``*_legacy_centroid``: on-disk centroid marginal (snapshot of the
          historical files, kept for bit-exact reproduction).
        - ``*_centroid_recompute``: compute centroid on the fly from raw
          combination .pt files (used when the legacy snapshot is absent).
        """
        key = (etype, name)
        cached = self._entity_cache.get(key)
        if cached is None:
            if etype.endswith("_theat_shifted"):
                base_etype = etype[: -len("_theat_shifted")]
                base = self.get_entity(base_etype, name)
                shift = self._theatricality_shift()
                cached = base + shift
            elif etype.endswith("_centroid_recompute"):
                cached = self._compute_combo_marginal(etype, name, mode="centroid")
            elif etype in ("r_goal", "r_nogoal", "t_goal", "t_nogoal"):
                cached = self._load_or_compute_residual_marginal(etype, name)
            else:
                cached = ca_data.load_vector(self.data_dir, etype, name)
            self._entity_cache[key] = cached
        return cached

    def _theatricality_shift(self) -> np.ndarray:
        """Cache + return the (n_slots, n_layers, hidden) theatricality
        shift vector loaded from disk."""
        key = ("__theatricality_shift__", "")
        cached = self._entity_cache.get(key)
        if cached is None:
            cached = ca_data.load_theatricality_shift(self.data_dir)
            self._entity_cache[key] = cached
        return cached

    def _load_or_compute_residual_marginal(self, etype: str, name: str
                                           ) -> np.ndarray:
        """Load the on-disk residual marginal, or recompute if missing."""
        path = ca_data.vector_path(self.data_dir, etype, name)
        if path.exists():
            return ca_data.load_vector(self.data_dir, etype, name)
        # Fallback: compute on the fly.
        return self._compute_combo_marginal(etype, name, mode="residual")

    def _compute_combo_marginal(self, etype: str, name: str, mode: str
                                ) -> np.ndarray:
        """Compute one combo marginal on demand from raw combinations.

        Decodes etype into ``kind`` ('r'|'t') and ``side`` ('goal'|'nogoal').
        For ``mode='residual'``: averages ``combo(A,B) - standalone(B)`` over
        partners B.  For ``mode='centroid'``: averages ``combo(A,B)`` over
        partners B.

        Note: ``mode='centroid'`` is the deprecated reduction (biased on
        incomplete grids); kept only for legacy reproduction paths.
        """
        # Strip the recompute suffix if present, so 'r_goal_centroid_recompute'
        # decodes to ('r', 'goal').
        base = etype
        for suffix in ("_centroid_recompute",):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        kind = base[0]
        side = "goal" if base.endswith("_goal") else "nogoal"
        goal_axis = ca_data.GOAL_AXIS[kind]
        nogoal_axis = "trait" if goal_axis == "role" else "role"
        index_axis = goal_axis if side == "goal" else nogoal_axis
        partner_axis = "trait" if index_axis == "role" else "role"
        partner_etype = "traits" if partner_axis == "trait" else "roles"

        combo_files = ca_data.all_combo_files(self.data_dir, kind)
        contribs: list[np.ndarray] = []
        for f in combo_files:
            role, trait = ca_data.parse_combo_stem(f.stem, kind)
            indexed = role if index_axis == "role" else trait
            if indexed != name:
                continue
            combo_vec = self.get_entity("combinations", f.stem)
            if mode == "residual":
                partner_name = trait if partner_axis == "trait" else role
                partner_vec = self.get_entity(partner_etype, partner_name)
                contribs.append(combo_vec - partner_vec)
            elif mode == "centroid":
                contribs.append(combo_vec)
            else:
                raise ValueError(f"Unknown combo marginal mode {mode!r}")
        if not contribs:
            raise RuntimeError(
                f"No combinations found contributing to {mode} marginal "
                f"{etype}/{name}"
            )
        return np.stack(contribs).mean(axis=0)

    def load_subspace_matrix(self, entries: tuple[tuple[str, str], ...],
                             slot_indices: tuple[int, ...],
                             slot_mode: str, layer: int) -> np.ndarray:
        """Load an N x D matrix where each row is one entity at the chosen
        slot/layer (after slot composition)."""
        rows = []
        for etype, name in entries:
            full = self.get_entity(etype, name)
            row = ca_data.compose_slots(full, tuple(slot_indices), slot_mode, layer)
            rows.append(row)
        return np.stack(rows)


# ---------------------------------------------------------------------------
# Origin computation
# ---------------------------------------------------------------------------

def _compute_origin(origin: OriginSpec, ca_spec: CASpec, env: ComputeEnv,
                    A: np.ndarray, B: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Compute centering origins for subspaces A and B.

    Returns ``(origin_a, origin_b)``.  Most modes use a single shared origin
    for both; ``'self_mean'`` and ``'pool_mean_per_side'`` give A and B
    different origins.

    ``self_mean`` uses a *weighted* mean where each entry is weighted by its
    ``n_partners_averaged`` (1 for standalone entries; 27-30 for residual
    marginals).  This properly downweights entries with smaller effective
    sample size.
    """
    # Loud warning for the one combination that's almost always a bug:
    # standalone + none leaves the global "AI assistant" component in both
    # subspaces, producing a near-zero first canonical angle that does not
    # reflect any meaningful alignment.  Allowed but warned.
    if origin.kind == "none" and ca_spec.aggregation.mode == "standalone":
        import warnings
        warnings.warn(
            "OriginSpec(kind='none') with AggSpec(mode='standalone') leaves "
            "the assistant-context baseline in both subspaces and yields a "
            "near-zero first canonical angle that is not a meaningful "
            "alignment.  Prefer 'default', 'self_mean', or 'pool_mean_per_side' "
            "for standalone aggregation.",
            stacklevel=2,
        )

    mode = origin.kind
    slot_indices = ca_spec.slot_indices
    slot_mode = ca_spec.slot_mode
    layer = ca_spec.layer

    if mode == "self_mean":
        wa = env.load_subspace_weights(ca_spec.subspace_a)
        wb = env.load_subspace_weights(ca_spec.subspace_b)
        origin_a = (A * wa[:, None]).sum(axis=0) / wa.sum()
        origin_b = (B * wb[:, None]).sum(axis=0) / wb.sum()
        return origin_a, origin_b

    if mode == "pool_mean_per_side":
        # Per-subspace pool: each subspace uses the standalone pool that
        # matches its own axis (kind=r goal -> roles; kind=r nogoal ->
        # traits; etc.).
        scope_a = _pool_scope_for_subspace(ca_spec.subspace_a)
        scope_b = _pool_scope_for_subspace(ca_spec.subspace_b)
        origin_a = _pool_mean(env, scope_a, slot_indices, slot_mode, layer)
        origin_b = _pool_mean(env, scope_b, slot_indices, slot_mode, layer)
        return origin_a, origin_b

    # All other modes use a single shared origin.
    cache_key = (
        mode, ca_spec.slot_label, layer,
        origin.combos_kind, origin.pool_scope,
        # For pool_mean we don't capture leave_out in the cache key here;
        # current callers set pool_mean via the same leave_out as the
        # whitening pool, but we recompute pool_mean per spec to be safe.
    )

    cached = env._origin_cache.get(cache_key)
    if cached is not None:
        return cached, cached

    if mode == "none":
        shared = np.zeros(A.shape[1])
    elif mode == "default":
        # Pick the default vector from the trait/role family that maps to
        # subspace A's first entry's etype, falling back to traits/default.
        default_etype = _default_etype_for(ca_spec.subspace_a)
        full = env.get_entity(default_etype, "default")
        shared = ca_data.compose_slots(full, tuple(slot_indices), slot_mode, layer)
    elif mode == "pool_mean":
        scope = origin.pool_scope or "roles+traits"
        shared = _pool_mean(env, scope, slot_indices, slot_mode, layer)
    elif mode == "full_combo_mean":
        if origin.combos_kind is None:
            raise ValueError("OriginSpec(kind='full_combo_mean') requires combos_kind")
        shared = _full_combo_mean(env, origin.combos_kind,
                                  tuple(slot_indices), slot_mode, layer)
    elif mode == "paired":
        raise NotImplementedError(
            "OriginSpec(kind='paired') requires combo_residual aggregation; "
            "not yet implemented."
        )
    else:
        raise ValueError(f"Unknown origin kind {mode!r}")

    env._origin_cache[cache_key] = shared
    return shared, shared


def _pool_mean(env: ComputeEnv, scope: str,
               slot_indices: tuple[int, ...], slot_mode: str, layer: int
               ) -> np.ndarray:
    """Mean of the standalone pool for a given scope at one slot/layer."""
    pool_entries = ca_data.build_whitening_pool(env.data_dir, scope=scope)
    pool_mat = env.load_subspace_matrix(tuple(pool_entries),
                                        tuple(slot_indices), slot_mode, layer)
    return pool_mat.mean(axis=0)


def _pool_scope_for_subspace(subspace: tuple[tuple[str, str], ...]) -> str:
    """Return the standalone pool scope ('roles' or 'traits') matching a
    subspace's axis.

    Looks at the first entry's etype: 'roles' -> 'roles', 'traits' -> 'traits',
    combo marginal etypes ('r_goal' etc. and their legacy/recompute siblings)
    are mapped via :func:`ca_data._combo_marginal_to_standalone`.
    Falls back to 'roles+traits' for anything we can't classify.
    """
    if not subspace:
        return "roles+traits"
    first_etype = subspace[0][0]
    if first_etype == "roles":
        return "roles"
    if first_etype == "traits":
        return "traits"
    underlying = ca_data._combo_marginal_to_standalone(first_etype)
    if underlying == "roles":
        return "roles"
    if underlying == "traits":
        return "traits"
    return "roles+traits"


def _full_combo_mean(env: ComputeEnv, combos_kind: str,
                     slot_indices: tuple[int, ...], slot_mode: str,
                     layer: int) -> np.ndarray:
    """Mean of all combination vectors of the given kind, plus default.

    For combos_kind in ('r', 't'): mean of all r_*__* (or t_*__*) plus
    combinations/default.pt.
    For combos_kind == 'combined': use both r and t combinations.
    """
    files = []
    if combos_kind in ("r", "combined"):
        files.extend(ca_data.all_combo_files(env.data_dir, "r"))
    if combos_kind in ("t", "combined"):
        files.extend(ca_data.all_combo_files(env.data_dir, "t"))
    if not files:
        raise RuntimeError(f"No combination files found for kind {combos_kind!r}")

    rows = []
    for f in files:
        # Each combination file lives at combinations/vectors/<stem>.pt.
        # Use the 'combinations' etype to load via the unified path resolver.
        full = env.get_entity("combinations", f.stem)
        rows.append(ca_data.compose_slots(full, slot_indices, slot_mode, layer))
    # Plus default
    full_default = env.get_entity("combinations", "default")
    rows.append(ca_data.compose_slots(full_default, slot_indices, slot_mode, layer))

    return np.stack(rows).mean(axis=0)


def _default_etype_for(subspace: tuple[tuple[str, str], ...]) -> str:
    """Best-effort guess at a 'default' etype for OriginSpec(kind='default')."""
    if not subspace:
        return "traits"
    first_etype = subspace[0][0]
    if first_etype in ("traits", "roles"):
        return first_etype
    if first_etype in ca_data._COMBO_MARGINAL_ETYPES \
            or first_etype.endswith("_centroid_recompute"):
        return "combinations"
    return "traits"


# ---------------------------------------------------------------------------
# Whitener get-or-fit, with caching
# ---------------------------------------------------------------------------

def _get_whitener(spec: WhiteningSpec, ca_spec: CASpec,
                  env: ComputeEnv) -> WhiteningBasis:
    """Look up or fit a whitening basis for the given spec.

    Cache key includes the slot composition + layer because the pool is
    sliced down to a single (slot_label, layer) when fit.
    """
    if spec.method == "raw":
        return WhiteningBasis(method="raw")

    cache_key = (spec.method, spec.K, spec.pool, ca_spec.slot_label, ca_spec.layer)
    cached = env._whitener_cache.get(cache_key)
    if cached is not None:
        return cached

    pool_mat = env.load_subspace_matrix(spec.pool,
                                        ca_spec.slot_indices,
                                        ca_spec.slot_mode,
                                        ca_spec.layer)
    basis = fit_whitening(spec.method, pool_mat, K=spec.K)
    env._whitener_cache[cache_key] = basis
    return basis


# ---------------------------------------------------------------------------
# SVD basis + canonical angles primitives
# ---------------------------------------------------------------------------

def _svd_basis(centered: np.ndarray, eps_factor: float = 1e-6) -> np.ndarray:
    """Return an orthonormal basis for the row-span of ``centered``.

    ``centered`` is shape ``(n_entities, hidden)``.  Output is shape
    ``(hidden, rank)`` where rank is determined by truncating singular
    values below ``eps_factor * sigma_max``.
    """
    if centered.shape[0] == 0:
        raise ValueError("Cannot compute basis of an empty matrix")
    _U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    if len(S) == 0 or S[0] == 0:
        return np.zeros((centered.shape[1], 0))
    rank = int(np.sum(S > S[0] * eps_factor))
    return Vt[:rank].T  # (hidden, rank)


def canonical_angles_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute canonical angles between two orthonormal bases, in degrees,
    sorted ascending.

    A: (D, rank_a), B: (D, rank_b).  Returns ``min(rank_a, rank_b)`` angles.
    """
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.array([])
    M = A.T @ B
    svs = np.linalg.svd(M, compute_uv=False)
    svs = np.clip(svs, 0.0, 1.0)
    angles = np.degrees(np.arccos(svs))
    angles.sort()
    return angles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ca(spec: CASpec, env: ComputeEnv) -> np.ndarray:
    """Compute the canonical angles for a single :class:`CASpec`.

    Returns an angle array in degrees, sorted ascending, of length
    ``min(N_a, N_b)`` where ``N_a`` and ``N_b`` are the entity counts
    of the two subspaces.

    NaN padding for rank deficiency
    --------------------------------
    Some origin choices (e.g. ``self_mean``) reduce each subspace's rank
    by 1, producing only ``min(N_a, N_b) - 1`` well-defined canonical
    angles instead of ``min(N_a, N_b)``.  In those cases the array is
    NaN-padded at the *start* (the smallest-angle position) so the result
    length is the same as the un-centered case.

    The convention is that the missing angle is *undefined* (the
    centered subspace inner product is 0/0 along the rank-deficient
    direction), which is honestly distinct from a genuine 0 deg
    "perfectly aligned" angle.  Use ``np.nanmedian`` / ``np.nanmin`` etc.
    when summarising.
    """
    A = env.load_subspace_matrix(spec.subspace_a, spec.slot_indices,
                                 spec.slot_mode, spec.layer)
    B = env.load_subspace_matrix(spec.subspace_b, spec.slot_indices,
                                 spec.slot_mode, spec.layer)

    origin_a, origin_b = _compute_origin(spec.origin, spec, env, A, B)

    A_centered = A - origin_a
    B_centered = B - origin_b

    if spec.whitening.method != "raw":
        whitener = _get_whitener(spec.whitening, spec, env)
        A_centered = whitener.apply(A_centered)
        B_centered = whitener.apply(B_centered)

    V_a = _svd_basis(A_centered)
    V_b = _svd_basis(B_centered)
    angles = canonical_angles_deg(V_a, V_b)
    # Pad with NaN at the start so the result length matches the
    # uncentered-rank expectation min(N_a, N_b).
    expected_len = min(len(spec.subspace_a), len(spec.subspace_b))
    n_pad = expected_len - len(angles)
    if n_pad > 0:
        angles = np.concatenate([np.full(n_pad, np.nan), angles])
    return angles


def compute_ca_grid(specs: list[CASpec],
                    data_dir: Path,
                    env: ComputeEnv | None = None,
                    ) -> dict[Hashable, np.ndarray]:
    """Compute CA for many specs, sharing cached intermediates.

    Returns a dict ``{key: angle_array}``.  The key is each spec's
    ``label`` if non-empty, else the spec itself (frozen dataclasses are
    hashable).
    """
    if env is None:
        env = ComputeEnv(data_dir=data_dir)

    out: dict[Hashable, np.ndarray] = {}
    for spec in specs:
        key: Hashable = spec.label if spec.label else spec
        out[key] = compute_ca(spec, env)
    return out
