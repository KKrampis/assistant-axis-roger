"""Entity loading and subspace construction for the canonical angles tool.

This module is responsible for resolving:

- raw .pt loading from any of the entity directories
  (``traits/vectors/``, ``roles/vectors/``, ``combinations/vectors/``,
  ``combinations/vectors/{r_goal,r_nogoal,t_goal,t_nogoal}/``).
- mapping a (kind, side, aggregation) triple to a list of ``(etype, name)``
  entries that define one canonical-angles subspace.
- building the whitening pool given a ``scope`` and ``leave_out`` set.

The core compute layer treats each subspace as a list of ``(etype, name)``
entries; this module owns all knowledge of where each one lives on disk and
what kind/side/aggregation combinations are valid.

Conventions
-----------

- Filenames in ``combinations/vectors/`` follow ``<kind>_<role>__<trait>.pt``.
  The order is always ``role__trait`` regardless of kind; only which one is
  the goal differs:

    * ``r`` : role supplies goal, trait is non-goal.
    * ``t`` : trait supplies goal, role is non-goal.

- Entity types (``etype``) recognised here:

    * ``traits``                 -> data_dir/traits/vectors/<name>.pt
    * ``roles``                  -> data_dir/roles/vectors/<name>.pt
    * ``r_goal``, ``r_nogoal``,
      ``t_goal``, ``t_nogoal``   -> data_dir/combinations/vectors/<etype>/<name>.pt
                                   (per-axis combo *residual* marginals; see
                                   results_analysis/compute_combo_marginals.py)
    * ``r_goal_legacy_centroid``, etc.
                                 -> data_dir/combinations/vectors/<etype>/<name>.pt
                                    (the original centroid marginals, kept for
                                    historical reproduction)
    * ``combinations``           -> data_dir/combinations/vectors/<name>.pt
                                   (raw r_*__* / t_*__* and default.pt)

- Loaded vectors are expected to be ``torch.bfloat16`` tensors of shape
  ``(n_slots, n_layers, hidden)`` stored under ``obj['vector']`` in a dict.

- Aggregation modes (in subspace constructors):

    * ``standalone``      : on-disk standalone trait/role vectors.
    * ``combo_residual``  (recommended default): on-disk per-axis residual
        marginals (``r_goal/<role>.pt``, etc.).  Each indexed entry is the
        mean over partners B of (combo(A,B) - standalone(B)).  Honest under
        partner-set imbalance.
    * ``combo_centroid``  (DEPRECATED): the partner-averaged combos,
        either from on-disk ``*_legacy_centroid/`` dirs (fast) or recomputed
        on the fly from the raw combination .pt files.  Biased on incomplete
        grids; kept only for legacy reproduction of older plots and is no
        longer exposed via the wrapper CLIs.
"""

from __future__ import annotations

import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# Default location of the cached activation vectors.  Wrappers should accept
# a CLI override.
DEFAULT_DATA_DIR = "runpod_workspace/qwen/qwen-3-32b Roger 8slot"

# Which axis (role or trait) supplies the goal for each combination kind.
GOAL_AXIS = {"r": "role", "t": "trait"}


@dataclass(frozen=True)
class Entry:
    """A pointer to one cached vector on disk.  Hashable for use as a cache key.

    We use ``(etype, name)`` tuples in most public APIs but keep this typed
    wrapper around for places where we need a stable order or hashable keys.
    """
    etype: str
    name: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.etype, self.name)


# ---------------------------------------------------------------------------
# Path resolution and raw loading
# ---------------------------------------------------------------------------

_COMBO_MARGINAL_ETYPES = (
    "r_goal", "r_nogoal", "t_goal", "t_nogoal",
    "r_goal_legacy_centroid", "r_nogoal_legacy_centroid",
    "t_goal_legacy_centroid", "t_nogoal_legacy_centroid",
    "r_goal_theat_shifted", "r_nogoal_theat_shifted",
    "t_goal_theat_shifted", "t_nogoal_theat_shifted",
)

_LEGACY_CENTROID_SUFFIX = "_legacy_centroid"

# ---------------------------------------------------------------------------
# Path resolution for post-pipeline derived files.
#
# Phase 1.0 (provenance redesign) split post-pipeline derivatives into a
# dedicated ``combinations/vectors/derived/`` subtree so they no longer share
# mtime/manifest fate with the raw pipeline outputs.  The new layout is::
#
#     combinations/vectors/derived/
#         marginals/{r_goal,r_nogoal,t_goal,t_nogoal}/<name>.pt
#         legacy_centroid/{r_goal,r_nogoal,t_goal,t_nogoal}/<name>.pt
#         aggregates/{mean_r_combos.pt, mean_t_combos.pt}
#         axis/theatricality_axis.pt
#
# Each helper below tries the new path first and falls back to the legacy
# location (flat under ``combinations/vectors/``) so the code change can land
# before any physical migration, and so each dataset can be migrated
# independently.  See audits/post_pipeline_derived_layout.md for the full
# mapping.
# ---------------------------------------------------------------------------

def _legacy_marginal_dir(data_dir: Path, etype: str) -> Path:
    """Pre-Phase-1.0 location of a combo-marginal etype's directory."""
    return data_dir / "combinations" / "vectors" / etype


def _derived_marginal_dir(data_dir: Path, etype: str) -> Path:
    """Resolve the on-disk directory holding the per-name .pt files for a
    combo-marginal etype, preferring the post-Phase-1.0 layout under
    ``combinations/vectors/derived/`` and falling back to the legacy flat
    layout if the new tree is absent.
    """
    cv = data_dir / "combinations" / "vectors"
    if etype.endswith(_LEGACY_CENTROID_SUFFIX):
        base = etype[: -len(_LEGACY_CENTROID_SUFFIX)]
        new = cv / "derived" / "legacy_centroid" / base
    else:
        new = cv / "derived" / "marginals" / etype
    if new.exists():
        return new
    return _legacy_marginal_dir(data_dir, etype)


def _derived_aggregate_path(data_dir: Path, filename: str) -> Path:
    """Resolve a top-level aggregate (mean_r_combos.pt / mean_t_combos.pt)."""
    cv = data_dir / "combinations" / "vectors"
    new = cv / "derived" / "aggregates" / filename
    if new.exists():
        return new
    return cv / filename


def _derived_axis_path(data_dir: Path) -> Path:
    """Resolve the theatricality_axis.pt artifact."""
    cv = data_dir / "combinations" / "vectors"
    new = cv / "derived" / "axis" / "theatricality_axis.pt"
    if new.exists():
        return new
    return cv / "theatricality_axis.pt"


def load_theatricality_shift(data_dir: Path) -> np.ndarray:
    """Load the per-(slot, layer) shift vector to apply to combo_residual
    marginals to relocate them to the *additive-model true origin*.

    Returns an array of shape ``(n_slots, n_layers, hidden)`` such that
    ``residual + load_theatricality_shift(...)`` cancels the systematic
    theatricality offset in each marginal.

    Concretely the shift equals
    ``default_offset[slot, layer] * axis_unit[slot, layer]`` where the
    artifact is produced by
    :func:`results_analysis.compute_combo_marginals.compute_theatricality_axis`.

    The shift is computed independently from the residual marginals (it is
    a function of raw combinations + standalone vectors only), so applying
    it downstream is not circular.
    """
    p = _derived_axis_path(data_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"Theatricality axis artifact not found at {p}. "
            f"Generate it via:\n"
            f"    uv run python results_analysis/compute_combo_marginals.py\n"
            f"(or call compute_theatricality_axis directly)."
        )
    obj = torch.load(p, weights_only=False)
    axis = obj["axis"].float().numpy()                # (n_slots, n_layers, hidden) unit
    offset = obj["default_offset"].float().numpy()    # (n_slots, n_layers)
    shift = offset[..., None] * axis                  # (n_slots, n_layers, hidden)
    return shift.astype(np.float32)


def vector_path(data_dir: Path, etype: str, name: str) -> Path:
    """Resolve an (etype, name) pair to its on-disk .pt path."""
    if etype in ("traits", "roles"):
        return data_dir / etype / "vectors" / f"{name}.pt"
    if etype in _COMBO_MARGINAL_ETYPES:
        return _derived_marginal_dir(data_dir, etype) / f"{name}.pt"
    if etype == "combinations":
        return data_dir / "combinations" / "vectors" / f"{name}.pt"
    raise ValueError(f"Unknown etype {etype!r}")


def load_vector(data_dir: Path, etype: str, name: str) -> np.ndarray:
    """Load a single .pt as a float32 numpy array of shape ``(n_slots, n_layers, hidden)``.

    Always converts bf16 -> float32 on load to keep downstream arithmetic in
    a precision that handles the SVD / mean reductions cleanly.
    """
    path = vector_path(data_dir, etype, name)
    obj = torch.load(path, weights_only=False)
    if isinstance(obj, dict):
        v = obj.get("vector", obj.get("axis", obj))
    else:
        v = obj
    return v.float().numpy()


def load_entity_weight(data_dir: Path, etype: str, name: str) -> int:
    """Return the effective sample size (``n_partners_averaged``) of an entry.

    For combo marginal etypes (``r_goal``, ``r_nogoal``, ``t_goal``,
    ``t_nogoal`` and their ``*_legacy_centroid`` variants) this reads
    ``metadata.n_partners_averaged`` from the .pt file, capturing how many
    raw combinations were averaged into the marginal (typically 30, but
    27-29 when the underlying grid is incomplete).

    For non-marginal etypes (``traits``, ``roles``, ``combinations``,
    ``*_centroid_recompute``) the weight is 1 -- they are single samples
    with no averaging.

    Used by the ``self_mean`` origin so that entries with smaller effective
    sample size contribute proportionally less to the centroid.
    """
    if etype not in _COMBO_MARGINAL_ETYPES:
        return 1
    path = vector_path(data_dir, etype, name)
    if not path.exists():
        # On-the-fly recompute path; we don't have a cached .pt to inspect,
        # so use 1 as a neutral fallback (the weighted_mean call site can
        # also choose to recompute the weight explicitly if it cares).
        return 1
    obj = torch.load(path, weights_only=False)
    if isinstance(obj, dict):
        meta = obj.get("metadata", {}) or {}
        n = meta.get("n_partners_averaged")
        if n is None:
            # Older files may store it under the legacy key.
            for k in ("n_traits_averaged", "n_roles_averaged"):
                if k in meta:
                    n = meta[k]
                    break
        if n is not None:
            return int(n)
    return 1


# ---------------------------------------------------------------------------
# Slot composition: turn a (slot_indices, slot_mode) into a single row vector
# ---------------------------------------------------------------------------

def compose_slots(vec: np.ndarray, slot_indices: tuple[int, ...],
                  slot_mode: str, layer: int) -> np.ndarray:
    """Reduce a ``(n_slots, n_layers, hidden)`` array to a single row vector
    according to the requested slot composition.

    Parameters
    ----------
    vec          : full (n_slots, n_layers, hidden) array (after .float())
    slot_indices : which slot(s) to pull
    slot_mode    : one of:
                     'single'   - len(slot_indices) must be 1; just pick that slot
                     'avg'      - average across the listed slots; output dim = hidden
                     'concat'   - concatenate the listed slots; output dim = hidden * len
    layer        : which transformer layer

    Returns
    -------
    np.ndarray of shape (hidden,) for 'single'/'avg', (hidden * k,) for 'concat'.
    """
    sliced = vec[list(slot_indices), layer, :]  # (k, hidden)
    if slot_mode == "single":
        if len(slot_indices) != 1:
            raise ValueError(f"slot_mode='single' requires exactly 1 slot, "
                             f"got {len(slot_indices)}")
        return sliced[0]
    if slot_mode in ("avg", "average"):
        return sliced.mean(axis=0)
    if slot_mode == "concat":
        return sliced.flatten()
    raise ValueError(f"Unknown slot_mode {slot_mode!r}")


# ---------------------------------------------------------------------------
# Listing entities (with optional default and exclusions)
# ---------------------------------------------------------------------------

def list_etype_names(data_dir: Path, etype: str,
                     include_default: bool = True,
                     exclude: set[str] = frozenset()) -> list[str]:
    """Return sorted entity names available for an etype.

    ``include_default`` controls whether ``default.pt`` is kept in the list
    (e.g. for the standalone ``traits`` and ``roles`` etypes, default is
    typically excluded from corpus statistics).
    """
    if etype in ("traits", "roles"):
        root = data_dir / etype / "vectors"
    elif etype == "combinations":
        root = data_dir / "combinations" / "vectors"
    elif etype in _COMBO_MARGINAL_ETYPES:
        root = _derived_marginal_dir(data_dir, etype)
    else:
        raise ValueError(f"Unknown etype {etype!r}")
    names = sorted(p.stem for p in root.glob("*.pt"))
    if not include_default:
        names = [n for n in names if n != "default"]
    return [n for n in names if n not in exclude]


def all_combo_files(data_dir: Path, kind: str) -> list[Path]:
    """All raw combination files for a kind: r_*__*.pt or t_*__*.pt."""
    pattern = data_dir / "combinations" / "vectors" / f"{kind}_*__*.pt"
    return sorted(Path(p) for p in glob.glob(str(pattern)))


def parse_combo_stem(stem: str, kind: str) -> tuple[str, str]:
    """Parse a combination filename stem like ``'r_activist__abstract'`` ->
    ``('activist', 'abstract')``."""
    prefix = f"{kind}_"
    if not stem.startswith(prefix):
        raise ValueError(f"stem {stem!r} does not start with {prefix!r}")
    rest = stem[len(prefix):]
    role, trait = rest.split("__", 1)
    return role, trait


def goal_aligned_names(data_dir: Path, kind: str) -> tuple[str, list[str], str, list[str]]:
    """Parse the combinations directory to find the goal/non-goal entity sets
    for a given combination kind.

    Returns ``(goal_axis, goal_names, nogoal_axis, nogoal_names)`` where
    ``goal_axis`` is 'role' or 'trait'.

    For ``kind='r'``: 30 goal-aligned roles, 30 non-goal-aligned traits.
    For ``kind='t'``: 30 goal-aligned traits, 30 non-goal-aligned roles.
    """
    goal_axis = GOAL_AXIS[kind]
    nogoal_axis = "trait" if goal_axis == "role" else "role"

    role_set: set[str] = set()
    trait_set: set[str] = set()
    for f in all_combo_files(data_dir, kind):
        role, trait = parse_combo_stem(f.stem, kind)
        role_set.add(role)
        trait_set.add(trait)

    if goal_axis == "role":
        return goal_axis, sorted(role_set), nogoal_axis, sorted(trait_set)
    return goal_axis, sorted(trait_set), nogoal_axis, sorted(role_set)


# ---------------------------------------------------------------------------
# Subspace construction (kind x side x aggregation -> list of Entry)
# ---------------------------------------------------------------------------

# Maps (kind, side) -> the etype that holds the appropriate combo marginals.
# E.g. ('r', 'goal') -> 'r_goal' (one .pt per goal-supplying role).
def _combo_marginal_etype(kind: str, side: str) -> str:
    return f"{kind}_{side}"


def _standalone_etype(axis: str) -> str:
    """Translate a logical axis ('role' or 'trait') into the etype used to
    load standalone vectors ('roles' or 'traits')."""
    return {"role": "roles", "trait": "traits"}[axis]


def standalone_subspace(data_dir: Path, kind: str, side: str,
                        include_default: bool = False) -> list[tuple[str, str]]:
    """Build a subspace from standalone trait/role vectors.

    For ``kind='r'``, side='goal': returns the 30 standalone goal-aligned
    roles (parsed from the r_*__* combination filenames).  Optionally
    appends ('roles', 'default').
    """
    if kind == "combined":
        return _combine_subspaces(
            standalone_subspace(data_dir, "r", side, include_default=include_default),
            standalone_subspace(data_dir, "t", side, include_default=include_default),
        )

    goal_axis, goal_names, nogoal_axis, nogoal_names = goal_aligned_names(data_dir, kind)
    if side == "goal":
        axis = goal_axis
        names = goal_names
    elif side == "nogoal":
        axis = nogoal_axis
        names = nogoal_names
    else:
        raise ValueError(f"Unknown side {side!r}")

    etype = _standalone_etype(axis)
    entries = [(etype, n) for n in names]
    if include_default:
        entries.append((etype, "default"))
    return entries


def combo_residual_subspace(data_dir: Path, kind: str, side: str,
                            include_default: bool = False
                            ) -> list[tuple[str, str]]:
    """Build a subspace of *residual* per-axis marginals from combinations.

    For ``kind='r'`` (role is goal, trait is non-goal):
        r_goal[A]   = mean over traits B of (r_A__B - traits[B])
            "average net effect of role A on top of its trait partner"
        r_nogoal[B] = mean over roles A  of (r_A__B - roles[A])
            "average net effect of trait B on top of its role partner"

    For ``kind='t'`` (trait is goal, role is non-goal): symmetric.

    Each residual marginal isolates the contribution of the indexed entity
    after subtracting the partner's standalone main effect.  These vectors
    are *deltas*, so common origins like ``full_combo_mean`` don't have
    their usual interpretation; ``self_mean`` or ``none`` are the natural
    centering choices.

    The residuals are read from on-disk files at
    ``data_dir/combinations/vectors/{kind}_{side}/<name>.pt``, produced by
    ``results_analysis/compute_combo_marginals.py`` (default mode).  If a
    ``mode='residual'`` field in metadata is missing or wrong, the loader
    falls back to recomputing on the fly inside
    :class:`.core.ComputeEnv`.

    ``include_default`` is rarely useful for residuals: a "default"
    residual would be zero by construction.  We omit it by default.
    """
    if kind == "combined":
        return _combine_subspaces(
            combo_residual_subspace(data_dir, "r", side, include_default=include_default),
            combo_residual_subspace(data_dir, "t", side, include_default=include_default),
        )

    etype = _combo_marginal_etype(kind, side)   # e.g. 'r_goal'
    # Names come from the on-disk dir, falling back to combo filenames if
    # the residual dir hasn't been generated yet (the env will compute them
    # on the fly in that case).
    on_disk = list_etype_names(data_dir, etype, include_default=False) \
        if _derived_marginal_dir(data_dir, etype).exists() else []
    if not on_disk:
        goal_axis, goal_names, nogoal_axis, nogoal_names = \
            goal_aligned_names(data_dir, kind)
        on_disk = goal_names if side == "goal" else nogoal_names
    entries = [(etype, n) for n in on_disk]
    if include_default:
        entries.append(("combinations", "default"))
    return entries


def combo_centroid_subspace(data_dir: Path, kind: str, side: str,
                            include_default: bool = False
                            ) -> list[tuple[str, str]]:
    """[DEPRECATED] Build a subspace from partner-averaged combination vectors.

    .. deprecated::
        ``combo_centroid`` aggregation is biased under partner-set
        imbalance.  Use :func:`combo_residual_subspace` instead -- it
        gives identical results on a complete grid (up to a global
        shift) and is strictly more honest on incomplete grids.  The
        wrapper CLIs no longer expose ``combo_centroid``; this function
        is kept for legacy reproduction only.

    Each indexed entry is ``mean over partners B of combo(A, B)`` -- the
    legacy reduction used by the original
    ``roger/canonical_angles_goal_vs_nogoal_mutual.png`` and the
    ``combos_vs_traits_roles_pooled_{r,t}.png`` family.

    Reads from ``data_dir/combinations/vectors/{kind}_{side}_legacy_centroid/``
    when present (a one-to-one snapshot of the historical r_goal/etc. files).
    Otherwise the env recomputes centroids on the fly from the raw
    combination .pt files.

    ``include_default``: when True, append the parent ``combinations/default.pt``
    (matching the historical 31-vector ``r_goal`` layout when used together
    with ``kind='r'``).  Default ``False``.
    """
    if kind == "combined":
        return _combine_subspaces(
            combo_centroid_subspace(data_dir, "r", side, include_default=include_default),
            combo_centroid_subspace(data_dir, "t", side, include_default=include_default),
        )

    legacy_etype = f"{kind}_{side}_legacy_centroid"
    legacy_dir = _derived_marginal_dir(data_dir, legacy_etype)
    if legacy_dir.exists():
        names = list_etype_names(data_dir, legacy_etype, include_default=False)
        entries = [(legacy_etype, n) for n in names]
    else:
        # Fall back to the same names the residual subspace would produce;
        # core.ComputeEnv recomputes centroids on the fly.
        goal_axis, goal_names, nogoal_axis, nogoal_names = \
            goal_aligned_names(data_dir, kind)
        names = goal_names if side == "goal" else nogoal_names
        entries = [(f"{kind}_{side}_centroid_recompute", n) for n in names]

    if include_default:
        entries.append(("combinations", "default"))
    return entries


def _combine_subspaces(a: list[tuple[str, str]],
                       b: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Union of two subspaces, deduplicating ``(*, 'default')`` so the default
    vector appears at most once.

    Used for ``kind='combined'`` to produce 60-entry subspaces.
    """
    seen_default = False
    result: list[tuple[str, str]] = []
    for entry in list(a) + list(b):
        _, name = entry
        if name == "default":
            if seen_default:
                continue
            seen_default = True
        result.append(entry)
    return result


def combo_residual_theat_shifted_subspace(
    data_dir: Path, kind: str, side: str, include_default: bool = False
) -> list[tuple[str, str]]:
    """Combo-residual marginals shifted by the theatricality offset to the
    additive model's true origin.

    Builds the same subspace as :func:`combo_residual_subspace`, but each
    entry's etype is renamed (``r_goal_theat_shifted`` etc.) to trigger
    the on-the-fly shift in :class:`.core.ComputeEnv.get_entity`.  The
    underlying file on disk is the same as for ``r_goal``; the env just
    adds the per-(slot, layer) theatricality shift after loading.
    """
    if kind == "combined":
        return _combine_subspaces(
            combo_residual_theat_shifted_subspace(
                data_dir, "r", side, include_default=include_default),
            combo_residual_theat_shifted_subspace(
                data_dir, "t", side, include_default=include_default),
        )

    base = combo_residual_subspace(data_dir, kind, side, include_default=False)
    shifted_etype = f"{kind}_{side}_theat_shifted"
    entries = [(shifted_etype, name) for _, name in base
               if name != "default"]
    if include_default:
        entries.append(("combinations", "default"))
    return entries


def build_subspace(data_dir: Path, kind: str, side: str,
                   aggregation: str = "combo_residual_theat_shifted",
                   include_default: bool = False) -> list[tuple[str, str]]:
    """Dispatch to the appropriate subspace constructor for an
    ``(aggregation, kind, side)`` triple.

    Default aggregation is ``combo_residual_theat_shifted`` (the on-disk
    per-axis residual marginals with the per-(slot, layer) theatricality
    shift applied on the fly).  The shift cancels the constant
    "performative-persona" offset shared by all four residual subspaces,
    relocating them to the additive origin where ``combo ≈ role + trait``.
    See :func:`combo_residual_theat_shifted_subspace` and the
    "Theatricality shift" section of ``canonical_angles/README.md`` for
    the rationale.

    Pass ``aggregation='combo_residual'`` to disable the shift (e.g. for
    slot-0 analyses where the shift hurts CA1, or for "before"
    comparisons).
    """
    if aggregation == "standalone":
        return standalone_subspace(data_dir, kind, side, include_default=include_default)
    if aggregation == "combo_centroid":
        return combo_centroid_subspace(data_dir, kind, side, include_default=include_default)
    if aggregation == "combo_residual":
        return combo_residual_subspace(data_dir, kind, side, include_default=include_default)
    if aggregation == "combo_residual_theat_shifted":
        return combo_residual_theat_shifted_subspace(
            data_dir, kind, side, include_default=include_default)
    raise ValueError(f"Unknown aggregation {aggregation!r}")


# ---------------------------------------------------------------------------
# Whitening pool construction
# ---------------------------------------------------------------------------

def build_whitening_pool(data_dir: Path,
                         scope: str = "roles+traits",
                         leave_out: set[tuple[str, str]] = frozenset(),
                         ) -> list[tuple[str, str]]:
    """Build a list of ``(etype, name)`` entries to use as a whitening pool.

    Parameters
    ----------
    scope     : 'roles' | 'traits' | 'roles+traits'
    leave_out : set of ``(etype, name)`` tuples to exclude.  When the
                wrapper passes the union of subspace_a + subspace_b here,
                the whitening basis is genuinely held-out.

    Notes
    -----
    - The pool excludes ``default.pt`` (it's a corpus average, not a sample).
    - Entries with the same name across different etypes (e.g. a name that
      appears in both the goal subspace's etype and standalone trait/role
      etype) are still kept distinct here since the whitening pool is built
      from standalone vectors regardless of which etype the held-out
      subspaces used.  In practice the wrapper passes leave-out as
      standalone (etype, name) pairs.
    """
    leave_out_names: dict[str, set[str]] = defaultdict(set)
    for etype, name in leave_out:
        # Map combo marginal etypes to their underlying standalone axis so
        # the pool exclusion is consistent regardless of which etype the
        # subspace was built from.
        std_etype = _combo_marginal_to_standalone(etype) or etype
        leave_out_names[std_etype].add(name)

    if scope == "roles":
        etypes = ["roles"]
    elif scope == "traits":
        etypes = ["traits"]
    elif scope == "roles+traits":
        etypes = ["roles", "traits"]
    else:
        raise ValueError(f"Unknown pool scope {scope!r}")

    entries: list[tuple[str, str]] = []
    for etype in etypes:
        names = list_etype_names(
            data_dir, etype, include_default=False,
            exclude=leave_out_names.get(etype, frozenset()),
        )
        entries.extend((etype, n) for n in names)
    return entries


#: Bump this whenever the augmentation logic changes.  Wrapper cache
#: keys include this string so disk caches auto-invalidate on a pool
#: change (necessary because cache keys hash CLI args, not pool
#: contents -- a same-args run with a different pool implementation
#: would otherwise return stale results).
AUGMENTED_POOL_VERSION = "v2-default-only"


def build_augmented_whitening_pool(
    data_dir: Path,
    leave_out: set[tuple[str, str]] = frozenset(),
    scope: str = "roles+traits",
) -> list[tuple[str, str]]:
    """Whitening pool: held-out standalones + corpus ``default.pt``.

    Just the standard ``build_whitening_pool`` extended by a single entry,
    ``('combinations', 'default')`` -- the corpus baseline activation.
    Including ``default`` gives the whitener a natural anchor without
    needing a same-kind held-out vector.

    Historical note: an earlier version of this helper also added
    ``mean_<other-kind>_combos.pt`` (the centroid of the OTHER kind's
    combinations) on the theory that it would inject the *theatricality*
    direction (the residual-after-additive-fit common-mode axis along
    which all four ``combo_{r,t}_{goal,nogoal}`` residual subspaces
    project) into the pool's PCA so soft-K whitening could shrink it.
    Empirically that augmentation was nearly inert (CA1 changes <0.1deg
    at K=4..16) -- the standalone roles+traits pool already covers
    ``v_theat`` adequately (variance along ``v_theat`` ~33 in the
    augmented pool vs ~32 in the un-augmented pool, top-5 PCs cover
    20.6% vs 20.9% of ``v_theat``).  An attempt to force the issue by
    adding 60 copies of ``mean_<other>_combos`` did move CA1 noticeably
    but distorted the rest of the PC structure heavily (the duplicated
    vector dominated multiple top PCs).  The principled fix turned out
    to be on the *subspace* side instead -- see
    :func:`combo_residual_theat_shifted_subspace` and the "Theatricality
    shift" section of ``canonical_angles/README.md``.  We therefore keep
    only the trivial ``default`` augmentation here.

    The previous ``kind`` parameter is removed (no longer needed since
    we don't pick a cross-kind centroid).
    """
    base = list(build_whitening_pool(data_dir, scope=scope, leave_out=leave_out))
    return base + [("combinations", "default")]


def all_combinations_subspace(data_dir: Path, kind: str,
                              include_default: bool = True
                              ) -> list[tuple[str, str]]:
    """Return all raw combination vectors for a kind as ``(etype, name)`` entries.

    For ``kind='r'``: every ``r_*__*.pt`` file in ``combinations/vectors/``
    (typically ~898 entries).  ``kind='t'`` similarly (~894).  ``kind='combined'``
    is r ∪ t.  Optionally appends ``('combinations', 'default')``.

    These entries use ``etype='combinations'`` since the path is
    ``combinations/vectors/<name>.pt`` (no per-axis subdir).
    """
    if kind == "combined":
        return _combine_subspaces(
            all_combinations_subspace(data_dir, "r", include_default=False),
            all_combinations_subspace(data_dir, "t", include_default=include_default),
        )

    files = all_combo_files(data_dir, kind)
    entries = [("combinations", f.stem) for f in files]
    if include_default:
        entries.append(("combinations", "default"))
    return entries


def all_traits_roles_subspace(data_dir: Path,
                              include_default: bool = False
                              ) -> list[tuple[str, str]]:
    """Return all standalone traits + roles as ``(etype, name)`` entries.

    Used as subspace B by the ``combos_vs_traits_roles_pooled`` wrapper.
    Default is excluded by default (it's a corpus average, not a sample);
    pass ``include_default=True`` to add ``('traits', 'default')`` once.
    """
    entries: list[tuple[str, str]] = []
    for etype in ("traits", "roles"):
        for name in list_etype_names(data_dir, etype, include_default=False):
            entries.append((etype, name))
    if include_default:
        entries.append(("traits", "default"))
    return entries


def _combo_marginal_to_standalone(etype: str) -> str | None:
    """For combo marginal etypes, return the underlying standalone etype.

    Recognizes the residual etypes ('r_goal', 'r_nogoal', 't_goal', 't_nogoal'),
    the legacy centroid etypes ('r_goal_legacy_centroid', etc.), and the
    on-the-fly centroid recomputation etypes ('r_goal_centroid_recompute',
    etc.).  Returns None for non-combo etypes.
    """
    # Strip known suffixes to get the canonical kind_side label.
    base = etype
    for suffix in ("_legacy_centroid", "_centroid_recompute", "_theat_shifted"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base not in ("r_goal", "r_nogoal", "t_goal", "t_nogoal"):
        return None
    kind = base[0]
    side = base[2:]  # 'goal' or 'nogoal'
    goal_axis = GOAL_AXIS[kind]
    nogoal_axis = "trait" if goal_axis == "role" else "role"
    axis = goal_axis if side == "goal" else nogoal_axis
    return _standalone_etype(axis)


# ---------------------------------------------------------------------------
# Auto-detect number of slots from the default vector on disk
# ---------------------------------------------------------------------------

def detect_n_slots(data_dir: Path) -> int:
    """Return ``n_slots`` from default.pt under traits/.

    Used by wrappers to default --slots to 'all' without hardcoding 4 vs 8.
    """
    v = load_vector(data_dir, "traits", "default")
    return v.shape[0]


# ---------------------------------------------------------------------------
# Goal vs no-goal subspaces for soft-shear fitting
# ---------------------------------------------------------------------------

def build_goal_nogoal_subspaces(
    data_dir: Path,
    slot: int,
    layer: int,
    kind: str = "combined",
    apply_theat_shift: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the ``(goal, nogoal)`` residual-marginal subspaces at a fixed
    ``(slot, layer)``, ready for :func:`.whitening.fit_shear`.

    This is the standard subspace constructor for soft-shear fitting.  The
    returned arrays are columns-as-vectors -- shape ``(D, n_g)`` and
    ``(D, n_n)`` -- where ``n_g + n_n = 60`` for ``kind='combined'``,
    ``30 + 30`` for ``kind='r_only'`` or ``kind='t_only'``.

    Parameters
    ----------
    data_dir : Path
        Standard data directory (the one with
        ``combinations/vectors/{r_goal,r_nogoal,t_goal,t_nogoal}/`` etc.).
    slot, layer : int
        The (slot, layer) at which to extract residuals from the (n_slots,
        n_layers, D) tensors stored in each ``.pt``.
    kind : str
        - ``'combined'`` (default): pool r and t residuals -> (D, 60) vs (D, 60).
          This is the standard soft-shear fit basis -- it captures the full
          common-mode goal/no-goal direction shared by both r and t halves.
        - ``'r_only'``: r-side only -> (D, 30) vs (D, 30).  Use when you
          want a shear specific to the r-grid (goal-roles vs non-goal-traits).
        - ``'t_only'``: t-side only -> (D, 30) vs (D, 30).  Use for the
          mirror-image t-grid.
    apply_theat_shift : bool
        If True (default), add the per-(slot, layer) theatricality shift to
        each residual vector before stacking.  This relocates the residuals
        to the additive-model true origin (same convention as the project's
        canonical-angles tooling).  Set False to keep the raw on-disk
        residual frame.

    Returns
    -------
    A_goal, A_nogoal : np.ndarray
        Shape ``(D, n_*)`` each, dtype float32.

    Examples
    --------
    >>> A_g, A_n = build_goal_nogoal_subspaces(data_dir, slot=3, layer=25)
    >>> from results_analysis.canonical_angles.whitening import fit_shear
    >>> shear = fit_shear(A_g, A_n, L=5)
    >>> M_done = shear.apply(M)  # M is (n_entities, D)
    """
    if kind not in ("combined", "r_only", "t_only"):
        raise ValueError(
            f"Unknown kind {kind!r}; expected 'combined', 'r_only', or 't_only'."
        )

    # Per-(slot, layer) theatricality shift (or zeros if disabled).
    if apply_theat_shift:
        ts_full = load_theatricality_shift(data_dir)        # (n_slots, n_layers, D)
        shift = np.asarray(ts_full[slot, layer], dtype=np.float32)
    else:
        # Use a default vector to get D without loading the shift artifact.
        any_vec = load_vector(data_dir, "traits", "default")
        D = int(any_vec.shape[-1])
        shift = np.zeros(D, dtype=np.float32)

    def _load_dir(etype: str) -> np.ndarray:
        d = _derived_marginal_dir(data_dir, etype)
        if not d.exists():
            raise FileNotFoundError(
                f"Goal/no-goal residual directory not found: {d}\n"
                f"Generate it via:\n"
                f"    uv run python results_analysis/compute_combo_marginals.py")
        names = sorted(fp.stem for fp in d.glob("*.pt") if fp.stem != "default")
        if not names:
            raise RuntimeError(f"No .pt files in {d}")
        cols = []
        for n in names:
            v = torch.load(d / f"{n}.pt", weights_only=False)
            v = v["vector"] if isinstance(v, dict) else v
            v = v.float().numpy()[slot, layer]   # (D,)
            cols.append(v + shift)
        return np.stack(cols, axis=1).astype(np.float32)     # (D, n)

    if kind == "r_only":
        return _load_dir("r_goal"), _load_dir("r_nogoal")
    if kind == "t_only":
        return _load_dir("t_goal"), _load_dir("t_nogoal")
    # combined: r ∪ t along the columns
    A_rg = _load_dir("r_goal");  A_tg = _load_dir("t_goal")
    A_rn = _load_dir("r_nogoal"); A_tn = _load_dir("t_nogoal")
    return (np.concatenate([A_rg, A_tg], axis=1),
            np.concatenate([A_rn, A_tn], axis=1))
