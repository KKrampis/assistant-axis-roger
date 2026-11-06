"""Compute marginal vectors from the role x trait combination grid.

For each goal-nogoal combination kind (`r_*__*` and `t_*__*`) we produce two
"marginal" subspaces by reducing the combination vectors along one axis of
the (up to) 30x30 grid.  Two reduction modes are supported:

    --mode residual  (default, recommended)
        For each indexed key A, average over partners B of:
            combo(A, B) - standalone(B)
        i.e., the *net effect of A on top of its partner's main effect*.
        This per-row debiasing is honest under partner-set imbalance: when
        different keys have different partner sets, the per-A subtraction
        correctly removes the partner-mean drift that would otherwise leak
        into the marginal.  When the grid is full and identical partner
        sets are used by every key, the residual subspace is identical
        (up to a global shift) to the centroid subspace, so this mode is
        a strict generalization.

    --mode centroid  (legacy, exact match for the original on-disk files)
        For each indexed key A, average over partners B of:
            combo(A, B)
        i.e. the partner-averaged combination vector.  Only equivalent to
        residual on a complete grid; differs (and is biased) on incomplete
        grids.  Kept for reproducibility of the historical canonical-angles
        plots in roger/.

For both modes:

    r_*__* combinations  (filename: r_<role>__<trait>.pt; role is the goal)
        r_goal/<role>.pt    : indexed by role A (one per goal-supplying role)
                              type='role_marginal'
        r_nogoal/<trait>.pt : indexed by trait B (one per non-goal trait)
                              type='trait_marginal'

    t_*__* combinations  (filename: t_<role>__<trait>.pt; trait is the goal)
        t_goal/<trait>.pt   : indexed by trait A (one per goal-supplying trait)
                              type='trait_marginal'
        t_nogoal/<role>.pt  : indexed by role B (one per non-goal role)
                              type='role_marginal'

Stored record schema::

    {
        'vector':   bf16 tensor (n_slots, n_layers, hidden),  # the mean
        'type':     'role_marginal' | 'trait_marginal',
        'role'/'trait': <name>,                               # the index
        'metadata': {
            'mode': 'residual' | 'centroid',
            'n_partners_averaged': N,
        },
    }

Averages are computed in float32 then cast back to bf16 for storage; this
avoids the cumulative bf16 rounding that would lose ~3 bits per add.

Default behaviour:

    - --mode defaults to 'residual' (the principled choice for canonical-
      angle analysis).
    - --include_default defaults to *empty* -- i.e., no default.pt is copied
      into the output dirs.  The historical r_goal/ and r_nogoal/ dirs used
      to include default.pt to match an ad-hoc 31-vector layout, but for
      residual marginals "default minus default" is zero by construction
      and adding it is at best meaningless.

Auxiliary artifacts also written by default (see ``--skip_*`` flags):

    - ``mean_r_combos.pt`` and ``mean_t_combos.pt``  -- single-vector
      cross-kind centroids.  Originally added to the canonical-angles
      tool's whitening pool as a held-out source of the theatricality
      direction; that augmentation turned out to be nearly inert and
      was dropped (the theatricality shift on the subspace side is the
      principled fix instead).  These files are still written for
      ad-hoc analyses; pass ``--skip_kind_centroids`` to skip them.
    - ``theatricality_axis.pt``  -- per-(slot, layer) unit axis ``v_theat``
      and signed default offset, used to "shift" the residual marginals to
      the additive model's true origin (where ``combo ≈ role + trait``).
      See :func:`compute_theatricality_axis` for the math; computed
      independently from raw combos + standalone vectors (no dependence on
      the residual marginals, so applying the shift downstream is not
      circular).

Usage::

    # Recommended: regenerate residual marginals into the canonical layout
    uv run python results_analysis/compute_combo_marginals.py

    # Reproduce the legacy centroid version exactly (incl. default.pt copy):
    uv run python results_analysis/compute_combo_marginals.py \\
        --mode centroid --include_default r_goal r_nogoal

    # Write to a temp dir for verification (no in-place changes):
    uv run python results_analysis/compute_combo_marginals.py \\
        --output_root /tmp/marginals_test

Provenance note (May 2026 -- Phase 6 fill-in)
---------------------------------------------

This script's outputs (``r_goal/``, ``r_nogoal/``, ``t_goal/``, ``t_nogoal/``,
``mean_r_combos.pt``, ``mean_t_combos.pt``, ``theatricality_axis.pt``) are
**dataset artifacts** living under ``combinations/vectors/derived/`` and are
tracked via the dataset's ``MANIFEST.json`` rather than per-file provenance
envelopes.  ``MANIFEST.json`` records one fingerprint per
``derived/{marginals/<r_goal|...|t_nogoal>,axis,aggregates}`` subtree, and
downstream consumers depend on those subtree fingerprints via
:func:`assistant_axis.provenance.current_data_subtree_input` calls (see e.g.
``results_analysis/canonical_angles/plots/*.py``).

Workflow when this script's output changes:

1. Re-run this script (``uv run python results_analysis/compute_combo_marginals.py``).
2. Regenerate the manifest::

       uv run python tools/regenerate_dataset_manifest.py \\
           --data-dir <data_dir>

   See :mod:`tools.regenerate_dataset_manifest` for flags; the schema
   lives in :mod:`assistant_axis.provenance` (``Manifest`` /
   ``ManifestSubtree``).
3. Downstream cache audits will then flag any consumer whose recorded
   subtree fingerprint no longer matches.

Adding a per-file ``json_metadata``-style envelope here would be redundant
with the manifest, so this script is intentionally NOT migrated to the
writer convention -- the manifest IS its provenance record.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch


# Updated to the 8-slot Roger data; the script is already slot-agnostic
# (n_slots is detected from the loaded tensor shape), so the same code
# handles 4-slot Christina-headers data via ``--data_dir`` override.
DEFAULT_DATA_DIR = "runpod_workspace/qwen/qwen-3-32b Roger 8slot/combinations/vectors"

# Which combination kinds use which side as the "goal".
# r_*__* : role supplies goal, trait is non-goal.
# t_*__* : trait supplies goal, role is non-goal.
GOAL_AXIS = {"r": "role", "t": "trait"}


def parse_combo_stem(stem: str, kind: str) -> tuple[str, str]:
    """Parse a combination filename stem like 'r_activist__abstract' into
    (role, trait).  Note: the filename order is always role__trait regardless
    of kind; only the meaning (which is goal) changes with kind."""
    prefix = f"{kind}_"
    assert stem.startswith(prefix), f"stem {stem!r} does not start with {prefix!r}"
    rest = stem[len(prefix):]
    role, trait = rest.split("__", 1)
    return role, trait


def axis_for(kind: str, side: str) -> str:
    """Return which axis of the (role, trait) pair is the *index* for this
    output subspace, i.e. what we group by when averaging.

    For r_*__*: side='goal' indexes by role; side='nogoal' indexes by trait.
    For t_*__*: side='goal' indexes by trait; side='nogoal' indexes by role.
    """
    goal_axis = GOAL_AXIS[kind]            # 'role' or 'trait'
    other_axis = "trait" if goal_axis == "role" else "role"
    return goal_axis if side == "goal" else other_axis


def _standalone_dir(data_dir: Path, axis: str) -> Path:
    """Resolve standalone vectors dir for a given axis.

    The combination data_dir typically lives under
    ``<root>/combinations/vectors``; standalone trait/role vectors live at
    ``<root>/{traits,roles}/vectors``.  We climb two levels up to find them.
    """
    root = data_dir.parent.parent     # combinations/vectors -> combinations -> root
    return root / ({"role": "roles", "trait": "traits"}[axis]) / "vectors"


def _load_standalone(data_dir: Path, axis: str, name: str) -> torch.Tensor:
    """Load a standalone trait/role vector tensor (any dtype, on CPU)."""
    p = _standalone_dir(data_dir, axis) / f"{name}.pt"
    return torch.load(p, weights_only=False)["vector"]


def compute_marginals(
    data_dir: Path,
    kind: str,
    side: str,
    output_dir: Path,
    include_default: bool,
    mode: str,
) -> dict[str, int]:
    """Compute marginal vectors for one (kind, side, mode) and write them to
    output_dir.  Returns a dict mapping each indexed name to the number of
    partners averaged over.

    Float32 accumulation then bf16 cast on save.
    """
    if mode not in ("residual", "centroid"):
        raise ValueError(f"Unknown mode {mode!r}")

    index_axis = axis_for(kind, side)               # 'role' or 'trait'
    type_label = f"{index_axis}_marginal"
    partner_axis = "trait" if index_axis == "role" else "role"

    pattern = str(data_dir / f"{kind}_*__*.pt")
    import glob
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No combination files matching {pattern}")

    # Group raw combination vectors by index, also remembering each
    # contribution's partner name so residual mode can subtract the partner's
    # standalone vector.
    grouped: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for f in files:
        stem = Path(f).stem
        role, trait = parse_combo_stem(stem, kind)
        index_name = role if index_axis == "role" else trait
        partner_name = trait if partner_axis == "trait" else role
        v = torch.load(f, weights_only=False)["vector"]
        grouped[index_name].append((partner_name, v))

    # Cache standalone partners so we don't re-load the same file 30x in
    # residual mode.  Only populated when needed.
    partner_cache: dict[str, torch.Tensor] = {}

    def get_partner(name: str) -> torch.Tensor:
        cached = partner_cache.get(name)
        if cached is None:
            cached = _load_standalone(data_dir, partner_axis, name).float()
            partner_cache[name] = cached
        return cached

    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for key, parts in sorted(grouped.items()):
        # Stack contributions into a (n_partners, *) tensor in float32.
        if mode == "centroid":
            stacked = torch.stack([v for _, v in parts]).float()
        else:  # residual
            stacked = torch.stack(
                [v.float() - get_partner(pn) for pn, v in parts]
            )
        mean_vec = stacked.mean(dim=0).bfloat16()
        record = {
            "vector": mean_vec,
            "type": type_label,
            index_axis: key,
            "metadata": {
                "mode": mode,
                "n_partners_averaged": len(parts),
                # Keep the legacy field name too so older readers don't break.
                f"n_{partner_axis}s_averaged": len(parts),
            },
        }
        torch.save(record, output_dir / f"{key}.pt")
        counts[key] = len(parts)

    if include_default:
        # Copy the data_dir's default.pt into the output_dir as a sibling.
        # We resolve any symlink to the actual file so the copy is independent.
        # (Note: this only makes sense for centroid mode -- a residual default
        # is zero by construction.  We honor the flag anyway and just copy
        # the raw default.pt; the metadata will indicate it is unmodified.)
        default_src = data_dir / "default.pt"
        if default_src.exists():
            real = Path(os.path.realpath(default_src))
            shutil.copy2(real, output_dir / "default.pt")

    return counts


def compute_kind_centroid(data_dir: Path, kind: str, output_path: Path) -> int:
    """Compute the simple unweighted mean of all ``{kind}_*__*.pt`` combination
    vectors and save it to ``output_path`` as a single-vector .pt file.

    Auxiliary artifact.  Originally added to the canonical-angles
    tool's whitening pool as a cross-kind, held-out source of the
    theatricality direction (since both r_ and t_ combinations share
    the same additive structure, ``mean_t_combos`` projects strongly
    onto ``v_theat`` and was meant to give the whitener access to it
    when analysing kind=r).  Empirically that augmentation barely
    shifted CA1 at the K values we care about (<0.1deg vs the
    standalone-only pool plus default), so the canonical-angles tool
    no longer reads these files; the principled fix is the
    theatricality shift on the subspace side
    (``combo_residual_theat_shifted``).  Kept for ad-hoc analyses.

    Stored format mirrors a single-vector entry:
        {'vector': bf16, 'type': 'kind_centroid', 'kind': <r|t>,
         'metadata': {'n_combos_averaged': N}}
    """
    import glob
    pattern = str(data_dir / f"{kind}_*__*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No combination files matching {pattern}")
    stacked = torch.stack([torch.load(f, weights_only=False)["vector"]
                           for f in files])
    mean_vec = stacked.float().mean(dim=0).bfloat16()
    record = {
        "vector": mean_vec,
        "type": "kind_centroid",
        "kind": kind,
        "metadata": {"n_combos_averaged": len(files)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, output_path)
    return len(files)


def compute_theatricality_axis(data_dir: Path, output_path: Path
                               ) -> dict[str, tuple]:
    """Compute the per-(slot, layer) theatricality axis and shift parameters,
    saving as a single multi-component .pt file.

    Background
    ----------
    The variance-decomposition analysis (see ``roger/variance_decomp_bars.png``)
    found that combination vectors are well modelled by an additive model
    plus a single shared offset along a particular direction we call the
    *theatricality axis*:

        combo(role, trait) ≈ role + trait − μ_pool + X · v_theatricality

    where ``μ_pool`` is the centroid of all standalone roles+traits and
    ``X · v_theatricality`` is a constant per-slot/layer offset.  The axis
    itself is computed as:

        v_theatricality(slot, layer) ∝
            mean over (role, trait) of (combo − role − trait + μ_pool)

    averaged across kinds (r and t).  We normalize to a unit vector for
    convenience.

    The accompanying *default offset* is the signed projection of
    ``(default − μ_pool)`` onto ``v_theatricality_unit``:

        offset(slot, layer) = (default − μ_pool) · v_theatricality_unit

    The natural correction for residual marginals (which sit at roughly
    ``+|offset|`` along the axis at the "combinations end") is to apply the
    shift ``offset × v_theatricality_unit`` per (slot, layer) -- this moves
    the additive-model origin from ``μ_pool`` to the actual centre of the
    additive model (``μ_pool + offset · v_theatricality_unit``).

    Stored format
    -------------
    Output dict has shape::

        {
            "axis":            (n_slots, n_layers, hidden)  bf16,
              # per-(slot, layer) UNIT theatricality direction
            "default_offset":  (n_slots, n_layers)          float32,
              # (default − μ_pool) · axis_unit, signed scalar per slot/layer
            "raw_norm":        (n_slots, n_layers)          float32,
              # ‖v_theatricality_raw‖ before normalization
            "metadata": {
                "n_combos_averaged": {"r": N_r, "t": N_t},
                "n_pool_entries":    N_pool,    # excludes default
                "kinds_used":        ["r", "t"],
                "doc": "v_theat = avg over kinds of mean(combo - role - trait + μ_pool)",
            },
        }

    Note: the axis is computed independently from the residual marginals
    (it is a function of raw combinations + standalone vectors only).  So
    applying the shift to the residual marginals does NOT introduce
    circularity in the axis definition.
    """
    import glob
    # Locate standalone dirs (assume DATA/<traits|roles>/vectors)
    traits_dir = _standalone_dir(data_dir, "trait")
    roles_dir  = _standalone_dir(data_dir, "role")

    def _stack_dir(d: Path) -> torch.Tensor:
        return torch.stack([torch.load(p, weights_only=False)["vector"]
                            for p in sorted(d.glob("*.pt"))
                            if p.stem != "default"])

    # μ_pool: mean of all standalone roles+traits at every (slot, layer)
    traits_all = _stack_dir(traits_dir).float()    # (n_traits, n_slots, n_layers, hidden)
    roles_all  = _stack_dir(roles_dir).float()
    pool = torch.cat([traits_all, roles_all], dim=0)   # (n_pool, n_slots, n_layers, hidden)
    mu_pool = pool.mean(dim=0)                          # (n_slots, n_layers, hidden)
    n_slots, n_layers, hidden = mu_pool.shape

    # Default vector (from data_dir)
    default_p = data_dir / "default.pt"
    if not default_p.exists():
        raise RuntimeError(f"default.pt not found in {data_dir}")
    default_full = torch.load(default_p, weights_only=False)["vector"].float()  # (n_slots, n_layers, hidden)

    # Per-kind residual mean: averaged combo - role - trait + μ_pool
    def _mean_residual(kind: str) -> torch.Tensor:
        files = sorted(glob.glob(str(data_dir / f"{kind}_*__*.pt")))
        if not files:
            raise RuntimeError(f"No combination files for kind={kind!r} in {data_dir}")
        # Accumulate sums in float32
        sum_vec = torch.zeros((n_slots, n_layers, hidden), dtype=torch.float32)
        n_pairs = 0
        for fp in files:
            stem = Path(fp).stem
            role, trait = parse_combo_stem(stem, kind)
            combo = torch.load(fp, weights_only=False)["vector"].float()
            R = torch.load(roles_dir / f"{role}.pt",  weights_only=False)["vector"].float()
            T = torch.load(traits_dir / f"{trait}.pt", weights_only=False)["vector"].float()
            sum_vec += combo - R - T + mu_pool
            n_pairs += 1
        return sum_vec / n_pairs, n_pairs

    v_r, n_r = _mean_residual("r")
    v_t, n_t = _mean_residual("t")
    v_raw = 0.5 * (v_r + v_t)                          # (n_slots, n_layers, hidden)
    raw_norm = torch.linalg.norm(v_raw, dim=-1)        # (n_slots, n_layers)
    # Avoid divide-by-zero on degenerate slots/layers
    safe_norm = torch.clamp(raw_norm, min=1e-12).unsqueeze(-1)
    axis_unit = v_raw / safe_norm                      # (n_slots, n_layers, hidden)

    # Default offset: signed projection of (default - μ_pool) on axis_unit
    default_centered = default_full - mu_pool
    default_offset = (default_centered * axis_unit).sum(dim=-1)   # (n_slots, n_layers)

    record = {
        "axis": axis_unit.bfloat16(),
        "default_offset": default_offset.float(),
        "raw_norm": raw_norm.float(),
        "metadata": {
            "n_combos_averaged": {"r": n_r, "t": n_t},
            "n_pool_entries": int(pool.shape[0]),
            "kinds_used": ["r", "t"],
            "doc": "v_theat = avg over kinds of mean(combo - role - trait + mu_pool); "
                   "axis = unit-normalized; default_offset = (default - mu_pool) . axis. "
                   "Apply default_offset * axis as shift to residual marginals.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, output_path)
    return {"r": n_r, "t": n_t, "n_pool": int(pool.shape[0])}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help="dir containing the r_*__*.pt / t_*__*.pt files and default.pt "
                        f"(default: {DEFAULT_DATA_DIR})")
    p.add_argument("--output_root", default=None,
                   help="where to write the derived/ tree; default = "
                        "data_dir/derived (which holds marginals/, "
                        "aggregates/, axis/ subdirs).  Pass an explicit "
                        "path to redirect, e.g. for legacy-centroid "
                        "snapshot reproduction.")
    p.add_argument("--layout", default="categorized",
                   choices=["categorized", "flat"],
                   help="On-disk layout under output_root.  'categorized' "
                        "(default, post-Phase-1.0) splits outputs into "
                        "marginals/{kind}_{side}/, aggregates/, axis/.  "
                        "'flat' writes everything directly under "
                        "output_root (the legacy pre-Phase-1.0 layout, "
                        "useful for legacy_centroid snapshot generation "
                        "into a custom dir).")
    p.add_argument("--kinds", nargs="+", default=["r", "t"], choices=["r", "t"],
                   help="which combination kinds to process (default: both)")
    p.add_argument("--sides", nargs="+", default=["goal", "nogoal"],
                   choices=["goal", "nogoal"],
                   help="which sides to compute (default: both)")
    p.add_argument("--mode", default="residual", choices=["residual", "centroid"],
                   help="reduction mode: 'residual' (default) subtracts the partner's "
                        "standalone vector before averaging; 'centroid' just averages "
                        "the raw combinations (legacy behaviour, biased on incomplete "
                        "grids)")
    p.add_argument("--include_default", nargs="*", default=[],
                   help="output subdirs that should additionally receive a copy of "
                        "default.pt; pass e.g. 'r_goal r_nogoal' to recreate the legacy "
                        "31-vector layout.  Default: empty (no default.pt copies).")
    p.add_argument("--skip_kind_centroids", action="store_true",
                   help="Skip writing mean_{r,t}_combos.pt (the cross-kind "
                        "centroids -- auxiliary; no longer used by the "
                        "canonical-angles tool, but cheap to produce so "
                        "they're written by default).")
    p.add_argument("--skip_theatricality_axis", action="store_true",
                   help="Skip writing theatricality_axis.pt (the per-(slot, "
                        "layer) theatricality direction + default offset). "
                        "Default: write it.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = data_dir / "derived"

    print(f"data_dir:    {data_dir}")
    print(f"output_root: {output_root}")
    print(f"layout:      {args.layout}")
    print(f"mode:        {args.mode}")
    if not data_dir.exists():
        raise SystemExit(f"data_dir does not exist: {data_dir}")

    if args.layout == "categorized":
        marginals_root = output_root / "marginals"
        aggregates_root = output_root / "aggregates"
        axis_root = output_root / "axis"
    else:  # flat
        marginals_root = output_root
        aggregates_root = output_root
        axis_root = output_root

    include_default_set = set(args.include_default)
    for kind in args.kinds:
        for side in args.sides:
            label = f"{kind}_{side}"
            out_dir = marginals_root / label
            inc_default = label in include_default_set
            print(f"\n=== {label} ({args.mode}) -> {out_dir} "
                  f"(include_default={inc_default}) ===")
            counts = compute_marginals(
                data_dir, kind, side, out_dir, inc_default, args.mode,
            )
            n_files = len(counts) + (1 if inc_default else 0)
            min_n = min(counts.values()) if counts else 0
            max_n = max(counts.values()) if counts else 0
            partial = {k: v for k, v in counts.items() if v != max_n}
            print(f"  wrote {n_files} files (incl. default = {inc_default})")
            print(f"  n_partners: min={min_n}, max={max_n}")
            if partial:
                print(f"  partial samples: {partial}")

    if not args.skip_kind_centroids:
        for kind in args.kinds:
            out_path = aggregates_root / f"mean_{kind}_combos.pt"
            n = compute_kind_centroid(data_dir, kind, out_path)
            print(f"\n  wrote {out_path} (mean of {n} {kind}_*__* combinations)")

    if not args.skip_theatricality_axis:
        out_path = axis_root / "theatricality_axis.pt"
        info = compute_theatricality_axis(data_dir, out_path)
        print(f"\n  wrote {out_path} (per-(slot, layer) axis from "
              f"{info['r']} r_ + {info['t']} t_ combos, "
              f"pool size {info['n_pool']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
