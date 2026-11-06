"""Mutual canonical angles between the goal and non-goal subspaces.

The default configuration uses ``combo_residual_theat_shifted``
aggregation with ``origin=none`` -- residual marginals translated to
the additive origin via the per-(slot, layer) theatricality shift; see
``canonical_angles/README.md`` for the math.  Residuals are deltas
with a semantic zero (``residual[A] = 0`` means "A adds nothing on top
of its partner"), so no further centering is needed and all 30
dimensions of the indexed axis are retained.  Pass
``--aggregation combo_residual`` to disable the shift (recommended for
slot-0 analyses where the shift hurts CA1).

Note: the legacy ``combo_centroid`` aggregation (used by the original
``roger/canonical_angles_goal_vs_nogoal_mutual.png``) is no longer
exposed via the CLI.  It is biased on incomplete grids; the residual
version is a strict generalization.  The deprecated code paths still
exist in :mod:`results_analysis.canonical_angles.data` and can be invoked
programmatically if needed.

What "goal" and "nogoal" mean
-----------------------------

For ``kind='r'`` (role supplies goal, trait is non-goal):

- Subspace A ("goal"):    span of the 30 role-marginals in ``r_goal/``.
- Subspace B ("nogoal"):  span of the 30 trait-marginals in ``r_nogoal/``.

The on-disk r_goal/, r_nogoal/, t_goal/, t_nogoal/ files now hold
*residual* marginals: each entry is the mean over partner B of
``combo(A,B) - standalone(B)``.  See ``compute_combo_marginals.py``.
The legacy centroid versions are preserved at ``*_legacy_centroid/`` and
are used by ``--aggregation combo_centroid``.

For ``kind='t'``: trait supplies goal, role is non-goal.

For ``kind='combined'``: subspace A = r_goal ∪ t_goal (60 vectors),
subspace B = r_nogoal ∪ t_nogoal (60 vectors).

Centering and whitening
-----------------------

The default centering origin is ``none``.  ``self_mean`` is available
for users who want to compare only the residual *spread*, not the
direction of average effect.  Other origin modes (``default``,
``pool_mean``, ``full_combo_mean``, ``paired``) are exposed via
``--origin``.

Whitening defaults to ``['raw', 'soft_K=2']`` -- raw and the project
soft-K default (see ``whitening.DEFAULT_SOFT_K``) plotted side by side.
Pass any subset of ``raw|soft_K=N|lw|oas|soft_shear=L`` to override; the
basis is fit on the ``--pool`` of standalone roles+traits (with held-out
entities optionally excluded) and applied to each subspace before SVD.

Examples
--------

    # Default (residual + origin=none), both kinds:
    uv run python -m results_analysis.canonical_angles.plots.goal_vs_nogoal_mutual \\
        --output roger/canonical_angles_goal_vs_nogoal_mutual.png

    # Whitening sweep, kind=combined:
    uv run python -m results_analysis.canonical_angles.plots.goal_vs_nogoal_mutual \\
        --kinds combined --whitening raw soft_K=4 soft_K=128 \\
        --slots 3 --output roger/ca_goal_nogoal_whitening_sweep.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import (
    AggSpec,
    CASpec,
    ComputeEnv,
    OriginSpec,
    WhiteningSpec,
    compute_ca_grid,
)
from ..data import (
    DEFAULT_DATA_DIR,
    build_subspace,
    build_augmented_whitening_pool,
    build_whitening_pool,
    detect_n_slots,
)
from ..plot_helpers import plot_per_slot_panels
from ..whitening import DEFAULT_SOFT_K, parse_whitening_spec
from assistant_axis import png_metadata
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_slot_spec(s: str) -> tuple[tuple[int, ...], str]:
    """Parse a CLI slot spec into ``(indices_tuple, mode)``.

    Accepted forms:
        '3'                 -> ((3,), 'single')
        'avg(0,3)'          -> ((0, 3), 'avg')
        'concat(3,7)'       -> ((3, 7), 'concat')
    """
    s = s.strip()
    if s.startswith("avg(") and s.endswith(")"):
        body = s[4:-1]
        idxs = tuple(int(x) for x in body.split(","))
        return idxs, "avg"
    if s.startswith("concat(") and s.endswith(")"):
        body = s[7:-1]
        idxs = tuple(int(x) for x in body.split(","))
        return idxs, "concat"
    return (int(s),), "single"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help=f"Cached vectors directory (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--kinds", nargs="+", default=["r", "t"],
                   choices=["r", "t", "combined"],
                   help="Combination kinds (default: r t)")
    p.add_argument("--slots", nargs="+", default=None,
                   help="Slot specs: int / 'avg(0,3)' / 'concat(3,7)'. "
                        "Default: all slots from default.pt.")
    p.add_argument("--layer", type=int, default=25,
                   help="Transformer layer (default: 25 -- Qwen-3-32B "
                        "optimum from rho_by_layer.py)")
    p.add_argument("--whitening", nargs="+",
                   default=["raw", f"soft_K={DEFAULT_SOFT_K}"],
                   help=f"Whitening regimes: 'raw' | 'soft_K=N' | 'lw' | 'oas' "
                        f"| 'soft_shear=L' (default: raw soft_K={DEFAULT_SOFT_K} "
                        f"-- see whitening.DEFAULT_SOFT_K)")
    p.add_argument("--pool", default="roles+traits",
                   choices=["roles", "traits", "roles+traits"],
                   help="Whitening pool scope (only used when whitening != raw)")
    p.add_argument("--no-heldout", dest="heldout", action="store_false",
                   help="Disable holding out subspace entities from the whitening "
                        "pool (default: held out)")
    p.set_defaults(heldout=True)
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Disable pool augmentation (default: augmented -- "
                        "pool gains the corpus default.pt; see "
                        "canonical_angles/README.md).")
    p.set_defaults(augment=True)
    p.add_argument("--aggregation", default="combo_residual_theat_shifted",
                   choices=["standalone", "combo_residual",
                            "combo_residual_theat_shifted"],
                   help="How to construct goal/nogoal subspaces "
                        "(default: combo_residual, the partner-debiased version). "
                        "The legacy 'combo_centroid' aggregation has been removed "
                        "from the CLI; see data.py if you need the deprecated "
                        "centroid path.")
    p.add_argument("--origin", default=None,
                   choices=["none", "default", "self_mean", "pool_mean",
                            "pool_mean_per_side", "full_combo_mean", "paired"],
                   help="Centering origin.  Default depends on --aggregation: "
                        "combo_residual -> 'none' (residuals already have a "
                        "semantic zero); standalone -> 'pool_mean_per_side' "
                        "(removes axis-specific assistant-context baseline). "
                        "Pass explicitly to override.")
    p.add_argument("--include_default", dest="include_default",
                   action="store_true",
                   help="Append default.pt to each subspace (default: omitted)")
    p.set_defaults(include_default=False)
    p.add_argument("--output", required=True, help="Output PNG path")
    p.add_argument("--ylim_max", type=float, default=95.0,
                   help="Y-axis upper limit for plots (default: 95)")
    p.add_argument("--bars", action="store_true",
                   help="Render as bar chart per panel. "
                        "Default: line chart, suitable for whitening sweeps.")
    args = p.parse_args()
    # Apply the per-aggregation default if --origin wasn't passed.
    if args.origin is None:
        args.origin = ("none" if args.aggregation in ("combo_residual",
                                                      "combo_residual_theat_shifted")
                       else "pool_mean_per_side")
    # Refuse the obviously-degenerate standalone + none combination.  The
    # underlying compute path also emits a warning for programmatic callers,
    # but at the CLI level we hard-fail to nudge users toward a sensible
    # origin (default / self_mean / pool_mean_per_side).
    if args.aggregation == "standalone" and args.origin == "none":
        raise SystemExit(
            "ERROR: --aggregation standalone --origin none leaves the "
            "assistant-context baseline in both subspaces and gives a "
            "near-zero first canonical angle that does NOT reflect any "
            "meaningful alignment.  Pick --origin default | self_mean | "
            "pool_mean_per_side instead."
        )
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)

    # Resolve slot list (default = all slots from default.pt)
    if args.slots is None:
        n_slots = detect_n_slots(data_dir)
        slot_specs = [((i,), "single") for i in range(n_slots)]
    else:
        slot_specs = [parse_slot_spec(s) for s in args.slots]

    # Default to line plots; users can opt in to bars with --bars.  (The
    # original ``roger/canonical_angles_goal_vs_nogoal_mutual.png`` rendered
    # as bars; that was a historical artefact, not a deliberate choice.)
    use_bars = args.bars

    # Build all CA specs in the (kind, slot, whitening) Cartesian product.
    specs = []
    metadata: dict[CASpec, dict] = {}  # for tagging the result records
    for kind in args.kinds:
        # Subspace A (goal) and B (nogoal) for this kind.
        sub_a = tuple(build_subspace(data_dir, kind, "goal",
                                     args.aggregation, args.include_default))
        sub_b = tuple(build_subspace(data_dir, kind, "nogoal",
                                     args.aggregation, args.include_default))
        # The set of entities to leave out of the whitening pool when
        # held-out is requested.  We map combo-marginal etypes to their
        # underlying standalone axis inside build_whitening_pool, so passing
        # the subspace entries directly is correct.
        leave_out = set(sub_a) | set(sub_b) if args.heldout else set()

        for slot_indices, slot_mode in slot_specs:
            for w_str in args.whitening:
                method, K = parse_whitening_spec(w_str)
                if method == "raw":
                    pool: tuple[tuple[str, str], ...] = ()
                elif args.augment:
                    pool = tuple(build_augmented_whitening_pool(
                        data_dir, leave_out=leave_out, scope=args.pool))
                else:
                    pool = tuple(build_whitening_pool(
                        data_dir, scope=args.pool, leave_out=leave_out))

                spec = CASpec(
                    subspace_a=sub_a,
                    subspace_b=sub_b,
                    slot_indices=tuple(slot_indices),
                    slot_mode=slot_mode,
                    layer=args.layer,
                    aggregation=AggSpec(mode=args.aggregation,
                                        include_default=args.include_default),
                    origin=OriginSpec(kind=args.origin, combos_kind=kind),
                    whitening=WhiteningSpec(method=method, K=K, pool=pool),
                )
                specs.append(spec)
                metadata[spec] = dict(
                    kind=kind,
                    slot_label=_slot_label(slot_indices, slot_mode),
                    whitening=w_str,
                )

    print(f"Computing {len(specs)} canonical-angle specs "
          f"({len(args.kinds)} kinds x {len(slot_specs)} slots x "
          f"{len(args.whitening)} whitenings) at layer {args.layer} ...")

    env = ComputeEnv(data_dir=data_dir)
    results = compute_ca_grid(specs, data_dir=data_dir, env=env)

    # Build long-format records for plot_helpers.
    records = []
    for spec in specs:
        meta = metadata[spec]
        angles = results[spec]
        # Panel: one per (kind, slot) so multiple whitenings overlay within
        # a panel.  Plot title shows kind and slot separately for clarity.
        panel = (meta["kind"], meta["slot_label"])
        records.append({
            "panel": panel,
            "kind": meta["kind"],
            "slot_label": meta["slot_label"],
            "whitening": meta["whitening"],
            "angles": angles,
        })

    # Plot.
    title = (
        f"Mutual Canonical Angles: Goal vs Non-goal "
        f"(layer {args.layer}, agg={args.aggregation}, origin={args.origin})"
    )

    fig = plot_per_slot_panels(
        records,
        panel_key="panel",
        group_key="whitening" if len(args.whitening) > 1 else None,
        as_bars=use_bars,
        layout=_choose_layout(len(args.kinds), len(slot_specs)),
        title=title,
        panel_title_fn=lambda v: _panel_title(v, len(args.kinds), len(slot_specs)),
        ylim=(0, args.ylim_max),
        legend_kwargs={"fontsize": 8, "loc": "lower right"},
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = _build_inputs(data_dir, args)
    fig.savefig(out_path, dpi=140, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    print(f"Wrote {out_path}")

    # Brief summary printed for sanity.
    for spec in specs:
        meta = metadata[spec]
        ang = results[spec]
        if len(ang) == 0:
            print(f"  {meta} -> 0 angles (empty subspace)")
        else:
            import numpy as np
            n_total = len(ang)
            n_nan = int(np.sum(np.isnan(ang)))
            n_eff = n_total - n_nan
            tag = f" ({n_nan} NaN-padded)" if n_nan else ""
            print(f"  {meta}  n={n_total:2d}{tag}  "
                  f"min={float(np.nanmin(ang)):5.1f}  "
                  f"med={_median(ang):5.1f}  "
                  f"max={float(np.nanmax(ang)):5.1f}")
    return 0


def _build_inputs(data_dir: Path, args: argparse.Namespace) -> list[InputSpec]:
    """Provenance inputs for one run.

    Declares the goal/nogoal marginal subtrees actually used (per
    ``--kinds``), plus the theatricality-axis subtree when residuals
    are theat-shifted, plus standalone subtrees when the aggregation
    is ``standalone`` OR when a non-raw whitening pool is in use.
    """
    extras = {"layer": str(args.layer),
              "kinds": ",".join(args.kinds),
              "aggregation": args.aggregation,
              "origin": args.origin,
              "whitening": ",".join(args.whitening),
              "pool": args.pool,
              "heldout": "1" if args.heldout else "0",
              "augment": "1" if args.augment else "0",
              "include_default": "1" if args.include_default else "0"}
    inputs: list[InputSpec] = []
    needs_r = "r" in args.kinds or "combined" in args.kinds
    needs_t = "t" in args.kinds or "combined" in args.kinds
    if args.aggregation in ("combo_residual", "combo_residual_theat_shifted"):
        if needs_r:
            inputs.append(current_data_subtree_input(
                data_dir, "combinations/vectors/derived/marginals/r_goal",
                dep_key="r_goal_marginals", extras=extras))
            inputs.append(current_data_subtree_input(
                data_dir, "combinations/vectors/derived/marginals/r_nogoal",
                dep_key="r_nogoal_marginals", extras=extras))
        if needs_t:
            inputs.append(current_data_subtree_input(
                data_dir, "combinations/vectors/derived/marginals/t_goal",
                dep_key="t_goal_marginals", extras=extras))
            inputs.append(current_data_subtree_input(
                data_dir, "combinations/vectors/derived/marginals/t_nogoal",
                dep_key="t_nogoal_marginals", extras=extras))
    if args.aggregation == "combo_residual_theat_shifted":
        inputs.append(current_data_subtree_input(
            data_dir, "combinations/vectors/derived/axis",
            dep_key="theatricality_axis", extras=extras))
    # Pool / standalone subspaces.  Whitening != "raw" pulls from
    # roles+traits per ``--pool``; ``standalone`` aggregation reads them
    # to build the subspaces themselves.
    needs_pool = any(w != "raw" for w in args.whitening)
    if args.aggregation == "standalone" or needs_pool:
        if "roles" in args.pool or args.aggregation == "standalone":
            inputs.append(current_data_subtree_input(
                data_dir, "roles/vectors", dep_key="roles_vectors",
                extras=extras))
        if "traits" in args.pool or args.aggregation == "standalone":
            inputs.append(current_data_subtree_input(
                data_dir, "traits/vectors", dep_key="traits_vectors",
                extras=extras))
    return inputs


def _slot_label(slot_indices: tuple[int, ...], slot_mode: str) -> str:
    if slot_mode == "single":
        return str(slot_indices[0])
    if slot_mode in ("avg", "average"):
        return f"avg({','.join(str(i) for i in slot_indices)})"
    if slot_mode == "concat":
        return f"concat({','.join(str(i) for i in slot_indices)})"
    return f"{slot_mode}{slot_indices}"


def _panel_title(panel_value, n_kinds: int, n_slots: int) -> str:
    """Build a panel title from the (kind, slot_label) tuple."""
    kind, slot_label = panel_value
    if n_kinds == 1:
        return _slot_pretty(slot_label)
    if n_slots == 1:
        return f"kind={kind}"
    return f"{kind} / slot {_slot_pretty(slot_label)}"


def _slot_pretty(slot_label: str) -> str:
    """Translate a slot index into the human-readable header label."""
    # Map known slot indices to human-readable headers (matches historical plot)
    pretty = {
        "0": "Slot 0 (Body mean)",
        "1": "Slot 1 (<|im_start|>)",
        "2": "Slot 2 (assistant)",
        "3": "Slot 3 (\\n)",
        "4": "Slot 4 (<think>)",
        "5": "Slot 5 (\\n\\n inside think)",
        "6": "Slot 6 (</think>)",
        "7": "Slot 7 (\\n\\n final)",
    }
    return pretty.get(slot_label, f"Slot {slot_label}")


def _choose_layout(n_kinds: int, n_slots: int) -> tuple[int, int]:
    """Default panel layout for (kinds x slots).

    Convention: rows = kinds (when more than one), cols = slots.  For one
    kind we collapse to a square-ish layout matching the historical 2x2 (4
    slots) or 2x4 (8 slots) plots.
    """
    if n_kinds == 1:
        if n_slots == 4:
            return (2, 2)
        if n_slots == 8:
            return (2, 4)
        # fallback for arbitrary n_slots with 1 kind: square-ish
        import math
        cols = int(math.ceil(math.sqrt(n_slots)))
        rows = int(math.ceil(n_slots / cols))
        return (rows, cols)
    # n_kinds >= 2: rows = kinds, cols = slots
    return (n_kinds, n_slots)


def _median(arr) -> float:
    import numpy as np
    return float(np.nanmedian(arr))


if __name__ == "__main__":
    raise SystemExit(main())
