"""Goal-vs-nogoal canonical angles, swept across multiple soft-K whitening regimes,
with the whitening pool being **all combinations of that kind**.

Reproduces the family ``roger/canonical_angles_combos_vs_traits_roles_pooled_{r,t}.png``
(historical title: *"Canonical Angles: raw vs all-combinations soft-whitening
at various K (<kind>_, Layer 24)"*).

Note on naming: the historical filename mentions "traits_roles_pooled" but
the plot title and structure are about a combinations-based pool, not a
trait+role pool.  The filename appears to be a misnomer kept for path
stability.

What it does
------------

The subspace pair is the same as :mod:`.goal_vs_nogoal_mutual` (goal
marginals vs non-goal marginals via on-disk ``r_goal/`` etc.), but instead
of a single regime per panel the wrapper plots one curve per K value:

- ``raw``                no whitening
- ``soft_K=16``          whitened against all combinations of this kind, K=16
- ``soft_K=32``          ditto, K=32
- ``soft_K=64``          K=64
- ``soft_K=128``         K=128

The whitening pool is the set of all ``r_*__*`` (or ``t_*__*``) combination
vectors plus ``default.pt`` -- the same dataset whose centroid we use as the
``full_combo_mean`` origin.  Useful as a pool because it's the natural
"data context" of the goal/nogoal subspaces being compared.

Despite the historical filename mentioning "traits_roles_pooled", the title
and content of the existing PNGs are about a combinations-based pool; the
filename appears to be a misnomer kept for path stability.

Examples
--------

    # Reproduce the r-side plot exactly
    uv run python -m results_analysis.canonical_angles.plots.combos_vs_traits_roles_pooled \
        --kind r --output /tmp/test_combos_r.png

    # Same for t
    uv run python -m results_analysis.canonical_angles.plots.combos_vs_traits_roles_pooled \
        --kind t --output /tmp/test_combos_t.png

    # Custom K sweep
    uv run python -m results_analysis.canonical_angles.plots.combos_vs_traits_roles_pooled \
        --kind r --K 4 8 16 32 64 128 --output /tmp/k_sweep.png
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
    all_combinations_subspace,
    build_subspace,
    detect_n_slots,
)
from ..plot_helpers import plot_per_slot_panels
from assistant_axis import png_metadata
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help=f"Cached vectors directory (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--kind", choices=["r", "t", "combined"], default="r",
                   help="Combination kind (default: r)")
    p.add_argument("--layer", type=int, default=25,
                   help="Transformer layer (default: 25 -- Qwen-3-32B "
                        "optimum from rho_by_layer.py)")
    p.add_argument("--K", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8],
                   help="K values for soft-K whitening (default: 16 32 64 128). "
                        "A 'raw' curve is always plotted as a baseline.")
    p.add_argument("--slots", nargs="+", type=int, default=None,
                   help="Slots to plot (default: all from default.pt)")
    p.add_argument("--n_angles", type=int, default=29,
                   help="Number of canonical angles to plot per panel (default: 29)")
    p.add_argument("--include_default", dest="include_default",
                   action="store_true",
                   help="Append default.pt to each goal/nogoal subspace "
                        "(default: omitted; including it yields a trivial "
                        "0-deg first canonical angle).")
    p.set_defaults(include_default=False)
    p.add_argument("--aggregation", default="combo_residual_theat_shifted",
                   choices=["standalone", "combo_residual",
                            "combo_residual_theat_shifted"],
                   help="How to construct the goal/nogoal subspaces "
                        "(default: combo_residual_theat_shifted, partner-"
                        "debiased residuals translated to the additive origin "
                        "via the theatricality shift; see "
                        "canonical_angles/README.md). Pass 'combo_residual' to "
                        "disable the shift (recommended for slot-0 analyses). "
                        "The legacy 'combo_centroid' aggregation has been removed "
                        "from the CLI; see data.py if you need the deprecated "
                        "centroid path.")
    p.add_argument("--origin", default=None,
                   choices=["none", "default", "self_mean", "pool_mean",
                            "pool_mean_per_side", "full_combo_mean"],
                   help="Centering origin.  Default depends on --aggregation: "
                        "combo_residual -> 'none' (residuals already have a "
                        "semantic zero); standalone -> 'pool_mean_per_side' "
                        "(removes axis-specific assistant-context baseline). "
                        "Pass explicitly to override.")
    p.add_argument("--output", required=True, help="Output PNG path")
    args = p.parse_args()
    if args.origin is None:
        args.origin = ("none" if args.aggregation in ("combo_residual",
                                                      "combo_residual_theat_shifted")
                       else "pool_mean_per_side")
    if args.aggregation == "standalone" and args.origin == "none":
        raise SystemExit(
            "ERROR: --aggregation standalone --origin none leaves the "
            "assistant-context baseline in both subspaces and gives a "
            "near-zero first canonical angle that does NOT reflect any "
            "meaningful alignment.  Pick --origin default | self_mean | "
            "pool_mean_per_side instead."
        )
    return args


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.slots is None:
        n_slots = detect_n_slots(data_dir)
        slot_indices_list = list(range(n_slots))
    else:
        slot_indices_list = args.slots

    # Goal vs non-goal subspaces from the chosen aggregation (default:
    # combo_residual -- the partner-debiased version, on disk under
    # r_goal/ etc.).
    sub_a = tuple(build_subspace(data_dir, args.kind, "goal",
                                 args.aggregation, args.include_default))
    sub_b = tuple(build_subspace(data_dir, args.kind, "nogoal",
                                 args.aggregation, args.include_default))

    # Whitening pool: all combinations of this kind + default.
    pool = tuple(all_combinations_subspace(data_dir, args.kind, include_default=True))
    print(f"Subspace A (goal): {len(sub_a)} entries")
    print(f"Subspace B (nogoal): {len(sub_b)} entries")
    print(f"Whitening pool (all combinations): {len(pool)} entries")

    # Whitening regimes: raw + each requested K.
    whitenings: list[tuple[str, WhiteningSpec]] = [
        ("raw", WhiteningSpec(method="raw")),
    ]
    for K in args.K:
        whitenings.append((
            f"combos K={K}",
            WhiteningSpec(method="soft_K", K=K, pool=pool),
        ))

    origin = OriginSpec(kind=args.origin, combos_kind=args.kind)
    aggregation = AggSpec(mode=args.aggregation,
                          include_default=args.include_default)

    specs: list[CASpec] = []
    metadata: dict[CASpec, dict] = {}
    for slot in slot_indices_list:
        for label, whitening in whitenings:
            spec = CASpec(
                subspace_a=sub_a,
                subspace_b=sub_b,
                slot_indices=(slot,),
                slot_mode="single",
                layer=args.layer,
                aggregation=aggregation,
                origin=origin,
                whitening=whitening,
            )
            specs.append(spec)
            metadata[spec] = dict(slot=slot, whitening=label)

    print(f"\nComputing {len(specs)} canonical-angle specs ...")
    env = ComputeEnv(data_dir=data_dir)
    results = compute_ca_grid(specs, data_dir=data_dir, env=env)

    # Build records, truncating to first n_angles for plot readability.
    # (Matches historical x-axis cap of ~29.)
    records = []
    for spec in specs:
        meta = metadata[spec]
        ang = results[spec][:args.n_angles]
        records.append({
            "panel": meta["slot"],
            "slot": meta["slot"],
            "whitening": meta["whitening"],
            "angles": ang,
        })

    title = (f"Canonical Angles: raw vs all-combinations soft-whitening at "
             f"various K ({args.kind}_, Layer {args.layer})")
    fig = plot_per_slot_panels(
        records,
        panel_key="panel",
        group_key="whitening",
        as_bars=False,
        layout=_layout_for_slots(len(slot_indices_list)),
        title=title,
        panel_title_fn=lambda s: _slot_pretty(s),
        ylim=(0, 95),
        legend_kwargs={"fontsize": 8, "loc": "lower right"},
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = _build_inputs(data_dir, args)
    fig.savefig(out_path, dpi=140, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    print(f"Wrote {out_path}")

    # Brief summary (NaN-aware, since rank-deficient origins NaN-pad the start)
    import numpy as np
    for spec in specs:
        meta = metadata[spec]
        ang = results[spec][:args.n_angles]
        n_nan = int(np.sum(np.isnan(ang)))
        tag = f" ({n_nan} NaN-padded)" if n_nan else ""
        print(f"  slot {meta['slot']}, {meta['whitening']:18s}{tag}  "
              f"first={float(np.nanmin(ang)):5.1f}  "
              f"med={_med(ang):5.1f}  "
              f"last={float(np.nanmax(ang)):5.1f}")
    return 0


def _build_inputs(data_dir: Path, args: argparse.Namespace) -> list[InputSpec]:
    """Provenance inputs for one run.

    Always declares ``combinations/vectors`` (the whitening pool).
    Adds the goal/nogoal marginal subtrees when the aggregation is one
    of the residual modes; adds the theatricality-axis subtree under
    ``combo_residual_theat_shifted``; adds standalone subtrees for the
    legacy ``standalone`` aggregation.
    """
    extras = {"layer": str(args.layer),
              "kind": args.kind,
              "aggregation": args.aggregation,
              "origin": args.origin,
              "include_default": "1" if args.include_default else "0"}
    inputs: list[InputSpec] = [
        current_data_subtree_input(
            data_dir, "combinations/vectors",
            dep_key="combinations_vectors", extras=extras),
    ]
    needs_r = args.kind in ("r", "combined")
    needs_t = args.kind in ("t", "combined")
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
    if args.aggregation == "standalone":
        inputs.append(current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras=extras))
        inputs.append(current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras=extras))
    return inputs


def _layout_for_slots(n_slots: int) -> tuple[int, int]:
    if n_slots == 4:
        return (2, 2)
    if n_slots == 8:
        return (2, 4)
    import math
    cols = int(math.ceil(math.sqrt(n_slots)))
    rows = int(math.ceil(n_slots / cols))
    return (rows, cols)


def _slot_pretty(slot: int) -> str:
    pretty = {
        0: "Slot 0 (Body mean)", 1: "Slot 1 (<|im_start|>)",
        2: "Slot 2 (assistant)", 3: "Slot 3 (\\n)",
        4: "Slot 4 (<think>)", 5: "Slot 5 (\\n\\n inside think)",
        6: "Slot 6 (</think>)", 7: "Slot 7 (\\n\\n final)",
    }
    return pretty.get(slot, f"Slot {slot}")


def _med(arr) -> float:
    import numpy as np
    return float(np.nanmedian(arr))


if __name__ == "__main__":
    raise SystemExit(main())
