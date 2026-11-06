"""Layer-sweep plot: how the goal/non-goal canonical-angle spectrum
varies across transformer layers, with two whitening regimes side by side.

Defaults: a 17-layer set spanning ``17, 21, 25, 27, 29, 31, ..., 53``
(centered on Qwen-3-32B's analysis layer 25 with 2-layer spacing on
the high side), slots 0 and 3 (2 rows), raw vs ``K=3 soft`` whitening
(2 columns).  Subspaces are ``combo_residual_theat_shifted`` marginals
(origin=none); whitening pool is ``roles+traits`` standalones (60
corresponding standalones held out) plus the corpus ``default.pt`` as
a single anchor row.

What this plot answers: "how does the goal/non-goal canonical-angle
spectrum change with depth, and is our analysis layer (25 for
Qwen-3-32B) a sensible single-layer choice?"

Cached results are pickled to ``--cache_dir`` (default ``/tmp``) keyed
on ``(kinds, slots, layers, K, scope, heldout)``; first run takes a few
seconds to compute, subsequent runs are instant.

Examples
--------

    # Default: 6 layers x 2 slots x 2 whitenings = 24 curves on a 2x2 grid
    uv run python -m results_analysis.canonical_angles.plots.layer_sweep \\
        --output roger/canonical_angles_layer_sweep.png

    # Different layer set, only slot 3
    uv run python -m results_analysis.canonical_angles.plots.layer_sweep \\
        --layers 12 24 36 48 60 --slots 3 \\
        --output /tmp/layer_sweep_slot3_only.png
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np

from .. import (
    AggSpec,
    CASpec,
    ComputeEnv,
    OriginSpec,
    WhiteningSpec,
    compute_ca_grid,
)
from ..whitening import DEFAULT_SOFT_K
from ..data import (
    AUGMENTED_POOL_VERSION,
    DEFAULT_DATA_DIR,
    build_augmented_whitening_pool,
    build_subspace,
    build_whitening_pool,
)
from assistant_axis import png_metadata, suptitle_with_specs
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
    p.add_argument("--kinds", nargs="+", default=["r", "t"],
                   choices=["r", "t"],
                   help="Combination kinds to compute and average (default: r t)")
    p.add_argument("--layers", nargs="+", type=int,
                   default=[17, 21, 25, 27, 29, 31, 33, 35, 37, 39,
                           41, 43, 45, 47, 49, 51, 53],
                   help="Transformer layers to sweep (default: a 17-layer "
                        "set centered on Qwen-3-32B layer 25 with 2-layer "
                        "spacing 21-53)")
    p.add_argument("--slots", nargs="+", type=int, default=[0, 6],
                   help="Slot indices (one panel row per slot; default: 0 6 -- "
                        "body-mean + </think>, the new judge-ρ winner).  Pass "
                        "--slots 0 3 to compare against historical \\n default, "
                        "or 0 7 for \\n\\n (post).  Slots 4-7 require 8-slot data.")
    p.add_argument("--K", type=int, default=DEFAULT_SOFT_K,
                   help=f"K for the soft-K whitening column (default: "
                        f"{DEFAULT_SOFT_K} -- the project soft-K default; "
                        f"see whitening.DEFAULT_SOFT_K).")
    p.add_argument("--scope", default="roles+traits",
                   choices=["roles", "traits", "roles+traits"],
                   help="Whitening pool scope (default: roles+traits)")
    p.add_argument("--no-heldout", dest="heldout", action="store_false",
                   help="Disable holding out subspace entities from the "
                        "whitening pool (default: held out).")
    p.set_defaults(heldout=True)
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Disable pool augmentation (default: augmented -- "
                        "pool gains the corpus default.pt; see "
                        "canonical_angles/README.md).")
    p.set_defaults(augment=True)
    p.add_argument("--cache_dir", default="/tmp",
                   help="Where to pickle CA results (default: /tmp)")
    p.add_argument("--aggregation", default="combo_residual_theat_shifted",
                   choices=["combo_residual", "combo_residual_theat_shifted"],
                   help="Subspace aggregation (default: "
                        "combo_residual_theat_shifted; see "
                        "canonical_angles/README.md). Pass 'combo_residual' "
                        "to disable the shift.")
    p.add_argument("--output", required=True, help="Output PNG path")
    return p.parse_args()


def _cache_key(prefix: str, *parts) -> str:
    h = hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}.pkl"


def compute_results(args, env: ComputeEnv, cache_path: Path) -> dict:
    """{(kind, slot, layer, label) -> angles}.  Loads from cache if present."""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data_dir = Path(args.data_dir)
    results: dict = {}
    for kind in args.kinds:
        sub_a = tuple(build_subspace(data_dir, kind, "goal",
                                     args.aggregation, include_default=False))
        sub_b = tuple(build_subspace(data_dir, kind, "nogoal",
                                     args.aggregation, include_default=False))
        leave_out = (set(sub_a) | set(sub_b)) if args.heldout else set()
        if args.augment:
            pool = tuple(build_augmented_whitening_pool(
                data_dir, leave_out=leave_out, scope=args.scope))
        else:
            pool = tuple(build_whitening_pool(
                data_dir, scope=args.scope, leave_out=leave_out))
        for slot in args.slots:
            for layer in args.layers:
                whitenings = [
                    ("raw", WhiteningSpec(method="raw")),
                    (f"K={args.K} soft",
                     WhiteningSpec(method="soft_K", K=args.K, pool=pool)),
                ]
                for label, w_spec in whitenings:
                    spec = CASpec(
                        subspace_a=sub_a, subspace_b=sub_b,
                        slot_indices=(slot,), slot_mode="single", layer=layer,
                        aggregation=AggSpec(mode=args.aggregation,
                                            include_default=False),
                        origin=OriginSpec(kind="none"),
                        whitening=w_spec,
                    )
                    out = compute_ca_grid([spec], data_dir=data_dir, env=env)
                    results[(kind, slot, layer, label)] = out[spec]
            print(f"  kind={kind} slot={slot}: {len(args.layers)} layers done")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Cached results to {cache_path}")
    return results


def make_plot(args, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def avg(slot: int, layer: int, label: str) -> np.ndarray:
        return np.mean([results[(k, slot, layer, label)]
                        for k in args.kinds], axis=0)

    labels_order = ["raw", f"K={args.K} soft"]
    rainbow = plt.cm.rainbow(np.linspace(0.0, 1.0, len(args.layers)))

    n_rows = len(args.slots)
    n_cols = len(labels_order)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for r, slot in enumerate(args.slots):
        for c, label in enumerate(labels_order):
            ax = axes[r, c]
            # Draw later layers first so EARLIER layers (purple) end up on top.
            for color, layer in reversed(list(zip(rainbow, args.layers))):
                ang = avg(slot, layer, label)
                x = np.arange(1, len(ang) + 1)
                ax.plot(x, ang, color=color, marker="o", markersize=4,
                        linewidth=1.5, alpha=0.95, label=f"L{layer}")
            ax.axhline(y=90, color="gray", linestyle=":", alpha=0.4)
            ax.set_xlabel("Canonical Angle Index (sorted)", fontsize=11)
            ax.set_ylabel("Canonical Angle (degrees)", fontsize=11)
            ax.set_title(f"{_slot_pretty(slot)}  -  {label}", fontsize=12)
            ax.set_ylim(0, 95)
            ax.set_xlim(0, 31)
            ax.grid(True, alpha=0.3)
            # Reorder legend to read ascending L<low> -> L<high>
            handles, lbls = ax.get_legend_handles_labels()
            order = [lbls.index(f"L{L}") for L in args.layers]
            ax.legend([handles[i] for i in order], [lbls[i] for i in order],
                      fontsize=10, loc="lower right", ncol=2, title="layer")

    kinds_str = " and ".join(args.kinds)
    pool_descr = (f"{args.scope}"
                  + (" (60 held out)" if args.heldout else " (no leave-out)")
                  + (" + default" if args.augment else ""))
    title_line = "Goal vs Non-goal canonical angles by transformer layer"
    spec_lines = [
        f"subspaces = {args.aggregation} (30 vs 30, {kinds_str} averaged), "
        f"origin=none",
        f"whitening pool = standalone {pool_descr}",
    ]
    _, top_rect = suptitle_with_specs(fig, title_line, spec_lines)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = _build_inputs(Path(args.data_dir), args)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    print(f"Wrote {out_path}")


def _build_inputs(data_dir: Path, args: argparse.Namespace) -> list[InputSpec]:
    """Provenance inputs for one run.

    Declares the goal/nogoal marginal subtrees actually used (per
    ``--kinds``); the theatricality axis when shifted; and the
    roles/traits subtrees feeding the whitening pool (always declared
    since the K-soft column always uses pool whitening).
    """
    extras = {"layers": ",".join(str(L) for L in args.layers),
              "slots": ",".join(str(s) for s in args.slots),
              "kinds": ",".join(args.kinds),
              "K": str(args.K),
              "aggregation": args.aggregation,
              "scope": args.scope,
              "heldout": "1" if args.heldout else "0",
              "augment": "1" if args.augment else "0"}
    inputs: list[InputSpec] = []
    needs_r = "r" in args.kinds
    needs_t = "t" in args.kinds
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
    # Whitening pool: always feeds the K-soft column.
    if "roles" in args.scope:
        inputs.append(current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras=extras))
    if "traits" in args.scope:
        inputs.append(current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras=extras))
    return inputs


def _slot_pretty(slot: int) -> str:
    pretty = {
        0: "Slot 0 (Body mean)",
        1: "Slot 1 (<|im_start|>)",
        2: "Slot 2 (assistant)",
        3: "Slot 3 (\\n)",
        4: "Slot 4 (<think>)",
        5: "Slot 5 (\\n\\n inside think)",
        6: "Slot 6 (</think>)",
        7: "Slot 7 (\\n\\n final)",
    }
    return pretty.get(slot, f"Slot {slot}")


def main() -> int:
    args = parse_args()
    cache_path = Path(args.cache_dir) / _cache_key(
        "ca_layer_sweep",
        args.data_dir, tuple(sorted(args.kinds)), tuple(sorted(args.slots)),
        tuple(sorted(args.layers)), args.K, args.scope, args.heldout,
        args.augment, args.aggregation,
        AUGMENTED_POOL_VERSION if args.augment else "no-aug",
    )
    print(f"Cache: {cache_path}\n")

    env = ComputeEnv(data_dir=Path(args.data_dir))
    results = compute_results(args, env, cache_path)
    make_plot(args, results)

    # Summary: first canonical angle per layer at slot 3 raw (most informative).
    if 3 in args.slots:
        print("\nSlot 3 raw, first CA by layer:")
        for L in args.layers:
            a = np.mean([results[(k, 3, L, "raw")] for k in args.kinds], axis=0)
            print(f"  L{L:2d}: first={float(np.nanmin(a)):5.2f}  "
                  f"med={float(np.nanmedian(a)):5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
