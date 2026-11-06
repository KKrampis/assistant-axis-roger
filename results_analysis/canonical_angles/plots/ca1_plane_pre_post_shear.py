"""CA-plane pre-shear vs post-shear visualization.

Two-panel figure showing one canonical-angle plane (CA1, CA2, ...) between
the goal and no-goal residual subspaces of one of three configurations:

* ``--kind r``        : r_combinations  (30 goal-roles  vs 30 nogoal-roles  marginals)
* ``--kind t``        : t_combinations  (30 goal-traits vs 30 nogoal-traits marginals)
* ``--kind combined`` : (r+t)_combinations  (60 vs 60 pooled)

The ``--ca_index`` flag selects the canonical-angle pair (default 1).  As
the index increases, theta -> 90 deg (the pair becomes more orthogonal),
the pre-shear panel becomes less elongated, and the shear's eigenvalues
``lambda_+ = sqrt(tan(theta/2))``, ``lambda_- = sqrt(cot(theta/2))`` both
approach 1 (the shear becomes near-identity).

All 579 standalone roles + traits are projected into the plane,
default-centered then pool-median-shifted (so origin = pool median),
coloured by goal-list membership (red = goal, blue = nogoal,
grey = neither), with the four corpus centroids overlaid as stars.

Examples::

    # Combined pooled CA1 plane:
    uv run python -m results_analysis.canonical_angles.plots.ca1_plane_pre_post_shear \\
        --kind combined --ca_index 1 --slot 3 --layer 25 \\
        --output roger/ca1_plane_combined_s3_L25.png

    # CA2 through CA5:
    for k in 2 3 4 5; do
        uv run python -m results_analysis.canonical_angles.plots.ca1_plane_pre_post_shear \\
            --kind combined --ca_index $k --slot 3 --layer 25 \\
            --output roger/ca${k}_plane_combined_s3_L25.png
    done
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from assistant_axis import png_metadata
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
    load_and_register,
)

from ..ca1_plane import (
    CAPlaneEntity,
    compute_ca_decomposition,
    plot_ca_plane_pre_post_shear,
)
from ..data import (
    DEFAULT_DATA_DIR,
    _derived_marginal_dir,
    list_etype_names,
    load_theatricality_shift,
    load_vector,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help=f"Cached vectors directory (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--kind", choices=["r", "t", "combined"], required=True,
                   help="Which goal/no-goal subspace pair to use")
    p.add_argument("--ca_index", type=int, default=1,
                   help="Which canonical-angle pair to plot (default: 1).")
    p.add_argument("--slot", type=int, default=6,
                   help="Token-slot index (default: 6 = </think>; new judge-ρ "
                        "winner from May 2026 rejudge).  Pass --slot 3 (\\n) "
                        "or 7 (\\n\\n post) to compare.  Slots 4-7 require "
                        "the 8-slot Roger dataset.")
    p.add_argument("--layer", type=int, default=25,
                   help="Transformer layer (default: 25)")
    p.add_argument("--theat_shift", action="store_true", default=True,
                   help="Apply theatricality shift to residuals before CCA "
                        "(default: enabled).  Use --no-theat_shift to disable.")
    p.add_argument("--no-theat_shift", dest="theat_shift", action="store_false")
    p.add_argument("--goal_list",
                   default="data/goal_roles_and_traits.json",
                   help="Path to JSON of goal/non-goal role+trait lists "
                        "(used for status colouring + corpus centroids).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output PNG path")
    p.add_argument("--pre_legend_pad_factor", type=float, default=3.5,
                   help="Pre-shear panel: how much extra horizontal margin "
                        "to leave on the legend side (default: 3.5x data extent).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_residual_block(data_dir: Path, folder: str, slot: int, layer: int,
                         shift: np.ndarray | None) -> np.ndarray:
    """Load all combo-marginal residuals from a folder, shifted if requested.

    ``folder`` is one of the four marginal etypes (``r_goal``, ``r_nogoal``,
    ``t_goal``, ``t_nogoal``); the on-disk path is resolved through the
    centralized ``_derived_marginal_dir`` so the new
    ``combinations/vectors/derived/marginals/`` layout is preferred when
    present.
    """
    cv = _derived_marginal_dir(data_dir, folder)
    vecs = []
    for fp in sorted(cv.glob("*.pt")):
        if fp.stem == "default":
            continue
        obj = torch.load(fp, map_location="cpu", weights_only=False)
        vec = (obj["vector"] if isinstance(obj, dict) else obj).float()[slot, layer].numpy()
        if shift is not None:
            vec = vec + shift
        vecs.append(vec)
    return np.stack(vecs, axis=0)


def _load_subspaces(data_dir: Path, kind: str, slot: int, layer: int,
                    theat_shift: bool):
    shift = (load_theatricality_shift(data_dir)[slot, layer]
             if theat_shift else None)
    if kind == "r":
        A = _load_residual_block(data_dir, "r_goal", slot, layer, shift)
        B = _load_residual_block(data_dir, "r_nogoal", slot, layer, shift)
    elif kind == "t":
        A = _load_residual_block(data_dir, "t_goal", slot, layer, shift)
        B = _load_residual_block(data_dir, "t_nogoal", slot, layer, shift)
    elif kind == "combined":
        A = np.concatenate([
            _load_residual_block(data_dir, "r_goal", slot, layer, shift),
            _load_residual_block(data_dir, "t_goal", slot, layer, shift),
        ], axis=0)
        B = np.concatenate([
            _load_residual_block(data_dir, "r_nogoal", slot, layer, shift),
            _load_residual_block(data_dir, "t_nogoal", slot, layer, shift),
        ], axis=0)
    else:
        raise SystemExit(f"Unknown --kind {kind!r}")
    return A, B


def _load_standalones(data_dir: Path, slot: int, layer: int,
                      goal_list_path: Path,
                      *,
                      inputs: list[InputSpec] | None = None):
    """Load all 579 default-centered standalones with goal-status tags.

    Threads ``inputs`` through ``load_and_register`` so the goal-list
    cache read and the InputSpec record happen together.
    """
    default_v = load_vector(data_dir, "traits", "default")[slot, layer]
    goal_master, _spec, _check = load_and_register(
        goal_list_path,
        dep_key="goal_list_json",
        inputs=inputs, policy="warn",
    )
    goal_set = {(et, n) for et in ("roles", "traits")
                for n in goal_master[et].get("goal", [])}
    nogoal_set = {(et, n) for et in ("roles", "traits")
                  for n in goal_master[et].get("non_goal", [])}

    entities = []
    for etype in ("roles", "traits"):
        type_letter = "role" if etype == "roles" else "trait"
        for name in list_etype_names(data_dir, etype, include_default=False):
            v = load_vector(data_dir, etype, name)[slot, layer] - default_v
            if (etype, name) in goal_set:    status = "goal"
            elif (etype, name) in nogoal_set: status = "nogoal"
            else:                              status = "neither"
            entities.append(CAPlaneEntity(
                name=name, etype=type_letter, vector=v, status=status,
            ))
    return entities, default_v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    goal_list_path = Path(args.goal_list)

    print(f"Loading {args.kind} goal/no-goal subspaces "
          f"(slot={args.slot}, layer={args.layer}, "
          f"theat_shift={args.theat_shift})")
    A, B = _load_subspaces(data_dir, args.kind, args.slot, args.layer,
                           args.theat_shift)
    print(f"  A: {A.shape}, B: {B.shape}")

    decomp = compute_ca_decomposition(A, B, ca_index=args.ca_index)
    print(f"  CA{args.ca_index}: theta = {np.degrees(decomp.theta):.2f} deg, "
          f"cos = {decomp.cos_theta:.4f}, "
          f"shear eigenvalues lambda_+ = {decomp.lambda_plus:.3f}, "
          f"lambda_- = {decomp.lambda_minus:.3f}")

    print("Loading 579 standalones + goal-status tags...")
    # Provenance accumulator -- threaded through _load_standalones so
    # the goal_list cache read and the InputSpec record happen together
    # (see AGENT_NOTES.md "Reader+registrar pattern").
    inputs: list[InputSpec] = []
    entities, _default_v = _load_standalones(
        data_dir, args.slot, args.layer, goal_list_path,
        inputs=inputs,
    )
    print(f"  loaded {len(entities)} entities; "
          f"goal={sum(1 for e in entities if e.status == 'goal')}, "
          f"nogoal={sum(1 for e in entities if e.status == 'nogoal')}, "
          f"neither={sum(1 for e in entities if e.status == 'neither')}")

    # Pool-median centring (historical convention; see canonical_angles
    # README).  default-centred standalones, then subtract their median.
    all_default_centered = np.stack([e.vector for e in entities], axis=0)
    pool_median = np.median(all_default_centered, axis=0)

    # Adjust entities so plot helper sees default-centered + pool-median-shifted
    for e in entities:
        e.vector = e.vector - pool_median   # in place is fine; CA1PlaneEntity
                                            # is a fresh dataclass per call

    # Centroids in default-centered + pool-median-shifted frame
    def _mean_filter(filter_fn):
        sel = [e for e in entities if filter_fn(e)]
        return np.mean([e.vector for e in sel], axis=0) if sel else None

    centroids = {
        "origin (pool median)": np.zeros_like(pool_median),
        "all roles":            _mean_filter(lambda e: e.etype == "role"),
        "all traits":           _mean_filter(lambda e: e.etype == "trait"),
        "goal corpus":          _mean_filter(lambda e: e.status == "goal"),
        "nogoal corpus":        _mean_filter(lambda e: e.status == "nogoal"),
    }

    title = (f"{_kind_label(args.kind)} CA{args.ca_index} plane: "
             f"pre-shear vs post-shear")
    spec_lines = (
        f"goal vs no-goal residuals "
        f"({'theatricality-shifted' if args.theat_shift else 'raw'}), "
        f"slot={args.slot}, layer={args.layer}, raw (no whitening); "
        f"$\\theta_{{{args.ca_index}}} = {np.degrees(decomp.theta):.2f}^\\circ$, "
        f"shear $\\lambda_+ = {decomp.lambda_plus:.3f}$, "
        f"$\\lambda_- = {decomp.lambda_minus:.3f}$\n"
        f"{len(entities)} standalones (default-centered, pool-median-shifted; "
        f"origin = pool median, yellow). Centroids: all roles (green), "
        f"all traits (purple), goal corpus (red), nogoal corpus (blue)."
    )

    fig = plot_ca_plane_pre_post_shear(
        decomposition=decomp,
        entities=entities,
        centroids=centroids,
        centering_vector=None,   # we already pre-shifted entity vectors
        title=title,
        spec_lines=spec_lines,
        pre_legend_pad_factor=args.pre_legend_pad_factor,
    )

    # Provenance inputs.  Per-kind, only the goal/nogoal subspaces
    # that were actually loaded contribute; ``combined`` reads all 4.
    extras = {"slot": str(args.slot), "layer": str(args.layer),
              "ca_index": str(args.ca_index),
              "theat_shift": "1" if args.theat_shift else "0"}
    # ``goal_list_json`` was already appended above by
    # ``_load_standalones`` → ``load_and_register``.
    inputs.extend([
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras=extras),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras=extras),
    ])
    needs_r = args.kind in ("r", "combined")
    needs_t = args.kind in ("t", "combined")
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
    if args.theat_shift:
        # Theatricality axis lives under combinations/vectors/derived/axis/.
        # Use the subtree (the manifest tracks the directory, not the file)
        # so the dep fingerprint covers any change to its contents.
        inputs.append(current_data_subtree_input(
            data_dir, "combinations/vectors/derived/axis",
            dep_key="theatricality_axis", extras=extras))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150,
                metadata=png_metadata(title=title, inputs=inputs))
    print(f"Wrote {args.output}")
    return 0


def _kind_label(kind: str) -> str:
    return {"r": "r_combinations",
            "t": "t_combinations",
            "combined": "Combined (pooled)"}[kind]


if __name__ == "__main__":
    raise SystemExit(main())
