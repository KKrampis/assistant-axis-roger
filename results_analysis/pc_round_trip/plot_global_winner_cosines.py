#!/usr/bin/env python3
"""Cosine of canonical Nth PC vs global nth_pc-search winner direction.

Both directions are mapped to "raw" space using the vector view (treat
direction as data-point analogue, apply inverse transforms in the
appropriate order) so the round-trip is consistent and any departure
from cos = 1 reflects a genuine geometric difference between the two
directions, not a basis/convention mismatch.

The canonical direction for PC N is Vt[N-1] of the L=2-sheared entity
matrix at (slot=3, layer=25).  The global winner is the (slot, layer,
L, K) reported by ``klm_sweep`` stage 2 (M=∞) for that (PC, style).

Reads:
- ``roger/pc_round_trip_klm_results.json`` (winners)

Writes:
- ``roger/pc_round_trip_global_winner_cosines.png``

Only valid sanity-check use: across different (slot, layer) we treat
R^D as a common coordinate system.  This is mathematically defensible
but interpretively dicey -- different model layers learn different
representations, so a low cosine across layers can mean either "same
concept, different coordinate system" or "different concept entirely".
We can't disambiguate purely geometrically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assistant_axis.plot_metadata import png_metadata, suptitle_with_specs
from assistant_axis.provenance import (
    CACHE_POLICIES,
    InputSpec,
    current_data_subtree_input,
    load_and_register,
)
from results_analysis.canonical_angles.data import DEFAULT_DATA_DIR
from results_analysis.canonical_angles.whitening import (
    fit_shear, fit_whitening, WhiteningBasis,
)
from results_analysis.pc_round_trip.klm_sweep import (
    setup_at, compute_pc_directions, DEFAULT_PCS,
)
from results_analysis.pc_round_trip.plot_direction_cosines import (
    woodbury_inverse_shear_apply, woodbury_inverse_whiten_apply,
)

# May 2026 migration: 8-slot dataset, slot 7, layer 25, raw (L=0).
CANON_SLOT, CANON_LAYER, CANON_L = 7, 25, 0
DEFAULT_OUTPUT = "roger/pc_round_trip_global_winner_cosines.png"


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two 1-D arrays.  Returns 0 if either is zero-norm."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--actual",
                   default="roger/pc_round_trip_klm_results.json")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--pcs", type=int, nargs="*", default=DEFAULT_PCS)
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to react to drift in the klm_results JSON's "
                        "recorded provenance: strict / warn (default) / "
                        "rebuild / off.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    pcs = list(args.pcs)

    # ---- Canonical directions, mapped to raw (vector view) ----
    print(f"Computing canonical directions at "
          f"(slot={CANON_SLOT}, layer={CANON_LAYER}, L={CANON_L}, K=0)...")
    names_canon, M_raw_canon, A_g_canon, A_n_canon, _ = setup_at(
        data_dir, CANON_SLOT, CANON_LAYER)
    sh_canon = fit_shear(A_g_canon, A_n_canon, L=CANON_L)
    M_canon = sh_canon.apply(M_raw_canon)
    Vt_canonical = compute_pc_directions(M_canon, max(pcs))
    inv_sh_canon = woodbury_inverse_shear_apply(sh_canon)

    d_canon_in_raw_per_pc: dict[int, np.ndarray] = {}
    for pc in pcs:
        if pc - 1 >= Vt_canonical.shape[0]:
            continue
        d = Vt_canonical[pc - 1]
        d_canon_in_raw_per_pc[pc] = inv_sh_canon(d[None, :])[0]

    # ---- Load global nth_pc winners (validated against current state) ----
    # ``inputs`` is built up via load_and_register so the same call
    # site handles read, envelope-unwrap, drift-check, and registering
    # the file as a dependency of this plot.  Subtree deps below are
    # appended explicitly (no read-and-register helper for subtrees
    # yet -- only the JSON cache file goes through load_and_register).
    inputs: list[InputSpec] = []
    nth_pc_data, _spec, _check = load_and_register(
        Path(args.actual), dep_key="klm_results_json",
        inputs=inputs, policy=args.cache_policy,
    )
    winners = nth_pc_data["stage2_K_refinement_M_inf"]

    # ---- For each (PC, style), recompute Vt at the winning config and
    #      inverse-map d_opt to raw via vector view (inv_wh, then inv_sh) ----
    setup_cache: dict = {}
    shear_cache: dict = {}
    whiten_cache: dict = {}
    cos_per_pc: dict[int, list[float]] = {pc: [] for pc in pcs}
    winners_per_pc: dict[int, list[tuple]] = {pc: [] for pc in pcs}
    rho_per_pc: dict[int, list[float]] = {pc: [] for pc in pcs}

    for k, v in sorted(winners.items()):
        pc = int(k[2:5])
        if pc not in d_canon_in_raw_per_pc:
            continue
        slot, layer, L, K = v["slot"], v["layer"], v["L"], v["K"]
        rho = v["rho"]

        if (slot, layer) not in setup_cache:
            setup_cache[(slot, layer)] = setup_at(data_dir, slot, layer)
        names, M_raw, A_g, A_n, pool = setup_cache[(slot, layer)]

        if (slot, layer, L) not in shear_cache:
            sh = (WhiteningBasis(method="raw", L=0) if L == 0
                  else fit_shear(A_g, A_n, L=L))
            shear_cache[(slot, layer, L)] = sh
        sh = shear_cache[(slot, layer, L)]

        if (slot, layer, L, K) not in whiten_cache:
            wh = (WhiteningBasis(method="raw", K=0) if K == 0
                  else fit_whitening("soft_K", sh.apply(pool), K=K))
            whiten_cache[(slot, layer, L, K)] = wh
        wh = whiten_cache[(slot, layer, L, K)]

        M_done = wh.apply(sh.apply(M_raw))
        Vt = compute_pc_directions(M_done, pc)
        if Vt is None or pc - 1 >= Vt.shape[0]:
            continue
        d_opt = Vt[pc - 1]

        # Vector-view inverse-map to raw: undo K-whitening, then undo L-shear.
        x = d_opt[None, :]
        if K > 0:
            x = woodbury_inverse_whiten_apply(wh)(x)
        if L > 0:
            x = woodbury_inverse_shear_apply(sh)(x)
        d_opt_in_raw = x[0]

        c = cos(d_canon_in_raw_per_pc[pc], d_opt_in_raw)
        cos_per_pc[pc].append(c)
        rho_per_pc[pc].append(rho)
        winners_per_pc[pc].append((slot, layer, L, K))

    # ---- Plot ----
    pcs_with_data = [pc for pc in pcs if cos_per_pc[pc]]
    cos_means = [float(np.mean(cos_per_pc[pc])) for pc in pcs_with_data]
    cos_min = [float(np.min(cos_per_pc[pc])) for pc in pcs_with_data]
    cos_max = [float(np.max(cos_per_pc[pc])) for pc in pcs_with_data]
    rho_means = [float(np.mean(rho_per_pc[pc])) for pc in pcs_with_data]

    # Color the markers by which (slot, layer) won (mode across styles)
    from collections import Counter
    sl_per_pc = []
    # Cell-colour map: canonical first, then the migration cells.
    sl_color = {
        (7, 25): "#4c72b0", (7, 49): "#55a868",
        (6, 25): "#c44e52", (6, 49): "#dd8452",
        (0, 26): "#8172b2", (0, 49): "#937860", (3, 25): "#888888",
    }
    for pc in pcs_with_data:
        c = Counter((s, l) for s, l, _, _ in winners_per_pc[pc])
        sl_per_pc.append(c.most_common(1)[0][0])
    point_colors = [sl_color.get(sl, "gray") for sl in sl_per_pc]

    fig, ax = plt.subplots(figsize=(10, 5.6))

    # Draw line in neutral color
    ax.plot(pcs_with_data, cos_means, linewidth=1.4, color="#333333",
            zorder=3)
    # Markers coloured by winning (slot, layer)
    for pc, val, color, sl in zip(pcs_with_data, cos_means,
                                    point_colors, sl_per_pc):
        ax.scatter([pc], [val], s=110, c=color, marker="o",
                   edgecolors="black", linewidths=0.7, zorder=5)
    ax.fill_between(pcs_with_data, cos_min, cos_max, color="#888888",
                    alpha=0.15, linewidth=0)

    for pc, val in zip(pcs_with_data, cos_means):
        offset = 0.04 if val >= 0 else -0.04
        ax.text(pc, val + offset, f"{val:+.2f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8.5, color="#333333")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="gray", linewidth=0.4, linestyle=":")
    ax.axhline(-1, color="gray", linewidth=0.4, linestyle=":")

    ax.set_xscale("log")
    ax.set_xticks(pcs_with_data)
    ax.set_xticklabels([str(pc) for pc in pcs_with_data])
    ax.minorticks_off()
    ax.set_xlabel("PC index (log scale)")
    ax.set_ylabel("Cosine in raw space\n"
                  "(canonical Nth PC vs global nth_pc winner Nth PC)")
    ax.set_ylim(min(min(cos_min), -0.15) - 0.1, 1.1)
    ax.grid(True, which="major", linestyle="-", alpha=0.25)

    # Legend by (slot, layer)
    handles = []
    for sl, color in sl_color.items():
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor=color, edgecolor="black",
                              label=f"winner at slot={sl[0]}, layer={sl[1]}"))
    ax.legend(handles=handles, loc="lower left", framealpha=0.92, fontsize=9)

    title = ("Direction cosine: canonical Nth PC vs global nth_pc-search "
             "winner Nth PC")
    spec = ("Both directions mapped to raw space via consistent vector-"
            "view (inverse-apply post→raw).  Round-trip is identity by "
            "construction; any deviation from cos=1 is a real "
            "geometric difference.\n"
            "Marker colour = winning (slot, layer); R^D treated as "
            "common coord system across layers (interpretively dicey, "
            "see source).")
    _, top_rect = suptitle_with_specs(fig, title, spec, line_height=0.030)
    fig.tight_layout(rect=(0, 0, 1, top_rect))

    # --- Provenance inputs ---
    # The ``--actual`` JSON ('klm_sweep' winners) is captured as a file
    # dep; its envelope (when written by an updated klm_sweep) carries
    # the upstream judge-cache provenance transitively.  We also pin
    # the four canonical-angles derived marginal subtrees + raw vector
    # subtrees, since this script's setup_at calls build_goal_nogoal_subspaces.
    inputs.extend([
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors"),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors"),
        current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/r_goal",
            dep_key="combos_r_goal"),
        current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/r_nogoal",
            dep_key="combos_r_nogoal"),
        current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/t_goal",
            dep_key="combos_t_goal"),
        current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/t_nogoal",
            dep_key="combos_t_nogoal"),
    ])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title,
                                       source_text=Path(__file__).read_text(),
                                       inputs=inputs))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
