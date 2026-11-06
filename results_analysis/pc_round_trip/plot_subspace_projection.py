#!/usr/bin/env python3
"""Project the global nth_pc winner direction onto the canonical
{PC_{N-1}, PC_N, PC_{N+1}} subspace.

For each PC N, computes the squared projection norm of the global
nth_pc winner direction (mapped into the canonical L=2 sheared space)
onto the span of the N-1, N, N+1 canonical PCs (with N-1 ≥ 1 and
N+1 ≤ max_pc).

If the value is close to 1, the winner direction lies in the
neighbour-PC subspace -- consistent with "the winner is just an
adjacent PC, off-by-one or off-by-two from the canonical".  If it's
small, the winner is genuinely outside the local canonical neighbourhood.

Output: ``roger/pc_round_trip_neighbour_subspace_projection.png``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assistant_axis.plot_metadata import png_metadata, suptitle_with_specs
from assistant_axis.provenance import (
    CACHE_POLICIES, InputSpec, current_data_subtree_input,
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
DEFAULT_OUTPUT = "roger/pc_round_trip_neighbour_subspace_projection.png"
WINDOW = 1  # half-window: project onto {N-WINDOW, ..., N+WINDOW}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--actual",
                   default="roger/pc_round_trip_klm_results.json")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--pcs", type=int, nargs="*", default=DEFAULT_PCS)
    p.add_argument("--window", type=int, default=WINDOW,
                   help="Half-window: project onto canonical PCs N-w..N+w.")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to handle stale or unrecognized inputs JSON envelopes.")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    actual_path = Path(args.actual)
    pcs = list(args.pcs)
    half = args.window
    max_pc = max(pcs) + half + 1  # need a few extra canonical PCs

    # ---- Canonical basis ----
    print(f"Computing canonical PCs at "
          f"(slot={CANON_SLOT}, layer={CANON_LAYER}, L={CANON_L}, K=0) ...")
    names_canon, M_raw_canon, A_g_canon, A_n_canon, _ = setup_at(
        data_dir, CANON_SLOT, CANON_LAYER)
    sh_canon = fit_shear(A_g_canon, A_n_canon, L=CANON_L)
    M_canon = sh_canon.apply(M_raw_canon)
    Vt_canonical = compute_pc_directions(M_canon, max_pc)
    inv_sh_canon = woodbury_inverse_shear_apply(sh_canon)
    print(f"  Vt_canonical shape: {Vt_canonical.shape}")

    # ---- Load global nth_pc winners ----
    # Read + envelope-unwrap + drift-check + register-as-input via
    # load_and_register so the inputs accumulator stays in lockstep with
    # actual reads (see AGENT_NOTES.md "Reader+registrar pattern").
    inputs: list[InputSpec] = []
    nth_pc_data, _spec, _check = load_and_register(
        actual_path, dep_key="klm_results_json",
        inputs=inputs, policy=args.cache_policy,
    )
    winners = nth_pc_data["stage2_K_refinement_M_inf"]

    # ---- Per (PC, style): compute winner direction, map to L=2 sheared
    # space (vector view), normalise, project onto neighbour subspace ----
    setup_cache: dict = {}
    sh_cache: dict = {}
    wh_cache: dict = {}

    proj_per_pc: dict[int, list[dict]] = {pc: [] for pc in pcs}

    for k, v in sorted(winners.items()):
        pc = int(k[2:5])
        if pc not in proj_per_pc:
            continue
        slot, layer = v["slot"], v["layer"]
        L_w, K_w = v["L"], v["K"]
        rho = v["rho"]

        if (slot, layer) not in setup_cache:
            setup_cache[(slot, layer)] = setup_at(data_dir, slot, layer)
        names, M_raw, A_g, A_n, pool = setup_cache[(slot, layer)]
        sh_key = (slot, layer, L_w)
        if sh_key not in sh_cache:
            sh_cache[sh_key] = (WhiteningBasis(method="raw", L=0)
                                 if L_w == 0
                                 else fit_shear(A_g, A_n, L=L_w))
        sh = sh_cache[sh_key]
        wh_key = (slot, layer, L_w, K_w)
        if wh_key not in wh_cache:
            wh_cache[wh_key] = (WhiteningBasis(method="raw", K=0)
                                  if K_w == 0
                                  else fit_whitening(
                                      "soft_K", sh.apply(pool), K=K_w))
        wh = wh_cache[wh_key]

        M_done = wh.apply(sh.apply(M_raw))
        Vt = compute_pc_directions(M_done, pc)
        if Vt is None or pc - 1 >= Vt.shape[0]:
            continue
        d_w = Vt[pc - 1]

        # Map d_w → raw via vector-view inverse, then → L=2 sheared via
        # canonical shear forward (so it lives in the same orthonormal
        # basis as the canonical Vt).
        x = d_w[None, :]
        if K_w > 0:
            x = woodbury_inverse_whiten_apply(wh)(x)
        if L_w > 0:
            x = woodbury_inverse_shear_apply(sh)(x)
        d_w_raw = x[0]
        # Forward into canonical shear space (identity at L=0).
        d_w_canon = sh_canon.apply(d_w_raw[None, :])[0]
        norm = np.linalg.norm(d_w_canon)
        if norm == 0:
            continue
        d_w_unit = d_w_canon / norm

        # Project onto canonical PCs N-half .. N+half.
        lo = max(1, pc - half)
        hi = min(Vt_canonical.shape[0], pc + half)
        cos_per_index = []
        for n in range(lo, hi + 1):
            v_n = Vt_canonical[n - 1]
            cos_per_index.append((n, float(np.dot(d_w_unit, v_n))))
        squared_norm = sum(c ** 2 for _, c in cos_per_index)

        proj_per_pc[pc].append({
            "rho": rho, "slot": slot, "layer": layer, "L": L_w, "K": K_w,
            "squared_norm": float(squared_norm),
            "norm": float(np.sqrt(squared_norm)),
            "cos_per_index": cos_per_index,
        })

    # ---- Print table ----
    print()
    print(f"{'PC':>4}  {'rho_mean':>9}  {'subspace_proj_norm':>18}  "
          f"{'best_cos_at':<24}")
    print("-" * 80)
    for pc in pcs:
        if not proj_per_pc[pc]:
            continue
        infos = proj_per_pc[pc]
        rho_mean = float(np.mean([info["rho"] for info in infos]))
        norm_mean = float(np.mean([info["norm"] for info in infos]))
        # Take the cosine table from the first style (they're typically
        # close); pick the index with max |cos|.
        cos_table = infos[0]["cos_per_index"]
        best = max(cos_table, key=lambda t: abs(t[1]))
        best_str = f"PC{best[0]} ({best[1]:+.3f})"
        # Show all cosines if there are 2-3
        all_cos_str = "  ".join(
            f"PC{n}={c:+.3f}" for n, c in cos_table
        )
        print(f"{pc:>4}  {rho_mean:>+9.4f}  {norm_mean:>18.4f}  "
              f"{best_str:<24}  {all_cos_str}")

    # ---- Plot ----
    pcs_with_data = [pc for pc in pcs if proj_per_pc[pc]]
    norm_means = [
        float(np.mean([info["norm"] for info in proj_per_pc[pc]]))
        for pc in pcs_with_data
    ]
    norm_min = [
        float(np.min([info["norm"] for info in proj_per_pc[pc]]))
        for pc in pcs_with_data
    ]
    norm_max = [
        float(np.max([info["norm"] for info in proj_per_pc[pc]]))
        for pc in pcs_with_data
    ]

    fig, ax = plt.subplots(figsize=(10, 5.6))

    ax.plot(pcs_with_data, norm_means, marker="o", markersize=8,
            linewidth=1.6, color="#4c72b0",
            markeredgecolor="black", markeredgewidth=0.7,
            zorder=5,
            label=f"projection norm onto canonical PCs "
                  f"{{N-{half}, ..., N+{half}}}")
    ax.fill_between(pcs_with_data, norm_min, norm_max,
                    color="#4c72b0", alpha=0.18, linewidth=0)
    for pc, val in zip(pcs_with_data, norm_means):
        ax.text(pc, val + 0.025, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="#4c72b0")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="gray", linewidth=0.4, linestyle=":")

    # Reference: random unit vector in R^D would have squared
    # projection norm ≈ k/D where k = neighbourhood size, D = 5120.
    D = Vt_canonical.shape[1]
    k_typical = 2 * half + 1
    rand_norm = float(np.sqrt(k_typical / D))
    ax.axhline(rand_norm, color="firebrick", linewidth=1.0, linestyle=":",
               label=f"random-direction baseline ≈ √({k_typical}/D) "
                     f"= {rand_norm:.3f} for D={D}")

    ax.set_xscale("log")
    ax.set_xticks(pcs_with_data)
    ax.set_xticklabels([str(pc) for pc in pcs_with_data])
    ax.minorticks_off()
    ax.set_xlabel("PC index N (log scale)")
    ax.set_ylabel(f"||projection of winner onto "
                  f"span(canonical PCs N-{half}..N+{half})||")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, which="major", linestyle="-", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)

    title = (f"Subspace projection: global nth_pc winner Nth PC onto "
             f"canonical {{N-{half}, ..., N+{half}}} subspace")
    spec = (f"Winner direction mapped to canonical (L={CANON_L}) space "
            "(vector view), normalised, projected onto canonical PCs "
            "N-w..N+w.\n"
            "Norm ≈ 1 means winner is in the canonical neighbourhood "
            "(possibly off-by-one); norm ≈ 0 means winner is outside "
            "the local neighbourhood; red dotted = D-dimensional null.")
    _, top_rect = suptitle_with_specs(fig, title, spec, line_height=0.030)
    fig.tight_layout(rect=(0, 0, 1, top_rect))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
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
    # ``klm_results_json`` was already appended above by load_and_register.
    fig.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
