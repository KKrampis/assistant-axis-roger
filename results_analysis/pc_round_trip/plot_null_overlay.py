#!/usr/bin/env python3
"""Overlay the actual PC round-trip ρ curve with the permutation-null
mean and 95% confidence band.

Reads:
- ``roger/pc_round_trip_klm_results.json`` (actual data, stage 2)
- ``roger/pc_round_trip_null_klm_results.json`` (null permutations)

Produces:
- ``roger/pc_round_trip_rho_loglin_with_null.png``

The actual data line is blue (matches the original plot).
The null distribution is shown as:
- a thick red line for the null mean
- a shaded red band for the 5%-95% percentile (across n_perms × 2 styles)
- a thin red line for the null max

The 1/√n noise-floor reference line is dropped -- it dramatically
underestimates the actual noise floor of the K/L search procedure
(the null curve typically lands at ρ ≈ 0.10-0.13, well above 1/√n
at n=550 ≈ 0.043).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assistant_axis.plot_metadata import png_metadata, suptitle_with_specs
from assistant_axis.provenance import (
    CACHE_POLICIES, InputSpec, load_and_register,
)

DEFAULT_ACTUAL = "roger/pc_round_trip_klm_results.json"
DEFAULT_NULL = "roger/pc_round_trip_null_klm_results.json"
DEFAULT_OUTPUT = "roger/pc_round_trip_rho_loglin_with_null.png"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--actual", default=DEFAULT_ACTUAL)
    p.add_argument("--null", default=DEFAULT_NULL)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--actual_source", default="stage2",
                   choices=["stage1", "stage2", "max"])
    p.add_argument("--null_source", default="stage2",
                   choices=["stage1", "stage2"],
                   help="Which null stage to use for the band.")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to handle stale or unrecognized inputs JSON envelopes.")
    args = p.parse_args()

    actual_path = Path(args.actual)
    null_path = Path(args.null)
    inputs: list[InputSpec] = []
    actual, _spec_a, _check_a = load_and_register(
        actual_path, dep_key="klm_results_json",
        inputs=inputs, policy=args.cache_policy,
    )
    null, _spec_n, _check_n = load_and_register(
        null_path, dep_key="null_klm_results_json",
        inputs=inputs, policy=args.cache_policy,
    )

    # ---- Actual data ----
    if args.actual_source == "stage1":
        a_items = actual["stage1_M_sweep"]
    elif args.actual_source == "stage2":
        a_items = actual["stage2_K_refinement_M_inf"]
    else:  # max
        s1 = actual["stage1_M_sweep"]
        s2 = actual["stage2_K_refinement_M_inf"]
        a_items = {}
        for k in set(s1) | set(s2):
            r1 = s1.get(k, {}).get("rho")
            r2 = s2.get(k, {}).get("rho")
            cand = [(r, src) for r, src in [(r1, "s1"), (r2, "s2")]
                    if r is not None]
            if not cand:
                continue
            best_r, src = max(cand, key=lambda t: t[0])
            a_items[k] = (s1[k] if src == "s1" else s2[k]) | {"rho": best_r}

    actual_per_pc = {}
    for k, v in a_items.items():
        if v.get("rho") is None:
            continue
        pc = int(k[2:5])
        actual_per_pc.setdefault(pc, []).append(v["rho"])
    actual_pcs = sorted(actual_per_pc)
    actual_means = [float(np.mean(actual_per_pc[pc])) for pc in actual_pcs]

    # ---- Null data ----
    n_items = null["stage2"] if args.null_source == "stage2" else null["stage1"]
    null_per_pc = {}
    for k, v in n_items.items():
        if v.get("rho") is None:
            continue
        pc = int(k[2:5])
        null_per_pc.setdefault(pc, []).append(v["rho"])
    null_pcs = sorted(null_per_pc)
    null_means = np.array([float(np.mean(null_per_pc[pc])) for pc in null_pcs])
    null_sds = np.array([float(np.std(null_per_pc[pc], ddof=1))
                         for pc in null_pcs])
    null_p2_lo = null_means - 2 * null_sds
    null_p2_hi = null_means + 2 * null_sds
    null_p3_lo = null_means - 3 * null_sds
    null_p3_hi = null_means + 3 * null_sds

    # Sample-size info per PC
    null_n = [len(null_per_pc[pc]) for pc in null_pcs]
    null_n_min, null_n_max = min(null_n), max(null_n)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    # Actual
    ax.plot(actual_pcs, actual_means, marker="o", markersize=8, linewidth=1.6,
            color="#4c72b0", markerfacecolor="#4c72b0",
            markeredgecolor="black", markeredgewidth=0.7,
            label=f"actual data ({args.actual_source}, glossary+inline mean)",
            zorder=5)
    for pc, val in zip(actual_pcs, actual_means):
        ax.text(pc, val + 0.025, f"{val:+.3f}",
                ha="center", va="bottom", fontsize=8.0, color="#4c72b0", zorder=6)

    # Null: mean + ±2sd band (filled) + ±3sd lines (dotted)
    ax.fill_between(null_pcs, null_p2_lo, null_p2_hi,
                    color="#c44e52", alpha=0.18, linewidth=0,
                    label="permutation null: mean ± 2 SD")
    ax.plot(null_pcs, null_p3_hi, linestyle=":", linewidth=1.0,
            color="#a02723", alpha=0.85, zorder=4,
            label="permutation null: ± 3 SD")
    ax.plot(null_pcs, null_p3_lo, linestyle=":", linewidth=1.0,
            color="#a02723", alpha=0.85, zorder=4)
    ax.plot(null_pcs, null_means, marker="s", markersize=5, linewidth=1.4,
            color="#c44e52", label="permutation null: mean", zorder=5)

    ax.axhline(0, color="black", linewidth=0.4)

    ax.set_xscale("log")
    all_pcs = sorted(set(actual_pcs) | set(null_pcs))
    ax.set_xticks(all_pcs)
    ax.set_xticklabels([str(pc) for pc in all_pcs])
    ax.minorticks_off()
    ax.set_xlabel("PC index (log scale)")
    ax.set_ylabel("Best round-trip Spearman ρ")
    y_lo = min(float(min(null_p3_lo)), min(actual_means)) - 0.05
    y_hi = max(max(actual_means), float(max(null_p3_hi))) + 0.10
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(all_pcs[0] * 0.85, all_pcs[-1] * 1.15)
    ax.grid(True, which="major", linestyle="-", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=8.5)

    n_perms = null.get("_meta", {}).get("n_perms", "?")
    n_stage2 = null.get("_meta", {}).get("n_stage2_requested", "?")
    title = "PC round-trip ρ vs PC index, with permutation-null band"
    spec = (f"actual: {args.actual_source}, glossary+inline mean.  "
            f"null: K/L sweep on {n_perms} random permutations of judge "
            f"scores per cell ({args.null_source}, M=∞ throughout)\n"
            f"null entries per PC: {null_n_min}–{null_n_max}; null measures "
            f"the search-procedure's noise floor (well above 1/√n for any "
            f"flexible search)")
    _, top_rect = suptitle_with_specs(fig, title, spec, line_height=0.030)
    fig.tight_layout(rect=(0, 0, 1, top_rect))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ``inputs`` was populated above by load_and_register.
    fig.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
