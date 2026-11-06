#!/usr/bin/env python3
"""Bar-chart histogram of best round-trip ρ per PC.

Reads the JSON cache produced by ``klm_sweep.py`` and emits a
companion-to-the-loglin-plot histogram showing the per-cell winner
labels (best stage / L / K / M) for diagnostic inspection.

By default takes ``max(stage1, stage2)`` per cell so finite-M wins (when
they beat the M=∞ refined-K result) are visible.  Bars are coloured by
whether the cell's avg ρ clears 2× the 1/√n noise floor.

Per-PC summary is also printed to stdout (slot/layer/L/K/M for the
winning cells of each style).
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

DEFAULT_CACHE_PATH = "roger/pc_round_trip_klm_results.json"
DEFAULT_OUTPUT = "roger/pc_round_trip_rho_histogram.png"
DEFAULT_NOISE_FLOOR_N = 550


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH,
                   help="JSON cache produced by klm_sweep.py.")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                   help="Output PNG path.")
    p.add_argument("--noise_floor_n", type=int, default=DEFAULT_NOISE_FLOOR_N,
                   help="Sample size n used to draw the 1/√n noise-floor "
                        "reference line (default 550).")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to handle stale or unrecognized inputs JSON envelopes.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache = Path(args.cache_path)
    out_path = Path(args.output)
    inputs: list[InputSpec] = []
    raw, _spec, _check = load_and_register(
        cache, dep_key="klm_results_json",
        inputs=inputs, policy=args.cache_policy,
    )
    s1 = raw.get("stage1_M_sweep", {})
    s2 = raw.get("stage2_K_refinement_M_inf", {})
    noise_floor = 1.0 / np.sqrt(args.noise_floor_n)

    # Per cell: take max(stage1, stage2).  Group per PC, average across styles.
    all_keys = set(s1.keys()) | set(s2.keys())
    per_pc: dict[int, list[float]] = {}
    labels_by_pc: dict[int, list[str]] = {}
    slot_layer_by_pc: dict[int, list[tuple[int, int]]] = {}
    for k in all_keys:
        pc = int(k[2:5])
        style = k.split("_", 1)[1]
        r1 = s1.get(k, {}).get("rho")
        r2 = s2.get(k, {}).get("rho")
        cands = [(r, src, s1.get(k, {}) if src == "s1" else s2.get(k, {}))
                 for r, src in [(r1, "s1"), (r2, "s2")] if r is not None]
        if not cands:
            continue
        best_r, best_src, best_meta = max(cands, key=lambda t: t[0])
        per_pc.setdefault(pc, []).append(best_r)
        # Distinguish: slot (token position), ly (transformer layer),
        # L (shear truncation), K (whitening depth), M (PC truncation).
        L = best_meta.get("L")
        K = best_meta.get("K")
        M = best_meta.get("M", "inf")
        slot = best_meta.get("slot")
        layer = best_meta.get("layer")
        slot_layer_by_pc.setdefault(pc, []).append((slot, layer))
        m_str = "" if M in ("inf", None) else f",M={M}"
        labels_by_pc.setdefault(pc, []).append(
            f"{style[:4]}({best_src}:slot={slot},ly={layer},L={L},K={K}{m_str})"
        )

    pcs = sorted(per_pc)
    avgs = [float(np.mean(per_pc[pc])) for pc in pcs]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = np.arange(len(pcs))
    colors = ["#4c72b0" if v >= noise_floor * 2 else "#d4856b" for v in avgs]
    ax.bar(x, avgs, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(noise_floor, linestyle=":", color="firebrick", linewidth=1.4,
               label=f"noise floor (1/√n at n={args.noise_floor_n}) ≈ {noise_floor:.3f}")
    ax.axhline(0, color="black", linewidth=0.5)

    for xi, val in zip(x, avgs):
        label_y = val + 0.012 if val >= 0 else val - 0.018
        ax.text(xi, label_y, f"{val:+.3f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"PC {pc}" for pc in pcs])
    ax.set_ylabel("Best round-trip Spearman ρ")
    ax.set_ylim(min(min(avgs) - 0.05, -0.05), max(max(avgs) + 0.08, 0.85))
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    title = "PC round-trip ρ (refined): auto-described axis vs. original projection"
    spec = ("bars averaged over glossary + inline; per-cell max(stage 1, stage 2); "
            "stage-2 K refined via bracket-and-bisect, rescuing PC 40 / 48 dips")
    _, top_rect = suptitle_with_specs(fig, title, spec)
    fig.tight_layout(rect=(0, 0, 1, top_rect))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ``inputs`` was populated above by load_and_register.
    fig.savefig(
        out_path, dpi=150, bbox_inches="tight",
        metadata=png_metadata(title=title, inputs=inputs),
    )
    print(f"Wrote {out_path}")
    print()
    print("Per-PC summary:")
    for pc, avg in zip(pcs, avgs):
        lbls = labels_by_pc[pc]
        print(f"  PC {pc:>3}: avg ρ = {avg:+.4f}   |   " + " | ".join(lbls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
