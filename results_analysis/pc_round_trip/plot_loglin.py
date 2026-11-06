#!/usr/bin/env python3
"""Log-x line+marker plot of best round-trip ρ vs PC index.

Reads the JSON cache produced by ``klm_sweep.py`` and emits the headline
figure for the PC round-trip experiment: average best round-trip Spearman
ρ (across glossary + inline description styles) vs PC index, on a
log-spaced x-axis.

Bar charts on log axes are visually misleading (bar widths distort);
markers + connecting line read cleanly on a log x-axis and emphasise the
trend across the full PC range.

Defaults to stage-2 (refined K, M=∞) since occasional finite-M wins are
within the ~1/√n noise band and not worth the extra hyperparameter on
this view; pass ``--source stage1`` to use the M-sweep best instead, or
``--source max`` to take ``max(stage1, stage2)`` per cell (matches the
companion histogram).
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
DEFAULT_OUTPUT = "roger/pc_round_trip_rho_loglin.png"
DEFAULT_NOISE_FLOOR_N = 550


def _per_pc_avgs(raw: dict, source: str) -> tuple[list[int], list[float]]:
    """Return (sorted PCs, per-PC mean ρ across styles) for the chosen source."""
    s1 = raw.get("stage1_M_sweep", {})
    s2 = raw.get("stage2_K_refinement_M_inf", {})

    if source == "stage1":
        items = s1
    elif source == "stage2":
        items = s2
    elif source == "max":
        # Per-cell max(stage1, stage2)
        items = {}
        for k in set(s1) | set(s2):
            r1 = s1.get(k, {}).get("rho")
            r2 = s2.get(k, {}).get("rho")
            cand = [(r, src) for r, src in [(r1, "s1"), (r2, "s2")] if r is not None]
            if not cand:
                continue
            best_r, src = max(cand, key=lambda t: t[0])
            items[k] = (s1[k] if src == "s1" else s2[k]) | {"rho": best_r}
    else:
        raise ValueError(f"Unknown --source: {source}")

    per_pc: dict[int, list[float]] = {}
    for k, v in items.items():
        pc = int(k[2:5])
        r = v.get("rho")
        if r is not None:
            per_pc.setdefault(pc, []).append(r)
    pcs = sorted(per_pc)
    avgs = [float(np.mean(per_pc[pc])) for pc in pcs]
    return pcs, avgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH,
                   help="JSON cache produced by klm_sweep.py.")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                   help="Output PNG path.")
    p.add_argument("--source", type=str, default="stage2",
                   choices=["stage1", "stage2", "max"],
                   help="Which sweep to plot: 'stage2' (refined K at M=∞, "
                        "default), 'stage1' (best M at coarse K), or 'max' "
                        "(per-cell max of the two).")
    p.add_argument("--noise_floor_n", type=int, default=DEFAULT_NOISE_FLOOR_N,
                   help="Sample size n used to draw the 1/√n noise-floor "
                        "reference line (default 550, matches the entity "
                        "count after intersecting with judge coverage).")
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
    pcs, avgs = _per_pc_avgs(raw, args.source)
    if not pcs:
        raise SystemExit(f"No data in {cache} for source={args.source}")

    noise_floor = 1.0 / np.sqrt(args.noise_floor_n)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.plot(pcs, avgs, marker="o", markersize=8, linewidth=1.5,
            color="#4c72b0", markerfacecolor="#4c72b0",
            markeredgecolor="black", markeredgewidth=0.8)

    for pc, val in zip(pcs, avgs):
        ax.text(pc, val + 0.02, f"{val:+.3f}",
                ha="center", va="bottom", fontsize=8.5)

    ax.axhline(noise_floor, linestyle=":", color="firebrick", linewidth=1.4,
               label=f"noise floor (1/√n at n={args.noise_floor_n}) ≈ {noise_floor:.3f}")
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_xticks(pcs)
    ax.set_xticklabels([str(pc) for pc in pcs])
    ax.minorticks_off()
    ax.set_xlabel("PC index (log scale)")
    ax.set_ylabel("Best round-trip Spearman ρ")
    ax.set_ylim(min(min(avgs) - 0.05, -0.05), max(max(avgs) + 0.10, 0.90))
    ax.set_xlim(pcs[0] * 0.85, pcs[-1] * 1.15)
    ax.grid(True, which="major", linestyle="-", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    source_spec = {
        "stage1": "best across coarse (L, K, M) — stage 1 only",
        "stage2": "refined K, M = ∞ — stage 2",
        "max":    "per-cell max(stage 1, stage 2)",
    }[args.source]
    title = "PC round-trip ρ vs PC index (log-lin)"
    spec = (f"averaged over glossary + inline description styles; "
            f"{source_spec}\n"
            f"K refined via bracket-and-bisect; finite-M wins (when "
            f"present) within ~1/√n noise band")
    _, top_rect = suptitle_with_specs(fig, title, spec)
    fig.tight_layout(rect=(0, 0, 1, top_rect))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ``inputs`` was populated above by load_and_register.
    fig.savefig(
        out_path, dpi=150, bbox_inches="tight",
        metadata=png_metadata(title=title, inputs=inputs),
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
