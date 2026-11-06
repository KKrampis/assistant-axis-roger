#!/usr/bin/env python3
"""Scatter plot of fitted peak K (desc+inst vs responses) with per-axis
weights derived from fit-quality heuristics.

For each axis we have two ρ-vs-K curves (desc+inst and responses, see
:mod:`results_analysis.whitening_k_sweep`).  Each curve is fit with a
log-K parabola

    ρ(K) ≈ a + b · log₂(K+1) + c · log₂(K+1)²

producing a peak location ``peak_log = -b / (2c)`` (only meaningful
when ``c < 0``) and a peak height ``peak_rho`` at that location.  This
script asks: **how well does the desc+inst peak K predict the
responses peak K?**  But rather than treating all axes equally, we
weight by per-axis fit quality so noisy / flat fits don't dominate.

Weighting heuristic
-------------------

A few moderately principled per-curve confidence factors, multiplied
together:

- ``peak_rho``   -- low signal shouldn't dominate.
- ``R²``         -- penalise curves the parabola fits poorly.
- ``√|c|``       -- penalise flat parabolas (vertex location is
                    ill-defined when curvature is small).

Combined per-curve confidence:

    confidence = peak_rho × R² × √|c|

Cross-source rebalancing for the desc+inst-vs-responses comparison:
when desc+inst peaks lower than responses (``peak_rho_di < peak_rho_rs``),
desc+inst can't outvote responses, so we multiply its confidence by
``min(1, peak_rho_di / peak_rho_rs)``.  This stops a barely-correlated
desc+inst signal from dragging the weighted fit around just because
its parabola happened to fit well.

Per-axis weight (geometric mean of the two confidences with the
cross-source rebalance baked in):

    w = √( confidence_di · min(1, peak_rho_di / peak_rho_rs) · confidence_rs )

Clipping
--------

When a parabola's vertex falls outside the measured K range
[0, 32] (e.g. ρ declines monotonically from raw, so the fitted vertex
is at log₂(K+1) ≈ -0.5 — i.e. K ≈ -0.3), we clip the displayed peak to
the boundary [0, log₂(33)].  A negative K is meaningless in this
context (raw is already K=0; you can't whiten "negative" PCs), so K=0
is the principled stand-in for "data never rose above raw".

Inputs (from ``--experiment_dir``)
-----------------------------------

- ``whitening_k_sweep.json``  -- produced by ``whitening_k_sweep.py``
- ``<pairs>``                 -- chosen by ``--pairs``
                                 (default ``pair_list_responses.json``)

Outputs (also to ``--experiment_dir``)
--------------------------------------

- ``rho_peak_K_weighted_scatter.png`` -- the scatter plot.
- ``whitening_k_peak_fit_weighted.json`` -- per-(axis, source) fit
  records including ``a, b, c, r2_log, peak_log, peak_rho, abs_c,
  confidence``, plus the per-axis weight.

Examples
--------

::

    # Default: responses cohort (``pair_list_responses.json``):
    uv run python results_analysis/whitening_k_weighted_scatter.py
    # writes whitening_k_peak_fit_weighted_responses_slot6.json + .png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from assistant_axis import cohort_from_pairs, json_metadata, png_metadata
from assistant_axis.provenance import (
    CACHE_POLICIES,
    InputSpec,
    load_and_register,
)

DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)
FIT_K = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
MIN_LOG = 0.0
MAX_LOG = float(np.log2(33))


def fit_log_parabola(ys: list[float]) -> dict:
    """Fit ``ρ ≈ a + b·log₂(K+1) + c·log₂(K+1)²`` and return all the
    bits the weighted-scatter needs."""
    xs = np.array([np.log2(K + 1) for K in FIT_K], dtype=float)
    ys = np.asarray(ys, dtype=float)
    c, b, a = np.polyfit(xs, ys, 2)  # highest order first
    yhat = a + b * xs + c * xs ** 2
    ss_res = float(((ys - yhat) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if c < 0:
        peak_log = float(-b / (2 * c))
        peak_rho = float(a + b * peak_log + c * peak_log ** 2)
    else:
        peak_log = float("nan")
        peak_rho = float("nan")
    abs_c = float(abs(c))
    confidence = (peak_rho * r2 * np.sqrt(abs_c)
                  if np.isfinite(peak_rho) and peak_rho > 0 else 0.0)
    return {
        "a": float(a), "b": float(b), "c": float(c),
        "r2_log": r2,
        "peak_log": peak_log,
        "peak_K": (2 ** peak_log - 1) if np.isfinite(peak_log) else float("nan"),
        "peak_rho": peak_rho,
        "abs_c": abs_c,
        "confidence": float(confidence),
    }


def _wpearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation."""
    wmx = np.average(x, weights=w)
    wmy = np.average(y, weights=w)
    cov = np.average((x - wmx) * (y - wmy), weights=w)
    vx = np.average((x - wmx) ** 2, weights=w)
    vy = np.average((y - wmy) ** 2, weights=w)
    return float(cov / np.sqrt(vx * vy))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR),
                   help=f"Directory holding the K-sweep input + pair list "
                        f"(default: {DEFAULT_EXPERIMENT_DIR}).")
    p.add_argument("--pairs", default="pair_list_responses.json",
                   help="Pair-list JSON filename (default: pair_list_responses.json -- "
                        "the cohort that has both desc+instr and responses ρ; "
                        "the scatter is only meaningful for axes with both).")
    p.add_argument("--slot", type=int, default=6,
                   help="Token-position slot (default: 6 = </think>).  Used to "
                        "auto-suffix --sweep / --plot / --fit_json defaults.  "
                        "Pass --slot 3 (\\n) or 7 (\\n\\n post) to compare.")
    p.add_argument("--sweep", default=None,
                   help="K-sweep input filename (default: "
                        "whitening_k_sweep_<cohort>_slot{N}.json -- cohort "
                        "is derived from --pairs).")
    p.add_argument("--fit_json", default=None,
                   help="Output fit-records filename (default: "
                        "whitening_k_peak_fit_weighted_<cohort>_slot{N}.json).")
    p.add_argument("--plot", default=None,
                   help="Output PNG filename (default: "
                        "rho_peak_K_weighted_scatter_<cohort>_slot{N}.png).")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to react to drift in the sweep JSON's "
                        "recorded provenance: strict / warn (default) / "
                        "rebuild / off.")
    args = p.parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    slot = int(args.slot)
    cohort = cohort_from_pairs(args.pairs)
    if args.sweep is None:
        args.sweep = f"whitening_k_sweep_{cohort}_slot{slot}.json"
    if args.plot is None:
        args.plot = f"rho_peak_K_weighted_scatter_{cohort}_slot{slot}.png"
    if args.fit_json is None:
        args.fit_json = f"whitening_k_peak_fit_weighted_{cohort}_slot{slot}.json"

    # ``inputs`` is built up via load_and_register: each cache read
    # both unwraps the envelope (legacy bare-list files pass through
    # with check=None), validates its recorded provenance under
    # ``--cache-policy``, AND appends an InputSpec to this list, so
    # the read and the dependency record can't fall out of sync.
    inputs: list[InputSpec] = []
    sweep, _spec_sweep, _check_sweep = load_and_register(
        experiment_dir / args.sweep,
        dep_key="sweep_json",
        inputs=inputs,
        policy=args.cache_policy,
    )
    pairs, _spec_pairs, _check_pairs = load_and_register(
        experiment_dir / args.pairs,
        dep_key="pairs_json",
        inputs=inputs,
        policy=args.cache_policy,
    )

    # axis -> source -> {K: rho}
    curves: dict = {}
    for r in sweep:
        curves.setdefault((r["pos"], r["neg"]), {}) \
              .setdefault(r["source"], {})[r["K"]] = r["rho"]

    # Refit each (axis, source) curve and collect summaries.
    fit_records: list[dict] = []
    meta: dict = {}  # (pos, neg, source) -> fit dict
    for (pos, neg), by_src in curves.items():
        for src, m in by_src.items():
            ys = [m[K] for K in FIT_K]
            fit = fit_log_parabola(ys)
            fit_records.append({
                "pos": pos, "neg": neg, "source": src, **fit, "ys": ys,
            })
            meta[(pos, neg, src)] = fit

    # Per-axis weight + clipped (x, y) for scatter.
    points: list[dict] = []
    for it in pairs:
        key = (it["pos"], it["neg"])
        di = meta[(*key, "desc_inst")]
        rs = meta[(*key, "responses")]

        peak_rho_di = di["peak_rho"]
        peak_rho_rs = rs["peak_rho"]
        rel = (min(1.0, peak_rho_di / peak_rho_rs)
               if (peak_rho_rs is not None and np.isfinite(peak_rho_rs)
                   and peak_rho_rs > 0
                   and np.isfinite(peak_rho_di)) else 1.0)
        w = float(np.sqrt(max(0.0, di["confidence"] * rel * rs["confidence"])))

        x_raw = di["peak_log"]; y_raw = rs["peak_log"]
        x_c = (min(MAX_LOG, max(MIN_LOG, x_raw))
               if np.isfinite(x_raw) else MIN_LOG)
        y_c = (min(MAX_LOG, max(MIN_LOG, y_raw))
               if np.isfinite(y_raw) else MIN_LOG)
        points.append({
            "key": key, "x": x_c, "y": y_c, "w": w,
            "x_raw": x_raw, "y_raw": y_raw,
            "peak_rho_di": peak_rho_di, "peak_rho_rs": peak_rho_rs,
        })

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    ws = np.array([p["w"] for p in points])

    pear_u = float(pearsonr(xs, ys).statistic)
    spear_u = float(spearmanr(xs, ys).correlation)
    pear_w = (_wpearson(xs, ys, ws) if ws.sum() > 0 else float("nan"))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    size_scale = 150.0 / ws.max() if ws.max() > 0 else 1.0
    sizes = size_scale * ws
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(points))))

    fig, ax = plt.subplots(figsize=(7, 7))
    for p_i, c, sz in zip(points, colors, sizes):
        ax.scatter(p_i["x"], p_i["y"], s=sz, color=c, edgecolor="black",
                   linewidth=0.8, zorder=3)
        ax.annotate(p_i["key"][0], (p_i["x"], p_i["y"]),
                    fontsize=8, xytext=(6, 4), textcoords="offset points")

    lims = [-0.2, MAX_LOG + 0.3]
    ax.plot(lims, lims, "k--", alpha=0.4, lw=0.8, label="y = x")
    if ws.sum() > 0:
        slope, intercept = np.polyfit(xs, ys, 1, w=ws)
        xx = np.linspace(lims[0], lims[1], 50)
        ax.plot(xx, slope * xx + intercept, "g-", alpha=0.6, lw=1.2,
                label="weighted OLS")

    tick_Ks = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    tick_logs = [np.log2(K + 1) for K in tick_Ks]
    ax.set_xticks(tick_logs)
    ax.set_xticklabels([f"K={K}" for K in tick_Ks], fontsize=8)
    ax.set_yticks(tick_logs)
    ax.set_yticklabels([f"K={K}" for K in tick_Ks], fontsize=8)

    ws_sorted = np.sort(ws)
    if len(ws_sorted) >= 3:
        ref_ws = [ws_sorted[0], ws_sorted[len(ws_sorted) // 2], ws_sorted[-1]]
    else:
        ref_ws = list(ws_sorted)
    for ref_w in ref_ws:
        ax.scatter([], [], s=size_scale * ref_w, color="grey",
                   edgecolor="black", linewidth=0.5,
                   label=f"w={ref_w:.3f}")

    ax.set_xlabel("desc+inst fitted peak  log₂(K+1)  (clipped to K ≥ 0)")
    ax.set_ylabel("responses fitted peak  log₂(K+1)  (clipped to K ≥ 0)")
    title_line = "Peak K desc+inst vs responses (circle area ∝ weight)"
    ax.set_title(
        title_line + "\n"
        f"Unweighted Pearson={pear_u:+.3f}, Spearman={spear_u:+.3f}  |  "
        f"Weighted Pearson={pear_w:+.3f}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.legend(loc="upper left", fontsize=7, scatterpoints=1, labelspacing=1.1)
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    # ``inputs`` was already populated above by the load_and_register
    # calls; both upstream files are recorded.

    plot_path = experiment_dir / args.plot
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # Save per-axis points (with weights and raw/clipped peaks).
    out_records = []
    for r in fit_records:
        # Find the matching scatter point if this is one of the listed axes.
        matched = next(
            (p for p in points
             if p["key"] == (r["pos"], r["neg"]) and r["source"] == "desc_inst"),
            None,
        )
        out_records.append({
            **r,
            "weight_when_di": matched["w"] if matched else None,
        })
    fit_path = experiment_dir / args.fit_json
    fit_envelope = json_metadata(
        out_records, inputs=inputs,
        title=f"whitening_k_weighted_scatter slot={slot}")
    with open(fit_path, "w") as _f:
        json.dump(fit_envelope, _f, indent=2, default=str)
    print(f"Wrote {fit_path}")

    # Summary print.
    print("\nPoint coordinates (log₂(K+1) space):")
    for p_i in points:
        clipped_note = ""
        if not np.isclose(p_i["x"], p_i["x_raw"]) or not np.isclose(p_i["y"], p_i["y_raw"]):
            clipped_note = (f"  [clipped: raw x,y = "
                            f"{p_i['x_raw']:+.2f}, {p_i['y_raw']:+.2f}]")
        print(f"  {p_i['key'][0]:16s} vs {p_i['key'][1]:16s}  "
              f"x={p_i['x']:+.2f}  y={p_i['y']:+.2f}  "
              f"w={p_i['w']:.3f}{clipped_note}")
    print(f"\nUnweighted Pearson = {pear_u:+.3f}")
    print(f"Unweighted Spearman = {spear_u:+.3f}")
    print(f"Weighted Pearson    = {pear_w:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
