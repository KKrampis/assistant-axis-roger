#!/usr/bin/env python3
"""Plot the B-size cost-vs-quality curve on a 1/(1-ρ) "quality" scale.

Reads the JSON cache produced by :mod:`batch_size_rho_curve` (default
``roger/batch_size_curve_rho.json``) and emits a single-panel scatter:

- x: cost per axis (USD), from the README's per-batch token model
- y: quality = ``1 / (1 − ρ)``, where ρ is the grand-mean ρ across
  the axes × (slot, layer) configs in the cache.

Why the 1/(1-ρ) scale?  Each +0.01 ρ matters exponentially more as you
approach the ρ=1 ceiling.  At ρ ≈ 0.77 (our typical responses-mode
operating point), ``d(1/(1−ρ))/dρ ≈ 18`` -- so each +0.01 ρ buys roughly
a 4 % improvement in effective signal.  The diminishing-returns shape
of the cost-vs-quality curve is much more visible here than on a linear
ρ axis.

Side-effect: prints a small table of marginal Δquality / Δ$ between
adjacent batch sizes (a cleaner way to read "where does the
cost-efficiency curve fall off?").

CLI
---

::

    # Default: read roger/batch_size_curve_rho.json, write
    # roger/batch_size_cost_vs_quality.png.
    uv run python results_analysis/plot_batch_size_quality_vs_cost.py

    # Custom JSON / output path:
    uv run python results_analysis/plot_batch_size_quality_vs_cost.py \\
      --input roger/b_curve_truthful/batch_size_curve_rho.json \\
      --output roger/b_curve_truthful/cost_vs_quality.png

Library
-------

::

    from results_analysis.plot_batch_size_quality_vs_cost import plot_quality_vs_cost
    plot_quality_vs_cost(input_path=..., output_path=...)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assistant_axis.plot_metadata import png_metadata, suptitle_with_specs
from assistant_axis.provenance import (
    CACHE_POLICIES,
    InputSpec,
    current_file_input,
    load_and_register,
)


DEFAULT_INPUT = "roger/batch_size_curve_rho.json"
DEFAULT_OUTPUT = "roger/batch_size_cost_vs_quality.png"


def _default_ensembles_path(input_path: Path) -> Path:
    """Side-car ensembles JSON path next to a batch-size cache.
    Mirrors :func:`judge_ensemble_rho_curve._default_output_for`.
    """
    p = Path(input_path)
    return p.with_name(f"{p.stem}_ensembles{p.suffix}")


# ---------------------------------------------------------------------------
# Multi-judge cost extras
# ---------------------------------------------------------------------------
# At B=10 the GPT-4.1-mini per-axis cost (both cohorts combined) is
# ~$49.40 (input $46.25 + output $3.15).  When we ensemble GPT-mini
# with a second judge, the second judge processes the same items with
# the same input/output token counts -- only the per-token rates
# change (and the question_subsample_modulo factor reduces items
# proportionally).
#
# Subsample q9 ⇒ 1/3 of items (per axis_judge_correlation.py:1761).
#
# Per-axis cost (both cohorts) = input_cost + output_cost
# Input model:   115.6M tokens × $rate_in  per axis at B=10 full
# Output model:    1.97M tokens × $rate_out per axis at B=10 full
#
# These constants come from a from-scratch dry-run (2026-05-09): 240
# real B=10 prompts reconstructed from v1-rubric cache batch keys +
# matching responses in `runpod_workspace/.../{roles,traits}/responses/
# <entity>.jsonl`, tokenized with `tiktoken` `o200k_base`.  Sample
# stats: per-batch input mean 4,700 tokens (median 5,018, σ 1,065),
# output mean 80 tokens.  Per-axis batches: 24,605 (mean over 12 v1
# axes).  These supersede the older 16.6M/2.08M figures, which were
# header-only estimates that under-counted the actual response
# content (~95% of every prompt) by ~5x.  See README "Judging cost
# model" for the derivation table at multiple B values.
#
# 2026-05-10 cross-axis validation: re-ran the dry-run on ALL 12 v2
# axes × both cohorts (295,258 reconstructed B=10 batches via
# tools/dry_run_response_token_count.py).  Per-axis cost ratio
# (measured / 49.40) was 1.006 ± 0.013 over the 12-axis sample with
# range [0.989, 1.034].  The 4,700/80/24,605 figures are correct
# within ±3% per axis and within 0.6% on average -- no recalibration
# needed.  See AGENT_NOTES "Judging cost model" for the per-axis
# breakdown.
B10_GPT_INPUT_M_TOK = 115.6       # millions of input tokens at B=10 full (both cohorts)
B10_GPT_OUTPUT_M_TOK = 1.97       # millions of output tokens at B=10 full (both cohorts)

# 2026-05-11 PHASE-5C/5D FULL-SWEEP EMPIRICAL HAIKU/GPT TOKEN-SCALES.
# ------------------------------------------------------------------
# 5c.1 (GPT full regen, B=7, 11 v2 axes × 2 cohorts, 372,652 calls):
#   per-call input  3,433 tok   (5c.0 canary: 3,402)
#   per-call output    74.0 tok (5c.0 canary: 59.8 -- canary was
#                                low-verbosity, full sweep raises)
# 5d.1 (Haiku full sweep, B=7, q9/t3 subsample, 11 axes × 2 cohorts,
#         8,927 calls):
#   per-call input  3,730 tok   (5d.0 canary: 3,727)
#   per-call output  160.2 tok  (5d.0 canary: 131)
#
# Per-call scale ratios (Haiku / GPT, full-sweep numbers preferred
# since they cover 11 distinct axes with N >> canary):
#   input  scale = 3730 / 3433 = 1.087   (canary said 1.18; full
#                                          sweep is more reliable)
#   output scale = 160.2 / 74.0 = 2.165  (canary said 2.19; we had
#                                          been using 2.00 as a
#                                          conservative mid-estimate
#                                          and that under-budgeted
#                                          Haiku cost by ~21% --
#                                          the 5d.1 actual/expected
#                                          ratio of 1.21 traces
#                                          directly to that).
#
# Mechanistic interpretation:
#   1. Tokenizer drift: Anthropic's tokenizer is ~9% chunkier than
#      o200k_base on response prompts dominated by raw text (the
#      tokenizer-gap is much smaller than the 1.35× seen on static
#      prompts in 5b, because response prompts are mostly verbatim
#      response text where both tokenizers do well).
#      HAIKU_INPUT_SCALE = 1.09.
#   2. Verbosity: Haiku produces ~2.17× the output tokens of GPT-mini
#      for the same item, in both static (1.93×) and response (2.16×)
#      prompts.  HAIKU_OUTPUT_SCALE = 2.17 (empirical full-sweep).
#
# Sonnet uses the SAME Anthropic tokenizer family as Haiku, so the
# input scale carries over directly.  Sonnet's verbosity has NOT
# been measured in this project; we apply the Haiku output scale as
# a placeholder (Sonnet is typically ≥ as verbose as Haiku, so this
# is more likely an under-estimate than over-estimate).
HAIKU_INPUT_SCALE = 1.09
HAIKU_OUTPUT_SCALE = 2.17
SONNET_INPUT_SCALE = HAIKU_INPUT_SCALE   # same Anthropic tokenizer
SONNET_OUTPUT_SCALE = HAIKU_OUTPUT_SCALE  # CAVEAT: untested for Sonnet

# Pricing (per 1M tokens), pinned to what the archived batch-size-curve
# experiment (roger/axis_judge_experiments/batch_size_curve_8slot/) actually
# used: gpt-4.1-mini and claude-sonnet-4 (not the current claude-sonnet-5,
# which is priced differently -- $2.00/$10.00 as of 2026-09 -- see
# assistant_axis/judge_pricing.py for current, multi-tier, versioned rates).
# Deliberately NOT updated to current rates: doing so would silently
# misprice a regeneration of this specific chart from its existing cached
# usage data, which was billed at these historical rates.
GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT = 0.40, 1.60
HAIKU_RATE_IN, HAIKU_RATE_OUT = 1.00, 5.00
SONNET_RATE_IN, SONNET_RATE_OUT = 3.00, 15.00  # claude-sonnet-4, not sonnet-5


def _judge_cost_per_axis(rate_in: float, rate_out: float,
                          input_scale: float = 1.0,
                          output_scale: float = 1.0,
                          subsample: float = 1.0) -> float:
    """Per-axis B=10 cost in USD for a judge with the given rates,
    per-judge input/output token-scale multipliers (relative to the
    GPT-4.1-mini per-axis token totals), and an optional subsample
    factor (1.0 = full, 1/3 ≈ q9)."""
    return subsample * (B10_GPT_INPUT_M_TOK * input_scale * rate_in
                        + B10_GPT_OUTPUT_M_TOK * output_scale * rate_out)


# Cost extras to draw on top of the B-curve.  Each entry has an
# ``ensemble_key`` matching the combo name in the JSON's
# ``ensemble_combos`` block (computed by judge_ensemble_rho_curve.py);
# if present, the extra point's ρ is read from there.  If absent (older
# JSON), we fall back to the B=10 ρ as a placeholder.
EXTRA_POINTS = [
    {"label": "+ Haiku q9",
     "extra_cost": _judge_cost_per_axis(
         HAIKU_RATE_IN, HAIKU_RATE_OUT,
         input_scale=HAIKU_INPUT_SCALE,
         output_scale=HAIKU_OUTPUT_SCALE,
         subsample=1/3),
     "color": "#dd8452",   # orange
     "dy_pt": -18,         # annotation below marker
     "ensemble_key": "gpt_b10__plus_haiku_q9"},
    {"label": "+ Haiku full",
     "extra_cost": _judge_cost_per_axis(
         HAIKU_RATE_IN, HAIKU_RATE_OUT,
         input_scale=HAIKU_INPUT_SCALE,
         output_scale=HAIKU_OUTPUT_SCALE,
         subsample=1.0),
     "color": "#c44e52",   # red
     "dy_pt": -18,
     "ensemble_key": "gpt_b10__plus_haiku_full"},
    {"label": "+ Sonnet q9",
     "extra_cost": _judge_cost_per_axis(
         SONNET_RATE_IN, SONNET_RATE_OUT,
         input_scale=SONNET_INPUT_SCALE,
         output_scale=SONNET_OUTPUT_SCALE,
         subsample=1/3),
     "color": "#8172b2",   # purple
     "dy_pt": +22,         # above marker (dodges coincident Haiku-full)
     "ensemble_key": "gpt_b10__plus_sonnet_q9"},
]


# Per-axis-set marker styling for the 4-Pareto-frontier overlay.  Set 1
# (initial) reuses the existing big black filled square that anchors the
# B-curve -- we don't draw a second marker for it.  Sets 2..4 get
# distinct hollow shapes so each set's anchor + ensemble rays are
# visually traceable when colours coincide between rays of different
# sets.  Order matches ``judge_ensemble_rho_curve.AXIS_SETS``.
SET_MARKERS: dict[str, dict] = {
    "set_1_initial": {
        "marker":      "s",          # filled square (matches B-curve)
        "size":        12,
        "facecolor":   "#333333",
        "edgecolor":   "black",
        "edgewidth":   0.6,
        "is_overlay":  False,        # already drawn by the B-curve
        "short_label": "set 1",
    },
    "set_2": {
        "marker":      "o",
        "size":        10,
        "facecolor":   "#333333",
        "edgecolor":   "black",
        "edgewidth":   0.6,
        "is_overlay":  True,
        "short_label": "set 2",
    },
    "set_3": {
        "marker":      "^",
        "size":        11,
        "facecolor":   "#333333",
        "edgecolor":   "black",
        "edgewidth":   0.6,
        "is_overlay":  True,
        "short_label": "set 3",
    },
    "set_4": {
        "marker":      "v",
        "size":        11,
        "facecolor":   "#333333",
        "edgecolor":   "black",
        "edgewidth":   0.6,
        "is_overlay":  True,
        "short_label": "set 4",
    },
}


def _best_across_cells_mean_axes(per_axis: dict, B_int: dict) -> dict:
    """Aggregate ``per_axis`` (one ρ per ``<axis>|b<N>|s<S>_l<L>``) into
    ``{b<N>: ρ}`` where ρ is the **mean across axes** of the **max ρ
    across cells**.  Captures "for each axis, what's the best cell at
    this B, averaged over axes" — a tighter quality metric than the
    grand mean (which averages over both axes and cells).
    """
    # axis -> {b<N> -> [ρ over cells]}
    by_axis: dict[str, dict[str, list[float]]] = {}
    for k, r in per_axis.items():
        try:
            axis_part, b_part, _cell_part = k.split("|", 2)
        except ValueError:
            continue
        by_axis.setdefault(axis_part, {}).setdefault(b_part, []).append(r)

    # max-over-cells per (axis, B)
    best_per_axis_B: dict[str, dict[str, float]] = {
        axis: {b: max(rhos) for b, rhos in d.items() if rhos}
        for axis, d in by_axis.items()
    }
    # mean-over-axes per B
    out: dict[str, float] = {}
    for b in B_int:
        vals = [best_per_axis_B[a][b]
                for a in best_per_axis_B
                if b in best_per_axis_B[a]]
        if vals:
            out[b] = sum(vals) / len(vals)
    return out


def _load_batch_size_json(
    path: Path, *,
    policy: str = "warn",
    inputs: list[InputSpec] | None = None,
    dep_key: str = "batch_size_curve_json",
) -> dict:
    """Load + (optionally) validate the batch-size cache.

    Tolerates legacy bare-dict caches (returns ``check=None``) and
    enforces the user-selected cache policy on envelope-wrapped ones.
    When an ``inputs`` accumulator is supplied, also appends an
    InputSpec for ``path`` so the read and the registration can't
    drift apart (see AGENT_NOTES.md "Reader+registrar pattern").
    """
    payload, _spec, _check = load_and_register(
        path, dep_key=dep_key, inputs=inputs, policy=policy,
    )
    return payload


def plot_quality_vs_cost(
    input_path: Path,
    output_path: Path,
    *,
    raw: dict | None = None,
    ensembles: dict | None = None,
    ensembles_path: Path | None = None,
    policy: str = "warn",
) -> Path:
    """Draw the cost-vs-quality plot from a batch_size_curve_rho.json cache.

    Y-axis: ρ aggregated as **best-across-cells, mean-across-axes**
    (per axis, take max over the 6 (slot, layer) cells; then mean
    across axes).  Tighter than the grand mean — represents the
    "axis-tuned" upper bound.

    If ``raw`` is supplied (the dict already loaded by the caller),
    we skip the disk read; otherwise we load + validate ``input_path``
    using ``policy``.  This avoids double-validation when the caller
    has already validated.

    Ensemble-extras handling (split-file design, May 2026):

    * Pre-Phase-6 caches stored ensembles inline under
      ``raw["ensemble_combos"]``.  Post-split they live in a side-car
      JSON written by :mod:`judge_ensemble_rho_curve`.
    * ``ensembles`` -- pre-loaded dict (the side-car's unwrapped
      ``result``).  Wins over the inline-fallback when both are present.
    * ``ensembles_path`` -- path to load when ``ensembles`` is None and
      the path exists; defaults to the side-car next to ``input_path``.

    Returns the output PNG path.
    """
    # Build the provenance ``inputs`` list as we go: load_and_register
    # handles read + register together when this function does the read,
    # and we fall back to a bare current_file_input when the caller
    # already loaded the data and only the path needs registering.
    # Either way every cache that contributes to the output PNG ends up
    # exactly once in ``inputs``.
    inputs: list[InputSpec] = []
    if raw is None:
        raw = _load_batch_size_json(
            input_path, policy=policy,
            inputs=inputs, dep_key="batch_size_curve_json",
        )
    else:
        inputs.append(current_file_input(
            dep_key="batch_size_curve_json", path=input_path))
    # Resolve ensemble data: pre-loaded > side-car file > inline-legacy.
    if ensembles is None:
        ens_path = ensembles_path or _default_ensembles_path(input_path)
        if Path(ens_path).exists():
            ensembles, _spec, _check = load_and_register(
                Path(ens_path), dep_key="ensembles_json",
                inputs=inputs, policy=policy,
            )
            ensembles_path = Path(ens_path)
        else:
            # Older caches kept ensemble_combos inline; still honour them
            # so historical batch_size_curve_rho.json files keep rendering.
            ensembles = {"ensemble_combos": raw.get("ensemble_combos") or {}}
            ensembles_path = None
    elif ensembles_path is not None:
        inputs.append(current_file_input(
            dep_key="ensembles_json", path=ensembles_path))
    cost = raw["cost_per_axis_usd"]
    B_int = raw["B_integer"]
    # Prefer best-across-cells aggregation if per_axis is in the JSON
    # (it has been since the May 2026 schema update).  Fall back to the
    # historical grand_mean for older caches.
    if "per_axis" in raw:
        grand = _best_across_cells_mean_axes(raw["per_axis"], B_int)
    else:
        grand = raw["grand_mean_per_b"]
    # ``axes`` was added to batch_size_rho_curve.py's JSON output in
    # May 2026; older caches don't have it, in which case we recover the
    # axis count from the unique ``<axis>|b<N>|s<S>_l<L>`` keys in
    # ``per_axis``.  ``configs`` is similarly fallback'd from per_axis.
    axes_field = raw.get("axes") or []
    if axes_field:
        n_axes = len(axes_field)
    else:
        n_axes = len({k.split("|", 1)[0] for k in raw.get("per_axis", {})})
    configs_field = raw.get("configs") or []
    if configs_field:
        n_configs = len(configs_field)
    else:
        n_configs = len({k.rsplit("|", 1)[1] for k in raw.get("per_axis", {})})

    # Sort by ascending cost (== descending B) so the line draws cheap → expensive.
    Bs = sorted((int(B_int[k]) for k in B_int), reverse=True)
    keys = [f"b{B}" for B in Bs]
    xs = [cost[k] for k in keys]
    rhos = [grand[k] for k in keys]
    qual = [1.0 / (1.0 - r) for r in rhos]

    fig, ax = plt.subplots(figsize=(11.0, 6.5))

    # Non-linear y-axis: underlying coordinate is y = 1/(1-ρ), but ticks
    # and labels are in ρ units.  Each +0.01 ρ matters exponentially more
    # as ρ approaches 1, and this scale makes the diminishing-returns
    # shape visible: low-ρ ticks are bunched near the bottom, high-ρ
    # ticks are spread out near the top.  forward(ρ) = 1/(1-ρ) is
    # monotonically increasing on (-∞, 1), so larger ρ naturally sits
    # higher on screen (better quality up) without an axis inversion.
    def _fwd(r):
        import numpy as np
        r = np.asarray(r, dtype=float)
        return 1.0 / np.maximum(1.0 - r, 1e-9)

    def _inv(y):
        import numpy as np
        y = np.asarray(y, dtype=float)
        return 1.0 - 1.0 / np.maximum(y, 1e-9)

    ax.set_yscale("function", functions=(_fwd, _inv))

    # Plot the raw ρ values (not the transformed quality); the FuncScale
    # handles the visual non-linearity for us.
    ax.plot(xs, rhos, marker="s", markersize=12, linewidth=2.0,
             color="#333333", zorder=5)

    # Annotate each point with B, ρ, cost, quality.
    for i, (B, x, r, q) in enumerate(zip(Bs, xs, rhos, qual)):
        # Alternate above/below to avoid overlap.
        dy = 14 if i % 2 == 0 else -14
        va = "bottom" if dy > 0 else "top"
        label = (f"B={B}\nρ = {r:+.4f}\n"
                 f"${x:.2f}/axis\n1/(1−ρ) = {q:.3f}")
        ax.annotate(
            label,
            xy=(x, r), xytext=(0, dy), textcoords="offset points",
            ha="center", va=va, fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#ffffff", edgecolor="#aaaaaa",
                       alpha=0.92),
            arrowprops=dict(arrowstyle="-",
                             connectionstyle="arc3,rad=0",
                             color="#666666", lw=0.6),
        )

    # Linear-endpoints reference line: drawn as a straight line in the
    # *transformed* (=quality) space so it remains a true chord between
    # the two endpoint points -- concave curves lie above their chord.
    # In ρ-space this looks slightly bowed because of the FuncScale, but
    # with set_yscale handling the transform we just plot in ρ values
    # corresponding to the linearly-interpolated quality values.
    chord_slope: float | None = None
    if len(xs) >= 2:
        chord_slope = (qual[-1] - qual[0]) / (xs[-1] - xs[0])
        # Sample the chord densely so it draws smoothly through the
        # nonlinear axis.
        import numpy as np
        x_chord = np.linspace(xs[0], xs[-1], 50)
        q_chord = qual[0] + chord_slope * (x_chord - xs[0])
        r_chord = 1.0 - 1.0 / q_chord
        ax.plot(x_chord, r_chord,
                 linestyle=":", color="#888888", linewidth=1.2)

    # ---- 4-Pareto-frontier overlay (per-axis-set ensemble rays) ------
    # Each axis set forms its own mini Pareto front.  The set's anchor
    # is at (B=10 GPT cost, set's GPT-only B=10 mean ρ); colored rays
    # extend to (anchor_x + Anthropic-judge cost, set's ensemble ρ for
    # that combo).  Set 1 reuses the existing big black square as its
    # anchor; sets 2..4 get hollow shape overlays (see SET_MARKERS).
    # Colors match across sets per ensemble combo, so visually a fan of
    # same-coloured rays from different anchors lets you read off
    # "is this combo's slope consistent across the full axis sample?".
    b10_idx = Bs.index(10) if 10 in Bs else None
    ensemble_combos = ensembles.get("ensemble_combos") or {}
    gpt_only_per_set = (
        (ensembles.get("gpt_only_b10_baseline") or {}).get("per_set") or []
    )
    # Track every per-set ρ that lands on the plot so the y-range pad
    # downstream covers all of them, not just set 1.
    per_set_rhos_to_plot: list[float] = []
    if b10_idx is not None and EXTRA_POINTS and gpt_only_per_set:
        anchor_x = xs[b10_idx]
        # Index ensemble per_set entries by set_id for O(1) lookup.
        combo_per_set: dict[str, dict[str, float]] = {}
        for ep in EXTRA_POINTS:
            entries = (
                ensemble_combos.get(ep["ensemble_key"], {}).get("per_set") or []
            )
            combo_per_set[ep["ensemble_key"]] = {
                e["set_id"]: float(e["rho_mean_best_cell"]) for e in entries
            }
        for set_entry in gpt_only_per_set:
            sid = set_entry["set_id"]
            anchor_y = float(set_entry["rho_mean_best_cell"])
            n_axes_set = int(set_entry.get("n_axes", 0))
            mark = SET_MARKERS.get(sid, SET_MARKERS["set_2"])
            per_set_rhos_to_plot.append(anchor_y)
            # Draw set anchor (sets 2..4; set 1 already has the B-curve square).
            if mark["is_overlay"]:
                ax.plot([anchor_x], [anchor_y],
                        marker=mark["marker"],
                        markersize=mark["size"],
                        markerfacecolor=mark["facecolor"],
                        markeredgecolor=mark["edgecolor"],
                        markeredgewidth=mark["edgewidth"],
                        linestyle="none", zorder=6)
                ax.annotate(
                    f"{mark['short_label']}\nρ={anchor_y:+.4f}",
                    xy=(anchor_x, anchor_y),
                    xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5,
                    color="#222222",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#ffffff", edgecolor="#888888",
                              alpha=0.85),
                )
            # Draw colored rays to each ensemble combo that has data
            # for this set.
            for ep in EXTRA_POINTS:
                ens_rho = combo_per_set[ep["ensemble_key"]].get(sid)
                if ens_rho is None:
                    continue   # combo has no data for this set
                x_extra = anchor_x + ep["extra_cost"]
                ax.plot([anchor_x, x_extra], [anchor_y, ens_rho],
                        linewidth=1.4,
                        color=ep["color"], alpha=0.85, zorder=4)
                ax.plot([x_extra], [ens_rho],
                        marker=mark["marker"],
                        markersize=mark["size"],
                        markerfacecolor=ep["color"],
                        markeredgecolor="black",
                        markeredgewidth=0.6,
                        linestyle="none", zorder=5)
                per_set_rhos_to_plot.append(ens_rho)
            # Annotate set 1's three endpoints (the canonical reference
            # picture); for sets 2..4 the legend + anchor label carries
            # the identification, so we skip per-endpoint boxes to avoid
            # clutter.
            if sid == "set_1_initial":
                for ep in EXTRA_POINTS:
                    ens_rho = combo_per_set[ep["ensemble_key"]].get(sid)
                    if ens_rho is None:
                        continue
                    x_extra = anchor_x + ep["extra_cost"]
                    ax.annotate(
                        f"{ep['label']}\nρ={ens_rho:+.4f}\n"
                        f"${x_extra:.2f}/axis",
                        xy=(x_extra, ens_rho),
                        xytext=(0, ep["dy_pt"]),
                        textcoords="offset points",
                        ha="center",
                        va=("bottom" if ep["dy_pt"] > 0 else "top"),
                        fontsize=8.5, color=ep["color"],
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#ffffff",
                                  edgecolor=ep["color"], alpha=0.92),
                        arrowprops=dict(arrowstyle="-",
                                        connectionstyle="arc3,rad=0",
                                        color=ep["color"], lw=0.6),
                    )

    ax.set_xlabel("Cost per axis (USD, responses-mode; gpt-4.1-mini base + "
                  "optional second judge)")
    ax.set_ylabel(r"$\rho$ (axis scaled by $1/(1-\rho)$)")
    ax.grid(True, alpha=0.25)
    # X-ticks: B-curve costs, plus the extra-point costs.
    extra_xs = (
        [xs[b10_idx] + ep["extra_cost"] for ep in EXTRA_POINTS]
        if (b10_idx is not None and EXTRA_POINTS) else []
    )
    all_xs = sorted(set(xs + extra_xs))
    ax.set_xticks(all_xs)
    ax.set_xticklabels([f"${c:.0f}" if c >= 20 else f"${c:.2f}"
                        for c in all_xs], rotation=30, ha="right")
    # X-axis range with padding so extras' annotation boxes don't get
    # clipped at the right edge.
    if all_xs:
        x_lo = all_xs[0] - 0.05 * (all_xs[-1] - all_xs[0])
        x_hi = all_xs[-1] + 0.10 * (all_xs[-1] - all_xs[0])
        ax.set_xlim(x_lo, x_hi)

    # Y-axis padding: pad in the *transformed* (= quality) space so the
    # data fills most of the figure regardless of where on the asymptotic
    # curve it sits.  Padding in raw ρ-space gets disproportionately
    # expanded near ρ=1 because the FuncScale stretches that region;
    # quality-space padding is roughly uniform on screen.
    #
    # Range covers the B-curve AND every per-set anchor + per-set
    # ensemble endpoint so all 4 mini Pareto fronts are visible
    # regardless of where the extras land.  ``per_set_rhos_to_plot``
    # was populated by the per-set rendering loop above.
    import numpy as np
    rhos_for_range = list(rhos) + list(per_set_rhos_to_plot)
    qual_for_range = [1.0 / (1.0 - r) for r in rhos_for_range]
    q_min, q_max = min(qual_for_range), max(qual_for_range)
    q_span = q_max - q_min
    q_lo = q_min - q_span * 0.20
    q_hi = q_max + q_span * 0.30
    rho_lo = max(0.0, 1.0 - 1.0 / q_lo)
    rho_hi = min(0.9999, 1.0 - 1.0 / q_hi)
    ax.set_ylim(rho_lo, rho_hi)

    # Y-ticks at "nice" round ρ values (multiples of 1, 2, or 5 × 10^k)
    # rather than uniformly spaced.  This means horizontal gridlines and
    # labels land at simple decimals like 0.770, 0.775, 0.780 rather
    # than weird values from a linspace.
    span = rho_hi - rho_lo
    raw_step = span / 6.0  # aim for ~6 ticks
    magnitude = 10.0 ** np.floor(np.log10(raw_step))
    norm = raw_step / magnitude
    nice_norm = 1.0 if norm <= 1.0 else 2.0 if norm <= 2.0 else 5.0 if norm <= 5.0 else 10.0
    step = nice_norm * magnitude
    tick_start = np.ceil(rho_lo / step) * step
    tick_end = np.floor(rho_hi / step) * step
    yticks = np.round(np.arange(tick_start, tick_end + step / 2, step),
                      max(0, int(-np.floor(np.log10(step))) + 1))
    if step >= 0.01:
        fmt = "{:.2f}"
    elif step >= 0.001:
        fmt = "{:.3f}"
    else:
        fmt = "{:.4f}"
    ax.set_yticks(yticks)
    ax.set_yticklabels([fmt.format(t) for t in yticks])

    # Two-block legend: combos (colours) on the left, axis sets
    # (marker shapes) on the right of the upper row.  Built from
    # proxy artists since the actual rendering paints the same colour
    # across multiple sets and the same shape across multiple combos.
    from matplotlib.lines import Line2D as _L2D
    combo_handles = [
        _L2D([], [], color=ep["color"], lw=2.0,
             marker="s", markersize=8,
             markerfacecolor=ep["color"], markeredgecolor="black",
             label=f"{ep['label']} (+${ep['extra_cost']:.2f})")
        for ep in EXTRA_POINTS
    ]
    # Linear-endpoints reference line proxy (the ":" line drawn earlier).
    if chord_slope is not None:
        combo_handles.append(
            _L2D([], [], color="#888888", lw=1.2, ls=":",
                 label=f"set-1 B-curve chord (slope ≈ "
                       f"{chord_slope:.3f} qual/$)")
        )
    set_handles = []
    set_labels_axes = {
        "set_1_initial": "set 1 (truthful, progressive, improvisational)",
        "set_2": "set 2 (concise, ecocentric, egalitarian)",
        "set_3": "set 3 (guileless, harmless, helpful)",
        "set_4": "set 4 (honest, relativist, systems_thinker)",
    }
    for set_entry in gpt_only_per_set:
        sid = set_entry["set_id"]
        mark = SET_MARKERS.get(sid, SET_MARKERS["set_2"])
        set_handles.append(
            _L2D([], [], linestyle="none",
                 marker=mark["marker"], markersize=mark["size"],
                 markerfacecolor=mark["facecolor"],
                 markeredgecolor=mark["edgecolor"],
                 markeredgewidth=mark["edgewidth"],
                 label=set_labels_axes.get(sid, sid))
        )
    leg_combos = ax.legend(handles=combo_handles,
                           loc="upper left", fontsize=8.5,
                           framealpha=0.9, title="ensemble combo",
                           title_fontsize=9)
    ax.add_artist(leg_combos)
    ax.legend(handles=set_handles,
              loc="lower right", fontsize=8.5, framealpha=0.9,
              title="axis set (3 axes each)", title_fontsize=9)

    n_sets = len(gpt_only_per_set)
    n_axes_total = sum(int(s.get("n_axes", 0)) for s in gpt_only_per_set)
    title = ("Responses-mode judging: quality vs cost — "
             f"{n_sets} per-axis-set Pareto fronts at B=10")
    # Pre-broken into ≤~110-char lines so matplotlib doesn't stretch
    # the figure horizontally trying to fit a single long line.
    spec = [
        (f"quality $= 1/(1-\\rho)$.  ρ = best-cell-per-axis, axis-mean "
         f"within each set ({n_axes_total} axes total, split into "
         f"{n_sets} alphabetical groups of 3)."),
        ("Black B=5/7/10/15 markers = GPT-4.1-mini alone, set 1 only "
         "(README cost model, $0.40/$1.60 per 1M)."),
        ("Coloured rays from each set's anchor = GPT B=10 + a 2nd judge "
         "on the same items (Haiku $1/$5, Sonnet $3/$15; q9 = 1/3 "
         "question subsample)."),
        ("Score combine: 0.6·gpt + 0.4·anthropic (rounded Haiku-q9 "
         "parabolic peak from the 12-axis response-mode weight sweep)."),
        ("Haiku-q9 is the Pareto winner; Sonnet-q9 shown at the same "
         "weight despite its own peak being higher, since it is "
         "dominated on cost-per-quality."),
        ("Set-2..4 anchor labels show per-set ρ; endpoint marker shape "
         "encodes which set a ray belongs to.  ρ from "
         "judge_ensemble_rho_curve.py."),
    ]
    # line_height bumped from the helper default (0.022) -- on this 6"-tall
    # figure the bold 14pt headline already occupies ~0.024 fig-fraction,
    # so the default puts the first spec line under the headline's
    # descender.  0.030 leaves ~0.005 clearance.
    _, top_rect = suptitle_with_specs(fig, title, spec, line_height=0.030)
    fig.tight_layout(rect=(0, 0, 1, top_rect))

    src_text = Path(__file__).read_text(encoding="utf-8")
    # ``inputs`` was populated above (load_and_register on internal
    # loads, current_file_input on caller-supplied dicts).  The audit
    # tools' transitive propagation will then mark this PNG stale if
    # any per-axis judge cache feeding judge_ensemble_rho_curve.py
    # drifts via the ``ensembles_json`` dep.
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                 metadata=png_metadata(title=title, source_text=src_text,
                                        inputs=inputs))
    plt.close(fig)
    return output_path


def print_marginal_table(
    input_path: Path,
    *,
    raw: dict | None = None,
    policy: str = "warn",
) -> None:
    """Helper: print a small table of marginal Δquality / Δ$ between
    adjacent batch sizes (cheap-to-expensive).  Uses the same best-
    across-cells / mean-across-axes aggregation as the plot.

    Accepts a pre-loaded ``raw`` dict to avoid double validation.
    """
    if raw is None:
        raw = _load_batch_size_json(input_path, policy=policy)
    cost = raw["cost_per_axis_usd"]
    B_int = raw["B_integer"]
    if "per_axis" in raw:
        grand = _best_across_cells_mean_axes(raw["per_axis"], B_int)
    else:
        grand = raw["grand_mean_per_b"]

    Bs = sorted((int(B_int[k]) for k in B_int), reverse=True)  # cheap → expensive
    rows = []
    prev_q = prev_c = None
    print()
    print(f"{'B':>3}  {'cost':>8}  {'ρ':>9}  {'1/(1−ρ)':>9}  {'+q / +$':>9}")
    for B in Bs:
        k = f"b{B}"
        r = grand[k]
        q = 1.0 / (1.0 - r)
        c = cost[k]
        if prev_q is None:
            row_ratio = ""
        else:
            dq = q - prev_q
            dc = c - prev_c
            row_ratio = f"{(dq / dc):>9.4f}" if dc > 0 else f"{'-':>9}"
        prev_q, prev_c = q, c
        print(f"{B:>3}  ${c:>6.2f}  {r:>+9.4f}  {q:>9.4f}  {row_ratio}")
        rows.append((B, c, r, q))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=str, default=DEFAULT_INPUT,
                   help=f"JSON cache from batch_size_rho_curve.py "
                        f"(default: {DEFAULT_INPUT}).")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                   help=f"Output PNG path (default: {DEFAULT_OUTPUT}).")
    p.add_argument("--ensembles", type=str, default=None,
                   help="Side-car JSON with ensemble ρ values produced by "
                        "judge_ensemble_rho_curve.py.  Default: "
                        "``<input_stem>_ensembles<input_suffix>`` next to "
                        "--input; if absent, the plot omits ensemble "
                        "extras (or honours legacy inline ones).")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to react to drift in the batch-size cache's "
                        "recorded provenance: strict / warn (default) / "
                        "rebuild / off.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Single validation pass; both downstream consumers reuse `raw`.
    # plot_quality_vs_cost will register its own inputs internally
    # via load_and_register/current_file_input, so we don't pass an
    # accumulator here -- main only needs the dicts.
    raw = _load_batch_size_json(inp, policy=args.cache_policy)
    # Resolve and (if possible) pre-load the side-car ensembles JSON.
    ens_path = Path(args.ensembles) if args.ensembles else _default_ensembles_path(inp)
    ensembles_dict: dict | None = None
    ensembles_path: Path | None = None
    if ens_path.exists():
        # Load via load_and_register too, but discard its InputSpec
        # (plot_quality_vs_cost re-registers under the canonical
        # ``ensembles_json`` dep_key from the caller-pre-loaded path).
        ensembles_dict, _spec, _check = load_and_register(
            ens_path, dep_key="ensembles_json",
            inputs=None, policy=args.cache_policy,
        )
        ensembles_path = ens_path
    plot_quality_vs_cost(input_path=inp, output_path=out, raw=raw,
                         ensembles=ensembles_dict,
                         ensembles_path=ensembles_path)
    print(f"Wrote {out}")
    print_marginal_table(inp, raw=raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
