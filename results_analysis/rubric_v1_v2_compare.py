#!/usr/bin/env python3
"""v1 vs v2 response-rubric ρ comparison plot.

The May 2026 rubric v1→v2 change anonymises the entity name in
``RUBRIC_RESPONSE_BATCH`` (see the version-history block at the top
of :mod:`results_analysis.axis_judge_correlation`).  Only response-
mode judging is affected; desc + inst rubrics are unchanged.

This script quantifies the rubric change by computing four per-
entity Spearman ρ values for each of the 12 response-mode axes:

* ``response_v1`` -- ρ between the response-mode ensemble
  ``0.6 * GPT_b10 + 0.4 * Haiku_q9`` (using the v1 snapshot at
  ``scores_responses__rubric_v1.json``) and the raw activation
  projection at (slot, layer).
* ``response_v2`` -- same ensemble but reading the canonical
  ``scores_responses.json`` (the freshly-rejudged v2 cache).
* ``combined_v1`` -- ρ between
  ``DEFAULT_RESPONSE_DI_WEIGHT · response_v1
  + (1 − DEFAULT_RESPONSE_DI_WEIGHT) · desc_inst`` and the
  projection.
* ``combined_v2`` -- same as ``combined_v1`` but with
  ``response_v2``.

Mixing weights come from
:mod:`assistant_axis.judge_score_combine` (single source of truth;
see that module's docstring for empirical derivations).  The
desc+inst score uses the project's canonical 4-way combiner
``combine_desc_inst_two_judges(GPT_d, GPT_i, Sonnet_d, Sonnet_i,
weights=DEFAULT_DI_WEIGHTS)`` -- identical between the v1 and v2
columns because the desc+inst rubric did not change.

The render is a two-panel grouped bar chart (top: response-only,
bottom: combined) with shared y-axis so the two panels are
comparable at a glance.  A 13th "mean" group at the right of each
panel shows the arithmetic mean of the 12 per-axis ρ values for
each of the v1/v2 series.

Outputs (default into ``--experiment_dir``):

* ``rubric_v1_v2_compare_slot{N}.png``
* ``rubric_v1_v2_compare_slot{N}.json`` (per-axis + mean ρ values
  in both panels, plus full provenance envelope)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from assistant_axis import (
    entity_id, json_metadata, png_metadata, suptitle_with_specs,
)
from assistant_axis.judge_score_combine import (
    declare_constants_dependency,
    DEFAULT_DI_WEIGHTS,
    DEFAULT_GPT_HAIKU_Q9_WEIGHT,
    DEFAULT_RESPONSE_DI_WEIGHT,
    combine_desc_inst_two_judges,
)
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
    current_file_input,
    load_and_register,
)
from results_analysis.axis_judge_correlation import _load_vector_file

_SCRIPT_PATH = Path(__file__).resolve()


# The 12 response-mode axes for which we have v2 data (B=10, GPT
# full + Haiku q9).  Mirrors ``ALL_RESPONSE_AXES`` in
# :mod:`results_analysis.judge_ensemble_rho_curve`.
DEFAULT_AXES: list[tuple[str, str, str]] = [
    ("truthful_vs_deceitful",          "truthful",        "deceitful"),
    ("progressive_vs_conservative",    "progressive",     "conservative"),
    ("improvisational_vs_methodical",  "improvisational", "methodical"),
    ("concise_vs_verbose",             "concise",         "verbose"),
    ("ecocentric_vs_anthropocentric",  "ecocentric",      "anthropocentric"),
    ("egalitarian_vs_elitist",         "egalitarian",     "elitist"),
    ("guileless_vs_scheming",          "guileless",       "scheming"),
    ("harmless_vs_harmful",            "harmless",        "harmful"),
    ("honest_vs_dishonest",            "honest",          "dishonest"),
    ("helpful_vs_unhelpful",           "helpful",         "unhelpful"),
    ("relativist_vs_absolutist",       "relativist",      "absolutist"),
    ("systems_thinker_vs_analytical",  "systems_thinker", "analytical"),
]

DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
    "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
)
# Operating-point cell: matches the 0.6 GPT + 0.4 Haiku-q9 default
# established in gpt_anthropic_response_weight_sweep / the Pareto
# plot at slot 6 (</think>) layer 25.
DEFAULT_SLOT, DEFAULT_LAYER = 6, 25

# Mixing ratios are imported from assistant_axis.judge_score_combine
# (single source of truth; see that module's docstring for the
# empirical derivation of each value).  Keep them as named locals
# here so the JSON / PNG provenance records reflect the actual
# numbers used at run-time even if the central constants are
# re-tuned later.
GPT_RESPONSE_WEIGHT = DEFAULT_GPT_HAIKU_Q9_WEIGHT
ANTH_RESPONSE_WEIGHT = 1.0 - GPT_RESPONSE_WEIGHT
RESPONSE_IN_COMBINED_WEIGHT = DEFAULT_RESPONSE_DI_WEIGHT
DI_IN_COMBINED_WEIGHT = 1.0 - RESPONSE_IN_COMBINED_WEIGHT

GPT_DIR_TEMPLATE = "gpt_responses_{side}_b10"
HAIKU_DIR_TEMPLATE = "haiku_responses_{side}_b10_q9"

# v1 snapshots live alongside the canonical caches with a __rubric_v1
# suffix.  scripts/rejudge_after_rubric_v2.sh writes them; see
# AGENT_NOTES.md "Snapshot-before-invalidate principle".
V1_SCORES_FILENAME = "scores_responses__rubric_v1.json"
V2_SCORES_FILENAME = "scores_responses.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _v(path: Path, slot: int, layer: int) -> torch.Tensor:
    return _load_vector_file(path).float()[slot, layer]


def _axis_unit(data_dir: Path, pos: str, neg: str,
               slot: int, layer: int) -> torch.Tensor:
    """Unit-norm direction in raw activation space at (slot, layer)."""
    p = _v(data_dir / "traits" / "vectors" / f"{pos}.pt", slot, layer)
    n = _v(data_dir / "traits" / "vectors" / f"{neg}.pt", slot, layer)
    d = p - n
    nrm = torch.linalg.vector_norm(d)
    return d / nrm if nrm > 0 else d


def _load_response_scores(
    experiment_dir: Path, axis: str, dir_template: str,
    *,
    scores_filename: str,
    judge_label: str,
    inputs: list[InputSpec],
) -> dict[str, float]:
    """Per-entity ``mean_score`` summed across the roles+traits sides
    for a single (judge, rubric-version) cell.  Mirrors the loader in
    :mod:`results_analysis.gpt_anthropic_response_weight_sweep`.
    """
    out: dict[str, float] = {}
    for side in ("roles", "traits"):
        sub = dir_template.format(side=side)
        path = experiment_dir / axis / sub / scores_filename
        if not path.exists():
            continue
        scores, _spec, _check = load_and_register(
            path,
            dep_key=f"resp_{judge_label}_{axis}_{side}",
            extras={"axis": axis, "side": side, "judge": judge_label,
                    "scores_filename": scores_filename},
            policy="warn",
            inputs=inputs,
        )
        for name, info in scores.items():
            ms = info.get("mean_score") if isinstance(info, dict) else None
            if ms is not None:
                out[entity_id(name, side)] = float(ms)
    return out


def _blend(a: dict[str, float], b: dict[str, float],
           w_a: float) -> dict[str, float]:
    """``w_a * a[n] + (1 - w_a) * b[n]`` over the intersection of keys."""
    common = set(a) & set(b)
    return {n: w_a * a[n] + (1.0 - w_a) * b[n] for n in common}


def _rho(score_by_name: dict[str, float],
         proj_by_name: dict[str, float]) -> tuple[float, int]:
    """Spearman ρ over the intersection of the two dicts."""
    common = sorted(set(score_by_name) & set(proj_by_name))
    if len(common) < 5:
        return float("nan"), len(common)
    x = np.array([score_by_name[n] for n in common])
    y = np.array([proj_by_name[n] for n in common])
    return float(spearmanr(x, y).correlation), len(common)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _short_axis_label(axis_name: str) -> str:
    """Compact axis label for x-tick rendering ("eco/anthro" etc.).

    Strategy: split on ``_vs_``, take the first 4 chars of each
    half (or the full word if shorter) and join with ``/``.  Keeps
    labels well under 12 chars so the 13-bin axis fits without
    needing a sidebar legend.
    """
    pos, neg = axis_name.split("_vs_", 1)
    short = lambda s: s.replace("_", " ")[:4]
    return f"{short(pos)}/{short(neg)}"


def _make_plot(
    axis_names: list[str],
    rho_table: dict[str, list[float]],
    out_path: Path,
    *,
    slot: int,
    layer: int,
    n_axes_for_mean: int,
    inputs: list[InputSpec],
) -> None:
    """Single-panel grouped-bar render: 4 bars per axis group.

    Within each group, left-to-right:
      1. response v1   (light blue)
      2. response v2   (dark blue)
      3. combined v1   (light orange)
      4. combined v2   (dark orange)

    A small gap separates the response pair from the combined pair
    so the two "categories" stay visually distinct while keeping all
    four versions in one eye-scan-able cluster.

    ``rho_table`` keys: ``response_v1``, ``response_v2``,
    ``combined_v1``, ``combined_v2``.  Each value is a list of length
    ``len(axis_names) + 1`` (per-axis ρ followed by the mean).
    """
    labels = [_short_axis_label(a) for a in axis_names] + ["mean"]
    n = len(labels)
    assert all(len(rho_table[k]) == n for k in rho_table), \
        "rho_table column lengths inconsistent"

    # X positions: 12 axes packed normally then a small gap before
    # the mean column so the eye can find the aggregate quickly.
    x = np.arange(n, dtype=float)
    x[-1] += 0.6                          # bigger gap before mean

    # Bar geometry inside each group: 4 bars, with a small extra
    # gap between the response pair and the combined pair.
    bar_w = 0.18
    mid_gap = 0.06
    half_pair = bar_w + mid_gap / 2.0     # half-width of one (v1+v2) pair
    offsets = {
        "response_v1": -half_pair - bar_w / 2.0,
        "response_v2": -half_pair + bar_w / 2.0,
        "combined_v1": +half_pair - bar_w / 2.0,
        "combined_v2": +half_pair + bar_w / 2.0,
    }

    # Colour scheme: response = blue family, combined = orange family;
    # within each family v1 = light, v2 = dark.  Keeps "rubric version"
    # and "score type" both legible at a glance without hatching.
    COLORS = {
        "response_v1": "#a5cae3",   # light blue
        "response_v2": "#1f77b4",   # dark blue
        "combined_v1": "#f8c69a",   # light orange
        "combined_v2": "#d96b1f",   # dark orange
    }
    EDGES = {
        "response_v1": "#5e8da7",
        "response_v2": "#0f4c7c",
        "combined_v1": "#a8763e",
        "combined_v2": "#8a4413",
    }
    LABELS = {
        "response_v1": "response v1",
        "response_v2": "response v2",
        "combined_v1": "combined v1",
        "combined_v2": "combined v2",
    }
    MEAN_LINE = "#222222"

    fig, ax = plt.subplots(figsize=(14.5, 6.8))

    bars: dict[str, list] = {}
    series_arrays: dict[str, np.ndarray] = {}
    for key in ("response_v1", "response_v2", "combined_v1", "combined_v2"):
        v = np.array(rho_table[key])
        series_arrays[key] = v
        bars[key] = ax.bar(
            x + offsets[key], v, bar_w,
            label=LABELS[key],
            color=COLORS[key], edgecolor=EDGES[key], linewidth=0.6,
            zorder=3,
        )

    # Vertical separator just before the mean column.
    ax.axvline(x[-1] - 0.55, color="#999999", lw=0.7, ls="--",
               alpha=0.7, zorder=2)

    ax.set_ylabel("Spearman ρ\n(score vs activation projection)")
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Per-axis Δ annotations: one above each (v1, v2) pair.  Group
    # by which pair the delta refers to so the "response Δ" and
    # "combined Δ" labels sit directly above their respective
    # paired bars.
    for i, x_i in enumerate(x):
        for k_v1, k_v2, anchor_offset, color in (
            ("response_v1", "response_v2", -half_pair, EDGES["response_v2"]),
            ("combined_v1", "combined_v2", +half_pair, EDGES["combined_v2"]),
        ):
            y1 = series_arrays[k_v1][i]
            y2 = series_arrays[k_v2][i]
            if np.isnan(y1) or np.isnan(y2):
                continue
            d = y2 - y1
            top = max(y1, y2)
            ax.annotate(
                f"{d:+.3f}",
                xy=(x_i + anchor_offset, top),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=6.8,
                color=color,
                fontweight="bold" if i == n - 1 else "normal",
            )

    # Cross-axis mean reference lines (one per series, dotted).
    # The explicit "mean" group at the right shows the same numbers,
    # but the dotted lines give the eye an immediate per-axis
    # comparison.  Top-left text box repeats the four μ values for
    # quick numerical reference.
    means = {k: float(np.nanmean(series_arrays[k][:-1]))
             for k in series_arrays}
    for k, mu in means.items():
        ax.axhline(mu, color=EDGES[k], lw=0.7, ls=":", alpha=0.55,
                   zorder=1)

    summary = (
        "cross-axis means:\n"
        f"  resp:      v1={means['response_v1']:.3f}   "
        f"v2={means['response_v2']:.3f}   "
        f"Δμ={means['response_v2']-means['response_v1']:+.3f}\n"
        f"  combined:  v1={means['combined_v1']:.3f}   "
        f"v2={means['combined_v2']:.3f}   "
        f"Δμ={means['combined_v2']-means['combined_v1']:+.3f}"
    )
    ax.text(0.011, 0.015, summary,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color="#222222", family="monospace",
            bbox=dict(boxstyle="round,pad=0.30",
                      fc="white", ec="#cccccc", lw=0.5, alpha=0.92))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    for tick_label, lab in zip(ax.get_xticklabels(), labels):
        if lab == "mean":
            tick_label.set_fontweight("bold")
            tick_label.set_color(MEAN_LINE)

    ax.legend(
        loc="lower right", fontsize=9, framealpha=0.95,
        ncol=2, handlelength=1.6, columnspacing=1.0,
    )

    title = (f"Response-mode rubric v1 vs v2: per-axis ρ, "
             f"B=10, slot {slot} layer {layer}")
    spec_lines = [
        (f"Spearman ρ between predicted per-entity score and raw "
         f"activation projection at (slot {slot}, layer {layer}, K=0), "
         f"for {n_axes_for_mean} response-mode axes plus their mean."),
        (f"Response = {GPT_RESPONSE_WEIGHT:g}·GPT_b10 + "
         f"{ANTH_RESPONSE_WEIGHT:g}·Haiku_q9.   "
         f"Combined = {RESPONSE_IN_COMBINED_WEIGHT:g}·response + "
         f"{DI_IN_COMBINED_WEIGHT:g}·desc+inst (GPT+Sonnet 4-way, "
         "inst-tiebreak)."),
        ("v1 = pre-anonymisation rubric "
         "(read from scores_responses__rubric_v1.json snapshots);   "
         "v2 = anonymised rubric "
         "(canonical scores_responses.json, rejudged 2026-05-09)."),
        ("Δ above each (v1, v2) bar pair = ρ(v2) − ρ(v1).   "
         "Dotted horizontals: cross-axis mean ρ for each of the "
         "four series."),
    ]
    _, top_rect = suptitle_with_specs(
        fig, title, spec_lines, line_height=0.030, spec_fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, top_rect))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path, dpi=150, bbox_inches="tight",
        metadata=png_metadata(title=title, inputs=inputs),
    )
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR))
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--plot", default=None,
                   help="Output PNG path "
                        "(default: rubric_v1_v2_compare_slot{N}.png inside "
                        "--experiment_dir).")
    p.add_argument("--rhos_json", default=None,
                   help="Output JSON path "
                        "(default: rubric_v1_v2_compare_slot{N}.json inside "
                        "--experiment_dir).")
    args = p.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    slot, layer = int(args.slot), int(args.layer)
    out_stem = f"rubric_v1_v2_compare_slot{slot}"
    plot_path = Path(args.plot) if args.plot else (
        experiment_dir / f"{out_stem}.png")
    json_path = Path(args.rhos_json) if args.rhos_json else (
        experiment_dir / f"{out_stem}.json")

    inputs: list[InputSpec] = [
        current_file_input(
            dep_key="producer_script",
            path=_SCRIPT_PATH,
            extras={
                "slot": str(slot), "layer": str(layer),
                "gpt_response_weight": str(GPT_RESPONSE_WEIGHT),
                "response_in_combined_weight":
                    str(RESPONSE_IN_COMBINED_WEIGHT),
            },
        ),
    ]
    # Track the project-wide constants this script imports so a future
    # retune of DEFAULT_GPT_HAIKU_Q9_WEIGHT (or any sibling default in
    # judge_score_combine.py) flags this cache as stale via
    # audit_caches.py.  See declare_constants_dependency docstring.
    declare_constants_dependency(inputs)
    inputs.extend([
        current_data_subtree_input(
            data_dir=data_dir, subtree_rel="traits/vectors",
            dep_key="traits_vectors"),
        current_data_subtree_input(
            data_dir=data_dir, subtree_rel="roles/vectors",
            dep_key="roles_vectors"),
    ])

    # Pre-cache standalone entity vectors at (slot, layer),
    # default-centered.  Mirrors gpt_anthropic_response_weight_sweep.
    default_v = _v(data_dir / "traits" / "vectors" / "default.pt", slot, layer)
    entity_vecs: dict[str, np.ndarray] = {}
    for et in ("traits", "roles"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            try:
                v = _load_vector_file(fp).float()[slot, layer]
                entity_vecs[entity_id(fp.stem, et)] = (v - default_v).numpy()
            except Exception:  # pragma: no cover -- skip unreadable .pt
                continue

    print(f"Rubric v1 vs v2 ρ comparison @ slot {slot} layer {layer}")
    print(f"  axes: {[a[0] for a in DEFAULT_AXES]}")
    print(f"  response ensemble: "
          f"{GPT_RESPONSE_WEIGHT:.2f}·GPT_b10  +  "
          f"{ANTH_RESPONSE_WEIGHT:.2f}·Haiku_q9")
    print(f"  combined mix: "
          f"{RESPONSE_IN_COMBINED_WEIGHT:.2f}·response  +  "
          f"{DI_IN_COMBINED_WEIGHT:.2f}·desc+inst (4-way)\n")

    per_axis: list[dict] = []
    rho_table: dict[str, list[float]] = {
        "response_v1": [], "response_v2": [],
        "combined_v1": [], "combined_v2": [],
    }
    n_table: dict[str, list[int]] = {k: [] for k in rho_table}

    for axis_name, pos, neg in DEFAULT_AXES:
        axis_dir = experiment_dir / axis_name
        # Response scores (4 dicts: 2 versions × 2 judges).
        gpt_v1 = _load_response_scores(
            experiment_dir, axis_name, GPT_DIR_TEMPLATE,
            scores_filename=V1_SCORES_FILENAME, judge_label="gpt_b10_v1",
            inputs=inputs)
        gpt_v2 = _load_response_scores(
            experiment_dir, axis_name, GPT_DIR_TEMPLATE,
            scores_filename=V2_SCORES_FILENAME, judge_label="gpt_b10_v2",
            inputs=inputs)
        haiku_v1 = _load_response_scores(
            experiment_dir, axis_name, HAIKU_DIR_TEMPLATE,
            scores_filename=V1_SCORES_FILENAME, judge_label="haiku_q9_v1",
            inputs=inputs)
        haiku_v2 = _load_response_scores(
            experiment_dir, axis_name, HAIKU_DIR_TEMPLATE,
            scores_filename=V2_SCORES_FILENAME, judge_label="haiku_q9_v2",
            inputs=inputs)

        # Desc + inst scores (4 flat dicts).  Bare legacy JSONs --
        # load_and_register short-circuits to no-validate on those.
        g_d, _, _ = load_and_register(
            axis_dir / "gpt" / "scores_descriptions.json",
            dep_key=f"di_gpt_d_{axis_name}",
            inputs=inputs, policy="warn")
        g_i, _, _ = load_and_register(
            axis_dir / "gpt" / "scores_instructions.json",
            dep_key=f"di_gpt_i_{axis_name}",
            inputs=inputs, policy="warn")
        s_d, _, _ = load_and_register(
            axis_dir / "sonnet" / "scores_descriptions.json",
            dep_key=f"di_sonnet_d_{axis_name}",
            inputs=inputs, policy="warn")
        s_i, _, _ = load_and_register(
            axis_dir / "sonnet" / "scores_instructions.json",
            dep_key=f"di_sonnet_i_{axis_name}",
            inputs=inputs, policy="warn")

        # Build the five derived per-entity score dicts.
        response_v1 = _blend(gpt_v1, haiku_v1, GPT_RESPONSE_WEIGHT)
        response_v2 = _blend(gpt_v2, haiku_v2, GPT_RESPONSE_WEIGHT)
        di_combined = combine_desc_inst_two_judges(
            g_d, g_i, s_d, s_i, weights=DEFAULT_DI_WEIGHTS)
        combined_v1 = _blend(response_v1, di_combined,
                              RESPONSE_IN_COMBINED_WEIGHT)
        combined_v2 = _blend(response_v2, di_combined,
                              RESPONSE_IN_COMBINED_WEIGHT)

        # Project entities onto the axis unit at (slot, layer).
        a = _axis_unit(data_dir, pos, neg, slot, layer).numpy()
        proj: dict[str, float] = {
            n: float(np.dot(entity_vecs[n], a))
            for n in entity_vecs
        }

        # Spearman ρ for each of the 4 score series vs the
        # projection (intersected with entity_vecs implicitly via
        # ``proj``).
        r_resp_v1, n_resp_v1 = _rho(response_v1, proj)
        r_resp_v2, n_resp_v2 = _rho(response_v2, proj)
        r_comb_v1, n_comb_v1 = _rho(combined_v1, proj)
        r_comb_v2, n_comb_v2 = _rho(combined_v2, proj)

        rho_table["response_v1"].append(r_resp_v1)
        rho_table["response_v2"].append(r_resp_v2)
        rho_table["combined_v1"].append(r_comb_v1)
        rho_table["combined_v2"].append(r_comb_v2)
        n_table["response_v1"].append(n_resp_v1)
        n_table["response_v2"].append(n_resp_v2)
        n_table["combined_v1"].append(n_comb_v1)
        n_table["combined_v2"].append(n_comb_v2)

        per_axis.append({
            "axis_name": axis_name, "pos": pos, "neg": neg,
            "response_v1": {"rho": r_resp_v1, "n": n_resp_v1},
            "response_v2": {"rho": r_resp_v2, "n": n_resp_v2},
            "combined_v1": {"rho": r_comb_v1, "n": n_comb_v1},
            "combined_v2": {"rho": r_comb_v2, "n": n_comb_v2},
        })
        print(f"  {axis_name:<32}  "
              f"resp v1={r_resp_v1:+.4f} v2={r_resp_v2:+.4f} "
              f"(Δ={r_resp_v2 - r_resp_v1:+.4f})   "
              f"comb v1={r_comb_v1:+.4f} v2={r_comb_v2:+.4f} "
              f"(Δ={r_comb_v2 - r_comb_v1:+.4f})  "
              f"[n_resp={n_resp_v2}, n_comb={n_comb_v2}]")

    # Cross-axis means (the 13th column).
    means = {k: float(np.nanmean(v)) for k, v in rho_table.items()}
    for k, m in means.items():
        rho_table[k].append(m)

    print()
    print(f"Cross-{len(per_axis)}-axis means (arithmetic mean of per-axis ρ):")
    print(f"  response: v1={means['response_v1']:+.4f}  "
          f"v2={means['response_v2']:+.4f}  "
          f"Δ={means['response_v2'] - means['response_v1']:+.4f}")
    print(f"  combined: v1={means['combined_v1']:+.4f}  "
          f"v2={means['combined_v2']:+.4f}  "
          f"Δ={means['combined_v2'] - means['combined_v1']:+.4f}")
    resp_delta = means["response_v2"] - means["response_v1"]
    comb_delta = means["combined_v2"] - means["combined_v1"]
    print(f"  response→combined Δρ ratio: "
          f"{comb_delta / (resp_delta + 1e-12):+.3f}  "
          f"(naive linear-in-score expectation = "
          f"{RESPONSE_IN_COMBINED_WEIGHT:.2f}; ρ is rank-based, so the "
          f"signal-mixing in combined mode can amplify or attenuate "
          f"depending on whether response and desc+inst rank entities "
          f"concordantly).")

    # Render plot.
    _make_plot(
        axis_names=[a["axis_name"] for a in per_axis],
        rho_table=rho_table,
        out_path=plot_path,
        slot=slot, layer=layer,
        n_axes_for_mean=len(per_axis),
        inputs=inputs,
    )

    # Write JSON sidecar with the full table.
    payload = {
        "config": {
            "slot": slot, "layer": layer, "K": 0,
            "gpt_response_weight": GPT_RESPONSE_WEIGHT,
            "anthropic_response_weight": ANTH_RESPONSE_WEIGHT,
            "response_in_combined_weight": RESPONSE_IN_COMBINED_WEIGHT,
            "di_in_combined_weight": DI_IN_COMBINED_WEIGHT,
            "di_weights_desc_inst": list(DEFAULT_DI_WEIGHTS),
            "v1_scores_filename": V1_SCORES_FILENAME,
            "v2_scores_filename": V2_SCORES_FILENAME,
        },
        "per_axis": per_axis,
        "means_over_axes": means,
        "n_axes": len(per_axis),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(
        json_metadata(payload, title="rubric_v1_v2_compare", inputs=inputs),
        indent=2,
    ))
    print(f"Wrote {json_path}  ({len(inputs)} inputs recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
