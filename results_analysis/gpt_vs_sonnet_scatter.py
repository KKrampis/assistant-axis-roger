#!/usr/bin/env python3
"""GPT-vs-Sonnet judge agreement: per-entity score scatter plots.

For each axis pair, we have desc+inst scores from two providers
(GPT-4.1-mini and Claude Sonnet 4) for the same set of entities.  This
script computes the per-entity ``both_mean = (descriptions + instructions) / 2``
score for each provider, then visualises the agreement two ways:

1. **Pooled scatter** -- one figure, all axes' points overlaid,
   colored by axis, with the overall Spearman ρ across the
   ``(entity, axis)`` pool.  Quick "do GPT and Sonnet agree across
   axes?" answer.

2. **Per-axis grid** -- one panel per axis with its own ρ.  Lets you
   spot axes where the two providers diverge.

A small Gaussian jitter (sigma=0.08) is added to both axes for visual
separation, since judge scores are integer-valued in [-3, 3] and many
points overlap exactly.

Inputs (from ``--experiment_dir``)
----------------------------------

The standard per-axis directory tree produced by
:mod:`results_analysis.axis_judge_correlation`::

    <experiment_dir>/
      pair_list_di.json   # or any pair list passed via --pairs
      <pos>_vs_<neg>/
        gpt/scores_descriptions.json
        gpt/scores_instructions.json
        sonnet/scores_descriptions.json
        sonnet/scores_instructions.json

Outputs (to ``--experiment_dir``)
---------------------------------

- ``gpt_vs_sonnet_scatter_pooled.png`` -- pooled scatter.
- ``gpt_vs_sonnet_scatter_grid.png``   -- N-panel grid.
- ``gpt_vs_sonnet_rhos.json``           -- per-axis Spearman ρ + pooled ρ.

Examples
--------

::

    # Default: 33 axes (every axis with desc+inst from both providers)
    uv run python results_analysis/gpt_vs_sonnet_scatter.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from assistant_axis import cohort_from_pairs, json_metadata, png_metadata
from assistant_axis.judge_loaders import migrate_v1_static_scores
from assistant_axis.provenance import InputSpec, load_and_register

DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
    "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
)


def _load_both_mean(
    axis_dir: Path, provider: str,
    *,
    inputs: list[InputSpec] | None = None,
    axis_id: str | None = None,
    kinds_for_name: dict[str, set] | None = None,
) -> dict[str, float]:
    """Load ``(descriptions + instructions) / 2`` scores for one provider.

    Threads ``inputs`` through ``load_and_register`` so each cache
    consumed is recorded as a dependency in lockstep with the read.

    When ``kinds_for_name`` is provided, v1 (bare-name) score dicts
    are migrated up to v2 (entity_id) keys in-memory.  This is needed
    when comparing a v2 judge cache (GPT post-Phase-5b) against a v1
    judge cache (deferred Sonnet rejudges); without migration the
    bare-name keys silently fail to intersect with ``name|R`` / ``name|T``
    keys, dropping the 12 collision-touching axes.  Migration is a
    no-op on already-v2 input, so applying it uniformly is safe.
    """
    label = axis_id or axis_dir.name
    d, _, _ = load_and_register(
        axis_dir / provider / "scores_descriptions.json",
        dep_key=f"judge_{label}_descriptions_{provider}",
        inputs=inputs, policy="warn",
    )
    i, _, _ = load_and_register(
        axis_dir / provider / "scores_instructions.json",
        dep_key=f"judge_{label}_instructions_{provider}",
        inputs=inputs, policy="warn",
    )
    if kinds_for_name is not None:
        d = migrate_v1_static_scores(d, kinds_for_name)
        i = migrate_v1_static_scores(i, kinds_for_name)
    common = sorted(set(d) & set(i))
    return {n: (d[n] + i[n]) / 2.0 for n in common}


def _layout_for(n_axes: int, n_cols: int | None = None) -> tuple[int, int]:
    """Choose ``(n_rows, n_cols)`` for an N-panel grid.

    By default we pick the near-square layout ``ceil(sqrt(N))`` columns,
    which gives roughly equal aspect ratios across plot sizes (e.g. 6x6
    for N=33, 4x3 for N=12, 3x3 for N=7).  Pass an explicit ``n_cols``
    to force a specific layout.
    """
    if n_cols is None:
        from math import ceil, sqrt
        n_cols = max(1, ceil(sqrt(n_axes)))
    n_rows = (n_axes + n_cols - 1) // n_cols
    return n_rows, n_cols


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR),
                   help=f"Directory holding the pair list and per-axis "
                        f"score subdirs (default: {DEFAULT_EXPERIMENT_DIR}).")
    p.add_argument("--pairs", default="pair_list_di.json",
                   help="Pair-list JSON filename (default: pair_list_di.json "
                        "-- every axis with desc+inst from both GPT and "
                        "Sonnet; the K-sweep tools default to "
                        "pair_list_responses.json since they additionally "
                        "require GPT response scores).")
    p.add_argument("--pooled", default=None,
                   help="Pooled-scatter PNG filename (default: "
                        "gpt_vs_sonnet_scatter_pooled_<cohort>.png).")
    p.add_argument("--grid", default=None,
                   help="Per-axis grid PNG filename (default: "
                        "gpt_vs_sonnet_scatter_grid_<cohort>.png).")
    p.add_argument("--rhos_json", default=None,
                   help="Output ρ-summary JSON filename (default: "
                        "gpt_vs_sonnet_rhos_<cohort>.json).")
    p.add_argument("--jitter_sigma", type=float, default=0.08,
                   help="Stddev of Gaussian jitter added to both axes for "
                        "visual separation of integer-valued points "
                        "(default: 0.08).")
    p.add_argument("--grid_cols", type=int, default=None,
                   help="Number of columns in the per-axis grid plot.  "
                        "Default: ceil(sqrt(N)) -- e.g. 6x6 for N=33, "
                        "4x3 for N=12, 3x3 for N=7.  Pass --grid_cols 4 "
                        "to force the historical 4-column layout.")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                   help="Corpus root used to build the kinds_for_name map "
                        "for v1->v2 cache key migration.  Only needed so "
                        "Sonnet v1 (bare-name) caches can be re-keyed to "
                        "v2 (entity_id) form before intersecting with GPT "
                        "v2 caches; otherwise the 12 collision-touching "
                        "axes are silently dropped.  Default: "
                        f"{DEFAULT_DATA_DIR}.")
    args = p.parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    cohort = cohort_from_pairs(args.pairs)
    if args.pooled is None:
        args.pooled = f"gpt_vs_sonnet_scatter_pooled_{cohort}.png"
    if args.grid is None:
        args.grid = f"gpt_vs_sonnet_scatter_grid_{cohort}.png"
    if args.rhos_json is None:
        args.rhos_json = f"gpt_vs_sonnet_rhos_{cohort}.json"

    # ``inputs`` accumulator -- every cache read goes through
    # load_and_register so the read AND the InputSpec record are
    # built together (see AGENT_NOTES.md "Reader+registrar pattern").
    inputs: list[InputSpec] = []
    pairs, _spec, _check = load_and_register(
        experiment_dir / args.pairs,
        dep_key="pairs_json",
        inputs=inputs, policy="warn",
    )
    print(f"Loaded {len(pairs)} axis pairs from {args.pairs}")

    # Build kinds_for_name from the corpus so legacy v1 (bare-name) caches
    # can be lifted to v2 (entity_id) keys in-memory before intersection.
    # The set-valued return matters for the 9 collision names that
    # appear in both kinds; migrate_v1_static_scores drops those (the
    # bare key is irrecoverably ambiguous) and we expect that loss.
    kinds_for_name: dict[str, set] = {}
    for et in ("traits", "roles"):
        vec_dir = data_dir / et / "vectors"
        if not vec_dir.is_dir():
            continue
        for fp in sorted(vec_dir.glob("*.pt")):
            if fp.stem == "default":
                continue
            kinds_for_name.setdefault(fp.stem, set()).add(et)

    # Per-axis (gpt_score_list, sonnet_score_list, common_names) and per-axis Spearman ρ.
    per_axis: dict[tuple[str, str], dict] = {}
    all_gpt: list[float] = []
    all_son: list[float] = []
    all_axis: list[tuple[str, str]] = []
    for pair in pairs:
        pos, neg = pair["pos"], pair["neg"]
        axis_id = f"{pos}_vs_{neg}"
        axis_dir = experiment_dir / axis_id
        gpt = _load_both_mean(axis_dir, "gpt", inputs=inputs, axis_id=axis_id,
                              kinds_for_name=kinds_for_name)
        son = _load_both_mean(axis_dir, "sonnet", inputs=inputs, axis_id=axis_id,
                              kinds_for_name=kinds_for_name)
        common = sorted(set(gpt) & set(son))
        if not common:
            print(f"  [skip] {pos}/{neg}: no shared entities between gpt and sonnet")
            continue
        g = [gpt[n] for n in common]
        s = [son[n] for n in common]
        rho = float(spearmanr(g, s).correlation)
        per_axis[(pos, neg)] = {"g": g, "s": s, "names": common, "rho": rho}
        all_gpt.extend(g)
        all_son.extend(s)
        all_axis.extend([(pos, neg)] * len(common))
        print(f"  {pos:18s} vs {neg:18s}  ρ = {rho:+.3f}  (n={len(common)})")

    pooled_rho = float(spearmanr(all_gpt, all_son).correlation)
    print(f"\nPooled (entity, axis) Spearman ρ = {pooled_rho:+.3f}  (n={len(all_gpt)})")

    # ---------- Provenance inputs (shared by JSON + both PNGs) ----------
    # ``inputs`` was already populated above by load_and_register at
    # every cache-read site (pairs_json + per-axis × per-judge ×
    # per-mode score caches).  Pre-retrofit this section duplicated
    # the registration in a separate post-load loop -- which not only
    # could fall out of sync with the actual reads but also recorded
    # deps for axes that the read pass had already skipped (no shared
    # entities).  Dropping that loop in favour of read-site
    # registration makes "what we recorded" exactly equal "what we
    # consumed".  See Phase 6c rationale in AGENT_NOTES.md.

    # ---------- Save JSON summary ----------
    rhos_out = {
        "pooled_rho": pooled_rho,
        "n_pooled": len(all_gpt),
        "per_axis": [
            {"pos": pos, "neg": neg, "rho": d["rho"], "n": len(d["names"])}
            for (pos, neg), d in per_axis.items()
        ],
    }
    rhos_path = experiment_dir / args.rhos_json
    envelope = json_metadata(
        rhos_out,
        inputs=inputs,
        title=f"gpt_vs_sonnet_scatter pairs={args.pairs}",
    )
    json.dump(envelope, open(rhos_path, "w"), indent=2)
    print(f"Wrote {rhos_path}")

    # ---------- Colors per axis ----------
    axes_list = list(per_axis.keys())
    cmap = plt.cm.tab10 if len(axes_list) <= 10 else plt.cm.tab20
    colors = cmap(np.linspace(0, 1, max(cmap.N, len(axes_list))))[:len(axes_list)]
    color_map = {k: c for k, c in zip(axes_list, colors)}

    # Deterministic jitter (sigma small enough to keep ±0.5 of integer grid).
    rng_x = np.random.RandomState(0)
    rng_y = np.random.RandomState(1)
    jitter = {
        k: (rng_x.normal(0, args.jitter_sigma, len(d["g"])),
            rng_y.normal(0, args.jitter_sigma, len(d["s"])))
        for k, d in per_axis.items()
    }

    # ============================================================
    # Plot 1: pooled scatter + per-axis ρ ranked bar chart (1x2)
    # ============================================================
    # Figure height auto-scales with axis count so 33 horizontal bars
    # stay readable; width fixed so combination is always wider than tall.
    fig_h = max(6.5, 0.22 * len(axes_list) + 1.0)
    fig, (ax, ax_h) = plt.subplots(
        1, 2, figsize=(14, fig_h),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )
    # -- Left: pooled scatter --
    # s=2 (point area in pt²) → marker radius half of the previous s=8.
    for k in axes_list:
        d = per_axis[k]
        jx, jy = jitter[k]
        ax.scatter(np.array(d["g"]) + jx, np.array(d["s"]) + jy,
                   s=2, alpha=0.45, color=color_map[k], edgecolor="none")
    ax.plot([-3.5, 3.5], [-3.5, 3.5], "k--", alpha=0.3, lw=0.8)
    ax.axvline(0, color="grey", lw=0.3)
    ax.axhline(0, color="grey", lw=0.3)
    ax.set_xlabel("GPT-4.1-mini both_mean score (desc+inst)")
    ax.set_ylabel("Sonnet 4 both_mean score (desc+inst)")
    ax.set_title(f"Pooled scatter (jitter added) -- "
                 f"overall Spearman ρ = {pooled_rho:.3f}, n={len(all_gpt)}",
                 fontsize=10)
    ax.text(0.02, 0.98,
            f"{len(per_axis)} axes (colors arbitrary;\n"
            f"see per-axis grid plot for labels)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.85))
    ax.grid(alpha=0.3)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")

    # -- Right: per-axis ρ as horizontal bars, sorted descending --
    rhos = np.array([per_axis[k]["rho"] for k in axes_list])
    order = np.argsort(rhos)[::-1]   # descending
    sorted_keys = [axes_list[i] for i in order]
    sorted_rhos = rhos[order]
    y_pos = np.arange(len(sorted_keys))
    bar_colors = [color_map[k] for k in sorted_keys]
    ax_h.barh(y_pos, sorted_rhos, color=bar_colors,
              edgecolor="black", linewidth=0.5)
    for y, r in zip(y_pos, sorted_rhos):
        ax_h.text(r + 0.005, y, f"{r:.3f}",
                  va="center", ha="left", fontsize=7)
    labels = [f"{k[0]} / {k[1]}" for k in sorted_keys]
    ax_h.set_yticks(y_pos)
    ax_h.set_yticklabels(labels, fontsize=8)
    ax_h.invert_yaxis()  # highest ρ at top
    ax_h.set_xlim(0.0, 1.0)
    ax_h.axvline(float(np.median(rhos)), color="red", linestyle="--",
                 lw=1.2, alpha=0.7,
                 label=f"median = {np.median(rhos):.3f}")
    ax_h.axvline(float(rhos.mean()), color="darkorange", linestyle=":",
                 lw=1.2, alpha=0.7,
                 label=f"mean   = {rhos.mean():.3f}")
    ax_h.set_xlabel(
        "Per-axis Spearman ρ (GPT-4.1-mini vs Sonnet 4, both_mean)")
    ax_h.set_title(
        f"Per-axis ρ ranked (n={len(rhos)}; "
        f"range [{rhos.min():.3f}, {rhos.max():.3f}])",
        fontsize=10)
    ax_h.grid(axis="x", alpha=0.3)
    ax_h.legend(loc="lower right", fontsize=8)

    pooled_title = (f"GPT-4.1-mini vs Sonnet 4 per-entity scores, "
                    f"{len(per_axis)} axes pooled")
    fig.suptitle(pooled_title, fontsize=14, fontweight="bold")
    # Reserve a top band for the suptitle, scaled to figure height.
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig_h))
    pooled_path = experiment_dir / args.pooled
    plt.savefig(pooled_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=pooled_title, inputs=inputs))
    plt.close(fig)
    print(f"Wrote {pooled_path}")

    # ============================================================
    # Plot 2: per-axis grid
    # ============================================================
    n_rows, n_cols = _layout_for(len(axes_list), n_cols=args.grid_cols)
    fig, axes_p = plt.subplots(n_rows, n_cols,
                               figsize=(4 * n_cols, 4 * n_rows),
                               sharex=True, sharey=True, squeeze=False)
    for i, k in enumerate(axes_list):
        ax = axes_p[i // n_cols, i % n_cols]
        d = per_axis[k]
        jx, jy = jitter[k]
        ax.scatter(np.array(d["g"]) + jx, np.array(d["s"]) + jy,
                   s=10, alpha=0.45, color=color_map[k], edgecolor="none")
        ax.plot([-3.5, 3.5], [-3.5, 3.5], "k--", alpha=0.3, lw=0.8)
        ax.set_title(f'{k[0]} vs {k[1]}\n'
                     f'ρ = {d["rho"]:.3f}, n={len(d["names"])}',
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
    for j in range(len(axes_list), n_rows * n_cols):
        axes_p[j // n_cols, j % n_cols].axis("off")
    fig.supxlabel("GPT-4.1-mini both_mean score", fontsize=11)
    fig.supylabel("Sonnet 4 both_mean score", fontsize=11)
    grid_title = (f"GPT-4.1-mini vs Sonnet 4 per-entity scores, "
                  f"by axis ({len(axes_list)} axes; jitter added)")
    fig.suptitle(grid_title, fontsize=14, fontweight="bold")
    # Leave a band at the top for the suptitle (otherwise it overlaps the
    # top row of per-panel titles when n_rows is large).
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.4 / max(n_rows, 1)))
    grid_path = experiment_dir / args.grid
    plt.savefig(grid_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=grid_title, inputs=inputs))
    plt.close(fig)
    print(f"Wrote {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
