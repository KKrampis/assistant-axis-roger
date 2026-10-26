#!/usr/bin/env python3
"""
Pooled judge-tier cost vs quality plot across multiple axes.

Reads all *_tier_eval/ directories found under roger/axis_judge_experiments/
(or a specified set via --tier_eval_dirs), aggregates per-axis rho and cost
for each judge model, then plots mean quality = 1/(1-rho) vs mean cost-per-axis
with the Pareto frontier highlighted.

Also optionally overlays the two legacy reference models (gpt-4.1-mini and
claude-sonnet-4) whose judge caches predate usage.json tracking: their rhos
come from the gpt/ and sonnet/ subdirs of the original axis directories and
their costs are ESTIMATED from the same token/pricing model as the dry-run.

Usage:
    # After running judge_tier_cost_eval.py on 18+ axes:
    uv run python results_analysis/judge_tier_cost_plot_pooled.py \\
        --experiments_dir roger/axis_judge_experiments \\
        --slot 6 \\
        --output roger/axis_judge_experiments/pareto_pooled_20axes.png

    # Restrict to a specific set of axes:
    uv run python results_analysis/judge_tier_cost_plot_pooled.py \\
        --tier_eval_dirs \\
            roger/axis_judge_experiments/angel_vs_demon_tier_eval \\
            roger/axis_judge_experiments/trustworthy_vs_untrustworthy_tier_eval \\
        --slot 6 \\
        --output /tmp/pareto_subset.png
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from assistant_axis.judge_pricing import (
    GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT, SONNET4_RATE_IN, SONNET4_RATE_OUT,
    price_for_model,
)
from assistant_axis.plot_metadata import png_metadata

PROVIDER_COLOR = {"openai": "#7FB069", "anthropic": "#D9636B"}

EST_IN_PER_ENTITY = 320 + 150
EST_OUT_PER_ENTITY = 80


def provider_of(model: str) -> str:
    return "openai" if model.startswith(("gpt-", "o1-", "o3-", "o4-")) else "anthropic"


def load_rho(correlations_path: Path, slot: int,
             mode: str = "descriptions", metric: str = "raw") -> Optional[Tuple[float, int]]:
    if not correlations_path.exists():
        return None
    with open(correlations_path) as f:
        d = json.load(f)
    r = d.get("result", d)
    try:
        cell = r[mode][str(slot)][metric]
        return float(cell["rho"]), int(cell["n"])
    except (KeyError, TypeError, ValueError):
        return None


def load_measured_cost(usage_path: Path) -> Optional[float]:
    if not usage_path.exists():
        return None
    with open(usage_path) as f:
        d = json.load(f)
    return d.get("total_cost_usd")


def estimate_cost(n_entities: int, rate_in: float, rate_out: float) -> float:
    return (n_entities * EST_IN_PER_ENTITY * rate_in
            + n_entities * EST_OUT_PER_ENTITY * rate_out) / 1e6


def pareto_frontier(points: List[Tuple[float, float]]) -> List[int]:
    """Return indices of Pareto-optimal points (min cost, max quality)."""
    indices = []
    for i, (cost_i, quality_i) in enumerate(points):
        dominated = False
        for j, (cost_j, quality_j) in enumerate(points):
            if j == i:
                continue
            if cost_j <= cost_i and quality_j >= quality_i and (cost_j < cost_i or quality_j > quality_i):
                dominated = True
                break
        if not dominated:
            indices.append(i)
    return sorted(indices, key=lambda i: points[i][0])


def collect_tier_eval_data(
    tier_eval_dirs: List[Path],
    slot: int,
) -> Dict[str, List[dict]]:
    """
    Returns {model: [{rho, cost, cost_is_measured, axis_name}, ...]}
    """
    per_model: Dict[str, List[dict]] = defaultdict(list)

    for tier_dir in tier_eval_dirs:
        if not tier_dir.is_dir():
            print(f"WARNING: {tier_dir} does not exist — skipping", file=sys.stderr)
            continue
        axis_name = tier_dir.name.replace("_tier_eval", "")
        for model_dir in sorted(tier_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            result = load_rho(model_dir / "correlations.json", slot)
            if result is None:
                continue
            rho, n = result
            cost = load_measured_cost(model_dir / "usage.json")
            if cost is None:
                print(f"  NOTE: {model} / {axis_name}: no usage.json — skipping this entry",
                      file=sys.stderr)
                continue
            per_model[model].append({
                "rho": rho, "n": n, "cost": cost,
                "cost_is_measured": True,
                "axis_name": axis_name,
            })

    return per_model


def collect_legacy_data(
    legacy_axis_dirs: List[Path],
    slot: int,
) -> Dict[str, List[dict]]:
    """
    Load gpt-4.1-mini and claude-sonnet-4 from the non-tier-eval axis dirs.
    These predate usage.json; costs are estimated.
    """
    per_model: Dict[str, List[dict]] = defaultdict(list)

    legacy_cells = [
        ("gpt-4.1-mini", "gpt", GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT),
        ("claude-sonnet-4", "sonnet", SONNET4_RATE_IN, SONNET4_RATE_OUT),
    ]

    for axis_dir in legacy_axis_dirs:
        axis_name = axis_dir.name
        for model, subdir, rin, rout in legacy_cells:
            corr = axis_dir / subdir / "correlations.json"
            result = load_rho(corr, slot)
            if result is None:
                continue
            rho, n = result
            cost = estimate_cost(n, rin, rout)
            per_model[model].append({
                "rho": rho, "n": n, "cost": cost,
                "cost_is_measured": False,
                "axis_name": axis_name,
            })

    return per_model


def aggregate(per_model: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Collapse per-axis entries into mean rho and mean cost per model."""
    agg: Dict[str, dict] = {}
    for model, entries in per_model.items():
        if not entries:
            continue
        mean_rho = float(np.mean([e["rho"] for e in entries]))
        mean_cost = float(np.mean([e["cost"] for e in entries]))
        n_axes = len(entries)
        measured = all(e["cost_is_measured"] for e in entries)
        agg[model] = {
            "mean_rho": mean_rho,
            "mean_cost": mean_cost,
            "n_axes": n_axes,
            "cost_is_measured": measured,
            "provider": provider_of(model),
        }
    return agg


def plot(agg: Dict[str, dict], slot: int, n_total_axes: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    model_list = sorted(agg.items(), key=lambda kv: kv[1]["mean_cost"])
    costs = [d["mean_cost"] for _, d in model_list]
    qualities = [1.0 / (1.0 - d["mean_rho"]) for _, d in model_list]

    frontier_idx = pareto_frontier(list(zip(costs, qualities)))
    frontier_costs = [costs[i] for i in frontier_idx]
    frontier_qualities = [qualities[i] for i in frontier_idx]

    # Draw Pareto frontier line
    ax.plot(frontier_costs, frontier_qualities, "k--", linewidth=1.2, alpha=0.5, zorder=2,
            label="Pareto frontier")

    # Assign vertical label offsets to avoid crowding among points with similar quality.
    # Strategy: for each point, if a previously placed label is within 0.08 quality units,
    # nudge this one down by 18pt per collision.
    placed: list[tuple[float, float, float]] = []  # (cost, quality, y_offset_pts)
    label_offsets: list[tuple[int, int]] = []
    for i, (model, d) in enumerate(model_list):
        cost = costs[i]
        quality = qualities[i]
        base_x = 12 if d["mean_cost"] < 0.5 else -12
        y_pts = 8
        for _, pq, py in placed:
            if abs(quality - pq) < 0.08:
                y_pts = py - 18  # push below the colliding label
        label_offsets.append((base_x, y_pts))
        placed.append((cost, quality, y_pts))

    for i, (model, d) in enumerate(model_list):
        quality = qualities[i]
        cost = costs[i]
        is_pareto = i in frontier_idx
        facecolor = PROVIDER_COLOR[d["provider"]] if d["cost_is_measured"] else "none"
        edgecolor = PROVIDER_COLOR[d["provider"]]
        marker = "D" if is_pareto else "o"
        size = 200 if is_pareto else 120
        ax.scatter(cost, quality, s=size, marker=marker,
                   facecolors=facecolor, edgecolors=edgecolor, linewidths=2.0, zorder=4)
        cost_note = "" if d["cost_is_measured"] else " (est.)"
        label = f"{model}  ρ={d['mean_rho']:.3f}  ${cost:.2f}{cost_note}  n={d['n_axes']}ax"
        ox, oy = label_offsets[i]
        ha = "left" if d["mean_cost"] < 0.5 else "right"
        ax.annotate(label, (cost, quality), textcoords="offset points",
                    xytext=(ox, oy), fontsize=9, ha=ha)

    ax.set_xlabel("Mean cost per axis (USD, descriptions mode)")
    ax.set_ylabel("Mean quality = 1 / (1 − ρ̄)")
    ax.set_title(
        f"Pooled judge-tier Pareto: {n_total_axes} axes, slot {slot}  "
        f"[◆ = Pareto-optimal, ● = dominated]\n"
        f"Filled = measured cost (usage.json)  Hollow = estimated (legacy tiers)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)

    provider_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PROVIDER_COLOR["anthropic"],
                   markeredgecolor=PROVIDER_COLOR["anthropic"],
                   markersize=10, label="Anthropic"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PROVIDER_COLOR["openai"],
                   markeredgecolor=PROVIDER_COLOR["openai"],
                   markersize=10, label="OpenAI"),
        plt.Line2D([0], [0], linestyle="--", color="black", alpha=0.5,
                   label="Pareto frontier"),
    ]
    ax.legend(handles=provider_handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(
                    title=f"Pooled judge-tier Pareto ({n_total_axes} axes, slot {slot})",
                    author="Claude",
                    source_text=Path(__file__).read_text(),
                ))
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiments_dir", type=str,
                   help="Root dir to glob for *_tier_eval subdirectories "
                        "(e.g. roger/axis_judge_experiments)")
    g.add_argument("--tier_eval_dirs", nargs="+", type=str,
                   help="Explicit list of tier_eval dirs")
    parser.add_argument("--legacy_dirs", nargs="*", type=str, default=None,
                        help="Optional: non-tier-eval axis dirs that have gpt/ and sonnet/ "
                             "subdirs for estimating legacy model cost+rho "
                             "(e.g. roger/axis_judge_experiments/angel_vs_demon). "
                             "If omitted with --experiments_dir, auto-detected from "
                             "*_tier_eval siblings that have a gpt/correlations.json.")
    parser.add_argument("--slot", type=int, default=6)
    parser.add_argument("--no_legacy", action="store_true",
                        help="Skip legacy model overlay entirely")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.experiments_dir:
        exp_dir = Path(args.experiments_dir)
        tier_eval_dirs = sorted(exp_dir.glob("*_tier_eval"))
    else:
        tier_eval_dirs = [Path(p) for p in args.tier_eval_dirs]

    if not tier_eval_dirs:
        print("ERROR: no *_tier_eval directories found. Run judge_tier_cost_eval.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tier_eval_dirs)} tier_eval dirs", file=sys.stderr)

    per_model = collect_tier_eval_data(tier_eval_dirs, args.slot)

    if not args.no_legacy:
        if args.legacy_dirs is not None:
            legacy_dirs = [Path(p) for p in args.legacy_dirs]
        elif args.experiments_dir:
            # Auto-detect: for each tier_eval dir, check if the base axis dir
            # has a gpt/correlations.json (meaning it has legacy judge data).
            exp_dir = Path(args.experiments_dir)
            legacy_dirs = []
            for td in tier_eval_dirs:
                base = exp_dir / td.name.replace("_tier_eval", "")
                if (base / "gpt" / "correlations.json").exists():
                    legacy_dirs.append(base)
        else:
            legacy_dirs = []

        if legacy_dirs:
            print(f"Loading legacy tiers from {len(legacy_dirs)} axis dirs", file=sys.stderr)
            legacy_data = collect_legacy_data(legacy_dirs, args.slot)
            for model, entries in legacy_data.items():
                per_model[model].extend(entries)

    if not per_model:
        print("ERROR: no data found at all.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(per_model)

    print(f"\nAggregated results ({args.slot=}):", file=sys.stderr)
    for model, d in sorted(agg.items(), key=lambda kv: -kv[1]["mean_rho"]):
        measured = "measured" if d["cost_is_measured"] else "estimated"
        print(f"  {model:30s}  mean_rho={d['mean_rho']:.4f}  "
              f"mean_cost=${d['mean_cost']:.2f}  n_axes={d['n_axes']}  cost={measured}",
              file=sys.stderr)

    n_total = max(len(v) for v in per_model.values())
    plot(agg, args.slot, n_total, Path(args.output))


if __name__ == "__main__":
    main()
