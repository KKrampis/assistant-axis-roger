#!/usr/bin/env python3
"""
Plot judge-tier cost vs. correlation quality, in the same visual language
as plot_batch_size_quality_vs_cost.py's batch_size_cost_vs_quality.png:
quality = 1/(1-rho) on the y-axis, $/axis on the x-axis, one annotated
point per tier.

Consumes the output of judge_tier_cost_eval.py (the 6 new tiers: measured
cost from usage.json, rho from correlations.json) plus the two legacy
tiers already committed under roger/axis_judge_experiments/{pair}/
{gpt,sonnet}/ (correlations.json exists there too -- same descriptions-
mode measurement, same 571 entities -- but no usage.json, since that
feature postdates those archived runs, so their cost is ESTIMATED from
the same token/pricing model rather than measured. The chart marks this
distinction with marker style (filled = measured, hollow = estimated) --
never claim more certainty than the data actually has.

Usage:
    uv run python results_analysis/judge_tier_cost_plot.py \\
        --pair angel demon \\
        --tier_eval_dir roger/axis_judge_experiments/angel_vs_demon_tier_eval \\
        --legacy_dir roger/axis_judge_experiments/angel_vs_demon \\
        --slot 6 \\
        --output roger/axis_judge_experiments/angel_vs_demon_tier_eval/cost_vs_quality.png

Requires judge_tier_cost_eval.py to have actually been run first (this
script makes no API calls itself -- it only reads existing JSON).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402

from assistant_axis.judge_pricing import (  # noqa: E402
    GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT, SONNET4_RATE_IN, SONNET4_RATE_OUT,
    price_for_model,
)
from assistant_axis.plot_metadata import png_metadata  # noqa: E402

# Same coarse per-entity token model used elsewhere in this project for
# description-mode judging (README "Judging cost model" / judge_tier_cost_eval.py
# --dry_run): ~320-token fixed rubric header, ~150 tokens per short entity
# description, ~80 tokens of judge output (reasoning + SCORE: line).
EST_IN_PER_ENTITY = 320 + 150
EST_OUT_PER_ENTITY = 80

PROVIDER_COLOR = {"openai": "#7FB069", "anthropic": "#D9636B"}  # sage / coral -- matches this project's existing palette


def estimate_legacy_cost(n_entities: int, rate_in: float, rate_out: float) -> float:
    in_tok = n_entities * EST_IN_PER_ENTITY
    out_tok = n_entities * EST_OUT_PER_ENTITY
    return in_tok * rate_in / 1e6 + out_tok * rate_out / 1e6


def load_rho(correlations_path: Path, slot: int, mode: str = "descriptions", metric: str = "raw") -> Optional[Tuple[float, int]]:
    if not correlations_path.exists():
        return None
    with open(correlations_path) as f:
        d = json.load(f)
    r = d.get("result", d)
    try:
        cell = r[mode][str(slot)][metric]
        return cell["rho"], cell["n"]
    except (KeyError, TypeError):
        return None


def load_measured_cost(usage_path: Path) -> Optional[float]:
    if not usage_path.exists():
        return None
    with open(usage_path) as f:
        d = json.load(f)
    return d.get("total_cost_usd")


def provider_of(model: str) -> str:
    return "openai" if model.startswith(("gpt-", "o1-", "o3-", "o4-")) else "anthropic"


def collect_points(args: argparse.Namespace) -> Dict[str, dict]:
    points: Dict[str, dict] = {}

    # Legacy tiers -- cost ESTIMATED (no usage.json exists for these).
    legacy_dir = Path(args.legacy_dir)
    legacy = {
        "gpt-4.1-mini": (legacy_dir / "gpt" / "correlations.json", GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT),
        "claude-sonnet-4": (legacy_dir / "sonnet" / "correlations.json", SONNET4_RATE_IN, SONNET4_RATE_OUT),
    }
    for model, (corr_path, rin, rout) in legacy.items():
        result = load_rho(corr_path, args.slot)
        if result is None:
            print(f"WARNING: no descriptions/slot{args.slot} rho found for legacy {model} at {corr_path}", file=sys.stderr)
            continue
        rho, n = result
        points[model] = {
            "rho": rho, "n": n,
            "cost": estimate_legacy_cost(n, rin, rout),
            "cost_is_measured": False,
            "provider": provider_of(model),
        }

    # New tiers -- cost MEASURED from usage.json.
    tier_eval_dir = Path(args.tier_eval_dir)
    if tier_eval_dir.exists():
        for model_dir in sorted(tier_eval_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            result = load_rho(model_dir / "correlations.json", args.slot)
            cost = load_measured_cost(model_dir / "usage.json")
            if result is None or cost is None:
                print(f"WARNING: incomplete data for {model} under {model_dir} -- skipping", file=sys.stderr)
                continue
            rho, n = result
            points[model] = {
                "rho": rho, "n": n, "cost": cost,
                "cost_is_measured": True,
                "provider": provider_of(model),
            }
    else:
        print(f"NOTE: {tier_eval_dir} does not exist yet -- run judge_tier_cost_eval.py first "
              f"for the 6 new tiers. Plotting legacy tiers only.", file=sys.stderr)

    return points


def plot(points: Dict[str, dict], pair: Tuple[str, str], slot: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    for model, d in sorted(points.items(), key=lambda kv: kv[1]["cost"]):
        quality = 1.0 / (1.0 - d["rho"])
        # Filled = measured cost (usage.json exists); hollow = estimated
        # (legacy tiers, no usage.json -- that feature postdates them).
        facecolor = PROVIDER_COLOR[d["provider"]] if d["cost_is_measured"] else "none"
        ax.scatter(d["cost"], quality, s=140, marker="o",
                   facecolors=facecolor, edgecolors=PROVIDER_COLOR[d["provider"]], linewidths=2, zorder=3)
        label = f"{model}\nrho={d['rho']:.3f}  ${d['cost']:.2f}" + ("" if d["cost_is_measured"] else " (est.)")
        ax.annotate(label, (d["cost"], quality), textcoords="offset points", xytext=(10, 6), fontsize=9)

    ax.set_xlabel(f"Cost per axis (USD, descriptions mode, n≈571 entities)")
    ax.set_ylabel("Quality = 1 / (1 - rho)")
    pos, neg = pair
    ax.set_title(f"Judge tier comparison: {pos} vs {neg}, slot {slot}\n"
                 f"Filled = measured cost (usage.json)   Hollow = estimated (legacy tiers, no usage.json existed)",
                 fontsize=11)
    ax.grid(True, alpha=0.3)

    # Legend for provider color, independent of the per-point labels above.
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PROVIDER_COLOR["anthropic"],
                   markeredgecolor=PROVIDER_COLOR["anthropic"], markersize=10, label="Anthropic"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PROVIDER_COLOR["openai"],
                   markeredgecolor=PROVIDER_COLOR["openai"], markersize=10, label="OpenAI"),
    ]
    ax.legend(handles=handles, loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(
                    title=f"Judge tier cost vs quality: {pos} vs {neg}",
                    author="Claude",
                    source_text=Path(__file__).read_text(),
                ))
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, metavar=("POS", "NEG"), required=True)
    parser.add_argument("--tier_eval_dir", type=str, required=True,
                         help="Output dir from judge_tier_cost_eval.py (the 6 new tiers)")
    parser.add_argument("--legacy_dir", type=str, required=True,
                         help="The existing roger/axis_judge_experiments/<pair>/ dir "
                              "(with gpt/ and sonnet/ subdirs) for the 2 legacy tiers")
    parser.add_argument("--slot", type=int, default=6,
                         help="Slot to plot (default 6, this project's tuned canonical operating point)")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    points = collect_points(args)
    if not points:
        print("No data found at all -- nothing to plot.", file=sys.stderr)
        sys.exit(1)

    plot(points, tuple(args.pair), args.slot, Path(args.output))


if __name__ == "__main__":
    main()
