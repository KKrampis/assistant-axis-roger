#!/usr/bin/env python3
"""
Pooled judge-tier cost vs quality plot across multiple axes.

Reads all *_tier_eval/ directories found under roger/axis_judge_experiments/
(or a specified set via --tier_eval_dirs), aggregates per-axis rho and cost
for each judge model, then plots mean quality = 1/(1-rho) vs mean cost-per-axis
with the Pareto frontier highlighted.

Also optionally overlays the two legacy reference models ("gpt-4.1-mini (May)" and
"claude-sonnet-4 (May)", the May 2026 cells: T=0, max_tokens 1024) whose judge caches predate usage.json tracking: their rhos
come from the gpt/ and sonnet/ subdirs of the original axis directories and
their costs are ESTIMATED from the same token/pricing model as the dry-run.

Coverage rule: the plot's headline "N axes" is the largest number of axes any
model was judged on.  A model judged on fewer axes than that is NOT plotted
(its mean rho is over a different, usually easier, subset and is not
comparable); the models left out are named in the title and on stderr.
--min_axes N lowers the bar, e.g. --min_axes 60 keeps a legacy judge that
is missing one axis, and --min_axes 1 shows everything as before.

Price basis: batch and Flex cells are billed at half the real-time price, so
their recorded cost (usage.json's total_cost_usd) is not comparable with a
real-time cell's.  By default (--price_basis list) every cell's cost is
recomputed from its recorded token counts at the model's real-time price;
--price_basis batch prices everything at the batch rate, and
--price_basis recorded plots the cost as billed.

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


PRICE_BASES = ("list", "batch", "recorded")


def _base_model(key: str) -> str:
    """``claude-haiku-4-5:batch`` -> ``claude-haiku-4-5`` (usage keys carry the transport)."""
    for suffix in (":batch", ":flex"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def load_measured_cost(usage_path: Path, price_basis: str = "list") -> Optional[float]:
    """A cell's cost from its ``usage.json``.

    ``recorded``: the cost the run was billed (``total_cost_usd``), which mixes
    real-time cells with batch and Flex cells billed at half price.  ``list``
    (the default) and ``batch`` recompute the cost from the recorded token
    counts at the model's real-time price or its batch price, whatever
    transport the cell was judged by, so that every point on one plot is
    priced the same way.  A model the pricing table does not know falls back
    to the recorded cost with a warning.
    """
    if not usage_path.exists():
        return None
    with open(usage_path) as f:
        d = json.load(f)
    if price_basis == "recorded" or not d.get("per_model"):
        return d.get("total_cost_usd")
    total = 0.0
    for key, entry in d["per_model"].items():
        model = _base_model(entry.get("model") or key)
        try:
            rate_in, rate_out = price_for_model(model if price_basis == "list" else f"{model}:batch")
        except KeyError:
            print(f"WARNING: {usage_path}: no {price_basis} price for {model!r}; using the recorded cost",
                  file=sys.stderr)
            return d.get("total_cost_usd")
        total += (entry.get("prompt_tokens", 0) * rate_in
                  + entry.get("completion_tokens", 0) * rate_out) / 1e6
    return total


#: The reply cap of judge_tier_cost_eval.py runs made before each cell recorded
#: its own (the script's --max_tokens default; the 20- and 41-axis drivers never
#: set it).  A reply cut at the cap usually loses its SCORE line and is unscored.
TIER_EVAL_DEFAULT_MAX_TOKENS = 200


def cell_max_tokens(cell_dir: Path, legacy: bool = False) -> Optional[int]:
    """The judge's reply cap (max_tokens) for one cell.

    A tier-eval cell: its ``scores_descriptions.json`` provenance
    ``max_tokens``, else :data:`TIER_EVAL_DEFAULT_MAX_TOKENS`.  A legacy
    (May 2026, axis_judge_correlation.py) cell: ``config.json``'s
    ``args.max_tokens``, else None (not recorded)."""
    try:
        if legacy:
            with open(cell_dir / "config.json") as f:
                value = (json.load(f).get("args") or {}).get("max_tokens")
            return int(value) if value is not None else None
        with open(cell_dir / "scores_descriptions.json") as f:
            prov = json.load(f).get("_provenance") or {}
        value = prov.get("max_tokens")
        return int(value) if value is not None else TIER_EVAL_DEFAULT_MAX_TOKENS
    except (OSError, ValueError, TypeError, AttributeError):
        return None if legacy else TIER_EVAL_DEFAULT_MAX_TOKENS


def cap_summary(caps: Dict[str, set]) -> str:
    """One title line naming the judges' reply caps, ``{model: {max_tokens, ...}}``,
    grouped by cap: the largest group by count, the others by name."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for model, values in caps.items():
        known = sorted(v for v in values if v is not None)
        key = "/".join(str(v) for v in known) or "unrecorded"
        if known and None in values:
            key += " or unrecorded"
        groups[key].append(model)
    tail = " (a reply cut at the cap gives no score)"
    if len(groups) == 1:
        key = next(iter(groups))
        return f"Judge reply cap: max_tokens {key} for every judge{tail}"
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    parts = []
    for i, (key, models) in enumerate(ordered):
        who = f"{len(models)} judges" if i == 0 and len(models) > 2 else ", ".join(sorted(models))
        parts.append(f"{key} for {who}")
    return "Judge reply cap (max_tokens): " + "; ".join(parts) + tail


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
    price_basis: str = "list",
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
            cost = load_measured_cost(model_dir / "usage.json", price_basis)
            if cost is None:
                print(f"  NOTE: {model} / {axis_name}: no usage.json — skipping this entry",
                      file=sys.stderr)
                continue
            per_model[model].append({
                "rho": rho, "n": n, "cost": cost,
                "cost_is_measured": True,
                "axis_name": axis_name,
                "max_tokens": cell_max_tokens(model_dir),
            })

    return per_model


def rho_from_stored_scores(cell_dir: Path, axis_name: str, slot: int,
                           mode: str = "descriptions", metric: str = "raw") -> Optional[Tuple[float, int]]:
    """Recompute a legacy cell's rho from its ``scores_<mode>.json`` and
    ``projections.json`` when its ``correlations.json`` lacks the mode.

    A later run on the same cell that scored only another mode rewrites
    correlations.json without this one (the Sonnet cell of
    aligned_artificial_intelligence_vs_paperclip_maximizer: descriptions
    scored 580/580 on 2026-05-13, then a rejudge of one role's instructions
    the same evening left correlations.json with instructions only).  Uses
    ``compute_correlations`` itself, with the axis's two poles excluded, so
    the result is what the original run computed.
    """
    scores_path = cell_dir / f"scores_{mode}.json"
    proj_path = cell_dir / "projections.json"
    if not (scores_path.exists() and proj_path.exists()):
        return None
    from results_analysis.axis_judge_correlation import compute_correlations  # heavy; only when needed

    with open(scores_path) as f:
        sd = json.load(f)
    scores = {k: v for k, v in sd.get("result", sd).items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    with open(proj_path) as f:
        pd_ = json.load(f)
    projections = {int(s): v for s, v in pd_.get("result", pd_).items() if s.lstrip("-").isdigit()}
    poles = set(axis_name.split("_vs_", 1))
    corr = compute_correlations({mode: scores}, projections, [slot], poles)
    cell = corr[mode][str(slot)][metric]
    if cell["n"] < 3:
        return None
    return float(cell["rho"]), int(cell["n"])


def collect_legacy_data(
    legacy_axis_dirs: List[Path],
    slot: int,
    price_basis: str = "list",
) -> Dict[str, List[dict]]:
    """
    Load gpt-4.1-mini and claude-sonnet-4 from the non-tier-eval axis dirs.
    These predate usage.json; costs are estimated at the real-time price
    (``list``, also used for ``recorded``: they were judged in real time) or
    at half of it (``batch``).
    """
    factor = 0.5 if price_basis == "batch" else 1.0
    per_model: Dict[str, List[dict]] = defaultdict(list)

    legacy_cells = [
        # Labelled "(May)": the May 2026 cells of axis_judge_correlation.py (T=0, max_tokens
        # 1024), kept apart from any tier-eval cell of the same model (gpt-4.1-mini rejudged at
        # T=0 and max_tokens 200 on 2026-11-01 sits under <axis>_tier_eval/gpt-4.1-mini/).
        ("gpt-4.1-mini (May)", "gpt", GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT),
        ("claude-sonnet-4 (May)", "sonnet", SONNET4_RATE_IN, SONNET4_RATE_OUT),
    ]

    for axis_dir in legacy_axis_dirs:
        axis_name = axis_dir.name
        for model, subdir, rin, rout in legacy_cells:
            corr = axis_dir / subdir / "correlations.json"
            result = load_rho(corr, slot)
            if result is None:
                result = rho_from_stored_scores(axis_dir / subdir, axis_name, slot)
                if result is not None:
                    print(f"  NOTE: {model} / {axis_name}: correlations.json has no descriptions "
                          f"result; rho recomputed from scores_descriptions.json and "
                          f"projections.json (rho={result[0]:.3f}, n={result[1]})", file=sys.stderr)
            if result is None:
                continue
            rho, n = result
            cost = estimate_cost(n, rin, rout) * factor
            per_model[model].append({
                "rho": rho, "n": n, "cost": cost,
                "cost_is_measured": False,
                "axis_name": axis_name,
                "max_tokens": cell_max_tokens(axis_dir / subdir, legacy=True),
            })

    return per_model


def rho_from_robustness(per_model: Dict[str, List[dict]], robustness_json: Path,
                        variant: str = "s6_l25_raw") -> Dict[str, List[dict]]:
    """Replace every entry's rho and n with judge_tier_robustness.py's for ``variant``.

    The robustness tool recomputes rho from the stored scores and drops the nine
    trait/role collision names on bare-keyed (v1) axes, where the tier-eval cells
    judged a collision name's role description against the trait's cached
    projection; its s6_l25_raw values are therefore the corrected version of the
    cells' correlations.json.  An entry the JSON lacks is dropped with a note."""
    with open(robustness_json) as f:
        d = json.load(f)
    r = d.get("result", d)
    rho, n = r["rho"], r.get("n", {})
    out: Dict[str, List[dict]] = defaultdict(list)
    for model, entries in per_model.items():
        for e in entries:
            value = (rho.get(model, {}).get(e["axis_name"]) or {}).get(variant)
            if value is None:
                print(f"  NOTE: {model} / {e['axis_name']}: not in {robustness_json} for {variant}; dropped",
                      file=sys.stderr)
                continue
            out[model].append({**e, "rho": float(value), "n": int(n.get(model, {}).get(e["axis_name"], e["n"]))})
    return out


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
            "max_tokens": {e.get("max_tokens") for e in entries},
        }
    return agg


def split_by_coverage(agg: Dict[str, dict], min_axes: int) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Split aggregated models into (plotted, excluded) by axis coverage.

    A model enters the plot only if it was judged on at least ``min_axes``
    axes.  The excluded models are returned so the plot can say they exist.
    """
    kept = {m: d for m, d in agg.items() if d["n_axes"] >= min_axes}
    excluded = {m: d for m, d in agg.items() if d["n_axes"] < min_axes}
    return kept, excluded


PRICE_BASIS_LABEL = {
    "list": "real-time list price",
    "batch": "batch price (half of list)",
    "recorded": "as billed (mixed transports)",
}


def plot(agg: Dict[str, dict], slot: int, n_total_axes: int, output_path: Path,
         excluded: Optional[Dict[str, dict]] = None, price_basis: str = "list",
         rho_note: str = "") -> None:
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

    ax.set_xlabel(f"Mean cost per axis (USD at {PRICE_BASIS_LABEL[price_basis]}, descriptions mode)")
    ax.set_ylabel("Mean quality = 1 / (1 − ρ̄)")
    title = (
        f"Pooled judge-tier Pareto: {n_total_axes} axes, slot {slot}  "
        f"[◆ = Pareto-optimal, ● = dominated]\n"
        f"Filled = measured cost (usage.json)  Hollow = estimated (legacy tiers)"
    )
    if rho_note:
        title += f"\n{rho_note}"
    if excluded:
        names = ", ".join(f"{m} ({d['n_axes']})"
                          for m, d in sorted(excluded.items(), key=lambda kv: -kv[1]["n_axes"]))
        title += f"\nNot shown, judged on fewer than {n_total_axes} axes: {names}"
    ax.set_title(title, fontsize=11)
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
    # Room for the labels of the lowest and highest points (a label pushed down by
    # the collision rule used to land on the x-axis title), and the legend in the
    # corner a cost/quality frontier leaves empty (high cost, high quality is
    # rare; low cost, low quality is where cheap judges sit).
    ax.margins(x=0.06, y=0.14)
    ax.legend(handles=provider_handles, loc="upper right", fontsize=9)

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
    parser.add_argument("--min_axes", type=int, default=None,
                        help="Plot only models judged on at least this many axes. Default: the "
                             "headline axis count (the most any model has), so every plotted "
                             "model covers the same axes; --min_axes 1 shows everything.")
    parser.add_argument("--price_basis", choices=PRICE_BASES, default="list",
                        help="How a point's cost is priced.  'list' (default): every cell recomputed "
                             "from its token counts at the model's real-time price, whatever transport "
                             "judged it (batch and Flex cells are billed at half price, so their recorded "
                             "cost is not comparable with real-time cells); 'batch': the same at the "
                             "batch price; 'recorded': usage.json's total_cost_usd as billed.")
    parser.add_argument("--robustness_json", type=str, default=None,
                        help="judge_tier_robustness.json: take each cell's rho from it (variant "
                             "--robustness_variant) instead of the cell's correlations.json.  Its rho is "
                             "recomputed from the stored scores with the nine trait/role collision names "
                             "dropped on bare-keyed axes, which the cells' correlations.json mismatch.")
    parser.add_argument("--robustness_variant", type=str, default="s6_l25_raw",
                        help="Variant key in --robustness_json (default s6_l25_raw: slot 6, layer 25, raw, "
                             "the cells' own projection).")
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

    per_model = collect_tier_eval_data(tier_eval_dirs, args.slot, args.price_basis)

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
            legacy_data = collect_legacy_data(legacy_dirs, args.slot, args.price_basis)
            for model, entries in legacy_data.items():
                per_model[model].extend(entries)

    if args.robustness_json:
        if not args.robustness_variant.startswith(f"s{args.slot}_"):
            print(f"WARNING: --robustness_variant {args.robustness_variant} is not slot {args.slot}",
                  file=sys.stderr)
        per_model = rho_from_robustness(per_model, Path(args.robustness_json), args.robustness_variant)

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
    min_axes = args.min_axes if args.min_axes is not None else n_total
    agg, excluded = split_by_coverage(agg, min_axes)
    if excluded:
        print(f"Not plotted (judged on fewer than {min_axes} axes): "
              + ", ".join(f"{m} (n_axes={d['n_axes']})" for m, d in sorted(excluded.items())),
              file=sys.stderr)
    if not agg:
        print(f"ERROR: no model has {min_axes} axes; lower --min_axes.", file=sys.stderr)
        sys.exit(1)
    rho_note = (f"rho from {Path(args.robustness_json).name}, variant {args.robustness_variant} "
                f"(collision names dropped on bare-keyed axes)") if args.robustness_json else ""
    plot(agg, args.slot, n_total, Path(args.output), excluded=excluded, price_basis=args.price_basis,
         rho_note=rho_note)


if __name__ == "__main__":
    main()
