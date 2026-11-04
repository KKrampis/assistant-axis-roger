"""The rubric-tuning figure: one panel per judge, every rubric variant as a point (cost per axis at
the robustness file's price basis, quality 1/(1 - rho) at the default variant) from a judge_tier_robustness.json, labelled by
the variant tokens.  No API calls.

Usage: uv run python scripts/judge_tier_tuning_panels.py --robustness_json RHO.json --output OUT.png"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from assistant_axis import png_metadata
from assistant_axis.provenance import CACHE_POLICIES, load_and_register
from results_analysis.judge_tier_cost_plot_pooled import (PRICE_BASIS_LABEL, PROVIDER_COLOR, cost_ticks, pareto_frontier,
                                                           place_labels, provider_of)

JUDGES = ("gpt-6-luna", "claude-haiku-5-5", "gpt-5.6-luna@none", "gpt-4.1-mini")
DEFAULT_TITLE = ("Rubric tuning, DESCRIPTION judging, 30 development axes: every variant per judge "
                 "(v3 old; c1-c10 single-entity; b10x2 batched percentile, 2 shuffles averaged)")
TOKEN_LABELS = {"b10x2": "batched ×2", "rep2": "v3 repeat"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robustness_json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variant", default="s6_l25_raw")
    ap.add_argument("--title", default=DEFAULT_TITLE)
    ap.add_argument("--cache-policy", dest="cache_policy", choices=CACHE_POLICIES, default="warn",
                    help="what to do when the rho table's recorded inputs have changed: strict | warn (default) | rebuild | off")
    args = ap.parse_args(argv)
    inputs: list = []
    r, _spec, _check = load_and_register(Path(args.robustness_json), dep_key="robustness", inputs=inputs, policy=args.cache_policy)
    r = r.get("result", r)
    per, models = r["per_variant"][args.variant]["models"], r["models"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, judge in zip(axes.flat, JUDGES):
        cells = [m for m in models if m == judge or m.startswith(judge + "@")]
        pts, labels = [], []
        for m in cells:
            token = m[len(judge):].lstrip("@") or "v3 (old)"
            pts.append((models[m]["mean_cost"], 1 / (1 - per[m]["mean_rho"])))
            labels.append(TOKEN_LABELS.get(token, token))
        fr = pareto_frontier(pts)
        bold = [i in fr for i in range(len(pts))]
        col = PROVIDER_COLOR[provider_of(judge)]
        for (c, q), b in zip(pts, bold):
            ax.scatter(c, q, s=90 if b else 50, marker="D" if b else "o", color=col, zorder=4)
        fc = sorted(pts[i] for i in fr)
        ax.plot([p[0] for p in fc], [p[1] for p in fc], "--", color="#555", lw=1, alpha=0.6)
        ax.set_xscale("log")
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.set_xlim(min(xs) / 1.5, max(xs) * 1.5)
        pad = 0.12 * (max(ys) - min(ys))
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ticks = cost_ticks(min(xs) / 1.5, max(xs) * 1.5)
        ax.set_xticks(ticks); ax.set_xticklabels([f"\\${t:g}" for t in ticks]); ax.minorticks_off()
        ax.grid(True, alpha=0.3); ax.set_title(judge, fontsize=11)
        ax.set_xlabel(f"cost per axis (USD, {PRICE_BASIS_LABEL[r.get('config', {}).get('price_basis', 'list')]}, log)"); ax.set_ylabel("1 / (1 − mean ρ)")
        fig.canvas.draw()
        place_labels(ax, pts, labels, bold, fontsize=8.5)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=150, bbox_inches="tight", metadata=png_metadata(title=args.title, inputs=inputs))
    plt.close(fig)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
