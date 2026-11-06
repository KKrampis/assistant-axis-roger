"""The optimisation grid of batched percentile judging: for each judge and arm, mean rho (slot 6,
layer 25, raw, from the cells' own correlations.json) and cost per axis from usage.json token counts, at
the real price (each judge's batch or Flex rate) since 2026-11-03, --price_basis list for the list price.  Arms: passes 1x-4x at N=10 (shuffles averaged in order), batch size 5/7/10/15/20 at
2 passes, reasoning none/low/medium at N=10 and 2 passes.  A judge or arm whose cells are not all
there is left out.  Writes <output_dir>/<stem>.json and .png.

Usage: uv run python scripts/judge_tier_grid_analysis.py --axes_file scripts/tier_eval_axes_dev30.tsv \\
           --stem judge_tier_grid_dev30 --axes_label "30 development axes\""""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from results_analysis.judge_tier_cost_plot_pooled import (load_measured_cost, place_labels, cost_ticks,
                                                           PROVIDER_COLOR, provider_of, pareto_frontier,
                                                           DEFAULT_PRICE_BASIS, PRICE_BASES, PRICE_BASIS_LABEL)
from assistant_axis import json_metadata, png_metadata
from assistant_axis.provenance import current_file_input, current_files_input

# judge: (the cell prefix under the batched method, the prefix for a reasoning arm, first shuffle seed)
JUDGES = {"gpt-5.6-luna": ("gpt-5.6-luna@none", "gpt-5.6-luna", 1), "gpt-6-luna": ("gpt-6-luna", "gpt-6-luna", 1),
          "claude-haiku-5-5": ("claude-haiku-5-5", "claude-haiku-5-5", 3),
          "gemini-3.1-flash-lite": ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite", 1),
          "gemini-3.5-flash-lite": ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite", 1)}
MARKERS = {"passes": "o", "batch size": "s", "reasoning": "^"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments_dir", default="roger/axis_judge_experiments")
    ap.add_argument("--axes_file", default="scripts/tier_eval_axes_dev30.tsv")
    ap.add_argument("--output_dir", default=None, help="default: <experiments_dir>/judge_tier_grid_dev30")
    ap.add_argument("--stem", default="judge_tier_grid_dev30")
    ap.add_argument("--axes_label", default="30 development axes")
    ap.add_argument("--price_basis", choices=PRICE_BASES, default=DEFAULT_PRICE_BASIS,
                    help=f"as judge_tier_cost_plot_pooled.py (default {DEFAULT_PRICE_BASIS}, the real price; list until "
                         f"2026-11-03)")
    args = ap.parse_args()
    E = Path(args.experiments_dir); out = Path(args.output_dir) if args.output_dir else E / "judge_tier_grid_dev30"
    axes = [l.split()[0] for l in open(args.axes_file) if l.strip() and not l.startswith("#")]

    read: list = []

    def cell_stats(cell):
        rhos, costs = [], []
        for a in axes:
            d = E / f"{a}_tier_eval" / cell
            if not (d / "correlations.json").exists():
                return None
            with open(d / "correlations.json") as f:
                rhos.append(json.load(f)["descriptions"]["6"]["raw"]["rho"])
            costs.append(load_measured_cost(d / "usage.json", args.price_basis))
            read.extend([d / "correlations.json", d / "usage.json"])
        return float(np.nanmean(rhos)), float(np.mean(costs))

    rows = []
    for judge, (base, plain, first) in JUDGES.items():
        arms = [("passes", f"{k}x", f"{base}@b10s{first}" if k == 1 else f"{base}@b10x{k}") for k in range(1, 5)]
        arms += [("batch size", f"N={n}", f"{base}@b10x2" if n == 10 else f"{base}@b10n{n}x2") for n in (5, 7, 10, 15, 20)]
        arms += [("reasoning", lvl, f"{base}@b10x2" if lvl == "none" else f"{plain}{tok}@b10x2")
                 for lvl, tok in (("none", ""), ("low", "@low"), ("medium", "@med"))]
        arms += [("old rubric", "v3", base)]
        for dimension, arm, cell in arms:
            st = cell_stats(cell)
            if st:
                rows.append(dict(judge=judge, dimension=dimension, arm=arm, cell=cell, mean_rho=st[0], cost_per_axis=st[1], price_basis=args.price_basis))

    print(f"{'judge':22s} {'dimension':11s} {'arm':8s} {'rho':>7s} {'$/axis':>7s}  cell")
    for r in rows:
        print(f"{r['judge']:22s} {r['dimension']:11s} {r['arm']:8s} {r['mean_rho']:7.4f} {r['cost_per_axis']:7.3f}  {r['cell']}")
    out.mkdir(parents=True, exist_ok=True)
    inputs = [current_file_input("axes_file", Path(args.axes_file)), current_files_input("cell_files", sorted(set(read)))]
    (out / f"{args.stem}.json").write_text(json.dumps(json_metadata(rows, title=f"{args.stem}: {args.axes_label}", inputs=inputs), indent=1))

    grid = [r for r in rows if r["dimension"] != "old rubric"]
    # A judge with one arm (Gemini, N=20 only) stays in the table but has no sweep to draw.
    judges = [j for j in JUDGES if sum(r["judge"] == j for r in grid) > 1]
    grid = [r for r in grid if r["judge"] in judges]
    ys = [1 / (1 - r["mean_rho"]) for r in grid]
    pad = 0.08 * (max(ys) - min(ys) or 0.1)
    fig, axs = plt.subplots(1, len(judges), figsize=(6.2 * len(judges), 6.6), sharey=True, squeeze=False)
    for ax, judge in zip(axs[0], judges):
        mine: dict = {}          # the shared base cell is one point with one label
        for r in grid:
            if r["judge"] == judge:
                mine.setdefault(r["cell"], []).append(r)
        pts = [(rs[0]["cost_per_axis"], 1 / (1 - rs[0]["mean_rho"])) for rs in mine.values()]
        labels = [("base: " + ", ".join(r["arm"] for r in rs)) if len(rs) > 1 else rs[0]["arm"] for rs in mine.values()]
        fr = pareto_frontier(pts); bold = [i in fr for i in range(len(pts))]
        col = PROVIDER_COLOR[provider_of(judge)]
        for i, ((c, q), rs) in enumerate(zip(pts, mine.values())):
            ax.scatter(c, q, s=130 if bold[i] else 70, marker="D" if len(rs) > 1 else MARKERS[rs[0]["dimension"]], color=col,
                       zorder=4, edgecolors="black" if bold[i] else "none", linewidths=1.0)
        fc = sorted(pts[i] for i in fr)
        ax.plot([p[0] for p in fc], [p[1] for p in fc], "--", color="#555", lw=1, alpha=0.6)
        ax.set_xscale("log"); xs = [p[0] for p in pts]; lo, hi = min(xs) / 1.35, max(xs) * 1.35; ax.set_xlim(lo, hi)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        tk = [t for t in (0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3) if lo <= t <= hi] or cost_ticks(lo, hi)
        ax.set_xticks(tk); ax.set_xticklabels([f"\\${t:g}" for t in tk]); ax.minorticks_off()
        ax.grid(True, alpha=0.3); ax.set_title(judge); ax.set_xlabel(f"cost per axis (USD, {PRICE_BASIS_LABEL[args.price_basis]}, log)")
        old = [r for r in rows if r["judge"] == judge and r["dimension"] == "old rubric"]
        if old:
            ax.text(0.97, 0.03, f"old rubric, off the chart: ρ {old[0]['mean_rho']:.3f} at \\${old[0]['cost_per_axis']:.3f}",
                    transform=ax.transAxes, ha="right", fontsize=8.5, color="#555")
        fig.canvas.draw(); place_labels(ax, pts, labels, bold, fontsize=8.5, leaders="all")
    axs[0][0].set_ylabel("1 / (1 − mean ρ)")
    title = f"Batched percentile judging, DESCRIPTION judging, {args.axes_label}"
    fig.suptitle(f"{title}\npasses at N=10 (circles), batch size at 2 passes (squares), reasoning effort at 2 passes (triangles); "
                 "diamond = the one cell all three share; black edge = on that judge's frontier", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"{args.stem}.png", dpi=150, bbox_inches="tight", metadata=png_metadata(title=title, inputs=inputs))
    print("Wrote", out / f"{args.stem}.png")


if __name__ == "__main__":
    main()
