"""The optimisation grid of batched judging under the 12 projection variants: one figure per
dimension (batch size, passes, reasoning effort) and one of the whole grid, each in the layout of
judge_tier_robustness.py (a heatmap of mean rho above the 12 Pareto panels).  Each figure's frontier
is taken among its own cells, so a setting that is on the batch-size frontier may be off the whole
grid's.  Writes judge_tier_grid_12way_<dimension>.png and judge_tier_grid_12way.json.

Usage: first the rho table, then the figures (neither makes API calls):
    uv run python results_analysis/judge_tier_robustness.py --axes_file scripts/tier_eval_axes_dev30.tsv \\
        --include_models $(uv run python scripts/judge_tier_grid_12way.py --print_cells) --output_dir OUT
    uv run python scripts/judge_tier_grid_12way.py --robustness_json OUT/judge_tier_robustness.json --output_dir DIR"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_axis import json_metadata
from assistant_axis.provenance import CACHE_POLICIES, load_and_register
from results_analysis import judge_tier_robustness as rob

# (name in labels, the cell prefix under the batched method, the prefix for a reasoning arm, first
# shuffle seed, passes measured).  Gemini 3.1 stops at 3 passes (Roger, 2026-11-02: "up to 3x").
JUDGES = (("Luna", "gpt-5.6-luna@none", "gpt-5.6-luna", 1, 4), ("GPT-6", "gpt-6-luna", "gpt-6-luna", 1, 4),
          ("Haiku", "claude-haiku-5-5", "claude-haiku-5-5", 3, 4),
          ("Gemini", "gemini-3.1-flash-lite", "gemini-3.1-flash-lite", 1, 3))
BLENDS = {"gpt-5.6-luna+claude-haiku-5-5@b10n20x2": "Luna + Haiku N=20",
          "gpt-5.6-luna+gemini-3.1-flash-lite@b10n20x2": "Luna + Gemini N=20",
          "gpt-5.6-luna+claude-haiku-5-5+gemini-3.1-flash-lite@b10n20x2": "Luna + Haiku + Gemini N=20"}
X_TICKS = (0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)


def dimension_arms(base: str, plain: str, first: int, max_passes: int = 4) -> dict:
    """{dimension: {arm: cell}} for one judge; every dimension holds the other two at the base
    setting, N=10, two passes, no reasoning."""
    return {"batch_size": {f"N={n}": f"{base}@b10x2" if n == 10 else f"{base}@b10n{n}x2" for n in (5, 7, 10, 15, 20)},
            "passes": {f"{k}x": f"{base}@b10s{first}" if k == 1 else f"{base}@b10x{k}" for k in range(1, max_passes + 1)},
            "reasoning": {lvl: f"{base}@b10x2" if lvl == "none" else f"{plain}{tok}@b10x2"
                          for lvl, tok in (("none", ""), ("low", "@low"), ("medium", "@med"))}}


def figures() -> dict:
    """{stem: (labels {cell: label}, shorter point labels, series, headline phrase, subtitle, frontier
    labels only, panel width, panel height)}."""
    figs = {}
    own = "the frontier is among this figure's settings only"
    text = {"batch_size": ("Batch size", f"N = entities per call, at 2 passes and no reasoning; arrows run from N=5 to N=20; {own}", 5.2, 4.2),
            "passes": ("Number of passes", f"passes = random batch orders averaged, at N=10 and no reasoning; arrows run from 1 pass to 4 "
                       f"(3 for Gemini); {own}", 4.6, 3.8),
            "reasoning": ("Reasoning effort", f"at N=10 and 2 passes; arrows run from no reasoning to medium; {own}", 4.6, 3.6)}
    for dim, (phrase, sub, w, h) in text.items():
        labels, short, series = {}, {}, []
        for name, base, plain, first, max_passes in JUDGES:
            arms = dimension_arms(base, plain, first, max_passes)[dim]
            labels.update({cell: f"{name} {arm}" for arm, cell in arms.items()})
            series.append(list(arms.values()))
            if dim == "batch_size":      # each judge's five points sit together, so one name is enough
                short.update({cell: arm[2:] for arm, cell in arms.items() if arm != "N=20"})
        figs[dim] = (labels, short, series, phrase, sub, False, w, h)
    labels = {}
    for name, base, plain, first, max_passes in JUDGES:
        for dim, arms in dimension_arms(base, plain, first, max_passes).items():
            for arm, cell in arms.items():
                labels[cell] = f"{name} base" if cell == f"{base}@b10x2" else f"{name} {arm}"
    labels.update(BLENDS)
    figs["all"] = (labels, {}, [], "The whole grid", "base = N=10, 2 passes, no reasoning; each other row changes one of those; "
                   "blends at N=20, 2 passes; only frontier points are labelled", True, 4.6, 4.2)
    return figs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print_cells", action="store_true", help="print the cells the rho table must hold, and stop")
    ap.add_argument("--robustness_json")
    ap.add_argument("--output_dir")
    ap.add_argument("--axes_label", default="30 development axes")
    ap.add_argument("--cache-policy", dest="cache_policy", choices=CACHE_POLICIES, default="warn",
                    help="what to do when the rho table's recorded inputs have changed: strict | warn (default) | rebuild | off")
    args = ap.parse_args()
    if args.print_cells:
        print(" ".join(figures()["all"][0]))
        return
    inputs: list = []
    payload, _spec, _check = load_and_register(Path(args.robustness_json), dep_key="robustness", inputs=inputs, policy=args.cache_policy)
    d = rob._unwrap(payload); cfg = d["config"]; out = Path(args.output_dir)
    variants = rob.build_variants(cfg["slots"], cfg["layers"], cfg["regimes"], K=cfg["soft_K"], L=cfg["shear_L"])
    default = cfg["default_variant"]; costs = {m: v["mean_cost"] for m, v in d["models"].items()}
    results = {}
    for stem, (labels, short, series, phrase, sub, frontier_only, panel_w, panel_h) in figures().items():
        cells = list(labels)
        missing = [c for c in cells if c not in d["rho"]]
        if missing:
            raise SystemExit(f"{stem}: the rho table has no {missing}")
        per_variant, frontiers = {}, {}
        for v in variants:
            per_variant[v.key], frontiers[v.key] = rob.summarise_variant(d["rho"], costs, cells, v.key)
        order = rob.ranking(per_variant[default], cells)
        counts = rob.frontier_counts(frontiers, cells)
        headline = (f"{phrase}, DESCRIPTION judging, under {len(variants)} projection variants: {args.axes_label} "
                    f"(default {default})")
        rob.render_figure(out / f"judge_tier_grid_12way_{stem}.png", order, variants, per_variant, frontiers,
                          {m: True for m in cells}, default, len(d["axes"]),
                          f"{sub}\ncost at {rob.PRICE_BASIS_LABEL[cfg.get('price_basis', 'list')]}; soft_K: K={cfg['soft_K']}; soft_shear: L={cfg['shear_L']}",
                          inputs, rob.FigureOptions(headline=headline, cell_labels=labels, point_labels=short, series=series,
                                                    label_frontier_only=frontier_only, own_y=True, x_log=True, x_ticks=X_TICKS, x_log_pad=2.0,
                                                    panel_w=panel_w, panel_h=panel_h))
        results[stem] = {"frontier_sets": frontiers, "cells": {
            m: {"label": labels[m], "mean_cost": costs[m], "mean_rho_default": per_variant[default][m]["mean_rho"],
                "on_frontier_of_12": counts[m]} for m in order}}
        print(f"\n{stem}: label, rho at {default}, $/axis, variants on the frontier")
        for m in order:
            print(f"  {labels[m]:24s} {per_variant[default][m]['mean_rho']:.4f}  ${costs[m]:.3f}  {counts[m]:2d}/{len(variants)}")
    (out / "judge_tier_grid_12way.json").write_text(json.dumps(
        json_metadata(results, title=f"Batched judging grid under {len(variants)} projection variants", inputs=inputs), indent=1))


if __name__ == "__main__":
    main()
