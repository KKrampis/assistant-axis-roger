"""Do the judges respond to the grid's settings in the same way?  Two figures from the rho table of
judge_tier_robustness.py (no API calls):

* judge_tier_grid_shapes.png: for each dimension (batch size, passes, reasoning), each judge's change
  in mean rho from its own base setting (N=10, 2 passes, no reasoning) at the default variant, with a
  95% paired bootstrap interval over the axes, and its cost as a multiple of the base's.
* judge_tier_grid_own_frontier.png: in how many of the 12 projection variants each setting is on that
  judge's own Pareto frontier, the frontier taken among that judge's settings up to 3 passes.

Usage: uv run python scripts/judge_tier_grid_shapes.py --robustness_json OUT/judge_tier_robustness.json --output_dir DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from assistant_axis import json_metadata, png_metadata
from assistant_axis.provenance import CACHE_POLICIES, load_and_register
from results_analysis import judge_tier_robustness as rob
from scripts.judge_tier_grid_12way import JUDGES, dimension_arms

BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 20261102
#: Colour and marker by judge.  The two OpenAI judges need telling apart, so GPT-6 Luna takes the one
#: colour of the Pareto plots' set this figure has no other use for (validated for colour blindness in
#: this order); every series also has its own marker.
STYLE = {"Luna": ("#7FB069", "o"), "GPT-6": ("#8E6BBF", "s"), "Haiku": ("#D9636B", "^"), "Gemini": ("#5B8DEF", "D")}
LONG = {"Luna": "GPT-5.6 Luna", "GPT-6": "GPT-6 Luna", "Haiku": "Haiku 5.5", "Gemini": "Gemini 3.1"}
DIMS = (("batch_size", "Batch size N (2 passes, no reasoning)", ("N=5", "N=7", "N=10", "N=15", "N=20")),
        ("passes", "Passes (N=10, no reasoning)", ("1x", "2x", "3x")),
        ("reasoning", "Reasoning effort (N=10, 2 passes)", ("none", "low", "medium")))
SETTINGS = (("batch_size", "N=5"), ("batch_size", "N=7"), ("batch_size", "N=10"), ("batch_size", "N=15"), ("batch_size", "N=20"),
            ("passes", "1x"), ("passes", "3x"), ("reasoning", "low"), ("reasoning", "medium"))
SETTING_LABEL = {"N=10": "N=10, 2 passes, no reasoning (base)", "N=5": "N=5", "N=7": "N=7", "N=15": "N=15", "N=20": "N=20",
                 "1x": "1 pass", "3x": "3 passes", "low": "low reasoning", "medium": "medium reasoning"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e3"


def paired(rho: dict, cell: str, base: str, vkey: str, rng: np.random.Generator):
    """Mean over the axes of rho(cell) - rho(base), with its 95% bootstrap interval."""
    axes = sorted(a for a in rho[base] if a in rho[cell] and np.isfinite(rho[cell][a].get(vkey, np.nan)))
    d = np.array([rho[cell][a][vkey] - rho[base][a][vkey] for a in axes])
    m = d[rng.integers(0, len(d), (BOOTSTRAP_DRAWS, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robustness_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--axes_label", default="30 development axes")
    ap.add_argument("--cache-policy", dest="cache_policy", choices=CACHE_POLICIES, default="warn",
                    help="what to do when the rho table's recorded inputs have changed: strict | warn (default) | rebuild | off")
    args = ap.parse_args()
    inputs: list = []
    payload, _spec, _check = load_and_register(Path(args.robustness_json), dep_key="robustness", inputs=inputs, policy=args.cache_policy)
    d = rob._unwrap(payload); cfg = d["config"]; out = Path(args.output_dir); rho = d["rho"]
    costs = {m: v["mean_cost"] for m, v in d["models"].items()}
    default = cfg["default_variant"]
    variants = rob.build_variants(cfg["slots"], cfg["layers"], cfg["regimes"], K=cfg["soft_K"], L=cfg["shear_L"])
    arms = {name: dimension_arms(base, plain, first, min(max_passes, 3)) for name, base, plain, first, max_passes in JUDGES}
    base_cell = {name: arms[name]["batch_size"]["N=10"] for name in arms}
    missing = [c for a in arms.values() for dim in a.values() for c in dim.values() if c not in rho]
    if missing:
        raise SystemExit(f"the rho table has no {missing}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # the shapes
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.4), sharey="row", gridspec_kw={"height_ratios": [3, 2], "width_ratios": [5, 3.4, 3.4]})
    table: dict = {}
    for col, (dim, title, order) in enumerate(DIMS):
        ax_rho, ax_cost = axs[0][col], axs[1][col]
        xs = np.arange(len(order))
        for j, name in enumerate(arms):
            color, marker = STYLE[name]
            dodge = (j - 1.5) * 0.045
            cells = [arms[name][dim][a] for a in order]
            stats = [paired(rho, c, base_cell[name], default, rng) for c in cells]
            means = [s[0] for s in stats]
            ax_rho.errorbar(xs + dodge, means, yerr=[[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]], color=color,
                            marker=marker, ms=7, lw=2, elinewidth=1, capsize=0, label=LONG[name], zorder=3)
            ratio = [costs[c] / costs[base_cell[name]] for c in cells]
            ax_cost.plot(xs + dodge, ratio, color=color, marker=marker, ms=7, lw=2, zorder=3)
            table.setdefault(name, {})[dim] = {a: {"cell": c, "drho": s[0], "ci": [s[1], s[2]], "cost_ratio": r, "cost": costs[c]}
                                               for a, c, s, r in zip(order, cells, stats, ratio)}
        ax_rho.axhline(0, color=MUTED, lw=1, zorder=1)
        ax_cost.axhline(1, color=MUTED, lw=1, zorder=1)
        ax_rho.set_title(title, fontsize=11, color=INK)
        for ax in (ax_rho, ax_cost):
            ax.set_xticks(xs); ax.set_xticklabels([a.replace("N=", "") for a in order]); ax.set_xlim(-0.35, len(order) - 0.65)
            ax.grid(True, axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, length=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
    axs[0][0].set_ylabel("change in mean ρ from the judge's base setting", color=INK)
    axs[1][0].set_ylabel("cost, multiple of the base", color=INK)
    axs[0][2].legend(loc="lower right", frameon=False, fontsize=9.5)
    title = f"Do the judges respond alike? DESCRIPTION judging, {args.axes_label}, {default}"
    fig.suptitle(f"{title}\nbase = N=10, 2 passes, no reasoning; bars are 95% paired bootstrap intervals over axes; "
                 "points dodged sideways so the bars can be read", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "judge_tier_grid_shapes.png", dpi=150, bbox_inches="tight", metadata=png_metadata(title=title, inputs=inputs))
    plt.close(fig)

    # each judge's own frontier, across the variants
    counts = np.zeros((len(SETTINGS), len(arms)), dtype=int)
    for jj, name in enumerate(arms):
        cells = [arms[name][dim][a] for dim, a in SETTINGS]
        frontiers = {v.key: rob.summarise_variant(rho, costs, cells, v.key)[1] for v in variants}
        c = rob.frontier_counts(frontiers, cells)
        counts[:, jj] = [c[m] for m in cells]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.imshow(counts, cmap="Blues", vmin=0, vmax=len(variants) * 1.35, aspect="auto")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            n = counts[i, j]
            ax.text(j, i, str(n) if n else "·", ha="center", va="center", fontsize=11, color="white" if n >= 9 else INK)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([LONG[n] for n in arms])
    ax.set_yticks(range(len(SETTINGS))); ax.set_yticklabels([SETTING_LABEL[a] for _, a in SETTINGS])
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(arms)), minor=True); ax.set_yticks(np.arange(-.5, len(SETTINGS)), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    title2 = f"Variants of {len(variants)} where a setting is on the judge's own frontier"
    fig.suptitle(f"{title2}\nDESCRIPTION judging, {args.axes_label}; settings up to 3 passes; · = never", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "judge_tier_grid_own_frontier.png", dpi=150, bbox_inches="tight", metadata=png_metadata(title=title2, inputs=inputs))
    plt.close(fig)

    (out / "judge_tier_grid_shapes.json").write_text(json.dumps(json_metadata(
        {"shapes": table, "own_frontier": {name: {a: int(counts[i, j]) for i, (_, a) in enumerate(SETTINGS)} for j, name in enumerate(arms)}},
        title=title, inputs=inputs), indent=1))
    print(f"{'':10s}" + "".join(f"{n:>26s}" for n in arms))
    for dim, _, order in DIMS:
        for a in order:
            print(f"{a:10s}" + "".join(f"{table[n][dim][a]['drho']:+8.4f} x{table[n][dim][a]['cost_ratio']:.2f} ${table[n][dim][a]['cost']:.3f} " .rjust(26) for n in arms))
    print("\nown frontier, of", len(variants))
    for i, (_, a) in enumerate(SETTINGS):
        print(f"{a:10s}" + "".join(f"{counts[i, j]:>26d}" for j in range(len(arms))))


if __name__ == "__main__":
    main()
