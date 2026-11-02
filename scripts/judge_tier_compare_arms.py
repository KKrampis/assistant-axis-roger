"""Compare judging arms on a set of axes (slot 6, layer 25, raw): each arm is a list of cells whose
scores are averaged per entity, so an arm can be several passes of one judge or a blend of judges.
Prints each arm's mean rho and cost per axis at list price, and for each --compare A:B the mean
per-axis difference with a 95% bootstrap interval over axes.  Reads scores and projections; no API calls.

Usage: uv run python scripts/judge_tier_compare_arms.py --axes_file scripts/tier_eval_axes_holdout31.tsv \\
           --arm "Luna=gpt-5.6-luna@none@b10n20s1,gpt-5.6-luna@none@b10n20s2" \\
           --arm "Luna+Haiku=gpt-5.6-luna@none@b10n20s1,gpt-5.6-luna@none@b10n20s2,claude-haiku-5-5@b10n20s1,claude-haiku-5-5@b10n20s2" \\
           --compare "Luna+Haiku:Luna\""""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from results_analysis.judge_tier_cost_plot_pooled import load_measured_cost

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--experiments_dir", default="roger/axis_judge_experiments")
ap.add_argument("--axes_file", default="scripts/tier_eval_axes.tsv")
ap.add_argument("--arm", action="append", required=True, metavar="NAME=CELL,CELL,...")
ap.add_argument("--compare", action="append", default=[], metavar="A:B")
ap.add_argument("--seed", type=int, default=20261102)
args = ap.parse_args()
E = Path(args.experiments_dir)
axes = [l.split()[0] for l in open(args.axes_file) if l.strip() and not l.startswith("#")]
arms = {a.rsplit("=", 1)[0]: a.rsplit("=", 1)[1].split(",") for a in args.arm}
proj = {}
for a in axes:
    d = json.load(open(E / a / "gpt" / "projections.json")); d = d.get("result", d)
    proj[a] = {name: v["raw"] for name, v in d["6"].items()}


def arm_rhos(cells):
    out = []
    for a in axes:
        loaded = [json.load(open(E / f"{a}_tier_eval" / c / "scores_descriptions.json"))["result"] for c in cells]
        blend = {}
        for name in proj[a]:
            vals = [s[name] for s in loaded if s.get(name) is not None]
            if vals:
                blend[name] = float(np.mean(vals))
        out.append(spearmanr(list(blend.values()), [proj[a][n] for n in blend])[0])
    return np.array(out)


rhos = {k: arm_rhos(c) for k, c in arms.items()}
width = max(len(k) for k in arms)
for k, cells in arms.items():
    cost = sum(np.mean([load_measured_cost(E / f"{a}_tier_eval" / c / "usage.json", "list") for a in axes]) for c in cells)
    print(f"{k:{width}s}  rho {np.nanmean(rhos[k]):.4f}  ${cost:.3f}/axis")
rng = np.random.default_rng(args.seed)
for spec in args.compare:
    x, y = spec.split(":")
    d = rhos[x] - rhos[y]
    means = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    print(f"{x} - {y}: {d.mean():+.4f} [{np.percentile(means, 2.5):+.4f}, {np.percentile(means, 97.5):+.4f}] over {len(d)} axes")
