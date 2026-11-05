"""The comparisons behind the response-judging decision (reports/judge_pareto/response_judging_plan.md § 4d).

Roger, 2026-11-05: plain stacking, 4 axes a call, the worst 20% of the axes judged one axis a call instead; both halves
of that choice marginal.  This script prints the readings the plan quotes, from the JSONs that
scripts/judge_response_pareto.py writes (no judging, no projections; seconds to run):

* On the 61 axes (judge_response_pareto_61.json, --stack_group g61):
  - leaving the worst out minus stacking every axis, at each k and share;
  - the shares against each other at each k, and the k against each other at each share;
  - 20% at k=4 against its neighbours;
  - the frontier of the measured points by share and k, with each panel's end points dropped (ends mislead: the
    cheapest point is always on a convex hull);
  - the chosen design's price and rho against one axis a call, and the axes it leaves out on each fold;
  - 40 answers a call against 20, at the same coverage and at equal cost.
* On the twelve axes (judge_response_pareto.json): plain stacking against labelled and axis by axis at each k, and
  plain k=2 against one axis a call, by slot.

"Equal cost": each arm's per-axis rho interpolated linearly in log cost along its measured points, on a grid of five
costs over the range both arms span, then averaged; the interval is a paired bootstrap over the axes (4,000
resamples, a fresh generator per comparison, seed 0), by slot (slot 0, slot 6: six projection variants each).  The
extrapolated points (1/3 and all, Spearman-Brown) carry no per-axis values, so they enter only as a mean ("with
extrapolated points").

Usage: uv run python scripts/response_stacking_checks.py [--judge luna|haiku|gemini] [--pareto_dir ...]
(the twelve-axis checks are Luna's and print only with --judge luna)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from results_analysis.judge_tier_cost_plot_pooled import pareto_frontier  # noqa: E402

LUNA = "gpt-5.6-luna"     # the judge read; main() sets it from the 61-axis JSON's config (--judge)
SHARES = (10, 15, 20, 25)
KS = (2, 4, 6)
N_BOOT = 4000
GRID = 5


class Points:
    """One Pareto JSON's points (judge_response_pareto.py's result['points'])."""

    def __init__(self, path: Path):
        d = json.load(open(path))["result"]
        self.config = d["config"]
        self.points: Dict[str, dict] = d["points"]
        self.variants: List[str] = d["config"]["variants"]
        self.left_out: Dict[str, Dict[str, int]] = d.get("left_out_folds", {})
        self.slot0 = np.array([v.startswith("s0") for v in self.variants])

    def has(self, arm: str) -> bool:
        return any(p["arm"] == arm for p in self.points.values())

    def curve(self, arm: str, measured_only: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """(log costs, {variant: array points x axes}); the extrapolated points as one column of mean rho."""
        ps = sorted([p for p in self.points.values() if p["arm"] == arm and (p["measured"] or not measured_only)],
                    key=lambda p: p["cost"])
        x = np.log([p["cost"] for p in ps])
        if measured_only:
            return x, {v: np.array([p["per_axis"][v] for p in ps]) for v in self.variants}
        return x, {v: np.array([[p["rho"][v]] for p in ps]) for v in self.variants}

    def equal_cost(self, a: str, b: str, measured_only: bool = True) -> Optional[Tuple[float, float, np.ndarray]]:
        """a minus b at equal cost: (lowest cost, highest cost, array grid x variants x axes), or None without overlap."""
        ca, cb = self.curve(a, measured_only), self.curve(b, measured_only)
        lo, hi = max(ca[0][0], cb[0][0]), min(ca[0][-1], cb[0][-1])
        if lo > hi:
            return None
        grid = np.linspace(lo, hi, GRID)
        n_axes = ca[1][self.variants[0]].shape[1]
        D = np.array([[[np.interp(g, ca[0], ca[1][v][:, j]) - np.interp(g, cb[0], cb[1][v][:, j]) for j in range(n_axes)]
                       for v in self.variants] for g in grid])
        return float(np.exp(lo)), float(np.exp(hi)), D

    def report(self, a: str, b: str, name: str) -> None:
        r = self.equal_cost(a, b)
        if r is None:
            print(f"  {name:38s} no common measured cost range")
            return
        lo, hi, D = r
        line = f"  {name:38s} ${lo:4.0f}-${hi:4.0f}: mean {D.mean():+.4f}"
        rng = np.random.default_rng(0)
        for lab, mask in (("slot 0", self.slot0), ("slot 6", ~self.slot0)):
            X = D[:, mask, :].mean(axis=(0, 1))                      # per axis
            bs = [X[rng.integers(0, len(X), len(X))].mean() for _ in range(N_BOOT)]
            line += f"  {lab} {X.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
        line += f"  ({(D.mean(axis=(0, 2)) > 0).sum()}/{len(self.variants)} variants)"
        rx = self.equal_cost(a, b, measured_only=False)
        if rx:
            line += f"; with extrapolated points {rx[2].mean():+.4f} over ${rx[0]:.0f}-${rx[1]:.0f}"
        print(line)

    def frontier_counts(self, figure: str, label_of, drop: int) -> collections.Counter:
        """Frontier points of the measured points of one figure, by label_of(arm), each panel's `drop` end points
        dropped, summed over the variants."""
        pts = {k: p for k, p in self.points.items() if p["measured"] and figure in p["figures"]}
        counts: collections.Counter = collections.Counter()
        for v in self.variants:
            keys = list(pts)
            idx = pareto_frontier([(pts[k]["cost"], 1 / (1 - pts[k]["rho"][v])) for k in keys])
            front = sorted((keys[i] for i in idx), key=lambda k: pts[k]["cost"])
            if drop:
                front = front[drop:-drop]
            counts.update(label_of(pts[k]["arm"]) for k in front)
        return counts


def share_of(arm: str) -> str:
    if arm.endswith("@k1"):
        return "one axis a call"
    return next((f"{s}%" for s in SHARES if f"x{s}k" in arm), "all stacked")


def stacked(share: Optional[int], k: int) -> str:
    return f"{LUNA}@rk_pnx{share}k{k}" if share else f"{LUNA}@rk_pnk{k}"


def mean_rho(p: dict, variants: List[str], mask=None) -> float:
    vs = variants if mask is None else [v for v, m in zip(variants, mask) if m]
    return float(np.mean([p["rho"][v] for v in vs]))


def sixty_one(P: Points) -> None:
    print("=" * 100)
    print(f"61 axes ({P.config.get('axes_file')}), GPT-5.6 Luna; equal cost, paired bootstrap over the axes")
    shares = [s for s in SHARES if P.has(stacked(s, 4))]
    print("\n(1) the worst left out, minus stacking every axis")
    for k in KS:
        for s in shares:
            P.report(stacked(s, k), stacked(None, k), f"k={k}: {s}% out minus all stacked")
    print("\n(2) the shares against each other, at each k")
    for k in KS:
        for i, a in enumerate(shares):
            for b in shares[:i]:
                P.report(stacked(a, k), stacked(b, k), f"k={k}: {a}% minus {b}%")
    print("\n(3) the k against each other, at each share")
    for s in shares:
        for a, b in ((2, 4), (4, 6), (2, 6)):
            P.report(stacked(s, a), stacked(s, b), f"{s}% out: k={a} minus k={b}")
    if 20 in shares:
        print("\n(4) the chosen design, 20% out at k=4, against its neighbours")
        for s, k in [(x, 4) for x in shares if x != 20] + [(20, 2), (20, 6), (10, 6), (25, 2)]:
            P.report(stacked(20, 4), stacked(s, k), f"20% k=4 minus {s}% k={k}")
    print("\n(5) the frontier of the measured points (one axis a call, every stacked arm), summed over the variants")
    for drop in (0, 1, 2):
        by_share = P.frontier_counts("luna61", share_of, drop)
        by_both = P.frontier_counts("luna61", lambda arm: f"{share_of(arm)} k={arm[-1]}" if "rk_pn" in arm else share_of(arm), drop)
        print(f"  {drop} end point(s) dropped per panel ({sum(by_share.values())} points): {dict(sorted(by_share.items()))}")
        print(f"      by share and k: {dict(sorted(by_both.items()))}")
    print("\n(6) the chosen design and one axis a call: price of a full run (200 October axes) and mean rho "
          "(12 variants; slot 0; slot 6)")
    for arm in (stacked(20, 4) if 20 in shares else None, f"{LUNA}@k1"):
        if arm is None:
            continue
        for cov in ("ninth x 3/3", "1/3", "all"):
            p = P.points[f"{arm}|{cov}"]
            print(f"  {arm.split('@')[1]:12s} {cov:12s} ${p['cost']:6,.0f}  {mean_rho(p, P.variants):.4f}  "
                  f"{mean_rho(p, P.variants, P.slot0):.4f}  {mean_rho(p, P.variants, ~P.slot0):.4f}")
    for key, lo in P.left_out.items():
        if key.endswith("x20k4"):
            print("  left out (folds of 9): " + ", ".join(f"{a} {n}" for a, n in sorted(lo.items(), key=lambda x: (-x[1], x[0]))))
    n40 = f"{LUNA}@n40"
    if P.has(n40):
        print("\n(7) 40 answers a call against 20 (one axis a call)")
        a, b = P.points[f"{n40}|ninth x 3/3"]["per_axis"], P.points[f"{LUNA}@k1|ninth x 3/3"]["per_axis"]
        D = np.array([[a[v][i] - b[v][i] for i in range(len(a[P.variants[0]]))] for v in P.variants])
        per_axis = D.mean(0)
        rng = np.random.default_rng(0)
        bs = [per_axis[rng.integers(0, len(per_axis), len(per_axis))].mean() for _ in range(N_BOOT)]
        print(f"  same coverage (the ninth): {per_axis.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, "
              f"{np.percentile(bs, 97.5):+.4f}]; worse on {(per_axis < 0).sum()}/{len(per_axis)} axes, "
              f"{(D.mean(1) < 0).sum()}/{len(P.variants)} variants; slot 0 {D[P.slot0].mean():+.4f}, "
              f"slot 6 {D[~P.slot0].mean():+.4f}")
        P.report(n40, f"{LUNA}@k1", "40 answers minus 20, equal cost")


def twelve(P: Points) -> None:
    print("=" * 100)
    print(f"twelve axes ({P.config.get('axes_file')}), GPT-5.6 Luna, every axis stacked; equal cost")
    print("\n(8) plain stacking against the other formats, at each k")
    for k in KS:
        for fmt, name in (("rk_pl", "labelled"), ("rk_pa", "axis by axis")):
            if P.has(f"{LUNA}@{fmt}k{k}") and P.has(f"{LUNA}@rk_pnk{k}"):
                P.report(f"{LUNA}@rk_pnk{k}", f"{LUNA}@{fmt}k{k}", f"k={k}: plain minus {name}")
    print("\n(9) plain k=2 against one axis a call")
    P.report(f"{LUNA}@rk_pnk2", f"{LUNA}@k1", "plain k=2 minus one axis a call")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pareto_dir", default="roger/axis_judge_experiments/response_pareto")
    ap.add_argument("--judge", default="luna", choices=("luna", "haiku", "gemini"))
    args = ap.parse_args(argv)
    global LUNA
    d = Path(args.pareto_dir)
    f61 = d / ("judge_response_pareto_61.json" if args.judge == "luna" else f"judge_response_pareto_61_{args.judge}.json")
    if f61.exists():
        P61 = Points(f61)
        LUNA = P61.config.get("model", LUNA)
        sixty_one(P61)
    if args.judge == "luna" and (d / "judge_response_pareto.json").exists():
        LUNA = "gpt-5.6-luna"
        twelve(Points(d / "judge_response_pareto.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
