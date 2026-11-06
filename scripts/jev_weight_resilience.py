"""Jev's weight in the four-judge blend of description and instruction judging: the evidence for 4 : 3 : 3 : 12.

Roger, 2026-11-04, on the mean-rho fit of 4 : 3 : 3 : 24: "That's not an ensemble, that's an addendum", and "part of
the reason for the ensemble was resilience": each judge has good and bad axes, which vary between models.  So the
weight is judged here by its worst axes as well as its mean.  Each judge's score of an entity is its mean over its
two passes, one each way (descriptions: seeds 1 and 2; instructions: rounds 0-4 and 5-9), on the balanced cells of
the 61 May axes; rho is Spearman against the projections, averaged over the 12 projection variants (slots 0 and 6,
layers 25 and 49, raw, soft-K and soft-shear), as in the weight fit.  Five parts, per mode:

1. The grid 4:3:3:X: mean rho with its difference from X = 24 and X = 12 (95% intervals from resampling the axes),
   the axes where the blend falls below the trio (Luna 4 : Haiku 3 : Gemini 3), and each axis's shortfall from the
   better of Jev alone and the trio, mean and worst.
2. Each judge's bad axes: Jev's gap to the trio per axis, its agreement between the two passes, and how often each
   judge alone trails the mean of the other three alone.
3. The tails: whether the per-axis gaps vary by more than noise (each axis's standard error from a paired bootstrap
   over its entities; Cochran's Q, I2 and the DerSimonian-Laird spread tau), and whether what is left after a normal
   spread is fat-tailed (standardised gaps beyond 3, kurtosis, Shapiro-Wilk, a Student-t fit), also with the
   judge noise of the pass split added, and without the three lowest axes.
4. The alternatives (Roger: "Where we have 4 results, can we do better by discarding the outlier and averaging the
   other 3?"): per entity, the score farthest from the other three dropped and the rest averaged 1:1:1:1, 1:1:1:2,
   4:3:3:12 or 4:3:3:24 (Roger: "Try 1:1:1:1 and 4:3:3:12 ... and 1:1:1:2 for the averaging"); the median of the
   four; per axis, the judge that agrees least with the other three dropped (chosen without the projections).
   Against the plain blends 1:1:1:1, 1:1:1:2, 4:3:3:12 and 4:3:3:24.
5. The effective shares: weight times each judge's spread over an axis's entities, which is what a weighted mean of
   raw scores gives each judge.

Writes roger/axis_judge_experiments/jev_weight/jev_weight_resilience.json (a provenance envelope) and prints the
tables.  No API calls; about ten minutes, most of it the bootstrap.  Evidence for
judge_score_combine.TIER_JUDGE_WEIGHTS_BY_MODE; the account is in reports/judge_pareto/jev_judging.md § "Jev's
weight".

Usage: uv run python scripts/jev_weight_resilience.py [--modes descriptions instructions] [--bootstrap 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_axis import json_metadata
from assistant_axis.entity_id import is_entity_id, parse_entity_id
from assistant_axis.judge_score_combine import TIER_JUDGE_DEFAULT_WEIGHT, TIER_JUDGE_WEIGHTS, combine_tier_judges
from assistant_axis.provenance import current_file_input, current_files_input
from results_analysis import judge_tier_robustness as rob
from results_analysis.judge_tier_cost_eval import cell_scores_path

JUDGES = ("gpt-5.6-luna", "claude-haiku-5-5", "gemini-3.1-flash-lite", "jev-1.14.0")
TRIO = JUDGES[:3]
JEV = JUDGES[3]
NAMES = dict(zip(JUDGES, ("Luna", "Haiku", "Gemini", "Jev")))
TRIO_W = np.array([TIER_JUDGE_WEIGHTS.get(m, TIER_JUDGE_DEFAULT_WEIGHT) for m in TRIO])     # 4 : 3 : 3
X_GRID = (3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32)
MODES = ("descriptions", "instructions")
DEFAULT_OUTPUT = Path("roger/axis_judge_experiments/jev_weight/jev_weight_resilience.json")


def blend_w(x: float) -> np.ndarray:
    """4 : 3 : 3 : x over (Luna, Haiku, Gemini, Jev)."""
    return np.append(TRIO_W, float(x))


def cells(mode: str, judge: str, passes: Sequence[int]) -> List[str]:
    """The balanced cells of ``judge``'s ``passes`` (1, 2): instruction pass p is rounds 5(p-1) .. 5p-1."""
    if mode == "instructions":
        rs = [r for p in passes for r in range(5 * (p - 1), 5 * p)]
        return [f"{JEV}@ji_noul_relative_n160d{r}" if judge == JEV else f"{judge}@i1_p0n20d{r}" for r in rs]
    return [f"{JEV}@jd_noul_relative_n160d{p}" if judge == JEV else f"{judge}@b10n20d{p}" for p in passes]


# ---------------------------------------------------------------------------
# Pure helpers (tested in scripts/tests/test_jev_weight_resilience.py)
# ---------------------------------------------------------------------------

def combine_rule(values: np.ndarray, weights: np.ndarray, drop: str = "none") -> Tuple[float, Optional[int]]:
    """One entity's score from its judges' scores (NaN where a judge did not score it): the mean weighted by
    ``weights`` over the judges present, with ``drop`` "mean" or "median" first removing, when every judge scored
    it, the score farthest from the mean of all (the same as farthest from the mean of the others) or from their
    median.  Returns ``(score, index dropped or None)``."""
    present = np.isfinite(values)
    if not present.any():
        return float("nan"), None
    if drop != "none" and present.all():
        centre = values.mean() if drop == "mean" else np.median(values)
        i = int(np.argmax(np.abs(values - centre)))
        keep = np.arange(len(values)) != i
        return float((weights[keep] * values[keep]).sum() / weights[keep].sum()), i
    return float((weights[present] * values[present]).sum() / weights[present].sum()), None


def combine_entities(per_judge: Sequence[Mapping[str, float]], weights: np.ndarray, drop: str = "none",
                     median: bool = False) -> Tuple[Dict[str, float], Dict[str, Optional[int]]]:
    """Every entity any judge scored: ``combine_rule`` (or, with ``median``, the median of the scores present).
    Returns the scores and, per entity, the judge index dropped."""
    out, dropped = {}, {}
    for k in sorted(set().union(*per_judge)):
        v = np.array([d.get(k, np.nan) for d in per_judge], dtype=float)
        if median:
            out[k], dropped[k] = float(np.median(v[np.isfinite(v)])), None
        else:
            out[k], dropped[k] = combine_rule(v, weights, drop)
    return out, dropped


def rank_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman along the last axis (average ranks), broadcasting."""
    rx = stats.rankdata(x, axis=-1)
    ry = stats.rankdata(y, axis=-1)
    rx = rx - rx.mean(-1, keepdims=True)
    ry = ry - ry.mean(-1, keepdims=True)
    return (rx * ry).sum(-1) / np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))


def least_agreeing(M: np.ndarray) -> Tuple[int, List[float]]:
    """The row (judge) of ``M`` [judges, entities] whose Spearman with the mean of the other rows is lowest."""
    agree = [float(stats.spearmanr(M[i], np.delete(M, i, 0).mean(0))[0]) for i in range(len(M))]
    return int(np.argmin(agree)), agree


def shortfall(blend: np.ndarray, jev: np.ndarray, trio: np.ndarray) -> dict:
    """Per-axis rho of a blend against the better of Jev alone and the trio, and against the trio."""
    s = np.maximum(jev, trio) - blend
    return {"mean_shortfall": float(s.mean()), "worst_shortfall": float(s.max()),
            "below_trio": int((blend < trio - 1e-12).sum()), "worst_vs_trio": float((blend - trio).min())}


def mean_ci(d: np.ndarray, rng: np.random.Generator, n_boot: int = 4000) -> Tuple[float, float, float]:
    """Mean of ``d`` with a 95% percentile interval from resampling its entries."""
    b = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def meta_analysis(gap: np.ndarray, se: np.ndarray) -> dict:
    """Fixed-effect mean, Cochran's Q, I2, DerSimonian-Laird tau, random-effects mean, and the standardised gaps:
    ``z_noise`` if the gaps differed only by their noise, ``z_tau`` with a normal spread tau added."""
    w = 1 / se ** 2
    mu_fe = float((w * gap).sum() / w.sum())
    Q = float((w * (gap - mu_fe) ** 2).sum())
    df = len(gap) - 1
    tau2 = max(0.0, (Q - df) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (se ** 2 + tau2)
    mu = float((wr * gap).sum() / wr.sum())
    return {"mu_fixed": mu_fe, "Q": Q, "df": df, "Q_p": float(stats.chi2.sf(Q, df)),
            "I2": max(0.0, (Q - df) / Q) if Q > 0 else 0.0, "tau": float(np.sqrt(tau2)), "mu_random": mu,
            "z_noise": (gap - mu_fe) / se, "z_tau": (gap - mu) / np.sqrt(se ** 2 + tau2)}


def tail_summary(z: np.ndarray) -> dict:
    """Counts beyond 2 and 3 against a normal's expectation, skew, excess kurtosis, Shapiro-Wilk p, t-fit df."""
    n = len(z)
    t_df = stats.t.fit(z)[0] if n >= 8 else float("nan")
    return {"beyond_2": int((np.abs(z) > 2).sum()), "normal_beyond_2": round(n * 2 * stats.norm.sf(2), 2),
            "beyond_3": int((np.abs(z) > 3).sum()), "normal_beyond_3": round(n * 2 * stats.norm.sf(3), 2),
            "min": float(z.min()), "max": float(z.max()), "skew": float(stats.skew(z)),
            "excess_kurtosis": float(stats.kurtosis(z)), "shapiro_p": float(stats.shapiro(z).pvalue),
            "t_df": float(t_df)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def bare(k: str) -> str:
    return parse_entity_id(k).name if is_entity_id(k) else k


def mean_of(runs: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = set().union(*runs)
    return {k: float(np.mean([r[k] for r in runs if k in r])) for k in keys}


class Mode:
    """One mode's scores (each judge's mean over passes (1, 2), (1,) and (2,)) and projections."""

    def __init__(self, mode: str, axes, E: Path, data_dir: Path, slices, variants, kinds):
        self.mode, self.axes, self.vkeys = mode, axes, [v.key for v in variants]
        self.paths: List[Path] = []
        self.sc: Dict[Tuple[str, str, Tuple[int, ...]], Dict[str, float]] = {}
        for ax in axes:
            for m in JUDGES:
                runs = {}
                for c in cells(mode, m, (1, 2)):
                    p = cell_scores_path(E / f"{ax.axis_dir}{rob.TIER_EVAL_SUFFIX}" / c)
                    runs[c] = rob.lift_scores(rob.numeric_scores(rob._unwrap(json.loads(p.read_text()))), kinds)
                    self.paths.append(p)
                for ps in ((1, 2), (1,), (2,)):
                    self.sc[ax.axis_dir, m, ps] = mean_of([runs[c] for c in cells(mode, m, ps)])
        names = {ax.axis_dir: set().union(*(self.sc[ax.axis_dir, m, (1, 2)] for m in JUDGES)) for ax in axes}
        self.proj = rob.project_variants(slices, axes, variants, names, data_dir)

    def judges(self, ax, ps=(1, 2)) -> List[Dict[str, float]]:
        return [self.sc[ax.axis_dir, m, ps] for m in JUDGES]

    def rho(self, ax, scores: Mapping[str, float]) -> float:
        """rho averaged over the 12 variants (poles left out by name)."""
        return float(np.mean([rob.cell_rho(dict(scores), self.proj[ax.axis_dir][vk], (ax.pos, ax.neg))[0]
                              for vk in self.vkeys]))

    def weighted(self, ax, w: np.ndarray, ps=(1, 2)) -> Dict[str, float]:
        js = [(m, s) for m, s, x in zip(JUDGES, self.judges(ax, ps), w) if x > 0]
        return combine_tier_judges({m: s for m, s in js}, {m: float(x) for m, x in zip(JUDGES, w) if x > 0})


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def analyse(md: Mode, n_boot: int, rng: np.random.Generator) -> dict:
    axes = md.axes
    names = [ax.axis_dir for ax in axes]
    rho: Dict[str, np.ndarray] = {}

    def put(label, per_axis):
        rho[label] = np.array(per_axis)

    for i, m in enumerate(JUDGES):
        put(NAMES[m], [md.rho(ax, md.sc[ax.axis_dir, m, (1, 2)]) for ax in axes])
    trio_w = np.append(TRIO_W, 0.0)
    put("trio", [md.rho(ax, md.weighted(ax, trio_w)) for ax in axes])
    for x in X_GRID:
        put(f"4:3:3:{x}", [md.rho(ax, md.weighted(ax, blend_w(x))) for ax in axes])
    jev, trio = rho["Jev"], rho["trio"]

    # 1. the grid
    grid = []
    for x in X_GRID:
        r = rho[f"4:3:3:{x}"]
        grid.append({"x": x, "mean_rho": float(r.mean()), "vs_24": mean_ci(r - rho["4:3:3:24"], rng),
                     "vs_12": mean_ci(r - rho["4:3:3:12"], rng), **shortfall(r, jev, trio)})

    # 2. bad axes: Jev against the trio, pass by pass, and each judge alone against the other three alone
    gap1 = np.array([md.rho(ax, md.sc[ax.axis_dir, JEV, (1,)]) - md.rho(ax, md.weighted(ax, trio_w, (1,))) for ax in axes])
    gap2 = np.array([md.rho(ax, md.sc[ax.axis_dir, JEV, (2,)]) - md.rho(ax, md.weighted(ax, trio_w, (2,))) for ax in axes])
    gap = jev - trio
    own_vs_others = {}
    for m in JUDGES:
        d = rho[NAMES[m]] - np.mean([rho[NAMES[o]] for o in JUDGES if o != m], axis=0)
        own_vs_others[NAMES[m]] = {"axes_worse": int((d < 0).sum()), "worst": float(d.min())}
    bad = {"gap_mean": float(gap.mean()), "jev_below_trio": int((gap < 0).sum()),
           "pass_split_pearson": float(stats.pearsonr(gap1, gap2)[0]),
           "pass_split_spearman": float(stats.spearmanr(gap1, gap2)[0]),
           "below_on_both_passes": int(((gap1 < 0) & (gap2 < 0)).sum()), "each_judge_vs_other_three": own_vs_others,
           "axes": sorted(({"axis": a, "gap": float(g), "gap_pass1": float(g1), "gap_pass2": float(g2),
                            "x12_minus_trio": float(rho["4:3:3:12"][i] - trio[i]),
                            "x24_minus_trio": float(rho["4:3:3:24"][i] - trio[i])}
                           for i, (a, g, g1, g2) in enumerate(zip(names, gap, gap1, gap2))), key=lambda r: r["gap"])}

    # 3. the tails: per-axis SE of the gap from a paired bootstrap over the axis's entities (the same resample for
    # both judges and all 12 projections), with every judge on the entities both have
    tgap, se = [], []
    for ax in axes:
        j = md.sc[ax.axis_dir, JEV, (1, 2)]
        t = md.weighted(ax, trio_w)
        P_ax = md.proj[ax.axis_dir]
        keys = sorted(k for k in set(j) & set(t) if bare(k) not in (ax.pos, ax.neg)
                      and all(np.isfinite(P_ax[vk].get(k, np.nan)) for vk in md.vkeys))
        J = np.array([j[k] for k in keys])
        T = np.array([t[k] for k in keys])
        P = np.array([[P_ax[vk][k] for k in keys] for vk in md.vkeys])
        tgap.append(float((rank_corr(J, P) - rank_corr(T, P)).mean()))
        idx = rng.integers(0, len(keys), (n_boot, len(keys)))
        gb = (rank_corr(J[idx][None], P[:, idx]) - rank_corr(T[idx][None], P[:, idx])).mean(0)
        se.append(float(gb.std(ddof=1)))
    tgap, se = np.array(tgap), np.array(se)
    judge_noise = float(np.sqrt(np.mean((gap1 - gap2) ** 2) / 4))     # the 2x gap's judge-noise SE, from the split

    def meta(g, s):
        ma = meta_analysis(g, s)
        return {k: v for k, v in ma.items() if not k.startswith("z_")} | {
            "z_noise": tail_summary(ma["z_noise"]), "z_tau": tail_summary(ma["z_tau"])}

    lowest3 = np.argsort(tgap)[3:]
    se_c = np.sqrt(se ** 2 + judge_noise ** 2)
    ma = meta_analysis(tgap, se)
    tails = {"gap_sd_across_axes": float(tgap.std(ddof=1)), "se_median": float(np.median(se)),
             "judge_noise_se_from_pass_split": judge_noise,
             "entity_bootstrap": meta(tgap, se), "with_pass_split_noise": meta(tgap, se_c),
             "with_pass_split_noise_without_lowest_3": meta(tgap[lowest3], se_c[lowest3]),
             "axes": sorted(({"axis": a, "gap": float(g), "se": float(s), "z_tau": float(z)}
                             for a, g, s, z in zip(names, tgap, se, ma["z_tau"])), key=lambda r: r["gap"])}

    # 4. the alternatives
    eq, w1112 = np.ones(4), np.array([1.0, 1.0, 1.0, 2.0])
    alts = {"1:1:1:1": ("plain", eq), "1:1:1:2": ("plain", w1112),
            "drop outlier, 1:1:1:1 on the rest": ("mean", eq), "drop outlier, 1:1:1:2 on the rest": ("mean", w1112),
            "drop outlier, 4:3:3:12 on the rest": ("mean", blend_w(12)),
            "drop outlier, 4:3:3:24 on the rest": ("mean", blend_w(24)), "median of 4": ("median4", eq)}
    # (dropping the score farthest from the median instead drops the same score whenever there are four: see the
    # tests)
    drop_share = {}
    for label, (how, w) in alts.items():
        per = []
        counts = np.zeros(4)
        for ax in axes:
            if how == "plain":
                s, dr = combine_entities(md.judges(ax), w)
            elif how == "median4":
                s, dr = combine_entities(md.judges(ax), w, median=True)
            else:
                s, dr = combine_entities(md.judges(ax), w, drop=how)
            for k, i in dr.items():
                if i is not None and bare(k) not in (ax.pos, ax.neg):
                    counts[i] += 1
            per.append(md.rho(ax, s))
        put(label, per)
        if counts.sum():
            drop_share[label] = dict(zip([NAMES[m] for m in JUDGES], (counts / counts.sum()).round(4).tolist()))
    axis_drops = []
    for label, w in (("axis: drop the least-agreeing judge, 1:1:1:1 on the rest", eq),
                     ("axis: drop the least-agreeing judge, 1:1:1:2 on the rest", w1112),
                     ("axis: drop the least-agreeing judge, 4:3:3:12 on the rest", blend_w(12))):
        per = []
        for ax in axes:
            keys = sorted(k for k in set.intersection(*(set(s) for s in md.judges(ax))) if bare(k) not in (ax.pos, ax.neg))
            M = np.array([[s[k] for k in keys] for s in md.judges(ax)])
            i, agree = least_agreeing(M)
            if w is eq:
                axis_drops.append({"axis": ax.axis_dir, "dropped": NAMES[JUDGES[i]],
                                   "agreement": dict(zip([NAMES[m] for m in JUDGES], np.round(agree, 4).tolist()))})
            per.append(md.rho(ax, md.weighted(ax, np.where(np.arange(4) == i, 0.0, w))))
        put(label, per)
    alternatives = []
    for label in ["4:3:3:24", "4:3:3:12"] + list(alts) + [k for k in rho if k.startswith("axis:")] + ["Jev", "trio"]:
        r = rho[label]
        alternatives.append({"method": label, "mean_rho": float(r.mean()), "vs_24": mean_ci(r - rho["4:3:3:24"], rng),
                             "vs_12": mean_ci(r - rho["4:3:3:12"], rng), **shortfall(r, jev, trio)})

    # 5. the effective shares: weight x each judge's sd over the axis's entities
    shares = {}
    for x in (12, 24):
        w = blend_w(x)
        per_axis = []
        for ax in axes:
            keys = sorted(set.intersection(*(set(s) for s in md.judges(ax))))
            sd = np.array([np.std([s[k] for k in keys]) for s in md.judges(ax)])
            per_axis.append(w * sd / (w * sd).sum())
        per_axis = np.array(per_axis)
        b = {k: [] for k in JUDGES}
        for ax in axes:
            bl = md.weighted(ax, w)
            keys = sorted(set.intersection(*(set(s) for s in md.judges(ax))))
            for m, s in zip(JUDGES, md.judges(ax)):
                b[m].append(stats.spearmanr([s[k] for k in keys], [bl[k] for k in keys])[0])
        shares[f"4:3:3:{x}"] = {NAMES[m]: {"nominal": float(w[i] / w.sum()), "effective": float(per_axis[:, i].mean()),
                                           "spearman_with_blend": float(np.mean(b[m]))} for i, m in enumerate(JUDGES)}
    return {"n_axes": len(axes), "grid": grid, "bad_axes": bad, "tails": tails, "alternatives": alternatives,
            "entity_drop_share": drop_share, "axis_drops": axis_drops, "effective_shares": shares,
            "per_axis_rho": {k: dict(zip(names, map(float, v))) for k, v in rho.items()}}


def print_report(mode: str, r: dict) -> None:
    print(f"\n=== {mode}, {r['n_axes']} axes, rho averaged over the 12 variants")
    print("  4:3:3:X   mean rho   vs 24 [95%]                vs 12 [95%]                below trio  mean / worst shortfall")
    for g in r["grid"]:
        a, b = g["vs_24"], g["vs_12"]
        print(f"  {g['x']:7d}   {g['mean_rho']:.4f}   {a[0]:+.4f} [{a[1]:+.4f}, {a[2]:+.4f}]   "
              f"{b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]   {g['below_trio']:3d}         {g['mean_shortfall']:.4f} / {g['worst_shortfall']:.4f}")
    b = r["bad_axes"]
    print(f"  Jev below the trio on {b['jev_below_trio']} axes; pass 1 vs pass 2 gaps: Pearson {b['pass_split_pearson']:.2f}, "
          f"below on both {b['below_on_both_passes']}; each judge alone vs the other three alone: "
          + ", ".join(f"{k} {v['axes_worse']} worse (worst {v['worst']:+.3f})" for k, v in b["each_judge_vs_other_three"].items()))
    print("  Jev's five worst: " + "; ".join(f"{x['axis']} {x['gap']:+.3f}" for x in b["axes"][:5]))
    t = r["tails"]
    for k in ("entity_bootstrap", "with_pass_split_noise", "with_pass_split_noise_without_lowest_3"):
        m = t[k]
        z = m["z_tau"]
        print(f"  tails, {k}: Q {m['Q']:.0f} on {m['df']} (p {m['Q_p']:.1e}), I2 {m['I2']:.2f}, tau {m['tau']:.4f}; "
              f"beyond 3 after tau {z['beyond_3']} (normal {z['normal_beyond_3']}), excess kurtosis {z['excess_kurtosis']:.1f}, "
              f"Shapiro p {z['shapiro_p']:.3f}, t df {z['t_df']:.1f}")
    print("  method                                                     mean rho   vs 24 [95%]                below trio  worst shortfall")
    for a in r["alternatives"]:
        v = a["vs_24"]
        print(f"  {a['method']:58s} {a['mean_rho']:.4f}   {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]   {a['below_trio']:3d}         {a['worst_shortfall']:.4f}")
    print("  dropped judge, share of entities: " + "; ".join(f"{k}: {v}" for k, v in r["entity_drop_share"].items() if "1:1:1:1" in k))
    for x, s in r["effective_shares"].items():
        print(f"  {x} effective shares: " + ", ".join(f"{k} {v['effective']:.2f} (nominal {v['nominal']:.2f})" for k, v in s.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--axes_file", default="scripts/tier_eval_axes.tsv")
    p.add_argument("--experiments_dir", default=str(rob.DEFAULT_EXPERIMENTS_DIR))
    p.add_argument("--data_dir", default=str(rob.DEFAULT_DATA_DIR))
    p.add_argument("--cache_dir", default=None, help="the projection slice cache (default: <experiments_dir>/"
                                                     "_tier_eval_scratch/robustness_cache, judge_tier_robustness.py's)")
    p.add_argument("--bootstrap", type=int, default=1000, help="entity resamples per axis for the gap's SE")
    p.add_argument("--seed", type=int, default=20261104)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args(argv)
    t0 = time.time()
    E, D = Path(args.experiments_dir), Path(args.data_dir)
    cache = Path(args.cache_dir) if args.cache_dir else E / "_tier_eval_scratch" / "robustness_cache"
    axes = rob.read_axes_tsv(Path(args.axes_file))
    variants = rob.build_variants(rob.DEFAULT_SLOTS, rob.DEFAULT_LAYERS, list(rob.REGIMES),
                                  K=rob.DEFAULT_SOFT_K, L=rob.DEFAULT_SOFT_SHEAR_L)
    meta = {"version": rob.CACHE_VERSION, "fingerprint": rob.data_fingerprint(D), "ca_kind": "combined"}
    slices = rob.load_slices(cache, rob.DEFAULT_SLOTS, rob.DEFAULT_LAYERS, meta, True)
    if slices is None:
        raise SystemExit(f"no current slice cache under {cache}: run judge_tier_robustness.py once to build it")
    kinds = rob.vector_kinds(D)
    rng = np.random.default_rng(args.seed)
    inputs = [rob._subtree_input(D, "traits/vectors", "traits_vectors"), rob._subtree_input(D, "roles/vectors", "roles_vectors"),
              current_file_input("axes_file", Path(args.axes_file))]
    report = {}
    for mode in args.modes:
        md = Mode(mode, axes, E, D, slices, variants, kinds)
        inputs.append(current_files_input(dep_key=f"{mode}_scores", paths=md.paths))
        report[mode] = analyse(md, args.bootstrap, rng)
        print_report(mode, report[mode])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_metadata(report, inputs=inputs, title="Jev's weight: the 4:3:3:X grid, bad axes, "
                                                                          "tails and alternatives"), indent=1))
    print(f"\nwrote {out} ({time.time() - t0:.0f} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
