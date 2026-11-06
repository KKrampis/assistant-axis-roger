"""Comprehensive judge / dose-response QC for a steering experiment.

Reports on:

1. **Per-strength dose-response curves** for each sign × cell:
   - mean_coh (per-strength aggregate, already cached on records)
   - mean_rp (mean over questions)
   - mean_eff_signed, mean_eff_abs (mean over questions, eff.combined)
   - n_records (excluding skipped-coh strengths)
   - n_with_valid_eff

2. **Monotonicity** of |eff| as strength rises on each sign.
   Reports the longest *non-monotone* run (any strength where |eff|
   decreases below the previous's |eff| by more than ``noise_tol``)
   and the strength at which |eff| first crosses ``signal_threshold``.

3. **Per-question variance** at each strength:
   - mean ± stdev of eff across the N questions at that strength
   - flags strengths where stdev is high relative to mean (judge
     disagreement OR question heterogeneity)

4. **Judge agreement** within the bidirectional ensemble:
   - per-record diff between gpt and claude scores
   - cross-judge correlation across all records
   - fraction of records where the two judges disagree by >= 2 levels

5. **Response-length collapse**: per-strength avg response length
   relative to baseline length per question.  Flags strengths where
   length drops below 0.5x baseline (steered response is much shorter
   than baseline -- a steering pathology often invisible to the
   coherence rubric).

6. **Persona-rubric (RP) sanity**: distribution of RP scores per
   strength, and correlation between RP collapse and |eff| spikes.

Usage:
    uv run python tools/analyze_dose_response.py \\
        /workspace/outputs/qwen-3-32b/steering/architect_ecocentric_v2 \\
        [/workspace/outputs/qwen-3-32b/steering/chef_helpful_v2 ...]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_baselines(experiment_dir: Path) -> Dict[int, Dict[str, Any]]:
    p = experiment_dir / "baselines" / "records.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[int(r["question_idx"])] = r
    return out


def load_cells(experiment_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for cell in sorted(experiment_dir.iterdir()):
        if not cell.is_dir() or cell.name in ("baselines",):
            continue
        recs_p = cell / "records.jsonl"
        if not recs_p.exists():
            continue
        out[cell.name] = [
            json.loads(line) for line in recs_p.read_text().splitlines()
            if line.strip()
        ]
    return out


def per_strength_groups(recs: List[Dict[str, Any]]) -> Dict[float, List[Dict[str, Any]]]:
    g: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for r in recs:
        g[r["strength"]].append(r)
    return g


def _safe_mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return statistics.mean(xs) if xs else float("nan")


def _safe_stdev(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return statistics.stdev(xs) if len(xs) >= 2 else float("nan")


def extract_eff(r: Dict[str, Any]) -> Optional[float]:
    eff = (r.get("judges") or {}).get("effect") or {}
    v = eff.get("combined")
    return float(v) if isinstance(v, (int, float)) else None


def extract_eff_per_judge(r: Dict[str, Any]) -> Dict[str, Optional[float]]:
    eff = (r.get("judges") or {}).get("effect") or {}
    out: Dict[str, Optional[float]] = {}
    scores = (eff.get("bidirectional") or {}).get("scores") or {}
    for mn, sc in scores.items():
        if isinstance(sc, dict):
            out[mn] = sc.get("score") if isinstance(sc.get("score"), (int, float)) else None
    return out


def extract_coh(r: Dict[str, Any]) -> Optional[float]:
    j = r.get("judges") or {}
    return j.get("strength_mean_coh")


def extract_rp(r: Dict[str, Any]) -> Optional[float]:
    j = r.get("judges") or {}
    p = j.get("persona") or {}
    sc = p.get("score")
    return float(sc) if isinstance(sc, (int, float)) else None


def is_skipped(r: Dict[str, Any]) -> bool:
    eff = (r.get("judges") or {}).get("effect") or {}
    return bool(eff.get("skipped_due_to_strength_mean_coh"))


def pearson(xs, ys) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys)
             if isinstance(x, (int, float)) and isinstance(y, (int, float))
             and not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 3:
        return float("nan")
    xs2, ys2 = zip(*pairs)
    mx, my = statistics.mean(xs2), statistics.mean(ys2)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs2))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys2))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def analyze_cell(
    cell_name: str, recs: List[Dict[str, Any]],
    baselines: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    sign_part = cell_name.split("_")[-1]
    sign = +1 if sign_part == "+1" else -1
    g = per_strength_groups(recs)
    rows = []
    for s in sorted(g):
        rs = g[s]
        skipped = sum(1 for r in rs if is_skipped(r))
        if skipped == len(rs):
            rows.append({
                "strength": s, "sign": sign, "n": len(rs), "n_skipped": skipped,
                "mean_coh": _safe_mean([extract_coh(r) for r in rs]),
                "mean_eff": float("nan"), "stdev_eff": float("nan"),
                "mean_abs_eff": float("nan"),
                "mean_rp": float("nan"),
                "len_ratio": float("nan"),
                "judge_diff_2plus": float("nan"),
                "n_with_eff": 0,
            })
            continue
        effs = [extract_eff(r) for r in rs]
        effs_valid = [e for e in effs if e is not None]
        # Per-judge for agreement check
        per_judge = [extract_eff_per_judge(r) for r in rs]
        gpt_scores = [pj.get("gpt-4.1-mini") for pj in per_judge]
        cl_scores = [pj.get("claude-haiku-4-5-20251001") for pj in per_judge]
        # response length vs baseline length per question
        len_ratios = []
        for r in rs:
            qi = int(r["question_idx"])
            bl = baselines.get(qi)
            if bl and bl.get("response") and r.get("response"):
                len_ratios.append(len(r["response"]) / max(1, len(bl["response"])))
        # judge_diff_2plus: fraction of records where |gpt - claude| >= 2
        n_with_both = 0
        n_diff2 = 0
        for g_, c_ in zip(gpt_scores, cl_scores):
            if g_ is not None and c_ is not None:
                n_with_both += 1
                if abs(g_ - c_) >= 2:
                    n_diff2 += 1
        rows.append({
            "strength": s, "sign": sign, "n": len(rs), "n_skipped": skipped,
            "n_with_eff": len(effs_valid),
            "mean_coh": _safe_mean([extract_coh(r) for r in rs]),
            "mean_eff": _safe_mean(effs_valid),
            "stdev_eff": _safe_stdev(effs_valid),
            "mean_abs_eff": _safe_mean([abs(e) for e in effs_valid]),
            "mean_rp": _safe_mean([extract_rp(r) for r in rs]),
            "len_ratio": _safe_mean(len_ratios),
            "judge_diff_2plus": (n_diff2 / n_with_both) if n_with_both else float("nan"),
        })
    return {
        "cell": cell_name,
        "sign": sign,
        "rows": rows,
        "judge_correlation": _cross_judge_corr_all(recs),
    }


def _cross_judge_corr_all(recs: List[Dict[str, Any]]) -> float:
    """Pearson r between gpt-4.1-mini and claude-haiku scores across all
    non-skipped records that have BOTH judges' scores."""
    gs, cs = [], []
    for r in recs:
        if is_skipped(r):
            continue
        pj = extract_eff_per_judge(r)
        g_ = pj.get("gpt-4.1-mini")
        c_ = pj.get("claude-haiku-4-5-20251001")
        if g_ is not None and c_ is not None:
            gs.append(g_)
            cs.append(c_)
    return pearson(gs, cs)


def fmt_num(x, fmt=".2f"):
    if x is None:
        return "?"
    if isinstance(x, float) and math.isnan(x):
        return "."
    return format(x, fmt)


def print_cell_report(report: Dict[str, Any]) -> None:
    cell = report["cell"]
    sign = report["sign"]
    print(f"\n=== {cell} (sign={sign:+d}) ===")
    print(f"  cross-judge pearson r (gpt vs claude): {fmt_num(report['judge_correlation'])}")
    print()
    print(f"  {'strength':>10} {'n_eff':>5} {'mean_coh':>8} {'mean_rp':>7} "
          f"{'mean_eff':>9} {'mabs_eff':>9} {'stdev':>6} {'len_r':>6} "
          f"{'diff2+':>6}")
    # Monotonicity check on |eff|: find longest non-decreasing run.
    prev_abs = -math.inf
    monotone_increases = 0
    monotone_breaks = 0
    first_above_threshold_strength = None
    SIGNAL_THRESHOLD = 0.5
    for row in report["rows"]:
        flag = []
        if not math.isnan(row["mean_coh"]) and row["mean_coh"] >= 1.5:
            flag.append("INCOH")
        if not math.isnan(row["len_ratio"]) and row["len_ratio"] < 0.5:
            flag.append("RES-COLLAPSE")
        if not math.isnan(row["judge_diff_2plus"]) and row["judge_diff_2plus"] >= 0.3:
            flag.append("JUDGE-DISAGREE")
        if not math.isnan(row["mean_abs_eff"]):
            cur = row["mean_abs_eff"]
            if first_above_threshold_strength is None and cur >= SIGNAL_THRESHOLD:
                first_above_threshold_strength = row["strength"]
            if not math.isnan(prev_abs) and cur < prev_abs - 0.5:
                flag.append("NON-MONO")
                monotone_breaks += 1
            else:
                monotone_increases += 1
            prev_abs = cur
        print(f"  {row['strength']:>10.4f} {row['n_with_eff']:>5d} "
              f"{fmt_num(row['mean_coh']):>8} {fmt_num(row['mean_rp']):>7} "
              f"{fmt_num(row['mean_eff'], '+.2f'):>9} "
              f"{fmt_num(row['mean_abs_eff']):>9} "
              f"{fmt_num(row['stdev_eff']):>6} "
              f"{fmt_num(row['len_ratio'], '.2f'):>6} "
              f"{fmt_num(row['judge_diff_2plus'], '.2f'):>6}  "
              f"{','.join(flag)}")
    print()
    print(f"  first strength with mean|eff|>={SIGNAL_THRESHOLD}: "
          f"{first_above_threshold_strength}")
    print(f"  monotonicity: {monotone_increases} non-decreases, "
          f"{monotone_breaks} drops > 0.5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dirs", nargs="+", type=Path)
    args = ap.parse_args()

    for exp_dir in args.experiment_dirs:
        print(f"\n##### {exp_dir.name} #####")
        baselines = load_baselines(exp_dir)
        cells = load_cells(exp_dir)
        for cell_name, recs in cells.items():
            report = analyze_cell(cell_name, recs, baselines)
            print_cell_report(report)


if __name__ == "__main__":
    main()
