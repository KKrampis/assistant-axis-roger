"""Per-question responsiveness audit across all available steering experiments.

For each (axis, cell=slot×layer, sign), we order kept strengths by
"steps remaining to incoherence" (x=0 = last coherent, x=1 = one step
weaker, ...).  At each x ∈ {0,1,2,4,8,16} (a biased subsample favouring
near-cliff strengths) we:

  1. Pick the strength S at that x (skip if absent).
  2. Within that (cell, sign, S) cohort, keep only per-record
     coherence.score == 0 entries.
  3. Compute directional response: dr = sign * effect.combined.
  4. Compute cohort mean μ and stdev σ of dr.
  5. Classify each kept question as:
        - "strong"  if dr - μ >  TAU  (or z > Z_HI when σ usable)
        - "weak"    if dr - μ < -TAU  (or z < Z_LO when σ usable)
        - "average" otherwise

Then aggregate per (axis, question_idx):
    counts of strong / average / weak across all (cell, sign, x)
    where that question appeared with coh==0.
    net = (#strong - #weak), normalised by appearances.

Print a per-axis ranking and a global summary (which question
*shapes* tend to land strong vs weak).

Outputs JSONL of per-question stats so we can post-process / sort
by net responsiveness.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path("/Users/roger/Documents/GitHub/assistant-axis/runpod_workspace/qwen/qwen-3-32b Roger 8slot/steering")
AXES = [
    "anthropologist_helpful_v1",
    "prodigy_harmless_v1",
    "novelist_honest_v1",
    "publisher_truthful_v1",
    "publisher_guileless_v1",
    "cartographer_egalitarian_v1",
    "navigator_progressive_v1",
    "anarchist_concise_v1",
    "architect_ecocentric_v3",
]
X_POSITIONS = [0, 1, 2, 4, 8, 16]
COH_FILTER = 1.0          # cell-level mean_coh anchor (matches plot default)
TAU = 0.5                 # absolute effect deviation threshold (fallback)
Z_HI, Z_LO = 0.5, -0.5    # z-score thresholds when σ usable
MIN_SIGMA = 0.4           # below this, fall back to TAU


def load_questions(axis_dir: Path) -> list[str]:
    return json.loads((axis_dir / "questions.json").read_text())


def load_cell_records(cell_dir: Path):
    """Return list of (strength, sign, question_idx, coh, eff_combined)."""
    out = []
    p = cell_dir / "records.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        j = r.get("judges") or {}
        coh = ((j.get("coherence") or {}).get("score"))
        eff = ((j.get("effect") or {}).get("combined"))
        if coh is None or eff is None:
            continue
        out.append((
            float(r["strength"]),
            int(r["sign"]),
            int(r["question_idx"]),
            int(coh),
            float(eff),
        ))
    return out


def order_x_positions(records):
    """Return [(x_idx, strength)] for kept strengths after coh anchor.

    Group by strength, compute mean_coh per strength, drop strengths
    whose mean_coh >= COH_FILTER, sort the remainder by |strength|
    descending so the strongest still-coherent is x=0.
    """
    by_str = defaultdict(list)
    for s, sg, qi, coh, eff in records:
        by_str[s].append(coh)
    kept = []
    for s, cohs in by_str.items():
        mean_coh = sum(cohs) / len(cohs)
        if mean_coh < COH_FILTER:
            kept.append((abs(s), s, mean_coh))
    kept.sort(key=lambda t: t[0], reverse=True)
    return [(i, s) for i, (_, s, _) in enumerate(kept)]


def classify(dr: float, mu: float, sigma: float) -> str:
    if sigma >= MIN_SIGMA:
        z = (dr - mu) / sigma
        if z > Z_HI:
            return "strong"
        if z < Z_LO:
            return "weak"
        return "average"
    d = dr - mu
    if d > TAU:
        return "strong"
    if d < -TAU:
        return "weak"
    return "average"


def cell_dirs(axis_dir: Path):
    for p in sorted(axis_dir.glob("s*_l*_*")):
        if not p.is_dir():
            continue
        yield p


def parse_cell_name(name: str):
    # eg s0_l25_+1 or s0_l25_-1
    parts = name.split("_")
    slot = int(parts[0][1:])
    layer = int(parts[1][1:])
    sign = int(parts[2])
    return slot, layer, sign


def main():
    per_q = defaultdict(lambda: {"strong": 0, "average": 0, "weak": 0, "n": 0,
                                 "dr_sum": 0.0, "dr_sq": 0.0,
                                 "n_at_x": Counter(), "strong_at_x": Counter(),
                                 "weak_at_x": Counter()})
    n_cohorts = 0
    n_dead_cohorts = 0  # cohort where all dr ≈ 0

    for axis in AXES:
        axis_dir = ROOT / axis
        if not axis_dir.exists():
            continue
        qs = load_questions(axis_dir)
        for cd in cell_dirs(axis_dir):
            slot, layer, sign = parse_cell_name(cd.name)
            recs = load_cell_records(cd)
            if not recs:
                continue
            x_to_s = dict(order_x_positions(recs))
            # Map: strength -> list of (qi, coh, eff)
            by_s = defaultdict(list)
            for s, sg, qi, coh, eff in recs:
                by_s[s].append((qi, coh, eff))
            for x in X_POSITIONS:
                s = x_to_s.get(x)
                if s is None:
                    continue
                cohort = [(qi, eff) for qi, coh, eff in by_s[s] if coh == 0]
                if len(cohort) < 4:
                    continue
                drs = [sign * eff for _, eff in cohort]
                mu = statistics.fmean(drs)
                sigma = statistics.pstdev(drs) if len(drs) > 1 else 0.0
                n_cohorts += 1
                if max(abs(x_ - 0) for x_ in drs) < 0.1:
                    n_dead_cohorts += 1
                for qi, eff in cohort:
                    dr = sign * eff
                    label = classify(dr, mu, sigma)
                    key = (axis, qi)
                    rec = per_q[key]
                    rec[label] += 1
                    rec["n"] += 1
                    rec["dr_sum"] += dr
                    rec["dr_sq"] += dr * dr
                    rec["n_at_x"][x] += 1
                    if label == "strong":
                        rec["strong_at_x"][x] += 1
                    elif label == "weak":
                        rec["weak_at_x"][x] += 1

    # Build per-axis ranking + global ranking
    rows = []
    for (axis, qi), rec in per_q.items():
        n = rec["n"]
        if n == 0:
            continue
        net = (rec["strong"] - rec["weak"]) / n
        dr_mean = rec["dr_sum"] / n
        dr_var = max(0.0, rec["dr_sq"] / n - dr_mean * dr_mean)
        dr_std = math.sqrt(dr_var)
        qs = load_questions(ROOT / axis)
        question = qs[qi] if qi < len(qs) else "<missing>"
        rows.append({
            "axis": axis,
            "qi": qi,
            "question": question,
            "n": n,
            "strong": rec["strong"],
            "average": rec["average"],
            "weak": rec["weak"],
            "net_per_appearance": net,
            "dr_mean": dr_mean,
            "dr_std": dr_std,
            "strong_at_x": dict(rec["strong_at_x"]),
            "weak_at_x": dict(rec["weak_at_x"]),
            "n_at_x": dict(rec["n_at_x"]),
        })

    rows.sort(key=lambda r: (r["axis"], -r["net_per_appearance"]))

    out_path = Path("/tmp/q_responsiveness.jsonl")
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Per-axis ranking
    print(f"\n{n_cohorts} cohorts examined, {n_dead_cohorts} ~ dead (|dr|<0.1)")
    print(f"\nPer-axis ranking (top 3 strong, top 3 weak):\n")
    by_axis = defaultdict(list)
    for r in rows:
        by_axis[r["axis"]].append(r)
    for axis in AXES:
        if axis not in by_axis:
            continue
        rs = sorted(by_axis[axis], key=lambda r: -r["net_per_appearance"])
        print(f"=== {axis} ===")
        for r in rs[:3]:
            print(f"  +{r['net_per_appearance']:+.2f} n={r['n']:3d} "
                  f"S{r['strong']:>2d}/A{r['average']:>2d}/W{r['weak']:>2d} "
                  f"dr={r['dr_mean']:+.2f}±{r['dr_std']:.2f}  "
                  f"q{r['qi']:>2d}: {r['question'][:90]}")
        print("  ...")
        for r in rs[-3:]:
            print(f"  {r['net_per_appearance']:+.2f} n={r['n']:3d} "
                  f"S{r['strong']:>2d}/A{r['average']:>2d}/W{r['weak']:>2d} "
                  f"dr={r['dr_mean']:+.2f}±{r['dr_std']:.2f}  "
                  f"q{r['qi']:>2d}: {r['question'][:90]}")
        print()

    # Global: top 15 strongest + top 15 weakest (cross-axis)
    glob = sorted(rows, key=lambda r: -r["net_per_appearance"])
    print("\n=== GLOBAL TOP 15 STRONGEST ===")
    for r in glob[:15]:
        print(f"  {r['net_per_appearance']:+.2f} n={r['n']:3d} "
              f"[{r['axis'][:24]:24s}] q{r['qi']:>2d}: {r['question'][:100]}")
    print("\n=== GLOBAL TOP 15 WEAKEST ===")
    for r in glob[-15:]:
        print(f"  {r['net_per_appearance']:+.2f} n={r['n']:3d} "
              f"[{r['axis'][:24]:24s}] q{r['qi']:>2d}: {r['question'][:100]}")


if __name__ == "__main__":
    main()
