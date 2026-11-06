#!/usr/bin/env python3
"""Analyse start-strength quality for completed steering sweeps.

For every (persona, slot, layer, sign) cell that has at least one
judged record, look up the strength CLOSEST to the configured
``s_init = weakest * multiplier ** start_strength_multiplier_steps``
and categorise the start as:

* ``GOOD``: at s_init the coherence stop predicate isn't tripped
  (mean(coh) <= 1.5) AND the effect stop predicate isn't tripped
  either (|mean(effect.combined)| >= 1/3).  The scan can productively
  walk both UP and DOWN from here.

* ``TOO_LOW``: |mean(eff)| < 1/3 at s_init -- the downward walk
  immediately eff-stops, wasting at least one strength on a barren
  bottom region.  Higher s_init would help.

* ``TOO_HIGH``: mean(coh) > 1.5 at s_init -- the upward walk
  immediately coh-stops, wasting at least one strength on an
  already-incoherent region.  Lower s_init would help.

This is the analyser that produced the empirical
``assistant_axis.sweep_start_heuristics.OVERRIDES`` / ``DEFAULTS``
table.  Re-run after every meaningful new sweep batch and update the
heuristic if the (mode, slot, layer) buckets shift -- a single flat
``start_strength_multiplier_steps`` only ever fits a subset of the
grid well.

Usage::

    python tools/analyse_start_strength.py
    # or with an explicit output root:
    python tools/analyse_start_strength.py --root "runpod_workspace/qwen/qwen-3-32b Roger 8slot/steering"

The default root matches the canonical (re-filtered) steering output directory.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover -- always present in the venv
    raise SystemExit("This tool needs pyyaml.  pip install pyyaml.") from exc


COH_STOP = 1.5
EFF_STOP = 1.0 / 3.0


def _parse_cfg_params(cfg_dir: Path) -> dict[str, dict[str, float]]:
    """Map experiment_id -> sweep params (s_init, weakest, mult, steps)."""
    out: dict[str, dict[str, float]] = {}
    for cfg in cfg_dir.glob("*.yaml"):
        try:
            y = yaml.safe_load(cfg.read_text())
        except Exception:
            continue
        eid = y.get("experiment_id") or cfg.stem
        sw = y.get("sweep") or {}
        weakest = float(sw.get("weakest_strength", 1.0))
        mult = float(sw.get("multiplier", 1.189))
        steps = int(sw.get("start_strength_multiplier_steps", 2))
        out[eid] = {
            "s_init": weakest * (mult ** steps),
            "weakest": weakest, "mult": mult, "steps": steps,
        }
    return out


def _analyse_cell(records_path: Path, sign: int,
                  s_init: float) -> Optional[dict]:
    """Return (category, mean_coh, abs_eff, n_records) for the start
    strength nearest s_init in this cell+sign, or None if no records."""
    by_strength: defaultdict[float, list[dict]] = defaultdict(list)
    if not records_path.exists():
        return None
    for line in records_path.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("abandoned"):
            continue
        if int(r.get("sign", 0)) != sign:
            continue
        s = float(r.get("strength", 0.0))
        if s <= 0:
            continue
        by_strength[s].append(r)
    if not by_strength:
        return None
    nearest = min(by_strength, key=lambda s: abs(s - s_init))
    recs = by_strength[nearest]
    coh: list[float] = []
    eff: list[float] = []
    for r in recs:
        j = r.get("judges") or {}
        c = j.get("coherence")
        if isinstance(c, dict) and isinstance(c.get("score"), (int, float)):
            coh.append(float(c["score"]))
        e = j.get("effect")
        if isinstance(e, dict):
            cb = e.get("combined")
            if isinstance(cb, (int, float)):
                eff.append(float(cb))
    if not coh:
        return None
    m_coh = sum(coh) / len(coh)
    m_eff = abs(sum(eff) / len(eff)) if eff else 0.0
    if m_coh > COH_STOP:
        cat = "TOO_HIGH"
    elif m_eff < EFF_STOP:
        cat = "TOO_LOW"
    else:
        cat = "GOOD"
    return {
        "category": cat, "actual_strength": nearest,
        "mean_coh": m_coh, "abs_eff": m_eff, "n_records": len(recs),
    }


def _persona_to_mode(name: str) -> str:
    return "prefill" if name.endswith("_prefill") else "all"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("runpod_workspace/qwen/qwen-3-32b Roger 8slot/steering"),
                    help="Sweep output root (contains <persona>/<cell>/records.jsonl).")
    ap.add_argument("--cfg-dir", type=Path,
                    default=Path("data/steering/configs"),
                    help="Directory with the YAML configs that produced the runs.")
    args = ap.parse_args()

    cfg = _parse_cfg_params(args.cfg_dir)
    print(f"Parsed {len(cfg)} configs from {args.cfg_dir}")

    rows: list[dict] = []
    for persona_dir in sorted(args.root.iterdir()):
        if not persona_dir.is_dir() or persona_dir.name == "baselines":
            continue
        cp = cfg.get(persona_dir.name)
        if cp is None:
            continue
        s_init = cp["s_init"]
        mode = _persona_to_mode(persona_dir.name)
        for cell_dir in sorted(persona_dir.iterdir()):
            if not cell_dir.is_dir() or not cell_dir.name.startswith("s"):
                continue
            parts = cell_dir.name.split("_")
            if len(parts) != 3:
                continue
            try:
                slot = int(parts[0][1:])
                layer = int(parts[1][1:])
                sign = int(parts[2])
            except ValueError:
                continue
            res = _analyse_cell(cell_dir / "records.jsonl", sign, s_init)
            if res is None:
                continue
            rows.append({
                "persona": persona_dir.name, "mode": mode,
                "slot": slot, "layer": layer, "sign": sign,
                "s_init": s_init, **res,
            })

    if not rows:
        print("No cells found.")
        return
    print(f"Cells analysed: {len(rows)}")
    print()

    cats = Counter(r["category"] for r in rows)
    print("Overall start quality (at configured s_init):")
    for c in ("GOOD", "TOO_LOW", "TOO_HIGH"):
        n = cats.get(c, 0)
        print(f"  {c:10s}  {n:4d}  ({n*100/len(rows):.1f}%)")
    print()

    print("By mode (all vs prefill):")
    print(f"{'mode':<10} {'GOOD':>6} {'TOO_LOW':>8} {'TOO_HIGH':>9} {'total':>7}")
    for mode in ("all", "prefill"):
        sub = [r for r in rows if r["mode"] == mode]
        c = Counter(r["category"] for r in sub)
        print(f"{mode:<10} {c.get('GOOD',0):>6} {c.get('TOO_LOW',0):>8} "
              f"{c.get('TOO_HIGH',0):>9} {len(sub):>7}")
    print()

    print("By (slot, layer, sign, mode):")
    print(f"{'slot':>4} {'layer':>5} {'sign':>4} {'mode':<8}  "
          f"{'GOOD':>4} {'TOO_LOW':>7} {'TOO_HIGH':>8} {'total':>5}  "
          f"{'eff@s_init':>10} {'coh@s_init':>10}")
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["slot"], r["layer"], r["sign"], r["mode"])].append(r)
    for (s, l, sn, m), rs in sorted(by_key.items()):
        c = Counter(r["category"] for r in rs)
        eff_med = sorted(r["abs_eff"] for r in rs)[len(rs)//2]
        coh_med = sorted(r["mean_coh"] for r in rs)[len(rs)//2]
        print(f"{s:>4} {l:>5} {sn:>+4} {m:<8}  {c.get('GOOD',0):>4} "
              f"{c.get('TOO_LOW',0):>7} {c.get('TOO_HIGH',0):>8} "
              f"{len(rs):>5}  {eff_med:>10.3f} {coh_med:>10.3f}")


if __name__ == "__main__":
    main()
