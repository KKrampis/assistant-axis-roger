#!/usr/bin/env python3
"""Phase 5b orchestrator (May 2026): surgical rejudge of the 9
trait/role collision names across all 12 v2 axes for GPT + Haiku
in static mode (``scores_descriptions.json`` + ``scores_instructions.json``).

Mechanics
---------

The fix is fully automatic: when ``axis_judge_correlation.py`` resumes
from a v1 (bare-name) cache it now MIGRATES the cache in-place via
:func:`score_static_mode`'s migration step (Phase 4):

* Bare names that map to a unique kind in the corpus get RELABELLED
  to ``entity_id(name, kind)`` keys (~571 entries per cache).
* The 9 collision names get DROPPED (Bug A means we can't trust
  the bare-name value).
* A subsequent ``score_static_mode`` pass re-judges only the
  dropped entries — exactly the 9 collisions × 2 kinds = 18
  entities per cell.

Across 12 axes × 2 judges × 2 modes = 48 cells × 18 calls = **864
calls**.  GPT-4.1-mini portion ~$0.16, Haiku portion ~$0.39, total
~$0.55 (per the Phase 4c cost model).  Per-cell budget cap is set
to **$5** (~3500× expected) to catch any rogue cell without
dragging on a tiny rejudge run.

This script is idempotent: re-running on already-v2 caches is a
no-op (nothing to migrate, nothing to re-judge — the producer
short-circuits via the cache hit).
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Tuple

REPO = Path(__file__).resolve().parent.parent
PAIR_LIST = REPO / "roger/axis_judge_experiments/pair_list_responses.json"
EXPERIMENT_ROOT = REPO / "roger/axis_judge_experiments"
DATA_DIR = REPO / "runpod_workspace/qwen/qwen-3-32b Roger 8slot"

JUDGES = [
    # (subdir, provider, model)
    ("gpt", "openai", "gpt-4.1-mini"),
    ("haiku", "anthropic", "claude-haiku-4-5-20251001"),
]
PER_CELL_BUDGET_USD = 5.0

# Per-cell expected cost (USD) -- one cell = one (axis, judge) pair,
# rejudging 18 collision entities × 2 modes (descriptions +
# instructions) = 36 single-shot calls.  Empirical token shape from
# the 2026-05-10 GPT canary on concise_vs_verbose: 351.8 input +
# 62.5 output tokens per call.  Haiku scaled from that via the rate
# card (input 2.5×, output 3.125×).  See AGENT_NOTES "Judging cost
# model".
PER_CELL_EXPECTED_USD = {
    "gpt":   0.0087,   # 36 × ($0.40/1M × 351.8 + $1.60/1M × 62.5)
    "haiku": 0.0240,   # 36 × ($1.00/1M × 351.8 + $5.00/1M × 62.5)
}

logging.basicConfig(
    format="%(asctime)s [phase5b] %(levelname)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("phase5b")


def _load_axes() -> list[Tuple[str, str]]:
    pairs = json.loads(PAIR_LIST.read_text())
    return [(p["pos"], p["neg"]) for p in pairs]


def _run_one_cell(
    pos: str, neg: str, judge_subdir: str, provider: str, model: str,
    *, dry_run: bool,
) -> dict:
    """Invoke ``axis_judge_correlation.py`` directly (bypassing
    ``run_axis_experiment_batch``'s correlations-exists SKIP) for a
    single (axis, judge) cell.  Returns a dict with cost / call counts.
    """
    axis_id = f"{pos}_vs_{neg}"
    out_dir = EXPERIMENT_ROOT / axis_id / judge_subdir
    cmd = [
        "uv", "run", "python",
        "results_analysis/axis_judge_correlation.py",
        "--pair", pos, neg, "--pair_type", "traits",
        # NB: --pair_type controls the AXIS direction vector (loaded from
        # the traits/ pole vectors); it does NOT restrict scoring scope.
        # score_static_mode iterates corpus.descriptions which is the full
        # mixed-kind 302-traits + 280-roles list, so all 9×2 collisions get
        # rejudged here regardless of pair_type.
        "--data_dir", str(DATA_DIR),
        "--provider", provider, "--judge_model", model,
        "--score_descriptions", "--score_instructions",
        "--output_dir", str(out_dir),
        "--budget_usd", str(PER_CELL_BUDGET_USD),
        "--expected_cost_usd", str(PER_CELL_EXPECTED_USD[judge_subdir]),
        "--usage_json", str(out_dir / "usage_phase5b.json"),
    ]
    log.info(f"[{axis_id}/{judge_subdir}] {shlex.join(cmd)}")
    if dry_run:
        return {"axis": axis_id, "judge": judge_subdir, "dry_run": True}
    proc = subprocess.run(
        cmd, cwd=str(REPO), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.error(
            f"[{axis_id}/{judge_subdir}] FAILED rc={proc.returncode}\n"
            f"STDOUT (tail): {proc.stdout[-1500:]}\n"
            f"STDERR (tail): {proc.stderr[-1500:]}"
        )
        return {
            "axis": axis_id, "judge": judge_subdir,
            "rc": proc.returncode, "ok": False,
        }
    # Parse the usage side-car for cost reporting.
    side_car = out_dir / "usage_phase5b.json"
    usage = json.loads(side_car.read_text()) if side_car.exists() else {}
    cost = usage.get("cost_usd", 0.0)
    n_calls = usage.get("n_calls", 0)
    log.info(
        f"[{axis_id}/{judge_subdir}] OK cost=${cost:.4f} "
        f"n_calls={n_calls} (expected ~36 = 18 desc + 18 instr)"
    )
    return {
        "axis": axis_id, "judge": judge_subdir,
        "rc": 0, "ok": True, "cost_usd": cost,
        "n_calls": n_calls, "usage": usage,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--axes", nargs="+", default=None,
        help="Restrict to specific axes (e.g. concise_vs_verbose). "
             "Default: all 12 v2 axes.",
    )
    p.add_argument(
        "--judges", nargs="+", default=None,
        choices=["gpt", "haiku"],
        help="Restrict to one judge family.  Default: both.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print commands, don't execute (useful for inspection).",
    )
    args = p.parse_args()

    axes = _load_axes()
    if args.axes:
        wanted = set(args.axes)
        axes = [(p, n) for (p, n) in axes if f"{p}_vs_{n}" in wanted]
        if not axes:
            log.error(f"--axes={args.axes} matched no v2 axes")
            return 2
    judges = [j for j in JUDGES if (args.judges is None or j[0] in args.judges)]

    n_cells = len(axes) * len(judges)
    log.info(
        f"Phase 5b: {n_cells} cells "
        f"({len(axes)} axes × {len(judges)} judges)"
    )
    expected_total = sum(
        len(axes) * PER_CELL_EXPECTED_USD[j[0]] for j in judges
    )
    log.info(f"Per-cell budget: ${PER_CELL_BUDGET_USD:.2f}; expected:")
    for jname, _, _ in judges:
        log.info(
            f"  {jname:<6} ~${PER_CELL_EXPECTED_USD[jname]:.4f}/cell × "
            f"{len(axes)} axes = ~${len(axes) * PER_CELL_EXPECTED_USD[jname]:.4f}"
        )
    log.info(f"Total budget cap: ${n_cells * PER_CELL_BUDGET_USD:.2f}")
    log.info(f"Total expected: ~${expected_total:.4f}")

    results = []
    for (pos, neg) in axes:
        for (sub, prov, model) in judges:
            results.append(_run_one_cell(
                pos, neg, sub, prov, model, dry_run=args.dry_run,
            ))

    if args.dry_run:
        return 0

    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    total_calls = sum(r.get("n_calls", 0) for r in results)
    n_failed = sum(1 for r in results if not r.get("ok", False))
    log.info("=" * 70)
    log.info(
        f"Phase 5b DONE: {n_cells - n_failed}/{n_cells} cells OK; "
        f"total_cost=${total_cost:.4f} (expected ~$0.55); "
        f"total_calls={total_calls} (expected ~864 incl. desc+instr)"
    )
    if n_failed:
        log.error(f"{n_failed} cells failed — see logs")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
