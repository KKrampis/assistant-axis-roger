#!/usr/bin/env python3
"""Refill gaps in axis-judge output directories.

This is a convenience wrapper around ``axis_judge_correlation.py``. Given one
or more axis-output directories, it reads each directory's ``config.json``
to recover the exact CLI arguments that produced the run, optionally checks
``gaps.json`` to confirm there *are* gaps to refill, and re-invokes
``axis_judge_correlation.py`` with the same args.

The inner script's resume logic does the right thing automatically:

- **static modes** (``descriptions``, ``instructions``): only re-issues calls
  for entities not already in the cache (parse failures aren't cached, so
  they're retried; successful entities are skipped).
- **response mode**: only re-issues calls for ``per_batch`` entries whose
  ``score`` is ``None`` (transient-failure batches; successful batches are
  skipped).

So a refill is **cheap**: it costs roughly the size of the gap, not a full
re-run. With the new retry-with-backoff in ``axis_judge_correlation.py``,
gaps should be rare in the first place; this tool exists for the legacy
caches that pre-date that change.

Use cases
---------

- "I just noticed ``gaps.json`` says these axes have unfilled batches --
  fix them": pass the affected dirs as ``--dir`` arguments.
- "Audit all my axis runs for gaps": pass a glob via ``--glob``.
- "Just *show* me what's incomplete; don't actually call the judge":
  ``--scan``.

CLI
---

::

    # Refill one specific dir.
    uv run python results_analysis/refill_judge_gaps.py \\
      --dir roger/axis_judge_experiments/truthful_vs_deceitful/gpt_responses_traits

    # Refill any GPT-response cache with gaps under the experiment root.
    uv run python results_analysis/refill_judge_gaps.py \\
      --glob 'roger/axis_judge_experiments/*/gpt_responses_*'

    # Audit-only: list dirs with gaps, do nothing.
    uv run python results_analysis/refill_judge_gaps.py \\
      --glob 'roger/axis_judge_experiments/*/*' --scan
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _gap_summary(gaps: Dict[str, Any]) -> Tuple[int, str]:
    """Return ``(total_gap_count, human_summary)`` for a gaps.json payload.

    Static modes contribute one gap per missing entity name; response mode
    contributes one gap per (entity, batch_idx) pair.
    """
    parts: List[str] = []
    total = 0
    for mode in ("descriptions", "instructions"):
        v = gaps.get(mode)
        if isinstance(v, list) and v:
            parts.append(f"{mode}: {len(v)} entities")
            total += len(v)
    resp = gaps.get("responses")
    if isinstance(resp, dict) and resp:
        n_batches = sum(len(v) for v in resp.values() if isinstance(v, list))
        parts.append(f"responses: {n_batches} batches across {len(resp)} entities")
        total += n_batches
    return total, "; ".join(parts) if parts else "no gaps"


def _scan_dir(d: Path) -> Optional[Tuple[int, str]]:
    """Inspect ``d`` for a gaps.json or fall back to inferring gaps from
    cache files. Returns ``(total_gaps, summary_str)`` or ``None`` if the
    directory doesn't look like an axis-judge output dir.
    """
    if not (d / "config.json").exists():
        return None
    gaps_path = d / "gaps.json"
    if gaps_path.exists():
        try:
            return _gap_summary(_load_json(gaps_path))
        except Exception as e:
            return 0, f"could not read gaps.json: {e}"

    # Legacy fallback: gaps.json doesn't exist (older caches). Inspect
    # scores_responses.json directly for None batches; for static modes we
    # don't know the universe size from this dir alone, so we just check
    # that the file exists and isn't empty.
    inferred: Dict[str, Any] = {}
    resp_path = d / "scores_responses.json"
    if resp_path.exists():
        try:
            obj = _load_json(resp_path)
            resp_gaps: Dict[str, List[int]] = {}
            for name, entry in obj.items():
                if not isinstance(entry, dict):
                    continue
                pb = entry.get("per_batch", []) or []
                idxs = [i for i, b in enumerate(pb) if b.get("score") is None]
                if idxs:
                    resp_gaps[name] = idxs
            if resp_gaps:
                inferred["responses"] = resp_gaps
        except Exception:
            pass
    if inferred:
        return _gap_summary(inferred)
    return 0, "no gaps (inferred from caches; no gaps.json)"


def _build_argv(config: Dict[str, Any]) -> List[str]:
    """Reconstruct the axis_judge_correlation.py argv from a config.json."""
    a = config["args"]
    cmd: List[str] = [
        sys.executable, "results_analysis/axis_judge_correlation.py",
    ]

    # Axis specification: --pair (P, N) plus --pair_type, OR --axis_file.
    pair = a.get("pair")
    if pair:
        cmd += ["--pair", pair[0], pair[1]]
        if a.get("pair_type"):
            cmd += ["--pair_type", a["pair_type"]]
    elif a.get("axis_file"):
        cmd += ["--axis_file", a["axis_file"]]

    if a.get("axis_name"):
        cmd += ["--axis_name", a["axis_name"]]
    if a.get("neg_pole"):
        cmd += ["--neg_pole", a["neg_pole"]]
    if a.get("pos_pole"):
        cmd += ["--pos_pole", a["pos_pole"]]
    if a.get("neg_examples"):
        cmd += ["--neg_examples", ",".join(a["neg_examples"])]
    if a.get("pos_examples"):
        cmd += ["--pos_examples", ",".join(a["pos_examples"])]

    # Data + judging knobs.
    cmd += ["--data_dir", a["data_dir"]]
    cmd += ["--instructions_dir", a.get("instructions_dir", "data")]
    cmd += ["--layer", str(a.get("layer", 25))]
    if a.get("slot") and a["slot"] != "all":
        cmd += ["--slot", str(a["slot"])]
    cmd += ["--whiten_K", str(a.get("whiten_K", 3))]
    cmd += ["--provider", a["provider"]]
    cmd += ["--judge_model", a["judge_model"]]
    cmd += ["--max_tokens", str(a.get("max_tokens", 1024))]
    cmd += ["--temperature", str(a.get("temperature", 0.0))]
    cmd += ["--rps", str(a.get("rps", 10.0))]
    cmd += ["--batch_size", str(a.get("batch_size", 20))]
    cmd += ["--save_every", str(a.get("save_every", 40))]

    # Scoring modes.
    if a.get("score_descriptions"):
        cmd += ["--score_descriptions"]
    if a.get("score_instructions"):
        cmd += ["--score_instructions"]
    if a.get("score_responses"):
        cmd += ["--score_responses"]
        if a.get("scores_dir"):
            cmd += ["--scores_dir", a["scores_dir"]]
        if a.get("responses_dir"):
            cmd += ["--responses_dir", a["responses_dir"]]
        if a.get("response_target_batch_size"):
            cmd += ["--response_target_batch_size",
                    str(a["response_target_batch_size"])]

    # Output goes back to the same directory as the original run.
    cmd += ["--output_dir", a["output_dir"]]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(
        description="Refill gaps in axis-judge output directories by "
                    "re-invoking axis_judge_correlation.py with the same args.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", action="append", default=[],
                   help="Path to an axis-judge output directory (repeatable). "
                        "Each must contain a config.json from a previous run.")
    p.add_argument("--glob", action="append", default=[],
                   help="Glob pattern to expand into output directories "
                        "(repeatable). Globs are evaluated relative to cwd.")
    p.add_argument("--scan", action="store_true",
                   help="Audit-only: list which dirs have gaps and how many; "
                        "don't actually call the judge.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the command(s) that would be run; don't execute.")
    args = p.parse_args()

    # Collect directories.
    dirs: List[Path] = [Path(d) for d in args.dir]
    for pat in args.glob:
        dirs.extend(Path(x) for x in sorted(_glob.glob(pat)))
    # Dedupe while preserving order.
    seen: set = set()
    unique_dirs: List[Path] = []
    for d in dirs:
        if d in seen:
            continue
        seen.add(d)
        unique_dirs.append(d)
    if not unique_dirs:
        p.error("specify --dir and/or --glob (no directories selected)")

    # Filter to only those that look like axis-judge output dirs.
    candidates: List[Tuple[Path, int, str]] = []
    for d in unique_dirs:
        info = _scan_dir(d)
        if info is None:
            continue  # not an axis-judge dir
        candidates.append((d, info[0], info[1]))

    if not candidates:
        print("No axis-judge output directories matched.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(candidates)} axis-judge directories:")
    for d, n_gaps, summary in candidates:
        marker = "  REFILL" if n_gaps > 0 else "  clean "
        print(f"{marker}  {d}    [{summary}]")
    print()

    if args.scan:
        n_dirty = sum(1 for _, n, _ in candidates if n > 0)
        print(f"Scan-only mode: {n_dirty} of {len(candidates)} dirs have gaps. "
              f"Re-run without --scan to refill.")
        return

    refill_targets = [(d, n, s) for d, n, s in candidates if n > 0]
    if not refill_targets:
        print("Nothing to do -- all selected directories are clean.")
        return

    print(f"Refilling {len(refill_targets)} directory/directories...\n")
    n_ok = 0
    n_fail = 0
    for d, n_gaps, summary in refill_targets:
        try:
            config = _load_json(d / "config.json")
        except Exception as e:
            print(f"  FAIL  {d}: cannot read config.json ({e})")
            n_fail += 1
            continue

        cmd = _build_argv(config)
        print(f"--- {d} ({summary}) ---")
        if args.dry_run:
            print("  (dry-run)  " + " ".join(repr(x) if " " in x else x for x in cmd))
            continue
        # Inherit stdout/stderr so the user sees retry/gap banners live.
        rc = subprocess.run(cmd, check=False).returncode
        if rc == 0:
            n_ok += 1
            print(f"  OK    {d}")
        else:
            n_fail += 1
            print(f"  FAIL  {d}: rc={rc}")
        print()

    if not args.dry_run:
        print(f"Refilled {n_ok} dir(s); {n_fail} failure(s).")


if __name__ == "__main__":
    main()
