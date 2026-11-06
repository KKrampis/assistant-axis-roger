#!/usr/bin/env python3
"""Post-hoc audit + fill of allowlisted judge-refusal gaps.

Scans axis-judge output directories, partitions any gaps recorded in
``gaps.json`` into (allowlisted -> fillable via fallback judge) vs
(unexpected -> loud banner), and -- when not in ``--scan`` mode --
calls the fallback judge to fill the allowlisted ones.

This tool covers the **post-hoc** path: caches produced before the
producer-side automatic fallback existed, or by code paths that bypass
it (manual one-off scoring scripts, for example).

The **producer-side automatic fallback** lives at
:func:`assistant_axis.judge_refusal_fallback.fill_allowlisted_gaps_inline`
and is invoked at the end of
:func:`results_analysis.axis_judge_correlation.score_static_mode`, so
fresh judging runs are self-healing for allowlisted entities without
needing this tool.  Both code paths share the same per-entity helpers
in :mod:`assistant_axis.judge_refusal_fallback`, so a fill via this
tool and a fill via the producer-side path produce byte-identical
results.

Use cases:

* "I just noticed ``gaps.json`` says these axes have unfilled
  allowlisted entries -- fill them": run the tool (default mode).
* "Just *show* me what would be filled; don't actually call the
  judge": pass ``--scan``.
* "Surface any UNEXPECTED refusals across all my caches": same scan;
  unexpected gaps trigger the loud banner regardless of mode.

CLI::

    # Default: scan all sonnet/ subdirs under the experiment root and
    # fill any allowlisted gaps.
    uv run python tools/fill_judge_refusal_gaps.py

    # Audit-only.
    uv run python tools/fill_judge_refusal_gaps.py --scan

    # Target a specific dir.
    uv run python tools/fill_judge_refusal_gaps.py \\
        --dir roger/axis_judge_experiments/aligned_artificial_intelligence_vs_paperclip_maximizer/sonnet
"""
from __future__ import annotations

import argparse
import asyncio
import glob as _glob
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import results_analysis.axis_judge_correlation as ajc  # noqa: E402
from assistant_axis.judge_refusal_fallback import (  # noqa: E402
    build_content_for_mode,
    fill_one_allowlisted_gap,
    format_unexpected_banner,
    get_allowlist_entry,
    kind_of_entity_id,
    load_allowlist,
    name_of_entity_id,
    partition_gaps_by_allowlist,
)

DEFAULT_EXPERIMENT_GLOB = "roger/axis_judge_experiments/*_vs_*/sonnet"


def _safe_load_json(p: Path) -> Any:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_env() -> None:
    """Source .env so the API keys are available.  Same pattern as
    the minimal probe tool: avoids needing python-dotenv as a hard dep."""
    env_file = REPO / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def _build_axis_spec_from_config(cfg: dict) -> ajc.AxisSpec:
    """Reconstruct an AxisSpec from a cache dir's config.json.

    config.json was written by the original judging run; it contains
    everything the prompt template needs without re-running the corpus
    inventory.  The vector-shaped fields (``axis_by_slot``) aren't
    needed for prompt building, so we stub them with empty containers.
    """
    return ajc.AxisSpec(
        axis_name=cfg["axis_name"],
        pos_pole=cfg["pos_pole"],
        neg_pole=cfg["neg_pole"],
        pos_examples=cfg["pos_examples"],
        neg_examples=cfg["neg_examples"],
        axis_by_slot={},
        source_description=cfg.get("axis_source", ""),
        exclusions=cfg.get("exclusions", []),
    )


def _gaps_from_dir(d: Path) -> dict[str, list[str]]:
    """Read gaps.json from an axis judge dir, intersected with what's
    actually missing from each ``scores_<mode>.json`` cache.

    Why intersect: ``gaps.json`` can drift out of sync with the
    scores caches when a previous fill patched the cache without
    updating gaps.json (e.g. earlier versions of this tool, or hand-
    edits).  The scores cache is the source of truth; gaps.json is a
    convenience index.  Always reconcile against the cache so a
    second run of this tool after a successful first run is a
    no-op (idempotent).
    """
    gj = _safe_load_json(d / "gaps.json")
    if not gj:
        return {}
    out: dict[str, list[str]] = {}
    for mode, missing in gj.items():
        if not isinstance(missing, list) or not missing:
            continue
        scores_path = d / f"scores_{mode}.json"
        scores = _safe_load_json(scores_path) or {}
        present = set((scores.get("result", scores) or {}).keys())
        real_missing = [eid for eid in missing if eid not in present]
        if real_missing:
            out[mode] = real_missing
    return out


def _drop_filled_from_gaps_file(
    cache_dir: Path, mode: str, filled_eids: set[str],
) -> None:
    """Remove ``filled_eids`` from the ``gaps.json`` entry for ``mode``.

    Called after a successful fill so the source-of-truth index reflects
    the new state.  Atomic write.  If the per-mode list becomes empty
    we keep the key (with an empty list) for schema stability.
    """
    if not filled_eids:
        return
    gj_path = cache_dir / "gaps.json"
    if not gj_path.exists():
        return
    try:
        gj = json.loads(gj_path.read_text())
    except json.JSONDecodeError:
        return
    if mode not in gj:
        return
    current = list(gj.get(mode, []))
    gj[mode] = [eid for eid in current if eid not in filled_eids]
    from assistant_axis.atomic_io import atomic_write_text
    atomic_write_text(json.dumps(gj, indent=2) + "\n", gj_path)


def _judge_model_for_dir(d: Path) -> str | None:
    cfg = _safe_load_json(d / "config.json")
    if not cfg:
        return None
    return cfg.get("args", {}).get("judge_model")


def _scan_dirs(globs: list[str], explicit_dirs: list[str]) -> list[Path]:
    dirs: list[Path] = [Path(d) for d in explicit_dirs]
    for pat in globs:
        dirs.extend(Path(x) for x in sorted(_glob.glob(pat)))
    seen: set = set()
    out: list[Path] = []
    for d in dirs:
        if d in seen or not d.is_dir():
            continue
        seen.add(d)
        out.append(d)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--allowlist", default=None,
                   help="Path to judge_refusal_allowlist.json "
                        "(default: data/judge_refusal_allowlist.json).")
    p.add_argument("--glob", action="append", default=[],
                   help=f"Glob for axis-judge dirs (default: "
                        f"{DEFAULT_EXPERIMENT_GLOB!r}).  Repeatable.")
    p.add_argument("--dir", action="append", default=[],
                   help="Explicit axis-judge dir (repeatable).")
    p.add_argument("--scan", action="store_true",
                   help="Audit-only: list what would be filled; no API calls.")
    p.add_argument("--concurrency", type=int, default=3,
                   help="Parallel API calls (default: 3).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Per-call status lines.")
    args = p.parse_args()

    if not args.glob and not args.dir:
        args.glob = [DEFAULT_EXPERIMENT_GLOB]

    _load_env()
    allowlist = load_allowlist(
        Path(args.allowlist) if args.allowlist else None)
    if not allowlist.get("judge_refusals"):
        print(f"NOTE: allowlist is empty -- no fills will happen.",
              file=sys.stderr)

    dirs = _scan_dirs(args.glob, args.dir)
    print(f"Scanning {len(dirs)} axis-judge dirs for gaps...")

    # Collect: (dir, mode, eid, judge, entry) per allowlisted gap, and
    # (axis_label, mode, eid) per unexpected gap for the banner.
    jobs: list[tuple[Path, str, str, dict]] = []
    unexpected_by_judge: dict[str, list[tuple[str, str, str]]] = {}
    for d in dirs:
        gaps = _gaps_from_dir(d)
        if not gaps:
            continue
        judge = _judge_model_for_dir(d) or "<unknown>"
        for mode, eids in gaps.items():
            allowlisted, unexpected = partition_gaps_by_allowlist(
                eids, judge_model=judge, mode=mode, allowlist=allowlist,
            )
            for eid, entry in allowlisted:
                jobs.append((d, mode, eid, entry))
            for eid in unexpected:
                unexpected_by_judge.setdefault(judge, []).append(
                    (d.parent.name, mode, eid))

    n_unexpected = sum(len(v) for v in unexpected_by_judge.values())
    print(f"  {len(jobs)} allowlisted gaps to fill")
    print(f"  {n_unexpected} unexpected gaps")
    print()

    if not jobs and not n_unexpected:
        print("No gaps found.  Done.")
        return 0

    # Loud banner FIRST so it's visible even if the fill phase is slow.
    for judge, items in sorted(unexpected_by_judge.items()):
        print(format_unexpected_banner(items, judge), file=sys.stderr)

    if args.scan:
        print("Scan-only: showing what would be filled, no API calls.")
        for d, mode, eid, _ in jobs:
            print(f"  WOULD FILL: {d.parent.name} / {mode} / {eid}")
        return 0

    if not jobs:
        return 0 if not n_unexpected else 1

    # Build per-dir AxisSpec cache so we don't re-load config.json per job.
    cfg_cache: dict[Path, ajc.AxisSpec] = {}
    for d, _, _, _ in jobs:
        if d not in cfg_cache:
            cfg = _safe_load_json(d / "config.json") or {}
            cfg_cache[d] = _build_axis_spec_from_config(cfg)

    # Lazy import: only needed when actually filling.
    from anthropic import Anthropic
    client = Anthropic()

    print(f"Filling {len(jobs)} allowlisted gaps with "
          f"concurrency={args.concurrency}...")

    sem = asyncio.Semaphore(args.concurrency)

    async def _one(d: Path, mode: str, eid: str, entry: dict) -> dict:
        async with sem:
            spec = cfg_cache[d]

            def _prompt_builder(_eid: str, _mode: str) -> str:
                kind = kind_of_entity_id(_eid)
                name = name_of_entity_id(_eid)
                content = build_content_for_mode(kind, name, _mode)
                return ajc.build_static_prompt(spec, kind, name, content)

            r = await fill_one_allowlisted_gap(
                entity_id=eid, mode=mode,
                cache_path=d / f"scores_{mode}.json",
                allowlist_entry=entry,
                prompt_builder=_prompt_builder,
                parse_score=ajc.parse_signed_score,
                client=client, dry_run=False,
            )
            r["axis"] = d.parent.name  # add axis label for the summary
            if args.verbose and r.get("status") == "filled":
                u = r.get("usage", {})
                print(f"  filled {d.parent.name} / {mode} / {eid} = "
                      f"{r['score']:+d}  ({u.get('input_tokens', '?')}in / "
                      f"{u.get('output_tokens', '?')}out tokens)")
            return r

    results = asyncio.run(asyncio.gather(*(
        _one(d, mode, eid, entry) for d, mode, eid, entry in jobs
    )))

    # Update each affected gaps.json so the filled entries are removed.
    # Grouping by (dir, mode) so the file is touched once per mode even
    # when multiple fills happen there.
    fills_per_dir_mode: dict[tuple[Path, str], set[str]] = {}
    for (d, mode, eid, _), r in zip(jobs, results):
        if r.get("status") == "filled":
            fills_per_dir_mode.setdefault((d, mode), set()).add(eid)
    for (d, mode), eids in fills_per_dir_mode.items():
        _drop_filled_from_gaps_file(d, mode, eids)

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print()
    print(f"Fill results: {by_status}")

    failed = [r for r in results if r["status"] != "filled"]
    if failed:
        print(f"\n{len(failed)} fills failed:", file=sys.stderr)
        for r in failed[:15]:
            print(f"  {r['axis']} / {r['mode']} / {r['entity_id']}: "
                  f"{r['status']}: {r.get('detail', '')[:100]}",
                  file=sys.stderr)
        return 1
    return 0 if not n_unexpected else 2


if __name__ == "__main__":
    raise SystemExit(main())
