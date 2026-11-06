#!/usr/bin/env python3
"""Migrate post-pipeline derived files in a dataset's combinations/vectors/
into the new categorized derived/ layout introduced in Phase 1.0 of the
provenance redesign.

Old (legacy) layout::

    <dataset>/combinations/vectors/
        r_goal/<role>.pt
        r_nogoal/<trait>.pt
        t_goal/<trait>.pt
        t_nogoal/<role>.pt
        mean_r_combos.pt
        mean_t_combos.pt
        theatricality_axis.pt
        [optional]
        r_goal_legacy_centroid/<role>.pt        \
        r_nogoal_legacy_centroid/<trait>.pt      } only present on Roger 4-slot
        t_goal_legacy_centroid/<trait>.pt        |
        t_nogoal_legacy_centroid/<role>.pt      /
        [accidents]
        r_goal copy/, r_nogoal copy/, t_goal copy/, t_nogoal copy/   (Finder dupes)

New (categorized) layout::

    <dataset>/combinations/vectors/derived/
        marginals/
            r_goal/, r_nogoal/, t_goal/, t_nogoal/
        aggregates/
            mean_r_combos.pt, mean_t_combos.pt
        axis/
            theatricality_axis.pt
        legacy_centroid/                        # only when source had *_legacy_centroid/
            r_goal/, r_nogoal/, t_goal/, t_nogoal/

Pipeline-generated files (the per-pair r_*__*.pt / t_*__*.pt, the default.pt
symlink, and missing_vectors_audit/rerun) **stay in place** under
combinations/vectors/.

Behaviour
---------
* Idempotent: each item is moved only if the source exists and the destination
  does not.  If both exist, that's an error (don't silently clobber).
* Atomic per-item: uses ``os.replace`` for whole-tree renames inside the same
  filesystem.  No copy-then-delete dance, so a partial run leaves either the
  old layout or the new -- not both.
* Finder ' copy/' dirs (Roger 4-slot only): byte-equal-diffed against their
  canonical sibling; pruned only when fully identical.  Any difference halts
  the run and reports the divergent files.
* Dry-run mode (``--dry-run``): prints the planned actions without touching
  disk.

CLI
---
    uv run python tools/migrate_dataset_to_derived_layout.py \\
        --dataset 'runpod_workspace/qwen/qwen-3-32b Roger 8slot' \\
        [--dataset 'runpod_workspace/qwen/qwen-3-32b Roger'] \\
        [--prune-finder-copies] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import filecmp
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

LEGACY_MARGINAL_NAMES = ("r_goal", "r_nogoal", "t_goal", "t_nogoal")
LEGACY_CENTROID_NAMES = tuple(f"{n}_legacy_centroid" for n in LEGACY_MARGINAL_NAMES)
AGGREGATE_NAMES = ("mean_r_combos.pt", "mean_t_combos.pt")
AXIS_NAME = "theatricality_axis.pt"
FINDER_COPY_SUFFIX = " copy"


@dataclass
class Action:
    kind: str  # "move" | "prune" | "skip"
    src: Path
    dst: Path | None = None
    note: str = ""


@dataclass
class MigrationPlan:
    dataset: Path
    actions: list[Action] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _plan_for_dataset(
    dataset_root: Path,
    *,
    prune_finder_copies: bool,
) -> MigrationPlan:
    plan = MigrationPlan(dataset=dataset_root)
    cv = dataset_root / "combinations" / "vectors"
    if not cv.exists():
        plan.issues.append(f"no combinations/vectors/ in {dataset_root}")
        return plan

    derived = cv / "derived"
    marginals_root = derived / "marginals"
    aggregates_root = derived / "aggregates"
    axis_root = derived / "axis"
    legacy_centroid_root = derived / "legacy_centroid"

    def _add_move(src: Path, dst: Path, note: str = "") -> None:
        if not src.exists():
            plan.actions.append(Action("skip", src, dst, note=f"src absent ({note})"))
            return
        if dst.exists():
            # Allow idempotent re-runs: if both exist and are the same path
            # (filesystem-level), treat as already-done.
            try:
                if src.resolve() == dst.resolve():
                    plan.actions.append(
                        Action("skip", src, dst, note=f"already at destination ({note})")
                    )
                    return
            except OSError:
                pass
            plan.issues.append(
                f"both src and dst exist (src={src} dst={dst}); "
                f"refusing to overwrite. Inspect manually.")
            return
        plan.actions.append(Action("move", src, dst, note=note))

    for et in LEGACY_MARGINAL_NAMES:
        _add_move(cv / et, marginals_root / et, note=f"marginal {et}")
    for et in LEGACY_CENTROID_NAMES:
        base = et[: -len("_legacy_centroid")]
        _add_move(cv / et, legacy_centroid_root / base, note=f"legacy_centroid {et}")
    for fname in AGGREGATE_NAMES:
        _add_move(cv / fname, aggregates_root / fname, note=f"aggregate {fname}")
    _add_move(cv / AXIS_NAME, axis_root / AXIS_NAME, note="theatricality axis")

    if prune_finder_copies:
        for et in LEGACY_MARGINAL_NAMES:
            copy_dir = cv / f"{et}{FINDER_COPY_SUFFIX}"
            canonical = cv / et
            if not copy_dir.exists():
                continue
            if not canonical.exists():
                # Could happen post-move (canonical already moved into derived/).
                # Fall back to the new location.
                canonical = marginals_root / et
            if not canonical.exists():
                plan.issues.append(
                    f"{copy_dir} present but no canonical counterpart found at "
                    f"{cv / et} or {marginals_root / et}; refusing to prune.")
                continue
            issues = _diff_dirs(copy_dir, canonical)
            if issues:
                plan.issues.append(
                    f"{copy_dir} differs from canonical {canonical}; refusing to "
                    f"prune. Differences: {issues[:5]}"
                    + (" (... truncated)" if len(issues) > 5 else ""))
                continue
            plan.actions.append(
                Action("prune", copy_dir, note="byte-identical Finder dupe"))
    return plan


def _diff_dirs(a: Path, b: Path) -> list[str]:
    """Return a list of human-readable difference descriptions; empty if dirs
    are byte-identical (same set of files, each file equal)."""
    cmp_a = {p.name for p in a.iterdir() if p.is_file()}
    cmp_b = {p.name for p in b.iterdir() if p.is_file()}
    issues: list[str] = []
    only_a = cmp_a - cmp_b
    only_b = cmp_b - cmp_a
    if only_a:
        issues.append(f"only_in_copy: {sorted(only_a)[:5]}")
    if only_b:
        issues.append(f"only_in_canonical: {sorted(only_b)[:5]}")
    common = cmp_a & cmp_b
    cmp_result = filecmp.cmpfiles(a, b, common, shallow=False)
    differing = cmp_result[1]
    errs = cmp_result[2]
    if differing:
        issues.append(f"differing: {sorted(differing)[:5]}")
    if errs:
        issues.append(f"errors: {sorted(errs)[:5]}")
    return issues


def _execute(plan: MigrationPlan, *, dry_run: bool) -> bool:
    """Returns True on success, False on any failure."""
    if plan.issues:
        print(f"  [PLANNING ISSUES, refusing to execute]:")
        for it in plan.issues:
            print(f"    - {it}")
        return False
    moves = [a for a in plan.actions if a.kind == "move"]
    prunes = [a for a in plan.actions if a.kind == "prune"]
    skips = [a for a in plan.actions if a.kind == "skip"]
    print(f"  plan: {len(moves)} moves, {len(prunes)} prunes, {len(skips)} skips")

    for a in skips:
        print(f"    skip:  {a.src.relative_to(plan.dataset)}  ({a.note})")
    for a in moves:
        rel_src = a.src.relative_to(plan.dataset)
        rel_dst = a.dst.relative_to(plan.dataset)
        print(f"    move:  {rel_src}  ->  {rel_dst}  ({a.note})")
        if not dry_run:
            a.dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(a.src, a.dst)
    for a in prunes:
        rel = a.src.relative_to(plan.dataset)
        print(f"    prune: {rel}  ({a.note})")
        if not dry_run:
            _rmtree(a.src)
    return True


def _rmtree(p: Path) -> None:
    """Remove a directory tree (file-by-file, then rmdir).  Preserves the
    'safe' invariant -- we never call this on anything outside the migration
    plan, and only after a byte-equal diff against the canonical sibling.
    """
    for child in sorted(p.rglob("*"), key=lambda x: -len(str(x))):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    p.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True, type=Path,
                        help="Dataset root (the dir whose child is "
                             "'combinations/vectors/').  Pass once per "
                             "dataset to migrate.")
    parser.add_argument("--prune-finder-copies", action="store_true",
                        help="In each dataset, also prune any Finder-accident "
                             "'<etype> copy/' dirs after byte-equal diffing "
                             "them against their canonical sibling.  "
                             "Halts if any differ.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan and print but don't move/delete anything.")
    args = parser.parse_args()

    any_failed = False
    for ds in args.dataset:
        print(f"=== {ds} ===")
        if not ds.exists():
            print(f"  [SKIP] dataset does not exist")
            any_failed = True
            continue
        plan = _plan_for_dataset(ds, prune_finder_copies=args.prune_finder_copies)
        if not _execute(plan, dry_run=args.dry_run):
            any_failed = True
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
