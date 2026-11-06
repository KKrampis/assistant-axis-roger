#!/usr/bin/env python3
"""Audit ``deferred_rejudges.yaml`` for category-specific invariant
violations.

Companion to :mod:`tools.audit_caches` and :mod:`tools.audit_pngs`,
which surface deferred files in their own status reports.  Where
those two ask "is this on-disk artifact current, stale, or
deferred?", *this* tool asks "is each deferral entry itself still
coherent?" -- it never inspects file contents, only the registry
plus a snapshot of ``git ls-files`` and the on-disk path index.

Why this matters
----------------

Deferral entries accumulate over time and can quietly fall out of
sync with the codebase.  Two failure modes the audit catches:

1. **Orphan promoted to git** -- an ``orphan_no_producer`` entry
   names ``producer_script: /tmp/_canonical_angles_v1.py`` (or
   similar one-off path).  If a later refactor lands the same
   producer in the tree (e.g. ``results_analysis/canonical_angles.py``
   is a known successor), the deferral becomes stale: the producer
   is now in scope, the cache should be re-run rather than deferred.

   We detect a weaker signal automatically: any
   ``producer_script`` path that now ``git ls-files`` knows about
   is an alarm.  Stronger semantic detection ("this script is the
   spiritual successor of that orphan") is left to the maintainer.

2. **Successor disappeared** -- a ``superseded`` entry's
   ``replaced_by`` artifact has been deleted, so the deferral is
   referencing a path that no longer exists.  Either the deletion
   was intentional (deferral entry should be removed too) or it was
   accidental (the successor needs regenerating).

3. **Frozen-snapshot twin disappeared** -- a ``frozen_snapshot``'s
   ``compares_to`` path glob no longer resolves.  Comparison plots
   that read both snapshot + live data have lost half their
   provenance; either the live half was retired (snapshot category
   may need revisiting) or the live producer broke.

4. **Uncategorized entries** -- v1-schema entries that the May 2026
   backfill missed, or new entries added with the default
   ``--category=uncategorized``.  Listed for follow-up; not a hard
   error.

Severity levels
---------------

* ``error``   -- invariant violation that almost certainly needs
  action (orphan promoted; replaced_by missing; uncategorized
  entry).  Audit exits non-zero in this case (or with --strict).
* ``warning`` -- coherence concern that may be intentional
  (compares_to glob doesn't match anything; both snapshot + twin
  exist on disk so categorisation is debatable).  Audit exits zero.
* ``info``    -- per-category roll-up counts only.

CLI
---

::

    # Walk the registry, print Markdown report to stdout.
    uv run python tools/audit_deferrals.py

    # Save to file.
    uv run python tools/audit_deferrals.py --output reports/audit_deferrals.md

    # Exit non-zero on any error (CI-friendly).
    uv run python tools/audit_deferrals.py --strict

    # Plain-text instead of Markdown.
    uv run python tools/audit_deferrals.py --format text
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis.deferral_registry import (  # noqa: E402
    DeferralCategory,
    DeferralEntry,
    load_registry,
)


# ---------------------------------------------------------------------------
# Issue records
# ---------------------------------------------------------------------------

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass
class Issue:
    """One invariant violation tied to a registry entry."""
    severity: str          # "error" | "warning"
    entry: DeferralEntry
    invariant: str         # short tag like "orphan_promoted"
    detail: str            # human-readable explanation


@dataclass
class AuditReport:
    """Full audit result."""
    issues: list[Issue] = field(default_factory=list)
    counts_by_category: dict[DeferralCategory, int] = field(default_factory=dict)

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == SEVERITY_ERROR]

    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == SEVERITY_WARNING]


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _git_ls_files(repo_root: Path) -> set[str]:
    """Snapshot of files tracked by git, repo-relative.

    Returns an empty set if ``git`` is not available or the repo
    isn't a git checkout (the audit then degrades to filesystem-
    only checks).
    """
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _path_glob_matches_any_disk_file(
    pattern: str, repo_root: Path,
) -> bool:
    """True iff some on-disk path matches the registry's ``path_glob``.

    Uses ``Path.rglob`` for the simple suffix case; falls back to
    walking and ``fnmatch.fnmatch`` for patterns with leading ``*/``
    or other awkward shapes.  This is best-effort -- the registry
    treats ``fnmatch`` as authoritative, but for "does the successor
    artifact still exist?" we just need a yes/no for any one match.
    """
    if not pattern:
        return False
    # If pattern has no glob characters, simple existence check.
    if not any(c in pattern for c in "*?["):
        return (repo_root / pattern).exists()
    # Try rglob with the simple basename pattern; cheap when the
    # pattern is just a directory + filename.
    try:
        # A pattern like "roger/foo/bar_*.json" is a relative glob.
        if not pattern.startswith("/") and not pattern.startswith("*"):
            for _ in repo_root.glob(pattern):
                return True
            return False
    except (OSError, ValueError):
        pass
    # Fallback: walk roger/ + runpod_workspace/ tops and fnmatch.
    for top in ("roger", "runpod_workspace", "results_analysis", "tools"):
        topdir = repo_root / top
        if not topdir.exists():
            continue
        for p in topdir.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(repo_root))
            if fnmatch.fnmatch(rel, pattern):
                return True
    return False


# ---------------------------------------------------------------------------
# Per-category invariants
# ---------------------------------------------------------------------------

def _check_orphan(
    entry: DeferralEntry, *, git_files: set[str], repo_root: Path,
) -> list[Issue]:
    """``orphan_no_producer`` invariants.

    * If ``producer_script`` is set AND it's now in ``git ls-files``,
      the orphan was promoted -- raise an error so the maintainer
      lifts the deferral and re-runs the producer.
    """
    issues: list[Issue] = []
    if entry.producer_script and entry.producer_script in git_files:
        issues.append(Issue(
            severity=SEVERITY_ERROR, entry=entry,
            invariant="orphan_promoted",
            detail=(f"producer_script {entry.producer_script!r} is now "
                    "tracked by git -- the orphan has been promoted.  "
                    "Lift the deferral, re-run the producer, and let the "
                    "fresh envelope land naturally."),
        ))
    return issues


def _check_superseded(
    entry: DeferralEntry, *, repo_root: Path,
) -> list[Issue]:
    """``superseded`` invariants.

    * ``replaced_by`` should be set (if not, it's a coherence
      warning -- the deferral can't point at its successor).
    * ``replaced_by`` glob should match at least one on-disk file
      (otherwise the successor either wasn't produced or got
      deleted, and the "superseded" framing is now incoherent).
    """
    issues: list[Issue] = []
    if not entry.replaced_by:
        issues.append(Issue(
            severity=SEVERITY_WARNING, entry=entry,
            invariant="superseded_missing_replaced_by",
            detail=("category=superseded but no replaced_by field set; "
                    "consumers can't navigate to the successor."),
        ))
        return issues
    if not _path_glob_matches_any_disk_file(entry.replaced_by, repo_root):
        issues.append(Issue(
            severity=SEVERITY_ERROR, entry=entry,
            invariant="superseded_replaced_by_missing",
            detail=(f"replaced_by={entry.replaced_by!r} matches no file "
                    "on disk -- successor was deleted or never produced.  "
                    "Either remove this deferral entry or regenerate the "
                    "successor."),
        ))
    return issues


def _check_frozen_snapshot(
    entry: DeferralEntry, *, repo_root: Path,
) -> list[Issue]:
    """``frozen_snapshot`` invariants.

    * ``compares_to`` is recommended but not required.
    * If set, it should match a live twin on disk; otherwise the
      paired comparison plots have lost half their provenance.
    """
    issues: list[Issue] = []
    if entry.compares_to is None:
        return issues
    if not _path_glob_matches_any_disk_file(entry.compares_to, repo_root):
        issues.append(Issue(
            severity=SEVERITY_WARNING, entry=entry,
            invariant="frozen_snapshot_twin_missing",
            detail=(f"compares_to={entry.compares_to!r} matches no file "
                    "on disk -- the live twin used by paired comparison "
                    "plots is missing.  If the live producer was retired "
                    "the snapshot may belong in 'archived' instead."),
        ))
    return issues


def _check_uncategorized(entry: DeferralEntry) -> list[Issue]:
    return [Issue(
        severity=SEVERITY_ERROR, entry=entry,
        invariant="uncategorized",
        detail=("category=uncategorized; backfill or pass --category "
                "when re-adding via tools/defer_rejudge.py."),
    )]


# ---------------------------------------------------------------------------
# Top-level audit
# ---------------------------------------------------------------------------

def audit(repo_root: Optional[Path] = None) -> AuditReport:
    """Run all category-specific invariant checks.

    Returns an :class:`AuditReport` with per-entry issues plus
    per-category counts.  Caller decides what to do with errors
    (CLI exits non-zero on errors).
    """
    root = repo_root or _REPO_ROOT
    registry = load_registry(repo_root=root)
    git_files = _git_ls_files(root)
    report = AuditReport()
    for entry in registry:
        report.counts_by_category[entry.category] = (
            report.counts_by_category.get(entry.category, 0) + 1
        )
        if entry.category is DeferralCategory.UNCATEGORIZED:
            report.issues.extend(_check_uncategorized(entry))
            continue
        if entry.category is DeferralCategory.ORPHAN_NO_PRODUCER:
            report.issues.extend(
                _check_orphan(entry, git_files=git_files, repo_root=root)
            )
        elif entry.category is DeferralCategory.SUPERSEDED:
            report.issues.extend(
                _check_superseded(entry, repo_root=root)
            )
        elif entry.category is DeferralCategory.FROZEN_SNAPSHOT:
            report.issues.extend(
                _check_frozen_snapshot(entry, repo_root=root)
            )
        # Other categories (legacy_bare, archived, operational,
        # manifest_tracked, external_pipeline, experimental_one_off,
        # hand_curated_input) currently have no invariants beyond
        # "have a category at all", which is checked above.
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_markdown(report: AuditReport) -> str:
    out: list[str] = []
    total = sum(report.counts_by_category.values())
    n_err = len(report.errors())
    n_warn = len(report.warnings())
    out.append("# Deferral registry audit\n")
    out.append(f"**{total} entries**, **{n_err} errors**, **{n_warn} warnings**.\n")

    out.append("## Category roll-up\n")
    out.append("| Category | Count |")
    out.append("|---|---:|")
    for cat in sorted(report.counts_by_category, key=lambda c: c.value):
        out.append(f"| `{cat.value}` | {report.counts_by_category[cat]} |")
    out.append("")

    if report.errors():
        out.append("## Errors\n")
        for iss in report.errors():
            out.append(_format_issue_md(iss))
    else:
        out.append("## Errors\n\n*(none)*\n")

    if report.warnings():
        out.append("## Warnings\n")
        for iss in report.warnings():
            out.append(_format_issue_md(iss))
    else:
        out.append("## Warnings\n\n*(none)*\n")
    return "\n".join(out) + "\n"


def _format_issue_md(iss: Issue) -> str:
    lines = [
        f"- **{iss.invariant}** ({iss.entry.category.value})",
        f"  - `path_glob`: `{iss.entry.path_glob}`",
        f"  - {iss.detail}",
    ]
    if iss.entry.producer_script:
        lines.append(f"  - `producer_script`: `{iss.entry.producer_script}`")
    if iss.entry.replaced_by:
        lines.append(f"  - `replaced_by`: `{iss.entry.replaced_by}`")
    if iss.entry.compares_to:
        lines.append(f"  - `compares_to`: `{iss.entry.compares_to}`")
    return "\n".join(lines)


def _format_text(report: AuditReport) -> str:
    out: list[str] = []
    total = sum(report.counts_by_category.values())
    out.append(f"== Deferral registry audit ({total} entries) ==")
    for cat in sorted(report.counts_by_category, key=lambda c: c.value):
        out.append(f"  {cat.value:25s} {report.counts_by_category[cat]:4d}")
    out.append("")
    if not report.issues:
        out.append("(no issues)")
        return "\n".join(out)
    for sev_label, items in (("ERRORS", report.errors()),
                              ("WARNINGS", report.warnings())):
        if not items:
            continue
        out.append(f"-- {sev_label} ({len(items)}) --")
        for iss in items:
            out.append(f"  [{iss.invariant}] ({iss.entry.category.value}) "
                       f"{iss.entry.path_glob}")
            out.append(f"      {iss.detail}")
        out.append("")
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", help="Path to write the report (default: stdout).")
    p.add_argument("--format", choices=("markdown", "text"), default="markdown")
    p.add_argument("--strict", action="store_true",
                   help=("Exit non-zero on any error (CI-friendly).  "
                         "Without --strict the CLI always exits 0; the "
                         "report still reflects errors / warnings."))
    args = p.parse_args(argv)

    report = audit(repo_root=_REPO_ROOT)
    body = (_format_markdown(report) if args.format == "markdown"
            else _format_text(report))
    if args.output:
        Path(args.output).write_text(body)
    else:
        sys.stdout.write(body)

    if args.strict and report.errors():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
