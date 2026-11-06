#!/usr/bin/env python3
"""Walk a tree of committed PNGs, validate each one's embedded ``Inputs``
chunk against current dataset / file state, and emit a markdown report.

This is the "Phase 5a" companion to
:mod:`tools.audit_committed_pngs` (which classifies by recorded
``DEFAULT_DATA_DIR`` vintage).  Where ``audit_committed_pngs`` answers
"which Roger / Christina dataset did this come from?", *this* tool
answers "are the recorded inputs of this plot still current?".

A PNG is classified into exactly one of four buckets based on its
embedded :class:`assistant_axis.provenance.InputSpec` list:

- **current**: every recorded input matches the current state of the
  dataset / cache it points at.  Plot is fresh.
- **stale**: at least one input has drifted, gone missing, or is a
  legacy multi-input we can't re-verify (``unverifiable``).  Plot
  should probably be regenerated.
- **legacy**: the PNG has no ``Inputs`` chunk.  Either pre-Phase-3
  (before the provenance system existed) or produced by a writer
  not yet migrated.  Audit can't say anything about it.
- **frozen**: the PNG carries an explicit ``Frozen`` or
  ``Frozen-At`` chunk marking it as intentionally pinned to a
  particular dataset version (e.g. historical snapshots in
  ``roger/pc_round_trip_4slot_archive/``).  Skipped from drift
  reporting.

The "frozen" marker is opt-in -- you add it via ``png_metadata(...,
extra={"Frozen": "snapshot-2026-04"})`` in the producing script when
you want to keep an artifact even after upstream changes.  No
existing PNGs use it yet; it's a forward-looking escape hatch.

CLI
---

::

    # Default: walk roger/, validate, write markdown to stdout.
    uv run python tools/audit_pngs.py

    # Save report to file.
    uv run python tools/audit_pngs.py --output reports/audit_pngs.md

    # Show only stale PNGs.
    uv run python tools/audit_pngs.py --status stale

    # Restrict to a subtree.
    uv run python tools/audit_pngs.py --root roger/axis_judge_experiments

    # Print one-line per PNG to stdout (no markdown), good for grep:
    uv run python tools/audit_pngs.py --format flat

Output
------

Markdown sections:

* **Summary** -- count per status, per-producing-script breakdown.
* **Stale** -- table of stale PNGs with the dep_keys that drifted /
  vanished / are unverifiable, formatted for clipboard-friendly
  re-run commands.
* **Legacy** -- one row per legacy PNG with its ``Software`` /
  recorded ``Source`` (git SHA) so you can find the producing
  commit.
* **Frozen** -- listed only if any are present.

Performance
-----------

Each PNG read pulls only metadata (PIL lazy-loads pixel data); a
manifest cache means ``read_manifest`` runs once per dataset root no
matter how many subtree InputSpecs reference it.  ``roger/`` has
~250 PNGs; full audit completes in a few seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Allow ``python tools/audit_pngs.py`` from the repo root without PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image  # noqa: E402

from assistant_axis.provenance import (  # noqa: E402
    InputSpec,
    ProvenanceCheck,
    inputs_from_jsonable,
    validate_recorded,
)
from assistant_axis.deferral_registry import (  # noqa: E402
    DeferralEntry,
    load_registry as load_deferral_registry,
    match_cache as match_deferral,
)


STATUSES = ("current", "stale", "deferred", "legacy", "frozen")


@dataclass
class PNGAudit:
    """Result of auditing one PNG."""
    path: Path
    status: str           # one of STATUSES
    title: Optional[str] = None
    software: Optional[str] = None
    creation_time: Optional[str] = None
    git_sha: Optional[str] = None
    inputs: list = field(default_factory=list)   # list[InputSpec]
    check: Optional[ProvenanceCheck] = None
    frozen_marker: Optional[str] = None
    notes: list = field(default_factory=list)
    deferral_matches: list = field(default_factory=list)  # list[DeferralEntry]
    pre_deferred_status: Optional[str] = None             # status before deferral pass

    def short_script(self) -> str:
        """Best-effort producing-script name; matches
        ``audit_committed_pngs._short_script`` so the two reports can be
        cross-referenced."""
        if not self.software:
            return "(unknown source)"
        if self.software.startswith("Matplotlib version"):
            return "(legacy: pre-provenance, no wrapper)"
        # ``uv run python results_analysis/foo.py --opt v`` -> ``results_analysis/foo.py``
        toks = self.software.split()
        for t in toks:
            if t.endswith(".py") or t.startswith("-m"):
                return t if t.endswith(".py") else " ".join(
                    toks[toks.index(t):toks.index(t) + 2])
        return self.software

    def stale_reasons(self) -> list[str]:
        """One human-readable reason per non-ok status."""
        if self.check is None:
            return []
        out: list[str] = []
        for s in self.check.statuses:
            if s.status != "ok":
                out.append(f"{s.dep_key}: {s.status}"
                           + (f" -- {s.detail}" if s.detail else ""))
        return out


# ---------------------------------------------------------------------------
# PNG inspection
# ---------------------------------------------------------------------------

def _read_png_info(path: Path) -> dict:
    """Extract the PNG iTXt/tEXt chunks as a dict.  PIL handles both."""
    with Image.open(path) as im:
        # ``info`` is populated lazily on open; force a load of header chunks.
        return dict(im.info)


def audit_png(path: Path) -> PNGAudit:
    """Inspect one PNG and return its audit record."""
    try:
        info = _read_png_info(path)
    except Exception as e:  # noqa: BLE001 -- we don't want to fail the whole walk
        return PNGAudit(path=path, status="legacy",
                        notes=[f"PIL failed to open: {e}"])

    title = info.get("Title")
    software = info.get("Software")
    creation_time = info.get("Creation Time")
    git_sha = info.get("Source")  # "git abc123" or "git abc123+dirty"
    frozen_marker = info.get("Frozen") or info.get("Frozen-At")

    if frozen_marker:
        return PNGAudit(path=path, status="frozen",
                        title=title, software=software,
                        creation_time=creation_time, git_sha=git_sha,
                        frozen_marker=frozen_marker)

    inputs_blob = info.get("Inputs")
    if inputs_blob is None:
        return PNGAudit(path=path, status="legacy",
                        title=title, software=software,
                        creation_time=creation_time, git_sha=git_sha)

    try:
        recorded_dicts = json.loads(inputs_blob)
    except json.JSONDecodeError as e:
        return PNGAudit(path=path, status="legacy",
                        title=title, software=software,
                        creation_time=creation_time, git_sha=git_sha,
                        notes=[f"Inputs chunk is not valid JSON: {e}"])

    if not isinstance(recorded_dicts, list):
        return PNGAudit(path=path, status="legacy",
                        title=title, software=software,
                        creation_time=creation_time, git_sha=git_sha,
                        notes=["Inputs chunk is not a JSON list."])

    if not recorded_dicts:
        # Writer migrated but declared no deps.  Treat as current; surface
        # via a note so the reader can investigate if surprising.
        return PNGAudit(path=path, status="current",
                        title=title, software=software,
                        creation_time=creation_time, git_sha=git_sha,
                        notes=["Inputs chunk is empty (writer migrated, "
                               "no deps recorded)."])

    try:
        recorded = inputs_from_jsonable(recorded_dicts)
    except Exception as e:  # noqa: BLE001
        return PNGAudit(path=path, status="legacy",
                        title=title, software=software,
                        creation_time=creation_time, git_sha=git_sha,
                        notes=[f"Failed to parse Inputs JSON: {e}"])

    check = validate_recorded(recorded)
    status = "current" if check.ok else "stale"
    return PNGAudit(path=path, status=status,
                    title=title, software=software,
                    creation_time=creation_time, git_sha=git_sha,
                    inputs=recorded, check=check)


# ---------------------------------------------------------------------------
# Walk + report
# ---------------------------------------------------------------------------

def apply_deferrals(
    audits: list[PNGAudit],
    *,
    registry: Optional[list[DeferralEntry]] = None,
    repo_root: Optional[Path] = None,
) -> None:
    """Mutate ``audits`` in-place: reclassify rows to ``deferred`` when
    an entry in ``deferred_rejudges.yaml`` matches the PNG's path (and
    optionally the drift-source dep_key).

    Reclassifies either ``stale`` ("this would otherwise need
    rerunning; defer instead") or ``legacy`` ("this never had an
    ``Inputs`` chunk -- pre-migration or a producer we've decided not
    to migrate; presume current as of mechanism introduction").
    ``current`` and ``frozen`` are never reclassified.
    """
    if registry is None:
        registry = load_deferral_registry(repo_root=repo_root)
    if not registry:
        return
    deferrable = ("stale", "legacy")
    for a in audits:
        if a.status not in deferrable:
            continue
        dep_key: Optional[str] = None
        if a.check is not None:
            offending = [
                s for s in a.check.statuses
                if s.status not in ("ok", "equivalent")
            ]
            if len(offending) == 1:
                dep_key = offending[0].dep_key
        matches = match_deferral(
            a.path.resolve(), dep_key=dep_key, registry=registry,
            repo_root=repo_root,
        )
        if matches:
            a.pre_deferred_status = a.status
            a.status = "deferred"
            a.deferral_matches = matches


def collect(root: Path) -> list[PNGAudit]:
    """Recursively audit every ``*.png`` under ``root``."""
    rows: list[PNGAudit] = []
    for path in sorted(root.rglob("*.png")):
        rows.append(audit_png(path))
    return rows


def render_markdown(
    rows: list[PNGAudit],
    *,
    root: Path,
    status_filter: Optional[str] = None,
) -> str:
    """Produce the markdown audit report."""
    if status_filter is not None:
        rows = [r for r in rows if r.status == status_filter]

    lines: list[str] = []
    lines.append(f"# PNG provenance audit: `{root}`\n")
    lines.append(f"Total PNGs: **{len(rows)}**.  "
                 f"Status legend: current / stale / deferred / legacy / "
                 f"frozen (see ``tools/audit_pngs.py`` docstring).\n")

    # Summary by status.
    by_status = Counter(r.status for r in rows)
    lines.append("## Summary by status\n")
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for st in STATUSES:
        lines.append(f"| {st} | {by_status.get(st, 0)} |")
    lines.append("")

    # Per-script breakdown.
    by_script: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_script[r.short_script()][r.status] += 1
    lines.append("## By producing script\n")
    lines.append("| Script | current | stale | deferred | legacy | frozen | total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for script in sorted(by_script):
        c = by_script[script]
        total = sum(c.values())
        lines.append(
            f"| `{script}` | {c.get('current', 0)} | {c.get('stale', 0)} "
            f"| {c.get('deferred', 0)} | {c.get('legacy', 0)} | "
            f"{c.get('frozen', 0)} | {total} |"
        )
    lines.append("")

    # Stale details.
    stale_rows = [r for r in rows if r.status == "stale"]
    if stale_rows:
        lines.append("## Stale PNGs\n")
        lines.append("Each row lists the dep_keys that drifted/vanished/are "
                     "unverifiable, plus the recorded ``Software`` so you "
                     "can re-run the producer.\n")
        for r in sorted(stale_rows, key=lambda r: str(r.path)):
            lines.append(f"### `{r.path}`")
            if r.creation_time:
                lines.append(f"- **Created**: {r.creation_time}"
                             + (f"  (git {r.git_sha})" if r.git_sha else ""))
            if r.software:
                lines.append(f"- **Software**: `{r.software}`")
            for reason in r.stale_reasons():
                lines.append(f"- {reason}")
            lines.append("")

    # Deferred details (matched by deferred_rejudges.yaml).
    deferred_rows = [r for r in rows if r.status == "deferred"]
    if deferred_rows:
        lines.append(f"## Deferred PNGs ({len(deferred_rows)})\n")
        lines.append("PNGs that would otherwise be ``stale`` but match an "
                     "entry in ``deferred_rejudges.yaml``.  Use "
                     "``tools/defer_rejudge.py --remove`` to lift.  Run "
                     "``tools/audit_deferrals.py`` for category-specific "
                     "invariant checks (e.g. promotion-detection on "
                     "``orphan_no_producer`` entries).\n")

        by_category: dict[str, list] = defaultdict(list)
        for r in deferred_rows:
            cat = (r.deferral_matches[0].category.value
                   if r.deferral_matches else "uncategorized")
            by_category[cat].append(r)
        lines.append("| Category | Count |")
        lines.append("|---|---:|")
        for cat in sorted(by_category):
            lines.append(f"| `{cat}` | {len(by_category[cat])} |")
        lines.append("")

        for cat in sorted(by_category):
            lines.append(f"### Category: `{cat}` "
                         f"({len(by_category[cat])})\n")
            for r in sorted(by_category[cat], key=lambda r: str(r.path)):
                lines.append(f"#### `{r.path}`")
                if r.pre_deferred_status:
                    lines.append(f"- **Was**: {r.pre_deferred_status}")
                for entry in r.deferral_matches:
                    lines.append(
                        f"- **Deferred by**: `{entry.path_glob}`"
                        + (f" (dep_key=`{entry.dep_key}`)"
                           if entry.dep_key else "")
                        + f" -- ({entry.category.value}) {entry.reason}"
                        + (f"  *(at {entry.deferred_at})*"
                           if entry.deferred_at else "")
                    )
                    if entry.producer_script:
                        lines.append(f"  - `producer_script`: "
                                     f"`{entry.producer_script}`")
                    if entry.replaced_by:
                        lines.append(f"  - `replaced_by`: "
                                     f"`{entry.replaced_by}`")
                    if entry.compares_to:
                        lines.append(f"  - `compares_to`: "
                                     f"`{entry.compares_to}`")
                lines.append("")

    # Legacy details (compact -- often dominant; kept short).
    legacy_rows = [r for r in rows if r.status == "legacy"]
    if legacy_rows:
        lines.append(f"## Legacy PNGs ({len(legacy_rows)})\n")
        lines.append("PNGs without an ``Inputs`` chunk -- pre-Phase-3, or "
                     "produced by a writer that hasn't been migrated yet.\n")
        lines.append("| Path | Software | Created | Git SHA | Note |")
        lines.append("|---|---|---|---|---|")
        for r in sorted(legacy_rows, key=lambda r: str(r.path)):
            note = "; ".join(r.notes) if r.notes else ""
            lines.append(
                f"| `{r.path}` | `{r.software or ''}` | "
                f"{r.creation_time or ''} | {r.git_sha or ''} | {note} |"
            )
        lines.append("")

    # Frozen details.
    frozen_rows = [r for r in rows if r.status == "frozen"]
    if frozen_rows:
        lines.append(f"## Frozen PNGs ({len(frozen_rows)})\n")
        lines.append("Intentionally pinned to a historical dataset version "
                     "(skipped from drift reporting).\n")
        for r in sorted(frozen_rows, key=lambda r: str(r.path)):
            lines.append(f"- `{r.path}` -- frozen marker: "
                         f"`{r.frozen_marker}`")
        lines.append("")

    return "\n".join(lines)


def render_flat(rows: list[PNGAudit]) -> str:
    """One line per PNG: ``status<TAB>path<TAB>note``.  For grep / awk."""
    out: list[str] = []
    for r in rows:
        if r.status == "stale":
            note = "; ".join(r.stale_reasons())
        elif r.status == "deferred":
            note = (f"was={r.pre_deferred_status}; "
                    + "; ".join(e.reason for e in r.deferral_matches))
        else:
            note = ""
        out.append(f"{r.status}\t{r.path}\t{note}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", type=str, default="roger",
                   help="Directory to walk recursively (default: roger).")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Write report to this file instead of stdout.")
    p.add_argument("--status", choices=STATUSES, default=None,
                   help="Filter to PNGs with this status only.")
    p.add_argument("--format", choices=("markdown", "flat"), default="markdown",
                   help="Output format (default: markdown).")
    p.add_argument("--ignore-deferrals", action="store_true",
                   help="Skip applying deferred_rejudges.yaml; show "
                        "underlying ``stale`` classification regardless of "
                        "registry entries.")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"error: --root {root!r} is not a directory", file=sys.stderr)
        return 2

    rows = collect(root)
    if not args.ignore_deferrals:
        apply_deferrals(rows)
    if args.status is not None:
        rows = [r for r in rows if r.status == args.status]

    if args.format == "markdown":
        body = render_markdown(rows, root=root, status_filter=None)
    else:
        body = render_flat(rows)

    if args.output:
        Path(args.output).write_text(body)
        # Stat for the user.
        cnts = Counter(r.status for r in rows)
        summary = ", ".join(f"{k}={v}" for k, v in sorted(cnts.items()))
        print(f"Wrote {args.output} ({len(rows)} PNGs: {summary}).",
              file=sys.stderr)
    else:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
