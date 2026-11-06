#!/usr/bin/env python3
"""Walk a tree of JSON caches, validate each provenance envelope against
current dataset/file state, and propagate stale-ness transitively
through ``kind="file"`` dependencies between caches.

This is the "Phase 5b" companion to :mod:`tools.audit_pngs` (which
classifies committed PNG outputs).  Where ``audit_pngs`` answers "is
this plot fresh?", *this* tool answers "are the caches that feed
plots / further analyses still current, including via transitive
dependencies?".

Direct vs transitive drift
--------------------------

A cache ``A`` records a file dep on cache ``B`` via ``InputSpec``::

    InputSpec(kind="file", path="roger/.../B.json",
              fingerprint="v1:<mtime>@<size>")

There are two ways ``A`` can become stale:

1. **Direct drift**: ``B``'s mtime/size has changed since ``A`` was
   produced -- ``A``'s recorded file-fingerprint no longer matches
   the current one.  This is what
   :func:`assistant_axis.provenance.validate_recorded` already
   reports.

2. **Transitive drift**: ``B`` is *itself* stale -- e.g. ``B``'s
   judge-cache deps drifted, but ``B`` has not been regenerated, so
   its mtime/size has not changed, so ``A``'s recorded file-fp for
   ``B`` still matches.  ``A``'s own validation would say "ok", but
   reading ``A`` actually gives you data computed from stale ``B``.

This tool fixes (2) by walking all envelope-wrapped JSONs in the
tree, building a directed graph (edge ``A -> B`` when ``A`` declares a
file dep on ``B``), and propagating stale-ness along that graph in a
second pass.

Status legend
-------------

- **current**: own provenance validates ok, and no transitive deps
  are stale.
- **stale_direct**: own provenance has at least one drifted /
  missing / unverifiable dep.
- **stale_transitive**: own provenance is ok, but at least one
  upstream cache (reachable via file deps) is stale.
- **legacy**: no ``_provenance`` envelope (pre-migration cache, or
  a hand-written / pipeline-produced JSON like ``MANIFEST.json``,
  ``pair_list_*.json``, judge-score caches).  Skipped from drift
  analysis; passes through transparently if downstream caches
  reference it.

CLI
---

::

    # Default: walk roger/, write markdown to stdout.
    uv run python tools/audit_caches.py

    # Save report to file:
    uv run python tools/audit_caches.py --output reports/audit_caches.md

    # Status filter (e.g. "show me everything stale, direct or transitive"):
    uv run python tools/audit_caches.py --status stale

    # Include runpod_workspace (warning: large; mostly legacy judge caches):
    uv run python tools/audit_caches.py --root roger --root runpod_workspace

Performance
-----------

Each JSON is read once and parsed once; ``read_manifest`` is cached
internally by :mod:`assistant_axis.provenance`.  ``roger/`` has a few
hundred JSONs and audits in <1s.  Adding ``runpod_workspace/`` is
slower (~thousands of judge caches) but only the envelope-wrapped
files are validated; legacy JSONs are inspected only enough to
classify them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis.provenance import (  # noqa: E402
    ProvenanceCheck,
    inputs_from_jsonable,
    validate_recorded,
)
from assistant_axis.deferral_registry import (  # noqa: E402
    DeferralEntry,
    load_registry as load_deferral_registry,
    match_cache as match_deferral,
)


STATUSES = ("current", "stale_direct", "stale_transitive", "deferred", "legacy")
STALE_STATUSES = ("stale_direct", "stale_transitive")


@dataclass
class CacheAudit:
    """Result of auditing one JSON cache."""
    path: Path
    abs_path: Path
    status: str           # one of STATUSES
    title: Optional[str] = None
    cmd: Optional[str] = None
    git_sha: Optional[str] = None
    produced_at: Optional[str] = None
    inputs: list = field(default_factory=list)         # list[InputSpec]
    check: Optional[ProvenanceCheck] = None
    notes: list = field(default_factory=list)
    transitive_stale_via: list = field(default_factory=list)  # list[Path]
    deferral_matches: list = field(default_factory=list)  # list[DeferralEntry]
    pre_deferred_status: Optional[str] = None             # status before deferral pass

    def short_script(self) -> str:
        """Producer script name from the recorded ``cmd``.

        ``cmd`` follows the convention used by
        :mod:`assistant_axis.plot_metadata`:
        ``uv run python results_analysis/foo.py --opt v``
        or ``uv run python -m foo.bar --opt v``.
        """
        if not self.cmd:
            return "(unknown source)"
        toks = self.cmd.split()
        for i, t in enumerate(toks):
            if t.endswith(".py"):
                return t
            if t == "-m" and i + 1 < len(toks):
                return f"-m {toks[i + 1]}"
        return self.cmd

    def stale_reasons(self) -> list[str]:
        """Direct drift reasons (one line per non-ok dep)."""
        if self.check is None:
            return []
        return [
            f"{s.dep_key}: {s.status}"
            + (f" -- {s.detail}" if s.detail else "")
            for s in self.check.statuses if s.status != "ok"
        ]


def _is_envelope(obj) -> bool:
    """Match the envelope schema used by ``json_metadata``."""
    return (isinstance(obj, dict)
            and "result" in obj
            and isinstance(obj.get("_provenance"), dict))


def _read_json(path: Path) -> tuple[Optional[dict], Optional[str]]:
    """Read JSON, return ``(obj, error)``.

    Returns ``(None, error_message)`` for unreadable / non-JSON files
    (we just classify those as legacy without crashing the walk)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"read failed: {e}"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"not valid JSON: {e}"


def audit_cache(path: Path) -> CacheAudit:
    """Inspect one JSON file and return its audit record."""
    abs_path = path.resolve()
    obj, err = _read_json(path)
    if err is not None:
        return CacheAudit(path=path, abs_path=abs_path, status="legacy",
                          notes=[err])
    if not _is_envelope(obj):
        return CacheAudit(path=path, abs_path=abs_path, status="legacy")

    prov = obj.get("_provenance", {})
    produced_by = prov.get("produced_by") or {}
    title = produced_by.get("title")
    cmd = produced_by.get("cmd")
    git_sha = produced_by.get("git_sha")
    produced_at = prov.get("produced_at")

    inputs_blob = prov.get("inputs") or []
    if not isinstance(inputs_blob, list):
        return CacheAudit(path=path, abs_path=abs_path, status="legacy",
                          title=title, cmd=cmd, git_sha=git_sha,
                          produced_at=produced_at,
                          notes=["inputs field is not a list"])

    if not inputs_blob:
        # Migrated writer with no declared deps; treat as current with a note.
        return CacheAudit(path=path, abs_path=abs_path, status="current",
                          title=title, cmd=cmd, git_sha=git_sha,
                          produced_at=produced_at,
                          notes=["empty inputs list"])

    try:
        recorded = inputs_from_jsonable(inputs_blob)
    except Exception as e:  # noqa: BLE001
        return CacheAudit(path=path, abs_path=abs_path, status="legacy",
                          title=title, cmd=cmd, git_sha=git_sha,
                          produced_at=produced_at,
                          notes=[f"failed to parse inputs JSON: {e}"])

    check = validate_recorded(recorded)
    status = "current" if check.ok else "stale_direct"
    return CacheAudit(path=path, abs_path=abs_path, status=status,
                      title=title, cmd=cmd, git_sha=git_sha,
                      produced_at=produced_at,
                      inputs=recorded, check=check)


# ---------------------------------------------------------------------------
# Transitive stale propagation
# ---------------------------------------------------------------------------

def apply_deferrals(
    audits: list[CacheAudit],
    *,
    registry: Optional[list[DeferralEntry]] = None,
    repo_root: Optional[Path] = None,
) -> None:
    """Mutate ``audits`` in-place: reclassify rows to ``deferred`` when
    a deferral entry in ``deferred_rejudges.yaml`` matches the cache's
    path (and optionally the drift-source dep_key).

    Reclassifies any of the following pre-deferred statuses:

    * ``stale_direct`` / ``stale_transitive`` -- "this would otherwise
      need rerunning; defer instead".
    * ``legacy`` -- "this never had an envelope (pre-migration bare
      JSON, or a producer we've decided not to migrate); presume
      current as of mechanism introduction".  Useful for the May 2026
      backfill of judge caches: ``axis_judge_correlation.py`` was
      migrated but its existing on-disk caches were written before
      that, and we don't intend to re-judge them.

    ``current`` is intentionally NOT reclassified -- a freshly
    validated cache shouldn't be "deferred" just because its path
    happens to match.

    Deferred caches retain their original status in
    ``pre_deferred_status`` for human inspection, and the matched
    entries are recorded in ``deferral_matches`` so reports can show
    why each cache was deferred.

    Run this BEFORE :func:`propagate_transitive_stale`: deferring an
    upstream cache means we've explicitly chosen to treat it as a
    fixed-baseline (not stale).  Propagating staleness through it
    would falsely taint every downstream consumer as
    ``stale_transitive``, even when the consumer's recorded
    fingerprint of the deferred cache is perfectly current.  By
    deferring first, the propagation BFS later skips deferred nodes
    entirely (their status is no longer in ``STALE_STATUSES``).
    """
    if registry is None:
        registry = load_deferral_registry(repo_root=repo_root)
    if not registry:
        return
    deferrable = (*STALE_STATUSES, "legacy")
    for a in audits:
        if a.status not in deferrable:
            continue
        # Drift-source dep_key: if the cache validated `stale_direct`
        # with exactly one drifted/missing input, pass that dep_key to
        # the matcher so dep-key-scoped registry entries can target it.
        # Otherwise (multi-drift, transitive, or legacy without an
        # envelope), pass None and let path_glob alone govern.
        dep_key: Optional[str] = None
        if a.check is not None:
            offending = [
                s for s in a.check.statuses
                if s.status not in ("ok", "equivalent")
            ]
            if len(offending) == 1:
                dep_key = offending[0].dep_key
        matches = match_deferral(
            a.abs_path, dep_key=dep_key, registry=registry,
            repo_root=repo_root,
        )
        if matches:
            a.pre_deferred_status = a.status
            a.status = "deferred"
            a.deferral_matches = matches


def propagate_transitive_stale(audits: list[CacheAudit]) -> None:
    """Mutate ``audits`` in-place: any audit currently ``current`` whose
    file deps reach (transitively) a stale audit gets reclassified as
    ``stale_transitive``, with ``transitive_stale_via`` listing the
    direct upstream caches that caused the propagation.
    """
    by_abs: dict[Path, CacheAudit] = {a.abs_path: a for a in audits}

    # Build directed graph: edge ``A -> B`` when A declares a file dep on B
    # (and B is in our audit set).  Only file kind matters here -- subtree
    # and multi inputs aren't intra-cache references.
    out_edges: dict[Path, list[Path]] = defaultdict(list)
    for a in audits:
        for spec in a.inputs:
            if spec.kind != "file":
                continue
            target = (_REPO_ROOT / spec.path).resolve()
            if target in by_abs:
                out_edges[a.abs_path].append(target)

    # Iteratively propagate: a node is "tainted" if it's stale_direct or
    # any of its out-edges points to a tainted node.  We BFS up from each
    # already-stale node along *reverse* edges to mark its dependents.
    rev_edges: dict[Path, list[Path]] = defaultdict(list)
    for src, dests in out_edges.items():
        for d in dests:
            rev_edges[d].append(src)

    initially_stale = [a.abs_path for a in audits
                       if a.status in STALE_STATUSES]
    queue: deque = deque(initially_stale)
    while queue:
        node = queue.popleft()
        for parent in rev_edges.get(node, []):
            pa = by_abs[parent]
            if pa.status == "current":
                pa.status = "stale_transitive"
                pa.transitive_stale_via.append(by_abs[node].path)
                queue.append(parent)
            elif pa.status == "stale_transitive":
                # Already tainted; just remember this contributor.
                if by_abs[node].path not in pa.transitive_stale_via:
                    pa.transitive_stale_via.append(by_abs[node].path)


# ---------------------------------------------------------------------------
# Walk + render
# ---------------------------------------------------------------------------

def collect(roots: list[Path]) -> list[CacheAudit]:
    """Walk every ``*.json`` under each root and audit it."""
    audits: list[CacheAudit] = []
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            ap = path.resolve()
            if ap in seen:
                continue
            seen.add(ap)
            audits.append(audit_cache(path))
    return audits


def render_markdown(
    rows: list[CacheAudit],
    *,
    roots: list[Path],
    status_filter: Optional[str] = None,
) -> str:
    """Build the markdown audit report."""
    if status_filter is not None:
        if status_filter == "stale":
            rows = [r for r in rows if r.status in STALE_STATUSES]
        else:
            rows = [r for r in rows if r.status == status_filter]

    lines: list[str] = []
    lines.append("# Cache provenance audit\n")
    lines.append("Roots: " + ", ".join(f"`{r}`" for r in roots) + "\n")
    lines.append(f"Total JSONs scanned: **{len(rows)}**.\n")

    by_status = Counter(r.status for r in rows)
    lines.append("## Summary by status\n")
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for st in STATUSES:
        lines.append(f"| {st} | {by_status.get(st, 0)} |")
    lines.append("")

    by_script: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_script[r.short_script()][r.status] += 1
    lines.append("## By producing script\n")
    lines.append("| Script | current | stale_direct | stale_transitive | deferred | legacy | total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for script in sorted(by_script):
        c = by_script[script]
        total = sum(c.values())
        lines.append(
            f"| `{script}` | {c.get('current', 0)} | "
            f"{c.get('stale_direct', 0)} | {c.get('stale_transitive', 0)} | "
            f"{c.get('deferred', 0)} | {c.get('legacy', 0)} | {total} |"
        )
    lines.append("")

    direct_stale = [r for r in rows if r.status == "stale_direct"]
    if direct_stale:
        lines.append(f"## Stale (direct): {len(direct_stale)}\n")
        for r in sorted(direct_stale, key=lambda r: str(r.path)):
            lines.append(f"### `{r.path}`")
            if r.produced_at:
                lines.append(f"- **Produced**: {r.produced_at}"
                             + (f"  (git {r.git_sha})" if r.git_sha else ""))
            if r.cmd:
                lines.append(f"- **Cmd**: `{r.cmd}`")
            for reason in r.stale_reasons():
                lines.append(f"- {reason}")
            lines.append("")

    transitive_stale = [r for r in rows if r.status == "stale_transitive"]
    if transitive_stale:
        lines.append(f"## Stale (transitive): {len(transitive_stale)}\n")
        lines.append("These caches' own provenance validates clean, but at "
                     "least one upstream cache (reachable via file deps) is "
                     "stale.  Regenerate the upstream first, then this.\n")
        for r in sorted(transitive_stale, key=lambda r: str(r.path)):
            lines.append(f"### `{r.path}`")
            if r.cmd:
                lines.append(f"- **Cmd**: `{r.cmd}`")
            via = ", ".join(f"`{p}`" for p in r.transitive_stale_via)
            lines.append(f"- **Stale via**: {via}")
            lines.append("")

    deferred_rows = [r for r in rows if r.status == "deferred"]
    if deferred_rows:
        lines.append(f"## Deferred: {len(deferred_rows)}\n")
        lines.append("These caches would otherwise be stale "
                     "(``stale_direct`` or ``stale_transitive``) but are "
                     "matched by an entry in ``deferred_rejudges.yaml`` -- "
                     "i.e. the maintainer has explicitly chosen not to "
                     "rerun judging for them right now.  Use "
                     "``tools/defer_rejudge.py --remove`` to lift the "
                     "deferral when ready.  Run "
                     "``tools/audit_deferrals.py`` for category-specific "
                     "invariant checks (e.g. promotion-detection on "
                     "``orphan_no_producer`` entries).\n")

        # Group by deferral category for easier scanning.  Each row
        # may match multiple entries (different categories); we use
        # the first match's category for grouping (callers can read
        # the full list under each row).
        by_category: dict[str, list] = defaultdict(list)
        for r in deferred_rows:
            cat = (r.deferral_matches[0].category.value
                   if r.deferral_matches else "uncategorized")
            by_category[cat].append(r)
        # Roll-up table:
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

    legacy_rows = [r for r in rows if r.status == "legacy"]
    if legacy_rows:
        lines.append(f"## Legacy ({len(legacy_rows)})\n")
        lines.append("JSONs without a ``_provenance`` envelope.  Either "
                     "pre-migration caches, hand-written config (pair lists, "
                     "manifests), or pipeline outputs (judge caches).  "
                     "Audit can't say anything about them; downstream caches "
                     "track their freshness via mtime/size only.\n")
        # Don't emit one row per legacy file -- that would be 1000s.
        # Show only the ones with explicit notes (parse errors, etc.).
        with_notes = [r for r in legacy_rows if r.notes]
        if with_notes:
            lines.append("### Legacy entries with parse problems\n")
            lines.append("| Path | Note |")
            lines.append("|---|---|")
            for r in sorted(with_notes, key=lambda r: str(r.path)):
                lines.append(f"| `{r.path}` | {'; '.join(r.notes)} |")
            lines.append("")

    return "\n".join(lines)


def render_flat(rows: list[CacheAudit]) -> str:
    """One line per JSON: ``status<TAB>path<TAB>note``."""
    out: list[str] = []
    for r in rows:
        if r.status == "stale_direct":
            note = "; ".join(r.stale_reasons())
        elif r.status == "stale_transitive":
            note = "via " + ", ".join(str(p) for p in r.transitive_stale_via)
        elif r.status == "deferred":
            note = (f"was={r.pre_deferred_status}; "
                    + "; ".join(e.reason for e in r.deferral_matches))
        else:
            note = ""
        out.append(f"{r.status}\t{r.path}\t{note}")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", action="append", default=None,
                   help="Directory to walk recursively.  May be passed "
                        "multiple times to audit multiple roots in one "
                        "report (default: roger).")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Write report to this file instead of stdout.")
    p.add_argument("--status", choices=("current", "stale", "stale_direct",
                                        "stale_transitive", "deferred",
                                        "legacy"),
                   default=None,
                   help="Filter to caches with this status.  ``stale`` is a "
                        "convenience alias matching either stale_direct or "
                        "stale_transitive (does NOT include ``deferred`` -- "
                        "those are intentionally held out by the maintainer).")
    p.add_argument("--format", choices=("markdown", "flat"),
                   default="markdown",
                   help="Output format (default: markdown).")
    p.add_argument("--ignore-deferrals", action="store_true",
                   help="Skip applying deferred_rejudges.yaml; show "
                        "underlying ``stale_direct`` / ``stale_transitive`` "
                        "classification regardless of registry entries.")
    args = p.parse_args()

    roots_arg = args.root or ["roger"]
    roots = [Path(r).resolve() for r in roots_arg]
    for r in roots:
        if not r.exists() or not r.is_dir():
            print(f"error: --root {r!r} is not a directory", file=sys.stderr)
            return 2

    rows = collect(roots)
    if not args.ignore_deferrals:
        apply_deferrals(rows)
    propagate_transitive_stale(rows)

    if args.format == "markdown":
        body = render_markdown(rows, roots=roots, status_filter=args.status)
    else:
        if args.status == "stale":
            rows = [r for r in rows if r.status in STALE_STATUSES]
        elif args.status is not None:
            rows = [r for r in rows if r.status == args.status]
        body = render_flat(rows)

    if args.output:
        Path(args.output).write_text(body)
        cnts = Counter(r.status for r in rows)
        summary = ", ".join(f"{k}={v}" for k, v in sorted(cnts.items()))
        print(f"Wrote {args.output} ({len(rows)} JSONs: {summary}).",
              file=sys.stderr)
    else:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
