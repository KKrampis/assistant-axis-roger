#!/usr/bin/env python3
"""Walk a tree of judge caches and report rubric_version drift.

Sister tool to :mod:`tools.audit_caches` (which validates provenance
envelope fingerprints) and :mod:`tools.audit_pngs` (which validates
PNGs).  Where those answer "is this cache / plot still computed from
fresh inputs?", *this* tool answers "is this cache's stamped rubric
version consistent with the rubric the code currently runs?".

Two granularities, reported side by side:

* **Cohort-level**: every envelope-wrapped judge cache carries a
  ``rubric_version`` label in its
  ``_provenance.inputs[…producer_script].extras``.  We compare that
  against the current
  :data:`results_analysis.axis_judge_correlation.RUBRIC_VERSION` per
  cache.  Mismatches are reported with the offending file path.

* **Per-entity**: post-May-2026 caches additionally stamp a
  ``per_entity_rubric_versions: {<eid_or_name>: <rubric_version>}``
  map in ``_provenance.notes``.  For each cache we walk the map (or
  fall back to the cohort stamp when missing) and tally how many
  entries are *current*, *equivalent* (per
  :mod:`assistant_axis.rubric_equivalence`), and *drifted*.

Two output formats:

* ``--format markdown`` (default): human-readable report with summary
  tables and a per-cache breakdown of drifted entities.
* ``--format flat``: one line per cache, useful for piping into grep
  / sort / shell-driven follow-up actions.

CLI
---

::

    # Default: walk roger/, write markdown to stdout, compare against
    # current axis_judge_correlation.RUBRIC_VERSION.
    uv run python tools/audit_rubric_versions.py

    # Save report to file:
    uv run python tools/audit_rubric_versions.py -o reports/rubric_audit.md

    # Filter to drifted caches only (skip current/equivalent):
    uv run python tools/audit_rubric_versions.py --status drifted

    # Compare against a specific rubric version (handy for "what would
    # rebumping v3 -> v4 invalidate?" what-if analyses).
    uv run python tools/audit_rubric_versions.py --current v4

    # Add ``runpod_workspace/`` for steering-judge caches too.
    # Currently those don't participate in this scheme (see
    # ``assistant_axis/steering_judges.py`` TODO block), so the
    # report will mostly classify them as legacy.
    uv run python tools/audit_rubric_versions.py --root roger --root runpod_workspace

Performance
-----------

Each JSON is read once; the rubric-equivalence registry is loaded
once and reused across the whole walk.  ``roger/`` audits in <2s on
a warm cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis.judge_loaders import (  # noqa: E402
    peek_per_entity_rubric_versions,
    peek_rubric_version,
)
from assistant_axis.rubric_equivalence import (  # noqa: E402
    RubricEquivalenceEdge,
    is_equivalent,
    load_registry,
)


# A cache is bucketed into exactly one of these statuses at the
# cohort level, plus we report per-entity tallies inside.
STATUSES = ("current", "equivalent", "drifted", "legacy")


@dataclass
class CacheRubricAudit:
    """Per-cache rubric audit result."""
    path: Path
    abs_path: Path
    status: str                              # one of STATUSES
    cohort_rubric: Optional[str] = None      # stamped cohort-level rubric
    current_rubric: str = ""
    axis: Optional[str] = None
    mode: Optional[str] = None
    n_total: int = 0
    n_current: int = 0
    n_equivalent: int = 0
    drifted_by_version: dict = field(default_factory=dict)  # rv -> [eids]

    @property
    def n_drifted(self) -> int:
        return sum(len(v) for v in self.drifted_by_version.values())


# ---------------------------------------------------------------------------
# Per-cache classification
# ---------------------------------------------------------------------------

def _is_envelope(obj) -> bool:
    return (
        isinstance(obj, dict)
        and "result" in obj
        and isinstance(obj.get("_provenance"), dict)
    )


def _read_envelope(path: Path) -> Optional[dict]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _infer_axis_and_mode(path: Path) -> tuple[Optional[str], Optional[str]]:
    """Best-effort recovery of (axis, mode) from a cache path.

    Mirrors :func:`results_analysis.axis_judge_correlation._axis_from_cache_path`
    plus a mode classification based on filename.  Returns ``(None, None)``
    for caches that don't fit the standard layout (those generally get
    classified as ``legacy`` and the missing dimensions don't matter).
    """
    parts = path.parts
    # Standard layout:
    #   <experiments_root>/<axis>/<judge_or_cohort>/scores_<mode>.json
    # axis = grandparent; mode = filename stem minus 'scores_' prefix.
    name = path.name
    mode: Optional[str] = None
    if name.startswith("scores_descriptions"):
        mode = "descriptions"
    elif name.startswith("scores_instructions"):
        mode = "instructions"
    elif name.startswith("scores_responses"):
        mode = "responses"
    axis: Optional[str] = None
    if len(parts) >= 3:
        # The grandparent is the axis dir when the path matches the
        # layout above.  Project axes are conventionally named like
        # ``concise_vs_verbose`` (underscores) so a "_" hint would be
        # paranoid-correct, but we accept any non-empty grandparent
        # name -- the cache filename pattern already filters the
        # walk to the standard layout.
        candidate = path.parent.parent.name
        if candidate:
            axis = candidate
    return axis, mode


def _count_entries(obj: Optional[dict]) -> int:
    """Total entry count in the cache's inner ``result`` payload, with
    a fallback for legacy bare-dict (non-envelope) caches that put
    entries at the top level.  Used for the audit's entry tallies so
    legacy caches show their size for context even though their drift
    can't be classified."""
    if not isinstance(obj, dict):
        return 0
    if _is_envelope(obj):
        result = obj.get("result")
    else:
        result = obj
    if isinstance(result, dict):
        return len(result)
    return 0


def _rel_to_repo(path: Path) -> Path:
    """Best-effort repo-relative path for display.

    Falls back to the absolute path when ``path`` isn't under the
    repo root (e.g. tests running against a ``tmp_path`` outside
    the workspace).  ``Path.is_relative_to`` is Python 3.9+ but the
    try/except form is portable to anything we care about.
    """
    if not path.is_absolute():
        return path
    try:
        return path.relative_to(_REPO_ROOT)
    except ValueError:
        return path


def audit_one(
    path: Path,
    *,
    current_rubric: str,
    registry: Sequence[RubricEquivalenceEdge],
) -> CacheRubricAudit:
    """Classify one cache JSON's rubric drift state."""
    rel = _rel_to_repo(path)
    axis, mode = _infer_axis_and_mode(path)
    obj = _read_envelope(path)
    if obj is None or not _is_envelope(obj):
        # Bare JSON / unparseable -- count entries from the top-level
        # dict if any (helpful for legacy caches), but classify as
        # legacy since we have no rubric stamp to compare against.
        return CacheRubricAudit(
            path=rel, abs_path=path.resolve(),
            status="legacy", current_rubric=current_rubric,
            axis=axis, mode=mode,
            n_total=_count_entries(obj),
        )

    cohort_rubric = peek_rubric_version(path)
    per_entity = peek_per_entity_rubric_versions(path)
    if cohort_rubric is None and not per_entity:
        return CacheRubricAudit(
            path=rel, abs_path=path.resolve(),
            status="legacy", current_rubric=current_rubric,
            axis=axis, mode=mode,
            n_total=_count_entries(obj),
        )

    result = obj.get("result")
    keys: List[str] = []
    if isinstance(result, dict):
        keys = list(result.keys())
    elif isinstance(result, list):
        # Unusual but possible for some aggregate caches; nothing to
        # tally per-entity for those.
        keys = []

    n_current = 0
    n_equivalent = 0
    drifted: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        rv = per_entity.get(key, cohort_rubric)
        if rv is None or rv == current_rubric:
            n_current += 1
            continue
        if is_equivalent(
            rv, current_rubric,
            axis=axis, mode=mode, entity_id=key,
            registry=registry,
        ):
            n_equivalent += 1
            continue
        drifted[rv].append(key)

    # Classify the cache as a whole.  Priority: drifted > equivalent > current.
    if drifted:
        status = "drifted"
    elif n_equivalent > 0:
        status = "equivalent"
    else:
        status = "current"

    return CacheRubricAudit(
        path=rel, abs_path=path.resolve(),
        status=status,
        cohort_rubric=cohort_rubric,
        current_rubric=current_rubric,
        axis=axis, mode=mode,
        n_total=len(keys),
        n_current=n_current,
        n_equivalent=n_equivalent,
        drifted_by_version=dict(drifted),
    )


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------

def iter_judge_caches(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield JSON files under ``roots`` whose names look judge-cache-like.

    We don't crack open every JSON under the roots — that would be
    most of ``roger/``.  Instead we restrict to the standard judge
    cache filenames (``scores_*.json``).  This keeps the walk fast
    AND avoids false ``legacy`` classifications on unrelated JSONs
    (axis aggregates, audit reports, etc.).
    """
    patterns = ("scores_descriptions*.json",
                "scores_instructions*.json",
                "scores_responses*.json")
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            yield from sorted(root.rglob(pat))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_markdown(
    audits: Sequence[CacheRubricAudit],
    *,
    current_rubric: str,
    roots: Sequence[Path],
    status_filter: Optional[str],
) -> str:
    out: list[str] = []
    out.append("# Rubric-version audit\n")
    out.append(f"Roots: {', '.join(f'`{r}`' for r in roots)}\n")
    out.append(f"Active ``RUBRIC_VERSION``: ``{current_rubric}``\n")
    out.append(f"Total judge caches scanned: **{len(audits)}**.\n")

    # Summary by status
    counts = Counter(a.status for a in audits)
    out.append("## Summary by cache status\n")
    out.append("| Status | Count |")
    out.append("|---|---:|")
    for s in STATUSES:
        out.append(f"| {s} | {counts.get(s, 0)} |")
    out.append("")

    # Cohort-stamp distribution (informational)
    cohort_counts = Counter(
        a.cohort_rubric for a in audits if a.cohort_rubric is not None
    )
    legacy_count = sum(1 for a in audits if a.cohort_rubric is None)
    out.append("## Cohort-level ``rubric_version`` distribution\n")
    out.append("| Stamp | Count |")
    out.append("|---|---:|")
    for rv, n in sorted(cohort_counts.items(), key=lambda kv: kv[0]):
        out.append(f"| `{rv}` | {n} |")
    if legacy_count:
        out.append(f"| (unstamped, legacy) | {legacy_count} |")
    out.append("")

    # Per-entity entry tallies
    total_entries = sum(a.n_total for a in audits)
    total_current = sum(a.n_current for a in audits)
    total_equiv = sum(a.n_equivalent for a in audits)
    total_drifted = sum(a.n_drifted for a in audits)
    out.append("## Per-entity entry tallies\n")
    out.append(f"- Total entries:    **{total_entries}**")
    out.append(f"- Current:          **{total_current}**")
    out.append(f"- Equivalent:       **{total_equiv}**")
    out.append(f"- Drifted:          **{total_drifted}**")
    out.append("")

    # Filtered per-cache breakdown
    shown = [
        a for a in audits
        if (status_filter is None or a.status == status_filter)
    ]
    if status_filter is not None:
        out.append(f"## Caches with status = ``{status_filter}``\n")
    else:
        out.append("## Per-cache breakdown\n")
    if not shown:
        out.append("*(none)*\n")
        return "\n".join(out)
    for a in shown:
        out.append(f"### `{a.path}`\n")
        out.append(f"- **Status**: {a.status}")
        out.append(f"- **Cohort rubric_version**: "
                   f"{a.cohort_rubric or '(unstamped)'}")
        out.append(f"- **Axis / mode**: {a.axis or '(unknown)'} / "
                   f"{a.mode or '(unknown)'}")
        out.append(
            f"- **Entries**: total={a.n_total}, "
            f"current={a.n_current}, equivalent={a.n_equivalent}, "
            f"drifted={a.n_drifted}"
        )
        if a.drifted_by_version:
            out.append("- **Drifted entries** (by recorded rubric):")
            for rv, eids in sorted(a.drifted_by_version.items()):
                head = ", ".join(eids[:8])
                tail = "..." if len(eids) > 8 else ""
                out.append(f"  - `{rv}` ({len(eids)}): {head}{tail}")
        out.append("")
    return "\n".join(out)


def render_flat(
    audits: Sequence[CacheRubricAudit],
    *,
    status_filter: Optional[str],
) -> str:
    """One line per cache.  Format:

        <status> <path> cohort=<rv> axis=<a> mode=<m> n_total=<N> drifted=<D>
    """
    rows: list[str] = []
    for a in audits:
        if status_filter is not None and a.status != status_filter:
            continue
        rows.append(
            f"{a.status:11s} {a.path}  "
            f"cohort={a.cohort_rubric!r:6s} "
            f"axis={a.axis or '-'} mode={a.mode or '-'} "
            f"n_total={a.n_total} drifted={a.n_drifted}"
        )
    return "\n".join(rows) + ("\n" if rows else "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_current_rubric() -> str:
    """Read the current ``RUBRIC_VERSION`` from axis_judge_correlation.

    Done by source-parsing rather than importing because importing
    that module pulls in matplotlib + every results_analysis dep,
    and we don't need any of that here.  Falls back to ``"v3"`` if
    the constant can't be parsed (loud-fails if neither found and
    no override was passed on the CLI; see ``main``).
    """
    candidate = _REPO_ROOT / "results_analysis" / "axis_judge_correlation.py"
    if not candidate.exists():
        return ""
    for line in candidate.read_text(encoding="utf-8").splitlines():
        # Match e.g. ``RUBRIC_VERSION = "v3"`` -- no fancy parsing.
        s = line.strip()
        if s.startswith("RUBRIC_VERSION") and "=" in s:
            _, rhs = s.split("=", 1)
            rhs = rhs.strip()
            if rhs.startswith(('"', "'")) and rhs[-1] in ('"', "'"):
                return rhs[1:-1]
    return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", action="append", default=None,
        help="Directory to walk recursively.  Pass multiple times to "
             "audit several roots in one report.  Default: ``roger``.",
    )
    parser.add_argument(
        "--current", default=None,
        help="Override the current rubric version (default: parsed "
             "from results_analysis/axis_judge_correlation.py "
             "RUBRIC_VERSION).  Useful for what-if audits, e.g. "
             "``--current v4`` to ask 'what would a v3 → v4 bump "
             "invalidate?'.",
    )
    parser.add_argument(
        "--status",
        choices=STATUSES,
        default=None,
        help="Filter the per-cache breakdown to one status.  Summary "
             "tables (status counts, cohort distribution, entry "
             "tallies) always show the full data.",
    )
    parser.add_argument(
        "--format", choices=("markdown", "flat"), default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write report to this file instead of stdout.",
    )
    args = parser.parse_args(argv)

    roots = [Path(r) for r in (args.root or ["roger"])]
    roots = [r if r.is_absolute() else _REPO_ROOT / r for r in roots]

    current_rubric = args.current or _default_current_rubric()
    if not current_rubric:
        raise SystemExit(
            "error: couldn't determine current rubric_version "
            "(neither --current passed nor RUBRIC_VERSION parseable "
            "from results_analysis/axis_judge_correlation.py)."
        )

    registry = load_registry()

    audits: list[CacheRubricAudit] = []
    for path in iter_judge_caches(roots):
        audits.append(audit_one(
            path,
            current_rubric=current_rubric,
            registry=registry,
        ))

    if args.format == "flat":
        body = render_flat(audits, status_filter=args.status)
    else:
        body = render_markdown(
            audits,
            current_rubric=current_rubric,
            roots=roots,
            status_filter=args.status,
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")
        # Echo a one-line summary to stderr so CI / shell can see it.
        counts = Counter(a.status for a in audits)
        print(
            f"Wrote {out_path} ({len(audits)} caches: "
            + ", ".join(f"{s}={counts.get(s, 0)}" for s in STATUSES)
            + ").",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(body)
        if not body.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
