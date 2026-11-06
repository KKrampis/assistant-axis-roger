#!/usr/bin/env python3
"""Walk a directory tree of committed PNGs and emit a markdown vintage
report classifying each by source dataset.

Uses :func:`tools.png_vintage.classify_png` per file (see that module for
classification details and limits).

The report groups results by (producing script, vintage) so it's easy to
spot, e.g., "all `whitening_k_sweep.py` outputs are 8slot but two
`batch_size_*.png` are still 4-slot Roger -- rerun those".

CLI
---

::

    # Default: walk roger/ recursively, emit markdown to stdout.
    uv run python tools/audit_committed_pngs.py

    # Custom root + write to a file:
    uv run python tools/audit_committed_pngs.py --root roger --output report.md

    # Skip git-show DEFAULT_DATA_DIR lookup (faster; more 'indeterminate'):
    uv run python tools/audit_committed_pngs.py --no-sha-lookup

    # Filter to a specific vintage (useful for "show me what's still 4-slot"):
    uv run python tools/audit_committed_pngs.py --only 4-slot Roger

Output
------

A markdown document with:

  - A summary table at the top: count per vintage.
  - A "By script" section: each producing script gets a subsection with
    its own count-by-vintage and (collapsed) per-PNG details.
  - A "Notes / issues" section listing any per-PNG resolution problems.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running as ``python tools/audit_committed_pngs.py`` from the repo
# root without needing the ``tools`` dir on PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.png_vintage import PNGInfo, classify_png  # noqa: E402


VINTAGE_PRIORITY = [
    "8slot",
    "4-slot Roger",
    "Christina headers",
    "Christina",
    "indeterminate (dirty)",
    "indeterminate",
]


def _vintage_key(v: str) -> tuple[int, str]:
    """Sort key putting known vintages in the predefined priority order
    and unknown ones (``other (...)``) at the end alphabetically."""
    if v in VINTAGE_PRIORITY:
        return (VINTAGE_PRIORITY.index(v), v)
    return (len(VINTAGE_PRIORITY), v)


def _short_script(info: PNGInfo) -> str:
    """Display name for grouping by producing script.

    Distinguishes:
      * scripts (the usual case),
      * legacy PNGs from before the png_metadata wrapper existed (their
        Software field is matplotlib's default ``Matplotlib version ...``),
      * truly unknown sources (no Software field at all).
    """
    if info.script:
        return info.script
    if info.software and info.software.startswith("Matplotlib version"):
        return "(legacy: pre-provenance, no wrapper)"
    return "(unknown source)"


def collect(root: Path, *, sha_lookup: bool) -> list[PNGInfo]:
    """Find all *.png under ``root`` and classify each."""
    rows: list[PNGInfo] = []
    for path in sorted(root.rglob("*.png")):
        rows.append(classify_png(path, sha_lookup=sha_lookup))
    return rows


def render_markdown(rows: list[PNGInfo], *, root: Path,
                    only: str | None = None) -> str:
    """Build the audit report markdown."""
    if only:
        rows = [r for r in rows if r.vintage == only]

    lines: list[str] = []
    lines.append(f"# PNG vintage audit: `{root}`\n")
    lines.append(f"Total PNGs scanned: **{len(rows)}**\n")

    # ---- Summary by vintage ------------------------------------------------
    by_vintage = Counter(r.vintage for r in rows)
    lines.append("## Summary by vintage\n")
    lines.append("| Vintage | Count |")
    lines.append("|---|---:|")
    for v in sorted(by_vintage, key=_vintage_key):
        lines.append(f"| {v} | {by_vintage[v]} |")
    lines.append("")

    # ---- By producing script ----------------------------------------------
    by_script: dict[str, list[PNGInfo]] = defaultdict(list)
    for r in rows:
        by_script[_short_script(r)].append(r)

    lines.append("## By producing script\n")
    for script in sorted(by_script):
        srows = by_script[script]
        # Per-script vintage counts.
        sc = Counter(r.vintage for r in srows)
        lines.append(f"### `{script}` ({len(srows)})")
        counts = ", ".join(
            f"{v}={sc[v]}" for v in sorted(sc, key=_vintage_key)
        )
        lines.append(f"_{counts}_\n")
        # Sort rows within the script by vintage then path.
        srows_sorted = sorted(
            srows, key=lambda r: (_vintage_key(r.vintage), r.path)
        )
        lines.append("| Path | Vintage | Source | Created | SHA |")
        lines.append("|---|---|---|---|---|")
        for r in srows_sorted:
            lines.append(
                f"| `{r.path}` | {r.vintage} | "
                f"{r.data_dir_source} | {r.creation_time or ''} | "
                f"{r.git_sha or ''} |"
            )
        lines.append("")

    # ---- Notes / issues ----------------------------------------------------
    issues = [(r.path, r.notes) for r in rows if r.notes]
    if issues:
        lines.append("## Notes / issues\n")
        for path, notes in issues:
            for note in notes:
                lines.append(f"- `{path}`: {note}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", type=str, default="roger",
                   help="Directory to walk (default: roger).")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Write the markdown report to this file instead of "
                        "stdout.")
    p.add_argument("--no-sha-lookup", action="store_true",
                   help="Skip the git lookup of DEFAULT_DATA_DIR at the "
                        "recorded SHA (faster, more 'indeterminate' rows).")
    p.add_argument("--only", type=str, default=None,
                   help="Filter the report to only a given vintage label "
                        "(e.g. '4-slot Roger').")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"error: --root {root!r} is not a directory", file=sys.stderr)
        return 2

    rows = collect(root, sha_lookup=not args.no_sha_lookup)
    md = render_markdown(rows, root=root, only=args.only)
    if args.output:
        Path(args.output).write_text(md)
        print(f"Wrote {args.output} ({len(rows)} PNGs scanned).",
              file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
