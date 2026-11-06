#!/usr/bin/env python3
"""Inspect a plot PNG's embedded metadata and classify its source dataset.

The repo's plot-saving convention (see ``assistant_axis/plot_metadata.py``)
embeds these PNG-text chunks:

  * ``Software``      -- full reproduction CLI, e.g. ``uv run python -m foo --data_dir 'X'``
  * ``Source``        -- ``git <short-sha>[+dirty]``
  * ``Creation Time`` -- ISO-8601 UTC timestamp (pre-2026-05 PNGs may
                         carry the older local-time format)
  * ``Title``, ``Author`` -- bookkeeping

This tool extracts those fields and tries to determine which *dataset*
(e.g. 4-slot Roger, 8slot Roger, Christina headers) the plot was rendered
against.

Determination strategy:

  1. If ``Software`` contains an explicit ``--data_dir VALUE`` argument,
     that's authoritative.
  2. Otherwise, the producing script's compile-time ``DEFAULT_DATA_DIR``
     is what was used.  We look up the script's source at the recorded
     git SHA via ``git show <sha>:<script_path>`` and parse the
     ``DEFAULT_DATA_DIR`` assignment.  This handles two common forms:

       a. ``DEFAULT_DATA_DIR = "runpod_workspace/qwen/qwen-3-32b X"``
       b. ``DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
              "runpod_workspace/qwen/qwen-3-32b X")``

     If the script doesn't define its own ``DEFAULT_DATA_DIR`` (i.e. it
     imports the canonical one from
     ``results_analysis/canonical_angles/data.py``), we look up that
     file at the same SHA.

  3. If we can't determine the dataset (no metadata, malformed
     ``Software``, SHA missing from local git), the result is
     ``indeterminate``.

The classification step maps the resolved data_dir string to a vintage
label: ``8slot`` / ``4-slot Roger`` / ``Christina headers`` / ``Christina``
/ ``other``.

CLI
---

::

    uv run python tools/png_vintage.py [PNG_PATH...]

    # Disable the git lookup for speed (more results will be 'indeterminate'):
    uv run python tools/png_vintage.py --no-sha-lookup [PNG_PATH...]

    # Emit JSON instead of a markdown table:
    uv run python tools/png_vintage.py --json [PNG_PATH...]

Library
-------

::

    from tools.png_vintage import classify_png
    info = classify_png(Path("roger/foo.png"))
    print(info.vintage, info.data_dir, info.script)
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image


# Path of the canonical default constant; consulted when the producing script
# imports DEFAULT_DATA_DIR rather than defining its own.
CANONICAL_DEFAULT_PATH = "results_analysis/canonical_angles/data.py"


# ---------------------------------------------------------------------------
# Vintage classification
# ---------------------------------------------------------------------------

def classify_dataset(data_dir: Optional[str]) -> str:
    """Map a data_dir string to a vintage label.

    Returns one of: ``8slot``, ``4-slot Roger``, ``Christina headers``,
    ``Christina``, ``other (...)``, ``indeterminate``.
    """
    if data_dir is None:
        return "indeterminate"
    dd = data_dir.replace("\\", "/").rstrip("/")
    if dd.endswith("Roger 8slot") or "Roger 8slot/" in dd:
        return "8slot"
    if dd.endswith("Christina headers") or "Christina headers/" in dd:
        return "Christina headers"
    if dd.endswith("Christina") or "Christina/" in dd:
        return "Christina"
    if dd.endswith("Roger") or "Roger/" in dd:
        return "4-slot Roger"
    return f"other ({dd})"


# ---------------------------------------------------------------------------
# Software-field parsing
# ---------------------------------------------------------------------------

def parse_software(software: str) -> dict[str, Optional[str]]:
    """Pull out the producing script, any explicit --data_dir, and any
    --slot value from the embedded Software command.

    Returns a dict with keys ``script`` (e.g. ``-m foo.bar`` or
    ``results_analysis/foo.py``), ``data_dir`` (None if not on the CLI),
    ``slot`` (string form of an explicit ``--slot`` value, None if not
    given), and ``argv`` (the post-script tokens, for diagnostic
    purposes).  Recognises the matplotlib default ``Software`` string
    (``Matplotlib version X.Y, https://matplotlib.org/``) and treats it
    as "no provenance" rather than parsing it as a script name.
    """
    out: dict[str, Optional[str]] = {"script": None, "data_dir": None,
                                      "slot": None, "argv": None}
    if not software:
        return out
    # matplotlib's savefig writes a default Software line when our
    # png_metadata wrapper wasn't used.  Detect and bail out so we don't
    # mis-group these as a "Matplotlib" script.
    if software.startswith("Matplotlib version"):
        return out
    try:
        toks = shlex.split(software)
    except ValueError:
        return out
    # Skip the interpreter prefix (e.g. ['uv', 'run', 'python', ...] or
    # ['python', ...]).  The script token is the first non-flag, non-prefix
    # entry.  Handle ``-m foo.bar`` specially.
    i = 0
    while i < len(toks) and toks[i] in {"uv", "run", "python", "python3"}:
        i += 1
    if i < len(toks) and toks[i] == "-m" and i + 1 < len(toks):
        out["script"] = "-m " + toks[i + 1]
        i += 2
    elif i < len(toks):
        out["script"] = toks[i]
        i += 1
    out["argv"] = " ".join(shlex.quote(t) for t in toks[i:]) or None
    # Pull --data_dir and --slot if present.
    rest = toks[i:]
    for j, tok in enumerate(rest):
        if tok == "--data_dir" and j + 1 < len(rest):
            out["data_dir"] = rest[j + 1]
        elif tok.startswith("--data_dir="):
            out["data_dir"] = tok.split("=", 1)[1]
        elif tok == "--slot" and j + 1 < len(rest):
            out["slot"] = rest[j + 1]
        elif tok.startswith("--slot="):
            out["slot"] = tok.split("=", 1)[1]
    return out


def parse_source_sha(source_field: str) -> tuple[Optional[str], bool]:
    """Extract the bare git SHA from a ``Source`` field like
    ``git abc1234+dirty``.

    Returns ``(sha, dirty)``.  ``dirty`` is True when the recorded
    ``Source`` field had a ``+dirty`` suffix, meaning the working tree
    had uncommitted changes when the artifact was produced.  This is a
    crucial signal: if ``DEFAULT_DATA_DIR`` was changed in the working
    tree but not yet committed, ``git show <sha>:`` returns the
    committed (stale) value, so a default-at-sha lookup will produce the
    wrong vintage.  Callers should treat dirty cases as a downgraded
    confidence level.
    """
    if not source_field:
        return None, False
    m = re.match(r"git\s+([0-9a-f]+)(\+dirty)?", source_field.strip())
    if not m:
        return None, False
    return m.group(1), bool(m.group(2))


# ---------------------------------------------------------------------------
# DEFAULT_DATA_DIR lookup at a given git SHA
# ---------------------------------------------------------------------------

# Match any of:
#   DEFAULT_DATA_DIR = "STRING"
#   DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / ("STRING")
#   DEFAULT_DATA_DIR = Path(...) / "STRING"
#   DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
#       "STRING"
#   )
# We grab the first string literal after the ``=``; everything between
# ``=`` and the opening quote is discarded.  ``[^"']*?`` is lazy so we
# stop at the first quote, and the character class matches newlines, so
# multi-line Path-style assignments work too.  ``re.MULTILINE`` makes
# ``^`` match at line starts (assignments are at column 0 in practice).
_DEFAULT_DATA_DIR_PAT = re.compile(
    r'^DEFAULT_DATA_DIR\s*=\s*[^"\']*?["\']([^"\']+)["\']',
    re.MULTILINE,
)


def script_to_path(script: str) -> Optional[str]:
    """Convert a Software-recorded script reference to a repo-relative
    Python file path.

    ``-m foo.bar`` -> ``foo/bar.py``
    ``results_analysis/foo.py`` -> as-is
    bare filenames -> as-is (caller may need to disambiguate)
    """
    if not script:
        return None
    if script.startswith("-m "):
        mod = script[3:].strip()
        return mod.replace(".", "/") + ".py"
    return script


# Tiny per-process cache so an audit over many PNGs doesn't shell out to
# git hundreds of times for the same (sha, path).  Keyed on (sha, path);
# values are the file's text or None when ``git show`` failed.
_GIT_SHOW_CACHE: dict[tuple[str, str], Optional[str]] = {}


def git_show(sha: str, path: str) -> Optional[str]:
    """Read the contents of ``path`` at ``sha``, or None if not available.

    Cached per process; safe to call repeatedly with the same args.
    """
    key = (sha, path)
    if key in _GIT_SHOW_CACHE:
        return _GIT_SHOW_CACHE[key]
    try:
        res = subprocess.run(
            ["git", "show", f"{sha}:{path}"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        result: Optional[str] = res.stdout if res.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result = None
    _GIT_SHOW_CACHE[key] = result
    return result


def lookup_default_data_dir(sha: str, script_path: str) -> Optional[str]:
    """Find the value of ``DEFAULT_DATA_DIR`` that the script saw at ``sha``.

    Tries the script file first; if it doesn't define ``DEFAULT_DATA_DIR``
    locally, falls back to ``results_analysis/canonical_angles/data.py``
    at the same SHA (the canonical default that most consumers inherit).

    Returns None if nothing matches (e.g. SHA missing from local git, or
    the regex doesn't fire).
    """
    src = git_show(sha, script_path)
    if src is not None:
        m = _DEFAULT_DATA_DIR_PAT.search(src)
        if m:
            return m.group(1)
        # Script didn't define its own; check if it imports DEFAULT_DATA_DIR.
        if "DEFAULT_DATA_DIR" not in src:
            return None
    canon = git_show(sha, CANONICAL_DEFAULT_PATH)
    if canon is None:
        return None
    m = _DEFAULT_DATA_DIR_PAT.search(canon)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# High-level classify(path) entry point
# ---------------------------------------------------------------------------

@dataclass
class PNGInfo:
    path: str
    title: Optional[str] = None
    software: Optional[str] = None
    script: Optional[str] = None
    cli_data_dir: Optional[str] = None     # explicit on the command line
    default_data_dir: Optional[str] = None  # resolved via SHA lookup
    data_dir: Optional[str] = None         # the effective one (cli > default)
    data_dir_source: str = "missing"       # "cli" / "default-at-sha" / "slot-heuristic" / "missing"
    git_sha: Optional[str] = None
    git_dirty: bool = False                # True if the recorded Source had +dirty
    creation_time: Optional[str] = None
    vintage: str = "indeterminate"
    notes: list[str] = field(default_factory=list)


def classify_png(path: Path, *, sha_lookup: bool = True) -> PNGInfo:
    """Read a PNG, parse its embedded metadata, return a classification.

    Parameters
    ----------
    path
        Path to the PNG.
    sha_lookup
        When True (default), if the embedded ``Software`` lacks an explicit
        ``--data_dir``, fall back to looking up ``DEFAULT_DATA_DIR`` in
        the script's source at the recorded git SHA.  Set False to skip
        the git calls (faster, but more results will be ``indeterminate``).
    """
    info = PNGInfo(path=str(path))

    if not path.exists():
        info.notes.append("file does not exist")
        return info

    try:
        meta = Image.open(path).info
    except Exception as e:                                       # noqa: BLE001
        info.notes.append(f"PIL error: {e}")
        return info

    info.title = meta.get("Title")
    info.software = meta.get("Software") or None
    info.creation_time = meta.get("Creation Time") or None
    info.git_sha, info.git_dirty = parse_source_sha(meta.get("Source", ""))

    parsed = parse_software(info.software or "")
    info.script = parsed["script"]
    info.cli_data_dir = parsed["data_dir"]
    cli_slot = parsed["slot"]

    if info.cli_data_dir is not None:
        info.data_dir = info.cli_data_dir
        info.data_dir_source = "cli"
    elif sha_lookup and info.git_sha and info.script:
        script_path = script_to_path(info.script)
        if script_path:
            resolved = lookup_default_data_dir(info.git_sha, script_path)
            if resolved is not None:
                info.default_data_dir = resolved
                info.data_dir = resolved
                info.data_dir_source = "default-at-sha"
            else:
                info.notes.append(
                    f"could not resolve DEFAULT_DATA_DIR at sha={info.git_sha} "
                    f"for script={script_path}")
        else:
            info.notes.append(f"could not coerce script ref {info.script!r} "
                              f"to a file path")

    info.vintage = classify_dataset(info.data_dir)

    # ---- Confidence corrections -----------------------------------------
    # 1. Slot heuristic: if --slot N was on the CLI and N >= 4, the data
    #    must be 8slot (the 4-slot datasets only have slots 0..3, so any
    #    successful run with slot >= 4 had to be reading 8slot data).
    #    This overrides any earlier classification because it's a hard
    #    structural fact, not an inference.
    if cli_slot is not None:
        try:
            slot_int = int(cli_slot)
        except ValueError:
            slot_int = -1
        if slot_int >= 4 and info.vintage != "8slot":
            info.vintage = "8slot"
            info.data_dir_source = "slot-heuristic"
            info.notes.append(
                f"forced 8slot by --slot={slot_int} (slot \u2265 4 cannot exist "
                f"in 4-slot data)")
    # 2. Dirty tree downgrade: when we resolved data_dir via the SHA
    #    lookup but the working tree was dirty at write time, the script
    #    may have used a working-tree-only DEFAULT_DATA_DIR that
    #    git-show can't see.  Downgrade to indeterminate unless the slot
    #    heuristic already nailed it down.
    if info.git_dirty and info.data_dir_source == "default-at-sha":
        info.vintage = "indeterminate (dirty)"
        info.notes.append(
            "working tree was dirty at write time; default DATA_DIR may "
            "have differed from the committed value at sha "
            f"{info.git_sha} (looked up: {info.default_data_dir!r})")

    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_table(rows: list[PNGInfo]) -> str:
    if not rows:
        return ""
    cols = ("path", "vintage", "data_dir", "src", "creation", "sha")
    out_rows: list[tuple[str, ...]] = []
    for r in rows:
        out_rows.append((
            r.path,
            r.vintage,
            r.data_dir or "",
            r.data_dir_source,
            r.creation_time or "",
            r.git_sha or "",
        ))
    widths = [
        max(len(c), max(len(row[i]) for row in out_rows))
        for i, c in enumerate(cols)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out_lines = [fmt.format(*cols), fmt.format(*("-" * w for w in widths))]
    for row in out_rows:
        out_lines.append(fmt.format(*row))
    return "\n".join(out_lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paths", nargs="+", help="PNG files to inspect.")
    p.add_argument("--no-sha-lookup", action="store_true",
                   help="Skip the git lookup of DEFAULT_DATA_DIR at the "
                        "recorded SHA (faster, more 'indeterminate' results).")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="Emit JSON (one record per line) instead of a "
                        "human-readable table.")
    p.add_argument("--show-software", action="store_true",
                   help="Print the full Software field for each PNG (table "
                        "mode only).")
    args = p.parse_args()

    results = [
        classify_png(Path(p), sha_lookup=not args.no_sha_lookup)
        for p in args.paths
    ]

    if args.as_json:
        for r in results:
            print(json.dumps(asdict(r)))
    else:
        print(_format_table(results))
        if args.show_software:
            print()
            for r in results:
                print(f"# {r.path}")
                print(f"  Software: {r.software or '(none)'}")
        # Footer with any issues.
        issues = [(r.path, r.notes) for r in results if r.notes]
        if issues:
            print()
            print("Issues:")
            for path, notes in issues:
                for note in notes:
                    print(f"  {path}: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
