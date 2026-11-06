#!/usr/bin/env python3
"""Show what changed in a producer script since a cache was written.

When ``audit_caches.py`` reports a cache as ``stale_direct`` because
its ``producer_script`` ``kind="file"`` dependency drifted, the
maintainer often wants to inspect the diff to decide whether the
edit was harmless (declare via ``mark_script_equivalent.py``) or
genuinely changed output (rerun the judge / regenerate the cache).

This tool resolves that diff in two ways and prints whichever
succeeds first:

1. **Git** (preferred): if the cache's envelope recorded a
   ``produced_by.git_sha`` and the working tree is in a clean-enough
   state, ``git diff <sha>:<script_rel> -- <script_rel>`` shows the
   exact content delta.

2. **Cursor Local History** (fallback): when git can't help (no
   recorded sha, or the sha is unreachable in the local clone), this
   tool consults Cursor's per-save snapshot store at
   ``~/Library/Application Support/Cursor/User/History/<hash>/``.
   It walks every ``entries.json`` to find the one whose ``resource``
   field matches the absolute path of the producer script, picks the
   snapshot whose ``timestamp`` is closest to the cache's recorded
   ``last_modified_at``, and diffs its content against the current
   on-disk file.

The tool also prints a ready-to-paste
``mark_script_equivalent.py`` command pre-filled with the recorded
fingerprint, so a maintainer who confirms the diff is harmless can
declare equivalence in one paste.

Usage
-----

::

    uv run python tools/diff_against_recorded.py \\
        roger/axis_judge_experiments/q9_angel_vs_demon/gpt/scores_descriptions.json

    # Force fallback path (skip git entirely):
    uv run python tools/diff_against_recorded.py CACHE.json --no-git

    # Force git path (skip History fallback):
    uv run python tools/diff_against_recorded.py CACHE.json --no-history

    # Override which producer dep is used (defaults to dep_key="producer_script";
    # useful when a cache records multiple kind="file" producers):
    uv run python tools/diff_against_recorded.py CACHE.json --dep-key custom_script
"""

from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis import provenance as prov  # noqa: E402


# Default Cursor Local History root on macOS.  Linux / Windows
# alternatives can be passed via --history-root.
_DEFAULT_HISTORY_ROOTS = (
    "~/Library/Application Support/Cursor/User/History",
    "~/.config/Cursor/User/History",
    "~/AppData/Roaming/Cursor/User/History",
)


# ---------------------------------------------------------------------------
# Envelope inspection
# ---------------------------------------------------------------------------

def _read_envelope(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "_provenance" not in raw:
        raise SystemExit(
            f"{path} has no ``_provenance`` envelope; nothing to diff "
            f"against (this tool is for envelope-bearing caches only)."
        )
    return raw["_provenance"]


def _find_producer_input(
    env: dict, *, dep_key: str = "producer_script",
) -> dict:
    """Locate the kind="file" InputSpec for the producer script.

    Falls back to the first kind="file" entry when the named dep_key
    isn't present (legacy producers that wrote files but not under
    that exact key).
    """
    inputs = env.get("inputs") or []
    for entry in inputs:
        if (isinstance(entry, dict)
                and entry.get("kind") == "file"
                and entry.get("dep_key") == dep_key):
            return entry
    for entry in inputs:
        if isinstance(entry, dict) and entry.get("kind") == "file":
            return entry
    raise SystemExit(
        f"No kind=\"file\" InputSpec found in this cache's envelope "
        f"(dep_key={dep_key!r}).  Pass --dep-key explicitly if the "
        f"producer recorded under a different name."
    )


# ---------------------------------------------------------------------------
# Resolver: git
# ---------------------------------------------------------------------------

def _git_show(git_sha: str, rel_path: str, repo_root: Path) -> Optional[str]:
    """Return ``git show <sha>:<rel_path>`` content, or ``None`` on failure."""
    if not git_sha:
        return None
    sha = git_sha.split("+")[0]  # strip any "+dirty" suffix
    try:
        out = subprocess.check_output(
            ["git", "show", f"{sha}:{rel_path}"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Resolver: Cursor Local History
# ---------------------------------------------------------------------------

def _abs_path_from_uri(resource: str) -> Optional[Path]:
    if not isinstance(resource, str):
        return None
    if resource.startswith("file://"):
        return Path(resource[len("file://"):])
    return None


def _parse_iso_to_epoch_ms(s: str) -> Optional[int]:
    if not s:
        return None
    try:
        return int(_dt.datetime.fromisoformat(s).timestamp() * 1000)
    except ValueError:
        return None


def _candidate_history_roots(extra: Optional[Path] = None) -> list[Path]:
    candidates: list[Path] = []
    if extra is not None:
        candidates.append(extra)
    for h in _DEFAULT_HISTORY_ROOTS:
        candidates.append(Path(os.path.expanduser(h)))
    return [c for c in candidates if c.exists() and c.is_dir()]


def _find_history_dir(
    abs_path: Path, history_root: Path,
) -> Optional[Tuple[Path, list]]:
    """Return ``(dir, entries)`` for the history directory recording
    ``abs_path``, or ``None`` if not found.
    """
    target = str(abs_path.resolve())
    for entry_file in sorted(history_root.glob("*/entries.json")):
        try:
            data = json.loads(entry_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        resource = data.get("resource")
        ap = _abs_path_from_uri(resource)
        if ap is not None and str(ap) == target:
            return entry_file.parent, data.get("entries", [])
    return None


def _pick_closest_snapshot(
    entries: list[dict], target_ms: Optional[int],
) -> Optional[dict]:
    if not entries:
        return None
    if target_ms is None:
        # No target timestamp; pick the most recent snapshot.
        return max(entries, key=lambda e: int(e.get("timestamp") or 0))
    return min(
        entries,
        key=lambda e: abs(int(e.get("timestamp") or 0) - target_ms),
    )


def _read_local_history(
    abs_path: Path,
    *,
    last_modified_at: Optional[str],
    history_root: Optional[Path] = None,
) -> Optional[Tuple[str, dict, Path]]:
    """Return ``(text, snapshot_entry, history_dir)`` from Cursor's
    Local History, or ``None`` when no matching snapshot exists.
    """
    target_ms = _parse_iso_to_epoch_ms(last_modified_at) if last_modified_at else None
    for root in _candidate_history_roots(history_root):
        found = _find_history_dir(abs_path, root)
        if found is None:
            continue
        hist_dir, entries = found
        snap = _pick_closest_snapshot(entries, target_ms)
        if snap is None:
            continue
        snap_id = snap.get("id")
        if not snap_id:
            continue
        snap_path = hist_dir / snap_id
        if not snap_path.exists():
            continue
        try:
            return snap_path.read_text(encoding="utf-8"), snap, hist_dir
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# Diff rendering
# ---------------------------------------------------------------------------

def _unified_diff(
    old_text: str, new_text: str, *, old_label: str, new_label: str,
) -> str:
    return "".join(difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=old_label,
        tofile=new_label,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("cache_path",
                   help="Path to the envelope-bearing cache JSON to inspect.")
    p.add_argument("--script",
                   help="Optional override for the producer script path "
                        "(repo-relative or absolute).  Defaults to the path "
                        "recorded in the envelope's producer_script InputSpec.")
    p.add_argument("--dep-key", dest="dep_key", default="producer_script",
                   help="Which kind=\"file\" InputSpec to treat as the "
                        "producer (default: producer_script).")
    p.add_argument("--no-git", action="store_true",
                   help="Skip the git diff path entirely.")
    p.add_argument("--no-history", action="store_true",
                   help="Skip the Cursor Local History fallback.")
    p.add_argument("--history-root",
                   help="Override the Local History directory (advanced).")
    args = p.parse_args(argv)

    cache_path = Path(args.cache_path).resolve()
    if not cache_path.exists():
        print(f"ERROR: cache not found: {cache_path}", file=sys.stderr)
        return 2

    env = _read_envelope(cache_path)
    producer = _find_producer_input(env, dep_key=args.dep_key)
    rec_script_rel = str(producer.get("path") or "").strip()
    rec_fingerprint = str(producer.get("fingerprint") or "")
    rec_last_modified = str(producer.get("last_modified_at") or "")
    git_sha = (env.get("produced_by") or {}).get("git_sha") or ""

    # Resolve absolute script path.
    if args.script:
        script_abs = (
            Path(args.script).resolve() if Path(args.script).is_absolute()
            else (_REPO_ROOT / args.script).resolve()
        )
        script_rel = (
            str(script_abs.relative_to(_REPO_ROOT))
            if script_abs.is_relative_to(_REPO_ROOT)
            else str(script_abs)
        )
    else:
        script_rel = rec_script_rel
        script_abs = (_REPO_ROOT / script_rel).resolve()

    if not script_abs.exists():
        print(f"ERROR: producer script not found at {script_abs}",
              file=sys.stderr)
        return 2

    print(f"Cache:           {cache_path}")
    print(f"Producer script: {script_rel}")
    print(f"Recorded fp:     {rec_fingerprint}")
    print(f"Recorded mtime:  {rec_last_modified}")
    print(f"Recorded git:    {git_sha or '(none)'}")
    print()

    current_text = script_abs.read_text(encoding="utf-8")
    diff_text: Optional[str] = None
    diff_source: Optional[str] = None

    if not args.no_git and git_sha:
        old = _git_show(git_sha, script_rel, _REPO_ROOT)
        if old is not None:
            diff_text = _unified_diff(
                old, current_text,
                old_label=f"a/{script_rel}@{git_sha}",
                new_label=f"b/{script_rel}",
            )
            diff_source = f"git show {git_sha}:{script_rel}"
        else:
            print(f"  (git show failed for {git_sha}; falling back to History)")

    if (diff_text is None or not diff_text.strip()) and not args.no_history:
        history_root = (
            Path(os.path.expanduser(args.history_root)).resolve()
            if args.history_root else None
        )
        snap = _read_local_history(
            script_abs, last_modified_at=rec_last_modified,
            history_root=history_root,
        )
        if snap is not None:
            old_text, snap_entry, hist_dir = snap
            ts_ms = int(snap_entry.get("timestamp") or 0)
            ts_iso = (
                _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.timezone.utc)
                .replace(microsecond=0).isoformat()
                if ts_ms else "(unknown)"
            )
            diff_text = _unified_diff(
                old_text, current_text,
                old_label=f"a/{script_rel}@{ts_iso}",
                new_label=f"b/{script_rel}",
            )
            diff_source = f"Cursor Local History  ({hist_dir / snap_entry.get('id', '?')})"

    if diff_text is None or not diff_text.strip():
        print("(no diff available -- script may be unchanged, or neither "
              "git nor Cursor Local History could resolve the recorded "
              "version)")
        return 1

    print(f"=== Diff source: {diff_source} ===")
    print(diff_text)

    # Suggested follow-up command (works for any kind="file" producer).
    cur_spec = prov.current_file_input(dep_key="probe", path=script_abs)
    if cur_spec.fingerprint != rec_fingerprint:
        print()
        print("If this diff is HARMLESS (e.g., docstring / comment / type-hint "
              "edits, internal refactor with identical I/O), declare it "
              "equivalent so downstream caches don't get flagged stale:")
        print()
        print(f"  uv run python tools/mark_script_equivalent.py \\")
        print(f"      --script {script_rel} \\")
        print(f"      --from-fp '{rec_fingerprint}' \\")
        print(f"      --to-fp   '{cur_spec.fingerprint}' \\")
        print(f"      --reason  '<short justification>'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
