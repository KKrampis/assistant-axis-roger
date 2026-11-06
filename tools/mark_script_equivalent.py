#!/usr/bin/env python3
"""Mark a code edit as output-preserving in script_equivalences.yaml.

The provenance system records producer scripts (e.g.,
``results_analysis/axis_judge_correlation.py``) as ``kind="file"``
dependencies of every cache they write.  Most edits don't actually
change output (docstrings, type hints, internal refactors); marking
them via this CLI keeps downstream caches from being flagged stale
in audits.

Typical use after a harmless edit:

    uv run python tools/mark_script_equivalent.py \\
        --script results_analysis/axis_judge_correlation.py \\
        --reason "Added type hints; no logic change."

By default ``--from-fp`` is auto-detected from the most recent
provenance-bearing cache that recorded this script (looking under
``roger/`` and ``runpod_workspace/``); ``--to-fp`` defaults to the
current on-disk fingerprint of the script.

To inspect existing edges:

    uv run python tools/mark_script_equivalent.py --list
    uv run python tools/mark_script_equivalent.py --list \\
        --script results_analysis/axis_judge_correlation.py

To check whether a specific edge is recognised (without mutating
the registry):

    uv run python tools/mark_script_equivalent.py --check \\
        --script results_analysis/axis_judge_correlation.py \\
        --from-fp 'v1:2026-04-15T12:34:56+00:00@98765'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

# Repo-root resolution so the script is runnable from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis import provenance as prov  # noqa: E402
from assistant_axis import script_equivalence as eq  # noqa: E402


# Default search roots for auto-detect of ``--from-fp``.  Order is
# significant: caches under ``roger/`` are typically the freshest.
_DEFAULT_SEARCH_ROOTS = ("roger", "runpod_workspace")


def _to_repo_relative(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(_REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _iter_envelope_jsons(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield JSON files under ``roots`` that look envelope-shaped."""
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.json")):
            try:
                # Cheap header-shaped sniff: only open if size is
                # plausible (skip 0-byte / huge mostly-non-JSON files).
                if p.stat().st_size > 200_000_000:
                    continue
                with p.open("rb") as f:
                    head = f.read(2048)
                if b'"_provenance"' not in head:
                    continue
            except OSError:
                continue
            yield p


def _autodetect_from_fp(script_rel: str) -> Optional[str]:
    """Scan envelope-bearing caches for the most recent recorded
    fingerprint of the named script.  Returns the fingerprint string,
    or ``None`` if the script doesn't appear in any envelope under the
    standard search roots.

    "Most recent" is determined by ``produced_at`` in the envelope,
    falling back to the file's mtime when ``produced_at`` is missing.
    """
    roots = [_REPO_ROOT / r for r in _DEFAULT_SEARCH_ROOTS]
    candidates: list[tuple[str, str]] = []  # (produced_at, fingerprint)
    for jpath in _iter_envelope_jsons(roots):
        try:
            data = json.loads(jpath.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        env = data.get("_provenance")
        if not isinstance(env, dict):
            continue
        produced_at = str(env.get("produced_at", ""))
        for entry in env.get("inputs", []) or []:
            if (
                isinstance(entry, dict)
                and entry.get("kind") == "file"
                and entry.get("path") == script_rel
                and entry.get("fingerprint")
            ):
                candidates.append((produced_at, str(entry["fingerprint"])))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _current_fp(script_path: Path) -> str:
    """Compute the *current* file fingerprint of the producer script."""
    spec = prov.current_file_input(dep_key="probe", path=script_path)
    return spec.fingerprint


def _cmd_add(args: argparse.Namespace) -> int:
    script_path = (Path(args.script).resolve()
                   if Path(args.script).is_absolute()
                   else (_REPO_ROOT / args.script).resolve())
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        return 2
    script_rel = _to_repo_relative(script_path)

    to_fp = args.to_fp or _current_fp(script_path)
    from_fp = args.from_fp
    if from_fp is None:
        from_fp = _autodetect_from_fp(script_rel)
        if from_fp is None:
            print(
                f"ERROR: could not auto-detect --from-fp for {script_rel!r}.\n"
                f"  No envelope-bearing cache under "
                f"{', '.join(_DEFAULT_SEARCH_ROOTS)} records this script.\n"
                f"  Pass --from-fp explicitly (recover from git history "
                f"or Cursor Local History; see Phase 6e tool).",
                file=sys.stderr,
            )
            return 2
        print(f"  auto-detected --from-fp: {from_fp}")

    if from_fp == to_fp:
        print(
            f"NOTE: --from-fp == --to-fp ({from_fp!r}); nothing to record. "
            f"The script's current fingerprint already matches the "
            f"recorded one; no equivalence edge needed.",
            file=sys.stderr,
        )
        return 0

    try:
        edge = eq.append_equivalence(
            script=script_rel,
            from_fp=from_fp,
            to_fp=to_fp,
            reason=args.reason,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Added equivalence:")
    print(f"  script:    {edge.script}")
    print(f"  from_fp:   {edge.from_fp}")
    print(f"  to_fp:     {edge.to_fp}")
    print(f"  reason:    {edge.reason}")
    print(f"  marked_at: {edge.marked_at}")
    print()
    print(f"Registry: {(_REPO_ROOT / eq.REGISTRY_FILENAME).relative_to(_REPO_ROOT)}")
    print("(commit this file alongside the code change.)")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    reg = eq.load_registry(repo_root=_REPO_ROOT)
    if not reg:
        print("(registry empty)")
        return 0
    scripts = (
        [args.script] if args.script
        else sorted(reg.keys())
    )
    n_total = 0
    for s in scripts:
        edges = reg.get(s, [])
        if not edges:
            if args.script:
                print(f"(no equivalences for {s})")
            continue
        print(f"== {s} ({len(edges)} edge{'s' if len(edges) != 1 else ''}) ==")
        for e in edges:
            print(f"  [{e.marked_at}]  {e.from_fp}  ->  {e.to_fp}")
            print(f"      reason: {e.reason}")
        print()
        n_total += len(edges)
    if not args.script:
        print(f"Total: {n_total} edges across {len(reg)} scripts.")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    if not args.script or not args.from_fp:
        print(
            "ERROR: --check requires --script and --from-fp; --to-fp "
            "defaults to the current fingerprint.",
            file=sys.stderr,
        )
        return 2
    script_path = (Path(args.script).resolve()
                   if Path(args.script).is_absolute()
                   else (_REPO_ROOT / args.script).resolve())
    script_rel = _to_repo_relative(script_path)
    to_fp = args.to_fp or _current_fp(script_path)
    found, path = eq.is_equivalent(
        script_rel, args.from_fp, to_fp, return_path=True,
    )
    if found:
        print(
            f"EQUIVALENT: {script_rel}\n"
            f"  {args.from_fp}  ->  {to_fp}\n"
            f"  via {len(path)} hop{'s' if len(path) != 1 else ''}:"
        )
        for e in path:
            print(f"    [{e.marked_at}]  {e.reason}")
        return 0
    print(
        f"NOT EQUIVALENT: {script_rel}\n"
        f"  {args.from_fp}  ->  {to_fp}\n"
        f"  No declared chain in {eq.REGISTRY_FILENAME}.\n"
        f"  Use 'mark_script_equivalent.py --script {script_rel} "
        f"--reason \"...\"' to record this edge as harmless."
    )
    return 1


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--script",
                   help="Repo-relative or absolute path of the producer script.")
    p.add_argument("--from-fp", dest="from_fp",
                   help="Older fingerprint (auto-detected from recent envelopes "
                        "if omitted).")
    p.add_argument("--to-fp", dest="to_fp",
                   help="Newer fingerprint (defaults to current on-disk).")
    p.add_argument("--reason",
                   help="Free-text justification (REQUIRED for add).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true",
                   help="List existing equivalences (filtered by --script if given).")
    g.add_argument("--check", action="store_true",
                   help="Test whether a specific edge is recognised; exit 0 yes, 1 no.")
    args = p.parse_args(argv)

    if args.list:
        return _cmd_list(args)
    if args.check:
        return _cmd_check(args)
    # default: add
    if not args.script:
        p.error("--script is required (unless --list).")
    if not args.reason:
        p.error("--reason is required when adding an equivalence edge.")
    return _cmd_add(args)


if __name__ == "__main__":
    sys.exit(main())
