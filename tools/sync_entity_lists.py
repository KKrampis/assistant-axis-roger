#!/usr/bin/env python3
"""Regenerate ``data/roles/role_list.json`` and ``data/traits/trait_list.json``
from the per-entity instruction files.

Both lists are ``{stem: description}`` indexes of the corpus.  Since
2026-09-07 they are **derived**, not hand-maintained: every
``data/<kind>/instructions/<stem>.json`` with a non-empty ``description``
contributes one entry, keyed by its file stem (the file-name form; see
``assistant_axis.entity_id``) and sorted by stem.  Hand edits are
overwritten on the next run, so never edit the lists directly -- edit the
instruction file and rerun this script.

Why derived: no code reads either list; they exist as a one-file,
human- and LLM-readable index of the corpus.  Hand maintenance let them
lag the instruction directories (6 roles and 57 traits missing by Sep
2026), and none of the 275 hand-written role one-liners matched the
instruction-file description that every prompt actually uses.  The
instruction files are the single source of truth; the lists are a view.

Files without a ``description`` (currently only the bare corpus-default
role ``default.json``) are skipped and reported.

Exit code:
* 0 = lists are up to date (``--check``) or were just regenerated
* 1 = ``--check`` found a stale or missing list
* 2 = an instruction directory is missing or a file is not valid JSON

Usage::

    uv run python tools/sync_entity_lists.py            # regenerate both
    uv run python tools/sync_entity_lists.py --check    # CI / pre-commit gate
    uv run python tools/sync_entity_lists.py --kinds traits
    ASSISTANT_AXIS_DATA_DIR=/workspace/data uv run python tools/sync_entity_lists.py

Added a trait or role?  Rerun this script (step 6 of AGENT_NOTES
§ "Adding New Trait Clean Pairs"; step 3 of ``ROLES_TO_ADD.md``).
``tools/tests/test_sync_entity_lists.py::test_repo_lists_are_in_sync``
fails whenever the checked-in lists lag the instruction directories.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis.atomic_io import atomic_write_text  # noqa: E402
from assistant_axis.entity_id import default_data_dir  # noqa: E402

KINDS: Tuple[str, ...] = ("roles", "traits")
LIST_FILE: Dict[str, str] = {"roles": "role_list.json", "traits": "trait_list.json"}
RERUN_HINT = "run `uv run python tools/sync_entity_lists.py`"


class SyncError(RuntimeError):
    """A missing instruction directory or an unreadable instruction file."""


def instructions_dir(data_dir: Union[str, Path], kind: str) -> Path:
    return Path(data_dir) / kind / "instructions"


def list_path(data_dir: Union[str, Path], kind: str) -> Path:
    return Path(data_dir) / kind / LIST_FILE[kind]


def build_list(data_dir: Union[str, Path], kind: str) -> Tuple[Dict[str, str], List[str]]:
    """Return ``({stem: description}, [skipped stems])`` for one kind.

    Entries are sorted by stem.  A file is skipped (and its stem returned
    in the second element) when it has no ``description`` or a blank one.
    Non-JSON files in the directory (the ``*_TO_ADD.md`` notes) are
    ignored.  Descriptions are copied verbatim so the list is an exact
    view of the instruction files.
    """
    src = instructions_dir(data_dir, kind)
    if not src.is_dir():
        raise SyncError(f"{src} is not a directory")
    entries: Dict[str, str] = {}
    skipped: List[str] = []
    for path in sorted(src.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            raise SyncError(f"{path}: {exc}") from exc
        desc = doc.get("description") if isinstance(doc, dict) else None
        if not isinstance(desc, str) or not desc.strip():
            skipped.append(path.stem)
            continue
        entries[path.stem] = desc
    return entries, skipped


def render(entries: Dict[str, str]) -> str:
    """Serialise a list exactly as it is written to disk."""
    return json.dumps(entries, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def describe_drift(old_text: Optional[str], entries: Dict[str, str]) -> str:
    """One-line summary of how the on-disk list differs from ``entries``."""
    if old_text is None:
        return "missing"
    try:
        old = json.loads(old_text)
    except ValueError:
        return "not valid JSON"
    if not isinstance(old, dict):
        return "not a {stem: description} object"
    added = sorted(set(entries) - set(old))
    removed = sorted(set(old) - set(entries))
    changed = sorted(k for k in entries.keys() & old.keys() if entries[k] != old[k])
    parts = [f"{len(added)} added", f"{len(removed)} removed", f"{len(changed)} changed"]
    if not (added or removed or changed):
        parts.append("formatting only")
    return ", ".join(parts)


def sync(
    data_dir: Union[str, Path],
    kinds: Sequence[str] = KINDS,
    *,
    check: bool = False,
    quiet: bool = False,
) -> int:
    """Regenerate (or, with ``check``, just compare) the lists for ``kinds``.

    Returns the process exit code documented in the module docstring.
    Raises :class:`SyncError` for unreadable input.
    """
    stale = 0
    for kind in kinds:
        entries, skipped = build_list(data_dir, kind)
        dest = list_path(data_dir, kind)
        new_text = render(entries)
        old_text = dest.read_text(encoding="utf-8") if dest.exists() else None
        if old_text == new_text:
            if not quiet:
                print(f"{dest}: up to date ({len(entries)} entries)")
                if skipped:
                    print(f"  skipped (no description): {', '.join(skipped)}")
            continue
        stale += 1
        drift = describe_drift(old_text, entries)
        if check:
            print(f"{dest}: STALE ({drift}); {RERUN_HINT}")
        else:
            atomic_write_text(new_text, dest)
            print(f"{dest}: rewrote ({len(entries)} entries; {drift})")
        if skipped:
            print(f"  skipped (no description): {', '.join(skipped)}")
    return 1 if (check and stale) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate role_list.json and trait_list.json from the "
            "instruction files (they are derived views; never hand-edit)."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare only; exit 1 if either list is stale or missing.  Never writes.",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=KINDS,
        default=list(KINDS),
        help="Which lists to sync (default: both).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Corpus data directory (default: $ASSISTANT_AXIS_DATA_DIR, else <repo>/data).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only when a list is stale or was rewritten.",
    )
    args = parser.parse_args(argv)
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()
    try:
        return sync(data_dir, args.kinds, check=args.check, quiet=args.quiet)
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
