#!/usr/bin/env python3
"""Manage entries in deferred_rejudges.yaml.

Judging is expensive and non-deterministic, so we sometimes
intentionally choose to live with a stale judge cache rather than
rerun.  This CLI adds, lists, and removes entries in the
deferred-rejudge registry; ``audit_caches.py`` and ``audit_pngs.py``
consult that registry and reclassify both ``stale_*`` AND ``legacy``
caches to ``deferred`` for matched paths.

The ``legacy`` reclassification covers the May 2026 backfill case:
existing on-disk judge caches predate the modern writer pattern and
read as ``legacy`` (no envelope).  We don't intend to re-judge, so a
deferral entry records "presume current as of mechanism introduction"
rather than letting them accumulate as uncategorised legacy noise.
The entry auto-becomes irrelevant if the producer is later re-run
with the modern writer (the cache then carries an envelope and reads
as ``current`` directly).

Examples
--------

Defer a specific cache file (path-glob match)::

    uv run python tools/defer_rejudge.py \\
        --path 'roger/axis_judge_experiments/q9_*/gpt/scores_descriptions.json' \\
        --reason "Desc-mode rerun deferred until 4-model audit completes."

Defer scoped to a particular dep_key (only that drift source counts)::

    uv run python tools/defer_rejudge.py \\
        --path 'roger/axis_judge_experiments/*/sonnet/scores_descriptions.json' \\
        --dep-key 'judge_*_descriptions_sonnet' \\
        --reason "Sonnet-desc dropped from canonical pipeline."

List all current deferrals::

    uv run python tools/defer_rejudge.py --list

Remove a deferral::

    uv run python tools/defer_rejudge.py --remove \\
        --path 'roger/axis_judge_experiments/q9_*/gpt/scores_descriptions.json'

Semantics
---------
* ``--path`` accepts a Unix shell glob (``fnmatch``) over repo-
  relative cache paths.
* ``--dep-key`` (optional) further scopes the deferral to drifts
  whose offending dep_key matches the given glob.
* Deferrals are ``deferred-until-removed``: there's no expiry; they
  stay active until you delete the entry.
* Audit tools default to honoring deferrals.  Pass
  ``--ignore-deferrals`` to ``audit_caches.py`` /
  ``audit_pngs.py`` to see the underlying ``stale_*`` classification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis import deferral_registry as deferral  # noqa: E402


def _cmd_add(args: argparse.Namespace) -> int:
    if not args.path:
        print("ERROR: --path is required when adding a deferral.", file=sys.stderr)
        return 2
    if not args.reason:
        print("ERROR: --reason is required when adding a deferral.", file=sys.stderr)
        return 2
    try:
        category = deferral.DeferralCategory(args.category)
    except ValueError:
        valid = ", ".join(c.value for c in deferral.DeferralCategory)
        print(f"ERROR: invalid --category {args.category!r}.  Valid: {valid}",
              file=sys.stderr)
        return 2
    try:
        entry = deferral.append_deferral(
            path_glob=args.path,
            dep_key=args.dep_key,
            reason=args.reason,
            category=category,
            producer_script=args.producer_script,
            replaced_by=args.replaced_by,
            compares_to=args.compares_to,
            repo_root=_REPO_ROOT,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("Added deferral:")
    print(f"  path_glob:   {entry.path_glob}")
    print(f"  dep_key:     {entry.dep_key or '(any)'}")
    print(f"  category:    {entry.category.value}")
    if entry.producer_script:
        print(f"  producer_script: {entry.producer_script}")
    if entry.replaced_by:
        print(f"  replaced_by: {entry.replaced_by}")
    if entry.compares_to:
        print(f"  compares_to: {entry.compares_to}")
    print(f"  reason:      {entry.reason}")
    print(f"  deferred_at: {entry.deferred_at}")
    print()
    print(f"Registry: {deferral.DEFERRAL_FILENAME}  (commit it).")
    if entry.category is deferral.DeferralCategory.UNCATEGORIZED:
        print()
        print("WARNING: entry was added with category=uncategorized; "
              "consider passing --category for clarity.", file=sys.stderr)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    reg = deferral.load_registry(repo_root=_REPO_ROOT)
    if not reg:
        print("(no deferrals declared)")
        return 0
    if args.path:
        reg = [e for e in reg if e.path_glob == args.path]
        if not reg:
            print(f"(no deferrals matching path_glob={args.path!r})")
            return 0
    print(f"== {len(reg)} deferred entries ==")
    for e in reg:
        print(f"  [{e.deferred_at}]  ({e.category.value})  path_glob = `{e.path_glob}`")
        if e.dep_key:
            print(f"      dep_key = `{e.dep_key}`")
        if e.producer_script:
            print(f"      producer_script = `{e.producer_script}`")
        if e.replaced_by:
            print(f"      replaced_by = `{e.replaced_by}`")
        if e.compares_to:
            print(f"      compares_to = `{e.compares_to}`")
        print(f"      reason  = {e.reason}")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    if not args.path:
        print("ERROR: --path is required when removing a deferral.", file=sys.stderr)
        return 2
    removed = deferral.remove_deferral(
        path_glob=args.path,
        dep_key=args.dep_key,
        repo_root=_REPO_ROOT,
    )
    if not removed:
        print(
            f"No matching deferral found "
            f"(path_glob={args.path!r}, dep_key={args.dep_key!r}).",
            file=sys.stderr,
        )
        return 1
    print(
        f"Removed deferral path_glob={args.path!r}"
        + (f" dep_key={args.dep_key!r}" if args.dep_key else "")
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--path",
                   help="Unix-shell glob over repo-relative cache paths.")
    p.add_argument("--dep-key", dest="dep_key",
                   help="Optional dep_key glob scoping the deferral.")
    p.add_argument("--reason",
                   help="Free-text justification (required for add).")
    valid_cats = ", ".join(c.value for c in deferral.DeferralCategory)
    p.add_argument(
        "--category", default=deferral.DeferralCategory.UNCATEGORIZED.value,
        help=("Structured intent tag.  One of: " + valid_cats + ".  "
              "See assistant_axis.deferral_registry docstring for "
              "category semantics.  Defaults to 'uncategorized' (a "
              "warning is printed; backfill or pass --category)."),
    )
    p.add_argument("--producer-script", dest="producer_script",
                   help=("(orphan_no_producer) repo-relative path to the "
                         "expected/historical producer script."))
    p.add_argument("--replaced-by", dest="replaced_by",
                   help=("(superseded) repo-relative path-glob of the "
                         "superseding artifact."))
    p.add_argument("--compares-to", dest="compares_to",
                   help=("(frozen_snapshot) repo-relative path-glob of the "
                         "live twin used by paired comparison plots."))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true",
                   help="List declared deferrals (filtered by --path if given).")
    g.add_argument("--remove", action="store_true",
                   help="Remove a deferral by exact (path, dep_key) match.")
    args = p.parse_args(argv)

    if args.list:
        return _cmd_list(args)
    if args.remove:
        return _cmd_remove(args)
    return _cmd_add(args)


if __name__ == "__main__":
    sys.exit(main())
