#!/usr/bin/env python3
"""Check the ``arrangement`` field across ``data/{roles,traits}/instructions/``.

The field records which set of same-type entities each role or trait belongs
to and the shape of that set (pair, triangle, square, N-orthoplex, sequence,
...).  Rules and vocabulary: ``AGENT_NOTES.md`` § "The ``arrangement``
field"; implementation: ``assistant_axis/arrangements.py``.

What is checked:

* the kind is in the vocabulary and the member count matches it;
* members are stems, include the entity itself, and are sorted (except
  ``sequence`` / ``ring``, whose order is the content);
* every member exists and records the same arrangement;
* trait pairs agree with reciprocal ``negative_label`` fields in both
  directions, octahedra / orthoplexes partition into clean pairs, and every
  clean pair by label is recorded in some arrangement of a classified trait;
* tree links are consistent (one root, children name their parent).

A missing field is *not yet classified* and is reported as a count, never
as an error.

Exit code: 0 clean, 1 problems found, 2 cannot read the corpus.

Usage::

    uv run python data_analysis/check_arrangements.py
    uv run python data_analysis/check_arrangements.py --kinds traits --list-unclassified
    uv run python data_analysis/check_arrangements.py --quiet   # print only problems
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis.arrangements import summarize_corpus, validate_corpus  # noqa: E402
from assistant_axis.entity_id import default_data_dir  # noqa: E402

KINDS = ("roles", "traits")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="Corpus data directory (default: $ASSISTANT_AXIS_DATA_DIR or <repo>/data).")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--list-unclassified", action="store_true",
                        help="Print the stems that carry no arrangement field.")
    parser.add_argument("--quiet", action="store_true", help="Print only problems.")
    args = parser.parse_args(argv)
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()

    total_problems = 0
    for etype in args.kinds:
        try:
            problems = validate_corpus(data_dir, etype)
            summary = summarize_corpus(data_dir, etype)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        total_problems += len(problems)
        for p in problems:
            print(f"PROBLEM {p}")
        if args.quiet:
            continue
        kinds = ", ".join(f"{k}: {n}" for k, n in summary["by_kind"].items()) or "none"
        print(f"{etype}: {summary['total']} files; arrangements by kind: {kinds}; "
              f"unclassified: {len(summary['unclassified'])}; malformed: {len(summary['malformed'])}; "
              f"problems: {len(problems)}")
        if args.list_unclassified and summary["unclassified"]:
            print("  unclassified: " + " ".join(summary["unclassified"]))
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(main())
