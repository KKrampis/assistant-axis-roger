#!/usr/bin/env python3
"""Build the two canonical clean-trait pair-list files from
``data/goal_roles_and_traits.json`` and the existing clean-pair source.

Outputs (default in ``roger/axis_judge_experiments/``):

* ``pair_list_clean.json`` -- ALL clean pairs (currently 56: 55 trait
  pairs + 1 role pair, defined by the "bidirectional negative_label
  match" cleanliness criterion).  No goal/non-goal filter.  Each entry
  preserves the original ``{pos, neg, pair_type, note}`` schema and
  ALSO carries a derived ``goal_type`` field (``"goal"``,
  ``"non_goal"``, or ``null``) so consumers that *do* want the
  filtered view can apply their own filter without reloading the
  lookup table.

* ``pair_list_goalnongoal.json`` -- clean pairs where at least one
  pole is listed in ``data/goal_roles_and_traits.json``'s
  ``<pair_type>.goal`` or ``<pair_type>.non_goal`` arrays.  Each entry
  has a non-null ``goal_type`` field set to ``"goal"`` or
  ``"non_goal"`` according to which list the matched pole(s) came
  from.  Currently 34 pairs: 1 role pair (``angel/demon``, both poles
  in ``roles.goal``), 18 ``traits.goal``, 15 ``traits.non_goal``.

The 22 trait pairs that survive the "clean" filter but have *neither*
pole listed in the goal/non-goal corpus are the "intermediate" axes
(style / method / affect rather than goal-pursuit) and appear ONLY in
``pair_list_clean.json``.

Pre-2026-05-13 the file currently named ``pair_list_goalnongoal.json``
held all 56 clean pairs (filename was a misnomer; despite the name,
no goal/non-goal filter was applied and ``goal_type`` was unset on
every entry).  This script writes the canonically-named replacement
plus the new properly-filtered file.

Usage::

    uv run python tools/build_pair_lists.py

Idempotent -- safe to rerun whenever ``data/goal_roles_and_traits.json``
or the source clean-pair list changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA = _REPO_ROOT / "data" / "goal_roles_and_traits.json"
_EXPERIMENT_DIR = _REPO_ROOT / "roger" / "axis_judge_experiments"
# Going forward ``pair_list_clean.json`` is the authoritative
# all-clean-pairs source; we fall back to the legacy
# ``pair_list_goalnongoal.json`` location for the first-time migration
# (when ``pair_list_clean.json`` doesn't exist yet).
_AUTH_SOURCE = _EXPERIMENT_DIR / "pair_list_clean.json"
_LEGACY_SOURCE = _EXPERIMENT_DIR / "pair_list_goalnongoal.json"
_OUT_CLEAN = _EXPERIMENT_DIR / "pair_list_clean.json"
_OUT_GOAL = _EXPERIMENT_DIR / "pair_list_goalnongoal.json"


def default_source() -> Path:
    """Pick the source-of-truth pair list to read from.

    Prefer the new ``pair_list_clean.json`` once it exists; fall back
    to the legacy ``pair_list_goalnongoal.json`` (which carried the 56
    pairs before the 2026-05-13 rename).
    """
    if _AUTH_SOURCE.exists():
        return _AUTH_SOURCE
    return _LEGACY_SOURCE


def classify_pole(pole: str, kind: str, lookup: dict) -> str | None:
    """Return ``"goal"`` / ``"non_goal"`` / ``None`` for a single pole.

    ``kind`` is ``"traits"`` or ``"roles"`` (the ``pair_type`` of the
    pair the pole belongs to).  Lookup is the parsed JSON of
    ``data/goal_roles_and_traits.json``.
    """
    if pole in lookup[kind]["goal"]:
        return "goal"
    if pole in lookup[kind]["non_goal"]:
        return "non_goal"
    return None


def classify_pair(pos: str, neg: str, kind: str, lookup: dict) -> str | None:
    """Combine per-pole classifications into a pair-level classification.

    Rule: at least one pole must be listed in either ``goal`` or
    ``non_goal``.  The pair's classification is the (unique)
    sub-list that contains the listed pole(s).  If poles span both
    sub-lists, returns ``"mixed"`` (we'd then have to pick a tiebreak,
    but on the current 56-pair clean set this case doesn't occur).
    If neither pole is listed, returns ``None`` (the pair is excluded
    from the goal/non-goal filter).
    """
    pos_cls = classify_pole(pos, kind, lookup)
    neg_cls = classify_pole(neg, kind, lookup)
    listed = {c for c in (pos_cls, neg_cls) if c is not None}
    if not listed:
        return None
    if len(listed) == 1:
        return next(iter(listed))
    return "mixed"


def build_lists(source_pairs: list[dict], lookup: dict
                ) -> tuple[list[dict], list[dict], list[dict]]:
    """Return ``(clean_pairs, goalnongoal_pairs, excluded_pairs)``.

    Both output lists are derived from ``source_pairs``; ``clean`` is a
    passthrough of every input pair with a derived ``goal_type`` field
    appended (possibly ``None``).  ``goalnongoal`` is the subset where
    ``goal_type`` is non-null and unambiguous (i.e. ``"goal"`` or
    ``"non_goal"``, not ``"mixed"`` or ``None``).  ``excluded`` lists
    the pairs that didn't make the goal/non-goal cut (for reporting).
    """
    clean_pairs: list[dict] = []
    goal_pairs: list[dict] = []
    excluded: list[dict] = []
    for it in source_pairs:
        pos, neg = it["pos"], it["neg"]
        kind = it.get("pair_type", "traits")
        cls = classify_pair(pos, neg, kind, lookup)
        out = dict(it)  # don't mutate input
        out["goal_type"] = cls
        clean_pairs.append(out)
        if cls in ("goal", "non_goal"):
            goal_pairs.append(out)
        else:
            excluded.append(out)
    return clean_pairs, goal_pairs, excluded


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", default=str(_DATA),
                   help=f"Goal/non-goal lookup JSON "
                        f"(default: {_DATA}).")
    p.add_argument("--source", default=None,
                   help="Input pair list of all clean pairs.  Defaults "
                        "to pair_list_clean.json if it exists "
                        "(post-migration), else the legacy "
                        "pair_list_goalnongoal.json (pre-migration "
                        "all-clean source, despite the misleading "
                        "filename).")
    p.add_argument("--out_clean", default=str(_OUT_CLEAN),
                   help=f"Output path for the all-clean pair list "
                        f"(default: {_OUT_CLEAN}).")
    p.add_argument("--out_goalnongoal", default=str(_OUT_GOAL),
                   help=f"Output path for the goal/non-goal-filtered "
                        f"pair list (default: {_OUT_GOAL}).")
    args = p.parse_args()

    lookup = json.loads(Path(args.data).read_text())
    source_path = Path(args.source) if args.source else default_source()
    print(f"Reading source: {source_path}")
    source = json.loads(source_path.read_text())
    if isinstance(source, dict):
        # Some legacy variants wrapped pairs in {"pairs": [...]}.
        source = source.get("pairs", source.get("result", source))

    clean_pairs, goal_pairs, excluded = build_lists(source, lookup)

    Path(args.out_clean).write_text(json.dumps(clean_pairs, indent=2))
    Path(args.out_goalnongoal).write_text(json.dumps(goal_pairs, indent=2))

    # Summary
    by_kind_cls = {}
    for p in clean_pairs:
        key = (p.get("pair_type", "traits"), p["goal_type"])
        by_kind_cls[key] = by_kind_cls.get(key, 0) + 1
    print(f"Wrote {len(clean_pairs)} pairs to {args.out_clean}")
    print(f"Wrote {len(goal_pairs)} pairs to {args.out_goalnongoal}")
    print(f"  ({len(excluded)} intermediate pairs in clean but excluded "
          f"from goalnongoal)")
    print("Breakdown:")
    for (kind, cls), n in sorted(by_kind_cls.items(),
                                  key=lambda x: (x[0][0], str(x[0][1]))):
        print(f"  {kind:8s}  {str(cls):12s}  n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
