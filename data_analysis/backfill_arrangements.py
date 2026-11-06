#!/usr/bin/env python3
"""One-off: write the initial ``arrangement`` field into the corpus (Sep 2026).

Dry run by default; ``--apply`` writes.  Rules, from AGENT_NOTES § "The
``arrangement`` field" and Roger's 2026-09-08 decisions:

Traits
  * reciprocal ``negative_label`` pairs -> ``pair``;
  * the two documented triangles (compassionate / malicious / callous,
    conformist / contrarian / nonconformist) -> ``triangle`` (a trait that is
    also in a pair gets a list of two arrangements, e.g. ``malicious``);
  * the moral-circle group -> ``sequence`` in the order below, carrying a
    ``note`` because the order is proposed, not yet confirmed;
  * a ``non-X`` placeholder label with nothing pointing at it and no other
    arrangement -> ``singleton``;
  * everything else (real-word labels with no file, one-way pointers,
    ``un-X`` style negations) -> left blank: *not yet classified*.

Roles
  * pairs listed with ``pair_type == "roles"`` in ``pair_list_clean.json``
    -> ``pair``; every other role -> ``singleton``.

Existing fields are left alone unless ``--overwrite``.  The field is
inserted after ``description`` (or first, for the bare ``default.json``),
keys otherwise untouched, 2-space indent and trailing newline as the
regenerate scripts write.

Usage::

    uv run python data_analysis/backfill_arrangements.py            # dry run: report only
    uv run python data_analysis/backfill_arrangements.py --apply
    uv run python data_analysis/backfill_arrangements.py --kinds traits --apply --overwrite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis.arrangements import (  # noqa: E402
    FIELD, Arrangement, arrangement_to_json, field_to_json,
    load_corpus_arrangements, reciprocal_pairs, validate_corpus,
)
from assistant_axis.atomic_io import atomic_write_text  # noqa: E402
from assistant_axis.entity_id import default_data_dir, normalize_to_file_name  # noqa: E402

KINDS = ("roles", "traits")
DEFAULT_PAIR_LIST = _REPO_ROOT / "roger" / "axis_judge_experiments" / "pair_list_clean.json"

#: The two triangles documented in AGENT_NOTES ("Clean pair validation results").
TRIANGLES: List[List[str]] = [
    ["callous", "compassionate", "malicious"],
    ["conformist", "contrarian", "nonconformist"],
]

#: Moral-circle sequence, ordered by the estimated population of the circle
#: (Roger, 2026-09-08: "guesstimate it by likely population; we then get to
#: find out if various LLMs agree with us").  Rough sizes: self 1; clan ~50;
#: clique ~100; local community ~10^3; parish / locality ~10^4; region
#: ~10^7; religious group anywhere from a sect to 10^9 (overlaps region and
#: nation); nation ~10^8; ethnic group ~10^8; all humans 8x10^9 for the
#: cosmopolitan / philanthropic / humanitarian / universalist group; all
#: sentient animals; all life and ecosystems.
MORAL_CIRCLE_SEQUENCE: List[str] = [
    "selfish", "clannish", "cliqueish", "insular", "parochial", "regionalist",
    "sectarian", "nationalist", "patriotic", "ethnocentric", "cosmopolitan",
    "philanthropic", "humanitarian", "universalist", "kind_to_animals", "ecocentric",
]
MORAL_CIRCLE_NOTE = ("moral-circle size, ordered by estimated population of the circle "
                     "(guesstimate 2026-09-08; sectarian overlaps region and nation)")

#: Sets the corpus already contains that were NOT written (candidates for Roger).
CANDIDATE_SETS_NOT_WRITTEN = {
    "social value orientation triangle": ["competitive", "cooperative", "selfish"],
    "Zimbardo time-perspective five": ["bitter", "fatalistic", "futuristic", "hedonistic", "nostalgic"],
    "Thomas-Kilmann five": ["accommodating", "avoidant", "collaborative", "confrontational", "moderate"],
}


def plan_traits(data_dir: Path) -> Dict[str, List[Arrangement]]:
    records = load_corpus_arrangements(data_dir, "traits")
    stems = set(records)
    plan: Dict[str, List[Arrangement]] = {s: [] for s in stems}
    for a, b in reciprocal_pairs(records):
        pair = Arrangement("pair", (a, b))
        plan[a].append(pair)
        plan[b].append(pair)
    for tri in TRIANGLES:
        members = tuple(sorted(tri))
        if not all(m in stems for m in members):
            continue
        for m in members:
            plan[m].append(Arrangement("triangle", members))
    seq_members = tuple(m for m in MORAL_CIRCLE_SEQUENCE if m in stems)
    if len(seq_members) >= 2:
        for m in seq_members:
            plan[m].append(Arrangement("sequence", seq_members, note=MORAL_CIRCLE_NOTE))
    # singletons: non-X placeholder, nothing points here, no other arrangement
    pointed_at = set()
    for s, rec in records.items():
        if rec.negative_label:
            target = normalize_to_file_name(rec.negative_label)
            if target in stems and target != s:
                pointed_at.add(target)
    for s, rec in records.items():
        if plan[s]:
            continue
        neg = normalize_to_file_name(rec.negative_label or "")
        if neg == f"non_{s}" and s not in pointed_at:
            plan[s].append(Arrangement("singleton"))
    return plan


def plan_roles(data_dir: Path, pair_list: Optional[Path]) -> Dict[str, List[Arrangement]]:
    records = load_corpus_arrangements(data_dir, "roles")
    stems = set(records)
    plan: Dict[str, List[Arrangement]] = {s: [] for s in stems}
    if pair_list is not None and pair_list.exists():
        doc = json.load(open(pair_list, encoding="utf-8"))
        items = doc if isinstance(doc, list) else doc.get("pairs", [])
        for p in items:
            if p.get("pair_type") != "roles":
                continue
            a, b = normalize_to_file_name(p["pos"]), normalize_to_file_name(p["neg"])
            if a in stems and b in stems:
                pair = Arrangement("pair", tuple(sorted((a, b))))
                plan[a].append(pair)
                plan[b].append(pair)
    for s in stems:
        if not plan[s]:
            plan[s].append(Arrangement("singleton"))
    return plan


def _insert_field(doc: Dict[str, Any], value: Any) -> Dict[str, Any]:
    """Return a new dict with ``arrangement`` placed after ``description``."""
    out: Dict[str, Any] = {}
    placed = False
    anchor = "description" if "description" in doc else ("negative_label" if "negative_label" in doc else None)
    if anchor is None:
        out[FIELD] = value
        placed = True
    for k, v in doc.items():
        if k == FIELD:
            continue
        out[k] = v
        if k == anchor and not placed:
            out[FIELD] = value
            placed = True
    if not placed:
        out[FIELD] = value
    return out


def apply_plan(data_dir: Path, etype: str, plan: Dict[str, List[Arrangement]], *, overwrite: bool, write: bool) -> Dict[str, int]:
    counts = {"written": 0, "skipped_existing": 0, "unclassified": 0}
    src = data_dir / etype / "instructions"
    for stem in sorted(plan):
        arrs = plan[stem]
        if not arrs:
            counts["unclassified"] += 1
            continue
        path = src / f"{stem}.json"
        doc = json.load(open(path, encoding="utf-8"))
        if FIELD in doc and not overwrite:
            counts["skipped_existing"] += 1
            continue
        new_doc = _insert_field(doc, field_to_json(arrs))
        if write:
            atomic_write_text(json.dumps(new_doc, indent=2, ensure_ascii=False) + "\n", path)
        counts["written"] += 1
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--pair_list", type=Path, default=DEFAULT_PAIR_LIST,
                        help="pair_list_clean.json supplying the role pairs.")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--apply", action="store_true", help="Write the files (default: dry run).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing arrangement fields.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()

    for etype in args.kinds:
        plan = plan_traits(data_dir) if etype == "traits" else plan_roles(data_dir, args.pair_list)
        by_kind: Dict[str, int] = {}
        for arrs in plan.values():
            for a in arrs:
                by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
        counts = apply_plan(data_dir, etype, plan, overwrite=args.overwrite, write=args.apply)
        if not args.quiet:
            mode = "APPLIED" if args.apply else "DRY RUN"
            kinds = ", ".join(f"{k}: {n}" for k, n in sorted(by_kind.items()))
            print(f"[{mode}] {etype}: {len(plan)} files; planned arrangements by kind: {kinds}; "
                  f"written: {counts['written']}, skipped (field present): {counts['skipped_existing']}, "
                  f"left unclassified: {counts['unclassified']}")
            multi = sorted(s for s, arrs in plan.items() if len(arrs) > 1)
            if multi:
                print(f"  in more than one arrangement: {' '.join(multi)}")
            if etype == "traits":
                print("  candidate sets present in the corpus but NOT written (TRAITS_TO_ADD § \"TODO: arrangement hunting\"): "
                      + "; ".join(f"{k} [{' '.join(v)}]" for k, v in CANDIDATE_SETS_NOT_WRITTEN.items()))
        if args.apply:
            problems = validate_corpus(data_dir, etype)
            if problems:
                print(f"validation after write found {len(problems)} problem(s):")
                for p in problems[:20]:
                    print(f"  {p}")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
