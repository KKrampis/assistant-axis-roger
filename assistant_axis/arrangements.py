"""The ``arrangement`` field: which set of same-type entities an entity belongs
to, and the shape of that set.

Sep 2026 convention (``AGENT_NOTES.md`` § "The ``arrangement`` field").  Each
role or trait instruction JSON may carry::

    "arrangement": {"kind": "pair", "members": ["callous", "compassionate"]}

or a *list* of such objects when the entity belongs to several sets (a trait
can be one pole of a clean pair and one corner of a triangle at the same
time).  A missing field means *not yet classified*; ``singleton`` is written
explicitly.

Kinds, canonical spellings::

    singleton                           no members
    pair                                2   one clean pair
    triangle, tetrahedron, N-simplex    3, 4, N+1 (N >= 4)
    square                              4   two axes (2-cube and 2-orthoplex are
                                            deliberately not distinguished)
    cube, N-cube                        8, 2**N (N >= 4): every combination of
                                            N binary axes
    octahedron, N-orthoplex             6, 2N (N >= 4): the poles of N clean pairs
    ring                                >= 3, ordered cyclically (circumplexes)
    tree                                >= 2, with ``parent`` / ``children`` links
    map                                 >= 2, unordered, expected to have a
                                            low-dimensional metric structure
    sequence                            >= 2, ordered along a rough axis
    set                                 >= 2, unstructured

Numeric aliases are accepted and canonicalised: ``2-simplex`` -> ``triangle``,
``1-orthoplex`` / ``1-cube`` / ``1-simplex`` -> ``pair``, ``2-cube`` /
``2-orthoplex`` -> ``square``, ``3-cube`` -> ``cube``, ``3-orthoplex`` ->
``octahedron``, ``0-simplex`` -> ``singleton``.

Members are file stems (never display labels), include the entity itself,
and are sorted, except for ``sequence`` and ``ring`` whose order is the
content.  Optional keys: ``axes`` (list of two-element lists naming the
clean pairs that form the axes of a square / cube / orthoplex), ``parent``
and ``children`` (tree only), ``source`` (provenance of an imported
structure), ``note`` (free text).  Unknown keys are preserved.

The pair convention is unchanged: a clean pair of traits is still two files
whose ``negative_label`` fields point at each other's ``positive_label``.
The arrangement field is authoritative for *shape*; the labels stay the
prompt-facing antonyms, and :func:`validate_corpus` refuses to let the two
disagree.  Roles have no ``negative_label``, so their pairs exist only here.

Entry points: :func:`canonical_kind`, :func:`expected_size`,
:func:`parse_arrangement`, :func:`load_corpus_arrangements`,
:func:`validate_corpus`, :func:`summarize_corpus`,
:func:`arrangement_to_json`.  The CLI wrappers are
``data_analysis/check_arrangements.py`` and
``data_analysis/backfill_arrangements.py``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .entity_id import normalize_to_file_name

FIELD = "arrangement"

_FIXED_KINDS = (
    "singleton", "pair", "triangle", "tetrahedron", "square", "cube",
    "octahedron", "ring", "tree", "map", "sequence", "set",
)
_NUMERIC = re.compile(r"^(\d+)-(simplex|cube|orthoplex)$")
_SMALL_NUMERIC: Dict[Tuple[str, int], str] = {
    ("simplex", 0): "singleton",
    ("simplex", 1): "pair",
    ("orthoplex", 1): "pair",
    ("cube", 1): "pair",
    ("simplex", 2): "triangle",
    ("simplex", 3): "tetrahedron",
    ("cube", 2): "square",
    ("orthoplex", 2): "square",
    ("cube", 3): "cube",
    ("orthoplex", 3): "octahedron",
}
ORDERED_KINDS = frozenset({"sequence", "ring"})
#: kinds whose members must partition into clean pairs (traits only)
PAIRWISE_KINDS_PREFIX = ("octahedron", "-orthoplex")
_JSON_KEY_ORDER = ("kind", "members", "axes", "parent", "children", "source", "note")


class ArrangementError(ValueError):
    """A malformed ``arrangement`` value."""


def canonical_kind(kind: str) -> str:
    """Return the canonical spelling of ``kind`` or raise :class:`ArrangementError`."""
    if not isinstance(kind, str):
        raise ArrangementError(f"kind must be a string, got {type(kind).__name__}")
    k = kind.strip().lower()
    if k in _FIXED_KINDS:
        return k
    m = _NUMERIC.match(k)
    if not m:
        raise ArrangementError(f"unknown arrangement kind {kind!r}")
    n, family = int(m.group(1)), m.group(2)
    small = _SMALL_NUMERIC.get((family, n))
    if small is not None:
        return small
    if n < 4:
        raise ArrangementError(f"unknown arrangement kind {kind!r}")
    return f"{n}-{family}"


def expected_size(kind: str) -> Tuple[int, Optional[int]]:
    """``(minimum, maximum)`` member count for a canonical kind; ``None`` = unbounded."""
    k = canonical_kind(kind)
    fixed = {
        "singleton": (0, 0), "pair": (2, 2), "triangle": (3, 3),
        "tetrahedron": (4, 4), "square": (4, 4), "cube": (8, 8),
        "octahedron": (6, 6), "ring": (3, None), "tree": (2, None),
        "map": (2, None), "sequence": (2, None), "set": (2, None),
    }
    if k in fixed:
        return fixed[k]
    n, family = k.split("-")
    n = int(n)
    size = {"simplex": n + 1, "cube": 2 ** n, "orthoplex": 2 * n}[family]
    return (size, size)


def is_ordered(kind: str) -> bool:
    return canonical_kind(kind) in ORDERED_KINDS


def is_pairwise(kind: str) -> bool:
    """True for octahedron and N-orthoplex: members are the poles of clean pairs."""
    k = canonical_kind(kind)
    return k == "octahedron" or k.endswith("-orthoplex")


@dataclass(frozen=True)
class Arrangement:
    kind: str
    members: Tuple[str, ...] = ()
    axes: Optional[Tuple[Tuple[str, str], ...]] = None
    parent: Optional[str] = None
    children: Tuple[str, ...] = ()
    source: Optional[str] = None
    note: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def member_key(self):
        """Identity of the set: ordered tuple for ordered kinds, frozenset otherwise."""
        return self.members if is_ordered(self.kind) else frozenset(self.members)


def _stems(values: Iterable[Any], what: str) -> Tuple[str, ...]:
    out: List[str] = []
    for v in values:
        if not isinstance(v, str) or not v:
            raise ArrangementError(f"{what} must be a list of non-empty strings")
        out.append(v)
    return tuple(out)


def parse_arrangement(obj: Any) -> Arrangement:
    """Validate one arrangement object and return an :class:`Arrangement`.

    Only the *shape of the object* is checked here (types, kind spelling,
    member count, key presence); cross-file consistency is
    :func:`validate_corpus`'s job.
    """
    if not isinstance(obj, Mapping):
        raise ArrangementError(f"arrangement must be an object, got {type(obj).__name__}")
    if "kind" not in obj:
        raise ArrangementError("arrangement has no 'kind'")
    kind = canonical_kind(obj["kind"])
    raw_members = obj.get("members", [])
    if not isinstance(raw_members, list):
        raise ArrangementError("'members' must be a list")
    members = _stems(raw_members, "'members'")
    for m in members:
        if normalize_to_file_name(m) != m:
            raise ArrangementError(f"member {m!r} is not in file-name form (use stems, not labels)")
    if len(set(members)) != len(members):
        raise ArrangementError("duplicate members")
    lo, hi = expected_size(kind)
    if len(members) < lo or (hi is not None and len(members) > hi):
        want = f"{lo}" if hi == lo else (f"at least {lo}" if hi is None else f"{lo}-{hi}")
        raise ArrangementError(f"kind {kind!r} expects {want} members, got {len(members)}")
    axes = None
    if obj.get("axes") is not None:
        if not isinstance(obj["axes"], list):
            raise ArrangementError("'axes' must be a list of two-element lists")
        parsed_axes = []
        for ax in obj["axes"]:
            if not isinstance(ax, list) or len(ax) != 2:
                raise ArrangementError("'axes' must be a list of two-element lists")
            a, b = _stems(ax, "'axes' entries")
            if a not in members or b not in members:
                raise ArrangementError(f"axis ({a}, {b}) names a non-member")
            parsed_axes.append((a, b))
        axes = tuple(parsed_axes)
    parent = obj.get("parent")
    if parent is not None and (not isinstance(parent, str) or not parent):
        raise ArrangementError("'parent' must be a non-empty string or null")
    children = _stems(obj.get("children", []) or [], "'children'")
    if kind != "tree" and (parent is not None or children):
        raise ArrangementError("'parent' / 'children' are only valid for kind 'tree'")
    for key in ("source", "note"):
        if obj.get(key) is not None and not isinstance(obj[key], str):
            raise ArrangementError(f"'{key}' must be a string")
    extra = {k: v for k, v in obj.items() if k not in _JSON_KEY_ORDER}
    return Arrangement(
        kind=kind, members=members, axes=axes, parent=parent,
        children=children, source=obj.get("source"), note=obj.get("note"),
        extra=extra,
    )


def parse_field(value: Any) -> List[Arrangement]:
    """Parse the raw field value: one object, or a list of objects."""
    if value is None:
        return []
    if isinstance(value, list):
        if not value:
            raise ArrangementError("empty arrangement list; omit the field instead")
        return [parse_arrangement(v) for v in value]
    return [parse_arrangement(value)]


def arrangement_to_json(arr: Arrangement) -> Dict[str, Any]:
    """Serialise in canonical key order (kind, members, axes, parent, children, source, note, extras)."""
    out: Dict[str, Any] = {"kind": arr.kind}
    if arr.kind != "singleton":
        out["members"] = list(arr.members)
    if arr.axes is not None:
        out["axes"] = [list(a) for a in arr.axes]
    if arr.kind == "tree":
        out["parent"] = arr.parent
        out["children"] = list(arr.children)
    if arr.source is not None:
        out["source"] = arr.source
    if arr.note is not None:
        out["note"] = arr.note
    for k, v in arr.extra.items():
        out[k] = v
    return out


def field_to_json(arrs: Sequence[Arrangement]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """One object for a single arrangement, a list for several."""
    if len(arrs) == 1:
        return arrangement_to_json(arrs[0])
    return [arrangement_to_json(a) for a in arrs]


# ---------------------------------------------------------------------------
# Corpus-level loading and validation
# ---------------------------------------------------------------------------

@dataclass
class EntityRecord:
    stem: str
    path: Path
    negative_label: Optional[str]          # traits only
    arrangements: List[Arrangement]
    parse_error: Optional[str] = None


@dataclass(frozen=True)
class Problem:
    entity_type: str
    stem: str
    message: str

    def __str__(self) -> str:
        return f"{self.entity_type}/{self.stem}: {self.message}"


def instructions_dir(data_dir: Union[str, Path], entity_type: str) -> Path:
    return Path(data_dir) / entity_type / "instructions"


def load_corpus_arrangements(data_dir: Union[str, Path], entity_type: str) -> Dict[str, EntityRecord]:
    """Read every ``<data_dir>/<entity_type>/instructions/*.json``.

    Entities without the field get an empty ``arrangements`` list (that is
    the *not yet classified* state).  A malformed field is recorded in
    ``parse_error`` rather than raised, so one bad file does not hide the
    rest of the report.
    """
    src = instructions_dir(data_dir, entity_type)
    if not src.is_dir():
        raise FileNotFoundError(f"{src} is not a directory")
    out: Dict[str, EntityRecord] = {}
    for path in sorted(src.glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        neg = doc.get("negative_label") if entity_type == "traits" else None
        rec = EntityRecord(stem=path.stem, path=path, negative_label=neg, arrangements=[])
        try:
            rec.arrangements = parse_field(doc.get(FIELD))
        except ArrangementError as exc:
            rec.parse_error = str(exc)
        out[path.stem] = rec
    return out


def reciprocal_pairs(records: Mapping[str, EntityRecord]) -> List[Tuple[str, str]]:
    """Clean pairs by the negative-label criterion (traits): sorted stem tuples."""
    pairs = set()
    for stem, rec in records.items():
        if not rec.negative_label:
            continue
        partner = normalize_to_file_name(rec.negative_label)
        other = records.get(partner)
        if other is None or not other.negative_label:
            continue
        if normalize_to_file_name(other.negative_label) == stem:
            pairs.add(tuple(sorted((stem, partner))))
    return sorted(pairs)


def _same_arrangement(a: Arrangement, b: Arrangement) -> bool:
    return a.kind == b.kind and a.member_key == b.member_key


def validate_corpus(data_dir: Union[str, Path], entity_type: str) -> List[Problem]:
    """Every consistency rule from AGENT_NOTES, as a list of problems (empty = clean).

    Unclassified entities are not problems; :func:`summarize_corpus` counts them.
    """
    records = load_corpus_arrangements(data_dir, entity_type)
    problems: List[Problem] = []
    add = lambda stem, msg: problems.append(Problem(entity_type, stem, msg))  # noqa: E731

    label_pairs = set(reciprocal_pairs(records)) if entity_type == "traits" else set()
    neg_of = {s: normalize_to_file_name(r.negative_label) for s, r in records.items() if r.negative_label}

    for stem, rec in records.items():
        if rec.parse_error:
            add(stem, f"malformed arrangement: {rec.parse_error}")
            continue
        seen_keys = set()
        for arr in rec.arrangements:
            key = (arr.kind, arr.member_key)
            if key in seen_keys:
                add(stem, f"duplicate {arr.kind} arrangement")
            seen_keys.add(key)
            if arr.kind == "singleton":
                if len(rec.arrangements) > 1:
                    add(stem, "singleton listed alongside other arrangements")
                if entity_type == "traits" and rec.negative_label and not rec.negative_label.lower().startswith("non-"):
                    # singleton is for non-X placeholders only; a real-word label
                    # with no reciprocal file is a pairing still to be decided
                    # and must stay unclassified (AGENT_NOTES § arrangement rule 2)
                    add(stem, f"singleton with a real-word negative_label {rec.negative_label!r}: "
                              "remove the arrangement (undecided) or record the pair")
                continue
            if stem not in arr.members:
                add(stem, f"{arr.kind}: members do not include the entity itself")
            if not is_ordered(arr.kind) and list(arr.members) != sorted(arr.members):
                add(stem, f"{arr.kind}: members are not sorted")
            missing = [m for m in arr.members if m not in records]
            if missing:
                add(stem, f"{arr.kind}: members not in corpus: {', '.join(missing)}")
            for m in arr.members:
                if m == stem or m not in records:
                    continue
                other = records[m]
                if other.parse_error:
                    continue
                if not any(_same_arrangement(arr, o) for o in other.arrangements):
                    add(stem, f"{arr.kind}: member {m} does not record the same arrangement")
            if entity_type == "traits":
                if arr.kind == "pair":
                    a, b = arr.members
                    if tuple(sorted((a, b))) not in label_pairs:
                        add(stem, f"pair {a} / {b} is not reciprocal by negative_label")
                elif is_pairwise(arr.kind):
                    unpaired = [m for m in arr.members
                                if neg_of.get(m) not in arr.members or neg_of.get(neg_of.get(m, ""), None) != m]
                    if unpaired:
                        add(stem, f"{arr.kind}: members not in reciprocal pairs: {', '.join(unpaired)}")
            if arr.axes is not None:
                for a, b in arr.axes:
                    if entity_type == "traits" and tuple(sorted((a, b))) not in label_pairs:
                        add(stem, f"{arr.kind}: axis {a} / {b} is not a clean pair")
            if arr.kind == "tree":
                if arr.parent is not None and arr.parent not in arr.members:
                    add(stem, f"tree: parent {arr.parent} is not a member")
                for c in arr.children:
                    if c not in arr.members:
                        add(stem, f"tree: child {c} is not a member")
                    elif c in records and not records[c].parse_error:
                        child_arrs = [o for o in records[c].arrangements if _same_arrangement(arr, o)]
                        if child_arrs and child_arrs[0].parent != stem:
                            add(stem, f"tree: child {c} does not name {stem} as parent")
                if stem in arr.members and arr.parent is None:
                    roots = [m for m in arr.members if m in records and any(
                        _same_arrangement(arr, o) and o.parent is None
                        for o in records[m].arrangements)]
                    if len(roots) != 1:
                        add(stem, f"tree: expected exactly one root, found {len(roots)}")

    if entity_type == "traits":
        for a, b in sorted(label_pairs):
            for x, y in ((a, b), (b, a)):
                rec = records[x]
                if rec.parse_error or not rec.arrangements:
                    continue  # unclassified is allowed; malformed reported above
                if not any(y in arr.members for arr in rec.arrangements):
                    add(x, f"clean pair with {y} by negative_label is not recorded in any arrangement")
    return problems


def summarize_corpus(data_dir: Union[str, Path], entity_type: str) -> Dict[str, Any]:
    """Counts by kind plus the unclassified list."""
    records = load_corpus_arrangements(data_dir, entity_type)
    by_kind: Dict[str, int] = {}
    unclassified: List[str] = []
    malformed: List[str] = []
    for stem, rec in sorted(records.items()):
        if rec.parse_error:
            malformed.append(stem)
        elif not rec.arrangements:
            unclassified.append(stem)
        else:
            for arr in rec.arrangements:
                by_kind[arr.kind] = by_kind.get(arr.kind, 0) + 1
    return {
        "entity_type": entity_type,
        "total": len(records),
        "by_kind": dict(sorted(by_kind.items())),
        "unclassified": unclassified,
        "malformed": malformed,
    }


__all__ = [
    "FIELD", "Arrangement", "ArrangementError", "EntityRecord", "Problem",
    "ORDERED_KINDS", "arrangement_to_json", "canonical_kind", "expected_size",
    "field_to_json", "instructions_dir", "is_ordered", "is_pairwise",
    "load_corpus_arrangements", "parse_arrangement", "parse_field",
    "reciprocal_pairs", "summarize_corpus", "validate_corpus",
]
