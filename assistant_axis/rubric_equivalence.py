"""Rubric-version equivalence registry.

Sister module to :mod:`assistant_axis.script_equivalence`.  Where
that one declares "this producer-script edit was output-preserving",
*this* one declares "this rubric-version bump was prompt-preserving
for the following (axes, modes) cells".

Motivation
----------

``axis_judge_correlation.RUBRIC_VERSION`` is bumped whenever any of
the rubric template strings (``_SCALE_TABLE``, ``_RUBRIC_HEADER``,
``RUBRIC_STATIC``, ``RUBRIC_RESPONSE_BATCH``) changes in a way that
alters what the judge sees.  A bump invalidates every previously-
written cache *in principle* -- but in practice the change often
only affects a subset of (axis, mode) cells.  Two recent
examples:

* **v1 → v2** (2026-05-09): response-mode prompt anonymises the
  entity (``a specific {entity}`` instead of ``{name}``).  Static-mode
  prompts unchanged.  → equivalent for static modes on every axis;
  not equivalent for response mode anywhere.

* **v2 → v3** (2026-05-10): entity-name field switches from
  file-form (``aligned_artificial_intelligence``) to display-form
  (``aligned artificial intelligence``).  Static-mode prompt
  contains an entity name → differs whenever the entity name has
  underscores.  Response-mode prompt has no entity-name slot, but
  the rubric header renders pole names through the same display
  pipeline → differs only when at least one pole is multi-word.
  Of the 12 v2 axes, only ``systems_thinker_vs_analytical`` has a
  multi-word pole.  → equivalent for response mode on the other 11
  axes; not equivalent for static modes (entity-level, not
  axis-level granularity, so cohort-level claim isn't safe).

Without this registry, ``_check_rubric_version_on_resume`` in
``axis_judge_correlation.py`` would conservatively rejudge every
v1/v2 cache after a bump to v3, spending money to reproduce
byte-identical results.  Declaring a v2↔v3 edge here lets the
producer skip the rejudge for the cells whose prompts didn't
actually change.

Schema
------

Stored at ``<repo_root>/rubric_equivalences.yaml``::

    schema_version: 1
    equivalences:
      - from_rubric: v2
        to_rubric:   v3
        modes:       [responses]
        except_axes: [systems_thinker_vs_analytical]
        reason:      "v2→v3 differs only in display-form vs file-form
                     name rendering. For response mode this only affects
                     prompts on axes with multi-word pole names; the 11
                     single-pole axes render byte-identical prompts."
        marked_at:   "2026-05-11T15:00:00+00:00"

Each edge is per-(``from_rubric``, ``to_rubric``) pair plus optional
scope restrictions:

* ``modes`` -- list of mode strings (``"descriptions"``,
  ``"instructions"``, ``"responses"``).  ``None`` / absent = applies
  to all modes.

* ``axes`` (allow-list) OR ``except_axes`` (deny-list) -- mutually
  exclusive.  ``axes`` restricts the edge to those axes only;
  ``except_axes`` covers all axes except the listed ones; absence of
  both = applies to all axes.

The equivalence relation is reflexive (``from == to`` is always
equivalent), transitive (a→b and b→c implies a→c at the
intersection of their scopes), but NOT symmetric in the data
structure -- maintainers wanting symmetry should declare both
directions (the CLI's ``--symmetric`` flag does this in one shot).

Public API
----------

* :func:`load_registry` -- parse the YAML file.
* :func:`is_equivalent` -- BFS lookup over scoped edges; answers
  "is ``from_rubric`` equivalent to ``to_rubric`` for this (axis,
  mode) cell?".
* :func:`append_equivalence` -- write a new edge (used by
  ``tools/mark_rubric_equivalent.py``).

The registry file is committed to git so equivalence claims travel
with the code.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

__all__ = [
    "REGISTRY_FILENAME",
    "SCHEMA_VERSION",
    "RubricEquivalenceEdge",
    "load_registry",
    "is_equivalent",
    "append_equivalence",
    "edge_covers",
]


REGISTRY_FILENAME = "rubric_equivalences.yaml"
SCHEMA_VERSION = 1

# All known mode strings; an edge with ``modes=None`` covers all of these.
# Kept in sync with axis_judge_correlation.py's mode literals.  Bump if a
# new mode is added (e.g. some future "ranking" mode).
_KNOWN_MODES: Tuple[str, ...] = ("descriptions", "instructions", "responses")


@dataclass(frozen=True)
class RubricEquivalenceEdge:
    """One declared "from_rubric -> to_rubric" rubric-bump claim.

    Scope is restricted by the optional ``modes`` / ``axes`` /
    ``except_axes`` / ``entity_ids`` / ``except_entity_ids`` fields.
    Within each scope dimension, at most one of allow-list or
    deny-list may be set (or neither, meaning "all").  The scope
    dimensions intersect: an edge covers a cell iff *every*
    dimension covers it.

    The entity dimension uses :func:`assistant_axis.entity_id`-format
    strings (``"name|R"`` / ``"name|T"``) rather than bare names,
    because the 9 trait/role name collisions otherwise can't be
    disambiguated.  Bare-name entries in legacy caches resolve
    against the entity scope by promoting them through the kind
    context the cache lives in (static-mode caches are mixed-kind
    but every entry carries the kind in its key; response-mode
    caches are kind-pure and the kind is implicit in the cohort
    directory).
    """

    from_rubric: str
    to_rubric: str
    modes: Optional[Tuple[str, ...]] = None
    axes: Optional[Tuple[str, ...]] = None
    except_axes: Optional[Tuple[str, ...]] = None
    entity_ids: Optional[Tuple[str, ...]] = None
    except_entity_ids: Optional[Tuple[str, ...]] = None
    reason: str = ""
    marked_at: str = ""

    def __post_init__(self) -> None:
        if self.axes is not None and self.except_axes is not None:
            raise ValueError(
                "RubricEquivalenceEdge: pass at most one of "
                "``axes`` (allow-list) or ``except_axes`` (deny-list); "
                f"got axes={self.axes!r}, except_axes={self.except_axes!r}."
            )
        if self.entity_ids is not None and self.except_entity_ids is not None:
            raise ValueError(
                "RubricEquivalenceEdge: pass at most one of "
                "``entity_ids`` (allow-list) or ``except_entity_ids`` "
                "(deny-list); got entity_ids="
                f"{self.entity_ids!r}, except_entity_ids="
                f"{self.except_entity_ids!r}."
            )
        if self.modes is not None:
            for m in self.modes:
                if m not in _KNOWN_MODES:
                    raise ValueError(
                        f"RubricEquivalenceEdge: unknown mode {m!r}; "
                        f"expected one of {_KNOWN_MODES}."
                    )


def edge_covers(
    edge: RubricEquivalenceEdge,
    *,
    axis: Optional[str],
    mode: Optional[str],
    entity_id: Optional[str] = None,
) -> bool:
    """Return True iff ``edge`` applies to the given ``(axis, mode,
    entity_id)`` cell.

    A ``None`` value for any dimension means "the caller doesn't
    care about this dimension"; the edge is treated as covering
    it.  This is the sensible behaviour when the registry is
    consulted with partial information (e.g. an audit pass that
    walks PNG provenance without per-entity granularity).

    When a caller IS being granular (passing both axis and
    entity_id), an edge whose entity scope excludes the entity will
    correctly return False even if every other dimension matches.
    """
    if mode is not None and edge.modes is not None:
        if mode not in edge.modes:
            return False
    if axis is not None:
        if edge.axes is not None and axis not in edge.axes:
            return False
        if edge.except_axes is not None and axis in edge.except_axes:
            return False
    if entity_id is not None:
        if edge.entity_ids is not None and entity_id not in edge.entity_ids:
            return False
        if (edge.except_entity_ids is not None
                and entity_id in edge.except_entity_ids):
            return False
    return True


def _repo_root_default() -> Path:
    """Locate the repository root; mirror provenance._repo_root logic."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def _registry_path(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or _repo_root_default()) / REGISTRY_FILENAME


def load_registry(
    repo_root: Optional[Path] = None,
) -> List[RubricEquivalenceEdge]:
    """Load the registry from disk, returning the full edge list.

    Returns an empty list if the registry file doesn't exist (the
    common case before the first edge is declared).  Unknown
    ``schema_version`` values raise :class:`ValueError`.

    Edges are returned in declaration order (the YAML file order).
    """
    path = _registry_path(repo_root)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    sv = raw.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unknown schema_version {sv!r}; "
            f"this assistant_axis build supports v{SCHEMA_VERSION}."
        )
    out: List[RubricEquivalenceEdge] = []
    for entry in raw.get("equivalences", []) or []:
        modes = entry.get("modes")
        axes = entry.get("axes")
        except_axes = entry.get("except_axes")
        entity_ids = entry.get("entity_ids")
        except_entity_ids = entry.get("except_entity_ids")
        out.append(RubricEquivalenceEdge(
            from_rubric=str(entry["from_rubric"]),
            to_rubric=str(entry["to_rubric"]),
            modes=tuple(modes) if modes else None,
            axes=tuple(axes) if axes else None,
            except_axes=tuple(except_axes) if except_axes else None,
            entity_ids=tuple(entity_ids) if entity_ids else None,
            except_entity_ids=tuple(except_entity_ids) if except_entity_ids else None,
            reason=str(entry.get("reason", "")),
            marked_at=str(entry.get("marked_at", "")),
        ))
    return out


def is_equivalent(
    from_rubric: str,
    to_rubric: str,
    *,
    axis: Optional[str] = None,
    mode: Optional[str] = None,
    entity_id: Optional[str] = None,
    registry: Optional[Sequence[RubricEquivalenceEdge]] = None,
    return_path: bool = False,
):
    """Answer "is ``from_rubric`` equivalent to ``to_rubric`` for this
    ``(axis, mode, entity_id)`` cell?".

    Reflexive (``from_rubric == to_rubric`` is always True), transitive
    (chains through intermediate rubric versions, but each hop must
    cover the same ``(axis, mode, entity_id)`` cell).  Not symmetric
    in the data structure: callers wanting "is X equivalent to Y *or*
    Y equivalent to X?" should call twice (or declare both directions
    in the registry).

    Args:
        from_rubric: Cached rubric-version label.
        to_rubric: Currently-active rubric-version label.
        axis: Axis name (e.g. ``"concise_vs_verbose"``); ``None`` to
            treat the axis dimension as wildcard.
        mode: Mode (``"descriptions"`` / ``"instructions"`` /
            ``"responses"``); ``None`` for wildcard.
        entity_id: Disambiguated entity id (e.g. ``"patient|R"``);
            ``None`` for wildcard.  Crucial for rubric bumps that
            varied per-entity rather than per-axis (e.g. v2→v3,
            where static-mode prompts differed only for entities
            whose names contained underscores).
        registry: Optional pre-loaded edge list; loaded from disk
            when omitted.  Pre-loading helps when calling in a hot
            loop (e.g. an audit pass over hundreds of caches or a
            per-entry filter inside ``score_static_mode``).
        return_path: When True, also return the list of edges
            traversed (useful for human-readable diagnostics).

    Returns:
        ``bool`` if ``return_path`` is False (default).  Otherwise
        ``(found, path)``; ``path`` is empty when ``from_rubric ==
        to_rubric`` and a list of :class:`RubricEquivalenceEdge`
        instances otherwise.
    """
    if from_rubric == to_rubric:
        return (True, []) if return_path else True
    edges = list(registry) if registry is not None else load_registry()
    # Pre-filter to edges that actually cover this (axis, mode,
    # entity_id) cell.
    scoped = [e for e in edges
              if edge_covers(e, axis=axis, mode=mode, entity_id=entity_id)]
    if not scoped:
        return (False, []) if return_path else False

    by_from: dict[str, List[RubricEquivalenceEdge]] = {}
    for e in scoped:
        by_from.setdefault(e.from_rubric, []).append(e)
    visited: set[str] = {from_rubric}
    frontier: list[Tuple[str, List[RubricEquivalenceEdge]]] = [
        (from_rubric, [])
    ]
    while frontier:
        cur, path = frontier.pop(0)
        for edge in by_from.get(cur, []):
            new_path = path + [edge]
            if edge.to_rubric == to_rubric:
                return (True, new_path) if return_path else True
            if edge.to_rubric not in visited:
                visited.add(edge.to_rubric)
                frontier.append((edge.to_rubric, new_path))
    return (False, []) if return_path else False


def append_equivalence(
    *,
    from_rubric: str,
    to_rubric: str,
    modes: Optional[Sequence[str]] = None,
    axes: Optional[Sequence[str]] = None,
    except_axes: Optional[Sequence[str]] = None,
    entity_ids: Optional[Sequence[str]] = None,
    except_entity_ids: Optional[Sequence[str]] = None,
    reason: str,
    marked_at: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> RubricEquivalenceEdge:
    """Append one new edge to the registry, creating the file if needed.

    The registry uses ``yaml.safe_dump`` with ``sort_keys=False`` to
    preserve maintainer-chosen field order (rubric-version pair
    first, scope next, reason and timestamp last) for human review.

    Returns the newly-added :class:`RubricEquivalenceEdge`.

    Raises ``ValueError`` if an identical edge already exists (same
    ``from_rubric``, ``to_rubric``, ``modes``, ``axes``,
    ``except_axes``, ``entity_ids``, ``except_entity_ids``; reason
    and timestamp may differ), or if both allow-list and deny-list
    forms of either the axis or entity dimensions are passed.
    """
    if marked_at is None:
        marked_at = (
            _dt.datetime.now(tz=_dt.timezone.utc)
            .replace(microsecond=0).isoformat()
        )
    edge = RubricEquivalenceEdge(
        from_rubric=from_rubric, to_rubric=to_rubric,
        modes=tuple(modes) if modes else None,
        axes=tuple(axes) if axes else None,
        except_axes=tuple(except_axes) if except_axes else None,
        entity_ids=tuple(entity_ids) if entity_ids else None,
        except_entity_ids=tuple(except_entity_ids) if except_entity_ids else None,
        reason=reason, marked_at=marked_at,
    )
    path = _registry_path(repo_root)
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        sv = raw.get("schema_version")
        if sv != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: refusing to write to unknown schema_version {sv!r}"
            )
    else:
        raw = {"schema_version": SCHEMA_VERSION, "equivalences": []}
    entries = raw.setdefault("equivalences", []) or []
    raw["equivalences"] = entries
    for existing in entries:
        if (
            existing.get("from_rubric") == edge.from_rubric
            and existing.get("to_rubric") == edge.to_rubric
            and tuple(existing.get("modes") or ()) == (edge.modes or ())
            and tuple(existing.get("axes") or ()) == (edge.axes or ())
            and tuple(existing.get("except_axes") or ()) == (edge.except_axes or ())
            and tuple(existing.get("entity_ids") or ()) == (edge.entity_ids or ())
            and tuple(existing.get("except_entity_ids") or ()) == (edge.except_entity_ids or ())
        ):
            raise ValueError(
                f"Duplicate rubric-equivalence edge: "
                f"{edge.from_rubric} -> {edge.to_rubric} "
                f"(modes={edge.modes}, axes={edge.axes}, "
                f"except_axes={edge.except_axes}, "
                f"entity_ids={edge.entity_ids}, "
                f"except_entity_ids={edge.except_entity_ids}) "
                f"already declared."
            )
    new_entry: dict = {
        "from_rubric": edge.from_rubric,
        "to_rubric": edge.to_rubric,
    }
    if edge.modes is not None:
        new_entry["modes"] = list(edge.modes)
    if edge.axes is not None:
        new_entry["axes"] = list(edge.axes)
    if edge.except_axes is not None:
        new_entry["except_axes"] = list(edge.except_axes)
    if edge.entity_ids is not None:
        new_entry["entity_ids"] = list(edge.entity_ids)
    if edge.except_entity_ids is not None:
        new_entry["except_entity_ids"] = list(edge.except_entity_ids)
    new_entry["reason"] = edge.reason
    new_entry["marked_at"] = edge.marked_at
    entries.append(new_entry)
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    return edge
