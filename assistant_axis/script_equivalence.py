"""Script-equivalence registry for the provenance system.

Producer scripts (notably ``results_analysis/axis_judge_correlation.py``)
are recorded as ``kind="file"`` dependencies of every cache they
write, so any edit to the script changes its mtime/size fingerprint
and downstream caches register as ``drift`` in ``audit_caches.py``.

Most edits, however, don't actually affect output: docstring
clean-ups, type hints, log-message tweaks, refactors that preserve
I/O.  Rejudging every cache after a docstring fix would be
prohibitively expensive (judges aren't cheap and aren't
deterministic, so a rerun isn't even a clean re-derivation).

This module implements a small declarative registry that lets a
maintainer mark a specific (from_fp, to_fp) edge as "equivalent" --
i.e., output-preserving for the named script.  Audits and validators
consult the registry and downgrade ``drift -> equivalent`` for
matched script edges.

Schema
------
Stored at ``<repo_root>/script_equivalences.yaml``::

    schema_version: 1
    equivalences:
      - script:    results_analysis/axis_judge_correlation.py
        from_fp:   v1:2026-04-15T12:34:56+00:00@98765
        to_fp:     v1:2026-05-08T04:03:52+00:00@103727
        reason:    "Refactored prompt builder; identical I/O."
        marked_at: "2026-05-08T04:30:00+00:00"
      - script:    results_analysis/axis_judge_correlation.py
        from_fp:   v1:2026-05-08T04:03:52+00:00@103727
        to_fp:     v1:2026-05-15T18:00:00+00:00@103900
        reason:    "Added type hints; no logic change."
        marked_at: "2026-05-15T18:30:00+00:00"

Edges are *per-edit pairs*, not "from a fixed baseline to current".
This keeps the registry append-only, lets us audit each edit
individually, and admits a transitive closure: an old cache pinning
``v1:...@98765`` is equivalent to any of the later fingerprints that
are reachable from it via a chain of declared-harmless edits.

Lookup is a BFS from ``from_fp`` across edges keyed by the same
``script`` path; the answer is True iff ``to_fp`` is reachable.

Equivalence is a directional relation in the registry, but the
practical test is symmetric (an edit that's harmless when applied
is also harmless when reverted).  Rather than auto-add reverse
edges, we accept the asymmetry: callers asking
"is X equivalent to Y?" and "is Y equivalent to X?" get separate
answers; the CLI may add both edges if the maintainer wants both
directions covered.

Public API
----------
* :func:`load_registry` -- parse the YAML file.
* :func:`is_equivalent` -- BFS lookup, optionally returning the
  derivation path.
* :func:`append_equivalence` -- write a new edge (used by
  ``tools/mark_script_equivalent.py``).

The registry file is committed to git so equivalence claims travel
with the code.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import yaml

__all__ = [
    "REGISTRY_FILENAME",
    "EquivalenceEdge",
    "load_registry",
    "is_equivalent",
    "append_equivalence",
]


REGISTRY_FILENAME = "script_equivalences.yaml"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EquivalenceEdge:
    """One declared "from_fp -> to_fp" harmless-edit pair for a script."""

    script: str          # Repo-relative POSIX path of the producer script.
    from_fp: str         # Older fingerprint (e.g. v1:<mtime>@<size>).
    to_fp: str           # Newer fingerprint.
    reason: str          # Free-text justification (commit ref / explanation).
    marked_at: str       # ISO-8601 timestamp of when the edge was declared.


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
) -> dict[str, List[EquivalenceEdge]]:
    """Load the registry from disk, returning a script-keyed edge map.

    Returns an empty dict if the registry file doesn't exist (the
    common case before the first edge is declared).  Unknown
    ``schema_version`` values raise ``ValueError``.
    """
    path = _registry_path(repo_root)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    sv = raw.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unknown schema_version {sv!r}; "
            f"this assistant_axis build supports v{SCHEMA_VERSION}."
        )
    out: dict[str, List[EquivalenceEdge]] = {}
    for entry in raw.get("equivalences", []) or []:
        edge = EquivalenceEdge(
            script=str(entry["script"]),
            from_fp=str(entry["from_fp"]),
            to_fp=str(entry["to_fp"]),
            reason=str(entry.get("reason", "")),
            marked_at=str(entry.get("marked_at", "")),
        )
        out.setdefault(edge.script, []).append(edge)
    return out


def is_equivalent(
    script_path: str,
    from_fp: str,
    to_fp: str,
    *,
    registry: Optional[dict[str, List[EquivalenceEdge]]] = None,
    return_path: bool = False,
):
    """Return ``True`` if ``from_fp`` is declared equivalent to ``to_fp``
    for the given ``script_path`` via a chain of registry edges.

    The relation is reflexive (``from_fp == to_fp`` is always equivalent)
    and transitive (A->B and B->C implies A->C).  It is not symmetric
    in the data structure, but callers wanting symmetry can either add
    the reverse edge or check both directions.

    Args:
        script_path: Repo-relative POSIX path of the producer script,
            matching the ``script`` field of the YAML edges.
        from_fp: Recorded fingerprint (older).
        to_fp: Current fingerprint (newer).
        registry: Optional pre-loaded registry; loaded from disk if
            omitted.  Pre-loading is useful when calling this in a hot
            loop (e.g., audit tools).
        return_path: When True, also return the list of edges traversed
            (useful for auditors / mark_script_equivalent.py inspection).

    Returns:
        ``bool`` if ``return_path`` is False (default).  Otherwise a
        tuple ``(found, path)`` where ``path`` is the list of
        :class:`EquivalenceEdge` instances traversed (empty when
        ``from_fp == to_fp`` and a single-step path otherwise).
    """
    if from_fp == to_fp:
        return (True, []) if return_path else True
    reg = registry if registry is not None else load_registry()
    edges = reg.get(script_path, [])
    if not edges:
        return (False, []) if return_path else False

    # BFS: from_fp -> ... -> to_fp.
    by_from: dict[str, list[EquivalenceEdge]] = {}
    for e in edges:
        by_from.setdefault(e.from_fp, []).append(e)
    visited: set[str] = {from_fp}
    # frontier elements: (current_fp, path_so_far)
    frontier: list[Tuple[str, List[EquivalenceEdge]]] = [(from_fp, [])]
    while frontier:
        cur, path = frontier.pop(0)
        for edge in by_from.get(cur, []):
            new_path = path + [edge]
            if edge.to_fp == to_fp:
                return (True, new_path) if return_path else True
            if edge.to_fp not in visited:
                visited.add(edge.to_fp)
                frontier.append((edge.to_fp, new_path))
    return (False, []) if return_path else False


def append_equivalence(
    *,
    script: str,
    from_fp: str,
    to_fp: str,
    reason: str,
    marked_at: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> EquivalenceEdge:
    """Append one new edge to the registry, creating the file if needed.

    The registry file uses YAML's ``safe_dump`` with ``sort_keys=False``
    to preserve the maintainer-chosen field order (script first,
    fingerprints next, reason and timestamp last) for human review.

    Returns the newly-added :class:`EquivalenceEdge`.

    Raises ``ValueError`` if an identical edge already exists (same
    script, from_fp, to_fp; reason and timestamp may differ).
    """
    if marked_at is None:
        marked_at = (_dt.datetime.now(tz=_dt.timezone.utc)
                     .replace(microsecond=0).isoformat())
    edge = EquivalenceEdge(
        script=script, from_fp=from_fp, to_fp=to_fp,
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
            existing.get("script") == edge.script
            and existing.get("from_fp") == edge.from_fp
            and existing.get("to_fp") == edge.to_fp
        ):
            raise ValueError(
                f"Duplicate equivalence edge: {edge.script} "
                f"{edge.from_fp} -> {edge.to_fp} already declared."
            )
    entries.append({
        "script": edge.script,
        "from_fp": edge.from_fp,
        "to_fp": edge.to_fp,
        "reason": edge.reason,
        "marked_at": edge.marked_at,
    })
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    return edge
