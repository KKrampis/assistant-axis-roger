"""Deferred-rejudge registry for the provenance system.

Judging is expensive and non-deterministic, so we sometimes
intentionally choose to live with a stale judge cache rather than
pay the cost (and accept the rng) of rerunning.  Audits should know
about these deferrals: a deferred cache shouldn't keep showing up in
"what's stale?" rollups, but it should be queryable as "what have I
deferred, and why?".

This module implements a small declarative registry that maps a
glob over cache paths (optionally with a ``dep_key`` filter) to a
deferral entry.  ``audit_caches.py`` and ``audit_pngs.py`` consult
the registry and reclassify both ``stale_direct`` /
``stale_transitive`` AND ``legacy`` rows to ``deferred`` when an
entry matches:

* ``stale_*`` -> ``deferred``: "this would otherwise need rerunning;
  defer the rerun instead".
* ``legacy`` -> ``deferred``: "this never had an envelope -- pre-
  migration bare JSON, or a producer we've decided not to migrate.
  Presume current as of mechanism introduction; would auto-clear if
  the producer were re-run with the modern writer pattern."

The legacy case is what supports the May 2026 backfill: judge caches
written by ``axis_judge_correlation.py`` predate the Phase 6
provenance wrap, and we deliberately don't re-run them; deferral
entries record that the staleness is known and intentional rather
than letting them accumulate as uncategorised ``legacy`` noise in
audits.

Categories
----------
Each deferral carries a structured :class:`DeferralCategory` that
records *why* the entry exists.  Categories drive both audit-report
grouping and category-specific invariant checks in
``tools/audit_deferrals.py``:

* ``LEGACY_BARE`` -- pre-Phase-6 bare JSON missing an envelope.
  Auto-clears when the producer is re-run (envelope replaces bare
  file).
* ``FROZEN_SNAPSHOT`` -- explicit point-in-time capture with a live
  twin (e.g. ``__rubric_v1.json`` snapshots taken before v2).
  Comparison plots typically read both snapshot AND live data.  Never
  auto-clears; optional ``compares_to`` points at the live twin so
  the audit can warn when the twin disappears.
* ``ARCHIVED`` -- standalone output of a retired pipeline
  configuration with NO live twin (pre-cohort-cutover backups, deprecated
  variants).  Kept on-disk for historical reference; never auto-
  clears, no live twin to verify.
* ``ORPHAN_NO_PRODUCER`` -- output whose producer script is not in
  the git tree (one-off ``/tmp/_*.py`` scripts, removed/superseded
  producers).  ⚠️ ``audit_deferrals`` raises if any matching file's
  recorded ``Software:`` / ``producer.cmdline`` references a path
  that has reappeared in ``git ls-files`` -- the maintainer probably
  promoted the producer and forgot to lift the deferral.  Optional
  ``producer_script`` records the expected/historical script path.
* ``OPERATIONAL`` -- config / API-usage / tracker files written
  without an envelope by design.  Not data dependencies; never
  auto-clears.
* ``MANIFEST_TRACKED`` -- files whose freshness is asserted via a
  per-dataset ``MANIFEST.json`` rather than per-file envelopes (see
  AGENT_NOTES.md "Manifest-tracked producers").  Different provenance
  regime; never auto-clears.
* ``SUPERSEDED`` -- replaced by a named newer artifact.  Optional
  ``replaced_by`` points at the successor; the audit can warn when
  the successor disappears (since the deferral becomes incoherent).
* ``EXTERNAL_PIPELINE`` -- output of a separate workflow outside the
  current provenance migration scope (e.g. coherence_eval pipeline).
* ``EXPERIMENTAL_ONE_OFF`` -- exploratory artifact with no plan to
  integrate (e.g. Sonnet response judging that was never carried over
  to v2 rubric).
* ``HAND_CURATED_INPUT`` -- hand-edited config consumed by producers
  but not produced by them (e.g. ``pair_list*.json``).  No envelope
  expected.

Schema
------
Stored at ``<repo_root>/deferred_rejudges.yaml``::

    schema_version: 2
    deferrals:
      - path_glob: "roger/axis_judge_experiments/q9_*/gpt/scores_descriptions.json"
        dep_key:   ~                # optional; matches any dep_key when null
        category:  legacy_bare
        reason:    "Desc-mode rejudge deferred until the 4-model audit completes."
        deferred_at: "2026-05-08T04:30:00+00:00"
      - path_glob: "roger/optimal_axis/truthful_vs_deceitful_responses_slot*/diagnostics.json"
        dep_key:   ~
        category:  archived
        reason:    "Slot-6 responses fit was the v1-rubric baseline ..."
        deferred_at: "2026-05-09T14:24:48+00:00"
      - path_glob: "roger/canonical_angles_*.png"
        dep_key:   ~
        category:  orphan_no_producer
        producer_script: "/tmp/_canonical_angles_v1.py"   # historical, not in git
        reason:    "Pre-Phase-6 one-off; producer no longer in codebase."
        deferred_at: "2026-05-09T03:49:18+00:00"

Backwards-compat: ``schema_version: 1`` files (no ``category``
field) are still readable; entries default to
``DeferralCategory.UNCATEGORIZED``.  Writers always emit v2.

Semantics
---------
* ``path_glob`` is a Unix-shell glob applied to repo-relative paths
  via ``fnmatch.fnmatch``; matches any file the audit walks.

  **fnmatch quirk worth knowing**: ``*`` matches across path
  separators (so ``*/scores_descriptions.json`` will match
  ``roger/axis_judge_experiments/q9_x_vs_y/gpt/scores_descriptions.json``),
  but it does NOT match the empty string at the start of a path.
  In practice that means ``*/roger/foo.json`` will NOT match the
  top-level ``roger/foo.json`` (no characters before ``roger/``),
  while it does match ``archive/roger/foo.json``.  When deferring
  top-level files in ``roger/``, write the pattern as
  ``roger/foo.json`` directly (no ``*/`` prefix).
* ``dep_key`` (when set) is a glob over the dep_key string of the
  drift-source InputSpec inside the cache; useful when one cache file
  depends on many sub-inputs and only some should be deferred.
* ``category`` -- :class:`DeferralCategory`; see Categories section
  above.  ``UNCATEGORIZED`` for v1-schema entries that haven't been
  backfilled.
* ``producer_script`` (optional) -- repo-relative path to the
  expected producer script.  Used by ``ORPHAN_NO_PRODUCER`` entries:
  the audit warns if this path now exists in git (potential
  promotion).
* ``replaced_by`` (optional) -- repo-relative path-glob of the
  superseding artifact.  Used by ``SUPERSEDED`` entries.
* ``compares_to`` (optional) -- repo-relative path-glob of the live
  twin.  Used by ``FROZEN_SNAPSHOT`` entries.
* ``reason`` is free-text; surfaced in audit reports.
* ``deferred_at`` is the ISO-8601 timestamp when the deferral was
  recorded.  Deferred-until-removed semantics: there's no
  ``expires_at``; deferrals stay active until the maintainer deletes
  the entry.

Public API
----------
* :func:`load_registry` -- parse the YAML.
* :func:`match_cache` -- given a cache path, return the list of
  matching :class:`DeferralEntry` records.
* :func:`append_deferral` -- add a new entry.
* :func:`remove_deferral` -- delete by index or by exact field match.
"""

from __future__ import annotations

import datetime as _dt
import enum as _enum
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

__all__ = [
    "DEFERRAL_FILENAME",
    "DeferralCategory",
    "DeferralEntry",
    "load_registry",
    "match_cache",
    "append_deferral",
    "remove_deferral",
]

DEFERRAL_FILENAME = "deferred_rejudges.yaml"
SCHEMA_VERSION = 2
"""Current writer schema version.  Reader supports v1 (no category) +
v2 (category required).  v1 entries load with category set to
:attr:`DeferralCategory.UNCATEGORIZED`."""

_SUPPORTED_READ_VERSIONS = (1, 2)


class DeferralCategory(str, _enum.Enum):
    """Structured taxonomy of deferral entries.

    Inherits from ``str`` so values round-trip as plain YAML strings
    (e.g. ``category: legacy_bare`` reads as
    :attr:`DeferralCategory.LEGACY_BARE`).
    """

    LEGACY_BARE = "legacy_bare"
    """Pre-Phase-6 bare JSON; auto-clears when producer is re-run."""

    FROZEN_SNAPSHOT = "frozen_snapshot"
    """Point-in-time capture with a live twin.  Optional
    ``compares_to`` field."""

    ARCHIVED = "archived"
    """Standalone output of a retired pipeline configuration; no
    live twin."""

    ORPHAN_NO_PRODUCER = "orphan_no_producer"
    """Producer script not in git.  ⚠️ Promotion-detection alarm in
    ``audit_deferrals``.  Optional ``producer_script`` records the
    expected path."""

    OPERATIONAL = "operational"
    """Config / API-usage / tracker files written without envelope by
    design."""

    MANIFEST_TRACKED = "manifest_tracked"
    """Freshness asserted via a per-dataset MANIFEST.json rather than
    per-file envelopes."""

    SUPERSEDED = "superseded"
    """Replaced by a named newer artifact (optional ``replaced_by``
    field)."""

    EXTERNAL_PIPELINE = "external_pipeline"
    """Output of a workflow outside the current provenance migration
    scope."""

    EXPERIMENTAL_ONE_OFF = "experimental_one_off"
    """Exploratory artifact with no plan to integrate."""

    HAND_CURATED_INPUT = "hand_curated_input"
    """Hand-edited config consumed by producers; not a producer
    output."""

    UNCATEGORIZED = "uncategorized"
    """Placeholder for v1-schema entries that haven't been
    backfilled.  New entries should always pick a real category."""


@dataclass(frozen=True)
class DeferralEntry:
    """One declared deferred-rejudge entry."""

    path_glob: str            # repo-relative glob over cache paths
    dep_key: Optional[str]    # optional dep_key glob; None = match any
    reason: str
    deferred_at: str
    category: DeferralCategory = DeferralCategory.UNCATEGORIZED
    """Structured intent tag; drives audit-report grouping and
    category-specific invariant checks.  Defaults to
    :attr:`DeferralCategory.UNCATEGORIZED` for v1-schema entries that
    haven't been backfilled."""

    producer_script: Optional[str] = None
    """For ``ORPHAN_NO_PRODUCER``: repo-relative path to the
    historical/expected producer.  ``audit_deferrals`` raises if this
    path is now present in ``git ls-files`` (likely promotion that
    needs the deferral lifted)."""

    replaced_by: Optional[str] = None
    """For ``SUPERSEDED``: repo-relative path-glob of the superseding
    artifact."""

    compares_to: Optional[str] = None
    """For ``FROZEN_SNAPSHOT``: repo-relative path-glob of the live
    twin used by paired comparison plots."""


def _repo_root_default() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def _registry_path(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or _repo_root_default()) / DEFERRAL_FILENAME


def _parse_category(raw_value: object) -> DeferralCategory:
    """Best-effort coercion of a YAML string to :class:`DeferralCategory`.

    Returns :attr:`DeferralCategory.UNCATEGORIZED` for missing or
    unknown values (forward compatibility: a newer schema version
    might add categories this build doesn't know about; we keep
    reading rather than crash).
    """
    if raw_value is None:
        return DeferralCategory.UNCATEGORIZED
    s = str(raw_value).strip().lower()
    try:
        return DeferralCategory(s)
    except ValueError:
        return DeferralCategory.UNCATEGORIZED


def load_registry(repo_root: Optional[Path] = None) -> List[DeferralEntry]:
    """Load the registry, returning the list of declared entries.

    Empty list when the YAML file doesn't exist.
    Raises ``ValueError`` for unsupported ``schema_version`` values.
    Reads both v1 (no ``category`` field) and v2 (category required
    on writes); v1 entries load with
    :attr:`DeferralCategory.UNCATEGORIZED`.
    """
    path = _registry_path(repo_root)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    sv = raw.get("schema_version")
    if sv not in _SUPPORTED_READ_VERSIONS:
        raise ValueError(
            f"{path}: unknown schema_version {sv!r}; "
            f"this assistant_axis build supports {_SUPPORTED_READ_VERSIONS!r}."
        )
    out: List[DeferralEntry] = []
    for entry in raw.get("deferrals", []) or []:
        out.append(DeferralEntry(
            path_glob=str(entry["path_glob"]),
            dep_key=str(entry["dep_key"]) if entry.get("dep_key") else None,
            reason=str(entry.get("reason", "")),
            deferred_at=str(entry.get("deferred_at", "")),
            category=_parse_category(entry.get("category")),
            producer_script=(str(entry["producer_script"])
                             if entry.get("producer_script") else None),
            replaced_by=(str(entry["replaced_by"])
                         if entry.get("replaced_by") else None),
            compares_to=(str(entry["compares_to"])
                         if entry.get("compares_to") else None),
        ))
    return out


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def match_cache(
    cache_path: Path,
    *,
    dep_key: Optional[str] = None,
    registry: Optional[List[DeferralEntry]] = None,
    repo_root: Optional[Path] = None,
) -> List[DeferralEntry]:
    """Return all registry entries matching ``cache_path`` (and optionally
    ``dep_key``).

    ``dep_key`` is the dep_key string of the drift-source InputSpec
    inside the cache; pass ``None`` to match any dep_key.  An entry
    with ``dep_key=None`` in the registry matches any caller-side
    dep_key value (intent: "defer this cache regardless of which sub-
    input changed").

    Returns an empty list when no entries match -- the cache is not
    deferred.  Multiple matches indicate overlapping rules; callers
    typically just check whether the list is non-empty.

    ``repo_root`` is auto-detected (via ``.git``) when omitted.
    """
    reg = registry if registry is not None else load_registry(repo_root)
    if not reg:
        return []
    root = repo_root or _repo_root_default()
    rel = _to_repo_relative(Path(cache_path), root)
    matches: List[DeferralEntry] = []
    for entry in reg:
        if not fnmatch.fnmatch(rel, entry.path_glob):
            continue
        if entry.dep_key is not None and dep_key is not None:
            if not fnmatch.fnmatch(dep_key, entry.dep_key):
                continue
        matches.append(entry)
    return matches


def append_deferral(
    *,
    path_glob: str,
    reason: str,
    dep_key: Optional[str] = None,
    deferred_at: Optional[str] = None,
    category: DeferralCategory = DeferralCategory.UNCATEGORIZED,
    producer_script: Optional[str] = None,
    replaced_by: Optional[str] = None,
    compares_to: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> DeferralEntry:
    """Append one entry to the registry, creating the file if needed.

    Returns the new :class:`DeferralEntry`.

    Raises ``ValueError`` if an identical entry (same ``path_glob`` +
    ``dep_key``) already exists.  Reason / timestamp may differ; we
    refuse silent overwrites to keep the registry append-only.

    Existing v1 schema files are upgraded in-place to v2 on first
    write; existing entries are preserved (they remain
    :attr:`DeferralCategory.UNCATEGORIZED` until backfilled).
    """
    if deferred_at is None:
        deferred_at = (_dt.datetime.now(tz=_dt.timezone.utc)
                       .replace(microsecond=0).isoformat())
    new = DeferralEntry(
        path_glob=path_glob, dep_key=dep_key,
        reason=reason, deferred_at=deferred_at,
        category=category,
        producer_script=producer_script,
        replaced_by=replaced_by,
        compares_to=compares_to,
    )
    path = _registry_path(repo_root)
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        sv = raw.get("schema_version")
        if sv not in _SUPPORTED_READ_VERSIONS:
            raise ValueError(
                f"{path}: refusing to write to unknown schema_version {sv!r}"
            )
    else:
        raw = {"schema_version": SCHEMA_VERSION, "deferrals": []}
    raw["schema_version"] = SCHEMA_VERSION
    entries = raw.setdefault("deferrals", []) or []
    raw["deferrals"] = entries
    for existing in entries:
        if (
            existing.get("path_glob") == new.path_glob
            and (existing.get("dep_key") or None) == new.dep_key
        ):
            raise ValueError(
                f"Duplicate deferral entry for path_glob={new.path_glob!r}, "
                f"dep_key={new.dep_key!r}; remove the old one first."
            )
    entries.append(_serialise_entry(new))
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    return new


def _serialise_entry(entry: DeferralEntry) -> dict:
    """Convert a :class:`DeferralEntry` into a YAML-friendly dict.

    Optional fields are emitted only when set, so v2 entries that
    don't need ``producer_script`` / ``replaced_by`` / ``compares_to``
    stay terse on disk.
    """
    out: dict = {
        "path_glob": entry.path_glob,
        "dep_key": entry.dep_key,
        "category": entry.category.value,
        "reason": entry.reason,
        "deferred_at": entry.deferred_at,
    }
    if entry.producer_script is not None:
        out["producer_script"] = entry.producer_script
    if entry.replaced_by is not None:
        out["replaced_by"] = entry.replaced_by
    if entry.compares_to is not None:
        out["compares_to"] = entry.compares_to
    return out


def remove_deferral(
    *,
    path_glob: str,
    dep_key: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> bool:
    """Remove one entry by exact (path_glob, dep_key) match.

    Returns True when an entry was removed, False when no match was
    found.
    """
    path = _registry_path(repo_root)
    if not path.exists():
        return False
    raw = yaml.safe_load(path.read_text()) or {}
    sv = raw.get("schema_version")
    if sv not in _SUPPORTED_READ_VERSIONS:
        raise ValueError(
            f"{path}: refusing to mutate unknown schema_version {sv!r}"
        )
    entries = raw.get("deferrals", []) or []
    new_entries = [
        e for e in entries
        if not (
            e.get("path_glob") == path_glob
            and (e.get("dep_key") or None) == dep_key
        )
    ]
    if len(new_entries) == len(entries):
        return False
    raw["schema_version"] = SCHEMA_VERSION
    raw["deferrals"] = new_entries
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    return True
