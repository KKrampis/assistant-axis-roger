"""Judge-side data loaders for response-mode and static-mode caches.

This module is the canonical reader-side interface for the per-axis
judge cache fan-out under ``roger/axis_judge_experiments/<axis>/``.

The two main entry points are:

* :func:`load_response_scores` — read response-mode scores for one
  ``(axis, kind, judge, rubric)`` combination, with **per-entity B
  fallback** for Haiku and **conditional provenance registration**
  (only registers cohort files that actually contributed an entity to
  the returned result).  This is the helper that wires Change D's
  B=10 → B=7 default switch through the consumer side.

* :func:`load_static_scores` / :func:`load_projections` /
  :func:`load_static_correlations` — readers for the static-mode
  caches under ``<axis>/<judge>/{scores_*,projections,correlations}.json``
  with strict ``schema_version`` enforcement (Option C from the plan).
  These will be added in Phase 4; for now this module only ships
  :func:`load_response_scores` (Phase 5e).

Suffix conventions for cohort directories
==========================================

* ``(no suffix)`` — full volume (every score==3 item judged).  GPT
  default after Phase 4b.
* ``_q<N>`` — uniform ``--question_subsample_modulo N``.  Used by the
  legacy ``_b10_q9`` Haiku and frozen Sonnet cohorts.  Going forward we
  don't write new ``_q<N>`` directories for Haiku (see 2026-05-21 note
  below); Sonnet ``_b10_q9`` is the still-canonical frozen cohort.
* ``_t<M>`` — tiered subsampling at ``--tiered_modulo_per_chunk M``.
  Used by the new Haiku ``_b7_t3`` cohorts and any future tiered
  Haiku/Sonnet runs.  The per-entity tier (1/2/3) is auto-detected at
  run time; it's recorded inside the cache, not in the directory name.

**2026-05-21 status of Haiku cohorts**

The Haiku ``_b10_q9`` cohort is OBSOLETE.  Its purpose was to serve as
a uniform-sample fallback alongside the tiered ``_b7_t3`` cohort, but
in practice the tiered cohort auto-escalates (tier 1 → tier 2 → tier 3
per-entity) until every persona has enough graded items, which
typically yields ~q9-equivalent coverage anyway.  Running both
cohorts is therefore mostly redundant: roughly twice the cost for
little marginal coverage.

The default ``_DEFAULT_PREFER_B`` for ``("haiku", *)`` was reduced to
``(7,)`` on 2026-05-21 and restored to ``(7, 10)`` on 2026-05-22 (the
original 12 axes still take most of their entities from ``_b10_q9``;
see the comment on ``_DEFAULT_PREFER_B``).  So existing ``_b10_q9``
Haiku caches ARE read, per entity, as the fallback after ``_b7_t3``.
No new ``_b10_q9`` Haiku cohorts should be written.

Conditional-provenance contract
================================

The naive approach to fallback (register both ``_b7_t3`` and
``_b10_q9`` as deps unconditionally) is *wrong* — it would mark every
``_b7_t3``-only artifact as stale every time ``_b10_q9`` changes,
even when the artifact never read a single byte from ``_b10_q9``.
:func:`load_response_scores` registers cohort files **only when they
actually contributed at least one entity** to the returned dict.

See ``trait_role_name_disambiguation_06e61abc.plan.md`` Phase 5e for
the full design rationale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from .entity_id import is_entity_id, kind_long
from .provenance import (
    InputSpec,
    ProvenanceCheck,
    current_file_input,
    load_validated_json,
)


__all__ = [
    "DEFAULT_EXPERIMENTS_ROOT",
    "EXPECTED_STATIC_SCHEMA_VERSION",
    "StaleSchemaError",
    "load_response_scores",
    "load_static_scores",
    "load_projections",
    "load_static_correlations",
    "cohort_dirname",
    "filename_for_rubric",
    "subsample_method_for_dirname",
    "dep_key_for_cohort",
    "peek_rubric_version",
    "peek_per_entity_rubric_versions",
    "RubricVersionReport",
    "rubric_version_report",
    "migrate_v1_static_scores",
]


#: The schema version this module's static-mode loaders accept.  Files
#: written by the post-Phase-4 producer carry this number at the top
#: level of the envelope; older (v1, bare-name-keyed) files do not, and
#: are loud-rejected via :class:`StaleSchemaError`.
EXPECTED_STATIC_SCHEMA_VERSION: int = 2


class StaleSchemaError(Exception):
    """Raised by the static-mode loaders when a cache lacks the expected
    ``schema_version`` (v1 or missing).

    The May 2026 trait/role disambiguation work bumped the on-disk
    schema for the AT-RISK mixed-kind cache families
    (``scores_descriptions``, ``scores_instructions``, ``projections``,
    ``correlations``).  Per Decision 6 / Option C in the plan, we
    reject stale schemas loudly rather than auto-migrating — the v1
    bare-name format silently dropped one side of every collision
    name, so an auto-migration would be ambiguous (we can't infer
    *which* side the cache holds).

    Carries the path, the recorded vs expected schema versions, and a
    concrete regenerate-via command so the operator can fix the issue
    without spelunking the codebase.
    """

    def __init__(
        self,
        path: Path,
        recorded: Optional[int],
        expected: int,
        regenerate_via: str,
    ) -> None:
        self.path = Path(path)
        self.recorded = recorded
        self.expected = expected
        self.regenerate_via = regenerate_via
        rec_str = (
            f"v{recorded}" if recorded is not None
            else "v1-or-unset (legacy bare-name format)"
        )
        super().__init__(
            f"{path}: schema {rec_str}; expected v{expected}.  "
            f"Regenerate via:\n    {regenerate_via}"
        )


#: Project-default root for the per-axis judge cache fan-out.  Override
#: via the ``experiments_root`` argument in tests.
DEFAULT_EXPERIMENTS_ROOT: Path = Path("roger/axis_judge_experiments")


# Map (judge, rubric) -> default prefer_b priority order.
#
# * GPT v2: only B=7 after Phase 5c (the broken b10 v2 caches are
#   deferred and explicitly NOT in the default fallback chain).
# * GPT v1: B=10 reference (B=15/10/7/5 all exist for the B-curve, but
#   the canonical reference is B=10).
# * Haiku v1/v2: B=7 t3 (the tiered surgical cohort) first, then the
#   legacy B=10 q9 cohort per entity.  q9 is obsolete for *writing*
#   (2026-05-21: the tiered cohort auto-escalates from tier 1 (1/3
#   modulo) to tier 2 (2/3) or tier 3 (full) per entity), but it is
#   still read as the fallback (restored 2026-05-22, below).
# * Sonnet: frozen at B=10 q9, no rejudging.
_DEFAULT_PREFER_B: dict[tuple[str, str], tuple[int, ...]] = {
    ("gpt", "v2"):    (7,),
    ("gpt", "v1"):    (10,),
    # Haiku: B=7 preferred (canonical), with B=10 as legacy fallback for
    # axes/entities not yet rejudged at B=7.  The B=10 cohort was
    # initially marked obsolete on 2026-05-21 with the intent "don't
    # write new q9", but the fallback was inadvertently dropped from
    # the read path too -- restored on 2026-05-22 once it became clear
    # that the 10 new Phase-1/2 axes have only B=7 data while the
    # original 12 axes still rely on B=10 for the bulk of their
    # entity coverage (only the ~14-29 collision-disambiguation
    # entities per axis got the b7_t3 surgical rejudge).
    ("haiku", "v2"):  (7, 10),
    ("haiku", "v1"):  (7, 10),
    ("sonnet", "v2"): (10,),
    ("sonnet", "v1"): (10,),
}


# Encodes the (judge, B) -> sub-suffix map.  After Phase 5c GPT
# always writes a no-suffix directory (full volume by default for B=7,
# and the legacy B=10 dir is also no-suffix); Haiku/Sonnet use the
# subsampling suffix to make the selection method explicit at the
# filename level.
#
# 2026-05-21: ``(haiku, 10)`` mapping ("_q9") is kept here for read-side
# compatibility with legacy caches, but the corresponding cohort is
# marked OBSOLETE and is no longer written by new runs (the tiered t3
# cohort auto-escalates to full-volume coverage per-entity as needed).
_COHORT_SUB_SUFFIX: dict[tuple[str, int], str] = {
    ("gpt",     5):  "",
    ("gpt",     7):  "",
    ("gpt",    10):  "",
    ("gpt",    15):  "",
    ("haiku",   7):  "_t3",          # canonical (tiered, auto-escalating)
    ("haiku",  10):  "_q9",          # OBSOLETE 2026-05-21 (kept for read-side compat)
    ("sonnet", 10):  "_q9",
}


def cohort_dirname(judge: str, kind: str, b: int) -> str:
    """Return the directory name for ``(judge, kind, B)`` under the
    per-axis ``<axis>/`` root.

    Encodes the project's suffix conventions (see module docstring):

    >>> cohort_dirname("gpt", "roles", 7)
    'gpt_responses_roles_b7'
    >>> cohort_dirname("haiku", "traits", 10)
    'haiku_responses_traits_b10_q9'
    >>> cohort_dirname("haiku", "roles", 7)
    'haiku_responses_roles_b7_t3'
    >>> cohort_dirname("sonnet", "traits", 10)
    'sonnet_responses_traits_b10_q9'

    Raises:
        ValueError: If ``judge`` is unknown, or if there's no known
            cohort directory for ``(judge, B)``.
    """
    judge = judge.lower()
    kind_canon = kind_long(kind)
    suffix = _COHORT_SUB_SUFFIX.get((judge, b))
    if suffix is None:
        raise ValueError(
            f"cohort_dirname: no known cohort directory for "
            f"judge={judge!r} B={b!r}; known combinations: "
            f"{sorted(_COHORT_SUB_SUFFIX)}"
        )
    return f"{judge}_responses_{kind_canon}_b{b}{suffix}"


def filename_for_rubric(rubric: str) -> str:
    """Return the ``scores_responses*.json`` filename for ``rubric``.

    >>> filename_for_rubric("v2")
    'scores_responses.json'
    >>> filename_for_rubric("v1")
    'scores_responses__rubric_v1.json'
    """
    rubric = rubric.lower()
    if rubric == "v2":
        return "scores_responses.json"
    if rubric == "v1":
        return "scores_responses__rubric_v1.json"
    raise ValueError(
        f"filename_for_rubric: unknown rubric {rubric!r}; "
        f"accepted: 'v1', 'v2'"
    )


def subsample_method_for_dirname(dirname: str) -> str:
    """Decode the subsampling-method tag from a cohort directory name.

    Returns one of:

    * ``"full_volume"`` — the cohort dir has no suffix (e.g.
      ``gpt_responses_traits_b7``).
    * ``"uniform_mod<N>"`` — ``_q<N>`` legacy uniform-modulo cohort.
    * ``"tiered_t<M>"`` — ``_t<M>`` tiered cohort (per-entity tier
      auto-detected at run time; not encoded in the dirname).
    * ``"unknown:<suffix>"`` — fallback for any future suffix we
      haven't taught the helper about.

    >>> subsample_method_for_dirname("gpt_responses_traits_b7")
    'full_volume'
    >>> subsample_method_for_dirname("haiku_responses_roles_b10_q9")
    'uniform_mod9'
    >>> subsample_method_for_dirname("haiku_responses_roles_b7_t3")
    'tiered_t3'
    """
    parts = dirname.split("_")
    b_idx: Optional[int] = None
    for i, p in enumerate(parts):
        if p.startswith("b") and p[1:].isdigit():
            b_idx = i
            break
    if b_idx is None:
        return "full_volume"
    suffix_parts = parts[b_idx + 1:]
    if not suffix_parts:
        return "full_volume"
    suffix = "_".join(suffix_parts)
    if suffix.startswith("q") and suffix[1:].isdigit():
        return f"uniform_mod{suffix[1:]}"
    if suffix.startswith("t") and suffix[1:].isdigit():
        return f"tiered_t{suffix[1:]}"
    return f"unknown:{suffix}"


def dep_key_for_cohort(dirname: str, rubric: str) -> str:
    """Build the InputSpec ``dep_key`` for a cohort × rubric.

    Format: ``<dirname>_<rubric>`` so the dep_key encodes the cohort's
    subsample method (via the dirname's suffix) and the rubric version
    in one short string.  Used by audit tools to trace artifacts back
    to their cohort sources without re-parsing the recorded path.

    >>> dep_key_for_cohort("haiku_responses_roles_b7_t3", "v2")
    'haiku_responses_roles_b7_t3_v2'
    >>> dep_key_for_cohort("gpt_responses_traits_b7", "v2")
    'gpt_responses_traits_b7_v2'
    """
    return f"{dirname}_{rubric}"


def load_response_scores(
    axis: str,
    kind: str,
    judge: str,
    rubric: str,
    *,
    experiments_root: Optional[Union[Path, str]] = None,
    prefer_b: Optional[tuple[int, ...]] = None,
    inputs: Optional[list[InputSpec]] = None,
    policy: str = "warn",
) -> tuple[dict[str, dict], dict[str, str]]:
    """Load response-mode judge scores with per-entity B fallback.

    The returned ``scores_by_entity`` maps **bare** entity names (kind
    is implicit in the cohort) to score-entry dicts.  Consumers that
    merge across kinds must apply
    :func:`assistant_axis.entity_id.entity_id` themselves at merge
    time.

    Args:
        axis: Axis name (e.g. ``"concise_vs_verbose"``); resolves to
            ``<experiments_root>/<axis>/<cohort_dirname>/<filename>``.
        kind: ``"roles"`` or ``"traits"`` (case-insensitive; any
            spelling :func:`assistant_axis.entity_id.kind_long`
            accepts).
        judge: ``"gpt"``, ``"haiku"``, or ``"sonnet"``.
        rubric: ``"v1"`` or ``"v2"``.
        experiments_root: Override for the per-axis cache root.
            Defaults to :data:`DEFAULT_EXPERIMENTS_ROOT`.  Useful in
            tests.
        prefer_b: Cohort B priority list, **most preferred first**.
            For each entity, the first value of ``B`` whose cohort
            directory contains a record for that entity wins.
            Defaults are encoded by ``(judge, rubric)``:

            * ``("gpt", "v2") -> (7,)`` (after Phase 5c covers all v2
              GPT axes at B=7; explicitly does NOT fall back to the
              broken v2 b10 caches).
            * ``("gpt", "v1") -> (10,)`` (legitimate v1 reference).
            * ``("haiku", *) -> (7, 10)`` (new tiered surgical
              cohort first, legacy uniform-mod-9 fallback).
            * ``("sonnet", *) -> (10,)`` (frozen, no rejudging).
        inputs: If non-None, the helper *appends* one
            :class:`InputSpec` for each cohort file that **actually
            contributed** at least one entity to the returned result.
            Cohorts that were opened but contributed nothing (e.g.
            ``_b10_q9`` opened to look up an entity that turned out
            to live in ``_b7_t3``) are NOT registered.  This is the
            "only register what you actually read" guarantee.
        policy: Drift-handling policy forwarded to
            :func:`assistant_axis.provenance.load_validated_json` for
            each cohort cache's *own* recorded inputs.  ``"warn"``
            (default), ``"strict"``, ``"rebuild"`` (no callback
            wired), or ``"off"``.

    Returns:
        ``(scores_by_entity, source_cohort_by_entity)`` where:

        * ``scores_by_entity`` — bare-name keyed dict.  Each entry
          carries the existing fields (``mean_score``, ``std_score``,
          ``n_batches``, ``n_total_items``, ``per_batch``, ...) **plus**
          two new fields populated at read time:

          - ``source_cohort`` — the cohort directory name (e.g.
            ``"haiku_responses_roles_b7_t3"``).
          - ``subsample_method`` — derived from the dirname suffix
            via :func:`subsample_method_for_dirname` (e.g.
            ``"tiered_t3"`` or ``"uniform_mod9"``).

          Score-entry mutation is on a *copy* of the on-disk dict, so
          repeated calls with the same cohort don't accrete metadata.
        * ``source_cohort_by_entity`` — bare-name → cohort dirname
          map.  Useful for stamping into an artifact's ``extras``
          (recommended field name: ``"haiku_fallback_map"``).

    Raises:
        ValueError: For unknown ``judge`` / ``rubric`` / ``kind`` or
            an empty ``prefer_b`` resolved to no defaults.

    Notes:
        Missing cohort files are silently skipped (we walk every B in
        ``prefer_b`` and just continue if the file isn't there).  An
        entity that's missing from every prefer_b cohort is absent
        from the returned dict — consumers should treat this as "no
        data for this entity in this (axis, judge, rubric, kind)
        cell" and decide what to do.

    See Also:
        ``trait_role_name_disambiguation_06e61abc.plan.md`` Phase 5e
        for the full design (especially the conditional-provenance
        rationale).
    """
    if experiments_root is None:
        experiments_root = DEFAULT_EXPERIMENTS_ROOT
    experiments_root = Path(experiments_root)
    judge_norm = judge.lower()
    rubric_norm = rubric.lower()
    kind_canon = kind_long(kind)

    # Validate rubric first so an invalid rubric always gets the
    # clearest possible error message regardless of which judge it
    # was paired with.
    filename = filename_for_rubric(rubric_norm)

    if prefer_b is None:
        defaults = _DEFAULT_PREFER_B.get((judge_norm, rubric_norm))
        if defaults is None:
            raise ValueError(
                f"load_response_scores: no default prefer_b for "
                f"judge={judge!r} rubric={rubric!r}; pass prefer_b "
                f"explicitly.  Known defaults: {sorted(_DEFAULT_PREFER_B)}"
            )
        prefer_b = defaults
    if not prefer_b:
        raise ValueError(
            "load_response_scores: prefer_b must be a non-empty tuple"
        )
    scores_by_entity: dict[str, dict] = {}
    source_cohort_by_entity: dict[str, str] = {}
    # (B, dirname, full_path) for every cohort we successfully opened.
    opened_cohorts: list[tuple[int, str, Path]] = []

    for b in prefer_b:
        dirname = cohort_dirname(judge_norm, kind_canon, b)
        cohort_path = experiments_root / axis / dirname / filename
        if not cohort_path.exists():
            continue
        # ``load_validated_json`` gives us drift-checking on the
        # cohort cache's own recorded inputs, plus envelope unwrap.
        payload, _check = load_validated_json(cohort_path, policy=policy)
        if not isinstance(payload, dict):
            raise ValueError(
                f"load_response_scores: expected dict payload at "
                f"{cohort_path}, got {type(payload).__name__}"
            )
        opened_cohorts.append((b, dirname, cohort_path))
        method = subsample_method_for_dirname(dirname)
        for entity_name, entry in payload.items():
            if entity_name in scores_by_entity:
                # Earlier-priority B already supplied this entity.
                continue
            if isinstance(entry, dict):
                entry_copy = dict(entry)
                entry_copy["source_cohort"] = dirname
                entry_copy["subsample_method"] = method
            else:
                # Defensive: payload could in principle be a flat
                # name->scalar map (it isn't currently, but the
                # interface doesn't mandate dict entries).
                entry_copy = entry
            scores_by_entity[entity_name] = entry_copy
            source_cohort_by_entity[entity_name] = dirname

    # Conditional-registration: only register cohort files that
    # actually contributed at least one entity to the result.
    contributing_dirnames = set(source_cohort_by_entity.values())
    if inputs is not None:
        for b, dirname, cohort_path in opened_cohorts:
            if dirname not in contributing_dirnames:
                continue
            is_preferred = (b == prefer_b[0])
            spec = current_file_input(
                dep_key=dep_key_for_cohort(dirname, rubric_norm),
                path=cohort_path,
                extras={
                    "resolution": "preferred" if is_preferred else "fallback",
                    "B": b,
                },
            )
            inputs.append(spec)

    return scores_by_entity, source_cohort_by_entity


# ---------------------------------------------------------------------------
# Static-mode loaders (Phase 4 — schema_version: 2 enforcement)
# ---------------------------------------------------------------------------

def _peek_schema_version(path: Path) -> Optional[int]:
    """Read just enough of ``path`` to find the top-level ``schema_version``
    field (added in Phase 4 — May 2026).

    Returns the integer schema version on success, or ``None`` if the
    file is missing the field (= legacy v1 bare-name format) or
    can't be parsed.

    Doesn't raise on parse failure; the caller (e.g.
    :func:`_assert_static_schema_v2`) decides what to do based on
    intent (mismatch + JSON error usually means the same thing:
    "this file isn't what we expect").
    """
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    raw = obj.get("schema_version")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _assert_static_schema_v2(
    path: Path,
    *,
    regenerate_via: str,
) -> None:
    """Raise :class:`StaleSchemaError` if ``path`` doesn't carry
    ``schema_version: 2`` at the top level of its envelope.

    Use as the first line of any static-mode loader to fail loudly on
    pre-Phase-4 caches.  ``FileNotFoundError`` propagates separately
    so the missing-cache case stays distinguishable from the
    wrong-schema case.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p}: file not found")
    recorded = _peek_schema_version(p)
    if recorded != EXPECTED_STATIC_SCHEMA_VERSION:
        raise StaleSchemaError(
            path=p,
            recorded=recorded,
            expected=EXPECTED_STATIC_SCHEMA_VERSION,
            regenerate_via=regenerate_via,
        )


def _assert_static_payload_disambiguated(
    path: Path,
    payload: Any,
    *,
    regenerate_via: str,
) -> None:
    """Sanity check that a v2 static-mode cache really uses
    disambiguated entity-id keys (covers the "schema_version field
    set but keys are bare names" forgot-to-disambiguate-on-write
    failure mode).

    Empty payloads pass (early-stage runs with zero entries).  A
    non-empty payload with **zero** entity-id keys raises
    :class:`StaleSchemaError`; a payload with at least one entity-id
    key passes (consistent with the "post-Phase-4 producer" intent).
    """
    if not isinstance(payload, dict) or not payload:
        return
    if any(is_entity_id(k) for k in payload):
        return
    raise StaleSchemaError(
        path=path,
        recorded=EXPECTED_STATIC_SCHEMA_VERSION,
        expected=EXPECTED_STATIC_SCHEMA_VERSION,
        regenerate_via=(
            regenerate_via
            + "  (declared schema_version=2 but no entity-id (name|R / "
            "name|T) keys present — producer wrote bare names by mistake.)"
        ),
    )


def peek_rubric_version(path: Union[Path, str]) -> Optional[str]:
    """Extract the recorded ``rubric_version`` label from a judge
    cache's provenance envelope.

    Returns the string stamped into the ``producer_script``
    :class:`InputSpec`'s ``extras`` at write time (e.g. ``"v3"``), or
    ``None`` if the file is missing, is not envelope-wrapped, or
    doesn't record a ``rubric_version`` (legacy pre-Phase-6 cache).

    Consumers use this to detect when caches across multiple
    (axis, judge, mode) cells were judged under inconsistent
    rubric versions — a soft warning, not an error, since "consistent
    within an experiment" is a property of the *collection* of
    caches, not of any single cache.  The producer side
    (:func:`results_analysis.axis_judge_correlation._check_rubric_version_on_resume`)
    handles the strict producer-resume case (drop the cache and
    rejudge under the current ``RUBRIC_VERSION``).
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    prov = obj.get("_provenance")
    if not isinstance(prov, dict):
        return None
    for inp in prov.get("inputs", []) or []:
        if not isinstance(inp, dict):
            continue
        if inp.get("dep_key") != "producer_script":
            continue
        extras = inp.get("extras") or {}
        v = extras.get("rubric_version")
        if isinstance(v, str):
            return v
    return None


def peek_per_entity_rubric_versions(
    path: Union[Path, str],
) -> dict[str, str]:
    """Read the per-entity ``rubric_version`` map stamped into a judge
    cache's ``_provenance.notes.per_entity_rubric_versions`` block.

    Consumer-side equivalent of
    :func:`results_analysis.axis_judge_correlation._peek_per_entity_rubric_versions`.
    Returns an empty dict on missing file / non-envelope JSON / cache
    without per-entity stamps (legacy pre-stamping data).  In that
    case callers typically fall back to the cohort-level
    :func:`peek_rubric_version` as the effective stamp for every
    entry.

    Map keys are whatever the producer keys the main cache by:
    disambiguated ``"name|R"`` / ``"name|T"`` for static-mode caches
    (``scores_descriptions.json``, ``scores_instructions.json``), bare
    ``name`` for the kind-pure response-mode caches
    (``scores_responses.json``).
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    if not isinstance(obj, dict):
        return {}
    prov = obj.get("_provenance")
    if not isinstance(prov, dict):
        return {}
    notes = prov.get("notes") or {}
    if not isinstance(notes, dict):
        return {}
    raw = notes.get("per_entity_rubric_versions")
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


@dataclass(frozen=True)
class RubricVersionReport:
    """Result of a consumer-side rubric-version freshness audit.

    Produced by :func:`rubric_version_report` for one judge cache.
    All counts are entry-level; the cohort-level ``cohort_rubric``
    stamp is included verbatim for diagnostic logging.

    Fields:
        path: The cache file inspected.
        cohort_rubric: Cohort-level ``rubric_version`` from the
            envelope's ``producer_script`` extras, or ``None`` for
            legacy caches predating stamping.
        current_rubric: The active ``RUBRIC_VERSION`` the caller is
            comparing against.
        n_total: Total entries in the inspected cache.
        n_current: Entries whose effective rubric (per-entity stamp
            if present, else cohort-level) matches ``current_rubric``.
        n_equivalent: Entries whose effective rubric differs from
            ``current_rubric`` but is declared equivalent for this
            (axis, mode, entity_id) cell in
            :mod:`assistant_axis.rubric_equivalence`.
        drifted: Map ``recorded_rubric -> [entity_keys]`` for entries
            that are neither current nor equivalent.  Empty when the
            cache is fully fresh-or-equivalent.

    ``ok`` (the bool conversion) is True iff ``drifted`` is empty;
    callers can short-circuit with ``if report:`` to mean "this
    cache is fresh enough to consume".
    """

    path: Path
    cohort_rubric: Optional[str]
    current_rubric: str
    n_total: int
    n_current: int
    n_equivalent: int
    drifted: dict[str, list[str]]

    def __bool__(self) -> bool:
        return not self.drifted

    @property
    def n_drifted(self) -> int:
        return sum(len(v) for v in self.drifted.values())


def rubric_version_report(
    path: Union[Path, str],
    *,
    current_rubric: str,
    axis: Optional[str] = None,
    mode: Optional[str] = None,
    entries: Optional[Iterable[str]] = None,
) -> RubricVersionReport:
    """Audit one judge cache for rubric-version drift.

    Reads the cohort-level ``rubric_version`` and per-entity stamp
    map from ``path``'s envelope, then for each entity in the cache
    (or in ``entries`` when provided) classifies it as:

    * **current**: effective rubric matches ``current_rubric``.
    * **equivalent**: effective rubric differs but is declared
      equivalent in :mod:`assistant_axis.rubric_equivalence` for the
      ``(axis, mode, entity_id)`` cell.
    * **drifted**: neither current nor equivalent.

    Returns a :class:`RubricVersionReport`.  Use it for soft-warn
    auditing of consumer-side reads; the report is informational and
    never mutates the cache or raises.

    Args:
        path: Cache file to inspect.
        current_rubric: The active ``RUBRIC_VERSION`` to compare
            against (typically read from
            ``results_analysis.axis_judge_correlation.RUBRIC_VERSION``;
            passed in explicitly so this module stays free of the
            heavyweight axis_judge_correlation import).
        axis: Axis name (e.g. ``"concise_vs_verbose"``); passed to
            the equivalence registry lookup.  ``None`` = wildcard
            (any axis-scoped edge applies).
        mode: ``"descriptions"`` / ``"instructions"`` /
            ``"responses"``; passed to the equivalence registry
            lookup.  ``None`` = wildcard.
        entries: Optional iterable of entity-key strings to restrict
            the audit to.  When omitted, every key in the cache's
            top-level ``result`` block is audited.  Useful when the
            caller knows exactly which entries it will consume and
            only cares about drift on those.
    """
    p = Path(path)
    cohort = peek_rubric_version(p)
    per_entity = peek_per_entity_rubric_versions(p)

    if entries is None:
        # Walk the cache's actual entry set.
        result_keys: list[str] = []
        try:
            obj = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            obj = {}
        if isinstance(obj, dict):
            inner = obj.get("result") if "_provenance" in obj else obj
            if isinstance(inner, dict):
                result_keys = list(inner.keys())
        target_keys = result_keys
    else:
        target_keys = list(entries)

    # Lazy-load registry once for this call (parallel to the producer side).
    from . import rubric_equivalence as _eq
    registry = _eq.load_registry()

    n_current = 0
    n_equivalent = 0
    drifted: dict[str, list[str]] = {}
    for key in target_keys:
        rv = per_entity.get(key, cohort)
        if rv is None or rv == current_rubric:
            n_current += 1
            continue
        if _eq.is_equivalent(
            rv, current_rubric,
            axis=axis, mode=mode, entity_id=key,
            registry=registry,
        ):
            n_equivalent += 1
            continue
        drifted.setdefault(rv, []).append(key)

    return RubricVersionReport(
        path=p,
        cohort_rubric=cohort,
        current_rubric=current_rubric,
        n_total=len(target_keys),
        n_current=n_current,
        n_equivalent=n_equivalent,
        drifted=drifted,
    )


def migrate_v1_static_scores(
    bare_scores: dict,
    kinds_for_name: dict,
) -> dict:
    """Re-key a v1 (bare-name) static-score dict to v2 (entity_id) form
    in-memory.

    Mirrors the producer-side migration in
    :func:`results_analysis.axis_judge_correlation.score_static_mode`,
    for use in **consumer** scripts that need to mix v1 caches
    (e.g. Sonnet desc/inst, which weren't included in Phase 5b/5c.1
    rejudging) with v2 caches (GPT, Haiku post-May 2026).

    Behaviour per key:

    * Already an entity_id (``"patient|R"``) → keep as-is (so already-v2
      caches pass through unchanged; idempotent).
    * Bare name uniquely belongs to one kind (per ``kinds_for_name``) →
      relabel as ``entity_id(name, kind)``.
    * Bare name is a known collision (in both ``traits`` and ``roles``)
      → DROP (Bug A: v1 silently overwrote one side, so we don't know
      which kind's score we have).
    * Bare name in no kind (orphan from a corpus edit) → DROP.

    ``kinds_for_name`` maps bare names to the set of kinds they
    belong to in the operative corpus.  Build it from any
    authoritative source available to the caller, typically the
    entity-vectors directory layout::

        kinds_for_name: dict[str, set[str]] = {}
        for et in ("traits", "roles"):
            for fp in (data_dir / et / "vectors").glob("*.pt"):
                if fp.stem != "default":
                    kinds_for_name.setdefault(fp.stem, set()).add(et)

    The migration is lossy by design for the collision names (stems with
    vectors in both kinds: nine in the May 2026 data; the corpus has had
    eleven since September 2026, parent and specialist joining);
    callers that need them must use a v2 cache for both sides.

    Returns the migrated dict (does not mutate the input).
    """
    from .entity_id import entity_id
    migrated: dict = {}
    for k, v in bare_scores.items():
        if is_entity_id(k):
            migrated[k] = v
            continue
        kinds = kinds_for_name.get(k)
        if kinds and len(kinds) == 1:
            only_kind = next(iter(kinds))
            migrated[entity_id(k, only_kind)] = v
        # else: collision (>=2 kinds) or orphan (0 kinds) → drop
    return migrated


def _read_envelope_or_bare(path: Path) -> tuple[Any, Optional[ProvenanceCheck]]:
    """Wrapper around :func:`load_validated_json` that hardens the
    file-not-found path with a more informative error.

    Static-mode loaders use this to read both the envelope-wrapped
    schema-v2 caches *and* fall through to bare JSON for any cache
    written before the envelope convention landed (those should fail
    at the schema-version check earlier, but if a hand-crafted
    legacy-equivalent v2 file lands without an envelope, this still
    works).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p}: file not found")
    return load_validated_json(p, policy="warn")


def _regenerate_via_for_static_scores(path: Path) -> str:
    """Construct a human-friendly regenerate command for a static-mode
    scores file.

    The path's stem (``scores_descriptions`` / ``scores_instructions``)
    determines the relevant ``--score_*`` flag; the parent directory
    is the ``--output_dir`` for the producer call.
    """
    p = Path(path)
    stem = p.stem
    if stem.startswith("scores_descriptions"):
        flag = "--score_descriptions"
    elif stem.startswith("scores_instructions"):
        flag = "--score_instructions"
    else:
        flag = "--score_descriptions [or --score_instructions]"
    return (
        f"uv run python results_analysis/axis_judge_correlation.py "
        f"{flag} --output_dir {p.parent} "
        f"--data_dir <DATA_DIR> --instructions_dir data --pair <POS> <NEG> "
        f"--pair_type <traits|roles> --provider <openai|anthropic> "
        f"--judge_model <MODEL>"
    )


def load_static_scores(
    path: Union[Path, str],
    *,
    dep_key: Optional[str] = None,
    extras: Optional[dict] = None,
    inputs: Optional[list[InputSpec]] = None,
    policy: str = "warn",
) -> tuple[dict[str, float], InputSpec, Optional[ProvenanceCheck]]:
    """Read a ``scores_descriptions.json`` / ``scores_instructions.json``
    cache from ``<axis>/<judge>/`` (a mixed-kind directory), enforcing
    the May-2026 ``schema_version: 2`` contract.

    Args:
        path: The cache JSON.
        dep_key: Writer-chosen short dep_key for the InputSpec the
            caller will record.  Defaults to
            ``"static_scores:<filename>"``.
        extras: Free-form discriminators on the InputSpec.
        inputs: If non-None, the freshly-built InputSpec is *appended*
            in place (matches the project's writer-side accumulation
            idiom).
        policy: Drift-handling policy forwarded to
            :func:`assistant_axis.provenance.load_validated_json` for
            the cache's *own* recorded inputs.

    Returns:
        ``(scores_by_eid, spec, check)`` where ``scores_by_eid`` is
        the disambiguated-key score map (e.g. ``{"patient|R": 1,
        "patient|T": -2, ...}``).  ``spec`` is the InputSpec for
        ``path``; ``check`` is the inner provenance check (or
        ``None`` if no envelope).

    Raises:
        StaleSchemaError: If ``path`` is v1 (no ``schema_version``) or
            mis-versioned, or if it claims v2 but has only bare-name
            keys (forgot-to-disambiguate sanity check).
        FileNotFoundError: If ``path`` doesn't exist.
    """
    p = Path(path)
    regenerate_via = _regenerate_via_for_static_scores(p)
    _assert_static_schema_v2(p, regenerate_via=regenerate_via)
    payload, check = load_validated_json(p, policy=policy)
    _assert_static_payload_disambiguated(
        p, payload, regenerate_via=regenerate_via,
    )
    spec = current_file_input(
        dep_key=dep_key or f"static_scores:{p.name}",
        path=p,
        extras=extras,
    )
    if inputs is not None:
        inputs.append(spec)
    if not isinstance(payload, dict):
        # Defensive: schema check should have caught this, but match
        # the declared return type.
        payload = {}
    return payload, spec, check


def _regenerate_via_for_projections(path: Path) -> str:
    p = Path(path)
    return (
        f"uv run python results_analysis/axis_judge_correlation.py "
        f"--score_descriptions --output_dir {p.parent} "
        f"--data_dir <DATA_DIR> --instructions_dir data --pair <POS> <NEG> "
        f"--pair_type <traits|roles>  "
        f"# (projections are recomputed as a side-effect of any scoring run)"
    )


def load_projections(
    path: Union[Path, str],
    *,
    dep_key: Optional[str] = None,
    extras: Optional[dict] = None,
    inputs: Optional[list[InputSpec]] = None,
    policy: str = "warn",
) -> tuple[dict[str, dict], InputSpec, Optional[ProvenanceCheck]]:
    """Read a ``projections.json`` cache from ``<axis>/<judge>/``
    (mixed-kind), enforcing the ``schema_version: 2`` contract.

    The returned dict is keyed by **slot index** (string, e.g.
    ``"0"``, ``"1"``, ...).  The inner per-slot dict is keyed by
    disambiguated entity_id (``"patient|R"`` / ``"patient|T"``) with
    values ``{"raw": float, "whitened": float}``.

    See :func:`load_static_scores` for argument semantics.
    """
    p = Path(path)
    regenerate_via = _regenerate_via_for_projections(p)
    _assert_static_schema_v2(p, regenerate_via=regenerate_via)
    payload, check = load_validated_json(p, policy=policy)
    # Sanity check on the **inner** dict (not the slot-index outer
    # one) — projections are mixed-kind even when the outer call only
    # touches one cohort.
    if isinstance(payload, dict):
        sample_inner = next(
            (v for v in payload.values() if isinstance(v, dict)),
            None,
        )
        if sample_inner is not None:
            _assert_static_payload_disambiguated(
                p, sample_inner, regenerate_via=regenerate_via,
            )
    spec = current_file_input(
        dep_key=dep_key or f"projections:{p.parent.name}",
        path=p,
        extras=extras,
    )
    if inputs is not None:
        inputs.append(spec)
    if not isinstance(payload, dict):
        payload = {}
    return payload, spec, check


def _regenerate_via_for_static_correlations(path: Path) -> str:
    p = Path(path)
    return (
        f"uv run python results_analysis/axis_judge_correlation.py "
        f"--all --output_dir {p.parent} "
        f"--data_dir <DATA_DIR> --instructions_dir data --pair <POS> <NEG> "
        f"--pair_type <traits|roles> --provider <openai|anthropic> "
        f"--judge_model <MODEL>  "
        f"# (correlations.json is regenerated as the final step of any --all run)"
    )


def load_static_correlations(
    path: Union[Path, str],
    *,
    dep_key: Optional[str] = None,
    extras: Optional[dict] = None,
    inputs: Optional[list[InputSpec]] = None,
    policy: str = "warn",
) -> tuple[dict, InputSpec, Optional[ProvenanceCheck]]:
    """Read a static-mode ``correlations.json`` from ``<axis>/<judge>/``
    (the mixed-kind one — NOT the kind-pure ones inside cohort
    directories).  Enforces the ``schema_version: 2`` contract; does
    NOT enforce key-shape on the payload (the structure is
    ``{mode: {slot: {raw|whitened: {rho, p, n, names, scores,
    projections}}}}`` and the disambiguation lives inside the inner
    ``names`` arrays, not at the dict-key level).

    See :func:`load_static_scores` for argument semantics.
    """
    p = Path(path)
    regenerate_via = _regenerate_via_for_static_correlations(p)
    _assert_static_schema_v2(p, regenerate_via=regenerate_via)
    payload, check = load_validated_json(p, policy=policy)
    spec = current_file_input(
        dep_key=dep_key or f"static_correlations:{p.parent.name}",
        path=p,
        extras=extras,
    )
    if inputs is not None:
        inputs.append(spec)
    if not isinstance(payload, dict):
        payload = {}
    return payload, spec, check
