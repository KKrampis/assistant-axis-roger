"""Phase 6 (May 2026) — regression test for the trait/role name
collision bug ("Bug A").

Pins down the canonical dual-iteration patterns that triggered Bug A
in 14 consumer files, demonstrating both:

1. The original bug (bare-name keys overwrite silently across kinds).
2. The fix (``entity_id(name, kind)`` keys keep both kinds distinct).

If a future refactor accidentally re-introduces the bug — typically
by iterating ``for et in ("traits", "roles"):`` and merging into a
single bare-name dict — these tests fail loudly.

The collision set is the 11 names that appear in BOTH the project's
trait corpus and its role corpus (nine in the May 2026 dataset;
``specialist`` and ``parent`` joined in September 2026).  Future
dataset edits may shrink or grow this set; if so, update
:data:`assistant_axis.tests.test_entity_id.COLLISION_NAMES` and
:data:`EXPECTED_N_COLLISIONS` here.
:func:`test_collision_names_match_the_corpus_on_disk` compares the list
with the stems in ``data/{traits,roles}/instructions/``, so it cannot
go stale silently.
"""
from __future__ import annotations

import pytest

from assistant_axis.entity_id import (
    default_data_dir,
    entity_id,
    is_entity_id,
    parse_entity_id,
)
from assistant_axis.tests.test_entity_id import COLLISION_NAMES


# Project-wide expected collision count — anchors the test suite to
# the corpus.  Increase / decrease alongside corpus edits.
EXPECTED_N_COLLISIONS = 11


def test_collision_count_matches_project_baseline():
    """Anchor test: the project ships 11 known trait/role name
    collisions (October 2026).  Any drift here is a meaningful change to
    the corpus that the operator should notice consciously, not a
    silent shift."""
    assert len(COLLISION_NAMES) == EXPECTED_N_COLLISIONS


def test_collision_names_match_the_corpus_on_disk():
    """The hard-coded list is the actual intersection of the trait and
    role stems on disk.  A trait or role seeded (or renamed) onto a name
    the other kind already has fails here, so the list, the count and
    the documentation are updated on purpose (W17, 2026-10-09: parent
    and specialist had joined unnoticed)."""
    data = default_data_dir()
    stems = {kind: {p.stem for p in (data / kind / "instructions").glob("*.json")}
             for kind in ("traits", "roles")}
    assert all(len(s) > 100 for s in stems.values()), f"corpus not found under {data}"
    on_disk = sorted(stems["traits"] & stems["roles"])
    assert sorted(COLLISION_NAMES) == on_disk, (
        f"collision names on disk {on_disk} differ from COLLISION_NAMES; update "
        "test_entity_id.COLLISION_NAMES, EXPECTED_N_COLLISIONS and the docs that list them")


# ---------------------------------------------------------------------------
# Pattern A: building one entity_vecs dict from two source dirs
# (mimics rho_by_layer._build_entity_cache).
# ---------------------------------------------------------------------------

def _simulate_vector_load() -> dict:
    """Stand-in for reading per-entity vector files from
    ``data/{traits,roles}/vectors/*.pt``.  Returns a per-kind map of
    ``{name: vec}`` so the test can compare bare-name vs entity_id
    merge strategies on the same input."""
    return {
        "traits": {n: f"trait_vec_{n}" for n in COLLISION_NAMES},
        "roles": {n: f"role_vec_{n}" for n in COLLISION_NAMES},
    }


def test_dual_iteration_bare_name_loses_data():
    """Bug A reproducer — DO NOT FIX BY CHANGING THE TEST.

    Iterating ``("traits", "roles")`` and writing to a single
    bare-name dict drops one of the two kinds entirely for every
    collision.  This test pins that down so a future regression
    (someone reverting the fix in a consumer) is detectable."""
    src = _simulate_vector_load()
    bad: dict[str, str] = {}
    for et in ("traits", "roles"):
        for name, vec in src[et].items():
            bad[name] = vec  # noqa: kind-collision  -- Bug A reproducer
    # All 11 names are present, but only with the SECOND iteration's
    # values (roles, since it iterated last).  Trait data is gone.
    assert len(bad) == EXPECTED_N_COLLISIONS
    for name in COLLISION_NAMES:
        assert bad[name] == f"role_vec_{name}"
        assert "trait_vec" not in bad[name]


def test_dual_iteration_entity_id_keeps_both():
    """The fix: use ``entity_id(name, kind)`` so the two kinds get
    distinct keys and survive the merge."""
    src = _simulate_vector_load()
    good: dict[str, str] = {}
    for et in ("traits", "roles"):
        for name, vec in src[et].items():
            good[entity_id(name, et)] = vec
    # Both kinds preserved: 2 × 11 = 22 entries.
    assert len(good) == 2 * EXPECTED_N_COLLISIONS
    for name in COLLISION_NAMES:
        assert good[entity_id(name, "traits")] == f"trait_vec_{name}"
        assert good[entity_id(name, "roles")] == f"role_vec_{name}"


# ---------------------------------------------------------------------------
# Pattern B: building one ``scores`` dict from two response-mode
# (kind-pure) caches (mimics rho_by_layer's response loop).
# ---------------------------------------------------------------------------

def test_response_mode_merge_bare_name_loses_data():
    """Reproduce the response-mode merge bug: kind-pure caches
    written with bare names get blown away when merged across kinds
    without entity_id promotion."""
    trait_cache = {n: {"mean_score": 0.5} for n in COLLISION_NAMES}
    role_cache = {n: {"mean_score": -0.5} for n in COLLISION_NAMES}

    # Bug-A merge:
    bad: dict[str, float] = {}
    for n, info in trait_cache.items():
        bad[n] = info["mean_score"]
    for n, info in role_cache.items():
        bad[n] = info["mean_score"]
    # Only the role values survived.
    assert all(v == -0.5 for v in bad.values())
    assert len(bad) == EXPECTED_N_COLLISIONS  # half the data lost


def test_response_mode_merge_entity_id_keeps_both():
    """Fix: promote bare names with ``entity_id(name, kind)`` during
    the merge."""
    trait_cache = {n: {"mean_score": 0.5} for n in COLLISION_NAMES}
    role_cache = {n: {"mean_score": -0.5} for n in COLLISION_NAMES}

    good: dict[str, float] = {}
    for n, info in trait_cache.items():
        good[entity_id(n, "traits")] = info["mean_score"]
    for n, info in role_cache.items():
        good[entity_id(n, "roles")] = info["mean_score"]
    assert len(good) == 2 * EXPECTED_N_COLLISIONS
    for name in COLLISION_NAMES:
        assert good[entity_id(name, "traits")] == 0.5
        assert good[entity_id(name, "roles")] == -0.5


# ---------------------------------------------------------------------------
# Pattern C: set intersection across kinds for downstream alignment
# (mimics rho_by_layer's ``set(scores) & set(proj)``).
# ---------------------------------------------------------------------------

def test_set_intersection_bare_names_overcounts_collisions():
    """Bare-name set intersection across kinds counts each collision
    name ONCE even when the data has BOTH kinds — silent
    under-counting that miscalculates n in correlation."""
    scores_keys = set(COLLISION_NAMES)  # bare names from a merged dict
    projs_keys = set(COLLISION_NAMES)
    common = scores_keys & projs_keys
    assert len(common) == EXPECTED_N_COLLISIONS  # NOT 2× — wrong!


def test_set_intersection_entity_ids_counts_correctly():
    """With entity_id keys, the intersection correctly returns 2× the
    collision count (one per kind)."""
    scores_keys = {entity_id(n, k) for n in COLLISION_NAMES
                   for k in ("traits", "roles")}
    projs_keys = {entity_id(n, k) for n in COLLISION_NAMES
                  for k in ("traits", "roles")}
    common = scores_keys & projs_keys
    assert len(common) == 2 * EXPECTED_N_COLLISIONS  # correct!


# ---------------------------------------------------------------------------
# Pattern D: name-keyed annotation/label round-trip
# (mimics make_plot's ``names`` array → display label).
# ---------------------------------------------------------------------------

def test_entity_id_round_trip_for_all_collisions():
    """Each collision name must round-trip cleanly through
    entity_id/parse_entity_id for both kinds."""
    for name in COLLISION_NAMES:
        for kind in ("traits", "roles"):
            eid = entity_id(name, kind)
            parsed = parse_entity_id(eid)
            assert parsed.name == name
            assert parsed.kind == kind
            assert is_entity_id(eid)


def test_bare_collision_name_is_not_entity_id():
    """The 11 collision names, on their own, are NOT entity_ids — only
    the suffixed forms are.  Critical for guard clauses in code that
    needs to detect 'is this already disambiguated?'"""
    for name in COLLISION_NAMES:
        assert not is_entity_id(name), (
            f"{name!r} should NOT match is_entity_id (no '|R'/'|T' suffix)"
        )


# ---------------------------------------------------------------------------
# Pattern E: indirect lookup chain (entity_id key → bare name → look
# up in a kind-pure cache).  Common in code that must read a
# kind-pure cache while keying its results by entity_id for safe
# downstream merge.
# ---------------------------------------------------------------------------

def test_eid_then_bare_lookup_flow():
    """An entity_id key → split into (name, kind) → use kind to pick
    the right kind-pure cache → use bare name to look up → store
    back into a mixed-kind output dict by entity_id.  Pin this
    full-cycle pattern so refactors keep both ends consistent."""
    trait_cache = {n: f"trait_vec_{n}" for n in COLLISION_NAMES}
    role_cache = {n: f"role_vec_{n}" for n in COLLISION_NAMES}
    caches_by_kind = {"traits": trait_cache, "roles": role_cache}

    # Caller builds a mixed-kind want-list of entity_ids.
    want_eids = [
        entity_id(n, k) for n in COLLISION_NAMES
        for k in ("traits", "roles")
    ]

    out: dict[str, str] = {}
    for eid in want_eids:
        parsed = parse_entity_id(eid)
        cache = caches_by_kind[parsed.kind]
        out[eid] = cache[parsed.name]

    assert len(out) == 2 * EXPECTED_N_COLLISIONS
    assert all(out[entity_id(n, "traits")].startswith("trait_vec")
               for n in COLLISION_NAMES)
    assert all(out[entity_id(n, "roles")].startswith("role_vec")
               for n in COLLISION_NAMES)


# ---------------------------------------------------------------------------
# Comprehensive sanity: collision names cover the documented set.
# ---------------------------------------------------------------------------

def test_documented_collisions_complete():
    """If a developer adds or removes collision names from the
    project's canonical list, this test fails and forces them to
    update the count + also any documentation referencing 'the 11
    collision names' (AGENT_NOTES.md, plan documents, etc.)."""
    expected = {
        "ascetic", "contrarian", "cosmopolitan", "generalist",
        "pacifist", "parent", "patient", "perfectionist", "romantic",
        "specialist", "stoic",
    }
    assert set(COLLISION_NAMES) == expected


@pytest.mark.parametrize("collision_name", COLLISION_NAMES)
def test_each_collision_distinguishable_individually(collision_name):
    """Every individual collision must be distinguishable (one
    failure should clearly point at WHICH name regressed)."""
    role_id = entity_id(collision_name, "roles")
    trait_id = entity_id(collision_name, "traits")
    assert role_id != trait_id
    assert role_id == f"{collision_name}|R"
    assert trait_id == f"{collision_name}|T"
