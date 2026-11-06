"""Tests for assistant_axis.deferral_registry (Phase 6d-1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from assistant_axis import deferral_registry as deferral


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _stub_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


def _write_registry(repo: Path, entries: list[dict]) -> None:
    (repo / deferral.DEFERRAL_FILENAME).write_text(yaml.safe_dump({
        "schema_version": deferral.SCHEMA_VERSION,
        "deferrals": entries,
    }, sort_keys=False))


# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------

def test_load_registry_missing_returns_empty(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    assert deferral.load_registry(repo_root=repo) == []


def test_load_registry_parses(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    _write_registry(repo, [
        {"path_glob": "roger/*/scores_*.json",
         "dep_key": None,
         "reason": "skip rejudges for now",
         "deferred_at": "2026-05-08T04:30:00+00:00"},
    ])
    reg = deferral.load_registry(repo_root=repo)
    assert len(reg) == 1
    e = reg[0]
    assert e.path_glob == "roger/*/scores_*.json"
    assert e.dep_key is None
    assert "skip" in e.reason


def test_load_registry_rejects_unknown_schema(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    (repo / deferral.DEFERRAL_FILENAME).write_text(yaml.safe_dump({
        "schema_version": 999, "deferrals": [],
    }))
    with pytest.raises(ValueError, match="schema_version"):
        deferral.load_registry(repo_root=repo)


# ---------------------------------------------------------------------------
# match_cache
# ---------------------------------------------------------------------------

def test_match_cache_path_glob_only(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    cache = repo / "roger" / "exp" / "scores_responses.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{}")
    _write_registry(repo, [
        {"path_glob": "roger/exp/scores_*.json", "dep_key": None,
         "reason": "ok", "deferred_at": "2026-05-08"},
    ])
    matches = deferral.match_cache(cache, repo_root=repo)
    assert len(matches) == 1


def test_match_cache_no_match(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    cache = repo / "roger" / "scores.json"
    cache.parent.mkdir()
    cache.write_text("{}")
    _write_registry(repo, [
        {"path_glob": "elsewhere/*.json", "dep_key": None,
         "reason": "x", "deferred_at": "2026-05-08"},
    ])
    assert deferral.match_cache(cache, repo_root=repo) == []


def test_match_cache_dep_key_filter(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    cache = repo / "roger" / "scores.json"
    cache.parent.mkdir()
    cache.write_text("{}")
    _write_registry(repo, [
        {"path_glob": "roger/*.json",
         "dep_key": "judge_*_descriptions_sonnet",
         "reason": "scoped", "deferred_at": "2026-05-08"},
    ])
    # Matching dep_key:
    m = deferral.match_cache(cache, dep_key="judge_q9_descriptions_sonnet",
                             repo_root=repo)
    assert len(m) == 1
    # Non-matching dep_key:
    m2 = deferral.match_cache(cache, dep_key="judge_q9_responses_gpt",
                              repo_root=repo)
    assert m2 == []
    # No dep_key passed: registry entry's dep_key is set, so we treat
    # the call as "any dep_key" (caller doesn't know which one drifted)
    # -> match.
    m3 = deferral.match_cache(cache, dep_key=None, repo_root=repo)
    assert len(m3) == 1


def test_match_cache_uses_explicit_registry(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    cache = repo / "x.json"; cache.write_text("{}")
    inline_reg = [deferral.DeferralEntry(
        path_glob="*.json", dep_key=None, reason="inline",
        deferred_at="2026-05-08",
    )]
    matches = deferral.match_cache(cache, registry=inline_reg, repo_root=repo)
    assert len(matches) == 1
    assert matches[0].reason == "inline"


# ---------------------------------------------------------------------------
# append_deferral / remove_deferral
# ---------------------------------------------------------------------------

def test_append_deferral_creates_and_appends(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    deferral.append_deferral(
        path_glob="roger/scores.json", reason="r1", repo_root=repo,
    )
    deferral.append_deferral(
        path_glob="roger/other.json", dep_key="judge_x", reason="r2",
        repo_root=repo,
    )
    reg = deferral.load_registry(repo_root=repo)
    assert len(reg) == 2
    assert reg[0].dep_key is None
    assert reg[1].dep_key == "judge_x"


def test_append_deferral_rejects_duplicate(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    deferral.append_deferral(path_glob="r.json", reason="r1", repo_root=repo)
    with pytest.raises(ValueError, match="Duplicate"):
        deferral.append_deferral(path_glob="r.json", reason="r2", repo_root=repo)


def test_remove_deferral(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    deferral.append_deferral(path_glob="a.json", reason="r1", repo_root=repo)
    deferral.append_deferral(path_glob="b.json", reason="r2", repo_root=repo)
    removed = deferral.remove_deferral(path_glob="a.json", repo_root=repo)
    assert removed is True
    reg = deferral.load_registry(repo_root=repo)
    assert len(reg) == 1
    assert reg[0].path_glob == "b.json"


def test_remove_deferral_nonexistent(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    deferral.append_deferral(path_glob="a.json", reason="r1", repo_root=repo)
    removed = deferral.remove_deferral(path_glob="missing.json", repo_root=repo)
    assert removed is False


# ---------------------------------------------------------------------------
# Schema v2: DeferralCategory + optional metadata fields
# ---------------------------------------------------------------------------

def test_v1_schema_loads_with_uncategorized(tmp_path: Path) -> None:
    """Schema v1 files (no ``category`` field) must still load; existing
    entries default to ``UNCATEGORIZED`` so the codebase can read
    pre-migration registries during the v1->v2 transition."""
    repo = _stub_repo(tmp_path)
    (repo / deferral.DEFERRAL_FILENAME).write_text(yaml.safe_dump({
        "schema_version": 1,
        "deferrals": [
            {"path_glob": "roger/x.json", "dep_key": None,
             "reason": "v1 entry", "deferred_at": "2026-05-08"},
        ],
    }))
    reg = deferral.load_registry(repo_root=repo)
    assert len(reg) == 1
    assert reg[0].category is deferral.DeferralCategory.UNCATEGORIZED


def test_v2_round_trip_with_category(tmp_path: Path) -> None:
    """Append + reload preserves category and optional metadata."""
    repo = _stub_repo(tmp_path)
    deferral.append_deferral(
        path_glob="roger/snap_v1.json",
        reason="frozen for v1-vs-v2 compare",
        category=deferral.DeferralCategory.FROZEN_SNAPSHOT,
        compares_to="roger/snap.json",
        repo_root=repo,
    )
    deferral.append_deferral(
        path_glob="roger/orphan_*.png",
        reason="one-off pre-Phase-6 producer",
        category=deferral.DeferralCategory.ORPHAN_NO_PRODUCER,
        producer_script="/tmp/_canonical_angles_v1.py",
        repo_root=repo,
    )
    reg = deferral.load_registry(repo_root=repo)
    assert {e.category for e in reg} == {
        deferral.DeferralCategory.FROZEN_SNAPSHOT,
        deferral.DeferralCategory.ORPHAN_NO_PRODUCER,
    }
    snap = next(e for e in reg
                if e.category is deferral.DeferralCategory.FROZEN_SNAPSHOT)
    assert snap.compares_to == "roger/snap.json"
    assert snap.producer_script is None
    orph = next(e for e in reg
                if e.category is deferral.DeferralCategory.ORPHAN_NO_PRODUCER)
    assert orph.producer_script == "/tmp/_canonical_angles_v1.py"
    assert orph.compares_to is None


def test_unknown_category_falls_back_to_uncategorized(tmp_path: Path) -> None:
    """Forward-compat: an unknown ``category`` string (e.g. one added
    by a future schema bump) must not crash existing readers; the
    entry loads with ``UNCATEGORIZED`` so newer YAMLs remain partially
    legible to older code."""
    repo = _stub_repo(tmp_path)
    (repo / deferral.DEFERRAL_FILENAME).write_text(yaml.safe_dump({
        "schema_version": deferral.SCHEMA_VERSION,
        "deferrals": [
            {"path_glob": "roger/x.json", "dep_key": None,
             "category": "future_category_we_dont_know",
             "reason": "from a newer schema", "deferred_at": "2099-01-01"},
        ],
    }))
    reg = deferral.load_registry(repo_root=repo)
    assert len(reg) == 1
    assert reg[0].category is deferral.DeferralCategory.UNCATEGORIZED


def test_writer_upgrades_v1_to_v2_in_place(tmp_path: Path) -> None:
    """First write to a v1 file must bump ``schema_version`` to v2 so
    the registry doesn't get stuck advertising v1 once any new
    category-bearing entry is appended."""
    repo = _stub_repo(tmp_path)
    (repo / deferral.DEFERRAL_FILENAME).write_text(yaml.safe_dump({
        "schema_version": 1,
        "deferrals": [
            {"path_glob": "old.json", "dep_key": None,
             "reason": "v1 entry", "deferred_at": "2026-05-08"},
        ],
    }))
    deferral.append_deferral(
        path_glob="new.json", reason="v2 entry",
        category=deferral.DeferralCategory.LEGACY_BARE,
        repo_root=repo,
    )
    on_disk = yaml.safe_load(
        (repo / deferral.DEFERRAL_FILENAME).read_text()
    )
    assert on_disk["schema_version"] == deferral.SCHEMA_VERSION
    assert on_disk["schema_version"] == 2
    cats = {e.get("category") for e in on_disk["deferrals"]}
    assert "legacy_bare" in cats


def test_serialise_omits_optional_fields_when_unset(tmp_path: Path) -> None:
    """Optional metadata fields (producer_script / replaced_by /
    compares_to) must NOT appear in the YAML when not provided, so
    the on-disk format stays terse for the common case."""
    repo = _stub_repo(tmp_path)
    deferral.append_deferral(
        path_glob="legacy.json", reason="bare",
        category=deferral.DeferralCategory.LEGACY_BARE,
        repo_root=repo,
    )
    on_disk = yaml.safe_load(
        (repo / deferral.DEFERRAL_FILENAME).read_text()
    )
    entry = on_disk["deferrals"][0]
    assert "producer_script" not in entry
    assert "replaced_by" not in entry
    assert "compares_to" not in entry
    assert entry["category"] == "legacy_bare"
