"""Tests for assistant_axis.script_equivalence and its integration with
``validate_recorded`` (Phase 6b-1)."""

from __future__ import annotations

import datetime as _dt
import os
import time
from pathlib import Path
from typing import Iterable

import pytest
import yaml

from assistant_axis import provenance as prov
from assistant_axis import script_equivalence as eq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_repo(tmp_path: Path) -> Path:
    """Create a fake repo (with a ``.git`` marker) and patch the
    equivalence module's repo-root resolver to point at it.

    Returns the path to the fake repo root.
    """
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


def _write_registry(repo: Path, equivalences: Iterable[dict]) -> None:
    (repo / eq.REGISTRY_FILENAME).write_text(yaml.safe_dump({
        "schema_version": eq.SCHEMA_VERSION,
        "equivalences": list(equivalences),
    }, sort_keys=False))


# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------

def test_load_registry_missing_returns_empty(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    assert eq.load_registry(repo_root=repo) == {}


def test_load_registry_parses_well_formed_yaml(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    _write_registry(repo, [
        {"script": "a.py", "from_fp": "fp1", "to_fp": "fp2",
         "reason": "tweak", "marked_at": "2026-05-08T00:00:00+00:00"},
    ])
    reg = eq.load_registry(repo_root=repo)
    assert "a.py" in reg
    assert len(reg["a.py"]) == 1
    e = reg["a.py"][0]
    assert (e.from_fp, e.to_fp, e.reason) == ("fp1", "fp2", "tweak")


def test_load_registry_rejects_unknown_schema_version(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    (repo / eq.REGISTRY_FILENAME).write_text(yaml.safe_dump({
        "schema_version": 999, "equivalences": [],
    }))
    with pytest.raises(ValueError, match="schema_version"):
        eq.load_registry(repo_root=repo)


# ---------------------------------------------------------------------------
# is_equivalent: BFS / transitive lookup
# ---------------------------------------------------------------------------

def test_is_equivalent_reflexive() -> None:
    # Same fingerprint always equivalent (no registry lookup needed).
    assert eq.is_equivalent("anything.py", "fp", "fp", registry={}) is True


def test_is_equivalent_one_hop() -> None:
    reg = {"a.py": [
        eq.EquivalenceEdge("a.py", "fp1", "fp2", "edit1", "2026-05-01"),
    ]}
    assert eq.is_equivalent("a.py", "fp1", "fp2", registry=reg) is True
    # Reverse direction not implied.
    assert eq.is_equivalent("a.py", "fp2", "fp1", registry=reg) is False


def test_is_equivalent_transitive_two_hops() -> None:
    reg = {"a.py": [
        eq.EquivalenceEdge("a.py", "fp1", "fp2", "r1", "t1"),
        eq.EquivalenceEdge("a.py", "fp2", "fp3", "r2", "t2"),
    ]}
    assert eq.is_equivalent("a.py", "fp1", "fp3", registry=reg) is True


def test_is_equivalent_fails_when_chain_breaks() -> None:
    reg = {"a.py": [
        eq.EquivalenceEdge("a.py", "fp1", "fp2", "r1", "t1"),
        # gap between fp2 and fp3
        eq.EquivalenceEdge("a.py", "fp3", "fp4", "r2", "t2"),
    ]}
    assert eq.is_equivalent("a.py", "fp1", "fp4", registry=reg) is False


def test_is_equivalent_per_script_isolation() -> None:
    reg = {
        "a.py": [eq.EquivalenceEdge("a.py", "fp1", "fp2", "r", "t")],
        "b.py": [eq.EquivalenceEdge("b.py", "fp3", "fp4", "r", "t")],
    }
    assert eq.is_equivalent("a.py", "fp1", "fp2", registry=reg) is True
    assert eq.is_equivalent("b.py", "fp1", "fp2", registry=reg) is False


def test_is_equivalent_return_path() -> None:
    reg = {"a.py": [
        eq.EquivalenceEdge("a.py", "fp1", "fp2", "r1", "t1"),
        eq.EquivalenceEdge("a.py", "fp2", "fp3", "r2", "t2"),
    ]}
    found, path = eq.is_equivalent("a.py", "fp1", "fp3",
                                    registry=reg, return_path=True)
    assert found is True
    assert [e.reason for e in path] == ["r1", "r2"]


def test_is_equivalent_handles_cycles() -> None:
    # A cycle in declared edges shouldn't hang the BFS.
    reg = {"a.py": [
        eq.EquivalenceEdge("a.py", "fp1", "fp2", "r", "t"),
        eq.EquivalenceEdge("a.py", "fp2", "fp1", "r", "t"),
    ]}
    assert eq.is_equivalent("a.py", "fp1", "fp3", registry=reg) is False


# ---------------------------------------------------------------------------
# append_equivalence
# ---------------------------------------------------------------------------

def test_append_equivalence_creates_file(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    e = eq.append_equivalence(
        script="a.py", from_fp="fp1", to_fp="fp2",
        reason="tweak", repo_root=repo,
    )
    assert (repo / eq.REGISTRY_FILENAME).exists()
    reg = eq.load_registry(repo_root=repo)
    assert reg["a.py"][0].from_fp == "fp1"
    assert e.marked_at  # ISO-formatted, non-empty


def test_append_equivalence_appends_to_existing(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    eq.append_equivalence(script="a.py", from_fp="f1", to_fp="f2",
                          reason="r1", repo_root=repo)
    eq.append_equivalence(script="a.py", from_fp="f2", to_fp="f3",
                          reason="r2", repo_root=repo)
    reg = eq.load_registry(repo_root=repo)
    assert len(reg["a.py"]) == 2


def test_append_equivalence_rejects_duplicates(tmp_path: Path) -> None:
    repo = _stub_repo(tmp_path)
    eq.append_equivalence(script="a.py", from_fp="f1", to_fp="f2",
                          reason="r1", repo_root=repo)
    with pytest.raises(ValueError, match="Duplicate"):
        eq.append_equivalence(script="a.py", from_fp="f1", to_fp="f2",
                              reason="r1-resaid", repo_root=repo)


# ---------------------------------------------------------------------------
# Integration with validate_recorded
# ---------------------------------------------------------------------------

def _make_kind_file_spec(script_path: Path, dep_key: str = "producer_script"):
    return prov.current_file_input(dep_key=dep_key, path=script_path)


def test_validate_recorded_drift_downgraded_to_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a script's fingerprint drifts but the registry has a matching
    edge, validate_recorded should report ``equivalent`` instead of
    ``drift`` and ``ProvenanceCheck.ok`` should remain True."""
    repo = _stub_repo(tmp_path)
    # Pretend the repo root is ``repo`` so script_equivalence finds the
    # YAML and provenance treats the script's path as repo-relative.
    monkeypatch.setattr(prov, "_repo_root", lambda: repo)
    monkeypatch.setattr(eq, "_repo_root_default", lambda: repo)

    script = repo / "fake_producer.py"
    script.write_text("# v1\n")
    spec_v1 = _make_kind_file_spec(script)

    # Wait a beat so the next write changes the mtime/size fingerprint.
    time.sleep(1.05)
    script.write_text("# v1 plus a comment\n")
    spec_v2 = _make_kind_file_spec(script)
    assert spec_v1.fingerprint != spec_v2.fingerprint

    # Without a registry entry: validate_recorded should report drift.
    check_no_reg = prov.validate_recorded([spec_v1])
    assert check_no_reg.statuses[0].status == "drift"
    assert check_no_reg.ok is False

    # Declare the edit equivalent: drift downgrades to equivalent.
    eq.append_equivalence(
        script="fake_producer.py",
        from_fp=spec_v1.fingerprint,
        to_fp=spec_v2.fingerprint,
        reason="harmless docstring tweak",
        repo_root=repo,
    )
    check_with_reg = prov.validate_recorded([spec_v1])
    assert check_with_reg.statuses[0].status == "equivalent"
    assert check_with_reg.ok is True
    # Detail should include the registry justification.
    assert "declared equivalent" in check_with_reg.statuses[0].detail


def test_validate_recorded_does_not_apply_equivalence_to_multi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``equivalent`` only applies to kind="file"; kind="multi" drifts
    stay as drifts even if a misnamed equivalence edge exists."""
    repo = _stub_repo(tmp_path)
    monkeypatch.setattr(prov, "_repo_root", lambda: repo)
    monkeypatch.setattr(eq, "_repo_root_default", lambda: repo)

    a = repo / "a.txt"; a.write_text("aa")
    spec_v1 = prov.current_files_input("dataset", [a])
    time.sleep(1.05)
    a.write_text("aaXX")
    spec_v2 = prov.current_files_input("dataset", [a])

    eq.append_equivalence(
        script="(some path)",
        from_fp=spec_v1.fingerprint,
        to_fp=spec_v2.fingerprint,
        reason="not applicable to multi",
        repo_root=repo,
    )
    check = prov.validate_recorded([spec_v1])
    assert check.statuses[0].status == "drift"


def test_provenance_check_equivalent_accessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ProvenanceCheck.equivalent()`` should expose the new status."""
    repo = _stub_repo(tmp_path)
    monkeypatch.setattr(prov, "_repo_root", lambda: repo)
    monkeypatch.setattr(eq, "_repo_root_default", lambda: repo)
    script = repo / "p.py"; script.write_text("v1")
    s1 = _make_kind_file_spec(script)
    time.sleep(1.05)
    script.write_text("v1 plus")
    s2 = _make_kind_file_spec(script)
    eq.append_equivalence(script="p.py", from_fp=s1.fingerprint,
                          to_fp=s2.fingerprint, reason="harmless",
                          repo_root=repo)
    check = prov.validate_recorded([s1])
    eqs = check.equivalent()
    assert len(eqs) == 1
    assert eqs[0].dep_key == "producer_script"
