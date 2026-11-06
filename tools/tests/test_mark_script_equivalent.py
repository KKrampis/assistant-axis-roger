"""Tests for tools/mark_script_equivalent.py.

Covers the CLI surface (add, --list, --check) plus the auto-detect
helper that scans envelope-bearing caches for the most recent
recorded fingerprint of a script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis import script_equivalence as eq  # noqa: E402
from tools import mark_script_equivalent as mse  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_envelope(
    script_rel: str,
    fingerprint: str,
    *,
    produced_at: str,
) -> dict:
    return {
        "result": {"foo": "bar"},
        "_provenance": {
            "schema_version": "1.0",
            "produced_at": produced_at,
            "produced_by": {"git_sha": "deadbeef"},
            "inputs": [
                {
                    "dep_key": "producer_script",
                    "kind": "file",
                    "path": script_rel,
                    "fingerprint": fingerprint,
                    "last_modified_at": produced_at,
                    "extras": {},
                    "dataset_id": None,
                    "member_paths": None,
                },
            ],
        },
    }


def _redirect_repo(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """Pretend the CLI's idea of repo root is ``repo``."""
    monkeypatch.setattr(mse, "_REPO_ROOT", repo)
    monkeypatch.setattr(eq, "_repo_root_default", lambda: repo)


# ---------------------------------------------------------------------------
# Auto-detect
# ---------------------------------------------------------------------------

def test_autodetect_returns_none_when_no_envelopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    _redirect_repo(monkeypatch, repo)
    assert mse._autodetect_from_fp("results_analysis/foo.py") is None


def test_autodetect_finds_most_recent_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "roger").mkdir()
    _redirect_repo(monkeypatch, repo)

    older = _make_envelope(
        "results_analysis/foo.py",
        "v1:2026-04-01T00:00:00+00:00@1000",
        produced_at="2026-04-01 00:10:00 +0000",
    )
    newer = _make_envelope(
        "results_analysis/foo.py",
        "v1:2026-05-01T00:00:00+00:00@1100",
        produced_at="2026-05-01 00:10:00 +0000",
    )
    (repo / "roger" / "old_cache.json").write_text(json.dumps(older))
    (repo / "roger" / "new_cache.json").write_text(json.dumps(newer))

    fp = mse._autodetect_from_fp("results_analysis/foo.py")
    assert fp == "v1:2026-05-01T00:00:00+00:00@1100"


def test_autodetect_ignores_unrelated_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "roger").mkdir()
    _redirect_repo(monkeypatch, repo)

    other = _make_envelope(
        "results_analysis/other.py", "v1:1@1",
        produced_at="2026-04-01 00:00:00 +0000",
    )
    (repo / "roger" / "other.json").write_text(json.dumps(other))
    assert mse._autodetect_from_fp("results_analysis/foo.py") is None


# ---------------------------------------------------------------------------
# CLI: add
# ---------------------------------------------------------------------------

def _make_repo_with_script(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "results_analysis").mkdir()
    script = repo / "results_analysis" / "foo.py"
    script.write_text("# v1\n")
    return repo, script


def test_cli_add_with_explicit_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo, script = _make_repo_with_script(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = mse.main([
        "--script", "results_analysis/foo.py",
        "--from-fp", "v1:OLD@100",
        "--to-fp", "v1:NEW@200",
        "--reason", "harmless",
    ])
    assert rc == 0
    reg = eq.load_registry(repo_root=repo)
    assert reg["results_analysis/foo.py"][0].from_fp == "v1:OLD@100"
    out = capsys.readouterr().out
    assert "Added equivalence" in out


def test_cli_add_autodetects_from_fp_from_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo, script = _make_repo_with_script(tmp_path)
    (repo / "roger").mkdir()
    _redirect_repo(monkeypatch, repo)

    env = _make_envelope(
        "results_analysis/foo.py", "v1:RECORDED@123",
        produced_at="2026-05-01 00:00:00 +0000",
    )
    (repo / "roger" / "cache.json").write_text(json.dumps(env))

    rc = mse.main([
        "--script", "results_analysis/foo.py",
        "--to-fp", "v1:CURRENT@456",
        "--reason", "auto-detected",
    ])
    assert rc == 0, capsys.readouterr().err
    reg = eq.load_registry(repo_root=repo)
    edge = reg["results_analysis/foo.py"][0]
    assert edge.from_fp == "v1:RECORDED@123"
    assert edge.to_fp == "v1:CURRENT@456"
    out = capsys.readouterr().out
    assert "auto-detected --from-fp" in out


def test_cli_add_errors_when_autodetect_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo, _ = _make_repo_with_script(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = mse.main([
        "--script", "results_analysis/foo.py",
        "--reason", "nope",
    ])
    assert rc != 0
    err = capsys.readouterr().err
    assert "could not auto-detect" in err.lower()


def test_cli_add_no_op_when_from_equals_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo, _ = _make_repo_with_script(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = mse.main([
        "--script", "results_analysis/foo.py",
        "--from-fp", "v1:SAME@1",
        "--to-fp", "v1:SAME@1",
        "--reason", "noop",
    ])
    assert rc == 0
    err = capsys.readouterr().err
    assert "nothing to record" in err.lower()
    reg = eq.load_registry(repo_root=repo)
    assert reg == {}


# ---------------------------------------------------------------------------
# CLI: --list
# ---------------------------------------------------------------------------

def test_cli_list_empty_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    _redirect_repo(monkeypatch, repo)
    rc = mse.main(["--list"])
    assert rc == 0
    assert "registry empty" in capsys.readouterr().out


def test_cli_list_filtered_by_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo, _ = _make_repo_with_script(tmp_path)
    _redirect_repo(monkeypatch, repo)
    eq.append_equivalence(script="results_analysis/foo.py", from_fp="f1",
                          to_fp="f2", reason="r1", repo_root=repo)
    eq.append_equivalence(script="results_analysis/bar.py", from_fp="b1",
                          to_fp="b2", reason="rB", repo_root=repo)
    rc = mse.main(["--list", "--script", "results_analysis/foo.py"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "results_analysis/foo.py" in out
    assert "results_analysis/bar.py" not in out


# ---------------------------------------------------------------------------
# CLI: --check
# ---------------------------------------------------------------------------

def test_cli_check_recognises_declared_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo, _ = _make_repo_with_script(tmp_path)
    _redirect_repo(monkeypatch, repo)
    eq.append_equivalence(script="results_analysis/foo.py", from_fp="f1",
                          to_fp="f2", reason="ok", repo_root=repo)
    rc = mse.main([
        "--check",
        "--script", "results_analysis/foo.py",
        "--from-fp", "f1",
        "--to-fp", "f2",
    ])
    assert rc == 0
    assert "EQUIVALENT" in capsys.readouterr().out


def test_cli_check_returns_nonzero_when_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo, _ = _make_repo_with_script(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = mse.main([
        "--check",
        "--script", "results_analysis/foo.py",
        "--from-fp", "f1",
        "--to-fp", "f2",
    ])
    assert rc == 1
    assert "NOT EQUIVALENT" in capsys.readouterr().out
