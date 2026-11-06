"""Tests for tools/defer_rejudge.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis import deferral_registry as deferral  # noqa: E402
from tools import defer_rejudge as dr  # noqa: E402


def _redirect_repo(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    monkeypatch.setattr(dr, "_REPO_ROOT", repo)
    monkeypatch.setattr(deferral, "_repo_root_default", lambda: repo)


def _stub_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

def test_cli_add_creates_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = dr.main([
        "--path", "roger/scores.json",
        "--reason", "skip for now",
    ])
    assert rc == 0
    reg = deferral.load_registry(repo_root=repo)
    assert len(reg) == 1
    assert "Added deferral" in capsys.readouterr().out


def test_cli_add_with_dep_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = dr.main([
        "--path", "roger/*.json",
        "--dep-key", "judge_q9_*",
        "--reason", "scoped",
    ])
    assert rc == 0
    reg = deferral.load_registry(repo_root=repo)
    assert reg[0].dep_key == "judge_q9_*"


def test_cli_add_requires_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = dr.main(["--reason", "x"])
    assert rc != 0
    assert "--path is required" in capsys.readouterr().err


def test_cli_add_requires_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = dr.main(["--path", "x.json"])
    assert rc != 0
    assert "--reason is required" in capsys.readouterr().err


def test_cli_add_rejects_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc1 = dr.main(["--path", "x.json", "--reason", "r1"])
    rc2 = dr.main(["--path", "x.json", "--reason", "r2"])
    assert rc1 == 0 and rc2 != 0


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------

def test_cli_list_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = dr.main(["--list"])
    assert rc == 0
    assert "no deferrals" in capsys.readouterr().out


def test_cli_list_shows_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    deferral.append_deferral(path_glob="a.json", reason="r1", repo_root=repo)
    deferral.append_deferral(path_glob="b.json", dep_key="dk",
                              reason="r2", repo_root=repo)
    rc = dr.main(["--list"])
    out = capsys.readouterr().out
    assert "a.json" in out and "b.json" in out
    assert "dk" in out
    assert rc == 0


def test_cli_list_filtered_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    deferral.append_deferral(path_glob="a.json", reason="r1", repo_root=repo)
    deferral.append_deferral(path_glob="b.json", reason="r2", repo_root=repo)
    rc = dr.main(["--list", "--path", "a.json"])
    out = capsys.readouterr().out
    assert "a.json" in out and "b.json" not in out
    assert rc == 0


# ---------------------------------------------------------------------------
# --remove
# ---------------------------------------------------------------------------

def test_cli_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    deferral.append_deferral(path_glob="a.json", reason="r", repo_root=repo)
    rc = dr.main(["--remove", "--path", "a.json"])
    assert rc == 0
    assert deferral.load_registry(repo_root=repo) == []


def test_cli_remove_nonexistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = dr.main(["--remove", "--path", "missing.json"])
    assert rc != 0
    assert "No matching" in capsys.readouterr().err


def test_cli_remove_requires_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = _stub_repo(tmp_path)
    _redirect_repo(monkeypatch, repo)
    rc = dr.main(["--remove"])
    assert rc != 0
    assert "--path is required" in capsys.readouterr().err
