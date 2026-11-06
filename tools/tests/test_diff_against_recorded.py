"""Tests for tools/diff_against_recorded.py.

Covers:
* Envelope inspection + producer-script resolution.
* Cursor Local History fallback (with a synthetic History fixture).
* git fallback (with a tiny in-temp-dir git repo).
* The "suggested mark_script_equivalent.py command" output path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis import provenance as prov  # noqa: E402
from tools import diff_against_recorded as dar  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_history_fixture(
    repo_root: Path,
    script_abs: Path,
    snapshots: list[tuple[int, str]],
) -> Path:
    """Create a fake Cursor Local History tree under ``repo_root/history/``.

    ``snapshots`` is a list of ``(timestamp_ms, content)`` pairs.

    Returns the path to the synthetic ``History`` root.
    """
    history_root = repo_root / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    hist_dir = history_root / "fakedir"
    hist_dir.mkdir()
    entries = []
    for i, (ts_ms, content) in enumerate(snapshots):
        snap_id = f"snap_{i:03d}.py"
        (hist_dir / snap_id).write_text(content, encoding="utf-8")
        entries.append({"id": snap_id, "timestamp": ts_ms})
    (hist_dir / "entries.json").write_text(json.dumps({
        "version": 1,
        "resource": f"file://{script_abs.resolve()}",
        "entries": entries,
    }), encoding="utf-8")
    return history_root


def _write_envelope_for_script(
    cache_path: Path, script_path: Path, *,
    git_sha: str = "deadbeef",
) -> None:
    """Build a minimal envelope around a payload, recording ``script_path``
    as the producer_script kind="file" input."""
    spec = prov.current_file_input(dep_key="producer_script", path=script_path)
    cache_path.write_text(json.dumps({
        "result": {"x": 1},
        "_provenance": {
            "schema_version": "1.0",
            "produced_by": {"git_sha": git_sha, "cmd": "uv run python ..."},
            "produced_at": "2026-05-08 00:00:00 +0000",
            "inputs": prov.inputs_to_jsonable([spec]),
        },
    }), encoding="utf-8")


def _redirect_repo(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    monkeypatch.setattr(dar, "_REPO_ROOT", repo)
    # Also redirect provenance's repo-root resolver so InputSpecs
    # built inside the test see the synthetic repo and store
    # repo-relative paths cleanly.
    monkeypatch.setattr(prov, "_repo_root", lambda: repo)


# ---------------------------------------------------------------------------
# _read_envelope / _find_producer_input
# ---------------------------------------------------------------------------

def test_read_envelope_rejects_non_envelope(tmp_path: Path) -> None:
    p = tmp_path / "plain.json"; p.write_text(json.dumps({"foo": 1}))
    with pytest.raises(SystemExit, match="no ``_provenance``"):
        dar._read_envelope(p)


def test_find_producer_input_picks_named_dep(tmp_path: Path) -> None:
    env = {"inputs": [
        {"kind": "multi", "dep_key": "data", "path": "data/", "fingerprint": "v1"},
        {"kind": "file", "dep_key": "producer_script",
         "path": "results_analysis/foo.py", "fingerprint": "v1:1@1"},
    ]}
    found = dar._find_producer_input(env)
    assert found["dep_key"] == "producer_script"


def test_find_producer_input_falls_back_to_first_file(tmp_path: Path) -> None:
    env = {"inputs": [
        {"kind": "multi", "dep_key": "data", "path": "data/", "fingerprint": "v1"},
        {"kind": "file", "dep_key": "axis_file",
         "path": "axes/x.pt", "fingerprint": "v1:1@1"},
    ]}
    found = dar._find_producer_input(env)
    assert found["dep_key"] == "axis_file"


def test_find_producer_input_raises_when_none(tmp_path: Path) -> None:
    env = {"inputs": [
        {"kind": "multi", "dep_key": "data", "path": "data/", "fingerprint": "v1"},
    ]}
    with pytest.raises(SystemExit, match="kind=\"file\""):
        dar._find_producer_input(env)


# ---------------------------------------------------------------------------
# Cursor Local History fallback
# ---------------------------------------------------------------------------

def test_history_fallback_finds_snapshot(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; repo.mkdir(); (repo / ".git").mkdir()
    script = repo / "results_analysis"; script.mkdir()
    script_path = script / "foo.py"
    script_path.write_text("# v3 (current)\n", encoding="utf-8")

    history_root = _make_history_fixture(
        repo, script_path,
        snapshots=[
            (1_700_000_000_000, "# v1\n"),
            (1_700_000_500_000, "# v2\n"),
            (1_700_000_900_000, "# v3 (current)\n"),
        ],
    )
    # Recorded last_modified_at maps to roughly the v2 timestamp.
    target_iso = "2023-11-14T22:21:40+00:00"  # ~1_700_000_500
    snap = dar._read_local_history(
        script_path, last_modified_at=target_iso, history_root=history_root,
    )
    assert snap is not None
    text, entry, hist_dir = snap
    assert text == "# v2\n"
    assert entry["timestamp"] == 1_700_000_500_000
    assert hist_dir.parent == history_root


def test_history_fallback_missing_dir_returns_none(tmp_path: Path) -> None:
    script = tmp_path / "x.py"; script.write_text("# x\n")
    history_root = tmp_path / "empty_history"; history_root.mkdir()
    result = dar._read_local_history(
        script, last_modified_at=None, history_root=history_root,
    )
    assert result is None


def test_history_fallback_picks_most_recent_when_no_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"; repo.mkdir()
    script_path = repo / "foo.py"; script_path.write_text("cur\n")
    history_root = _make_history_fixture(
        repo, script_path,
        snapshots=[
            (1_000_000_000_000, "old\n"),
            (1_500_000_000_000, "newest_snapshot\n"),
        ],
    )
    snap = dar._read_local_history(
        script_path, last_modified_at=None, history_root=history_root,
    )
    assert snap is not None
    text, entry, _ = snap
    assert text == "newest_snapshot\n"
    assert entry["timestamp"] == 1_500_000_000_000


# ---------------------------------------------------------------------------
# git fallback (mocked via subprocess)
# ---------------------------------------------------------------------------

def _mock_git_show(monkeypatch, *, sha_to_text: dict[str, str]) -> None:
    def fake_check_output(cmd, **kwargs):
        # cmd is like ["git", "show", "<sha>:<rel>"]
        if cmd[:2] == ["git", "show"]:
            target = cmd[2]
            sha = target.split(":", 1)[0]
            if sha in sha_to_text:
                return sha_to_text[sha]
            raise subprocess.CalledProcessError(1, cmd)
        raise subprocess.CalledProcessError(127, cmd)
    monkeypatch.setattr(subprocess, "check_output", fake_check_output)


def test_git_show_returns_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_git_show(monkeypatch, sha_to_text={"abc123": "# v1 from git\n"})
    out = dar._git_show("abc123", "results_analysis/foo.py", tmp_path)
    assert out == "# v1 from git\n"


def test_git_show_strips_dirty_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_git_show(monkeypatch, sha_to_text={"abc123": "from clean sha\n"})
    out = dar._git_show("abc123+dirty", "x.py", tmp_path)
    assert out == "from clean sha\n"


def test_git_show_returns_none_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_git_show(monkeypatch, sha_to_text={})
    assert dar._git_show("missing_sha", "x.py", tmp_path) is None


# ---------------------------------------------------------------------------
# End-to-end CLI: prefers git, falls back to History
# ---------------------------------------------------------------------------

def test_cli_uses_git_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import time
    repo = tmp_path / "repo"; repo.mkdir(); (repo / ".git").mkdir()
    script = repo / "results_analysis" / "foo.py"
    script.parent.mkdir()
    script.write_text("# v1\n", encoding="utf-8")

    _redirect_repo(monkeypatch, repo)

    cache = repo / "cache.json"
    _write_envelope_for_script(cache, script, git_sha="abc123")

    # Mutate the script so the current fingerprint differs from the
    # recorded one (otherwise the "suggested mark_script_equivalent"
    # block is intentionally skipped -- nothing to declare).
    time.sleep(1.1)
    script.write_text("# current version\n# extra line\n", encoding="utf-8")

    _mock_git_show(monkeypatch, sha_to_text={
        "abc123": "# original version\n",
    })

    rc = dar.main([str(cache)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Diff source: git show abc123" in out
    assert "+# current version" in out
    assert "-# original version" in out
    # Suggested mark_script_equivalent command appears (since current
    # fingerprint differs from recorded).
    assert "mark_script_equivalent.py" in out


def test_cli_falls_back_to_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = tmp_path / "repo"; repo.mkdir(); (repo / ".git").mkdir()
    script = repo / "foo.py"
    script.write_text("# current version\n", encoding="utf-8")

    # Build envelope first (recorded last_modified_at, fingerprint) BEFORE we
    # change the file contents.  Then mutate the file so the diff has
    # something to show.
    cache = repo / "cache.json"
    _write_envelope_for_script(cache, script, git_sha="missing_sha")
    rec_lmt = json.loads(cache.read_text())["_provenance"]["inputs"][0]["last_modified_at"]

    # Mutate current.
    script.write_text("# current version\n# new line\n", encoding="utf-8")

    history_root = _make_history_fixture(
        repo, script,
        snapshots=[(1_700_000_000_000, "# original version\n")],
    )

    # Mock git as failing for the recorded sha.
    _mock_git_show(monkeypatch, sha_to_text={})
    _redirect_repo(monkeypatch, repo)

    rc = dar.main([str(cache), "--history-root", str(history_root)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Cursor Local History" in out
    assert "+# new line" in out


def test_cli_no_diff_available_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    repo = tmp_path / "repo"; repo.mkdir(); (repo / ".git").mkdir()
    script = repo / "foo.py"; script.write_text("# x\n")
    cache = repo / "cache.json"
    _write_envelope_for_script(cache, script, git_sha="missing_sha")
    _mock_git_show(monkeypatch, sha_to_text={})
    _redirect_repo(monkeypatch, repo)

    rc = dar.main([
        str(cache),
        "--history-root", str(tmp_path / "no_history_here"),
    ])
    out = capsys.readouterr().out
    assert rc == 1
    assert "no diff available" in out
