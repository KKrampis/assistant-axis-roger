"""Unit tests for the HF-cache tmpfs check in pipeline/2_activations.py.

The check protects against a real failure mode: launching 2_activations.py
directly (bypassing run_pipeline.sh's setup_tmpfs) caused a 4-hour silent
hang loading Qwen3-32B from an NFS-backed HF cache.  These tests lock in
the mount-point matching logic so the warning fires when it should.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_2_activations_module():
    """Import pipeline/2_activations.py despite its leading-digit filename.

    Python identifiers can't start with a digit so a normal `import` fails;
    we go through importlib instead.  Module is loaded fresh per test
    invocation but that's fine -- the module body is small.
    """
    spec = importlib.util.spec_from_file_location(
        "_pipeline_two_activations",
        REPO_ROOT / "pipeline" / "2_activations.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_2_activations_module()


# ---------------------------------------------------------------------------
# _is_path_on_tmpfs
# ---------------------------------------------------------------------------

def _write_mounts(tmp_path: Path, lines: list[str]) -> Path:
    f = tmp_path / "mounts"
    f.write_text("\n".join(lines) + "\n")
    return f


def test_is_path_on_tmpfs_returns_none_when_mounts_missing(mod, tmp_path):
    # Simulates macOS / containers without /proc/mounts.
    missing = tmp_path / "no_such_file"
    assert mod._is_path_on_tmpfs(tmp_path, mounts_file=missing) is None


def test_is_path_on_tmpfs_true_when_path_under_tmpfs_mount(mod, tmp_path):
    # /tmp is tmpfs, /home is ext4.  A subdir under /tmp should match tmpfs.
    mounts = _write_mounts(tmp_path, [
        "tmpfs /tmp tmpfs rw,nosuid 0 0",
        "/dev/sda1 /home ext4 rw 0 0",
        "/dev/sda2 / ext4 rw 0 0",
    ])
    # Use a directory that actually exists so the resolve loop finds it;
    # the exact path doesn't matter for the test since we override mounts.
    # Use tmp_path itself as the probe; declare /<tmp_path> as tmpfs.
    mounts2 = _write_mounts(tmp_path, [
        f"tmpfs {tmp_path} tmpfs rw,nosuid 0 0",
        "/dev/sda2 / ext4 rw 0 0",
    ])
    assert mod._is_path_on_tmpfs(tmp_path, mounts_file=mounts2) is True


def test_is_path_on_tmpfs_false_when_path_under_disk_mount(mod, tmp_path):
    mounts = _write_mounts(tmp_path, [
        f"/dev/sda1 {tmp_path} ext4 rw 0 0",
        "/dev/sda2 / ext4 rw 0 0",
        "tmpfs /dev/shm tmpfs rw,nosuid 0 0",
    ])
    assert mod._is_path_on_tmpfs(tmp_path, mounts_file=mounts) is False


def test_is_path_on_tmpfs_picks_longest_matching_prefix(mod, tmp_path):
    # Path falls under both `/` (ext4) and a deeper tmpfs mount; the deeper
    # mount must win.  This is the failure mode for the naive "any match"
    # implementation.
    sub = tmp_path / "deep_sub"
    sub.mkdir()
    mounts = _write_mounts(tmp_path, [
        "/dev/sda2 / ext4 rw 0 0",
        f"tmpfs {sub} tmpfs rw 0 0",
    ])
    assert mod._is_path_on_tmpfs(sub, mounts_file=mounts) is True


def test_is_path_on_tmpfs_root_only_disk_mount(mod, tmp_path):
    # When only `/` is mounted (no specific entry for our path) and it's
    # ext4, we should report False -- nothing tmpfs-y about the resolved
    # path.
    mounts = _write_mounts(tmp_path, [
        "/dev/sda2 / ext4 rw 0 0",
    ])
    assert mod._is_path_on_tmpfs(tmp_path, mounts_file=mounts) is False


def test_is_path_on_tmpfs_handles_nonexistent_path_gracefully(mod, tmp_path):
    mounts = _write_mounts(tmp_path, [
        "tmpfs / tmpfs rw 0 0",
    ])
    # Path doesn't exist; we walk up until we find an existing ancestor.
    # Since `/` is tmpfs in this synthetic mount table, the answer is True.
    assert mod._is_path_on_tmpfs(
        tmp_path / "does" / "not" / "exist",
        mounts_file=mounts,
    ) is True


# ---------------------------------------------------------------------------
# _warn_if_hf_cache_not_tmpfs end-to-end (logging side effects)
# ---------------------------------------------------------------------------

def test_warn_fires_loudly_when_hf_cache_not_tmpfs(mod, tmp_path, caplog, monkeypatch):
    # Simulate Linux with HF_HOME on an ext4 mount.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    mounts = _write_mounts(tmp_path, [
        f"/dev/sda1 {tmp_path} ext4 rw 0 0",
        "/dev/sda2 / ext4 rw 0 0",
    ])
    # Patch the default mounts_file in the helper to point at our synthetic
    # one for this test.  Easiest way: monkey-patch _is_path_on_tmpfs to
    # return False directly.
    monkeypatch.setattr(mod, "_is_path_on_tmpfs", lambda p: False)

    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        mod._warn_if_hf_cache_not_tmpfs("Qwen/Qwen3-32B")

    # The warning block should mention the model name and the fix path.
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Qwen/Qwen3-32B" in text
    assert "tmpfs" in text
    assert "run_pipeline.sh" in text


def test_warn_silent_when_hf_cache_is_tmpfs(mod, tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(mod, "_is_path_on_tmpfs", lambda p: True)

    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        mod._warn_if_hf_cache_not_tmpfs("Qwen/Qwen3-32B")

    # No WARNING-level records when tmpfs detected.
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warns == []


def test_warn_silent_when_indeterminate(mod, tmp_path, caplog, monkeypatch):
    # The macOS / non-Linux case: don't spam warnings on dev machines.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(mod, "_is_path_on_tmpfs", lambda p: None)

    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        mod._warn_if_hf_cache_not_tmpfs("Qwen/Qwen3-32B")

    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warns == []
