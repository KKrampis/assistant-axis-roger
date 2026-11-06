"""Tests for assistant_axis.atomic_io."""
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from assistant_axis import atomic_io


def test_atomic_write_text_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "stage"))
    dest = tmp_path / "out" / "x.txt"
    atomic_io.atomic_write_text("hello world", dest)
    assert dest.read_text() == "hello world"
    # Staging dir should be cleaned up (no .tmp leftovers)
    stage = tmp_path / "stage"
    if stage.exists():
        leftovers = [p for p in stage.iterdir() if p.name.startswith(".")]
        assert leftovers == []


def test_atomic_write_text_overwrite_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    dest = tmp_path / "out.txt"
    atomic_io.atomic_write_text("first", dest)
    atomic_io.atomic_write_text("second", dest)
    assert dest.read_text() == "second"


def test_atomic_write_bytes_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    dest = tmp_path / "blob.bin"
    payload = bytes(range(256))
    atomic_io.atomic_write_bytes(payload, dest)
    assert dest.read_bytes() == payload


def test_append_jsonl_creates_and_appends(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    dest = tmp_path / "records.jsonl"
    atomic_io.append_jsonl({"a": 1}, dest)
    atomic_io.append_jsonl({"a": 2, "b": "x"}, dest)
    lines = dest.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"a": 2, "b": "x"}


def test_write_jsonl_replaces(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    dest = tmp_path / "records.jsonl"
    atomic_io.write_jsonl([{"a": 1}, {"b": 2}], dest)
    atomic_io.write_jsonl([{"c": 3}], dest)
    lines = dest.read_text().splitlines()
    assert lines == ['{"c": 3}']


def test_retry_copy_succeeds_after_failures(tmp_path, monkeypatch):
    """The retry loop should swallow transient OSError and eventually succeed."""
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.write_text("payload")

    n_calls = {"n": 0}
    real_copy2 = __import__("shutil").copy2

    def flaky_copy2(s, d):
        n_calls["n"] += 1
        if n_calls["n"] < 3:
            raise OSError("simulated NFS hiccup")
        return real_copy2(s, d)

    with mock.patch("assistant_axis.atomic_io.shutil.copy2", side_effect=flaky_copy2):
        # short delay so the test is fast
        atomic_io._retry_copy(src, dest, attempts=5, base_delay_s=0.01)

    assert dest.read_text() == "payload"
    assert n_calls["n"] == 3


def test_retry_copy_gives_up_after_max_attempts(tmp_path):
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.write_text("payload")

    def always_fail(s, d):
        raise OSError("permanent failure")

    with mock.patch("assistant_axis.atomic_io.shutil.copy2", side_effect=always_fail):
        with pytest.raises(OSError, match="permanent failure"):
            atomic_io._retry_copy(src, dest, attempts=3, base_delay_s=0.01)
    # No half-written file left behind.
    assert not dest.exists()
    inprogress = dest.with_suffix(dest.suffix + ".inprogress")
    assert not inprogress.exists()


def test_atomic_write_uses_tmpdir_env(tmp_path, monkeypatch):
    """Verify TMPDIR is honoured for staging."""
    staging = tmp_path / "my_staging"
    staging.mkdir()
    monkeypatch.setenv("TMPDIR", str(staging))
    dest = tmp_path / "elsewhere" / "out.txt"

    # Patch the staging dir helper so we can observe what it returns.
    seen = {"path": None}
    real = atomic_io._staging_dir
    def spy():
        p = real()
        seen["path"] = p
        return p
    with mock.patch("assistant_axis.atomic_io._staging_dir", side_effect=spy):
        atomic_io.atomic_write_text("hi", dest)

    assert seen["path"] == staging
    assert dest.read_text() == "hi"


# ---------------------------------------------------------------------------
# Opt-in fsync and per-thread staging names (review 7 D6, walk-through E4)
# ---------------------------------------------------------------------------

class TestFsyncAndThreadStaging:
    WRITERS = {
        "text": lambda dest, **kw: atomic_io.atomic_write_text("t", dest, **kw),
        "bytes": lambda dest, **kw: atomic_io.atomic_write_bytes(b"b", dest, **kw),
        "append_jsonl": lambda dest, **kw: atomic_io.append_jsonl({"a": 1}, dest, **kw),
        "write_jsonl": lambda dest, **kw: atomic_io.write_jsonl([{"a": 1}], dest, **kw),
    }

    @pytest.mark.parametrize("writer", sorted(WRITERS))
    def test_fsync_is_off_by_default(self, tmp_path, monkeypatch, writer):
        monkeypatch.setenv("TMPDIR", str(tmp_path / "stage"))
        with mock.patch("assistant_axis.atomic_io.os.fsync") as fs:
            self.WRITERS[writer](tmp_path / "out.json")
        fs.assert_not_called()

    @pytest.mark.parametrize("writer", sorted(WRITERS))
    def test_fsync_flushes_the_copy_before_the_rename(
        self, tmp_path, monkeypatch, writer,
    ):
        monkeypatch.setenv("TMPDIR", str(tmp_path / "stage"))
        dest = tmp_path / "out.json"
        inprogress = dest.with_suffix(dest.suffix + ".inprogress")
        seen = []

        def spy(fd):
            # At the flush the copy sits beside dest, not yet renamed.
            seen.append((inprogress.exists(), dest.exists()))

        with mock.patch("assistant_axis.atomic_io.os.fsync", side_effect=spy):
            self.WRITERS[writer](dest, fsync=True)
        assert seen == [(True, False)]
        assert dest.exists() and not inprogress.exists()

    def test_a_failed_fsync_is_retried_like_a_failed_copy(self, tmp_path):
        src = tmp_path / "src"
        dest = tmp_path / "dest"
        src.write_text("payload")
        calls = {"n": 0}

        def flaky(fd):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("simulated fsync EIO")

        with mock.patch("assistant_axis.atomic_io.os.fsync", side_effect=flaky):
            atomic_io._retry_copy(src, dest, attempts=3, base_delay_s=0.01,
                                  fsync=True)
        assert dest.read_text() == "payload"
        assert calls["n"] == 2

    def test_staging_name_carries_pid_and_thread_id(self, tmp_path, monkeypatch):
        import threading
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        dest = Path("somewhere/config.json")
        here = atomic_io._staging_path(dest)
        there = {}
        t = threading.Thread(
            target=lambda: there.setdefault("p", atomic_io._staging_path(dest)))
        t.start()
        t.join()
        assert here != there["p"]
        assert f".{os.getpid()}.{threading.get_ident()}." in here.name
        assert atomic_io._staging_path(dest, ".pt.tmp").name.endswith(".pt.tmp")

    def test_two_threads_writing_the_same_name_keep_their_own_bytes(
        self, tmp_path, monkeypatch,
    ):
        """Both threads stage before either copies (a barrier forces the
        interleaving).  With a per-process staging name the second write
        overwrote the first's staging file, so both destinations got the
        same bytes (or the first cleanup removed the other's staging)."""
        import threading
        monkeypatch.setenv("TMPDIR", str(tmp_path / "stage"))
        barrier = threading.Barrier(2, timeout=10)
        real_copy = atomic_io._retry_copy

        def copy_after_both_staged(*a, **kw):
            barrier.wait()
            return real_copy(*a, **kw)

        errors = []

        def write(sub, text):
            try:
                atomic_io.atomic_write_text(text, tmp_path / sub / "config.json")
            except BaseException as e:  # noqa: BLE001 -- surfaced below
                errors.append(e)

        with mock.patch("assistant_axis.atomic_io._retry_copy",
                        side_effect=copy_after_both_staged):
            ts = [threading.Thread(target=write, args=(s, s * 3))
                  for s in ("a", "b")]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        assert errors == []
        assert (tmp_path / "a" / "config.json").read_text() == "aaa"
        assert (tmp_path / "b" / "config.json").read_text() == "bbb"


# ---------------------------------------------------------------------------
# Read retry helpers
# ---------------------------------------------------------------------------

class TestReadRetries:
    def test_read_text_round_trip(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("hello\nworld\n")
        assert atomic_io.read_text_with_retry(p) == "hello\nworld\n"

    def test_read_text_retries_then_succeeds(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("payload")
        n_calls = {"n": 0}
        real_read = Path.read_text

        def flaky_read(self_, encoding="utf-8"):
            n_calls["n"] += 1
            if n_calls["n"] < 3:
                raise OSError("simulated NFS hiccup")
            return real_read(self_, encoding=encoding)

        with mock.patch.object(Path, "read_text", flaky_read):
            result = atomic_io.read_text_with_retry(p, attempts=5, base_delay_s=0.01)
        assert result == "payload"
        assert n_calls["n"] == 3

    def test_read_text_gives_up_and_raises(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("payload")

        def always_fail(self_, encoding="utf-8"):
            raise OSError("permanent failure")

        with mock.patch.object(Path, "read_text", always_fail):
            with pytest.raises(OSError, match="permanent failure"):
                atomic_io.read_text_with_retry(p, attempts=3, base_delay_s=0.01)

    def test_a_missing_file_is_raised_at_once(self, tmp_path):
        """P8.4 (2026-10-12): FileNotFoundError is not an NFS hiccup; each
        missing file used to cost the full 265 s retry schedule."""
        import time as _time
        sleeps = []
        with mock.patch.object(_time, "sleep", lambda s: sleeps.append(s)):
            with pytest.raises(FileNotFoundError):
                atomic_io.read_text_with_retry(tmp_path / "absent.txt")
            with pytest.raises(FileNotFoundError):
                atomic_io.torch_load_with_retry(tmp_path / "absent.pt")
        assert sleeps == []
        assert FileNotFoundError in atomic_io.NON_TRANSIENT_READ_ERRORS

    def test_estale_is_still_retried(self, tmp_path):
        """A stale NFS handle is an OSError with errno ESTALE, not a
        FileNotFoundError: it stays transient."""
        import errno
        p = tmp_path / "x.txt"
        p.write_text("payload")
        n_calls = {"n": 0}
        real_read = Path.read_text

        def stale_then_ok(self_, encoding="utf-8"):
            n_calls["n"] += 1
            if n_calls["n"] < 2:
                raise OSError(errno.ESTALE, "Stale file handle")
            return real_read(self_, encoding=encoding)

        with mock.patch.object(Path, "read_text", stale_then_ok):
            assert atomic_io.read_text_with_retry(p, attempts=3, base_delay_s=0.01) == "payload"
        assert n_calls["n"] == 2
        assert not isinstance(OSError(errno.ESTALE, "x"), FileNotFoundError)

    def test_read_text_does_not_retry_unicode_decode_error(self, tmp_path):
        """Non-OSError exceptions (e.g. corrupt content) should propagate immediately."""
        p = tmp_path / "x.txt"
        p.write_bytes(b"\xff\xfe\xfd")  # invalid utf-8

        # Should raise UnicodeDecodeError (not OSError) and NOT retry.
        n_attempts = {"n": 0}
        real_read = Path.read_text
        def counting_read(self_, encoding="utf-8"):
            n_attempts["n"] += 1
            return real_read(self_, encoding=encoding)
        with mock.patch.object(Path, "read_text", counting_read):
            with pytest.raises(UnicodeDecodeError):
                atomic_io.read_text_with_retry(p, attempts=5, base_delay_s=0.01)
        assert n_attempts["n"] == 1  # no retries

    def test_read_jsonl_parses(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text('{"a": 1}\n{"b": 2}\n')
        assert atomic_io.read_jsonl_with_retry(p) == [{"a": 1}, {"b": 2}]

    def test_read_jsonl_skips_malformed_by_default(self, tmp_path, caplog):
        import logging as py_logging
        p = tmp_path / "x.jsonl"
        # Last line is half-written / garbage
        p.write_text('{"a": 1}\n{"b": 2}\n{not json\n')
        with caplog.at_level(py_logging.WARNING, logger=atomic_io.logger.name):
            result = atomic_io.read_jsonl_with_retry(p)
        assert result == [{"a": 1}, {"b": 2}]
        assert any("malformed JSON" in r.getMessage() for r in caplog.records)

    def test_read_jsonl_skip_malformed_false_raises(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text('{"a": 1}\nnot json\n')
        with pytest.raises(ValueError, match="malformed JSON"):
            atomic_io.read_jsonl_with_retry(p, skip_malformed=False)

    def test_read_jsonl_handles_empty_lines(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text('\n{"a": 1}\n\n{"b": 2}\n\n')
        assert atomic_io.read_jsonl_with_retry(p) == [{"a": 1}, {"b": 2}]

    def test_torch_load_with_retry_round_trip(self, tmp_path):
        import torch
        p = tmp_path / "x.pt"
        torch.save({"foo": torch.tensor([1, 2, 3])}, p)
        loaded = atomic_io.torch_load_with_retry(p, map_location="cpu")
        assert torch.equal(loaded["foo"], torch.tensor([1, 2, 3]))

    def test_torch_load_retries_then_succeeds(self, tmp_path):
        import torch
        p = tmp_path / "x.pt"
        torch.save([1, 2, 3], p)

        n_calls = {"n": 0}
        real_torch_load = torch.load

        def flaky_load(*args, **kwargs):
            n_calls["n"] += 1
            if n_calls["n"] < 3:
                raise OSError("simulated NFS hiccup")
            return real_torch_load(*args, **kwargs)

        with mock.patch("torch.load", flaky_load):
            result = atomic_io.torch_load_with_retry(p, attempts=5, base_delay_s=0.01)
        assert result == [1, 2, 3]
        assert n_calls["n"] == 3

    def test_torch_load_retries_runtime_error_short_read(self, tmp_path):
        """RuntimeError("storage has wrong byte size of dtype...") is the
        actual failure mode NFS short-reads produce in torch.load -- the
        retry helper must treat it as transient (legacy behaviour only
        retried OSError, which missed exactly this case).
        """
        import torch
        p = tmp_path / "x.pt"
        torch.save([1, 2, 3], p)

        n_calls = {"n": 0}
        real_torch_load = torch.load

        def flaky_load(*args, **kwargs):
            n_calls["n"] += 1
            if n_calls["n"] < 3:
                raise RuntimeError(
                    "storage has wrong byte size of dtype Float "
                    "(expected 1234567890, got 5)"
                )
            return real_torch_load(*args, **kwargs)

        with mock.patch("torch.load", flaky_load):
            result = atomic_io.torch_load_with_retry(
                p, attempts=5, base_delay_s=0.01,
            )
        assert result == [1, 2, 3]
        assert n_calls["n"] == 3

    def test_default_backoff_uses_project_standard(self):
        """Calling _retry_read with no explicit attempts/base_delay_s
        should fall through to the project-standard exponential backoff
        ``[5, 20, 60, 180]`` (matches axis_judge_correlation's
        RETRY_DELAYS).
        """
        # Inspect the exposed constant directly.
        assert atomic_io.DEFAULT_RETRY_DELAYS_S == (5.0, 20.0, 60.0, 180.0)
        # And RuntimeError / EOFError are in the default transient set.
        assert OSError in atomic_io.TRANSIENT_READ_ERRORS
        assert RuntimeError in atomic_io.TRANSIENT_READ_ERRORS
        assert EOFError in atomic_io.TRANSIENT_READ_ERRORS


# ---------------------------------------------------------------------------
# torch_save_with_retry
# ---------------------------------------------------------------------------

class TestTorchSaveWithRetry:
    def test_round_trip(self, tmp_path):
        import torch
        p = tmp_path / "out.pt"
        atomic_io.torch_save_with_retry({"x": torch.tensor([1, 2])}, p)
        assert p.exists()
        loaded = torch.load(p, weights_only=False)
        assert torch.equal(loaded["x"], torch.tensor([1, 2]))

    def test_retries_copy_then_succeeds(self, tmp_path, monkeypatch):
        """Save's staging-to-dest copy retries on transient OSError."""
        import torch

        # Force staging to land in tmp_path so we can observe both files.
        monkeypatch.setenv("TMPDIR", str(tmp_path / "stage"))
        (tmp_path / "stage").mkdir()

        p = tmp_path / "dest" / "out.pt"

        n_calls = {"n": 0}
        import shutil
        real_copy = shutil.copy2

        def flaky_copy(src, dst, *a, **kw):
            n_calls["n"] += 1
            if n_calls["n"] < 3:
                raise OSError("simulated NFS hiccup")
            return real_copy(src, dst, *a, **kw)

        with mock.patch("assistant_axis.atomic_io.shutil.copy2", flaky_copy):
            atomic_io.torch_save_with_retry(
                {"x": torch.tensor([1, 2])}, p,
                attempts=5,
            )
        assert p.exists()
        assert n_calls["n"] == 3

    def test_size_mismatch_after_copy_raises_and_unlinks(self, tmp_path):
        """Post-copy size check catches silent NFS truncation."""
        import torch
        p = tmp_path / "out.pt"

        import shutil
        real_copy = shutil.copy2

        def truncating_copy(src, dst, *a, **kw):
            # Copy partial bytes -- simulate a short write that didn't raise.
            real_copy(src, dst, *a, **kw)
            # Now truncate the destination after copy completes.
            with open(dst, "ab") as f:
                pass
            with open(dst, "r+b") as f:
                f.truncate(10)  # arbitrary smaller size

        with mock.patch("assistant_axis.atomic_io.shutil.copy2", truncating_copy):
            with pytest.raises(OSError, match="size mismatch"):
                atomic_io.torch_save_with_retry(
                    {"x": torch.tensor([1, 2, 3, 4, 5])}, p,
                    attempts=1,
                )
        # Partial dest should have been removed.
        assert not p.exists()

    def test_sha256_mismatch_triggers_retry_then_succeeds(self, tmp_path):
        """SHA-256 verification catches silent byte-flips inside a
        correct-size file (e.g. corrupted NFS chunks that pass length
        checks -- the failure mode behind the 2.6 GB-but-unloadable
        ``r_guardian__casual.pt``).
        """
        import torch
        import shutil
        p = tmp_path / "out.pt"

        n_calls = {"n": 0}
        real_copy = shutil.copy2

        def byte_flipping_copy(src, dst, *a, **kw):
            n_calls["n"] += 1
            real_copy(src, dst, *a, **kw)
            if n_calls["n"] < 3:
                # Flip one byte in the middle of the dest, preserving
                # length but changing content.  Byte position chosen
                # to land in pickle payload territory for any
                # non-trivial save (skip the small zip-file header).
                size = os.path.getsize(dst)
                with open(dst, "r+b") as f:
                    f.seek(size // 2)
                    orig = f.read(1)
                    f.seek(size // 2)
                    f.write(bytes([orig[0] ^ 0xFF]))

        with mock.patch("assistant_axis.atomic_io.shutil.copy2", byte_flipping_copy):
            atomic_io.torch_save_with_retry(
                {"x": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])}, p,
                attempts=5,
                verify_load=False,  # isolate the SHA check
            )
        assert p.exists()
        assert n_calls["n"] == 3
        # Final dest matches what we asked to save.
        loaded = torch.load(p, weights_only=False)
        assert torch.equal(loaded["x"], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_sha256_mismatch_exhausts_retries(self, tmp_path):
        """Persistent byte-flip surfaces as a final OSError after all
        retries exhaust, with the partial dest unlinked.
        """
        import torch
        import shutil
        p = tmp_path / "out.pt"

        real_copy = shutil.copy2

        def always_flipping_copy(src, dst, *a, **kw):
            real_copy(src, dst, *a, **kw)
            with open(dst, "r+b") as f:
                f.seek(os.path.getsize(dst) // 2)
                orig = f.read(1)
                f.seek(os.path.getsize(dst) // 2)
                f.write(bytes([orig[0] ^ 0xFF]))

        with mock.patch("assistant_axis.atomic_io.shutil.copy2", always_flipping_copy):
            with pytest.raises(OSError, match="sha256 mismatch"):
                atomic_io.torch_save_with_retry(
                    {"x": torch.tensor([1.0, 2.0, 3.0])}, p,
                    attempts=2,
                    verify_load=False,
                )
        assert not p.exists()

    def test_verify_load_catches_unloadable_dest(self, tmp_path):
        """When SHA matches but torch.load raises (e.g. version drift,
        pickle bug), verify_load triggers a retry.  Simulated by
        patching torch.load itself to raise on first calls.
        """
        import torch
        p = tmp_path / "out.pt"

        n_load_calls = {"n": 0}
        real_torch_load = torch.load

        def flaky_load(*args, **kwargs):
            n_load_calls["n"] += 1
            if n_load_calls["n"] < 3:
                raise RuntimeError(
                    "storage has wrong byte size (simulated)"
                )
            return real_torch_load(*args, **kwargs)

        with mock.patch("torch.load", flaky_load):
            atomic_io.torch_save_with_retry(
                {"x": torch.tensor([1, 2, 3])}, p,
                attempts=5,
                verify_sha256=False,  # isolate the load check
            )
        assert p.exists()
        assert n_load_calls["n"] == 3

    def test_dest_never_holds_unverified_bytes(self, tmp_path, monkeypatch):
        """Verification runs on the ``.inprogress`` copy and the rename comes
        last (since 2026-10-12): while the load check runs, ``dest`` does
        not exist yet, so a crash mid-verification cannot leave a file that
        skip-if-exists resume logic takes for a finished one."""
        import torch
        monkeypatch.setattr(atomic_io.time, "sleep", lambda s: None)
        p = tmp_path / "out.pt"
        seen = []
        real_torch_load = torch.load

        def spying_load(f, *args, **kwargs):
            seen.append((str(f), p.exists()))
            return real_torch_load(f, *args, **kwargs)

        with mock.patch("torch.load", spying_load):
            atomic_io.torch_save_with_retry({"x": torch.tensor([1, 2])}, p)
        assert seen == [(str(p) + ".inprogress", False)]
        assert p.exists()
        assert not Path(str(p) + ".inprogress").exists()

    def test_failed_overwrite_keeps_the_previous_file(self, tmp_path, monkeypatch):
        """A save that fails every attempt used to unlink ``dest`` -- for an
        ``--overwrite`` run, the previous good file."""
        import shutil
        import torch
        monkeypatch.setattr(atomic_io.time, "sleep", lambda s: None)
        p = tmp_path / "out.pt"
        torch.save({"x": torch.tensor([7.0])}, p)
        before = p.read_bytes()
        real_copy = shutil.copy2

        def always_flipping_copy(src, dst, *a, **kw):
            real_copy(src, dst, *a, **kw)
            with open(dst, "r+b") as f:
                f.seek(os.path.getsize(dst) // 2)
                orig = f.read(1)
                f.seek(os.path.getsize(dst) // 2)
                f.write(bytes([orig[0] ^ 0xFF]))

        with mock.patch("assistant_axis.atomic_io.shutil.copy2", always_flipping_copy):
            with pytest.raises(OSError, match="sha256 mismatch"):
                atomic_io.torch_save_with_retry({"x": torch.tensor([1.0, 2.0])}, p,
                                                attempts=2)
        assert p.read_bytes() == before
        assert not Path(str(p) + ".inprogress").exists()

    def test_unpickling_error_in_verification_is_retried_and_cleaned(
            self, tmp_path, monkeypatch):
        """torch.load raises pickle.UnpicklingError on some corrupt files;
        it escaped the retry and the cleanup before 2026-10-12."""
        import pickle
        import torch
        monkeypatch.setattr(atomic_io.time, "sleep", lambda s: None)
        p = tmp_path / "out.pt"
        n = {"calls": 0}
        real_torch_load = torch.load

        def flaky_load(*args, **kwargs):
            n["calls"] += 1
            if n["calls"] == 1:
                raise pickle.UnpicklingError("invalid load key, 'x'.")
            return real_torch_load(*args, **kwargs)

        with mock.patch("torch.load", flaky_load):
            atomic_io.torch_save_with_retry({"x": torch.tensor([1])}, p,
                                            attempts=3, verify_sha256=False)
        assert n["calls"] == 2
        assert p.exists()

        q = tmp_path / "never.pt"
        with mock.patch("torch.load", side_effect=pickle.UnpicklingError("bad")):
            with pytest.raises(pickle.UnpicklingError):
                atomic_io.torch_save_with_retry({"x": torch.tensor([1])}, q,
                                                attempts=1, verify_sha256=False)
        assert not q.exists()
        assert not Path(str(q) + ".inprogress").exists()

    def test_verify_off_skips_checks(self, tmp_path):
        """With both verifications disabled, no re-hash and no load
        happens after copy -- file is treated as written-correctly
        on first successful os.replace.  (Useful for tests / known-
        local writes where the hot-path cost isn't warranted.)
        """
        import torch
        p = tmp_path / "out.pt"

        # Patch torch.load so we can detect if verify_load runs.
        n_load_calls = {"n": 0}
        real_torch_load = torch.load

        def counting_load(*args, **kwargs):
            n_load_calls["n"] += 1
            return real_torch_load(*args, **kwargs)

        with mock.patch("torch.load", counting_load):
            atomic_io.torch_save_with_retry(
                {"x": torch.tensor([1, 2])}, p,
                attempts=1,
                verify_load=False,
                verify_sha256=False,
            )
        assert p.exists()
        # No verification load happened (the test's torch.load below
        # is a separate call that bumps the counter to 1).
        assert n_load_calls["n"] == 0
        loaded = real_torch_load(p, weights_only=False)
        assert torch.equal(loaded["x"], torch.tensor([1, 2]))


def test_read_jsonl_keeps_a_line_holding_a_raw_unicode_line_separator(tmp_path):
    """A JSON string may hold U+2028 / U+2029 / U+0085 raw; str.splitlines()
    would break the record in two malformed halves and drop it (2026-10-12)."""
    from assistant_axis.atomic_io import read_jsonl_with_retry
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": "x y"}\n{"b": "p\u0085q"}\r\n{"c": 1}\n', encoding="utf-8")
    assert read_jsonl_with_retry(path, skip_malformed=False) == [
        {"a": "x y"}, {"b": "p\u0085q"}, {"c": 1},
    ]
