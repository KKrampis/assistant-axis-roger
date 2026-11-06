"""Tests for assistant_axis.tmpfs's mirror-marker fast-path.

The high-level ``setup_model_tmpfs_cache`` call needs ``/dev/shm``, an
NFS-backed source HF cache, and rsync, so it isn't unit-testable on
laptops -- it's exercised end-to-end on RunPod.  These tests cover the
two pure-Python helpers added for the fast-path: ``_check_existing_mirror``
and ``_write_mirror_marker``.  They're enough to catch regressions in
the marker-format / drift-tolerance logic without booting a GPU box.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from assistant_axis import tmpfs


# ---------------------------------------------------------------------------
# _write_mirror_marker
# ---------------------------------------------------------------------------

class TestWriteMirrorMarker:
    def test_writes_expected_fields(self, tmp_path: Path):
        # Populate the mirror dir with some bytes so _du_kb returns a
        # nonzero size that the marker can record.
        (tmp_path / "blob.bin").write_bytes(b"x" * 4096)
        tmpfs._write_mirror_marker(tmp_path, source_size_kb=8)
        marker = json.loads(
            (tmp_path / tmpfs.MIRROR_MARKER_FILENAME).read_text()
        )
        assert marker["marker_version"] == 1
        assert marker["model_subdir"] == tmp_path.name
        assert marker["source_size_kb"] == 8
        assert isinstance(marker["mirror_size_kb"], int)
        assert marker["mirror_size_kb"] > 0
        # completed_at is roughly now (within 60s)
        assert abs(marker["completed_at"] - time.time()) < 60

    def test_swallows_oserror(self, tmp_path: Path):
        # Marker dir doesn't exist -> open() raises -> we should not
        # propagate (best-effort write).  Caller would otherwise have
        # no way to recover.
        nonexistent = tmp_path / "does_not_exist"
        # Should NOT raise, just log a warning.
        tmpfs._write_mirror_marker(nonexistent, source_size_kb=1)


# ---------------------------------------------------------------------------
# _check_existing_mirror
# ---------------------------------------------------------------------------

class TestCheckExistingMirror:
    def test_returns_none_when_marker_missing(self, tmp_path: Path):
        assert tmpfs._check_existing_mirror(tmp_path) is None

    def test_returns_none_when_marker_unreadable(self, tmp_path: Path):
        (tmp_path / tmpfs.MIRROR_MARKER_FILENAME).write_text("not-json{")
        assert tmpfs._check_existing_mirror(tmp_path) is None

    def test_returns_none_when_size_field_missing(self, tmp_path: Path):
        (tmp_path / tmpfs.MIRROR_MARKER_FILENAME).write_text(
            json.dumps({"marker_version": 1, "model_subdir": "x"})
        )
        assert tmpfs._check_existing_mirror(tmp_path) is None

    def test_returns_marker_when_size_matches(self, tmp_path: Path):
        # Real fixture: write some bytes, measure with _du_kb, write a
        # marker recording that size, then read it back.  Round-trip
        # exercises the full happy path.
        (tmp_path / "blob.bin").write_bytes(b"x" * 32_000)
        tmpfs._write_mirror_marker(tmp_path, source_size_kb=32)
        marker = tmpfs._check_existing_mirror(tmp_path)
        assert marker is not None
        assert marker["source_size_kb"] == 32
        assert marker["mirror_size_kb"] >= 32

    def test_tolerates_growth(self, tmp_path: Path):
        # Lay down 32 KB of payload, write the marker, then add another
        # 32 KB.  Growth doesn't invalidate the mirror -- the model files
        # we care about are still present, the extra is just noise (e.g.
        # the marker file itself, or another model coexisting in the
        # cache).  Should still return the marker.
        (tmp_path / "blob.bin").write_bytes(b"x" * 32_000)
        tmpfs._write_mirror_marker(tmp_path, source_size_kb=32)
        (tmp_path / "extra.bin").write_bytes(b"y" * 32_000)
        assert tmpfs._check_existing_mirror(tmp_path) is not None

    def test_returns_none_when_files_deleted(self, tmp_path: Path):
        # Lay down 200 KB across two blobs, mark, then delete one blob
        # so tmpfs is now ~100 KB while marker still says 200 KB.
        # Shortfall > tolerance -> mirror is broken, should re-sync.
        (tmp_path / "a.bin").write_bytes(b"a" * 100_000)
        (tmp_path / "b.bin").write_bytes(b"b" * 100_000)
        tmpfs._write_mirror_marker(tmp_path, source_size_kb=200)
        (tmp_path / "b.bin").unlink()
        assert tmpfs._check_existing_mirror(tmp_path) is None

    def test_tolerates_marker_file_overhead(self, tmp_path: Path):
        # The marker file itself contributes ~4 KB on disk (one block).
        # On a real 60 GB model that's a 0.000007% overhead, well below
        # MIRROR_SIZE_TOLERANCE.  Verified here by re-writing the marker
        # without changing the tree contents -- the second write should
        # update the marker in place, and a subsequent check should still
        # succeed because actual >= recorded (no shortfall).
        (tmp_path / "blob.bin").write_bytes(b"x" * 32_000)
        tmpfs._write_mirror_marker(tmp_path, source_size_kb=32)
        tmpfs._write_mirror_marker(tmp_path, source_size_kb=32)
        assert tmpfs._check_existing_mirror(tmp_path) is not None
