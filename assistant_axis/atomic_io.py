"""NFS-safe file I/O helpers.

On RunPod and similar setups where /workspace is a network-mounted volume
(MooseFS or NFS), direct file ops can be slow and occasionally flaky:
partial copies, transient EIO, slow flushes, half-written files visible
to readers.  The pattern that has proven robust across
pipeline/2_activations.py and the steering driver is:

    Writes:
      1. Serialise to a local-disk temp file (TMPDIR or /tmp).  Local
         writes are fast and don't depend on the network filesystem.
      2. Copy the temp file to the NFS destination with retry/backoff:
         hiccups usually resolve within a few seconds.
      3. Atomically rename into place (os.replace), so partial writes
         never leave the destination in a half-written state.
      4. Unlink the temp.

    Reads:
      Direct read inside a 5-attempt retry loop with exponential
      backoff.  On final failure: log error and raise (the caller's
      outer loop is responsible for "continue to next item" semantics).

Write entry points:

    atomic_write_text(text, dest)        small JSON / config / log content
    atomic_write_bytes(buf, dest)        binary blobs
    append_jsonl(record, dest)           append one JSONL line
    write_jsonl(records, dest)           rewrite full JSONL

Read entry points:

    read_text_with_retry(path)           plain text
    read_jsonl_with_retry(path)          parsed list of dicts
    torch_load_with_retry(path, ...)     torch.load with the same retry shape

Retry schedules:
  * **Reads** (read_text / read_jsonl / torch_load) default to the
    project-standard exponential backoff
    ``DEFAULT_RETRY_DELAYS_S = (5, 20, 60, 180)`` seconds between
    attempts (5 attempts total, cumulative wall-clock ~265 s).  This
    matches the API-retry standard used in
    ``results_analysis/axis_judge_correlation.py`` and is the right
    schedule when transient ``RuntimeError`` short-reads from
    ``torch.load`` need a long-enough window for chunkserver-side
    recovery.
  * **Text and byte writes** (atomic_write_* / append_jsonl /
    write_jsonl) default to a shorter linear backoff
    ``10 * (attempt + 1)`` seconds (5 attempts, cumulative ~100 s).
    These only ever hit ``OSError`` on copy failures (``shutil.copy2``
    doesn't raise the torch-load ``RuntimeError`` flavours), so a
    tighter retry window is fine and keeps long-running flush hot
    paths from blocking on a wedged network for >4 minutes per failure.
  * **Tensor writes** (torch_save_with_retry) use the read schedule
    :data:`DEFAULT_RETRY_DELAYS_S`, since each attempt ends in a
    ``torch.load`` verification that can hit the same short-read
    ``RuntimeError`` as any read.

Both shapes are configurable per call via explicit ``attempts`` and
``base_delay_s`` / ``delays_s`` keywords.  Add new entry points here
rather than re-rolling the pattern at each call site.

Durability (review 7 D6, walk-through E4, 2026-10-12): the small-write
entry points (``atomic_write_text`` / ``atomic_write_bytes`` /
``append_jsonl`` / ``write_jsonl``) take ``fsync=True`` to flush the
``.inprogress`` copy to stable storage before it is renamed into place, so
a power loss cannot leave an empty file under the final name.  It is off by
default until it has been timed on the pod's NFS (pod TODO item 4).  The
staging file's name carries the process id *and* the thread id, so two
threads of one process writing files of the same name (in different
directories) never share a staging file.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Type, Union

logger = logging.getLogger(__name__)


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of ``path``, read in 1 MiB chunks.

    Used by :func:`torch_save_with_retry` to verify destination bytes
    match the staging file byte-for-byte.  Streams the file so the
    memory cost is constant (one chunk) regardless of file size.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# Project-standard exponential backoff for transient filesystem errors.
# Matches RETRY_DELAYS in results_analysis/axis_judge_correlation.py.
# Five attempts total (1 + 4 retries), cumulative wall-clock ~265 s.
DEFAULT_RETRY_DELAYS_S: Tuple[float, ...] = (5.0, 20.0, 60.0, 180.0)


# Exception classes treated as transient for read retries.  We
# explicitly include RuntimeError here because torch.load surfaces
# NFS / MooseFS short-reads as RuntimeError("storage has wrong byte
# size of dtype ...") or RuntimeError("PytorchStreamReader failed
# reading zip archive: ... unexpected EOF, expected N more bytes"),
# *not* OSError -- the older policy of retrying only on OSError missed
# exactly the failure mode that motivated this audit.  EOFError covers
# raw pickle stream truncation.
TRANSIENT_READ_ERRORS: Tuple[Type[BaseException], ...] = (
    OSError, RuntimeError, EOFError,
)

# ...except these, which are raised at once although they are OSErrors.  A
# file that does not exist is not an NFS hiccup: retrying it cost 265 s per
# missing file (13 files, ~57 min, in step 4 on 2026-05-03; review P4 D5,
# walk-through P8.4, 2026-10-12).  ESTALE (a stale NFS handle) is an OSError
# with errno ESTALE, not a FileNotFoundError (errno ENOENT), so it stays
# transient.
NON_TRANSIENT_READ_ERRORS: Tuple[Type[BaseException], ...] = (
    FileNotFoundError,
)


def _staging_dir() -> Path:
    """TMPDIR (if set) or /tmp.

    On RunPod we typically export TMPDIR=/dev/shm to keep the staging area
    on a RAM-backed tmpfs -- avoids filling the small container /tmp and
    keeps the local-disk write fast.  Honoured automatically.
    """
    return Path(os.environ.get("TMPDIR", "/tmp"))


def _staging_path(dest_path: Path, suffix: str = ".tmp") -> Path:
    """The staging file for ``dest_path``, unique per process *and* thread.

    The name used to carry the process id only, so two threads writing
    files of the same name (``config.json`` in two directories) shared one
    staging file and could publish each other's bytes (review 7 D6,
    walk-through E4).
    """
    return _staging_dir() / (
        f".{dest_path.name}.{os.getpid()}.{threading.get_ident()}{suffix}"
    )


def _fsync_path(path: Path) -> None:
    """Flush ``path``'s data to stable storage (raises OSError on failure)."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _retry_copy(src: Path, dest: Path, *, attempts: int = 5,
                base_delay_s: float = 10.0,
                logger_obj: Optional[logging.Logger] = None,
                fsync: bool = False) -> None:
    """Copy src -> dest with retry/backoff for NFS flakiness.

    Uses an intermediate ``dest.inprogress`` path then os.replace for atomic
    rename, so dest never appears as a partial file to readers.  With
    ``fsync=True`` the ``.inprogress`` copy is flushed to stable storage
    before the rename (a failed flush is retried like a failed copy).
    """
    log = logger_obj or logger
    inprogress = dest.with_suffix(dest.suffix + ".inprogress")

    for attempt in range(attempts):
        try:
            shutil.copy2(str(src), str(inprogress))
            if fsync:
                _fsync_path(inprogress)
            os.replace(str(inprogress), str(dest))
            return
        except OSError as e:
            if attempt < attempts - 1:
                wait = base_delay_s * (attempt + 1)
                # INFO rather than WARNING for intermediate retries: the
                # vast majority recover within 1-2 attempts and would
                # otherwise drown out the logs that actually matter.
                # The final-failure branch below escalates to ERROR.
                log.info(
                    f"Copy to {dest} failed (attempt {attempt + 1}/{attempts}): {e}. "
                    f"Retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                log.error(f"Copy to {dest} failed after {attempts} attempts: {e}")
                # Best-effort cleanup of any half-written inprogress file
                try:
                    inprogress.unlink(missing_ok=True)
                except OSError:
                    pass
                raise


def atomic_write_text(
    text: str,
    dest: Union[str, Path],
    *,
    encoding: str = "utf-8",
    attempts: int = 5,
    logger_obj: Optional[logging.Logger] = None,
    fsync: bool = False,
) -> None:
    """Write `text` to `dest` atomically, via TMPDIR staging.

    Suitable for JSON, JSONL, config files, logs.  For large binary blobs
    (e.g. torch tensors) use `atomic_write_bytes` or call `_retry_copy`
    directly after writing the staging file with the appropriate serialiser.
    ``fsync=True`` flushes the copy to stable storage before the rename
    (opt-in; see the module docstring).
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    staging = _staging_path(dest_path)
    staging.parent.mkdir(parents=True, exist_ok=True)

    try:
        staging.write_text(text, encoding=encoding)
        _retry_copy(staging, dest_path, attempts=attempts,
                    logger_obj=logger_obj, fsync=fsync)
    finally:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_bytes(
    data: bytes,
    dest: Union[str, Path],
    *,
    attempts: int = 5,
    logger_obj: Optional[logging.Logger] = None,
    fsync: bool = False,
) -> None:
    """Write `data` (bytes) to `dest` atomically, via TMPDIR staging.

    ``fsync=True`` as in :func:`atomic_write_text`.
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    staging = _staging_path(dest_path)
    staging.parent.mkdir(parents=True, exist_ok=True)

    try:
        staging.write_bytes(data)
        _retry_copy(staging, dest_path, attempts=attempts,
                    logger_obj=logger_obj, fsync=fsync)
    finally:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass


def append_jsonl(
    record: dict,
    dest: Union[str, Path],
    *,
    attempts: int = 5,
    logger_obj: Optional[logging.Logger] = None,
    fsync: bool = False,
) -> None:
    """Append a single JSONL record to `dest` atomically.

    Reads the existing file (if any), appends the new record, and rewrites
    via the atomic-staging path.  This is O(N) per append where N is the
    file size, which is fine for our records.jsonl scale (hundreds to
    thousands of small JSON records per file).  A more scalable approach
    would batch appends, but for the steering driver's per-batch flush
    cadence the simple rewrite is plenty.
    """
    import json
    dest_path = Path(dest)

    existing = ""
    if dest_path.exists():
        existing = dest_path.read_text(encoding="utf-8")
        if existing and not existing.endswith("\n"):
            existing += "\n"

    new_line = json.dumps(record, ensure_ascii=False) + "\n"
    atomic_write_text(existing + new_line, dest_path,
                      attempts=attempts, logger_obj=logger_obj, fsync=fsync)


def write_jsonl(
    records: list,
    dest: Union[str, Path],
    *,
    attempts: int = 5,
    logger_obj: Optional[logging.Logger] = None,
    fsync: bool = False,
) -> None:
    """Write a list of dicts as JSONL atomically (replaces any existing file)."""
    import json
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    atomic_write_text(body, dest, attempts=attempts, logger_obj=logger_obj,
                      fsync=fsync)


# ---------------------------------------------------------------------------
# NFS-safe reads with retry
# ---------------------------------------------------------------------------

def _retry_read(
    fn,
    path: Union[str, Path],
    *,
    attempts: Optional[int] = None,
    base_delay_s: Optional[float] = None,
    delays_s: Optional[Sequence[float]] = None,
    transient: Tuple[Type[BaseException], ...] = TRANSIENT_READ_ERRORS,
    non_transient: Tuple[Type[BaseException], ...] = NON_TRANSIENT_READ_ERRORS,
    logger_obj: Optional[logging.Logger] = None,
):
    """Run ``fn(path)`` inside a retry/backoff loop.

    By default treats ``OSError``, ``RuntimeError`` and ``EOFError`` as
    transient -- this includes the ``RuntimeError("storage has wrong
    byte size...")`` thrown by ``torch.load`` when an NFS / MooseFS
    short-read truncates the pickle stream mid-tensor (the failure
    mode that motivated tightening this helper).  ``JSONDecodeError``
    is *not* in the default set: a malformed JSON file is almost
    always corrupt-on-disk, not transient, so retrying just delays
    the inevitable.  ``non_transient`` (default ``FileNotFoundError``,
    since 2026-10-12) is raised at once even though it is an
    ``OSError``; an ``OSError`` with errno ``ESTALE`` is still retried.

    Backoff schedule:
      * If ``delays_s`` is given, it's used directly (length determines
        retry count: ``attempts = len(delays_s) + 1``).
      * Else if ``attempts`` and/or ``base_delay_s`` are given, falls
        back to the legacy linear schedule
        ``base_delay_s * (attempt + 1)`` for backwards compatibility.
      * Else uses the project-standard
        :data:`DEFAULT_RETRY_DELAYS_S` (=``(5, 20, 60, 180)``).
    """
    log = logger_obj or logger

    if delays_s is not None:
        delays = list(delays_s)
        n_attempts = len(delays) + 1
    elif attempts is not None or base_delay_s is not None:
        # Legacy linear-backoff path: preserve the old behaviour for
        # explicit callers.
        n_attempts = int(attempts) if attempts is not None else 5
        base = float(base_delay_s) if base_delay_s is not None else 10.0
        delays = [base * (i + 1) for i in range(n_attempts - 1)]
    else:
        delays = list(DEFAULT_RETRY_DELAYS_S)
        n_attempts = len(delays) + 1

    last_err: Optional[BaseException] = None
    for attempt in range(n_attempts):
        try:
            return fn(path)
        except transient as e:
            if non_transient and isinstance(e, non_transient):
                raise
            last_err = e
            if attempt < n_attempts - 1:
                wait = delays[attempt]
                # INFO rather than WARNING: same reasoning as
                # _retry_copy -- intermediate retries are noisy and
                # almost always recover.  The final failure below
                # escalates to ERROR.
                log.info(
                    f"Read failed for {path} ({type(e).__name__}: {e}) "
                    f"(attempt {attempt + 1}/{n_attempts}). "
                    f"Retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                log.error(
                    f"Read failed for {path} after {n_attempts} attempts. "
                    f"Last error: {type(e).__name__}: {e}"
                )
    assert last_err is not None
    raise last_err


def read_text_with_retry(
    path: Union[str, Path],
    *,
    encoding: str = "utf-8",
    attempts: Optional[int] = None,
    base_delay_s: Optional[float] = None,
    delays_s: Optional[Sequence[float]] = None,
    logger_obj: Optional[logging.Logger] = None,
) -> str:
    """Read text content with NFS-flakiness retry.

    Defaults route through the project-standard exponential backoff
    (:data:`DEFAULT_RETRY_DELAYS_S`).  Pass ``delays_s`` for an
    explicit schedule (e.g. ``delays_s=[5.0]`` for a single 5 s
    retry, useful for diagnostic tools that don't want to wait
    out the full project-standard 265 s on persistent corruption).
    Pass ``attempts`` and/or ``base_delay_s`` to fall back to the
    legacy linear schedule (mainly for tests).
    """
    def _read(p):
        return Path(p).read_text(encoding=encoding)
    return _retry_read(
        _read, path, attempts=attempts, base_delay_s=base_delay_s,
        delays_s=delays_s, logger_obj=logger_obj,
    )


def read_jsonl_with_retry(
    path: Union[str, Path],
    *,
    encoding: str = "utf-8",
    skip_malformed: bool = True,
    attempts: Optional[int] = None,
    base_delay_s: Optional[float] = None,
    delays_s: Optional[Sequence[float]] = None,
    logger_obj: Optional[logging.Logger] = None,
) -> list:
    """Read a JSONL file with NFS-flakiness retry.

    The retry loop wraps the OS-level read; once we have the bytes, JSON
    parsing happens once.  By default malformed lines are skipped with a
    warning (the records.jsonl format may briefly contain a half-written
    last line if a writer crashed mid-flush before the atomic rename
    landed -- though atomic_write_text avoids that on the writer side).
    Set `skip_malformed=False` to raise instead.
    """
    import json
    log = logger_obj or logger
    text = read_text_with_retry(
        path, encoding=encoding, attempts=attempts,
        base_delay_s=base_delay_s, delays_s=delays_s,
        logger_obj=logger_obj,
    )
    out: list = []
    # split on "\n" only: str.splitlines() also breaks at U+2028, U+2029 and
    # U+0085, which a valid JSON string may hold raw, and the two halves would
    # both be malformed (found by the 2026-10-12 pipeline review)
    for line_num, line in enumerate(text.split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            msg = f"{path}: malformed JSON on line {line_num}: {e}"
            if skip_malformed:
                log.warning(msg + " (skipping)")
            else:
                raise ValueError(msg) from e
    return out


def torch_load_with_retry(
    path: Union[str, Path],
    *,
    map_location: Optional[Any] = None,
    weights_only: bool = False,
    attempts: Optional[int] = None,
    base_delay_s: Optional[float] = None,
    delays_s: Optional[Sequence[float]] = None,
    logger_obj: Optional[logging.Logger] = None,
):
    """torch.load wrapped in the NFS retry loop.

    Defaults route through the project-standard exponential backoff
    (:data:`DEFAULT_RETRY_DELAYS_S` ``= (5, 20, 60, 180)`` seconds, 5
    attempts total).  This catches not just ``OSError`` but also the
    ``RuntimeError("storage has wrong byte size of dtype...")`` and
    ``RuntimeError("PytorchStreamReader failed reading zip archive
    ... unexpected EOF, expected N more bytes")`` variants that
    NFS / MooseFS short-reads produce on read-back -- the failure
    mode that motivated this audit.

    Pass ``delays_s`` for an explicit schedule -- diagnostic tools
    auditing many files (e.g. ``pipeline/scan_missing_vectors.py``)
    typically pass ``delays_s=[5.0]`` to fast-fail on persistent
    corruption rather than waiting out the full 265 s per file.

    `weights_only=False` matches existing axis/role-vector loaders,
    which need to deserialise dict-wrapped checkpoints.
    """
    import torch
    def _load(p):
        return torch.load(str(p), map_location=map_location,
                          weights_only=weights_only)
    return _retry_read(
        _load, path, attempts=attempts, base_delay_s=base_delay_s,
        delays_s=delays_s, logger_obj=logger_obj,
    )


def torch_save_with_retry(
    obj: Any,
    dest: Union[str, Path],
    *,
    use_zipfile_serialization: Optional[bool] = None,
    attempts: int = 5,
    verify_load: bool = True,
    verify_sha256: bool = True,
    logger_obj: Optional[logging.Logger] = None,
    **save_kwargs: Any,
) -> None:
    """Atomically save a torch object to ``dest`` with retry/backoff.

    Mirrors the ``atomic_write_*`` family but with strong post-copy
    integrity verification.  Each attempt:

      1. Serialise ``obj`` to a TMPDIR staging file with
         ``torch.save`` (once per call -- staging is reused across
         retries).
      2. Copy staging -> ``dest.inprogress`` (beside ``dest``, so on
         the destination filesystem).
      3. Verify the ``.inprogress`` size matches staging byte-for-byte.
      4. If ``verify_sha256``: stream-hash both files and compare
         SHA-256.  Catches silent byte-flips / partial bad chunks
         where the copy is the right *length* but contains different
         bytes than staging.  This is the failure mode behind the
         apparently-healthy-2.6 GB ``r_guardian__casual.pt`` that
         later refused to ``torch.load`` with
         ``"storage has wrong byte size of dtype"`` -- size and
         even pickle-header parsing succeeded, but one storage's
         payload bytes were wrong.
      5. If ``verify_load``: ``torch.load(..., weights_only=False)``
         to confirm the file deserialises end-to-end.  Tests at the
         actual usage level, catching cases where bytes match but
         downstream torch versions disagree.
      6. Only then rename ``.inprogress`` -> ``dest`` (``os.replace``,
         atomic, the bytes do not change).

    Verifying before the rename (since 2026-10-12; it used to rename
    first and verify ``dest``) means ``dest`` never holds unverified
    bytes: a crash during the minute-long verification of a 2.6 GB file
    no longer leaves a file that skip-if-exists resume logic takes for
    a finished one, and a save that fails every attempt leaves an
    existing ``dest`` (an ``--overwrite`` run's previous file) in place
    instead of deleting it.

    On any verification failure the bad ``.inprogress`` copy is
    unlinked and the copy + verify cycle retries with the
    project-standard exponential backoff (:data:`DEFAULT_RETRY_DELAYS_S`).
    After ``attempts`` failures the staging file is cleaned up and the
    last error is re-raised.

    Parameters
    ----------
    obj : object to serialise.
    dest : final on-disk location (may be on NFS / MooseFS).
    use_zipfile_serialization : if not None, forwarded to
        ``torch.save`` as ``_use_new_zipfile_serialization``.  Step 2
        sets this to ``False`` to avoid a zipfile bug that surfaces as
        "iostream error" on RunPod -- callers serialising large
        activation files should set the same.  Defaults to torch's
        own default for everything else.
    attempts : total attempts (1 = no retry).  Default 5.
    verify_load : run ``torch.load(dest)`` after each successful copy
        to confirm deserialisation works.  Default True.  Set False
        for write paths where the cost of an extra full read isn't
        warranted (small files written to local disk).
    verify_sha256 : compare staging vs dest SHA-256 after each copy.
        Default True.  Set False if you've measured that the extra
        read is dominating wall-clock and you trust ``verify_load``
        (which independently catches byte corruption that affects
        deserialisation -- the relevant case for tensor files).
    save_kwargs : extra kwargs forwarded to ``torch.save``.
    """
    import torch
    log = logger_obj or logger

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_path(dest_path, ".pt.tmp")
    staging.parent.mkdir(parents=True, exist_ok=True)

    save_call_kwargs = dict(save_kwargs)
    if use_zipfile_serialization is not None:
        save_call_kwargs["_use_new_zipfile_serialization"] = use_zipfile_serialization

    inprogress = dest_path.with_suffix(dest_path.suffix + ".inprogress")
    delays = list(DEFAULT_RETRY_DELAYS_S)
    n_attempts = max(1, int(attempts))

    try:
        # Serialise once to staging; reuse across copy+verify retries.
        torch.save(obj, str(staging), **save_call_kwargs)
        staging_size = staging.stat().st_size
        # Hash staging once (cheap on RAM-backed /dev/shm or local disk)
        # so per-retry verification only re-hashes the destination.
        staging_sha = _sha256_file(staging) if verify_sha256 else None

        last_err: Optional[BaseException] = None
        for attempt in range(n_attempts):
            try:
                shutil.copy2(str(staging), str(inprogress))

                # 1. Size check (cheapest -- one stat call).
                dest_size = inprogress.stat().st_size
                if dest_size != staging_size:
                    raise OSError(
                        f"size mismatch at {dest_path}: "
                        f"staging {staging_size} bytes, dest {dest_size} bytes "
                        f"(silent short-write)"
                    )

                # 2. SHA-256 byte-equality check (catches bit-flips
                #    inside a correct-size file -- e.g. corrupted
                #    NFS chunks that pass length checks).
                if verify_sha256:
                    dest_sha = _sha256_file(inprogress)
                    if dest_sha != staging_sha:
                        raise OSError(
                            f"sha256 mismatch at {dest_path}: "
                            f"staging {staging_sha}, dest {dest_sha} "
                            f"(silent byte-flip / chunk corruption)"
                        )

                # 3. End-to-end load check (catches torch-level
                #    deserialisation failures even if byte-equality
                #    held; also a smoke-test that the consumer's
                #    code path will actually work).
                if verify_load:
                    torch.load(str(inprogress),
                               map_location="cpu", weights_only=False)

                # 4. Publish: rename the verified copy into place.
                os.replace(str(inprogress), str(dest_path))
                return  # success
            except (OSError, RuntimeError, EOFError,
                    pickle.UnpicklingError, ValueError) as e:
                # UnpicklingError / ValueError: what torch.load raises
                # on some corrupt pickles; without them such a failure
                # escaped the cleanup below.
                last_err = e
                # Best-effort cleanup of the bad copy before the next
                # attempt so the next iteration's copy starts from a
                # clean slate.  ``dest`` itself is never touched here:
                # it is either absent or the previous, untouched file.
                try:
                    if inprogress.exists():
                        inprogress.unlink()
                except OSError:
                    pass
                if attempt < n_attempts - 1:
                    wait = delays[attempt] if attempt < len(delays) else delays[-1]
                    log.warning(
                        f"Save+verify to {dest_path} failed "
                        f"(attempt {attempt+1}/{n_attempts}, "
                        f"{type(e).__name__}: {e}). "
                        f"Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                else:
                    log.error(
                        f"Save+verify to {dest_path} failed after "
                        f"{n_attempts} attempts. "
                        f"Last error: {type(e).__name__}: {e}"
                    )
        assert last_err is not None
        raise last_err
    finally:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
