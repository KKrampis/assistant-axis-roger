"""OpenAI Batch API for chat-completion judge calls (pipeline step 3).

Pipeline step 3 (``pipeline/3_judge.py``) judges every response of an
extraction: for the October 2026 corpus about 1.5 million calls with the
30 x 30 combinations (466,000 trait, 169,500 role and 899,700 combination
responses), 636,000 without them.  Since
2026-10-10 it sends them through OpenAI's Batch API by default (Roger:
"make batch mode the default, with a flag to override that"), at half the
real-time price (``gpt-4.1-mini``: $0.20 / $0.80 per million tokens against
$0.40 / $1.60).  This module is the transport: it writes request lines into
chunk files, uploads and submits them, polls, and hands every result back to
the caller by ``custom_id``.

Service limits (OpenAI batch guide, read 2026-10-10): a batch holds at most
50,000 requests and 200 MB, the completion window is 24 hours, and each model
has a queue limit on the prompt tokens enqueued per organisation, shown only
on the organisation's limits page.  A batch over that limit fails validation
with the code ``token_limit_exceeded``; :meth:`OpenAIBatchRunner.run` then
waits for a batch in flight to finish and tries again, or, with none of its
own in flight, splits the chunk in half, so the run adapts to whatever the
limit is.  Several runs on one API key (the four-pod extraction, walk-through
P1, 2026-10-12) share the queue: a run refused with none of its own batches in
flight first reads the organisation's batch list, and when other runs'
batches are queued it waits ``poll_seconds`` and sends the same upload again
instead of splitting (a split would halve the chunk down to
``MIN_SPLIT_REQUESTS`` and stop the run, though nothing was wrong with its
size).  Size ``--batch_requests`` from the queue limit and the number of runs
(pipeline/README.md, "Several runs on one organisation's queue").

Restart.  The state file (``batch_state.json`` beside the scores) records
every chunk as it is submitted.  A run started again finds the batches still
in flight, excludes their requests from what it submits (:meth:`resume`), and
collects them; chunks written but never submitted are discarded and rebuilt.
A chunk's upload is recorded before a batch is created from it, so a run
stopped between the two calls leaves an upload with no batch recorded;
:meth:`resume` then looks the upload up among the organisation's batches and
collects the batch it finds instead of submitting its requests again (before
2026-10-12 they were, and billed twice).  The same upload is reused when the
queue limit refuses a batch and it is tried again, and deleted once it is of
no further use.

Network errors.  A status check or a result download that fails is tried
again at the next poll (the batch stays in flight); only
``MAX_FAILED_POLL_ROUNDS`` rounds in a row with every check failing stop the
run, so a network outage of a few minutes during a wait of hours does not
end it (before 2026-10-12 one failed check did).

Only the standard library, ``openai`` and the stdlib-only
``assistant_axis/atomic_io.py`` are imported: step 3 has no use for torch,
whose import can take minutes on the pod's network volume (the package
``__init__`` is lazy since 2026-10-12, so importing this module through the
package no longer pulls torch in either).
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

# The state file lives beside the scores, on the pod's network volume: written
# through TMPDIR staging with retry and backoff (AGENT_NOTES "NFS-safe file
# I/O"), since a failed write right after a submission is what would make a
# restart submit the same requests again.
from .atomic_io import atomic_write_text

ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_BYTES = 200 * 1024 * 1024
# A chunk well inside both limits, and about 10 million prompt tokens of step-3
# requests (about 1,000 tokens each), so a modest queue limit takes a few at once.
DEFAULT_CHUNK_REQUESTS = 10_000
DEFAULT_CHUNK_BYTES = 100 * 1024 * 1024
# The smallest chunk the token-limit split goes down to before giving up.
MIN_SPLIT_REQUESTS = 50
# Poll rounds in a row in which every status check or download failed before
# the run gives up (about half an hour at the default 60 s poll).
MAX_FAILED_POLL_ROUNDS = 30
# Looking an upload up among the organisation's batches (newest first): scan
# back to this long before the upload, and no further than this many batches.
LOOKUP_MARGIN_SECONDS = 3600
LOOKUP_MAX_BATCHES = 2000
TERMINAL_STATUSES = ("completed", "failed", "expired", "cancelled")
# A batch in these states holds its tokens in the organisation's queue.
ACTIVE_STATUSES = ("validating", "in_progress", "finalizing", "cancelling")
# How far back the organisation's batch list is read for other runs'
# batches in the queue (a batch lives at most its 24-hour window).
OTHERS_LOOKBACK_SECONDS = 25 * 3600
TOKEN_LIMIT_CODE = "token_limit_exceeded"

STATE_VERSION = 1


class BatchFailedError(RuntimeError):
    """A batch failed validation for a reason other than the queue limit (a
    bad model name, a malformed line): nothing is retried or sent in real
    time, because the same requests would fail the same way."""


def request_line(custom_id: str, body: Dict[str, Any]) -> str:
    """One line of a batch input file."""
    return json.dumps({"custom_id": custom_id, "method": "POST", "url": ENDPOINT,
                       "body": body}, ensure_ascii=False) + "\n"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _error_codes(batch: Any) -> List[str]:
    errors = _get(batch, "errors")
    data = _get(errors, "data") or []
    return [str(_get(e, "code") or "") for e in data]


def _error_text(batch: Any) -> str:
    errors = _get(batch, "errors")
    data = _get(errors, "data") or []
    return "; ".join(f"{_get(e, 'code')}: {_get(e, 'message')}" for e in data[:5]) or "(no detail)"


def _content_text(content: Any) -> str:
    text = _get(content, "text")
    if callable(text):
        text = text()
    if text is None and hasattr(content, "read"):
        raw = content.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return text or ""


def _read_custom_ids(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["custom_id"] for line in f if line.strip()]


def _read_prompts(path: Path, wanted: Set[str]) -> Dict[str, Dict[str, Any]]:
    """``custom_id -> body`` for the wanted requests of a chunk file."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["custom_id"] in wanted:
                out[rec["custom_id"]] = rec["body"]
    return out


class OpenAIBatchRunner:
    """Write, submit, poll and collect OpenAI batches for one output directory.

    ``on_result(custom_id, body, error)`` is called for every request a batch
    returns: ``body`` is the chat-completion response (a dict) on success,
    ``error`` a short string otherwise.  ``on_batch_done()`` is called after
    each batch's results, before the batch is marked collected, so the caller
    can make them durable first.  Requests a batch never answered (expired,
    cancelled, or an error line) come back from :meth:`run` as
    ``custom_id -> request body``, for the caller to send in real time or
    leave for a rerun.
    """

    def __init__(self, client: Any, work_dir: Path, state_path: Path, *,
                 poll_seconds: float = 60.0, validation_poll_seconds: float = 10.0,
                 max_failed_poll_rounds: int = MAX_FAILED_POLL_ROUNDS,
                 metadata: Optional[Dict[str, str]] = None,
                 logger: Optional[logging.Logger] = None):
        self.client = client
        self.work_dir = Path(work_dir)
        self.state_path = Path(state_path)
        self.poll_seconds = poll_seconds
        self.validation_poll_seconds = validation_poll_seconds
        self.max_failed_poll_rounds = max_failed_poll_rounds
        self.metadata = dict(metadata or {})
        self.log = logger or logging.getLogger(__name__)
        self.state: Dict[str, Any] = self._load_state()

    # -- state ------------------------------------------------------------

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get("version") != STATE_VERSION:
                raise RuntimeError(f"{self.state_path}: unknown state version {state.get('version')!r}")
            return state
        return {"version": STATE_VERSION, "chunks": []}

    def _save_state(self) -> None:
        atomic_write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n",
                          self.state_path, logger_obj=self.log)

    def _record(self, path: Path) -> Dict[str, Any]:
        for rec in self.state["chunks"]:
            if rec["path"] == str(path):
                return rec
        rec = {"path": str(path)}
        self.state["chunks"].append(rec)
        return rec

    def _inflight_records(self) -> List[Dict[str, Any]]:
        return [r for r in self.state["chunks"] if r.get("batch_id") and not r.get("collected")]

    async def resume(self) -> Set[str]:
        """Prepare a restart: drop chunks never submitted, and return the
        ``custom_id`` of every request in a batch still to be collected (the
        caller must not submit those again).  A chunk file that is gone is
        read back from the batch's input file on the service.

        A chunk with an upload recorded and no batch (the run stopped
        between uploading and recording the batch, or after the queue limit
        refused it) is looked up among the organisation's batches: a batch
        created from that upload and not failed is collected like any other;
        otherwise the upload is deleted and the chunk dropped with the rest."""
        orphans = [r for r in self.state["chunks"]
                   if not r.get("batch_id") and r.get("input_file_id")]
        if orphans:
            found = await self._find_batches(
                {r["input_file_id"] for r in orphans},
                since=min(float(r.get("uploaded_ts") or 0) for r in orphans))
            for rec in orphans:
                batch = found.get(rec["input_file_id"])
                if batch is not None:
                    self.log.warning(f"{Path(rec['path']).name}: batch {_get(batch, 'id')} was "
                                     f"created from its upload before the last run stopped; "
                                     f"collecting it instead of submitting the chunk again")
                    rec.update(batch_id=_get(batch, "id"), status=_get(batch, "status"),
                               submitted_at=_now(), recovered=True)
                    rec.pop("collected", None)
        keep = []
        for rec in self.state["chunks"]:
            if rec.get("batch_id"):
                keep.append(rec)
            else:
                Path(rec["path"]).unlink(missing_ok=True)
                await self._delete_files(rec.get("input_file_id"))
        self.state["chunks"] = keep
        ids: Set[str] = set()
        for rec in self._inflight_records():
            path = Path(rec["path"])
            if not path.exists():
                self.log.warning(f"batch {rec['batch_id']}: chunk file {path} is gone; "
                                 f"reading the batch's input file back from the service")
                content = await self.client.files.content(rec["input_file_id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(_content_text(content), encoding="utf-8")
            ids.update(_read_custom_ids(path))
        self._save_state()
        if ids:
            self.log.info(f"resuming {len(self._inflight_records())} batch(es) in flight, "
                          f"{len(ids)} requests")
        return ids

    def inflight_models(self) -> Dict[str, List[str]]:
        """``model -> [chunk names]`` over the batches still to be collected,
        read from each chunk file's first request (a chunk is written by one
        run, with one model).  Call after :meth:`resume`, which restores a
        missing chunk file from the service.  Step 3 refuses to resume when a
        model here differs from its ``--judge_model``: the batches would be
        collected and charged under the new name (walk-through P5,
        2026-10-12)."""
        out: Dict[str, List[str]] = {}
        for rec in self._inflight_records():
            path = Path(rec["path"])
            model = "(unreadable chunk)"
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            model = str(json.loads(line).get("body", {}).get("model"))
                            break
            except (OSError, ValueError):
                pass
            out.setdefault(model, []).append(path.name)
        return out

    # -- chunk files --------------------------------------------------------

    def write_chunks(self, requests: Iterable[Tuple[str, Dict[str, Any]]], *,
                     max_requests: int = DEFAULT_CHUNK_REQUESTS,
                     max_bytes: int = DEFAULT_CHUNK_BYTES) -> List[Path]:
        """Stream ``(custom_id, body)`` pairs into chunk files of at most
        ``max_requests`` lines and ``max_bytes`` bytes; return their paths."""
        max_requests = min(max_requests, MAX_BATCH_REQUESTS)
        max_bytes = min(max_bytes, MAX_BATCH_BYTES)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # A name no earlier chunk had: two runs started in the same second
        # used to share names, and the second run's chunk then took over the
        # first one's record, collected flag and batch included (2026-10-12).
        known = {Path(r["path"]).name for r in self.state["chunks"]}
        base = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        stamp, k = base, 0
        while any(n.startswith(f"chunk_{stamp}_") for n in known) \
                or any(self.work_dir.glob(f"chunk_{stamp}_*")):
            k += 1
            stamp = f"{base}-{k}"
        paths: List[Path] = []
        f = None
        n = size = 0
        for custom_id, body in requests:
            line = request_line(custom_id, body).encode("utf-8")
            if f is None or n >= max_requests or size + len(line) > max_bytes:
                if f is not None:
                    f.close()
                path = self.work_dir / f"chunk_{stamp}_{len(paths):04d}.jsonl"
                paths.append(path)
                f = open(path, "wb")
                n = size = 0
            f.write(line)
            n += 1
            size += len(line)
        if f is not None:
            f.close()
        for path in paths:
            with open(path, "rb") as fh:
                self._record(path)["n_requests"] = sum(1 for _ in fh)
        self._save_state()
        return paths

    def _split(self, path: Path) -> List[Path]:
        lines = path.read_bytes().splitlines(keepends=True)
        half = len(lines) // 2
        out = []
        for i, part in enumerate((lines[:half], lines[half:])):
            p = path.with_name(f"{path.stem}_{'ab'[i]}.jsonl")
            p.write_bytes(b"".join(part))
            rec = self._record(p)
            rec["n_requests"] = len(part)
            out.append(p)
        self.state["chunks"] = [r for r in self.state["chunks"] if r["path"] != str(path)]
        path.unlink(missing_ok=True)
        self._save_state()
        return out

    # -- submission and collection -----------------------------------------

    async def _find_batches(self, file_ids: Set[str], *, since: float) -> Dict[str, Any]:
        """``input_file_id -> batch`` for the newest batch not failed that was
        created from each of ``file_ids``, scanning the organisation's
        batches (listed newest first: checked against the real API by the
        2026-10-12 canary) back to ``LOOKUP_MARGIN_SECONDS``
        before ``since`` (epoch seconds; 0 scans up to the cap)."""
        found: Dict[str, Any] = {}
        n = 0
        async for batch in self.client.batches.list(limit=100):
            n += 1
            created = float(_get(batch, "created_at") or 0)
            if (since and created and created < since - LOOKUP_MARGIN_SECONDS) \
                    or n > LOOKUP_MAX_BATCHES:
                break
            fid = _get(batch, "input_file_id")
            if fid in file_ids and fid not in found and _get(batch, "status") != "failed":
                found[fid] = batch
        return found

    async def _others_active(self) -> int:
        """How many batches of the organisation not recorded by this runner
        hold tokens in the queue now (another pod's step 3 on the same API
        key, for one).  Read from ``batches.list`` (newest first), back to
        :data:`OTHERS_LOOKBACK_SECONDS`."""
        own = {r.get("batch_id") for r in self.state["chunks"] if r.get("batch_id")}
        cutoff = time.time() - OTHERS_LOOKBACK_SECONDS
        n = scanned = 0
        async for batch in self.client.batches.list(limit=100):
            scanned += 1
            created = float(_get(batch, "created_at") or 0)
            if (created and created < cutoff) or scanned > LOOKUP_MAX_BATCHES:
                break
            if _get(batch, "status") in ACTIVE_STATUSES and _get(batch, "id") not in own:
                n += 1
        return n

    async def _submit(self, path: Path) -> Any:
        rec = self._record(path)
        if not rec.get("input_file_id") or rec.get("collected"):
            # Recorded before the batch is created, so that a run stopped
            # between the two calls can find the batch on restart (resume).
            uploaded = await self.client.files.create(file=(path.name, path.read_bytes()),
                                                      purpose="batch")
            rec.update(input_file_id=uploaded.id, uploaded_at=_now(), uploaded_ts=time.time())
            self._save_state()
        # else: the queue limit refused a batch of this upload; send it again
        batch = await self.client.batches.create(
            input_file_id=rec["input_file_id"], endpoint=ENDPOINT,
            completion_window=COMPLETION_WINDOW,
            metadata={**self.metadata, "chunk": path.name})
        rec.update(batch_id=batch.id, submitted_at=_now(), status=_get(batch, "status"))
        self._save_state()
        self.log.info(f"submitted {path.name}: batch {batch.id}, {rec.get('n_requests')} requests")
        return batch

    async def _count_others_or_assume(self) -> int:
        """:meth:`_others_active`, or 1 when the list cannot be read (then
        waiting is the safe guess: a split cannot be undone).  After
        ``max_failed_poll_rounds`` failed reads in a row the error is raised."""
        try:
            n = await self._others_active()
            self._failed_lists = 0
            return n
        except Exception as e:  # noqa: BLE001 -- network; decided below
            self._failed_lists = getattr(self, "_failed_lists", 0) + 1
            self.log.warning(f"could not list the organisation's batches ({type(e).__name__}: "
                             f"{e}); assuming other runs hold the queue")
            if self._failed_lists >= self.max_failed_poll_rounds:
                raise
            return 1

    async def _wait_validation(self, batch: Any) -> Any:
        while _get(batch, "status") == "validating":
            await asyncio.sleep(self.validation_poll_seconds)
            batch = await self.client.batches.retrieve(batch.id)
        return batch

    async def _delete_files(self, *file_ids: Optional[str]) -> None:
        for file_id in file_ids:
            if file_id:
                try:
                    await self.client.files.delete(file_id)
                except Exception as e:  # noqa: BLE001 -- tidying only
                    self.log.debug(f"could not delete file {file_id}: {e}")

    async def _download(self, batch: Any) -> List[str]:
        """The text of a finished batch's output and error files.  Network
        only, so that a failure here leaves nothing half processed and the
        batch can be collected at the next poll."""
        texts = []
        for file_id in (_get(batch, "output_file_id"), _get(batch, "error_file_id")):
            if file_id:
                texts.append(_content_text(await self.client.files.content(file_id)))
        return texts

    async def _collect(self, rec: Dict[str, Any], batch: Any, texts: List[str],
                       on_result: Callable[[str, Optional[dict], Optional[str]], None],
                       on_batch_done: Callable[[], None],
                       unanswered: Dict[str, Dict[str, Any]]) -> None:
        path = Path(rec["path"])
        expected = set(_read_custom_ids(path))
        seen: Set[str] = set()
        failed: Set[str] = set()
        n_malformed = 0
        for text in texts:
            # Split at "\n" only: str.splitlines() also breaks at U+2028,
            # U+2029, U+0085 and others, which JSON allows unescaped inside a
            # string, and one broken line used to stop every collection of
            # the batch (2026-10-12).
            for line in text.split("\n"):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    n_malformed += 1
                    continue
                cid = item.get("custom_id") if isinstance(item, dict) else None
                if cid not in expected or cid in seen:
                    continue
                seen.add(cid)
                response = item.get("response") or {}
                if item.get("error") is None and response.get("status_code") == 200:
                    on_result(cid, response.get("body") or {}, None)
                else:
                    err = item.get("error") or (response.get("body") or {}).get("error") or {}
                    failed.add(cid)
                    on_result(cid, None, str(err.get("message") or err or "error")[:200])
        missing = expected - seen
        status = _get(batch, "status")
        if n_malformed:
            self.log.warning(f"batch {batch.id}: {n_malformed} unreadable result line(s); "
                             f"their requests count as not answered")
        if failed or missing:
            self.log.warning(f"batch {batch.id} ({status}): {len(failed)} errored, "
                             f"{len(missing)} not answered, of {len(expected)}")
        unanswered.update(_read_prompts(path, failed | missing))
        on_batch_done()
        rec.update(collected=True, status=status, collected_at=_now(),
                   n_answered=len(seen) - len(failed))
        self._save_state()
        path.unlink(missing_ok=True)
        await self._delete_files(rec.get("input_file_id"), _get(batch, "output_file_id"),
                                 _get(batch, "error_file_id"))

    async def run(self, chunks: List[Path],
                  on_result: Callable[[str, Optional[dict], Optional[str]], None],
                  on_batch_done: Callable[[], None] = lambda: None) -> Dict[str, Dict[str, Any]]:
        """Submit ``chunks`` (and collect any batches :meth:`resume` found in
        flight) and wait until every batch has been collected.  Returns the
        requests no batch answered, as ``custom_id -> body``."""
        pending: Deque[Path] = deque(chunks)
        inflight: Dict[str, Dict[str, Any]] = {r["batch_id"]: r for r in self._inflight_records()}
        unanswered: Dict[str, Dict[str, Any]] = {}
        hold = False
        failed_rounds = 0
        seen_empty_queue: Set[Path] = set()   # refused once with no other batch queued
        while pending or inflight:
            while pending and not hold:
                path = pending.popleft()
                batch = await self._wait_validation(await self._submit(path))
                rec = self._record(path)
                if _get(batch, "status") == "failed":
                    rec.update(status="failed", collected=True)
                    if TOKEN_LIMIT_CODE in _error_codes(batch):
                        rec.pop("batch_id", None)
                        rec.pop("collected", None)
                        if inflight:
                            # held: the retry creates a batch from the same upload
                            self.log.info(f"{path.name}: queue limit reached; waiting for one "
                                          f"of {len(inflight)} batch(es) in flight")
                            pending.appendleft(path)
                            hold = True
                            self._save_state()
                            continue
                        others = await self._count_others_or_assume()
                        if others:
                            # Several runs on one key (walk-through P1): the
                            # queue is full of other runs' batches, not too
                            # small for this chunk.  Splitting would halve it
                            # down to MIN_SPLIT_REQUESTS and stop the run;
                            # wait for the queue and send it again instead.
                            self.log.info(f"{path.name}: queue limit reached while {others} "
                                          f"batch(es) of other runs are queued; trying again in "
                                          f"{self.poll_seconds:.0f} s")
                            self._save_state()
                            await asyncio.sleep(self.poll_seconds)
                            pending.appendleft(path)
                            continue
                        if path not in seen_empty_queue:
                            # The queue may have emptied between the refusal
                            # and the list: one more try before splitting.
                            seen_empty_queue.add(path)
                            self._save_state()
                            pending.appendleft(path)
                            continue
                        if rec.get("n_requests", 0) > MIN_SPLIT_REQUESTS:
                            self.log.info(f"{path.name}: over the queue limit on its own "
                                          f"({rec.get('n_requests')} requests); splitting it")
                            await self._delete_files(rec.pop("input_file_id", None))
                            self._save_state()
                            pending.extendleft(reversed(self._split(path)))
                        else:
                            await self._delete_files(rec.pop("input_file_id", None))
                            self._save_state()
                            raise BatchFailedError(f"{path.name}: over the queue limit even at "
                                                   f"{rec.get('n_requests')} requests: {_error_text(batch)}")
                        self._save_state()
                        continue
                    # The batch never ran and will not be retried as it is:
                    # its upload is of no further use (the chunk file stays).
                    await self._delete_files(rec.get("input_file_id"))
                    self._save_state()
                    raise BatchFailedError(f"batch {batch.id} ({path.name}) failed validation: "
                                           f"{_error_text(batch)}")
                inflight[batch.id] = rec
            if not inflight:
                continue
            await asyncio.sleep(self.poll_seconds)
            n_ok = 0
            last_error: Optional[BaseException] = None
            for batch_id in list(inflight):
                try:
                    batch = await self.client.batches.retrieve(batch_id)
                    status = _get(batch, "status")
                    texts = (await self._download(batch)
                             if status in TERMINAL_STATUSES and status != "failed" else [])
                except Exception as e:  # noqa: BLE001 -- tried again at the next poll
                    last_error = e
                    self.log.warning(f"batch {batch_id}: status check or download failed "
                                     f"({type(e).__name__}: {e}); trying again at the next poll")
                    continue
                n_ok += 1
                counts = _get(batch, "request_counts")
                if status not in TERMINAL_STATUSES:
                    self.log.info(f"batch {batch_id}: {status} "
                                  f"({_get(counts, 'completed', 0)}/{_get(counts, 'total', '?')})")
                    continue
                rec = inflight.pop(batch_id)
                if status == "failed" and TOKEN_LIMIT_CODE in _error_codes(batch) \
                        and Path(rec["path"]).exists():
                    # Refused by the queue limit after a restart (a batch still
                    # validating when the last run stopped): submit it again.
                    rec.pop("batch_id", None)
                    rec.update(status="failed")
                    self._save_state()
                    pending.append(Path(rec["path"]))
                    continue
                if status == "failed":
                    rec.update(status="failed", collected=True)
                    self._save_state()
                    await self._delete_files(rec.get("input_file_id"))
                    raise BatchFailedError(f"batch {batch_id} failed: {_error_text(batch)}")
                await self._collect(rec, batch, texts, on_result, on_batch_done, unanswered)
                hold = False
            if last_error is not None and n_ok == 0:
                failed_rounds += 1
                if failed_rounds >= self.max_failed_poll_rounds:
                    self.log.error(f"{failed_rounds} poll rounds in a row failed; stopping (the "
                                   f"batches stay recorded, and a rerun collects them)")
                    raise last_error
            else:
                failed_rounds = 0
        return unanswered
