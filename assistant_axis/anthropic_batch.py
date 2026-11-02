"""Anthropic Message Batches for pipeline step 3's judge calls.

The counterpart of :class:`assistant_axis.openai_batch.OpenAIBatchRunner`,
with the same interface (``resume``, ``inflight_models``, ``write_chunks``,
``run``), so that ``judge_in_batches`` in ``pipeline/3_judge.py`` drives
either.  Added 2026-10-26 for the step-3 judge comparison (Roger: "let's do
Haiku in batch"): half the real-time price, results within 24 hours, most
within one.

Differences from the OpenAI runner:

* **Requests go inline** in ``messages.batches.create(requests=[...])``;
  there is no upload.  A batch holds at most 100,000 requests and 256 MB.
* **Custom ids are restricted** (letters, digits, ``_`` and ``-``, at most 64
  characters), and step 3's are ``entity|key``.  Each line of a chunk file
  carries the step-3 id, a short batch id (``r000123``) and the request
  parameters; results are mapped back through the chunk file.
* **No queue validation.**  A create refused for the organisation's batch
  queue (HTTP 429) is tried again after ``poll_seconds``, up to
  ``max_create_retries`` times.
* **A created batch that was not recorded** (the run stopped between the
  create call and the state write) cannot be found by an upload id.  The
  chunk is marked ``submitting_ts`` before the call; :meth:`resume` looks
  for a batch created within ``LOOKUP_WINDOW_SECONDS`` of that time with the
  same number of requests, and collects it when there is exactly one;
  otherwise the chunk is dropped and its requests sent again.

Results are handed back as plain dicts of the message (``content``,
``stop_reason``, ``stop_details``, ``usage``).  Only the standard library and
``assistant_axis/atomic_io.py`` are imported; the client is passed in.
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

from .atomic_io import atomic_write_text

MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 256 * 1024 * 1024
DEFAULT_CHUNK_REQUESTS = 10_000
DEFAULT_CHUNK_BYTES = 100 * 1024 * 1024
MAX_FAILED_POLL_ROUNDS = 30
MAX_CREATE_RETRIES = 60
LOOKUP_WINDOW_SECONDS = 900
LOOKUP_MAX_BATCHES = 500
STATE_VERSION = 1


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_dict(obj: Any) -> Any:
    """An SDK object as plain data (``model_dump`` where it has one)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    return {k: _to_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}


def _epoch(value: Any) -> float:
    """A batch's ``created_at`` (a datetime or an ISO string) as epoch seconds; 0 when unreadable."""
    if isinstance(value, _dt.datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return _dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _read_lines(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _maybe_await(x: Any) -> Any:
    return await x if asyncio.iscoroutine(x) or isinstance(x, asyncio.Future) else x


async def _iterate(x: Any):
    """Items of a sync or async iterable (the SDK's results decoder and list paginator)."""
    x = await _maybe_await(x)
    if hasattr(x, "__aiter__"):
        async for item in x:
            yield item
    else:
        for item in x:
            yield item


def _status_code(exc: BaseException) -> Optional[int]:
    return getattr(exc, "status_code", None)


class AnthropicBatchRunner:
    """Write, submit, poll and collect Anthropic Message Batches for one
    output directory; see :class:`assistant_axis.openai_batch.OpenAIBatchRunner`
    for the contract of ``on_result(custom_id, body, error)``,
    ``on_batch_done()`` and the requests :meth:`run` returns unanswered."""

    def __init__(self, client: Any, work_dir: Path, state_path: Path, *,
                 poll_seconds: float = 60.0,
                 max_failed_poll_rounds: int = MAX_FAILED_POLL_ROUNDS,
                 max_create_retries: int = MAX_CREATE_RETRIES,
                 logger: Optional[logging.Logger] = None):
        self.client = client
        self.work_dir = Path(work_dir)
        self.state_path = Path(state_path)
        self.poll_seconds = poll_seconds
        self.max_failed_poll_rounds = max_failed_poll_rounds
        self.max_create_retries = max_create_retries
        self.log = logger or logging.getLogger(__name__)
        self.state: Dict[str, Any] = self._load_state()

    # -- state ------------------------------------------------------------

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get("version") != STATE_VERSION:
                raise RuntimeError(f"{self.state_path}: unknown state version {state.get('version')!r}")
            return state
        return {"version": STATE_VERSION, "provider": "anthropic", "chunks": []}

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

    async def _find_batch(self, rec: Dict[str, Any]) -> Optional[Any]:
        """The one batch created within ``LOOKUP_WINDOW_SECONDS`` of the
        chunk's submission attempt with its number of requests, or None."""
        t0 = float(rec.get("submitting_ts") or 0)
        matches = []
        n = 0
        async for b in _iterate(self.client.messages.batches.list(limit=100)):
            n += 1
            created = _epoch(_get(b, "created_at"))
            if n > LOOKUP_MAX_BATCHES or (created and created < t0 - LOOKUP_WINDOW_SECONDS):
                break
            counts = _get(b, "request_counts")
            total = sum(int(_get(counts, k, 0) or 0)
                        for k in ("processing", "succeeded", "errored", "canceled", "expired"))
            if created and abs(created - t0) <= LOOKUP_WINDOW_SECONDS and total == rec.get("n_requests"):
                matches.append(b)
        return matches[0] if len(matches) == 1 else None

    async def resume(self) -> Set[str]:
        """Prepare a restart: adopt a batch created but not recorded when
        exactly one matches, drop chunks never submitted, and return the
        step-3 ``custom_id`` of every request in a batch still to be
        collected (the caller must not submit those again)."""
        for rec in [r for r in self.state["chunks"] if r.get("submitting_ts") and not r.get("batch_id")]:
            try:
                batch = await self._find_batch(rec)
            except Exception as e:  # noqa: BLE001 -- then the chunk is sent again
                self.log.warning(f"{Path(rec['path']).name}: could not list batches ({type(e).__name__}: {e})")
                batch = None
            if batch is not None:
                self.log.warning(f"{Path(rec['path']).name}: batch {_get(batch, 'id')} was created before "
                                 f"the last run stopped; collecting it instead of submitting the chunk again")
                rec.update(batch_id=_get(batch, "id"), submitted_at=_now(), recovered=True)
        keep = []
        for rec in self.state["chunks"]:
            if rec.get("batch_id"):
                keep.append(rec)
            else:
                Path(rec["path"]).unlink(missing_ok=True)
        self.state["chunks"] = keep
        ids: Set[str] = set()
        for rec in self._inflight_records():
            path = Path(rec["path"])
            if not path.exists():
                raise RuntimeError(f"batch {rec['batch_id']}: chunk file {path} is gone, so its results "
                                   f"cannot be mapped back to step-3 ids")
            ids.update(line["custom_id"] for line in _read_lines(path))
        self._save_state()
        if ids:
            self.log.info(f"resuming {len(self._inflight_records())} batch(es) in flight, {len(ids)} requests")
        return ids

    def inflight_models(self) -> Dict[str, List[str]]:
        """``model -> [chunk names]`` over the batches still to be collected."""
        out: Dict[str, List[str]] = {}
        for rec in self._inflight_records():
            path = Path(rec["path"])
            model = "(unreadable chunk)"
            try:
                lines = _read_lines(path)
                if lines:
                    model = str(lines[0].get("params", {}).get("model"))
            except (OSError, ValueError):
                pass
            out.setdefault(model, []).append(path.name)
        return out

    # -- chunk files --------------------------------------------------------

    def write_chunks(self, requests: Iterable[Tuple[str, Dict[str, Any]]], *,
                     max_requests: int = DEFAULT_CHUNK_REQUESTS,
                     max_bytes: int = DEFAULT_CHUNK_BYTES) -> List[Path]:
        """Stream ``(custom_id, params)`` pairs into chunk files of at most
        ``max_requests`` lines and ``max_bytes`` bytes; return their paths."""
        max_requests = min(max_requests, MAX_BATCH_REQUESTS)
        max_bytes = min(max_bytes, MAX_BATCH_BYTES)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        known = {Path(r["path"]).name for r in self.state["chunks"]}
        base = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        stamp, k = base, 0
        while any(n.startswith(f"chunk_{stamp}_") for n in known) or any(self.work_dir.glob(f"chunk_{stamp}_*")):
            k += 1
            stamp = f"{base}-{k}"
        def encode(custom_id: str, i: int, params: Dict[str, Any]) -> bytes:
            return (json.dumps({"custom_id": custom_id, "batch_custom_id": f"r{i:06d}", "params": params},
                               ensure_ascii=False) + "\n").encode("utf-8")

        paths: List[Path] = []
        f = None
        n = size = 0
        for custom_id, params in requests:
            line = encode(custom_id, n, params)
            if f is None or n >= max_requests or (n and size + len(line) > max_bytes):
                if f is not None:
                    f.close()
                path = self.work_dir / f"chunk_{stamp}_{len(paths):04d}.jsonl"
                paths.append(path)
                f = open(path, "wb")
                n = size = 0
                line = encode(custom_id, 0, params)
            f.write(line)
            n += 1
            size += len(line)
        if f is not None:
            f.close()
        for path in paths:
            self._record(path)["n_requests"] = len(_read_lines(path))
        self._save_state()
        return paths

    # -- submission and collection -----------------------------------------

    async def _submit(self, path: Path) -> Any:
        rec = self._record(path)
        requests = [{"custom_id": line["batch_custom_id"], "params": line["params"]} for line in _read_lines(path)]
        for attempt in range(self.max_create_retries + 1):
            rec["submitting_ts"] = time.time()
            self._save_state()
            try:
                batch = await _maybe_await(self.client.messages.batches.create(requests=requests))
                break
            except Exception as e:  # noqa: BLE001 -- 429 retried, anything else raised
                if _status_code(e) == 429 and attempt < self.max_create_retries:
                    self.log.info(f"{path.name}: batch queue full ({e}); trying again in {self.poll_seconds:.0f} s")
                    rec.pop("submitting_ts", None)
                    self._save_state()
                    await asyncio.sleep(self.poll_seconds)
                    continue
                rec.pop("submitting_ts", None)
                self._save_state()
                raise
        rec.update(batch_id=_get(batch, "id"), submitted_at=_now())
        self._save_state()
        self.log.info(f"submitted {path.name}: batch {_get(batch, 'id')}, {rec.get('n_requests')} requests")
        return batch

    async def _results(self, batch_id: str) -> List[Any]:
        return [r async for r in _iterate(self.client.messages.batches.results(batch_id))]

    def _collect(self, rec: Dict[str, Any], results: List[Any],
                 on_result: Callable[[str, Optional[dict], Optional[str]], None],
                 on_batch_done: Callable[[], None],
                 unanswered: Dict[str, Dict[str, Any]]) -> None:
        path = Path(rec["path"])
        lines = _read_lines(path)
        by_bcid = {line["batch_custom_id"]: line for line in lines}
        seen: Set[str] = set()
        failed: Set[str] = set()
        for res in results:
            bcid = _get(res, "custom_id")
            line = by_bcid.get(bcid)
            if line is None or bcid in seen:
                continue
            seen.add(bcid)
            result = _get(res, "result")
            if _get(result, "type") == "succeeded":
                on_result(line["custom_id"], _to_dict(_get(result, "message")), None)
            else:
                failed.add(bcid)
                err = _get(result, "error")
                on_result(line["custom_id"], None, f"{_get(result, 'type')}: {_to_dict(err)}"[:200])
        missing = set(by_bcid) - seen
        if failed or missing:
            self.log.warning(f"batch {rec['batch_id']}: {len(failed)} not succeeded, {len(missing)} not "
                             f"answered, of {len(lines)}")
        for bcid in failed | missing:
            unanswered[by_bcid[bcid]["custom_id"]] = by_bcid[bcid]["params"]
        on_batch_done()
        rec.update(collected=True, collected_at=_now(), n_answered=len(seen) - len(failed))
        self._save_state()
        path.unlink(missing_ok=True)

    async def run(self, chunks: List[Path],
                  on_result: Callable[[str, Optional[dict], Optional[str]], None],
                  on_batch_done: Callable[[], None] = lambda: None) -> Dict[str, Dict[str, Any]]:
        """Submit ``chunks`` (and collect any batches :meth:`resume` found in
        flight) and wait until every batch has been collected.  Returns the
        requests no batch answered, as ``custom_id -> params``."""
        pending: Deque[Path] = deque(chunks)
        inflight: Dict[str, Dict[str, Any]] = {r["batch_id"]: r for r in self._inflight_records()}
        unanswered: Dict[str, Dict[str, Any]] = {}
        failed_rounds = 0
        while pending:
            path = pending.popleft()
            batch = await self._submit(path)
            inflight[_get(batch, "id")] = self._record(path)
        while inflight:
            await asyncio.sleep(self.poll_seconds)
            n_ok = 0
            last_error: Optional[BaseException] = None
            for batch_id in list(inflight):
                try:
                    batch = await _maybe_await(self.client.messages.batches.retrieve(batch_id))
                    status = _get(batch, "processing_status")
                    results = await self._results(batch_id) if status == "ended" else None
                except Exception as e:  # noqa: BLE001 -- tried again at the next poll
                    last_error = e
                    self.log.warning(f"batch {batch_id}: status check or download failed "
                                     f"({type(e).__name__}: {e}); trying again at the next poll")
                    continue
                n_ok += 1
                if results is None:
                    counts = _get(batch, "request_counts")
                    self.log.info(f"batch {batch_id}: {status} ({_get(counts, 'succeeded', 0)} succeeded, "
                                  f"{_get(counts, 'processing', '?')} processing)")
                    continue
                self._collect(inflight.pop(batch_id), results, on_result, on_batch_done, unanswered)
            if last_error is not None and n_ok == 0:
                failed_rounds += 1
                if failed_rounds >= self.max_failed_poll_rounds:
                    self.log.error(f"{failed_rounds} poll rounds in a row failed; stopping (the batches stay "
                                   f"recorded, and a rerun collects them)")
                    raise last_error
            else:
                failed_rounds = 0
        return unanswered
