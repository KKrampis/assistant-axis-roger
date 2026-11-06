"""Gemini Batch API (``batchGenerateContent``) for the judge-tier tool's batch transport.

The counterpart of :class:`assistant_axis.anthropic_batch.AnthropicBatchRunner`, with the same
interface (``resume``, ``inflight_models``, ``write_chunks``, ``run``), so that ``run_batch`` in
``results_analysis/judge_tier_cost_eval.py`` drives it as it drives the other two.  Added 2026-11-03
(Roger: "we're not done experimenting with Gemini, so add batch mode for it"): half the standard
price (Gemini 3.1 Flash-Lite $0.125 / $0.75 per 1M against $0.25 / $1.50; paid tier only), a target
of 24 hours, and a batch still pending or running after 48 hours expires without results
(aistudio.google.com/docs/batch-api, read 2026-11-03).

Differences from the Anthropic runner:

* **Requests go inline** in the create call, ``input_config.requests.requests[]``, each
  ``{"request": <generateContent body>, "metadata": {"key": ...}}``.  Google allows under 20 MB of
  inline requests, so a chunk holds at most :data:`DEFAULT_CHUNK_BYTES` of request JSON.  The file
  route (an upload through the File API, up to 2 GB) is not wired.
* **The model is in the URL**, not the request, so each chunk line records it.
* **A lost submission is found by name.**  Every batch is created with a display name made unique
  by a random token and recorded before the call, so a batch created but not recorded (the run
  stopped between the create call and the state write) is found exactly by listing the account's
  batches, where the Anthropic runner can only match by time and size.  Creating a batch is not
  idempotent; when no batch has the name, the chunk is sent again.
* **Reading the result defensively.**  Google's guide shows the finished batch's results both as
  ``response.inlinedResponses`` and as ``dest.inlinedResponses``, and its states both as
  ``JOB_STATE_*`` and ``BATCH_STATE_*``: all are read, states by their suffix.  Each result is
  matched by its ``metadata.key``, or by position when no result carries a key.  A batch that names
  a responses file instead is downloaded and read line by line.  Seen live (2026-11-03, one request,
  gemini-3.1-flash-lite): ``response.inlinedResponses.inlinedResponses[]`` of ``{"response",
  "metadata": {"key"}}``, the same list under ``metadata.output``, and ``metadata.state``
  ``BATCH_STATE_SUCCEEDED``; done in 1 min 47 s.
* **Explicit caches** (2026-11-05, ``cache_create`` and ``cache_delete`` given): a request written
  with a prefix (``write_chunks`` items ``(custom_id, body, prefix)``, the body carrying only the
  prompt's suffix) goes on an explicit cache of that prefix.  The chunk line keeps the prefix's
  digest and the chunk's prefixes go in a side file, ``<chunk>.prefixes.json``, so that the inline
  submission carries the suffixes only (far more requests fit the 15 MB).  Just before a chunk is
  submitted its prefixes' caches are created (a cache still alive from an earlier chunk is reused)
  and recorded in the state file's ``caches``, by digest, with every chunk and custom id that uses
  it, and the chunk's lines are rewritten with the caches' names; a prefix whose cache cannot be
  created goes whole instead (logged), and lines that would then pass the inline limit go into a
  chunk of their own.  A cache is deleted once every chunk that uses it is collected, its creation
  and storage handed to ``on_cache_done``; :meth:`resume` deletes the caches no batch in flight
  needs any more.  The TTL (``cache_ttl_s``, six hours by default) is the backstop.  Requests a
  batch leaves unanswered are returned whole, never on a cache.
* **HTTP.**  A create answered 429 is tried again after ``poll_seconds``, up to
  ``max_create_retries`` times; 402 (prepaid credit used up) and any other refusal raise
  :class:`GeminiBatchError`.  A failed status check is tried again at the next poll, as in the
  Anthropic runner.

Results are handed to ``on_result`` as the generateContent response dicts
(``judge.gemini_reply`` reads them).  Only the standard library and ``atomic_io`` are imported; the
client (``judge.make_gemini_client``: httpx with the v1beta base URL and the key header) is passed in.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import logging
import secrets
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

from .atomic_io import atomic_write_text

#: Google: "under 20MB" of inline requests per batch; a chunk stays well below it.
DEFAULT_CHUNK_BYTES = 15 * 1024 * 1024
#: What a prepared chunk (explicit caches' names in, a failed cache's requests whole) may hold inline; the rest goes
#: in chunks of their own.
INLINE_LIMIT_BYTES = DEFAULT_CHUNK_BYTES
DEFAULT_CHUNK_REQUESTS = 10_000
MAX_FAILED_POLL_ROUNDS = 30
MAX_CREATE_RETRIES = 60
LIST_MAX_PAGES = 20
DOWNLOAD_BASE_URL = "https://generativelanguage.googleapis.com/download/v1beta"
TERMINAL_STATES = ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")
#: An explicit cache's TTL in a batch run: long enough for most batches (the usual wait is minutes to an
#: hour), short enough that a cache a stopped run leaves behind costs little (storage is billed by the hour
#: until it is deleted or expires).  A request whose cache has expired comes back as an error and is judged
#: in real time.
DEFAULT_CACHE_TTL_S = 6 * 3600.0
DEFAULT_CACHE_CONCURRENCY = 16


def prefix_digest(prefix: str) -> str:
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:24]


def full_request(request: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """A request on an explicit cache made whole: the prefix and the suffix joined in one user turn, as
    ``judge.gemini_request_body`` sends a prompt without one, the cache's name dropped."""
    out = {k: v for k, v in request.items() if k != "cachedContent"}
    suffix = "".join(part.get("text", "") for c in request.get("contents", []) for part in c.get("parts", []))
    out["contents"] = [{"parts": [{"text": prefix + suffix}]}]
    return out


def _prefixes_path(path: Path) -> Path:
    return path.with_name(path.stem + ".prefixes.json")
STATE_VERSION = 1


class GeminiBatchError(RuntimeError):
    """The Batch API refused a request in a way no retry within the run gets past."""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _read_lines(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _field(d: Any, *names: str) -> Any:
    """The first of ``names`` (camelCase as the API writes, snake_case as its guide does) in ``d``."""
    if not isinstance(d, dict):
        return None
    for name in names:
        if name in d:
            return d[name]
    return None


def batch_state(op: Dict[str, Any]) -> str:
    """The state's suffix (``SUCCEEDED``, ``RUNNING``, ...) of a batch as the API returns it, or ``""``."""
    state = _field(_field(op, "metadata") or {}, "state") or _field(op, "state") or ""
    return str(state).rsplit("_STATE_", 1)[-1]


def batch_output(op: Dict[str, Any]) -> Dict[str, Any]:
    """The output record of a finished batch, wherever the API puts it."""
    for holder in (_field(op, "response"), _field(op, "dest"), _field(op, "output"),
                   _field(_field(op, "metadata") or {}, "output")):
        if isinstance(holder, dict) and holder:
            return holder
    return {}


def inlined_responses(output: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The inline results of a batch output (the list itself, or its ``inlinedResponses`` wrapper)."""
    x = _field(output, "inlinedResponses", "inlined_responses")
    if isinstance(x, dict):
        x = _field(x, "inlinedResponses", "inlined_responses")
    return x if isinstance(x, list) else None


def _result_key(item: Dict[str, Any]) -> Optional[str]:
    return _field(_field(item, "metadata") or {}, "key") or _field(item, "key")


def _status_text(resp: Any) -> str:
    return f"HTTP {resp.status_code}: {str(getattr(resp, 'text', ''))[:300]}"


class GeminiBatchRunner:
    """Write, submit, poll and collect Gemini batches for one output directory; see
    :class:`assistant_axis.openai_batch.OpenAIBatchRunner` for the contract of
    ``on_result(custom_id, body, error)``, ``on_batch_done()`` and the requests :meth:`run`
    returns unanswered."""

    def __init__(self, client: Any, work_dir: Path, state_path: Path, *, model: str,
                 poll_seconds: float = 60.0,
                 max_failed_poll_rounds: int = MAX_FAILED_POLL_ROUNDS,
                 max_create_retries: int = MAX_CREATE_RETRIES,
                 logger: Optional[logging.Logger] = None,
                 cache_create: Optional[Callable[[str, float], Awaitable[Optional[Tuple[str, int]]]]] = None,
                 cache_delete: Optional[Callable[[str], Awaitable[bool]]] = None,
                 cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
                 on_cache_done: Optional[Callable[[List[str], Dict[str, Any]], None]] = None,
                 cache_concurrency: int = DEFAULT_CACHE_CONCURRENCY):
        """``cache_create(prefix, ttl_s)`` gives ``(name, tokens)`` or None, ``cache_delete(name)`` whether it
        went (the tool passes ``judge.gemini_create_cache`` and ``judge.gemini_delete_cache``); ``on_cache_done(
        custom_ids, record)`` receives each cache once it is deleted, ``record`` with its ``tokens``, its
        ``storage_token_hours`` and its ``custom_ids``, so the caller can charge it to those requests' cells."""
        self.client = client
        self.model = model
        self.work_dir = Path(work_dir)
        self.state_path = Path(state_path)
        self.poll_seconds = poll_seconds
        self.max_failed_poll_rounds = max_failed_poll_rounds
        self.max_create_retries = max_create_retries
        self.log = logger or logging.getLogger(__name__)
        self.cache_create, self.cache_delete = cache_create, cache_delete
        self.cache_ttl_s, self.on_cache_done = float(cache_ttl_s), on_cache_done
        self.cache_concurrency = max(1, int(cache_concurrency))
        self.state: Dict[str, Any] = self._load_state()

    # -- state ------------------------------------------------------------

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get("version") != STATE_VERSION:
                raise RuntimeError(f"{self.state_path}: unknown state version {state.get('version')!r}")
            return state
        return {"version": STATE_VERSION, "provider": "gemini", "chunks": []}

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

    async def _find_batch(self, display_name: str) -> Optional[str]:
        """The name (``batches/...``) of the account's batch with ``display_name``, or None."""
        token: Optional[str] = None
        for _ in range(LIST_MAX_PAGES):
            resp = await self.client.get("/batches", params={"pageSize": 100, **({"pageToken": token} if token else {})})
            if resp.status_code != 200:
                raise GeminiBatchError(f"listing batches: {_status_text(resp)}")
            data = resp.json()
            for op in (_field(data, "operations", "batches") or []):
                meta = _field(op, "metadata") or {}
                if (_field(meta, "displayName", "display_name") or _field(op, "displayName", "display_name")) == display_name:
                    return _field(op, "name")
            token = _field(data, "nextPageToken", "next_page_token")
            if not token:
                return None
        return None

    async def resume(self) -> Set[str]:
        """Prepare a restart: adopt a batch created but not recorded (found by its display name),
        drop chunks never submitted, and return the ``custom_id`` of every request in a batch still
        to be collected (the caller must not submit those again)."""
        for rec in [r for r in self.state["chunks"] if r.get("submitting_ts") and not r.get("batch_id")]:
            try:
                name = await self._find_batch(rec["display_name"]) if rec.get("display_name") else None
            except Exception as e:  # noqa: BLE001 -- then the chunk is sent again
                self.log.warning(f"{Path(rec['path']).name}: could not list batches ({type(e).__name__}: {e})")
                name = None
            if name:
                self.log.warning(f"{Path(rec['path']).name}: batch {name} was created before the last run "
                                 f"stopped; collecting it instead of submitting the chunk again")
                rec.update(batch_id=name, submitted_at=_now(), recovered=True)
        keep = []
        for rec in self.state["chunks"]:
            if rec.get("batch_id"):
                keep.append(rec)
            else:
                Path(rec["path"]).unlink(missing_ok=True)
                _prefixes_path(Path(rec["path"])).unlink(missing_ok=True)
        self.state["chunks"] = keep
        await self._release_caches(None, at_resume=True)
        ids: Set[str] = set()
        for rec in self._inflight_records():
            path = Path(rec["path"])
            if not path.exists():
                raise RuntimeError(f"batch {rec['batch_id']}: chunk file {path} is gone, so its results "
                                   f"cannot be mapped back to custom ids")
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
                    model = str(lines[0].get("model"))
            except (OSError, ValueError):
                pass
            out.setdefault(model, []).append(path.name)
        return out

    # -- chunk files --------------------------------------------------------

    def write_chunks(self, requests: Iterable[Tuple[Any, ...]], *,
                     max_requests: int = DEFAULT_CHUNK_REQUESTS,
                     max_bytes: int = DEFAULT_CHUNK_BYTES) -> List[Path]:
        """Stream ``(custom_id, generateContent body)`` pairs into chunk files of at most
        ``max_requests`` lines and ``max_bytes`` bytes; return their paths.  An item
        ``(custom_id, body, prefix)`` with a prefix goes on an explicit cache of it (the body
        holding the suffix only; the module docstring), its line counting without the prefix."""
        max_bytes = min(max_bytes, DEFAULT_CHUNK_BYTES)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        known = {Path(r["path"]).name for r in self.state["chunks"]}
        base = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        stamp, k = base, 0
        while any(n.startswith(f"chunk_{stamp}_") for n in known) or any(self.work_dir.glob(f"chunk_{stamp}_*")):
            k += 1
            stamp = f"{base}-{k}"

        def encode(custom_id: str, i: int, request: Dict[str, Any], digest: Optional[str]) -> bytes:
            line = {"custom_id": custom_id, "key": f"r{i:06d}", "model": self.model, "request": request}
            if digest is not None:
                line["prefix"] = digest
            return (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8")

        paths: List[Path] = []
        f = None
        n = size = 0
        prefixes: Dict[str, str] = {}

        def close() -> None:
            if f is not None:
                f.close()
                if prefixes:
                    _prefixes_path(paths[-1]).write_text(json.dumps(prefixes, ensure_ascii=False), encoding="utf-8")

        for item in requests:
            custom_id, request = item[0], item[1]
            prefix = item[2] if len(item) > 2 else None
            if prefix is not None and self.cache_create is None:
                raise ValueError("a request with a prefix to cache needs a runner with cache_create")
            digest = prefix_digest(prefix) if prefix is not None else None
            line = encode(custom_id, n, request, digest)
            if f is None or n >= max_requests or (n and size + len(line) > max_bytes):
                close()
                path = self.work_dir / f"chunk_{stamp}_{len(paths):04d}.jsonl"
                paths.append(path)
                f = open(path, "wb")
                n = size = 0
                prefixes = {}
                line = encode(custom_id, 0, request, digest)
            if digest is not None:
                prefixes[digest] = prefix
            f.write(line)
            n += 1
            size += len(line)
        close()
        for path in paths:
            self._record(path)["n_requests"] = len(_read_lines(path))
        self._save_state()
        return paths

    # -- explicit caches (2026-11-05) -------------------------------------

    def _caches(self) -> Dict[str, Dict[str, Any]]:
        return self.state.setdefault("caches", {})

    def _alive(self, c: Dict[str, Any], margin_s: float = 600.0) -> bool:
        return bool(c.get("name")) and not c.get("deleted") and c["created"] + c["ttl_s"] > time.time() + margin_s

    async def _prepare(self, path: Path) -> List[Path]:
        """Before ``path`` is submitted: create the caches of its prefixes (reusing any still alive), rewrite
        its lines with their names (a prefix whose cache failed, whole), and move what then passes the inline
        limit into new chunks, returned for submission after it."""
        lines = _read_lines(path)
        wanted = sorted({line["prefix"] for line in lines if line.get("prefix") and not line.get("prepared")})
        if not wanted:
            return []
        prefixes = json.loads(_prefixes_path(path).read_text(encoding="utf-8"))
        caches = self._caches()
        sem = asyncio.Semaphore(self.cache_concurrency)

        async def make(digest: str) -> None:
            c = caches.get(digest)
            if c is not None and self._alive(c):
                return
            async with sem:
                made = await self.cache_create(prefixes[digest], self.cache_ttl_s)
            if made is None:
                caches[digest] = {"name": None, "failed": True, "chunks": [], "custom_ids": []}
            else:
                name, tokens = made
                caches[digest] = {"name": name, "tokens": int(tokens), "created": time.time(),
                                  "ttl_s": self.cache_ttl_s, "chunks": [], "custom_ids": []}
            self._save_state()      # every name on record at once, so a stopped run can delete it

        await asyncio.gather(*(make(d) for d in wanted))
        failed = [d for d in wanted if not caches[d].get("name")]
        if failed:
            self.log.warning(f"{path.name}: {len(failed)} of {len(wanted)} explicit caches not created; those "
                             f"prefixes' requests go whole")
        keep: List[Dict[str, Any]] = []
        overflow: List[Tuple[str, Dict[str, Any]]] = []
        size = 0
        for line in lines:
            d = line.get("prefix")
            if d and not line.get("prepared"):
                c = caches[d]
                if c.get("name"):
                    line["request"] = {**line["request"], "cachedContent": c["name"]}
                    c["chunks"] = sorted(set(c["chunks"]) | {str(path)})
                    c["custom_ids"] = sorted(set(c["custom_ids"]) | {line["custom_id"]})
                else:
                    line["request"] = full_request(line["request"], prefixes[d])
                line["prepared"] = True
            encoded = len(json.dumps(line, ensure_ascii=False).encode("utf-8")) + 1
            if keep and size + encoded > INLINE_LIMIT_BYTES:
                overflow.append((line["custom_id"], line["request"]))
                continue
            keep.append(line)
            size += encoded
        atomic_write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in keep), path,
                          logger_obj=self.log)
        self._record(path)["n_requests"] = len(keep)
        self._save_state()
        if not overflow:
            return []
        self.log.warning(f"{path.name}: {len(overflow)} requests sent whole no longer fit the inline limit; "
                         f"they go in a chunk of their own")
        return self.write_chunks(overflow)

    async def _release_caches(self, collecting: Optional[Dict[str, Any]], *, at_resume: bool = False) -> None:
        """Delete every cache no chunk still to be collected uses (``collecting``, the chunk being collected,
        counts as collected), and hand each to ``on_cache_done`` (at a resume only logged: the cells its
        requests were charged to are written or not loaded)."""
        caches = self.state.get("caches") or {}
        if not caches or self.cache_delete is None:
            return
        open_paths = {r["path"] for r in self.state["chunks"]
                      if not r.get("collected") and r is not collecting}
        done = [(d, c) for d, c in caches.items()
                if c.get("name") and not c.get("deleted") and not (set(c.get("chunks") or []) & open_paths)]
        if not done:
            return
        sem = asyncio.Semaphore(self.cache_concurrency)

        async def drop(digest: str, c: Dict[str, Any]) -> None:
            async with sem:
                ok = await self.cache_delete(c["name"])
            end = time.time() if ok else c["created"] + c["ttl_s"]      # undeleted: it lives to its TTL
            end = min(end, c["created"] + c["ttl_s"])
            c.update(deleted=True, delete_ok=bool(ok),
                     storage_token_hours=c["tokens"] * max(0.0, end - c["created"]) / 3600.0)
            if at_resume or self.on_cache_done is None:
                self.log.warning(f"cache {c['name']} ({c['tokens']} tokens) left by an earlier run "
                                 f"{'deleted' if ok else 'not deleted (it expires at its TTL)'}; its creation and "
                                 f"{c['storage_token_hours']:.1f} token-hours of storage are in no cell's usage")
            else:
                self.on_cache_done(list(c.get("custom_ids") or []), dict(c))

        await asyncio.gather(*(drop(d, c) for d, c in done))
        self._save_state()

    # -- submission and collection -----------------------------------------

    async def _submit(self, path: Path) -> str:
        rec = self._record(path)
        lines = _read_lines(path)
        model = lines[0]["model"] if lines else self.model
        rec.setdefault("display_name", f"judge-tier {self.work_dir.parent.name} {path.stem} {secrets.token_hex(4)}")
        body = {"batch": {"display_name": rec["display_name"], "input_config": {"requests": {"requests": [
            {"request": line["request"], "metadata": {"key": line["key"]}} for line in lines]}}}}
        for attempt in range(self.max_create_retries + 1):
            rec["submitting_ts"] = time.time()
            self._save_state()
            resp = await self.client.post(f"/models/{model}:batchGenerateContent", json=body)
            if resp.status_code == 200:
                break
            rec.pop("submitting_ts", None)     # refused: no batch was created
            self._save_state()
            if resp.status_code == 429 and attempt < self.max_create_retries:
                self.log.info(f"{path.name}: batch refused for now ({_status_text(resp)}); trying again in "
                              f"{self.poll_seconds:.0f} s")
                await asyncio.sleep(self.poll_seconds)
                continue
            if resp.status_code == 402:
                raise GeminiBatchError(f"{model}: {_status_text(resp)}; a rerun resumes once the credit is topped up")
            raise GeminiBatchError(f"{path.name}: batch create refused: {_status_text(resp)}")
        name = _field(resp.json(), "name")
        if not name:
            raise GeminiBatchError(f"{path.name}: the create answer names no batch: {str(resp.text)[:300]}")
        rec.update(batch_id=name, submitted_at=_now())
        self._save_state()
        self.log.info(f"submitted {path.name}: {name}, {rec.get('n_requests')} requests")
        return name

    async def _results(self, op: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """The result items of a finished batch: inline, or read from its responses file."""
        output = batch_output(op)
        items = inlined_responses(output)
        if items is not None:
            return items
        file_name = _field(output, "responsesFile", "responses_file")
        if not file_name:
            return None
        resp = await self.client.get(f"{DOWNLOAD_BASE_URL}/{file_name}:download", params={"alt": "media"})
        if resp.status_code != 200:
            raise GeminiBatchError(f"downloading {file_name}: {_status_text(resp)}")
        return [json.loads(line) for line in str(resp.text).splitlines() if line.strip()]

    def _collect(self, rec: Dict[str, Any], items: List[Dict[str, Any]],
                 on_result: Callable[[str, Optional[dict], Optional[str]], None],
                 on_batch_done: Callable[[], None],
                 unanswered: Dict[str, Dict[str, Any]]) -> None:
        path = Path(rec["path"])
        lines = _read_lines(path)
        by_key = {line["key"]: line for line in lines}
        keyed = any(_result_key(item) for item in items)
        if not keyed and len(items) != len(lines):
            self.log.warning(f"batch {rec['batch_id']}: {len(items)} results without keys for {len(lines)} "
                             f"requests; none can be matched, all are sent again")
            items = []
        seen: Set[str] = set()
        failed: Set[str] = set()
        for i, item in enumerate(items):
            key = _result_key(item) if keyed else lines[i]["key"]
            line = by_key.get(key)
            if line is None or key in seen:
                continue
            seen.add(key)
            response = _field(item, "response")
            if isinstance(response, dict) and not _field(item, "error"):
                on_result(line["custom_id"], response, None)
            else:
                failed.add(key)
                on_result(line["custom_id"], None, f"error: {json.dumps(_field(item, 'error'))}"[:200])
        missing = set(by_key) - seen
        if failed or missing:
            self.log.warning(f"batch {rec['batch_id']}: {len(failed)} errored, {len(missing)} not answered, "
                             f"of {len(lines)}")
        prefixes: Dict[str, str] = {}
        if _prefixes_path(path).exists():
            prefixes = json.loads(_prefixes_path(path).read_text(encoding="utf-8"))
        for key in failed | missing:
            line = by_key[key]
            request = line["request"]
            if "cachedContent" in request and line.get("prefix") in prefixes:
                request = full_request(request, prefixes[line["prefix"]])     # whole: its cache is going
            unanswered[line["custom_id"]] = request
        on_batch_done()
        rec.update(collected=True, collected_at=_now(), n_answered=len(seen) - len(failed))
        self._save_state()
        path.unlink(missing_ok=True)
        _prefixes_path(path).unlink(missing_ok=True)

    async def run(self, chunks: List[Path],
                  on_result: Callable[[str, Optional[dict], Optional[str]], None],
                  on_batch_done: Callable[[], None] = lambda: None) -> Dict[str, Dict[str, Any]]:
        """Submit ``chunks`` (and collect any batches :meth:`resume` found in flight) and wait
        until every batch has been collected.  Returns the requests no batch answered, as
        ``custom_id -> generateContent body``."""
        pending: Deque[Path] = deque(chunks)
        inflight: Dict[str, Dict[str, Any]] = {r["batch_id"]: r for r in self._inflight_records()}
        unanswered: Dict[str, Dict[str, Any]] = {}
        failed_rounds = 0
        while pending:
            path = pending.popleft()
            if self.cache_create is not None:
                pending.extend(await self._prepare(path))
            inflight[await self._submit(path)] = self._record(path)
        while inflight:
            await asyncio.sleep(self.poll_seconds)
            n_ok = 0
            last_error: Optional[BaseException] = None
            for name in list(inflight):
                try:
                    resp = await self.client.get(f"/{name}")
                    if resp.status_code != 200:
                        raise GeminiBatchError(f"status of {name}: {_status_text(resp)}")
                    op = resp.json()
                    state = batch_state(op)
                    done = bool(_field(op, "done")) or state in TERMINAL_STATES
                    items = await self._results(op) if done and state not in ("FAILED", "CANCELLED", "EXPIRED") else None
                except Exception as e:  # noqa: BLE001 -- tried again at the next poll
                    last_error = e
                    self.log.warning(f"batch {name}: status check or download failed ({type(e).__name__}: {e}); "
                                     f"trying again at the next poll")
                    continue
                n_ok += 1
                if not done:
                    stats = _field(_field(op, "metadata") or {}, "batchStats", "batch_stats") or {}
                    self.log.info(f"batch {name}: {state or 'state unknown'} ({_field(stats, 'successfulRequestCount') or 0} "
                                  f"succeeded, {_field(stats, 'pendingRequestCount') or '?'} pending)")
                    continue
                if items is None:
                    self.log.warning(f"batch {name} ended {state or 'without results'} "
                                     f"({json.dumps(_field(op, 'error'))[:200]}); its requests are unanswered")
                    items = []
                rec = inflight.pop(name)
                await self._release_caches(rec)     # its results are in hand: caches only it still used can go
                self._collect(rec, items, on_result, on_batch_done, unanswered)
            if last_error is not None and n_ok == 0:
                failed_rounds += 1
                if failed_rounds >= self.max_failed_poll_rounds:
                    self.log.error(f"{failed_rounds} poll rounds in a row failed; stopping (the batches stay "
                                   f"recorded, and a rerun collects them)")
                    raise last_error
            else:
                failed_rounds = 0
        return unanswered
