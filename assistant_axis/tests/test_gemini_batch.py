"""Tests for assistant_axis/gemini_batch.py (2026-11-03): the Gemini Batch API runner.

No network: the client is a fake with httpx's ``post``/``get`` shape.  The fake follows Google's
batch guide (aistudio.google.com/docs/batch-api, read 2026-11-03), and each variant the guide shows
(results under ``response`` or ``dest``, ``JOB_STATE_*`` or ``BATCH_STATE_*``, inline results or a
responses file) is exercised.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from assistant_axis import gemini_batch as gb
from assistant_axis.gemini_batch import GeminiBatchError, GeminiBatchRunner

MODEL = "gemini-3.1-flash-lite"


def _resp(status, data=None, text=None):
    return SimpleNamespace(status_code=status, text=text if text is not None else json.dumps(data or {}),
                           json=lambda: data)


def _answer(prompt):
    return {"candidates": [{"content": {"parts": [{"text": f"re: {prompt}"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}}


class FakeBatchClient:
    """Batches finish at their second status check.  ``shape`` picks where results go: ``response``
    (``response.inlinedResponses.inlinedResponses``), ``dest`` (``dest.inlinedResponses``), ``nokeys``
    (as ``response`` without metadata keys) or ``file`` (a responses file)."""

    def __init__(self, *, shape="response", prefix="BATCH", create_codes=(), errored=(), end_state="SUCCEEDED"):
        self.shape, self.prefix, self.create_codes = shape, prefix, list(create_codes)
        self.errored, self.end_state = set(errored), end_state
        self.batches, self.posts, self.gets, self.files = {}, [], [], {}

    async def post(self, path, json):
        self.posts.append((path, json))
        if self.create_codes:
            code = self.create_codes.pop(0)
            return _resp(code, {"error": {"code": code, "message": "no"}})
        assert path == f"/models/{MODEL}:batchGenerateContent"
        name = f"batches/b{len(self.batches)}"
        reqs = json["batch"]["input_config"]["requests"]["requests"]
        self.batches[name] = {"display": json["batch"]["display_name"], "requests": reqs, "checks": 0}
        return _resp(200, {"name": name, "metadata": {"state": f"{self.prefix}_STATE_PENDING"}})

    async def get(self, path, params=None):
        self.gets.append((path, params))
        if path == "/batches":
            ops = [{"name": n, "metadata": {"displayName": b["display"]}} for n, b in self.batches.items()]
            return _resp(200, {"operations": ops})
        if path.startswith(gb.DOWNLOAD_BASE_URL):
            return _resp(200, None, text=self.files[path[len(gb.DOWNLOAD_BASE_URL) + 1:-len(":download")]])
        b = self.batches[path[1:]]
        b["checks"] += 1
        if b["checks"] < 2:
            return _resp(200, {"name": path[1:], "metadata": {"state": f"{self.prefix}_STATE_RUNNING",
                                                              "batchStats": {"pendingRequestCount": len(b["requests"])}}})
        items = []
        for r in b["requests"]:
            prompt = r["request"]["contents"][0]["parts"][0]["text"]
            item = {"error": {"code": 13, "message": "internal"}} if prompt in self.errored else {"response": _answer(prompt)}
            if self.shape != "nokeys":
                item["metadata"] = {"key": r["metadata"]["key"]}
            items.append(item)
        op = {"name": path[1:], "done": True, "metadata": {"state": f"{self.prefix}_STATE_{self.end_state}"}}
        if self.end_state != "SUCCEEDED":
            op["error"] = {"code": 4, "message": "deadline"}
        elif self.shape == "dest":
            op["dest"] = {"inlinedResponses": items}
        elif self.shape == "file":
            fname = f"files/out-{path[1:].split('/')[1]}"
            self.files[fname] = "\n".join(json.dumps({"key": it.pop("metadata")["key"], **it}) for it in items)
            op["response"] = {"responsesFile": fname}
        else:
            op["response"] = {"inlinedResponses": {"inlinedResponses": items}}
        return _resp(200, op)


def _body(prompt):
    return {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 9}}


def _runner(tmp_path, client, **kw):
    return GeminiBatchRunner(client, tmp_path / "cell" / "chunks", tmp_path / "cell" / "batch_state.json",
                             model=MODEL, poll_seconds=0, **kw)


def _go(runner, requests, **chunk_kw):
    got = {}

    def on_result(cid, body, error):
        got[cid] = (body, error)
    chunks = runner.write_chunks(requests, **chunk_kw)
    unanswered = asyncio.run(runner.run(chunks, on_result))
    return got, unanswered


REQS = [(f"a0|e{i}", _body(f"p{i}")) for i in range(5)]


@pytest.mark.parametrize("shape,prefix", [("response", "BATCH"), ("dest", "JOB"), ("nokeys", "BATCH"), ("file", "JOB")])
def test_every_result_shape_comes_back_to_its_request(tmp_path, shape, prefix):
    client = FakeBatchClient(shape=shape, prefix=prefix)
    got, unanswered = _go(_runner(tmp_path, client), REQS)
    assert not unanswered
    assert {cid: body["candidates"][0]["content"]["parts"][0]["text"] for cid, (body, _) in got.items()} == \
        {f"a0|e{i}": f"re: p{i}" for i in range(5)}
    (path, body), = client.posts
    reqs = body["batch"]["input_config"]["requests"]["requests"]
    assert [r["metadata"]["key"] for r in reqs] == [f"r{i:06d}" for i in range(5)]
    assert reqs[0]["request"] == _body("p0") and body["batch"]["display_name"].startswith("judge-tier cell chunk_")
    state = json.loads((tmp_path / "cell" / "batch_state.json").read_text())
    assert state["provider"] == "gemini" and state["chunks"][0]["collected"] is True


def test_errored_items_and_failed_batches_come_back_unanswered(tmp_path):
    got, unanswered = _go(_runner(tmp_path, FakeBatchClient(errored={"p1", "p3"})), REQS)
    assert sorted(unanswered) == ["a0|e1", "a0|e3"] and unanswered["a0|e1"] == _body("p1")
    assert got["a0|e1"][0] is None and "internal" in got["a0|e1"][1]
    _, unanswered = _go(_runner(tmp_path / "x", FakeBatchClient(end_state="EXPIRED")), REQS)
    assert sorted(unanswered) == [cid for cid, _ in REQS]


def test_chunks_split_by_bytes_and_requests(tmp_path):
    client = FakeBatchClient()
    runner = _runner(tmp_path, client)
    paths = runner.write_chunks(REQS, max_requests=2)
    assert [len(p.read_text().splitlines()) for p in paths] == [2, 2, 1]
    assert json.loads(paths[0].read_text().splitlines()[0])["model"] == MODEL
    assert runner.inflight_models() == {}
    one = len(paths[0].read_bytes().splitlines(keepends=True)[0])
    paths = _runner(tmp_path / "y", client).write_chunks(REQS, max_bytes=one * 2 + 1)
    assert len(paths) == 3


def test_create_waits_out_a_429_and_stops_on_402(tmp_path):
    client = FakeBatchClient(create_codes=[429])
    got, unanswered = _go(_runner(tmp_path, client), REQS)
    assert len(got) == 5 and not unanswered and len(client.posts) == 2
    client = FakeBatchClient(create_codes=[402])
    runner = _runner(tmp_path / "z", client)
    with pytest.raises(GeminiBatchError, match="topped up"):
        _go(runner, REQS)
    assert not runner._inflight_records() and "submitting_ts" not in runner.state["chunks"][0]


def test_a_batch_created_but_not_recorded_is_found_by_its_name(tmp_path):
    client = FakeBatchClient()
    runner = _runner(tmp_path, client)
    path, = runner.write_chunks(REQS)

    async def lost():               # the run stops between the create call and the state write
        rec = runner._record(path)
        rec["display_name"] = "judge-tier cell lost 1234"
        rec["submitting_ts"] = 1.0
        runner._save_state()
        body = {"batch": {"display_name": rec["display_name"], "input_config": {"requests": {"requests": [
            {"request": r, "metadata": {"key": f"r{i:06d}"}} for i, (_, r) in enumerate(REQS)]}}}}
        await client.post(f"/models/{MODEL}:batchGenerateContent", json=body)
    asyncio.run(lost())
    again = _runner(tmp_path, client)
    assert asyncio.run(again.resume()) == {cid for cid, _ in REQS}
    assert again.inflight_models() == {MODEL: [path.name]}
    got = {}
    assert asyncio.run(again.run([], lambda cid, body, err: got.setdefault(cid, body))) == {}
    assert len(got) == 5 and len(client.batches) == 1        # collected, not created twice

    # a submission that left no batch is dropped and sent again
    client2 = FakeBatchClient()
    runner2 = _runner(tmp_path / "w", client2)
    p2, = runner2.write_chunks(REQS)
    runner2._record(p2).update(display_name="never created", submitting_ts=1.0)
    runner2._save_state()
    assert asyncio.run(_runner(tmp_path / "w", client2).resume()) == set() and not p2.exists()


def test_a_status_check_failure_is_tried_again(tmp_path):
    client = FakeBatchClient()
    real_get = client.get
    fails = {"n": 2}

    async def flaky(path, params=None):
        if path.startswith("/batches/") and fails["n"] > 0:
            fails["n"] -= 1
            return _resp(503, {"error": "busy"})
        return await real_get(path, params)
    client.get = flaky
    got, unanswered = _go(_runner(tmp_path, client), REQS)
    assert len(got) == 5 and not unanswered
