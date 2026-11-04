"""``assistant_axis.anthropic_batch.AnthropicBatchRunner`` (2026-10-26: Haiku 5.5 as a step-3 judge in
batch).  No network: the client is a fake with the async SDK's shapes."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from assistant_axis.anthropic_batch import AnthropicBatchRunner


class _HTTP429(Exception):
    status_code = 429


class FakeBatches:
    """``client.messages.batches``: create / retrieve / results / list.  ``outcome(bcid, params)`` decides
    each request's result: a reply text, or ("errored" | "expired"), or None for no result line."""

    def __init__(self, outcome=None, end_after=1, fail_create=0, fail_retrieve=0):
        self.created, self.batches = [], {}
        self.outcome = outcome or (lambda bcid, params: "SCORE: 3")
        self.end_after, self.fail_create, self.fail_retrieve = end_after, fail_create, fail_retrieve
        self.polls = {}

    async def create(self, requests):
        if self.fail_create:
            self.fail_create -= 1
            raise _HTTP429("queue full")
        bid = f"msgbatch_{len(self.batches)}"
        self.batches[bid] = {"requests": requests, "created_at": datetime.now(timezone.utc)}
        self.created.append(bid)
        return SimpleNamespace(id=bid, processing_status="in_progress")

    async def retrieve(self, bid):
        if self.fail_retrieve:
            self.fail_retrieve -= 1
            raise ConnectionError("network")
        self.polls[bid] = self.polls.get(bid, 0) + 1
        status = "ended" if self.polls[bid] >= self.end_after else "in_progress"
        return SimpleNamespace(id=bid, processing_status=status,
                               request_counts=SimpleNamespace(processing=0, succeeded=0))

    async def results(self, bid):
        out = []
        for r in self.batches[bid]["requests"]:
            o = self.outcome(r["custom_id"], r["params"])
            if o is None:
                continue
            if o in ("errored", "expired"):
                out.append(SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type=o, error=None)))
            else:
                msg = SimpleNamespace(content=[SimpleNamespace(type="text", text=o)], stop_reason="end_turn",
                                      usage=SimpleNamespace(input_tokens=100, output_tokens=10))
                out.append(SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg)))

        async def gen():
            for x in out:
                yield x
        return gen()

    def list(self, limit=100):
        items = [SimpleNamespace(id=bid, created_at=b["created_at"],
                                 request_counts=SimpleNamespace(processing=len(b["requests"]), succeeded=0,
                                                                errored=0, canceled=0, expired=0))
                 for bid, b in reversed(self.batches.items())]

        async def gen():
            for x in items:
                yield x
        return gen()


def _client(**kw):
    return SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches(**kw)))


def _requests(n, model="claude-haiku-5-5"):
    return [(f"patient|pos_p0_q{i}", {"model": model, "max_tokens": 200,
                                      "messages": [{"role": "user", "content": f"prompt {i}"}]}) for i in range(n)]


def _runner(client, tmp_path, **kw):
    return AnthropicBatchRunner(client, tmp_path / "_batches", tmp_path / "batch_state.json", poll_seconds=0, **kw)


def test_results_come_back_under_the_step3_ids(tmp_path):
    client = _client()
    runner = _runner(client, tmp_path)
    chunks = runner.write_chunks(_requests(5), max_requests=3)
    assert len(chunks) == 2
    got, done = {}, []
    unanswered = asyncio.run(runner.run(chunks, lambda cid, body, err: got.__setitem__(cid, (body, err)),
                                        lambda: done.append(1)))
    assert unanswered == {} and len(done) == 2
    assert set(got) == {f"patient|pos_p0_q{i}" for i in range(5)}
    body, err = got["patient|pos_p0_q4"]
    assert err is None and body["content"][0]["text"] == "SCORE: 3" and body["usage"]["input_tokens"] == 100
    sent = client.messages.batches.batches["msgbatch_0"]["requests"]
    assert [r["custom_id"] for r in sent] == ["r000000", "r000001", "r000002"]   # Anthropic-safe ids
    state = json.loads((tmp_path / "batch_state.json").read_text())
    assert all(r["collected"] for r in state["chunks"]) and not any(p.exists() for p in chunks)


def test_errored_expired_and_missing_requests_are_returned_unanswered(tmp_path):
    def outcome(bcid, params):
        return {"r000001": "errored", "r000002": "expired", "r000003": None}.get(bcid, "SCORE: 2")
    runner = _runner(_client(outcome=outcome), tmp_path)
    chunks = runner.write_chunks(_requests(4))
    got = {}
    unanswered = asyncio.run(runner.run(chunks, lambda cid, body, err: got.__setitem__(cid, (body, err))))
    assert set(unanswered) == {"patient|pos_p0_q1", "patient|pos_p0_q2", "patient|pos_p0_q3"}
    assert unanswered["patient|pos_p0_q2"]["messages"][0]["content"] == "prompt 2"
    assert got["patient|pos_p0_q1"][0] is None and got["patient|pos_p0_q1"][1].startswith("errored")
    assert got["patient|pos_p0_q0"][0]["content"][0]["text"] == "SCORE: 2"


def test_a_full_queue_is_tried_again_and_failed_polls_tolerated(tmp_path):
    client = _client(fail_create=2, fail_retrieve=3, end_after=2)
    runner = _runner(client, tmp_path)
    got = {}
    asyncio.run(runner.run(runner.write_chunks(_requests(2)), lambda cid, b, e: got.__setitem__(cid, b)))
    assert len(client.messages.batches.created) == 1 and len(got) == 2


def test_a_create_that_keeps_failing_raises_and_records_nothing_sent(tmp_path):
    runner = _runner(_client(fail_create=5), tmp_path, max_create_retries=2)
    chunks = runner.write_chunks(_requests(2))
    with pytest.raises(_HTTP429):
        asyncio.run(runner.run(chunks, lambda *a: None))
    rec = json.loads((tmp_path / "batch_state.json").read_text())["chunks"][0]
    assert "batch_id" not in rec and "submitting_ts" not in rec


def test_resume_collects_batches_in_flight_and_drops_unsent_chunks(tmp_path):
    client = _client(end_after=99)
    runner = _runner(client, tmp_path)
    sent, unsent = runner.write_chunks(_requests(3)), None
    asyncio.run(runner._submit(sent[0]))                # submitted, never collected (the run stopped)
    unsent = runner.write_chunks(_requests(2))           # written, never submitted
    again = _runner(client, tmp_path)
    ids = asyncio.run(again.resume())
    assert ids == {f"patient|pos_p0_q{i}" for i in range(3)}
    assert not unsent[0].exists() and sent[0].exists()
    assert again.inflight_models() == {"claude-haiku-5-5": [sent[0].name]}


def test_resume_adopts_a_batch_created_but_not_recorded(tmp_path):
    client = _client(end_after=1)
    runner = _runner(client, tmp_path)
    chunk = runner.write_chunks(_requests(3))[0]
    asyncio.run(client.messages.batches.create(          # created on the service ...
        requests=[{"custom_id": f"r{i:06d}", "params": p} for i, (_, p) in enumerate(_requests(3))]))
    runner._record(chunk)["submitting_ts"] = time.time()  # ... but the run stopped before recording it
    runner._save_state()
    again = _runner(client, tmp_path)
    ids = asyncio.run(again.resume())
    assert ids == {f"patient|pos_p0_q{i}" for i in range(3)}
    rec = again.state["chunks"][0]
    assert rec["batch_id"] == "msgbatch_0" and rec["recovered"]
    got = {}
    asyncio.run(again.run([], lambda cid, b, e: got.__setitem__(cid, b)))
    assert len(got) == 3 and len(client.messages.batches.created) == 1   # not submitted twice


def test_resume_leaves_a_lost_batch_alone_when_told_to(tmp_path):
    """Another run's batch of the same size looks just like a lost submission (2026-11-02: sibling judging
    cells of exactly 870 requests each, seven uploads lost among them)."""
    client = _client(end_after=1)
    runner = _runner(client, tmp_path)
    chunk = runner.write_chunks(_requests(3))[0]
    asyncio.run(client.messages.batches.create(          # a sibling's batch, the only one of this size
        requests=[{"custom_id": f"r{i:06d}", "params": p} for i, (_, p) in enumerate(_requests(3))]))
    runner._record(chunk)["submitting_ts"] = time.time()
    runner._save_state()
    again = _runner(client, tmp_path)
    again.adopt_lost_batches = False
    assert asyncio.run(again.resume()) == set()
    assert again.state["chunks"] == [] and not chunk.exists()


def test_resume_sends_again_when_no_single_batch_matches(tmp_path):
    client = _client()
    runner = _runner(client, tmp_path)
    chunk = runner.write_chunks(_requests(3))[0]
    runner._record(chunk)["submitting_ts"] = time.time()
    runner._save_state()
    assert asyncio.run(_runner(client, tmp_path).resume()) == set()
    assert not chunk.exists()
