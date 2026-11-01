"""Tests for judge_tier_cost_eval.py's transports (Flex, batch, real time) and
judge_tier_transport.py (2026-11-01).

No network: every client is a fake with the async SDKs' shapes, and an autouse
fixture makes the script's client factories fail unless a test replaces them.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from assistant_axis.judge import RateLimiter
from assistant_axis.judge_pricing import MultiModelUsage, price_for_model
from results_analysis import judge_tier_cost_eval as ev
from results_analysis import judge_tier_transport as tt

REPO = Path(__file__).resolve().parents[2]

# The fake judges answer by entity: the score each entity gets on every axis.
SCORES = {"calm": 2, "bold": -1, "shy": 3, "kind": 0, "pilot|R": -3}
ENTITIES = sorted(SCORES)
_NAME_RE = re.compile(r"to score\n\n\*\*(.+?)\*\*: ")


def reply_for(prompt: str):
    name = _NAME_RE.search(prompt).group(1)
    return f"It leans one way.\nSCORE: {SCORES[name]}", name


class HTTPError(Exception):
    """An SDK error: both SDKs set ``status_code``; OpenAI's also ``param`` and ``code``."""

    def __init__(self, status_code, message, *, param=None, code=None):
        super().__init__(message)
        self.status_code, self.message, self.param, self.code, self.body = status_code, message, param, code, None


class FakeOpenAI:
    """``openai.AsyncOpenAI``: chat.completions.create, and the Batch API's
    files and batches (a batch completes at its first retrieve)."""

    def __init__(self, *, flex_error=None, flex_fail_times=None, error_names=(), fail_retrieve=False,
                 prompt_tokens=300, completion_tokens=40):
        self.flex_error, self.flex_fail_times = flex_error, flex_fail_times
        self.error_names, self.fail_retrieve = set(error_names), fail_retrieve
        self.pt, self.ct = prompt_tokens, completion_tokens
        self.calls, self.events, self.deleted = [], [], []
        self.uploaded, self._batches, self.batches_created = {}, {}, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.files = SimpleNamespace(create=self._fcreate, content=self._fcontent, delete=self._fdelete)
        self.batches = SimpleNamespace(create=self._bcreate, retrieve=self._bretrieve, list=self._blist)

    async def _create(self, **kw):
        self.calls.append(kw)
        if kw.get("service_tier") == "flex" and self.flex_error is not None:
            if self.flex_fail_times is None or self.flex_fail_times > 0:
                if self.flex_fail_times is not None:
                    self.flex_fail_times -= 1
                raise self.flex_error
        text, _ = reply_for(kw["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=self.pt, completion_tokens=self.ct))

    async def _fcreate(self, file, purpose):
        assert purpose == "batch"
        fid = f"file-{len(self.uploaded)}"
        self.uploaded[fid] = file[1].decode("utf-8")
        return SimpleNamespace(id=fid)

    async def _fcontent(self, fid):
        return SimpleNamespace(text=self.uploaded[fid])

    async def _fdelete(self, fid):
        self.deleted.append(fid)

    def lines(self, fid):
        return [json.loads(x) for x in self.uploaded[fid].splitlines() if x.strip()]

    def batch_lines(self, i):
        return self.lines(self._batches[self.batches_created[i]]["input"])

    async def _bcreate(self, input_file_id, endpoint, completion_window, metadata):
        assert endpoint == "/v1/chat/completions" and completion_window == "24h"
        bid = f"batch-{len(self._batches)}"
        self._batches[bid] = {"input": input_file_id, "status": "in_progress", "created_at": int(time.time())}
        self.batches_created.append(bid)
        self.events.append(("create", bid))
        return SimpleNamespace(id=bid, status="in_progress", errors=None)

    async def _blist(self, limit=20):
        for bid in reversed(list(self._batches)):
            b = self._batches[bid]
            yield SimpleNamespace(id=bid, status=b["status"], input_file_id=b["input"], created_at=b["created_at"])

    async def _bretrieve(self, bid):
        self.events.append(("retrieve", bid))
        if self.fail_retrieve:
            raise ConnectionError("network unreachable")
        b = self._batches[bid]
        if b["status"] == "in_progress":
            out = []
            for line in self.lines(b["input"]):
                text, name = reply_for(line["body"]["messages"][0]["content"])
                if name in self.error_names:
                    out.append({"custom_id": line["custom_id"], "error": None,
                                "response": {"status_code": 500, "body": {"error": {"message": "server"}}}})
                else:
                    out.append({"custom_id": line["custom_id"], "error": None, "response": {"status_code": 200, "body": {
                        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": self.pt, "completion_tokens": self.ct}}}})
            ofid = f"file-out-{bid}"
            self.uploaded[ofid] = "\n".join(json.dumps(o) for o in out) + "\n"
            b.update(status="completed", output_file_id=ofid)
        return SimpleNamespace(id=bid, status=b["status"], errors=None, output_file_id=b.get("output_file_id"),
                               error_file_id=None, request_counts=SimpleNamespace(completed=0, total=0))


class FakeAnthropic:
    """``anthropic.AsyncAnthropic``: messages.create and messages.batches."""

    def __init__(self, *, refuse_names=(), errored_names=()):
        self.refuse_names, self.errored_names = set(refuse_names), set(errored_names)
        self.calls, self._batches, self.created = [], {}, []
        self.messages = SimpleNamespace(create=self._create, batches=SimpleNamespace(
            create=self._bcreate, retrieve=self._bretrieve, results=self._bresults, list=self._blist))

    def _message(self, prompt):
        text, name = reply_for(prompt)
        if name in self.refuse_names:
            return SimpleNamespace(content=[], stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"),
                                   usage=SimpleNamespace(input_tokens=300, output_tokens=0))
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                               stop_details=None, usage=SimpleNamespace(input_tokens=300, output_tokens=40))

    async def _create(self, **kw):
        self.calls.append(kw)
        return self._message(kw["messages"][0]["content"])

    async def _bcreate(self, requests):
        bid = f"msgbatch_{len(self._batches)}"
        self._batches[bid] = {"requests": requests, "created_at": datetime.now(timezone.utc)}
        self.created.append(bid)
        return SimpleNamespace(id=bid, processing_status="in_progress")

    async def _bretrieve(self, bid):
        return SimpleNamespace(id=bid, processing_status="ended",
                               request_counts=SimpleNamespace(processing=0, succeeded=0))

    async def _bresults(self, bid):
        out = []
        for r in self._batches[bid]["requests"]:
            prompt = r["params"]["messages"][0]["content"]
            if reply_for(prompt)[1] in self.errored_names:
                out.append(SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="errored", error=None)))
            else:
                out.append(SimpleNamespace(custom_id=r["custom_id"],
                                           result=SimpleNamespace(type="succeeded", message=self._message(prompt))))

        async def gen():
            for x in out:
                yield x
        return gen()

    def _blist(self, limit=100):
        async def gen():
            return
            yield
        return gen()


# ---------------------------------------------------------------------------
# A small world: three axes of five entities, a corpus, two axes files
# ---------------------------------------------------------------------------

AXES = [("alpha_vs_beta", "alpha", "beta", "traits"), ("hero_vs_villain", "hero", "villain", "roles"),
        ("gentle_vs_harsh", "gentle", "harsh", "traits")]


def _describe(data: Path, kind: str, name: str) -> None:
    d = data / kind / "instructions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps({"description": f"The {name} one."}))


@pytest.fixture
def world(tmp_path, monkeypatch):
    exp, data = tmp_path / "experiments", tmp_path / "data"
    for name in ("calm", "bold", "shy", "kind", "alpha", "beta", "gentle", "harsh"):
        _describe(data, "traits", name)
    for name in ("pilot", "hero", "villain"):
        _describe(data, "roles", name)
    for k, (axis_dir, *_rest) in enumerate(AXES):
        proj = {str(slot): {e: {"raw": SCORES[e] * (slot + 1) + 0.1 * i + k, "whitened": -float(i)}
                            for i, e in enumerate(ENTITIES)} for slot in (0, 1)}
        p = exp / axis_dir / "gpt" / "projections.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"result": proj, "_provenance": {}}))
    axes2, axes3 = tmp_path / "axes2.tsv", tmp_path / "axes3.tsv"
    axes2.write_text("# two axes\n" + "".join("\t".join(a) + "\n" for a in AXES[:2]))
    axes3.write_text("".join("\t".join(a) + "\n" for a in AXES) + "  # trailing comment line\n")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    def no_real_client(*a, **k):
        raise AssertionError("a test reached a real client factory")
    monkeypatch.setattr(ev, "make_openai_client", no_real_client)
    monkeypatch.setattr(ev, "make_anthropic_client", no_real_client)
    return SimpleNamespace(exp=exp, data=data, axes2=axes2, axes3=axes3)


def _run(world, *models, extra=(), axes=None) -> int:
    argv = ["--axes_file", str(axes or world.axes2), "--experiments_dir", str(world.exp),
            "--instructions_dir", str(world.data), "--requests_per_second", "1000", "--batch_poll_seconds", "0"]
    for m in models:
        argv += ["--judge_model", m]
    return asyncio.run(ev.main_async([*argv, *extra]))


def _cell(world, axis_dir, model):
    d = world.exp / f"{axis_dir}_tier_eval" / model
    scores = json.loads((d / "scores_descriptions.json").read_text())
    usage = json.loads((d / "usage.json").read_text())
    return scores, usage, d


def _openai(monkeypatch, **clients):
    """Install fake OpenAI clients by tier: ``flex=`` and ``std=``."""
    table = {"flex": clients.get("flex"), None: clients.get("std")}

    def factory(service_tier=None):
        client = table[service_tier]
        assert client is not None, f"no fake client for tier {service_tier!r}"
        return client
    monkeypatch.setattr(ev, "make_openai_client", factory)


# ---------------------------------------------------------------------------
# Transport choice
# ---------------------------------------------------------------------------

@pytest.fixture
def luna_untested(monkeypatch):
    """GPT-6 Luna as a model whose Flex support is untested (the table has recorded
    it as taking Flex since the 2026-11-01 probe), so the probe path keeps its tests."""
    monkeypatch.delitem(tt.MODEL_CAPABILITIES, "gpt-6-luna")


def test_transport_selection(luna_untested):
    c = tt.choose_transport("claude-haiku-4-5")
    assert (c.transport, c.probe) == ("batch", False)
    assert tt.choose_transport("claude-sonnet-5").transport == "batch"
    c = tt.choose_transport("gpt-5.6-luna")                       # MODEL_CAPABILITIES: Flex taken
    assert (c.transport, c.probe) == ("flex", False)
    c = tt.choose_transport("gpt-4.1-mini")                       # MODEL_CAPABILITIES: Flex refused
    assert (c.transport, c.probe) == ("batch", False)
    c = tt.choose_transport("gpt-6-luna")                         # untested (fixture): probe
    assert (c.transport, c.probe) == ("flex", True)
    for model in ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.4-mini"):  # probed 2026-11-01: Flex taken
        c = tt.choose_transport(model)
        assert (c.transport, c.probe) == ("flex", False)
    assert tt.choose_transport("claude-haiku-4-5", "realtime").transport == "realtime"
    assert tt.choose_transport("gpt-5.6-luna", "batch").transport == "batch"
    c = tt.choose_transport("gpt-6-luna", "flex")                 # forced: no probe, a refusal stops the model
    assert (c.transport, c.probe) == ("flex", False)
    with pytest.raises(tt.JudgeRequestError):
        tt.choose_transport("claude-haiku-4-5", "flex")
    with pytest.raises(tt.JudgeRequestError):
        tt.choose_transport("gpt-4.1-mini", "flex")


def test_cli_defaults_and_exclusions(world):
    args = ev.build_parser().parse_args(["--axes_file", "x", "--judge_model", "m"])
    assert (args.max_tokens, args.requests_per_second, args.transport) == (200, 10, "auto")
    with pytest.raises(SystemExit) as e:
        _run(world, "gpt-4.1-mini", extra=["--pair", "alpha", "beta", "--dry_run"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:   # single-axis mode needs all four
        asyncio.run(ev.main_async(["--pair", "alpha", "beta", "--judge_model", "gpt-4.1-mini", "--dry_run"]))
    assert e.value.code == 2


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

def test_default_request_settings_reproduce_the_earlier_cells():
    """Coordinator's pin (2026-11-01): the models of the existing cells send what
    this tree's judge.py sent; new models follow step 3's convention."""
    body = tt.request_for("gpt-5.6-sol", "P", 200, tt.judge_request_settings("gpt-5.6-sol"))
    assert body == {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "P"}],
                    "max_completion_tokens": 200}             # no reasoning_effort, no temperature
    for m in ("gpt-5.6-luna", "gpt-5.6-terra"):
        assert tt.judge_request_settings(m) == {}
    params = tt.request_for("claude-sonnet-5", "P", 200, tt.judge_request_settings("claude-sonnet-5"))
    assert params["output_config"] == {"effort": "low"} and "thinking" not in params
    assert "temperature" not in params and params["max_tokens"] == 200
    assert tt.judge_request_settings("claude-opus-5") == {"output_config": {"effort": "low"}}
    params = tt.request_for("claude-haiku-5-5", "P", 200, tt.judge_request_settings("claude-haiku-5-5"))
    assert params["thinking"] == {"type": "disabled"} and "output_config" not in params
    assert set(tt.anthropic_request_params("P", "claude-haiku-4-5", 200)) == {"model", "max_tokens", "messages"}
    body = tt.request_for("gpt-6-luna", "P", 200, tt.judge_request_settings("gpt-6-luna"))
    assert body["reasoning_effort"] == "none" and "temperature" not in body
    assert tt.judge_request_settings("gpt-5.4-mini") == {"reasoning_effort": "none"}
    flex = tt.chat_request_body("P", "gpt-5.6-luna", 200, service_tier="flex")
    assert flex["service_tier"] == "flex"


def test_request_setting_override():
    o = [tt.parse_request_setting('gpt-6:reasoning_effort="low"'),
         tt.parse_request_setting("opus-5:output_config=null"),
         tt.parse_request_setting("metadata={\"run\": 1}")]
    assert tt.judge_request_settings("gpt-6-luna", o) == {"reasoning_effort": "low", "metadata": {"run": 1}}
    assert tt.judge_request_settings("claude-opus-5", o) == {"metadata": {"run": 1}}
    assert tt.judge_request_settings("claude-sonnet-5", o) == {"output_config": {"effort": "low"},
                                                                "metadata": {"run": 1}}
    for bad in ("no_equals", "reasoning_effort=none", "=1"):
        with pytest.raises(ValueError):
            tt.parse_request_setting(bad)
    # the table itself is never mutated by an override
    assert tt.judge_request_settings("claude-opus-5") == {"output_config": {"effort": "low"}}


def test_cli_override_reaches_the_requests(world, monkeypatch):
    std = FakeOpenAI()
    _openai(monkeypatch, std=std)
    assert _run(world, "gpt-6-luna", extra=["--transport", "realtime",
                                            "--request_setting", 'gpt-6:reasoning_effort="low"']) == 0
    assert std.calls and all(c["reasoning_effort"] == "low" for c in std.calls)
    scores, _, _ = _cell(world, "alpha_vs_beta", "gpt-6-luna")
    assert scores["_provenance"]["request_settings"] == {"reasoning_effort": "low"}


# ---------------------------------------------------------------------------
# Single calls
# ---------------------------------------------------------------------------

def test_call_openai_errors():
    usage, stats, rl = MultiModelUsage(), tt.CallStats(), RateLimiter(1000)
    bad = FakeOpenAI(flex_error=HTTPError(400, "Unsupported parameter: 'reasoning_effort'", param="reasoning_effort"))
    with pytest.raises(tt.JudgeRequestError) as e:
        asyncio.run(tt.call_openai(bad, "to score\n\n**calm**: x", "gpt-5.6-luna", 200, rl, usage,
                                   service_tier="flex", stats=stats))
    assert not isinstance(e.value, tt.FlexRefusedError)
    refused = FakeOpenAI(flex_error=HTTPError(400, "Invalid service_tier argument"))
    with pytest.raises(tt.FlexRefusedError):
        asyncio.run(tt.call_openai(refused, "to score\n\n**calm**: x", "gpt-6-luna", 200, rl, usage,
                                   service_tier="flex", stats=stats))
    flagged = FakeOpenAI(flex_error=HTTPError(400, "flagged", code="invalid_prompt"))
    assert asyncio.run(tt.call_openai(flagged, "to score\n\n**calm**: x", "gpt-6-luna", 200, rl, usage,
                                      service_tier="flex", stats=stats)) is None
    assert stats.content_flagged == 1
    with pytest.raises(tt.FlexRefusedError):        # known refused: nothing is sent
        asyncio.run(tt.call_openai(FakeOpenAI(), "x", "gpt-4.1-mini", 200, rl, usage, service_tier="flex"))
    assert usage.n_calls == 0


def test_flex_retries_then_succeeds_without_fallback(monkeypatch):
    monkeypatch.setattr(tt, "FLEX_RETRY_SLEEP_S", 0.0)
    flex = FakeOpenAI(flex_error=ConnectionError("overloaded"), flex_fail_times=2)
    std = FakeOpenAI()
    usage, stats = MultiModelUsage(), tt.CallStats()
    text = asyncio.run(tt.call_openai(flex, "to score\n\n**shy**: x", "gpt-5.6-luna", 200, RateLimiter(1000), usage,
                                      service_tier="flex", fallback_client=std, stats=stats))
    assert text.endswith("SCORE: 3") and len(flex.calls) == 3 and not std.calls
    assert set(usage.per_model) == {"gpt-5.6-luna:flex"} and stats.flex_fallbacks == 0


def test_billing_refusal_is_fatal_at_once(monkeypatch):
    """OpenAI's 'no credits' 429 (2026-11-01) and Anthropic's low-balance 400
    stop the call at once: no Flex retries, no standard-tier fallback."""
    monkeypatch.setattr(tt, "FLEX_RETRY_SLEEP_S", 0.0)
    usage, stats, rl = MultiModelUsage(), tt.CallStats(), RateLimiter(1000)
    broke = FakeOpenAI(flex_error=HTTPError(429, "You have no credits remaining. Add credits to continue "
                                                 "using the API", code="credit_balance_exhausted"))
    std = FakeOpenAI()
    with pytest.raises(tt.BillingError):
        asyncio.run(tt.call_openai(broke, "to score\n\n**calm**: x", "gpt-6-luna", 200, rl, usage,
                                   service_tier="flex", fallback_client=std, stats=stats))
    assert len(broke.calls) == 1 and not std.calls and stats.flex_fallbacks == 0
    assert tt.is_billing_error(HTTPError(429, "You exceeded your current quota", code="insufficient_quota"))
    assert tt.is_billing_error(HTTPError(400, "Your credit balance is too low to access the Anthropic API."))
    assert not tt.is_billing_error(HTTPError(429, "Rate limit reached for requests"))     # an ordinary 429
    assert not tt.is_billing_error(ConnectionError("overloaded"))


def test_billing_refusal_stops_a_run_without_scoring(world, monkeypatch):
    monkeypatch.setattr(tt, "FLEX_RETRY_SLEEP_S", 0.0)
    flex = FakeOpenAI(flex_error=HTTPError(429, "You have no credits remaining", code="credit_balance_exhausted"))
    std = FakeOpenAI()
    _openai(monkeypatch, flex=flex, std=std)
    assert _run(world, "gpt-5.6-luna") != 0
    assert len(flex.calls) <= 2 and not std.calls       # stopped, not retried for every request


def test_anthropic_refusal_is_a_counted_failure():
    usage, stats = MultiModelUsage(), tt.CallStats()
    client = FakeAnthropic(refuse_names={"calm"})
    out = asyncio.run(tt.call_anthropic(client, "to score\n\n**calm**: x", "claude-haiku-5-5", 200,
                                        RateLimiter(1000), usage, stats=stats))
    assert out is None and stats.refused == 1 and stats.refusal_categories == {"cyber": 1}
    assert usage.n_calls == 1                       # charged all the same
    assert client.calls[0]["thinking"] == {"type": "disabled"}


# ---------------------------------------------------------------------------
# Custom ids, the axis map, the axes file
# ---------------------------------------------------------------------------

def test_custom_ids_and_axis_map_round_trip(tmp_path):
    cid = ev.custom_id(12, "pilot|R")
    assert cid == "a12|pilot|R" and ev.split_custom_id(cid) == (12, "pilot|R")
    with pytest.raises(ValueError):
        ev.custom_id(0, "x" * 70)
    with pytest.raises(ValueError):
        ev.custom_id(0, "has space")
    job_a = ev.AxisJob("a_vs_b", "a", "b", "traits", tmp_path / "a" / "gpt" / "projections.json", tmp_path / "a_te")
    job_b = ev.AxisJob("c_vs_d", "c", "d", "roles", tmp_path / "c" / "gpt" / "projections.json", tmp_path / "c_te")
    m = ev.AxisMap(tmp_path / "work" / "axis_map.json")
    assert (m.index_for(job_a), m.index_for(job_b), m.index_for(job_a)) == (0, 1, 0)
    m.settings = {"model": "x"}
    m.save()
    m2 = ev.AxisMap(tmp_path / "work" / "axis_map.json")
    assert m2.settings == {"model": "x"} and m2.axes == m.axes
    assert m2.index_for(job_b) == 1
    job_c = ev.AxisJob("e_vs_f", "e", "f", "traits", tmp_path / "e.json", tmp_path / "e_te")
    assert m2.index_for(job_c) == 2                 # never reuses an index
    back = ev.AxisJob.from_map_entry(m2.axes[1])
    assert (back.axis_dir, back.pair_type, back.output_dir) == ("c_vs_d", "roles", (tmp_path / "c_te").resolve())


def test_axes_file(tmp_path):
    good = tmp_path / "a.tsv"
    good.write_text("# c\n\na_vs_b a b traits  # trailing\nc_vs_d\tc\td\troles\n")
    assert ev.read_axes_file(good) == [("a_vs_b", "a", "b", "traits"), ("c_vs_d", "c", "d", "roles")]
    for text in ("a_vs_b a b\n", "a_vs_b a b things\n", "a_vs_b a b traits\na_vs_b a b traits\n"):
        good.write_text(text)
        with pytest.raises(ValueError):
            ev.read_axes_file(good)
    rows = ev.read_axes_file(REPO / "scripts" / "tier_eval_axes.tsv")
    assert len(rows) == 61 and len({r[0] for r in rows}) == 61
    assert rows[0] == ("angel_vs_demon", "angel", "demon", "roles")
    assert rows[1] == ("decisive_vs_indecisive", "decisive", "indecisive", "traits")
    assert rows[2][0] == "trustworthy_vs_untrustworthy" and rows[20][0] == "accessible_vs_esoteric"
    assert sorted(r[0] for r in rows if r[3] == "roles") == [
        "aligned_artificial_intelligence_vs_paperclip_maximizer", "angel_vs_demon", "guardian_vs_destroyer",
        "predator_vs_prey", "symbiont_vs_parasite"]


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def test_pricing_batch_flex_and_new_rows():
    assert price_for_model("claude-haiku-5-5") == (0.10, 0.50)
    assert price_for_model("claude-haiku-4-5-20251001") == (1.00, 5.00)    # the bare "haiku" row still
    assert price_for_model("gpt-6-luna") == (0.10, 0.50)
    assert price_for_model("gpt-5.4-mini") == (0.75, 4.50)
    for m in ("gpt-6-luna", "gpt-5.6-sol", "gpt-4.1-mini", "claude-haiku-5-5", "claude-sonnet-5", "claude-opus-5"):
        rt_in, rt_out = price_for_model(m)
        assert price_for_model(f"{m}:batch") == pytest.approx((rt_in / 2, rt_out / 2))
    assert price_for_model("gpt-5.6-luna:flex") == price_for_model("gpt-5.6-luna:batch")
    with pytest.raises(KeyError):
        price_for_model("claude-haiku-5-5:flex")
    u = MultiModelUsage()
    u.charge("gpt-6-luna:batch", 1_000_000, 1_000_000)
    assert u.total_cost_usd == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Flex
# ---------------------------------------------------------------------------

def test_flex_run_writes_cells_with_flex_usage(world, monkeypatch, capsys):
    flex, std = FakeOpenAI(), FakeOpenAI()
    _openai(monkeypatch, flex=flex, std=std)
    assert _run(world, "gpt-5.6-luna") == 0
    assert len(flex.calls) == 10 and not std.calls
    assert all(c["service_tier"] == "flex" and "reasoning_effort" not in c and "temperature" not in c
               and c["max_completion_tokens"] == 200 for c in flex.calls)
    for axis_dir in ("alpha_vs_beta", "hero_vs_villain"):
        scores, usage, d = _cell(world, axis_dir, "gpt-5.6-luna")
        assert scores["result"] == SCORES
        prov = scores["_provenance"]
        assert (prov["transport"], prov["request_settings"], prov["max_tokens"]) == ("flex", {}, 200)
        assert prov["judge_model"] == "gpt-5.6-luna" and prov["rubric_version"]
        assert set(usage["per_model"]) == {"gpt-5.6-luna:flex"} and usage["n_calls"] == 5
        assert usage["total_cost_usd"] > 0
        corr = json.loads((d / "correlations.json").read_text())
        assert corr["descriptions"]["0"]["raw"]["n"] == 5
    out = capsys.readouterr().out
    assert "parsed 10/10" in out and "flex fallbacks 0" in out


def test_flex_falls_back_to_the_standard_tier(world, monkeypatch, capsys):
    monkeypatch.setattr(tt, "FLEX_FALLBACK_AFTER_S", 0.0)
    flex, std = FakeOpenAI(flex_error=ConnectionError("overloaded")), FakeOpenAI()
    _openai(monkeypatch, flex=flex, std=std)
    assert _run(world, "gpt-5.6-luna") == 0
    assert len(flex.calls) == 10 and len(std.calls) == 10
    assert all("service_tier" not in c for c in std.calls)
    _, usage, _ = _cell(world, "alpha_vs_beta", "gpt-5.6-luna")
    assert set(usage["per_model"]) == {"gpt-5.6-luna"}            # charged at the full rate
    assert "flex fallbacks 10" in capsys.readouterr().out


def test_probe_switches_to_batch_without_losing_the_request(world, monkeypatch, luna_untested):
    flex = FakeOpenAI(flex_error=HTTPError(400, "Invalid service_tier argument", param="service_tier"))
    std = FakeOpenAI()
    _openai(monkeypatch, flex=flex, std=std)
    assert _run(world, "gpt-6-luna") == 0
    assert len(flex.calls) == 1                                   # the probe alone
    lines = std.batch_lines(0)
    assert sorted(line["custom_id"] for line in lines) == sorted(
        f"a{i}|{e}" for i in (0, 1) for e in ENTITIES)            # the probe's request included
    assert all(line["body"]["reasoning_effort"] == "none" and "service_tier" not in line["body"] for line in lines)
    assert not std.calls                                          # nothing in real time
    scores, usage, _ = _cell(world, "alpha_vs_beta", "gpt-6-luna")
    assert scores["result"] == SCORES and scores["_provenance"]["transport"] == "batch"
    assert "refused Flex on the probe" in scores["_provenance"]["transport_reason"]
    assert set(usage["per_model"]) == {"gpt-6-luna:batch"}


def test_probe_accepted_continues_on_flex(world, monkeypatch, luna_untested):
    flex = FakeOpenAI()
    _openai(monkeypatch, flex=flex, std=FakeOpenAI())
    assert _run(world, "gpt-6-luna") == 0
    assert len(flex.calls) == 10 and all(c["reasoning_effort"] == "none" for c in flex.calls)
    scores, usage, _ = _cell(world, "hero_vs_villain", "gpt-6-luna")
    assert scores["_provenance"]["transport"] == "flex"
    assert "probe was answered" in scores["_provenance"]["transport_reason"]
    assert set(usage["per_model"]) == {"gpt-6-luna:flex"}


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def test_openai_batch_custom_ids_axis_map_and_realtime_remainder(world, monkeypatch):
    std = FakeOpenAI(error_names={"bold"})
    _openai(monkeypatch, std=std)
    assert _run(world, "gpt-4.1-mini") == 0                      # auto: Flex refused -> batch
    assert len(std.batches_created) == 1                          # one submission for both axes
    lines = std.batch_lines(0)
    assert sorted(line["custom_id"] for line in lines) == sorted(f"a{i}|{e}" for i in (0, 1) for e in ENTITIES)
    body = lines[0]["body"]
    assert body == tt.chat_request_body(body["messages"][0]["content"], "gpt-4.1-mini", 200, settings={})
    work = world.exp / "_tier_eval_batches" / "gpt-4.1-mini"
    amap = json.loads((work / "axis_map.json").read_text())
    assert {k: v["axis_dir"] for k, v in amap["axes"].items()} == {"0": "alpha_vs_beta", "1": "hero_vs_villain"}
    assert amap["settings"]["max_tokens"] == 200 and amap["settings"]["request_settings"] == {}
    assert not list(world.exp.glob("*_tier_eval/_tier_eval_batches"))
    # the two errored requests ("bold" on both axes) went in real time, on the standard tier
    assert len(std.calls) == 2 and all("service_tier" not in c for c in std.calls)
    for axis_dir in ("alpha_vs_beta", "hero_vs_villain"):
        scores, usage, _ = _cell(world, axis_dir, "gpt-4.1-mini")
        assert scores["result"] == SCORES
        assert usage["per_model"]["gpt-4.1-mini:batch"]["n_calls"] == 4
        assert usage["per_model"]["gpt-4.1-mini"]["n_calls"] == 1
    state = json.loads((work / "batch_state.json").read_text())
    assert [r["collected"] for r in state["chunks"]] == [True]
    assert not list((work / "collected").glob("*.json")) if (work / "collected").exists() else True


def test_anthropic_batch_collects_cells_and_counts_refusals(world, monkeypatch, caplog):
    client = FakeAnthropic(refuse_names={"kind"}, errored_names={"shy"})
    monkeypatch.setattr(ev, "make_anthropic_client", lambda: client)
    assert _run(world, "claude-haiku-5-5") == 0
    reqs = client._batches[client.created[0]]["requests"]
    assert len(client.created) == 1 and len(reqs) == 10
    assert all(r["custom_id"].startswith("r") and "|" not in r["custom_id"] for r in reqs)
    assert all(r["params"]["thinking"] == {"type": "disabled"} and "temperature" not in r["params"] for r in reqs)
    assert len(client.calls) == 2                                 # the errored "shy" requests, in real time
    for axis_dir in ("alpha_vs_beta", "hero_vs_villain"):
        scores, usage, _ = _cell(world, axis_dir, "claude-haiku-5-5")
        assert scores["result"] == {k: v for k, v in SCORES.items() if k != "kind"}   # the refusal: no score
        assert usage["per_model"]["claude-haiku-5-5:batch"]["n_calls"] == 4
        assert usage["per_model"]["claude-haiku-5-5"]["n_calls"] == 1
    assert "HIGH FAIL RATE" in caplog.text and "8/10" in caplog.text


def test_rerun_collects_batches_in_flight_before_submitting(world, monkeypatch):
    std = FakeOpenAI(fail_retrieve=True)
    _openai(monkeypatch, std=std)
    assert _run(world, "gpt-4.1-mini") == 1                      # the poll keeps failing: the model stops
    assert std.batches_created == ["batch-0"]
    assert not (world.exp / "alpha_vs_beta_tier_eval" / "gpt-4.1-mini" / "usage.json").exists()
    std.fail_retrieve = False
    std.events.clear()
    assert _run(world, "gpt-4.1-mini", axes=world.axes3) == 0    # a third axis joins
    assert std.events.index(("retrieve", "batch-0")) < std.events.index(("create", "batch-1"))
    assert sorted(line["custom_id"] for line in std.batch_lines(1)) == sorted(f"a2|{e}" for e in ENTITIES)
    for axis_dir, *_ in AXES:
        scores, usage, _ = _cell(world, axis_dir, "gpt-4.1-mini")
        assert scores["result"] == SCORES and usage["n_calls"] == 5
    assert not std.calls


def test_model_drift_refuses_to_collect(world, monkeypatch, caplog):
    work = world.exp / "_tier_eval_batches" / "gpt-4.1-mini"
    (work / "chunks").mkdir(parents=True)
    chunk = work / "chunks" / "chunk_x_0000.jsonl"
    chunk.write_text(json.dumps({"custom_id": "a0|calm", "method": "POST", "url": "/v1/chat/completions",
                                 "body": {"model": "gpt-other"}}) + "\n")
    (work / "batch_state.json").write_text(json.dumps({"version": 1, "chunks": [
        {"path": str(chunk), "batch_id": "batch-9", "input_file_id": "file-9", "n_requests": 1}]}))
    std = FakeOpenAI()
    _openai(monkeypatch, std=std)
    assert _run(world, "gpt-4.1-mini") == 1
    assert "ModelDriftError" in caplog.text and not std.batches_created
    # a live transport is refused too while batches are in flight
    caplog.clear()
    assert _run(world, "gpt-4.1-mini", extra=["--transport", "realtime"]) == 1
    assert "still in flight" in caplog.text and not std.calls


# ---------------------------------------------------------------------------
# Resume, budget, dry run
# ---------------------------------------------------------------------------

def test_axis_with_usage_json_is_skipped(world, monkeypatch):
    done = world.exp / "alpha_vs_beta_tier_eval" / "gpt-4.1-mini"
    done.mkdir(parents=True)
    (done / "usage.json").write_text('{"sentinel": true}')
    std = FakeOpenAI()
    _openai(monkeypatch, std=std)
    assert _run(world, "gpt-4.1-mini", extra=["--transport", "realtime"]) == 0
    assert len(std.calls) == 5
    assert {reply_for(c["messages"][0]["content"])[1] for c in std.calls} == set(ENTITIES)
    assert (done / "usage.json").read_text() == '{"sentinel": true}'
    assert not (done / "scores_descriptions.json").exists()
    scores, usage, _ = _cell(world, "hero_vs_villain", "gpt-4.1-mini")
    assert scores["_provenance"]["transport"] == "realtime" and set(usage["per_model"]) == {"gpt-4.1-mini"}


def test_cell_label_writes_beside_earlier_cells(world, monkeypatch):
    """--cell_label @t0: a rerun under other settings goes to <model>@t0, is not
    skipped because the plain cell is done, and records its label."""
    done = world.exp / "alpha_vs_beta_tier_eval" / "gpt-4.1-mini"
    done.mkdir(parents=True)
    (done / "usage.json").write_text('{"sentinel": true}')
    std = FakeOpenAI()
    _openai(monkeypatch, std=std)
    assert _run(world, "gpt-4.1-mini", extra=["--transport", "realtime", "--cell_label", "@t0",
                                              "--request_setting", "gpt-4.1-mini:temperature=0"]) == 0
    assert len(std.calls) == 10 and all(c["temperature"] == 0 for c in std.calls)
    assert (done / "usage.json").read_text() == '{"sentinel": true}'      # the plain cell is untouched
    for axis_dir in ("alpha_vs_beta", "hero_vs_villain"):
        scores, usage, d = _cell(world, axis_dir, "gpt-4.1-mini@t0")
        assert scores["_provenance"]["cell_label"] == "@t0"
        assert scores["_provenance"]["request_settings"] == {"temperature": 0}
        assert set(usage["per_model"]) == {"gpt-4.1-mini"}                # usage keyed by the model, not the label
    assert ev.CELL_LABEL == "@t0"
    assert _run(world, "gpt-4.1-mini", extra=["--transport", "realtime"]) == 0   # the next run resets the label
    assert ev.CELL_LABEL == ""
    with pytest.raises(SystemExit):
        _run(world, "gpt-4.1-mini", extra=["--cell_label", "bad/label"])


def test_budget_stop(world, monkeypatch):
    # each call costs $0.40 at gpt-4.1-mini's $0.40 per million input tokens
    std = FakeOpenAI(prompt_tokens=1_000_000, completion_tokens=0)
    _openai(monkeypatch, std=std)
    assert _run(world, "gpt-4.1-mini", extra=["--transport", "realtime", "--budget_usd", "1.0"]) == 2
    assert len(std.calls) == 3                                    # $1.20 > $1.00: no fourth request
    d = world.exp / "alpha_vs_beta_tier_eval" / "gpt-4.1-mini"
    assert not (d / "usage.json").exists()
    partial = json.loads((d / "scores_descriptions.partial.json").read_text())
    assert len(partial["result"]) == 3 and partial["_provenance"]["complete"] is False
    assert json.loads((d / "usage.partial.json").read_text())["n_calls"] == 3
    assert not (world.exp / "hero_vs_villain_tier_eval").exists()


def test_dry_run_writes_nothing_and_reports(world, capsys, luna_untested):
    before = sorted(p for p in world.exp.rglob("*"))
    assert _run(world, "gpt-4.1-mini", "gpt-6-luna", "claude-haiku-5-5", extra=["--dry_run"]) == 0
    assert sorted(p for p in world.exp.rglob("*")) == before
    out = capsys.readouterr().out
    assert "alpha_vs_beta" in out and "5 entities" in out
    assert re.search(r"gpt-4\.1-mini .*transport=batch .*axes 2/2\s+~10 calls", out)
    assert re.search(r"gpt-6-luna .*transport=flex \(probe\)", out)
    assert re.search(r"claude-haiku-5-5 .*transport=batch", out)
    assert '"thinking": {"type": "disabled"}' in out
