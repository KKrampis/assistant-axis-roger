"""judge.py as the one request layer (2026-11-02): the Flex limits, the billing stop, cut replies,
overrides and Gemini, through the calls step 3 and the Pareto tools use."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from assistant_axis import judge as J
from assistant_axis.judge_pricing import MultiModelUsage, price_for_model


class HTTPError(Exception):
    def __init__(self, status_code, message, *, code=None, param=None, type=None):
        super().__init__(message)
        self.status_code, self.message, self.code, self.param, self.type, self.body = status_code, message, code, param, type, None


def chat_reply(text="SCORE: 2", finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))


class FakeOpenAI:
    """Raises the queued errors in turn, then answers; records each attempt's client options."""

    def __init__(self, *errors, reply=None):
        self.errors, self.reply, self.calls, self.options = list(errors), reply or chat_reply(), [], []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, **kw):
        self.options.append(kw)
        return self

    async def _create(self, **kw):
        self.calls.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        return self.reply


class FakeAnthropic:
    def __init__(self, *errors, content=None, stop="end_turn"):
        self.errors, self.calls = list(errors), []
        self.content = content if content is not None else [SimpleNamespace(type="text", text="SCORE: 2")]
        self.stop = stop
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        self.calls.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(content=self.content, stop_reason=self.stop, stop_details=None,
                               usage=SimpleNamespace(input_tokens=10, output_tokens=2))


class FakeGemini:
    def __init__(self, *responses):
        self.responses, self.posts = list(responses), []

    async def post(self, url, json):
        self.posts.append((url, json))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def http(status, data):
    return SimpleNamespace(status_code=status, text=json.dumps(data), json=lambda: data)


def gemini_ok(text="1. x SCORE: 40", finish="STOP", thoughts=0):
    return http(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
                      "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3, "thoughtsTokenCount": thoughts}})


RATE = J.RateLimiter(1000)
CONNECTION = ConnectionError("no route")
NO_CREDIT = HTTPError(429, "You have no credits remaining", code="credit_balance_exhausted", type="insufficient_quota")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(J, "FLEX_RETRY_SLEEP_S", 0.0)
    monkeypatch.setattr(J, "GEMINI_RETRY_SLEEP_S", (0,))


def run(coro):
    return asyncio.run(coro)


# ---- routing, capabilities, settings

def test_provider_routing():
    assert J.provider_for_model("gemini-3.1-flash-lite") == "google"
    assert J.provider_for_model("gpt-5.6-luna") == "openai" and J.provider_for_model("claude-haiku-5-5") == "anthropic"
    with pytest.raises(ValueError, match="gemini-"):
        J.provider_for_model("llama-3")


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-6-luna", "gpt-5.4-mini"])
def test_flex_is_known_for_the_models_that_answered_it(model):
    assert J.accepts_flex(model) is True


def test_flex_is_refused_for_mini_and_gemini():
    assert J.accepts_flex("gpt-4.1-mini") is False and J.accepts_flex("gemini-3.1-flash-lite") is False


@pytest.mark.parametrize("model", ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gpt-5.6-terra", "gpt-5.6-sol"])
def test_the_tool_s_models_have_prices(model):
    assert price_for_model(model)[0] > 0


def test_overrides_add_change_and_remove_fields_and_leave_the_table_alone():
    overrides = [J.parse_request_setting('gpt-6:reasoning_effort="low"'), J.parse_request_setting("gpt-5.6:reasoning_effort=null"),
                 J.parse_request_setting('metadata={"run": 1}')]
    assert J.judge_request_settings("gpt-6-luna", overrides) == {"reasoning_effort": "low", "metadata": {"run": 1}}
    assert J.judge_request_settings("gpt-5.6-luna", overrides) == {"metadata": {"run": 1}}
    assert J.judge_request_settings("gpt-5.6-luna") == {"reasoning_effort": "none"}
    with pytest.raises(ValueError):
        J.parse_request_setting("reasoning_effort=none")


def test_settings_replace_the_table_in_a_request():
    assert "reasoning_effort" not in J.chat_request_body("P", "gpt-5.6-luna", 9, settings={})
    assert J.chat_request_body("P", "gpt-5.6-luna", 9)["reasoning_effort"] == "none"
    assert J.anthropic_request_params("P", "claude-haiku-5-5", 9, settings={"metadata": {"a": 1}})["metadata"] == {"a": 1}
    assert "thinking" not in J.anthropic_request_params("P", "claude-haiku-5-5", 9, settings={})


# ---- Flex

def test_a_flex_attempt_goes_out_with_no_sdk_retries_and_the_time_left(monkeypatch):
    client = FakeOpenAI(CONNECTION)
    assert run(J.call_judge_single(client, "P", "gpt-5.6-luna", 9, RATE, service_tier="flex", fallback_client=FakeOpenAI())) == "SCORE: 2"
    assert len(client.options) == 2
    for o in client.options:
        assert o["max_retries"] == 0 and J.FLEX_MIN_ATTEMPT_S <= o["timeout"] <= J.FLEX_FALLBACK_AFTER_S


def test_a_flex_attempt_near_the_limit_still_gets_the_minimum(monkeypatch):
    client = FakeOpenAI()
    monkeypatch.setattr(J, "FLEX_FALLBACK_AFTER_S", 0.0)
    run(J.call_judge_single(client, "P", "gpt-5.6-luna", 9, RATE, service_tier="flex", fallback_client=FakeOpenAI()))
    assert client.options[0]["timeout"] == J.FLEX_MIN_ATTEMPT_S


def test_the_flex_clock_starts_at_the_first_send_not_in_the_queue(monkeypatch):
    class SlowQueue:
        async def acquire(self):
            await asyncio.sleep(0.05)
    monkeypatch.setattr(J, "FLEX_FALLBACK_AFTER_S", 0.02)
    flex, std = FakeOpenAI(), FakeOpenAI()
    # queued for longer than the whole allowance: its first attempt must still go to Flex
    assert run(J.call_judge_single(flex, "P", "gpt-5.6-luna", 9, SlowQueue(), service_tier="flex", fallback_client=std)) == "SCORE: 2"
    assert len(flex.calls) == 1 and not std.calls


def test_flex_falls_back_once_after_the_allowance(monkeypatch):
    monkeypatch.setattr(J, "FLEX_FALLBACK_AFTER_S", 0.0)
    before = J.FLEX_FALLBACKS.get("gpt-5.6-luna", 0)
    flex, std, usage = FakeOpenAI(CONNECTION), FakeOpenAI(), MultiModelUsage()
    assert run(J.call_judge_single(flex, "P", "gpt-5.6-luna", 9, RATE, usage, service_tier="flex", fallback_client=std)) == "SCORE: 2"
    assert "service_tier" not in std.calls[0] and set(usage.per_model) == {"gpt-5.6-luna"}
    assert J.FLEX_FALLBACKS["gpt-5.6-luna"] == before + 1


def test_a_refused_service_tier_says_so():
    with pytest.raises(J.FlexRefusedError):
        run(J.call_judge_single(FakeOpenAI(HTTPError(400, "Invalid service_tier argument")), "P", "gpt-6-luna", 9, RATE, service_tier="flex"))
    with pytest.raises(J.FlexRefusedError):
        run(J.call_judge_single(FakeOpenAI(), "P", "gpt-4.1-mini", 9, RATE, service_tier="flex"))


# ---- the account

def test_no_credit_stops_at_once_even_on_flex():
    client = FakeOpenAI(NO_CREDIT)
    with pytest.raises(J.BillingError):
        run(J.call_judge_single(client, "P", "gpt-5.6-luna", 9, RATE, service_tier="flex", fallback_client=FakeOpenAI()))
    assert len(client.calls) == 1


def test_no_credit_stops_a_step_3_batch():
    clients = FakeOpenAI(NO_CREDIT)
    with pytest.raises(J.JudgeRequestError):
        run(J.call_judge_batch(clients, ["P"], "gpt-6-luna", 9, RATE))


def test_a_low_anthropic_balance_stops_at_once():
    with pytest.raises(J.BillingError):
        run(J.call_anthropic_judge_single(FakeAnthropic(HTTPError(400, "Your credit balance is too low")), "P", "claude-haiku-5-5", 9, RATE))


def test_a_429_that_is_only_a_rate_limit_is_no_billing_error():
    assert not J.is_billing_error(HTTPError(429, "Rate limit reached", code="rate_limit_exceeded"))
    assert not J.is_billing_error(ConnectionError("no route"))
    assert J.is_billing_error(NO_CREDIT)


# ---- what a call returns

def test_outcomes_of_the_detailed_call():
    stats = J.CallStats()
    assert run(J.call_openai_judge(FakeOpenAI(CONNECTION), "P", "gpt-5.6-luna", 9, RATE, stats=stats)) is J.FAILED
    assert stats.errors == 1
    assert run(J.call_openai_judge(FakeOpenAI(), "P", "gpt-5.6-luna", 9, RATE, gate=lambda: True, stats=stats)) is J.SKIPPED
    flagged = HTTPError(400, "flagged", code="invalid_prompt")
    assert run(J.call_openai_judge(FakeOpenAI(flagged), "P", "gpt-5.6-luna", 9, RATE, stats=stats)) is None
    assert stats.content_flagged == 1
    # OpenAI's biological-risk flag (2026-11-04) is a flag on the prompt, not a broken call
    bio = HTTPError(400, "This content was flagged for possible biological risk.", code="bio_policy")
    assert run(J.call_openai_judge(FakeOpenAI(bio), "P", "gpt-5.6-luna", 9, RATE, stats=stats)) is None
    assert stats.content_flagged == 2
    cut = run(J.call_openai_judge(FakeOpenAI(reply=chat_reply("SCORE", "length")), "P", "gpt-5.6-luna", 9, RATE))
    assert isinstance(cut, J.Truncated) and cut.text == "SCORE"
    assert run(J.call_anthropic_judge(FakeAnthropic(CONNECTION), "P", "claude-haiku-5-5", 9, RATE)) is J.FAILED
    assert run(J.call_anthropic_judge(FakeAnthropic(stop="refusal"), "P", "claude-haiku-5-5", 9, RATE)) is None


def test_the_plain_calls_keep_their_meaning():
    assert run(J.call_judge_single(FakeOpenAI(CONNECTION), "P", "gpt-5.6-luna", 9, RATE)) is None
    cut = chat_reply("It leans 2 ways", "length")
    assert run(J.call_judge_single(FakeOpenAI(reply=cut), "P", "gpt-5.6-luna", 9, RATE)) == "It leans 2 ways"
    before = J.TRUNCATED_REPLIES.get("gpt-5.6-luna", 0)
    assert run(J.call_judge_single(FakeOpenAI(reply=cut), "P", "gpt-5.6-luna", 9, RATE, reject_truncated=True)) is None
    assert J.TRUNCATED_REPLIES["gpt-5.6-luna"] == before + 1
    before = J.REFUSED_REPLIES.get("claude-haiku-5-5", 0)
    assert run(J.call_anthropic_judge_single(FakeAnthropic(stop="refusal"), "P", "claude-haiku-5-5", 9, RATE)) is None
    assert J.REFUSED_REPLIES["claude-haiku-5-5"] == before + 1
    thinking = [SimpleNamespace(type="thinking", thinking="hm"), SimpleNamespace(type="text", text="SCORE: 1")]
    assert run(J.call_anthropic_judge_single(FakeAnthropic(content=thinking), "P", "claude-haiku-5-5", 9, RATE)) == "SCORE: 1"


# ---- Gemini

def test_gemini_request_and_charge():
    client, usage = FakeGemini(gemini_ok(thoughts=4)), MultiModelUsage()
    assert run(J.call_gemini_judge(client, "P", "gemini-3.1-flash-lite", 99, RATE, usage)) == "1. x SCORE: 40"
    url, body = client.posts[0]
    assert url == "/models/gemini-3.1-flash-lite:generateContent"
    assert body == {"contents": [{"parts": [{"text": "P"}]}],
                    "generationConfig": {"maxOutputTokens": 99, "temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}}
    entry = usage.per_model["gemini-3.1-flash-lite"]
    assert (entry.prompt_tokens, entry.completion_tokens) == (11, 7)     # thought tokens bill as output


def test_gemini_charges_its_implicit_cache_reads():
    """2026-11-04: a reply's cachedContentTokenCount (a repeated prefix read from Gemini's implicit cache) is
    charged as cache reads, priced at a tenth of the input rate."""
    reply = http(200, {"candidates": [{"content": {"parts": [{"text": "1. 40"}]}, "finishReason": "STOP"}],
                       "usageMetadata": {"promptTokenCount": 10_000, "candidatesTokenCount": 100,
                                         "cachedContentTokenCount": 6_000}})
    client, usage = FakeGemini(reply), MultiModelUsage()
    assert run(J.call_gemini_judge(client, "P", "gemini-3.1-flash-lite", 99, RATE, usage)) == "1. 40"
    entry = usage.per_model["gemini-3.1-flash-lite"]
    assert (entry.prompt_tokens, entry.cache_read_tokens, entry.completion_tokens) == (10_000, 6_000, 100)
    assert entry.cost_usd == pytest.approx((4_000 * 0.25 + 6_000 * 0.025 + 100 * 1.50) / 1e6)


def test_gemini_retries_then_fails_without_an_answer(monkeypatch):
    monkeypatch.setattr(J, "GEMINI_ATTEMPTS", 2)
    client, stats = FakeGemini(http(503, {}), CONNECTION), J.CallStats()
    assert run(J.call_gemini_judge(client, "P", "gemini-3.1-flash-lite", 9, RATE, stats=stats)) is J.FAILED
    assert stats.errors == 1 and len(client.posts) == 2
    assert run(J.call_gemini_judge(FakeGemini(http(503, {}), gemini_ok()), "P", "gemini-3.1-flash-lite", 9, RATE)) == "1. x SCORE: 40"


def test_gemini_retries_an_unreadable_200():
    def not_json():
        raise ValueError("Expecting value")
    broken = SimpleNamespace(status_code=200, text="<html>", json=not_json)
    assert run(J.call_gemini_judge(FakeGemini(broken, gemini_ok()), "P", "gemini-3.1-flash-lite", 9, RATE)) == "1. x SCORE: 40"


def test_gemini_stops_on_a_daily_quota_and_a_bad_request():
    daily = {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
    with pytest.raises(J.BillingError):
        run(J.call_gemini_judge(FakeGemini(http(429, daily)), "P", "gemini-3.1-flash-lite", 9, RATE))
    depleted = {"error": {"code": 402, "message": "Your prepayment credits are depleted."}}
    client = FakeGemini(http(402, depleted))
    with pytest.raises(J.BillingError, match="topped up"):
        run(J.call_gemini_judge(client, "P", "gemini-3.1-flash-lite", 9, RATE))
    assert len(client.posts) == 1
    with pytest.raises(J.JudgeRequestError):
        run(J.call_gemini_judge(FakeGemini(http(404, {"error": "no such model"})), "P", "gemini-2.5-flash-lite", 9, RATE))


def test_gemini_cut_and_blocked_replies():
    cut = run(J.call_gemini_judge(FakeGemini(gemini_ok(finish="MAX_TOKENS")), "P", "gemini-3.1-flash-lite", 9, RATE))
    assert isinstance(cut, J.Truncated)
    blocked = http(200, {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 5}})
    stats = J.CallStats()
    assert run(J.call_gemini_judge(FakeGemini(blocked), "P", "gemini-3.1-flash-lite", 9, RATE, stats=stats)) is None
    assert stats.content_flagged == 1


def test_gemini_in_the_batch_call_and_the_dispatch():
    assert run(J.call_judge_batch(FakeGemini(gemini_ok(), gemini_ok()), ["A", "B"], "gemini-3.1-flash-lite", 9, RATE)) == ["1. x SCORE: 40"] * 2
    assert run(J.call_judge_model("gemini-3.1-flash-lite", "P", 9, RATE, client=FakeGemini(gemini_ok()))) == "1. x SCORE: 40"
    assert run(J.call_judge_model("gpt-5.6-luna", "P", 9, RATE, client=FakeOpenAI())) == "SCORE: 2"
    assert run(J.call_judge_model("claude-haiku-5-5", "P", 9, RATE, client=FakeAnthropic())) == "SCORE: 2"


def test_the_unified_call_says_gemini_is_not_its_own():
    with pytest.raises(ValueError, match="call_judge_model"):
        run(J.call_judge_single_unified(prompt="P", model="gemini-3.1-flash-lite", max_tokens=9, rate_limiter=RATE))
