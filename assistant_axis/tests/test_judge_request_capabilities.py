"""Request parameters by model in assistant_axis/judge.py (code review
2026-10-12, B1: reviews 7 D1 and 8 D3).

* A temperature equal to the provider default (1) is never sent, on either
  provider, so a default call works on models that refuse the field.
* ``MODEL_CAPABILITIES`` lists the non-default cases (models that refuse a
  temperature); a non-default temperature for one of them raises
  ``JudgeRequestError`` before anything is sent.
* An HTTP 400 from either SDK is raised as ``JudgeRequestError``, not logged
  and turned into None (which made every item of a run unparseable).

No network: the clients are fakes.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from assistant_axis import judge as J
from assistant_axis.judge import (
    DEFAULT_TEMPERATURE,
    MODEL_CAPABILITIES,
    JudgeRequestError,
    RateLimiter,
    accepts_flex,
    accepts_temperature,
    call_anthropic_judge_single,
    call_judge_batch,
    call_judge_single,
    call_judge_single_unified,
    chat_request_body,
    model_capabilities,
    request_temperature,
)


class _Status400(Exception):
    status_code = 400


class _Status500(Exception):
    status_code = 500


def _openai_response(text="SCORE: 2"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=None)


def _anthropic_response(text="SCORE: 2"):
    return SimpleNamespace(content=[SimpleNamespace(text=text)], usage=None)


class FakeOpenAI:
    def __init__(self, fail_on=None, exc=None):
        self.calls = []
        self.fail_on, self.exc = fail_on, exc

        async def create(**kw):
            self.calls.append(kw)
            if self.exc is not None and (self.fail_on is None or self.fail_on in kw["messages"][0]["content"]):
                raise self.exc
            return _openai_response()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class FakeAnthropic:
    def __init__(self, exc=None):
        self.calls = []
        self.exc = exc

        async def create(**kw):
            self.calls.append(kw)
            if self.exc is not None:
                raise self.exc
            return _anthropic_response()
        self.messages = SimpleNamespace(create=create)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# the capability table
# ---------------------------------------------------------------------------

class TestModelCapabilities:
    @pytest.mark.parametrize("model", [
        "gpt-4.1-mini", "gpt-4o-mini", "claude-haiku-4-5-20251001", "claude-haiku-4-5",
        "claude-3-5-haiku-20241022", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-opus-4-6",
        "claude-3-5-sonnet-20241022",
    ])
    def test_models_that_take_a_temperature(self, model):
        assert accepts_temperature(model)
        assert model_capabilities(model)["temperature"] is True

    @pytest.mark.parametrize("model", [
        "claude-haiku-5-5", "CLAUDE-HAIKU-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-4-7",
        "claude-opus-4-8", "o1-mini", "o3-mini", "o4-mini", "gpt-5-mini", "gpt-5.6-luna", "gpt-6-luna",
    ])
    def test_models_that_refuse_one(self, model):
        assert not accepts_temperature(model)

    def test_the_platform_list_is_covered(self):
        """judge.py keeps its own copy (pipeline/3_judge.py imports judge.py without the package, so it cannot
        import gapgen); every fragment of the platform's list refuses here too."""
        from assistant_axis.gapgen.llm import NO_TEMPERATURE_FRAGMENTS, accepts_temperature as platform_accepts
        for frag in NO_TEMPERATURE_FRAGMENTS:
            assert MODEL_CAPABILITIES.get(frag) == {"temperature": False}, frag
        for model in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-4-7",
                      "claude-opus-4-8", "claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"):
            assert accepts_temperature(model) == platform_accepts(model), model


class TestFlex:
    """``"flex"`` in the table (2026-10-30, with run_pipeline.sh judging by Flex
    by default): OpenAI refused gpt-4.1-mini Flex with HTTP 400 on 2026-10-28 and
    has taken gpt-5.6-luna's since the cap pilot; any other model is untested."""

    def test_the_table_says_which_models_take_flex(self):
        assert accepts_flex("gpt-4.1-mini") is False
        assert accepts_flex("gpt-5.6-luna") is True
        assert accepts_flex("gpt-4.1") is None and accepts_flex("gpt-6-luna") is None
        assert model_capabilities("gpt-4.1-mini") == {"temperature": True, "flex": False, "cache_mode": False}
        assert model_capabilities("gpt-5.6-luna") == {"temperature": False, "flex": True, "cache_mode": True}

    def test_a_flex_call_to_a_refused_model_raises_before_sending(self):
        client = FakeOpenAI()
        with pytest.raises(JudgeRequestError, match="refuses Flex processing for gpt-4.1-mini"):
            _run(call_judge_single(client, "p", "gpt-4.1-mini", 50, RateLimiter(1000), service_tier="flex"))
        assert client.calls == []
        # the same model without the tier, and an untested model on Flex, are sent
        assert _run(call_judge_single(client, "p", "gpt-4.1-mini", 50, RateLimiter(1000))) == "SCORE: 2"
        assert _run(call_judge_single(client, "p", "gpt-4.1", 50, RateLimiter(1000),
                                      service_tier="flex")) == "SCORE: 2"
        assert [c.get("service_tier") for c in client.calls] == [None, "flex"]


class TestRequestTemperature:
    def test_default_and_none_are_left_out(self):
        for model in ("gpt-4.1-mini", "claude-haiku-4-5", "claude-haiku-5-5", "claude-opus-4-7", "gpt-6-luna"):
            assert request_temperature(model, None) is None
            assert request_temperature(model, DEFAULT_TEMPERATURE) is None
            assert request_temperature(model, 1) is None

    def test_non_default_is_sent_where_taken(self):
        assert request_temperature("claude-haiku-4-5", 0.0) == 0.0
        assert request_temperature("gpt-4.1-mini", 0.7) == 0.7

    def test_non_default_on_a_refusing_model_raises(self):
        with pytest.raises(JudgeRequestError, match="claude-haiku-5-5 refuses a temperature"):
            request_temperature("claude-haiku-5-5", 0.0)
        with pytest.raises(JudgeRequestError):
            request_temperature("gpt-6-luna", 0.0)


# ---------------------------------------------------------------------------
# the request builders
# ---------------------------------------------------------------------------

class TestChatRequestBody:
    def test_no_temperature_at_the_default(self):
        body = chat_request_body("p", "gpt-4.1-mini", 200)
        assert body == {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "p"}],
                        "max_completion_tokens": 200}
        assert "temperature" not in body

    def test_a_non_default_temperature_is_sent(self):
        assert chat_request_body("p", "gpt-4.1-mini", 200, temperature=0.0)["temperature"] == 0.0

    def test_refusing_model_raises_before_sending(self):
        with pytest.raises(JudgeRequestError):
            chat_request_body("p", "o3-mini", 200, temperature=0.0)

    def test_live_call_sends_the_same_body(self):
        client = FakeOpenAI()
        out = _run(call_judge_single(client, "p", "gpt-4.1-mini", 200, RateLimiter(1000)))
        assert out == "SCORE: 2"
        assert client.calls == [chat_request_body("p", "gpt-4.1-mini", 200)]
        assert "temperature" not in client.calls[0]


class TestAnthropicRequest:
    def test_default_temperature_is_not_sent(self):
        client = FakeAnthropic()
        _run(call_anthropic_judge_single(client, "p", "claude-haiku-4-5", 100, RateLimiter(1000)))
        assert client.calls == [{"model": "claude-haiku-4-5", "max_tokens": 100,
                                 "messages": [{"role": "user", "content": "p"}]}]

    def test_default_temperature_on_a_refusing_model_is_not_sent(self):
        """Opus 4.7 and later refuse any temperature, the default included: the call goes out without it."""
        client = FakeAnthropic()
        out = _run(call_anthropic_judge_single(client, "p", "claude-opus-4-7", 100, RateLimiter(1000),
                                               temperature=1.0))
        assert out == "SCORE: 2" and "temperature" not in client.calls[0]

    def test_non_default_temperature_is_sent_where_taken(self):
        client = FakeAnthropic()
        _run(call_anthropic_judge_single(client, "p", "claude-haiku-4-5", 100, RateLimiter(1000), temperature=0.0))
        assert client.calls[0]["temperature"] == 0.0

    def test_non_default_on_a_refusing_model_raises_before_any_call(self):
        client = FakeAnthropic()
        with pytest.raises(JudgeRequestError):
            _run(call_anthropic_judge_single(client, "p", "claude-haiku-5-5", 100, RateLimiter(1000),
                                             temperature=0.0))
        assert client.calls == []


class TestUnified:
    def test_temperature_reaches_the_openai_path(self):
        client = FakeOpenAI()
        _run(call_judge_single_unified(prompt="p", model="gpt-4.1-mini", max_tokens=50,
                                       rate_limiter=RateLimiter(1000), openai_client=client, temperature=0.0))
        assert client.calls[0]["temperature"] == 0.0

    def test_default_sends_none_on_either_path(self):
        oa, an = FakeOpenAI(), FakeAnthropic()
        for model in ("gpt-4.1-mini", "claude-haiku-4-5-20251001"):
            _run(call_judge_single_unified(prompt="p", model=model, max_tokens=50, rate_limiter=RateLimiter(1000),
                                           openai_client=oa, anthropic_client=an))
        assert "temperature" not in oa.calls[0] and "temperature" not in an.calls[0]


# ---------------------------------------------------------------------------
# an HTTP 400 is raised, other failures still give None
# ---------------------------------------------------------------------------

def _sdk_400(sdk):
    req = httpx.Request("POST", "https://api.example.invalid/v1")
    return sdk.BadRequestError(message="`temperature` is deprecated for this model",
                               response=httpx.Response(400, request=req), body=None)


class TestBadRequestIsSurfaced:
    def test_openai_400_raises(self):
        import openai
        client = FakeOpenAI(exc=_sdk_400(openai))
        with pytest.raises(JudgeRequestError, match="HTTP 400 from judge model gpt-4.1-mini"):
            _run(call_judge_single(client, "p", "gpt-4.1-mini", 200, RateLimiter(1000)))

    def test_anthropic_400_raises(self):
        import anthropic
        client = FakeAnthropic(exc=_sdk_400(anthropic))
        with pytest.raises(JudgeRequestError, match="deprecated for this model"):
            _run(call_anthropic_judge_single(client, "p", "claude-opus-4-7", 100, RateLimiter(1000)))

    def test_unified_400_raises(self):
        client = FakeAnthropic(exc=_Status400("bad"))
        with pytest.raises(JudgeRequestError):
            _run(call_judge_single_unified(prompt="p", model="claude-haiku-5-5", max_tokens=50,
                                           rate_limiter=RateLimiter(1000), anthropic_client=client))

    def test_other_failures_still_give_none(self):
        assert _run(call_judge_single(FakeOpenAI(exc=_Status500("server")), "p", "gpt-4.1-mini", 200,
                                      RateLimiter(1000))) is None
        assert _run(call_anthropic_judge_single(FakeAnthropic(exc=RuntimeError("network")), "p",
                                                "claude-haiku-4-5", 100, RateLimiter(1000))) is None

    def test_batch_raises_on_a_400_and_keeps_none_for_other_failures(self):
        client = FakeOpenAI(fail_on="bad", exc=_Status400("bad request"))
        with pytest.raises(JudgeRequestError):
            _run(call_judge_batch(client, ["ok 1", "bad 2", "ok 3"], "gpt-4.1-mini", 200, RateLimiter(1000)))
        client = FakeOpenAI(fail_on="bad", exc=_Status500("server"))
        out = _run(call_judge_batch(client, ["ok 1", "bad 2", "ok 3"], "gpt-4.1-mini", 200, RateLimiter(1000)))
        assert out == ["SCORE: 2", None, "SCORE: 2"]

    def test_the_error_is_a_value_error(self):
        """Callers that catch ValueError for a bad request keep working."""
        assert issubclass(JudgeRequestError, ValueError)
        assert J._is_bad_request(_Status400()) and not J._is_bad_request(_Status500())


# ---------------------------------------------------------------------------
# B1b: a 400 that flags one prompt's content stays per-item
# ---------------------------------------------------------------------------

#: The inner error objects the OpenAI SDK puts in ``body`` (and whose ``code`` it copies onto the exception).
OPENAI_FLAGGED = {"message": "Invalid prompt: your prompt was flagged as potentially violating our usage policy. "
                             "Please try again with a different prompt.",
                  "type": "invalid_request_error", "param": None, "code": "invalid_prompt"}
OPENAI_TEMPERATURE = {"message": "Unsupported value: 'temperature' does not support 0 with this model. Only the "
                                 "default (1) value is supported.",
                      "type": "invalid_request_error", "param": "temperature", "code": "unsupported_value"}
#: The Anthropic SDK keeps the whole error document in ``body`` and sets no ``code``.
ANTHROPIC_TEMPERATURE = {"type": "error", "error": {"type": "invalid_request_error",
                                                   "message": "`temperature` is deprecated for this model."}}


def _openai_400(body):
    import openai
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(f"Error code: 400 - {{'error': {body}}}", response=httpx.Response(400, request=req),
                                  body=body)


def _anthropic_400(body):
    import anthropic
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(f"Error code: 400 - {body}", response=httpx.Response(400, request=req),
                                     body=body)


@pytest.fixture
def flagged(monkeypatch, caplog):
    monkeypatch.setattr(J, "CONTENT_FLAGGED", {})
    caplog.set_level("WARNING", logger="assistant_axis.judge")
    return caplog


class TestContentFlaggedPromptIsPerItem:
    def test_the_two_openai_400s_are_told_apart(self):
        assert J.content_flag(_openai_400(OPENAI_FLAGGED)) == "invalid_prompt"
        assert J.content_flag(_openai_400(OPENAI_TEMPERATURE)) is None
        assert J.content_flag(_openai_400({"message": "x", "code": "content_policy_violation"})) \
            == "content_policy_violation"
        assert J.content_flag(_anthropic_400(ANTHROPIC_TEMPERATURE)) is None
        # a code nested in the body, and the phrase fallback where no code says it
        assert J.content_flag(_Status400("bad")) is None
        e = _Status400("x")
        e.body = {"error": {"code": "content_filter", "message": "filtered"}}
        assert J.content_flag(e) == "content_filter"
        assert J.content_flag(_Status400("Your prompt was flagged as potentially violating our usage policy")) \
            == "flagged as potentially violating"
        assert J.content_flag(_Status500("flagged as potentially violating")) is None   # only a 400

    def test_single_call_gives_none_and_warns(self, flagged):
        out = _run(call_judge_single(FakeOpenAI(exc=_openai_400(OPENAI_FLAGGED)), "p", "gpt-4.1-mini", 200,
                                     RateLimiter(1000)))
        assert out is None
        assert J.CONTENT_FLAGGED == {"gpt-4.1-mini": 1}
        assert "content flag (invalid_prompt) from judge model gpt-4.1-mini" in flagged.text
        assert "1 flagged so far" in flagged.text

    def test_batch_keeps_the_other_items(self, flagged):
        client = FakeOpenAI(fail_on="bad", exc=_openai_400(OPENAI_FLAGGED))
        out = _run(call_judge_batch(client, ["ok 1", "bad 2", "ok 3", "bad 4"], "gpt-4.1-mini", 200,
                                    RateLimiter(1000)))
        assert out == ["SCORE: 2", None, "SCORE: 2", None]
        assert J.CONTENT_FLAGGED == {"gpt-4.1-mini": 2} and "2 flagged so far" in flagged.text

    def test_anthropic_and_unified_paths(self, flagged):
        flag_doc = {"type": "error", "error": {"type": "invalid_request_error",
                                               "message": "Your prompt was flagged as potentially violating our "
                                                          "usage policy."}}
        assert _run(call_anthropic_judge_single(FakeAnthropic(exc=_anthropic_400(flag_doc)), "p",
                                                "claude-haiku-4-5", 100, RateLimiter(1000))) is None
        assert _run(call_judge_single_unified(prompt="p", model="gpt-4.1-mini", max_tokens=50,
                                              rate_limiter=RateLimiter(1000),
                                              openai_client=FakeOpenAI(exc=_openai_400(OPENAI_FLAGGED)))) is None
        assert _run(call_judge_single_unified(prompt="p", model="claude-haiku-4-5", max_tokens=50,
                                              rate_limiter=RateLimiter(1000),
                                              anthropic_client=FakeAnthropic(exc=_anthropic_400(flag_doc)))) is None
        assert J.CONTENT_FLAGGED == {"claude-haiku-4-5": 2, "gpt-4.1-mini": 1}
        assert flagged.text.count("HTTP 400 content flag") == 3

    def test_an_unsupported_parameter_400_still_raises_in_every_path(self, flagged):
        rl = RateLimiter(1000)
        with pytest.raises(JudgeRequestError, match="does not support 0"):
            _run(call_judge_single(FakeOpenAI(exc=_openai_400(OPENAI_TEMPERATURE)), "p", "gpt-4.1-mini", 200, rl))
        with pytest.raises(JudgeRequestError):
            _run(call_judge_batch(FakeOpenAI(fail_on="bad", exc=_openai_400(OPENAI_TEMPERATURE)),
                                  ["ok 1", "bad 2"], "gpt-4.1-mini", 200, rl))
        with pytest.raises(JudgeRequestError, match="deprecated for this model"):
            _run(call_anthropic_judge_single(FakeAnthropic(exc=_anthropic_400(ANTHROPIC_TEMPERATURE)), "p",
                                             "claude-opus-4-7", 100, rl))
        with pytest.raises(JudgeRequestError):
            _run(call_judge_single_unified(prompt="p", model="gpt-4.1-mini", max_tokens=50, rate_limiter=rl,
                                           openai_client=FakeOpenAI(exc=_openai_400(OPENAI_TEMPERATURE))))
        with pytest.raises(JudgeRequestError):
            _run(call_judge_single_unified(prompt="p", model="claude-opus-4-7", max_tokens=50, rate_limiter=rl,
                                           anthropic_client=FakeAnthropic(exc=_anthropic_400(ANTHROPIC_TEMPERATURE))))
        assert J.CONTENT_FLAGGED == {}
        assert "content flag" not in flagged.text


# ---------------------------------------------------------------------------
# judge request settings: judges run without thinking or reasoning
# (2026-10-26, the step-3 judge comparison)
# ---------------------------------------------------------------------------

class _ScriptedAnthropic:
    """An Anthropic client whose reply is set per test: content blocks, stop reason, stop details."""

    def __init__(self, blocks, stop_reason="end_turn", stop_details=None):
        self.calls = []

        async def create(**kw):
            self.calls.append(kw)
            return SimpleNamespace(content=blocks, stop_reason=stop_reason, stop_details=stop_details,
                                   usage=SimpleNamespace(input_tokens=300, output_tokens=40))
        self.messages = SimpleNamespace(create=create)


def _text(t):
    return SimpleNamespace(type="text", text=t)


class TestJudgeRequestSettings:
    def test_settings_by_model(self):
        for model in ("gpt-6-luna", "gpt-5.4-mini", "gpt-5.4-mini-2026-03-17", "gpt-5.6-luna"):
            assert J.judge_request_settings(model) == {"reasoning_effort": "none"}, model
        assert J.judge_request_settings("claude-haiku-5-5") == {"thinking": {"type": "disabled"}}
        # GPT-5 and GPT-5 mini refuse effort "none": they get nothing.
        for model in ("gpt-4.1-mini", "claude-haiku-4-5-20251001", "claude-sonnet-4-6", "o3-mini",
                      "gpt-5", "gpt-5-mini"):
            assert J.judge_request_settings(model) == {}, model

    def test_luna_chat_body_and_batch_line_carry_reasoning_effort_none(self):
        body = chat_request_body("p", "gpt-6-luna", 200)
        assert body["reasoning_effort"] == "none" and "temperature" not in body
        client = FakeOpenAI()
        _run(call_judge_single(client, "p", "gpt-6-luna", 200, RateLimiter(1000)))
        # the live call is the body with prompt_cache_options moved into extra_body (2026-11-01)
        assert client.calls == [J.chat_create_kwargs(body)]
        assert body["prompt_cache_options"] == {"mode": "explicit"}

    def test_gpt_mini_body_is_unchanged(self):
        assert chat_request_body("p", "gpt-4.1-mini", 200) == {
            "model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "p"}], "max_completion_tokens": 200}

    def test_haiku_5_5_request_turns_thinking_off(self):
        client = _ScriptedAnthropic([_text("Reason.\nSCORE: 3")])
        out = _run(call_anthropic_judge_single(client, "p", "claude-haiku-5-5", 200, RateLimiter(1000)))
        assert out == "Reason.\nSCORE: 3"
        assert client.calls[0]["thinking"] == {"type": "disabled"} and "temperature" not in client.calls[0]

    def test_haiku_4_5_request_is_unchanged(self):
        client = _ScriptedAnthropic([_text("SCORE: 1")])
        _run(call_anthropic_judge_single(client, "p", "claude-haiku-4-5", 100, RateLimiter(1000)))
        assert client.calls == [{"model": "claude-haiku-4-5", "max_tokens": 100,
                                 "messages": [{"role": "user", "content": "p"}]}]


class TestAnthropicReplyReading:
    def test_text_is_read_by_block_type(self):
        """A model that thinks can begin its content with thinking blocks; the answer is the text blocks."""
        blocks = [SimpleNamespace(type="thinking", thinking="", signature="sig"), _text("SCORE: "), _text("2")]
        out = _run(call_anthropic_judge_single(_ScriptedAnthropic(blocks), "p", "claude-haiku-5-5", 200,
                                               RateLimiter(1000)))
        assert out == "SCORE: 2"

    def test_a_refusal_gives_none_and_is_counted(self, monkeypatch, caplog):
        monkeypatch.setattr(J, "REFUSED_REPLIES", {})
        client = _ScriptedAnthropic([], stop_reason="refusal",
                                    stop_details=SimpleNamespace(type="refusal", category="cyber", explanation="x"))
        from assistant_axis.judge_pricing import MultiModelUsage
        u = MultiModelUsage()
        out = _run(call_anthropic_judge_single(client, "p", "claude-haiku-5-5", 200, RateLimiter(1000), usage=u))
        assert out is None and J.REFUSED_REPLIES == {"claude-haiku-5-5": 1}
        assert "refused a judge prompt (category cyber" in caplog.text
        assert u.n_calls == 1                        # a refusal is charged all the same

    def test_a_truncated_reply_is_rejected_only_when_asked(self, monkeypatch):
        monkeypatch.setattr(J, "TRUNCATED_REPLIES", {})
        blocks = [_text("The response plays the role because")]
        kept = _run(call_anthropic_judge_single(_ScriptedAnthropic(blocks, stop_reason="max_tokens"), "p",
                                                "claude-haiku-5-5", 10, RateLimiter(1000)))
        assert kept == "The response plays the role because" and J.TRUNCATED_REPLIES == {}
        cut = _run(call_anthropic_judge_single(_ScriptedAnthropic(blocks, stop_reason="max_tokens"), "p",
                                               "claude-haiku-5-5", 10, RateLimiter(1000), reject_truncated=True))
        assert cut is None and J.TRUNCATED_REPLIES == {"claude-haiku-5-5": 1}

    def test_an_empty_text_gives_none(self):
        assert _run(call_anthropic_judge_single(_ScriptedAnthropic([]), "p", "claude-haiku-5-5", 200,
                                                RateLimiter(1000))) is None


class TestBatchRoutesByProvider:
    def test_a_claude_model_goes_to_the_anthropic_client(self):
        client = _ScriptedAnthropic([_text("SCORE: 3")])
        out = _run(call_judge_batch(client, ["a", "b"], "claude-haiku-5-5", 200, RateLimiter(1000)))
        assert out == ["SCORE: 3", "SCORE: 3"] and len(client.calls) == 2
        assert all(c["thinking"] == {"type": "disabled"} for c in client.calls)

    def test_a_gpt_model_and_an_unknown_name_go_to_the_client_given(self):
        for model in ("gpt-6-luna", "some-model"):
            client = FakeOpenAI()
            assert _run(call_judge_batch(client, ["a"], model, 200, RateLimiter(1000))) == ["SCORE: 2"]
            assert len(client.calls) == 1

    def test_reject_truncated_reaches_the_anthropic_path(self, monkeypatch):
        monkeypatch.setattr(J, "TRUNCATED_REPLIES", {})
        client = _ScriptedAnthropic([_text("cut")], stop_reason="max_tokens")
        out = _run(call_judge_batch(client, ["a"], "claude-haiku-5-5", 10, RateLimiter(1000), reject_truncated=True))
        assert out == [None] and J.TRUNCATED_REPLIES == {"claude-haiku-5-5": 1}


def test_luna_is_priced():
    from assistant_axis.judge_pricing import price_for_model
    assert price_for_model("gpt-6-luna") == (0.10, 0.50)
    assert price_for_model("gpt-6-luna:batch") == (0.05, 0.25)
    assert price_for_model("claude-haiku-5-5") == (0.10, 0.50)


# ---------------------------------------------------------------------------
# OpenAI Flex processing (2026-10-26: Luna's batches did not move for hours)
# ---------------------------------------------------------------------------

class _UsageOpenAI(FakeOpenAI):
    def __init__(self):
        super().__init__()

        async def create(**kw):
            self.calls.append(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="SCORE: 3"),
                                                            finish_reason="stop")],
                                   usage=SimpleNamespace(prompt_tokens=700, completion_tokens=50))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class TestFlex:
    def test_body_carries_the_tier_only_when_asked(self):
        assert "service_tier" not in chat_request_body("p", "gpt-6-luna", 200)
        assert chat_request_body("p", "gpt-6-luna", 200, service_tier="flex")["service_tier"] == "flex"

    def test_an_unknown_tier_is_refused(self):
        with pytest.raises(ValueError, match="service_tier"):
            chat_request_body("p", "gpt-6-luna", 200, service_tier="priority")

    def test_flex_calls_are_charged_under_model_flex_at_the_batch_rate(self):
        from assistant_axis.judge_pricing import MultiModelUsage, price_for_model
        client, u = _UsageOpenAI(), MultiModelUsage()
        out = _run(call_judge_batch(client, ["a", "b"], "gpt-6-luna", 200, RateLimiter(1000), usage=u,
                                    service_tier="flex"))
        assert out == ["SCORE: 3", "SCORE: 3"]
        assert all(c["service_tier"] == "flex" and c["reasoning_effort"] == "none" for c in client.calls)
        assert set(u.as_dict()["per_model"]) == {"gpt-6-luna:flex"}
        assert price_for_model("gpt-6-luna:flex") == price_for_model("gpt-6-luna:batch") == (0.05, 0.25)

    def test_flex_is_openai_only_in_pricing(self):
        from assistant_axis.judge_pricing import price_for_model
        with pytest.raises(KeyError, match="Flex"):
            price_for_model("claude-haiku-5-5:flex")


# ---------------------------------------------------------------------------
# OpenAI prompt caching (2026-11-01): off by default on GPT-5.6 and later
# ---------------------------------------------------------------------------

class _CacheUsageOpenAI(FakeOpenAI):
    """Replies with the usage block luna returns (checked live on Flex, 2026-11-01): 1,218 prompt tokens
    of which ``cache_write`` were written to the cache or ``cached`` read from it."""

    def __init__(self, cache_write=0, cached=0):
        super().__init__()

        async def create(**kw):
            self.calls.append(kw)
            details = SimpleNamespace(cached_tokens=cached, cache_write_tokens=cache_write, audio_tokens=0)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="SCORE: 3"),
                                                            finish_reason="stop")],
                                   usage=SimpleNamespace(prompt_tokens=1218, completion_tokens=4,
                                                         prompt_tokens_details=details))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class TestPromptCache:
    """GPT-5.6 and later cache by default and bill a cache write at 1.25x the input rate; a judge prompt
    is never read back (luna Flex, 29 Oct to 1 Nov 2026: $198 of writes for $0.05 of reads).  "off", the
    default, sends those models ``prompt_cache_options`` explicit mode with no breakpoints, which writes
    and reads nothing; gpt-4.1-mini refuses the field (HTTP 400) and writes free, so it is sent nothing."""

    def test_the_table_says_which_models_take_the_field(self):
        assert J.accepts_cache_options("gpt-5.6-luna") is True
        assert J.accepts_cache_options("gpt-6-luna") is True
        assert J.accepts_cache_options("gpt-4.1-mini") is False
        assert J.accepts_cache_options("gpt-5.4-mini") is None
        assert J.accepts_cache_options("claude-haiku-5-5") is None
        assert not accepts_temperature("gpt-6-luna")      # the gpt-6 row keeps its temperature entry
        assert J.DEFAULT_PROMPT_CACHE == "off" and J.PROMPT_CACHE_MODES == ("off", "implicit")

    def test_off_sends_explicit_mode_with_no_breakpoints_to_a_model_that_takes_it(self):
        assert J.prompt_cache_fields("gpt-5.6-luna") == {"prompt_cache_options": {"mode": "explicit"}}
        assert J.prompt_cache_fields("gpt-5.6-luna", "implicit") == {}
        for model in ("gpt-4.1-mini", "gpt-5.4-mini", "claude-haiku-5-5"):
            assert J.prompt_cache_fields(model) == {} and J.prompt_cache_fields(model, "implicit") == {}
        with pytest.raises(ValueError, match="prompt_cache"):
            J.prompt_cache_fields("gpt-5.6-luna", "explicit")

    def test_the_body_carries_the_field_and_the_sdk_call_moves_it_to_extra_body(self):
        body = chat_request_body("p", "gpt-5.6-luna", 200, service_tier="flex")
        assert body["prompt_cache_options"] == {"mode": "explicit"} and "extra_body" not in body
        kwargs = J.chat_create_kwargs(body)
        assert kwargs["extra_body"] == {"prompt_cache_options": {"mode": "explicit"}}
        assert "prompt_cache_options" not in kwargs
        assert {k: v for k, v in kwargs.items() if k != "extra_body"} \
            == {k: v for k, v in body.items() if k != "prompt_cache_options"}
        assert "prompt_cache_options" not in chat_request_body("p", "gpt-5.6-luna", 200, prompt_cache="implicit")
        mini = chat_request_body("p", "gpt-4.1-mini", 200)
        assert "prompt_cache_options" not in mini and J.chat_create_kwargs(mini) == mini

    def test_live_calls_send_it_and_implicit_leaves_it_out(self):
        client = _CacheUsageOpenAI()
        _run(call_judge_single(client, "p", "gpt-5.6-luna", 200, RateLimiter(1000), service_tier="flex"))
        assert client.calls[0]["extra_body"] == {"prompt_cache_options": {"mode": "explicit"}}
        assert "prompt_cache_options" not in client.calls[0] and client.calls[0]["service_tier"] == "flex"
        client = _CacheUsageOpenAI()
        _run(call_judge_single(client, "p", "gpt-5.6-luna", 200, RateLimiter(1000), prompt_cache="implicit"))
        assert "extra_body" not in client.calls[0] and "prompt_cache_options" not in client.calls[0]
        client = _CacheUsageOpenAI()
        _run(call_judge_batch(client, ["a", "b"], "gpt-5.6-luna", 200, RateLimiter(1000)))
        assert all(c["extra_body"] == {"prompt_cache_options": {"mode": "explicit"}} for c in client.calls)
        client = _CacheUsageOpenAI()
        _run(call_judge_batch(client, ["a", "b"], "gpt-5.6-luna", 200, RateLimiter(1000), prompt_cache="implicit"))
        assert all("extra_body" not in c for c in client.calls)

    def test_the_fallback_to_the_standard_tier_keeps_the_mode(self, monkeypatch):
        monkeypatch.setattr(J, "FLEX_FALLBACK_AFTER_S", 0.0)
        monkeypatch.setattr(J, "FLEX_RETRY_SLEEP_S", 0.0)
        monkeypatch.setattr(J, "FLEX_FALLBACKS", {})
        flex, standard = FakeOpenAI(exc=_Status500("busy")), _CacheUsageOpenAI()
        out = _run(call_judge_single(flex, "p", "gpt-5.6-luna", 200, RateLimiter(1000), service_tier="flex",
                                     fallback_client=standard))
        assert out == "SCORE: 3" and J.FLEX_FALLBACKS == {"gpt-5.6-luna": 1}
        assert standard.calls[0]["extra_body"] == {"prompt_cache_options": {"mode": "explicit"}}
        assert "service_tier" not in standard.calls[0]

    def test_cache_tokens_are_charged_at_their_factors(self):
        from assistant_axis.judge_pricing import MultiModelUsage
        client, u = _CacheUsageOpenAI(cache_write=1215), MultiModelUsage()
        _run(call_judge_single(client, "p", "gpt-5.6-luna", 200, RateLimiter(1000), usage=u, service_tier="flex",
                               prompt_cache="implicit"))
        t = u.per_model["gpt-5.6-luna:flex"]
        assert (t.prompt_tokens, t.completion_tokens, t.cache_write_tokens, t.cache_read_tokens) == (1218, 4, 1215, 0)
        # 3 plain tokens at the $0.10 Flex rate, 1,215 written at 1.25 x $0.10, 4 out at $0.60 (per 1M)
        assert t.cost_usd == pytest.approx((3 * 0.10 + 1215 * 0.125 + 4 * 0.60) / 1e6)
        client, u = _CacheUsageOpenAI(cached=1215), MultiModelUsage()
        _run(call_judge_single(client, "p", "gpt-5.6-luna", 200, RateLimiter(1000), usage=u))
        t = u.per_model["gpt-5.6-luna"]
        assert t.cache_read_tokens == 1215
        assert t.cost_usd == pytest.approx((3 * 0.20 + 1215 * 0.02 + 4 * 1.20) / 1e6)
        assert "cache_write=1,215" not in u.log_line() and "cache_read=1,215" in u.log_line()
