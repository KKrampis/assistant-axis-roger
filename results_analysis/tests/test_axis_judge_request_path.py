"""The axis judge's requests go through judge.py (2026-11-02): before that, its own requests sent
temperature 0 and no reasoning setting, GPT-5.6 Luna and Haiku 5.5 answered HTTP 400, and the run
"completed" with no score at all."""
import asyncio
from types import SimpleNamespace

import pytest

from assistant_axis import judge as J
from results_analysis import axis_judge_correlation as A


class HTTPError(Exception):
    def __init__(self, status_code, message, *, code=None):
        super().__init__(message)
        self.status_code, self.message, self.code, self.body = status_code, message, code, None


class Fake:
    """Either SDK's client, with the queued errors raised first."""

    def __init__(self, *errors, finish="stop", stop="end_turn"):
        self.errors, self.calls, self.finish, self.stop = list(errors), [], finish, stop
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.messages = SimpleNamespace(create=self._messages)

    async def _chat(self, **kw):
        self.calls.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="SCORE: 2"), finish_reason=self.finish)],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    async def _messages(self, **kw):
        self.calls.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="SCORE: 2")], stop_reason=self.stop, stop_details=None,
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))


@pytest.fixture
def client(monkeypatch):
    holder = {}

    def install(fake):
        import anthropic
        import openai
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda *a, **k: fake)
        monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: fake)
        holder["fake"] = fake
        return fake
    real_sleep = asyncio.sleep
    monkeypatch.setattr(A.asyncio, "sleep", lambda s: real_sleep(0))      # the retry backoff, not waited out
    return install


def _openai(prompts, model, T):
    return asyncio.run(A._call_openai_batch(prompts, model, 50, T, J.RateLimiter(1000), 5))


def _anthropic(prompts, model, T):
    return asyncio.run(A._call_anthropic_batch(prompts, model, 50, T, J.RateLimiter(1000), 5))


def test_luna_goes_out_as_judge_py_sends_it(client):
    fake = client(Fake())
    assert _openai(["P"], "gpt-5.6-luna", 1.0) == ["SCORE: 2"]
    kw = fake.calls[0]
    assert "temperature" not in kw and kw["reasoning_effort"] == "none" and kw["max_completion_tokens"] == 50
    assert kw["extra_body"] == {"prompt_cache_options": {"mode": "explicit"}}


def test_haiku_goes_out_as_judge_py_sends_it(client):
    fake = client(Fake())
    assert _anthropic(["P"], "claude-haiku-5-5", 1.0) == ["SCORE: 2"]
    kw = fake.calls[0]
    assert "temperature" not in kw and kw["thinking"] == {"type": "disabled"} and kw["max_tokens"] == 50


def test_the_old_judges_requests_are_unchanged(client):
    fake = client(Fake())
    _openai(["P"], "gpt-4.1-mini", 0.0)
    assert fake.calls[0] == {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "P"}],
                             "max_completion_tokens": 50, "temperature": 0.0}
    fake = client(Fake())
    _anthropic(["P"], "claude-sonnet-4-6", 0.0)
    assert fake.calls[0] == {"model": "claude-sonnet-4-6", "max_tokens": 50,
                             "messages": [{"role": "user", "content": "P"}], "temperature": 0.0}


@pytest.mark.parametrize("call,model", [(_openai, "gpt-5.6-luna"), (_anthropic, "claude-haiku-5-5")])
def test_a_temperature_the_model_refuses_stops_before_anything_is_sent(client, call, model):
    fake = client(Fake())
    with pytest.raises(J.JudgeRequestError):
        call(["P"], model, 0.0)
    assert not fake.calls


def test_a_rejected_request_stops_the_run(client):
    client(Fake(HTTPError(400, "Unsupported parameter")))
    with pytest.raises(J.JudgeRequestError):
        _openai(["P"], "gpt-4.1-mini", 0.0)


def test_no_credit_stops_the_run(client):
    client(Fake(HTTPError(429, "You exceeded your current quota", code="insufficient_quota")))
    with pytest.raises(J.BillingError):
        _openai(["P"], "gpt-4.1-mini", 0.0)


def test_a_content_flag_stays_with_its_item(client):
    client(Fake(HTTPError(400, "flagged", code="invalid_prompt")))
    assert _openai(["P", "Q"], "gpt-4.1-mini", 0.0).count(None) == 1


def test_a_transient_error_is_retried(client):
    fake = client(Fake(ConnectionError("no route")))
    assert _openai(["P"], "gpt-4.1-mini", 0.0) == ["SCORE: 2"] and len(fake.calls) == 2


def test_a_cut_reply_is_counted_and_never_parsed(client):
    before = J.TRUNCATED_REPLIES.get("gpt-4.1-mini", 0)
    client(Fake(finish="length"))
    assert _openai(["P"], "gpt-4.1-mini", 0.0) == [None]
    assert J.TRUNCATED_REPLIES["gpt-4.1-mini"] == before + 1
    client(Fake(stop="max_tokens"))
    assert _anthropic(["P"], "claude-sonnet-4-6", 0.0) == [None]
    client(Fake(stop="refusal"))
    assert _anthropic(["P"], "claude-sonnet-4-6", 0.0) == [None]


BASE = ["x", "--data_dir", "d", "--output_dir", "o", "--pair", "a", "b", "--pair_type", "traits"]


def _args(monkeypatch, *extra):
    monkeypatch.setattr("sys.argv", [*BASE, *extra])
    return A.parse_args()


@pytest.mark.parametrize("extra,temperature", [
    (("--score_descriptions", "--provider", "openai"), 0.0),                                         # gpt-4.1-mini
    (("--score_descriptions",), 0.0),                                                                # claude-sonnet-4-6
    (("--score_descriptions", "--provider", "openai", "--judge_model", "gpt-5.6-luna"), 1.0),        # refuses any other
    (("--score_descriptions", "--judge_model", "claude-haiku-5-5"), 1.0),
    (("--score_descriptions", "--provider", "openai", "--temperature", "0.5"), 0.5),
])
def test_the_default_temperature_is_0_where_the_model_takes_one(monkeypatch, extra, temperature):
    assert _args(monkeypatch, *extra).temperature == temperature


def test_a_temperature_the_model_refuses_is_refused_on_the_command_line(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _args(monkeypatch, "--score_descriptions", "--provider", "openai", "--judge_model", "gpt-5.6-luna", "--temperature", "0")
    assert "refuses a temperature" in capsys.readouterr().err


def test_a_model_of_another_provider_is_refused(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _args(monkeypatch, "--score_descriptions", "--judge_model", "gemini-3.1-flash-lite")
    assert "judge_descriptions.py" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _args(monkeypatch, "--score_descriptions", "--judge_model", "gpt-5.6-luna")     # --provider defaults to anthropic


def test_projections_only_needs_no_score_mode_and_excludes_them(monkeypatch):
    assert _args(monkeypatch, "--projections_only").projections_only
    with pytest.raises(SystemExit):
        _args(monkeypatch, "--projections_only", "--score_descriptions")
    with pytest.raises(SystemExit):
        _args(monkeypatch)
