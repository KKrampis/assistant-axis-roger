"""Requests, model capabilities and real-time / Flex calls for
``judge_tier_cost_eval.py``'s three transports (OpenAI Flex processing, the
providers' batch APIs, real time).

Adapted (2026-11-01) from the main checkout's ``assistant_axis/judge.py``,
where pipeline step 3 has had these pieces since October 2026:
``MODEL_CAPABILITIES``, ``JUDGE_REQUEST_SETTINGS``, ``chat_request_body``,
``anthropic_request_params``, ``anthropic_reply``, ``usage_key`` and the Flex
retry-then-fallback loop of ``call_judge_single``.  They live here rather
than in this tree's ``assistant_axis/judge.py`` (a September 2026 version,
left unchanged) so that the tier evaluation can change transport without
changing what every other judge call site in this tree sends.

**Request settings differ from step 3's on purpose.**  The tier evaluation's
new cells extend cells already judged by this tree's ``judge.py`` (branch
``personas-konstantinos``), so for the models those cells used the defaults
reproduce the requests that code sent: no ``reasoning_effort`` for the GPT-5.6
tiers (step 3 sends "none"; its ``"gpt-5.6"`` row would match gpt-5.6-sol, so
it is not copied), ``output_config: {"effort": "low"}`` for the Claude models
that code sent it to (``_ANTHROPIC_EFFORT_CAPABLE`` in this tree's
``judge.py``), nothing extra for claude-haiku-4-5.  Models those cells never
used follow step 3's convention: thinking disabled for Haiku 5.x, reasoning
effort "none" for GPT-5.4 and GPT-6.  The one difference from that code: a
temperature equal to the provider default (1) is left out of the request
instead of being sent as 1 (the same sampling, and it works on models that
refuse the field).

Only the standard library and this tree's ``assistant_axis`` are imported;
clients are passed in, so tests use fakes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from assistant_axis.judge import RateLimiter, provider_for_model
from assistant_axis.judge_pricing import BATCH_SUFFIX, FLEX_SUFFIX

logger = logging.getLogger(__name__)

#: Both providers' default sampling temperature (Chat Completions and Messages).
DEFAULT_TEMPERATURE = 1.0

#: Model-id fragments (lower case) and what the model takes, as in the main
#: checkout's ``judge.MODEL_CAPABILITIES`` (read 2026-11-01).  A fragment
#: matches at the start of the model id or after a hyphen.
#:
#: * ``"temperature"``: False for the models that refuse a ``temperature`` with
#:   an HTTP 400.  Only a non-default temperature consults it (the default is
#:   never sent), and the tier evaluation sends only the default.
#: * ``"flex"``: whether OpenAI takes ``service_tier="flex"`` for the model.
#:   False for gpt-4.1-mini (HTTP 400 "Invalid service_tier argument",
#:   2026-10-28), True for gpt-5.6-luna (step 3's judging since 2026-10-28); a
#:   model in no ``"flex"`` entry is untested (:func:`accepts_flex` gives None),
#:   and under ``--transport auto`` its first request probes it.  Add a row once
#:   a probe has told.
MODEL_CAPABILITIES: Dict[str, Dict[str, Optional[bool]]] = {
    # Anthropic
    "sonnet-5": {"temperature": False},
    "opus-5": {"temperature": False},
    "opus-4-7": {"temperature": False},
    "opus-4-8": {"temperature": False},
    "haiku-5": {"temperature": False},
    "fable": {"temperature": False},
    "mythos": {"temperature": False},
    # OpenAI reasoning models
    "o1": {"temperature": False},
    "o3": {"temperature": False},
    "o4": {"temperature": False},
    "gpt-5": {"temperature": False},
    "gpt-6": {"temperature": False},
    # OpenAI Flex processing
    "gpt-4.1-mini": {"flex": False},
    "gpt-5.6-luna": {"flex": True},
    # Probed 2026-11-01 (judge_tier_cost_eval.py --transport auto: first request on Flex, answered)
    "gpt-5.6-terra": {"flex": True},
    "gpt-5.6-sol": {"flex": True},
    "gpt-6-luna": {"flex": True},
    "gpt-5.4-mini": {"flex": True},
}

_EFFORT_LOW = {"output_config": {"effort": "low"}}

#: Fields a judge request adds for a model, by model-id fragment (matched as in
#: :data:`MODEL_CAPABILITIES`; every matching row applies, in table order).  To
#: add a model, add a row; to override for one run, ``--request_setting``
#: (:func:`parse_request_setting`), applied on top of this table.
#:
#: * Models the existing tier-evaluation cells used reproduce the requests of
#:   this tree's ``judge.py``: the GPT-5.6 tiers get nothing (so gpt-5.6-* are
#:   deliberately absent: step 3's ``"gpt-5.6"`` row would match all three),
#:   the Claude models in its ``_ANTHROPIC_EFFORT_CAPABLE`` get
#:   ``output_config.effort`` "low", claude-haiku-4-5 gets nothing.
#: * New models follow step 3: Haiku 5.5 thinks adaptively unless told not to
#:   and its thinking counts toward ``max_tokens``, so thinking is disabled;
#:   GPT-5.4 mini and GPT-6 Luna are reasoning models and take
#:   ``reasoning_effort`` "none" (their model pages, read in the main checkout
#:   2026-10-26/27).
JUDGE_REQUEST_SETTINGS: Dict[str, Dict[str, Any]] = {
    # new models, step 3's convention
    "gpt-5.4": {"reasoning_effort": "none"},
    "gpt-6": {"reasoning_effort": "none"},
    "haiku-5": {"thinking": {"type": "disabled"}},
    # this tree's judge.py, _ANTHROPIC_EFFORT_CAPABLE
    "opus-5": _EFFORT_LOW,
    "sonnet-5": _EFFORT_LOW,
    "fable-5": _EFFORT_LOW,
    "mythos-5": _EFFORT_LOW,
    "opus-4-6": _EFFORT_LOW,
    "opus-4-7": _EFFORT_LOW,
    "opus-4-8": _EFFORT_LOW,
    "sonnet-4-6": _EFFORT_LOW,
}


def _matches(model: str, fragment: str) -> bool:
    m = str(model).lower()
    return m.startswith(fragment) or f"-{fragment}" in m


#: One ``--request_setting``: (model fragment or None for every model, key, value);
#: a value of None removes the key.
RequestOverride = Tuple[Optional[str], str, Any]


def parse_request_setting(text: str) -> RequestOverride:
    """``[MODEL_FRAGMENT:]key=<json>`` -> ``(fragment, key, value)``.

    ``reasoning_effort="low"`` applies to every model of the run;
    ``gpt-6:reasoning_effort="low"`` only to models matching ``gpt-6`` (as
    :data:`JUDGE_REQUEST_SETTINGS` matches).  The value is JSON
    (``'{"type": "disabled"}'``, ``'"none"'``, ``200``); ``null`` removes a
    field the table would add (``opus-5:output_config=null``)."""
    if "=" not in text:
        raise ValueError(f"--request_setting {text!r}: expected [MODEL_FRAGMENT:]key=<json>")
    left, raw = text.split("=", 1)
    fragment: Optional[str] = None
    if ":" in left:
        fragment, left = left.split(":", 1)
        fragment = fragment.strip().lower() or None
    key = left.strip()
    if not key:
        raise ValueError(f"--request_setting {text!r}: empty key")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--request_setting {text!r}: the value is not JSON ({e}); "
                         f"quote a string, e.g. {key}='\"none\"'") from e
    return fragment, key, value


def judge_request_settings(model: str, overrides: Iterable[RequestOverride] = ()) -> Dict[str, Any]:
    """The fields a request to ``model`` adds: the :data:`JUDGE_REQUEST_SETTINGS`
    rows that match it, then the ``--request_setting`` overrides that apply to
    it (a None value removes the key).  A fresh dict (deep-copied values)."""
    out: Dict[str, Any] = {}
    for fragment, entry in JUDGE_REQUEST_SETTINGS.items():
        if _matches(model, fragment):
            out.update(json.loads(json.dumps(entry)))
    for fragment, key, value in overrides:
        if fragment is not None and not _matches(model, fragment):
            continue
        if value is None:
            out.pop(key, None)
        else:
            out[key] = json.loads(json.dumps(value))
    return out


class JudgeRequestError(ValueError):
    """A judge request that is wrong for its model: an HTTP 400 from the API,
    or a non-default temperature for a model that refuses one.  Raised, never
    swallowed, because every further call would fail the same way."""


class FlexRefusedError(JudgeRequestError):
    """OpenAI refused ``service_tier="flex"`` for the model (HTTP 400 naming the
    service tier), or :data:`MODEL_CAPABILITIES` says it would."""


def model_capabilities(model: str) -> Dict[str, Optional[bool]]:
    """``{"temperature": bool, "flex": bool or None}`` by :data:`MODEL_CAPABILITIES`."""
    caps: Dict[str, Optional[bool]] = {"temperature": True, "flex": None}
    for fragment, entry in MODEL_CAPABILITIES.items():
        if _matches(model, fragment):
            caps.update(entry)
    return caps


def accepts_temperature(model: str) -> bool:
    return bool(model_capabilities(model)["temperature"])


def accepts_flex(model: str) -> Optional[bool]:
    """True / False when known for ``model``, None when untested."""
    return model_capabilities(model)["flex"]


def request_temperature(model: str, temperature: Optional[float]) -> Optional[float]:
    """None (leave the field out) for None or the provider default, else the
    temperature; raises :class:`JudgeRequestError` for a non-default one on a
    model that refuses it."""
    if temperature is None or float(temperature) == DEFAULT_TEMPERATURE:
        return None
    if not accepts_temperature(model):
        raise JudgeRequestError(f"{model} refuses a temperature (MODEL_CAPABILITIES), so "
                                f"temperature={temperature} cannot be honoured")
    return float(temperature)


def _is_bad_request(exc: BaseException) -> bool:
    """An HTTP 400 from either SDK (both set ``status_code`` on the error)."""
    return getattr(exc, "status_code", None) == 400


#: An HTTP 400 that flags one prompt's content rather than the request's shape
#: (main checkout's judge.py, B1b): that item gives None with a WARNING.
CONTENT_FLAG_CODES: Tuple[str, ...] = ("invalid_prompt", "content_policy_violation", "content_filter")
CONTENT_FLAG_PHRASES: Tuple[str, ...] = ("flagged as potentially violating", "violating our usage policy",
                                         "content management policy")


def _error_field(exc: BaseException, name: str) -> Optional[str]:
    value = getattr(exc, name, None)
    if value:
        return str(value)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        if inner.get(name):
            return str(inner[name])
    return None


def content_flag(exc: BaseException) -> Optional[str]:
    """For an HTTP 400 that flags the prompt's content, the code or phrase that says so; else None."""
    if not _is_bad_request(exc):
        return None
    code = _error_field(exc, "code")
    if code in CONTENT_FLAG_CODES:
        return code
    text = f"{getattr(exc, 'message', '')} {exc}".lower()
    return next((p for p in CONTENT_FLAG_PHRASES if p in text), None)


def is_flex_refusal(exc: BaseException) -> bool:
    """An HTTP 400 refusing the service tier (gpt-4.1-mini, 2026-10-28:
    "Invalid service_tier argument"): its ``param`` is ``service_tier``, or its
    message names the tier."""
    if not _is_bad_request(exc):
        return False
    if _error_field(exc, "param") == "service_tier":
        return True
    text = f"{getattr(exc, 'message', '')} {exc}".lower()
    return "service_tier" in text or "service tier" in text


class BillingError(JudgeRequestError):
    """The provider refuses the account, not the request: out of credits or
    over quota.  Every further call fails the same way, so it stops the run at
    once instead of being retried (OpenAI, 2026-11-01: HTTP 429 with code
    ``credit_balance_exhausted``, type ``insufficient_quota``, "You have no
    credits remaining", which the Flex loop used to retry for 10 minutes per
    request and then leave unscored)."""


BILLING_ERROR_CODES: Tuple[str, ...] = ("insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached")
BILLING_ERROR_PHRASES: Tuple[str, ...] = ("no credits remaining", "exceeded your current quota",
                                          "credit balance is too low", "billing")


def is_billing_error(exc: BaseException) -> bool:
    """An error saying the account cannot pay: an OpenAI 429 whose code or type
    is a quota code, or any 4xx whose message says so (Anthropic answers a low
    credit balance with HTTP 400 "Your credit balance is too low")."""
    status = getattr(exc, "status_code", None)
    if status is None or not (400 <= int(status) < 500):
        return False
    if _error_field(exc, "code") in BILLING_ERROR_CODES or _error_field(exc, "type") in BILLING_ERROR_CODES:
        return True
    text = f"{getattr(exc, 'message', '')} {exc}".lower()
    return any(p in text for p in BILLING_ERROR_PHRASES[:3])


def _billing_error(model: str, exc: BaseException) -> BillingError:
    logger.error(f"Billing refusal from judge model {model}: {exc}")
    return BillingError(f"{model}: the provider refuses the account ({exc}).  Every call would fail the same "
                        f"way: add credits or raise the quota, then rerun (finished cells are skipped)")


def _bad_request_error(model: str, exc: BaseException) -> JudgeRequestError:
    logger.error(f"HTTP 400 from judge model {model}: {exc}")
    return JudgeRequestError(f"HTTP 400 from judge model {model}: {exc}.  The API rejected the request "
                             f"itself, so every call would fail the same way (check the request settings "
                             f"and MODEL_CAPABILITIES in results_analysis/judge_tier_transport.py)")


#: The OpenAI service tiers a call may name.  ``"flex"``: lower priority,
#: priced at Batch rates, answered like a real-time call but slower.
SERVICE_TIERS = ("flex",)

#: Flex is tried this long before a request falls back to the standard tier
#: (main checkout, Roger 2026-10-29: "only after trying for a while"); between
#: attempts the call sleeps FLEX_RETRY_SLEEP_S.  Each attempt is the Flex
#: client's own retries (FLEX_MAX_RETRIES, with a FLEX_TIMEOUT_S timeout, as
#: step 3's ``judge_client``).
FLEX_FALLBACK_AFTER_S = 600.0
FLEX_RETRY_SLEEP_S = 30.0
FLEX_TIMEOUT_S = 900.0
FLEX_MAX_RETRIES = 8


def usage_key(model: str, service_tier: Optional[str] = None) -> str:
    """``<model>:flex`` for a Flex call (priced at the batch rate), else ``model``."""
    return f"{model}:{service_tier}" if service_tier else model


def chat_request_body(prompt: str, model: str, max_tokens: int, *,
                      settings: Optional[Dict[str, Any]] = None,
                      temperature: Optional[float] = DEFAULT_TEMPERATURE,
                      service_tier: Optional[str] = None) -> Dict[str, Any]:
    """The chat-completion request: the keyword arguments of
    ``client.chat.completions.create`` and, unchanged, the ``body`` of an
    OpenAI Batch API line, so a batched request is exactly the live one.
    ``settings``: the request settings to add (None: the table's,
    :func:`judge_request_settings`).  The default temperature is left out."""
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
    }
    t = request_temperature(model, temperature)
    if t is not None:
        body["temperature"] = t
    body.update(judge_request_settings(model) if settings is None else json.loads(json.dumps(settings)))
    if service_tier is not None:
        if service_tier not in SERVICE_TIERS:
            raise ValueError(f"service_tier {service_tier!r}: expected one of {SERVICE_TIERS}")
        body["service_tier"] = service_tier
    return body


def anthropic_request_params(prompt: str, model: str, max_tokens: int, *,
                             settings: Optional[Dict[str, Any]] = None,
                             temperature: Optional[float] = DEFAULT_TEMPERATURE) -> Dict[str, Any]:
    """The Messages API parameters: the keyword arguments of
    ``client.messages.create`` and, unchanged, the ``params`` of a Message
    Batches request."""
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    t = request_temperature(model, temperature)
    if t is not None:
        params["temperature"] = t
    params.update(judge_request_settings(model) if settings is None else json.loads(json.dumps(settings)))
    return params


def request_for(model: str, prompt: str, max_tokens: int, settings: Dict[str, Any],
                service_tier: Optional[str] = None) -> Dict[str, Any]:
    """The provider's request for ``model`` (a chat body or Messages params)."""
    if provider_for_model(model) == "anthropic":
        return anthropic_request_params(prompt, model, max_tokens, settings=settings)
    return chat_request_body(prompt, model, max_tokens, settings=settings, service_tier=service_tier)


def prompt_of_request(request: Dict[str, Any]) -> str:
    """The prompt of a request built by :func:`request_for`."""
    return request["messages"][0]["content"]


def _g(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


#: The finish reason of a reply cut at ``max_tokens``: OpenAI's
#: ``finish_reason``, and what :func:`anthropic_reply` makes of Anthropic's
#: ``stop_reason == "max_tokens"``.
CUT_AT_CAP = "length"


@dataclass(frozen=True)
class Truncated:
    """What the call functions return, instead of the text, for a reply cut at
    ``max_tokens``.  Such a reply usually lacks its ``SCORE:`` line, and
    ``parse_signed_score`` would then score it from the first small integer in
    its reasoning, so it is never parsed: it scores as missing (Roger,
    2026-11-01; the main checkout's step 3 does the same, ``reject_truncated``).
    ``text`` is kept for inspection only."""
    text: Optional[str] = None


def anthropic_reply(message: Any) -> Tuple[Optional[str], Optional[str], int, int, bool]:
    """``(text, finish_reason, input_tokens, output_tokens, refused)`` of an
    Anthropic message, an SDK object or a plain dict (a Message Batches
    result): the text of its text blocks (None when empty; thinking blocks are
    skipped by type), ``"length"`` for a reply cut at ``max_tokens``, and
    whether it stopped as a refusal."""
    stop = _g(message, "stop_reason")
    text = "".join(_g(b, "text", "") or "" for b in (_g(message, "content") or [])
                   if _g(b, "type", "text") == "text")
    usage = _g(message, "usage") or {}
    return (text or None, "length" if stop == "max_tokens" else stop,
            int(_g(usage, "input_tokens", 0) or 0), int(_g(usage, "output_tokens", 0) or 0),
            stop == "refusal")


def openai_reply(response: Any) -> Tuple[Optional[str], Optional[str], int, int]:
    """``(text, finish_reason, prompt_tokens, completion_tokens)`` of a chat
    completion, an SDK object or a plain dict (a Batch API result body)."""
    choices = _g(response, "choices") or []
    first = choices[0] if choices else None
    text = _g(_g(first, "message"), "content") if first is not None else None
    finish = _g(first, "finish_reason") if first is not None else None
    usage = _g(response, "usage") or {}
    return (text or None, finish, int(_g(usage, "prompt_tokens", 0) or 0),
            int(_g(usage, "completion_tokens", 0) or 0))


@dataclass
class CallStats:
    """What happened to one model's calls in a run (beyond scores)."""
    refused: int = 0          # Anthropic stop_reason "refusal": counted, unparseable
    content_flagged: int = 0  # HTTP 400 content flags: counted, unparseable
    errors: int = 0           # failed calls (network, 5xx...): counted, unparseable
    flex_fallbacks: int = 0   # Flex calls sent to the standard tier after FLEX_FALLBACK_AFTER_S
    skipped: int = 0          # not issued (--budget_usd reached, or a fatal error elsewhere)
    truncated: int = 0        # replies cut at max_tokens: counted, never parsed (score None)
    refusal_categories: Dict[str, int] = field(default_factory=dict)

    def note_refusal(self, model: str, category: Optional[str]) -> None:
        self.refused += 1
        key = str(category)
        self.refusal_categories[key] = self.refusal_categories.get(key, 0) + 1
        logger.warning(f"{model} refused a judge prompt (category {category}; {self.refused} refusals so far)")


class _Skipped:
    def __repr__(self) -> str:
        return "SKIPPED"


#: Returned by the call functions when ``gate()`` said not to issue the request.
SKIPPED = _Skipped()


async def call_openai(client: Any, prompt: str, model: str, max_tokens: int, rate_limiter: RateLimiter,
                      usage: Any, *, settings: Optional[Dict[str, Any]] = None,
                      service_tier: Optional[str] = None, fallback_client: Any = None,
                      stats: Optional[CallStats] = None,
                      gate: Optional[Callable[[], bool]] = None) -> Any:
    """One chat-completion judge call: the reply text, a :class:`Truncated` for
    a reply cut at ``max_tokens`` (``finish_reason == "length"``), None for a
    failed call, or :data:`SKIPPED` when ``gate()`` (checked once the rate
    limiter lets the request go, before it is first sent) is True.

    The Flex loop is the main checkout's ``call_judge_single``: a Flex call
    that fails with anything but an HTTP 400 is tried again every
    :data:`FLEX_RETRY_SLEEP_S` until :data:`FLEX_FALLBACK_AFTER_S` has passed
    since the first attempt, then sent once through ``fallback_client`` with
    no tier and charged under the plain model name (``stats.flex_fallbacks``).
    An HTTP 400 raises :class:`JudgeRequestError` (:class:`FlexRefusedError`
    when it refuses the service tier), except a content flag, which gives
    None.  ``usage.charge(key, prompt_tokens, completion_tokens)`` is called
    under :func:`usage_key`."""
    stats = stats if stats is not None else CallStats()
    if service_tier == "flex" and accepts_flex(model) is False:
        raise FlexRefusedError(f"OpenAI refuses Flex processing for {model} (MODEL_CAPABILITIES): "
                               f"judge it through the Batch API or in real time")
    body = chat_request_body(prompt, model, max_tokens, settings=settings, service_tier=service_tier)
    started = time.monotonic()
    first = True
    while True:
        await rate_limiter.acquire()
        if first:
            first = False
            if gate is not None and gate():
                stats.skipped += 1
                return SKIPPED
        try:
            response = await client.chat.completions.create(**body)
            break
        except Exception as e:  # noqa: BLE001 -- classified below
            if is_billing_error(e):
                raise _billing_error(model, e) from e
            if _is_bad_request(e):
                if service_tier == "flex" and is_flex_refusal(e):
                    raise FlexRefusedError(f"OpenAI refused Flex processing for {model}: {e}") from e
                flag = content_flag(e)
                if flag is None:
                    raise _bad_request_error(model, e) from e
                stats.content_flagged += 1
                logger.warning(f"HTTP 400 content flag ({flag}) from {model}: this item gives None ({e})")
                return None
            if service_tier == "flex" and fallback_client is not None:
                if gate is not None and gate():
                    # another request hit a fatal error (or the budget): stop retrying
                    stats.skipped += 1
                    return SKIPPED
                waited = time.monotonic() - started
                if waited < FLEX_FALLBACK_AFTER_S:
                    logger.warning(f"Flex call to {model} failed after {waited:.0f}s ({e}); trying Flex "
                                   f"again in {FLEX_RETRY_SLEEP_S:.0f}s")
                    await asyncio.sleep(FLEX_RETRY_SLEEP_S)
                    continue
                logger.warning(f"Flex call to {model} failed for {waited:.0f}s ({e}); sending it to the "
                               f"standard tier")
                stats.flex_fallbacks += 1
                client, service_tier = fallback_client, None
                body = chat_request_body(prompt, model, max_tokens, settings=settings)
                continue
            stats.errors += 1
            logger.error(f"Error calling judge model {model}: {e}")
            return None
    text, finish, pt, ct = openai_reply(response)
    if usage is not None:
        usage.charge(usage_key(model, service_tier), pt, ct)
    if finish == CUT_AT_CAP:
        return Truncated(text)
    return text


async def call_anthropic(client: Any, prompt: str, model: str, max_tokens: int, rate_limiter: RateLimiter,
                         usage: Any, *, settings: Optional[Dict[str, Any]] = None,
                         stats: Optional[CallStats] = None,
                         gate: Optional[Callable[[], bool]] = None) -> Any:
    """One Messages API judge call: the reply text, a :class:`Truncated` for a
    reply cut at ``max_tokens`` (``stop_reason == "max_tokens"``), None for a
    failed call (a refusal included, counted in ``stats.refused`` and charged),
    or :data:`SKIPPED`.  An HTTP 400 raises :class:`JudgeRequestError` unless it
    flags the prompt's content."""
    stats = stats if stats is not None else CallStats()
    params = anthropic_request_params(prompt, model, max_tokens, settings=settings)
    await rate_limiter.acquire()
    if gate is not None and gate():
        stats.skipped += 1
        return SKIPPED
    try:
        response = await client.messages.create(**params)
    except Exception as e:  # noqa: BLE001 -- classified below
        if is_billing_error(e):
            raise _billing_error(model, e) from e
        if _is_bad_request(e):
            flag = content_flag(e)
            if flag is None:
                raise _bad_request_error(model, e) from e
            stats.content_flagged += 1
            logger.warning(f"HTTP 400 content flag ({flag}) from {model}: this item gives None ({e})")
            return None
        stats.errors += 1
        logger.error(f"Error calling Anthropic judge model {model}: {e}")
        return None
    text, finish, n_in, n_out, refused = anthropic_reply(response)
    if usage is not None:
        usage.charge(model, n_in, n_out)
    if refused:
        stats.note_refusal(model, _g(_g(response, "stop_details"), "category"))
        return None
    if finish == CUT_AT_CAP:
        return Truncated(text)
    return text


async def call_model(model: str, prompt: str, max_tokens: int, rate_limiter: RateLimiter, usage: Any, *,
                     client: Any, settings: Dict[str, Any], service_tier: Optional[str] = None,
                     fallback_client: Any = None, stats: Optional[CallStats] = None,
                     gate: Optional[Callable[[], bool]] = None) -> Any:
    """:func:`call_openai` or :func:`call_anthropic` by the model's provider."""
    if provider_for_model(model) == "anthropic":
        return await call_anthropic(client, prompt, model, max_tokens, rate_limiter, usage,
                                    settings=settings, stats=stats, gate=gate)
    return await call_openai(client, prompt, model, max_tokens, rate_limiter, usage, settings=settings,
                             service_tier=service_tier, fallback_client=fallback_client, stats=stats,
                             gate=gate)


# ---------------------------------------------------------------------------
# Transport choice
# ---------------------------------------------------------------------------

TRANSPORT_CHOICES = ("auto", "flex", "batch", "realtime")


@dataclass(frozen=True)
class TransportChoice:
    transport: str      # "flex", "batch" or "realtime"
    probe: bool         # Flex untested: the first request probes it, batch if refused
    reason: str

    def describe(self) -> str:
        return f"{self.transport}{' (probe)' if self.probe else ''}: {self.reason}"


def choose_transport(model: str, requested: str = "auto") -> TransportChoice:
    """The transport for ``model`` under ``--transport requested``.

    ``auto``: an Anthropic model goes by Message Batches; an OpenAI model by
    Flex when :data:`MODEL_CAPABILITIES` says OpenAI takes it, by the Batch API
    when it says OpenAI refuses it, and, untested, by Flex with a probe (the
    first request alone; a service-tier HTTP 400 switches the model to batch
    for the run).  ``flex`` on a model known to refuse it, or on an Anthropic
    model, raises :class:`JudgeRequestError`."""
    if requested not in TRANSPORT_CHOICES:
        raise ValueError(f"--transport {requested!r}: expected one of {TRANSPORT_CHOICES}")
    provider = provider_for_model(model)
    if requested == "realtime":
        return TransportChoice("realtime", False, "--transport realtime")
    if requested == "batch":
        return TransportChoice("batch", False, "--transport batch")
    flex = accepts_flex(model) if provider == "openai" else False
    if requested == "flex":
        if provider != "openai":
            raise JudgeRequestError(f"--transport flex: Flex processing is OpenAI's, not available for {model}")
        if flex is False:
            raise JudgeRequestError(f"--transport flex: OpenAI refuses Flex processing for {model} "
                                    f"(MODEL_CAPABILITIES); use --transport batch")
        return TransportChoice("flex", False, "--transport flex" + (
            "" if flex else " (Flex untested for this model: a refusal stops the model)"))
    if provider == "anthropic":
        return TransportChoice("batch", False, "Anthropic model: Message Batches")
    if flex is True:
        return TransportChoice("flex", False, "MODEL_CAPABILITIES: OpenAI takes Flex for this model")
    if flex is False:
        return TransportChoice("batch", False, "MODEL_CAPABILITIES: OpenAI refuses Flex for this model")
    return TransportChoice("flex", True, "Flex untested for this model: the first request probes it, "
                                         "batch if OpenAI refuses the tier")


def price_key(model: str, transport: str) -> str:
    """The usage key a transport's calls are charged under (and priced by)."""
    if transport == "batch":
        return f"{model}{BATCH_SUFFIX}"
    if transport == "flex":
        return f"{model}{FLEX_SUFFIX}"
    return model


__all__: List[str] = [
    "DEFAULT_TEMPERATURE", "MODEL_CAPABILITIES", "JUDGE_REQUEST_SETTINGS", "parse_request_setting",
    "judge_request_settings", "JudgeRequestError", "FlexRefusedError", "model_capabilities",
    "accepts_temperature", "accepts_flex", "request_temperature", "content_flag", "is_flex_refusal",
    "SERVICE_TIERS", "FLEX_FALLBACK_AFTER_S", "FLEX_RETRY_SLEEP_S", "FLEX_TIMEOUT_S", "FLEX_MAX_RETRIES",
    "usage_key", "chat_request_body", "anthropic_request_params", "request_for", "prompt_of_request",
    "anthropic_reply", "openai_reply", "CUT_AT_CAP", "Truncated", "CallStats", "SKIPPED", "call_openai",
    "call_anthropic",
    "call_model", "TRANSPORT_CHOICES", "TransportChoice", "choose_transport", "price_key",
]
