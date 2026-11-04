"""
Judge LLM utilities for scoring role responses.

This module provides functions for scoring how well model responses
adhere to their assigned roles using an LLM judge (e.g., GPT-4).

Score Scale:
    0: Model refused to answer
    1: Model says it can't be the role, but can help with related tasks
    2: Model identifies as AI/LLM but has some role attributes
    3: Model is fully playing the role

Example:
    from assistant_axis.judge import score_responses

    scores = await score_responses(
        responses=[{"question": "...", "response": "..."}],
        eval_prompt_template="Rate how well...",
        judge_model="gpt-4.1-mini"
    )
"""

import asyncio
import json
import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Iterable, Tuple, Sequence, TYPE_CHECKING

import openai
from dotenv import load_dotenv

if TYPE_CHECKING:
    # Forward reference only -- the runtime import lives inside the
    # call sites that actually need to extract usage, to keep the
    # judge.py import surface minimal.
    from .judge_pricing import MultiModelUsage  # noqa: F401

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------
#
# Model names route to providers by prefix.  Steering judges (Phase 2) call
# both OpenAI and Anthropic models, so judge.py exposes a unified
# call_judge_single_unified() that hides the SDK split.

_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")
_ANTHROPIC_PREFIXES = ("claude-",)
_GOOGLE_PREFIXES = ("gemini-",)


def provider_for_model(model: str) -> str:
    """Return 'openai', 'anthropic' or 'google' based on the model name prefix.

    Raises ValueError for unrecognised prefixes so we fail loudly rather
    than silently routing a typo to the wrong SDK.
    """
    if any(model.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    if any(model.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic"
    if any(model.startswith(p) for p in _GOOGLE_PREFIXES):
        return "google"
    raise ValueError(
        f"unknown judge model {model!r}; expected a prefix in "
        f"{_OPENAI_PREFIXES + _ANTHROPIC_PREFIXES + _GOOGLE_PREFIXES}"
    )


# ---------------------------------------------------------------------------
# Request parameters by model (code review 2026-10-12, B1)
# ---------------------------------------------------------------------------
#
# A temperature equal to the provider's default is never sent, so a call at
# the default works on every model, those that refuse the field included.
# Only a non-default temperature consults MODEL_CAPABILITIES, which lists the
# non-default cases: models that refuse a temperature.  A model in no entry
# takes one.  Check the table, and add the model, before switching a judge to
# a model this project has not used.  An HTTP 400 from either provider is
# raised as JudgeRequestError rather than logged and turned into None: a
# request the API rejects as malformed fails the same way for every item, and
# swallowing it made every item of a run unparseable (review 7 D1).  The one
# exception is a 400 that flags a single prompt's content
# (CONTENT_FLAG_CODES): that item gives None with a WARNING, as before.

#: Both providers' default sampling temperature (Chat Completions and Messages).
DEFAULT_TEMPERATURE = 1.0

#: Model-id fragments (lower case) of the models that refuse a ``temperature``
#: with an HTTP 400.  A fragment matches at the start of the model id or after
#: a hyphen.  The Anthropic rows are those of the trait-gap platform's
#: ``assistant_axis.gapgen.llm.NO_TEMPERATURE_FRAGMENTS`` (Sonnet 5.5 checked
#: 2026-09-29; every Opus from 4.7 on returned 400 "`temperature` is deprecated
#: for this model" on 2026-10-10; Haiku 5.5 from its model page of 2026-10-07);
#: ``test_judge_request_capabilities.py`` keeps the two lists in step.  The
#: OpenAI rows are its reasoning models, which take only the default
#: temperature (GPT-6 Luna's support is unstated in OpenAI's docs as of
#: 2026-10-07: test before use).  judge.py sends no reasoning-effort setting,
#: so a reasoning model would bill its reasoning at the default effort; add
#: that capability here with the first switch to one.
#: ``"flex"`` (since 2026-10-30, with run_pipeline.sh judging by Flex by
#: default): whether OpenAI takes ``service_tier="flex"`` for the model.
#: False for gpt-4.1-mini (HTTP 400 "Invalid service_tier argument",
#: 2026-10-28), True for gpt-5.6-luna (the cap pilot's judging, 2026-10-28
#: on); a model in no ``"flex"`` entry is untested (``accepts_flex`` gives
#: None), and its first Flex request tells: a 400 raises JudgeRequestError.
#: The pricing table cannot tell (it prices every OpenAI model's ``:flex``).
#: ``"cache_mode"`` (since 2026-11-01): whether OpenAI takes
#: ``prompt_cache_options`` for the model, the field that turns its default
#: prompt caching off (:data:`DEFAULT_PROMPT_CACHE`).  True for GPT-5.6 and
#: later (the prompt-caching guide; gpt-5.6-luna checked on Flex,
#: 2026-11-01), False for gpt-4.1-mini (HTTP 400 "prompt_cache_options is
#: not supported on this model", the same day); a model in no entry is
#: untested and is sent nothing, and its ``usage.json`` shows what it wrote.
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
    "gpt-6": {"temperature": False, "cache_mode": True},
    # OpenAI Flex processing, and prompt_cache_options (GPT-5.6 and later; a
    # fragment appears once in this table, a later duplicate would replace
    # the earlier entry)
    "gpt-4.1-mini": {"flex": False, "cache_mode": False},
    "gpt-5.6": {"cache_mode": True},
    "gpt-5.6-luna": {"flex": True},
    # Each answered its first Flex request (judge tier evaluation, 2026-11-01)
    "gpt-5.6-terra": {"flex": True},
    "gpt-5.6-sol": {"flex": True},
    "gpt-6-luna": {"flex": True},
    "gpt-5.4-mini": {"flex": True},
    # Google: temperature accepted (the judges send 0), and no Flex tier
    "gemini": {"flex": False},
}

#: How a judge request treats OpenAI's prompt caching (2026-11-01).  GPT-5.6
#: and later cache by default (implicit mode: every prompt of 1,024 tokens or
#: more is written, at 1.25x the input rate, and a repeat within 30 minutes
#: reads it at 0.1x).  A judge prompt ends in a unique response and is never
#: repeated, so the writes are a premium for nothing: on luna Flex from 29 Oct
#: to 1 Nov 2026, $198 of writes against $0.05 of reads, about 10% of the
#: bill (Roger, through the Judging Pareto Handoff session); and the shared
#: part of a judge prompt (a rubric of 500 to 600 tokens) is under the 1,024
#: minimum, so no breakpoint could make a cache pay.  ``"off"``: a model that
#: takes the field (``MODEL_CAPABILITIES`` ``"cache_mode"``) is sent explicit
#: mode with no breakpoints, which writes and reads nothing (checked on Flex,
#: 2026-11-01).  ``"implicit"``: the field is left out and the provider's
#: default applies.  A model listed as refusing the field, or untested, is
#: sent nothing in either mode (gpt-4.1-mini writes free and discounts reads,
#: so nothing is lost there).
PROMPT_CACHE_MODES = ("off", "implicit")
DEFAULT_PROMPT_CACHE = "off"
PROMPT_CACHE_OFF_OPTIONS: Dict[str, str] = {"mode": "explicit"}

#: Fields a judge request adds for a model, by model-id fragment (matched as in
#: :data:`MODEL_CAPABILITIES`).  Judges run without thinking or reasoning, the
#: mode gpt-4.1-mini works in, so that the reasoning the rubric asks for is in
#: the reply and a short ``max_tokens`` holds the answer (Roger, 2026-10-26: the
#: step-3 judge comparison).  GPT-6 Luna takes ``reasoning_effort`` "none" on
#: Chat Completions and in the Batch API (its model page, read 2026-10-26).
#: GPT-5.4 mini (effort "none" its default) and GPT-5.6 Luna (default "medium")
#: take "none" too (their model pages, read 2026-10-27); GPT-5 and GPT-5 mini do
#: not, so the fragments name the versions, not "gpt-5".
#: Haiku 5.5 thinks adaptively unless the request says otherwise, its thinking
#: counts toward ``max_tokens``, and it accepts thinking disabled at its default
#: effort (medium).  Gemini takes a temperature, so T=0 (Roger's rule: T=0 where
#: a model takes it), and its lowest thinking level, ``minimal``, which spent no
#: thought tokens on 3.1 Flash-Lite in a probe of 2026-11-02 (``low`` about 120 a
#: call); the fields go in ``generationConfig`` (:func:`gemini_request_body`).
#: A model in no entry adds nothing.
JUDGE_REQUEST_SETTINGS: Dict[str, Dict[str, Any]] = {
    "gpt-5.4": {"reasoning_effort": "none"},
    "gpt-5.6": {"reasoning_effort": "none"},
    "gpt-6": {"reasoning_effort": "none"},
    "haiku-5": {"thinking": {"type": "disabled"}},
    "gemini": {"temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}},
}

#: One override of a request's settings: (model fragment or None for every model,
#: key, value); a value of None removes the key.  :func:`parse_request_setting`.
RequestOverride = Tuple[Optional[str], str, Any]


def _fragment_matches(model: str, fragment: str) -> bool:
    m = str(model).lower()
    return m.startswith(fragment) or f"-{fragment}" in m


def judge_request_settings(model: str, overrides: Iterable[RequestOverride] = ()) -> Dict[str, Any]:
    """The fields a request to ``model`` adds: the :data:`JUDGE_REQUEST_SETTINGS`
    rows that match it, then the ``overrides`` that apply to it.  A fresh dict."""
    out: Dict[str, Any] = {}
    for fragment, entry in JUDGE_REQUEST_SETTINGS.items():
        if _fragment_matches(model, fragment):
            out.update(json.loads(json.dumps(entry)))
    for fragment, key, value in overrides:
        if fragment is not None and not _fragment_matches(model, fragment):
            continue
        if value is None:
            out.pop(key, None)
        else:
            out[key] = json.loads(json.dumps(value))
    return out


def parse_request_setting(text: str) -> RequestOverride:
    """``[MODEL_FRAGMENT:]key=<json>`` -> ``(fragment, key, value)``.

    ``reasoning_effort="low"`` applies to every model of a run;
    ``gpt-6:reasoning_effort="low"`` only to models matching ``gpt-6`` (as
    :data:`JUDGE_REQUEST_SETTINGS` matches).  The value is JSON
    (``'{"type": "disabled"}'``, ``'"none"'``, ``200``); ``null`` removes a
    field the table would add."""
    if "=" not in text:
        raise ValueError(f"request setting {text!r}: expected [MODEL_FRAGMENT:]key=<json>")
    left, raw = text.split("=", 1)
    fragment: Optional[str] = None
    if ":" in left:
        fragment, left = left.split(":", 1)
        fragment = fragment.strip().lower() or None
    key = left.strip()
    if not key:
        raise ValueError(f"request setting {text!r}: empty key")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"request setting {text!r}: the value is not JSON ({e}); "
                         f"quote a string, e.g. {key}='\"none\"'") from e
    return fragment, key, value


class JudgeRequestError(ValueError):
    """A judge request that is wrong for its model: an HTTP 400 from the API,
    or a non-default temperature for a model that refuses one.  Raised, never
    swallowed, because every further call would fail the same way."""


class FlexRefusedError(JudgeRequestError):
    """OpenAI refused ``service_tier="flex"`` for the model (HTTP 400 naming the
    service tier), or :data:`MODEL_CAPABILITIES` says it would."""


class BillingError(JudgeRequestError):
    """The provider refuses the account, not the request: out of credits, over
    quota, or a daily quota used up.  Every further call fails the same way, so
    it stops a run at once instead of being retried (OpenAI, 2026-11-01: HTTP
    429 with code ``credit_balance_exhausted``, type ``insufficient_quota``,
    which a Flex loop used to retry for ten minutes per request and then leave
    unscored)."""


def model_capabilities(model: str) -> Dict[str, Optional[bool]]:
    """What ``model`` takes, by :data:`MODEL_CAPABILITIES`: ``{"temperature":
    bool, "flex": bool or None}``.  A model no entry names takes a temperature,
    and its Flex support is untested (None)."""
    caps: Dict[str, Optional[bool]] = {"temperature": True, "flex": None, "cache_mode": None}
    for fragment, entry in MODEL_CAPABILITIES.items():
        if _fragment_matches(model, fragment):
            caps.update(entry)
    return caps


def accepts_cache_options(model: str) -> Optional[bool]:
    """Whether OpenAI takes ``prompt_cache_options`` for ``model``
    (:data:`MODEL_CAPABILITIES` ``"cache_mode"``): True for GPT-5.6 and
    later, False for gpt-4.1-mini (HTTP 400), None when untested."""
    return model_capabilities(model)["cache_mode"]


def prompt_cache_fields(model: str, prompt_cache: str = DEFAULT_PROMPT_CACHE) -> Dict[str, Any]:
    """The request fields that set prompt caching for ``model`` under
    ``prompt_cache`` (one of :data:`PROMPT_CACHE_MODES`): ``"off"`` gives
    ``prompt_cache_options`` explicit mode with no breakpoints for a model that
    takes the field, nothing otherwise; ``"implicit"`` gives nothing."""
    if prompt_cache not in PROMPT_CACHE_MODES:
        raise ValueError(f"prompt_cache {prompt_cache!r}: expected one of {PROMPT_CACHE_MODES}")
    if prompt_cache == "off" and accepts_cache_options(model):
        return {"prompt_cache_options": dict(PROMPT_CACHE_OFF_OPTIONS)}
    return {}


def chat_create_kwargs(body: Dict[str, Any]) -> Dict[str, Any]:
    """The keyword arguments of ``client.chat.completions.create`` for a
    request body from :func:`chat_request_body`.  The OpenAI SDK (2.17) does
    not know ``prompt_cache_options``, so that field travels in
    ``extra_body``; a Batch API line carries the body as it is."""
    kwargs = dict(body)
    extra = {k: kwargs.pop(k) for k in ("prompt_cache_options",) if k in kwargs}
    if extra:
        kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), **extra}
    return kwargs


def accepts_temperature(model: str) -> bool:
    """False for a model that refuses a ``temperature`` (:data:`MODEL_CAPABILITIES`)."""
    return bool(model_capabilities(model)["temperature"])


def accepts_flex(model: str) -> Optional[bool]:
    """Whether OpenAI takes ``service_tier="flex"`` for ``model``
    (:data:`MODEL_CAPABILITIES`): False for one it refuses with HTTP 400
    (gpt-4.1-mini), True for one it has taken (gpt-5.6-luna), None when
    untested.  A Flex call to a model listed False raises
    :class:`JudgeRequestError` before anything is sent."""
    return model_capabilities(model)["flex"]


def request_temperature(model: str, temperature: Optional[float]) -> Optional[float]:
    """The ``temperature`` a request to ``model`` carries: ``None`` (leave the
    field out) for ``None`` or the provider default, else ``temperature``.
    Raises :class:`JudgeRequestError` for a non-default temperature on a model
    that refuses one, before anything is sent."""
    if temperature is None or float(temperature) == DEFAULT_TEMPERATURE:
        return None
    if not accepts_temperature(model):
        raise JudgeRequestError(
            f"{model} refuses a temperature (judge.MODEL_CAPABILITIES), so "
            f"temperature={temperature} cannot be honoured: call it at the "
            f"default ({DEFAULT_TEMPERATURE:g}), or choose a model that takes one"
        )
    return float(temperature)


def _is_bad_request(exc: BaseException) -> bool:
    """An HTTP 400 from either SDK (both set ``status_code`` on the error)."""
    return getattr(exc, "status_code", None) == 400


#: An HTTP 400 that flags one prompt's *content*, not the request's shape (B1b,
#: 2026-10-12).  This project judges persona responses that can be deliberately
#: harmful, so such a 400 keeps the per-item behaviour: that item gives None, a
#: WARNING names the model and the code, :data:`CONTENT_FLAGGED` counts it, and
#: the run goes on; every other 400 raises :class:`JudgeRequestError`.
#:
#: * OpenAI sets the error's ``code`` (the SDK copies the error object's
#:   ``code`` onto the exception): ``invalid_prompt`` ("Invalid prompt: your
#:   prompt was flagged as potentially violating our usage policy", reasoning
#:   models) and ``content_policy_violation``; Azure OpenAI uses
#:   ``content_filter`` ("... triggering Azure OpenAI's content management
#:   policy").  A refused temperature is ``unsupported_value`` /
#:   ``unsupported_parameter`` with ``param: "temperature"``, so it still raises.
#: * Anthropic answers a prompt it will not judge with a refusal (HTTP 200,
#:   ``stop_reason: "refusal"``), not a 400, and has no code here; the phrases
#:   would still catch such a message if one ever came as a 400.
#: ``bio_policy`` (2026-11-04): GPT-5.6 Luna refused one batch of response-judging prompts as "flagged for
#: possible biological risk"; until then any 400 stopped the run, so a 220,000-response arm stopped at one prompt.
CONTENT_FLAG_CODES: Tuple[str, ...] = ("invalid_prompt", "content_policy_violation", "content_filter", "bio_policy")
CONTENT_FLAG_PHRASES: Tuple[str, ...] = ("flagged as potentially violating", "violating our usage policy",
                                         "content management policy")

#: Content-flagged 400s in this process, by model (a WARNING carries the running count).
CONTENT_FLAGGED: Dict[str, int] = {}

#: Replies cut at ``max_tokens`` (``finish_reason == "length"``) that a call
#: with ``reject_truncated=True`` returned as None, per model (pipeline step
#: 3 since 2026-10-12, walk-through P4: a truncated reply is unparseable even
#: when its text holds a ``SCORE:`` line, since the reasoning was cut).
TRUNCATED_REPLIES: Dict[str, int] = {}

#: Anthropic replies that stopped with ``stop_reason == "refusal"`` (a safety
#: classifier declined the prompt; HTTP 200 and no answer), per model.  Such a
#: call gives None, an unparseable reply, and is charged all the same.
REFUSED_REPLIES: Dict[str, int] = {}


def _error_field(exc: BaseException, name: str) -> Optional[str]:
    """A field of a provider's error: OpenAI's SDK copies ``code``, ``param`` and
    ``type`` onto the exception; otherwise from the error body (``{name: ...}`` or
    ``{"error": {name: ...}}``)."""
    value = getattr(exc, name, None)
    if value:
        return str(value)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        if inner.get(name):
            return str(inner[name])
    return None


def _error_code(exc: BaseException) -> Optional[str]:
    """The provider's error code."""
    return _error_field(exc, "code")


def content_flag(exc: BaseException) -> Optional[str]:
    """For an HTTP 400 that flags the prompt's content, the code or phrase that
    says so (:data:`CONTENT_FLAG_CODES`, :data:`CONTENT_FLAG_PHRASES`); else None."""
    if not _is_bad_request(exc):
        return None
    code = _error_code(exc)
    if code in CONTENT_FLAG_CODES:
        return code
    text = f"{getattr(exc, 'message', '')} {exc}".lower()
    return next((p for p in CONTENT_FLAG_PHRASES if p in text), None)


def _note_content_flag(model: str, flag: str, exc: BaseException) -> None:
    CONTENT_FLAGGED[model] = CONTENT_FLAGGED.get(model, 0) + 1
    logger.warning(f"HTTP 400 content flag ({flag}) from judge model {model}: this item gives None and the run "
                   f"goes on ({CONTENT_FLAGGED[model]} flagged so far for {model}): {exc}")


def _bad_request_error(model: str, exc: BaseException) -> JudgeRequestError:
    logger.error(f"HTTP 400 from judge model {model}: {exc}")
    return JudgeRequestError(
        f"HTTP 400 from judge model {model}: {exc}.  The API rejected the "
        f"request itself, so every call would fail the same way (check "
        f"judge.MODEL_CAPABILITIES for this model)")


def is_flex_refusal(exc: BaseException) -> bool:
    """An HTTP 400 refusing the service tier (gpt-4.1-mini, 2026-10-28: "Invalid
    service_tier argument"): its ``param`` is ``service_tier``, or its message
    names the tier."""
    if not _is_bad_request(exc):
        return False
    if _error_field(exc, "param") == "service_tier":
        return True
    text = f"{getattr(exc, 'message', '')} {exc}".lower()
    return "service_tier" in text or "service tier" in text


BILLING_ERROR_CODES: Tuple[str, ...] = ("insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached")
BILLING_ERROR_PHRASES: Tuple[str, ...] = ("no credits remaining", "exceeded your current quota",
                                          "credit balance is too low")


def is_billing_error(exc: BaseException) -> bool:
    """An error saying the account cannot pay: a 4xx whose code or type is a
    quota code (OpenAI answers HTTP 429), or whose message says so (Anthropic
    answers a low credit balance with HTTP 400 "Your credit balance is too low")."""
    status = getattr(exc, "status_code", None)
    if status is None or not 400 <= int(status) < 500:
        return False
    if _error_field(exc, "code") in BILLING_ERROR_CODES or _error_field(exc, "type") in BILLING_ERROR_CODES:
        return True
    text = f"{getattr(exc, 'message', '')} {exc}".lower()
    return any(p in text for p in BILLING_ERROR_PHRASES)


def _billing_error(model: str, exc: BaseException) -> BillingError:
    logger.error(f"Billing refusal from judge model {model}: {exc}")
    return BillingError(f"{model}: the provider refuses the account ({exc}).  Every call would fail the same "
                        f"way: add credits or raise the quota, then rerun")


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, rate: float):
        """
        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.time()
            self.tokens = min(self.rate, self.tokens + (now - self.last_update) * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0


# Project-standard threshold for "judge parse failures are bad enough
# to warrant a loud warning".  Set to 1% (i.e. <99% OK rate) per the
# May 2026 audit which found that even ~1% UNPARSEABLE on batched
# effect judging silently biases ensemble means -- the surviving
# judge dominates whenever its partner fails, which doesn't show up
# in mean-of-means but does show up in per-(kind, model) scatter.
# See AGENT_NOTES.md "Judge prompts: reason BEFORE score" for the
# full rule context, and ``warn_if_low_parse_rate`` below for the
# central emit point that all judge call sites should funnel through.
PARSE_RATE_LOUD_THRESHOLD = 0.99  # OK >= 99% = quiet; OK < 99% = loud


def warn_if_low_parse_rate(
    *,
    label: str,
    n_ok: int,
    n_total: int,
    threshold: float = PARSE_RATE_LOUD_THRESHOLD,
    logger_obj: Optional[logging.Logger] = None,
    extra: Optional[str] = None,
) -> None:
    """Log judge-call parse-rate, loudly if OK rate falls below
    ``threshold`` (default :data:`PARSE_RATE_LOUD_THRESHOLD` = 0.99).

    Use this from every judge driver (axis_judge_correlation,
    pipeline/3_judge.py, steering_judges.py, ...) at end-of-run so a
    single grep for ``HIGH FAIL RATE`` finds every degraded run.

    Examples::

        warn_if_low_parse_rate(
            label="axis_judge_correlation:descriptions",
            n_ok=98, n_total=100, logger_obj=logger,
        )
        # -> WARNING: *** HIGH FAIL RATE *** [axis_judge_correlation:descriptions]
        #             parse rate: 98/100 OK (2/100 = 2% UNPARSEABLE)

        warn_if_low_parse_rate(
            label="pipeline/3_judge", n_ok=5000, n_total=5000,
            logger_obj=logger,
        )
        # -> INFO: [pipeline/3_judge] parse rate: 5000/5000 OK

    Parameters
    ----------
    label : short identifier for the judge call site (no spaces preferred)
    n_ok : count of calls that produced a valid parsed score
    n_total : total calls made (n_ok + n_unparseable [+ n_missing])
    threshold : minimum OK rate to stay quiet (default 0.99)
    logger_obj : logger to emit through (defaults to this module's logger)
    extra : optional one-line continuation, e.g. "see gaps.json for the list"
    """
    log = logger_obj or logger
    if n_total == 0:
        return
    fail = n_total - n_ok
    if fail == 0:
        msg = f"[{label}] parse rate: {n_total}/{n_total} OK"
        if extra:
            msg += f"  ({extra})"
        log.info(msg)
        return
    pct = fail / n_total
    body = (
        f"[{label}] parse rate: {n_ok}/{n_total} OK "
        f"({fail}/{n_total} = {pct:.1%} UNPARSEABLE)"
    )
    if extra:
        body += f"  ({extra})"
    if (n_ok / n_total) < threshold:
        log.warning(f"*** HIGH FAIL RATE *** {body}")
    else:
        log.info(body)


def parse_judge_score(response_text: str, *, require_marker: bool = False) -> Optional[int]:
    """
    Parse the judge's response to extract the 0-3 score.

    See AGENT_NOTES.md "Judge prompts: reason BEFORE score" for the
    project rule any new judge prompts must follow.  This parser
    supports the reasoning-first format introduced under that rule
    AND the older "just emit a number" format (for backward compat
    with legacy ``eval_prompt`` fields).

    Two-stage parser:

    1. Look for an explicit ``SCORE: <int>`` marker (case-insensitive).
       This is the format used by the reasoning-first templates rolled
       out after the original "just emit a number" prompts.
    2. Fall back to the LAST integer in [0, 3] anywhere in the response.
       This handles both:
         - the original "just emit a number" templates (response is a
           single digit, last == first), and
         - templates that elicit reasoning without using the SCORE:
           marker (the conclusion-bearing integer typically lands at
           the end of the response, not in the middle of the reasoning
           where mention of e.g. "0 if no traits, 3 if fully" would
           otherwise mislead a first-integer parser).

    A marker whose value is out of range (``SCORE: 4``) gives None: the
    judge's answer was invalid, and an integer from its reasoning is not
    its answer.  (Before 2026-10-12 such a reply fell through to stage 2,
    so "shows 2 of the traits ... SCORE: 4" was scored 2.)

    ``require_marker=True`` (pipeline step 3 since 2026-10-12, its
    ``JUDGE_RUBRIC_VERSION`` 3; walk-through P4 / D5, review P3 D1): exactly
    one ``SCORE: <int>`` marker is the answer, and a reply with none or
    with several is None (unparseable), never a digit of its reasoning
    ("... lists 3 examples" read as 3, a truncated reply most often).  The
    default keeps the two-stage parse for the other callers.

    Returns the integer in [0, 3], or None if nothing parseable.
    """
    if not response_text:
        return None

    # Stage 1: explicit SCORE: marker (preferred for new prompts).
    markers = re.findall(r"SCORE\s*:\s*(\d+)", response_text, flags=re.IGNORECASE)
    if require_marker and len(markers) != 1:
        return None
    if markers:
        score = int(markers[0])
        return score if 0 <= score <= 3 else None

    # Stage 2: walk integers from the end backwards, return the first
    # one in [0, 3].  This is forgiving for old "just the number"
    # responses (the only integer is the score) and reasoning-included
    # responses without the SCORE: marker (the score lands at the end).
    for tok in reversed(re.findall(r'\d+', response_text)):
        try:
            v = int(tok)
        except ValueError:
            continue
        if 0 <= v <= 3:
            return v
    return None


# ---------------------------------------------------------------------------
# Robust JSON extraction for structured judge output
# ---------------------------------------------------------------------------
#
# Used by steering judges, which ask for {"score": ..., "reason": ...}.
# Models occasionally wrap JSON in ```json ... ``` fences or add prose; we
# strip both before parsing.

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

# LLMs often write "score": +1 (with literal "+" prefix) when the rubric
# describes the scale as -3..+3.  RFC 8259 JSON does NOT permit "+" as a
# numeric prefix; this trips json.loads -> the whole batch is dropped as
# UNPARSEABLE.  We strip "+" ONLY when it appears in JSON number-after-
# colon context (i.e. `: +N`), preserving "+" inside string bodies like
# `"talks about +5 reward"`.  Diagnosed 2026-05-15 on
# architect_ecocentric_v2 effect-judge UNPARSEABLE batches; rubric also
# bumped to v6 to explicitly request plain-integer scores.
_JSON_PLUS_NUM_AFTER_COLON_RE = re.compile(r":(\s*)\+(\d)")


def _repair_json_blob(blob: str) -> str:
    """Best-effort repair of common LLM-JSON deviations before parsing.

    Currently fixes:
      - ``"score": +1`` numeric-after-colon prefixes (drops the ``+``).

    Future deviations (trailing commas, unquoted keys, etc.) can be
    added here.  Keep the repair list minimal and well-targeted; over-
    aggressive rewriting risks corrupting valid payloads -- in
    particular avoid touching characters that appear inside string
    bodies, since those can be legitimate prose content.
    """
    return _JSON_PLUS_NUM_AFTER_COLON_RE.sub(r":\1\2", blob)


def extract_json_blob(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON object or array from a model response.

    Handles three common shapes:
      - markdown-fenced ``` ```json ... ``` ```
      - prose-wrapped object/array
      - bare JSON

    For raw text, prefers an object match if both an object and an array
    look plausible -- objects are the more common judge shape -- but
    falls back to array when no object is present.
    """
    if not text:
        return None
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    obj = _JSON_OBJECT_RE.search(text)
    arr = _JSON_ARRAY_RE.search(text)
    # Prefer whichever appears earlier in the text; avoids picking the
    # inner object out of a top-level array.
    if obj and arr:
        return (obj if obj.start() < arr.start() else arr).group(0).strip()
    if obj:
        return obj.group(0).strip()
    if arr:
        return arr.group(0).strip()
    return text.strip()


def parse_score_reason_json(
    text: str,
    score_range: Tuple[int, int] = (0, 3),
) -> Optional[Dict[str, Any]]:
    """Parse ``{"score": int, "reason": str}`` out of a model response.

    Returns ``{"score": int, "reason": str}`` on success, ``None`` if the
    response can't be parsed or the score is out of range.  ``score_range``
    is inclusive on both ends; for the bidirectional effect judge use
    ``(-3, 3)`` instead of the default ``(0, 3)``.
    """
    blob = extract_json_blob(text)
    if blob is None:
        return None
    try:
        parsed = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        # Retry with common-LLM-deviation repair (e.g. "+1" -> "1").
        try:
            parsed = json.loads(_repair_json_blob(blob))
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(parsed, dict) or "score" not in parsed:
        return None
    raw = parsed["score"]
    try:
        score = int(raw) if isinstance(raw, (int, float, str)) else None
    except (ValueError, TypeError):
        return None
    if score is None:
        return None
    lo, hi = score_range
    if not (lo <= score <= hi):
        return None
    reason = parsed.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return {"score": score, "reason": reason}


def parse_batch_scores_json(
    text: str,
    expected_ids: Sequence[Any],
    score_range: Tuple[int, int] = (-3, 3),
) -> Optional[Dict[Any, Dict[str, Any]]]:
    """Parse a batched-effect judge response.

    Expected payload shape (top-level field name flexible):

        {"items": [{"id": <id>, "score": int, "reason": str}, ...]}

    Returns a dict keyed by id with ``{"score": int, "reason": str}``
    values.  Returns ``None`` if the payload is unparseable or doesn't
    cover ``expected_ids``.  Order does not matter in the output; ids that
    come back outside ``score_range`` are dropped (logged).
    """
    blob = extract_json_blob(text)
    if blob is None:
        return None
    try:
        parsed = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        # Retry with common-LLM-deviation repair (e.g. "+1" -> "1" inside
        # numeric score fields; see _repair_json_blob).  Diagnosed
        # 2026-05-15 as the dominant UNPARSEABLE root cause on the
        # bidirectional effect rubric where the scale is described as
        # -3..+3 with explicit "+" signs that the LLM faithfully echoes.
        try:
            parsed = json.loads(_repair_json_blob(blob))
        except (json.JSONDecodeError, ValueError):
            return None
    items = None
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        for key in ("items", "responses", "scores", "results"):
            if isinstance(parsed.get(key), list):
                items = parsed[key]
                break
    if not items:
        return None

    lo, hi = score_range
    out: Dict[Any, Dict[str, Any]] = {}
    for entry in items:
        if not isinstance(entry, dict) or "id" not in entry or "score" not in entry:
            continue
        id_ = entry["id"]
        try:
            score = int(entry["score"])
        except (ValueError, TypeError):
            continue
        if not (lo <= score <= hi):
            logger.warning(
                f"batched judge returned score={score} out of range "
                f"[{lo},{hi}] for id={id_}; dropping"
            )
            continue
        reason = entry.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        out[id_] = {"score": score, "reason": reason}

    # Verify we got every expected id; missing ones come back as Nones in
    # the caller's output dict so it can decide how to handle them.
    expected_set = set(expected_ids)
    missing = expected_set - set(out.keys())
    if missing:
        logger.warning(
            f"batched judge missing {len(missing)} expected ids: "
            f"{sorted(list(missing))[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return out


#: The OpenAI service tiers a judge call may name (2026-10-26).  ``"flex"``:
#: lower priority, priced at Batch rates, answered like a real-time call
#: but slower, and refused with an uncharged HTTP 429 when capacity is short
#: (developers.openai.com Flex processing guide, read 2026-10-26).
SERVICE_TIERS = ("flex",)

#: Flex processing (pipeline step 3's ``--service_tier flex``) is tried for this
#: long before a request falls back to the standard tier (Roger, 2026-10-29:
#: "if flex keeps failing, it should eventually fall back to an actual
#: real-time query, but only after trying for a while").  The clock starts when a
#: request is first sent, not when it was queued, and each attempt goes out with
#: no SDK retries and a timeout of the time left, at least :data:`FLEX_MIN_ATTEMPT_S`
#: (:func:`flex_attempt_client`), so that ten minutes means ten minutes: until
#: 2026-11-02 one attempt could hold a request for ``FLEX_TIMEOUT_S`` x
#: (``FLEX_MAX_RETRIES`` + 1), over two hours (the stuck Luna and Terra run of
#: 2026-11-01).  Between attempts the call sleeps :data:`FLEX_RETRY_SLEEP_S`.  The
#: fallback call goes through the standard client with no tier and is charged
#: under the plain model name, so ``usage.json`` records how much went each way;
#: :data:`FLEX_FALLBACKS` counts the fallbacks by model.
FLEX_FALLBACK_AFTER_S = 600.0
FLEX_RETRY_SLEEP_S = 30.0
FLEX_MIN_ATTEMPT_S = 60.0
#: The Flex client's own limits (:func:`judge_client`): Flex calls can wait in a
#: queue and be refused with an uncharged 429.
FLEX_TIMEOUT_S = 900.0
FLEX_MAX_RETRIES = 8
FLEX_FALLBACKS: Dict[str, int] = {}


def flex_attempt_client(client: Any, started: float) -> Any:
    """``client`` for one Flex attempt: no SDK retries, and a timeout of the time
    left before :data:`FLEX_FALLBACK_AFTER_S` (at least :data:`FLEX_MIN_ATTEMPT_S`).
    A client without ``with_options`` (a test fake) is used as it is."""
    with_options = getattr(client, "with_options", None)
    if with_options is None:
        return client
    left = FLEX_FALLBACK_AFTER_S - (time.monotonic() - started)
    return with_options(max_retries=0, timeout=max(left, FLEX_MIN_ATTEMPT_S))


def make_gemini_client(api_key: str, timeout_s: float = 120.0) -> Any:
    """An httpx client for generateContent; the key travels only in the
    x-goog-api-key header (never in the URL, which an HTTP log prints)."""
    import httpx
    return httpx.AsyncClient(base_url=GEMINI_BASE_URL, headers={"x-goog-api-key": api_key},
                             timeout=httpx.Timeout(timeout_s, connect=20.0))


def judge_client(model: str, service_tier: Optional[str] = None) -> Any:
    """The async client for the judge ``model``'s provider (for a Flex run, with
    :data:`FLEX_TIMEOUT_S` and :data:`FLEX_MAX_RETRIES`).  Keys come from the
    environment (the caller's ``load_dotenv``)."""
    provider = provider_for_model(model)
    if provider == "anthropic":
        import anthropic
        return anthropic.AsyncAnthropic()
    if provider == "google":
        return make_gemini_client(os.environ["GEMINI_API_KEY"])
    if service_tier == "flex":
        return openai.AsyncOpenAI(timeout=FLEX_TIMEOUT_S, max_retries=FLEX_MAX_RETRIES)
    return openai.AsyncOpenAI()


def usage_key(model: str, service_tier: Optional[str] = None) -> str:
    """The name a call's tokens are charged under: ``<model>:flex`` for a Flex
    call (priced at the batch rate, ``judge_pricing``), else ``model``."""
    return f"{model}:{service_tier}" if service_tier else model


def prompt_parts(prompt: Any) -> Tuple[str, Optional[str]]:
    """A prompt as ``(prefix, suffix)``: a two-part prompt (a tuple or list of two strings) whose prefix the
    provider is asked to cache, or ``(text, None)`` for a plain string.  Response judging sends the question
    and the answers as the prefix and the axis last, so that the k axes' calls on one batch read the prefix
    from the cache (Roger, 2026-11-04; OpenAI reads at 0.1x the input rate, Anthropic the same)."""
    if isinstance(prompt, (tuple, list)) and len(prompt) == 2:
        return str(prompt[0]), str(prompt[1])
    return str(prompt), None


def prompt_text(prompt: Any) -> str:
    """The whole prompt text, the two parts joined."""
    a, b = prompt_parts(prompt)
    return a if b is None else a + b


def chat_content(prompt: Any, model: Optional[str] = None) -> Any:
    """A chat message's content: the string, or two text parts with an explicit cache breakpoint on the
    first (``prompt_cache_breakpoint``, GPT-5.6 and later; checked live on 2026-11-04: the second call of a
    2,172-token prefix read it all back as cached).  The request's explicit cache mode
    (:func:`prompt_cache_fields`) then caches exactly that prefix.  A model that refuses
    ``prompt_cache_options`` (gpt-4.1-mini, which caches on its own) gets the parts joined."""
    a, b = prompt_parts(prompt)
    if b is None:
        return a
    if model is not None and not accepts_cache_options(model):
        return a + b
    return [{"type": "text", "text": a, "prompt_cache_breakpoint": {"mode": "explicit"}}, {"type": "text", "text": b}]


def anthropic_content(prompt: Any) -> Any:
    """A Messages API content: the string, or two text blocks with ``cache_control`` on the first."""
    a, b = prompt_parts(prompt)
    if b is None:
        return a
    return [{"type": "text", "text": a, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": b}]


def chat_request_body(prompt: Any, model: str, max_tokens: int, *,
                      temperature: Optional[float] = DEFAULT_TEMPERATURE,
                      service_tier: Optional[str] = None,
                      prompt_cache: str = DEFAULT_PROMPT_CACHE,
                      settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The chat-completion request a judge call sends: the ``body`` of an
    OpenAI Batch API line (``assistant_axis.openai_batch``), unchanged, and
    through :func:`chat_create_kwargs` the keyword arguments of
    ``client.chat.completions.create``, so that a response judged in a batch
    got exactly the request a live call sends.

    ``temperature`` is left out at the default, 1 (every request until
    2026-10-12 sent ``"temperature": 1`` explicitly: the same sampling), and
    is checked against :data:`MODEL_CAPABILITIES` otherwise
    (:func:`request_temperature`).  ``settings`` when given, else the model's
    :data:`JUDGE_REQUEST_SETTINGS` are added (GPT-6 Luna's reasoning effort),
    ``service_tier`` when given (one of :data:`SERVICE_TIERS`), and the
    prompt-caching field of ``prompt_cache`` (:func:`prompt_cache_fields`;
    :data:`DEFAULT_PROMPT_CACHE`)."""
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": chat_content(prompt, model)}],
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
    body.update(prompt_cache_fields(model, prompt_cache))
    return body


@dataclass(frozen=True)
class Truncated:
    """What a call function returns, instead of the text, for a reply cut at
    ``max_tokens``.  It usually lacks its ``SCORE:`` line, so it is never parsed:
    it scores as missing.  ``text`` is kept for inspection only."""
    text: Optional[str] = None


class _CallOutcome:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


#: Returned by a call function when its ``gate()`` said not to issue the request.
SKIPPED = _CallOutcome("SKIPPED")
#: Returned by a call function for a call that got no answer (network, 5xx, a 429
#: past its retries).  Nothing was charged and nothing is known about the item, so
#: unlike None (a refusal or a content flag, which asking again would repeat) a
#: caller keeping a record must leave the item open and ask it again.
FAILED = _CallOutcome("FAILED")


@dataclass
class CallStats:
    """What happened to one model's calls in a run (beyond scores)."""
    refused: int = 0          # Anthropic stop_reason "refusal": counted, unparseable
    content_flagged: int = 0  # HTTP 400 content flags, blocked Gemini prompts: counted, unparseable
    errors: int = 0           # calls that got no answer (network, 5xx...)
    flex_fallbacks: int = 0   # Flex calls sent to the standard tier after FLEX_FALLBACK_AFTER_S
    skipped: int = 0          # not issued (a budget stop, or a fatal error elsewhere)
    truncated: int = 0        # replies cut at max_tokens: counted by the caller, which records them
    refusal_categories: Dict[str, int] = field(default_factory=dict)

    def note_refusal(self, model: str, category: Optional[str]) -> None:
        self.refused += 1
        key = str(category)
        self.refusal_categories[key] = self.refusal_categories.get(key, 0) + 1
        REFUSED_REPLIES[model] = REFUSED_REPLIES.get(model, 0) + 1
        logger.warning("%s refused a judge prompt (category %s; %d refusals so far)", model, category,
                       REFUSED_REPLIES[model])


async def call_openai_judge(
    client: Any,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    usage: Any = None,
    *,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    settings: Optional[Dict[str, Any]] = None,
    service_tier: Optional[str] = None,
    fallback_client: Any = None,
    prompt_cache: str = DEFAULT_PROMPT_CACHE,
    stats: Optional[CallStats] = None,
    gate: Optional[Callable[[], bool]] = None,
) -> Any:
    """One chat-completion judge call: the reply text, a :class:`Truncated` for a
    reply cut at ``max_tokens``, None for a content flag or an empty reply,
    :data:`FAILED` for a call that got no answer, or :data:`SKIPPED` when
    ``gate()`` (checked once the rate limiter lets the request go, before it is
    first sent, and between Flex attempts) is True.

    ``usage`` is charged under :func:`usage_key` with
    :func:`assistant_axis.judge_pricing.openai_usage`.  A Flex call that fails
    with anything but an HTTP 400 is tried again every :data:`FLEX_RETRY_SLEEP_S`
    until :data:`FLEX_FALLBACK_AFTER_S` has passed since it was first sent, then
    sent once through ``fallback_client`` with no tier (without one it fails as
    any other call does).  Raises :class:`BillingError` when the provider refuses
    the account, and for an HTTP 400 :class:`JudgeRequestError`
    (:class:`FlexRefusedError` when it refuses the service tier), except a content
    flag, and likewise before sending for what :data:`MODEL_CAPABILITIES` says the
    model refuses."""
    from .judge_pricing import openai_usage
    stats = stats if stats is not None else CallStats()
    if service_tier == "flex" and accepts_flex(model) is False:
        raise FlexRefusedError(f"OpenAI refuses Flex processing for {model} (HTTP 400; judge.MODEL_CAPABILITIES): "
                               f"judge it through the Batch API or in real time without a service tier")
    body = chat_request_body(prompt, model, max_tokens, temperature=temperature, settings=settings,
                             service_tier=service_tier, prompt_cache=prompt_cache)
    started: Optional[float] = None
    while True:
        await rate_limiter.acquire()
        if started is None:
            started = time.monotonic()    # the clock starts at the first send, not when it was queued
            if gate is not None and gate():
                stats.skipped += 1
                return SKIPPED
        sender = flex_attempt_client(client, started) if service_tier == "flex" else client
        try:
            response = await sender.chat.completions.create(**chat_create_kwargs(body))
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
                _note_content_flag(model, flag, e)
                return None
            if service_tier == "flex" and fallback_client is not None:
                if gate is not None and gate():
                    stats.skipped += 1    # a fatal error elsewhere, or the budget: stop retrying
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
                FLEX_FALLBACKS[model] = FLEX_FALLBACKS.get(model, 0) + 1
                client, service_tier = fallback_client, None
                body = chat_request_body(prompt, model, max_tokens, temperature=temperature, settings=settings,
                                         prompt_cache=prompt_cache)
                continue
            stats.errors += 1
            logger.error(f"Error calling judge model {model}: {e}")
            return FAILED
    try:
        if usage is not None:
            usage.charge_usage(usage_key(model, service_tier), openai_usage(response))
        first = response.choices[0] if response.choices else None
        text = first.message.content if first is not None else None
        if first is not None and getattr(first, "finish_reason", None) == "length":
            return Truncated(text or None)
        return text or None
    except Exception as e:  # noqa: BLE001 -- a reply that cannot be read: billed, so not asked again
        logger.error(f"Error reading the judge model's reply: {e}")
        return None


async def call_judge_single(
    client: openai.AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    usage: Optional["MultiModelUsage"] = None,
    *,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    reject_truncated: bool = False,
    service_tier: Optional[str] = None,
    fallback_client: Optional[openai.AsyncOpenAI] = None,
    prompt_cache: str = DEFAULT_PROMPT_CACHE,
) -> Optional[str]:
    """Call the judge model with a single prompt (:func:`call_openai_judge`).

    Returns the response text, or None when the call failed (logged).  An
    HTTP 400 is raised as :class:`JudgeRequestError` instead, as is a
    non-default ``temperature`` for a model that refuses one (before sending),
    except a 400 that flags this prompt's content (:func:`content_flag`),
    which gives None with a WARNING.  An account the provider refuses raises
    :class:`BillingError`.  ``service_tier`` (``"flex"``) is sent and the tokens
    are charged under :func:`usage_key`.

    Args:
        usage: Optional :class:`assistant_axis.judge_pricing.MultiModelUsage`
            accumulator.  When provided, the response's token counts
            (:func:`assistant_axis.judge_pricing.openai_usage`, the cache
            writes and reads included) are added to ``usage[model]``.
            Pass one for every automated/batched call site — see AGENT_NOTES
            "Token usage logging is mandatory on batched LLM call sites".
        temperature: sent only when it is not the default
            (:func:`chat_request_body`).
        prompt_cache: the request's prompt-caching mode
            (:data:`PROMPT_CACHE_MODES`; :data:`DEFAULT_PROMPT_CACHE`).
        reject_truncated: return None for a reply cut at ``max_tokens``
            (``finish_reason == "length"``), counted in
            :data:`TRUNCATED_REPLIES` (the usage is charged all the same).
        fallback_client: a standard-tier client for a Flex call (step 3
            passes one with ``--service_tier flex``), as
            :func:`call_openai_judge`.  Without one, a failed Flex call gives
            None as any other failed call does.
    """
    reply = await call_openai_judge(client, prompt, model, max_tokens, rate_limiter, usage,
                                    temperature=temperature, service_tier=service_tier,
                                    fallback_client=fallback_client, prompt_cache=prompt_cache)
    return _plain_reply(reply, model, reject_truncated)


def _plain_reply(reply: Any, model: str, reject_truncated: bool) -> Optional[str]:
    """A call function's result as the plain callers take it: the text, or None."""
    if reply is FAILED or reply is SKIPPED:
        return None
    if isinstance(reply, Truncated):
        if reject_truncated:
            TRUNCATED_REPLIES[model] = TRUNCATED_REPLIES.get(model, 0) + 1
            return None
        return reply.text
    return reply


async def call_judge_batch(
    client: Any,
    prompts: List[str],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    batch_size: int = 50,
    usage: Optional["MultiModelUsage"] = None,
    *,
    reject_truncated: bool = False,
    service_tier: Optional[str] = None,
    fallback_client: Optional[openai.AsyncOpenAI] = None,
    prompt_cache: str = DEFAULT_PROMPT_CACHE,
) -> List[Optional[str]]:
    """Call the judge model with multiple prompts concurrently.

    ``service_tier`` (``"flex"``, OpenAI models only), ``fallback_client``
    (the standard-tier client a Flex call falls back to, :func:`call_judge_single`)
    and ``prompt_cache`` (:data:`PROMPT_CACHE_MODES`) are passed to every call.

    ``client`` is the provider's async client for ``model``
    (:func:`provider_for_model`): an ``openai.AsyncOpenAI`` for a GPT model,
    an ``anthropic.AsyncAnthropic`` for a Claude model (pipeline step 3 in
    real time, since 2026-10-26).

    ``usage``: optional :class:`assistant_axis.judge_pricing.MultiModelUsage`
    accumulator, passed to every single call (AGENT_NOTES "Token usage
    logging is mandatory on batched LLM call sites").

    A failed call gives None in its place (a content-flagged prompt too),
    except a :class:`JudgeRequestError` (any other HTTP 400), which is raised
    once its chunk has finished.
    """
    results = []
    try:
        anthropic_model = provider_for_model(model) == "anthropic"
        google_model = provider_for_model(model) == "google"
    except ValueError:      # a name no prefix covers goes to the client given, as before
        anthropic_model = google_model = False

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        if anthropic_model:
            tasks = [
                call_anthropic_judge_single(client, prompt, model, max_tokens, rate_limiter, usage=usage,
                                            reject_truncated=reject_truncated)
                for prompt in batch
            ]
        elif google_model:
            async def one_gemini(prompt: str) -> Optional[str]:
                reply = await call_gemini_judge(client, prompt, model, max_tokens, rate_limiter, usage)
                return _plain_reply(reply, model, reject_truncated)
            tasks = [one_gemini(prompt) for prompt in batch]
        else:
            tasks = [
                call_judge_single(client, prompt, model, max_tokens, rate_limiter, usage=usage,
                                  reject_truncated=reject_truncated, service_tier=service_tier,
                                  fallback_client=fallback_client, prompt_cache=prompt_cache)
                for prompt in batch
            ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        bad_request = next((r for r in batch_results if isinstance(r, JudgeRequestError)), None)
        if bad_request is not None:
            raise bad_request

        processed = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Exception in batch: {result}")
                processed.append(None)
            else:
                processed.append(result)

        results.extend(processed)

    return results


async def score_responses(
    responses: List[Dict[str, str]],
    eval_prompt_template: str,
    judge_model: str = "gpt-4.1-mini",
    # 200, as pipeline/3_judge.py: the reasoning-first rubrics need room for
    # 2-3 sentences and the SCORE line (10 truncated every reply, and the
    # parser's fallback then read a digit out of the cut-off reasoning).
    max_tokens: int = 200,
    requests_per_second: int = 100,
    batch_size: int = 50,
) -> List[Optional[int]]:
    """
    Score a list of responses using an LLM judge.

    Args:
        responses: List of dicts with 'question' and 'response' keys
        eval_prompt_template: Template string with {question} and {answer} placeholders
        judge_model: OpenAI model to use as judge
        max_tokens: Max tokens for judge response
        requests_per_second: Rate limit for API calls
        batch_size: Concurrent batch size

    Returns:
        List of scores (0-3) or None for failed parsing
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Build prompts
    prompts = []
    for resp in responses:
        prompt = eval_prompt_template.format(
            question=resp["question"],
            answer=resp["response"]
        )
        prompts.append(prompt)

    # Initialize client and rate limiter
    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(requests_per_second)

    # Call judge
    judge_responses = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )

    # Parse scores
    scores = []
    for response_text in judge_responses:
        score = parse_judge_score(response_text) if response_text else None
        scores.append(score)

    return scores


def score_responses_sync(
    responses: List[Dict[str, str]],
    eval_prompt_template: str,
    judge_model: str = "gpt-4.1-mini",
    max_tokens: int = 200,
    requests_per_second: int = 100,
    batch_size: int = 50,
) -> List[Optional[int]]:
    """
    Synchronous wrapper for score_responses.

    Args:
        responses: List of dicts with 'question' and 'response' keys
        eval_prompt_template: Template string with {question} and {answer} placeholders
        judge_model: OpenAI model to use as judge
        max_tokens: Max tokens for judge response
        requests_per_second: Rate limit for API calls
        batch_size: Concurrent batch size

    Returns:
        List of scores (0-3) or None for failed parsing
    """
    return asyncio.run(score_responses(
        responses=responses,
        eval_prompt_template=eval_prompt_template,
        judge_model=judge_model,
        max_tokens=max_tokens,
        requests_per_second=requests_per_second,
        batch_size=batch_size
    ))


# ---------------------------------------------------------------------------
# Anthropic equivalent of call_judge_single
# ---------------------------------------------------------------------------
#
# Steering judges (Phase 2) use the bidirectional GPT+Haiku effect
# ensemble, so judge.py needs an Anthropic single-call helper.  Kept
# minimal to mirror call_judge_single() shape; structured output is
# delivered via prompt instruction + JSON parsing rather than
# response_format (Anthropic supports a tools-based json mode but the
# prompt-instruction approach is simpler and works across both providers
# uniformly).

def anthropic_request_params(prompt: str, model: str, max_tokens: int, *,
                             temperature: Optional[float] = DEFAULT_TEMPERATURE,
                             settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The Messages API parameters a judge call sends: the keyword arguments
    of ``client.messages.create`` and, unchanged, the ``params`` of a Message
    Batches request (``assistant_axis.anthropic_batch``), as
    :func:`chat_request_body` is for OpenAI.  ``temperature`` is left out at
    the default; ``settings`` when given, else the model's
    :data:`JUDGE_REQUEST_SETTINGS` are added (Haiku 5.5's thinking off)."""
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": anthropic_content(prompt)}],
    }
    t = request_temperature(model, temperature)
    if t is not None:
        params["temperature"] = t
    params.update(judge_request_settings(model) if settings is None else json.loads(json.dumps(settings)))
    return params


def anthropic_reply(message: Any) -> Tuple[Optional[str], Optional[str], int, int, bool]:
    """``(text, finish_reason, input_tokens, output_tokens, refused)`` of an
    Anthropic message, an SDK object or a plain dict (a Message Batches
    result): the text of its text blocks (None when empty), ``"length"`` for
    a reply cut at ``max_tokens`` (OpenAI's word, which step 3 reads), and
    whether it stopped as a refusal."""
    def g(obj: Any, name: str, default: Any = None) -> Any:
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    stop = g(message, "stop_reason")
    text = "".join(g(b, "text", "") or "" for b in (g(message, "content") or [])
                   if g(b, "type", "text") == "text")
    usage = g(message, "usage") or {}
    return (text or None, "length" if stop == "max_tokens" else stop,
            int(g(usage, "input_tokens", 0) or 0), int(g(usage, "output_tokens", 0) or 0),
            stop == "refusal")


async def call_anthropic_judge(
    client: Any,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    usage: Any = None,
    *,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    settings: Optional[Dict[str, Any]] = None,
    stats: Optional[CallStats] = None,
    gate: Optional[Callable[[], bool]] = None,
) -> Any:
    """One Messages API judge call: the reply text (of the text blocks, read by
    type: a model that thinks can begin with ``thinking`` blocks), a
    :class:`Truncated` for a reply cut at ``max_tokens``, None for a refusal
    (``stop_reason == "refusal"``, counted and charged), a content flag or an
    empty reply, :data:`FAILED` for a call that got no answer, or :data:`SKIPPED`.
    Raises :class:`BillingError` when the provider refuses the account, and
    :class:`JudgeRequestError` for an HTTP 400 unless it flags the prompt's
    content, or before sending for a temperature the model refuses."""
    from .judge_pricing import anthropic_usage
    stats = stats if stats is not None else CallStats()
    params = anthropic_request_params(prompt, model, max_tokens, temperature=temperature, settings=settings)
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
            _note_content_flag(model, flag, e)
            return None
        stats.errors += 1
        logger.error(f"Error calling Anthropic judge model {model}: {e}")
        return FAILED
    try:
        if usage is not None:
            usage.charge_usage(model, anthropic_usage(response))
        text, finish, _n_in, _n_out, refused = anthropic_reply(response)
        if refused:
            details = getattr(response, "stop_details", None)
            stats.note_refusal(model, getattr(details, "category", None))
            return None
        if finish == "length":
            return Truncated(text)
        return text
    except Exception as e:  # noqa: BLE001 -- a reply that cannot be read: billed, so not asked again
        logger.error(f"Error reading the judge model's reply: {e}")
        return None


async def call_anthropic_judge_single(
    client: "anthropic.AsyncAnthropic",
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    usage: Optional["MultiModelUsage"] = None,
    *,
    reject_truncated: bool = False,
) -> Optional[str]:
    """Call an Anthropic judge model with a single prompt (:func:`call_anthropic_judge`).

    Returns the response text, or None on error.  Errors are logged at
    ERROR level and swallowed so a single bad call doesn't take down a
    whole batch of judging, except an HTTP 400, raised as
    :class:`JudgeRequestError` because every call would fail alike (a 400
    that flags this prompt's content, :func:`content_flag`, gives None),
    and an account the provider refuses, raised as :class:`BillingError`.

    A refusal (``stop_reason == "refusal"``) gives None, counted in
    :data:`REFUSED_REPLIES` with a WARNING naming its category.

    Args:
        temperature: sent only when it is not the default; a model that
            refuses one (:data:`MODEL_CAPABILITIES`) raises
            :class:`JudgeRequestError` for any other value, before sending.
        usage: Optional :class:`assistant_axis.judge_pricing.MultiModelUsage`
            accumulator.  Same semantics as :func:`call_judge_single`.
        reject_truncated: return None for a reply cut at ``max_tokens``
            (``stop_reason == "max_tokens"``), counted in
            :data:`TRUNCATED_REPLIES`, as :func:`call_judge_single` does.
    """
    reply = await call_anthropic_judge(client, prompt, model, max_tokens, rate_limiter, usage,
                                       temperature=temperature)
    return _plain_reply(reply, model, reject_truncated)


# ---------------------------------------------------------------------------
# Gemini (2026-11-02): generateContent over HTTPS, reached directly with its own key
# ---------------------------------------------------------------------------

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_RETRY_STATUSES = (429, 500, 502, 503, 504)
GEMINI_ATTEMPTS = 8
GEMINI_RETRY_SLEEP_S = (5, 10, 20, 30, 45, 60, 60)


def gemini_request_body(prompt: str, model: str, max_tokens: int, *,
                        settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The generateContent request: ``settings`` when given, else the model's
    :data:`JUDGE_REQUEST_SETTINGS`, go in ``generationConfig``."""
    config: Dict[str, Any] = {"maxOutputTokens": max_tokens}
    config.update(judge_request_settings(model) if settings is None else json.loads(json.dumps(settings)))
    return {"contents": [{"parts": [{"text": prompt_text(prompt)}]}], "generationConfig": config}


def gemini_quota_ids(body: str) -> List[str]:
    """The quotas a 429 names (``GenerateRequestsPerDayPerProjectPerModel-FreeTier``
    and the like), from its ``google.rpc.QuotaFailure`` detail; empty when the body
    says nothing readable."""
    try:
        details = json.loads(body)["error"]["details"]
        return [v["quotaId"] for d in details for v in d.get("violations", []) if "quotaId" in v]
    except (ValueError, KeyError, TypeError, AttributeError):
        return []


def gemini_reply(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], int, int]:
    """``(text, finish_reason, prompt_tokens, output_tokens)`` of a generateContent
    response; output tokens include the thought tokens, which Google bills as
    output, and the text leaves out the thought parts."""
    u = data.get("usageMetadata") or {}
    n_in = int(u.get("promptTokenCount") or 0)
    n_out = int(u.get("candidatesTokenCount") or 0) + int(u.get("thoughtsTokenCount") or 0)
    candidates = data.get("candidates") or []
    if not candidates:
        return None, (data.get("promptFeedback") or {}).get("blockReason"), n_in, n_out
    c = candidates[0]
    parts = (c.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")) or None
    return text, c.get("finishReason"), n_in, n_out


async def call_gemini_judge(
    client: Any,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    usage: Any = None,
    *,
    settings: Optional[Dict[str, Any]] = None,
    stats: Optional[CallStats] = None,
    gate: Optional[Callable[[], bool]] = None,
) -> Any:
    """One generateContent call (``client``: :func:`make_gemini_client`): the reply
    text, a :class:`Truncated` when it hit ``maxOutputTokens``, None for a blocked
    prompt, :data:`FAILED` for a call that got no answer, or :data:`SKIPPED`.  A
    connection error, a 429 or a 5xx (rate limits, "high demand" 503s) is tried
    :data:`GEMINI_ATTEMPTS` times with growing sleeps.  An HTTP 400, 401, 403 or 404
    raises :class:`JudgeRequestError`, and a 402 (prepaid credit used up) or a 429
    naming a daily quota :class:`BillingError`, since no retry within a run gets past
    it."""
    from .judge_pricing import gemini_usage
    stats = stats if stats is not None else CallStats()
    body = gemini_request_body(prompt, model, max_tokens, settings=settings)
    last: Optional[str] = None
    for attempt in range(GEMINI_ATTEMPTS):
        await rate_limiter.acquire()
        if attempt == 0 and gate is not None and gate():
            stats.skipped += 1
            return SKIPPED
        try:
            resp = await client.post(f"/models/{model}:generateContent", json=body)
        except Exception as e:  # noqa: BLE001 -- a connection error: tried again
            last = f"{type(e).__name__}: {e}"
            resp = None
        if resp is not None:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    text, finish, n_in, n_out = gemini_reply(data)
                except ValueError as e:     # a truncated or non-JSON body: no answer yet
                    text, finish, n_in, n_out = None, None, 0, 0
                    last = f"HTTP 200 with an unreadable body: {e}"
                    sleep = GEMINI_RETRY_SLEEP_S[min(attempt, len(GEMINI_RETRY_SLEEP_S) - 1)]
                    logger.warning(f"Gemini call to {model} failed ({last}); attempt {attempt + 1} of "
                                   f"{GEMINI_ATTEMPTS}, trying again in {sleep}s")
                    await asyncio.sleep(sleep)
                    continue
                if usage is not None:
                    usage.charge_usage(model, gemini_usage(data))     # the cache reads too
                if finish == "MAX_TOKENS":
                    return Truncated(text)
                if text is None:
                    stats.content_flagged += 1
                    logger.warning(f"{model}: no text in the reply (finishReason {finish}); this item gives None")
                return text
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code in (400, 401, 403, 404):
                logger.error(f"Error calling judge model {model}: {last}")
                raise JudgeRequestError(f"{model}: {last}")
            if resp.status_code == 402:     # "Your prepayment credits are depleted" (2026-11-03): every call would fail alike
                raise BillingError(f"{model}: {last}; a rerun resumes once the credit is topped up")
            if resp.status_code == 429:
                quotas = gemini_quota_ids(resp.text)
                if any("PerDay" in q for q in quotas):
                    raise BillingError(f"{model}: daily quota used up ({', '.join(quotas)}); a rerun resumes "
                                       f"once it resets")
                if quotas:
                    last = f"HTTP 429: {', '.join(quotas)}"
            if resp.status_code not in GEMINI_RETRY_STATUSES:
                break
        sleep = GEMINI_RETRY_SLEEP_S[min(attempt, len(GEMINI_RETRY_SLEEP_S) - 1)]
        logger.warning(f"Gemini call to {model} failed ({last}); attempt {attempt + 1} of {GEMINI_ATTEMPTS}, "
                       f"trying again in {sleep}s")
        await asyncio.sleep(sleep)
    stats.errors += 1
    logger.error(f"Error calling judge model {model}: {last}")
    return FAILED


async def call_judge_model(
    model: str,
    prompt: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    usage: Any = None,
    *,
    client: Any,
    settings: Optional[Dict[str, Any]] = None,
    service_tier: Optional[str] = None,
    fallback_client: Any = None,
    stats: Optional[CallStats] = None,
    gate: Optional[Callable[[], bool]] = None,
) -> Any:
    """:func:`call_openai_judge`, :func:`call_anthropic_judge` or
    :func:`call_gemini_judge` by the model's provider, at the provider's default
    temperature unless ``settings`` says otherwise (Gemini's T=0 is a setting)."""
    provider = provider_for_model(model)
    if provider == "google":
        return await call_gemini_judge(client, prompt, model, max_tokens, rate_limiter, usage,
                                       settings=settings, stats=stats, gate=gate)
    if provider == "anthropic":
        return await call_anthropic_judge(client, prompt, model, max_tokens, rate_limiter, usage,
                                          settings=settings, stats=stats, gate=gate)
    return await call_openai_judge(client, prompt, model, max_tokens, rate_limiter, usage, settings=settings,
                                   service_tier=service_tier, fallback_client=fallback_client, stats=stats,
                                   gate=gate)


async def call_judge_single_unified(
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    openai_client: Optional["openai.AsyncOpenAI"] = None,
    anthropic_client: Optional["anthropic.AsyncAnthropic"] = None,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    usage: Optional["MultiModelUsage"] = None,
    reject_truncated: bool = False,
) -> Optional[str]:
    """Provider-agnostic single judge call.

    Routes by ``model`` prefix (see :func:`provider_for_model`).  Caller is
    responsible for instantiating the relevant client(s) and rate
    limiter(s); this is a thin dispatch helper, not a connection pool.

    For mixed-provider workloads, the caller typically constructs one
    ``AsyncOpenAI`` and one ``AsyncAnthropic`` and passes both -- the
    routing picks whichever is needed per call.

    Args:
        temperature: for either provider, sent only when it is not the
            default (:func:`request_temperature`; until 2026-10-12 the
            OpenAI path ignored it and always sent 1).
        usage: Optional :class:`assistant_axis.judge_pricing.MultiModelUsage`
            accumulator -- ticks ``usage[model]`` with the per-call
            token counts.  Pass one on every batched/automated call site.

    Raises :class:`JudgeRequestError` on an HTTP 400 from either provider,
    other than one that flags this prompt's content (None, with a WARNING).
    """
    provider = provider_for_model(model)
    if provider == "openai":
        if openai_client is None:
            raise ValueError(
                f"openai_client required for model {model}"
            )
        return await call_judge_single(
            client=openai_client,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            rate_limiter=rate_limiter,
            usage=usage,
            temperature=temperature,
            reject_truncated=reject_truncated,
        )
    if provider == "anthropic":
        if anthropic_client is None:
            raise ValueError(
                f"anthropic_client required for model {model}"
            )
        return await call_anthropic_judge_single(
            client=anthropic_client,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            rate_limiter=rate_limiter,
            temperature=temperature,
            usage=usage,
            reject_truncated=reject_truncated,
        )
    raise ValueError(f"call_judge_single_unified has no client for {provider} judge model {model!r}: "
                     f"use call_judge_model")
