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
from typing import Dict, List, Optional, Any, Iterable, Tuple, Sequence, TYPE_CHECKING

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


def provider_for_model(model: str) -> str:
    """Return 'openai' or 'anthropic' based on the model name prefix.

    Raises ValueError for unrecognised prefixes so we fail loudly rather
    than silently routing a typo to the wrong SDK.
    """
    if any(model.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    if any(model.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic"
    raise ValueError(
        f"unknown judge model {model!r}; expected a prefix in "
        f"{_OPENAI_PREFIXES + _ANTHROPIC_PREFIXES}"
    )


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


def parse_judge_score(response_text: str) -> Optional[int]:
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

    Returns the integer in [0, 3], or None if nothing parseable.
    """
    if not response_text:
        return None

    # Stage 1: explicit SCORE: marker (preferred for new prompts).
    m = re.search(r"SCORE\s*:\s*(\d+)", response_text, flags=re.IGNORECASE)
    if m:
        try:
            score = int(m.group(1))
            if 0 <= score <= 3:
                return score
        except ValueError:
            pass

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


async def call_judge_single(
    client: openai.AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    usage: Optional["MultiModelUsage"] = None,
) -> Optional[str]:
    """Call the judge model with a single prompt.

    Args:
        usage: Optional :class:`assistant_axis.judge_pricing.MultiModelUsage`
            accumulator.  When provided, the (prompt_tokens, completion_tokens)
            tuple extracted from the response is added to ``usage[model]``.
            Pass one for every automated/batched call site — see AGENT_NOTES
            "Token usage logging is mandatory on batched LLM call sites".
    """
    await rate_limiter.acquire()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=1
        )

        if usage is not None:
            # Lazy import to avoid a hard dep cycle (judge_pricing imports
            # nothing from judge but judge.py is imported from many places).
            from .judge_pricing import extract_usage_openai
            pt, ct = extract_usage_openai(response)
            usage.charge(model, pt, ct)

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return None

    except Exception as e:
        logger.error(f"Error calling judge model: {e}")
        return None


async def call_judge_batch(
    client: openai.AsyncOpenAI,
    prompts: List[str],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    batch_size: int = 50
) -> List[Optional[str]]:
    """Call the judge model with multiple prompts concurrently."""
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        tasks = [
            call_judge_single(client, prompt, model, max_tokens, rate_limiter)
            for prompt in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Exception in batch: {result}")
                processed.append(None)
            else:
                processed.append(result)

        results.extend(processed)

    return results


async def call_judge_batch_unified(
    prompts: List[str],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    batch_size: int = 50,
    openai_client: Optional["openai.AsyncOpenAI"] = None,
    anthropic_client: Optional["anthropic.AsyncAnthropic"] = None,
    temperature: float = 1.0,
    usage: Optional["MultiModelUsage"] = None,
) -> List[Optional[str]]:
    """Provider-agnostic equivalent of :func:`call_judge_batch`.

    Same batching/gather shape as :func:`call_judge_batch`, but dispatches
    each call through :func:`call_judge_single_unified` so ``model`` can be
    an OpenAI or an Anthropic model name -- pass whichever client(s) the
    models you intend to use actually need (unused clients can stay None).

    Use this instead of :func:`call_judge_batch` at any call site that
    should support judging with a Claude model, not just OpenAI models.
    """
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        tasks = [
            call_judge_single_unified(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                rate_limiter=rate_limiter,
                openai_client=openai_client,
                anthropic_client=anthropic_client,
                temperature=temperature,
                usage=usage,
            )
            for prompt in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

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
    max_tokens: int = 10,
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
    max_tokens: int = 10,
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

# Models where adaptive thinking runs BY DEFAULT (thinking param omitted
# still thinks) and which accept output_config.effort to bound it.
# Diagnosed 2026-09-23: Opus 5 (and Sonnet 5) put a ThinkingBlock first in
# response.content by default -- code that blindly reads content[0].text
# raises AttributeError on every call, which the broad except swallows as
# a silent parse failure while the call still gets billed (thinking tokens
# included). Passing effort="low" bounds the thinking-token spend for a
# short rating task like judging; models NOT in this set (Haiku 4.5,
# older Sonnet/Opus) reject the effort param outright, so it must stay
# gated to only the models known to accept it.
_ANTHROPIC_EFFORT_CAPABLE = ("opus-5", "sonnet-5", "fable-5", "mythos-5",
                             "opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6")


def _anthropic_supports_effort(model: str) -> bool:
    m = model.lower()
    return any(frag in m for frag in _ANTHROPIC_EFFORT_CAPABLE)


def _extract_text_block(content) -> Optional[str]:
    """Find the actual text block in an Anthropic response's content list.

    NOT content[0] -- models with thinking on by default (Opus 5, Sonnet 5,
    ...) put a ThinkingBlock (or more than one) before the TextBlock.
    """
    for block in content or []:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            return block.text
    return None


async def call_anthropic_judge_single(
    client: "anthropic.AsyncAnthropic",
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    temperature: float = 1.0,
    usage: Optional["MultiModelUsage"] = None,
) -> Optional[str]:
    """Call an Anthropic judge model with a single prompt.

    Returns the response text, or None on error.  Errors are logged at
    ERROR level and swallowed so a single bad call doesn't take down a
    whole batch of judging.

    Args:
        usage: Optional :class:`assistant_axis.judge_pricing.MultiModelUsage`
            accumulator.  Same semantics as :func:`call_judge_single`.
    """
    await rate_limiter.acquire()

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if _anthropic_supports_effort(model):
        kwargs["output_config"] = {"effort": "low"}

    try:
        response = await client.messages.create(**kwargs)
        if usage is not None:
            from .judge_pricing import extract_usage_anthropic
            pt, ct = extract_usage_anthropic(response)
            usage.charge(model, pt, ct)
        text = _extract_text_block(response.content)
        if text:
            return text
        logger.error(f"No text block in response content for model {model} "
                     f"(block types: {[getattr(b, 'type', '?') for b in (response.content or [])]})")
        return None
    except Exception as e:  # noqa: BLE001 - we deliberately catch everything
        logger.error(f"Error calling Anthropic judge model {model}: {e}")
        return None


async def call_judge_single_unified(
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    openai_client: Optional["openai.AsyncOpenAI"] = None,
    anthropic_client: Optional["anthropic.AsyncAnthropic"] = None,
    temperature: float = 1.0,
    usage: Optional["MultiModelUsage"] = None,
) -> Optional[str]:
    """Provider-agnostic single judge call.

    Routes by ``model`` prefix (see :func:`provider_for_model`).  Caller is
    responsible for instantiating the relevant client(s) and rate
    limiter(s); this is a thin dispatch helper, not a connection pool.

    For mixed-provider workloads, the caller typically constructs one
    ``AsyncOpenAI`` and one ``AsyncAnthropic`` and passes both -- the
    routing picks whichever is needed per call.

    Args:
        usage: Optional :class:`assistant_axis.judge_pricing.MultiModelUsage`
            accumulator -- ticks ``usage[model]`` with the per-call
            token counts.  Pass one on every batched/automated call site.
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
        )
    raise ValueError(f"unknown provider for model {model!r}")
