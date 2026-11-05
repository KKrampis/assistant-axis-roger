"""Judge pricing constants + budget tracking utilities.

Phase 4c (May 2026) — promotes the per-token rates (previously
duplicated in ``results_analysis/plot_batch_size_quality_vs_cost.py``)
into a single project-wide source of truth, and adds:

* :class:`UsageTotals` — accumulator for prompt/completion tokens
  and resolved USD cost.
* :class:`BudgetTracker` — wraps a UsageTotals with a hard budget
  cap (``--budget_usd``).  Each :meth:`charge` call updates the
  totals and raises :class:`BudgetExceededError` if the running
  cost crosses the cap.
* :func:`extract_usage_anthropic`, :func:`extract_usage_openai` —
  extract the token-count tuple from a provider's response object,
  surviving SDK version drift (the field names differ across major
  OpenAI / Anthropic SDK versions).
* :func:`price_for_model` — look up (input, output) USD per 1M
  rates by canonical model name; aliases handled.

USAGE
-----
The budget tracker is owned by :func:`run` in
``results_analysis/axis_judge_correlation.py`` and threaded through
to :func:`_call_anthropic_batch` / :func:`_call_openai_batch` so
every API call ticks the accumulator.  Callers MUST handle
:class:`BudgetExceededError` — they should re-raise with
``SystemExit(2)`` (cap-exceeded, structured exit code distinct from
cmd-line-error 1).

PRICING
-------
USD per 1M tokens, separated into input and output streams.  These
numbers are documented in README.md "Judging cost model" and are
the same numbers the cost-vs-quality plot uses.  Update both places
together if rates change.

Prompt caching (2026-11-01): a cached prompt token is billed at a multiple
of the input rate, and the multiples differ by model (``cache_rate_factors``):
OpenAI's GPT-5.6 and later charge a cache write at 1.25x and a read at 0.1x,
earlier OpenAI models write free and read at their cached-input rate, and
Anthropic writes at 1.25x and reads at 0.1x when a request asks for caching.
Every accumulator here takes the write and read counts beside the prompt
total (:class:`TokenUsage`, :func:`openai_usage`, :func:`anthropic_usage`),
so ``usage.json`` prices what the bill prices.  Until then every prompt token
was priced at the plain rate, which understated a luna Flex run by the write
premium (about 10%; see judge.DEFAULT_PROMPT_CACHE for why the judges now
send those models explicit mode with no breakpoints).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing constants (USD per 1,000,000 tokens, input / output split)
#
# Source: official OpenAI / Anthropic public pricing pages, May 2026.
# Mirrors the constants in
#   results_analysis/plot_batch_size_quality_vs_cost.py
# (those are the canonical definitions; this module re-exports them
# under stable names so the call-site code doesn't need to import the
# plotting script).
# ---------------------------------------------------------------------------

GPT_MINI_RATE_IN = 0.40   # USD / 1M input tokens (gpt-4.1-mini)
GPT_MINI_RATE_OUT = 1.60  # USD / 1M output tokens
HAIKU_RATE_IN = 1.00      # claude-haiku-4-5
HAIKU_RATE_OUT = 5.00
SONNET_RATE_IN = 3.00     # claude-sonnet-4
SONNET_RATE_OUT = 15.00


# --------------------------------------------------------------------------
# Roles-vs-traits response-length asymmetry (response-mode judging only).
# --------------------------------------------------------------------------
# Empirical: at the same B and the same N entities, response-mode judging
# of ROLES costs ~1.22× more per entity than TRAITS, because Qwen-3-32B
# produces ~22% longer responses when role-played than trait-modulated
# (mean 2094 vs 1718 chars, median 2351 vs 1842 chars across the 29 roles
# + 14 traits residual cohort used in 5d.1 = 14,500 + 7,000 responses).
# The 5d.1 actual/expected = 1.21 reconciles cleanly once this factor is
# applied (see ROLES_RESPONSE_LENGTH_FACTOR usage in surgical-rejudge
# budget formulae).  Output-mode tokens are NOT affected -- the judge
# emits a fixed-format score per response regardless of cohort.
#
# This factor applies ONLY to response-mode judging (where per-call
# input is dominated by raw response text).  Static-mode prompts
# (descriptions / instructions) don't include responses and have no
# such asymmetry; use 1.0 there.
ROLES_RESPONSE_LENGTH_FACTOR: float = 1.22
"""Per-entity response-mode cost multiplier for ``roles`` cohorts vs
``traits``.  Multiply the per-entity cost of a traits-only budget by
this factor to get the per-entity cost for roles, or use the helper
:func:`surgical_rejudge_cost_split` below to split a per-axis budget
into role/trait cohort shares.
"""


def surgical_rejudge_cost_split(
    per_axis_cost_usd: float,
    n_roles: int,
    n_traits: int,
    *,
    mode: str = "response",
) -> tuple[float, float]:
    """Split a per-axis judging budget between roles and traits cohorts,
    accounting for the response-mode roles-vs-traits cost asymmetry.

    Surgical rejudges (Phase 5b / 5c / 5d / etc.) score only N_r roles
    + N_t traits entities per axis; the cost-per-entity is NOT equal
    across cohorts in response-mode because Qwen role-played responses
    are ~22% longer than trait-modulated ones (see
    :data:`ROLES_RESPONSE_LENGTH_FACTOR`).

    Args:
        per_axis_cost_usd: Total expected cost for the axis (full
            volume per-axis cost from the cost table × N_total / 580).
        n_roles, n_traits: Number of entities surgically rejudged in
            each cohort.
        mode: ``"response"`` (default) or ``"static"``.  Response mode
            applies the 1.22× asymmetry; static mode treats roles and
            traits as equally expensive per entity (no per-response
            text in the prompt).

    Returns:
        ``(roles_cost_usd, traits_cost_usd)`` summing to
        ``per_axis_cost_usd``.

    Example: 5d.1 used N_r=29 + N_t=14 per axis at $3.50/axis
    (revised estimate); ``surgical_rejudge_cost_split(3.50, 29, 14)``
    returns ``($2.50, $1.00)``, matching the observed 5d.1 cell
    ratios within ~5%.
    """
    if mode == "response":
        weight_r = n_roles * ROLES_RESPONSE_LENGTH_FACTOR
        weight_t = n_traits
    elif mode == "static":
        weight_r = float(n_roles)
        weight_t = float(n_traits)
    else:
        raise ValueError(f"mode must be 'response' or 'static', got {mode!r}")
    w_total = weight_r + weight_t
    if w_total <= 0:
        return (0.0, 0.0)
    return (
        per_axis_cost_usd * weight_r / w_total,
        per_axis_cost_usd * weight_t / w_total,
    )

# (input_rate, output_rate) keyed by canonical model name.  Lookups
# are case-insensitive and use substring matching (so e.g.
# "gpt-4.1-mini-2024" still resolves to GPT_MINI rates).
_MODEL_RATES: tuple[tuple[str, float, float], ...] = (
    # Embedding models (trait-gap platform, M2/M3): input-only pricing, so
    # embedding calls are counted in usage.json like judge calls.  Local
    # models are free but are still recorded (n_calls, tokens).
    ("text-embedding-3-large", 0.13, 0.0),
    ("text-embedding-3-small", 0.02, 0.0),
    ("qwen3-embedding", 0.0, 0.0),
    ("bge-large", 0.0, 0.0),
    ("embeddinggemma", 0.0, 0.0),   # google/embeddinggemma-300m, local (M2 third arm)
    # Local sampler of the trait-gap base-model generator (Qwen/Qwen3-8B-Base on the Mac): free, but its
    # tokens are recorded (coding_plan_03_base_model.md, decision "no generator-side paid calls").
    ("qwen3-8b-base", 0.0, 0.0),
    ("gpt-4.1-mini", GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT),
    ("gpt-4o-mini", GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT),
    # GPT-6 Luna (developers.openai.com model page, read 2026-10-26): $0.10 / $0.50, cached input
    # $0.01, batch at half (the "gpt-" batch factor below).  A reasoning model: judges send
    # reasoning_effort "none" (judge.JUDGE_REQUEST_SETTINGS), else reasoning is billed as output.
    ("gpt-6-luna", 0.10, 0.50),
    # GPT-5.4 mini and GPT-5.6 Luna (OpenAI pricing page, read 2026-10-27; cached input $0.075 and
    # $0.02), candidates to replace gpt-4.1-mini, which shuts on 2027-01-14.  Reasoning models, sent
    # reasoning_effort "none" as Luna is.
    ("gpt-5.4-mini", 0.75, 4.50),
    ("gpt-5.6-luna", 0.20, 1.20),
    # GPT-5.6 Terra and Sol (OpenAI pricing page, read 2026-11-01), judged for the tier evaluation.
    ("gpt-5.6-terra", 2.00, 12.00),
    ("gpt-5.6-sol", 4.00, 20.00),
    # Gemini (ai.google.dev/gemini-api/docs/pricing, read 2026-11-01; Batch and Flex at half, the
    # "gemini-" batch factor below, read 2026-11-03).  2.5 Flash-Lite answers HTTP 404 to new users
    # (2026-11-02).
    ("gemini-3.1-flash-lite", 0.25, 1.50),
    # TypeSafe AI's Jev (typesafe.ai and the docs' parallel-questions cookbook, read 2026-11-04): $0.042 per
    # million input tokens, output "currently free of charge"; a request's state is billed once whatever the
    # number of questions.  No batch or flex tier.
    ("jev-", 0.042, 0.0),
    ("gemini-3.5-flash-lite", 0.30, 2.50),
    ("gemini-2.5-flash-lite", 0.10, 0.40),
    ("gemini-3-flash", 0.50, 3.00),
    ("gemini-2.5-flash", 0.30, 2.50),
    # Haiku 5.5 (claude-haiku-5-5, released 2026-10-07; pricing page read that day): $0.10 / $0.50 for
    # prompts up to 100,000 tokens, cache reads at the standard 0.1x ($0.01), batch at half (the "haiku"
    # batch factor below).  It must come before the bare "haiku" fragment, which would price it at 4.5's
    # rates.  Same tokenizer as Opus 4.7 and later (about 30% more tokens of the same text).
    ("haiku-5-5", 0.10, 0.50),
    ("haiku", HAIKU_RATE_IN, HAIKU_RATE_OUT),
    # Claude 5 and 5.5 (platform.claude.com/docs/en/about-claude/pricing,
    # read 2026-09-30).  These lines must come before the bare "sonnet" and
    # "opus" fragments, which would otherwise match first.  Models from 4.7 on
    # use a tokenizer that makes about 30% more tokens of the same text.
    ("sonnet-5", 2.00, 10.00),   # Sonnet 5 and Sonnet 5.5
    ("fable", 10.00, 50.00),     # Fable 5 and 5.1 (same page, read 2026-10-04); the M3 overlap test's fourth judge
    ("opus-5-5", 4.00, 20.00),
    ("opus-5", 5.00, 25.00),     # Opus 5 (and 4.5 to 4.8 at the same price, below)
    ("sonnet", SONNET_RATE_IN, SONNET_RATE_OUT),
    ("opus-4-5", 5.00, 25.00),
    ("opus-4-6", 5.00, 25.00),
    ("opus-4-7", 5.00, 25.00),
    ("opus-4-8", 5.00, 25.00),
    # Not a judge: the subject model, when it is reached through OpenRouter
    # instead of the pod (`qwen/qwen3-32b` served by DeepInfra; its price on
    # 2026-09-29).  Used by data_analysis/opening_form_experiment.py.
    ("qwen3-32b", 0.08, 0.28),
    # Retired Opus 4 and 4.1 ($15 / $75) are left out on purpose: a lookup
    # fails loud rather than pricing them as a current Opus.
)


# Batch requests are billed at a fraction of the real-time price.  A call
# site that runs in batch mode keeps its usage under "<model>:batch", which is
# priced with the factor below, so that a usage record says how the tokens
# were bought and a merged record recomputes to the right total.
BATCH_SUFFIX = ":batch"
_BATCH_PRICE_FACTORS: tuple[tuple[str, float], ...] = (
    ("qwen3-32b", 0.8),   # DeepInfra's own batch API, 2026-09-29
    ("gpt-", 0.5),        # OpenAI Batch API
    ("haiku", 0.5),       # Anthropic Message Batches
    ("sonnet", 0.5),
    ("opus", 0.5),
    ("fable", 0.5),
    ("gemini-", 0.5),     # Gemini Batch API (aistudio.google.com/docs/pricing, read 2026-11-03: 3.1 and 3.5 Flash-Lite)
)


#: OpenAI Flex processing (``service_tier="flex"``) is billed at Batch API rates
#: (developers.openai.com Flex guide, read 2026-10-26); its usage is kept under
#: ``<model>:flex`` and priced with the batch factors.  Gemini's Flex tier too
#: (``"serviceTier": "flex"``, 2026-11-05): input, cached input, storage and output
#: all at half the standard price, as its Batch API (aistudio.google.com/docs/pricing,
#: read 2026-11-05: 3.1 Flash-Lite $0.125 / $0.0125 cached / $0.75 against $0.25 /
#: $0.025 / $1.50).
FLEX_SUFFIX = ":flex"
#: The providers whose ``<model>:flex`` has a price.
_FLEX_PREFIXES = ("gpt-", "gemini-")


# Explicit-cache storage (Gemini, 2026-11-05): USD per 1M cached tokens per hour, standard tier
# (aistudio.google.com/docs/pricing, read that day: 3.1 Flash-Lite $1.00 an hour, Flex and Batch $0.50, the
# "gemini-" batch factor).  A usage record keeps it as token-hours (UsageTotals.cache_storage_token_hours),
# priced at cache_storage_rate of its key.  Only explicit caches bill storage; the implicit ones and the
# other providers' caches bill none.
_CACHE_STORAGE_RATES: tuple[tuple[str, float], ...] = (
    ("gemini-3.1-flash-lite", 1.00),
)


def cache_storage_rate(model: str) -> float:
    """USD per 1M token-hours of explicit-cache storage for a usage key (``:batch`` and ``:flex`` at the batch
    factor).  Raises ``KeyError`` for a model with no rate, so that a run cannot store a cache it prices as
    free."""
    m = model.lower()
    factor = 1.0
    for suffix in (FLEX_SUFFIX, BATCH_SUFFIX):
        if m.endswith(suffix):
            m = m[: -len(suffix)]
            factor = next((f for fragment, f in _BATCH_PRICE_FACTORS if fragment in m), None)
            if factor is None:
                raise KeyError(f"No batch price factor for {model!r} (cache storage)")
    for fragment, rate in _CACHE_STORAGE_RATES:
        if fragment in m:
            return rate * factor
    raise KeyError(f"No cache storage rate for {model!r}.  Add one to "
                   f"assistant_axis/judge_pricing.py:_CACHE_STORAGE_RATES before caching with it.")


# ---------------------------------------------------------------------------
# Prompt caching (2026-11-01)
# ---------------------------------------------------------------------------
#
# A cached prompt token is billed at a multiple of the model's input rate.
# OpenAI's GPT-5.6 and later charge a cache WRITE at 1.25x and a READ at
# 0.1x (developers.openai.com prompt-caching guide, read 2026-11-01: "not an
# additive fee", the written tokens are billed at 1.25x instead of 1x); in
# their default, implicit, mode they write every prompt of 1,024 tokens or
# more, so a judge prompt, which ends in a unique response and is never read
# back, pays the premium for nothing (Roger's dashboard, luna Flex 29 Oct to
# 1 Nov 2026: 1,584M tokens written at $0.125 per 1M = 1.25 x the $0.10 Flex
# rate, 4.9M read; judge.DEFAULT_PROMPT_CACHE turns the writes off).  Earlier
# OpenAI models write free and read at their cached-input rate (gpt-4.1-mini
# $0.10 of $0.40; gpt-5.4-mini $0.075 of $0.75, pricing page 2026-10-27).
# Anthropic writes at 1.25x (the five-minute cache) and reads at 0.1x, only
# when a request sends cache_control, which no judge here does.  The factors
# multiply the rate a usage key resolves to, batch and Flex rates included.
# A model in no row takes (1.0, 1.0): no discount and no premium.
_CACHE_RATE_FACTORS: tuple[tuple[str, float, float], ...] = (
    ("gpt-5.6", 1.25, 0.10),
    ("gpt-6", 1.25, 0.10),
    ("gpt-5.4-mini", 1.0, 0.10),
    ("gpt-4.1-mini", 1.0, 0.25),
    ("claude", 1.25, 0.10),
    ("haiku", 1.25, 0.10),
    ("sonnet", 1.25, 0.10),
    ("opus", 1.25, 0.10),
    ("fable", 1.25, 0.10),
    # Gemini's implicit cache (2026-11-04): no write premium, reads at a tenth (3.1 Flash-Lite $0.025 of $0.25,
    # batch $0.0125 of $0.125; aistudio.google.com/docs/pricing, read that day); a live probe of Flash-Lite
    # read 6,125 of 10,145 prompt tokens of a repeated prefix.  Explicit caches (2026-11-05,
    # judge_tier_cost_eval --gemini_cache explicit) read at the same tenth ($0.0125 on Flex and Batch) and bill
    # storage by the token-hour (cache_storage_rate); their creation is charged as cache-written tokens at the
    # factor 1.0, the tier's input rate, an assumption (not verified against a bill).
    ("gemini", 1.0, 0.10),
)


def cache_rate_factors(model: str) -> tuple[float, float]:
    """``(write_factor, read_factor)``: what a cache-written and a cache-read
    prompt token cost as multiples of ``model``'s input rate
    (:data:`_CACHE_RATE_FACTORS`; a ``:batch`` or ``:flex`` suffix is ignored,
    the factors apply to that rate).  ``(1.0, 1.0)`` for a model in no row."""
    m = model.lower()
    for suffix in (FLEX_SUFFIX, BATCH_SUFFIX):
        if m.endswith(suffix):
            m = m[: -len(suffix)]
    for fragment, write, read in _CACHE_RATE_FACTORS:
        if fragment in m:
            return write, read
    return 1.0, 1.0


def price_for_model(model: str) -> tuple[float, float]:
    """Return ``(input_rate, output_rate)`` in USD per 1M tokens for
    ``model`` (canonical name).

    Substring-match: ``"gpt-4.1-mini-2024-07-18"`` resolves to
    ``gpt-4.1-mini`` rates, ``"claude-haiku-4-5-20251001"`` to
    ``haiku`` rates, etc.

    Raises ``KeyError`` for unrecognised models — callers MUST
    register new models here before judging with them, otherwise
    cost will silently appear free in budget reports.
    """
    m = model.lower()
    if m.endswith(FLEX_SUFFIX):
        if not m.startswith(_FLEX_PREFIXES):
            raise KeyError(f"Flex processing is OpenAI's and Gemini's; no flex price for {model!r}")
        return price_for_model(m[: -len(FLEX_SUFFIX)] + BATCH_SUFFIX)
    if m.endswith(BATCH_SUFFIX):
        rate_in, rate_out = price_for_model(m[: -len(BATCH_SUFFIX)])
        for fragment, factor in _BATCH_PRICE_FACTORS:
            if fragment in m:
                return rate_in * factor, rate_out * factor
        raise KeyError(
            f"No batch price factor for {model!r}.  Add one to "
            f"assistant_axis/judge_pricing.py:_BATCH_PRICE_FACTORS."
        )
    for fragment, rate_in, rate_out in _MODEL_RATES:
        if fragment in m:
            return rate_in, rate_out
    raise KeyError(
        f"Unknown judge model {model!r} for pricing.  Add an entry "
        f"to assistant_axis/judge_pricing.py:_MODEL_RATES (and the "
        f"corresponding constants in plot_batch_size_quality_vs_cost.py)."
    )


def cost_for_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_storage_token_hours: float = 0.0,
) -> float:
    """USD cost of one call's tokens at the model's rates.  ``prompt_tokens``
    is every input token, the cache-written and cache-read ones included;
    those are priced at :func:`cache_rate_factors` times the input rate, the
    rest at the input rate.  ``cache_storage_token_hours`` (an explicit Gemini
    cache's storage, 2026-11-05) is priced at :func:`cache_storage_rate`.
    """
    rate_in, rate_out = price_for_model(model)
    write_factor, read_factor = cache_rate_factors(model)
    plain = prompt_tokens - cache_write_tokens - cache_read_tokens
    storage = cache_storage_token_hours * cache_storage_rate(model) / 1_000_000.0 if cache_storage_token_hours else 0.0
    return (
        (plain + cache_write_tokens * write_factor + cache_read_tokens * read_factor)
        * rate_in / 1_000_000.0
        + completion_tokens * rate_out / 1_000_000.0
        + storage
    )


# ---------------------------------------------------------------------------
# Usage extractors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenUsage:
    """One call's token counts, as the accumulators charge them:
    ``prompt_tokens`` is every input token, the cache-written and cache-read
    ones included (OpenAI's ``prompt_tokens`` counts them in; Anthropic's
    ``input_tokens`` leaves them out, and :func:`anthropic_usage` adds them)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


def _field(obj: Any, *names: str) -> Any:
    """The first of ``names`` that ``obj`` (an SDK object or a dict) carries
    with a value that is not None."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _usage_of(resp: Any) -> Any:
    """The usage block of a response (an SDK object or a Batch API body)."""
    return resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)


def openai_usage(resp: Any) -> TokenUsage:
    """The token counts of an OpenAI chat-completions response: an SDK
    response object or a Batch API result body (a dict).  The cache counts
    come from ``prompt_tokens_details`` (``cached_tokens`` for reads,
    ``cache_write_tokens`` for writes, which the SDK keeps as an extra field),
    or ``input_tokens_details`` (the Responses API).  Zeros when usage is
    missing; never raises (a partial response from a retry)."""
    usage = _usage_of(resp)
    if usage is None:
        return TokenUsage()
    details = _field(usage, "prompt_tokens_details", "input_tokens_details")
    return TokenUsage(
        prompt_tokens=int(_field(usage, "prompt_tokens", "input_tokens") or 0),
        completion_tokens=int(_field(usage, "completion_tokens", "output_tokens") or 0),
        cache_write_tokens=int(_field(details, "cache_write_tokens") or 0) if details is not None else 0,
        cache_read_tokens=int(_field(details, "cached_tokens") or 0) if details is not None else 0,
    )


def anthropic_usage(resp: Any) -> TokenUsage:
    """The token counts of an Anthropic Messages response (an SDK object or a
    Message Batches result dict).  ``input_tokens`` excludes the cached
    tokens, so ``prompt_tokens`` is their sum with
    ``cache_creation_input_tokens`` and ``cache_read_input_tokens``.  Zeros
    when usage is missing; never raises."""
    usage = _usage_of(resp)
    if usage is None:
        return TokenUsage()
    plain = int(_field(usage, "input_tokens", "prompt_tokens") or 0)
    write = int(_field(usage, "cache_creation_input_tokens") or 0)
    read = int(_field(usage, "cache_read_input_tokens") or 0)
    return TokenUsage(
        prompt_tokens=plain + write + read,
        completion_tokens=int(_field(usage, "output_tokens", "completion_tokens") or 0),
        cache_write_tokens=write,
        cache_read_tokens=read,
    )


def gemini_usage(data: Any) -> TokenUsage:
    """The token counts of a Gemini generateContent response (a dict, live or a Batch API result):
    ``promptTokenCount`` counts the cached tokens in, ``cachedContentTokenCount`` is the part read from the
    implicit cache, and the output includes the thought tokens, which Google bills as output.  Zeros when
    usage is missing; never raises."""
    u = (data.get("usageMetadata") if isinstance(data, dict) else None) or {}
    return TokenUsage(
        prompt_tokens=int(u.get("promptTokenCount") or 0),
        completion_tokens=int(u.get("candidatesTokenCount") or 0) + int(u.get("thoughtsTokenCount") or 0),
        cache_read_tokens=int(u.get("cachedContentTokenCount") or 0),
    )


def extract_usage_anthropic(resp: Any) -> tuple[int, int]:
    """``(prompt_tokens, completion_tokens)`` of an Anthropic Messages
    response (:func:`anthropic_usage`; the tuple form of the older callers).
    Tolerates SDK version drift (``input_tokens`` / ``output_tokens``, or the
    legacy ``prompt_tokens`` / ``completion_tokens``) and a missing usage block
    (``(0, 0)``, never raises).
    """
    u = anthropic_usage(resp)
    return u.prompt_tokens, u.completion_tokens


def extract_usage_openai(resp: Any) -> tuple[int, int]:
    """``(prompt_tokens, completion_tokens)`` of an OpenAI chat-completions
    response (:func:`openai_usage`; the tuple form of the older callers).
    """
    u = openai_usage(resp)
    return u.prompt_tokens, u.completion_tokens


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------

@dataclass
class UsageTotals:
    """Running totals for one judge invocation (one
    ``axis_judge_correlation.py`` run = one axis × one cohort).

    Attributes are mutated in place via :meth:`charge`.

    Attributes:
        model: Canonical model name (used for re-deriving the cost).
        prompt_tokens: Sum of input/prompt tokens across all calls, the
            cache-written and cache-read ones included.
        completion_tokens: Sum of output/completion tokens.
        cache_write_tokens: Of the prompt tokens, those written to the
            provider's prompt cache (priced at :func:`cache_rate_factors`
            times the input rate; 2026-11-01).
        cache_read_tokens: Of the prompt tokens, those read from it.
        cache_storage_token_hours: Explicit-cache storage (Gemini, 2026-11-05),
            in token-hours, priced at :func:`cache_storage_rate`.
        cost_usd: Running total cost in USD, computed at charge time
            using :func:`price_for_model` rates.
        n_calls: Number of API calls accumulated (for log lines).
    """
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    n_calls: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cache_storage_token_hours: float = 0.0

    def charge(self, prompt_tokens: int, completion_tokens: int,
               cache_write_tokens: int = 0, cache_read_tokens: int = 0, *,
               cache_storage_token_hours: float = 0.0, n_calls: int = 1) -> float:
        """Add one API call's tokens; return its USD cost contribution.
        ``prompt_tokens`` includes the cache-written and cache-read tokens.
        ``n_calls=0`` for a charge that is not a judge call (an explicit cache's
        creation or storage, 2026-11-05)."""
        delta_cost = cost_for_usage(self.model, int(prompt_tokens), int(completion_tokens),
                                    int(cache_write_tokens), int(cache_read_tokens),
                                    float(cache_storage_token_hours))
        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)
        self.cache_write_tokens += int(cache_write_tokens)
        self.cache_read_tokens += int(cache_read_tokens)
        self.cache_storage_token_hours += float(cache_storage_token_hours)
        self.cost_usd += delta_cost
        self.n_calls += int(n_calls)
        return delta_cost

    def as_dict(self) -> dict:
        """Serialisable form for ``usage.json`` side-car / provenance notes.
        The cache counts are written only when they are not zero, so a record
        of a run that cached nothing reads as before."""
        out = {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "n_calls": self.n_calls,
        }
        if self.cache_write_tokens:
            out["cache_write_tokens"] = self.cache_write_tokens
        if self.cache_read_tokens:
            out["cache_read_tokens"] = self.cache_read_tokens
        if self.cache_storage_token_hours:
            out["cache_storage_token_hours"] = round(self.cache_storage_token_hours, 4)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "UsageTotals":
        """Inverse of :meth:`as_dict`.  Tolerates missing fields (rounded
        ``cost_usd`` is recomputed precisely from the token counts).
        """
        model = str(d.get("model") or "")
        if not model:
            raise ValueError("UsageTotals.from_dict: missing 'model' field")
        pt = int(d.get("prompt_tokens") or 0)
        ct = int(d.get("completion_tokens") or 0)
        n  = int(d.get("n_calls") or 0)
        cw = int(d.get("cache_write_tokens") or 0)
        cr = int(d.get("cache_read_tokens") or 0)
        sh = float(d.get("cache_storage_token_hours") or 0.0)
        # Recompute cost from tokens (the stored value is rounded; we
        # keep full precision in memory).
        try:
            cost = cost_for_usage(model, pt, ct, cw, cr, sh)
        except KeyError:
            # Unknown model: fall back to the persisted cost so we
            # don't silently zero out historic totals.
            cost = float(d.get("cost_usd") or 0.0)
        return cls(model=model, prompt_tokens=pt, completion_tokens=ct,
                   cost_usd=cost, n_calls=n, cache_write_tokens=cw, cache_read_tokens=cr,
                   cache_storage_token_hours=sh)


# ---------------------------------------------------------------------------
# Multi-model accumulator (for workloads that hit multiple judges per item,
# e.g. steering's bidirectional GPT+Haiku effect ensemble).
# ---------------------------------------------------------------------------

@dataclass
class MultiModelUsage:
    """Per-model usage accumulator for mixed-judge workloads.

    Wraps ``dict[model_name, UsageTotals]`` with convenient
    aggregation helpers, JSON round-trip, and merge-on-load
    semantics for resume-friendly persistence.

    Use this on any automated LLM call site that may issue calls to
    more than one model, OR is run repeatedly enough that an aggregate
    cost report is useful (project rule -- see AGENT_NOTES "Token
    usage logging is mandatory on batched LLM call sites").

    Example::

        usage = MultiModelUsage()
        # ... in your call site, after each successful API response:
        usage.charge("gpt-4.1-mini", prompt_tokens, completion_tokens)
        usage.charge("claude-haiku-4-5", prompt_tokens, completion_tokens)
        # End-of-run:
        usage.write_json(output_dir / "usage.json")
    """
    per_model: dict[str, UsageTotals] = field(default_factory=dict)

    def charge(
        self, model: str, prompt_tokens: int, completion_tokens: int,
        cache_write_tokens: int = 0, cache_read_tokens: int = 0, *,
        cache_storage_token_hours: float = 0.0, n_calls: int = 1,
    ) -> float:
        """Tick the (model) sub-accumulator.  Returns the USD cost
        contribution of this single call (sum across all token streams).
        The cache counts are the part of ``prompt_tokens`` written to or
        read from the provider's prompt cache (:class:`TokenUsage`).

        The model entry is auto-created on first use; subsequent calls
        with the same name accumulate.  Never raises (no budget cap).
        """
        if model not in self.per_model:
            self.per_model[model] = UsageTotals(model=model)
        return self.per_model[model].charge(prompt_tokens, completion_tokens,
                                            cache_write_tokens, cache_read_tokens,
                                            cache_storage_token_hours=cache_storage_token_hours,
                                            n_calls=n_calls)

    def charge_usage(self, model: str, usage: TokenUsage) -> float:
        """:meth:`charge` from a :class:`TokenUsage`."""
        return self.charge(model, usage.prompt_tokens, usage.completion_tokens,
                           usage.cache_write_tokens, usage.cache_read_tokens)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.per_model.values())

    @property
    def total_prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.per_model.values())

    @property
    def total_completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self.per_model.values())

    @property
    def n_calls(self) -> int:
        return sum(t.n_calls for t in self.per_model.values())

    def as_dict(self) -> dict:
        """Serialisable form: per-model breakdown plus aggregate totals.

        Aggregate totals are written for human convenience; they are
        recomputed from ``per_model`` on :meth:`from_dict` rather than
        trusted as the source of truth.
        """
        return {
            "per_model": {m: t.as_dict() for m, t in sorted(self.per_model.items())},
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "n_calls": self.n_calls,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MultiModelUsage":
        """Inverse of :meth:`as_dict`.  Ignores aggregate totals (they
        are recomputed from per-model entries on read)."""
        per_model = {}
        for m, sub in (d.get("per_model") or {}).items():
            per_model[m] = UsageTotals.from_dict(sub)
        return cls(per_model=per_model)

    def merge_from(self, other: "MultiModelUsage") -> None:
        """In-place merge: add ``other``'s per-model totals into self.

        Used by resume paths (load existing usage.json, then continue
        ticking the live tracker).  Token counts are summed; cost is
        recomputed from tokens to absorb any pricing-table changes
        between runs (running-total cost is then accurate at current
        rates, not historical rates).
        """
        for model, sub in other.per_model.items():
            if model not in self.per_model:
                self.per_model[model] = UsageTotals(model=model)
            tgt = self.per_model[model]
            tgt.prompt_tokens += sub.prompt_tokens
            tgt.completion_tokens += sub.completion_tokens
            tgt.cache_write_tokens += sub.cache_write_tokens
            tgt.cache_read_tokens += sub.cache_read_tokens
            tgt.cache_storage_token_hours += sub.cache_storage_token_hours
            tgt.n_calls += sub.n_calls
            # Recompute cost from totals at current rates.
            try:
                tgt.cost_usd = cost_for_usage(model, tgt.prompt_tokens, tgt.completion_tokens,
                                              tgt.cache_write_tokens, tgt.cache_read_tokens,
                                              tgt.cache_storage_token_hours)
            except KeyError:
                # Unknown model: keep sum-of-stored-cost as best-effort.
                tgt.cost_usd += sub.cost_usd

    def log_line(self, prefix: str = "[usage]") -> str:
        """One-line human-readable summary suitable for end-of-run logs
        (the cache counts shown when they are not zero)."""
        if not self.per_model:
            return f"{prefix} (no API calls)"
        per = "; ".join(
            f"{m}: ${t.cost_usd:.3f} in={t.prompt_tokens:,} out={t.completion_tokens:,} n={t.n_calls:,}"
            + (f" cache_write={t.cache_write_tokens:,}" if t.cache_write_tokens else "")
            + (f" cache_read={t.cache_read_tokens:,}" if t.cache_read_tokens else "")
            + (f" cache_storage={t.cache_storage_token_hours:,.0f} token-h" if t.cache_storage_token_hours else "")
            for m, t in sorted(self.per_model.items())
        )
        return (
            f"{prefix} total=${self.total_cost_usd:.3f} "
            f"calls={self.n_calls:,} [{per}]"
        )

    # -------- File I/O helpers (idempotent, resume-friendly) --------

    @classmethod
    def load_or_create(cls, path) -> "MultiModelUsage":
        """Read an existing ``usage.json`` if it exists, else return an
        empty tracker.  ``path`` may be a str or Path.

        The file may be a bare :meth:`as_dict` payload or a
        ``json_metadata``-wrapped envelope (we look in both shapes).
        """
        import json
        from pathlib import Path as _P
        p = _P(path)
        if not p.is_file():
            return cls()
        try:
            obj = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            # Callers merge this into the record they write back, so an
            # empty tracker here resets the cumulative history: say so
            # (before 2026-10-12 this was silent).
            logger.warning(f"usage record {p} is unreadable ({type(e).__name__}: {e}); "
                           f"starting from zero -- the file will be overwritten")
            return cls()
        if not isinstance(obj, dict):
            logger.warning(f"usage record {p} is not a JSON object; starting from zero")
            return cls()
        # Top-level per_model: the bare as_dict() form, or as_dict() with a
        # _provenance key beside it.
        if "per_model" in obj:
            return cls.from_dict(obj)
        # Wrapped forms: "result" is json_metadata's envelope (the shape the
        # docstring promises; missing before 2026-10-12), the others are
        # older ad-hoc wrappers.
        for k in ("result", "payload", "_payload", "usage", "data"):
            sub = obj.get(k)
            if isinstance(sub, dict) and "per_model" in sub:
                return cls.from_dict(sub)
        if obj:
            logger.warning(f"usage record {p} has no per_model block; starting from zero")
        return cls()

    def write_json(self, path) -> None:
        """Write the tracker to ``path`` (str or Path), creating
        parent dirs as needed.  Overwrites any existing file.
        """
        import json
        from pathlib import Path as _P
        p = _P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))


class BudgetExceededError(Exception):
    """Raised by :meth:`BudgetTracker.charge` when the running cost
    crosses the configured ``budget_usd`` cap.

    The orchestrator (``results_analysis/axis_judge_correlation.py``
    :func:`run`) catches this, flushes any in-progress caches, logs a
    summary, and exits with code 2 (= cap exceeded; distinct from
    exit code 1 for command-line errors).
    """

    def __init__(self, totals: UsageTotals, budget_usd: float) -> None:
        self.totals = totals
        self.budget_usd = budget_usd
        super().__init__(
            f"Budget cap exceeded: ${totals.cost_usd:.2f} > ${budget_usd:.2f} "
            f"(prompt={totals.prompt_tokens}, completion="
            f"{totals.completion_tokens}, n_calls={totals.n_calls}). "
            f"Exiting; partial results have been flushed to disk."
        )


@dataclass
class BudgetTracker:
    """Cost-cap-enforcing accumulator over judge API calls.

    Wraps a :class:`UsageTotals` with an optional hard cap
    (:attr:`budget_usd`) and an optional expected-cost reference
    (:attr:`expected_cost_usd`) used for the
    actual/expected-ratio reporting in canary deployments.

    Attributes:
        totals: The :class:`UsageTotals` accumulator.
        budget_usd: Hard cap.  When set and exceeded, :meth:`charge`
            raises :class:`BudgetExceededError`.  ``None`` disables
            the cap (legacy behaviour).
        expected_cost_usd: Ratio reference (canary deployments
            compute ``actual / expected`` against this).  Advisory
            only; doesn't affect cap enforcement.
    """
    totals: UsageTotals
    budget_usd: Optional[float] = None
    expected_cost_usd: Optional[float] = None
    # Internal: log-line throttling.
    _last_logged_cost: float = field(default=0.0, repr=False)

    def charge(self, prompt_tokens: int, completion_tokens: int,
               cache_write_tokens: int = 0, cache_read_tokens: int = 0) -> float:
        """Tick the underlying totals; raise if the cap is crossed.

        Returns the USD cost contribution for this single call.
        """
        delta = self.totals.charge(prompt_tokens, completion_tokens, cache_write_tokens, cache_read_tokens)
        if self.budget_usd is not None and self.totals.cost_usd > self.budget_usd:
            raise BudgetExceededError(self.totals, self.budget_usd)
        return delta

    def ratio(self) -> Optional[float]:
        """``actual / expected`` ratio when ``expected_cost_usd`` is set,
        else ``None``.  Used for canary GREEN/AMBER/RED gating."""
        if not self.expected_cost_usd:
            return None
        if self.expected_cost_usd <= 0:
            return None
        return self.totals.cost_usd / self.expected_cost_usd

    def log_line(self, prefix: str = "[budget]") -> str:
        """One-line human-readable status (suitable for periodic logs)."""
        parts = [f"cost=${self.totals.cost_usd:.2f}"]
        if self.budget_usd is not None:
            pct = (self.totals.cost_usd / self.budget_usd * 100.0
                   if self.budget_usd > 0 else float("inf"))
            parts.append(f"cap=${self.budget_usd:.2f} ({pct:.0f}%)")
        if self.expected_cost_usd:
            ratio = self.ratio()
            parts.append(
                f"expected=${self.expected_cost_usd:.2f} "
                f"(actual/expected={ratio:.2f})"
            )
        parts.append(
            f"tokens=in:{self.totals.prompt_tokens:,}/"
            f"out:{self.totals.completion_tokens:,}"
        )
        parts.append(f"calls={self.totals.n_calls:,}")
        return f"{prefix} " + " ".join(parts)

    def maybe_log(self, logger_obj: logging.Logger,
                  threshold_usd: float = 0.10) -> None:
        """Emit a ``[budget]`` log line iff the running cost has
        increased by ``threshold_usd`` since the last log call.

        Cheap to call from a tight loop; only the periodic save_every
        path actually emits a line.
        """
        if self.totals.cost_usd - self._last_logged_cost >= threshold_usd:
            logger_obj.info(self.log_line())
            self._last_logged_cost = self.totals.cost_usd

    def as_dict(self) -> dict:
        """Serialisable form for ``usage.json`` and provenance notes."""
        out: dict[str, Any] = {**self.totals.as_dict()}
        if self.budget_usd is not None:
            out["budget_usd"] = round(self.budget_usd, 2)
        if self.expected_cost_usd is not None:
            out["expected_cost_usd"] = round(self.expected_cost_usd, 2)
            ratio = self.ratio()
            if ratio is not None:
                out["actual_over_expected"] = round(ratio, 3)
        return out


__all__ = [
    "GPT_MINI_RATE_IN",
    "GPT_MINI_RATE_OUT",
    "HAIKU_RATE_IN",
    "HAIKU_RATE_OUT",
    "SONNET_RATE_IN",
    "SONNET_RATE_OUT",
    "ROLES_RESPONSE_LENGTH_FACTOR",
    "price_for_model",
    "cache_rate_factors",
    "cache_storage_rate",
    "cost_for_usage",
    "surgical_rejudge_cost_split",
    "TokenUsage",
    "openai_usage",
    "anthropic_usage",
    "extract_usage_anthropic",
    "extract_usage_openai",
    "UsageTotals",
    "MultiModelUsage",
    "BudgetTracker",
    "BudgetExceededError",
]
