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
numbers are documented in README.md "Judging cost model".

results_analysis/plot_batch_size_quality_vs_cost.py has its OWN copy of
the legacy (gpt-4.1-mini / claude-sonnet-4) rates, deliberately NOT kept
in sync with this module -- that script regenerates a chart from cached
usage data that was actually billed at those historical rates, so its
constants are pinned to history rather than updated. This module is the
one to update when adding a new judge model or refreshing current prices;
see the 2026-09 update note below.

2026-09 update: Anthropic rates verified live via the claude-api skill's
current-models table (Haiku 4.5 unchanged; Sonnet 5 and Opus 5 added --
note Sonnet 5 is CHEAPER than Sonnet 4 was, and Opus 5 is far cheaper than
this module's original May-2026 placeholder guess of $15/$75). OpenAI
GPT-5.6 tier rates came from web search cross-confirming two independent
queries against developers.openai.com model pages -- lower confidence
than the Anthropic numbers (could not fetch developers.openai.com
directly; it's a client-rendered app that returns an empty shell to non-
JS fetchers). Re-verify at https://developers.openai.com/api/docs/pricing
before trusting for a real budget-tracked run, particularly gpt-5.6-sol,
whose rate is explicitly promotional through 2026-11-21 per the source.

Note on caching: gpt-4.1-mini / gpt-4-class models support prompt
caching (~5x cheaper for the cached prefix portion).  We do NOT
account for cache hits here — running totals will slightly
overestimate cost for cache-friendly workloads, which is
conservative-by-default for a budget cap.
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

# --- Legacy Anthropic rate, kept for pricing pre-existing cached runs ---
# (e.g. roger/axis_judge_experiments/angel_vs_demon/sonnet/ actually used
# claude-sonnet-4-20250514 at this rate; do not use for new judging).
SONNET4_RATE_IN = 3.00    # claude-sonnet-4 / claude-sonnet-4-6 (unchanged through 4.6)
SONNET4_RATE_OUT = 15.00

# --- Current rates (verified 2026-09; see module docstring) ---
HAIKU_RATE_IN = 1.00      # claude-haiku-4-5 -- unchanged since May 2026
HAIKU_RATE_OUT = 5.00
SONNET_RATE_IN = 2.00     # claude-sonnet-5 -- price CUT vs sonnet-4's $3.00/$15.00
SONNET_RATE_OUT = 10.00
OPUS_RATE_IN = 5.00       # claude-opus-5 (same rate as opus-4-6/4-7/4-8)
OPUS_RATE_OUT = 25.00     # supersedes the never-verified $15/$75 May-2026 placeholder

# --- OpenAI GPT-5.6 tier ladder (Sol/Terra/Luna), short-context rates ---
# Lower-confidence sourcing than the Anthropic rates above -- see module
# docstring. gpt-5.6-sol's rate is promotional through 2026-11-21.
GPT_LUNA_RATE_IN = 0.20   # gpt-5.6-luna -- budget tier (~matches Haiku 4.5's cost profile)
GPT_LUNA_RATE_OUT = 1.20
GPT_TERRA_RATE_IN = 2.00  # gpt-5.6-terra -- mid tier (~matches Sonnet 5)
GPT_TERRA_RATE_OUT = 12.00
GPT_SOL_RATE_IN = 5.00    # gpt-5.6-sol -- flagship tier (~matches Opus 5); promotional pricing
GPT_SOL_RATE_OUT = 30.00


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
# "gpt-4.1-mini-2024" still resolves to GPT_MINI rates), checked in
# order -- entries here are deliberately ordered MOST-specific first,
# so an exact dated/versioned model id resolves to its own generation's
# rate rather than falling through to a newer generic fallback that
# would silently misprice it (e.g. "claude-sonnet-4-20250514" must hit
# the "sonnet-4" entry, not the generic "sonnet" -> current-rate one).
_MODEL_RATES: tuple[tuple[str, float, float], ...] = (
    # OpenAI
    ("gpt-4.1-mini", GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT),
    ("gpt-4o-mini", GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT),
    ("gpt-5.6-luna", GPT_LUNA_RATE_IN, GPT_LUNA_RATE_OUT),
    ("gpt-5.6-terra", GPT_TERRA_RATE_IN, GPT_TERRA_RATE_OUT),
    ("gpt-5.6-sol", GPT_SOL_RATE_IN, GPT_SOL_RATE_OUT),
    # Anthropic -- versioned entries before the generic "sonnet"/"opus"
    # fallbacks (see ordering note above).
    ("sonnet-4-6", SONNET4_RATE_IN, SONNET4_RATE_OUT),
    ("sonnet-4", SONNET4_RATE_IN, SONNET4_RATE_OUT),
    ("sonnet-5", SONNET_RATE_IN, SONNET_RATE_OUT),
    ("opus-5", OPUS_RATE_IN, OPUS_RATE_OUT),
    ("haiku", HAIKU_RATE_IN, HAIKU_RATE_OUT),
    ("sonnet", SONNET_RATE_IN, SONNET_RATE_OUT),  # generic fallback -> current (5) rate
    ("opus", OPUS_RATE_IN, OPUS_RATE_OUT),        # generic fallback -> current (5) rate; also correct for 4-6/4-7/4-8 (same price)
)


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
    for fragment, rate_in, rate_out in _MODEL_RATES:
        if fragment in m:
            return rate_in, rate_out
    raise KeyError(
        f"Unknown judge model {model!r} for pricing.  Add an entry "
        f"to assistant_axis/judge_pricing.py:_MODEL_RATES -- this is the "
        f"one module to update (results_analysis/plot_batch_size_quality_vs_cost.py "
        f"has its own copy deliberately pinned to the historical rates its "
        f"archived chart data was actually billed at; see that file's comment)."
    )


def cost_for_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """USD cost of a single (prompt, completion) token pair at the
    model's rates.  Convenience wrapper around :func:`price_for_model`.
    """
    rate_in, rate_out = price_for_model(model)
    return (
        prompt_tokens * rate_in / 1_000_000.0
        + completion_tokens * rate_out / 1_000_000.0
    )


# ---------------------------------------------------------------------------
# Usage extractors
# ---------------------------------------------------------------------------

def extract_usage_anthropic(resp: Any) -> tuple[int, int]:
    """Return ``(prompt_tokens, completion_tokens)`` from an Anthropic
    Messages response.  Tolerates SDK version drift: the canonical
    field is ``resp.usage`` with ``input_tokens`` / ``output_tokens``,
    but legacy SDKs used ``prompt_tokens`` / ``completion_tokens``.

    Returns ``(0, 0)`` when usage is missing entirely — never raises
    (keeps the call path resilient to partial response objects from
    error retries).
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    in_tokens = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", None)
        or 0
    )
    out_tokens = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None)
        or 0
    )
    return int(in_tokens), int(out_tokens)


def extract_usage_openai(resp: Any) -> tuple[int, int]:
    """Return ``(prompt_tokens, completion_tokens)`` from an OpenAI
    chat-completions response.  Same SDK-drift tolerance as the
    Anthropic helper.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    in_tokens = (
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
        or 0
    )
    out_tokens = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
        or 0
    )
    return int(in_tokens), int(out_tokens)


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
        prompt_tokens: Sum of input/prompt tokens across all calls.
        completion_tokens: Sum of output/completion tokens.
        cost_usd: Running total cost in USD, computed at charge time
            using :func:`price_for_model` rates.
        n_calls: Number of API calls accumulated (for log lines).
    """
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    n_calls: int = 0

    def charge(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Add one API call's tokens; return its USD cost contribution."""
        rate_in, rate_out = price_for_model(self.model)
        delta_cost = (
            prompt_tokens * rate_in / 1_000_000.0
            + completion_tokens * rate_out / 1_000_000.0
        )
        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)
        self.cost_usd += delta_cost
        self.n_calls += 1
        return delta_cost

    def as_dict(self) -> dict:
        """Serialisable form for ``usage.json`` side-car / provenance notes."""
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "n_calls": self.n_calls,
        }

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
        # Recompute cost from tokens (the stored value is rounded; we
        # keep full precision in memory).
        try:
            cost = cost_for_usage(model, pt, ct)
        except KeyError:
            # Unknown model: fall back to the persisted cost so we
            # don't silently zero out historic totals.
            cost = float(d.get("cost_usd") or 0.0)
        return cls(model=model, prompt_tokens=pt, completion_tokens=ct,
                   cost_usd=cost, n_calls=n)


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
    ) -> float:
        """Tick the (model) sub-accumulator.  Returns the USD cost
        contribution of this single call (sum across all token streams).

        The model entry is auto-created on first use; subsequent calls
        with the same name accumulate.  Never raises (no budget cap).
        """
        if model not in self.per_model:
            self.per_model[model] = UsageTotals(model=model)
        return self.per_model[model].charge(prompt_tokens, completion_tokens)

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
            tgt.n_calls += sub.n_calls
            # Recompute cost from totals at current rates.
            try:
                rate_in, rate_out = price_for_model(model)
                tgt.cost_usd = (
                    tgt.prompt_tokens * rate_in / 1_000_000.0
                    + tgt.completion_tokens * rate_out / 1_000_000.0
                )
            except KeyError:
                # Unknown model: keep sum-of-stored-cost as best-effort.
                tgt.cost_usd += sub.cost_usd

    def log_line(self, prefix: str = "[usage]") -> str:
        """One-line human-readable summary suitable for end-of-run logs."""
        if not self.per_model:
            return f"{prefix} (no API calls)"
        per = "; ".join(
            f"{m}: ${t.cost_usd:.3f} in={t.prompt_tokens:,} out={t.completion_tokens:,} n={t.n_calls:,}"
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
        except Exception:
            return cls()
        # Envelope-wrapped form (json_metadata) keeps the payload under
        # the same top-level keys; we just look for "per_model" first.
        if "per_model" in obj:
            return cls.from_dict(obj)
        # If wrapped, the payload may be under "_payload" or similar;
        # try common shapes before giving up.
        for k in ("payload", "_payload", "usage", "data"):
            sub = obj.get(k)
            if isinstance(sub, dict) and "per_model" in sub:
                return cls.from_dict(sub)
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

    def charge(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Tick the underlying totals; raise if the cap is crossed.

        Returns the USD cost contribution for this single call.
        """
        delta = self.totals.charge(prompt_tokens, completion_tokens)
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
    "cost_for_usage",
    "surgical_rejudge_cost_split",
    "extract_usage_anthropic",
    "extract_usage_openai",
    "UsageTotals",
    "MultiModelUsage",
    "BudgetTracker",
    "BudgetExceededError",
]
