"""Tests for ``assistant_axis.judge_pricing``.

The most important property is that :class:`BudgetTracker` raises
:class:`BudgetExceededError` IMMEDIATELY when the cap is crossed —
not at the next save_every chunk, not at end-of-run.  Otherwise a
runaway cost surge can blow past the cap by orders of magnitude
before the orchestrator notices.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from assistant_axis.judge_pricing import (
    BudgetExceededError,
    BudgetTracker,
    GPT_MINI_RATE_IN,
    GPT_MINI_RATE_OUT,
    HAIKU_RATE_IN,
    HAIKU_RATE_OUT,
    MultiModelUsage,
    ROLES_RESPONSE_LENGTH_FACTOR,
    SONNET_RATE_IN,
    SONNET_RATE_OUT,
    UsageTotals,
    cache_rate_factors,
    cost_for_usage,
    extract_usage_anthropic,
    gemini_usage,
    extract_usage_openai,
    price_for_model,
    surgical_rejudge_cost_split,
)


# ---------------------------------------------------------------------------
# price_for_model
# ---------------------------------------------------------------------------

class TestPriceForModel:
    def test_gpt_mini_canonical(self):
        assert price_for_model("gpt-4.1-mini") == (
            GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT,
        )

    def test_gpt_mini_with_date_suffix(self):
        """Substring match: dated model snapshots resolve to base rates."""
        assert price_for_model("gpt-4.1-mini-2024-07-18") == (
            GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT,
        )

    def test_gpt_4o_mini_alias(self):
        """4o-mini shares the gpt-4.1-mini rate sheet — both should resolve."""
        assert price_for_model("gpt-4o-mini") == (
            GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT,
        )

    def test_haiku_canonical(self):
        assert price_for_model("claude-haiku-4-5-20251001") == (
            HAIKU_RATE_IN, HAIKU_RATE_OUT,
        )

    def test_sonnet_canonical(self):
        assert price_for_model("claude-sonnet-4-20250514") == (
            SONNET_RATE_IN, SONNET_RATE_OUT,
        )

    def test_sonnet_4_6(self):
        """The trait-gap filter's second-opinion model."""
        assert price_for_model("claude-sonnet-4-6") == (SONNET_RATE_IN, SONNET_RATE_OUT)

    def test_fable_5_1(self):
        """The M3 overlap test's fourth judge (2026-10-04): $10 / $50, batch at half."""
        assert price_for_model("claude-fable-5-1") == (10.00, 50.00)
        assert price_for_model("claude-fable-5-1:batch") == (5.00, 25.00)

    def test_haiku_5_5(self):
        """Haiku 5.5 (2026-10-07): $0.10 / $0.50, batch at half; not priced at Haiku 4.5's rates by the
        generic "haiku" row, which must come after it."""
        assert price_for_model("claude-haiku-5-5") == (0.10, 0.50)
        assert price_for_model("CLAUDE-HAIKU-5-5") == (0.10, 0.50)
        assert price_for_model("claude-haiku-5-5:batch") == (0.05, 0.25)
        assert cost_for_usage("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
        # Haiku 4.5 keeps its own rates
        assert price_for_model("claude-haiku-4-5-20251001") == (HAIKU_RATE_IN, HAIKU_RATE_OUT)
        assert price_for_model("claude-haiku-4-5") == (HAIKU_RATE_IN, HAIKU_RATE_OUT)

    def test_gpt_5_4_mini_and_gpt_5_6_luna(self):
        """The candidates to replace gpt-4.1-mini (2026-10-27): batch and Flex at half."""
        assert price_for_model("gpt-5.4-mini") == (0.75, 4.50)
        assert price_for_model("gpt-5.4-mini-2026-03-17") == (0.75, 4.50)
        assert price_for_model("gpt-5.4-mini:batch") == (0.375, 2.25)
        assert price_for_model("gpt-5.6-luna") == (0.20, 1.20)
        assert price_for_model("gpt-5.6-luna:flex") == (0.10, 0.60)

    def test_haiku_5_5_usage_records(self):
        """A usage record of both Haikus prices each at its own rates (and a merge recomputes them so)."""
        u = MultiModelUsage()
        u.charge("claude-haiku-5-5", 1_000_000, 100_000)            # $0.10 + $0.05
        u.charge("claude-haiku-4-5-20251001", 1_000_000, 100_000)   # $1.00 + $0.50
        assert u.per_model["claude-haiku-5-5"].cost_usd == pytest.approx(0.15)
        assert u.per_model["claude-haiku-4-5-20251001"].cost_usd == pytest.approx(1.50)
        again = MultiModelUsage.from_dict(u.as_dict())
        assert again.total_cost_usd == pytest.approx(1.65)

    @pytest.mark.parametrize("model,rates", [
        ("text-embedding-3-large", (0.13, 0.0)),
        ("text-embedding-3-small", (0.02, 0.0)),
        ("Qwen/Qwen3-Embedding-0.6B", (0.0, 0.0)),
        ("qwen3-embedding-0.6b", (0.0, 0.0)),
        ("BAAI/bge-large-en-v1.5", (0.0, 0.0)),
        ("google/embeddinggemma-300m", (0.0, 0.0)),
    ])
    def test_embedding_models(self, model, rates):
        """Embedding calls are counted in usage.json (trait-gap platform)."""
        assert price_for_model(model) == rates

    def test_embedding_cost_is_input_only(self):
        assert cost_for_usage("text-embedding-3-large", 1_000_000, 0) == pytest.approx(0.13)

    def test_case_insensitive(self):
        assert price_for_model("GPT-4.1-Mini") == (
            GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT,
        )

    def test_unknown_model_raises_keyerror(self):
        """Unknown models MUST raise — silently treating as $0/token
        would corrupt budget reports.  Loud-fail keeps the operator
        in the loop."""
        with pytest.raises(KeyError) as exc:
            price_for_model("ada-001")
        assert "ada-001" in str(exc.value)
        assert "judge_pricing.py" in str(exc.value)


# ---------------------------------------------------------------------------
# cost_for_usage
# ---------------------------------------------------------------------------

class TestCostForUsage:
    def test_zero_tokens_is_zero_cost(self):
        assert cost_for_usage("gpt-4.1-mini", 0, 0) == 0.0

    def test_known_arithmetic_gpt_mini(self):
        """1M input + 1M output @ $0.40 + $1.60 = $2.00."""
        cost = cost_for_usage("gpt-4.1-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(2.00, rel=1e-9)

    def test_known_arithmetic_haiku(self):
        """1M input + 1M output @ $1 + $5 = $6."""
        cost = cost_for_usage("claude-haiku-4-5-20251001",
                              1_000_000, 1_000_000)
        assert cost == pytest.approx(6.00, rel=1e-9)


# ---------------------------------------------------------------------------
# extract_usage_anthropic / extract_usage_openai
# ---------------------------------------------------------------------------

class TestExtractUsage:
    def test_anthropic_modern_field_names(self):
        """Modern Anthropic SDK: ``input_tokens`` / ``output_tokens``."""
        resp = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )
        assert extract_usage_anthropic(resp) == (100, 20)

    def test_anthropic_legacy_field_names(self):
        """Older SDKs used ``prompt_tokens`` / ``completion_tokens``."""
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10),
        )
        assert extract_usage_anthropic(resp) == (50, 10)

    def test_anthropic_missing_usage(self):
        """Partial response from a retry — must not raise."""
        resp = SimpleNamespace()
        assert extract_usage_anthropic(resp) == (0, 0)

    def test_anthropic_usage_none(self):
        resp = SimpleNamespace(usage=None)
        assert extract_usage_anthropic(resp) == (0, 0)

    def test_openai_modern_field_names(self):
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=40),
        )
        assert extract_usage_openai(resp) == (200, 40)

    def test_openai_input_output_aliases(self):
        """Some OpenAI SDK paths expose ``input_tokens`` / ``output_tokens``."""
        resp = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=200, output_tokens=40),
        )
        assert extract_usage_openai(resp) == (200, 40)

    def test_openai_missing_usage(self):
        resp = SimpleNamespace()
        assert extract_usage_openai(resp) == (0, 0)


# ---------------------------------------------------------------------------
# UsageTotals
# ---------------------------------------------------------------------------

class TestUsageTotals:
    def test_initial_zero(self):
        u = UsageTotals(model="gpt-4.1-mini")
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.cost_usd == 0.0
        assert u.n_calls == 0

    def test_charge_accumulates(self):
        u = UsageTotals(model="gpt-4.1-mini")
        u.charge(1_000_000, 1_000_000)
        assert u.prompt_tokens == 1_000_000
        assert u.completion_tokens == 1_000_000
        assert u.cost_usd == pytest.approx(2.00, rel=1e-9)
        assert u.n_calls == 1

    def test_charge_returns_delta_cost(self):
        u = UsageTotals(model="gpt-4.1-mini")
        delta = u.charge(1_000, 0)
        assert delta == pytest.approx(0.0004, rel=1e-9)

    def test_multiple_charges_accumulate(self):
        u = UsageTotals(model="claude-haiku-4-5-20251001")
        u.charge(100, 20)
        u.charge(200, 40)
        u.charge(50, 10)
        assert u.n_calls == 3
        assert u.prompt_tokens == 350
        assert u.completion_tokens == 70

    def test_as_dict_serialisable(self):
        u = UsageTotals(model="gpt-4.1-mini")
        u.charge(1_000_000, 1_000_000)
        d = u.as_dict()
        assert d == {
            "model": "gpt-4.1-mini",
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "cost_usd": 2.0,
            "n_calls": 1,
        }


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------

class TestBudgetTracker:
    def test_no_cap_does_not_raise(self):
        """Without budget_usd, charge() never raises (legacy behaviour)."""
        tracker = BudgetTracker(
            totals=UsageTotals(model="gpt-4.1-mini"),
        )
        for _ in range(100):
            tracker.charge(10_000, 10_000)
        # Still alive; cost is huge but no cap.
        assert tracker.totals.cost_usd > 0

    def test_cap_not_yet_reached(self):
        tracker = BudgetTracker(
            totals=UsageTotals(model="gpt-4.1-mini"),
            budget_usd=10.00,
        )
        # 1k+1k tokens = $0.002 — well under $10.
        tracker.charge(1_000, 1_000)
        assert tracker.totals.cost_usd < 10.00

    def test_cap_exceeded_raises_immediately(self):
        """The cap MUST trigger on the first call that crosses it,
        not at end-of-loop."""
        tracker = BudgetTracker(
            totals=UsageTotals(model="gpt-4.1-mini"),
            budget_usd=1.00,
        )
        # First call: 1M+0 = $0.40 — under cap.
        tracker.charge(1_000_000, 0)
        # Second call: pushes total to $0.80 — under cap.
        tracker.charge(1_000_000, 0)
        # Third call: pushes total to $1.20 > $1.00 → must raise NOW.
        with pytest.raises(BudgetExceededError) as exc:
            tracker.charge(1_000_000, 0)
        assert exc.value.budget_usd == 1.00
        assert exc.value.totals.cost_usd > 1.00
        # The totals object accumulated *this* charge (it's already
        # spent by the time we know about the cap), so n_calls should
        # be 3 — caller knows exactly how many calls happened.
        assert exc.value.totals.n_calls == 3

    def test_cap_exceeded_error_message_actionable(self):
        tracker = BudgetTracker(
            totals=UsageTotals(model="gpt-4.1-mini"),
            budget_usd=0.001,
        )
        with pytest.raises(BudgetExceededError) as exc:
            tracker.charge(10_000_000, 0)
        msg = str(exc.value)
        assert "$" in msg  # has dollar sign(s)
        assert "exceeded" in msg.lower()
        # Should reference both the actual and the cap amount (so the
        # operator can see how far over they went without re-deriving).
        assert "$0.00" in msg or "0.00" in msg  # cap value
        # And surface the call count for forensic correlation.
        assert "n_calls" in msg.lower() or "calls" in msg.lower()

    def test_ratio_actual_over_expected(self):
        tracker = BudgetTracker(
            totals=UsageTotals(model="gpt-4.1-mini"),
            expected_cost_usd=10.00,
        )
        # Spend $5 → ratio = 0.5
        tracker.charge(12_500_000, 0)  # 12.5M * $0.40 / 1M = $5
        assert tracker.ratio() == pytest.approx(0.5, rel=1e-6)

    def test_ratio_none_when_expected_unset(self):
        tracker = BudgetTracker(totals=UsageTotals(model="gpt-4.1-mini"))
        assert tracker.ratio() is None

    def test_ratio_none_when_expected_zero(self):
        tracker = BudgetTracker(
            totals=UsageTotals(model="gpt-4.1-mini"),
            expected_cost_usd=0.0,
        )
        assert tracker.ratio() is None

    def test_log_line_format(self):
        tracker = BudgetTracker(
            totals=UsageTotals(model="gpt-4.1-mini"),
            budget_usd=10.00,
            expected_cost_usd=5.00,
        )
        tracker.charge(2_500_000, 0)  # $1.00
        line = tracker.log_line()
        assert "[budget]" in line
        assert "cost=$1.00" in line
        assert "cap=$10.00" in line
        assert "(10%)" in line  # 1/10 = 10%
        assert "expected=$5.00" in line
        assert "(actual/expected=0.20)" in line
        assert "tokens=in:2,500,000/out:0" in line
        assert "calls=1" in line

    def test_log_line_with_no_cap(self):
        tracker = BudgetTracker(totals=UsageTotals(model="gpt-4.1-mini"))
        tracker.charge(1_000_000, 0)
        line = tracker.log_line()
        assert "[budget]" in line
        assert "cap=" not in line
        assert "expected=" not in line

    def test_maybe_log_throttles(self, caplog):
        tracker = BudgetTracker(totals=UsageTotals(model="gpt-4.1-mini"))
        log = logging.getLogger("test.budget")
        with caplog.at_level(logging.INFO, logger=log.name):
            # First call: tiny charge, well below threshold (default $0.10).
            tracker.charge(1_000, 0)  # ~$0.0004
            tracker.maybe_log(log)
            n_after_tiny = len(caplog.records)
            # Second call: big charge that crosses threshold.
            tracker.charge(1_000_000, 0)  # ~$0.40
            tracker.maybe_log(log)
            n_after_big = len(caplog.records)
        assert n_after_tiny == 0  # tiny charge didn't trigger log
        assert n_after_big == 1  # big charge did

    def test_as_dict_includes_cap_and_expected(self):
        tracker = BudgetTracker(
            totals=UsageTotals(model="gpt-4.1-mini"),
            budget_usd=10.00,
            expected_cost_usd=5.00,
        )
        tracker.charge(2_500_000, 0)  # $1
        d = tracker.as_dict()
        assert d["model"] == "gpt-4.1-mini"
        assert d["cost_usd"] == 1.0
        assert d["budget_usd"] == 10.0
        assert d["expected_cost_usd"] == 5.0
        assert d["actual_over_expected"] == pytest.approx(0.2, rel=1e-6)

    def test_as_dict_omits_unset_fields(self):
        tracker = BudgetTracker(totals=UsageTotals(model="gpt-4.1-mini"))
        tracker.charge(1_000, 0)
        d = tracker.as_dict()
        assert "budget_usd" not in d
        assert "expected_cost_usd" not in d
        assert "actual_over_expected" not in d


# ---------------------------------------------------------------------------
# MultiModelUsage
# ---------------------------------------------------------------------------

class TestMultiModelUsage:
    """``MultiModelUsage`` is the per-model accumulator used by mixed
    workloads (steering's GPT+Haiku effect ensemble, anything that
    calls more than one model in a single run).  Critical invariants:
    per-model totals stay separate, ``charge`` returns the per-call USD
    delta, and ``write_json``/``load_or_create`` round-trip exactly.
    """

    def test_charge_separates_models(self):
        u = MultiModelUsage()
        u.charge("gpt-4.1-mini", 1_000_000, 0)   # = $0.40 input
        u.charge("claude-haiku-4-5-20251001", 1_000_000, 0)  # = $1.00 input
        assert u.per_model["gpt-4.1-mini"].cost_usd == pytest.approx(0.40)
        assert u.per_model["claude-haiku-4-5-20251001"].cost_usd == pytest.approx(1.00)
        assert u.total_cost_usd == pytest.approx(1.40)

    def test_charge_accumulates_within_model(self):
        u = MultiModelUsage()
        u.charge("gpt-4.1-mini", 500_000, 100_000)   # 0.20 in + 0.16 out = $0.36
        u.charge("gpt-4.1-mini", 500_000, 100_000)   # +$0.36 = $0.72
        sub = u.per_model["gpt-4.1-mini"]
        assert sub.prompt_tokens == 1_000_000
        assert sub.completion_tokens == 200_000
        assert sub.n_calls == 2
        assert sub.cost_usd == pytest.approx(0.72)
        assert u.n_calls == 2

    def test_charge_returns_per_call_delta(self):
        u = MultiModelUsage()
        delta = u.charge("gpt-4.1-mini", 1_000_000, 0)
        assert delta == pytest.approx(GPT_MINI_RATE_IN)

    def test_aggregate_properties(self):
        u = MultiModelUsage()
        u.charge("gpt-4.1-mini", 100, 200)
        u.charge("claude-haiku-4-5", 300, 400)
        assert u.total_prompt_tokens == 400
        assert u.total_completion_tokens == 600
        assert u.n_calls == 2

    def test_as_dict_roundtrip(self):
        u = MultiModelUsage()
        u.charge("gpt-4.1-mini", 1_234_567, 89_012)
        u.charge("claude-haiku-4-5-20251001", 50_000, 5_000)
        d = u.as_dict()
        u2 = MultiModelUsage.from_dict(d)
        # Per-model token counts and call counts exact match.
        for m in u.per_model:
            assert u2.per_model[m].prompt_tokens == u.per_model[m].prompt_tokens
            assert u2.per_model[m].completion_tokens == u.per_model[m].completion_tokens
            assert u2.per_model[m].n_calls == u.per_model[m].n_calls
        # Aggregate cost preserved to float precision.
        assert u2.total_cost_usd == pytest.approx(u.total_cost_usd)

    def test_write_json_load_or_create_roundtrip(self, tmp_path):
        u = MultiModelUsage()
        u.charge("gpt-4.1-mini", 2_000, 500)
        path = tmp_path / "usage.json"
        u.write_json(path)
        assert path.exists()
        loaded = MultiModelUsage.load_or_create(path)
        assert loaded.per_model["gpt-4.1-mini"].prompt_tokens == 2_000
        assert loaded.per_model["gpt-4.1-mini"].completion_tokens == 500
        assert loaded.per_model["gpt-4.1-mini"].n_calls == 1

    def test_load_or_create_missing_returns_empty(self, tmp_path):
        loaded = MultiModelUsage.load_or_create(tmp_path / "does_not_exist.json")
        assert loaded.per_model == {}
        assert loaded.n_calls == 0
        assert loaded.total_cost_usd == 0.0

    def test_load_or_create_reads_a_json_metadata_envelope(self, tmp_path):
        """Regression (2026-10-12): the docstring promised the json_metadata
        envelope ({"result": ..., "_provenance": ...}) but "result" was not
        among the keys looked at, so such a file loaded as empty and the
        next write erased its history."""
        u = MultiModelUsage()
        u.charge("gpt-4.1-mini", 2_000, 500)
        path = tmp_path / "usage.json"
        path.write_text(json.dumps({"result": u.as_dict(), "_provenance": {"inputs": []}}))
        assert MultiModelUsage.load_or_create(path).per_model["gpt-4.1-mini"].prompt_tokens == 2_000

    def test_load_or_create_warns_on_an_unreadable_record(self, tmp_path, caplog):
        path = tmp_path / "usage.json"
        path.write_text('{"per_model": {')                      # truncated
        with caplog.at_level("WARNING"):
            assert MultiModelUsage.load_or_create(path).n_calls == 0
        assert "unreadable" in caplog.text

    def test_merge_from_accumulates(self):
        u1 = MultiModelUsage()
        u1.charge("gpt-4.1-mini", 1_000_000, 0)   # $0.40
        u2 = MultiModelUsage()
        u2.charge("gpt-4.1-mini", 1_000_000, 0)   # +$0.40
        u2.charge("claude-haiku-4-5", 1_000_000, 0)  # +$1.00
        u1.merge_from(u2)
        # gpt-4.1-mini summed; haiku created fresh.
        assert u1.per_model["gpt-4.1-mini"].prompt_tokens == 2_000_000
        assert u1.per_model["gpt-4.1-mini"].n_calls == 2
        assert u1.per_model["claude-haiku-4-5"].prompt_tokens == 1_000_000
        assert u1.total_cost_usd == pytest.approx(0.80 + 1.00)

    def test_resume_flow_via_load_then_charge(self, tmp_path):
        # Simulate: first invocation writes usage.json, second invocation
        # loads it, charges additional calls, and writes back.
        path = tmp_path / "usage.json"
        first = MultiModelUsage()
        first.charge("gpt-4.1-mini", 1_000_000, 0)
        first.write_json(path)
        # Resume:
        second = MultiModelUsage()
        second.merge_from(MultiModelUsage.load_or_create(path))
        second.charge("gpt-4.1-mini", 500_000, 0)
        second.write_json(path)
        # Re-read: total = 1.5M input tokens, 2 calls, $0.60.
        reloaded = MultiModelUsage.load_or_create(path)
        sub = reloaded.per_model["gpt-4.1-mini"]
        assert sub.prompt_tokens == 1_500_000
        assert sub.n_calls == 2
        assert sub.cost_usd == pytest.approx(0.60)

    def test_log_line_empty(self):
        u = MultiModelUsage()
        assert "no API calls" in u.log_line()

    def test_log_line_non_empty_mentions_each_model(self):
        u = MultiModelUsage()
        u.charge("gpt-4.1-mini", 100, 50)
        u.charge("claude-haiku-4-5", 100, 50)
        line = u.log_line(prefix="[test]")
        assert "[test]" in line
        assert "gpt-4.1-mini" in line
        assert "claude-haiku-4-5" in line
        assert "total=$" in line


# ---------------------------------------------------------------------------
# Roles-vs-traits response-length asymmetry
# ---------------------------------------------------------------------------

class TestSurgicalRejudgeCostSplit:
    """``surgical_rejudge_cost_split`` divides a per-axis judging budget
    across roles and traits cohorts, applying the
    :data:`ROLES_RESPONSE_LENGTH_FACTOR` multiplier in response mode
    (where Qwen role-played outputs are ~22% longer than
    trait-modulated) and treating cohorts symmetrically in static mode.
    """

    def test_constant_is_documented_value(self):
        # Anchor: the empirical 1.22 from 5d.1 (29 roles + 14 traits).
        # Lock in to catch accidental drift.
        assert ROLES_RESPONSE_LENGTH_FACTOR == 1.22

    def test_response_mode_applies_factor(self):
        r, t = surgical_rejudge_cost_split(
            per_axis_cost_usd=3.50,
            n_roles=29, n_traits=14,
            mode="response",
        )
        # weight_r = 29 * 1.22 = 35.38; weight_t = 14; total = 49.38
        # r = 3.50 * 35.38 / 49.38 ≈ 2.508
        # t = 3.50 * 14    / 49.38 ≈ 0.992
        assert r == pytest.approx(2.508, abs=1e-3)
        assert t == pytest.approx(0.992, abs=1e-3)
        assert r + t == pytest.approx(3.50, rel=1e-9)

    def test_response_mode_matches_5d1_empirical(self):
        # Sanity check vs 5d.1 actuals: $40.45 across 11 axes ≈
        # $3.68/axis.  Applying the formula with the *observed*
        # per-axis cost should give a roles share consistent with
        # the per-cell mean of $2.69 and traits share $0.99.
        r, t = surgical_rejudge_cost_split(
            per_axis_cost_usd=3.68,
            n_roles=29, n_traits=14,
            mode="response",
        )
        assert r == pytest.approx(2.638, abs=0.05)   # observed ~$2.69
        assert t == pytest.approx(1.043, abs=0.05)   # observed ~$0.99

    def test_static_mode_treats_cohorts_symmetrically(self):
        r, t = surgical_rejudge_cost_split(
            per_axis_cost_usd=0.42,
            n_roles=18, n_traits=18,
            mode="static",
        )
        # 50/50 split when N is equal in static mode.
        assert r == pytest.approx(0.21, abs=1e-6)
        assert t == pytest.approx(0.21, abs=1e-6)

    def test_static_mode_no_factor_unequal_N(self):
        r, t = surgical_rejudge_cost_split(
            per_axis_cost_usd=3.50,
            n_roles=29, n_traits=14,
            mode="static",
        )
        # Static: weight_r = 29, weight_t = 14, total = 43.
        # r = 3.50 * 29/43 ≈ 2.361; t = 3.50 * 14/43 ≈ 1.139.
        assert r == pytest.approx(2.361, abs=1e-3)
        assert t == pytest.approx(1.139, abs=1e-3)
        assert r + t == pytest.approx(3.50, rel=1e-9)

    def test_zero_total_returns_zeros(self):
        r, t = surgical_rejudge_cost_split(
            per_axis_cost_usd=10.0, n_roles=0, n_traits=0,
        )
        assert (r, t) == (0.0, 0.0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            surgical_rejudge_cost_split(
                per_axis_cost_usd=1.0, n_roles=5, n_traits=5,
                mode="bogus",
            )


class TestBatchPrices:
    """A usage record kept under "<model>:batch" is priced at the batch rate
    (Sep 2026: batch is a per-run choice for runs over about $20)."""

    def test_anthropic_and_openai_batches_are_half_price(self):
        from assistant_axis.judge_pricing import price_for_model
        assert price_for_model("claude-sonnet-4-6:batch") == (1.5, 7.5)
        assert price_for_model("claude-haiku-4-5-20251001:batch") == (0.5, 2.5)
        assert price_for_model("gpt-4.1-mini:batch") == (0.2, 0.8)

    def test_the_hosted_subject_model(self):
        from assistant_axis.judge_pricing import price_for_model
        assert price_for_model("qwen/qwen3-32b") == (0.08, 0.28)
        rate_in, rate_out = price_for_model("qwen/qwen3-32b:batch")
        assert rate_in == pytest.approx(0.064) and rate_out == pytest.approx(0.224)

    def test_real_time_prices_are_unchanged(self):
        from assistant_axis.judge_pricing import price_for_model
        assert price_for_model("claude-sonnet-4-6") == (3.0, 15.0)

    def test_a_merged_record_keeps_the_two_apart(self):
        from assistant_axis.judge_pricing import MultiModelUsage
        a, b = MultiModelUsage(), MultiModelUsage()
        a.charge("claude-sonnet-4-6", 1_000_000, 0)
        b.charge("claude-sonnet-4-6:batch", 1_000_000, 0)
        a.merge_from(b)
        assert a.per_model["claude-sonnet-4-6"].cost_usd == pytest.approx(3.0)
        assert a.per_model["claude-sonnet-4-6:batch"].cost_usd == pytest.approx(1.5)

    def test_an_unknown_model_in_batch_fails_loudly(self):
        from assistant_axis.judge_pricing import price_for_model
        with pytest.raises(KeyError):
            price_for_model("some-new-model:batch")


class TestClaude5Prices:
    """The prices read from platform.claude.com on 2026-09-30.  The "-5" lines
    must win over the bare "sonnet" and "opus" fragments."""

    def test_sonnet_5_and_5_5_are_two_and_ten(self):
        from assistant_axis.judge_pricing import price_for_model
        assert price_for_model("claude-sonnet-5") == (2.0, 10.0)
        assert price_for_model("claude-sonnet-5-5") == (2.0, 10.0)
        assert price_for_model("claude-sonnet-4-6") == (3.0, 15.0)

    def test_opus_5_5_is_four_and_twenty(self):
        from assistant_axis.judge_pricing import price_for_model
        assert price_for_model("claude-opus-5-5") == (4.0, 20.0)
        assert price_for_model("claude-opus-5") == (5.0, 25.0)
        assert price_for_model("claude-opus-4-8") == (5.0, 25.0)
        assert price_for_model("claude-opus-5-5:batch") == (2.0, 10.0)

    def test_retired_opus_is_not_priced_as_a_current_one(self):
        from assistant_axis.judge_pricing import price_for_model
        with pytest.raises(KeyError):
            price_for_model("claude-opus-4-1")


class TestCacheTokens:
    """Prompt caching (2026-11-01): the cache-written and cache-read parts of a prompt are priced at
    ``cache_rate_factors`` times the input rate (GPT-5.6 and later: writes 1.25x, reads 0.1x), so that
    usage.json prices what the bill prices; a record of a run that cached nothing reads as before."""

    def test_factors_by_model(self):
        from assistant_axis.judge_pricing import cache_rate_factors
        assert cache_rate_factors("gpt-5.6-luna") == (1.25, 0.10)
        assert cache_rate_factors("gpt-5.6-luna:flex") == (1.25, 0.10)
        assert cache_rate_factors("gpt-6-luna:batch") == (1.25, 0.10)
        assert cache_rate_factors("gpt-4.1-mini") == (1.0, 0.25)
        assert cache_rate_factors("gpt-5.4-mini") == (1.0, 0.10)
        assert cache_rate_factors("claude-haiku-5-5") == (1.25, 0.10)
        assert cache_rate_factors("claude-sonnet-4-6:batch") == (1.25, 0.10)
        assert cache_rate_factors("text-embedding-3-large") == (1.0, 1.0)

    def test_the_luna_flex_dashboard_arithmetic(self):
        """Roger's OpenAI dashboard, luna Flex 29 Oct to 1 Nov 2026: 1,584M tokens written at $0.125 per
        1M = $198.0, 4.9M read at $0.01 = $0.049, 1,187M uncached at $0.10 = $118.7; priced plain, as every
        record was until 2026-11-01, the same tokens come to $39 less (the write premium less the reads'
        discount), about 10% of the Flex bill."""
        written, read, plain = 1_584_000_000, 4_900_000, 1_187_000_000
        total = written + read + plain
        cost = cost_for_usage("gpt-5.6-luna:flex", total, 0, written, read)
        assert cost == pytest.approx(198.0 + 0.049 + 118.7, rel=1e-3)
        assert cost - cost_for_usage("gpt-5.6-luna:flex", total, 0) == pytest.approx(39.6 - 0.44, abs=0.05)

    def test_charge_record_merge_and_log_line(self, tmp_path):
        u = MultiModelUsage()
        u.charge("gpt-5.6-luna:flex", 1218, 4, cache_write_tokens=1215)
        u.charge("gpt-5.6-luna:flex", 1218, 4, cache_read_tokens=1215)
        t = u.per_model["gpt-5.6-luna:flex"]
        assert (t.prompt_tokens, t.cache_write_tokens, t.cache_read_tokens, t.n_calls) == (2436, 1215, 1215, 2)
        assert t.cost_usd == pytest.approx((6 * 0.10 + 1215 * 0.125 + 1215 * 0.01 + 8 * 0.60) / 1e6)
        d = t.as_dict()
        assert d["cache_write_tokens"] == 1215 and d["cache_read_tokens"] == 1215
        path = tmp_path / "usage.json"
        u.write_json(path)
        again = MultiModelUsage.load_or_create(path)
        assert again.per_model["gpt-5.6-luna:flex"].cache_write_tokens == 1215
        assert again.total_cost_usd == pytest.approx(u.total_cost_usd)
        merged = MultiModelUsage()
        merged.merge_from(u)
        merged.merge_from(u)
        m = merged.per_model["gpt-5.6-luna:flex"]
        assert (m.cache_write_tokens, m.cache_read_tokens) == (2430, 2430)
        assert m.cost_usd == pytest.approx(2 * t.cost_usd)
        assert "cache_write=1,215" in u.log_line() and "cache_read=1,215" in u.log_line()

    def test_a_record_without_cache_counts_is_unchanged(self):
        u = UsageTotals(model="gpt-4.1-mini")
        u.charge(1_000_000, 1_000_000)
        assert u.as_dict() == {"model": "gpt-4.1-mini", "prompt_tokens": 1_000_000,
                               "completion_tokens": 1_000_000, "cost_usd": 2.0, "n_calls": 1}
        assert "cache" not in MultiModelUsage.from_dict({"per_model": {"gpt-4.1-mini": u.as_dict()}}).log_line()

    def test_extractors_read_sdk_objects_and_batch_bodies(self):
        from assistant_axis.judge_pricing import TokenUsage, anthropic_usage, openai_usage
        details = SimpleNamespace(cached_tokens=0, cache_write_tokens=1215, audio_tokens=0)
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1218, completion_tokens=4,
                                                     prompt_tokens_details=details))
        assert openai_usage(resp) == TokenUsage(1218, 4, 1215, 0)
        body = {"usage": {"prompt_tokens": 1218, "completion_tokens": 4,
                          "prompt_tokens_details": {"cached_tokens": 1215, "cache_write_tokens": 0}}}
        assert openai_usage(body) == TokenUsage(1218, 4, 0, 1215)
        assert extract_usage_openai(body) == (1218, 4)
        # gpt-4.1-mini reports no cache_write_tokens; no details, or no usage, gives zeros
        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1219, completion_tokens=1,
                                                     prompt_tokens_details=SimpleNamespace(cached_tokens=1024)))
        assert openai_usage(resp) == TokenUsage(1219, 1, 0, 1024)
        assert openai_usage(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))) \
            == TokenUsage(10, 2)
        assert openai_usage(SimpleNamespace()) == TokenUsage()
        # Anthropic's input_tokens leaves the cached tokens out: the prompt total adds them back
        resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=300, output_tokens=40,
                                                     cache_creation_input_tokens=600, cache_read_input_tokens=0))
        assert anthropic_usage(resp) == TokenUsage(900, 40, 600, 0)
        assert extract_usage_anthropic(resp) == (900, 40)
        assert anthropic_usage({"usage": {"input_tokens": 300, "output_tokens": 40}}) == TokenUsage(300, 40)

    def test_budget_tracker_charges_cache_tokens(self):
        tracker = BudgetTracker(totals=UsageTotals(model="gpt-5.6-luna"))
        tracker.charge(1218, 4, 1215, 0)
        assert tracker.totals.cache_write_tokens == 1215
        assert tracker.totals.cost_usd == pytest.approx((3 * 0.20 + 1215 * 0.25 + 4 * 1.20) / 1e6)


def test_gemini_usage_reads_the_cache_and_thought_tokens():
    """2026-11-04: Gemini's implicit cache, no write premium and reads at a tenth (batch rates included)."""
    u = gemini_usage({"usageMetadata": {"promptTokenCount": 10_145, "candidatesTokenCount": 110,
                                        "thoughtsTokenCount": 5, "cachedContentTokenCount": 6_125}})
    assert (u.prompt_tokens, u.completion_tokens, u.cache_write_tokens, u.cache_read_tokens) == (10_145, 115, 0, 6_125)
    assert gemini_usage({}) == gemini_usage(None) == type(u)()
    assert cache_rate_factors("gemini-3.1-flash-lite:batch") == (1.0, 0.10)
    assert cost_for_usage("gemini-3.1-flash-lite:batch", 1_000_000, 0, 0, 1_000_000) == pytest.approx(0.0125)
