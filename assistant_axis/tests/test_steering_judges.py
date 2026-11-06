"""Tests for assistant_axis/steering_judges.py.

Covers:
- PersonaSpec / SteeringSpec dataclasses (label/description rendering)
- Prompt builders (coherence, RP, effect bidirectional, effect pole)
- NoOpJudgeDispatcher's protocol surface
- RealJudgeDispatcher batch-planning and skip-or-judge decision logic
  (with the network calls mocked)

Network-touching paths (real OpenAI/Anthropic calls, the asyncio loop
thread, retries) are exercised end-to-end on the pod, not here.  These
unit tests pin the prompt structure and the in-process branching.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from assistant_axis import steering_judges as sj
from assistant_axis.steering_judges import (
    EFFECT_MODE_BIDIRECTIONAL,
    EFFECT_MODE_BOTH,
    EFFECT_MODE_SEPARATE_POLES,
    NoOpJudgeDispatcher,
    PersonaSpec,
    RealJudgeDispatcher,
    SteeringSpec,
    build_baseline_lookup,
    build_coherence_prompt,
    build_effect_bidir_batch_prompt,
    build_effect_pole_batch_prompt,
    build_rp_prompt,
)


# ---------------------------------------------------------------------------
# PersonaSpec / SteeringSpec
# ---------------------------------------------------------------------------

class TestSpecs:
    def test_persona_simple(self):
        p = PersonaSpec(role="counselor", description="provides emotional support")
        assert p.display_label() == "counselor"
        assert p.display_description() == "provides emotional support"

    def test_persona_with_traits(self):
        p = PersonaSpec(
            role="counselor", description="emotional support",
            extra_traits=[("compassionate", "warmth"), ("traditional", "old-school")],
        )
        assert p.display_label() == "counselor + compassionate + traditional"
        # Description preserves all parts via " | " join
        desc = p.display_description()
        assert "counselor: emotional support" in desc
        assert "compassionate: warmth" in desc
        assert "traditional: old-school" in desc

    def test_steering_spec(self):
        s = SteeringSpec(
            axis_name="compassionate-callous",
            pos_label="callous", pos_description="hard, indifferent",
            neg_label="compassionate", neg_description="warm, sensitive",
        )
        assert s.axis_name == "compassionate-callous"
        assert s.pos_label == "callous"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

@pytest.fixture
def persona():
    return PersonaSpec(role="counselor",
                       description="provides compassionate emotional support")


@pytest.fixture
def steering():
    return SteeringSpec(
        axis_name="compassionate-callous",
        pos_label="callous", pos_description="hard, indifferent",
        neg_label="compassionate", neg_description="warm, sensitive",
    )


class TestCoherencePrompt:
    def test_includes_all_required_fields(self, persona, steering):
        p = build_coherence_prompt(
            persona=persona, steering=steering, sign=+1, strength=4.0,
            question="How do I cope with grief?",
            baseline_response="I'm so sorry for your loss...",
            steered_response="Get over it.",
        )
        assert "counselor" in p
        assert "compassionate emotional support" in p
        assert "compassionate-callous" in p
        assert "callous" in p
        assert "hard, indifferent" in p
        assert "compassionate" in p
        assert "How do I cope with grief?" in p
        assert "I'm so sorry for your loss" in p
        assert "Get over it." in p
        # Block delimiters present and used to wrap the response
        assert "[QUESTION START]" in p and "[QUESTION END]" in p
        assert "[BASELINE START]" in p and "[BASELINE END]" in p
        assert "[RESPONSE START]" in p and "[RESPONSE END]" in p
        # Rubric anchors visible
        assert "0-3" in p
        assert "drunk" in p.lower()
        assert "repetition" in p.lower()
        # New v2 anchors: empty/echo as level-2 (c)/level-3 (b)
        assert "echo" in p.lower()                         # echo-of-question
        assert "empty" in p.lower()                        # empty case
        assert "wrong language" in p.lower()               # explicit wrong-lang clause
        # JSON output instruction visible
        assert '"score"' in p

    def test_sign_zero_omits_steering_sentence(self, persona, steering):
        """sign=0 corresponds to an unsteered baseline.  Project
        convention (steering/post_judge.py) is to skip baselines for
        coherence-judging entirely, but the prompt builder must still
        be honest if it ever gets called with sign=0: the steering-
        direction sentence is omitted so we don't lie to the judge.

        See the build_coherence_prompt inline comment for full
        context."""
        p = build_coherence_prompt(
            persona=persona, steering=steering, sign=0, strength=0.0,
            question="q", baseline_response="b", steered_response="r",
        )
        # No claim that any steering happened.  Note: the phrase
        # "steered toward" appears in the level-2 scoring-criterion
        # example body ("an accountant not steered toward anything
        # mystical sounding mystical..."), so we can't ban that
        # substring globally; the assertion is just that the
        # response-specific claim ("This response was steered
        # toward X.") is absent.  Pole labels still appear
        # elsewhere (persona description, axis poles in the
        # "STEERING AXIS:" block) which is fine.
        assert "This response was steered" not in p
        # The rest of the prompt (persona, axis definition, scale, etc.)
        # is still present so the judge has enough context to rate
        # coherence, even though it's not told which direction was
        # steered (because none was).
        assert "PERSONA" in p
        assert "STEERING AXIS" in p
        assert "[QUESTION START]" in p
        assert "[BASELINE START]" in p
        assert "[RESPONSE START]" in p
        assert "0-3" in p

    def test_sign_resolved_to_pole_label_and_strength_hidden(
        self, persona, steering,
    ):
        """In v5 (2026-05-13) the coherence prompt no longer shows the
        raw sign or the numeric strength.  Instead:

        * The sign is pre-resolved to its pole label so the judge sees
          ``This response was steered toward callous.`` (or
          ``toward compassionate.``) without doing the +1/-1 mapping
          in its head.
        * The strength is omitted entirely -- coherence should be
          judged on its own merits, and showing the strength would
          prejudice the judge (high strength → expect incoherence,
          low strength → expect coherence) before it reads the
          response.

        See module docstring for the rationale.  Previous version
        of this test (pre-2026-05-13) asserted the OLD behaviour
        (``+1``, ``-1``, ``4.0``, ``2.0`` strings present); this
        rewrite is paired with the code change to drop those
        fields from COHERENCE_RUBRIC.
        """
        p_pos = build_coherence_prompt(
            persona=persona, steering=steering, sign=+1, strength=4.0,
            question="q", baseline_response="b", steered_response="r",
        )
        p_neg = build_coherence_prompt(
            persona=persona, steering=steering, sign=-1, strength=2.0,
            question="q", baseline_response="b", steered_response="r",
        )
        # Direction is shown as a pole name, not a signed number.
        assert "steered toward callous" in p_pos
        assert "steered toward compassionate" in p_neg
        # Old fields are gone -- judge cannot see sign or strength.
        assert "+1" not in p_pos
        assert "-1" not in p_neg
        assert "4.0" not in p_pos
        assert "2.0" not in p_neg
        # Defensive: make sure the strength values aren't sneaking
        # in via some other substring (e.g. "0.4" being substring of
        # "4.0"); the prompt should contain neither.
        assert "strength" not in p_pos.lower()
        assert "strength" not in p_neg.lower()


class TestRPPrompt:
    def test_includes_persona_and_axis(self, persona, steering):
        p = build_rp_prompt(
            persona=persona, steering=steering, sign=+1,
            question="How do I deal with grief?",
            steered_response="Death comes for us all.",
        )
        assert "counselor" in p
        assert "compassionate emotional support" in p
        # Mentions both poles for "may legitimately display..."
        assert "callous" in p
        assert "compassionate" in p
        # Question + answer present
        assert "deal with grief" in p
        assert "Death comes for us all" in p
        # 0-3 rubric scaffolding present
        assert "0-3" in p

    def test_rp_does_not_show_baseline(self, persona, steering):
        # RP rubric only takes the steered response, not the baseline.
        p = build_rp_prompt(
            persona=persona, steering=steering, sign=+1,
            question="q", steered_response="r",
        )
        assert "BASELINE" not in p

    def test_steering_direction_sentence_shown_for_nonzero_sign(
        self, persona, steering,
    ):
        """v3 (2026-05-13): RP rubric now appends a "This response was
        steered toward {pole}." sentence after the axis-poles bullets,
        so the judge knows which way the response was pulled when
        assessing persona-embodiment.  The sign is pre-resolved to a
        pole label same as in the coherence rubric."""
        p_pos = build_rp_prompt(
            persona=persona, steering=steering, sign=+1,
            question="q", steered_response="r",
        )
        p_neg = build_rp_prompt(
            persona=persona, steering=steering, sign=-1,
            question="q", steered_response="r",
        )
        assert "This response was steered toward callous" in p_pos
        assert "This response was steered toward compassionate" in p_neg
        # Sign / strength numbers MUST NOT appear (same rationale as
        # coherence rubric -- pre-resolve to direction name to reduce
        # judge cognitive load).
        assert "+1" not in p_pos
        assert "-1" not in p_neg
        assert "strength" not in p_pos.lower()

    def test_sign_zero_omits_steering_direction(self, persona, steering):
        """For the baseline (sign=0), RP judging is still appropriate
        (a persona-embodying baseline should still score 3), but the
        steering-direction sentence must be omitted so we don't claim
        the baseline was steered.  See build_rp_prompt inline comment
        for full context."""
        p = build_rp_prompt(
            persona=persona, steering=steering, sign=0,
            question="q", steered_response="r",
        )
        # Per-record direction-of-steering claim is absent.
        assert "This response was steered" not in p
        # Axis-poles bulleted-list paragraph still present (the persona
        # context is relevant even without a per-record direction).
        assert "callous" in p
        assert "compassionate" in p
        # 0-3 scoring scale still present.
        assert "0-3" in p


class TestEffectBidirBatchPrompt:
    def test_n_items_visible(self, persona, steering):
        items = [
            {"id": i, "question": f"Q{i}",
             "baseline_response": f"B{i}", "steered_response": f"R{i}"}
            for i in range(5)
        ]
        p = build_effect_bidir_batch_prompt(
            persona=persona, steering=steering, sign=-1, strength=8.0,
            items=items,
        )
        assert "5" in p          # n_items
        for i in range(5):
            assert f"Q{i}" in p
            assert f"R{i}" in p
            assert f"B{i}" in p
            # id label appears in the per-item header
            assert f"id={i}" in p
        # -3..+3 scale visible
        assert "-3" in p
        assert "+3" in p

    def test_v4_hides_direction_and_strength(self, persona, steering):
        """v4 (2026-05-13): the bidirectional effect rubric scores
        responses on a signed -3..+3 scale, requiring the judge to
        decide BOTH magnitude and direction.  Telling the judge the
        steering direction (sign) and strength up front gives a
        strong two-dimensional prior that biases the score in both
        respects: judge expects positive-sign scores AND large-
        magnitude scores when told "direction=+1, strength=8".
        Both fields are now hidden so the judge has to read the
        responses cold.

        Compare ``TestEffectPoleBatchPrompt`` -- the unidirectional
        rubric scores a fixed trait on 0..3, so direction is just
        contextual and is kept; only strength is hidden there.

        Test rewrite paired with code change v3 → v4; previously
        these checks would have asserted the OPPOSITE (sign/strength
        ARE present).  Pre-2026-05-13 test history if interested:
        the rubric body had a literal line
        ``This batch was steered with direction={sign:+d} at strength {strength}.``
        which has been removed.
        """
        items = [{"id": 0, "question": "q",
                  "baseline_response": "b", "steered_response": "r"}]
        p_pos = build_effect_bidir_batch_prompt(
            persona=persona, steering=steering, sign=+1, strength=8.0,
            items=items,
        )
        p_neg = build_effect_bidir_batch_prompt(
            persona=persona, steering=steering, sign=-1, strength=4.0,
            items=items,
        )
        # Direction-of-steering claim absent.  The score-scale lines
        # ("+1: slightly more ..." etc.) contain "+1"/"-1" as bullet
        # labels, so we check for direction CLAIM PATTERNS instead of
        # raw "+1"/"-1": no "direction=", no "sign=", no
        # "steered with direction", no "steered toward X" sentence.
        assert "direction=" not in p_pos
        assert "direction=" not in p_neg
        assert "sign=" not in p_pos
        assert "sign=" not in p_neg
        assert "steered with" not in p_pos.lower()
        assert "steered with" not in p_neg.lower()
        assert "steered toward" not in p_pos
        assert "steered toward" not in p_neg
        # Numeric strength absent.  Same guard pattern: don't check
        # raw "8"/"4" since the n_items count could include those;
        # check for "at strength", "strength=", and the word
        # "strength" in lowered form.
        assert "8.0" not in p_pos
        assert "4.0" not in p_neg
        assert "at strength" not in p_pos
        assert "strength=" not in p_pos
        assert "strength" not in p_pos.lower()
        # Pole-DEFINITIONS still present (the axis-definition block,
        # which is reference context, not direction-of-this-batch info).
        assert "callous" in p_pos
        assert "compassionate" in p_pos
        # -3..+3 scoring scale still present.
        assert "-3" in p_pos
        assert "+3" in p_pos


class TestEffectPoleBatchPrompt:
    def test_pos_pole(self, persona, steering):
        items = [{"id": 7, "question": "q",
                  "baseline_response": "b", "steered_response": "r"}]
        p = build_effect_pole_batch_prompt(
            persona=persona, steering=steering, sign=+1, strength=2.0,
            pole="pos", items=items,
        )
        # The pos_label is the "trait to measure" -- that's "callous"
        assert "callous" in p
        # The other-pole label "compassionate" should also appear (for context)
        assert "compassionate" in p
        # 0-3 scale, not -3..+3
        assert "0-3" in p

    def test_neg_pole_swaps_labels(self, persona, steering):
        items = [{"id": 0, "question": "q",
                  "baseline_response": "b", "steered_response": "r"}]
        p = build_effect_pole_batch_prompt(
            persona=persona, steering=steering, sign=-1, strength=1.0,
            pole="neg", items=items,
        )
        # When pole=neg, the trait being measured is the neg pole = compassionate
        # (other_pole_label is callous)
        assert "TRAIT TO MEASURE: compassionate" in p
        assert "the other pole is callous" in p

    def test_v4_hides_strength_keeps_direction(self, persona, steering):
        """v4 (2026-05-13): unidirectional pole rubric drops the
        numeric ``strength`` from the prompt (priming the judge to
        expect large magnitudes biases scores upward) but keeps the
        steering direction as plain-text "steered toward X".  The
        trait being measured is FIXED by ``pole=`` so direction is
        contextual rather than a strong score-shaping prior --
        unlike the bidirectional rubric (see
        ``test_v4_hides_direction_and_strength``).

        Also verifies the v3-style ``sign`` → ``steered_pole_label``
        resolution used by the COHERENCE rubric v5: when sign is
        +1, the steered-toward label is the POS pole; when sign is
        -1, it's the NEG pole.  This holds regardless of which
        pole is being scored (``pole=``).
        """
        items = [{"id": 0, "question": "q",
                  "baseline_response": "b", "steered_response": "r"}]
        # sign=+1 → steered toward pos pole (callous), measuring pos pole
        p_pos_steer_pos = build_effect_pole_batch_prompt(
            persona=persona, steering=steering, sign=+1, strength=8.0,
            pole="pos", items=items,
        )
        assert "steered toward callous" in p_pos_steer_pos
        # No numeric strength claim.
        assert "8.0" not in p_pos_steer_pos
        assert "at strength" not in p_pos_steer_pos
        # No literal sign value.
        assert "+1" not in p_pos_steer_pos
        assert "direction=" not in p_pos_steer_pos

        # sign=-1 → steered toward neg pole (compassionate), measuring pos pole
        p_pos_steer_neg = build_effect_pole_batch_prompt(
            persona=persona, steering=steering, sign=-1, strength=4.0,
            pole="pos", items=items,
        )
        assert "steered toward compassionate" in p_pos_steer_neg
        assert "4.0" not in p_pos_steer_neg

        # sign=+1 → steered toward pos pole (callous), measuring neg pole
        p_neg_steer_pos = build_effect_pole_batch_prompt(
            persona=persona, steering=steering, sign=+1, strength=2.0,
            pole="neg", items=items,
        )
        assert "steered toward callous" in p_neg_steer_pos
        assert "TRAIT TO MEASURE: compassionate" in p_neg_steer_pos

    def test_v4_sign_zero_treated_as_pos(self, persona, steering):
        """Defensive sanity check: sign=0 (baseline batches) still
        produces a syntactically-valid prompt.  Edge case mainly to
        guard against KeyError if someone wires baselines through the
        pole rubric (which is unusual but technically supported by
        the function signature)."""
        items = [{"id": 0, "question": "q",
                  "baseline_response": "b", "steered_response": "r"}]
        p = build_effect_pole_batch_prompt(
            persona=persona, steering=steering, sign=0, strength=0.0,
            pole="pos", items=items,
        )
        # Tie-breaks to pos pole when sign=0 (sign>=0 branch).
        assert "steered toward callous" in p

    def test_invalid_pole_raises(self, persona, steering):
        items = [{"id": 0, "question": "q",
                  "baseline_response": "b", "steered_response": "r"}]
        with pytest.raises(ValueError, match="pole"):
            build_effect_pole_batch_prompt(
                persona=persona, steering=steering, sign=+1, strength=1.0,
                pole="middle", items=items,
            )


# ---------------------------------------------------------------------------
# NoOpJudgeDispatcher
# ---------------------------------------------------------------------------

class TestNoOpDispatcher:
    def test_full_protocol(self):
        d = NoOpJudgeDispatcher()
        rec = {"strength": 1.0, "sign": +1, "question_idx": 0,
               "question": "?", "response": "x", "judges": {}}
        # judge_coherence_blocking returns 0 (no work, sentinel)
        assert d.judge_coherence_blocking(rec) == 0
        # enqueue + drain are no-ops
        d.enqueue_strength_group(
            cell_dir="x", slot=3, layer=25, sign=+1, strength=1.0,
            records=[rec],
        )
        # should_stop_at always False -- sweep runs to max
        assert d.should_stop_at(1.0) is False
        d.drain()


# ---------------------------------------------------------------------------
# RealJudgeDispatcher: pure logic (mocked APIs)
# ---------------------------------------------------------------------------
#
# We mock call_judge_single_unified at the module boundary.  No real
# HTTP requests; no real OpenAI/Anthropic SDK construction (we patch the
# clients to mock instances, never imported real).

@pytest.fixture
def mock_clients(monkeypatch):
    """Patch the AsyncOpenAI/AsyncAnthropic constructors to return dummies.

    The dispatcher only needs them to be non-None; the actual API call
    goes through call_judge_single_unified which we patch separately.
    """
    fake_openai = mock.MagicMock(name="AsyncOpenAI-instance")
    fake_anthropic = mock.MagicMock(name="AsyncAnthropic-instance")

    import openai as openai_mod
    monkeypatch.setattr(openai_mod, "AsyncOpenAI",
                        lambda: fake_openai)
    import anthropic as anthropic_mod
    monkeypatch.setattr(anthropic_mod, "AsyncAnthropic",
                        lambda: fake_anthropic)
    return fake_openai, fake_anthropic


@pytest.fixture
def dispatcher(tmp_path, mock_clients, persona, steering):
    """A dispatcher with stub clients, no baseline."""
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("")
    d = RealJudgeDispatcher(
        records_path=records_path,
        persona=persona, steering=steering,
        baseline_lookup=lambda q_idx: f"baseline-{q_idx}",
        coherence_model="gpt-4.1-mini",
        rp_model="gpt-4.1-mini",
        effect_models=("gpt-4.1-mini", "claude-haiku-4-5-20251001"),
        effect_mode=EFFECT_MODE_BIDIRECTIONAL,
        effect_target_batch_size=10,
        skip_threshold=1.0,
        coh_stop_threshold=1.5,
        max_concurrency=4,
    )
    yield d
    d.shutdown()


def _make_record(q_idx: int, *, strength: float = 1.0, sign: int = +1,
                 coh: int | None = None) -> dict:
    judges = {"coherence": None, "persona": None, "effect": None}
    if coh is not None:
        judges["coherence"] = {
            "score": coh, "reason": "", "model": "test", "ts": 0,
            "rubric_version": 1,
        }
    return {
        "strength": float(strength), "sign": int(sign),
        "slot": 3, "layer": 25,
        "question_idx": int(q_idx), "question": f"Q{q_idx}",
        "response": f"R{q_idx}",
        "n_tokens": 10,
        "judges": judges,
        "timing": {"gen_s": 1.0},
        "abandoned": False,
    }


class TestRealDispatcherSkipLogic:
    def test_skip_when_strength_mean_above_threshold(self, dispatcher, tmp_path):
        """mean(coh) > skip_threshold -> records flagged skipped, no API calls."""
        records = [
            _make_record(0, coh=2),
            _make_record(1, coh=2),
            _make_record(2, coh=2),
        ]
        # mean = 2.0 > skip_threshold=1.0 -> SKIP
        dispatcher.enqueue_strength_group(
            cell_dir=str(tmp_path), slot=3, layer=25, sign=+1,
            strength=1.0, records=records,
        )
        for r in records:
            j = r["judges"]
            assert j["strength_mean_coh"] == pytest.approx(2.0)
            assert j["persona"]["skipped_due_to_strength_mean_coh"] is True
            assert j["effect"]["skipped_due_to_strength_mean_coh"] is True
            # No real persona/effect score populated
            assert j["persona"]["score"] is None
        # Should-stop True (mean >= 1.5)
        assert dispatcher.should_stop_at(1.0) is True

    def test_no_skip_when_strength_mean_below_threshold(self, dispatcher,
                                                       tmp_path, monkeypatch):
        """mean(coh) <= skip_threshold -> async judge work enqueued."""
        # Mock the async call so we don't hit the network; record what got
        # passed in.
        seen_calls = []

        async def fake_call(*, model, prompt, max_tokens, rate_limiter,
                             openai_client=None, anthropic_client=None,
                             temperature=1.0, usage=None):
            seen_calls.append({"model": model, "prompt_len": len(prompt)})
            # Return a bidirectional batch JSON for both models
            if "items" in prompt and "+3" in prompt:
                # Bidirectional batch
                ids_in_prompt = []
                for i in range(20):
                    if f"id={i}" in prompt:
                        ids_in_prompt.append(i)
                items = [{"id": i, "score": 1, "reason": "test"}
                         for i in ids_in_prompt]
                return json.dumps({"items": items})
            # Persona (RP) call
            return json.dumps({"score": 2, "reason": "test"})

        monkeypatch.setattr(sj, "call_judge_single_unified", fake_call)

        records = [
            _make_record(0, coh=0),
            _make_record(1, coh=1),
            _make_record(2, coh=0),
        ]
        # mean = 1/3 < 1.0 -> JUDGE
        dispatcher.enqueue_strength_group(
            cell_dir=str(tmp_path), slot=3, layer=25, sign=+1,
            strength=1.0, records=records,
        )
        # Drain the async work
        dispatcher.drain(timeout_s=10.0)

        for r in records:
            j = r["judges"]
            assert j["strength_mean_coh"] == pytest.approx(1 / 3, rel=1e-3)
            assert j["persona"]["skipped_due_to_strength_mean_coh"] is False
            assert j["persona"]["score"] == 2
            assert j["effect"]["skipped_due_to_strength_mean_coh"] is False
            assert j["effect"]["combined"] is not None
        # No stop signal (mean << 1.5)
        assert dispatcher.should_stop_at(1.0) is False

        # Verify both effect models were called (one bidir prompt each).
        # Plus persona calls (one per record per RP_model).
        models_called = [c["model"] for c in seen_calls]
        assert "gpt-4.1-mini" in models_called
        assert "claude-haiku-4-5-20251001" in models_called

    def test_eager_effect_future_registration_no_race(
            self, dispatcher, tmp_path, monkeypatch):
        """Regression: the bidirectional-scan runner calls
        ``judge_effect_for_strength_async`` BEFORE
        ``enqueue_strength_group`` (the latter is chained off the
        coherence-future's done callback, so it runs later).  Before
        the 2026-05-15 fix, ``judge_effect_for_strength_async`` would
        find no registered future and resolve to NaN immediately,
        even though ``enqueue_strength_group`` eventually ran and
        produced a real effect score.

        Fix verified: pre-call ``judge_effect_for_strength_async``,
        then later call ``enqueue_strength_group`` -- the SAME future
        object is reused, and it resolves to a non-NaN mean once
        async work completes.
        """
        import math

        async def fake_call(*, model, prompt, max_tokens, rate_limiter,
                             openai_client=None, anthropic_client=None,
                             temperature=1.0, usage=None):
            if "items" in prompt and "+3" in prompt:
                ids_in_prompt = []
                for i in range(20):
                    if f"id={i}" in prompt:
                        ids_in_prompt.append(i)
                # All items get +2 -> mean abs = 2.0
                items = [{"id": i, "score": 2, "reason": "test"}
                         for i in ids_in_prompt]
                return json.dumps({"items": items})
            return json.dumps({"score": 2, "reason": "test"})

        monkeypatch.setattr(sj, "call_judge_single_unified", fake_call)

        records = [
            _make_record(0, coh=0, strength=2.0, sign=+1),
            _make_record(1, coh=0, strength=2.0, sign=+1),
        ]

        # Step 1 (race-trigger order): call eff-async FIRST.
        eff_fut_eager = dispatcher.judge_effect_for_strength_async(records)
        # Future is pending (not pre-resolved to NaN) -- the eager
        # registration path stashed it in _effect_mean_futures.
        assert not eff_fut_eager.done(), (
            "eff future should be pending after eager register, not NaN"
        )

        # Step 2: NOW call enqueue_strength_group (simulating the
        # coh-future done callback firing).
        dispatcher.enqueue_strength_group(
            cell_dir=str(tmp_path), slot=3, layer=25, sign=+1,
            strength=2.0, records=records,
        )

        # Step 3: drain async work, then check the future resolved
        # to a real (non-NaN) mean.
        dispatcher.drain(timeout_s=10.0)
        result = eff_fut_eager.result(timeout=5.0)
        assert not math.isnan(result), f"eff future resolved to NaN: {result}"
        # v7 (2026-05-16): the dispatcher fires BOTH straight and swap
        # rubrics.  This test's fake_call returns the same +2 score
        # regardless of which rubric variant ran, so straight.mean = +2
        # and swap.mean = +2.  Averaged = (straight - swap) / 2 = 0.
        # combined for each record is the averaged value, so
        # mean(|combined|) = 0.0.  (Pre-v7 this test asserted 2.0 --
        # that was the straight-mean, not the bias-cancelled signal.)
        assert result == pytest.approx(0.0, abs=1e-6)

        # A second call should return the SAME future object (idempotent).
        eff_fut_again = dispatcher.judge_effect_for_strength_async(records)
        assert eff_fut_again is eff_fut_eager, (
            "second eager call returned a different Future object"
        )

    def test_rerun_existing_effect_does_not_trip_over_prior_mean(
            self, dispatcher, tmp_path, monkeypatch):
        """Regression: re-judging effect over records that already have an
        effect dict (with a 'mean' float left from the previous run) used
        to crash with 'float object has no attribute get' because the
        mean-computation iteration treated the prior mean float as a
        per-model entry.  Fix: skip the 'mean' key explicitly when
        recomputing.
        """
        # Build a record with PREVIOUSLY-populated effect in the v6
        # schema (bidirectional.scores directly, no straight/swap
        # sub-blocks).  Re-run should both migrate the schema to v7
        # AND overwrite the per-model scores without tripping on the
        # legacy float ``mean``.
        prior = {
            "mode": "bidirectional",
            "bidirectional": {
                "scores": {
                    "gpt-4.1-mini": {"score": 1, "reason": "old"},
                    "claude-haiku-4-5-20251001": {"score": None,
                                                   "reason": "UNPARSEABLE"},
                    "mean": 1.0,
                },
            },
            "combined": 1.0,
            "ts": 0.0,
            "rubric_version": 6,
            "skipped_due_to_strength_mean_coh": False,
        }
        rec = _make_record(0, coh=0)
        rec["judges"]["effect"] = prior

        async def fake_call(*, model, prompt, max_tokens, rate_limiter,
                             openai_client=None, anthropic_client=None,
                             temperature=1.0, usage=None):
            # Return a valid bidirectional batch payload.
            return json.dumps({
                "items": [{"id": 0, "score": 2, "reason": "new"}]
            })

        # Re-run: dispatcher with rerun_existing=True
        monkeypatch.setattr(sj, "call_judge_single_unified", fake_call)
        dispatcher.rerun_existing = True
        dispatcher.enqueue_strength_group(
            cell_dir=str(tmp_path), slot=3, layer=25, sign=+1,
            strength=1.0, records=[rec],
        )
        dispatcher.drain(timeout_s=10.0)

        # Should NOT have crashed.  Migration to v7 happened, so
        # per-model scores now live under bidirectional.straight.scores
        # (and the dispatcher additionally populates bidirectional.swap
        # because v7 fires both rubrics).
        eff = rec["judges"]["effect"]
        bid = eff["bidirectional"]
        straight = bid["straight"]["scores"]
        assert straight["gpt-4.1-mini"]["score"] == 2
        assert straight["mean"] == 2.0
        # Swap half also populated by the dispatcher's v7 path.
        swap = bid["swap"]["scores"]
        assert swap["gpt-4.1-mini"]["score"] == 2
        # Both halves returning the same fake score => averaged = 0
        # ((straight - swap) / 2) -- exactly what content-driven
        # judging would yield if straight and swap genuinely agreed.
        assert bid["averaged"] == 0.0

    def test_strength_mean_coh_stamped_on_disk(self, dispatcher, tmp_path):
        records = [_make_record(i, coh=2) for i in range(2)]
        dispatcher.enqueue_strength_group(
            cell_dir=str(tmp_path), slot=3, layer=25, sign=+1,
            strength=1.0, records=records,
        )
        # The dispatcher merged updates to records.jsonl
        on_disk = (tmp_path / "records.jsonl").read_text().splitlines()
        on_disk = [json.loads(l) for l in on_disk if l.strip()]
        assert len(on_disk) == 2
        for r in on_disk:
            assert r["judges"]["strength_mean_coh"] == pytest.approx(2.0)


class TestSchemaMigrationV6ToV7:
    """v7 (2026-05-16) renames bidirectional.scores -> bidirectional.straight.scores
    and adds bidirectional.swap + bidirectional.averaged.  Old records
    on disk stay readable; combined falls back through three layers.
    """

    def test_migrate_v6_legacy_to_v7(self):
        """A pure v6 effect dict gains a ``straight`` sub-block whose
        ``scores`` is the old ``scores`` dict, with the legacy
        top-level ``scores`` key removed (so neither schema's reader
        sees both copies)."""
        eff = {
            "bidirectional": {
                "scores": {
                    "gpt-4.1-mini": {"score": 2, "reason": "ok"},
                    "mean": 2.0,
                },
            },
            "mode": "bidirectional",
            "combined": 2.0,
        }
        changed = sj.migrate_effect_dict_in_place(eff)
        assert changed is True
        bidir = eff["bidirectional"]
        assert "scores" not in bidir, \
            "legacy 'scores' key must be removed after migration"
        assert bidir["straight"]["scores"]["gpt-4.1-mini"]["score"] == 2
        assert bidir["straight"]["scores"]["mean"] == 2.0

    def test_migrate_v7_idempotent(self):
        """Already-v7 dicts (have straight/swap/averaged) are untouched."""
        eff = {
            "bidirectional": {
                "straight": {"scores": {"gpt": {"score": 1}, "mean": 1.0}},
                "swap": {"scores": {"gpt": {"score": -1}, "mean": -1.0}},
                "averaged": 1.0,
            },
        }
        before = json.dumps(eff, sort_keys=True)
        changed = sj.migrate_effect_dict_in_place(eff)
        assert changed is False
        assert json.dumps(eff, sort_keys=True) == before

    def test_migrate_no_bidirectional_section(self):
        """Effect dicts without a bidirectional sub-block (e.g., pure
        separate_poles mode) are untouched and return False."""
        eff = {
            "separate_poles": {
                "pos": {"gpt": {"score": 2}, "mean": 2.0},
                "neg": {"gpt": {"score": 0}, "mean": 0.0},
            },
            "mode": "separate_poles",
        }
        before = json.dumps(eff, sort_keys=True)
        assert sj.migrate_effect_dict_in_place(eff) is False
        assert json.dumps(eff, sort_keys=True) == before

    def test_averaged_estimator_math(self):
        """Bias-cancelling estimator: (straight.mean - swap.mean) / 2.

        - Content-driven judge: swap = -straight => averaged == straight.
        - Pure label bias: swap = straight => averaged == 0.
        """
        bidir = {
            "straight": {"scores": {"gpt": {"score": 2}, "mean": 2.0}},
            "swap": {"scores": {"gpt": {"score": -2}, "mean": -2.0}},
        }
        # Content-driven: averaged passes signal through.
        assert sj._averaged_eff_from_bidir(bidir) == 2.0

        bidir["swap"]["scores"]["mean"] = 2.0
        # Pure label bias: averaged collapses to 0.
        assert sj._averaged_eff_from_bidir(bidir) == 0.0

        # Mixed signal + bias: (1 - (-2))/2 = 1.5.
        bidir["straight"]["scores"]["mean"] = 1.0
        bidir["swap"]["scores"]["mean"] = -2.0
        assert sj._averaged_eff_from_bidir(bidir) == 1.5

    def test_averaged_estimator_returns_none_when_half_missing(self):
        # Swap absent.
        bidir = {"straight": {"scores": {"mean": 1.0}}}
        assert sj._averaged_eff_from_bidir(bidir) is None
        # Straight absent.
        bidir = {"swap": {"scores": {"mean": 1.0}}}
        assert sj._averaged_eff_from_bidir(bidir) is None
        # Means are None.
        bidir = {
            "straight": {"scores": {"mean": None}},
            "swap": {"scores": {"mean": 1.0}},
        }
        assert sj._averaged_eff_from_bidir(bidir) is None

    def test_compute_combined_prefers_averaged(self):
        """Three-layer fallback: averaged > straight.scores.mean > legacy."""
        # v7 with averaged.
        eff = {
            "bidirectional": {
                "straight": {"scores": {"mean": 1.5}},
                "swap": {"scores": {"mean": -1.5}},
                "averaged": 1.5,
            },
        }
        assert sj.RealJudgeDispatcher._compute_combined(
            eff, sj.EFFECT_MODE_BIDIRECTIONAL
        ) == 1.5

        # v7 with swap-half missing (averaged is None) -> straight fallback.
        eff = {
            "bidirectional": {
                "straight": {"scores": {"mean": 1.5}},
            },
        }
        assert sj.RealJudgeDispatcher._compute_combined(
            eff, sj.EFFECT_MODE_BIDIRECTIONAL
        ) == 1.5

        # Legacy v6 -> falls back to bidirectional.scores.mean.
        eff = {
            "bidirectional": {"scores": {"mean": 2.5}},
        }
        assert sj.RealJudgeDispatcher._compute_combined(
            eff, sj.EFFECT_MODE_BIDIRECTIONAL
        ) == 2.5


class TestRealDispatcherBatching:
    """The dispatcher reuses plan_response_batches semantics for effect."""

    def test_plan_batches_22_questions_target_10(self, dispatcher):
        records = [_make_record(i) for i in range(22)]
        batches = dispatcher._plan_batches(records)
        # round(22/10) = 2 batches
        assert len(batches) == 2
        # near-equal sizes; both 11
        assert all(len(b) == 11 for b in batches)
        # ids interleave by question_idx (sorted ascending)
        ids = [r["question_idx"] for batch in batches for r in batch]
        assert ids == list(range(22))

    def test_plan_batches_36_questions_target_10(self, dispatcher):
        records = [_make_record(i) for i in range(36)]
        batches = dispatcher._plan_batches(records)
        # round(36/10) = 4 batches; sizes 9,9,9,9
        assert len(batches) == 4
        assert all(len(b) == 9 for b in batches)

    def test_plan_batches_5_questions_target_10(self, dispatcher):
        records = [_make_record(i) for i in range(5)]
        batches = dispatcher._plan_batches(records)
        # round(5/10) = 0 -> max(1, 0) = 1 batch
        assert len(batches) == 1
        assert len(batches[0]) == 5


# ---------------------------------------------------------------------------
# build_baseline_lookup
# ---------------------------------------------------------------------------

class TestBuildBaselineLookup:
    def test_missing_file_returns_empty_string(self, tmp_path, caplog):
        lookup = build_baseline_lookup(tmp_path / "nope.jsonl")
        assert lookup(0) == ""
        assert lookup(99) == ""

    def test_lookup_by_question_idx(self, tmp_path):
        path = tmp_path / "baselines.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"question_idx": 0, "response": "first"}) + "\n")
            f.write(json.dumps({"question_idx": 7, "response": "seventh"}) + "\n")
        lookup = build_baseline_lookup(path)
        assert lookup(0) == "first"
        assert lookup(7) == "seventh"
        assert lookup(99) == ""  # missing -> empty string


# ---------------------------------------------------------------------------
# Judge-outcome tally + summary
# ---------------------------------------------------------------------------

class TestJudgeOutcomeSummary:
    """The May 2026 audit found that ``gpt-4.1-mini`` was failing to
    produce parseable JSON on ~17-31% of effect-batch calls in
    production smoke_test_v3 records, silently degrading the
    bidirectional ensemble to single-judge for those records.
    Nothing in run.log surfaced this -- only a per-record dig into
    the ``UNPARSEABLE`` reason field would have caught it.  The
    tally + summary makes that failure mode loud at end-of-run.
    """

    def test_empty_tally_returns_empty_string(self, dispatcher):
        # No judging has happened yet.
        assert dispatcher.judge_outcome_summary() == ""

    def test_records_ok_outcomes(self, dispatcher):
        dispatcher._record_judge_outcome(
            kind="coherence", model="gpt-4.1-mini", outcome="ok",
        )
        dispatcher._record_judge_outcome(
            kind="coherence", model="gpt-4.1-mini", outcome="ok",
        )
        s = dispatcher.judge_outcome_summary()
        assert "coherence / gpt-4.1-mini" in s
        assert "2/2 OK" in s
        assert "UNPARSEABLE" not in s

    def test_records_unparseable_outcomes(self, dispatcher):
        for _ in range(7):
            dispatcher._record_judge_outcome(
                kind="effect", model="gpt-4.1-mini", outcome="ok",
            )
        for _ in range(3):
            dispatcher._record_judge_outcome(
                kind="effect", model="gpt-4.1-mini", outcome="unparseable",
            )
        s = dispatcher.judge_outcome_summary()
        assert "effect / gpt-4.1-mini" in s
        assert "7/10 OK" in s
        assert "3/10 = 30% UNPARSEABLE" in s
        # 30% > 10% threshold -> loud marker
        assert "*** HIGH FAIL RATE ***" in s

    def test_loud_marker_threshold(self, dispatcher):
        # Project rule (May 2026): warn loudly when OK rate falls below
        # 99% (i.e. UNPARSEABLE >= 1%).  100/100 OK -> quiet.
        for _ in range(100):
            dispatcher._record_judge_outcome(
                kind="effect", model="claude-haiku-4-5-20251001", outcome="ok",
            )
        s = dispatcher.judge_outcome_summary()
        assert "100/100 OK" in s
        assert "*** HIGH FAIL RATE ***" not in s

    def test_loud_marker_at_one_percent(self, dispatcher):
        # 1% UNPARSEABLE: at threshold, loud marker fires.
        for _ in range(99):
            dispatcher._record_judge_outcome(
                kind="effect", model="claude-haiku-4-5-20251001", outcome="ok",
            )
        for _ in range(1):
            dispatcher._record_judge_outcome(
                kind="effect", model="claude-haiku-4-5-20251001", outcome="unparseable",
            )
        s = dispatcher.judge_outcome_summary()
        assert "1/100 = 1% UNPARSEABLE" in s
        assert "*** HIGH FAIL RATE ***" in s

    def test_separate_tally_per_kind_and_model(self, dispatcher):
        dispatcher._record_judge_outcome(kind="coherence", model="gpt-4.1-mini", outcome="ok")
        dispatcher._record_judge_outcome(kind="persona", model="gpt-4.1-mini", outcome="ok")
        dispatcher._record_judge_outcome(kind="effect", model="gpt-4.1-mini", outcome="unparseable")
        dispatcher._record_judge_outcome(kind="effect", model="claude-haiku-4-5-20251001", outcome="ok")
        s = dispatcher.judge_outcome_summary()
        # Four distinct (kind, model) pairs reported separately.
        assert "coherence / gpt-4.1-mini" in s
        assert "persona / gpt-4.1-mini" in s
        assert "effect / gpt-4.1-mini" in s
        assert "effect / claude-haiku-4-5-20251001" in s

    def test_missing_outcomes_reported(self, dispatcher):
        dispatcher._record_judge_outcome(kind="effect", model="gpt-4.1-mini", outcome="ok")
        dispatcher._record_judge_outcome(kind="effect", model="gpt-4.1-mini", outcome="missing")
        s = dispatcher.judge_outcome_summary()
        assert "1/2 MISSING" in s


# ---------------------------------------------------------------------------
# Token-usage tracking
# ---------------------------------------------------------------------------
#
# Every steering judge call goes through RealJudgeDispatcher._call_unified,
# which threads ``usage=self._usage`` into call_judge_single_unified.  The
# response.usage block from the SDK feeds the per-model accumulator, and
# shutdown() writes <cell_dir>/usage.json.  We exercise the wiring with a
# stub that fakes a usage block; the actual extract_usage_* helpers are
# covered by test_judge_pricing.py.


class TestUsageTracking:
    def test_dispatcher_constructs_with_empty_usage(self, dispatcher):
        # Fresh dispatcher: no calls have happened.
        assert dispatcher._usage.n_calls == 0
        assert dispatcher._usage.total_cost_usd == 0.0

    def test_call_unified_passes_usage_through(
        self, dispatcher, monkeypatch,
    ):
        """The dispatcher's own usage tracker is threaded into every
        call_judge_single_unified invocation."""
        captured = {}

        async def fake_call(*, model, prompt, max_tokens, rate_limiter,
                             openai_client=None, anthropic_client=None,
                             temperature=1.0, usage=None):
            captured["usage_is_dispatcher_tracker"] = usage is dispatcher._usage
            captured["model"] = model
            # Simulate the SDK having extracted tokens from a response.
            if usage is not None:
                usage.charge(model, prompt_tokens=1_000, completion_tokens=200)
            return '{"score": 0, "reason": "stub"}'

        monkeypatch.setattr(sj, "call_judge_single_unified", fake_call)

        # Drive _call_unified once via the dispatcher's event loop.
        fut = asyncio.run_coroutine_threadsafe(
            dispatcher._call_unified(
                model="gpt-4.1-mini", prompt="hello", max_tokens=10,
            ),
            dispatcher._loop,
        )
        result = fut.result(timeout=5)

        assert result == '{"score": 0, "reason": "stub"}'
        assert captured["usage_is_dispatcher_tracker"] is True
        assert dispatcher._usage.n_calls == 1
        # 1,000 input tokens at $0.40/M = $0.0004; 200 output at $1.60/M = $0.00032
        assert dispatcher._usage.total_cost_usd == pytest.approx(0.00072)

    def test_shutdown_writes_usage_json(self, tmp_path, mock_clients,
                                        persona, steering, monkeypatch):
        """At end-of-cell, the dispatcher serialises its usage tracker
        to <cell_dir>/usage.json (next to records.jsonl)."""
        records_path = tmp_path / "records.jsonl"
        records_path.write_text("")
        d = RealJudgeDispatcher(
            records_path=records_path,
            persona=persona, steering=steering,
            baseline_lookup=lambda q_idx: f"baseline-{q_idx}",
            coherence_model="gpt-4.1-mini",
            rp_model="gpt-4.1-mini",
            effect_models=("gpt-4.1-mini", "claude-haiku-4-5-20251001"),
            effect_mode=EFFECT_MODE_BIDIRECTIONAL,
            effect_target_batch_size=10,
            skip_threshold=1.0,
            coh_stop_threshold=1.5,
            max_concurrency=4,
        )
        try:
            # Pretend we made a couple of API calls.
            d._usage.charge("gpt-4.1-mini", 5_000, 1_000)
            d._usage.charge("claude-haiku-4-5-20251001", 8_000, 1_500)
        finally:
            d.shutdown()

        usage_path = tmp_path / "usage.json"
        assert usage_path.exists(), "shutdown() should have written usage.json"
        payload = json.loads(usage_path.read_text())
        assert "per_model" in payload
        assert "gpt-4.1-mini" in payload["per_model"]
        assert "claude-haiku-4-5-20251001" in payload["per_model"]
        assert payload["n_calls"] == 2
        assert payload["total_cost_usd"] > 0

    def test_shutdown_no_calls_no_file(self, tmp_path, mock_clients,
                                       persona, steering):
        """No file written when no API calls happened (avoids polluting
        cell dirs in --no-live-judging runs)."""
        records_path = tmp_path / "records.jsonl"
        records_path.write_text("")
        d = RealJudgeDispatcher(
            records_path=records_path,
            persona=persona, steering=steering,
            baseline_lookup=lambda q_idx: "",
            coherence_model="gpt-4.1-mini",
            rp_model="gpt-4.1-mini",
            effect_models=("gpt-4.1-mini", "claude-haiku-4-5-20251001"),
            effect_mode=EFFECT_MODE_BIDIRECTIONAL,
            effect_target_batch_size=10,
            skip_threshold=1.0,
            coh_stop_threshold=1.5,
            max_concurrency=4,
        )
        d.shutdown()
        assert not (tmp_path / "usage.json").exists()

    def test_resume_merges_historic_usage(self, tmp_path, mock_clients,
                                          persona, steering):
        """A second dispatcher constructed against a cell that already
        has a usage.json picks up where the first left off, so the
        persisted file accumulates across resume cycles rather than
        overwriting.
        """
        records_path = tmp_path / "records.jsonl"
        records_path.write_text("")
        usage_path = tmp_path / "usage.json"

        # First invocation: charge some tokens and shut down.
        d1 = RealJudgeDispatcher(
            records_path=records_path,
            persona=persona, steering=steering,
            baseline_lookup=lambda q_idx: "",
            coherence_model="gpt-4.1-mini",
            rp_model="gpt-4.1-mini",
            effect_models=("gpt-4.1-mini",),
            effect_mode=EFFECT_MODE_BIDIRECTIONAL,
            effect_target_batch_size=10,
            skip_threshold=1.0,
            coh_stop_threshold=1.5,
            max_concurrency=4,
        )
        try:
            d1._usage.charge("gpt-4.1-mini", 1_000, 200)
        finally:
            d1.shutdown()
        assert usage_path.exists()

        # Second invocation: should load the historic totals on init.
        d2 = RealJudgeDispatcher(
            records_path=records_path,
            persona=persona, steering=steering,
            baseline_lookup=lambda q_idx: "",
            coherence_model="gpt-4.1-mini",
            rp_model="gpt-4.1-mini",
            effect_models=("gpt-4.1-mini",),
            effect_mode=EFFECT_MODE_BIDIRECTIONAL,
            effect_target_batch_size=10,
            skip_threshold=1.0,
            coh_stop_threshold=1.5,
            max_concurrency=4,
        )
        try:
            # Pre-shutdown, d2._usage already reflects d1's tokens.
            assert d2._usage.per_model["gpt-4.1-mini"].prompt_tokens == 1_000
            assert d2._usage.per_model["gpt-4.1-mini"].n_calls == 1
            # Add more.
            d2._usage.charge("gpt-4.1-mini", 500, 100)
        finally:
            d2.shutdown()

        # Final file has the SUM of both runs.
        payload = json.loads(usage_path.read_text())
        sub = payload["per_model"]["gpt-4.1-mini"]
        assert sub["prompt_tokens"] == 1_500
        assert sub["completion_tokens"] == 300
        assert sub["n_calls"] == 2

    def test_judge_usage_summary_format(self, dispatcher):
        """Single-line summary is suitable for end-of-run logs."""
        dispatcher._usage.charge("gpt-4.1-mini", 100, 50)
        dispatcher._usage.charge("claude-haiku-4-5-20251001", 200, 100)
        s = dispatcher.judge_usage_summary()
        assert "[steering judge usage]" in s
        assert "gpt-4.1-mini" in s
        assert "claude-haiku-4-5-20251001" in s
        assert "total=$" in s
