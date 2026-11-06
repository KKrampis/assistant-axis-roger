"""Unit tests for results_analysis/infer_axis_description.py.

Covers the pure (no-API) helpers: input normalization, z-scoring, glossary
loading, label-map resolution, prompt building, and response parsing. The
network call to Anthropic is exercised via a mock in test_summarize_axis_e2e.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from results_analysis.infer_axis_description import (
    DEFAULT_INSTRUCTIONS_DIR,
    LabelMap,
    _call_opus,
    _format_z,
    _normalize_type,
    _resolve_examples,
    _validate_and_normalize_scores,
    _z_normalize,
    build_prompt,
    parse_response,
    summarize_axis,
)


# ---------------------------------------------------------------------------
# Fixtures: a small set of entities that actually exist in data/
# ---------------------------------------------------------------------------

REAL_ENTITIES = [
    {"name": "helpful",             "type": "trait", "score":  3.2},
    {"name": "advocate",            "type": "role",  "score":  2.7},
    {"name": "guileless",           "type": "trait", "score":  1.8},
    {"name": "systems_thinker",     "type": "trait", "score":  0.5},
    {"name": "warrior",             "type": "role",  "score": -0.3},
    {"name": "paperclip_maximizer", "type": "R",     "score": -1.4},
    {"name": "deceitful",           "type": "T",     "score": -2.1},
    {"name": "unhelpful",           "type": "trait", "score": -3.5},
]


@pytest.fixture
def label_map() -> LabelMap:
    z = _z_normalize(_validate_and_normalize_scores(REAL_ENTITIES))
    return LabelMap(z, instructions_dir=DEFAULT_INSTRUCTIONS_DIR)


# ---------------------------------------------------------------------------
# Type normalization
# ---------------------------------------------------------------------------

class TestTypeNormalization:
    def test_aliases(self):
        for s in ("R", "r", "role", "Role", "ROLES"):
            assert _normalize_type(s) == "role"
        for s in ("T", "t", "trait", "Trait", "TRAITS"):
            assert _normalize_type(s) == "trait"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _normalize_type("entity")


# ---------------------------------------------------------------------------
# z-normalization
# ---------------------------------------------------------------------------

class TestZNormalization:
    def test_validate_and_normalize_scores_dedupes(self):
        rows = [
            {"name": "a", "type": "trait", "score": 1.0},
            {"name": "a", "type": "trait", "score": 2.0},  # dup; last wins
            {"name": "b", "type": "trait", "score": 3.0},
            {"name": "c", "type": "trait", "score": 4.0},
            {"name": "d", "type": "trait", "score": 5.0},
            {"name": "e", "type": "trait", "score": 6.0},
        ]
        out = _validate_and_normalize_scores(rows)
        assert ("a", "trait", 2.0) in out
        assert len(out) == 5

    def test_z_normalize_mean_zero_sd_one(self):
        rows = [(f"x{i}", "trait", float(i)) for i in range(10)]
        z = _z_normalize(rows)
        zs = [v for _, _, v in z]
        assert abs(sum(zs) / len(zs)) < 1e-9
        # std (ddof=0) should be 1
        mean = sum(zs) / len(zs)
        var = sum((x - mean) ** 2 for x in zs) / len(zs)
        assert abs(var - 1.0) < 1e-9

    def test_z_normalize_constant_input_raises(self):
        rows = [(f"x{i}", "trait", 1.0) for i in range(10)]
        with pytest.raises(ValueError, match="z-normalize"):
            _z_normalize(rows)

    def test_validate_min_size(self):
        with pytest.raises(ValueError, match="at least 5"):
            _validate_and_normalize_scores([
                {"name": "a", "type": "trait", "score": 1.0},
                {"name": "b", "type": "trait", "score": 2.0},
            ])


# ---------------------------------------------------------------------------
# Display label conversion
# ---------------------------------------------------------------------------

class TestDisplayLabels:
    def test_trait_uses_positive_label(self, label_map):
        # systems_thinker is hyphenated in its positive_label
        st = next(e for e in label_map.entries if e["name"] == "systems_thinker")
        assert st["label"] == "systems-thinker"
        assert st["etype"] == "trait"

    def test_role_underscore_to_space(self, label_map):
        # paperclip_maximizer is a role; no positive_label, so _ -> space
        pm = next(e for e in label_map.entries if e["name"] == "paperclip_maximizer")
        assert pm["label"] == "paperclip maximizer"
        assert pm["etype"] == "role"

    def test_single_word_unchanged(self, label_map):
        h = next(e for e in label_map.entries if e["name"] == "helpful")
        assert h["label"] == "helpful"
        w = next(e for e in label_map.entries if e["name"] == "warrior")
        assert w["label"] == "warrior"


# ---------------------------------------------------------------------------
# LabelMap.filename_for_label
# ---------------------------------------------------------------------------

class TestLabelMapResolution:
    def test_exact_filename(self, label_map):
        assert label_map.filename_for_label("paperclip_maximizer") == "paperclip_maximizer"

    def test_display_label(self, label_map):
        assert label_map.filename_for_label("paperclip maximizer") == "paperclip_maximizer"
        assert label_map.filename_for_label("systems-thinker") == "systems_thinker"

    def test_fuzzy_punctuation_and_case(self, label_map):
        # Stray punctuation, mixed case
        assert label_map.filename_for_label("SYSTEMS_THINKER") == "systems_thinker"
        assert label_map.filename_for_label("Paperclip-Maximizer") == "paperclip_maximizer"

    def test_unknown_returns_none(self, label_map):
        assert label_map.filename_for_label("NotInList") is None


# ---------------------------------------------------------------------------
# z formatting
# ---------------------------------------------------------------------------

class TestFormatZ:
    def test_positive(self):
        assert _format_z(1.34) == "+1.3"
        assert _format_z(0.05) == "+0.1"  # round-half-to-even gives 0.0 sometimes

    def test_zero(self):
        assert _format_z(0.0) == "+0.0"
        assert _format_z(-0.04) == "+0.0"  # rounds to 0; we explicitly avoid -0.0

    def test_negative(self):
        assert _format_z(-2.55) == "-2.5"  # banker's rounding


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_glossary_style_contains_both_blocks(self, label_map):
        prompt = build_prompt(label_map, style="glossary")
        assert "--- ranking ---" in prompt
        assert "--- glossary ---" in prompt
        # All entities appear in both ranking and glossary
        for e in label_map.entries:
            assert e["label"] in prompt
        # Schema instructions present
        assert "<axis_name>" in prompt
        assert "<pos_examples>" in prompt

    def test_inline_style_has_descriptions_in_ranking(self, label_map):
        prompt = build_prompt(label_map, style="inline")
        assert "--- ranking ---" in prompt
        assert "--- glossary ---" not in prompt
        # Description text should be inline
        for e in label_map.entries:
            assert e["description"][:30] in prompt

    def test_ranking_is_descending(self, label_map):
        prompt = build_prompt(label_map, style="glossary")
        # helpful (z=+1.4) should precede unhelpful (z=-1.6)
        i_pos = prompt.index("helpful")
        i_neg = prompt.index("unhelpful")
        assert i_pos < i_neg

    def test_unknown_style_raises(self, label_map):
        with pytest.raises(ValueError):
            build_prompt(label_map, style="banana")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

CANNED_RESPONSE = """\
Some preamble the model emitted before the tags. Should be ignored.
<axis_name>prosocial vs antisocial</axis_name>
<pos_pole>Concepts at this end are oriented toward helping and being transparent with others.</pos_pole>
<neg_pole>Concepts at this end are oriented toward harm or single-minded goal pursuit at others' expense.</neg_pole>
<pos_examples>helpful, advocate, guileless, systems-thinker</pos_examples>
<neg_examples>unhelpful, deceitful, paperclip maximizer, NotInList</neg_examples>
A concluding remark, also ignored.
"""


class TestParseResponse:
    def test_extracts_all_tags(self):
        parsed = parse_response(CANNED_RESPONSE)
        assert parsed["axis_name"] == "prosocial vs antisocial"
        assert parsed["pos_pole"].startswith("Concepts at this end")
        assert parsed["neg_pole"].startswith("Concepts at this end")
        assert parsed["pos_examples"] == ["helpful", "advocate", "guileless", "systems-thinker"]
        assert parsed["neg_examples"] == ["unhelpful", "deceitful", "paperclip maximizer", "NotInList"]

    def test_missing_tag_raises(self):
        with pytest.raises(ValueError, match="missing/empty"):
            parse_response("<axis_name>x</axis_name>")  # missing the rest


# ---------------------------------------------------------------------------
# Example resolution (post-parse)
# ---------------------------------------------------------------------------

class TestResolveExamples:
    def test_resolves_mixed_label_forms(self, label_map):
        parsed = parse_response(CANNED_RESPONSE)
        resolved_pos = _resolve_examples(parsed["pos_examples"], label_map, "pos_examples")
        resolved_neg = _resolve_examples(parsed["neg_examples"], label_map, "neg_examples")
        assert resolved_pos == ["helpful", "advocate", "guileless", "systems_thinker"]
        # 'NotInList' is dropped with a warning
        assert resolved_neg == ["unhelpful", "deceitful", "paperclip_maximizer"]


# ---------------------------------------------------------------------------
# End-to-end with mocked Anthropic call
# ---------------------------------------------------------------------------

class TestSummarizeAxisE2E:
    def test_full_pipeline_with_mock(self, monkeypatch):
        # Pretend we have a key set so we don't hit the EnvironmentError guard.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

        async def fake_call_opus(prompt, *, model, thinking_budget, max_tokens):
            # Sanity: the prompt should mention at least one entity in display form
            assert "helpful" in prompt
            assert "paperclip maximizer" in prompt  # role got space-substituted
            return CANNED_RESPONSE

        with patch(
            "results_analysis.infer_axis_description._call_opus",
            new=AsyncMock(side_effect=fake_call_opus),
        ):
            result = summarize_axis(
                scores=REAL_ENTITIES,
                instructions_dir=DEFAULT_INSTRUCTIONS_DIR,
                style="glossary",
                thinking_budget=0,         # avoid the max-tokens guard
                max_tokens=2_000,
            )

        assert result["axis_name"] == "prosocial vs antisocial"
        assert result["pos_examples"] == ["helpful", "advocate", "guileless", "systems_thinker"]
        # filename format on output, not the display labels
        assert "paperclip_maximizer" in result["neg_examples"]
        assert "paperclip maximizer" not in result["neg_examples"]
        # Schema keys exhaustively
        assert set(result.keys()) == {
            "axis_name", "pos_pole", "neg_pole", "pos_examples", "neg_examples"
        }


# ---------------------------------------------------------------------------
# Streaming integration: verify _call_opus uses messages.stream(...) correctly
# ---------------------------------------------------------------------------

def _build_mock_anthropic_client(captured_kwargs: dict, content_blocks: list):
    """Build a mock AsyncAnthropic whose messages.stream(**kw) is an async
    context manager yielding a stream object whose get_final_message() returns
    a Message with the given content blocks.  Captures the kwargs passed to
    .stream() into ``captured_kwargs`` for assertion.
    """
    final_message = SimpleNamespace(content=content_blocks)

    mock_stream = MagicMock()
    mock_stream.get_final_message = AsyncMock(return_value=final_message)

    class FakeStreamCtx:
        async def __aenter__(self):
            return mock_stream

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def stream_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeStreamCtx()

    mock_messages = SimpleNamespace(stream=stream_factory)
    mock_client = SimpleNamespace(messages=mock_messages)
    return mock_client


class TestCallOpusStreaming:
    """The streaming path is what unblocks long thinking budgets that would
    otherwise hit Anthropic's 10-min non-streaming timeout.  These tests verify
    the wiring at the messages.stream() boundary.
    """

    def test_streaming_call_concatenates_text_blocks_skips_thinking(self, monkeypatch):
        import anthropic
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

        captured: dict = {}
        content_blocks = [
            SimpleNamespace(type="thinking", thinking="this is reasoning we ignore"),
            SimpleNamespace(type="text",     text="part one "),
            SimpleNamespace(type="thinking", thinking="more reasoning we ignore"),
            SimpleNamespace(type="text",     text="part two"),
        ]
        mock_client = _build_mock_anthropic_client(captured, content_blocks)
        monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda: mock_client)

        result = asyncio.run(_call_opus(
            prompt="hello",
            model="claude-opus-4-6",
            thinking_budget=5000,
            max_tokens=10000,
        ))

        # Text blocks concatenated in order; thinking blocks dropped.
        assert result == "part one part two"
        # Streaming kwargs include the thinking config and temperature=1.
        assert captured["model"] == "claude-opus-4-6"
        assert captured["max_tokens"] == 10000
        assert captured["thinking"] == {"type": "enabled", "budget_tokens": 5000}
        assert captured["temperature"] == 1.0
        # System prompt and message both present.
        assert "system" in captured
        assert captured["messages"] == [{"role": "user", "content": "hello"}]

    def test_streaming_with_thinking_disabled_uses_temperature_zero(self, monkeypatch):
        import anthropic
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

        captured: dict = {}
        content_blocks = [SimpleNamespace(type="text", text="deterministic output")]
        mock_client = _build_mock_anthropic_client(captured, content_blocks)
        monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda: mock_client)

        result = asyncio.run(_call_opus(
            prompt="hello",
            model="claude-opus-4-6",
            thinking_budget=0,
            max_tokens=2000,
        ))

        assert result == "deterministic output"
        assert captured["temperature"] == 0.0
        # No thinking config when budget is 0.
        assert "thinking" not in captured

    def test_streaming_raises_when_no_text_blocks_returned(self, monkeypatch):
        import anthropic
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

        captured: dict = {}
        # Only thinking blocks, no text -- pathological but possible if the
        # model exhausted its budget without producing visible output.
        content_blocks = [
            SimpleNamespace(type="thinking", thinking="endless rumination"),
        ]
        mock_client = _build_mock_anthropic_client(captured, content_blocks)
        monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda: mock_client)

        with pytest.raises(RuntimeError, match="no text content blocks"):
            asyncio.run(_call_opus(
                prompt="hello",
                model="claude-opus-4-6",
                thinking_budget=5000,
                max_tokens=10000,
            ))

    def test_streaming_max_tokens_must_exceed_thinking_budget(self, monkeypatch):
        import anthropic
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

        # No need for a working mock -- guard fires before the API is touched.
        with pytest.raises(ValueError, match="must exceed thinking_budget"):
            asyncio.run(_call_opus(
                prompt="hello",
                model="claude-opus-4-6",
                thinking_budget=5000,
                max_tokens=4000,  # less than thinking_budget
            ))
