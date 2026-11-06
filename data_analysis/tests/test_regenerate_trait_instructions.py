"""Tests for regenerate_trait_instructions.py.

Mocks the Anthropic client to verify merge logic: eval_prompt and labels must
be preserved, instruction (and optionally questions) must be replaced.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import data_analysis.regenerate_trait_instructions as module

from data_analysis.regenerate_trait_instructions import (
    _extract_step_one,
    _replace_step_one,
    _CHRISTINA_TEMPLATE,
    _ROGER_TEMPLATE,
    atomic_write_json,
    build_christina_instruction_prompt,
    build_roger_instruction_prompt,
    build_eval_prompt,
    extract_definition,
    regenerate_one,
    strip_markdown_fences,
)

SAMPLE_EVAL_PROMPT = (
    "You are evaluating whether the model's response displays the trait: "
    "**arrogant**. This involves showing excessive confidence.\n\n"
    "Prompt:\n[QUESTION START]\n{question}\n[QUESTION END]"
)

SAMPLE_DESCRIPTION = "This involves showing excessive confidence."

SAMPLE_TRAIT = {
    "positive_label": "arrogant",
    "negative_label": "humble",
    "description": SAMPLE_DESCRIPTION,
    "instruction": [
        {"pos": "Old pos 1.", "neg": "Old neg 1."},
        {"pos": "Old pos 2.", "neg": "Old neg 2."},
        {"pos": "Old pos 3.", "neg": "Old neg 3."},
        {"pos": "Old pos 4.", "neg": "Old neg 4."},
        {"pos": "Old pos 5.", "neg": "Old neg 5."},
    ],
    "questions": [f"Old question {i}?" for i in range(40)],
    "eval_prompt": SAMPLE_EVAL_PROMPT,
}

FAKE_INSTRUCTIONS = [
    {"pos": "New pos 1.", "neg": "New neg 1."},
    {"pos": "New pos 2.", "neg": "New neg 2."},
    {"pos": "New pos 3.", "neg": "New neg 3."},
    {"pos": "New pos 4.", "neg": "New neg 4."},
    {"pos": "New pos 5.", "neg": "New neg 5."},
]

FAKE_QUESTIONS = [f"New question {i}?" for i in range(40)]


def _make_response(text: str) -> MagicMock:
    msg = MagicMock()
    block = MagicMock(type="text", text=text)
    msg.content = [block]
    return msg


@pytest.fixture
def trait_file(tmp_path: Path) -> Path:
    p = tmp_path / "arrogant.json"
    p.write_text(json.dumps(SAMPLE_TRAIT, indent=2))
    return p


FAKE_ROGER_RESPONSE = {
    "instruction": FAKE_INSTRUCTIONS,
    "questions": FAKE_QUESTIONS,
    "eval_prompt": "You are evaluating whether the model's response displays the trait: arrogant.",
}


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.messages.create = AsyncMock(
        return_value=_make_response(json.dumps(FAKE_ROGER_RESPONSE))
    )
    return client


class TestExtractDefinition:
    def test_extracts_definition(self):
        result = extract_definition(SAMPLE_EVAL_PROMPT)
        assert result == "This involves showing excessive confidence."

    def test_returns_empty_on_missing_bold(self):
        assert extract_definition("No bold markers here.") == ""

    def test_returns_empty_on_empty_string(self):
        assert extract_definition("") == ""


class TestBuildEvalPrompt:
    def test_contains_label_and_description(self):
        result = build_eval_prompt("arrogant", "Shows excessive confidence.")
        assert "**arrogant**" in result
        assert "Shows excessive confidence." in result

    def test_preserves_question_answer_placeholders(self):
        result = build_eval_prompt("arrogant", "Shows excessive confidence.")
        assert "{question}" in result
        assert "{answer}" in result

    def test_roundtrip_with_extract(self):
        desc = "This involves showing excessive confidence."
        rebuilt = build_eval_prompt("arrogant", desc)
        extracted = extract_definition(rebuilt)
        assert extracted == desc


class TestStripMarkdownFences:
    def test_strips_json_fence(self):
        assert strip_markdown_fences('```json\n[1,2]\n```') == "[1,2]"

    def test_strips_bare_fence(self):
        assert strip_markdown_fences("```\nfoo\n```") == "foo"

    def test_noop_on_clean_json(self):
        assert strip_markdown_fences('[{"a":1}]') == '[{"a":1}]'


class TestRegenerateOne:
    """Core merge-logic tests."""

    def test_preserves_eval_prompt_and_labels(self, trait_file, mock_client):
        result = asyncio.run(
            regenerate_one(
                mock_client,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        data = json.loads(trait_file.read_text())
        expected_eval = build_eval_prompt("arrogant", SAMPLE_DESCRIPTION)
        assert data["eval_prompt"] == expected_eval
        assert data["positive_label"] == "arrogant"
        assert data["negative_label"] == "humble"
        assert data["description"] == SAMPLE_DESCRIPTION
        assert data["instruction"] == FAKE_INSTRUCTIONS
        assert data["questions"] == FAKE_QUESTIONS

    def test_instructions_only_preserves_questions(self, trait_file):
        client = AsyncMock()
        client.messages.create = AsyncMock(
            return_value=_make_response(json.dumps(FAKE_ROGER_RESPONSE))
        )

        result = asyncio.run(
            regenerate_one(
                client,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=True,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        data = json.loads(trait_file.read_text())
        assert data["instruction"] == FAKE_INSTRUCTIONS
        assert data["questions"] == SAMPLE_TRAIT["questions"]
        assert client.messages.create.call_count == 1

    def test_skip_when_counts_match(self, trait_file, mock_client):
        result = asyncio.run(
            regenerate_one(
                mock_client,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=False,
                dry_run=False,
            )
        )

        assert result.startswith("SKIP")
        data = json.loads(trait_file.read_text())
        assert data["instruction"] == SAMPLE_TRAIT["instruction"]
        mock_client.messages.create.assert_not_called()

    def test_force_overrides_skip(self, trait_file, mock_client):
        result = asyncio.run(
            regenerate_one(
                mock_client,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        data = json.loads(trait_file.read_text())
        assert data["instruction"] == FAKE_INSTRUCTIONS

    def test_dry_run_no_writes(self, trait_file, mock_client):
        original = trait_file.read_text()

        result = asyncio.run(
            regenerate_one(
                mock_client,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=True,
            )
        )

        assert result.startswith("DRY-RUN")
        assert trait_file.read_text() == original
        mock_client.messages.create.assert_not_called()

    def test_missing_labels_skipped(self, tmp_path):
        p = tmp_path / "broken.json"
        p.write_text(json.dumps({"instruction": []}))

        client = AsyncMock()
        result = asyncio.run(
            regenerate_one(
                client,
                p,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("SKIP")
        client.messages.create.assert_not_called()

    def test_preserves_unknown_fields(self, trait_file, mock_client):
        """Extra fields in the JSON are carried through."""
        data = json.loads(trait_file.read_text())
        data["custom_field"] = "keep me"
        trait_file.write_text(json.dumps(data))

        # Reset mock for fresh calls
        mock_client.messages.create = AsyncMock(
            return_value=_make_response(json.dumps(FAKE_ROGER_RESPONSE))
        )

        asyncio.run(
            regenerate_one(
                mock_client,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        data = json.loads(trait_file.read_text())
        assert data["custom_field"] == "keep me"

    def test_key_order(self, trait_file, mock_client):
        """Output keys should be: positive_label, negative_label, description, instruction, questions, eval_prompt."""
        asyncio.run(
            regenerate_one(
                mock_client,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        data = json.loads(trait_file.read_text())
        keys = list(data.keys())
        assert keys[0] == "positive_label"
        assert keys[1] == "negative_label"
        assert keys[2] == "description"
        assert keys[3] == "instruction"
        assert keys[4] == "questions"
        assert keys[5] == "eval_prompt"


class TestBuildChristinaInstructionPrompt:
    def test_substitutes_trait_and_description(self):
        result = build_christina_instruction_prompt(
            "sarcastic", "sincere", "This means using irony.", 5, 40
        )
        assert "<trait>\nsarcastic\n</trait>" in result
        assert "This means using irony." in result

    def test_parameterizes_counts(self):
        result = build_christina_instruction_prompt(
            "sarcastic", "sincere", "Desc.", 3, 20
        )
        assert "list of 3 instruction pairs" in result
        assert "Design 20 questions" in result
        assert "Generate 20 diverse questions" in result
        assert 'question 20' in result

    def test_eval_template_has_single_brace_placeholders(self):
        result = build_christina_instruction_prompt(
            "sarcastic", "sincere", "Desc.", 5, 40
        )
        assert "{TRAIT}" in result
        assert "{question}" in result
        assert "{answer}" in result
        # Should NOT have double braces (those are .format() escapes, not output)
        assert "{{TRAIT}}" not in result

    def test_json_format_has_single_braces(self):
        result = build_christina_instruction_prompt(
            "sarcastic", "sincere", "Desc.", 5, 40
        )
        idx = result.find("<output_format>")
        section = result[idx:]
        assert '{\n  "instruction"' in section
        assert '{"pos":' in section


class TestReplaceStepOne:
    """Tests for --use-original-step-one surgical replacement."""

    def test_extract_step_one_from_both_templates(self):
        roger_s1, _, _ = _extract_step_one(_ROGER_TEMPLATE)
        christina_s1, _, _ = _extract_step_one(_CHRISTINA_TEMPLATE)
        assert roger_s1.startswith("Step 1:")
        assert christina_s1.startswith("Step 1:")
        assert roger_s1 != christina_s1

    def test_extract_step_one_raises_on_missing_markers(self):
        with pytest.raises(ValueError, match="Could not find"):
            _extract_step_one("No steps here.")

    def test_surgical_replacement_preserves_preamble_and_step2(self, monkeypatch):
        monkeypatch.setattr(module, "USE_ANTONYM", True)

        monkeypatch.setattr(module, "USE_ORIGINAL_STEP_ONE", False)
        normal = build_roger_instruction_prompt("x", "y", "def", 5, 40)

        monkeypatch.setattr(module, "USE_ORIGINAL_STEP_ONE", True)
        swapped = build_roger_instruction_prompt("x", "y", "def", 5, 40)

        preamble_n = normal[:normal.index("Step 1:")]
        preamble_s = swapped[:swapped.index("Step 1:")]
        assert preamble_n == preamble_s, "Preamble should be identical"

        step2_n = normal[normal.index("Step 2:"):]
        step2_s = swapped[swapped.index("Step 2:"):]
        assert step2_n == step2_s, "Step 2+ should be identical"

    def test_surgical_replacement_swaps_step1_content(self, monkeypatch):
        monkeypatch.setattr(module, "USE_ANTONYM", True)

        monkeypatch.setattr(module, "USE_ORIGINAL_STEP_ONE", False)
        normal = build_roger_instruction_prompt("brave", "cowardly", "def", 5, 40)

        monkeypatch.setattr(module, "USE_ORIGINAL_STEP_ONE", True)
        swapped = build_roger_instruction_prompt("brave", "cowardly", "def", 5, 40)

        # Roger's step 1 has dual examples; Christina's has single
        assert 'Example pairs for "selfish"' in normal
        assert 'Example pairs for "selfish"' not in swapped
        # Christina's template uses "example pair for the trait"
        assert 'example pair for the trait' in swapped


FAKE_COMBINED_RESPONSE = {
    "instruction": FAKE_INSTRUCTIONS,
    "questions": FAKE_QUESTIONS,
    "eval_prompt": "You are evaluating whether the model's response displays the trait: sarcastic.",
}


@pytest.fixture
def christina_style(monkeypatch):
    """Set PROMPT_STYLE to Christina for the test, restore after."""
    monkeypatch.setattr(module, "PROMPT_STYLE", "Christina")


@pytest.fixture
def mock_client_combined() -> AsyncMock:
    """Mock client that returns a single combined response."""
    client = AsyncMock()
    client.messages.create = AsyncMock(
        return_value=_make_response(json.dumps(FAKE_COMBINED_RESPONSE))
    )
    return client


class TestRegenerateOneChristina:
    """Tests for regenerate_one with Christina prompt style."""

    def test_combined_call(self, trait_file, mock_client_combined, christina_style):
        result = asyncio.run(
            regenerate_one(
                mock_client_combined,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        data = json.loads(trait_file.read_text())
        assert data["instruction"] == FAKE_INSTRUCTIONS
        assert data["questions"] == FAKE_QUESTIONS
        # eval_prompt rebuilt deterministically, not from LLM response
        expected_eval = build_eval_prompt("arrogant", SAMPLE_DESCRIPTION)
        assert data["eval_prompt"] == expected_eval
        # Only one API call for combined
        assert mock_client_combined.messages.create.call_count == 1

    def test_instructions_only_discards_questions(
        self, trait_file, mock_client_combined, christina_style
    ):
        result = asyncio.run(
            regenerate_one(
                mock_client_combined,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=True,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        data = json.loads(trait_file.read_text())
        assert data["instruction"] == FAKE_INSTRUCTIONS
        # Questions preserved from original file, not from LLM
        assert data["questions"] == SAMPLE_TRAIT["questions"]
        # Still only one API call (combined prompt, questions discarded)
        assert mock_client_combined.messages.create.call_count == 1

    def test_dry_run(self, trait_file, mock_client_combined, christina_style):
        original = trait_file.read_text()

        result = asyncio.run(
            regenerate_one(
                mock_client_combined,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=True,
            )
        )

        assert result.startswith("DRY-RUN")
        assert "Christina" in result
        assert trait_file.read_text() == original
        mock_client_combined.messages.create.assert_not_called()


@pytest.fixture
def jacob_style(monkeypatch):
    """Set PROMPT_STYLE to Jacob for the test, restore after."""
    monkeypatch.setattr(module, "PROMPT_STYLE", "Jacob")


@pytest.fixture
def mock_client_jacob() -> AsyncMock:
    """Mock client that returns separate instruction and question responses."""
    client = AsyncMock()
    client.messages.create = AsyncMock(
        side_effect=[
            _make_response(json.dumps(FAKE_INSTRUCTIONS)),
            _make_response(json.dumps(FAKE_QUESTIONS)),
        ]
    )
    return client


class TestRegenerateOneJacob:
    """Tests for regenerate_one with Jacob prompt style (separate API calls)."""

    def test_separate_calls(self, trait_file, mock_client_jacob, jacob_style):
        result = asyncio.run(
            regenerate_one(
                mock_client_jacob,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        data = json.loads(trait_file.read_text())
        assert data["instruction"] == FAKE_INSTRUCTIONS
        assert data["questions"] == FAKE_QUESTIONS
        assert data["positive_label"] == "arrogant"
        assert data["negative_label"] == "humble"
        assert data["description"] == SAMPLE_DESCRIPTION
        assert mock_client_jacob.messages.create.call_count == 2

    def test_instructions_only(self, trait_file, jacob_style):
        client = AsyncMock()
        client.messages.create = AsyncMock(
            side_effect=[_make_response(json.dumps(FAKE_INSTRUCTIONS))]
        )

        result = asyncio.run(
            regenerate_one(
                client,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=True,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        data = json.loads(trait_file.read_text())
        assert data["instruction"] == FAKE_INSTRUCTIONS
        assert data["questions"] == SAMPLE_TRAIT["questions"]
        assert client.messages.create.call_count == 1

    def test_dry_run(self, trait_file, mock_client_jacob, jacob_style):
        original = trait_file.read_text()

        result = asyncio.run(
            regenerate_one(
                mock_client_jacob,
                trait_file,
                n_variants=5,
                n_questions=40,
                instructions_only=False,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=True,
            )
        )

        assert result.startswith("DRY-RUN")
        assert "Jacob" in result
        assert trait_file.read_text() == original
        mock_client_jacob.messages.create.assert_not_called()


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        p = tmp_path / "out.json"
        atomic_write_json(p, {"a": 1})
        assert json.loads(p.read_text()) == {"a": 1}

    def test_trailing_newline(self, tmp_path):
        p = tmp_path / "out.json"
        atomic_write_json(p, {"a": 1})
        assert p.read_text().endswith("\n")


class TestUsageTracking:
    """Every API response ticks the MultiModelUsage tracker (AGENT_NOTES
    "Token usage logging is mandatory on batched LLM call sites")."""

    def test_regenerate_one_charges_tracker(self, trait_file):
        from assistant_axis.judge_pricing import MultiModelUsage

        resp = _make_response(json.dumps(FAKE_ROGER_RESPONSE))
        resp.usage = MagicMock(input_tokens=1200, output_tokens=3400)
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=resp)
        tracker = MultiModelUsage()
        result = asyncio.run(
            regenerate_one(
                client, trait_file, n_variants=5, n_questions=40,
                instructions_only=False, model="claude-sonnet-4-6",
                semaphore=asyncio.Semaphore(10), temperature=1.0,
                force=True, dry_run=False, usage=tracker,
            )
        )
        assert result.startswith("OK")
        assert tracker.n_calls == 1
        assert tracker.total_prompt_tokens == 1200
        assert tracker.total_completion_tokens == 3400
        assert list(tracker.per_model) == ["claude-sonnet-4-6"]
        assert tracker.total_cost_usd > 0

    def test_persist_usage_merges_into_existing_file(self, tmp_path):
        from assistant_axis.judge_pricing import MultiModelUsage

        path = tmp_path / "regeneration_usage.json"
        first = MultiModelUsage(); first.charge("claude-sonnet-4-6", 100, 200)
        module.persist_usage(first, path)
        second = MultiModelUsage(); second.charge("claude-sonnet-4-6", 10, 20)
        module.persist_usage(second, path)
        total = MultiModelUsage.load_or_create(path)
        assert total.n_calls == 2
        assert total.total_prompt_tokens == 110
        assert total.total_completion_tokens == 220

    def test_dry_run_and_default_do_not_need_a_tracker(self, trait_file, mock_client):
        # usage=None (the default) must remain a no-op for existing callers
        result = asyncio.run(
            regenerate_one(
                mock_client, trait_file, n_variants=5, n_questions=40,
                instructions_only=False, model="test-model",
                semaphore=asyncio.Semaphore(10), temperature=1.0,
                force=True, dry_run=False,
            )
        )
        assert result.startswith("OK")


class TestGeneratorProvenance:
    def test_generator_field_written(self, trait_file, mock_client, monkeypatch):
        monkeypatch.setattr(module, "PROMPT_STYLE", "Roger")
        monkeypatch.setattr(module, "USE_ANTONYM", True)
        asyncio.run(regenerate_one(
            mock_client, trait_file, n_variants=5, n_questions=40, instructions_only=False,
            model="test-model", semaphore=asyncio.Semaphore(10), temperature=1.0,
            force=True, dry_run=False,
        ))
        g = json.loads(trait_file.read_text())["generator"]
        assert g["style"] == "Roger" and g["model"] == "test-model" and g["use_antonym"] is True
        assert g["template_sha256"] == module.template_sha256("Roger") and len(g["template_sha256"]) == 12
        assert g["script"] == "regenerate_trait_instructions.py"

    def test_jacob_style_has_no_template_hash(self):
        assert module.template_sha256("Jacob") is None
