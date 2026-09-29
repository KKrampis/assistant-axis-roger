"""Tests for regenerate_role_instructions.py.

Mocks the Anthropic client to verify merge logic: eval_prompt rebuilt from
description, instructions and questions replaced, pos-only format.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import data_analysis.regenerate_role_instructions as module

from data_analysis.regenerate_role_instructions import (
    atomic_write_json,
    build_christina_role_prompt,
    build_eval_prompt,
    extract_description,
    regenerate_one,
    role_display_name,
    strip_markdown_fences,
)

SAMPLE_DESCRIPTION = "A financial professional who manages numerical data."

SAMPLE_ROLE = {
    "description": SAMPLE_DESCRIPTION,
    "instruction": [
        {"pos": "Old pos 1."},
        {"pos": "Old pos 2."},
        {"pos": "Old pos 3."},
        {"pos": "Old pos 4."},
        {"pos": "Old pos 5."},
    ],
    "questions": [f"Old question {i}?" for i in range(40)],
    "eval_prompt": build_eval_prompt("accountant", SAMPLE_DESCRIPTION),
}

FAKE_INSTRUCTIONS = [
    {"pos": "New pos 1."},
    {"pos": "New pos 2."},
    {"pos": "New pos 3."},
    {"pos": "New pos 4."},
    {"pos": "New pos 5."},
]

FAKE_QUESTIONS = [f"New question {i}?" for i in range(40)]

FAKE_COMBINED_RESPONSE = {
    "instruction": FAKE_INSTRUCTIONS,
    "questions": FAKE_QUESTIONS,
    "eval_prompt": "You are evaluating whether the model's response displays the role: accountant.",
}


def _make_response(text: str) -> MagicMock:
    msg = MagicMock()
    block = MagicMock(type="text", text=text)
    msg.content = [block]
    return msg


@pytest.fixture
def role_file(tmp_path: Path) -> Path:
    p = tmp_path / "accountant.json"
    p.write_text(json.dumps(SAMPLE_ROLE, indent=2))
    return p


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.messages.create = AsyncMock(
        return_value=_make_response(json.dumps(FAKE_COMBINED_RESPONSE))
    )
    return client


@pytest.fixture
def roger_style(monkeypatch):
    monkeypatch.setattr(module, "PROMPT_STYLE", "Roger")


@pytest.fixture
def christina_style(monkeypatch):
    monkeypatch.setattr(module, "PROMPT_STYLE", "Christina")


class TestRoleDisplayName:
    def test_simple_name(self):
        assert role_display_name("accountant") == "accountant"

    def test_underscore_to_space(self):
        assert role_display_name("coral_reef") == "coral reef"

    def test_override(self):
        assert role_display_name("devils_advocate") == "devil's advocate"


class TestExtractDescription:
    def test_extracts_description(self):
        ep = (
            "You are evaluating whether the model's response displays the role: "
            "**accountant**. A financial professional who manages data.\n\n"
            "Prompt:\n[QUESTION START]"
        )
        assert extract_description(ep) == "A financial professional who manages data."

    def test_returns_empty_on_missing_bold(self):
        assert extract_description("No bold markers here.") == ""

    def test_returns_empty_on_empty_string(self):
        assert extract_description("") == ""


class TestBuildEvalPrompt:
    def test_contains_role_and_description(self):
        result = build_eval_prompt("accountant", "Manages financial data.")
        assert "**accountant**" in result
        assert "Manages financial data." in result

    def test_preserves_question_answer_placeholders(self):
        result = build_eval_prompt("accountant", "Manages financial data.")
        assert "{question}" in result
        assert "{answer}" in result

    def test_uses_0_to_3_scale(self):
        result = build_eval_prompt("accountant", "Manages financial data.")
        assert "0 if" in result
        assert "1 if" in result
        assert "2 if" in result
        assert "3 if" in result
        # Reason-before-score ending (AGENT_NOTES "Judge prompts: reason BEFORE
        # score"); the pre-May-2026 "number between 0 and 3 ... just the number"
        # ending is the forbidden anti-pattern and must not come back.
        assert "SCORE: <integer from 0 to 3>" in result
        assert "briefly reason" in result
        assert "just the number" not in result

    def test_roundtrip_with_extract(self):
        desc = "A financial professional who manages data."
        rebuilt = build_eval_prompt("accountant", desc)
        extracted = extract_description(rebuilt)
        assert extracted == desc


class TestStripMarkdownFences:
    def test_strips_json_fence(self):
        assert strip_markdown_fences('```json\n[1,2]\n```') == "[1,2]"

    def test_strips_bare_fence(self):
        assert strip_markdown_fences("```\nfoo\n```") == "foo"

    def test_noop_on_clean_json(self):
        assert strip_markdown_fences('[{"a":1}]') == '[{"a":1}]'


class TestBuildChristinaRolePrompt:
    def test_substitutes_role_and_description(self):
        result = build_christina_role_prompt("accountant", "Manages data.", 5, 40)
        assert "<role>\naccountant\n</role>" in result
        assert "Manages data." in result

    def test_parameterizes_counts(self):
        result = build_christina_role_prompt("chef", "Cooks food.", 3, 20)
        assert "list of 3 instructions" in result
        assert "Design 20 questions" in result
        assert "Generate 20 diverse questions" in result
        assert "question 20" in result

    def test_no_neg_instructions(self):
        result = build_christina_role_prompt("chef", "Cooks food.", 5, 40)
        assert '"neg"' not in result
        assert "negative" not in result.lower()

    def test_eval_template_has_single_brace_placeholders(self):
        result = build_christina_role_prompt("chef", "Cooks food.", 5, 40)
        assert "{ROLE}" in result
        assert "{question}" in result
        assert "{answer}" in result
        assert "{{ROLE}}" not in result

    def test_json_format_has_single_braces(self):
        result = build_christina_role_prompt("chef", "Cooks food.", 5, 40)
        idx = result.find("<output_format>")
        section = result[idx:]
        assert '{\n  "instruction"' in section
        assert '{"pos":' in section


class TestRegenerateOne:
    """Core merge-logic tests for role regeneration."""

    def test_replaces_instructions_and_questions(self, role_file, mock_client, roger_style):
        result = asyncio.run(
            regenerate_one(
                mock_client,
                role_file,
                n_variants=5,
                n_questions=40,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        data = json.loads(role_file.read_text())
        expected_eval = build_eval_prompt("accountant", SAMPLE_DESCRIPTION)
        assert data["eval_prompt"] == expected_eval
        assert data["description"] == SAMPLE_DESCRIPTION
        assert data["instruction"] == FAKE_INSTRUCTIONS
        assert data["questions"] == FAKE_QUESTIONS
        assert mock_client.messages.create.call_count == 1

    def test_skip_when_counts_match(self, role_file, mock_client, roger_style):
        result = asyncio.run(
            regenerate_one(
                mock_client,
                role_file,
                n_variants=5,
                n_questions=40,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=False,
                dry_run=False,
            )
        )

        assert result.startswith("SKIP")
        data = json.loads(role_file.read_text())
        assert data["instruction"] == SAMPLE_ROLE["instruction"]
        mock_client.messages.create.assert_not_called()

    def test_force_overrides_skip(self, role_file, mock_client, roger_style):
        result = asyncio.run(
            regenerate_one(
                mock_client,
                role_file,
                n_variants=5,
                n_questions=40,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        data = json.loads(role_file.read_text())
        assert data["instruction"] == FAKE_INSTRUCTIONS

    def test_dry_run_no_writes(self, role_file, mock_client, roger_style):
        original = role_file.read_text()

        result = asyncio.run(
            regenerate_one(
                mock_client,
                role_file,
                n_variants=5,
                n_questions=40,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=True,
            )
        )

        assert result.startswith("DRY-RUN")
        assert role_file.read_text() == original
        mock_client.messages.create.assert_not_called()

    def test_skip_default_role(self, tmp_path, mock_client, roger_style):
        p = tmp_path / "default.json"
        p.write_text(json.dumps({"instruction": [{"pos": ""}]}))

        result = asyncio.run(
            regenerate_one(
                mock_client,
                p,
                n_variants=5,
                n_questions=40,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("SKIP")
        assert "default" in result
        mock_client.messages.create.assert_not_called()

    def test_preserves_unknown_fields(self, role_file, mock_client, roger_style):
        data = json.loads(role_file.read_text())
        data["custom_field"] = "keep me"
        role_file.write_text(json.dumps(data))

        mock_client.messages.create = AsyncMock(
            return_value=_make_response(json.dumps(FAKE_COMBINED_RESPONSE))
        )

        asyncio.run(
            regenerate_one(
                mock_client,
                role_file,
                n_variants=5,
                n_questions=40,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        data = json.loads(role_file.read_text())
        assert data["custom_field"] == "keep me"

    def test_key_order(self, role_file, mock_client, roger_style):
        """Output keys should be: description, instruction, questions, eval_prompt."""
        asyncio.run(
            regenerate_one(
                mock_client,
                role_file,
                n_variants=5,
                n_questions=40,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        data = json.loads(role_file.read_text())
        keys = list(data.keys())
        assert keys[0] == "description"
        assert keys[1] == "instruction"
        assert keys[2] == "questions"
        assert keys[3] == "eval_prompt"

    def test_pos_only_no_neg(self, role_file, mock_client, roger_style):
        """Verify output instructions are pos-only (no neg key)."""
        asyncio.run(
            regenerate_one(
                mock_client,
                role_file,
                n_variants=5,
                n_questions=40,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        data = json.loads(role_file.read_text())
        for item in data["instruction"]:
            assert "pos" in item
            assert "neg" not in item

    def test_christina_style(self, role_file, mock_client, christina_style):
        result = asyncio.run(
            regenerate_one(
                mock_client,
                role_file,
                n_variants=5,
                n_questions=40,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        data = json.loads(role_file.read_text())
        assert data["instruction"] == FAKE_INSTRUCTIONS
        assert data["questions"] == FAKE_QUESTIONS

    def test_underscore_role_name(self, tmp_path, mock_client, roger_style):
        """Role with underscore in filename uses display name."""
        p = tmp_path / "coral_reef.json"
        role_data = {
            "description": "A living underwater ecosystem.",
            "instruction": [{"pos": f"Old {i}."} for i in range(5)],
            "questions": [f"Q{i}?" for i in range(40)],
            "eval_prompt": build_eval_prompt("coral reef", "A living underwater ecosystem."),
        }
        p.write_text(json.dumps(role_data))

        mock_client.messages.create = AsyncMock(
            return_value=_make_response(json.dumps(FAKE_COMBINED_RESPONSE))
        )

        result = asyncio.run(
            regenerate_one(
                mock_client,
                p,
                n_variants=5,
                n_questions=40,
                model="test-model",
                semaphore=asyncio.Semaphore(10),
                temperature=1.0,
                force=True,
                dry_run=False,
            )
        )

        assert result.startswith("OK")
        assert "coral reef" in result


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

    def test_regenerate_one_charges_tracker(self, role_file, roger_style):
        from assistant_axis.judge_pricing import MultiModelUsage

        resp = _make_response(json.dumps(FAKE_COMBINED_RESPONSE))
        resp.usage = MagicMock(input_tokens=1100, output_tokens=3300)
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=resp)
        tracker = MultiModelUsage()
        result = asyncio.run(
            regenerate_one(
                client, role_file, n_variants=5, n_questions=40,
                model="claude-sonnet-4-6", semaphore=asyncio.Semaphore(10),
                temperature=1.0, force=True, dry_run=False, usage=tracker,
            )
        )
        assert result.startswith("OK")
        assert tracker.n_calls == 1
        assert tracker.total_prompt_tokens == 1100
        assert tracker.total_completion_tokens == 3300
        assert list(tracker.per_model) == ["claude-sonnet-4-6"]

    def test_persist_usage_merges_into_existing_file(self, tmp_path):
        from assistant_axis.judge_pricing import MultiModelUsage

        path = tmp_path / "regeneration_usage.json"
        first = MultiModelUsage(); first.charge("claude-sonnet-4-6", 100, 200)
        module.persist_usage(first, path)
        second = MultiModelUsage(); second.charge("claude-sonnet-4-6", 10, 20)
        module.persist_usage(second, path)
        total = MultiModelUsage.load_or_create(path)
        assert total.n_calls == 2 and total.total_prompt_tokens == 110


class TestRogerV2StyleAndProvenance:
    """The Sep 2026 RogerV2 template and the ``generator`` provenance field."""

    def test_v2_prompt_substitutes_and_carries_the_rules(self):
        p = module.build_roger_role_prompt_v2("smuggler", "A smuggler moves contraband.", 5, 40)
        assert "<role>\nsmuggler\n</role>" in p and "A smuggler moves contraband." in p
        for phrase in ("case-worker", "poisoner", "15 to 25", "self-justification", '"question 40"'):
            assert phrase in p, phrase
        assert "{n_variants}" not in p and "{n_questions}" not in p

    def test_v2_examples_obey_the_register_rule(self):
        p = module.build_roger_role_prompt_v2("x", "y", 5, 40)
        block = p.split("<example_instructions>")[1].split("</example_instructions>")[0]
        for marker in ("navigating", "individuals", "demonstrate", "appropriate", "engage with", "the challenges of", "in a healthy way"):
            assert marker not in block, marker

    def test_v2_json_format_has_single_braces(self):
        p = module.build_roger_role_prompt_v2("x", "y", 5, 40)
        assert '{\n  "instruction"' in p and "{{" not in p and "}}" not in p

    def test_style_dispatch_sends_v2_prompt_and_writes_generator(self, role_file, mock_client, monkeypatch):
        monkeypatch.setattr(module, "PROMPT_STYLE", "RogerV2")
        result = asyncio.run(regenerate_one(
            mock_client, role_file, n_variants=5, n_questions=40, model="test-model",
            semaphore=asyncio.Semaphore(10), temperature=1.0, force=True, dry_run=False,
        ))
        assert result.startswith("OK")
        sent = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "case-worker" in sent and "<example_instructions>" in sent
        g = json.loads(role_file.read_text())["generator"]
        assert g["style"] == "RogerV2" and g["model"] == "test-model"
        assert g["template_sha256"] == module.template_sha256("RogerV2") and len(g["template_sha256"]) == 12
        assert g["temperature"] == 1.0 and g["thinking_budget"] == 0 and g["script"] == "regenerate_role_instructions.py"

    def test_generator_written_for_v1_and_replaces_stale_value(self, role_file, mock_client, roger_style):
        d = json.loads(role_file.read_text()); d["generator"] = {"style": "stale"}
        role_file.write_text(json.dumps(d))
        asyncio.run(regenerate_one(
            mock_client, role_file, n_variants=5, n_questions=40, model="m",
            semaphore=asyncio.Semaphore(10), temperature=1.0, force=True, dry_run=False,
        ))
        g = json.loads(role_file.read_text())["generator"]
        assert g["style"] == "Roger" and g["template_sha256"] != module.template_sha256("RogerV2")

    def test_template_hashes_are_distinct_per_style(self):
        shas = {s: module.template_sha256(s) for s in ("Christina", "Roger", "RogerV2")}
        assert shas["Roger"] != shas["RogerV2"]

    def test_parse_args_accepts_v2(self):
        assert module.parse_args(["--roles", "x", "--style", "RogerV2"]).style == "RogerV2"
