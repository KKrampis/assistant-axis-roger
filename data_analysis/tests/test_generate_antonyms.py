"""Tests for generate_antonyms.py: the classify call parses reasoning-first
JSON and ticks the usage tracker."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from assistant_axis.judge_pricing import MultiModelUsage
from data_analysis import generate_antonyms as module

INSTRUCTIONS = [{"pos": f"pos {i}", "neg": f"neg {i}"} for i in range(5)]
REPLY = {"reasoning": "the neg pole is humility", "negative_label": "humble", "antonym_score": 4}


def _client(text: str, usage=None) -> AsyncMock:
    resp = MagicMock()
    resp.content = [MagicMock(type="text", text=text)]
    resp.usage = usage if usage is not None else MagicMock(input_tokens=500, output_tokens=80)
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=resp)
    return client


class TestClassifyOne:
    def test_parses_reply_and_charges_usage(self):
        tracker = MultiModelUsage()
        result = asyncio.run(module.classify_one(
            _client(json.dumps(REPLY)), "arrogant", "Excessive confidence.", INSTRUCTIONS,
            asyncio.Semaphore(2), tracker,
        ))
        assert result["negative_label"] == "humble" and result["antonym_score"] == 4
        assert tracker.n_calls == 1
        assert tracker.total_prompt_tokens == 500 and tracker.total_completion_tokens == 80
        assert list(tracker.per_model) == [module.MODEL]

    def test_strips_fences_and_tolerates_no_tracker(self):
        text = "```json\n" + json.dumps(REPLY) + "\n```"
        result = asyncio.run(module.classify_one(
            _client(text), "arrogant", "Excessive confidence.", INSTRUCTIONS, asyncio.Semaphore(2),
        ))
        assert result["negative_label"] == "humble"

    def test_user_message_carries_definition_and_both_poles(self):
        msg = module.build_user_message("arrogant", "Excessive confidence.", INSTRUCTIONS)
        assert "positive_label: arrogant" in msg and "Definition: Excessive confidence." in msg
        assert "1. pos 0" in msg and "5. neg 4" in msg
