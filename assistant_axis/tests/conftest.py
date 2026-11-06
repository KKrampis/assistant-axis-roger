"""Tests that need the main repository's tree, skipped on this branch.

The judge modules here are copies of the main repository's (``RogerDOX14w/assistant-axis``, branch
``anthropic-vllm-uv``), and so are their test files, unchanged.  A few of those tests check the main
repository's own October 2026 corpus (``careless (from HEXACO)`` and other standard-suffixed traits), its
data-generation scripts, or its ``gapgen`` package.  This branch carries the corpus of 2026-09-28 and
neither of the others, so those tests are skipped while ``assistant_axis/gapgen`` is absent.  Judging
against the 8slot data reads the May corpus pinned at commit 1448a84c anyway
(``reports/judge_pareto/handoff_2026-10-21.md``).
"""
from pathlib import Path

import pytest

NEED_MAIN_TREE = {
    "test_axis_judge_correlation_phase4.py::TestRubricsRenderDisplayForm::test_standards_pair_header_nests_parentheses",
    "test_entity_id.py::TestCorpusDisplayName::test_overrides_in_sync_with_data_analysis",
    "test_entity_id.py::TestJudgeLabel::test_every_capitalised_suffix_in_the_corpus_is_in_the_table",
    "test_entity_id.py::TestJudgeLabel::test_every_capitalised_suffix_in_the_live_queue_is_in_the_table",
    "test_entity_id.py::TestJudgeLabel::test_every_role_renders_like_the_role_generator",
    "test_entity_id.py::TestJudgeLabel::test_examples",
    "test_entity_id.py::TestJudgeNegativeLabel::test_examples",
    "test_judge_request_capabilities.py::TestModelCapabilities::test_the_platform_list_is_covered",
}
MAIN_TREE = (Path(__file__).resolve().parents[1] / "gapgen").is_dir()


def pytest_collection_modifyitems(config, items):
    if MAIN_TREE:
        return
    skip = pytest.mark.skip(reason="needs the main repository's corpus, data scripts or gapgen package")
    for item in items:
        if item.nodeid.split("/")[-1] in NEED_MAIN_TREE:
            item.add_marker(skip)
