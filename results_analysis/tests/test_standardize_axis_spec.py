"""Unit tests for results_analysis.standardize_axis_spec."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from results_analysis.standardize_axis_spec import (
    build_prompt,
    parse_response,
    standardize_axis_spec,
)


# ---------------------------------------------------------------------------
# Pure helpers (no API)
# ---------------------------------------------------------------------------

def test_build_prompt_includes_examples():
    p = build_prompt("This pole represents X", "This pole represents Y")
    assert "This pole represents X" in p
    assert "This pole represents Y" in p
    # Multi-shot examples present
    assert "This means" in p
    assert "Someone who" in p
    assert "<pos_pole>" in p


def test_parse_response_well_formed():
    text = (
        "<pos_pole>This means X.</pos_pole>\n"
        "<neg_pole>This means Y.</neg_pole>\n"
    )
    pos, neg = parse_response(text)
    assert pos == "This means X."
    assert neg == "This means Y."


def test_parse_response_strips_whitespace():
    text = "<pos_pole>\n  This means X.\n  </pos_pole>\n<neg_pole>This means Y.</neg_pole>"
    pos, neg = parse_response(text)
    assert pos == "This means X."
    assert neg == "This means Y."


def test_parse_response_missing_tag_raises():
    with pytest.raises(ValueError, match="pos_pole"):
        parse_response("<neg_pole>only one</neg_pole>")
    with pytest.raises(ValueError, match="neg_pole"):
        parse_response("<pos_pole>only one</pos_pole>")


def test_parse_response_handles_multiline_tags():
    text = """\
<pos_pole>
This means being dedicated to reliably helping, guiding, and nurturing others
through structured care, behaving in ways that are trustworthy.
</pos_pole>
<neg_pole>
This means mocking, disrupting, or harming others rather than serving.
</neg_pole>
"""
    pos, neg = parse_response(text)
    assert pos.startswith("This means being dedicated")
    assert pos.endswith("trustworthy.")
    assert neg.startswith("This means mocking")


# ---------------------------------------------------------------------------
# standardize_axis_spec idempotence + force
# ---------------------------------------------------------------------------

def test_standardize_skips_when_already_done():
    """If pos_pole_standardized and neg_pole_standardized already exist,
    no API call is made."""
    spec = {
        "axis_name": "X",
        "pos_pole": "raw",
        "neg_pole": "raw neg",
        "pos_pole_standardized": "This means X.",
        "neg_pole_standardized": "This means Y.",
        "pos_examples": [], "neg_examples": [],
    }
    # Patch _call_sonnet to detect any unexpected API call
    with patch(
        "results_analysis.standardize_axis_spec._call_sonnet",
    ) as mock_call:
        out = standardize_axis_spec(spec)
        mock_call.assert_not_called()
    assert out["pos_pole_standardized"] == "This means X."
    assert out["pos_pole"] == "raw"  # original preserved


def test_standardize_force_reruns():
    spec = {
        "axis_name": "X",
        "pos_pole": "This pole represents X",
        "neg_pole": "This pole represents Y",
        "pos_pole_standardized": "stale",
        "neg_pole_standardized": "stale",
        "pos_examples": [], "neg_examples": [],
    }
    fake_response = (
        "<pos_pole>This means X (rerun).</pos_pole>"
        "<neg_pole>This means Y (rerun).</neg_pole>"
    )
    async def fake_call(*_args, **_kw): return fake_response
    with patch(
        "results_analysis.standardize_axis_spec._call_sonnet",
        side_effect=fake_call,
    ):
        out = standardize_axis_spec(spec, force=True)
    assert out["pos_pole_standardized"] == "This means X (rerun)."
    assert out["neg_pole_standardized"] == "This means Y (rerun)."
    assert out["pos_pole"] == "This pole represents X"  # original unchanged


def test_standardize_adds_fields_without_overwriting_originals():
    spec = {
        "axis_name": "X",
        "pos_pole": "This pole represents X (raw)",
        "neg_pole": "This pole represents Y (raw)",
        "pos_examples": ["a", "b"], "neg_examples": ["c", "d"],
        "_metadata": {"some": "info"},
    }
    fake_response = (
        "<pos_pole>This means X.</pos_pole>"
        "<neg_pole>This means Y.</neg_pole>"
    )
    async def fake_call(*_args, **_kw): return fake_response
    with patch(
        "results_analysis.standardize_axis_spec._call_sonnet",
        side_effect=fake_call,
    ):
        out = standardize_axis_spec(spec)
    # Original pole text preserved
    assert out["pos_pole"] == "This pole represents X (raw)"
    assert out["neg_pole"] == "This pole represents Y (raw)"
    # Standardized fields added
    assert out["pos_pole_standardized"] == "This means X."
    assert out["neg_pole_standardized"] == "This means Y."
    # All other fields passed through
    assert out["axis_name"] == "X"
    assert out["pos_examples"] == ["a", "b"]
    assert out["neg_examples"] == ["c", "d"]
    assert out["_metadata"] == {"some": "info"}


def test_standardize_validates_input():
    """Missing pos_pole or neg_pole raises ValueError."""
    bad = {"axis_name": "X", "pos_examples": [], "neg_examples": []}
    with pytest.raises(ValueError, match="pos_pole"):
        standardize_axis_spec(bad)


def test_parse_response_with_extra_text_around_tags():
    """If the model adds preamble or trailing text, the regex still works."""
    text = """Here's the output:

<pos_pole>This means X.</pos_pole>
<neg_pole>This means Y.</neg_pole>

Done!"""
    pos, neg = parse_response(text)
    assert pos == "This means X."
    assert neg == "This means Y."
