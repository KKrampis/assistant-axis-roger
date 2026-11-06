"""Tests for assistant_axis.pair_list_cohort."""
import pytest

from assistant_axis.pair_list_cohort import (
    DEFAULT_PAIR_TYPE,
    PAIR_TYPES,
    cohort_from_pairs,
    pair_type_of,
)


def test_cohort_from_pairs_canonical():
    assert cohort_from_pairs("pair_list_responses.json") == "responses"
    assert cohort_from_pairs("pair_list_di.json") == "di"
    # ``pair_list_goalnongoal.json`` carries the goal/non-goal-FILTERED
    # subset since the 2026-05-13 rename; ``pair_list_clean.json`` is
    # the comprehensive all-clean-pairs superset.  Both consume via
    # the same cohort_from_pairs strip-prefix logic.
    assert cohort_from_pairs("pair_list_goalnongoal.json") == "goalnongoal"
    assert cohort_from_pairs("pair_list_clean.json") == "clean"


def test_cohort_from_pairs_legacy_one_offs():
    assert cohort_from_pairs("/abs/path/pair_list_3_cohort_b.json") == "3_cohort_b"
    assert cohort_from_pairs("pair_list_13_new.json") == "13_new"


def test_cohort_from_pairs_falls_back_to_stem():
    assert cohort_from_pairs("custom_pairs.json") == "custom_pairs"


def test_pair_type_default_is_traits():
    """Pre-May-2026 entries lack the field; must default to ``traits`` so
    legacy pair lists keep working without modification."""
    assert pair_type_of({"pos": "helpful", "neg": "unhelpful"}) == "traits"
    assert DEFAULT_PAIR_TYPE == "traits"
    assert "traits" in PAIR_TYPES and "roles" in PAIR_TYPES


def test_pair_type_explicit_traits():
    assert pair_type_of({"pos": "x", "neg": "y", "pair_type": "traits"}) == "traits"


def test_pair_type_explicit_roles():
    assert pair_type_of({"pos": "angel", "neg": "demon",
                          "pair_type": "roles"}) == "roles"


def test_pair_type_case_normalised():
    assert pair_type_of({"pos": "x", "neg": "y", "pair_type": "TRAITS"}) == "traits"
    assert pair_type_of({"pos": "x", "neg": "y", "pair_type": "Roles"}) == "roles"


def test_pair_type_rejects_unknown():
    with pytest.raises(ValueError, match="pair_type must be one of"):
        pair_type_of({"pos": "x", "neg": "y", "pair_type": "combinations"})


def test_pair_type_rejects_non_string():
    with pytest.raises(ValueError, match="must be a string"):
        pair_type_of({"pos": "x", "neg": "y", "pair_type": 1})
