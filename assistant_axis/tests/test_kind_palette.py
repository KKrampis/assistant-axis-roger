"""Tests for the trait/role kind palette in
``assistant_axis.plot_palette``.

The colour/marker conventions are pinned to specific string values so a
silent style refactor can't drift the convention without a corresponding
test update.  See ``results_analysis/pair_slice_plots.py`` lines 336–410
for the canonical reference implementation.
"""
import pytest

from assistant_axis.plot_palette import (
    display_label,
    kind_color,
    kind_marker,
    kind_text_color,
    kind_text_style,
)


# ---------------------------------------------------------------------------
# Canonical values (locked-in by spec)
# ---------------------------------------------------------------------------

def test_kind_color_traits_is_lightgrey():
    assert kind_color("traits") == "lightgrey"


def test_kind_color_roles_is_lightsteelblue():
    assert kind_color("roles") == "lightsteelblue"


def test_kind_text_color_traits_is_dimgrey():
    assert kind_text_color("traits") == "dimgrey"


def test_kind_text_color_roles_is_navy():
    assert kind_text_color("roles") == "navy"


def test_kind_marker_traits_is_circle():
    assert kind_marker("traits") == "o"


def test_kind_marker_roles_is_square():
    assert kind_marker("roles") == "s"


def test_kind_text_style_traits():
    style = kind_text_style("traits")
    assert style == {"color": "dimgrey"}


def test_kind_text_style_roles():
    style = kind_text_style("roles")
    assert style == {"color": "navy", "fontstyle": "italic"}


# ---------------------------------------------------------------------------
# Accept all kind spellings the entity_id module accepts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kind_token", ["R", "role", "roles", "Roles", "ROLES"]
)
def test_kind_color_accepts_all_role_spellings(kind_token):
    assert kind_color(kind_token) == "lightsteelblue"


@pytest.mark.parametrize(
    "kind_token", ["T", "trait", "traits", "Traits", "TRAITS"]
)
def test_kind_color_accepts_all_trait_spellings(kind_token):
    assert kind_color(kind_token) == "lightgrey"


def test_kind_color_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        kind_color("flavour")


# ---------------------------------------------------------------------------
# Mutation safety
# ---------------------------------------------------------------------------

def test_kind_text_style_returns_fresh_dict():
    """Caller mutation must not pollute the canonical style."""
    s1 = kind_text_style("roles")
    s1["fontsize"] = 24
    s2 = kind_text_style("roles")
    assert "fontsize" not in s2


# ---------------------------------------------------------------------------
# display_label re-export
# ---------------------------------------------------------------------------

def test_display_label_reexported():
    """``display_label`` is convenience-re-exported alongside the
    palette helpers so plotting code only needs one import line."""
    assert display_label("patient|R") == "patient"
    assert display_label("patient") == "patient"
