"""Per-slot colour palette and trait/role kind palette.

This module hosts two related but distinct palettes:

1.  The 8-slot palette (:func:`slot_color`, :func:`slot_colors_8`,
    :func:`slot_colors`) — described in detail below.
2.  The trait/role *kind* palette (:func:`kind_color`,
    :func:`kind_marker`, :func:`kind_text_color`,
    :func:`kind_text_style`) used in any plot that mixes traits and
    roles in the same axes.  See the docstrings of those functions
    and the canonical reference implementation in
    ``results_analysis/pair_slice_plots.py`` (lines 336–410) for the
    convention.

8-slot palette
==============

Adopted May 2026 after the PCA-scree-tail analysis revealed three
structurally-distinct slot groups:

* **Slot 0** (body-mean) sits alone with a far more concentrated
  spectrum than any header slot.  Rendered in **black** to mark it
  as semantically distinct from the chat-template scaffold.

* **Slots 2, 3, 5, 7** (the "regular-text cluster": ``assistant`` +
  the three ``\\n``/``\\n\\n`` newline-rich tokens) have moderate
  PCA-tail variance and form the primary candidate pool for the
  project's working-slot default (per the May 2026 OOO + tail
  analyses).  Rendered on the **plasma** colormap with **slot 7 at
  the dark-purple end**, sweeping through magenta/red/orange to
  yellow at slot 2.  This puts the latest slot in the chat template
  (``\\n\\n (post)``, immediately before the model's response) at
  the cool end and the earliest of the cluster (``assistant``) at
  the warm end.

* **Slots 1, 4, 6** (the "special-token cluster": ``<|im_start|>``,
  ``<think>``, ``</think>``) have higher PCA-tail variance -- their
  representations are spectrally more diffuse, consistent with these
  being unique chat-template markers.  Rendered on the **winter**
  colormap with **slot 6 at the pure-blue end**, sweeping through
  cyan to green at slot 1.

Use this convention everywhere a plot encodes slot identity by
colour, EXCEPT for plots that use colour to encode *slot pairs*
(e.g. ``all_roles_pairwise_slots.py``), which use a different
two-colormap convention -- see that script's docstring.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .entity_id import corpus_display_name, display_label, kind_long


# Group memberships, derived from the May 2026 cluster analyses.
_BODY_SLOTS = (0,)
_REGULAR_TEXT_SLOTS = (2, 3, 5, 7)   # assistant + 3 newline-rich slots
_SPECIAL_MARKER_SLOTS = (1, 4, 6)    # <|im_start|>, <think>, </think>


def slot_color(slot: int) -> tuple[float, float, float, float]:
    """Return the RGBA colour for a given slot index in the 8-slot layout.

    Convenience for code that processes one slot at a time; equivalent
    to ``slot_colors_8()[slot]`` but indexes into the canonical
    palette directly.

    For 4-slot data (Christina-headers etc.) this still works:
    slots 0, 1, 2, 3 each have their canonical colour assigned by
    the 8-slot mapping (slot 0 = black, 1 = green, 2 = yellow,
    3 = orange).  Out-of-range indices raise ``IndexError``.
    """
    return slot_colors_8()[slot]


def slot_colors_8() -> list[tuple[float, float, float, float]]:
    """Return the canonical 8-slot colour palette as a list indexed by
    slot index.

    Length-8 list.  See module docstring for the colour convention.
    Subsequent calls return the same list (recomputed cheaply each
    time).
    """
    colors: list[tuple[float, float, float, float] | None] = [None] * 8

    # Slot 0: body-mean, black.
    colors[0] = (0.0, 0.0, 0.0, 1.0)

    # Slots 2/3/5/7 on plasma, slot 7 at the dark-purple (low-value)
    # end.  Order is descending (7 -> 5 -> 3 -> 2) so slot 7 gets
    # plasma(0.05) and slot 2 gets plasma(0.90).
    plasma_pts = np.linspace(0.05, 0.90, 4)
    for plasma_idx, slot in enumerate((7, 5, 3, 2)):
        colors[slot] = tuple(plt.cm.plasma(plasma_pts[plasma_idx]))

    # Slots 1/4/6 on winter, slot 6 at the pure-blue (low-value) end.
    # Order descending (6 -> 4 -> 1).
    winter_pts = np.linspace(0.0, 1.0, 3)
    for winter_idx, slot in enumerate((6, 4, 1)):
        colors[slot] = tuple(plt.cm.winter(winter_pts[winter_idx]))

    # Defensive: every slot 0..7 should now have a colour.
    assert all(c is not None for c in colors), \
        f"slot_colors_8() left None entries: {colors}"
    return colors  # type: ignore[return-value]


def slot_colors(num_slots: int) -> list[tuple[float, float, float, float]]:
    """Return the per-slot palette truncated to the first ``num_slots``
    entries.

    For ``num_slots <= 8`` this slices the canonical 8-slot palette;
    for ``num_slots > 8`` (not currently used in the project) it
    falls back to the matplotlib default cycle for the extra slots.
    """
    base = slot_colors_8()
    if num_slots <= 8:
        return base[:num_slots]
    extra = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    extras_needed = num_slots - 8
    extra_rgba = [
        tuple(int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5)) + (1.0,)
        for c in extra[:extras_needed]
    ]
    return base + extra_rgba  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Trait / role kind palette
# ---------------------------------------------------------------------------
#
# Mixed-kind scatter plots (any plot with both trait dots and role dots in the
# same axes) follow the convention established in
# ``results_analysis/pair_slice_plots.py``:
#
#   * trait — fill ``lightgrey``, marker ``o``, label text ``dimgrey``
#   * role  — fill ``lightsteelblue``, marker ``s``, label text ``navy`` italic
#
# Plot dot labels show the **bare name only** (no ``|R`` / ``|T``); kind is
# encoded visually via these helpers.  Use :func:`assistant_axis.entity_id.display_label`
# (re-exported from here for convenience) to strip the ``|R`` / ``|T`` suffix.

_KIND_FILL_COLOR: dict[str, str] = {
    "roles": "lightsteelblue",
    "traits": "lightgrey",
}

_KIND_TEXT_COLOR: dict[str, str] = {
    "roles": "navy",
    "traits": "dimgrey",
}

_KIND_MARKER: dict[str, str] = {
    "roles": "s",
    "traits": "o",
}

_KIND_TEXT_STYLE: dict[str, dict[str, object]] = {
    # Italic for roles per the pair_slice_plots convention; traits are upright.
    "roles": {"color": "navy", "fontstyle": "italic"},
    "traits": {"color": "dimgrey"},
}


def kind_color(kind: str) -> str:
    """Return the canonical fill colour for a trait/role kind.

    Accepts any spelling that :func:`assistant_axis.entity_id.kind_long`
    accepts (``"R"``, ``"T"``, ``"role"``, ``"roles"``, ``"trait"``,
    ``"traits"``; case-insensitive).
    """
    return _KIND_FILL_COLOR[kind_long(kind)]


def kind_text_color(kind: str) -> str:
    """Return the canonical *annotation text* colour for a trait/role
    kind (``"navy"`` for roles, ``"dimgrey"`` for traits).
    """
    return _KIND_TEXT_COLOR[kind_long(kind)]


def kind_marker(kind: str) -> str:
    """Return the canonical scatter marker for a trait/role kind
    (``"s"`` square for roles, ``"o"`` circle for traits).
    """
    return _KIND_MARKER[kind_long(kind)]


def kind_text_style(kind: str) -> dict[str, object]:
    """Return canonical kwargs for ``ax.annotate`` on a trait/role
    kind label.

    Returns a fresh dict each call so callers can mutate it freely
    (e.g. add a ``"fontsize"`` key without polluting the canonical
    style).  Roles are italicised in ``navy``; traits are upright in
    ``dimgrey``.
    """
    return dict(_KIND_TEXT_STYLE[kind_long(kind)])


# Re-export ``display_label`` so plotting code can do
# ``from assistant_axis.plot_palette import display_label, kind_color``
# alongside the colour helpers, without a second import line.
__all__ = (
    "slot_color",
    "slot_colors",
    "slot_colors_8",
    "kind_color",
    "kind_marker",
    "kind_text_color",
    "kind_text_style",
    "display_label",
    "corpus_display_name",
)
