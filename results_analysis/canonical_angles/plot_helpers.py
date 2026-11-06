"""Composable matplotlib helpers for canonical-angles plots.

The compute layer (:mod:`.core`) returns a ``dict[label, np.ndarray]`` of
canonical-angle curves.  Each plot wrapper decides how to slice and arrange
those curves into figures.  This module provides three reusable patterns:

- :func:`plot_per_slot_panels`
    One panel per "panel key" (typically slot, but can be anything).
    Within each panel, plot one or more curves either as lines or bars.
    Wrappers compose this for the standard 2x2 / 2x4 slot grid layouts.

- :func:`plot_overlay_curves`
    All curves on a single plot, one curve per group key (e.g.
    "raw" vs "soft_K=4").  Used when comparing whitening regimes
    on the same axis.

- :func:`plot_grid`
    Two-axis facet grid: rows by one key, columns by another, one curve
    per remaining group key inside each cell.  Used for sweeps over
    e.g. (whitening x layer x slot).

All three accept a ``records`` argument shaped as a list of dicts:
``[{'panel': str, 'group': str, 'angles': np.ndarray, ...}, ...]``.
This is a long-format intermediate that wrappers build by mapping
:class:`.core.CASpec` keys onto these tag fields.

The functions return a matplotlib ``Figure`` so wrappers can do additional
formatting / save it to disk themselves.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# A long-format record describing one CA curve: angles + a set of tags
# the helpers can use to group / facet / style.
Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Panel-grid layout (one panel per panel-key value)
# ---------------------------------------------------------------------------

def plot_per_slot_panels(records: list[Record],
                         panel_key: str = "panel",
                         group_key: str | None = None,
                         layout: tuple[int, int] | None = None,
                         as_bars: bool = False,
                         x_label: str = "Canonical Angle Index (sorted)",
                         y_label: str = "Canonical Angle (degrees)",
                         title: str | None = None,
                         panel_title_fn: Callable[[str], str] | None = None,
                         ylim: tuple[float, float] | None = (0, 95),
                         draw_orthogonal_line: bool = True,
                         figsize_per_panel: tuple[float, float] = (3.5, 2.7),
                         legend_kwargs: dict | None = None,
                         ) -> matplotlib.figure.Figure:
    """One panel per unique value of ``records[i][panel_key]``.

    Parameters
    ----------
    records      : list of records; each must contain ``panel_key`` and
                   ``'angles'``.  ``group_key`` if provided is used to draw
                   multiple curves per panel.
    panel_key    : record field whose values become panels (typically 'slot')
    group_key    : record field whose values become curves within each panel
                   (e.g. 'whitening' or 'kind').  If None, one curve per
                   panel.
    layout       : (rows, cols).  Default: square-ish layout fitting all panels.
    as_bars      : if True, bar chart per panel (single group only); useful
                   for the historical mutual-CA plot.
    panel_title_fn : function panel_value -> displayed title; default uses
                     the panel value as-is.
    ylim         : y-axis range; pass None to auto-scale.
    draw_orthogonal_line : if True, draw a dashed line at 90 degrees for reference.
    figsize_per_panel : matplotlib figsize per panel (multiplied by layout).
    legend_kwargs : passed through to ``ax.legend()``.  None disables legends.
    """
    if not records:
        raise ValueError("plot_per_slot_panels: no records to plot")

    panel_values = _unique_preserve_order([r[panel_key] for r in records])
    n_panels = len(panel_values)
    rows, cols = layout if layout else _auto_grid(n_panels)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(figsize_per_panel[0] * cols,
                                      figsize_per_panel[1] * rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    panel_title_fn = panel_title_fn or (lambda v: f"{v}")

    for ax, panel_val in zip(axes_flat, panel_values):
        panel_records = [r for r in records if r[panel_key] == panel_val]
        ax.set_title(panel_title_fn(panel_val))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if draw_orthogonal_line:
            ax.axhline(y=90, color="gray", linestyle="--", alpha=0.4,
                       label="Orthogonal (90 deg)")
        ax.grid(True, alpha=0.3, axis="y")

        if as_bars:
            if group_key is not None:
                # Multiple groups in bars: side-by-side bars per index.
                _plot_grouped_bars(ax, panel_records, group_key)
            else:
                # Single curve as bars
                _plot_single_bars(ax, panel_records)
        else:
            if group_key is not None:
                _plot_curves(ax, panel_records, group_key)
            else:
                _plot_single_curve(ax, panel_records)

        if legend_kwargs is not None and (group_key is not None
                                          or draw_orthogonal_line):
            ax.legend(**legend_kwargs)

    # Hide leftover axes if layout has empty cells
    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    # tight_layout with rect leaves room for the suptitle
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1.0))
    return fig


# ---------------------------------------------------------------------------
# Overlay layout (all curves on one panel)
# ---------------------------------------------------------------------------

def plot_overlay_curves(records: list[Record],
                        group_key: str = "group",
                        x_label: str = "Canonical Angle Index (sorted)",
                        y_label: str = "Canonical Angle (degrees)",
                        title: str | None = None,
                        ylim: tuple[float, float] | None = (0, 95),
                        figsize: tuple[float, float] = (8, 5),
                        legend_kwargs: dict | None = None,
                        ) -> matplotlib.figure.Figure:
    """All records on one axes; one curve per ``group_key`` value."""
    fig, ax = plt.subplots(figsize=figsize)
    _plot_curves(ax, records, group_key)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if legend_kwargs is not None:
        ax.legend(**legend_kwargs)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Two-axis grid (rows x cols of panels)
# ---------------------------------------------------------------------------

def plot_grid(records: list[Record],
              row_key: str,
              col_key: str,
              group_key: str | None = None,
              x_label: str = "Canonical Angle Index (sorted)",
              y_label: str = "Canonical Angle (degrees)",
              title: str | None = None,
              ylim: tuple[float, float] | None = (0, 95),
              figsize_per_cell: tuple[float, float] = (3.5, 2.7),
              ) -> matplotlib.figure.Figure:
    """A faceted grid of panels indexed by (row_key, col_key) values.

    Within each cell, draw one curve per ``group_key`` value (if provided),
    else a single curve.
    """
    if not records:
        raise ValueError("plot_grid: no records to plot")

    row_values = _unique_preserve_order([r[row_key] for r in records])
    col_values = _unique_preserve_order([r[col_key] for r in records])
    n_rows, n_cols = len(row_values), len(col_values)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_cell[0] * n_cols,
                                      figsize_per_cell[1] * n_rows),
                             squeeze=False)

    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            ax = axes[i, j]
            cell_records = [r for r in records
                            if r[row_key] == row_val and r[col_key] == col_val]
            ax.set_title(f"{row_key}={row_val}, {col_key}={col_val}", fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(y_label)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
            if cell_records:
                if group_key is not None:
                    _plot_curves(ax, cell_records, group_key)
                else:
                    _plot_single_curve(ax, cell_records)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1.0))
    return fig


# ---------------------------------------------------------------------------
# Internal plotting primitives
# ---------------------------------------------------------------------------

def _plot_curves(ax, records: list[Record], group_key: str) -> None:
    """Draw one line per unique value of ``group_key``."""
    groups = _unique_preserve_order([r[group_key] for r in records])
    for g in groups:
        rs = [r for r in records if r[group_key] == g]
        # Average if there's more than one record in the group at the same
        # x positions (shouldn't happen for normal inputs but be safe).
        if len(rs) == 1:
            angles = rs[0]["angles"]
        else:
            angles = np.mean(np.stack([r["angles"] for r in rs]), axis=0)
        x = np.arange(1, len(angles) + 1)
        ax.plot(x, angles, marker="o", markersize=2, linewidth=1.3, label=f"{g}")


def _plot_single_curve(ax, records: list[Record]) -> None:
    """Draw one line; if multiple records, plot each unlabeled."""
    for r in records:
        angles = r["angles"]
        x = np.arange(1, len(angles) + 1)
        ax.plot(x, angles, marker="o", markersize=2, linewidth=1.3)


def _plot_single_bars(ax, records: list[Record]) -> None:
    """Bar chart from a single record (typical for mutual-CA plots)."""
    if len(records) != 1:
        # If the wrapper passed multiple, just stack their angles in groups
        # but really this case shouldn't happen for as_bars=True without group_key.
        _plot_grouped_bars(ax, records, group_key="_idx")
        return
    angles = records[0]["angles"]
    x = np.arange(1, len(angles) + 1)
    ax.bar(x, angles, color="purple", alpha=0.7, edgecolor="black", linewidth=0.5)


def _plot_grouped_bars(ax, records: list[Record], group_key: str) -> None:
    """Side-by-side grouped bar chart."""
    groups = _unique_preserve_order([r.get(group_key, "") for r in records])
    n_groups = len(groups)
    if n_groups == 0:
        return
    width = 0.8 / n_groups
    for i, g in enumerate(groups):
        rs = [r for r in records if r.get(group_key, "") == g]
        if not rs:
            continue
        angles = rs[0]["angles"]
        x = np.arange(1, len(angles) + 1) + (i - (n_groups - 1) / 2) * width
        ax.bar(x, angles, width=width, alpha=0.7, edgecolor="black", linewidth=0.5,
               label=f"{g}")


def _unique_preserve_order(values: Iterable) -> list:
    """Return the unique values preserving first-seen order."""
    seen: set = set()
    out: list = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _auto_grid(n: int) -> tuple[int, int]:
    """Pick a roughly-square (rows, cols) layout for ``n`` panels."""
    if n <= 1:
        return (1, 1)
    if n == 2:
        return (1, 2)
    if n == 3:
        return (1, 3)
    if n == 4:
        return (2, 2)
    if n <= 6:
        return (2, 3)
    if n == 8:
        return (2, 4)
    if n == 9:
        return (3, 3)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return (rows, cols)
