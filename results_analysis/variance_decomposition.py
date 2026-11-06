"""Variance decomposition of role+trait combination activations.

For each (slot, layer, kind) and each metric (raw / whitened), explain the
variance of ``combo(role, trait)`` activations as a sequence of nested
models that progressively absorb more structure:

    (a) **role + trait - pool_mean**: the bare additive baseline.  How much
        of ``combo(A, B)``'s variance is captured by simply adding A's and
        B's standalone vectors and subtracting the corpus baseline?

    (b) **+ heuristic theatricality shift**: add ``X * v_theat`` where
        ``X = pool_mean . v_theat`` is the projection of the pool mean onto
        the theatricality unit vector.  This is the "no-fit" guess at the
        residual-after-additive direction.

    (c) **+ optimised theatricality shift**: replace ``X`` with the scalar
        that minimises SSE on the pooled (r_, t_) residual after step (a).

    (d) **+ 4-parameter weighted fit**: fit (a, b, c, d) in [0, 1] s.t.
        the predicted combo for r_ is ``(a+b)*role + (c+d)*trait`` and for
        t_ is ``(a+d)*role + (b+c)*trait``.  Captures any per-kind weight
        asymmetry between role and trait contributions.  Origin is
        ``pool_mean - (optimised X) * v_theat``.

    (rem) **remainder**: 1 - R^2 of step (d) -- the unexplained residual.

Reported separately for r_ and t_ kinds, even though steps (c) and (d) use
a shared optimisation.

What the plot tells you
-----------------------

All header slots (1..N-1, where N is the slot count) show a non-trivial
step (b): the heuristic theatricality shift along ``v_theat`` alone
captures 12-30% of the per-step variance reduction, confirming that
combination activations carry a non-additive offset along a single
common-mode direction.  Step (c) adds little (the heuristic was already
a good guess), and step (d) adds little more (the additive model is
already nearly weight-symmetric).  Slot 0 (body mean) does NOT show
this pattern -- the heuristic shift is essentially zero (raw) or
weakly positive (whitened) -- consistent with the theatricality offset
being a header-slot phenomenon (see ``canonical_angles/README.md``
"Theatricality shift").

In the 8-slot Roger layout (slots 4..7 are ``<think>``, ``\\n\\n (in)``,
``</think>``, ``\\n\\n (post)``), the new slots follow the same
header-slot pattern as 1..3, with one notable exception: slot 6
(``</think>``) has a body-mean-like additive R² (~67 %) while still
showing a moderate theatricality offset (b ~ +12 %).  The other six
header slots cluster around (a ~ 42-51 %, b ~ +12-29 %).

Whitening
---------

The whitened columns apply soft-K PCA whitening to ``role``, ``trait``,
``combo``, ``default``, and ``v_theat`` before the variance
decomposition.  Default is ``--K 3`` (one column per K, plus a
"raw" column at the left -- a 4 x 3 grid).  The whitening basis is fit
per-slot on the pool returned by
``canonical_angles.data.build_augmented_whitening_pool`` (held-out
roles+traits standalones + ``default.pt``); pass ``--no-augment`` to
drop the default augmentation.

Default
-------

``default.pt`` is used for centering (subtracted from data); the
"pool_mean" used in the additive baseline is the mean of the held-out
roles+traits pool minus default.

Output
------

An ``n_metrics x num_slots`` grid of bar charts (rows = whitening
metrics, cols = slots) showing ``\\Delta R^2`` per step for r_ (blue)
and t_ (orange), with a final "remainder" bar in muted grey/brown.
``num_slots`` is auto-detected from the loaded data (4 on the older
Christina-headers data, 8 on the new Qwen-3 non-thinking 8-slot data).
Per-slot panel width compresses from 7 in (S=4) to 3.5 in (S=8) so
the figure stays readable at any slot count.

Usage
-----

::

    # Default (8 slots, layer 25, K=[2], augmented pool, Roger 8slot data):
    # produces a 2 x 8 grid (raw row, K=2 wht row)
    uv run python results_analysis/variance_decomposition.py \\
        --output roger/variance_decomp_bars.png

    # Pies version (each cell = r_ + t_ pie pair)
    uv run python results_analysis/variance_decomposition.py \\
        --style pies \\
        --output roger/variance_decomp_pies.png

    # Single-K column reproducing the historic Apr 23 layout
    uv run python results_analysis/variance_decomposition.py \\
        --K 128 --output /tmp/var_decomp_K128.png

    # Different layer
    uv run python results_analysis/variance_decomposition.py \\
        --layer 32 --K 4 8 16 64 \\
        --output /tmp/var_decomp_l32_Ksweep.png

    # Reproduce the original 4-slot Christina-headers figure
    uv run python results_analysis/variance_decomposition.py \\
        --data_dir 'runpod_workspace/qwen/qwen-3-32b Christina headers' \\
        --output /tmp/var_decomp_4slot.png

The historical April 23 reconstruction lives at
``roger/variance_decomp_bars_apr23.png`` for comparison; the current
default-augmented pool causes only minor changes to the whitened panels.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

from results_analysis.canonical_angles.data import (
    DEFAULT_DATA_DIR,
    build_augmented_whitening_pool,
    build_whitening_pool,
)
from results_analysis.canonical_angles.whitening import (
    DEFAULT_SOFT_K,
    WhiteningBasis,
    fit_whitening,
)
from assistant_axis import png_metadata, suptitle_with_specs
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_vec(path: Path | str) -> np.ndarray:
    """Load a vector .pt file and return its tensor field as float32 numpy."""
    obj = torch.load(str(path), weights_only=False)
    return obj["vector"].float().numpy()


def collect_combos(combos_dir: Path, prefix: str
                   ) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
    """Scan ``combinations/vectors/{prefix}_*__*.pt`` and return
    (sorted_unique_roles, sorted_unique_traits, list_of_(role, trait, path))."""
    files = sorted(glob.glob(str(combos_dir / f"{prefix}_*.pt")))
    pairs = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        m = re.match(r"^[rt]_(.+)__(.+)$", stem)
        if m:
            pairs.append((m.group(1), m.group(2), f))
    roles = sorted({p[0] for p in pairs})
    traits = sorted({p[1] for p in pairs})
    return roles, traits, pairs


def load_named_subset(stand_dir: Path, names: Iterable[str]) -> np.ndarray:
    """Stack ``stand_dir/<name>.pt`` for each name in ``names`` (skip missing)."""
    arrs = []
    for n in names:
        p = stand_dir / f"{n}.pt"
        if p.exists():
            arrs.append(load_vec(p))
    return np.stack(arrs)


# ---------------------------------------------------------------------------
# The variance decomposition itself
# ---------------------------------------------------------------------------

def per_dataset_R2(combo_r_c: np.ndarray, combo_t_c: np.ndarray,
                   pred_r: np.ndarray, pred_t: np.ndarray,
                   ss_r_total: float, ss_t_total: float
                   ) -> tuple[float, float]:
    sse_r = float(((combo_r_c - pred_r) ** 2).sum())
    sse_t = float(((combo_t_c - pred_t) ** 2).sum())
    return 1 - sse_r / ss_r_total, 1 - sse_t / ss_t_total


def _whiten(W: WhiteningBasis | None, X: np.ndarray) -> np.ndarray:
    return X if W is None else W.apply(X)


def compute_decomp(slot: int, layer: int, *,
                   default: np.ndarray,
                   pool_mean_full: np.ndarray,
                   r_role_vecs: np.ndarray, r_trait_vecs: np.ndarray,
                   r_combo_vecs: np.ndarray, r_pair_idx: list[tuple[int, int]],
                   t_role_vecs: np.ndarray, t_trait_vecs: np.ndarray,
                   t_combo_vecs: np.ndarray, t_pair_idx: list[tuple[int, int]],
                   v_unit_raw: np.ndarray,
                   whitener: WhiteningBasis | None,
                   ) -> dict[str, tuple[float, float]]:
    """Compute the per-step \\Delta R^2 (r_, t_) for one (slot, metric).

    Returns a dict with keys 'a', 'b', 'c', 'd', 'rem'; each value is
    a (r_increment, t_increment) tuple in [0, 1].
    """
    d = default[slot]
    pool_mean = pool_mean_full[slot]

    # Per-pair role/trait vectors at this slot/layer
    R_r = np.stack([r_role_vecs[ri, slot, layer]
                    for (ri, _) in r_pair_idx])
    T_r = np.stack([r_trait_vecs[ti, slot, layer]
                    for (_, ti) in r_pair_idx])
    combo_r = r_combo_vecs[:, slot, layer]
    R_t = np.stack([t_role_vecs[ri, slot, layer]
                    for (ri, _) in t_pair_idx])
    T_t = np.stack([t_trait_vecs[ti, slot, layer]
                    for (_, ti) in t_pair_idx])
    combo_t = t_combo_vecs[:, slot, layer]

    # Apply whitening to all vectors that will appear in the model.
    R_r = _whiten(whitener, R_r); T_r = _whiten(whitener, T_r)
    combo_r_w = _whiten(whitener, combo_r)
    R_t = _whiten(whitener, R_t); T_t = _whiten(whitener, T_t)
    combo_t_w = _whiten(whitener, combo_t)
    d_w = _whiten(whitener, d[None])[0]
    pool_mean_w = _whiten(whitener, pool_mean[None])[0]
    v_unit_w = _whiten(whitener, v_unit_raw[None])[0]
    v_unit_w = v_unit_w / np.linalg.norm(v_unit_w)

    # Center on default.
    R_r_c = R_r - d_w; T_r_c = T_r - d_w; combo_r_c = combo_r_w - d_w
    R_t_c = R_t - d_w; T_t_c = T_t - d_w; combo_t_c = combo_t_w - d_w
    pool_mean_c = pool_mean_w - d_w

    ss_r_total = float(((combo_r_c - pool_mean_c) ** 2).sum())
    ss_t_total = float(((combo_t_c - pool_mean_c) ** 2).sum())

    # Step (a) role + trait - pool_mean
    pred_r_a = R_r_c + T_r_c - pool_mean_c
    pred_t_a = R_t_c + T_t_c - pool_mean_c
    R2a_r, R2a_t = per_dataset_R2(combo_r_c, combo_t_c, pred_r_a, pred_t_a,
                                  ss_r_total, ss_t_total)

    # Step (b) heuristic theatricality shift: X = pool_mean . v_theat
    X_heuristic = float(pool_mean_c @ v_unit_w)
    pred_r_b = pred_r_a + X_heuristic * v_unit_w
    pred_t_b = pred_t_a + X_heuristic * v_unit_w
    R2b_r, R2b_t = per_dataset_R2(combo_r_c, combo_t_c, pred_r_b, pred_t_b,
                                  ss_r_total, ss_t_total)

    # Step (c) optimal scalar shift along v_theat (from step-(a) residual)
    resid_a_r = combo_r_c - pred_r_a
    resid_a_t = combo_t_c - pred_t_a
    all_resid_v = np.concatenate([resid_a_r @ v_unit_w, resid_a_t @ v_unit_w])
    X_opt = float(all_resid_v.mean())
    pred_r_c = pred_r_a + X_opt * v_unit_w
    pred_t_c = pred_t_a + X_opt * v_unit_w
    R2c_r, R2c_t = per_dataset_R2(combo_r_c, combo_t_c, pred_r_c, pred_t_c,
                                  ss_r_total, ss_t_total)

    # Step (d) 4-param weighted fit (a, b, c, d) in [0, 1] each
    origin_shifted = pool_mean_c - X_opt * v_unit_w

    def sse_of(params):
        aa, bb, cc, dd = params
        pr = (aa + bb) * R_r_c + (cc + dd) * T_r_c - origin_shifted
        pt = (aa + dd) * R_t_c + (bb + cc) * T_t_c - origin_shifted
        return (float(((combo_r_c - pr) ** 2).sum())
                + float(((combo_t_c - pt) ** 2).sum()))

    result = minimize(sse_of, x0=[0.5, 0.5, 0.5, 0.5], method="L-BFGS-B",
                      bounds=[(0, 1), (0, 1), (0, 1), (0, 1)])
    aa, bb, cc, dd = result.x
    pred_r_d = (aa + bb) * R_r_c + (cc + dd) * T_r_c - origin_shifted
    pred_t_d = (aa + dd) * R_t_c + (bb + cc) * T_t_c - origin_shifted
    R2d_r, R2d_t = per_dataset_R2(combo_r_c, combo_t_c, pred_r_d, pred_t_d,
                                  ss_r_total, ss_t_total)

    return {
        "a":   (R2a_r,           R2a_t),
        "b":   (R2b_r - R2a_r,   R2b_t - R2a_t),
        "c":   (R2c_r - R2b_r,   R2c_t - R2b_t),
        "d":   (R2d_r - R2c_r,   R2d_t - R2c_t),
        "rem": (1 - R2d_r,       1 - R2d_t),
    }


def compute_v_theat_raw(slot: int, layer: int, *,
                        pool_mean_full: np.ndarray,
                        r_role_vecs, r_trait_vecs, r_combo_vecs, r_pair_idx,
                        t_role_vecs, t_trait_vecs, t_combo_vecs, t_pair_idx,
                        ) -> np.ndarray:
    """Mean residual-after-additive direction (raw, slot-specific unit vector).

    Per-row least-squares fit ``a*R + b*T ~= combo + pool_mean`` (i.e. the
    additive model with origin at ``pool_mean``); the row-mean residual
    is the theatricality direction.  See
    ``compute_theatricality_axis`` in :mod:`compute_combo_marginals`.
    """
    pool_mean = pool_mean_full[slot]

    def per_kind_residual(role_vecs, trait_vecs, combo_vecs, pair_idx):
        R = np.stack([role_vecs[ri, slot, layer] for (ri, _) in pair_idx])
        T = np.stack([trait_vecs[ti, slot, layer] for (_, ti) in pair_idx])
        combo = combo_vecs[:, slot, layer]
        rr = (R * R).sum(1); rt_ = (R * T).sum(1); tt = (T * T).sum(1)
        y = combo + pool_mean
        Ry = (R * y).sum(1); Ty = (T * y).sum(1)
        det = rr * tt - rt_ ** 2
        a = (tt * Ry - rt_ * Ty) / det
        b = (rr * Ty - rt_ * Ry) / det
        return (combo + pool_mean
                - a[:, None] * R - b[:, None] * T).mean(0)

    mr = per_kind_residual(r_role_vecs, r_trait_vecs,
                           r_combo_vecs, r_pair_idx)
    mt = per_kind_residual(t_role_vecs, t_trait_vecs,
                           t_combo_vecs, t_pair_idx)
    v = (mr + mt) / 2
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Qwen-3 chat-template slot layout in the 8-slot data.  Slots 5 and 7
# are both ``\n\n`` text but at different chat-template positions
# (inside vs after ``<think>...</think>``); we use ``(in)``/``(post)``
# suffixes for legend disambiguation, matching the convention in
# ``token_position_noise_analysis.py`` and ``all_roles_pairwise_slots.py``.
# For 4-slot data, only the first 4 entries are used; the script
# slices ``SLOT_LABELS[:num_slots]`` based on the loaded tensor shape.
SLOT_LABELS = [
    "Slot 0 (Body mean)", "Slot 1 (<|im_start|>)",
    "Slot 2 (assistant)", "Slot 3 (\\n)",
    "Slot 4 (<think>)",   "Slot 5 (\\n\\n in)",
    "Slot 6 (</think>)",  "Slot 7 (\\n\\n post)",
]
STEP_LABELS = ["(a)\nrole+trait", "(b)\nheuristic\ntheat_offset",
               "(c)\noptimal\ntheat_offset", "(d)\n4-param", "remainder"]
STEP_KEYS = ["a", "b", "c", "d", "rem"]

# Per-step colors used by both bars (in saturated form) and pies.
STEP_COLORS = {
    "a":   "#4c72b0",  # role+trait: blue
    "b":   "#55a868",  # heuristic theat_offset: green
    "c":   "#8172b3",  # optimal theat_offset: purple
    "d":   "#dd8452",  # 4-param: orange
    "rem": "#bbbbbb",  # remainder: light grey
}
# Short labels for pie wedges (kept terse so they fit).
PIE_LABELS = {
    "a":   "(a) role+trait",
    "b":   "(b) heuristic theat_offset",
    "c":   "(c) optimal theat_offset",
    "d":   "(d) 4-param",
    "rem": "rem",
}


def make_pie_plot(decomp_by_slot_metric: dict[tuple[int, str], dict],
                  output: Path, metric_labels: list[str],
                  layer: int, *, num_slots: int = 4,
                  inputs: list[InputSpec] | None = None) -> None:
    """Render a len(metric_labels) x num_slots grid of (r_, t_) pie pairs.

    Slots span the columns, whitening metrics span the rows.  Each
    cell is split into two pie charts side by side; pie wedges are
    the five variance-decomposition components (a, b, c, d, rem) in
    consistent colors.  Negative increments (rare; can occur on slot 0
    or with extreme K) are clipped to zero and the remaining wedges
    are renormalised to sum to 100% -- a small white-text annotation
    flags any clipped wedge.
    """
    n_rows = len(metric_labels)
    # Width per slot: at S=4 we keep 4 in/slot to match the historical
    # figure; at S=8 we use 3.5 in/slot (matching the bar plot below)
    # so each pie pair has ~1.75 in per pie -- big enough for the
    # wedge labels to stay readable.  Total fig width: 16 in at S=4,
    # 28 in at S=8.
    slot_width = 4.0 if num_slots <= 4 else 3.5
    fig_w = max(16.0, slot_width * num_slots)
    fig = plt.figure(figsize=(fig_w, 3.6 * n_rows + 1.4),
                     constrained_layout=False)
    metrics_str = ", ".join(metric_labels)
    title_line = (f"Variance decomposition of combination activations "
                  f"(layer={layer}).  "
                  f"{n_rows} metrics x {num_slots} slots ({metrics_str}).")
    # Single-line suptitle.  The legend below (drawn into the figure
    # explicitly after the subfigures grid) carries the colour key for
    # the (a)/(b)/(c)/(d)/rem step labels, so we don't duplicate them
    # in the title.
    fig.suptitle(title_line, fontsize=14, fontweight="bold", y=0.985)
    # Outer grid: one subfigure per (metric, slot) cell.  Reserve a
    # band at the top for the suptitle and shared legend; the band
    # needs to be a *fixed* slice of total height (~1 inch) so it
    # doesn't shrink as n_rows grows.
    title_frac = max(0.10, 1.4 / (3.6 * n_rows + 1.4))
    outer = fig.subfigures(2, 1, height_ratios=[title_frac, 1.0 - title_frac])
    body = outer[1]
    subfigs = body.subfigures(n_rows, num_slots, hspace=0.0, wspace=0.05)
    if n_rows == 1:
        subfigs = np.array([subfigs])
    if num_slots == 1:
        subfigs = subfigs.reshape(n_rows, 1)
    colors = [STEP_COLORS[k] for k in STEP_KEYS]

    # Pie label fontsize scales with panel size.  At S=4 each pie is
    # ~2 in across (slot_width=4.0 / 2 pies); at S=8 each is ~1.75 in
    # (slot_width=3.5 / 2).  Both fit "26.4" labels comfortably at the
    # respective fontsize.
    pie_label_fontsize = 8.5 if num_slots <= 4 else 6.5
    pie_kind_fontsize = 9 if num_slots <= 4 else 7.5

    def _pie(ax, values: np.ndarray, kind_label: str):
        """Draw one pie on axis ``ax``.  ``values`` is a length-5 array
        of Δ R² in percent (one per step in STEP_KEYS).  Negative
        entries are clipped to zero and the remaining wedges are
        renormalised to sum to 100%."""
        v = np.where(values < 0, 0.0, values)
        s = float(v.sum())
        if s <= 0:
            ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            return
        v_norm = v * 100.0 / s

        def fmt(pct):
            if pct >= 5:
                return f"{pct:.1f}%"
            if pct >= 1:
                return f"{pct:.1f}"
            return ""

        _, _, autotexts = ax.pie(
            v_norm, colors=colors, startangle=90, counterclock=False,
            autopct=fmt,
            wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
            textprops={"fontsize": pie_label_fontsize, "color": "black"},
        )
        for at in autotexts:
            at.set_fontweight("bold")
        ax.set_title(kind_label, fontsize=pie_kind_fontsize)

    pie_title_fontsize = 10 if num_slots <= 4 else 8
    for metric_idx, metric in enumerate(metric_labels):
        for slot_idx in range(num_slots):
            sf = subfigs[metric_idx, slot_idx]
            sf.suptitle(f"{SLOT_LABELS[slot_idx]} -- {metric}",
                        fontsize=pie_title_fontsize)
            inner = sf.subplots(1, 2)
            result = decomp_by_slot_metric[(slot_idx, metric)]
            r_vals = np.array([result[k][0] * 100 for k in STEP_KEYS])
            t_vals = np.array([result[k][1] * 100 for k in STEP_KEYS])
            _pie(inner[0], r_vals, "r_")
            _pie(inner[1], t_vals, "t_")
            # Surface clipped negatives in the script log so the user
            # sees the underlying numbers.
            neg_parts = []
            for kind, vals in (("r_", r_vals), ("t_", t_vals)):
                for k, val in zip(STEP_KEYS, vals):
                    if val < 0:
                        neg_parts.append(f"{kind}/{k}={val:+.1f}%")
            if neg_parts:
                print(f"  [clip] slot {slot_idx} {metric}: "
                      + ", ".join(neg_parts))

    # One shared legend at the top, drawn into the figure rather than
    # any individual subfigure so it doesn't get clipped.
    legend_handles = [
        plt.matplotlib.patches.Patch(facecolor=STEP_COLORS[k],
                                     label=PIE_LABELS[k])
        for k in STEP_KEYS
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.965), fontsize=9, ncol=len(STEP_KEYS),
               frameon=True, framealpha=0.92)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    print(f"Wrote {output}")


def make_plot(decomp_by_slot_metric: dict[tuple[int, str], dict],
              output: Path, metric_labels: list[str],
              layer: int, *, num_slots: int = 4,
              inputs: list[InputSpec] | None = None) -> None:
    """Render a len(metric_labels) x num_slots bar grid with a shared
    y-axis scale across all panels (so the bars are visually
    comparable).  Slots span the columns, whitening metrics span the
    rows."""
    n_rows = len(metric_labels)
    # Per-slot panel width compresses at high slot counts so the
    # figure doesn't blow past 32-inch total width.  S=4 stays at 7"
    # (matches original); S=8 -> 3.5" each = 28" total.
    slot_width = 7.0 if num_slots <= 4 else 3.5
    fig, axes = plt.subplots(
        n_rows, num_slots,
        figsize=(slot_width * num_slots, 3 * n_rows + 1.0),
        squeeze=False,
    )
    metrics_str = ", ".join(metric_labels)
    title_line = (f"Variance decomposition of combination activations "
                  f"(layer={layer}).  "
                  f"{n_rows} metrics x {num_slots} slots ({metrics_str}).")
    spec_line = (
        "(a) role+trait-pool_mean  |  (b) heuristic theat_offset along "
        "theatricality axis  |  (c) optimal theat_offset  |  "
        "(d) 4-param weights  |  (rem) unexplained"
    )
    _, top_rect = suptitle_with_specs(fig, title_line, spec_line,
                                      title_y=0.99, line_height=0.05)

    # Compute a shared (ymin, ymax) across all 12 panels with headroom for
    # the per-bar value labels.
    all_vals = np.concatenate([
        np.array([decomp_by_slot_metric[(s, m)][k][i] * 100
                  for s in range(num_slots) for m in metric_labels
                  for k in STEP_KEYS for i in (0, 1)])
    ])
    ymin = min(0.0, float(all_vals.min()) - 5.0)
    ymax = float(all_vals.max()) + 5.0
    bar_title_fontsize = 10 if num_slots <= 4 else 8
    bar_step_fontsize = 9 if num_slots <= 4 else 7
    for metric_idx, metric in enumerate(metric_labels):
        for slot_idx in range(num_slots):
            ax = axes[metric_idx, slot_idx]
            result = decomp_by_slot_metric[(slot_idx, metric)]
            x = np.arange(len(STEP_KEYS))
            width = 0.4
            r_vals = np.array([result[k][0] * 100 for k in STEP_KEYS])
            t_vals = np.array([result[k][1] * 100 for k in STEP_KEYS])
            bars_r = ax.bar(x - width / 2, r_vals, width,
                            label="r_", color="tab:blue", alpha=0.85)
            bars_t = ax.bar(x + width / 2, t_vals, width,
                            label="t_", color="tab:orange", alpha=0.85)
            # Highlight remainder bars distinctly.
            bars_r[-1].set_color("#6b8cae"); bars_r[-1].set_edgecolor("black")
            bars_r[-1].set_linewidth(1.2)
            bars_t[-1].set_color("#d18d5c"); bars_t[-1].set_edgecolor("black")
            bars_t[-1].set_linewidth(1.2)
            ax.set_xticks(x)
            ax.set_xticklabels(STEP_LABELS, fontsize=bar_step_fontsize)
            ax.set_ylabel("Δ R² per step (%)" if slot_idx == 0 else "")
            ax.set_title(f"{SLOT_LABELS[slot_idx]} -- {metric}",
                         fontsize=bar_title_fontsize)
            ax.set_ylim(ymin, ymax)
            ax.axhline(0, color="black", lw=0.6)
            ax.grid(True, axis="y", alpha=0.3)
            for bar, v in zip(bars_r, r_vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + (0.5 if v >= 0 else -1.5),
                        f"{v:+.1f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=7)
            for bar, v in zip(bars_t, t_vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + (0.5 if v >= 0 else -1.5),
                        f"{v:+.1f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=7)
            if metric_idx == 0 and slot_idx == 0:
                ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, top_rect))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    print(f"Wrote {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Local default override pointing at the 8-slot data, since the
    # script auto-detects num_slots from the loaded tensor.  We don't
    # change the shared ``canonical_angles.data.DEFAULT_DATA_DIR``
    # here because other (judge-dependent) scripts share that constant
    # and may need coordinated updates.
    LOCAL_DEFAULT_DATA_DIR = "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
    p.add_argument("--data_dir", default=LOCAL_DEFAULT_DATA_DIR,
                   help=f"Vectors directory (default: "
                        f"{LOCAL_DEFAULT_DATA_DIR}; pass the older "
                        f"{DEFAULT_DATA_DIR!r} to reproduce 4-slot figures).")
    p.add_argument("--layer", type=int, default=25,
                   help="Transformer layer (default: 25 -- Qwen-3-32B "
                        "optimum from rho_by_layer.py)")
    p.add_argument("--K", type=int, nargs="+", default=[DEFAULT_SOFT_K],
                   help=f"Soft-K whitening order(s) for the whitened "
                        f"panels.  One column per K value, plus a 'raw' "
                        f"column at the left.  Default: [{DEFAULT_SOFT_K}] "
                        f"(project soft-K default; see whitening.DEFAULT_SOFT_K).")
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Disable pool augmentation with default.pt "
                        "(default: augmented).  See "
                        "canonical_angles/README.md.")
    p.set_defaults(augment=True)
    p.add_argument("--style", choices=["bars", "pies"], default="bars",
                   help="Bar chart (default) or pie chart per (slot, "
                        "metric).  In 'pies' mode each cell is split "
                        "into two pies (r_ and t_); negative Δ R² "
                        "increments are clipped to zero and the "
                        "remaining wedges are renormalised.")
    p.add_argument("--output", required=True, help="Output PNG path")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    traits_dir = data_dir / "traits" / "vectors"
    roles_dir = data_dir / "roles" / "vectors"
    combos_dir = data_dir / "combinations" / "vectors"

    # default vector (across all S slots, used for centering)
    default = load_vec(combos_dir / "default.pt")[:, args.layer, :]  # (S, D)
    num_slots = default.shape[0]
    if num_slots > len(SLOT_LABELS):
        # Defensive: data has more slots than we have human-readable
        # labels for.  Should never happen with the canonical 4- or
        # 8-slot Qwen layouts, but bail loudly rather than render an
        # IndexError mid-plot.
        raise SystemExit(
            f"Loaded data has {num_slots} slots, but only "
            f"{len(SLOT_LABELS)} entries in SLOT_LABELS.  Extend "
            f"SLOT_LABELS at the top of this file."
        )
    print(f"Detected {num_slots} slots in the loaded data.")

    # Combination grid metadata
    r_roles, r_traits, r_pairs = collect_combos(combos_dir, "r")
    t_roles, t_traits, t_pairs = collect_combos(combos_dir, "t")
    print(f"Found {len(r_pairs)} r_-combos ({len(r_roles)} roles x "
          f"{len(r_traits)} traits) and {len(t_pairs)} t_-combos "
          f"({len(t_roles)} roles x {len(t_traits)} traits)")

    # Per-pair role/trait vectors and combo vectors.
    r_role_vecs = load_named_subset(roles_dir, r_roles)
    r_trait_vecs = load_named_subset(traits_dir, r_traits)
    t_role_vecs = load_named_subset(roles_dir, t_roles)
    t_trait_vecs = load_named_subset(traits_dir, t_traits)
    r_combo_vecs = np.stack([load_vec(p[2]) for p in r_pairs])
    t_combo_vecs = np.stack([load_vec(p[2]) for p in t_pairs])

    # Restrict to the chosen layer to save memory (still need all 4 slots).
    r_role_vecs   = r_role_vecs[:, :, args.layer:args.layer + 1, :]
    r_trait_vecs  = r_trait_vecs[:, :, args.layer:args.layer + 1, :]
    t_role_vecs   = t_role_vecs[:, :, args.layer:args.layer + 1, :]
    t_trait_vecs  = t_trait_vecs[:, :, args.layer:args.layer + 1, :]
    r_combo_vecs  = r_combo_vecs[:, :, args.layer:args.layer + 1, :]
    t_combo_vecs  = t_combo_vecs[:, :, args.layer:args.layer + 1, :]
    layer_idx = 0  # local index within the truncated tensor

    # Per-pair indices into the (deduplicated) role / trait arrays.
    r_role_pos = {r: i for i, r in enumerate(r_roles)}
    r_trait_pos = {t: i for i, t in enumerate(r_traits)}
    t_role_pos = {r: i for i, r in enumerate(t_roles)}
    t_trait_pos = {t: i for i, t in enumerate(t_traits)}
    r_pair_idx = [(r_role_pos[r], r_trait_pos[t]) for (r, t, _) in r_pairs]
    t_pair_idx = [(t_role_pos[r], t_trait_pos[t]) for (r, t, _) in t_pairs]

    # Held-out roles+traits pool vectors.  We mirror the
    # canonical_angles tool's leave-out logic: leave out every standalone
    # name that appears in the combination grid as either role or trait.
    leave_out_role_names = set(r_roles) | set(t_roles)
    leave_out_trait_names = set(r_traits) | set(t_traits)
    leave_out_set: set[tuple[str, str]] = set()
    leave_out_set |= {("roles", n) for n in leave_out_role_names}
    leave_out_set |= {("traits", n) for n in leave_out_trait_names}

    if args.augment:
        pool_entries = build_augmented_whitening_pool(
            data_dir, leave_out=leave_out_set, scope="roles+traits")
    else:
        pool_entries = build_whitening_pool(
            data_dir, leave_out=leave_out_set, scope="roles+traits")
    print(f"Whitening / pool-mean pool: {len(pool_entries)} entries")

    # Materialize the pool (across all S slots, layer fixed) into a
    # (n_pool, S, D) array.
    def _entry_path(et: str, name: str) -> Path:
        if et == "combinations":
            return combos_dir / f"{name}.pt"
        return data_dir / et / "vectors" / f"{name}.pt"

    pool_arr = np.stack([
        load_vec(_entry_path(et, n))[:, args.layer, :]
        for (et, n) in pool_entries
    ])  # (n_pool, S, D)

    # Pool mean, per slot.
    pool_mean_full = pool_arr.mean(axis=0)  # (S, D)

    # Theatricality direction per slot, in raw activation space.
    print("Computing theatricality unit vectors...")
    v_theat_raw = np.stack([
        compute_v_theat_raw(
            slot, layer_idx, pool_mean_full=pool_mean_full,
            r_role_vecs=r_role_vecs, r_trait_vecs=r_trait_vecs,
            r_combo_vecs=r_combo_vecs, r_pair_idx=r_pair_idx,
            t_role_vecs=t_role_vecs, t_trait_vecs=t_trait_vecs,
            t_combo_vecs=t_combo_vecs, t_pair_idx=t_pair_idx,
        )
        for slot in range(num_slots)
    ])  # (S, D)

    # Fit a soft-K=K whitener per slot on the pool, for each requested K.
    print(f"Fitting soft-K whiteners per slot for K in {args.K}...")
    whiteners: dict[int, list[WhiteningBasis]] = {}
    for K in args.K:
        whiteners[K] = [fit_whitening("soft_K", pool_arr[:, slot, :], K=K)
                        for slot in range(num_slots)]

    # Compute the decomposition for each (slot, metric).
    metric_labels = ["raw"] + [f"K={K} wht" for K in args.K]
    decomp: dict[tuple[int, str], dict] = {}
    for slot in range(num_slots):
        for metric in metric_labels:
            if metric == "raw":
                wht = None
            else:
                K_val = int(metric.split("=")[1].split()[0])
                wht = whiteners[K_val][slot]
            decomp[(slot, metric)] = compute_decomp(
                slot, layer_idx,
                default=default, pool_mean_full=pool_mean_full,
                r_role_vecs=r_role_vecs, r_trait_vecs=r_trait_vecs,
                r_combo_vecs=r_combo_vecs, r_pair_idx=r_pair_idx,
                t_role_vecs=t_role_vecs, t_trait_vecs=t_trait_vecs,
                t_combo_vecs=t_combo_vecs, t_pair_idx=t_pair_idx,
                v_unit_raw=v_theat_raw[slot], whitener=wht,
            )
        summary_parts = []
        for metric in metric_labels:
            r = decomp[(slot, metric)]
            summary_parts.append(
                f"{metric}: a={r['a'][0]*100:5.1f}% b={r['b'][0]*100:+5.1f}%")
        print(f"  slot {slot}:  " + "  |  ".join(summary_parts))

    # Provenance inputs: traits, roles, and combinations subtrees -- all
    # three are read at the chosen layer (combinations gives combo + the
    # corpus default; the standalones supply v_theat regression and the
    # whitening pool).  augmented-pool augmentation status is captured
    # in extras so changing --no-augment flips dependence accordingly.
    extras = {"layer": str(args.layer),
              "augment": "1" if args.augment else "0",
              "K": ",".join(str(k) for k in args.K)}
    inputs: list[InputSpec] = [
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras=extras),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras=extras),
        current_data_subtree_input(
            data_dir, "combinations/vectors", dep_key="combinations_vectors",
            extras=extras),
    ]
    if args.style == "pies":
        make_pie_plot(decomp, Path(args.output), metric_labels, args.layer,
                      num_slots=num_slots, inputs=inputs)
    else:
        make_plot(decomp, Path(args.output), metric_labels, args.layer,
                  num_slots=num_slots, inputs=inputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
