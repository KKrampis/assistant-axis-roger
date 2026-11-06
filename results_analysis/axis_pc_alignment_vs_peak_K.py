#!/usr/bin/env python3
"""Two-panel diagnostic of where each axis lives in held-out-pool PC
space, and how that translates into observed ρ at small whitening K.

For each (pos, neg) axis at the configured (slot, layer):

  1. Build the same held-out pool the K-sweep uses (every standalone
     role + trait + ``combinations/default``, default-centred, minus
     the 2 pair endpoints).
  2. PCA the pool.
  3. Compute ``|cos(axis, PC_k)|`` for the first ``N_PCS`` PCs.
  4. Compute "axis-energy in top-N PCs" = ``Σ_{k=1}^N cos²_k`` for
     several ``N`` (defaults: 1, 2, 3, 4).
  5. Read the fitted whitening sweet spot ``peak_K_log`` from the
     non-weighted di peak-fit JSON; for axes with responses scoring,
     blend desc+inst and responses at the project-default
     ``DEFAULT_RESPONSE_DI_WEIGHT`` (0.80 response / 0.20 desc+inst);
     otherwise use desc+inst only.  Convert to ``log₂(K+1)`` for the
     y-axis.
  6. Read the observed ρ at each K from the K-sweep JSON; combine
     desc+inst and responses at the same 0.80 / 0.20 ratio; fall back
     to desc+inst only when responses is unavailable.

The output PNG has TWO stacked panels sharing x = "axis energy in
top-N PCs":

  * Top panel  -- y = mean fitted peak K (log₂(K+1) clipped ≥ 0).
                 Per-axis horizontal line walks through top-N
                 markers.  Tied peak-K rows are vertically staggered
                 so a multi-axis pile-up at peak=0 doesn't fuse.
  * Bottom    -- y = observed ρ.  Marker shape encodes which K the
                 ρ is at: circle=K=0, square=K=1, diamond=K=2,
                 triangle=K=3 (line shape is no longer horizontal).

Per-axis colour is consistent across both panels.  Axes with BOTH
desc+inst AND responses data are drawn solid + full opacity ("primary
cohort"); axes available only in desc+inst are drawn dotted + faint
("extra cohort"), so the eye can immediately tell which lines depend
on responses scoring.

If the hypothesis holds, axes that peak at K=0 (raw) should have high
top-N cumulative cos² (most of the axis lives in the high-variance
directions soft-K shrinks first), and the relationship should be
monotonic across axes.

Inputs (read from ``--experiment_dir``):

- ``--pairs`` pair-list JSON (default ``pair_list_di.json`` -- the
  full ~35-axis cohort).
- ``--pairs_primary`` pair-list JSON (default
  ``pair_list_responses.json`` -- the 12-axis subset with response
  scoring as well).
- ``whitening_k_peak_fit_<cohort>_slot{N}.json`` -- non-weighted
  parabola fits (every axis in --pairs has at least one source).
- ``whitening_k_sweep_<cohort>_slot{N}.json`` -- raw observed ρ at
  every ``K ∈ FIT_K``, both sources.

Outputs (written to ``--experiment_dir``):

- ``axis_pc_alignment_vs_peak_K_<cohort>_slot{N}.png`` -- the
  two-panel scatter.

Examples
--------

::

    # Default: pair_list_di.json + pair_list_responses.json overlay,
    # slot 6, top-N ∈ {1, 2, 3, 4}.
    uv run python results_analysis/axis_pc_alignment_vs_peak_K.py

    # Compare slot 3:
    uv run python results_analysis/axis_pc_alignment_vs_peak_K.py --slot 3

    # Custom N values (the bottom panel maps marker_idx -> K =
    # (0, 1, 2, 3, ...), so it's most useful with consecutive small
    # integers; the top panel just wants increasing N):
    uv run python results_analysis/axis_pc_alignment_vs_peak_K.py \\
        --top_ns 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from assistant_axis.entity_id import corpus_display_name
from assistant_axis import (
    cohort_from_pairs,
    display_form_name,
    png_metadata,
    suptitle_with_specs,
)
from assistant_axis.judge_score_combine import DEFAULT_RESPONSE_DI_WEIGHT
from assistant_axis.provenance import (
    CACHE_POLICIES,
    InputSpec,
    current_data_subtree_input,
    load_and_register,
)
from results_analysis.axis_judge_correlation import _load_vector_file
from results_analysis.canonical_angles.data import (
    build_augmented_whitening_pool,
)

DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
    "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
)

# Match results_analysis/whitening_k_sweep.py: layer 25 is the tuned
# Qwen-3-32B layer (rho_by_layer.py) and slot 6 is the post-May-2026
# rejudge default (</think>); slot 3 (\n) remains a useful comparison.
LAYER = 25
DEFAULT_SLOT = 6
DEFAULT_PAIR_LIST = "pair_list_di.json"  # full ~35-axis cohort.
DEFAULT_PRIMARY_PAIR_LIST = "pair_list_responses.json"  # subset w/ responses.
DEFAULT_TOP_NS = (1, 2, 3, 4)
N_PCS = 16  # we only need up to max(top_ns), but 16 keeps headroom + cheap.

# Bottom-panel marker -> K mapping: the i-th marker in `top_ns` reads
# ρ at K = i  (so circle->K=0, square->K=1, diamond->K=2, ^->K=3 for
# the default top_ns = (1, 2, 3, 4)).  Independent of the actual N
# values, by user request.
def _bottom_panel_K_for_index(i: int) -> int:
    return i

# Marker progression for the per-axis horizontal lines.  Distinct
# shapes give the eye an anchor for "which N is which" without needing
# a second legend; the right-to-left ordering on the inverted x-axis
# means the line starts at top-1 (rightmost, smallest cos²) and walks
# leftward to top-4 (largest cos²).  Cycle if the user passes more
# than four top-N values.
_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")


def _resolve_pair_type(data_dir: Path, pos: str, neg: str) -> str:
    """Auto-detect whether the pair vectors live under ``traits/`` or
    ``roles/`` (the di cohort mixes both).  Falls back to ``"traits"``
    on ambiguous matches with a stderr warning -- silent ambiguity
    here was the root cause of the early-2026 axis-direction bug
    where role-pair axes were quietly read from a non-existent
    ``traits/`` slot.
    """
    in_traits = (data_dir / "traits" / "vectors" / f"{pos}.pt").exists() and \
                (data_dir / "traits" / "vectors" / f"{neg}.pt").exists()
    in_roles = (data_dir / "roles" / "vectors" / f"{pos}.pt").exists() and \
               (data_dir / "roles" / "vectors" / f"{neg}.pt").exists()
    if in_traits and not in_roles:
        return "traits"
    if in_roles and not in_traits:
        return "roles"
    if in_traits and in_roles:
        import sys as _sys
        print(f"[warn] {pos}/{neg} ambiguous: present in both traits and "
              f"roles, defaulting to traits", file=_sys.stderr)
        return "traits"
    raise FileNotFoundError(
        f"{pos}/{neg}: vectors not found in either {data_dir}/traits/vectors "
        f"or {data_dir}/roles/vectors"
    )


def _axis_unit_vector(
    data_dir: Path, pos: str, neg: str, *, slot: int, layer: int,
    pair_type: str | None = None,
) -> np.ndarray:
    """Unit axis direction at ``(slot, layer)``.

    ``pair_type`` is ``"traits"`` / ``"roles"`` / ``None``; ``None``
    auto-resolves via :func:`_resolve_pair_type`.
    """
    pt = pair_type or _resolve_pair_type(data_dir, pos, neg)
    vp = _load_vector_file(data_dir / pt / "vectors" / f"{pos}.pt").float()
    vn = _load_vector_file(data_dir / pt / "vectors" / f"{neg}.pt").float()
    d = (vp[slot, layer] - vn[slot, layer]).numpy()
    nrm = float(np.linalg.norm(d))
    if not np.isfinite(nrm) or nrm < 1e-12:
        raise ValueError(f"degenerate axis {pos!r} - {neg!r} at slot={slot}, "
                         f"layer={layer}: ||vp - vn|| = {nrm}")
    return d / nrm


def _pool_top_PCs(
    data_dir: Path, pos: str, neg: str, *, slot: int, layer: int, n_pcs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the held-out pool (excluding pair endpoints), centre, SVD,
    and return ``(Vt, S)`` for the top ``n_pcs`` components.

    ``Vt`` is the ``(n_pcs, hidden)`` right-singular-vector matrix
    (PCs as rows).  ``S`` is the corresponding length-``n_pcs``
    singular-value vector.  Singular values are returned so callers
    can construct the project's soft-K whitening transform
    (``W_K = I + sum_{j<K} (sigma_K/sigma_j - 1) PC_j PC_j^T``;
    see :mod:`results_analysis.canonical_angles.whitening`).

    Uses the *same* pool builder as
    :func:`results_analysis.whitening_k_sweep.load_pool` so PC content
    matches the whitener PC content the K-sweep is fit against.
    """
    leave_out: set[tuple[str, str]] = {
        ("traits", pos), ("roles", pos),
        ("traits", neg), ("roles", neg),
    }
    pool_entries = build_augmented_whitening_pool(
        data_dir, leave_out=leave_out, scope="roles+traits"
    )

    def _path(etype: str, name: str) -> Path:
        if etype == "combinations":
            return data_dir / "combinations" / "vectors" / f"{name}.pt"
        return data_dir / etype / "vectors" / f"{name}.pt"

    pool_rows = [
        _load_vector_file(_path(et, n)).float()[slot, layer].numpy()
        for (et, n) in pool_entries
    ]
    M = np.stack(pool_rows, axis=0)
    M_c = M - M.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(M_c, full_matrices=False)
    return Vt[:n_pcs], S[:n_pcs]


def _post_whitening_cosines(
    axis: np.ndarray, Vt: np.ndarray, S: np.ndarray, n_pcs_shown: int,
) -> np.ndarray:
    """``|cos(axis after soft-K=k-1 whitening, PC_k)|`` for k=1..n_pcs_shown.

    For each plotted PC index ``k`` (1-based), apply soft-K
    whitening with ``K = k-1`` to the *axis* vector, then compute
    its cosine with the *pre-whitening* ``PC_k`` (which is untouched
    by ``W_{k-1}`` because soft-K only rescales the top ``K`` PCs).

    With ``axis`` already a unit vector, ``coefs[j] = axis . PC_j``,
    and ``scales[j] = sigma_K / sigma_j``::

        ||axis_w||^2 = 1 + sum_{j<K} (scales[j]^2 - 1) * coefs[j]^2

    The numerator ``axis_w . PC_k`` equals ``coefs[k-1]`` (PC_k
    untouched by ``W_{k-1}`` and orthogonal to PC_j for j<K, j!=k-1).
    So the cosine is ``|coefs[k-1]| / ||axis_w||``.  For ``K=0`` this
    reduces to the bare pre-whitening ``|cos(axis, PC_1)|``.
    """
    coefs = Vt @ axis  # (n_pcs,) signed projections axis . PC_j
    out = np.zeros(n_pcs_shown, dtype=float)
    for k in range(n_pcs_shown):
        K = k  # whitening level for PC index k+1
        num = abs(float(coefs[k]))
        if K == 0 or K >= len(S):
            out[k] = num
            continue
        sigma_K = float(S[K])
        sigmas = np.maximum(S[:K].astype(float), 1e-12)
        scales = sigma_K / sigmas
        delta = float(np.sum((scales ** 2 - 1.0)
                             * (coefs[:K].astype(float) ** 2)))
        norm_sq = max(1.0 + delta, 1e-24)
        out[k] = num / float(np.sqrt(norm_sq))
    return out


def _clip_peak_log(x: float) -> float:
    """Clip a ``log₂(K+1)`` value to ``[0, log₂(33)]``.

    Off-grid (negative or >32) values are pushed onto the boundary; NaN
    is treated as 0 (i.e. "no preference for whitening" -- a defensible
    default given the y-axis is "amount of whitening that helps").
    """
    if not np.isfinite(x):
        return 0.0
    return float(min(np.log2(33.0), max(0.0, x)))


def _peak_log_from_fit_record(rec: dict) -> float:
    """Extract a comparable ``log₂(K+1)`` peak from either the weighted
    or non-weighted peak-fit record format.

    The weighted-fit JSON exposes ``peak_log`` directly (parabola peak
    in log₂(K+1) space, possibly off-grid).  The non-weighted di JSON
    instead exposes ``peak_K_log`` (the K value, clipped to [0, 32]);
    we convert via ``log₂(K+1)``.  Returned value is always clipped to
    the y-axis range used by the plot.
    """
    if "peak_log" in rec and rec["peak_log"] is not None:
        return _clip_peak_log(float(rec["peak_log"]))
    if "peak_K_log" in rec and rec["peak_K_log"] is not None:
        K = float(rec["peak_K_log"])
        if not np.isfinite(K):
            return 0.0
        return _clip_peak_log(np.log2(K + 1.0))
    return 0.0


def _per_axis_record(
    pair: dict, *,
    fits_by_pair: dict[tuple[str, str], dict[str, dict]],
    sweep_by_pair: dict[tuple[str, str], dict[str, dict[int, float]]],
    primary_set: set[tuple[str, str]],
    data_dir: Path, slot: int, layer: int, n_pcs: int,
    bottom_panel_Ks: tuple[int, ...],
) -> dict:
    """One row: cos²s, peak_log_avg, ρ@K, primary flag.

    ``primary_set`` -- pairs with both desc+inst AND responses
    available; rows for non-primary pairs fall back to desc+inst only
    (and are styled faint+dotted by :func:`make_plot`).
    ``bottom_panel_Ks`` -- which K values the bottom panel needs ρ
    for; we only persist those rather than the whole sweep.
    """
    pos, neg = pair["pos"], pair["neg"]
    is_primary = (pos, neg) in primary_set

    a = _axis_unit_vector(data_dir, pos, neg, slot=slot, layer=layer)
    Vt, S = _pool_top_PCs(data_dir, pos, neg,
                          slot=slot, layer=layer, n_pcs=n_pcs)
    cos = np.abs(Vt @ a)  # |cos| with each PC (pre-whitening)
    cos_sq = (cos ** 2).astype(float)
    # Panel 3 wants ``|cos(axis after soft-K=k-1 whitening, PC_k)|`` so
    # we compute that here against the same pool's singular values.
    cos_post_whitening = _post_whitening_cosines(
        a, Vt, S, n_pcs_shown=n_pcs,
    ).astype(float)

    # ----- peak_log: average two sources or fall back to desc_inst -----
    fits_here = fits_by_pair.get((pos, neg), {})
    pk_di = _peak_log_from_fit_record(fits_here["desc_inst"]) \
        if "desc_inst" in fits_here else 0.0
    pk_rs = _peak_log_from_fit_record(fits_here["responses"]) \
        if "responses" in fits_here else float("nan")
    # Project-default cross-mode blend: 0.80 * response + 0.20 * desc+inst
    # (assistant_axis.judge_score_combine.DEFAULT_RESPONSE_DI_WEIGHT,
    # picked from response_di_weight_sweep at slot 6 / layer 25).  Fall
    # back to desc+inst alone for axes without responses scoring.
    w_rs = DEFAULT_RESPONSE_DI_WEIGHT
    if is_primary and np.isfinite(pk_rs):
        pk_avg = w_rs * pk_rs + (1.0 - w_rs) * pk_di
    else:
        pk_avg = pk_di

    # ----- ρ@K: 0.80 response + 0.20 desc+inst, or desc+inst only -----
    rho_at_K: dict[int, float] = {}
    sweep_here = sweep_by_pair.get((pos, neg), {})
    rho_di_by_K = sweep_here.get("desc_inst", {})
    rho_rs_by_K = sweep_here.get("responses", {})
    for K in bottom_panel_Ks:
        rdi = rho_di_by_K.get(K, float("nan"))
        rrs = rho_rs_by_K.get(K, float("nan"))
        rdi_f = float(rdi) if rdi is not None else float("nan")
        rrs_f = float(rrs) if rrs is not None else float("nan")
        if is_primary and np.isfinite(rrs_f) and np.isfinite(rdi_f):
            rho_at_K[K] = w_rs * rrs_f + (1.0 - w_rs) * rdi_f
        elif np.isfinite(rdi_f):
            rho_at_K[K] = rdi_f
        else:
            rho_at_K[K] = float("nan")

    return {
        "pos": pos,
        "neg": neg,
        "name": pos,
        "primary": is_primary,
        "cos_sq": cos_sq.tolist(),
        "cos_post_whitening": cos_post_whitening.tolist(),
        "peak_log_di": pk_di,
        "peak_log_rs": pk_rs if np.isfinite(pk_rs) else None,
        "peak_log_avg": pk_avg,
        "rho_at_K": rho_at_K,
    }


def _color_for(idx: int, n: int) -> tuple[float, float, float, float]:
    """Legacy qualitative-palette colour (unused; kept for back-compat)."""
    cmap = plt.colormaps["tab20"] if n > 10 else plt.colormaps["tab10"]
    return cmap(idx % cmap.N)


# Category colour-scheme: classify each axis by the *shape* of its
# post-whitening |cos| sequence x_k = |cos(axis_w_{K=k-1}, PC_k)| for
# k = 1..4, using a cascading threshold rule.  The second condition
# in each level demands that the *next* level not be "as strong in
# its own threshold band" -- i.e. that the axis hasn't already
# graduated to the next category:
#
# - red    -- x_1 > 0.33  AND  x_2 * 0.33 < x_1 * 0.25
#               (PC1 dominates after K=0; x_2/x_1 < 0.25/0.33 ≈ 0.758)
# - yellow -- x_2 > 0.25  AND  x_3 * 0.25 < x_2 * 0.20
#               (PC2 dominates after K=1; x_3/x_2 < 0.20/0.25 = 0.800)
# - green  -- x_3 > 0.20  AND  x_4 < x_3
#               (PC3 dominates after K=2)
# - blue   -- everything else (no clear peak in PC1-3, or axis lives
#                              in PC4+)
# - gray   -- post-whitening cos sequence unavailable (shouldn't happen)
from matplotlib.colors import LinearSegmentedColormap as _LSC

# Custom yellow→gold ramp; built-in matplotlib has no pure-yellow
# sequential map (Wistia / YlOrBr drift to orange/brown at the top).
# Lower bound is a readable mid-yellow rather than near-white.
_YELLOW_CMAP = _LSC.from_list("Yellows_custom", ["#FFE34D", "#B58900"])

# Custom red ramp keyed by K=0 preference (ρ@K=0 − ρ@K=1).  Orange end
# = "most K=1-loving" red (smallest, possibly negative delta), dark red
# end = "most K=0-loving" red (largest delta).  Avoids the built-in
# ``Reds`` near-white pastel low-end and lets reds carry sub-category
# information without changing the category boundary.
_RED_K0_CMAP = _LSC.from_list("RedK0",
                              ["#FF8C00", "#E25A1C", "#B71C1C", "#7F0000"])

_CATEGORY_CMAPS: dict[str, object] = {
    "red":    _RED_K0_CMAP,
    "yellow": _YELLOW_CMAP,
    "green":  plt.colormaps["Greens"],
    "blue":   plt.colormaps["Blues"],
    "gray":   plt.colormaps["Greys"],
}
_CATEGORY_LABEL = {
    "red":    "x₁ > 0.33  ∧  x₂·0.33 < x₁·0.25",
    "yellow": "x₂ > 0.25  ∧  x₃·0.25 < x₂·0.20",
    "green":  "x₃ > 0.20  ∧  x₄ < x₃",
    "blue":   "else",
    "gray":   "(no post-whitening cos)",
}

_CATEGORY_ORDER: tuple[str, ...] = ("red", "yellow", "green", "blue", "gray")


def _categorize_row(row: dict) -> tuple[str, float]:
    """Return ``(category, strength)`` for the threshold colour scheme.

    The cascading rule on the post-whitening cosine sequence
    ``x_k = |cos(axis_w_{K=k-1}, PC_k)|`` for k=1..4::

        if x_1 > 0.33 and x_2 * 0.33 < x_1 * 0.25:  red    (strength = x_1)
        elif x_2 > 0.25 and x_3 * 0.25 < x_2 * 0.20: yellow (strength = x_2)
        elif x_3 > 0.20 and x_4 < x_3:               green  (strength = x_3)
        else:                                        blue   (strength = max)

    The second condition for red/yellow demands that the next level
    not be "as strong in its own threshold band" -- equivalently,
    ``x_{k+1} / x_k < t_{k+1} / t_k`` where ``t = (0.33, 0.25, 0.20)``.
    That way an axis only gets the higher (lower-k) category if it
    really hasn't graduated to the next level.

    ``strength`` is the magnitude at the category-defining index
    (the value that "won"); for blue we use the overall max because
    blue has no single defining position.  Returned strength is used
    later to pick the within-category shade.
    """
    cw = np.asarray(row.get("cos_post_whitening", []), dtype=float)
    if cw.size < 4:
        return "gray", 0.0
    x1, x2, x3, x4 = (float(cw[0]), float(cw[1]),
                      float(cw[2]), float(cw[3]))
    if x1 > 0.33 and x2 * 0.33 < x1 * 0.25:
        return "red", x1
    if x2 > 0.25 and x3 * 0.25 < x2 * 0.20:
        return "yellow", x2
    if x3 > 0.20 and x4 < x3:
        return "green", x3
    return "blue", max(x1, x2, x3, x4)


def _compute_category_colors(rows: list[dict]) -> list:
    """Per-row RGBA colours under the threshold category scheme.

    Within each non-red category, the strengths (the category-defining
    cosine magnitude) are normalised to the colormap's ``[0.40,
    0.95]`` band so all rows are readable (no near-white pastels)
    without saturating to black.

    The red category is special-cased: shade reflects K=0 preference
    (``ρ@K=0 − ρ@K=1``), so dark red = "most K=0-loving" red and
    orange = "most K=1-loving" red.  The red ramp itself already
    starts at orange, so reds use the full ``[0.0, 1.0]`` colormap
    range instead of the [0.40, 0.95] band.
    """
    cats = [_categorize_row(r) for r in rows]
    # Override red strength with K=0 preference (rho@K=0 − rho@K=1).
    effective: list[tuple[str, float]] = []
    for (cat, val), row in zip(cats, rows):
        if cat == "red":
            rho_k = row.get("rho_at_K", {})
            r0 = float(rho_k.get(0, float("nan")))
            r1 = float(rho_k.get(1, float("nan")))
            delta = 0.0 if (np.isnan(r0) or np.isnan(r1)) else (r0 - r1)
            effective.append((cat, delta))
        else:
            effective.append((cat, val))
    per_cat_min: dict[str, float] = {}
    per_cat_max: dict[str, float] = {}
    for cat, val in effective:
        per_cat_min[cat] = min(per_cat_min.get(cat, val), val)
        per_cat_max[cat] = max(per_cat_max.get(cat, val), val)
    colors = []
    for cat, val in effective:
        cmap = _CATEGORY_CMAPS[cat]
        lo, hi = per_cat_min[cat], per_cat_max[cat]
        if cat == "red":
            v_norm = 0.5 if hi <= lo else (val - lo) / (hi - lo)
        else:
            v_norm = 0.7 if hi <= lo else 0.40 + 0.55 * (val - lo) / (hi - lo)
        colors.append(cmap(v_norm))
    return colors


def _stagger_y_for_ties(
    rows: list[dict], *, bin_width: float = 0.05, step: float = 0.06,
) -> dict[str, float]:
    """Assign each row a ``y_plot`` that visually separates ties.

    All four (top-N) markers for a given entity share the same y
    (``peak_log_avg`` is fixed per entity), so groups of entities at
    the same fitted peak K produce a single overlapping horizontal
    band.  This helper bins entities by ``round(peak_log_avg /
    bin_width)`` and spreads the rows in each multi-member bin
    symmetrically about the bin centre with spacing ``step``.

    Sort within a bin is by ``name`` for a deterministic stagger
    across re-renders.

    Returns a ``{name: y_plot}`` map; rows not in any tied bin map to
    their original ``peak_log_avg``.
    """
    from collections import defaultdict
    bins: "dict[int, list[dict]]" = defaultdict(list)
    for r in rows:
        b = int(round(float(r["peak_log_avg"]) / bin_width))
        bins[b].append(r)
    y_plot: dict[str, float] = {}
    for b, group in bins.items():
        if len(group) == 1:
            y_plot[group[0]["name"]] = float(group[0]["peak_log_avg"])
            continue
        ordered = sorted(group, key=lambda r: r["name"])
        n = len(ordered)
        offsets = (np.arange(n) - (n - 1) / 2.0) * step
        for r, off in zip(ordered, offsets):
            y_plot[r["name"]] = float(r["peak_log_avg"] + off)
    return y_plot


def _per_axis_xs(row: dict, top_ns: tuple[int, ...]) -> list[float]:
    """The 4 x-coordinates (cumulative cos² at top-N) for a row."""
    return [float(np.sum(row["cos_sq"][:N])) for N in top_ns]


def _draw_top_panel(
    ax, primary_rows, extra_rows, primary_colors, extra_colors,
    top_ns: tuple[int, ...],
) -> dict[int, float]:
    """Top panel: y = peak_log_avg, horizontal line per axis.

    Returns ``{N: spearman_rho}`` over the PRIMARY-cohort rows for
    use in the shared legend.
    """
    # Stagger ties only within the union (so a primary row and a
    # faint row at the same peak K still separate visually).
    y_plot_by_name = _stagger_y_for_ties(primary_rows + extra_rows)

    rho_per_N: dict[int, float] = {}
    for N in top_ns:
        xs = np.array([float(np.sum(r["cos_sq"][:N])) for r in primary_rows])
        ys = np.array([float(r["peak_log_avg"]) for r in primary_rows])
        rho_per_N[N] = float(spearmanr(xs, ys).correlation)

    def _plot_group(rows, colors, *, alpha, lw, ls, label):
        for row, color in zip(rows, colors):
            xs = _per_axis_xs(row, top_ns)
            y_actual = float(row["peak_log_avg"])
            y = y_plot_by_name[row["name"]]
            ax.plot(xs, [y] * len(top_ns), color=color, linewidth=lw,
                    alpha=alpha, linestyle=ls, zorder=2 if alpha > 0.5 else 1)
            for x, mk in zip(xs, _MARKERS):
                ax.scatter(x, y, s=60, marker=mk, color=color,
                           edgecolor="black", linewidth=0.5, alpha=alpha,
                           zorder=3 if alpha > 0.5 else 2)
            if label:
                ax.annotate(
                    corpus_display_name(row["name"]),
                    (xs[0], y), fontsize=7,
                    xytext=(6, 0), textcoords="offset points",
                    va="center", color="black", alpha=alpha,
                )
            if abs(y - y_actual) > 1e-6:
                tick_dx = 0.005
                ax.plot([xs[-1] - tick_dx, xs[-1] + tick_dx],
                        [y_actual, y_actual],
                        color=color, linewidth=0.5, alpha=0.35 * alpha,
                        zorder=1)

    # Extras drawn first (under primaries) WITHOUT labels -- colour
    # alone gives enough trace to follow individual lines and labels
    # would crush the middle of the panel.  Primaries get labels.
    _plot_group(extra_rows, extra_colors, alpha=0.80, lw=0.9, ls="--",
                label=False)
    _plot_group(primary_rows, primary_colors, alpha=0.95, lw=1.2, ls="-",
                label=True)

    ax.set_ylabel("Mean fitted peak  log₂(K+1)  (clipped to ≥ 0)",
                  fontsize=10)
    ax.grid(alpha=0.3)
    return rho_per_N


def _draw_pc_cosine_panel(
    ax, primary_rows, extra_rows, primary_colors, extra_colors,
    *, n_pcs_shown: int,
) -> None:
    """Bottom panel: |cos(axis_w, PC_k)| vs PC index k = 1..n_pcs_shown,
    where ``axis_w`` is the axis after soft-K=(k-1) whitening (PCs
    themselves are pre-whitening).

    x=1: K=0, i.e. no whitening, plain |cos(axis, PC_1)|.
    x=2: K=1 whitening of the axis, then cos with the (untouched) PC_2.
    x=3: K=2 whitening of the axis, then cos with PC_3.
    x=4: K=3 whitening of the axis, then cos with PC_4.

    Same per-axis colour scheme as the upper two panels.  No
    per-marker shape variation here -- the x-position already encodes
    which PC we're looking at, so plain dots are the cleanest read.
    """
    xs = np.arange(1, n_pcs_shown + 1)  # PC indices on the x-axis (1-based)

    def _plot_group(rows, colors, *, alpha, lw, ls):
        for row, color in zip(rows, colors):
            ys_arr = np.asarray(row["cos_post_whitening"], dtype=float)
            ys = np.clip(ys_arr[:n_pcs_shown], 0.0, None)
            ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                    linestyle=ls,
                    zorder=2 if alpha > 0.5 else 1)
            ax.scatter(xs, ys, s=40, marker="o", color=color,
                       edgecolor="black", linewidth=0.4, alpha=alpha,
                       zorder=3 if alpha > 0.5 else 2)

    _plot_group(extra_rows, extra_colors, alpha=0.80, lw=0.9, ls="--")
    _plot_group(primary_rows, primary_colors, alpha=0.95, lw=1.2, ls="-")

    # Label EVERY axis (primary + extra) at x=1 and x=4 so all 35
    # lines can be identified at a glance.  Stagger label y-positions
    # within each x-column so labels with similar data-y don't
    # overlap.  Primary labels use full opacity; extra labels use
    # 0.80 (matching the line opacity) so the eye can still tell the
    # two cohorts apart.
    def _stagger(yvs: list[float], min_sep: float) -> list[float]:
        """Return y positions with at least ``min_sep`` between adjacent
        labels (after sorting by desired y).  Greedy bottom-up pass."""
        order = sorted(range(len(yvs)), key=lambda i: yvs[i])
        out = [0.0] * len(yvs)
        last = -np.inf
        for i in order:
            y = max(yvs[i], last + min_sep)
            out[i] = y
            last = y
        return out

    all_rows = list(primary_rows) + list(extra_rows)
    all_colors = list(primary_colors) + list(extra_colors)
    all_alphas = [0.95] * len(primary_rows) + [0.80] * len(extra_rows)

    # x=1 labels go LEFT (toward y-axis margin); x=4 labels go RIGHT
    # (into the right margin past the data area).  bbox_inches="tight"
    # expands the figure to fit either side.
    for x_pos, ha, dx in ((1, "right", -0.08), (4, "left", +0.08)):
        if x_pos > n_pcs_shown:
            continue
        ys_at_x = [float(np.clip(np.asarray(r["cos_post_whitening"],
                                            dtype=float)[x_pos - 1],
                                 0.0, None))
                   for r in all_rows]
        ys_label = _stagger(ys_at_x, min_sep=0.018)
        for row, color, alpha, y_data, y_lbl in zip(
                all_rows, all_colors, all_alphas, ys_at_x, ys_label):
            ax.annotate(
                corpus_display_name(row["name"]),
                xy=(x_pos, y_data),
                xytext=(x_pos + dx, y_lbl),
                fontsize=6.5, ha=ha, va="center",
                color=color, alpha=alpha, zorder=4,
                arrowprops=dict(arrowstyle="-",
                                color=color, alpha=alpha * 0.4, lw=0.5,
                                shrinkA=0, shrinkB=2),
            )

    ax.set_xlabel("PC index $k$  (whitening soft-K = $k-1$)", fontsize=10)
    ax.set_ylabel("|cos(axis$_w$, $PC_k$)|", fontsize=10)
    ax.set_xticks(list(xs))
    ax.set_xlim(xs[0] - 0.3, xs[-1] + 0.3)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)

    # Category legend: one swatch per category at a mid-saturation
    # shade.  Counts come from the full union of rows.
    cat_counts: dict[str, int] = {c: 0 for c in _CATEGORY_ORDER}
    for r in primary_rows + extra_rows:
        cat, _ = _categorize_row(r)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    cat_handles = []
    for cat in _CATEGORY_ORDER:
        if cat_counts.get(cat, 0) == 0:
            continue
        cmap = _CATEGORY_CMAPS[cat]
        h = ax.plot([], [], marker="o", linestyle="-", color=cmap(0.75),
                    markeredgecolor="black", markeredgewidth=0.4,
                    label=f"{_CATEGORY_LABEL[cat]}   (n={cat_counts[cat]})")[0]
        cat_handles.append(h)
    ax.legend(handles=cat_handles,
              title="Colour: post-whitening cos shape",
              loc="upper right", fontsize=8, title_fontsize=8,
              framealpha=0.85)


def _draw_bottom_panel(
    ax, primary_rows, extra_rows, primary_colors, extra_colors,
    top_ns: tuple[int, ...], bottom_panel_Ks: tuple[int, ...],
) -> dict[int, float]:
    """Bottom panel: y = ρ at K (one K per marker, indexed by marker).

    Marker i sits at ``x = top-N_i cos²`` and ``y = ρ at K_i``, where
    ``K_i = bottom_panel_Ks[i]``.  Returns ``{K: spearman_rho}`` over
    the PRIMARY rows (paired x = top-N cos² of marker i, y = ρ@K_i).
    """
    rho_per_K: dict[int, float] = {}
    for i, K in enumerate(bottom_panel_Ks):
        N = top_ns[i] if i < len(top_ns) else top_ns[-1]
        xs_p = np.array([float(np.sum(r["cos_sq"][:N])) for r in primary_rows])
        ys_p = np.array([float(r["rho_at_K"].get(K, np.nan))
                         for r in primary_rows])
        mask = np.isfinite(xs_p) & np.isfinite(ys_p)
        if mask.sum() >= 3:
            rho_per_K[K] = float(spearmanr(xs_p[mask], ys_p[mask]).correlation)
        else:
            rho_per_K[K] = float("nan")

    def _plot_group(rows, colors, *, alpha, lw, ls):
        for row, color in zip(rows, colors):
            xs = _per_axis_xs(row, top_ns)
            ys = [float(row["rho_at_K"].get(K, np.nan))
                  for K in bottom_panel_Ks]
            xs_finite, ys_finite, mks_finite = [], [], []
            for i, (x, y) in enumerate(zip(xs, ys)):
                if np.isfinite(y):
                    xs_finite.append(x)
                    ys_finite.append(y)
                    mks_finite.append(_MARKERS[i % len(_MARKERS)])
            if not xs_finite:
                continue
            ax.plot(xs_finite, ys_finite, color=color, linewidth=lw,
                    alpha=alpha, linestyle=ls,
                    zorder=2 if alpha > 0.5 else 1)
            for x, y, mk in zip(xs_finite, ys_finite, mks_finite):
                ax.scatter(x, y, s=60, marker=mk, color=color,
                           edgecolor="black", linewidth=0.5, alpha=alpha,
                           zorder=3 if alpha > 0.5 else 2)

    _plot_group(extra_rows, extra_colors, alpha=0.80, lw=0.9, ls="--")
    _plot_group(primary_rows, primary_colors, alpha=0.95, lw=1.2, ls="-")

    ax.set_ylabel(
        "Observed ρ  (0.80 responses + 0.20 desc+instr)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    return rho_per_K


def make_plot(
    rows: list[dict],
    top_ns: tuple[int, ...],
    bottom_panel_Ks: tuple[int, ...],
    *,
    out_path: Path,
    title_extra: str,
    inputs: list[InputSpec],
    n_pcs_panel3: int = 4,
) -> None:
    """Render the three-panel scatter and write to ``out_path``.

    Panels (top -> bottom):
      1. peak K vs cumulative cos²  (shared x with panel 2, inverted)
      2. observed ρ@K vs cumulative cos²  (shared x with panel 1)
      3. |cos(axis, PC_k)| vs PC index k = 1..``n_pcs_panel3``
         (independent x-axis -- different scale from panels 1/2).
    """
    primary_rows = [r for r in rows if r["primary"]]
    extra_rows = [r for r in rows if not r["primary"]]
    # Per-axis colour comes from the cos-shape category of each row
    # (see _categorize_row).  Category-shade normalisation runs over
    # the FULL union so a primary and an extra in the same category
    # get comparable saturation; primary vs extra distinction is
    # carried entirely by line style + opacity in the plotting
    # helpers.
    all_colors = _compute_category_colors(rows)
    by_name_color = {r["name"]: c for r, c in zip(rows, all_colors)}
    primary_colors = [by_name_color[r["name"]] for r in primary_rows]
    extra_colors = [by_name_color[r["name"]] for r in extra_rows]

    # 3-panel stacked layout.  Panel 3 has a different x-axis from 1/2
    # (PC index, not cumulative cos²) so we use an explicit gridspec
    # instead of sharex=True everywhere; the third panel is given a
    # bit less vertical room because it only spans 4 x-ticks.
    fig = plt.figure(figsize=(12.0, 16.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.7])
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    ax_pc = fig.add_subplot(gs[2])  # independent x-axis.

    rho_top = _draw_top_panel(
        ax_top, primary_rows, extra_rows, primary_colors, extra_colors,
        top_ns,
    )
    rho_bot = _draw_bottom_panel(
        ax_bot, primary_rows, extra_rows, primary_colors, extra_colors,
        top_ns, bottom_panel_Ks,
    )
    _draw_pc_cosine_panel(
        ax_pc, primary_rows, extra_rows, primary_colors, extra_colors,
        n_pcs_shown=n_pcs_panel3,
    )

    # Shared x-axis (panels 1 & 2): invert ONCE (sharex propagates).
    ax_top.invert_xaxis()
    ax_bot.set_xlabel(
        r"Axis energy in top-$N$ PCs of held-out pool: "
        r"$\sum_{k=1}^{N} \cos^2(\mathrm{axis}, PC_k)$",
        fontsize=10,
    )

    # Marker legend on the TOP panel (shape -> N + per-panel ρ),
    # plus a second legend on the BOTTOM panel (shape -> K + per-K ρ).
    n_handles = []
    for N, mk in zip(top_ns, _MARKERS):
        n_handles.append(ax_top.scatter(
            [], [], s=70, marker=mk,
            color="white", edgecolor="black", linewidth=0.8,
            label=f"top-{N}   ρ = {rho_top[N]:+.2f}",
        ))
    ax_top.legend(
        handles=n_handles,
        title="Top panel:  N  (Spearman ρ vs peak K, primary cohort)",
        loc="upper left", fontsize=9, title_fontsize=9, framealpha=0.85,
    )

    k_handles = []
    for i, K in enumerate(bottom_panel_Ks):
        mk = _MARKERS[i % len(_MARKERS)]
        rho = rho_bot.get(K, float("nan"))
        rho_str = f"{rho:+.2f}" if np.isfinite(rho) else " n/a"
        k_handles.append(ax_bot.scatter(
            [], [], s=70, marker=mk,
            color="white", edgecolor="black", linewidth=0.8,
            label=f"K = {K}   ρ = {rho_str}",
        ))
    # Style sample handles for solid (primary) and dashed (extra).
    style_handles = [
        ax_bot.plot([], [], color="black", linewidth=1.2, linestyle="-",
                    label=f"primary cohort  "
                          f"({DEFAULT_RESPONSE_DI_WEIGHT:.2f} responses + "
                          f"{1 - DEFAULT_RESPONSE_DI_WEIGHT:.2f} desc+instr)")[0],
        ax_bot.plot([], [], color="black", linewidth=0.9, linestyle="--",
                    alpha=0.8,
                    label="desc+instr only (responses unavailable)")[0],
    ]
    leg_K = ax_bot.legend(
        handles=k_handles,
        title="Bottom panel:  K  (Spearman ρ vs ρ@K, primary cohort)",
        loc="upper left", fontsize=9, title_fontsize=9, framealpha=0.85,
    )
    ax_bot.add_artist(leg_K)
    ax_bot.legend(handles=style_handles, loc="lower left",
                  fontsize=8, framealpha=0.85)

    title_line = "Axis–PC alignment vs whitening sweet spot vs ρ@K"
    spec_line = (
        f"Top: peak K  (log₂(K+1)).   Middle: observed ρ@K, "
        f"K=({','.join(str(K) for K in bottom_panel_Ks)}).   "
        f"Bottom: |cos(axis$_w$, $PC_k$)| after soft-K=k-1 whitening, "
        f"k=1..{n_pcs_panel3}.   "
        f"Solid = primary cohort; dashed = desc+instr only"
    )
    if title_extra:
        spec_line = f"{spec_line};   {title_extra}"
    _, top_rect = suptitle_with_specs(fig, title_line, spec_line)
    plt.tight_layout(rect=(0, 0, 1, top_rect))

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)
    print(f"\nWrote {out_path}")
    print(
        "  Top panel ρ (top-N cos² vs peak K):  "
        + ";  ".join(f"top-{N} ρ = {rho_top[N]:+.3f}" for N in top_ns)
    )
    print(
        "  Bottom panel ρ (top-N cos² vs ρ@K):  "
        + ";  ".join(
            f"K={K} ρ = {rho_bot.get(K, float('nan')):+.3f}"
            for K in bottom_panel_Ks
        )
    )


def _index_fits(fits: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    """``[{pos, neg, source, ...}]`` -> ``{(pos, neg): {source: rec}}``."""
    out: dict[tuple[str, str], dict[str, dict]] = {}
    for r in fits:
        out.setdefault((r["pos"], r["neg"]), {})[r["source"]] = r
    return out


def _index_sweep(
    sweep: list[dict],
) -> dict[tuple[str, str], dict[str, dict[int, float]]]:
    """``[{pos, neg, source, K, rho, ...}]`` -> nested ``{(pos, neg):
    {source: {K: rho}}}``."""
    out: dict[tuple[str, str], dict[str, dict[int, float]]] = {}
    for r in sweep:
        ent = out.setdefault((r["pos"], r["neg"]), {})
        per_src = ent.setdefault(r["source"], {})
        per_src[int(r["K"])] = r["rho"]
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR),
                   help=f"Directory holding pair-list + peak-fit JSONs and "
                        f"output PNG (default: {DEFAULT_EXPERIMENT_DIR}).")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                   help=f"Vector tree (default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--pairs", default=DEFAULT_PAIR_LIST,
                   help=f"Pair-list JSON filename for the FULL cohort, "
                        f"relative to --experiment_dir "
                        f"(default: {DEFAULT_PAIR_LIST}).")
    p.add_argument("--pairs_primary", default=DEFAULT_PRIMARY_PAIR_LIST,
                   help=f"Pair-list JSON filename for the PRIMARY cohort "
                        f"(axes with both desc+inst AND responses scoring); "
                        f"axes in --pairs but not in --pairs_primary are "
                        f"styled faint+dotted.  Default: "
                        f"{DEFAULT_PRIMARY_PAIR_LIST}.")
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT,
                   help=f"Token slot (default: {DEFAULT_SLOT}).")
    p.add_argument("--layer", type=int, default=LAYER,
                   help=f"Hidden-state layer (default: {LAYER}).")
    p.add_argument("--top_ns", type=int, nargs="+",
                   default=list(DEFAULT_TOP_NS),
                   help="Which top-N cumulative cos² windows to use for the "
                        f"x-coordinate of each marker (default: "
                        f"{list(DEFAULT_TOP_NS)}).")
    p.add_argument("--bottom_panel_Ks", type=int, nargs="+", default=None,
                   help="Bottom-panel marker -> K mapping (default: 0..N-1, "
                        "where N = len(--top_ns)).")
    p.add_argument("--n_pcs_panel3", type=int, default=4,
                   help="How many PC indices to show in the 3rd panel "
                        "(|cos(axis, PC_k)| vs k = 1..N).  Default: 4.")
    p.add_argument("--out", default=None,
                   help="Output PNG filename (default: "
                        "axis_pc_alignment_vs_peak_K_<cohort>_slot{N}.png).")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to react to drift in the peak-fit JSON's "
                        "recorded provenance (default: warn).")
    args = p.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"--data_dir does not exist: {data_dir}")
    top_ns = tuple(args.top_ns)
    if any(n < 1 for n in top_ns):
        raise SystemExit(f"--top_ns must be positive integers, got {top_ns}")
    n_pcs = max(N_PCS, max(top_ns))
    bottom_panel_Ks = tuple(
        args.bottom_panel_Ks
        if args.bottom_panel_Ks is not None
        else [_bottom_panel_K_for_index(i) for i in range(len(top_ns))]
    )
    if len(bottom_panel_Ks) != len(top_ns):
        raise SystemExit(
            f"--bottom_panel_Ks length ({len(bottom_panel_Ks)}) must match "
            f"--top_ns length ({len(top_ns)})"
        )

    cohort = cohort_from_pairs(args.pairs)
    fit_filename = f"whitening_k_peak_fit_{cohort}_slot{args.slot}.json"
    sweep_filename = f"whitening_k_sweep_{cohort}_slot{args.slot}.json"
    out_filename = args.out or (
        f"axis_pc_alignment_vs_peak_K_{cohort}_slot{args.slot}.png"
    )

    # ---- Load inputs with provenance ----
    inputs: list[InputSpec] = []

    pairs, _, _ = load_and_register(
        experiment_dir / args.pairs, dep_key="pairs_json",
        inputs=inputs, policy=args.cache_policy,
    )
    pairs_primary, _, _ = load_and_register(
        experiment_dir / args.pairs_primary, dep_key="pairs_primary_json",
        inputs=inputs, policy=args.cache_policy,
    )
    fits, _, _ = load_and_register(
        experiment_dir / fit_filename, dep_key="peak_fit_json",
        inputs=inputs, policy=args.cache_policy,
    )
    sweep, _, _ = load_and_register(
        experiment_dir / sweep_filename, dep_key="k_sweep_json",
        inputs=inputs, policy=args.cache_policy,
    )
    inputs.extend([
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras={"slot": str(args.slot), "layer": str(args.layer)}),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras={"slot": str(args.slot), "layer": str(args.layer)}),
    ])

    primary_set = {(it["pos"], it["neg"]) for it in pairs_primary}
    fits_by_pair = _index_fits(fits)
    sweep_by_pair = _index_sweep(sweep)

    # ---- Per-axis records ----
    rows: list[dict] = []
    for pair in pairs:
        rows.append(_per_axis_record(
            pair,
            fits_by_pair=fits_by_pair, sweep_by_pair=sweep_by_pair,
            primary_set=primary_set,
            data_dir=data_dir, slot=args.slot, layer=args.layer,
            n_pcs=n_pcs, bottom_panel_Ks=bottom_panel_Ks,
        ))

    n_primary = sum(1 for r in rows if r["primary"])
    n_extra = len(rows) - n_primary
    print(f"Loaded {len(rows)} axes total ({n_primary} primary + "
          f"{n_extra} desc+instr-only extras)")

    # ---- Console summary (sorted by top-Nmax cos²) ----
    header_n = "  ".join(f"top{N:>2d}" for N in top_ns)
    header_K = "  ".join(f"ρ@K{K:>2d}" for K in bottom_panel_Ks)
    print(f"\n{'axis':35s} | P | {header_n} | {header_K} | {'pkAvg':>6s}")
    print("-" * (40 + len(header_n) + len(header_K) + 12))
    for r in sorted(rows,
                    key=lambda x: -float(np.sum(x['cos_sq'][:max(top_ns)]))):
        cells_n = "  ".join(f"{float(np.sum(r['cos_sq'][:N])):5.3f}"
                            for N in top_ns)
        cells_K = "  ".join(f"{r['rho_at_K'].get(K, float('nan')):+5.2f}"
                            for K in bottom_panel_Ks)
        flag = "*" if r["primary"] else " "
        print(f"{r['pos']+'/'+r['neg']:35s} | {flag} | {cells_n} | "
              f"{cells_K} | {r['peak_log_avg']:>6.2f}")

    # ---- Plot ----
    title_extra = (
        f"slot {args.slot}, layer {args.layer}, "
        f"n={n_primary}+{n_extra} axes"
    )
    out_path = experiment_dir / out_filename
    make_plot(rows, top_ns, bottom_panel_Ks,
              out_path=out_path, title_extra=title_extra, inputs=inputs,
              n_pcs_panel3=args.n_pcs_panel3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
