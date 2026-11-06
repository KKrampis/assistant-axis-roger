"""CA-plane decomposition + visualisation library.

Tools for inspecting any single canonical-angle plane between two
subspaces (CA1, CA2, ...): the math (canonical vectors a_k / b_k,
common-mode bisector e_+, differential bisector e_-, the orthogonalising
shear within that 2-plane) and the corresponding figure (side-by-side
pre-shear / post-shear panels with all 579 standalone entities projected
into the plane, semantic-priority labels, and the four corpus centroids
overlaid).

This module is geometry-agnostic about how the goal / no-goal subspaces are
constructed; the caller hands in two arrays of shape ``(n, D)`` and the
list of entities to plot.  Standard usage from the package's plot wrappers:

    from results_analysis.canonical_angles.ca1_plane import (
        compute_ca_decomposition,
        plot_ca_plane_pre_post_shear,
    )

    decomp = compute_ca_decomposition(goal_subspace, nogoal_subspace,
                                       ca_index=1)
    fig = plot_ca_plane_pre_post_shear(
        decomposition=decomp,
        entities=[{"name": ..., "etype": ..., "vector": ..., "status": ...}, ...],
        centroids={"all roles": ..., "all traits": ..., ...},
        centering_vector=default_v + pool_median,
        ...
    )
    fig.savefig(out_path, ...)

The wrapper(s) in ``plots/`` handle CLI parsing, data loading, and PNG
metadata embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from assistant_axis.entity_id import corpus_display_name


# ---------------------------------------------------------------------------
# CA decomposition: math + parametric description of one CA plane
# ---------------------------------------------------------------------------

@dataclass
class CADecomposition:
    """One canonical-angle pair (CA_k) plus the volume-preserving shear that
    orthogonalises it.

    Conventions: ``a`` is the k-th canonical vector inside subspace A
    (often "goal"), ``b`` inside subspace B (often "non-goal").  Both
    are unit vectors in the ambient space ``R^D``.  ``e_+`` and ``e_-``
    are the eigenvectors of the orthogonalising shear, lying in the
    ``span(a, b)`` plane:

        e_+ = (a + b) / (2 cos(theta_k / 2))     # common-mode bisector
        e_- = (a - b) / (2 sin(theta_k / 2))     # differential bisector

    The shear scales ``e_+`` by ``lambda_+ = sqrt(tan(theta_k / 2))`` and
    ``e_-`` by ``lambda_- = sqrt(cot(theta_k / 2))``.  Post-shear,
    ``a -> a' = (e_+ + e_-) / sqrt(2)`` and
    ``b -> b' = (e_+ - e_-) / sqrt(2)``, which are now orthogonal.

    The ``ca_index`` field records which canonical-angle pair this is
    (1 = first / smallest theta, 2 = second, ...) for downstream
    titling / metadata.
    """
    a: np.ndarray
    b: np.ndarray
    eplus: np.ndarray
    eminus: np.ndarray
    theta: float        # radians
    cos_theta: float
    a_sheared: np.ndarray
    b_sheared: np.ndarray
    lambda_plus: float
    lambda_minus: float
    ca_index: int = 1


def compute_ca_decomposition(
    A: np.ndarray, B: np.ndarray, ca_index: int = 1,
) -> CADecomposition:
    """Compute the ``ca_index``-th canonical-angle pair between ``span(A)`` and
    ``span(B)``.

    Parameters
    ----------
    A : (n_a, D) array.  Rows are samples in subspace A.
    B : (n_b, D) array.  Rows are samples in subspace B.
    ca_index : int (default 1)
        Which canonical-angle pair to return.  ``1`` = smallest angle;
        ``2`` = next smallest; etc.  Must satisfy
        ``1 <= ca_index <= min(rank(A), rank(B))``.

    Returns
    -------
    CADecomposition with all unit-vector fields normalised in float64.
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B must be 2-D with matching D; got {A.shape}, {B.shape}")
    if ca_index < 1:
        raise ValueError(f"ca_index must be >= 1; got {ca_index}")
    Qa, _ = np.linalg.qr(A.T)            # (D, n_a)
    Qb, _ = np.linalg.qr(B.T)
    U, sigma, Vt = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    if ca_index > len(sigma):
        raise ValueError(
            f"ca_index={ca_index} exceeds the number of canonical pairs "
            f"({len(sigma)}); reduce ca_index or use larger subspaces."
        )
    k = ca_index - 1
    sigma = np.clip(sigma, -1.0, 1.0)
    cos_th = float(sigma[k])
    theta = float(np.arccos(cos_th))
    half = theta / 2.0
    a = (Qa @ U[:, k]).astype(np.float64)
    b = (Qb @ Vt[k]).astype(np.float64)
    eplus  = (a + b) / (2.0 * np.cos(half))
    eminus = (a - b) / (2.0 * np.sin(half))
    eplus  = eplus  / np.linalg.norm(eplus)
    eminus = eminus / np.linalg.norm(eminus)
    a_sheared = (eplus + eminus) / np.sqrt(2.0)
    b_sheared = (eplus - eminus) / np.sqrt(2.0)
    return CADecomposition(
        a=a, b=b,
        eplus=eplus, eminus=eminus,
        theta=theta, cos_theta=cos_th,
        a_sheared=a_sheared, b_sheared=b_sheared,
        lambda_plus=float(np.sqrt(np.tan(half))),
        lambda_minus=float(np.sqrt(1.0 / np.tan(half))),
        ca_index=ca_index,
    )


# Backward-compat aliases (so any external callers from yesterday's commit
# don't break).  Just thin shims around the generalized API.
CA1Decomposition = CADecomposition

def compute_ca1_decomposition(A: np.ndarray, B: np.ndarray) -> CADecomposition:
    return compute_ca_decomposition(A, B, ca_index=1)


# ---------------------------------------------------------------------------
# Labelling utilities for the in-plane scatter plot
# ---------------------------------------------------------------------------

# Semantic priority bonuses for the collision-aware labeller.  Higher-priority
# entries are placed first; later candidates that would overlap a placed label
# are silently dropped.
SEMANTIC_BASE_BONUS: Mapping[str, int] = {
    "assistant": 200, "helpful": 100, "unhelpful": 100,
    "harmless": 90, "harmful": 90, "honest": 90, "dishonest": 90,
    "truthful": 80, "deceitful": 80, "trustworthy": 80, "untrustworthy": 80,
    "compassionate": 70, "callous": 70, "malicious": 70, "benign": 70,
    "constructive": 60, "destructive": 60,
    "angel": 60, "demon": 50, "saint": 60, "bodhisattva": 60,
    "paperclip_maximizer": 80, "aligned_artificial_intelligence": 80,
    # aligned_artificial_intelligence was renamed 2026-09-28; both stems
    # are listed so plots of older extractions keep their priority
    "instrumentally_aligned_ai": 80, "virtue_aligned_ai": 80,
    "ecocentric": 40, "anthropocentric": 40, "individualistic": 40,
    "collectivistic": 40, "progressive": 40, "conservative": 40,
    "egalitarian": 35, "elitist": 35, "selfish": 30, "nihilistic": 30,
    "sociopathic": 50, "vindictive": 35, "manipulative": 35, "cruel": 35,
    "concise": 25, "verbose": 25, "casual": 25, "formal": 25,
    "improvisational": 25, "methodical": 25, "systems_thinker": 25,
    "analytical": 25, "reductionist": 25, "holistic": 25,
    "convergent": 25, "divergent": 25, "practical": 25, "theoretical": 25,
    "relativist": 25, "absolutist": 25, "forgiving": 25, "unforgiving": 25,
}


def collision_label(ax, candidates, *, fontsize: int = 7) -> int:
    """Greedy non-overlapping label placement.

    candidates : iterable of ``(etype, name, x_screen, y_screen, priority)``.
        Sorted descending by ``priority + magnitude / 6`` so high-priority
        and far-from-origin entries are placed first; later candidates that
        would overlap an already-placed label are silently dropped.

    Returns the number of labels actually placed.
    """
    candidates = sorted(
        candidates, key=lambda c: -(c[4] + np.hypot(c[2], c[3]) / 6),
    )
    kept = []
    for etype, name, x, y, _prio in candidates:
        side = "L" if x < 0 else "R"
        offset = (-4, 4) if side == "L" else (4, 3)
        ha = "right" if side == "L" else "left"
        ann = ax.annotate(
            corpus_display_name(name, etype), (x, y),
            fontsize=fontsize, xytext=offset, textcoords="offset points",
            ha=ha, color="#333", alpha=0.9, zorder=5,
            fontstyle="italic" if etype == "role" else "normal",
        )
        ax.figure.canvas.draw()
        bb = ann.get_window_extent().padded(1)
        if any(bb.overlaps(kb) for kb in kept):
            ann.remove()
        else:
            kept.append(bb)
    return len(kept)


# ---------------------------------------------------------------------------
# Pre-shear vs post-shear figure
# ---------------------------------------------------------------------------

@dataclass
class CAPlaneEntity:
    """One entity to plot in a CA plane.

    ``vector`` is in the ambient space; the plot helper subtracts
    ``centering_vector`` (if given) before projecting onto e_+ / e_-.
    """
    name: str
    etype: str          # "role" or "trait"
    vector: np.ndarray  # (D,)
    status: str = "neither"  # "goal" | "nogoal" | "neither"


# Backward-compat alias
CA1PlaneEntity = CAPlaneEntity


_STATUS_COLOURS = {
    "goal":    "#d62728",
    "nogoal":  "#1f77b4",
    "neither": "#888888",
}

_CENTROID_COLOURS = {
    "origin (pool median)": "#ffcc00",   # yellow star
    "all roles":            "#2ca02c",
    "all traits":           "#9467bd",
    "goal corpus":          "#d62728",
    "nogoal corpus":        "#1f77b4",
}


def plot_ca_plane_pre_post_shear(
    decomposition: CADecomposition,
    entities: Sequence[CAPlaneEntity],
    *,
    centroids: Mapping[str, Optional[np.ndarray]] | None = None,
    centering_vector: Optional[np.ndarray] = None,
    title: str | None = None,
    spec_lines: str = "",
    extra_label_priority: Mapping[str, int] | None = None,
    pre_panel_height_in: float = 7.5,
    pre_legend_pad_factor: float = 3.5,
):
    """Render a two-panel pre-shear / post-shear visualization of the CA plane.

    The figure is laid out in a 90-degree-CCW-rotated orientation:
    e_+ is vertical (positive up), e_- is horizontal (positive to the LEFT,
    achieved via ``ax.invert_xaxis()``).  Pre-shear is a tall narrow strip
    on the left (data hugs the e_+ axis); post-shear is a square on the
    right (data fills the panel after the e_-/e_+ shear redistribution).

    Parameters
    ----------
    decomposition : CA1Decomposition
        Output of :func:`compute_ca1_decomposition`.
    entities : sequence of CA1PlaneEntity
        Standalone entities to plot.  Each provides a vector in the
        ambient space, a name, an etype ("role" / "trait"), and a status
        ("goal" / "nogoal" / "neither") for colour-coding.
    centroids : optional mapping {label: vector or None}
        Special markers drawn as large stars on top of the scatter.
        Recognised labels (with their colours): ``"origin (pool median)"``
        (yellow), ``"all roles"`` (green), ``"all traits"`` (purple),
        ``"goal corpus"`` (red), ``"nogoal corpus"`` (blue).  Pass a
        zero-vector or skip the entry to omit a centroid.
    centering_vector : (D,) array, optional
        Subtracted from each entity vector and centroid before projection.
        Use e.g. ``default + pool_median`` to place origin at the pool
        median (the historical convention).  Default: no centering (origin
        = ambient zero).
    title : str
        Figure suptitle.
    spec_lines : str
        Multi-line spec block for the suptitle (e.g. layer / slot / shear
        eigenvalues).  Use newlines for multi-line.
    extra_label_priority : optional mapping {name: bonus}
        Extra priority bumps merged into :data:`SEMANTIC_BASE_BONUS` for
        this call only (caller doesn't mutate the module-level dict).
    pre_panel_height_in : float
        Height of the pre-shear panel in inches.  Post-shear panel is
        rendered at the same height for a square layout.
    pre_legend_pad_factor : float
        How many data-extents of empty horizontal margin to leave on the
        legend side of the pre-shear panel.  Larger -> more legend room.

    Returns
    -------
    matplotlib.figure.Figure
        Caller saves with ``fig.savefig(...)``.
    """
    from assistant_axis.plot_metadata import suptitle_with_specs

    if centering_vector is None:
        centering_vector = np.zeros_like(decomposition.eplus)
    if title is None:
        title = f"CA{decomposition.ca_index} plane: pre-shear vs post-shear"

    # ----------- Project entities and centroids into the CA plane -----------
    eplus, eminus = decomposition.eplus, decomposition.eminus
    half = decomposition.theta / 2.0
    eig_plus  = decomposition.lambda_plus
    eig_minus = decomposition.lambda_minus

    points = []
    for ent in entities:
        v = np.asarray(ent.vector, dtype=np.float64) - centering_vector
        points.append((ent.name, ent.etype, float(v @ eplus), float(v @ eminus), ent.status))

    centroid_xy: dict[str, np.ndarray] = {}
    if centroids:
        for name, vec in centroids.items():
            if vec is None:
                continue
            cv = np.asarray(vec, dtype=np.float64) - centering_vector
            centroid_xy[name] = np.array([float(cv @ eplus), float(cv @ eminus)])

    # ----------- Layout (per-panel scales, equal aspect) -----------
    all_eplus  = [abs(p[2]) for p in points] or [1.0]
    all_eminus = [abs(p[3]) for p in points] or [1.0]
    data_extent = max(all_eminus) * 1.10
    # Pre-shear xlim: tight on the left (data side) but with a small extra
    # margin so labels protruding to the screen-left of points have room
    # to render; generous on the right (legend side) per
    # ``pre_legend_pad_factor``.
    xlim_pre_left  = +data_extent + 2.0
    xlim_pre_right = -data_extent * pre_legend_pad_factor
    ylim_pre = max(all_eplus) * 1.05
    post_eplus_max  = max(all_eplus)  * eig_plus
    post_eminus_max = max(all_eminus) * eig_minus
    square_lim = max(post_eplus_max, post_eminus_max) * 1.10

    pre_h_in   = pre_panel_height_in
    pre_xrange = abs(xlim_pre_left - xlim_pre_right)
    pre_w_in   = pre_h_in * (pre_xrange / 2.0 / ylim_pre)
    post_h_in  = pre_h_in
    post_w_in  = post_h_in
    gap_in     = 1.2
    left_pad   = 0.8
    right_pad  = 1.6
    top_pad    = 1.5
    bottom_pad = 0.8
    fig_height = top_pad + max(pre_h_in, post_h_in) + bottom_pad
    fig_width  = left_pad + pre_w_in + gap_in + post_w_in + right_pad

    fig = plt.figure(figsize=(fig_width, fig_height))

    def _rect(left_in, bottom_in, w_in, h_in):
        return [left_in / fig_width, bottom_in / fig_height,
                w_in / fig_width, h_in / fig_height]

    ax_pre  = fig.add_axes(_rect(left_pad, bottom_pad, pre_w_in, pre_h_in))
    ax_post = fig.add_axes(_rect(left_pad + pre_w_in + gap_in,
                                  bottom_pad + (pre_h_in - post_h_in) / 2,
                                  post_w_in, post_h_in))

    # ----------- Render each panel -----------
    label_priority = dict(SEMANTIC_BASE_BONUS)
    if extra_label_priority:
        label_priority.update(extra_label_priority)

    def _rotate(eplus_v, eminus_v):
        # transpose: e_+ becomes y, e_- becomes x; combined with
        # ax.invert_xaxis() this gives a 90 deg CCW rotation of the
        # original (e_+ horizontal, e_- vertical) plot.
        return eminus_v, eplus_v

    def _render_panel(ax, post_shear: bool):
        sx_eplus  = eig_plus  if post_shear else 1.0
        sy_eminus = eig_minus if post_shear else 1.0

        # Background scatter (low priority statuses drawn first / underneath)
        for status, alpha, size in (("neither", 0.15, 6),
                                    ("nogoal",  0.55, 12),
                                    ("goal",    0.55, 12)):
            xs, ys = [], []
            for p in points:
                if p[4] != status: continue
                x, y = _rotate(sx_eplus * p[2], sy_eminus * p[3])
                xs.append(x); ys.append(y)
            ax.scatter(xs, ys, c=_STATUS_COLOURS[status], s=size, alpha=alpha,
                       edgecolors="none", label=status, zorder=2)

        # Centroid stars
        for label, c in centroid_xy.items():
            x_s, y_s = _rotate(sx_eplus * c[0], sy_eminus * c[1])
            ax.scatter([x_s], [y_s], c=_CENTROID_COLOURS.get(label, "#888"),
                       s=200, marker="*", edgecolors="black", linewidth=0.6,
                       zorder=6, label=label)
        # Origin marker (yellow star) -- always drawn at (0, 0) regardless
        # of whether a centroid was supplied at the same point.  This is
        # the *visual* origin of the plot; if the caller passed a
        # centroid named "origin (pool median)", the two stars coincide.
        if "origin (pool median)" not in centroid_xy:
            ax.scatter([0], [0], c=_CENTROID_COLOURS["origin (pool median)"],
                       s=200, marker="*", edgecolors="black", linewidth=0.6,
                       zorder=6, label="origin")

        # a_k / b_k direction lines through origin
        a_pre = np.array([np.cos(half),  np.sin(half)])
        b_pre = np.array([np.cos(half), -np.sin(half)])
        a_xy  = np.array([sx_eplus * a_pre[0], sy_eminus * a_pre[1]])
        b_xy  = np.array([sx_eplus * b_pre[0], sy_eminus * b_pre[1]])
        a_unit = a_xy / np.linalg.norm(a_xy)
        b_unit = b_xy / np.linalg.norm(b_xy)
        a_screen = np.array(_rotate(*a_unit))
        b_screen = np.array(_rotate(*b_unit))

        if post_shear:
            x_lo, x_hi = -square_lim, square_lim
            Lx_for_rays = square_lim
            Ly = square_lim
        else:
            x_lo, x_hi = xlim_pre_right, xlim_pre_left
            Lx_for_rays = max(abs(x_lo), abs(x_hi))
            Ly = ylim_pre
        L_ray = max(Lx_for_rays, Ly)
        sub = "{" + str(decomposition.ca_index) + "}"
        ax.plot([-L_ray * a_screen[0], L_ray * a_screen[0]],
                [-L_ray * a_screen[1], L_ray * a_screen[1]],
                color="#d62728", lw=1.3, alpha=0.7, zorder=3,
                label=rf"$a_{sub}$ (goals)")
        ax.plot([-L_ray * b_screen[0], L_ray * b_screen[0]],
                [-L_ray * b_screen[1], L_ray * b_screen[1]],
                color="#1f77b4", lw=1.3, alpha=0.7, zorder=3,
                label=rf"$b_{sub}$ (non-goals)")
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.axvline(0, color="black", lw=0.6, alpha=0.4)

        # Axis-pole text labels
        ax.text(0, Ly * 0.95, r"$e_+$", fontsize=11, ha="right", va="top",
                color="#222")
        ax.text(max(abs(x_lo), abs(x_hi)) * 0.95, 0,
                r"$e_-$", fontsize=11, ha="right", va="bottom", color="#222")

        # Axis limits + invert + equal aspect (set BEFORE label collision math)
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(-Ly, Ly)
        ax.invert_xaxis()
        ax.set_aspect("equal")

        # Collision-aware label placement.  Goal/no-goal members get a +500
        # priority bump so they're tried first; "neither" entities fill in
        # only where space remains.
        candidates = []
        for name, et, eplus_p, eminus_p, st in points:
            prio = label_priority.get(name, 0)
            if st in ("goal", "nogoal"):
                prio += 500
            x_s, y_s = _rotate(sx_eplus * eplus_p, sy_eminus * eminus_p)
            candidates.append((et, name, x_s, y_s, prio))
        collision_label(ax, candidates, fontsize=7)

        # Title -- describe angles in original (e_+, e_-) frame
        angle_a = np.degrees(np.arctan2(a_unit[1], a_unit[0]))
        angle_b = np.degrees(np.arctan2(b_unit[1], b_unit[0]))
        if post_shear:
            t = (f"Post-shear  ($a_{decomposition.ca_index}$/$b_{decomposition.ca_index}$ "
                 f"at $\\pm{abs(angle_a):.1f}^\\circ$ to $e_+$, "
                 f"${abs(angle_a) + abs(angle_b):.1f}^\\circ$ apart)")
        else:
            t = f"Pre-shear\n({abs(angle_a) + abs(angle_b):.1f}$^\\circ$ apart)"
        ax.set_title(t, fontsize=11)
        ax.set_xlabel(r"projection onto $e_-$  (positive $\leftarrow$)" +
                      ("  $\\times \\sqrt{\\cot(\\theta/2)}$" if post_shear else ""))
        ax.set_ylabel(r"projection onto $e_+$" +
                      ("  $\\times \\sqrt{\\tan(\\theta/2)}$" if post_shear else ""))
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85,
                  markerscale=0.7, handlelength=1.4)

    _render_panel(ax_pre, post_shear=False)
    _render_panel(ax_post, post_shear=True)

    suptitle_with_specs(fig, title=title, specs=spec_lines, line_height=0.022)
    return fig


# Back-compat alias for the rendering function as well
plot_ca1_plane_pre_post_shear = plot_ca_plane_pre_post_shear


__all__ = [
    "CADecomposition",
    "compute_ca_decomposition",
    "CAPlaneEntity",
    "SEMANTIC_BASE_BONUS",
    "collision_label",
    "plot_ca_plane_pre_post_shear",
    # Backward-compat aliases (CA1-only flavour, identical to the general API)
    "CA1Decomposition",
    "compute_ca1_decomposition",
    "CA1PlaneEntity",
    "plot_ca1_plane_pre_post_shear",
]
