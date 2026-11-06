"""Per-(slot, layer, sign) response curves AVERAGED across multiple axes.

Companion to ``steering_response_curves.py`` (per-axis plots).  This
script aggregates several completed experiments' per-cell curves and
plots, for each (slot, layer, sign), the MEAN effect curve across all
listed axes.

Alignment convention: each axis's per-(cell, sign) curve runs from
x=0 (last coherent strength = highest swept |s| still under
skip_threshold) leftward to higher x (weaker strengths, further from
the coherence cliff).  Axes differ in cliff strength AND in how far
the downward sweep ran (esp. after the bias-cancelled corrected stop
cutoff -- post_judge.py::apply_corrected_stop_cutoff -- moves weak-
strength records below the corrected predicate's anchor to
``_excluded_records.jsonl``).

Averaging rules:

* **Pad short curves with 0 on the high-x (weak-strength) end.**  The
  rationale: if axis A's corrected predicate stopped the sweep at
  x=5 because the bias-cancelled signal had dropped below the noise
  threshold, the unobserved x=6..N values are best estimated as 0
  (no signal, no movement).  This matches the empirical observation
  that strengths past the corrected anchor are at the noise floor.

* **Interpolate interior gaps linearly.**  An axis-cell can have a
  missing strength mid-curve (e.g. ``novelist_honest_v1/s6_l49_+1``
  is missing the strength ``|s|=1.681`` between 1.414 and 1.999 --
  presumably from a partial-flush during the original killed-and-
  resumed sweep on the two novelist cells that resumed with prior
  data).  Linear interpolation in the rank dimension fills the gap.

* **Per-axis curves are flipped before averaging when sign=-1.**  We
  reuse the per-axis-plot's normalisation (UP = toward steered pole)
  so positive-y means "the steering worked" universally.

Usage::

    uv run results_analysis/steering_response_curves_averaged.py \\
        --experiments outputs/qwen-3-32b/steering/anthropologist_helpful_v1 \\
                      outputs/qwen-3-32b/steering/prodigy_harmless_v1 \\
                      outputs/qwen-3-32b/steering/novelist_honest_v1 \\
                      outputs/qwen-3-32b/steering/publisher_truthful_v1 \\
                      outputs/qwen-3-32b/steering/publisher_guileless_v1 \\
        --output averaged_axes_response_curves.png

Each axis is named by its experiment directory's leaf (e.g.
``anthropologist_helpful_v1``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse aggregate_cell_sign / gather_cells from the per-axis script
# rather than duplicate the parsing logic.  We import via importlib so
# the per-axis file isn't required to be a package.
_per_axis_spec = importlib.util.spec_from_file_location(
    "_steering_response_curves",
    REPO_ROOT / "results_analysis" / "steering_response_curves.py",
)
assert _per_axis_spec is not None and _per_axis_spec.loader is not None
_per_axis = importlib.util.module_from_spec(_per_axis_spec)
sys.modules[_per_axis_spec.name] = _per_axis
_per_axis_spec.loader.exec_module(_per_axis)

from assistant_axis.plot_metadata import png_metadata  # noqa: E402

logger = logging.getLogger("steering_response_curves_averaged")


# ---------------------------------------------------------------------------
# Curve extraction + cross-axis alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignedCurve:
    """One axis's contribution to a (slot, layer, sign) averaged curve.

    ``x`` is "steps remaining to incoherence", 0 at the right (last
    coherent strength), increasing leftward (weaker strengths).
    ``y`` is the per-x mean effect score, oriented so positive = "toward
    the steered pole" (i.e. the sign=-1 cells have been flipped).
    Gaps in the source aggs (interior None values) are linearly
    interpolated; this struct holds the post-fill values.
    """
    axis_name: str
    eff: np.ndarray            # shape (n_strengths,), no NaNs
    coh: np.ndarray            # mean coherence per strength (raw 0..3)
    rp: np.ndarray             # 3 - mean_rp per strength (0..3, up = bad)
    n_strengths: int           # = len(eff)


def _signed_eff_for_plot(agg, sign: int) -> Optional[float]:
    """Return the per-strength mean eff oriented so positive = 'toward
    the steered pole'.  Matches the per-axis plot's normalisation:
    sign=+1 cells keep the rubric's sign; sign=-1 cells are flipped.
    """
    v = agg.mean_eff_all
    if v is None:
        return None
    return v if sign == +1 else -v


def _interpolate_interior_gaps(values: List[Optional[float]]) -> np.ndarray:
    """Linearly interpolate over interior None values in ``values``.

    Leading/trailing Nones are NOT extrapolated (they should be padded
    separately by the caller).  Returns an array with shape
    ``(len(values),)`` where every entry is finite (NaN only at the
    leading/trailing ends if those were None).
    """
    arr = np.array(
        [np.nan if v is None else float(v) for v in values],
        dtype=float,
    )
    n = len(arr)
    finite_mask = ~np.isnan(arr)
    if finite_mask.sum() == 0:
        return arr  # all NaN, nothing to interpolate
    finite_idx = np.where(finite_mask)[0]
    first, last = finite_idx[0], finite_idx[-1]
    # Interior NaNs: linearly interp between bracketing finite points
    for i in range(first, last + 1):
        if not np.isnan(arr[i]):
            continue
        # Find nearest finite to the left and right
        lo = i - 1
        while np.isnan(arr[lo]):
            lo -= 1
        hi = i + 1
        while np.isnan(arr[hi]):
            hi += 1
        # Linear interp on rank-index (not strength) -- the x-axis is
        # rank-from-cliff so this is the natural metric.
        t = (i - lo) / (hi - lo)
        arr[i] = arr[lo] * (1 - t) + arr[hi] * t
    return arr


def build_aligned_curve(
    cell_sign,
    axis_name: str,
    target_length: int,
) -> AlignedCurve:
    """Convert a ``CellSign`` (from steering_response_curves.aggregate_cell_sign)
    into an ``AlignedCurve`` of length ``target_length``.

    Output array convention: ASCENDING |s| order, matching
    ``cell_sign.aggs`` and the per-axis plot's ``cs.aggs`` ordering.
    So index 0 = weakest |s| (highest x = leftmost on the plotted
    inverted x-axis), index target_length-1 = cliff (x=0, rightmost
    on the plot).  Cells with fewer retained strengths than
    ``target_length`` (because their corrected stop predicate fired
    closer to the cliff) get PADDED ON THE LEFT with 0: those padded
    entries represent weaker strengths that the corrected predicate
    would have excluded, and "no observed signal" maps to 0.

    Interior gaps (rare; e.g. a partial-flush during a resumed sweep
    leaves a missing strength mid-curve) are linearly interpolated
    in rank space before padding.
    """
    aggs = cell_sign.aggs  # sorted ASCENDING magnitude
    # eff oriented so positive = toward steered pole.
    eff_raw = [_signed_eff_for_plot(a, cell_sign.sign) for a in aggs]
    coh_raw = [a.mean_coh_all for a in aggs]
    rp_raw = [
        (3.0 - a.mean_rp_all) if a.mean_rp_all is not None else None
        for a in aggs
    ]

    eff_interp = _interpolate_interior_gaps(eff_raw)
    coh_interp = _interpolate_interior_gaps(coh_raw)
    rp_interp = _interpolate_interior_gaps(rp_raw)

    # Pad to target_length on the LEFT (weak-strength end).  Effect
    # padding follows the user's spec ("treat data missing because
    # it's off the low end as 0"); coh and rp padding stays NaN so
    # the diagnostic bottom row reflects "no data" rather than
    # spuriously implying coherent / in-persona at unobserved
    # strengths.
    n = len(eff_interp)
    if n < target_length:
        pad_n = target_length - n
        pad_eff = np.zeros(pad_n, dtype=float)
        pad_coh = np.full(pad_n, np.nan, dtype=float)
        pad_rp = np.full(pad_n, np.nan, dtype=float)
        eff_full = np.concatenate([pad_eff, eff_interp])
        coh_full = np.concatenate([pad_coh, coh_interp])
        rp_full = np.concatenate([pad_rp, rp_interp])
    else:
        # If n > target_length (shouldn't happen since we set
        # target_length = max), keep the cliff-aligned right tail.
        eff_full = eff_interp[n - target_length:]
        coh_full = coh_interp[n - target_length:]
        rp_full = rp_interp[n - target_length:]

    return AlignedCurve(
        axis_name=axis_name,
        eff=eff_full, coh=coh_full, rp=rp_full,
        n_strengths=n,
    )


# ---------------------------------------------------------------------------
# Cross-axis averaging
# ---------------------------------------------------------------------------

@dataclass
class AveragedCellSign:
    """Mean eff/coh/rp curve across N axes for one (slot, layer, sign)."""
    slot: int
    layer: int
    sign: int
    target_length: int
    eff_mean: np.ndarray       # shape (target_length,) -- mean across axes
    eff_std: np.ndarray        # cross-axis stdev (informational)
    coh_mean: np.ndarray       # nanmean across axes (NaN where every axis was NaN)
    rp_mean: np.ndarray        # nanmean across axes
    n_axes_at_x: np.ndarray    # how many axes had real (non-padded) eff at each x
    contributing_axis_names: List[str]


def average_across_axes(
    cells_per_axis: Dict[str, List[Any]],   # axis_name -> List[CellSign]
    slot: int, layer: int, sign: int,
) -> Optional[AveragedCellSign]:
    """Build one AveragedCellSign by combining all axes that have data
    for (slot, layer, sign).

    Returns None if no axis has data for this combination.
    """
    matching: List[Tuple[str, Any]] = []
    for axis_name, cells in cells_per_axis.items():
        for cs in cells:
            if cs.slot == slot and cs.layer == layer and cs.sign == sign:
                if cs.aggs:  # skip empty-curve cells (no retained data)
                    matching.append((axis_name, cs))
                break  # at most one (cs.slot, cs.layer, cs.sign) per axis
    if not matching:
        return None

    target_length = max(len(cs.aggs) for _, cs in matching)
    aligned = [
        build_aligned_curve(cs, axis_name, target_length)
        for axis_name, cs in matching
    ]

    # Stack into (n_axes, target_length) and compute means
    eff_mat = np.stack([a.eff for a in aligned], axis=0)
    coh_mat = np.stack([a.coh for a in aligned], axis=0)
    rp_mat = np.stack([a.rp for a in aligned], axis=0)
    eff_mean = eff_mat.mean(axis=0)
    eff_std = eff_mat.std(axis=0)
    # coh / rp: padded values are NaN; nanmean handles gracefully
    with np.errstate(invalid="ignore"):
        coh_mean = np.nanmean(coh_mat, axis=0)
        rp_mean = np.nanmean(rp_mat, axis=0)
    # n_axes_at_x: how many axes had REAL (= non-padded) eff at each
    # entry (index in ascending-|s| order, so right tail = near
    # cliff).  Real data lives at the right end of the array; padded
    # zeros on the left.  Used to label "thin coverage" regions where
    # only a handful of axes contribute (the rest are padded 0s).
    n_axes_at_x = np.zeros(target_length, dtype=int)
    for a in aligned:
        n_axes_at_x[target_length - a.n_strengths:] += 1

    return AveragedCellSign(
        slot=slot, layer=layer, sign=sign,
        target_length=target_length,
        eff_mean=eff_mean, eff_std=eff_std,
        coh_mean=coh_mean, rp_mean=rp_mean,
        n_axes_at_x=n_axes_at_x,
        contributing_axis_names=[a.axis_name for a in aligned],
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _x_steps_from_cliff(n_strengths: int) -> np.ndarray:
    """Same as the per-axis script: x=0 at the right (highest swept |s|
    still coherent), increasing leftward."""
    return np.arange(n_strengths - 1, -1, -1)


def plot_averaged_curves(
    averaged_by_signature: Dict[Tuple[int, int, int], AveragedCellSign],
    *,
    output_path: Path,
    title: str,
    n_axes_total: int,
) -> None:
    """Render the averaged-across-axes figure: 2x2 layout matching
    steering_response_curves.plot_response_curves.

    Top row: averaged effect curves, one line per (slot, layer), with
    the cross-axis stdev shown as a translucent band.  Bottom row:
    averaged coh (solid) + 3-mean_rp (dotted), per cell.
    """
    unique_cells = sorted({(slot, layer) for (slot, layer, _) in averaged_by_signature.keys()})
    cmap = plt.get_cmap("tab10")
    color_for_cell = {(s, l): cmap(i % 10) for i, (s, l) in enumerate(unique_cells)}

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 11),
        sharey="row", sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )
    top_axes = axes[0]
    bot_axes = axes[1]

    for ax, sign in zip(top_axes, (+1, -1)):
        for (slot, layer) in unique_cells:
            key = (slot, layer, sign)
            if key not in averaged_by_signature:
                continue
            avg = averaged_by_signature[key]
            x = _x_steps_from_cliff(avg.target_length)
            y = avg.eff_mean
            std = avg.eff_std
            color = color_for_cell[(slot, layer)]
            base_label = f"s{slot}_l{layer}"
            ax.plot(
                x, y, color=color, linewidth=1.7, label=base_label,
                marker="o", markersize=4, markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=1.0,
            )
            ax.fill_between(
                x, y - std, y + std,
                color=color, alpha=0.12, linewidth=0,
            )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
        sign_label = "sign = +1  (toward pos pole)" if sign == +1 else "sign = -1  (toward neg pole)"
        ax.set_title(
            f"{sign_label}    "
            f"\u2191 = moved toward steered pole, \u2193 = opposite"
        )
        ax.grid(True, alpha=0.3)

    for ax, sign in zip(bot_axes, (+1, -1)):
        for (slot, layer) in unique_cells:
            key = (slot, layer, sign)
            if key not in averaged_by_signature:
                continue
            avg = averaged_by_signature[key]
            x = _x_steps_from_cliff(avg.target_length)
            color = color_for_cell[(slot, layer)]
            ax.plot(x, avg.coh_mean, color=color, linestyle="-", linewidth=1.4)
            ax.plot(x, avg.rp_mean, color=color, linestyle=":", linewidth=1.4)
        ax.axhline(1.0, color="gray", linewidth=0.7, linestyle=":",
                   alpha=0.6, label="skip_threshold (1.0)")
        ax.axhline(1.5, color="gray", linewidth=0.7, linestyle="--",
                   alpha=0.6, label="coh_stop_threshold (1.5)")
        ax.set_ylim(-0.1, 3.2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Steps remaining to incoherence (0 = last coherent)")

    # x=0 (cliff) on the RIGHT; increasing x to the LEFT.
    for ax in top_axes:
        ax.invert_xaxis()
    for ax in bot_axes:
        ax.xaxis.set_major_locator(MultipleLocator(1))

    top_axes[0].set_ylabel(
        "Mean effect score    "
        "(+: toward the steered pole    "
        "\u2212: away from the steered pole)"
    )
    bot_axes[0].set_ylabel(
        "Coherence / inverted-RP    (0 = good, 3 = bad)"
    )

    handles_cells, labels_cells = top_axes[0].get_legend_handles_labels()
    top_axes[0].legend(
        handles_cells, labels_cells,
        title=f"cell  (slot, layer)\nshaded band = ±cross-axis stdev",
        loc="upper left",
        fontsize=8,
    )
    metric_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-",
                   linewidth=1.4, label="mean coherence"),
        plt.Line2D([0], [0], color="black", linestyle=":",
                   linewidth=1.4, label="3 - mean RP"),
        plt.Line2D([0], [0], color="gray", linestyle=":",
                   linewidth=0.7, label="skip_threshold (1.0)"),
        plt.Line2D([0], [0], color="gray", linestyle="--",
                   linewidth=0.7, label="coh_stop_threshold (1.5)"),
    ]
    bot_axes[0].legend(
        handles=metric_handles,
        title="bottom-row metric",
        loc="upper left",
        fontsize=8,
    )

    fig.suptitle(f"{title}\nMean across {n_axes_total} axes "
                 f"(short curves padded with 0 on weak-strength end; "
                 f"interior gaps interpolated)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=140, bbox_inches="tight",
        metadata=png_metadata(title=title.splitlines()[0]),
    )
    logger.info(f"wrote {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--experiments", nargs="+", type=Path, required=True,
        help="Experiment directories to average across (each containing "
             "s<slot>_l<layer>_<sign> cell subdirs).",
    )
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--title", type=str, default="Averaged response curves")
    p.add_argument(
        "--coh-filter-threshold", type=float,
        default=_per_axis.COH_FILTER_THRESHOLD_DEFAULT,
        help=f"Display-time mean_coh cutoff (default "
             f"{_per_axis.COH_FILTER_THRESHOLD_DEFAULT}); see the "
             f"per-axis script for full semantics.  Independent of "
             f"the runner's skip_threshold.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cells_per_axis: Dict[str, List[Any]] = {}
    for exp_dir in args.experiments:
        if not exp_dir.exists():
            logger.error(f"missing: {exp_dir}")
            return 2
        cells = _per_axis.gather_cells(
            exp_dir, coh_filter_threshold=args.coh_filter_threshold,
        )
        if not cells:
            logger.warning(f"no cell subdirs found under {exp_dir}; skipping")
            continue
        cells_per_axis[exp_dir.name] = cells
        logger.info(f"loaded {len(cells)} (cell, sign) groups from {exp_dir.name}")

    if not cells_per_axis:
        logger.error("no experiments produced cell data")
        return 1

    # Unique (slot, layer, sign) across all axes
    signatures = set()
    for cells in cells_per_axis.values():
        for cs in cells:
            signatures.add((cs.slot, cs.layer, cs.sign))

    averaged: Dict[Tuple[int, int, int], AveragedCellSign] = {}
    for sig in sorted(signatures):
        result = average_across_axes(cells_per_axis, *sig)
        if result is not None:
            averaged[sig] = result

    logger.info(
        f"averaging across {len(cells_per_axis)} axes; produced "
        f"{len(averaged)} (slot, layer, sign) averaged curves"
    )
    plot_averaged_curves(
        averaged, output_path=args.output,
        title=args.title, n_axes_total=len(cells_per_axis),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
