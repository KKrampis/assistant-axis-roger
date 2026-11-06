"""Per-cell response curves vs steps-to-incoherence.

Reads a steering experiment directory (one (role, axis) pair, one or
more bidirectional-scan cells) and plots one curve per
(cell, sign, filter) combination of:

    Y axis : mean effect score across the 14 questions at one strength
    X axis : "steps remaining to incoherence" -- 0 at the right edge
             (the cliff itself); positive integers leftward count how
             many ranked-by-magnitude strengths are between the cell's
             current strength and its incoherence cliff.

Aligning all cells at x=0 lets you compare dose-response curves whose
absolute strength scales differ (different (slot, layer) cells have
different cliff strengths).

Four filter variants per (cell, sign):

    a) all-responses  : raw mean of effect scores at each strength
    b) coh-0 only     : mean over responses with coherence score == 0
                        (no judged incoherence at all)
    c) rp-3 only      : mean over responses with persona score == 3
                        (strongly in-persona)
    d) coh-0 & rp-3   : mean over responses with both conditions

Comparing (a) vs the filtered variants reveals how much the average
effect at each strength is being dragged around by responses that
were either drifting from persona or borderline-incoherent.

Two side-by-side subplots:
    Left : sign = +1 (steers toward the negative pole; role_to)
    Right: sign = -1 (steers toward the positive pole; role_from)

Within each subplot:
    color    : cell = (slot, layer)
    linestyle: filter variant (a solid / b dashed / c dotted / d dash-dot)

Usage:
    python results_analysis/steering_response_curves.py \\
        outputs/qwen-3-32b/steering/anthropologist_helpful_v1 \\
        --output anthropologist_helpful_v1_response_curves.png
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from assistant_axis.plot_metadata import png_metadata

logger = logging.getLogger("steering_response_curves")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrengthAgg:
    """One strength's worth of mean effect under each of the four
    response-filter regimes.

    Each entry is ``None`` if that filter regime had zero qualifying
    responses (so the resulting line has a gap at this strength).
    """

    strength: float       # absolute magnitude (positive)
    mean_eff_all: Optional[float]
    mean_eff_coh0: Optional[float]
    mean_eff_rp3: Optional[float]
    mean_eff_coh0_rp3: Optional[float]
    # Unfiltered means of the gating dimensions, for the bottom-row
    # coh/rp panels.  Both range 0..3 on the rubrics' integer scales;
    # ``mean_rp_all`` is the raw persona-fit score (3 = strongly in
    # persona, 0 = AI self-id / off-character).  The bottom-row plot
    # shows ``3 - mean_rp_all`` so "up = bad" matches coherence's
    # orientation.  May be None if every record at this strength is
    # missing the relevant score (e.g. coh judging crashed, or
    # persona judging was skipped because of strength_mean_coh).
    mean_coh_all: Optional[float]
    mean_rp_all: Optional[float]
    # Bookkeeping for the legend / sanity checks.
    n_total: int          # number of records at this strength
    n_with_eff: int       # number with a non-null effect score
    n_coh0: int           # number with coh == 0


@dataclass
class CellSign:
    """One (slot, layer, sign) sweep's worth of strength-aggregates,
    ranked by magnitude with the largest at index N (= incoherence cliff).
    """

    slot: int
    layer: int
    sign: int            # +1 or -1
    aggs: List[StrengthAgg]  # sorted ascending by magnitude
    # Trailing strengths that were trimmed from ``aggs`` because rp+effect
    # judging was skipped (their mean_coh exceeded the runner's
    # skip_threshold).  Coherence WAS judged at these strengths (coh is
    # the gating dimension), so the bottom-row coh panel can plot them
    # to the right of x=0 (negative x) to show how mean_coh ramps PAST
    # the eff-judged "cliff" before the sweep's coh-stop fired.
    # Sorted ascending by magnitude so ``extra_aggs[0]`` is the
    # smallest-magnitude trimmed strength (= "first step past the
    # cliff", x = -1).
    extra_aggs: List[StrengthAgg] = None  # type: ignore[assignment]
    # Whether the largest swept strength hit the incoherence ceiling
    # (vs. running into max_strength without crossing the coh stop).
    # Used to label the line's right endpoint in the legend / tooltips.
    up_blocked_reason: Optional[str] = None

    def __post_init__(self) -> None:
        # Default to empty list (dataclass field defaults that are
        # mutable need this idiom -- we can't use ``field(default_factory=list)``
        # on a non-frozen dataclass without bumping the import surface).
        if self.extra_aggs is None:
            self.extra_aggs = []


# ---------------------------------------------------------------------------
# Disk IO + score extraction (mirrors tools/sheet_layout.py conventions)
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _coh_score(rec: Dict[str, Any]) -> Optional[int]:
    j = rec.get("judges") or {}
    c = j.get("coherence")
    if not c:
        return None
    s = c.get("score")
    return None if s is None else int(s)


def _rp_score(rec: Dict[str, Any]) -> Optional[int]:
    j = rec.get("judges") or {}
    p = j.get("persona") or {}
    if p.get("skipped_due_to_strength_mean_coh"):
        return None
    s = p.get("score")
    return None if s is None else int(s)


def _eff_score(rec: Dict[str, Any]) -> Optional[float]:
    j = rec.get("judges") or {}
    e = j.get("effect") or {}
    if e.get("skipped_due_to_strength_mean_coh"):
        return None
    s = e.get("combined")
    return None if s is None else float(s)


def _safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    f = [x for x in xs if x is not None]
    return mean(f) if f else None


# ---------------------------------------------------------------------------
# Per-cell aggregation
# ---------------------------------------------------------------------------

def _load_swap_averaged(cell_dir: Path) -> Dict[Tuple[int, float], float]:
    """Build a lookup ``(question_idx, rounded_strength) -> averaged_eff``
    from this cell's ``records_effect_swap.jsonl`` (if it exists).

    The side file is produced by ``tools/test_effect_order_bias.py``
    and carries the bias-cancelled estimator
    ``averaged_eff = (orig_combined - swap_combined) / 2`` per record.
    When present, ``aggregate_cell_sign`` substitutes this for the
    record's raw ``judges.effect.combined`` -- the plot then shows
    bias-corrected signal instead of the rubric's raw output.

    Returns an empty dict if the file doesn't exist, so callers fall
    back to the raw ``effect.combined`` gracefully.
    """
    path = cell_dir / "records_effect_swap.jsonl"
    out: Dict[Tuple[int, float], float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            q = int(r["question_idx"])
            s = round(float(r["strength"]), 6)
            avg = r.get("averaged_eff")
            if avg is None:
                continue
            out[(q, s)] = float(avg)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
    return out


#: Default display-time cutoff for the per-cell plot's x=0 anchor.
#: Independent of the runner's ``skip_threshold`` (which is now 1.5):
#: this filter is pure data analysis and lets you tighten the
#: "coherent enough to show at the cliff" definition without rerunning
#: the sweep.  A strength is dropped from the top-row eff panels if
#: ``mean_coh_all >= COH_FILTER_THRESHOLD_DEFAULT``; it remains on the
#: bottom-row coh/rp diagnostic panels (via ``extra_aggs``).
COH_FILTER_THRESHOLD_DEFAULT: float = 1.0


def aggregate_cell_sign(
    cell_dir: Path,
    *,
    coh_filter_threshold: float = COH_FILTER_THRESHOLD_DEFAULT,
) -> CellSign:
    """Read one ``s<slot>_l<layer>_<+1|-1>`` directory and group its
    records by strength, computing each of the four filter means.

    If the cell has a ``records_effect_swap.jsonl`` sibling file
    (produced by ``tools/test_effect_order_bias.py``), the
    bias-cancelled ``averaged_eff = (orig - swap) / 2`` substitutes
    for the raw ``judges.effect.combined``.  Records at strengths the
    swap descent stopped above (because the cancelled signal had
    dropped below the noise floor) keep their original effect score
    -- those rows lie outside the bias-corrected analytical zone and
    the plot can still show them.
    """
    name = cell_dir.name  # "s3_l25_+1"
    # Parse slot/layer/sign back from dir name -- robust to any
    # signed-integer slot or layer.
    parts = name.split("_")
    slot = int(parts[0][1:])           # "s3" -> 3
    layer = int(parts[1][1:])          # "l25" -> 25
    sign_str = parts[2]                # "+1" or "-1"
    sign = int(sign_str)

    records = _read_jsonl(cell_dir / "records.jsonl")
    swap_avg_lookup = _load_swap_averaged(cell_dir)
    # Group by exact strength value (records use floats; rounding errors
    # are tolerable because the runner reuses the same float each time
    # it writes a strength).
    by_strength: Dict[float, List[Dict[str, Any]]] = {}
    for r in records:
        try:
            s = float(r.get("strength", 0.0))
        except (TypeError, ValueError):
            continue
        if s == 0.0:
            continue  # baseline rows -- excluded from steering curves
        by_strength.setdefault(s, []).append(r)

    # Strengths that have at least one swap-averaged eff in the side
    # file -- if a swap.jsonl exists at all, restrict the plotted
    # strengths to these so the curve only shows bias-corrected data.
    # When no swap.jsonl exists (legacy experiments or experiments
    # we haven't filled in yet) fall back to plotting every strength
    # with the raw rubric effect.
    strengths_with_swap: Optional[set] = None
    if swap_avg_lookup:
        strengths_with_swap = {k[1] for k in swap_avg_lookup}

    aggs: List[StrengthAgg] = []
    for s in sorted(by_strength.keys()):
        s_key = round(s, 6)
        recs = by_strength[s]
        # Two distinct reasons a strength might be absent from the
        # swap descent's coverage:
        #   (A) Bias-floor end: the swap descent stopped at some
        #       cutoff and didn't judge weaker strengths.  Under the
        #       corrected pipeline these wouldn't have been kept, so
        #       drop them entirely.
        #   (B) Past-cliff end: the runner skipped rp+effect judging
        #       at this strength because mean_coh exceeded
        #       skip_threshold, so the swap descent had nothing to
        #       correct.  These rows STILL carry useful coherence
        #       data (coh is always judged) and we want them visible
        #       in the bottom-row coh ramp -- they get trimmed into
        #       ``extra_aggs`` below.
        # The distinguishing signal: case (A) rows have eff scores
        # populated (just bias-contaminated); case (B) rows have
        # all-None eff.
        any_eff = any(_eff_score(r) is not None for r in recs)
        if (
            strengths_with_swap is not None
            and s_key not in strengths_with_swap
            and any_eff
        ):
            continue  # case (A): bias-contaminated low end
        coh = [_coh_score(r) for r in recs]
        rp = [_rp_score(r) for r in recs]
        # Use the swap-averaged (bias-cancelled) eff if available for
        # this (question_idx, strength); else fall back to the raw
        # rubric effect score.  Strength rounded to 6 dp to absorb
        # FP serialisation drift between the live runner and side file.
        eff: List[Optional[float]] = []
        for r in recs:
            try:
                q = int(r["question_idx"])
            except (KeyError, ValueError, TypeError):
                eff.append(None)
                continue
            cached = swap_avg_lookup.get((q, s_key))
            if cached is not None:
                eff.append(cached)
            else:
                eff.append(_eff_score(r))

        # Filter regimes -- each returns the eff value or None.
        def filter_eff(predicate: Callable[[int, int], bool]) -> List[Optional[float]]:
            out: List[Optional[float]] = []
            for c, p, e in zip(coh, rp, eff):
                # Drop rows where the filter predicate is unknown
                # (skipped / null score on the gating dimension).
                if c is None or p is None or e is None:
                    out.append(None)
                    continue
                out.append(e if predicate(c, p) else None)
            return out

        agg = StrengthAgg(
            strength=s,
            mean_eff_all=_safe_mean(eff),
            mean_eff_coh0=_safe_mean(filter_eff(lambda c, p: c == 0)),
            mean_eff_rp3=_safe_mean(filter_eff(lambda c, p: p == 3)),
            mean_eff_coh0_rp3=_safe_mean(
                filter_eff(lambda c, p: c == 0 and p == 3)
            ),
            # Unfiltered means of the gating dimensions for the bottom
            # row.  Cast int->float via _safe_mean so the mean is a
            # well-defined float regardless of the underlying schema.
            mean_coh_all=_safe_mean([float(c) if c is not None else None for c in coh]),
            mean_rp_all=_safe_mean([float(p) if p is not None else None for p in rp]),
            n_total=len(recs),
            n_with_eff=sum(1 for e in eff if e is not None),
            n_coh0=sum(1 for c in coh if c == 0),
        )
        aggs.append(agg)

    # Trim trailing strengths for the per-cell display.  Two reasons
    # a strength gets trimmed off the top-row effect panels:
    #
    # 1. The runner skipped rp+effect judging at that strength
    #    (``mean_eff_all is None``).  Pre-2026-05-17 this happened
    #    at ``mean_coh > 1.0`` (runner's skip_threshold default);
    #    post-2026-05-17 the runner default is 1.5, matching
    #    coh_stop_threshold, so judging and stop-counting are
    #    mutually exclusive.
    #
    # 2. ``mean_coh_all >= coh_filter_threshold`` for a display-time
    #    filter that's INDEPENDENT of the runner's skip_threshold.
    #    The runner now keeps judging up to its stop point; this
    #    threshold lets the plot display a stricter "fully coherent"
    #    cutoff without rerunning the sweep (cheap to change since
    #    it's pure analysis).  Default 1.0 matches the historical
    #    plot anchor; values up to ``DEFAULT_COH_STOP_THRESHOLD = 1.5``
    #    show progressively more of the borderline zone.
    #
    # The trimmed strengths are NOT lost: they're stashed on
    # ``CellSign.extra_aggs`` (ascending magnitude) so the bottom-row
    # coh/rp panels can show how mean_coh ramps PAST the display
    # cutoff before the sweep's coh-stop fires.
    trimmed_tail: List[StrengthAgg] = []
    while aggs and (
        aggs[-1].mean_eff_all is None
        or (aggs[-1].mean_coh_all is not None
            and aggs[-1].mean_coh_all >= coh_filter_threshold)
    ):
        trimmed_tail.append(aggs.pop())
    # ``aggs.pop()`` returned strengths in descending magnitude;
    # reverse so ``extra_aggs[0]`` is the smallest-magnitude trimmed
    # strength (the "first step past the cliff").
    trimmed_tail.reverse()

    summary_path = cell_dir / "summary.json"
    up_reason = None
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            up_reason = summary.get("up_blocked_reason")
        except (OSError, json.JSONDecodeError):
            pass

    return CellSign(
        slot=slot, layer=layer, sign=sign,
        aggs=aggs, extra_aggs=trimmed_tail, up_blocked_reason=up_reason,
    )


def gather_cells(
    experiment_dir: Path,
    *,
    coh_filter_threshold: float = COH_FILTER_THRESHOLD_DEFAULT,
) -> List[CellSign]:
    """Find every ``s<slot>_l<layer>_<sign>`` subdir under
    ``experiment_dir`` and aggregate it.  Sorted by (slot, layer, sign)
    for stable legend ordering.
    """
    out: List[CellSign] = []
    for d in experiment_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        # Pattern: s<digits>_l<digits>_(+1|-1)
        if not (name.startswith("s") and "_l" in name and "_" in name[name.index("_l"):]):
            continue
        try:
            cs = aggregate_cell_sign(d, coh_filter_threshold=coh_filter_threshold)
        except (ValueError, IndexError) as e:
            logger.warning(f"skipping {d}: {e}")
            continue
        if cs.aggs:
            out.append(cs)
    out.sort(key=lambda c: (c.slot, c.layer, c.sign))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

LINESTYLES = {
    "all":  "solid",
    "coh0": (0, (5, 2)),       # dashed
}

FILTER_LABEL = {
    "all":  "(a) all",
    "coh0": "(b) coh=0",
}

# RP-based filters c) ``rp=3`` and d) ``coh=0 & rp=3`` were dropped
# 2026-05-17: they don't reflect normal use of steering (callers
# don't filter by persona retention) and were just adding clutter to
# the plots.  The underlying mean_eff_rp3 / mean_eff_coh0_rp3 fields
# on StrengthAgg are still computed, in case future analysis wants
# to bring them back.
FILTER_KEYS = ("all", "coh0")


def _x_steps_from_cliff(n_strengths: int) -> np.ndarray:
    """Convert step-index (0..N-1, ascending strength) to "steps
    remaining to incoherence" (N-1..0).  The cell's largest strength
    with usable eff data -- the "data cliff" just inside the runner's
    skip-threshold region -- sits at x=0; the smallest at x=N-1.

    Trailing strengths whose mean_eff_all is None (rp+effect judging
    skipped because mean_coh exceeded skip_threshold) are already
    trimmed by ``aggregate_cell_sign`` so x=0 always corresponds to
    a real data point on each (cell, sign) curve.
    """
    return np.arange(n_strengths - 1, -1, -1)


def plot_response_curves(
    cells: List[CellSign],
    *,
    output_path: Path,
    title: str,
    pos_pole_name: Optional[str] = None,
    neg_pole_name: Optional[str] = None,
) -> None:
    """Render one figure with a 2x2 grid:

      top row    : per-cell effect curves (one panel per sign,
                   filter regimes encoded as linestyle).
      bottom row : per-cell unfiltered ``mean_coh`` (solid) and
                   ``3 - mean_rp`` (dotted), same colour per cell.
                   X-axis shared with the corresponding top panel so
                   the reader can eyeball whether coh/rp ramp before
                   the eff-judged cliff at x=0.

    Color encodes (slot, layer); linestyle encodes either filter
    regime (top row) or metric (bottom row -- coh vs inverted-rp).

    The bottom row also plots any ``CellSign.extra_aggs`` (strengths
    past the cliff where coh was judged but rp+effect were skipped)
    at negative x, so the reader can see the runner's full coh
    trajectory through and past the skip_threshold transition.

    ``pos_pole_name`` / ``neg_pole_name`` are the human-readable
    pole labels (from the experiment's config; e.g. "unhelpful" and
    "helpful").  These follow the runner's convention:
    sign=+1 steers toward the rubric's ``pos_label`` (= config
    ``role_to`` for role_transplant axes); sign=-1 steers toward the
    rubric's ``neg_label`` (= config ``role_from``).  When set, the
    subplot titles surface them so the reader doesn't have to
    mentally translate sign to direction.
    """
    # Distinct colour per (slot, layer).  Tab10 covers 7 comfortably.
    unique_cells = sorted({(c.slot, c.layer) for c in cells})
    cmap = plt.get_cmap("tab10")
    color_for_cell = {
        (s, l): cmap(i % 10) for i, (s, l) in enumerate(unique_cells)
    }

    # 2x2 grid.  Top row shares its y axis (effect score).  Bottom
    # row shares its y axis (gating dimensions, 0..3).  Each column
    # shares x with its paired top panel so the bottom-row coh ramp
    # lines up vertically with the top-row eff curves.
    fig, axes = plt.subplots(
        2, 2, figsize=(16, 11),
        sharey="row", sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )
    pos_suffix = f"  (toward {pos_pole_name})" if pos_pole_name else "  (toward neg pole)"
    neg_suffix = f"  (toward {neg_pole_name})" if neg_pole_name else "  (toward pos pole)"
    sign_label = {+1: f"sign = +1{pos_suffix}",
                  -1: f"sign = -1{neg_suffix}"}

    top_axes = axes[0]
    bot_axes = axes[1]

    for ax, sign in zip(top_axes, (+1, -1)):
        cell_signs_here = [c for c in cells if c.sign == sign]
        for cs in cell_signs_here:
            x = _x_steps_from_cliff(len(cs.aggs))
            color = color_for_cell[(cs.slot, cs.layer)]
            base_label = f"s{cs.slot}_l{cs.layer}"
            for filt_key in FILTER_KEYS:
                ys = [_filter_value(a, filt_key) for a in cs.aggs]
                # Normalise so UP = response moved in the STEERED
                # direction on both subplots.  The effect rubric's
                # absolute convention is +eff = more pos_label, -eff
                # = more neg_label, regardless of which way the cell
                # was steered.  Sign=+1 cells were steered toward
                # pos_label, so the rubric's sign already matches
                # "moved in steered direction" -- no flip.  Sign=-1
                # cells were steered toward neg_label, so -eff is the
                # "steering working" direction and we flip the sign
                # so it appears as positive Y.  This makes "up = the
                # steering experiment succeeded" universally and
                # removes the per-panel sign-flip the viewer would
                # otherwise have to do in their head.
                if sign == -1:
                    ys_arr = np.array(
                        [np.nan if y is None else -y for y in ys]
                    )
                else:
                    ys_arr = np.array(
                        [np.nan if y is None else y for y in ys]
                    )
                # The legend label only carries the cell name for the
                # "all" variant; otherwise the per-filter legend would
                # double up to 28 entries per subplot.
                label = base_label if filt_key == "all" else None
                # Markers so single-strength "curves" (= a cell with
                # only one retained x) are visible (a lone line
                # segment of length 0 renders to nothing without a
                # marker).  Filled circles for the solid "all" line;
                # hollow circles for the dashed "coh=0" line.
                marker_face = color if filt_key == "all" else "none"
                ax.plot(
                    x, ys_arr,
                    color=color,
                    linestyle=LINESTYLES[filt_key],
                    linewidth=1.6 if filt_key == "all" else 1.0,
                    alpha=0.95 if filt_key == "all" else 0.75,
                    marker="o",
                    markersize=4 if filt_key == "all" else 3,
                    markerfacecolor=marker_face,
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    label=label,
                )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
        # Per-subplot title carries both the sign and what "up" means
        # there, since each panel has been normalised so positive Y =
        # "response moved toward the steered pole".
        steered_pole = (pos_pole_name if sign == +1 else neg_pole_name)
        if steered_pole:
            ax.set_title(
                f"{sign_label[sign]}    "
                f"\u2191 = more {steered_pole}, \u2193 = opposite"
            )
        else:
            ax.set_title(sign_label[sign])
        ax.grid(True, alpha=0.3)

    # Bottom row: unfiltered coherence (solid) + inverted persona
    # score (dotted).  Same colour per cell as the top row.  Both
    # metrics are oriented so "up = bad" (more incoherent / more
    # out-of-persona), making cross-cell pattern matching easy.
    for ax, sign in zip(bot_axes, (+1, -1)):
        cell_signs_here = [c for c in cells if c.sign == sign]
        for cs in cell_signs_here:
            color = color_for_cell[(cs.slot, cs.layer)]
            # Combine retained + trimmed-trail strengths so the coh
            # curve continues PAST the eff-judged cliff.  Retained
            # strengths occupy x = [N-1 .. 0]; trimmed strengths
            # (ascending magnitude) occupy x = [-1 .. -M].
            n_ret = len(cs.aggs)
            n_extra = len(cs.extra_aggs)
            x_retained = list(_x_steps_from_cliff(n_ret))
            x_extra = list(range(-1, -(n_extra + 1), -1))
            x_all = x_retained + x_extra
            all_aggs = list(cs.aggs) + list(cs.extra_aggs)

            ys_coh = np.array(
                [np.nan if a.mean_coh_all is None else a.mean_coh_all
                 for a in all_aggs]
            )
            ys_rp_inv = np.array(
                [np.nan if a.mean_rp_all is None else (3.0 - a.mean_rp_all)
                 for a in all_aggs]
            )
            ax.plot(
                x_all, ys_coh,
                color=color, linestyle="-", linewidth=1.4, alpha=0.95,
            )
            ax.plot(
                x_all, ys_rp_inv,
                color=color, linestyle=":", linewidth=1.4, alpha=0.85,
            )
            # Mark the eff-judged cliff (x=0) once per panel.  Skip
            # the line if no extra strengths exist (no past-cliff
            # data on this side).
            if n_extra > 0:
                ax.axvline(
                    -0.5, color="black", linewidth=0.8,
                    linestyle="--", alpha=0.4,
                )

        # skip_threshold visual reference: the runner skips rp+effect
        # judging when mean_coh > 1.0 (= the eff-judged cliff).
        # coh_stop_threshold = 1.5 is the actual "stop the sweep"
        # line.  Both rendered as light horizontal references.
        ax.axhline(1.0, color="gray", linewidth=0.7, linestyle=":",
                   alpha=0.6, label="skip_threshold (1.0)")
        ax.axhline(1.5, color="gray", linewidth=0.7, linestyle="--",
                   alpha=0.6, label="coh_stop_threshold (1.5)")
        ax.set_ylim(-0.1, 3.2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Steps remaining to incoherence (0 = last coherent)")

    # Invert x for ALL columns (sharex propagates).  x=0 (cliff) on
    # the RIGHT; increasing x to the LEFT; trimmed past-cliff
    # strengths at negative x sit furthest right.
    for ax in top_axes:
        ax.invert_xaxis()
    # X-axis values are step indices -- only integer ticks are
    # meaningful.  Without this, matplotlib auto-picks half-integer
    # ticks (0.5, 1.5, ...) on the right panel when the x range is
    # narrow.  Applied to the bottom axes so the labels are drawn
    # on the visible (non-shared) tick row.
    for ax in bot_axes:
        ax.xaxis.set_major_locator(MultipleLocator(1))

    # Shared Y labels.
    top_axes[0].set_ylabel(
        "Mean effect score    "
        "(+: toward the steered pole    "
        "\u2212: away from the steered pole)"
    )
    bot_axes[0].set_ylabel(
        "Coherence / inverted-RP    (0 = good, 3 = bad)"
    )

    # Top-left legend: cell colours.
    handles_cells, labels_cells = top_axes[0].get_legend_handles_labels()
    top_axes[0].legend(
        handles_cells, labels_cells,
        title="cell  (slot, layer)",
        loc="upper left",
        fontsize=8,
    )
    # Top-right legend: filter linestyles.
    style_handles = [
        plt.Line2D(
            [0], [0], color="black",
            linestyle=LINESTYLES[k],
            linewidth=1.6 if k == "all" else 1.0,
            alpha=0.95 if k == "all" else 0.75,
            marker="o",
            markersize=4 if k == "all" else 3,
            markerfacecolor=("black" if k == "all" else "none"),
            markeredgecolor="black",
            markeredgewidth=1.0,
            label=FILTER_LABEL[k],
        )
        for k in FILTER_KEYS
    ]
    top_axes[1].legend(
        handles=style_handles,
        title="filter",
        loc="upper left",
        fontsize=8,
    )
    # Bottom-left legend: metric linestyles (coh vs inverted rp).
    metric_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-",
                   linewidth=1.4, label="mean coherence (0 = coherent)"),
        plt.Line2D([0], [0], color="black", linestyle=":",
                   linewidth=1.4, label="3 - mean RP (0 = in persona)"),
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

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Project convention: every tracked-script plot embeds PNG-text-chunk
    # provenance via assistant_axis.png_metadata (title + uv-run invocation
    # + ISO-8601 timestamp + git SHA).  See AGENT_NOTES.md "Plot
    # Provenance Metadata".
    fig.savefig(
        output_path, dpi=140, bbox_inches="tight",
        metadata=png_metadata(title=title.splitlines()[0]),
    )
    logger.info(f"wrote {output_path}")


def _filter_value(agg: StrengthAgg, key: str) -> Optional[float]:
    return {
        "all":  agg.mean_eff_all,
        "coh0": agg.mean_eff_coh0,
        "rp3":  agg.mean_eff_rp3,
        "both": agg.mean_eff_coh0_rp3,
    }[key]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("experiment_dir", type=Path,
                   help="Steering experiment output directory.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output PNG path.  Default: "
                        "<experiment_dir>/response_curves.png")
    p.add_argument("--title", type=str, default=None,
                   help="Plot title (default: experiment_id).")
    p.add_argument(
        "--coh-filter-threshold", type=float,
        default=COH_FILTER_THRESHOLD_DEFAULT,
        help=f"Display-time mean_coh cutoff for the per-cell x=0 "
             f"anchor (default {COH_FILTER_THRESHOLD_DEFAULT}).  A "
             f"strength is trimmed from the top-row effect panels if "
             f"its mean_coh is >= this value; trimmed strengths move "
             f"to the bottom-row coh/rp diagnostic panels as "
             f"past-cliff data.  Independent of the runner's "
             f"skip_threshold (currently 1.5).  Useful for "
             f"tightening the display to 'fully coherent' (e.g. "
             f"0.5) without rerunning the sweep.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    if not args.experiment_dir.exists():
        logger.error(f"missing: {args.experiment_dir}")
        return 2

    cells = gather_cells(
        args.experiment_dir,
        coh_filter_threshold=args.coh_filter_threshold,
    )
    if not cells:
        logger.error(f"no cell subdirs found under {args.experiment_dir}")
        return 1

    config_path = args.experiment_dir / "config.json"
    experiment_id = args.experiment_dir.name
    pos_pole_name = None
    neg_pole_name = None
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            experiment_id = config.get("experiment_id", experiment_id)
            # Pole labels follow the runner's convention for
            # role_transplant axes (see _resolve_steering_spec in
            # steering/run_sweep.py): the rubric's pos_label (= the
            # "+1 sign" pole) is the config's role_to; the rubric's
            # neg_label is the config's role_from.
            axis_source = config.get("axis_source", {}) or {}
            if axis_source.get("type") == "role_transplant":
                pos_pole_name = axis_source.get("role_to") or None
                neg_pole_name = axis_source.get("role_from") or None
            else:
                pos_pole_name = axis_source.get("pos_label") or None
                neg_pole_name = axis_source.get("neg_label") or None
        except (OSError, json.JSONDecodeError):
            pass

    title = args.title or f"{experiment_id}: per-cell response curves"
    output_path = args.output or (args.experiment_dir / "response_curves.png")

    logger.info(
        f"found {len(cells)} (cell, sign) groups across "
        f"{len({(c.slot, c.layer) for c in cells})} cells "
        f"for {experiment_id}"
    )
    plot_response_curves(
        cells, output_path=output_path, title=title,
        pos_pole_name=pos_pole_name, neg_pole_name=neg_pole_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
