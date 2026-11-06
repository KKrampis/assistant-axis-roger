#!/usr/bin/env python3
"""Sweep the GPT-4.1-mini / second-judge weight in the desc+inst score
average and plot mean per-axis projection-ρ vs the blend weight.

The second judge defaults to Sonnet-4 (`--second_judge sonnet`); pass
`--second_judge haiku` (or any other provider-named subdir found under
each axis directory) to compare a different second judge.

For each axis we compute, per entity::

    gpt_score   = combine_desc_inst_one_judge(GPT_d, GPT_i, weights=di_weights)
    other_score = combine_desc_inst_one_judge(Other_d, Other_i, weights=di_weights)
    score(w)    = w · gpt_score + (1 - w) · other_score

where ``di_weights`` is the standard desc/inst tiebreak weighting
(default: ``inst_tie`` = ``0.499*desc + 0.501*inst``; pass
``--di_weights`` to override).  Spearman ρ is taken against the
projection of each entity vector onto the axis direction at the
chosen ``(slot, layer)``, after applying the ``--whitening`` regime
(default: project-canonical ``soft_shear=3``; pass ``--whitening raw``
for the historical no-whitening view).

::

    w = 1   → pure GPT (one judge, desc+inst-combined)
    w = 0.5 → 50/50 average (≈ 4-way average across GPT and Sonnet)
    w = 0   → pure Sonnet (one judge, desc+inst-combined)

Historical empirical finding (33 axes, slot 3, raw projection):

==================== ============
weight on GPT (w)    mean ρ
==================== ============
0.0  (pure Sonnet)    0.5693
0.5  (50/50)          **0.5953**
1.0  (pure GPT)       0.5758
==================== ============

The 50/50 mix was the principled default at the time -- mean ρ was
essentially flat over ``w ∈ [0.1, 0.9]`` (parabolic fit on the
interior gives a peak at ``w = 0.530``, ρ ≈ 0.5949, which is *0.0004
lower* than the discrete 50/50 value).  Endpoints show "kinks"
because the mode flips from "averaging two judges" to "using one
judge alone" -- a categorical regime change, not a smooth blend,
which is why the parabola fit is restricted to the interior.  Re-run
with the canonical defaults (35 axes, slot 6, soft_shear=3) to get
the current operating point; the parabolic-peak interpretation still
applies under any whitening regime since whitening is a per-axis
linear preprocess and the cross-judge blend happens after it.

Per-axis interior slope ``Δρ = ρ(w=0.9) − ρ(w=0.1)`` ranks each axis
by which provider was a stronger judge for it; that ranking drives
the colormap and the legend ordering on the plot
(blue = Sonnet-better, red = GPT-better).

Inputs (from ``--experiment_dir``)
----------------------------------

The standard per-axis directory tree produced by
:mod:`results_analysis.axis_judge_correlation`::

    <experiment_dir>/
      pair_list_di.json    # or any pair list passed via --pairs
      <pos>_vs_<neg>/
        gpt/scores_descriptions.json
        gpt/scores_instructions.json
        sonnet/scores_descriptions.json
        sonnet/scores_instructions.json

Plus the standard activation-vectors directory passed via ``--data_dir``.

Outputs (to ``--experiment_dir``)
---------------------------------

- ``gpt_sonnet_weight_sweep.png`` -- the plot.
- ``gpt_sonnet_weight_sweep.json`` -- per-axis ρ-vs-w table, the
  interior parabolic fit (a, b, c, R², peak, peak_rho), and the
  per-axis interior slope used for the legend ordering.

Examples
--------

::

    # Default: 33 axes
    uv run python results_analysis/gpt_sonnet_weight_sweep.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from assistant_axis import (
    cohort_from_pairs, entity_id, json_metadata, pair_type_of, png_metadata,
)
from assistant_axis.judge_loaders import migrate_v1_static_scores
from assistant_axis.judge_score_combine import (
    add_di_weights_arg,
    combine_desc_inst_one_judge,
    declare_constants_dependency,
    parse_di_weights_arg,
)
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
    load_and_register,
)
from results_analysis.axis_judge_correlation import _load_vector_file
from results_analysis.canonical_angles.data import build_goal_nogoal_subspaces
from results_analysis.canonical_angles.whitening import (
    DEFAULT_WHITENING_SPEC,
    WhiteningBasis,
    fit_shear,
    fit_whitening,
    parse_whitening_spec,
)


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
    "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
)
DEFAULT_SLOT, LAYER = 6, 25  # New default after May 2026 rejudge: slot 6
                             # (</think>) beats slot 3 (\\n) on judge ρ.
                             # Pass --slot 3 (\\n) or 7 (\\n\\n post) to compare.
PARABOLA_COLOR = "#1faa4f"


def _v(path: Path, *, slot: int) -> torch.Tensor:
    """Load (n_slots, n_layers, hidden) and slice to (slot, LAYER) as float."""
    return _load_vector_file(path).float()[slot, LAYER]


def axis_unit(data_dir: Path, pos: str, neg: str, *, slot: int,
              pair_type: str = "traits") -> torch.Tensor:
    """Unit axis direction at (slot, LAYER); ``pair_type`` selects the
    ``traits/`` vs ``roles/`` subdir under ``data_dir``."""
    p = _v(data_dir / pair_type / "vectors" / f"{pos}.pt", slot=slot)
    n = _v(data_dir / pair_type / "vectors" / f"{neg}.pt", slot=slot)
    d = p - n
    return d / torch.linalg.vector_norm(d)


def _rho_at_weight(d: dict, w: float) -> float:
    score = w * d["g2"] + (1 - w) * d["s2"]
    return float(spearmanr(score, d["proj"]).correlation)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR),
                   help=f"Directory holding the pair list and per-axis "
                        f"score subdirs (default: {DEFAULT_EXPERIMENT_DIR}).")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                   help=f"Activation vectors directory "
                        f"(default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--pairs", default="pair_list_di.json",
                   help="Pair-list JSON filename "
                        "(default: pair_list_di.json -- every axis with "
                        "desc+inst judging from both GPT and Sonnet).")
    p.add_argument("--second_judge", default="sonnet",
                   help="Subdir name for the second-judge scores under each "
                        "axis directory (default: 'sonnet'; common "
                        "alternative: 'haiku'). Filenames inside that subdir "
                        "are still scores_descriptions.json / "
                        "scores_instructions.json.")
    p.add_argument("--second_judge_label", default=None,
                   help="Display label for the second judge in plot titles, "
                        "axis labels, and JSON output. Defaults to a "
                        "Title-cased version of --second_judge.")
    p.add_argument("--plot", default=None,
                   help="Output plot filename (default: "
                        "gpt_<second_judge>_weight_sweep_<cohort>_slot{N}.png; "
                        "cohort comes from --pairs, slot from --slot).")
    p.add_argument("--rhos_json", default=None,
                   help="Output JSON filename (default: "
                        "gpt_<second_judge>_weight_sweep_<cohort>_slot{N}.json).")
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT,
                   help=f"Token-position slot to project onto "
                        f"(default: {DEFAULT_SLOT} = </think>).  Pass --slot 3 "
                        f"(\\n) or 7 (\\n\\n post) to compare.")
    p.add_argument("--whitening", default=DEFAULT_WHITENING_SPEC,
                   help=f"Whitening regime applied to both entity vectors "
                        f"and the axis direction before computing ρ. "
                        f"Forms: 'raw', 'soft_K=N', 'lw', 'oas', "
                        f"'soft_shear=L'.  Default: "
                        f"{DEFAULT_WHITENING_SPEC!r} (project canonical -- "
                        f"see assistant_axis/canonical_angles/whitening.py "
                        f"selection-history block).")
    p.add_argument("--ca_kind", default="combined",
                   choices=("combined", "traits", "roles"),
                   help="Goal/no-goal subspaces for soft-shear fitting "
                        "(passed to ``build_goal_nogoal_subspaces``). "
                        "Ignored for other whitening regimes. "
                        "Default ``combined`` matches the L-sweep and "
                        "shear_l_vs_k_comparison defaults.")
    p.add_argument("--w_step", type=float, default=0.025,
                   help="Step size on the w grid (default: 0.025 -> "
                        "41 points on [0, 1]).  Pass --w_step 0.05 to "
                        "reproduce the historical coarser sweep.")
    add_di_weights_arg(p)
    args = p.parse_args()
    di_weights = parse_di_weights_arg(args.di_weights)
    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    slot = int(args.slot)
    second_judge = args.second_judge
    second_label = (args.second_judge_label
                     or second_judge[:1].upper() + second_judge[1:])
    cohort = cohort_from_pairs(args.pairs)
    wh_method, wh_n = parse_whitening_spec(args.whitening)
    # Filename tag: 'raw' is the historical default and gets no suffix
    # so old v1-archive plots and the new raw view share a slot.  Any
    # other regime gets a short, filename-safe suffix that mirrors the
    # spec (e.g. 'soft_shear=3' -> '_softshear3', 'soft_K=2' -> '_softK2').
    if wh_method == "raw":
        wh_suffix = ""
    elif wh_n is not None:
        wh_suffix = f"_{wh_method.replace('_', '')}{wh_n}"
    else:
        wh_suffix = f"_{wh_method.replace('_', '')}"
    if args.plot is None:
        args.plot = (f"gpt_{second_judge}_weight_sweep_{cohort}_"
                     f"slot{slot}{wh_suffix}.png")
    if args.rhos_json is None:
        args.rhos_json = (f"gpt_{second_judge}_weight_sweep_{cohort}_"
                          f"slot{slot}{wh_suffix}.json")

    # ``inputs`` accumulator: every cache read below goes through
    # load_and_register, so the read AND the InputSpec record are
    # built atomically (single call site can't drift -- see
    # AGENT_NOTES.md "Reader+registrar pattern").  Vector subtrees
    # are added later, before the savefig.
    inputs: list[InputSpec] = []
    # Centralised constants we inherit (DEFAULT_WHITENING_SPEC's value
    # is project-wide; if it changes, downstream plots are stale).
    declare_constants_dependency(inputs)
    pairs, _spec, _check = load_and_register(
        experiment_dir / args.pairs,
        dep_key="pairs_json",
        inputs=inputs, policy="warn",
    )
    print(f"Loaded {len(pairs)} axis pairs from {args.pairs}")

    # Cache standalone entity vectors at (slot, LAYER), default-centered.
    # Build kinds_for_name alongside so we can migrate any v1 (bare-name)
    # second-judge cache (e.g. Sonnet, which wasn't included in Phase 5b
    # rejudging) up to v2 in-memory before intersecting with the v2
    # GPT/Haiku caches.  See assistant_axis.judge_loaders.migrate_v1_static_scores.
    default_sl = _v(data_dir / "traits" / "vectors" / "default.pt", slot=slot)
    entity_vecs: dict[str, np.ndarray] = {}
    kinds_for_name: dict[str, set] = {}
    for et in ("traits", "roles"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            try:
                v = _load_vector_file(fp).float()[slot, LAYER]
                entity_vecs[entity_id(fp.stem, et)] = (v - default_sl).numpy()
                kinds_for_name.setdefault(fp.stem, set()).add(et)
            except Exception:  # pragma: no cover -- skip unreadable files
                continue

    # ----- Whitening basis ---------------------------------------------
    # Fit a whitening / shearing transform once and apply to both the
    # entity pool and per-axis directions.  The mapping is linear, so
    # applying it pre-projection is equivalent to applying it inside the
    # dot product -- and pre-applying keeps the hot loop tiny.
    #
    # ``soft_shear`` (the project default) needs the goal/no-goal CA
    # subspaces from ``build_goal_nogoal_subspaces``; the other regimes
    # fit on the entity pool itself.  ``raw`` returns ``None`` and the
    # downstream code falls back to the original vectors.
    basis: WhiteningBasis | None = None
    if wh_method == "soft_shear":
        A_g, A_n = build_goal_nogoal_subspaces(
            data_dir, slot=slot, layer=LAYER, kind=args.ca_kind)
        n_pairs_max = min(A_g.shape[1], A_n.shape[1])
        if wh_n is not None and wh_n > n_pairs_max:
            raise SystemExit(
                f"--whitening soft_shear={wh_n} exceeds the canonical-angle "
                f"pair budget for kind={args.ca_kind!r} "
                f"(min(n_g,n_n)={n_pairs_max}).  Pick a smaller L or a "
                f"different ca_kind."
            )
        basis = fit_shear(A_g, A_n, L=int(wh_n))
        print(f"Whitening: soft_shear L={wh_n} on '{args.ca_kind}' "
              f"subspaces (n_pairs_max={n_pairs_max})")
    elif wh_method == "soft_K":
        pool = np.stack(list(entity_vecs.values()))
        basis = fit_whitening("soft_K", pool, K=int(wh_n))
        print(f"Whitening: soft_K K={wh_n} on default-centered entity pool "
              f"(n={pool.shape[0]})")
    elif wh_method in ("lw", "oas"):
        pool = np.stack(list(entity_vecs.values()))
        basis = fit_whitening(wh_method, pool)
        print(f"Whitening: {wh_method} cov^(-1/2) on default-centered "
              f"entity pool (n={pool.shape[0]})")
    else:  # 'raw'
        print("Whitening: raw (identity)")

    if basis is not None and basis.method != "raw":
        # Apply once: entity pool is (n, D); rebind in place.
        for n, v in entity_vecs.items():
            entity_vecs[n] = basis.apply(v[None, :])[0]

    # For each axis, precompute the per-entity (gpt_2way, sonnet_2way,
    # projection) on the common entity set, since these don't depend on
    # the weight.
    per_axis: dict[tuple[str, str], dict] = {}
    skipped_missing: list[str] = []
    for it in pairs:
        pos, neg = it["pos"], it["neg"]
        axis_id = f"{pos}_vs_{neg}"
        axis_dir = experiment_dir / axis_id
        # The second judge may not have judged every axis in the pair
        # list (e.g. Haiku covers only the 12 axes that appear in
        # pair_list_responses.json, but pair_list_di.json has 35).
        # Skip cleanly instead of FileNotFoundError'ing the whole run.
        sj_d = axis_dir / second_judge / "scores_descriptions.json"
        sj_i = axis_dir / second_judge / "scores_instructions.json"
        if not (sj_d.exists() and sj_i.exists()):
            skipped_missing.append(axis_id)
            continue
        g_d, _, _ = load_and_register(
            axis_dir / "gpt" / "scores_descriptions.json",
            dep_key=f"judge_{axis_id}_descriptions_gpt",
            inputs=inputs, policy="warn",
        )
        g_i, _, _ = load_and_register(
            axis_dir / "gpt" / "scores_instructions.json",
            dep_key=f"judge_{axis_id}_instructions_gpt",
            inputs=inputs, policy="warn",
        )
        s_d, _, _ = load_and_register(
            sj_d,
            dep_key=f"judge_{axis_id}_descriptions_{second_judge}",
            inputs=inputs, policy="warn",
        )
        s_i, _, _ = load_and_register(
            sj_i,
            dep_key=f"judge_{axis_id}_instructions_{second_judge}",
            inputs=inputs, policy="warn",
        )
        # Per-judge desc/inst combination using the standard tiebreak weights.
        # The cross-judge sweep below is independent of this choice -- it sweeps
        # GPT vs <second_judge>, treating each as a single (already
        # desc+inst-combined) score.
        #
        # ``migrate_v1_static_scores`` is a no-op when the input is
        # already v2 (entity_id-keyed), but rescues mixed-format runs:
        # post-Phase-5b GPT/Haiku caches are v2, Sonnet desc/inst
        # caches are still v1 (bare names, 9 collisions dropped).
        # Without migration, the v2/v1 intersection is empty.
        gpt_scores = migrate_v1_static_scores(
            combine_desc_inst_one_judge(g_d, g_i, weights=di_weights),
            kinds_for_name,
        )
        son_scores = migrate_v1_static_scores(
            combine_desc_inst_one_judge(s_d, s_i, weights=di_weights),
            kinds_for_name,
        )
        common = sorted(set(gpt_scores) & set(son_scores) & set(entity_vecs))
        if len(common) < 3:
            print(f"  [skip] {pos}/{neg}: only {len(common)} shared entities")
            continue
        g2 = np.array([gpt_scores[n] for n in common])
        s2 = np.array([son_scores[n] for n in common])
        a = axis_unit(data_dir, pos, neg, slot=slot,
                       pair_type=pair_type_of(it)).numpy()
        # Whitening is linear, so equivalent to applying inside the dot
        # product.  We've already whitened entity_vecs once above; just
        # whiten the axis direction here per-axis.
        if basis is not None and basis.method != "raw":
            a = basis.apply(a[None, :])[0]
        proj = np.array([float(np.dot(entity_vecs[n], a)) for n in common])
        per_axis[(pos, neg)] = {
            "g2": g2, "s2": s2, "proj": proj, "n": len(common)}

    if skipped_missing:
        print(f"  [info] {second_judge} caches absent for "
              f"{len(skipped_missing)}/{len(pairs)} axes; "
              f"skipped: {skipped_missing}")
    if not per_axis:
        raise SystemExit(
            f"No axes had complete {second_judge} caches; "
            f"check --pairs (e.g. Haiku has data for the 12-axis "
            f"pair_list_responses.json subset)."
        )

    # Sort axes by interior slope Δρ = ρ(0.9) - ρ(0.1).  Ascending order
    # = most Sonnet-favored first → most GPT-favored last.
    slopes = {k: _rho_at_weight(d, 0.9) - _rho_at_weight(d, 0.1)
              for k, d in per_axis.items()}
    sorted_axis_keys = sorted(per_axis.keys(), key=lambda k: slopes[k])
    per_axis = {k: per_axis[k] for k in sorted_axis_keys}

    # Sweep w from 0 to 1 in --w_step increments.  Default 0.025 ->
    # 41 points; pass --w_step 0.05 for the historical coarser 21-point
    # grid (the early-2026 baseline).  Step must divide 1.0 exactly or
    # we won't land on integer-multiple grid points.
    if args.w_step <= 0 or args.w_step > 0.5:
        raise SystemExit(
            f"--w_step must be in (0, 0.5], got {args.w_step}")
    n_w = int(round(1.0 / args.w_step)) + 1
    if abs(n_w - 1 - 1.0 / args.w_step) > 1e-9:
        raise SystemExit(
            f"--w_step={args.w_step} doesn't divide 1.0 evenly; "
            f"pick a divisor of 1.0 (e.g. 0.025, 0.05, 0.02, 0.01).")
    ws_arr = np.linspace(0.0, 1.0, n_w)
    i_half = int(round(0.5 / args.w_step))
    all_rhos: list[list[float]] = []
    mean_rhos: list[float] = []
    for w in ws_arr:
        rhos = [_rho_at_weight(d, w) for d in per_axis.values()]
        mean_rhos.append(float(np.mean(rhos)))
        all_rhos.append(rhos)
    mean_arr = np.array(mean_rhos)
    all_arr = np.array(all_rhos)  # (n_w, n_axes)

    # Parabolic fit on w ∈ [0.1, 0.9] only -- the endpoint kinks at
    # w=0 / w=1 (single-judge regime) distort a global fit.
    mask = (ws_arr >= 0.10 - 1e-9) & (ws_arr <= 0.90 + 1e-9)
    ws_fit = ws_arr[mask]
    mean_fit = mean_arr[mask]
    c_fit, b_fit, a_fit = np.polyfit(ws_fit, mean_fit, 2)
    yhat_fit = a_fit + b_fit * ws_fit + c_fit * ws_fit ** 2
    ss_res = float(((mean_fit - yhat_fit) ** 2).sum())
    ss_tot = float(((mean_fit - mean_fit.mean()) ** 2).sum())
    r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if c_fit < 0:
        w_peak = float(-b_fit / (2 * c_fit))
        rho_peak = float(a_fit + b_fit * w_peak + c_fit * w_peak ** 2)
    else:
        i_best = int(np.argmax(mean_arr))
        w_peak = float(ws_arr[i_best])
        rho_peak = float(mean_arr[i_best])

    print(f"\nSweep w_GPT from 0 to 1 in steps of {args.w_step} "
          f"({n_w} points), n={len(per_axis)} axes "
          f"(whitening={args.whitening}):")
    print(f"  pure {second_label} (w=0.0):  mean ρ = {mean_arr[0]:+.4f}")
    print(f"  50/50      (w=0.5):  mean ρ = {mean_arr[i_half]:+.4f}")
    print(f"  pure GPT   (w=1.0):  mean ρ = {mean_arr[-1]:+.4f}")
    print(f"  parabolic peak (interior fit on w∈[0.1, 0.9], "
          f"R²={r2_fit:.4f}): w={w_peak:.3f}, mean ρ ≈ {rho_peak:+.4f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    n_axes = len(per_axis)
    axis_keys = list(per_axis.keys())
    axis_colors = plt.cm.RdBu_r(np.linspace(0.05, 0.95, n_axes))

    for j, k in enumerate(axis_keys):
        ax.plot(ws_arr, all_arr[:, j], color=axis_colors[j],
                lw=1.0, alpha=0.75)

    mean_handle, = ax.plot(
        ws_arr, mean_arr, color="black", lw=2.5, marker="o", markersize=5,
        zorder=20, label=f"mean ({n_axes} axes)")

    # Parabola fit (drawn on TOP of the per-axis curves and the mean).
    ws_fit_plot_in = np.linspace(0.10, 0.90, 50)
    ws_fit_plot_out = np.linspace(0.0, 1.0, 100)
    y_fit_in = a_fit + b_fit * ws_fit_plot_in + c_fit * ws_fit_plot_in ** 2
    y_fit_out = a_fit + b_fit * ws_fit_plot_out + c_fit * ws_fit_plot_out ** 2
    extrap_handle, = ax.plot(
        ws_fit_plot_out, y_fit_out, color="#444444", lw=1.0, ls=":",
        alpha=0.9, zorder=25, label="parabola (extrapolated)")
    fit_handle, = ax.plot(
        ws_fit_plot_in, y_fit_in, color=PARABOLA_COLOR, lw=2.0, ls="-",
        alpha=0.95, zorder=26,
        label=f"parabola fit on [0.1, 0.9] (R²={r2_fit:.3f})")
    w50_handle = ax.axvline(
        0.5, color="grey", linestyle=":", lw=1, alpha=0.7, label="w=0.5")
    peak_handle = ax.axvline(
        w_peak, color=PARABOLA_COLOR, linestyle="--", lw=1.4,
        label=f"interior peak w={w_peak:.3f}")

    ax.set_xlabel(f"Weight on GPT-4.1-mini\n(remaining on {second_label})",
                  fontsize=9)
    ax.set_ylabel(f"Per-axis Spearman ρ (slot {slot}, "
                  f"whitening={args.whitening})", fontsize=9)
    title_line = (f"GPT/{second_label} score-blend sweep -- mean ρ across "
                  f"{n_axes} axes")
    # set_title here functions as a suptitle (single-panel figure).
    # Two lines only -- the legend's "Axes: sorted Sonnet-best to
    # GPT-best" annotation already explains the slope-based ordering,
    # so we don't repeat it in the title.
    second_initial = second_label[0]
    ax.set_title(
        title_line + "\n"
        f"{second_initial}={mean_arr[0]:.3f}  "
        f"50/50={mean_arr[i_half]:.3f}  "
        f"G={mean_arr[-1]:.3f}",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    y_hi = float(min(1.0, all_arr.max() + 0.02))
    ax.set_ylim(0.30, y_hi)
    n_below = int((all_arr.min(axis=0) < 0.30).sum())
    below_msg = (f"{n_below} axis below ρ=0.30 not shown"
                 if n_below else "")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.3)

    # Two stacked legends on the right.
    axis_handles = [
        plt.Line2D([], [], color=axis_colors[i], lw=1.0, alpha=0.75)
        for i in range(n_axes)
    ]
    axis_labels = [f"{k[0]}/{k[1]} ({slopes[k]:+.3f})" for k in axis_keys]
    leg_axes = ax.legend(
        axis_handles, axis_labels,
        loc="upper left", bbox_to_anchor=(1.01, 0.98),
        fontsize=9, ncol=1, frameon=True, framealpha=0.9,
        handlelength=1.5, borderpad=0.6, labelspacing=0.32)
    ax.add_artist(leg_axes)
    # Heading via plain annotation (matplotlib's tight-bbox doesn't
    # account for legend titles wider than the legend body and silently
    # clips them).
    ax.annotate(
        f"Axes: sorted {second_label}-best to GPT-best",
        xy=(1.01, 1.005), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(
        [mean_handle, fit_handle, extrap_handle, w50_handle, peak_handle],
        [mean_handle.get_label(), fit_handle.get_label(),
         extrap_handle.get_label(), w50_handle.get_label(),
         peak_handle.get_label()],
        loc="lower left", bbox_to_anchor=(1.01, 0.0),
        fontsize=7, frameon=True, framealpha=0.9, handlelength=1.5)
    if below_msg:
        ax.text(0.02, 0.02, below_msg, transform=ax.transAxes,
                fontsize=7, va="bottom", color="#555555")
    plt.tight_layout()

    # --- Provenance inputs (used by both PNG + JSON writes) ---
    # ``inputs`` was already populated above by load_and_register at
    # every cache-read site (pairs_json + per-axis × per-judge ×
    # per-mode scores caches).  Subtree deps are appended here.
    inputs.extend([
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras={"slot": str(slot), "layer": str(LAYER)}),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras={"slot": str(slot), "layer": str(LAYER)}),
    ])

    out_path = experiment_dir / args.plot
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    json_out = {
        "n_axes": n_axes,
        "first_judge": "gpt",
        "second_judge": second_judge,
        "second_judge_label": second_label,
        "slot": int(slot),
        "layer": int(LAYER),
        "whitening": args.whitening,
        "ca_kind": args.ca_kind if wh_method == "soft_shear" else None,
        "ws": [float(w) for w in ws_arr],
        "mean_rho_by_w": [float(v) for v in mean_arr],
        "parabola_fit": {
            "domain": [0.10, 0.90],
            "a": float(a_fit), "b": float(b_fit), "c": float(c_fit),
            "r2": float(r2_fit),
            "peak_w": float(w_peak), "peak_rho": float(rho_peak),
        },
        "per_axis": [
            {"pos": k[0], "neg": k[1],
             "interior_slope_delta_rho": float(slopes[k]),
             "rho_by_w": [float(v) for v in all_arr[:, i]],
             "n": int(per_axis[k]["n"])}
            for i, k in enumerate(axis_keys)
        ],
    }
    json_path = experiment_dir / args.rhos_json
    envelope = json_metadata(
        json_out, inputs=inputs,
        title=f"gpt_{second_judge}_weight_sweep slot={slot} pairs={args.pairs}")
    json.dump(envelope, open(json_path, "w"), indent=2)
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
