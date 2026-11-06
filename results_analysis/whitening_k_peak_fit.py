#!/usr/bin/env python3
"""Fit a parabola to each ρ-vs-K curve produced by
:mod:`results_analysis.whitening_k_sweep` (N axes x 2 sources) over
``K ∈ [0, 32]``.

Tests two parametrizations:

(a) **linear-K**:   ``ρ(K) ≈ a + b·K + c·K²``
(b) **log2(K+1)**:  ``ρ(K) ≈ a + b·log2(K+1) + c·log2(K+1)²``

Reports R² per fit, the fitted peak K per curve, and the correlation
between the desc+inst peak and the responses peak across axes where both
sources have finite peaks.

Inputs (read from ``--experiment_dir``):

- ``whitening_k_sweep_<cohort>_slot{N}.json`` -- produced by
  ``whitening_k_sweep.py`` with the **same** ``--pairs`` list.  The
  ``<cohort>`` token comes from the pair-list filename
  (``cohort_from_pairs``) and is shared across both scripts so they
  always agree on input/output names.
- ``pair_list_di.json`` (default ``--pairs``) -- desc+instr cohort:
  every axis with desc+inst judging from both GPT and Sonnet.  Many
  of those axes do not have GPT response scores, so ρ for ``responses``
  is NaN on those panels.  Use ``--pairs pair_list_responses.json`` for
  the smaller cohort that has response judging too.

Outputs (written to ``--experiment_dir``):

- ``whitening_k_peak_fit_<cohort>_slot{N}.json`` -- per-curve fit
  records (R², fitted peak K, y-hat).
- ``rho_vs_K_parabolic_fits_<cohort>_slot{N}.png`` -- one panel per
  axis via ``--pairs``.

Examples
--------

::

    # Default cohort (pair_list_di.json) at three token slots:
    uv run python results_analysis/whitening_k_sweep.py --slot 3
    uv run python results_analysis/whitening_k_sweep.py --slot 6
    uv run python results_analysis/whitening_k_sweep.py --slot 7

    uv run python results_analysis/whitening_k_peak_fit.py --slot 3
    uv run python results_analysis/whitening_k_peak_fit.py --slot 6
    uv run python results_analysis/whitening_k_peak_fit.py --slot 7

    # Responses cohort (smaller mosaic; auto-names use cohort=responses):
    uv run python results_analysis/whitening_k_sweep.py \\
        --pairs pair_list_responses.json --slot 6
    uv run python results_analysis/whitening_k_peak_fit.py \\
        --pairs pair_list_responses.json --slot 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from assistant_axis import (
    cohort_from_pairs,
    json_metadata,
    png_metadata,
    suptitle_with_specs,
)
from assistant_axis.provenance import (
    CACHE_POLICIES,
    InputSpec,
    load_and_register,
)

DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)

# K values used for the parabola fit.  Subset of the K_VALUES sweep in
# whitening_k_sweep.py; restrict to the K range where a parabola is a
# reasonable fit (the "sweet spot" we're trying to localise).  Past
# K~32 the curves flatten and a parabola is a poor match.
FIT_K = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

DEFAULT_PAIR_LIST = "pair_list_di.json"
# Sweep / plot / fit_json defaults are derived from --pairs and --slot in
# ``_resolve_io_names`` (see below) using ``cohort_from_pairs(pairs)``.
# The constants below are kept as sentinels so the resolver can detect
# "user did not pass an explicit override" via the CLI parser default.
DEFAULT_SWEEP = "<auto: whitening_k_sweep_<cohort>_slot{N}.json>"
DEFAULT_PLOT = "<auto: rho_vs_K_parabolic_fits_<cohort>_slot{N}.png>"
DEFAULT_FIT_JSON = "<auto: whitening_k_peak_fit_<cohort>_slot{N}.json>"

# Lay out many axes in a moderately wide grid (33 → 6×6 with 3 blanks).
GRID_N_COLS = 6


def fit_parabola(xs, ys):
    """Least-squares fit ``y = p2*x² + p1*x + p0`` using finite pairs only.

    ``np.polyfit`` returns highest-degree first ``[p2, p1, p0]``. Returns
    ``((p0, p1, p2), r2, peak_x, yhat_full)`` with ``yhat_full`` evaluating
    the fit at every ``xs`` (all ``xs`` should be finite for FIT_K grids).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    yhat_full = np.full(xs.shape[0], np.nan, dtype=float)
    xv, yv = xs[mask], ys[mask]
    if xv.size < 3:
        return (np.nan, np.nan, np.nan), np.nan, np.nan, yhat_full

    coef = np.polyfit(xv, yv, 2)
    p2, p1, p0 = coef
    yhat_full = np.polyval(coef, xs)
    yhat_v = yhat_full[mask]
    ss_res = float(((yv - yhat_v) ** 2).sum())
    ym = float(np.mean(yv))
    ss_tot = float(((yv - ym) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else np.nan

    peak_x = (-p1 / (2 * p2)) if p2 < 0 else np.nan
    return (p0, p1, p2), r2, peak_x, yhat_full


def _resolve_io_names(args: argparse.Namespace) -> tuple[str, str, str]:
    """Auto-derive sweep/plot/fit_json filenames from --pairs and --slot
    when the user hasn't explicitly overridden them.

    Output names embed the cohort token (extracted from --pairs via
    ``cohort_from_pairs``) and the slot, e.g.
    ``whitening_k_sweep_di_slot6.json`` /
    ``whitening_k_sweep_responses_slot6.json``.  This avoids the silent
    cross-cohort collisions we used to get when the auto-derived name
    encoded slot only.  Slot defaults to 6 when --slot is not passed.
    """
    cohort = cohort_from_pairs(args.pairs)
    slot = args.slot if args.slot is not None else 6
    sweep_f = (args.sweep
               if args.sweep != DEFAULT_SWEEP
               else f"whitening_k_sweep_{cohort}_slot{slot}.json")
    plot_f = (args.plot
              if args.plot != DEFAULT_PLOT
              else f"rho_vs_K_parabolic_fits_{cohort}_slot{slot}.png")
    fit_f = (args.fit_json
             if args.fit_json != DEFAULT_FIT_JSON
             else f"whitening_k_peak_fit_{cohort}_slot{slot}.json")
    return sweep_f, plot_f, fit_f


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR),
                   help=f"Directory holding the K-sweep input + pair list "
                        f"(default: {DEFAULT_EXPERIMENT_DIR}).")
    p.add_argument("--pairs", default=DEFAULT_PAIR_LIST,
                   help=f"Pair-list JSON filename (default: {DEFAULT_PAIR_LIST}).")
    p.add_argument("--sweep", default=DEFAULT_SWEEP,
                   help=f"K-sweep JSON filename (default: {DEFAULT_SWEEP}).")
    p.add_argument("--slot", type=int, default=None,
                   help="If set, and sweep/plot/fit_json are still their "
                        "generic defaults, use …_slot{N}.json/png filenames. "
                        "Also adds slot N to the figure title.")
    p.add_argument("--fit_json", default=DEFAULT_FIT_JSON,
                   help="Output fit-results JSON filename "
                        f"(default: {DEFAULT_FIT_JSON}).")
    p.add_argument("--plot", default=DEFAULT_PLOT,
                   help="Output plot filename "
                        f"(default: {DEFAULT_PLOT}).")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to react to drift in the sweep JSON's "
                        "recorded provenance: strict (raise), warn (stderr "
                        "+ proceed; default), rebuild (raise unless a "
                        "rebuild_callback is wired in by a future version), "
                        "off (skip validation).")
    args = p.parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    sweep_f, plot_f, fit_f = _resolve_io_names(args)

    # ``inputs`` is built up via load_and_register: each cache read
    # both unwraps the envelope (if present), validates its recorded
    # provenance under ``--cache-policy``, AND appends an InputSpec to
    # this list.  Single call site means we can't accidentally read a
    # cache without registering it (the failure mode that left
    # gpt_anthropic_response_weight_sweep.py without a provenance
    # envelope at all -- see AGENT_NOTES.md "Reader+registrar pattern").
    inputs: list[InputSpec] = []

    pairs, _spec_pairs, _check_pairs = load_and_register(
        experiment_dir / args.pairs,
        dep_key="pairs_json",
        inputs=inputs,
        policy=args.cache_policy,
    )
    pair_keys_ordered = [(it["pos"], it["neg"]) for it in pairs]

    # Legacy (bare-list) sweep files are detected by the absence of an
    # envelope and pass through unvalidated (returns check=None).
    records, _spec_sweep, _check_sweep = load_and_register(
        experiment_dir / sweep_f,
        dep_key="sweep_json",
        inputs=inputs,
        policy=args.cache_policy,
    )
    curves: dict[tuple[str, str], dict] = {}
    for r in records:
        curves.setdefault((r["pos"], r["neg"]), {}).setdefault(
            r["source"], {})[r["K"]] = r["rho"]

    rows: list[dict] = []
    for (pos, neg) in pair_keys_ordered:
        cmap = curves.get((pos, neg), {})
        for source in ("desc_inst", "responses"):
            ys_at_K = cmap.get(source, {})
            ys = [ys_at_K.get(K, float("nan")) for K in FIT_K]

            xs_lin = FIT_K
            (_a_l, _b_l, _c_l), r2_lin, peak_lin, yhat_lin = fit_parabola(
                xs_lin, ys)
            peak_lin_clipped = (max(0, min(32, peak_lin))
                                if np.isfinite(peak_lin) else float("nan"))

            xs_log = [np.log2(K + 1) for K in FIT_K]
            (_a_g, _b_g, _c_g), r2_log, peak_log_x, yhat_log = fit_parabola(
                xs_log, ys)
            peak_log_K = (2 ** peak_log_x - 1
                          if np.isfinite(peak_log_x) else float("nan"))
            peak_log_K_clipped = (max(0, min(32, peak_log_K))
                                  if np.isfinite(peak_log_K) else float("nan"))

            rows.append({
                "pos": pos, "neg": neg, "source": source,
                "r2_linear": r2_lin,
                "r2_log": r2_log,
                "peak_K_linear": peak_lin_clipped,
                "peak_K_log": peak_log_K_clipped,
                "ys": ys,
                "yhat_lin": yhat_lin.tolist(),
                "yhat_log": yhat_log.tolist(),
            })

    row_lookup = {(r["pos"], r["neg"], r["source"]): r for r in rows}

    print(f"{'axis':37s} {'source':11s}  {'R²_lin':>7s}  {'R²_log':>7s}  "
          f"{'peak_lin':>9s}  {'peak_log':>9s}")
    print("-" * 87)
    for r in rows:
        print(f"{r['pos']+' vs '+r['neg']:37s} {r['source']:11s}  "
              f"{r['r2_linear']:>7.3f}  {r['r2_log']:>7.3f}  "
              f"{r['peak_K_linear']:>9.2f}  {r['peak_K_log']:>9.2f}")

    r2_lin_vals = [r["r2_linear"] for r in rows if np.isfinite(r["r2_linear"])]
    r2_log_vals = [r["r2_log"] for r in rows if np.isfinite(r["r2_log"])]
    mean_r2_lin = float(np.mean(r2_lin_vals)) if r2_lin_vals else float("nan")
    mean_r2_log = float(np.mean(r2_log_vals)) if r2_log_vals else float("nan")
    print(f"\nMean R² across {len(rows)} curves:   "
          f"linear-K = {mean_r2_lin:.3f}   log2(K+1) = {mean_r2_log:.3f}")
    n_compare = sum(1 for r in rows
                    if np.isfinite(r["r2_log"]) and np.isfinite(r["r2_linear"]))
    n_log_better = sum(1 for r in rows if np.isfinite(r["r2_log"])
                       and np.isfinite(r["r2_linear"])
                       and r["r2_log"] > r["r2_linear"])
    if n_compare > 0:
        print(f"log parametrization fits better than linear on "
              f"{n_log_better}/{n_compare} curves")

    def peaks_agreement(rows, key) -> list[tuple[tuple[str, str], float, float]]:
        aligned: dict[tuple[str, str], dict] = {}
        for r in rows:
            k = (r["pos"], r["neg"])
            aligned.setdefault(k, {})[r["source"]] = r[key]
        out: list[tuple[tuple[str, str], float, float]] = []
        for k, dmap in aligned.items():
            if "desc_inst" not in dmap or "responses" not in dmap:
                continue
            dv, rv = dmap["desc_inst"], dmap["responses"]
            if np.isfinite(dv) and np.isfinite(rv):
                out.append((k, float(dv), float(rv)))
        return out

    for key, label in [("peak_K_linear", "linear-K"),
                       ("peak_K_log",    "log2(K+1)")]:
        triples = peaks_agreement(rows, key)
        if len(triples) >= 3:
            xd = np.array([t[1] for t in triples])
            xr = np.array([t[2] for t in triples])
            r_pear = float(pearsonr(xd, xr).statistic)
            r_spear = float(spearmanr(xd, xr).correlation)
            print(f"\nPeak-K agreement desc+inst vs responses ({label}): "
                  f"Pearson ρ = {r_pear:.3f}, Spearman ρ = {r_spear:.3f}  "
                  f"(n={len(triples)})")
            print("  axis                                 desc+inst peak   "
                  "responses peak")
            for (k1, k2), d, r in triples:
                print(f"    {k1+' vs '+k2:33s}  {d:>14.2f}   {r:>14.2f}")

    # ``inputs`` was already populated above by the load_and_register
    # calls.  The K-sweep envelope's own inputs (vectors, judge caches,
    # etc.) are tracked transitively via the sweep_json file
    # fingerprint -- if any of those change, sweep_json's mtime/size
    # changes too, and the readers of *this* fit_json see drift on the
    # sweep_json dep.

    fit_envelope = json_metadata(
        rows, inputs=inputs,
        title=f"whitening_k_peak_fit slot={args.slot} pairs={args.pairs}")
    with open(experiment_dir / fit_f, "w") as _f:
        json.dump(fit_envelope, _f, indent=2, default=str)

    # ---- Per-axis panels with observed data and log-space fits ----
    n_axes = len(pair_keys_ordered)
    n_cols = GRID_N_COLS
    n_rows = int(np.ceil(n_axes / n_cols))
    fs_title = max(7, 11 - max(0, n_axes - 12) // 6)
    fs_tick = max(7, 9 - max(0, n_axes - 12) // 8)
    fs_legend = max(5, fs_tick - 1)

    fig, axes_p = plt.subplots(n_rows, n_cols,
                               figsize=(4 * n_cols, 3.35 * n_rows),
                               sharey=True, squeeze=False)

    x_plot_dense = np.linspace(np.log2(0 + 1), np.log2(32 + 1), 200)
    x_plot_orig = np.array([np.log2(K + 1) for K in FIT_K])

    any_point = False
    for i, (pos, neg) in enumerate(pair_keys_ordered):
        ax = axes_p[i // n_cols, i % n_cols]
        panel_has_curve = False
        for source, marker, linestyle, color in [
            ("desc_inst", "s", ":", "C0"),
            ("responses", "o", "-", "C3"),
        ]:
            row = row_lookup.get((pos, neg, source))
            if row is None:
                continue
            ys = np.asarray(row["ys"], dtype=float)
            mask = np.isfinite(ys)
            if mask.sum() < 3:
                continue
            coef = np.polyfit(x_plot_orig[mask], ys[mask], 2)
            yhat_dense = np.polyval(coef, x_plot_dense)

            lab_r2 = row["r2_log"]
            r2_lab = lab_r2 if np.isfinite(lab_r2) else float("nan")
            ax.plot(x_plot_orig, ys, marker=marker, color=color,
                    linestyle="", markersize=max(4, fs_tick),
                    label=f'{source} (R²={r2_lab:.2f})')
            ax.plot(x_plot_dense, yhat_dense, color=color,
                    linestyle=linestyle, alpha=0.7)
            peak_K = row["peak_K_log"]
            if np.isfinite(peak_K) and 0 <= peak_K <= 32:
                ax.axvline(np.log2(peak_K + 1), color=color,
                           linestyle="--", alpha=0.35, lw=1)
            panel_has_curve = True
            any_point = True

        if not panel_has_curve:
            ax.text(
                0.5, 0.52, "no ρ data\n(desc+instr / responses overlap)",
                transform=ax.transAxes, fontsize=fs_tick + 1, ha="center",
                va="center", clip_on=False)

        ax.set_xticks([np.log2(K + 1) for K in FIT_K])
        ax.set_xticklabels([str(K) for K in FIT_K], fontsize=fs_tick)
        ax.set_title(f"{pos} vs {neg}", fontsize=fs_title)
        ax.grid(alpha=0.3)
        ax.set_xlabel("K (log2(K+1) spacing)", fontsize=fs_tick)
        if i % n_cols == 0:
            ax.set_ylabel("Spearman ρ", fontsize=fs_tick)
        if panel_has_curve:
            ax.legend(fontsize=fs_legend, loc="lower left")
    for j in range(n_axes, n_rows * n_cols):
        axes_p[j // n_cols, j % n_cols].axis("off")

    title_line = (f"Per-axis parabolic fits over K ∈ {FIT_K[0]}..{FIT_K[-1]} "
                  f"(log2(K+1) space) — {n_axes} axes")
    if args.slot is not None:
        title_line += f"; token slot {args.slot}"

    spec_line = (
        "Solid = responses where available (NaN omitted); dotted = desc+instr; "
        "vertical dashed = fitted peak K (log parametrisation)."
    )
    if not any_point:
        print("Warning: no finite ρ points plotted (check sweep vs pair list).")
    _, top_rect = suptitle_with_specs(fig, title_line, spec_line)
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    out_path = experiment_dir / plot_f
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)
    print(f"\nWrote {experiment_dir / fit_f}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
