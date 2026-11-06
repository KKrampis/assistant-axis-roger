#!/usr/bin/env python3
"""Sweep the (response × desc+inst) blend weight to find the optimal
mix for the v2-rubric era.

Companion to :mod:`results_analysis.gpt_anthropic_response_weight_sweep`
(which sweeps the GPT/Anthropic weight inside the response ensemble)
and :mod:`results_analysis.gpt_sonnet_weight_sweep` (which sweeps the
GPT/Sonnet weight inside the desc+inst ensemble).  Where those two
fix one ingredient and sweep the other within a single mode, this
script fixes BOTH within-mode ensembles at their established
operating points and sweeps the weight across modes::

    score(w)  =  w · response_ensemble  +  (1 − w) · desc_inst_ensemble

with::

    response_ensemble  =  0.6 · GPT_b10  +  0.4 · Haiku_q9       (v2 caches)
    desc_inst_ensemble =  combine_desc_inst_two_judges(
                              gpt_d, gpt_i, sonnet_d, sonnet_i,
                              weights = inst-tiebreak)

Per-axis Spearman ρ vs the raw activation projection at
``(--slot, --layer)`` (default 6, 25 — the operating-point cell).
A parabolic fit on the interior ``w ∈ [0.0, 0.9]`` finds the peak
without being skewed by the w=1 endpoint kink (where desc+inst
drops out of the blend entirely).

Outputs (default into ``--experiment_dir``):

* ``response_di_weight_sweep_slot{N}.png`` -- per-axis lines + mean + parabola
* ``response_di_weight_sweep_slot{N}.json`` -- full per-axis ρ table,
  parabola coefficients, peak w, peak ρ, plus full provenance envelope.
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

from assistant_axis import entity_id, json_metadata, png_metadata
from assistant_axis.judge_loaders import migrate_v1_static_scores
from assistant_axis.judge_score_combine import (
    declare_constants_dependency,
    DEFAULT_DI_WEIGHTS,
    DEFAULT_GPT_HAIKU_Q9_WEIGHT,
    combine_desc_inst_two_judges,
)
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
    current_file_input,
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
from results_analysis.gpt_anthropic_response_weight_sweep import (
    AXIS_COLORS, DEFAULT_AXES, GPT_DIR_TEMPLATE, PARABOLA_COLOR,
)

_SCRIPT_PATH = Path(__file__).resolve()


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
    "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
)
DEFAULT_SLOT, DEFAULT_LAYER = 6, 25

HAIKU_DIR_TEMPLATE = "haiku_responses_{side}_b10_q9"

# Within-response ensemble weight on GPT-4.1-mini B=10 (the rest goes
# on Haiku-q9).  Imported from assistant_axis.judge_score_combine for
# a single source of truth; that module's docstring carries the
# empirical derivation.  Kept as a named local so the JSON / PNG
# provenance records reflect the value actually used at run-time.
GPT_RESPONSE_WEIGHT = DEFAULT_GPT_HAIKU_Q9_WEIGHT
HAIKU_RESPONSE_WEIGHT = 1.0 - GPT_RESPONSE_WEIGHT

# Default sweep grid: w from 0 to 1 in 0.025 steps (41 points).
# Matches the GPT/Sonnet and GPT/Anthropic weight sweeps for visual
# consistency.  Override via --w_step.
DEFAULT_W_STEP = 0.025

# Parabolic fit domain (per user spec): [0.0, 0.9], excluding the
# w=1 endpoint where desc+inst drops out of the blend (categorical
# regime change, not a smooth blend; mirrors the [0.1, 0.9] interior
# fit choice in gpt_anthropic_response_weight_sweep.py).
FIT_W_LO, FIT_W_HI = 0.0, 0.9


# ---------------------------------------------------------------------------
# Helpers (mirror gpt_anthropic_response_weight_sweep where possible)
# ---------------------------------------------------------------------------

def _v(path: Path, slot: int, layer: int) -> torch.Tensor:
    return _load_vector_file(path).float()[slot, layer]


def _axis_unit(data_dir: Path, pos: str, neg: str,
               slot: int, layer: int) -> torch.Tensor:
    p = _v(data_dir / "traits" / "vectors" / f"{pos}.pt", slot, layer)
    n = _v(data_dir / "traits" / "vectors" / f"{neg}.pt", slot, layer)
    d = p - n
    nrm = torch.linalg.vector_norm(d)
    return d / nrm if nrm > 0 else d


def _load_response_scores(
    experiment_dir: Path, axis: str,
    _dir_template_legacy: str,   # noqa: ARG001  retained for caller back-compat
    *,
    judge_label: str,
    inputs: list[InputSpec],
) -> dict[str, float]:
    """Per-entity ``mean_score`` across roles+traits sides for one
    (judge, axis) pair.

    Reads via :func:`assistant_axis.judge_loaders.load_response_scores`
    so the canonical B=7 → B=10 per-entity fallback works for axes
    that don't yet have B=7 data on every entity (e.g. the original
    12 axes' bulk lives in ``_b10_q9`` while only collision
    disambiguation entities have ``_b7_t3``).  The third positional
    arg ``_dir_template_legacy`` is retained so existing call sites
    don't break, but it's ignored — the canonical loader picks the
    cohort itself.  2026-05-22 retrofit: previously this function
    hardcoded ``haiku_responses_{side}_b10_q9`` which silently
    skipped the 10 Phase-1/2 axes that only have ``_b7_t3``.
    """
    from assistant_axis.judge_loaders import load_response_scores

    # Normalise the judge label that the rest of the script uses
    # (``gpt_b10`` / ``haiku_q9``) to the loader's ``judge`` arg.
    judge = "gpt" if judge_label.startswith("gpt") else "haiku"
    out: dict[str, float] = {}
    for side in ("roles", "traits"):
        scores, _sources = load_response_scores(
            axis=axis, kind=side, judge=judge, rubric="v2",
            experiments_root=experiment_dir, inputs=inputs,
        )
        for name, info in scores.items():
            ms = info.get("mean_score") if isinstance(info, dict) else None
            if ms is not None:
                out[entity_id(name, side)] = float(ms)
    return out


def _rho_at_w(d: dict, w: float) -> float:
    """Spearman ρ of (w·response + (1-w)·di) vs the projection."""
    score = w * d["response"] + (1.0 - w) * d["di"]
    return float(spearmanr(score, d["proj"]).correlation)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR))
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--plot", default=None,
                   help="Output PNG (default: response_di_weight_sweep_"
                        "slot{N}.png inside --experiment_dir).")
    p.add_argument("--rhos_json", default=None)
    p.add_argument(
        "--exclude_axes", default="", metavar="AXIS[,AXIS...]",
        help="Comma-separated axis names (e.g. "
             "'ecocentric_vs_anthropocentric') to drop from the "
             "sweep.  Useful when one axis is an outlier dragging "
             "the optimum (eco/anthro is the canonical example -- "
             "Haiku-q9 has ρ ≈ 0.11 there on desc+inst, far below "
             "every other axis, so its presence pulls the parabolic "
             "peak toward 100%% response).  When non-empty, an "
             "auto-suffix like '_no_eco_anthro' is appended to the "
             "default output filenames so the canonical run is not "
             "clobbered.")
    p.add_argument("--whitening", default=DEFAULT_WHITENING_SPEC,
                   help=f"Whitening regime applied to both entity vectors "
                        f"and the axis direction before computing ρ.  "
                        f"Forms: 'raw', 'soft_K=N', 'lw', 'oas', "
                        f"'soft_shear=L'.  Default: "
                        f"{DEFAULT_WHITENING_SPEC!r} (project canonical).")
    p.add_argument("--ca_kind", default="combined",
                   choices=("combined", "traits", "roles"),
                   help="Goal/no-goal subspaces for soft-shear fitting. "
                        "Ignored for non-shear regimes.")
    p.add_argument("--w_step", type=float, default=DEFAULT_W_STEP,
                   help=f"Step size on the w grid (default: "
                        f"{DEFAULT_W_STEP} -> 41 points on [0, 1]).  "
                        f"Pass --w_step 0.05 for the historical "
                        f"coarser 21-point sweep.")
    args = p.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    slot, layer = int(args.slot), int(args.layer)
    excluded = {a.strip() for a in args.exclude_axes.split(",") if a.strip()}
    if excluded:
        # Compact suffix: first 3 chars of each excluded pos pole,
        # joined by underscores.  E.g. "ecocentric_vs_anthropocentric"
        # → "eco_anthro" via the pos/neg split below.
        def _short(name: str) -> str:
            pos, neg = name.split("_vs_", 1)
            return f"{pos[:3]}_{neg[:6]}"
        exc_suffix = "_no_" + "__".join(sorted(_short(a) for a in excluded))
    else:
        exc_suffix = ""

    wh_method, wh_n = parse_whitening_spec(args.whitening)
    # Filename tag: 'raw' is the historical default and gets no suffix
    # so the v1-archive and the new raw view share a slot.  Any other
    # regime appends a short filename-safe suffix.
    if wh_method == "raw":
        wh_suffix = ""
    elif wh_n is not None:
        wh_suffix = f"_{wh_method.replace('_', '')}{wh_n}"
    else:
        wh_suffix = f"_{wh_method.replace('_', '')}"

    out_stem = f"response_di_weight_sweep_slot{slot}{wh_suffix}{exc_suffix}"
    if args.plot is None:
        args.plot = f"{out_stem}.png"
    if args.rhos_json is None:
        args.rhos_json = f"{out_stem}.json"

    # w grid (mirrors gpt_sonnet / gpt_anthropic for consistency).
    if args.w_step <= 0 or args.w_step > 0.5:
        raise SystemExit(
            f"--w_step must be in (0, 0.5], got {args.w_step}")
    n_w = int(round(1.0 / args.w_step)) + 1
    if abs(n_w - 1 - 1.0 / args.w_step) > 1e-9:
        raise SystemExit(
            f"--w_step={args.w_step} doesn't divide 1.0 evenly; "
            f"pick a divisor of 1.0 (e.g. 0.025, 0.05, 0.02, 0.01).")
    i_half = int(round(0.5 / args.w_step))

    # Provenance accumulator (threaded through every cache read).
    inputs: list[InputSpec] = [
        current_file_input(
            dep_key="producer_script",
            path=_SCRIPT_PATH,
            extras={
                "slot": str(slot), "layer": str(layer),
                "gpt_response_weight": str(GPT_RESPONSE_WEIGHT),
                "fit_domain": f"[{FIT_W_LO}, {FIT_W_HI}]",
                "excluded_axes": ",".join(sorted(excluded)) or "none",
                "whitening": args.whitening,
                "ca_kind": (args.ca_kind
                            if wh_method == "soft_shear" else ""),
            },
        ),]
    # Constants-file dependency: GPT_RESPONSE_WEIGHT is read from
    # ``DEFAULT_GPT_HAIKU_Q9_WEIGHT`` at import time, so the cache
    # this script writes is sensitive to retunings of that (and any
    # other) project-wide default in judge_score_combine.py.  See
    # the helper's docstring for the lazy-vs-accurate trade-off
    # discussion.  Without this entry, audit_caches.py would NOT
    # flag this script's outputs as stale after a default retune --
    # which is the exact bug that bit us on the 2026-05-11 0.60→0.41
    # GPT/Haiku retune.
    declare_constants_dependency(inputs)
    inputs.extend([
        current_data_subtree_input(
            data_dir=data_dir, subtree_rel="traits/vectors",
            dep_key="traits_vectors"),
        current_data_subtree_input(
            data_dir=data_dir, subtree_rel="roles/vectors",
            dep_key="roles_vectors"),
    ])

    # Pre-cache standalone entity vectors at (slot, layer), default-centered.
    # Build kinds_for_name alongside so v1 (bare-name) caches -- e.g.
    # the deferred Sonnet desc/inst caches that weren't included in
    # Phase 5b/5c.1 -- can be migrated up to v2 in-memory before
    # intersecting with the post-Phase-5 v2 GPT/Haiku caches.  See
    # assistant_axis.judge_loaders.migrate_v1_static_scores.
    default_v = _v(data_dir / "traits" / "vectors" / "default.pt", slot, layer)
    entity_vecs: dict[str, np.ndarray] = {}
    kinds_for_name: dict[str, set] = {}
    for et in ("traits", "roles"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            try:
                v = _load_vector_file(fp).float()[slot, layer]
                entity_vecs[entity_id(fp.stem, et)] = (v - default_v).numpy()
                kinds_for_name.setdefault(fp.stem, set()).add(et)
            except Exception:  # pragma: no cover -- skip unreadable .pt
                continue

    # Whitening basis -- fit once, apply once to the entity pool, then
    # whiten each per-axis direction in-place inside the loop.  Linear,
    # so applying pre-projection is equivalent to applying inside the
    # dot product.
    basis: WhiteningBasis | None = None
    if wh_method == "soft_shear":
        A_g, A_n = build_goal_nogoal_subspaces(
            data_dir, slot=slot, layer=layer, kind=args.ca_kind)
        n_pairs_max = min(A_g.shape[1], A_n.shape[1])
        if wh_n is not None and wh_n > n_pairs_max:
            raise SystemExit(
                f"--whitening soft_shear={wh_n} exceeds CA-pair budget "
                f"for kind={args.ca_kind!r} (min(n_g,n_n)={n_pairs_max})."
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
        for n, v in entity_vecs.items():
            entity_vecs[n] = basis.apply(v[None, :])[0]

    axes_in_use = [a for a in DEFAULT_AXES if a[0] not in excluded]
    if excluded:
        unknown = excluded - {a[0] for a in DEFAULT_AXES}
        if unknown:
            print(f"  [warn] --exclude_axes named axes that aren't in "
                  f"DEFAULT_AXES: {sorted(unknown)} (typo?  ignored)")
        print(f"Response × desc+inst weight sweep @ slot {slot} "
              f"layer {layer}  (excluding {sorted(excluded)})")
    else:
        print(f"Response × desc+inst weight sweep @ slot {slot} "
              f"layer {layer}")
    print(f"  axes: {[a[0] for a in axes_in_use]}")
    print(f"  response ensemble: "
          f"{GPT_RESPONSE_WEIGHT:.2f}·GPT_b10 + "
          f"{HAIKU_RESPONSE_WEIGHT:.2f}·Haiku_q9   (v2 canonical caches)")
    print("  desc+inst ensemble: GPT+Sonnet 4-way, inst-tiebreak")
    print(f"  whitening: {args.whitening}")
    print(f"  sweep: w from 0 to 1 in {args.w_step} steps "
          f"({n_w} points)")
    print(f"  parabola fit on w ∈ [{FIT_W_LO}, {FIT_W_HI}]\n")

    # Per-axis precompute: (response_ensemble, di_ensemble, projection)
    # arrays over the common entity set.  These do not depend on w,
    # so we compute them once and re-use across the 21 sweep points.
    per_axis: dict[tuple[str, str], dict] = {}
    for axis_name, pos, neg in axes_in_use:
        axis_dir = experiment_dir / axis_name

        gpt_resp = _load_response_scores(
            experiment_dir, axis_name, GPT_DIR_TEMPLATE,
            judge_label="gpt_b10", inputs=inputs)
        haiku_resp = _load_response_scores(
            experiment_dir, axis_name, HAIKU_DIR_TEMPLATE,
            judge_label="haiku_q9", inputs=inputs)
        if not gpt_resp or not haiku_resp:
            print(f"  [skip] {axis_name}: response data missing "
                  f"(gpt n={len(gpt_resp)}, haiku n={len(haiku_resp)})")
            continue

        g_d, _, _ = load_and_register(
            axis_dir / "gpt" / "scores_descriptions.json",
            dep_key=f"di_gpt_d_{axis_name}",
            inputs=inputs, policy="warn")
        g_i, _, _ = load_and_register(
            axis_dir / "gpt" / "scores_instructions.json",
            dep_key=f"di_gpt_i_{axis_name}",
            inputs=inputs, policy="warn")
        s_d, _, _ = load_and_register(
            axis_dir / "sonnet" / "scores_descriptions.json",
            dep_key=f"di_sonnet_d_{axis_name}",
            inputs=inputs, policy="warn")
        s_i, _, _ = load_and_register(
            axis_dir / "sonnet" / "scores_instructions.json",
            dep_key=f"di_sonnet_i_{axis_name}",
            inputs=inputs, policy="warn")
        # Migrate Sonnet's v1 (bare-name) keys to v2 (entity_id) form
        # before combining; GPT caches are already v2 post-Phase-5.
        # Without migration the 4-way intersection is empty (v1 vs v2
        # keys never overlap).  ``migrate_v1_static_scores`` is a no-op
        # on already-v2 input, so applying it uniformly is safe.
        s_d = migrate_v1_static_scores(s_d, kinds_for_name)
        s_i = migrate_v1_static_scores(s_i, kinds_for_name)
        di = combine_desc_inst_two_judges(
            g_d, g_i, s_d, s_i, weights=DEFAULT_DI_WEIGHTS)

        # Within-response ensemble (fixed at the 0.6/0.4 operating
        # point; this sweep is across modes, not within response).
        common_resp = set(gpt_resp) & set(haiku_resp)
        response = {
            n: GPT_RESPONSE_WEIGHT * gpt_resp[n]
               + HAIKU_RESPONSE_WEIGHT * haiku_resp[n]
            for n in common_resp
        }

        common = sorted(set(response) & set(di) & set(entity_vecs))
        if len(common) < 5:
            print(f"  [skip] {axis_name}: only {len(common)} entities "
                  f"shared across response, desc+inst, and vectors")
            continue
        resp_arr = np.array([response[n] for n in common])
        di_arr = np.array([di[n] for n in common])
        a = _axis_unit(data_dir, pos, neg, slot, layer).numpy()
        if basis is not None and basis.method != "raw":
            a = basis.apply(a[None, :])[0]
        proj = np.array([float(np.dot(entity_vecs[n], a)) for n in common])
        per_axis[(pos, neg)] = {
            "axis_name": axis_name,
            "response": resp_arr, "di": di_arr, "proj": proj,
            "n": len(common),
        }
        print(f"  {axis_name:<32}  n={len(common):3d}  "
              f"di-only ρ={_rho_at_w(per_axis[(pos, neg)], 0.0):+.4f}  "
              f"response-only ρ={_rho_at_w(per_axis[(pos, neg)], 1.0):+.4f}")

    if not per_axis:
        print("No axes had complete data; exiting.")
        return 1

    # Sort axes by interior slope Δρ = ρ(w=0.9) − ρ(w=0.1) so colours
    # in the legend run di-favored → response-favored, mirroring the
    # convention in the within-response sweep.
    slopes = {k: _rho_at_w(d, 0.9) - _rho_at_w(d, 0.1)
              for k, d in per_axis.items()}
    sorted_keys = sorted(per_axis.keys(), key=lambda k: slopes[k])
    per_axis = {k: per_axis[k] for k in sorted_keys}

    ws = np.linspace(0.0, 1.0, n_w)
    all_rhos = np.array([
        [_rho_at_w(d, w) for d in per_axis.values()]
        for w in ws
    ])
    mean_rhos = all_rhos.mean(axis=1)

    # Parabolic fit on [FIT_W_LO, FIT_W_HI] (default [0.0, 0.9]).
    mask = (ws >= FIT_W_LO - 1e-9) & (ws <= FIT_W_HI + 1e-9)
    ws_fit = ws[mask]
    mean_fit = mean_rhos[mask]
    c, b, a = np.polyfit(ws_fit, mean_fit, 2)
    yhat = a + b * ws_fit + c * ws_fit ** 2
    ss_res = float(((mean_fit - yhat) ** 2).sum())
    ss_tot = float(((mean_fit - mean_fit.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if c < 0:
        w_peak = float(-b / (2 * c))
        rho_peak = float(a + b * w_peak + c * w_peak ** 2)
    else:
        # Concave-up: peak is at one of the fit endpoints.
        i_best = int(np.argmax(mean_fit))
        w_peak = float(ws_fit[i_best])
        rho_peak = float(mean_fit[i_best])

    print(f"\nSweep summary ({len(per_axis)} axes, "
          f"whitening={args.whitening}, w_step={args.w_step}):")
    print(f"  pure desc+inst (w=0):      mean ρ = {mean_rhos[0]:+.4f}")
    print(f"  50/50 mix    (w=0.5):    mean ρ = {mean_rhos[i_half]:+.4f}")
    print(f"  pure response  (w=1):      mean ρ = {mean_rhos[-1]:+.4f}")
    i_disc = int(np.argmax(mean_rhos))
    print(f"  discrete grid peak:        w={ws[i_disc]:.3f}, "
          f"mean ρ = {mean_rhos[i_disc]:+.4f}")
    print(f"  parabolic peak (fit on [{FIT_W_LO}, {FIT_W_HI}], R²={r2:.3f}): "
          f"w={w_peak:.3f}, ρ ≈ {rho_peak:+.4f}")

    # ---- Plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    n_axes = len(per_axis)
    keys = list(per_axis.keys())
    fallback = plt.cm.plasma(np.linspace(0.05, 0.80, max(n_axes, 2)))
    colors = [
        AXIS_COLORS.get(per_axis[k]["axis_name"], fallback[i])
        for i, k in enumerate(keys)
    ]

    for j, k in enumerate(keys):
        ax.plot(ws, all_rhos[:, j], color=colors[j], lw=1.2, alpha=0.85)

    mean_handle, = ax.plot(
        ws, mean_rhos, color="black", lw=2.5, marker="o", markersize=5,
        zorder=20, label=f"mean ({n_axes} axes)")

    # Parabola fit + extrapolation outside the fit domain.
    ws_in = np.linspace(FIT_W_LO, FIT_W_HI, 50)
    ws_out = np.linspace(0.0, 1.0, 100)
    y_in = a + b * ws_in + c * ws_in ** 2
    y_out = a + b * ws_out + c * ws_out ** 2
    extrap_handle, = ax.plot(
        ws_out, y_out, color="#444444", lw=1.0, ls=":",
        alpha=0.9, zorder=25, label="parabola (extrapolated)")
    fit_handle, = ax.plot(
        ws_in, y_in, color=PARABOLA_COLOR, lw=2.0, ls="-",
        alpha=0.95, zorder=26,
        label=f"parabola fit on [{FIT_W_LO}, {FIT_W_HI}] "
              f"(R²={r2:.3f})")
    w50_handle = ax.axvline(0.5, color="grey", linestyle=":", lw=1,
                              alpha=0.7, label="w=0.5")
    peak_handle = ax.axvline(
        w_peak, color=PARABOLA_COLOR, linestyle="--", lw=1.4,
        label=f"interior peak w={w_peak:.3f}")

    ax.set_xlabel(
        f"Weight on response ensemble  "
        f"({GPT_RESPONSE_WEIGHT:.3f}·GPT_b10 + "
        f"{HAIKU_RESPONSE_WEIGHT:.3f}·Haiku_q9, v2)\n"
        f"(remaining 1−w on desc+inst, GPT+Sonnet 4-way inst-tiebreak)",
        fontsize=10,
    )
    ax.set_ylabel(f"Per-axis Spearman ρ "
                  f"(slot {slot}, layer {layer}, "
                  f"whitening={args.whitening})", fontsize=10)
    excl_tag = (
        f"  (excluding {', '.join(sorted(excluded))})"
        if excluded else ""
    )
    title = (f"Response × desc+inst score-blend sweep — "
             f"mean ρ across {n_axes} axes{excl_tag}")
    ax.set_title(
        title + "\n"
        f"DI={mean_rhos[0]:.3f}  "
        f"50/50={mean_rhos[i_half]:.3f}  "
        f"R={mean_rhos[-1]:.3f}  →  peak w={w_peak:.3f} (ρ={rho_peak:.3f})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.02, 1.02)
    y_hi = float(min(1.0, all_rhos.max() + 0.02))
    y_lo = float(max(0.0, all_rhos.min() - 0.02))
    ax.set_ylim(y_lo, y_hi)
    ax.grid(alpha=0.3)

    # Axis legend (sidebar) sorted di-favored to response-favored, with
    # interior-slope Δρ = ρ(0.9) − ρ(0.1) shown next to each axis name.
    axis_handles = [
        plt.Line2D([], [], color=colors[i], lw=1.2, alpha=0.85)
        for i in range(n_axes)
    ]
    axis_labels = [
        f"{per_axis[k]['axis_name']} (Δ{slopes[k]:+.3f})"
        for k in keys
    ]
    leg_axes = ax.legend(
        axis_handles, axis_labels,
        loc="upper left", bbox_to_anchor=(1.01, 0.98),
        fontsize=9, ncol=1, frameon=True, framealpha=0.9,
        handlelength=1.5, borderpad=0.6, labelspacing=0.32)
    ax.add_artist(leg_axes)
    ax.annotate(
        "Axes: sorted di-best → response-best",
        xy=(1.01, 1.005), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(
        [mean_handle, fit_handle, extrap_handle, w50_handle, peak_handle],
        [mean_handle.get_label(), fit_handle.get_label(),
         extrap_handle.get_label(), w50_handle.get_label(),
         peak_handle.get_label()],
        loc="lower left", bbox_to_anchor=(1.01, 0.0),
        fontsize=8, frameon=True, framealpha=0.9, handlelength=1.5)
    plt.tight_layout()

    out_path = experiment_dir / args.plot
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # ---- JSON ------------------------------------------------------------
    json_payload = {
        "n_axes": n_axes,
        "excluded_axes": sorted(excluded),
        "left_endpoint_label": "desc+inst (GPT+Sonnet 4-way, inst-tiebreak)",
        "right_endpoint_label": (
            f"response ({GPT_RESPONSE_WEIGHT:.3f}·GPT_b10 + "
            f"{HAIKU_RESPONSE_WEIGHT:.3f}·Haiku_q9, v2)"
        ),
        "slot": slot, "layer": layer,
        "whitening": args.whitening,
        "ca_kind": args.ca_kind if wh_method == "soft_shear" else None,
        "w_step": float(args.w_step),
        "ws": [float(w) for w in ws],
        "mean_rho_by_w": [float(v) for v in mean_rhos],
        "parabola_fit": {
            "domain": [FIT_W_LO, FIT_W_HI],
            "a": float(a), "b": float(b), "c": float(c),
            "r2": float(r2),
            "peak_w": float(w_peak), "peak_rho": float(rho_peak),
        },
        "per_axis": [
            {"axis_name": per_axis[k]["axis_name"],
             "pos": k[0], "neg": k[1],
             "interior_slope_delta_rho": float(slopes[k]),
             "rho_by_w": [float(v) for v in all_rhos[:, i]],
             "n": int(per_axis[k]["n"])}
            for i, k in enumerate(keys)
        ],
    }
    json_out = json_metadata(
        json_payload, title=title, inputs=inputs,
    )
    json_path = experiment_dir / args.rhos_json
    json_path.write_text(json.dumps(json_out, indent=2))
    print(f"Wrote {json_path}  ({len(inputs)} inputs recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
