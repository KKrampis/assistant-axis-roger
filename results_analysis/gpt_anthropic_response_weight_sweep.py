#!/usr/bin/env python3
"""GPT-vs-Anthropic *response*-mode weight sweep, analogous to
:mod:`results_analysis.gpt_sonnet_weight_sweep` for desc+inst judging.

For each axis we load the per-entity ``mean_score`` from
``<experiment_dir>/<axis>/<judge>_responses_{roles,traits}<suffix>/scores_responses.json``
for both judges (combined roles + traits sides), then sweep::

    score(w) = w · gpt_b10[name] + (1 − w) · anthropic[name]

over ``w ∈ [0, 1]``.  Per-axis Spearman ρ vs the raw activation
projection at ``(--slot, --layer)``.

Anthropic combo selectors (``--anthropic_combo``):
- ``haiku_q9``    → ``haiku_responses_{side}_b10_q9``    (1/3 question subsample)
- ``haiku_full``  → ``haiku_responses_{side}_b10``       (all questions)
- ``sonnet_q9``   → ``sonnet_responses_{side}_b10_q9``   (1/3 question subsample)

Outputs ``gpt_<combo>_response_weight_sweep_slot{N}.png`` + ``.json``
inside ``--experiment_dir``.

CLI
---

::

    # One sweep per Anthropic candidate (3 plots from 3 invocations).
    for combo in haiku_q9 haiku_full sonnet_q9; do
        uv run python -m results_analysis.gpt_anthropic_response_weight_sweep \\
            --anthropic_combo "$combo"
    done
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
from assistant_axis.judge_score_combine import declare_constants_dependency
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

_SCRIPT_PATH = Path(__file__).resolve()


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)


def _load_axes_from_pair_list(
    pair_list_path: Path,
) -> list[tuple[str, str, str]]:
    """Load axes from a pair_list JSON file.

    Reads e.g. ``pair_list_responses.json`` and returns the canonical
    ``[(axis_name, pos, neg), ...]`` tuple list used by the sweep
    loaders.  See :data:`DEFAULT_AXES` for the canonical 12 → 22
    expansion history (2026-05-22: 10 new axes added once their
    response judging completed at the canonical B=7 t3 cohort).
    """
    with open(pair_list_path, encoding="utf-8") as f:
        pairs = json.load(f)
    return [
        (f"{p['pos']}_vs_{p['neg']}", p["pos"], p["neg"])
        for p in pairs
    ]


# Axes available for response judging.  Loaded from
# ``pair_list_responses.json`` so future axis additions propagate
# automatically.  As of 2026-05-22 the set is 22 axes (was 12 prior
# to Phases 1+2 response-judging campaign): the original HHH-adjacent
# 12 plus 10 personality/style axes (proactive, descriptive,
# religious, fragile, introverted, benign, accessible, precise,
# blunt, earnest).  Override with --pair_list <path> for ablations.
DEFAULT_AXES: list[tuple[str, str, str]] = _load_axes_from_pair_list(
    DEFAULT_EXPERIMENT_DIR / "pair_list_responses.json"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
    "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
)
DEFAULT_SLOT, DEFAULT_LAYER = 6, 25  # Match gpt_sonnet_weight_sweep.py default.

# Combo selector → (label, anthropic-dir-suffix-template).
COMBO_DIRS: dict[str, tuple[str, str]] = {
    "haiku_q9":   ("Haiku-q9",   "haiku_responses_{side}_b10_q9"),
    "haiku_full": ("Haiku-full", "haiku_responses_{side}_b10"),
    "sonnet_q9":  ("Sonnet-q9",  "sonnet_responses_{side}_b10_q9"),
}

# Per-axis colours (consistent across the 3 weight-sweep plots so that a
# given axis keeps the same colour regardless of how the slopes happen to
# sort within a plot).  12 independent colours -- no set-membership
# grouping (each axis stands alone visually).  Picked to stay distinct
# from the mean line (black), the w=0.5 reference line (grey), and the
# parabola fit colour (PARABOLA_COLOR below).  Falls back to the plasma
# colormap for axes not listed here.
AXIS_COLORS: dict[str, str] = {
    "truthful_vs_deceitful":         "#4c72b0",  # blue
    "progressive_vs_conservative":   "#55a868",  # green
    "improvisational_vs_methodical": "#c44e52",  # red
    "concise_vs_verbose":            "#8172b2",  # purple
    "ecocentric_vs_anthropocentric": "#dd8452",  # orange
    "egalitarian_vs_elitist":        "#937860",  # warm brown
    "guileless_vs_scheming":         "#da8bc3",  # pink
    "harmless_vs_harmful":           "#17becf",  # cyan
    "honest_vs_dishonest":           "#bcbd22",  # mustard
    "helpful_vs_unhelpful":          "#2c5d8c",  # navy
    "relativist_vs_absolutist":      "#b04a72",  # rose
    "systems_thinker_vs_analytical": "#6b4ec1",  # violet
}

GPT_DIR_TEMPLATE = "gpt_responses_{side}_b10"
PARABOLA_COLOR = "#1faa4f"


def _load_response_scores(
    experiment_dir: Path, axis: str, dir_template: str,
    *,
    scores_filename: str = "scores_responses.json",
    judge_label: str,
    inputs: list[InputSpec] | None = None,
) -> dict[str, float]:
    """Per-entity mean response-judge score, summed across roles + traits.

    **v1-mode loader** (legacy path): reads a single hardcoded cohort
    directory per (axis, side).  See :func:`_load_response_scores_v2`
    for the v2-mode equivalent that uses the canonical
    :func:`assistant_axis.judge_loaders.load_response_scores` helper
    with per-entity B=7→B=10 fallback.

    Uses :func:`assistant_axis.provenance.load_and_register` to do the
    read + envelope-unwrap + drift-check + InputSpec construction in one
    call.  When ``inputs`` is supplied, every successfully-read cache is
    appended to it (so the caller can record exactly the dependencies
    actually consumed -- a missing per-side file results in no spec for
    that side, matching the old skip-on-missing behaviour).

    ``scores_filename`` defaults to the canonical cache name but can be
    overridden (e.g. ``scores_responses__rubric_v1.json``) so v1 and v2
    plots can be regenerated side-by-side from snapshotted data.
    ``judge_label`` (e.g. ``"gpt"``, ``"haiku_q9"``) is used to
    namespace the per-side ``dep_key`` so accumulated InputSpecs stay
    unique across both judges of a single sweep.
    """
    out: dict[str, float] = {}
    for side in ("roles", "traits"):
        sub = dir_template.format(side=side)
        path = experiment_dir / axis / sub / scores_filename
        if not path.exists():
            continue
        scores, _spec, _check = load_and_register(
            path,
            dep_key=f"scores_{judge_label}_{axis}_{side}",
            extras={"axis": axis, "side": side, "judge": judge_label},
            policy="warn",
            inputs=inputs,
        )
        for name, info in scores.items():
            ms = info.get("mean_score") if isinstance(info, dict) else None
            if ms is not None:
                out[entity_id(name, side)] = float(ms)
    return out


def _load_response_scores_v2(
    experiment_dir: Path, axis: str, judge: str,
    *,
    inputs: list[InputSpec] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """v2-mode loader: canonical
    :func:`assistant_axis.judge_loaders.load_response_scores` for both
    sides, with the project-default ``prefer_b`` per (judge, rubric).

    For ``judge="gpt"`` returns Phase-5c-regenerated B=7 full-volume
    data (no fallback).  For ``judge="haiku"`` returns the Phase-5d
    surgical B=7 tiered-M=3 cohort per-entity, falling back to the
    legacy B=10 q9 uniform-mod-9 cohort for entities not in the
    surgical set.  Roles + traits sides are merged via
    :func:`assistant_axis.entity_id.entity_id` so each entity is keyed
    by its disambiguated ``"name|R"`` / ``"name|T"`` id.

    Returns ``(scores_by_eid, source_cohort_by_eid)``; the latter maps
    each contributing entity-id to the cohort directory it came from,
    useful for the cohort-mix annotation in the rendered plot.
    """
    from assistant_axis.judge_loaders import load_response_scores
    out: dict[str, float] = {}
    sources: dict[str, str] = {}
    for side in ("roles", "traits"):
        scores_by_name, cohort_by_name = load_response_scores(
            axis=axis, kind=side, judge=judge, rubric="v2",
            experiments_root=experiment_dir,
            inputs=inputs, policy="warn",
        )
        for name, info in scores_by_name.items():
            ms = info.get("mean_score") if isinstance(info, dict) else None
            if ms is None:
                continue
            eid = entity_id(name, side)
            out[eid] = float(ms)
            sources[eid] = cohort_by_name.get(name, "?")
    return out, sources


def _v(path: Path, slot: int, layer: int) -> torch.Tensor:
    return _load_vector_file(path).float()[slot, layer]


def _axis_unit(data_dir: Path, pos: str, neg: str,
               slot: int, layer: int) -> torch.Tensor:
    p = _v(data_dir / "traits" / "vectors" / f"{pos}.pt", slot, layer)
    n = _v(data_dir / "traits" / "vectors" / f"{neg}.pt", slot, layer)
    d = p - n
    nrm = torch.linalg.vector_norm(d)
    return d / nrm if nrm > 0 else d


def _rho_at_weight(d: dict, w: float) -> float:
    score = w * d["gpt"] + (1 - w) * d["anth"]
    return float(spearmanr(score, d["proj"]).correlation)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--anthropic_combo", required=True,
                   choices=sorted(COMBO_DIRS.keys()),
                   help="Which Anthropic judge × subsampling combo to "
                        "compare GPT-4.1-mini against.  In ``--rubric v2`` "
                        "mode, ``haiku_q9`` and ``haiku_full`` collapse to "
                        "the same call (load_response_scores picks the "
                        "best-available per-entity cohort), so they "
                        "produce identical data but differently-named "
                        "output files.  ``sonnet_q9`` is rejected in "
                        "v2 mode because Sonnet has no v2 cache.")
    p.add_argument("--rubric", choices=("v1", "v2"), default="v1",
                   help="Data-loading regime (default: v1).  "
                        "* v1: legacy hardcoded B=10 paths "
                        "(GPT full-volume v1 archive, Haiku B=10 q9 OR "
                        "B=10 full per combo, Sonnet B=10 q9).  Reads "
                        "``--scores_filename`` from each hard-wired "
                        "cohort dir.  Matches the pre-May-2026 plots "
                        "exactly.  "
                        "* v2: canonical "
                        "``assistant_axis.judge_loaders.load_response_scores`` "
                        "for both sides.  GPT side reads the Phase-5c "
                        "B=7 full-volume cohort (no fallback); Haiku side "
                        "reads the Phase-5d ``_b7_t3`` (tiered M=3, "
                        "surgical-rejudge subset of entities) per-entity, "
                        "falling back to ``_b10_q9`` for entities not in "
                        "the surgical set.  Effectively a mixed-B view "
                        "of the most-current data.  Output filename "
                        "auto-suffixes ``__rubric_v2``.")
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR))
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--plot", default=None,
                   help="Output PNG (default: gpt_<combo>_response_weight_"
                        "sweep_slot{N}.png inside --experiment_dir).")
    p.add_argument("--rhos_json", default=None)
    p.add_argument("--scores_filename",
                   default="scores_responses__rubric_v1.json",
                   help="Filename within each <axis>/<judge>_responses_*/ "
                        "subdir to read.  Defaults to the v1 archive "
                        "(scores_responses__rubric_v1.json) -- this "
                        "script's GPT-vs-Anthropic weight sweep is "
                        "intrinsically v1-bound because Sonnet was never "
                        "rejudged at v2 (per the trait/role "
                        "disambiguation plan's 'no Sonnet rejudging' "
                        "guardrail), so the only intent-coherent "
                        "cross-judge comparison is v1 vs v1.  The "
                        "default also avoids reading the deferred-broken "
                        "v2 b10 GPT cache (Bug B, 1/3 subsample) and "
                        "the silent-skip on Sonnet axes for which no v2 "
                        "scores_responses.json exists.  Pass "
                        "scores_responses.json explicitly for the v2 "
                        "view at the (deferred-broken) b10 GPT + v2 "
                        "Haiku-q9 combinations.")
    p.add_argument("--out_stem_suffix", default="",
                   help="Optional suffix appended to the output stem (e.g. "
                        "'__rubric_v1' to write "
                        "gpt_<combo>_response_weight_sweep_slot{N}__rubric_v1"
                        ".{json,png}).  Useful for v1<->v2 side-by-side "
                        "snapshots.  Default: empty (canonical names).")
    p.add_argument("--whitening", default=DEFAULT_WHITENING_SPEC,
                   help=f"Whitening regime applied to both entity vectors "
                        f"and the axis direction before computing ρ.  "
                        f"Forms: 'raw', 'soft_K=N', 'lw', 'oas', "
                        f"'soft_shear=L'.  Default: "
                        f"{DEFAULT_WHITENING_SPEC!r} (project canonical -- "
                        f"see assistant_axis/canonical_angles/whitening.py "
                        f"selection-history block).")
    p.add_argument("--ca_kind", default="combined",
                   choices=("combined", "traits", "roles"),
                   help="Goal/no-goal subspaces for soft-shear fitting "
                        "(passed to ``build_goal_nogoal_subspaces``). "
                        "Ignored for non-shear regimes.  Default "
                        "``combined`` matches the L-sweep / "
                        "shear_l_vs_k_comparison defaults.")
    p.add_argument("--w_step", type=float, default=0.025,
                   help="Step size on the w grid (default: 0.025 -> "
                        "41 points on [0, 1]).  Pass --w_step 0.05 to "
                        "reproduce the historical coarser sweep.")
    args = p.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    slot, layer = int(args.slot), int(args.layer)
    combo_label, anth_template = COMBO_DIRS[args.anthropic_combo]
    wh_method, wh_n = parse_whitening_spec(args.whitening)

    # v2 mode validation + filename auto-suffixing.
    is_v2 = (args.rubric == "v2")
    if is_v2:
        if args.anthropic_combo == "sonnet_q9":
            raise SystemExit(
                "error: --rubric v2 is incompatible with "
                "--anthropic_combo sonnet_q9 (Sonnet was never rejudged "
                "at v2 per the trait/role disambiguation plan's "
                "'no Sonnet rejudging' guardrail).  Use --rubric v1 for "
                "the Sonnet-side weight sweep."
            )
        # Auto-append __rubric_v2 when caller didn't pass an explicit
        # --out_stem_suffix.  Mirrors the existing __rubric_v1
        # convention used for side-by-side regen.
        if not args.out_stem_suffix:
            args.out_stem_suffix = "__rubric_v2"

    # Whitening tag: 'raw' is the historical default and gets no suffix
    # so the v1-archive and the new raw view share a slot.  Any other
    # regime appends a short filename-safe suffix.
    if wh_method == "raw":
        wh_suffix = ""
    elif wh_n is not None:
        wh_suffix = f"_{wh_method.replace('_', '')}{wh_n}"
    else:
        wh_suffix = f"_{wh_method.replace('_', '')}"

    out_stem = (
        f"gpt_{args.anthropic_combo}_response_weight_sweep_slot{slot}"
        f"{wh_suffix}{args.out_stem_suffix}"
    )
    if args.plot is None:
        args.plot = f"{out_stem}.png"
    if args.rhos_json is None:
        args.rhos_json = f"{out_stem}.json"

    # w grid validation -- mirror gpt_sonnet_weight_sweep so the two
    # scripts move in lockstep.
    if args.w_step <= 0 or args.w_step > 0.5:
        raise SystemExit(
            f"--w_step must be in (0, 0.5], got {args.w_step}")
    n_w = int(round(1.0 / args.w_step)) + 1
    if abs(n_w - 1 - 1.0 / args.w_step) > 1e-9:
        raise SystemExit(
            f"--w_step={args.w_step} doesn't divide 1.0 evenly; "
            f"pick a divisor of 1.0 (e.g. 0.025, 0.05, 0.02, 0.01).")
    i_half = int(round(0.5 / args.w_step))

    # Cache standalone entity vectors at (slot, layer), default-centered.
    default_v = _v(data_dir / "traits" / "vectors" / "default.pt", slot, layer)
    entity_vecs: dict[str, np.ndarray] = {}
    for et in ("traits", "roles"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            try:
                v = _load_vector_file(fp).float()[slot, layer]
                entity_vecs[entity_id(fp.stem, et)] = (v - default_v).numpy()
            except Exception:  # pragma: no cover -- skip unreadable files
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
                f"--whitening soft_shear={wh_n} exceeds the canonical-angle "
                f"pair budget for kind={args.ca_kind!r} "
                f"(min(n_g,n_n)={n_pairs_max})."
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

    print(f"GPT-4.1-mini B=10 vs {combo_label} response-mode weight sweep, "
          f"slot {slot} layer {layer}, whitening={args.whitening}")
    print(f"  axes: {[a[0] for a in DEFAULT_AXES]}")
    print(f"  anthropic dir template: {anth_template}")

    # Provenance accumulator.  Threaded through every cache read so the
    # output JSON's _provenance.inputs reflects exactly the files this
    # run actually consumed.  Includes the producer script (this file)
    # and the entity-vector subtrees up front; per-axis scores caches
    # are appended inside ``_load_response_scores`` via load_and_register.
    inputs: list[InputSpec] = [
        current_file_input(
            dep_key="producer_script",
            path=_SCRIPT_PATH,
            extras={
                "anthropic_combo": args.anthropic_combo,
                "scores_filename": args.scores_filename,
                "slot": str(slot), "layer": str(layer),
                "whitening": args.whitening,
                "ca_kind": args.ca_kind if wh_method == "soft_shear" else "",
                "rubric": args.rubric,
            },
        ),
        current_data_subtree_input(
            data_dir=data_dir, subtree_rel="traits/vectors",
            dep_key="traits_vectors",
        ),
        current_data_subtree_input(
            data_dir=data_dir, subtree_rel="roles/vectors",
            dep_key="roles_vectors",
        ),
    ]
    declare_constants_dependency(inputs)

    per_axis: dict[tuple[str, str], dict] = {}
    cohort_mix: dict[str, dict[str, str]] = {"gpt": {}, "anth": {}}
    for axis_name, pos, neg in DEFAULT_AXES:
        if is_v2:
            # Canonical v2 loaders.  GPT reads Phase-5c b7 full volume;
            # Haiku reads b7_t3 per entity with b10_q9 fallback.  See
            # _load_response_scores_v2 for the cohort-source map we
            # also collect for the "cohort mix" annotation in the plot.
            anth_judge = "haiku"  # both haiku_q9 / haiku_full collapse in v2
            gpt, gpt_sources = _load_response_scores_v2(
                experiment_dir, axis_name, judge="gpt", inputs=inputs)
            anth, anth_sources = _load_response_scores_v2(
                experiment_dir, axis_name, judge=anth_judge, inputs=inputs)
            cohort_mix["gpt"].update(gpt_sources)
            cohort_mix["anth"].update(anth_sources)
        else:
            gpt = _load_response_scores(
                experiment_dir, axis_name, GPT_DIR_TEMPLATE,
                scores_filename=args.scores_filename,
                judge_label="gpt_b10",
                inputs=inputs,
            )
            anth = _load_response_scores(
                experiment_dir, axis_name, anth_template,
                scores_filename=args.scores_filename,
                judge_label=args.anthropic_combo,
                inputs=inputs,
            )
        if not gpt or not anth:
            print(f"  [skip] {axis_name}: gpt n={len(gpt)} "
                  f"anth n={len(anth)}")
            continue
        common = sorted(set(gpt) & set(anth) & set(entity_vecs))
        if len(common) < 5:
            print(f"  [skip] {axis_name}: only {len(common)} shared entities")
            continue
        gpt_arr = np.array([gpt[n] for n in common])
        anth_arr = np.array([anth[n] for n in common])
        a = _axis_unit(data_dir, pos, neg, slot, layer).numpy()
        # Whiten the per-axis direction in lockstep with the entity pool
        # (which was whitened once before this loop).
        if basis is not None and basis.method != "raw":
            a = basis.apply(a[None, :])[0]
        proj = np.array([float(np.dot(entity_vecs[n], a)) for n in common])
        per_axis[(pos, neg)] = {
            "axis_name": axis_name,
            "gpt": gpt_arr, "anth": anth_arr, "proj": proj,
            "n": len(common),
        }
        print(f"  {axis_name}: n={len(common)} "
              f"(pure GPT ρ={_rho_at_weight(per_axis[(pos, neg)], 1.0):+.4f}, "
              f"pure {combo_label} ρ="
              f"{_rho_at_weight(per_axis[(pos, neg)], 0.0):+.4f})")

    if not per_axis:
        print("No axes had data; exiting.")
        return 1

    # Sort axes by interior slope Δρ = ρ(0.9) − ρ(0.1) (Anthropic-best to
    # GPT-best), matching the original sweep's colour convention.
    slopes = {k: _rho_at_weight(d, 0.9) - _rho_at_weight(d, 0.1)
              for k, d in per_axis.items()}
    sorted_keys = sorted(per_axis.keys(), key=lambda k: slopes[k])
    per_axis = {k: per_axis[k] for k in sorted_keys}

    ws = np.linspace(0.0, 1.0, n_w)
    all_rhos = np.array([
        [_rho_at_weight(d, w) for d in per_axis.values()]
        for w in ws
    ])
    mean_rhos = all_rhos.mean(axis=1)

    # Parabolic fit on interior [0.1, 0.9] only (endpoint kinks distort
    # a global fit when w=0 / w=1 = single judge regime).
    mask = (ws >= 0.10 - 1e-9) & (ws <= 0.90 + 1e-9)
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
        i_best = int(np.argmax(mean_rhos))
        w_peak = float(ws[i_best])
        rho_peak = float(mean_rhos[i_best])

    print(f"\nSweep summary ({len(per_axis)} axes, "
          f"whitening={args.whitening}, w_step={args.w_step}):")
    print(f"  pure {combo_label} (w=0):    mean ρ = {mean_rhos[0]:+.4f}")
    print(f"  50/50         (w=0.5):  mean ρ = {mean_rhos[i_half]:+.4f}")
    print(f"  pure GPT      (w=1):    mean ρ = {mean_rhos[-1]:+.4f}")
    print(f"  parabolic peak (interior fit, R²={r2:.3f}): "
          f"w={w_peak:.3f}, ρ ≈ {rho_peak:+.4f}")

    # ---- Plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    n_axes = len(per_axis)
    keys = list(per_axis.keys())
    # Per-axis colours from the global ``AXIS_COLORS`` lookup so that a
    # given axis keeps the same colour across all three Anthropic combo
    # plots (the slope-based sort order can flip the position of an
    # axis between plots; fixed colours preserve cross-plot identity).
    # Fall back to plasma for any axis not in the lookup.
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

    ws_in = np.linspace(0.10, 0.90, 50)
    ws_out = np.linspace(0.0, 1.0, 100)
    y_in = a + b * ws_in + c * ws_in ** 2
    y_out = a + b * ws_out + c * ws_out ** 2
    extrap_handle, = ax.plot(
        ws_out, y_out, color="#444444", lw=1.0, ls=":",
        alpha=0.9, zorder=25, label="parabola (extrapolated)")
    fit_handle, = ax.plot(
        ws_in, y_in, color=PARABOLA_COLOR, lw=2.0, ls="-",
        alpha=0.95, zorder=26,
        label=f"parabola fit on [0.1, 0.9] (R²={r2:.3f})")
    w50_handle = ax.axvline(0.5, color="grey", linestyle=":", lw=1,
                              alpha=0.7, label="w=0.5")
    peak_handle = ax.axvline(
        w_peak, color=PARABOLA_COLOR, linestyle="--", lw=1.4,
        label=f"interior peak w={w_peak:.3f}")

    ax.set_xlabel(f"Weight on GPT-4.1-mini B=10\n"
                  f"(remaining on {combo_label})", fontsize=10)
    ax.set_ylabel(f"Per-axis Spearman ρ "
                  f"(slot {slot}, layer {layer}, "
                  f"whitening={args.whitening})", fontsize=10)
    title = (f"GPT-4.1-mini B=10 / {combo_label} response-mode score-blend "
             f"sweep — mean ρ across {n_axes} axes")
    initial = combo_label[0]
    ax.set_title(
        title + "\n"
        f"{initial}={mean_rhos[0]:.3f}  "
        f"50/50={mean_rhos[i_half]:.3f}  "
        f"G={mean_rhos[-1]:.3f}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.02, 1.02)
    y_hi = float(min(1.0, all_rhos.max() + 0.02))
    y_lo = float(max(0.0, all_rhos.min() - 0.02))
    ax.set_ylim(y_lo, y_hi)
    ax.grid(alpha=0.3)

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
        f"Axes: sorted {combo_label}-best to GPT-best",
        xy=(1.01, 1.005), xycoords="axes fraction",
        ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(
        [mean_handle, fit_handle, extrap_handle, w50_handle, peak_handle],
        [mean_handle.get_label(), fit_handle.get_label(),
         extrap_handle.get_label(), w50_handle.get_label(),
         peak_handle.get_label()],
        loc="lower left", bbox_to_anchor=(1.01, 0.0),
        fontsize=8, frameon=True, framealpha=0.9, handlelength=1.5)
    # Provenance banner.  Two variants depending on --rubric:
    #
    # * v1 mode (default): "outdated archive" banner.  GPT-vs-Anthropic
    #   weight sweep is intrinsically v1-bound here -- Sonnet was never
    #   rejudged at v2, and the haiku v2 caches mostly stay at B=10 q9.
    #   The plot is kept for historical reference; the cost-benefit of
    #   rerunning everything at v2 / B=7 against Haiku is poor.
    #
    # * v2 mode: "current data" banner.  GPT side at Phase-5c B=7
    #   full-volume, Haiku side mixed B=7 t3 (surgical) → B=10 q9
    #   fallback.  Cohort-mix summary is computed from ``cohort_mix``
    #   so the operator can see what fraction of each side came from
    #   each cohort.
    if is_v2:
        gpt_cohorts = sorted(set(cohort_mix["gpt"].values()))
        anth_dict = cohort_mix["anth"]
        anth_total = len(anth_dict) or 1
        anth_b7 = sum(1 for c in anth_dict.values() if "_b7" in c)
        anth_b10 = sum(1 for c in anth_dict.values() if "_b10" in c)
        banner_text = (
            f"v2/v3 current-data view.\n"
            f"GPT: {', '.join(gpt_cohorts) or '(no data)'} (full volume).\n"
            f"Haiku: per-entity mix -- "
            f"{anth_b7} of {anth_total} from _b7_t3 (Phase-5d surgical), "
            f"{anth_b10} from _b10_q9 (legacy fallback)."
        )
        banner_face, banner_edge, banner_text_color = (
            "#ecf6ff", "#5588cc", "#1a3d6a")
    else:
        rubric_tag = (
            "rubric_v1" if "rubric_v1" in str(args.scores_filename)
            else str(args.scores_filename)
        )
        banner_text = (
            f"v1-archive view ({rubric_tag} caches; "
            f"GPT B=10 full, Haiku B=10 q9 ~1/3 subsample).\n"
            f"Not regenerated under v2/v3 rubric or B=7 default; data "
            f"and conclusions reflect the pre-May-2026 judging regime."
        )
        banner_face, banner_edge, banner_text_color = (
            "#fff4ec", "#cc6655", "#882200")
    ax.text(
        0.99, 0.02, banner_text,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, style="italic", color=banner_text_color,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor=banner_face,
                  edgecolor=banner_edge, linewidth=0.7, alpha=0.92),
        zorder=10,
    )
    plt.tight_layout()

    out_path = experiment_dir / args.plot
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # ---- JSON ------------------------------------------------------------
    json_payload = {
        "n_axes": n_axes,
        "first_judge": "gpt_b10",
        "second_judge_label": combo_label,
        "anthropic_combo": args.anthropic_combo,
        "anthropic_dir_template": anth_template,
        "slot": slot, "layer": layer,
        "whitening": args.whitening,
        "ca_kind": args.ca_kind if wh_method == "soft_shear" else None,
        "rubric": args.rubric,
        "w_step": float(args.w_step),
        "ws": [float(w) for w in ws],
        "mean_rho_by_w": [float(v) for v in mean_rhos],
        "parabola_fit": {
            "domain": [0.10, 0.90],
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
    print(f"Wrote {json_path} ({len(inputs)} inputs recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
