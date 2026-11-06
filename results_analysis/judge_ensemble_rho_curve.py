#!/usr/bin/env python3
"""Compute response-mode ρ for **judge ensembles**: GPT-4.1-mini at B=10
weighted with each of three Anthropic options on the same items.

For each ensemble combo × axis × (slot, layer) cell, we:

1. Load per-entity mean ``mean_score`` from
   ``<experiment_dir>/<axis>/<dir>/scores_responses.json`` for both
   judges (e.g. ``gpt_responses_<side>_b10`` and
   ``haiku_responses_<side>_b10_q9``), combining roles + traits sides.
2. Combine them with
   ``combined[name] = w * gpt[name] + (1 - w) * anth[name]``
   over the entities present in BOTH judges' caches.  The default
   weight is the project-wide
   :data:`assistant_axis.judge_score_combine.DEFAULT_GPT_HAIKU_Q9_WEIGHT`
   (= 0.60 as of May 2026; rounded from the parabolic peak of the
   12-axis sweep at slot 6).  Override per-run with ``--gpt_weight``.
3. Run the same ``(L, K)`` sweep that ``batch_size_rho_curve`` uses
   (coarse grid + bracket-and-bisect on K), taking the best ρ as that
   (combo, axis, cell)'s score.

Aggregation matches ``plot_batch_size_quality_vs_cost.py``: per axis, max
ρ across cells; then mean across axes per combo.

Provenance: split-file design (May 2026)
----------------------------------------

This script reads the upstream ``batch_size_curve_rho.json`` (produced by
:mod:`results_analysis.batch_size_rho_curve`) and writes its results to a
**separate, side-car JSON file** -- ``batch_size_curve_rho_ensembles.json``
by default -- rather than mutating the upstream cache in place.

Why split?  Each output file in the provenance system has a single
producer and one provenance envelope; jamming two stages' outputs into
one file would mean either losing one stage's input list or inventing
per-block sub-envelopes.  Splitting keeps each stage's provenance crisp
and lets ``audit_caches.py`` flag drift on whichever stage's inputs have
moved.  See ``AGENT_NOTES.md`` for the broader rationale.

Output schema (in the side-car JSON's ``result`` block)::

    {
      "ensemble_combos": {
        "gpt_b10__plus_haiku_q9":   {"per_axis_cell": {"<axis>|s<S>_l<L>": ρ, ...},
                                      "best_per_axis": {"<axis>": ρ_max, ...},
                                      "mean_across_axes_best_cell": ρ,
                                      "per_set": [...]},
        "gpt_b10__plus_haiku_full": ...,
        "gpt_b10__plus_sonnet_q9":  ...,
        "_meta": {"gpt_weight": ..., "axis_sets": [...]}
      },
      "gpt_only_b10_baseline": {
        "per_axis_cell": ..., "best_per_axis": ...,
        "mean_across_axes_best_cell": ρ, "per_set": [...]
      }
    }

CLI
---

::

    uv run python results_analysis/judge_ensemble_rho_curve.py \\
        --input  roger/axis_judge_experiments/batch_size_curve_8slot/batch_size_curve_rho.json
        # output defaults to batch_size_curve_rho_ensembles.json next to --input
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from assistant_axis import entity_id, json_metadata
from assistant_axis.judge_score_combine import (
    DEFAULT_GPT_HAIKU_Q9_WEIGHT,
    declare_constants_dependency,
)
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
    load_and_register,
)
from results_analysis.batch_size_rho_curve import (
    DEFAULT_AXES, DEFAULT_CONFIGS, DEFAULT_DATA_DIR, DEFAULT_EXPERIMENT_DIR,
    axis_direction, best_rho, setup_at,
)


def _default_output_for(input_path: Path) -> Path:
    """Derive the side-car ensembles JSON path from the input path:
    ``foo/bar/batch_size_curve_rho.json`` →
    ``foo/bar/batch_size_curve_rho_ensembles.json``.

    Kept lossless so callers passing a non-default ``--input`` still get a
    sensible auto-default for ``--output``.
    """
    p = Path(input_path)
    return p.with_name(f"{p.stem}_ensembles{p.suffix}")


# Combos to evaluate.  Each entry: (label, anthropic-dir-prefix).  The
# GPT side is fixed at ``gpt_responses_{side}_b10``.
COMBOS: list[tuple[str, str]] = [
    ("gpt_b10__plus_haiku_q9",   "haiku_responses_{side}_b10_q9"),
    ("gpt_b10__plus_haiku_full", "haiku_responses_{side}_b10"),
    ("gpt_b10__plus_sonnet_q9",  "sonnet_responses_{side}_b10_q9"),
]
GPT_DIR_TEMPLATE = "gpt_responses_{side}_b10"


# All 12 traits-based response-mode axes that have GPT B=10 data
# (the 3 original "set 1" axes plus the 9 axes added by Phase 8 in
# May 2026).  ``axes`` is a list of ``(axis_dir_name, pos, neg)``
# triples, matching ``batch_size_rho_curve``'s shape.
ALL_RESPONSE_AXES: list[tuple[str, str, str]] = [
    ("truthful_vs_deceitful",         "truthful",        "deceitful"),
    ("progressive_vs_conservative",   "progressive",     "conservative"),
    ("improvisational_vs_methodical", "improvisational", "methodical"),
    ("concise_vs_verbose",            "concise",         "verbose"),
    ("ecocentric_vs_anthropocentric", "ecocentric",      "anthropocentric"),
    ("egalitarian_vs_elitist",        "egalitarian",     "elitist"),
    ("guileless_vs_scheming",         "guileless",       "scheming"),
    ("harmless_vs_harmful",           "harmless",        "harmful"),
    ("honest_vs_dishonest",           "honest",          "dishonest"),
    ("helpful_vs_unhelpful",          "helpful",         "unhelpful"),
    ("relativist_vs_absolutist",      "relativist",      "absolutist"),
    ("systems_thinker_vs_analytical", "systems_thinker", "analytical"),
]


# Four sets of three axes.  Set 1 is the original triplet (the three
# axes that have full B-size sweep data).  Sets 2..4 cover the nine
# May-2026 additions, alphabetically.  Used by the Pareto plot to
# show per-set slope variability of the ensemble lift on top of the
# B=10 GPT-only baseline.
AXIS_SETS: list[dict] = [
    {"set_id": "set_1_initial",
     "label": "set 1 (initial)",
     "axes": ["truthful_vs_deceitful",
              "progressive_vs_conservative",
              "improvisational_vs_methodical"]},
    {"set_id": "set_2",
     "label": "set 2",
     "axes": ["concise_vs_verbose",
              "ecocentric_vs_anthropocentric",
              "egalitarian_vs_elitist"]},
    {"set_id": "set_3",
     "label": "set 3",
     "axes": ["guileless_vs_scheming",
              "harmless_vs_harmful",
              "helpful_vs_unhelpful"]},
    {"set_id": "set_4",
     "label": "set 4",
     "axes": ["honest_vs_dishonest",
              "relativist_vs_absolutist",
              "systems_thinker_vs_analytical"]},
]


def _load_response_scores(
    experiment_dir: Path, axis: str, dir_template: str,
    *,
    scores_filename: str = "scores_responses.json",
    judge_label: str,
    inputs: list[InputSpec] | None = None,
) -> dict[str, float]:
    """Load per-entity mean response-judge scores from one judge×axis,
    summing across the roles+traits sides.  Returns ``{name: ρ}``.

    ``scores_filename`` defaults to the canonical cache name; pass
    ``scores_responses__rubric_v1.json`` to read from the v1 snapshot
    after a rubric version bump.  ``judge_label`` (e.g. ``"gpt_b10"``,
    ``"gpt_b10__plus_haiku_q9"``) is used to namespace the
    InputSpecs so multiple judges over the same axis stay distinct in
    the consumer's recorded inputs.

    Uses :func:`assistant_axis.provenance.load_and_register` to do
    the read + envelope-unwrap + drift-check + InputSpec construction
    in one call; when ``inputs`` is supplied, every successfully-read
    cache is appended to it (per the read+register pattern in
    AGENT_NOTES.md).
    """
    out: dict[str, float] = {}
    for side in ("roles", "traits"):
        sub = dir_template.format(side=side)
        path = experiment_dir / axis / sub / scores_filename
        if not path.exists():
            continue
        scores, _spec, _check = load_and_register(
            path,
            dep_key=f"judge_{axis}_responses_{judge_label}_{side}",
            extras={"axis": axis, "side": side, "judge": judge_label},
            policy="warn",
            inputs=inputs,
        )
        for name, info in scores.items():
            ms = info.get("mean_score") if isinstance(info, dict) else None
            if ms is not None:
                out[entity_id(name, side)] = float(ms)
    return out


def _combine_weighted(a: dict[str, float], b: dict[str, float],
                       w_a: float) -> dict[str, float]:
    """``w_a * a[name] + (1 - w_a) * b[name]`` for entities in BOTH dicts."""
    common = a.keys() & b.keys()
    return {n: w_a * a[n] + (1.0 - w_a) * b[n] for n in common}


def _aggregate_per_set(
    best_per_axis: dict[str, float],
    axis_sets: Sequence[dict] = tuple(AXIS_SETS),
) -> list[dict]:
    """Group ``best_per_axis`` (axis_name -> max-over-cells ρ) by ``axis_sets``
    and return one entry per set with the mean ρ over the set's available
    axes.  Sets with no available axes are skipped."""
    out: list[dict] = []
    for s in axis_sets:
        members = [a for a in s["axes"] if a in best_per_axis]
        if not members:
            continue
        rho_mean = float(np.mean([best_per_axis[a] for a in members]))
        out.append({
            "set_id": s["set_id"],
            "label": s["label"],
            "axes": members,
            "n_axes": len(members),
            "rho_mean_best_cell": rho_mean,
        })
    return out


def _compute_gpt_only_b10_per_axis_best_cell(
    *,
    experiment_dir: Path,
    data_dir: Path,
    axes: Sequence[tuple[str, str, str]],
    configs: Sequence[tuple[int, int]],
    geom: dict,
    scores_filename: str = "scores_responses.json",
    inputs: list[InputSpec] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """For each axis, max ρ across cells using **GPT-only** B=10 scores.

    Returns ``(per_axis_cell, best_per_axis)`` -- the same shape that
    ``compute_ensemble_rho`` returns per combo, but using just the GPT
    judge (no Anthropic blend).  Provides the natural per-set baseline
    against which each ensemble combo's per-set lift can be measured
    on a like-axis-set basis.
    """
    per_axis_cell: dict[str, float] = {}
    best_per_axis: dict[str, float] = {}
    for axis_name, pos_name, neg_name in axes:
        gpt = _load_response_scores(
            experiment_dir, axis_name, GPT_DIR_TEMPLATE,
            scores_filename=scores_filename,
            judge_label="gpt_b10", inputs=inputs,
        )
        if not gpt:
            continue
        cell_rhos: list[float] = []
        for slot, layer in configs:
            names, M_raw, A_g, A_n, pool = geom[(slot, layer)]
            axis_dir = axis_direction(data_dir, pos_name, neg_name, slot, layer)
            rho = best_rho(M_raw, axis_dir, A_g, A_n, pool, names, gpt)
            if np.isnan(rho):
                rho = float("nan")
            per_axis_cell[f"{axis_name}|s{slot}_l{layer}"] = float(rho)
            if np.isfinite(rho):
                cell_rhos.append(float(rho))
        if cell_rhos:
            best_per_axis[axis_name] = max(cell_rhos)
    return per_axis_cell, best_per_axis


def compute_ensemble_rho(
    *,
    experiment_dir: Path,
    data_dir: Path,
    axes: Sequence[tuple[str, str, str]],
    configs: Sequence[tuple[int, int]],
    combos: Sequence[tuple[str, str]] = COMBOS,
    gpt_weight: float = 0.625,
    scores_filename: str = "scores_responses.json",
    _geom: dict | None = None,
    inputs: list[InputSpec] | None = None,
) -> dict:
    """Compute ensemble ρ for every (combo, axis, cell).

    ``axes`` is a sequence of ``(axis_name, pos, neg)`` triples (the
    same shape that ``batch_size_rho_curve`` uses).  ``configs`` is a
    sequence of ``(slot, layer)`` pairs.

    ``_geom`` (optional) is a pre-computed ``{(slot, layer): setup_at(...)}``
    cache.  When supplied, skip the per-cell ``setup_at`` calls (each costs
    a few seconds).  Used by ``main`` to share geometry between the
    GPT-only baseline pass and the ensemble pass.

    Returns a dict shaped for direct insertion under ``ensemble_combos``
    in the batch-size curve JSON.
    """
    # ---- pre-cache geometry per (slot, layer) (re-fits are expensive) ----
    if _geom is not None:
        geom = _geom
    else:
        geom = {}
        for slot, layer in configs:
            print(f"  geom: setting up cell ({slot}, {layer}) ...")
            geom[(slot, layer)] = setup_at(data_dir, slot, layer)

    out: dict[str, dict] = {}
    for combo_label, anth_template in combos:
        print(f"\n=== combo: {combo_label} ===")
        per_axis_cell: dict[str, float] = {}
        best_per_axis: dict[str, float] = {}

        for axis_triple in axes:
            axis_name = axis_triple[0]
            pos_name, neg_name = axis_triple[1], axis_triple[2]
            print(f"  axis: {axis_name}")

            gpt = _load_response_scores(
                experiment_dir, axis_name, GPT_DIR_TEMPLATE,
                scores_filename=scores_filename,
                judge_label="gpt_b10", inputs=inputs,
            )
            anth = _load_response_scores(
                experiment_dir, axis_name, anth_template,
                scores_filename=scores_filename,
                judge_label=combo_label, inputs=inputs,
            )
            if not gpt or not anth:
                print(f"    skip: gpt n={len(gpt)} anth n={len(anth)}")
                continue
            combined = _combine_weighted(gpt, anth, w_a=gpt_weight)
            print(f"    combined entities: {len(combined)} "
                  f"(gpt {len(gpt)} ∩ anth {len(anth)}; "
                  f"weights: {gpt_weight:.3f}*gpt + "
                  f"{1.0 - gpt_weight:.3f}*anth)")

            cell_rhos: list[float] = []
            for slot, layer in configs:
                names, M_raw, A_g, A_n, pool = geom[(slot, layer)]
                axis_dir = axis_direction(
                    data_dir, pos_name, neg_name, slot, layer)
                rho = best_rho(M_raw, axis_dir, A_g, A_n, pool,
                               names, combined)
                if np.isnan(rho):
                    rho = float("nan")
                per_axis_cell[f"{axis_name}|s{slot}_l{layer}"] = float(rho)
                cell_rhos.append(float(rho))
                print(f"    ({slot},{layer}): ρ = {rho:+.4f}")
            best_per_axis[axis_name] = max(
                r for r in cell_rhos if np.isfinite(r)
            ) if any(np.isfinite(r) for r in cell_rhos) else float("nan")

        if best_per_axis:
            grand = float(np.mean(list(best_per_axis.values())))
        else:
            grand = float("nan")
        per_set = _aggregate_per_set(best_per_axis)
        print(f"  combo {combo_label}: best-per-axis mean = {grand:+.4f} "
              f"(over {len(best_per_axis)} axes)")
        for entry in per_set:
            print(f"    {entry['label']:>20s}: ρ = "
                  f"{entry['rho_mean_best_cell']:+.4f}  "
                  f"(n_axes={entry['n_axes']})")

        out[combo_label] = {
            "per_axis_cell": per_axis_cell,
            "best_per_axis": best_per_axis,
            "mean_across_axes_best_cell": grand,
            "per_set": per_set,
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", type=str, required=True,
        help="Existing batch_size_curve_rho.json -- read for "
             "axes / configs metadata.  Not modified.")
    p.add_argument(
        "--output", type=str, default=None,
        help="Side-car JSON path where ensemble results are written.  "
             "Default: ``<input_stem>_ensembles<input_suffix>`` next to "
             "--input (split-file design; see module docstring).")
    p.add_argument(
        "--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
        help=f"Vectors / pool data dir.  Default: {DEFAULT_DATA_DIR}")
    p.add_argument(
        "--experiment_dir", type=str, default=DEFAULT_EXPERIMENT_DIR,
        help=f"Where the per-axis judge dirs live.  Default: "
             f"{DEFAULT_EXPERIMENT_DIR}")
    p.add_argument(
        "--gpt_weight", type=float, default=DEFAULT_GPT_HAIKU_Q9_WEIGHT,
        help=f"Weight on GPT-mini in the ensemble; the Anthropic weight "
             f"is 1 - this.  Default "
             f"{DEFAULT_GPT_HAIKU_Q9_WEIGHT:g} (single source of truth: "
             f"``DEFAULT_GPT_HAIKU_Q9_WEIGHT`` in "
             f"``assistant_axis/judge_score_combine.py``; rounded from "
             f"the 12-axis response-mode GPT/Haiku-q9 parabolic peak at "
             f"w=0.609).  Haiku-q9 is the operating-point winner on "
             f"cost-per-quality across the 4-Pareto-set view, see "
             f"roger/axis_judge_experiments/batch_size_curve_8slot/"
             f"batch_size_cost_vs_quality.png.  Sonnet-q9's own peak "
             f"is higher (w≈0.73), but Sonnet is dominated by Haiku-q9 "
             f"on the Pareto frontier and is kept only for diagnostic "
             f"comparison.  Use 0.5 for even-50/50.")
    p.add_argument(
        "--axes_source", choices=["all_response_axes", "input_json"],
        default="all_response_axes",
        help="Which axes to evaluate ensembles over.  Default: "
             "ALL_RESPONSE_AXES (12 traits axes; combos with missing data "
             "auto-skip per axis).  'input_json' reuses raw['axes'] from the "
             "input JSON (the legacy 3-axis B-curve set).")
    p.add_argument(
        "--scores_filename", type=str,
        default="scores_responses__rubric_v1.json",
        help="Per-cell judge score cache filename (default: "
             "scores_responses__rubric_v1.json -- the v1 archive).  "
             "This ensemble script is intrinsically v1-bound because "
             "the Sonnet and Haiku-full combos don't have v2 caches "
             "(Sonnet was never rejudged at v2 per the trait/role "
             "disambiguation plan's guardrail; Haiku-full v2 only "
             "exists for a subset of axes), AND the v2 b10 GPT cache "
             "is the deferred-broken Bug-B 1/3-subsample.  Pass "
             "scores_responses.json explicitly for the v2 view at "
             "the (limited) combos where both judges have a v2 cache.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else _default_output_for(in_path)
    # Provenance accumulator: load_and_register threads through every
    # cache read (the upstream batch-size curve here, plus per-axis
    # judge scores via _load_response_scores → load_and_register
    # inside the compute helpers below).  Single accumulator keeps
    # reads and dependency records in lockstep -- if a cache wasn't
    # consumed it doesn't end up in inputs, and vice versa.
    inputs: list[InputSpec] = [
        current_data_subtree_input(
            Path(args.data_dir), "traits/vectors",
            dep_key="traits_vectors"),
        current_data_subtree_input(
            Path(args.data_dir), "roles/vectors",
            dep_key="roles_vectors"),
    ]
    # The default --gpt_weight value is read from
    # DEFAULT_GPT_HAIKU_Q9_WEIGHT at import time; record the constants
    # file as a cache-invalidation dependency so a retune flags this
    # script's outputs as stale via audit_caches.py.
    declare_constants_dependency(inputs)
    # Tolerate legacy bare-JSON during the rollout window; once
    # batch_size_rho_curve.py has been re-run post-Phase-D.3 the input
    # carries an envelope and ``warn`` will validate it.
    raw, _spec, _check = load_and_register(
        in_path, dep_key="batch_size_curve_json",
        inputs=inputs, policy="warn",
    )

    if args.axes_source == "all_response_axes":
        axes_t = list(ALL_RESPONSE_AXES)
    else:
        axes = raw.get("axes") or [list(t) for t in DEFAULT_AXES]
        axes_t = [(a[0], a[1], a[2]) for a in axes]
    configs_raw = raw.get("configs") or [list(c) for c in DEFAULT_CONFIGS]
    configs = [(int(c[0]), int(c[1])) for c in configs_raw]

    print(f"Computing ensemble ρ for up to {len(axes_t)} axes × "
          f"{len(configs)} cells × {len(COMBOS)} combos "
          f"(axes_source={args.axes_source}); "
          f"per-combo axes filter to those with both GPT and Anthropic data.")

    # Precompute geometry once and share between GPT-only baseline and ensemble.
    geom: dict[tuple[int, int], tuple] = {}
    for slot, layer in configs:
        print(f"  geom: setting up cell ({slot}, {layer}) ...")
        geom[(slot, layer)] = setup_at(Path(args.data_dir), slot, layer)

    print("\n=== GPT-only B=10 per-axis baseline ===")
    gpt_only_per_axis_cell, gpt_only_best_per_axis = (
        _compute_gpt_only_b10_per_axis_best_cell(
            experiment_dir=Path(args.experiment_dir),
            data_dir=Path(args.data_dir),
            axes=axes_t, configs=configs, geom=geom,
            scores_filename=args.scores_filename,
            inputs=inputs,
        )
    )
    gpt_only_per_set = _aggregate_per_set(gpt_only_best_per_axis)
    if gpt_only_best_per_axis:
        gpt_only_overall = float(np.mean(list(gpt_only_best_per_axis.values())))
    else:
        gpt_only_overall = float("nan")
    print(f"  GPT-only B=10 mean (over {len(gpt_only_best_per_axis)} axes): "
          f"{gpt_only_overall:+.4f}")
    for entry in gpt_only_per_set:
        print(f"    {entry['label']:>20s}: ρ = "
              f"{entry['rho_mean_best_cell']:+.4f}  "
              f"(n_axes={entry['n_axes']})")

    print("\n=== Ensemble combos ===")
    ensemble = compute_ensemble_rho(
        experiment_dir=Path(args.experiment_dir),
        data_dir=Path(args.data_dir),
        axes=axes_t, configs=configs,
        gpt_weight=float(args.gpt_weight),
        scores_filename=args.scores_filename,
        # Reuse pre-computed geometry to avoid the duplicate setup_at cost.
        _geom=geom,
        inputs=inputs,
    )
    # Stamp the weights into the JSON so the plotter / future readers
    # know what blend produced these ρ values.
    ensemble["_meta"] = {
        "gpt_weight": float(args.gpt_weight),
        "anthropic_weight": 1.0 - float(args.gpt_weight),
        "axes_source": args.axes_source,
        "n_axes_evaluated": len(axes_t),
        "axis_sets": [
            {"set_id": s["set_id"], "label": s["label"], "axes": s["axes"]}
            for s in AXIS_SETS
        ],
    }

    # ---- Side-car JSON output ------------------------------------------
    # ``inputs`` was populated above by load_and_register at every read
    # site:
    #   * the upstream batch_size_curve_rho.json,
    #   * the traits/roles vectors subtrees (axis directions + entity
    #     projections),
    #   * one InputSpec per (axis, judge_dir, side) for every
    #     scores_responses.json that was actually consumed (per-axis
    #     dep_keys so audit_caches.py can pinpoint which judge cache
    #     changed).  Note this naturally registers only the caches
    #     that succeeded; pre-retrofit the manual register loop here
    #     could add deps for files we never read (e.g. axes that the
    #     anth combo's compute pass would skip due to missing data).
    # Caveat: `_compute_gpt_only_b10_per_axis_best_cell` and
    # `compute_ensemble_rho` each load the GPT side independently, so
    # the same gpt path is registered twice with two different
    # InputSpecs (identical fingerprints, different list positions).
    # That's harmless for audit purposes -- both rows refer to the
    # same file -- and matches the previous behaviour of recording
    # one entry per intended consumer rather than per unique path.

    side_car = {
        "ensemble_combos": ensemble,
        "gpt_only_b10_baseline": {
            "per_axis_cell": gpt_only_per_axis_cell,
            "best_per_axis": gpt_only_best_per_axis,
            "mean_across_axes_best_cell": gpt_only_overall,
            "per_set": gpt_only_per_set,
        },
    }
    envelope = json_metadata(
        side_car,
        inputs=inputs,
        title=f"judge_ensemble_rho_curve gpt_weight={args.gpt_weight} "
              f"axes_source={args.axes_source}",
    )
    out_path.write_text(json.dumps(envelope, indent=2))
    print(f"\nWrote {out_path}")
    print()
    print(f"{'combo':<28}  {'mean(best-cell-per-axis)':>26}  {'n_axes':>6}")
    for k, v in ensemble.items():
        if k == "_meta" or "mean_across_axes_best_cell" not in v:
            continue
        n = len(v.get("best_per_axis", {}))
        print(f"  {k:<28}  {v['mean_across_axes_best_cell']:>+26.4f}  "
              f"{n:>6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
