#!/usr/bin/env python3
"""For each axis pair, vary the whitening K and compute Spearman ρ between
judge scores and the projection of entity vectors onto the axis at K.

Per axis we report two ρ-vs-K curves:

- ``desc_inst`` -- per entity, combine (GPT-descriptions,
  GPT-instructions, Sonnet-descriptions, Sonnet-instructions) via
  :func:`assistant_axis.judge_score_combine.combine_desc_inst_two_judges`
  (default: inst-tiebreak weighting `0.499*desc + 0.501*inst`; pass
  ``--di_weights {inst_tie,equal,desc_tie}`` to override), then ρ vs
  the projection.
- ``responses`` -- GPT response-mode mean score per entity, then ρ vs
  the projection.  (Roles and traits are merged when both are scored.)

Soft-K whitening is fit on the same augmented pool the canonical-angles
tool uses (held-out roles+traits standalones plus ``default.pt``;
:func:`results_analysis.canonical_angles.data.build_augmented_whitening_pool`),
with leave-out reduced to just the 2 pair endpoints (since this is a
single-axis projection task, not a residual-subspace task).  An A/B
test on 12 axes confirmed switching to this shared pool helper
changes ρ values by ≤0.007 at K ≤ 16 and ≤0.016 at K=128 (mean 0.002)
-- well below judge noise.  Using the shared helper keeps the
whitening infrastructure consistent across both tools.

Inputs
------

The script expects the following layout in ``--experiment_dir``:

::

    <experiment_dir>/
        <pairs>.json              # list of {pos, neg, ...};
                                  # default pair_list_responses.json
        <pos>_vs_<neg>/
            gpt/scores_descriptions.json
            gpt/scores_instructions.json
            sonnet/scores_descriptions.json
            sonnet/scores_instructions.json
            gpt_responses_traits_b{N}/scores_responses.json
            gpt_responses_roles_b{N}/scores_responses.json
            # ``{N}`` = assistant_axis.judge_batch.RESPONSE_BATCH_SIZE

These are produced by
:mod:`results_analysis.axis_judge_correlation` (see its README section).

Outputs (also written to ``--experiment_dir``)
----------------------------------------------

- ``<sweep>.json`` (default
  ``whitening_k_sweep_<cohort>_slot{N}.json`` -- cohort comes from
  ``--pairs`` via :func:`assistant_axis.cohort_from_pairs`): a list of
  ``{pos, neg, source, K, rho}`` records (one per axis x source x K).
- ``<plot>.png`` (default ``rho_vs_whitening_K_<cohort>_slot{N}.png``):
  an N-line overlay plot (one solid line per axis for ``responses``,
  one dotted line per axis for ``desc_inst``) with K on the x-axis
  (categorical positions ``raw, 1, 2, 4, 8, 16, 32, 64, 128``).

Examples
--------

::

    # Default: responses cohort (axes with GPT response judging in addition
    # to desc+instr), default K grid, default slot 6:
    uv run python results_analysis/whitening_k_sweep.py
    # writes whitening_k_sweep_responses_slot6.json + rho_vs_whitening_K_responses_slot6.png

    # Desc+instr cohort at slot 7:
    uv run python results_analysis/whitening_k_sweep.py \\
        --pairs pair_list_di.json --slot 7
    # writes whitening_k_sweep_di_slot7.json + rho_vs_whitening_K_di_slot7.png

    # Custom experiment directory
    uv run python results_analysis/whitening_k_sweep.py \\
        --experiment_dir /tmp/my_axis_experiments
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
    cohort_from_pairs,
    entity_id,
    json_metadata,
    pair_type_of,
    png_metadata,
    response_subdir,
    suptitle_with_specs,
)
from assistant_axis.judge_loaders import migrate_v1_static_scores
from assistant_axis.judge_score_combine import (
    DEFAULT_RESPONSE_DI_WEIGHT,
    PRIMARY_AXIS_SAMPLE_WEIGHT,
    add_di_weights_arg,
    cohort_mean_curves,
    combine_desc_inst_two_judges,
    parse_di_weights_arg,
)
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
    load_and_register,
)
from results_analysis.axis_judge_correlation import _load_vector_file
from results_analysis.canonical_angles.data import (
    build_augmented_whitening_pool,
)
from results_analysis.canonical_angles.whitening import fit_whitening

DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
    "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
)
LAYER = 25  # Qwen-3-32B; tuned via rho_by_layer.py.  Other models TBD.
DEFAULT_SLOT = 6  # New default after May 2026 rejudge run: slot 6 (</think>)
                  # beats slot 3 (\\n) by judge ρ across most axes -- see
                  # roger/axis_judge_experiments/rho_by_slot_and_K.png and
                  # rho_by_layer.png.  Pass --slot 3 (or 7) to compare.
                  # Output filenames auto-include _slot{N} when the user
                  # leaves --plot / --sweep at their generic defaults.
DEFAULT_K_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32]
K_VALUES = list(DEFAULT_K_VALUES)  # may be rebound by main() via --Ks


def load_pool(data_dir: Path, exclude_names: set[str], *, slot: int):
    """Build the held-out whitening pool for an axis-judge run.

    Returns ``(entity_vecs, pool, default_sl, kinds_for_name)`` where:

    - ``entity_vecs`` -- dict ``{name: tensor at (slot, LAYER)}`` for
      every standalone role + trait, with ``default.pt`` SUBTRACTED so
      projections are taken in the default-centered frame.
    - ``pool`` -- (n, hidden) numpy array used to fit the soft-K
      whitener; built via
      :func:`results_analysis.canonical_angles.data.build_augmented_whitening_pool`
      with ``leave_out = {("traits", n), ("roles", n) for n in exclude_names}``
      so only the pair endpoints are removed.  This is the same pool
      builder used by the canonical-angles tool, but with a different
      leave-out convention appropriate to a single-axis projection
      task: hold out ONLY the 2 names that defined the axis direction,
      not the 30+30 names that anchor a residual subspace.
    - ``default_sl`` -- the default activation at (SLOT, LAYER), used
      for centering.
    - ``kinds_for_name`` -- map ``{bare_name: {kind, ...}}`` over the
      corpus, used by callers as the input to
      :func:`assistant_axis.judge_loaders.migrate_v1_static_scores`
      so v1 (bare-name) judge caches can be lifted to v2 (entity_id)
      keys in-memory before intersection.  Built incidentally from
      the same corpus walk so callers don't need a second pass.

    A/B test on 12 axes confirmed switching to this pool helper
    changes ρ values by at most 0.004 (mean 0.0006) -- well below
    judge noise.  Switching over keeps the whitening infrastructure
    consistent across both tools.
    """
    default = _load_vector_file(
        data_dir / "traits" / "vectors" / "default.pt").float()
    default_sl = default[slot, LAYER]

    entity_vecs: dict[str, torch.Tensor] = {}
    kinds_for_name: dict[str, set] = {}
    for etype in ("traits", "roles"):
        vdir = data_dir / etype / "vectors"
        if not vdir.exists():
            continue
        for f in sorted(vdir.glob("*.pt")):
            if f.stem == "default":
                continue
            try:
                v = _load_vector_file(f).float()
            except Exception:  # pragma: no cover -- skip unreadable files
                continue
            entity_vecs[entity_id(f.stem, etype)] = v[slot, LAYER] - default_sl
            kinds_for_name.setdefault(f.stem, set()).add(etype)

    # Build the pool via the canonical-angles helper.  Pass leave-out
    # entries for both etypes since we don't know which one the pair
    # endpoints live in (build_augmented_whitening_pool silently
    # ignores names that don't exist in a given etype).
    leave_out: set[tuple[str, str]] = set()
    for n in exclude_names:
        leave_out.add(("traits", n))
        leave_out.add(("roles", n))
    pool_entries = build_augmented_whitening_pool(
        data_dir, leave_out=leave_out, scope="roles+traits")

    def _path(etype: str, name: str) -> Path:
        if etype == "combinations":
            return data_dir / "combinations" / "vectors" / f"{name}.pt"
        return data_dir / etype / "vectors" / f"{name}.pt"

    pool_rows = [
        _load_vector_file(_path(et, n)).float()[slot, LAYER].numpy()
        for (et, n) in pool_entries
    ]
    pool = np.stack(pool_rows, axis=0)
    return entity_vecs, pool, default_sl, kinds_for_name


def axis_direction(data_dir: Path, pos: str, neg: str, *,
                   slot: int,
                   pair_type: str = "traits") -> torch.Tensor:
    """Load pair vectors and return the unit axis direction at (slot, LAYER).

    ``pair_type`` selects the subdirectory under ``data_dir`` (``"traits"``
    or ``"roles"``); pass ``"roles"`` for role-pair axes.  Defaults to
    ``"traits"`` for backward compatibility with legacy callers.
    """
    vp = _load_vector_file(data_dir / pair_type / "vectors" / f"{pos}.pt").float()
    vn = _load_vector_file(data_dir / pair_type / "vectors" / f"{neg}.pt").float()
    d = vp[slot, LAYER] - vn[slot, LAYER]
    d = d / torch.linalg.vector_norm(d)
    return d


def project_at_K(entity_vecs, pool, axis_unit: torch.Tensor, K: int) -> dict:
    """Return ``{name: float projection}`` under a soft-K whitener fit
    on ``pool``.  ``K=0`` means raw (no whitening).

    ``pool`` is a ``(n, hidden)`` numpy array; the canonical-angles
    :func:`fit_whitening` mean-centres it internally before SVD.  We
    apply the whitener to default-centred entity vectors and the unit
    axis direction.
    """
    if K == 0:
        return {n: float((v @ axis_unit).item())
                for n, v in entity_vecs.items()}
    basis = fit_whitening("soft_K", pool, K=K)
    axis_np = axis_unit.numpy()
    axw = basis.apply(axis_np[None, :])[0]
    axw_n = float(np.linalg.norm(axw))
    proj: dict[str, float] = {}
    for n, v in entity_vecs.items():
        vw = basis.apply(v.numpy()[None, :])[0]
        proj[n] = float(np.dot(vw, axw)) / axw_n
    return proj


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR),
                   help=f"Directory holding pair lists, per-axis "
                        f"score subdirs, and the JSON/PNG outputs "
                        f"(default: {DEFAULT_EXPERIMENT_DIR}).")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                   help=f"Activation-vectors directory "
                        f"(default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--pairs", default="pair_list_responses.json",
                   help="Pair-list JSON filename within --experiment_dir "
                        "(default: pair_list_responses.json -- axes with "
                        "desc+inst from both providers AND GPT response "
                        "scores cached on disk).  Use ``pair_list_di.json`` "
                        "for the wider desc+inst cohort.")
    p.add_argument("--sweep", default=None,
                   help="Output JSON filename within --experiment_dir "
                        "(default: whitening_k_sweep_<cohort>_slot{N}.json -- "
                        "the cohort token comes from --pairs and the slot "
                        "suffix matches --slot).  Pass an explicit filename "
                        "to override.")
    p.add_argument("--plot", default=None,
                   help="Output PNG filename within --experiment_dir "
                        "(default: rho_vs_whitening_K_<cohort>_slot{N}.png).")
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT,
                   help=f"Token-position slot to project onto "
                        f"(default: {DEFAULT_SLOT} = </think>).  Slot 6 is "
                        f"the new judge-ρ winner; pass --slot 3 (\\n) or "
                        f"--slot 7 (\\n\\n post) to compare.")
    p.add_argument("--Ks", default=None,
                   help="Comma-separated whitening-K grid to sweep (default: "
                        f"{','.join(str(k) for k in DEFAULT_K_VALUES)}).  "
                        "Override e.g. for the historical 7-axis cohort, "
                        "which used K=64 and K=128.  Downstream consumers "
                        "(whitening_k_peak_fit.py, whitening_k_weighted_"
                        "scatter.py) extract whichever subset they need.")
    add_di_weights_arg(p)
    args = p.parse_args()
    if args.Ks is not None:
        global K_VALUES
        K_VALUES = sorted({int(k.strip()) for k in args.Ks.split(",") if k.strip()})
        print(f"Overriding K grid via --Ks: {K_VALUES}")
    di_weights = parse_di_weights_arg(args.di_weights)
    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    slot = int(args.slot)
    cohort = cohort_from_pairs(args.pairs)
    if args.sweep is None:
        args.sweep = f"whitening_k_sweep_{cohort}_slot{slot}.json"
    if args.plot is None:
        args.plot = f"rho_vs_whitening_K_{cohort}_slot{slot}.png"

    # Provenance accumulator -- threaded through every cache read via
    # load_and_register so the read AND the InputSpec record happen
    # together (see AGENT_NOTES.md "Reader+registrar pattern").
    inputs: list[InputSpec] = [
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras={"slot": str(slot), "layer": str(LAYER)}),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras={"slot": str(slot), "layer": str(LAYER)}),
    ]
    pairs, _, _ = load_and_register(
        experiment_dir / args.pairs,
        dep_key="pairs_json", inputs=inputs, policy="warn",
    )
    print(f"Loaded {len(pairs)} axis pairs from {args.pairs}")

    # Corpus-wide kinds_for_name map, used to lift v1 (bare-name)
    # judge caches to v2 (entity_id) keys in the per-axis loop
    # below.  Cheap (just a filesystem listing) and corpus-wide
    # rather than per-axis, so compute it once before the loop.
    kinds_for_name: dict[str, set] = {}
    for etype in ("traits", "roles"):
        vdir = data_dir / etype / "vectors"
        if not vdir.exists():
            continue
        for f in sorted(vdir.glob("*.pt")):
            if f.stem != "default":
                kinds_for_name.setdefault(f.stem, set()).add(etype)

    data: dict = {}
    for it in pairs:
        pos, neg = it["pos"], it["neg"]
        ptype = pair_type_of(it)
        axis_id = f"{pos}_vs_{neg}"
        axis_dir = experiment_dir / axis_id
        print(f"[{pos} vs {neg}] (pair_type={ptype}) loading...")

        # --- Scores ---
        g_d, _, _ = load_and_register(
            axis_dir / "gpt" / "scores_descriptions.json",
            dep_key=f"judge_{axis_id}_descriptions_gpt",
            inputs=inputs, policy="warn")
        g_i, _, _ = load_and_register(
            axis_dir / "gpt" / "scores_instructions.json",
            dep_key=f"judge_{axis_id}_instructions_gpt",
            inputs=inputs, policy="warn")
        s_d, _, _ = load_and_register(
            axis_dir / "sonnet" / "scores_descriptions.json",
            dep_key=f"judge_{axis_id}_descriptions_sonnet",
            inputs=inputs, policy="warn")
        s_i, _, _ = load_and_register(
            axis_dir / "sonnet" / "scores_instructions.json",
            dep_key=f"judge_{axis_id}_instructions_sonnet",
            inputs=inputs, policy="warn")
        # Lift any v1 (bare-name) score dicts to v2 (entity_id) keys
        # in-memory before combining, so a four-way intersection between
        # GPT v2 + Sonnet v1 doesn't silently come out empty (which it
        # would do for all 35 axes today: GPT was rejudged into v2 in
        # Phase 5b, Sonnet wasn't touched per the plan's "no Sonnet
        # rejudging" guardrail).  Without this, every desc_inst rho in
        # the JSON / PNG ends up NaN and the di-cohort plot silently
        # shows only the 12 response curves, not the 35 desc+inst
        # curves it advertises.  Migration is a no-op on already-v2
        # input, so applying it uniformly is safe.
        # See assistant_axis.judge_loaders.migrate_v1_static_scores docstring.
        g_d = migrate_v1_static_scores(g_d, kinds_for_name)
        g_i = migrate_v1_static_scores(g_i, kinds_for_name)
        s_d = migrate_v1_static_scores(s_d, kinds_for_name)
        s_i = migrate_v1_static_scores(s_i, kinds_for_name)
        desc_inst = combine_desc_inst_two_judges(g_d, g_i, s_d, s_i,
                                                 weights=di_weights)

        # Response scores merged from traits + roles GPT runs.  Many of
        # the 33-axis cohort don't have response judging yet -- skip
        # silently and leave the responses dict empty so ρ_responses comes
        # out NaN and whitening_k_peak_fit.py just plots desc_inst there.
        # Path-suffix tracks the canonical project-wide response batch
        # size (assistant_axis.judge_batch.RESPONSE_BATCH_SIZE); bump
        # that constant to retire the b={N} corpus.
        responses: dict[str, float] = {}
        for mode in ("traits", "roles"):
            sub = response_subdir("gpt", mode)
            fp = axis_dir / sub / "scores_responses.json"
            if not fp.exists():
                continue
            payload, _, _ = load_and_register(
                fp,
                dep_key=f"judge_{axis_id}_responses_{mode}",
                inputs=inputs, policy="warn",
            )
            for n, info in payload.items():
                if info.get("mean_score") is not None:
                    responses[entity_id(n, mode)] = info["mean_score"]

        # --- Projections at each K ---
        entity_vecs, pool, _default, _kinds = load_pool(
            data_dir, exclude_names={pos, neg}, slot=slot)
        axis_unit = axis_direction(data_dir, pos, neg, slot=slot,
                                   pair_type=ptype)
        projections_by_K = {K: project_at_K(entity_vecs, pool, axis_unit, K)
                            for K in K_VALUES}

        data[(pos, neg)] = {
            "desc_inst": desc_inst,
            "responses": responses,
            "projections_by_K": projections_by_K,
        }

    # --- Compute ρ per (axis, source, K) ---
    rho_table: dict[tuple[str, str, str], list[float]] = {}
    for (pos, neg), d in data.items():
        for source_name, score_map in [("desc_inst", d["desc_inst"]),
                                       ("responses", d["responses"])]:
            rhos: list[float] = []
            for K in K_VALUES:
                proj = d["projections_by_K"][K]
                names = sorted(set(score_map) & set(proj))
                if len(names) < 3:
                    rhos.append(float("nan"))
                    continue
                x = np.array([score_map[n] for n in names], dtype=float)
                y = np.array([proj[n] for n in names], dtype=float)
                rhos.append(spearmanr(x, y).correlation)
            rho_table[(pos, neg, source_name)] = rhos
            print(f"  {pos:16s} vs {neg:16s}  {source_name:9s}  rhos: "
                  + "  ".join(f"{r:+.3f}" for r in rhos))

    # --- Provenance inputs ----------------------------------------------
    # ``inputs`` was populated above by load_and_register at every
    # cache-read site (subtree deps + pair list + per-axis × per-judge
    # × per-mode score caches that were actually consumed).

    # --- Save JSON ---
    records = []
    for (pos, neg, source), rhos in rho_table.items():
        for K, r in zip(K_VALUES, rhos):
            records.append({"pos": pos, "neg": neg, "source": source,
                            "K": K, "rho": r})
    sweep_path = experiment_dir / args.sweep
    envelope = json_metadata(
        records,
        inputs=inputs,
        title=f"whitening_k_sweep slot={slot} pairs={args.pairs}")
    json.dump(envelope, open(sweep_path, "w"), indent=2)
    print(f"\nWrote {len(records)} records to {sweep_path}")

    # --- Plot ---
    pair_keys = [(it["pos"], it["neg"]) for it in pairs]
    axis_colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(pair_keys))))

    # Taller-than-wide aspect ratio (~3:4) gives the dense
    # per-axis stack of curves more vertical room to discriminate
    # at similar-ρ axes -- by-eye legibility at 35+ overlaid lines
    # is dominated by the y-resolution, not x.
    fig, ax = plt.subplots(figsize=(11, 14))
    # x positions on a log(K+1) scale: ``log1p(0)=0`` puts the K=0 (raw,
    # no whitening) point at the origin, the geometrically-spaced higher
    # K values fan out compressively to the right.  Empirically these
    # curves are roughly parabolic in log(K+1), so making the x-axis
    # actually that quantity is what reads off a peak K* by eye.
    x_pos = np.log1p(K_VALUES)
    for i, (pos, neg) in enumerate(pair_keys):
        color = axis_colors[i % 20]
        ax.plot(x_pos, rho_table[(pos, neg, "responses")], color=color, lw=2,
                linestyle="-", marker="o", markersize=5,
                label=f"{pos}/{neg} (responses)")
        ax.plot(x_pos, rho_table[(pos, neg, "desc_inst")], color=color, lw=1.5,
                linestyle=":", marker="s", markersize=4,
                label=f"{pos}/{neg} (desc+inst)")

    # Cohort-mean overlays (black) -- read the cross-axis trend at a
    # glance through the per-axis spaghetti.  See judge_score_combine.
    # cohort_mean_curves for the per-axis blend + weighting rules.
    avg_rs, avg_di, avg_blend = cohort_mean_curves(rho_table, pair_keys)
    ax.plot(x_pos, avg_rs, color="black", lw=2.5, linestyle="-",
            zorder=5,
            label=f"mean over {len(pair_keys)} axes (responses)")
    ax.plot(x_pos, avg_di, color="black", lw=2.5, linestyle=":",
            zorder=5,
            label=f"mean over {len(pair_keys)} axes (desc+inst)")
    ax.plot(x_pos, avg_blend, color="black", lw=2.5, linestyle="--",
            marker="^", markersize=7, zorder=6,
            label=(f"blended mean: {DEFAULT_RESPONSE_DI_WEIGHT:.2f}·rs + "
                   f"{1 - DEFAULT_RESPONSE_DI_WEIGHT:.2f}·di per axis, "
                   f"{PRIMARY_AXIS_SAMPLE_WEIGHT:g}x sample weight on "
                   f"axes with responses"))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["raw\n(K=0)"] + [str(K) for K in K_VALUES[1:]])
    ax.set_xlabel("Soft-whitening K, log(K+1) scale "
                  "(raw = no whitening; higher K = more PCs scaled down)")
    ax.set_ylabel(f"Spearman ρ (judge scores vs. projection, slot={slot})")
    title_line = f"ρ vs whitening K across {len(pair_keys)} axes"
    spec_line = ("solid = GPT responses; "
                 "dotted = desc+inst (GPT+Sonnet averaged); "
                 "black = cohort means (solid=rs, dotted=di, "
                 "dashed▲=blend)")
    _, top_rect = suptitle_with_specs(fig, title_line, spec_line)
    ax.axhline(0, color="grey", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7,
              ncol=1)
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    plot_path = experiment_dir / args.plot
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)
    print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
