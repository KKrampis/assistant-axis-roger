#!/usr/bin/env python3
"""For each axis pair, vary the soft-shear truncation depth L and compute
Spearman ρ between judge scores and the projection of entity vectors
onto the axis after applying an ``L``-shear of the goal / no-goal
subspaces.

Sister script to :mod:`results_analysis.whitening_k_sweep`.  Where that
sweeps the soft-K *whitening* exponent (which shrinks individual PCs of
the augmented entity pool), *this* sweeps the soft-shear *truncation
depth* L (which orthogonalises the top-L canonical-angle pairs of the
goal vs no-goal subspaces, then projects).  Both share the same
desc+inst / responses score cohort plumbing, the same per-axis
Spearman-ρ computation, and the same plot family layout -- so the
two output PNGs (``rho_vs_whitening_K_*.png`` and
``rho_vs_shear_L_*.png``) can be eyeballed side by side.

Why L is interesting on its own
-------------------------------

In a May 2026 head-to-head (`shear_l_vs_k_comparison.py`), L=1 by
itself beat a per-axis Oracle-K policy across K∈{0..3}: 0.681 vs
0.679 over 12 v2 axes, weighted by response-judging cohort
membership.  L=2 and L=3 added another ~0.005 each.  L is the
right operational parameter for the "did the underlying axis
geometry need any inter-subspace orthogonalisation?" question;
the curves below let you read off the per-axis peak L* by eye and
compare to the per-axis peak K* the K sweep produces.

Per-axis content matches the K sweep exactly:

- ``desc_inst`` -- per entity, combine (GPT-descriptions,
  GPT-instructions, Sonnet-descriptions, Sonnet-instructions) via
  :func:`assistant_axis.judge_score_combine.combine_desc_inst_two_judges`
  (default: inst-tiebreak weighting `0.499*desc + 0.501*inst`; pass
  ``--di_weights {inst_tie,equal,desc_tie}`` to override), then ρ vs
  the L-shear projection.
- ``responses`` -- GPT response-mode mean score per entity, then ρ vs
  the L-shear projection.  (Roles and traits are merged when both are
  scored.)

The goal / no-goal subspaces are built once via
:func:`results_analysis.canonical_angles.data.build_goal_nogoal_subspaces`
with ``kind='combined'`` (the same default that
``shear_l_vs_k_comparison.py`` uses), then re-used across every axis
in the sweep.  ``L=0`` is the identity transform (raw projection,
same as ``K=0`` in the K sweep) so the two PNGs share their left-most
data point per axis.

Inputs
------

Mirror the K-sweep layout under ``--experiment_dir``::

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

Outputs (also written to ``--experiment_dir``)
----------------------------------------------

- ``<sweep>.json`` (default
  ``shear_l_sweep_<cohort>_slot{N}.json`` -- cohort from
  :func:`assistant_axis.cohort_from_pairs`): a list of
  ``{pos, neg, source, L, rho}`` records (one per axis x source x L).
- ``<plot>.png`` (default ``rho_vs_shear_L_<cohort>_slot{N}.png``):
  an N-line overlay plot (one solid line per axis for ``responses``,
  one dotted line per axis for ``desc_inst``) with L on a
  log(L+1)-scaled x-axis (same convention as the K sweep, so the
  parabolic shape reads directly).

Examples
--------

::

    # Default: responses cohort, default L grid, default slot 6:
    uv run python results_analysis/shear_l_sweep.py
    # writes shear_l_sweep_responses_slot6.json +
    #        rho_vs_shear_L_responses_slot6.png

    # Desc+instr cohort at slot 7:
    uv run python results_analysis/shear_l_sweep.py \\
        --pairs pair_list_di.json --slot 7
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
from results_analysis.canonical_angles.data import build_goal_nogoal_subspaces
from results_analysis.canonical_angles.whitening import fit_shear


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
    "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
)
LAYER = 25  # Qwen-3-32B; matched to whitening_k_sweep.
DEFAULT_SLOT = 6  # </think>; matches K sweep so the two outputs are
                  # directly comparable.

# Same shape as DEFAULT_K_VALUES in whitening_k_sweep.py.  L values
# beyond ``min(n_g, n_n)`` (the canonical-angle pair count produced by
# build_goal_nogoal_subspaces) get clamped silently to NaN -- fit_shear
# rejects them.  With the default kind='combined' the subspaces are
# typically (D, 60) (30 goal + 30 no-goal pairs), so L up to 30 is
# always safe and L=32 may or may not fit.
DEFAULT_L_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32]
L_VALUES = list(DEFAULT_L_VALUES)  # may be rebound by main() via --Ls


def _load_entity_vecs_and_kinds(
    data_dir: Path, *, slot: int,
) -> tuple[dict[str, torch.Tensor], dict[str, set]]:
    """Return ``(entity_vecs, kinds_for_name)``.

    ``entity_vecs`` is default-centered at (slot, LAYER), keyed by the
    disambiguated entity_id (``"name|R"`` / ``"name|T"``).  Mirror of
    the corresponding block in
    :func:`results_analysis.whitening_k_sweep.load_pool`, minus the
    augmented-whitening pool (which the shear path doesn't use).
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
            except Exception:  # pragma: no cover -- skip unreadable
                continue
            entity_vecs[entity_id(f.stem, etype)] = v[slot, LAYER] - default_sl
            kinds_for_name.setdefault(f.stem, set()).add(etype)
    return entity_vecs, kinds_for_name


def axis_direction(data_dir: Path, pos: str, neg: str, *,
                   slot: int,
                   pair_type: str = "traits") -> torch.Tensor:
    """Load pair vectors and return the unit axis direction at (slot, LAYER).

    Same convention as the K sweep: ``pair_type`` selects ``traits`` vs
    ``roles`` for axes defined on role pairs.
    """
    vp = _load_vector_file(
        data_dir / pair_type / "vectors" / f"{pos}.pt").float()
    vn = _load_vector_file(
        data_dir / pair_type / "vectors" / f"{neg}.pt").float()
    d = vp[slot, LAYER] - vn[slot, LAYER]
    d = d / torch.linalg.vector_norm(d)
    return d


def project_at_L(
    entity_vecs: dict[str, torch.Tensor],
    A_goal: np.ndarray,
    A_nogoal: np.ndarray,
    axis_unit: torch.Tensor,
    L: int,
) -> dict[str, float]:
    """Return ``{name: float projection}`` after applying an ``L``-shear.

    ``L=0`` is the identity (raw projection); for ``L >= 1``
    :func:`assistant_axis.canonical_angles.whitening.fit_shear`
    orthogonalises the top-L canonical-angle pairs of the
    goal / no-goal subspaces, then we project the sheared entity vectors
    onto the sheared axis unit.

    Returns an empty dict (caller treats as NaN) when ``L`` exceeds
    ``min(n_g, n_n)`` (``fit_shear`` raises in that case) -- handled
    here so the per-axis loop in :func:`main` can sweep a single L
    grid across axes whose CA-pair count varies.
    """
    if L == 0:
        return {n: float((v @ axis_unit).item())
                for n, v in entity_vecs.items()}
    try:
        basis = fit_shear(A_goal, A_nogoal, L=L)
    except ValueError:  # L > min(n_g, n_n)
        return {}
    axis_np = axis_unit.numpy()
    axw = basis.apply(axis_np[None, :])[0]
    axw_n = float(np.linalg.norm(axw))
    if axw_n == 0:  # degenerate; nothing to project against
        return {}
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
                   help=f"Directory holding pair lists, per-axis score "
                        f"subdirs, and the JSON/PNG outputs "
                        f"(default: {DEFAULT_EXPERIMENT_DIR}).")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                   help=f"Activation-vectors directory "
                        f"(default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--pairs", default="pair_list_responses.json",
                   help="Pair-list JSON filename within --experiment_dir "
                        "(default: pair_list_responses.json).  Use "
                        "``pair_list_di.json`` for the wider desc+inst "
                        "cohort.")
    p.add_argument("--sweep", default=None,
                   help="Output JSON filename within --experiment_dir "
                        "(default: shear_l_sweep_<cohort>_slot{N}.json).")
    p.add_argument("--plot", default=None,
                   help="Output PNG filename within --experiment_dir "
                        "(default: rho_vs_shear_L_<cohort>_slot{N}.png).")
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT,
                   help=f"Token-position slot to project onto "
                        f"(default: {DEFAULT_SLOT} = </think>).")
    p.add_argument("--Ls", default=None,
                   help="Comma-separated L grid to sweep (default: "
                        f"{','.join(str(L_) for L_ in DEFAULT_L_VALUES)}).")
    p.add_argument(
        "--ca_kind", default="combined",
        choices=("combined", "traits", "roles"),
        help="Which goal/no-goal subspaces to use (passed through to "
             "``build_goal_nogoal_subspaces``).  Default ``combined`` "
             "matches the K sweep's augmented pool semantically and "
             "the existing shear_l_vs_k_comparison default.",
    )
    add_di_weights_arg(p)
    args = p.parse_args()

    if args.Ls is not None:
        global L_VALUES
        L_VALUES = sorted(
            {int(L_.strip()) for L_ in args.Ls.split(",") if L_.strip()}
        )
        print(f"Overriding L grid via --Ls: {L_VALUES}")

    di_weights = parse_di_weights_arg(args.di_weights)
    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    slot = int(args.slot)
    cohort = cohort_from_pairs(args.pairs)
    if args.sweep is None:
        args.sweep = f"shear_l_sweep_{cohort}_slot{slot}.json"
    if args.plot is None:
        args.plot = f"rho_vs_shear_L_{cohort}_slot{slot}.png"

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

    # Build goal / no-goal CA subspaces once -- they're corpus-wide
    # and identical across axes.  Same call site as
    # shear_l_vs_k_comparison.py for cross-script consistency.
    A_goal, A_nogoal = build_goal_nogoal_subspaces(
        data_dir, slot=slot, layer=LAYER, kind=args.ca_kind)
    n_pairs_max = min(A_goal.shape[1], A_nogoal.shape[1])
    print(f"CA subspaces ({args.ca_kind}): A_goal {A_goal.shape}, "
          f"A_nogoal {A_nogoal.shape}; L_max = min(n_g, n_n) = "
          f"{n_pairs_max}")
    if max(L_VALUES) > n_pairs_max:
        clamped = [L_ for L_ in L_VALUES if L_ > n_pairs_max]
        print(f"  [info] L grid exceeds CA-pair budget; values {clamped} "
              f"will silently return NaN for every axis.")

    # Entity vectors (default-centered) shared across axes; we still
    # rebuild per-axis to honor exclude_names = {pos, neg}, but the
    # corpus walk is so cheap (filesystem listing) that doing it once
    # then deep-copying is more trouble than re-listing.
    entity_vecs, kinds_for_name = _load_entity_vecs_and_kinds(
        data_dir, slot=slot)

    rho_table: dict[tuple[str, str, str], list[float]] = {}
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
        # Lift v1 (bare-name) caches up to v2 (entity_id) keys so the
        # 4-way intersection isn't empty (GPT is v2, Sonnet still v1).
        # No-op on already-v2 input.  Same pattern as whitening_k_sweep
        # and the other consumer scripts post-Phase-5b.
        g_d = migrate_v1_static_scores(g_d, kinds_for_name)
        g_i = migrate_v1_static_scores(g_i, kinds_for_name)
        s_d = migrate_v1_static_scores(s_d, kinds_for_name)
        s_i = migrate_v1_static_scores(s_i, kinds_for_name)
        desc_inst = combine_desc_inst_two_judges(g_d, g_i, s_d, s_i,
                                                 weights=di_weights)

        # GPT response scores merged from traits + roles cohorts.
        responses: dict[str, float] = {}
        for mode in ("traits", "roles"):
            sub = response_subdir("gpt", mode)
            fp = axis_dir / sub / "scores_responses.json"
            if not fp.exists():
                continue
            payload, _, _ = load_and_register(
                fp, dep_key=f"judge_{axis_id}_responses_{mode}",
                inputs=inputs, policy="warn")
            for n, info in payload.items():
                if isinstance(info, dict) and info.get("mean_score") is not None:
                    responses[entity_id(n, mode)] = info["mean_score"]

        # --- Projections at each L ---
        axis_unit = axis_direction(data_dir, pos, neg, slot=slot,
                                   pair_type=ptype)

        for source_name, score_map in (("desc_inst", desc_inst),
                                       ("responses", responses)):
            rhos: list[float] = []
            for L in L_VALUES:
                proj = project_at_L(
                    entity_vecs, A_goal, A_nogoal, axis_unit, L)
                if not proj or not score_map:
                    rhos.append(float("nan"))
                    continue
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

    # --- Save JSON ---
    records = []
    for (pos, neg, source), rhos in rho_table.items():
        for L, r in zip(L_VALUES, rhos):
            records.append({"pos": pos, "neg": neg, "source": source,
                            "L": L, "rho": r})
    sweep_path = experiment_dir / args.sweep
    envelope = json_metadata(
        records,
        inputs=inputs,
        title=f"shear_l_sweep slot={slot} pairs={args.pairs}")
    json.dump(envelope, open(sweep_path, "w"), indent=2)
    print(f"\nWrote {len(records)} records to {sweep_path}")

    # --- Plot ---
    pair_keys = [(it["pos"], it["neg"]) for it in pairs]
    axis_colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(pair_keys))))

    # Taller-than-wide aspect ratio (~3:4) gives the dense
    # per-axis stack of curves more vertical room to discriminate
    # at similar-ρ axes; mirrors :mod:`results_analysis.whitening_k_sweep`.
    fig, ax = plt.subplots(figsize=(11, 14))
    # log(L+1) x-positions: same convention as the K sweep, so the
    # two PNGs are visually aligned and the parabolic-in-log(L+1)
    # shape that the K sweep also shows reads directly.
    x_pos = np.log1p(L_VALUES)
    for i, (pos, neg) in enumerate(pair_keys):
        color = axis_colors[i % 20]
        ax.plot(x_pos, rho_table[(pos, neg, "responses")], color=color, lw=2,
                linestyle="-", marker="o", markersize=5,
                label=f"{pos}/{neg} (responses)")
        ax.plot(x_pos, rho_table[(pos, neg, "desc_inst")], color=color, lw=1.5,
                linestyle=":", marker="s", markersize=4,
                label=f"{pos}/{neg} (desc+inst)")

    # Cohort-mean overlays (black) -- read the cross-axis trend at a
    # glance through the per-axis spaghetti.  Same helper / weighting
    # convention as :mod:`results_analysis.whitening_k_sweep` so the K
    # and L versions are directly comparable.
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
    ax.set_xticklabels(["raw\n(L=0)"] + [str(L_) for L_ in L_VALUES[1:]])
    ax.set_xlabel("Soft-shear truncation depth L, log(L+1) scale "
                  "(raw = no shear; higher L = more canonical-angle "
                  "pairs orthogonalised)")
    ax.set_ylabel(f"Spearman ρ (judge scores vs. projection, slot={slot})")
    title_line = f"ρ vs shear L across {len(pair_keys)} axes"
    spec_line = (f"solid = GPT responses; "
                 f"dotted = desc+inst (GPT+Sonnet averaged); "
                 f"CA subspaces = {args.ca_kind} "
                 f"(L_max = {n_pairs_max}); "
                 f"black = cohort means (solid=rs, dotted=di, dashed▲=blend)")
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
