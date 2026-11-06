#!/usr/bin/env python3
"""Ablate the desc/inst tiebreak weighting choice
(``inst_tie`` = 0.499/0.501, ``equal`` = 0.500/0.500, ``desc_tie`` =
0.501/0.499) at the project default operating point.

Re-confirms / re-tunes :data:`assistant_axis.judge_score_combine.DEFAULT_DI_WEIGHTS`
under whatever whitening the caller asks for.  Mirrors the historical
ablation that picked ``inst_tie`` as default, but with two upgrades:

* **Configurable projection**.  The historical ablation was on
  raw projections (``--whitening raw``).  Today's project default is
  soft-shear at ``L = DEFAULT_SOFT_SHEAR_L = 3``, so this script
  defaults to ``--whitening soft_shear=3`` so re-tuning happens at the
  geometry consumers will actually use downstream.

* **v2 cohort handling**.  The 4 desc/inst inputs are lifted to
  ``entity_id`` keys before combining (Phase-5b disambiguation) so
  the GPT-v2 × Sonnet-v1 mismatch doesn't silently drop the 9
  collisions.

The script reports, for each of the 3 weight choices:

* Per-axis Spearman ρ.
* Cohort-mean ρ across the 35-axis di cohort.
* Cohort-mean ρ across the 12-axis responses cohort (the same axes the
  judge ensemble's response-mode operates on, so the choice carries
  through to the downstream blends).
* Δρ vs the current default (``inst_tie``).

CLI examples::

    # Default re-tune at slot 6 layer 25, L=3 sheared:
    uv run python -m results_analysis.di_weights_ablation

    # Reproduce the historical ablation at raw projection:
    uv run python -m results_analysis.di_weights_ablation --whitening raw

    # Sweep at slot 3 instead:
    uv run python -m results_analysis.di_weights_ablation --slot 3
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
    entity_id,
    json_metadata,
    pair_type_of,
    png_metadata,
)
from assistant_axis.judge_loaders import migrate_v1_static_scores
from assistant_axis.judge_score_combine import (
    DI_WEIGHT_CHOICES,
    combine_desc_inst_two_judges,
    declare_constants_dependency,
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
    DEFAULT_SOFT_SHEAR_L,
    fit_shear,
    fit_whitening,
)


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
    "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
)
LAYER = 25
DEFAULT_SLOT = 6
_SCRIPT_PATH = Path(__file__).resolve()


def _load_at(path: Path, slot: int) -> np.ndarray:
    return _load_vector_file(path).float()[slot, LAYER].numpy()


def _axis_unit_at(data_dir: Path, pos: str, neg: str, slot: int,
                  pair_type: str = "traits") -> np.ndarray:
    p = _load_at(data_dir / pair_type / "vectors" / f"{pos}.pt", slot)
    n = _load_at(data_dir / pair_type / "vectors" / f"{neg}.pt", slot)
    d = p - n
    return d / np.linalg.norm(d)


def _build_entity_cache(
    data_dir: Path, slot: int,
) -> tuple[dict[str, np.ndarray], dict[str, set]]:
    """Per-entity (default-centered) activation vectors at the given
    slot + corpus kinds_for_name map for v1→v2 migration."""
    default = _load_at(data_dir / "traits" / "vectors" / "default.pt", slot)
    vecs: dict[str, np.ndarray] = {}
    kinds_for_name: dict[str, set] = {}
    for et in ("traits", "roles"):
        vdir = data_dir / et / "vectors"
        if not vdir.is_dir():
            continue
        for fp in sorted(vdir.glob("*.pt")):
            if fp.stem == "default":
                continue
            try:
                v = _load_vector_file(fp).float()[slot, LAYER].numpy()
            except Exception:  # pragma: no cover -- skip unreadable
                continue
            vecs[entity_id(fp.stem, et)] = v - default
            kinds_for_name.setdefault(fp.stem, set()).add(et)
    return vecs, kinds_for_name


def _build_projector(
    whitening: str,
    *,
    data_dir: Path,
    slot: int,
    entity_vecs: dict[str, np.ndarray],
):
    """Return a callable ``project(axis_unit, names) -> dict``.

    ``whitening`` accepts:
    * ``"raw"`` -- identity projection (np.dot(v, axis_unit)).
    * ``"soft_shear=N"`` -- fit_shear on the combined goal/no-goal
      subspaces, L=N.
    * ``"soft_K=N"`` -- fit_whitening on the augmented entity pool
      with K=N.  NOTE: requires a per-axis leave-out which this
      ablation doesn't currently thread; raise ValueError for now.
    """
    if whitening == "raw":
        def project(axis_unit: np.ndarray, names: list[str]) -> dict[str, float]:
            return {n: float(np.dot(entity_vecs[n], axis_unit))
                    for n in names if n in entity_vecs}
        return project

    if whitening.startswith("soft_shear="):
        L = int(whitening.split("=", 1)[1])
        if L == 0:
            return _build_projector("raw", data_dir=data_dir, slot=slot,
                                     entity_vecs=entity_vecs)
        A_g, A_n = build_goal_nogoal_subspaces(
            data_dir, slot=slot, layer=LAYER, kind="combined")
        basis = fit_shear(A_g, A_n, L=L)

        def project(axis_unit: np.ndarray, names: list[str]) -> dict[str, float]:
            axw = basis.apply(axis_unit[None, :])[0]
            axw_n = float(np.linalg.norm(axw))
            if axw_n == 0:
                return {}
            out: dict[str, float] = {}
            for n in names:
                v = entity_vecs.get(n)
                if v is None:
                    continue
                vw = basis.apply(v[None, :])[0]
                out[n] = float(np.dot(vw, axw)) / axw_n
            return out
        return project

    if whitening.startswith("soft_K="):
        raise ValueError(
            "soft_K projection in di_weights_ablation requires a "
            "per-axis leave-out (which this script doesn't currently "
            "thread).  Use soft_shear=N or raw, or extend this "
            "projector to consume rho_by_slot_and_K's pool helpers."
        )
    raise ValueError(f"Unknown --whitening {whitening!r}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR))
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--pairs_di", default="pair_list_di.json",
                   help="Desc+inst cohort pair list (default: 35 axes).")
    p.add_argument("--pairs_resp", default="pair_list_responses.json",
                   help="Responses cohort pair list (default: 12 axes, "
                        "subset of pairs_di -- used for the secondary "
                        "table column to track the di-tiebreak's effect "
                        "on the same 12 axes that feed the response-mode "
                        "ratio retunings).")
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT,
                   help=f"Slot to project at (default: {DEFAULT_SLOT}).")
    p.add_argument("--whitening", type=str,
                   default=f"soft_shear={DEFAULT_SOFT_SHEAR_L}",
                   help=f"Projection regime (default: soft_shear="
                        f"{DEFAULT_SOFT_SHEAR_L}, the project default).  "
                        f"Accepted: 'raw', 'soft_shear=N'.  The historical "
                        f"ablation that picked inst_tie used 'raw'; this "
                        f"script defaults to the canonical whitening "
                        f"because the di-tiebreak feeds downstream "
                        f"analyses that run at L="
                        f"{DEFAULT_SOFT_SHEAR_L}.")
    p.add_argument("--plot", default="di_weights_ablation.png",
                   help="Output PNG filename inside --experiment_dir.")
    p.add_argument("--json", default="di_weights_ablation.json",
                   help="Output JSON filename inside --experiment_dir.")
    args = p.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    slot = int(args.slot)
    whitening = args.whitening

    # Provenance accumulator.
    inputs: list[InputSpec] = [
        current_file_input(
            dep_key="producer_script",
            path=_SCRIPT_PATH,
            extras={
                "slot": str(slot),
                "layer": str(LAYER),
                "whitening": whitening,
            },
        ),
    ]
    declare_constants_dependency(inputs)
    inputs.extend([
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras={"slot": str(slot), "layer": str(LAYER)}),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras={"slot": str(slot), "layer": str(LAYER)}),
    ])

    pairs_di, _, _ = load_and_register(
        experiment_dir / args.pairs_di,
        dep_key="pairs_di_json", inputs=inputs, policy="warn",
    )
    pairs_resp, _, _ = load_and_register(
        experiment_dir / args.pairs_resp,
        dep_key="pairs_resp_json", inputs=inputs, policy="warn",
    )
    primary_set = {(it["pos"], it["neg"]) for it in pairs_resp}
    print(f"Loaded {len(pairs_di)} di axes, {len(pairs_resp)} response axes "
          f"(primary subset).")
    print(f"Projection regime: {whitening}  (slot={slot}, layer={LAYER})")

    print("\nBuilding entity vectors + projector...")
    entity_vecs, kinds_for_name = _build_entity_cache(data_dir, slot)
    project = _build_projector(
        whitening, data_dir=data_dir, slot=slot, entity_vecs=entity_vecs)

    # Per-axis: load g_d, g_i, s_d, s_i; for each weight choice, compute rho.
    print("\nPer-axis ρ by weight choice...")
    per_axis: dict[tuple[str, str], dict[str, float]] = {}
    choices = ["inst_tie", "equal", "desc_tie"]
    for it in pairs_di:
        pos, neg = it["pos"], it["neg"]
        axis_id = f"{pos}_vs_{neg}"
        axis_dir = experiment_dir / axis_id
        try:
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
        except FileNotFoundError:
            continue
        # Lift v1 bare-name to v2 entity_id keys.
        g_d = migrate_v1_static_scores(g_d, kinds_for_name)
        g_i = migrate_v1_static_scores(g_i, kinds_for_name)
        s_d = migrate_v1_static_scores(s_d, kinds_for_name)
        s_i = migrate_v1_static_scores(s_i, kinds_for_name)
        au = _axis_unit_at(data_dir, pos, neg, slot,
                           pair_type=pair_type_of(it))

        per_axis_rhos: dict[str, float] = {}
        for choice in choices:
            weights = DI_WEIGHT_CHOICES[choice]
            di = combine_desc_inst_two_judges(g_d, g_i, s_d, s_i,
                                              weights=weights)
            if not di:
                per_axis_rhos[choice] = float("nan")
                continue
            common = sorted(di)
            proj = project(au, common)
            names = sorted(set(di) & set(proj))
            if len(names) < 3:
                per_axis_rhos[choice] = float("nan")
                continue
            x = np.array([di[n] for n in names], dtype=float)
            y = np.array([proj[n] for n in names], dtype=float)
            per_axis_rhos[choice] = float(spearmanr(x, y).correlation)
        per_axis[(pos, neg)] = per_axis_rhos
        print(f"  {pos:25s} vs {neg:25s}  "
              + "  ".join(f"{c}={per_axis_rhos[c]:+.4f}" for c in choices))

    # Cohort means.
    def _cohort_mean(pairs):
        out: dict[str, float] = {}
        for c in choices:
            vals = [per_axis[(p['pos'], p['neg'])][c]
                    for p in pairs
                    if (p['pos'], p['neg']) in per_axis
                    and np.isfinite(per_axis[(p['pos'], p['neg'])].get(c, float("nan")))]
            out[c] = float(np.mean(vals)) if vals else float("nan")
        return out

    mean_di = _cohort_mean(pairs_di)
    mean_resp = _cohort_mean([{"pos": p, "neg": n} for (p, n) in primary_set])
    baseline = "inst_tie"

    print("\n=== Cohort means by di-weights choice ===")
    print(f"projection: {whitening}  slot={slot} layer={LAYER}")
    print(f"{'choice':<10s}  {'(d, i)':<14s}  "
          f"{'di mean (35 axes)':>20s}  {'Δ vs inst_tie':>14s}  "
          f"{'resp subset (12)':>18s}  {'Δ vs inst_tie':>14s}")
    for c in choices:
        d, i = DI_WEIGHT_CHOICES[c]
        delta_di = mean_di[c] - mean_di[baseline]
        delta_resp = mean_resp[c] - mean_resp[baseline]
        print(
            f"{c:<10s}  ({d:.3f}, {i:.3f})  "
            f"{mean_di[c]:+.6f}  "
            f"{delta_di:+.6f}  "
            f"{mean_resp[c]:+.6f}  "
            f"{delta_resp:+.6f}"
        )

    # Save JSON.
    payload = {
        "slot": slot, "layer": LAYER, "whitening": whitening,
        "choices": [
            {"name": c, "weights": list(DI_WEIGHT_CHOICES[c]),
             "mean_rho_di_cohort": mean_di[c],
             "mean_rho_resp_subset": mean_resp[c]}
            for c in choices
        ],
        "per_axis": [
            {"pos": pos, "neg": neg,
             **{c: per_axis[(pos, neg)][c] for c in choices}}
            for (pos, neg) in sorted(per_axis)
        ],
    }
    json_path = experiment_dir / args.json
    envelope = json_metadata(
        payload, inputs=inputs,
        title=f"di_weights_ablation slot={slot} whitening={whitening}")
    json_path.write_text(json.dumps(envelope, indent=2))
    print(f"\nWrote {json_path}")

    # Plot: 2-panel bar chart (di cohort + responses subset).
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=False)
    x_pos = np.arange(len(choices))
    colors = ["#3471a8", "#888888", "#d4663c"]  # inst-tie / equal / desc-tie
    for ax, cohort_mean, label, n in (
        (axes[0], mean_di, "desc+inst cohort", len(pairs_di)),
        (axes[1], mean_resp, "responses subset", len(pairs_resp)),
    ):
        vals = [cohort_mean[c] for c in choices]
        for xi, (c, v) in enumerate(zip(choices, vals)):
            ax.bar(xi, v, 0.7, color=colors[xi], edgecolor="black",
                   linewidth=0.6)
            ax.text(xi, v + 0.001, f"{v:+.5f}", ha="center", va="bottom",
                    fontsize=8, rotation=0)
        # Δ vs baseline annotations under each bar
        for xi, c in enumerate(choices):
            delta = cohort_mean[c] - cohort_mean[baseline]
            ax.text(xi, -0.015, f"Δ={delta:+.5f}", ha="center", va="top",
                    fontsize=7, transform=ax.get_xaxis_transform(),
                    color="#444444")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([
            f"inst_tie\n(0.499, 0.501)",
            f"equal\n(0.500, 0.500)",
            f"desc_tie\n(0.501, 0.499)",
        ], fontsize=9)
        ax.set_title(f"{label} ({n} axes)", fontsize=11)
        ax.set_ylabel("Mean per-axis Spearman ρ", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        y_min = min(vals) - 0.005
        y_max = max(vals) + 0.008
        ax.set_ylim(y_min, y_max)
        # Faded reference line at baseline (inst_tie).
        ax.axhline(cohort_mean[baseline], color="black",
                   linestyle=":", alpha=0.35, lw=1.0, zorder=0)
    fig.suptitle(
        f"DI tiebreak ablation at projection={whitening}, "
        f"slot={slot} layer={LAYER}\n"
        f"(re-confirms DEFAULT_DI_WEIGHTS choice; baseline = inst_tie, "
        f"dotted line)",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plot_path = experiment_dir / args.plot
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(
                    title=f"di_weights_ablation slot={slot} "
                          f"whitening={whitening}",
                    inputs=inputs))
    plt.close(fig)
    print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
