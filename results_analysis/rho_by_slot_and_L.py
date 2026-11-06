#!/usr/bin/env python3
"""Per-slot grouped bar chart: mean Spearman ρ vs soft-shear truncation
depth L, for two judge sources (desc+inst and GPT responses).

Sister to :mod:`results_analysis.rho_by_slot_and_K`.  Where that one
sweeps the soft-K *whitening* exponent, this one sweeps the soft-shear
*truncation depth* L.  Same per-slot subplot layout, same red/blue
two-group structure, same gradient-coloured bars per L level.

For each slot ∈ {0..n_slots-1} and each L ∈ {0, 1, 2, 3, 4, 5, 6} we
compute the mean per-axis Spearman ρ between the L-shear-projected
activation projection at ``(slot, layer=25)`` and two score sources:

- **desc+inst** -- desc+instr cohort (``pair_list_di.json``), score per
  entity is combined across the four (judge, mode) sources
  ``GPT_d, GPT_i, Son_d, Son_i`` using
  :func:`assistant_axis.judge_score_combine.combine_desc_inst_two_judges`,
  with the default inst-tiebreak weighting (0.499*desc + 0.501*inst).
  Pass ``--di_weights {inst_tie,equal,desc_tie}`` to override.
- **responses** -- responses cohort (``pair_list_responses.json``),
  score per entity is the GPT-only mean over response-mode evals,
  read from the canonical ``gpt_responses_{kind}_b{N}/`` directories
  (N = :data:`assistant_axis.judge_batch.RESPONSE_BATCH_SIZE`).

L=0 means raw projection (no shear); L>0 orthogonalises the top-L
canonical-angle pairs of the goal vs no-goal subspaces fit on the
combined-kind pool from
:func:`results_analysis.canonical_angles.data.build_goal_nogoal_subspaces`.

The default L grid (0..6) deliberately stops at L=6 because the
L=3 → L=4 transition is a universal cliff across all tested slots
(see ``shear_l_sweep`` + paired t-tests; L=6 still on the post-cliff
plateau so it's a useful "saturated" reference).  Add higher L values
via ``--ls`` if you want to inspect the saturated plateau.

Inputs (from ``--experiment_dir``)
----------------------------------

Same layout as :mod:`rho_by_slot_and_K`::

    <experiment_dir>/
      pair_list_di.json
      pair_list_responses.json
      <pos>_vs_<neg>/
        gpt/scores_{descriptions,instructions}.json
        sonnet/scores_{descriptions,instructions}.json
        gpt_responses_traits_b{N}/scores_responses.json
        gpt_responses_roles_b{N}/scores_responses.json

Plus the standard activation-vectors directory passed via ``--data_dir``.

Output (to ``--experiment_dir``)
--------------------------------

- ``rho_by_slot_and_L.png`` -- the plot.

Examples
--------

::

    # Default: LS = [0..6], 35 + 12 axis pair lists
    uv run python results_analysis/rho_by_slot_and_L.py

    # Stretch into the saturated plateau
    uv run python results_analysis/rho_by_slot_and_L.py --ls 0 1 2 3 6 16 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from assistant_axis import entity_id, pair_type_of, png_metadata, response_subdir
from assistant_axis.judge_loaders import migrate_v1_static_scores
from assistant_axis.judge_score_combine import (
    add_di_weights_arg,
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
LAYER = 25  # Qwen-3-32B; matched to rho_by_slot_and_K.
DEFAULT_LS = [0, 1, 2, 3, 4, 5, 6]

# Same slot labels as rho_by_slot_and_K (keeps the two figures visually
# congruent so they can be opened side by side).
SLOT_LABELS_ALL = [
    "Slot 0 (body mean)",       "Slot 1 (<|im_start|>)",
    "Slot 2 (assistant)",       "Slot 3 (\\n)",
    "Slot 4 (<think>)",         "Slot 5 (\\n\\n in)",
    "Slot 6 (</think>)",        "Slot 7 (\\n\\n post)",
]


def _load_at(path: Path, slot: int) -> np.ndarray:
    return _load_vector_file(path).float()[slot, LAYER].numpy()


def _axis_unit_at(data_dir: Path, pos: str, neg: str, slot: int,
                  pair_type: str = "traits") -> np.ndarray:
    """Unit axis direction at (slot, LAYER); ``pair_type`` selects the
    ``traits/`` vs ``roles/`` subdir."""
    p = _load_at(data_dir / pair_type / "vectors" / f"{pos}.pt", slot)
    n = _load_at(data_dir / pair_type / "vectors" / f"{neg}.pt", slot)
    d = p - n
    return d / np.linalg.norm(d)


def _build_entity_cache(
    data_dir: Path, slots: list[int]
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, set]]:
    """Default-centered (slot, layer) entity vectors for traits + roles.

    Returns ``(cache, kinds_for_name)`` -- see
    :func:`results_analysis.rho_by_slot_and_K._build_entity_cache` for
    why ``kinds_for_name`` is bundled in.
    """
    defaults = {
        s: _load_at(data_dir / "traits" / "vectors" / "default.pt", s)
        for s in slots
    }
    cache: dict[int, dict[str, np.ndarray]] = {s: {} for s in slots}
    kinds_for_name: dict[str, set] = {}
    for et in ("traits", "roles"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            try:
                t = _load_vector_file(fp).float()
            except Exception:  # pragma: no cover -- skip unreadable
                continue
            eid = entity_id(fp.stem, et)
            for s in slots:
                v = t[s, LAYER].numpy()
                cache[s][eid] = v - defaults[s]
            kinds_for_name.setdefault(fp.stem, set()).add(et)
    return cache, kinds_for_name


class _ShearCache:
    """Cache fit_shear bases keyed by (slot, L).

    Shears are fit on the corpus-wide combined-kind goal/no-goal
    subspaces (one pair per slot), so the cache key is just
    ``(slot, L)`` -- no per-axis leave-out (unlike the K-sweep's
    augmented whitening pool, where the axis endpoints are held
    out to avoid contaminating the fit).
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._subspaces: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._cache: dict[tuple[int, int], object] = {}

    def _get_subspaces(self, slot: int) -> tuple[np.ndarray, np.ndarray]:
        if slot in self._subspaces:
            return self._subspaces[slot]
        A_g, A_n = build_goal_nogoal_subspaces(
            self.data_dir, slot=slot, layer=LAYER, kind="combined")
        self._subspaces[slot] = (A_g, A_n)
        return A_g, A_n

    def get(self, slot: int, L: int):
        if L == 0:
            return None
        key = (slot, L)
        if key in self._cache:
            return self._cache[key]
        A_g, A_n = self._get_subspaces(slot)
        try:
            basis = fit_shear(A_g, A_n, L=L)
        except ValueError:  # L > min(n_g, n_n)
            basis = None
        self._cache[key] = basis
        return basis


def _project(
    entity_vecs: dict[int, dict[str, np.ndarray]],
    shear_cache: _ShearCache,
    slot: int,
    L: int,
    axis_unit: np.ndarray,
    names: list[str],
) -> dict[str, float]:
    basis = shear_cache.get(slot, L)
    if basis is None:
        return {n: float(np.dot(entity_vecs[slot][n], axis_unit))
                for n in names if n in entity_vecs[slot]}
    axis_w = basis.apply(axis_unit[None, :])[0]
    axis_n = float(np.linalg.norm(axis_w))
    if axis_n == 0:
        return {}
    out: dict[str, float] = {}
    for n in names:
        v = entity_vecs[slot].get(n)
        if v is None:
            continue
        vw = basis.apply(v[None, :])[0]
        out[n] = float(np.dot(vw, axis_w)) / axis_n
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR),
                   help=f"Directory with pair lists and per-axis score "
                        f"subdirs (default: {DEFAULT_EXPERIMENT_DIR}).")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                   help=f"Activation vectors directory "
                        f"(default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--pairs_di", default="pair_list_di.json",
                   help="Pair-list JSON for desc+inst source "
                        "(default: pair_list_di.json -- 35 axes with "
                        "desc+inst judging from both GPT and Sonnet).")
    p.add_argument("--pairs_resp", default="pair_list_responses.json",
                   help="Pair-list JSON for GPT responses source "
                        "(default: pair_list_responses.json -- 12 axes "
                        "that additionally have GPT response judging).")
    p.add_argument("--ls", type=int, nargs="+", default=DEFAULT_LS,
                   help=f"L values to plot; 0 = raw, L>0 = soft-shear "
                        f"truncation depth (default: {DEFAULT_LS}; the "
                        f"cap at L=6 brackets the universally-significant "
                        f"L=3→L=4 cliff with two post-cliff anchor points).")
    p.add_argument("--plot", default="rho_by_slot_and_L.png",
                   help="Output plot filename "
                        "(default: rho_by_slot_and_L.png).")
    add_di_weights_arg(p)
    args = p.parse_args()
    di_weights = parse_di_weights_arg(args.di_weights)

    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    LS: list[int] = list(args.ls)

    inputs: list[InputSpec] = [
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras={"layer": str(LAYER)}),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras={"layer": str(LAYER)}),
    ]
    pairs_di, _, _ = load_and_register(
        experiment_dir / args.pairs_di,
        dep_key="pairs_di_json", inputs=inputs, policy="warn",
    )
    pairs_resp, _, _ = load_and_register(
        experiment_dir / args.pairs_resp,
        dep_key="pairs_resp_json", inputs=inputs, policy="warn",
    )
    print(f"Loaded {len(pairs_di)} desc+inst axis pairs from {args.pairs_di}")
    print(f"Loaded {len(pairs_resp)} responses axis pairs from {args.pairs_resp}")

    default_pt = data_dir / "traits" / "vectors" / "default.pt"
    n_slots = _load_vector_file(default_pt).shape[0]
    SLOTS = list(range(n_slots))
    if n_slots > len(SLOT_LABELS_ALL):
        raise SystemExit(
            f"default.pt has {n_slots} slots but only {len(SLOT_LABELS_ALL)} "
            f"entries in SLOT_LABELS_ALL; extend the constant at the top "
            f"of this file."
        )

    print(f"Caching entity vectors at {n_slots} slots...", flush=True)
    entity_vecs, kinds_for_name = _build_entity_cache(data_dir, SLOTS)
    shear_cache = _ShearCache(data_dir)

    results: dict = {"desc_inst": {}, "responses": {}}

    print("\nLoading per-axis desc+inst scores...", flush=True)
    di_scores_per_axis: dict[tuple[str, str], dict[str, float]] = {}
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
        # Lift v1 (bare-name) caches to v2 (entity_id) keys -- same
        # pattern as rho_by_slot_and_K.py, see comment there.
        g_d = migrate_v1_static_scores(g_d, kinds_for_name)
        g_i = migrate_v1_static_scores(g_i, kinds_for_name)
        s_d = migrate_v1_static_scores(s_d, kinds_for_name)
        s_i = migrate_v1_static_scores(s_i, kinds_for_name)
        di_scores_per_axis[(pos, neg)] = combine_desc_inst_two_judges(
            g_d, g_i, s_d, s_i, weights=di_weights)

    print("\nLoading per-axis response scores...", flush=True)
    rs_scores_per_axis: dict[tuple[str, str], dict[str, float]] = {}
    for it in pairs_resp:
        pos, neg = it["pos"], it["neg"]
        axis_id = f"{pos}_vs_{neg}"
        axis_dir = experiment_dir / axis_id
        scores_acc: dict[str, float] = {}
        # Same canonical-helper path as rho_by_slot_and_K -- tracks the
        # current RESPONSE_BATCH_SIZE automatically.
        for sub_label in ("traits", "roles"):
            sub = response_subdir("gpt", sub_label)
            fp = axis_dir / sub / "scores_responses.json"
            if not fp.exists():
                continue
            payload, _, _ = load_and_register(
                fp, dep_key=f"judge_{axis_id}_responses_{sub_label}",
                inputs=inputs, policy="warn",
            )
            for n, info in payload.items():
                if isinstance(info, dict) and info.get("mean_score") is not None:
                    scores_acc[entity_id(n, sub_label)] = info["mean_score"]
        if scores_acc:
            rs_scores_per_axis[(pos, neg)] = scores_acc

    print("\nComputing desc+inst ρ...", flush=True)
    for slot in SLOTS:
        for L in LS:
            rhos: list[float] = []
            for it in pairs_di:
                pos, neg = it["pos"], it["neg"]
                if (pos, neg) not in di_scores_per_axis:
                    continue
                scores = di_scores_per_axis[(pos, neg)]
                common = sorted(scores)
                au = _axis_unit_at(data_dir, pos, neg, slot,
                                   pair_type=pair_type_of(it))
                proj = _project(entity_vecs, shear_cache, slot, L, au, common)
                names = sorted(set(scores) & set(proj))
                if len(names) < 3:
                    continue
                x = np.array([scores[n] for n in names])
                y = np.array([proj[n] for n in names])
                rhos.append(spearmanr(x, y).correlation)
            mean_rho = float(np.mean(rhos)) if rhos else float("nan")
            results["desc_inst"][(slot, L)] = (mean_rho, len(rhos))
            print(f"  desc_inst slot={slot} L={L}: "
                  f"mean ρ = {mean_rho:+.4f} (n={len(rhos)})", flush=True)

    print("\nComputing responses ρ...", flush=True)
    for slot in SLOTS:
        for L in LS:
            rhos = []
            for it in pairs_resp:
                pos, neg = it["pos"], it["neg"]
                if (pos, neg) not in rs_scores_per_axis:
                    continue
                scores = rs_scores_per_axis[(pos, neg)]
                common = sorted(scores)
                au = _axis_unit_at(data_dir, pos, neg, slot,
                                   pair_type=pair_type_of(it))
                proj = _project(entity_vecs, shear_cache, slot, L, au, common)
                names = sorted(set(scores) & set(proj))
                if len(names) < 3:
                    continue
                x = np.array([scores[n] for n in names])
                y = np.array([proj[n] for n in names])
                rhos.append(spearmanr(x, y).correlation)
            mean_rho = float(np.mean(rhos)) if rhos else float("nan")
            results["responses"][(slot, L)] = (mean_rho, len(rhos))
            print(f"  responses slot={slot} L={L}: "
                  f"mean ρ = {mean_rho:+.4f} (n={len(rhos)})", flush=True)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    SLOT_LABELS = SLOT_LABELS_ALL[:n_slots]
    L_LABELS = ["raw"] + [f"L={L}" for L in LS[1:]]
    n_L = len(LS)
    L_COLORS_DI = [plt.cm.Blues(v) for v in np.linspace(0.85, 0.32, n_L)]
    L_COLORS_RS = [plt.cm.Oranges(v) for v in np.linspace(0.85, 0.32, n_L)]

    if n_slots <= 4:
        n_rows, n_cols = 1, n_slots
    elif n_slots <= 8:
        n_rows, n_cols = 2, 4
    else:  # pragma: no cover
        import math
        n_cols = int(math.ceil(math.sqrt(n_slots)))
        n_rows = int(math.ceil(n_slots / n_cols))
    fig, axes_arr = plt.subplots(
        n_rows, n_cols, figsize=(3.75 * n_cols, 5.5 * n_rows),
        sharey=True, squeeze=False,
    )
    axes = axes_arr.flatten()
    x_labels = [f"desc+inst\n({len(pairs_di)} axes)",
                f"responses\n(GPT, {len(pairs_resp)} axes)"]
    x_pos = np.arange(len(x_labels))
    bar_width = 0.78 / n_L

    y_max = 0.0
    for slot, ax in zip(SLOTS, axes):
        di_rhos = [results["desc_inst"][(slot, L)][0] for L in LS]
        rs_rhos = [results["responses"][(slot, L)][0] for L in LS]
        finite_di = [v for v in di_rhos if np.isfinite(v)]
        finite_rs = [v for v in rs_rhos if np.isfinite(v)]
        if finite_di:
            y_max = max(y_max, max(finite_di))
        if finite_rs:
            y_max = max(y_max, max(finite_rs))
        for li in range(n_L):
            offset = (li - (n_L - 1) / 2) * bar_width
            ax.bar(x_pos[0] + offset, di_rhos[li], bar_width,
                   color=L_COLORS_DI[li], edgecolor="black", linewidth=0.5)
            ax.bar(x_pos[1] + offset, rs_rhos[li], bar_width,
                   color=L_COLORS_RS[li], edgecolor="black", linewidth=0.5)
            if np.isfinite(di_rhos[li]):
                ax.text(x_pos[0] + offset, di_rhos[li] + 0.005,
                        f"{di_rhos[li]:.3f}", ha="center", va="bottom",
                        fontsize=6, rotation=90)
            if np.isfinite(rs_rhos[li]):
                ax.text(x_pos[1] + offset, rs_rhos[li] + 0.005,
                        f"{rs_rhos[li]:.3f}", ha="center", va="bottom",
                        fontsize=6, rotation=90)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(SLOT_LABELS[slot], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[n_slots:]:
        ax.set_visible(False)
    for ax in axes[:n_slots]:
        ax.set_ylim(0, y_max * 1.25)
        # Faded dotted line at the global max ρ across all (slot, mode, L)
        # cells.  Mirror of the rho_by_slot_and_K convention -- lets the
        # eye see at a glance how each (slot, source, L) bar compares to
        # the project-best operating point.
        ax.axhline(y_max, color="black", linestyle=":", alpha=0.35,
                   lw=1.0, zorder=0)

    for row_first in axes[::n_cols]:
        if row_first.get_visible():
            row_first.set_ylabel("Mean per-axis Spearman ρ", fontsize=10)

    legend_handles = []
    legend_labels = []
    for li, L_label in enumerate(L_LABELS):
        legend_handles.append(Patch(facecolor=L_COLORS_DI[li], edgecolor="black"))
        legend_labels.append(f"desc+inst, {L_label}")
        legend_handles.append(Patch(facecolor=L_COLORS_RS[li], edgecolor="black"))
        legend_labels.append(f"responses, {L_label}")
    fig.legend(legend_handles, legend_labels,
               loc="upper center", ncol=n_L, fontsize=8,
               bbox_to_anchor=(0.5, 0.96), frameon=True, framealpha=0.9,
               handlelength=1.4, columnspacing=1.2)
    title_line = (f"Mean per-axis ρ by slot, source, and soft-shear depth L "
                  f"(layer={LAYER}; desc+inst = {len(pairs_di)} axes, "
                  f"GPT+Sonnet 4-way mean; "
                  f"responses = {len(pairs_resp)} axes, GPT-only)")
    fig.suptitle(title_line, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(rect=(0, 0, 1, 0.88))

    out = experiment_dir / args.plot
    plt.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
