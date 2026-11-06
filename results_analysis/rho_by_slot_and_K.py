#!/usr/bin/env python3
"""4-panel grouped bar chart: per-slot mean Spearman ρ vs whitening level
for two judge sources (desc+inst and GPT responses).

For each slot ∈ {0, 1, 2, 3} and each K ∈ {0, 1, 2, 3, 4, 5} we compute
the mean per-axis Spearman ρ between the soft-K-whitened activation
projection at ``(slot, layer=25)`` and two score sources:

- **desc+inst** -- desc+instr cohort (``pair_list_di.json``), score per
  entity is combined across the four (judge, mode) sources
  ``GPT_d, GPT_i, Son_d, Son_i`` using
  :func:`assistant_axis.judge_score_combine.combine_desc_inst_two_judges`,
  with the default inst-tiebreak weighting (0.499*desc + 0.501*inst). Use
  ``--di_weights {inst_tie,equal,desc_tie}`` to override;
- **responses** -- responses cohort (``pair_list_responses.json``), score
  per entity is the GPT-only mean over response-mode evals.

K=0 means raw projections (no whitening); K>0 uses soft-K whitening
fit on the augmented held-out pool from
:mod:`results_analysis.canonical_angles.data` (held-out = the 2
endpoints of each axis pair).

The plot is a grouped histogram with one panel per slot, two source
groups per panel, and ``len(KS)`` bars per group (raw + soft-K).
Legend laid out as ``len(KS)`` columns × 2 rows so the (source, K)
combinations form a small explicit matrix.

Inputs (from ``--experiment_dir``)
----------------------------------

The standard per-axis directory tree produced by
:mod:`results_analysis.axis_judge_correlation`::

    <experiment_dir>/
      pair_list_di.json         # or any pair list passed via --pairs_di
      pair_list_responses.json  # or any pair list passed via --pairs_resp
      <pos>_vs_<neg>/
        gpt/scores_{descriptions,instructions}.json
        sonnet/scores_{descriptions,instructions}.json
        gpt_responses_traits/scores_responses.json   # responses cohort only
        gpt_responses_roles/scores_responses.json    # responses cohort only

Plus the standard activation-vectors directory passed via ``--data_dir``.

Output (to ``--experiment_dir``)
--------------------------------

- ``rho_by_slot_and_K.png`` -- the plot.

Examples
--------

::

    # Default: KS = [0, 1, 2, 3, 4, 5], 33 + 12 axis pair lists
    uv run python results_analysis/rho_by_slot_and_K.py

    # Custom K range
    uv run python results_analysis/rho_by_slot_and_K.py --ks 0 2 4 8 16
"""
from __future__ import annotations

import argparse
import json
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
DEFAULT_KS = [0, 1, 2, 3, 4, 5]

# Disambiguated slot labels for the 8-slot Qwen-3 layout (slots 5/7 are
# both ``\\n\\n`` text but at different chat-template positions).  For
# 4-slot data only the first 4 entries are used; the script slices
# ``SLOT_LABELS_ALL[:n_slots]`` based on what's loaded from default.pt.
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

    Returns ``(cache, kinds_for_name)`` where ``kinds_for_name`` maps
    each bare entity stem to the set of kinds it appears in.  Used
    downstream by :func:`migrate_v1_static_scores` to lift v1
    (bare-name) judge caches up to v2 (entity_id) keys -- without
    this, the Sonnet desc/inst caches (still v1 post-Phase-5b) fail
    to intersect with the GPT v2 caches and the desc+inst combined
    score comes back empty (which used to silently empty the
    desc+inst bars on the slot/K histogram).
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
            except Exception:  # pragma: no cover -- skip unreadable files
                continue
            eid = entity_id(fp.stem, et)
            for s in slots:
                v = t[s, LAYER].numpy()
                cache[s][eid] = v - defaults[s]
            kinds_for_name.setdefault(fp.stem, set()).add(et)
    return cache, kinds_for_name


class _Whitener:
    """Cache soft-K whiteners keyed by (slot, K, frozenset(leave_out))."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._cache: dict[tuple[int, int, frozenset], object] = {}

    def get(self, slot: int, K: int, leave_out_names: frozenset):
        if K == 0:
            return None
        key = (slot, K, leave_out_names)
        if key in self._cache:
            return self._cache[key]
        leave_out = set()
        for n in leave_out_names:
            leave_out.add(("traits", n))
            leave_out.add(("roles", n))
        pool_entries = build_augmented_whitening_pool(
            self.data_dir, leave_out=leave_out, scope="roles+traits")
        rows = []
        for (et, n) in pool_entries:
            sub = "combinations" if et == "combinations" else et
            path = self.data_dir / sub / "vectors" / f"{n}.pt"
            try:
                rows.append(_load_vector_file(path).float()[slot, LAYER].numpy())
            except Exception:  # pragma: no cover
                pass
        pool = np.stack(rows, axis=0)
        basis = fit_whitening("soft_K", pool, K=K)
        self._cache[key] = basis
        return basis


def _project(
    entity_vecs: dict[int, dict[str, np.ndarray]],
    whitener: _Whitener,
    slot: int,
    K: int,
    axis_unit: np.ndarray,
    names: list[str],
    leave_out_names: frozenset,
) -> dict[str, float]:
    basis = whitener.get(slot, K, leave_out_names)
    if basis is None:
        return {n: float(np.dot(entity_vecs[slot][n], axis_unit))
                for n in names if n in entity_vecs[slot]}
    axis_w = basis.apply(axis_unit[None, :])[0]
    axis_n = float(np.linalg.norm(axis_w))
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
                        "(default: pair_list_di.json -- axes with desc+inst "
                        "judging from both GPT and Sonnet).")
    p.add_argument("--pairs_resp", default="pair_list_responses.json",
                   help="Pair-list JSON for GPT responses source "
                        "(default: pair_list_responses.json -- axes that "
                        "additionally have GPT response judging).")
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS,
                   help=f"K values to plot; 0 = raw, K>0 = soft-K "
                        f"whitening (default: {DEFAULT_KS}).")
    p.add_argument("--plot", default="rho_by_slot_and_K.png",
                   help="Output plot filename "
                        "(default: rho_by_slot_and_K.png).")
    add_di_weights_arg(p)
    args = p.parse_args()
    di_weights = parse_di_weights_arg(args.di_weights)

    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    KS: list[int] = list(args.ks)

    # Provenance accumulator -- threaded through every cache read via
    # load_and_register so the read AND the InputSpec record happen
    # together (see AGENT_NOTES.md "Reader+registrar pattern").
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

    # Auto-detect num_slots from default.pt so the script handles both
    # 4-slot Christina-headers data and 8-slot Roger-8slot data.
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
    whitener = _Whitener(data_dir)

    results: dict = {"desc_inst": {}, "responses": {}}

    # Pre-load per-axis scores once via load_and_register (also
    # populates ``inputs`` with one InputSpec per consumed cache).
    # Pre-retrofit the script re-read these files on every (slot, K)
    # iteration -- functionally equivalent but wasteful, and the
    # separate post-load register loop could fall out of sync with
    # the actual reads if axes were missing.
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
        # Lift v1 (bare-name) caches to v2 (entity_id) keys in-memory
        # before combining.  Without this, the 4-way intersection
        # between GPT v2 (post-Phase-5b, ``name|R``/``name|T`` keys)
        # and Sonnet v1 (bare names, not rejudged per the plan
        # guardrail) collapses to {} and ``di_scores_per_axis`` ends
        # up empty -- which on the plot used to silently produce
        # missing desc+inst bars.  Migration is a no-op on
        # already-v2 input, so applying it uniformly is safe.
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
        # The bare-suffix ``gpt_responses_{kind}/`` dirs only contain
        # the v1-archive ``scores_responses__rubric_v1.json`` post-
        # Phase-5c; the canonical v2 GPT response data lives under
        # the batch-size-suffixed ``gpt_responses_{kind}_b{N}/``
        # directories where N = RESPONSE_BATCH_SIZE (currently 7).
        # Use the canonical helper so this script tracks the project
        # default batch size automatically.
        for sub_label in ("traits", "roles"):
            sub = response_subdir("gpt", sub_label)
            fp = axis_dir / sub / "scores_responses.json"
            if not fp.exists():
                continue
            payload, _, _ = load_and_register(
                fp,
                dep_key=f"judge_{axis_id}_responses_{sub_label}",
                inputs=inputs, policy="warn",
            )
            for n, info in payload.items():
                if isinstance(info, dict) and info.get("mean_score") is not None:
                    scores_acc[entity_id(n, sub_label)] = info["mean_score"]
        if scores_acc:
            rs_scores_per_axis[(pos, neg)] = scores_acc

    print("\nComputing desc+inst ρ...", flush=True)
    for slot in SLOTS:
        for K in KS:
            rhos: list[float] = []
            for it in pairs_di:
                pos, neg = it["pos"], it["neg"]
                if (pos, neg) not in di_scores_per_axis:
                    continue
                scores = di_scores_per_axis[(pos, neg)]
                common = sorted(scores)
                au = _axis_unit_at(data_dir, pos, neg, slot,
                                   pair_type=pair_type_of(it))
                proj = _project(entity_vecs, whitener, slot, K, au, common,
                                frozenset({pos, neg}))
                names = sorted(set(scores) & set(proj))
                if len(names) < 3:
                    continue
                x = np.array([scores[n] for n in names])
                y = np.array([proj[n] for n in names])
                rhos.append(spearmanr(x, y).correlation)
            results["desc_inst"][(slot, K)] = (float(np.mean(rhos)), len(rhos))
            print(f"  desc_inst slot={slot} K={K}: "
                  f"mean ρ = {np.mean(rhos):+.4f} (n={len(rhos)})", flush=True)

    print("\nComputing responses ρ...", flush=True)
    for slot in SLOTS:
        for K in KS:
            rhos = []
            for it in pairs_resp:
                pos, neg = it["pos"], it["neg"]
                if (pos, neg) not in rs_scores_per_axis:
                    continue
                scores = rs_scores_per_axis[(pos, neg)]
                common = sorted(scores)
                au = _axis_unit_at(data_dir, pos, neg, slot,
                                   pair_type=pair_type_of(it))
                proj = _project(entity_vecs, whitener, slot, K, au, common,
                                frozenset({pos, neg}))
                names = sorted(set(scores) & set(proj))
                if len(names) < 3:
                    continue
                x = np.array([scores[n] for n in names])
                y = np.array([proj[n] for n in names])
                rhos.append(spearmanr(x, y).correlation)
            results["responses"][(slot, K)] = (float(np.mean(rhos)), len(rhos))
            print(f"  responses slot={slot} K={K}: "
                  f"mean ρ = {np.mean(rhos):+.4f} (n={len(rhos)})", flush=True)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    SLOT_LABELS = SLOT_LABELS_ALL[:n_slots]
    K_LABELS = ["raw"] + [f"K={K}" for K in KS[1:]]
    n_K = len(KS)
    # Smooth dark→light gradient with a floor at ~0.32 so the lightest
    # shade is still readable against white.
    K_COLORS_DI = [plt.cm.Blues(v) for v in np.linspace(0.85, 0.32, n_K)]
    K_COLORS_RS = [plt.cm.Oranges(v) for v in np.linspace(0.85, 0.32, n_K)]

    # Layout scales with n_slots: 1x4 for 4 slots (matches historical
    # figure), 2x4 for 8 slots so each panel keeps its ~3.75-in width.
    if n_slots <= 4:
        n_rows, n_cols = 1, n_slots
    elif n_slots <= 8:
        n_rows, n_cols = 2, 4
    else:  # fallback
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
    bar_width = 0.78 / n_K

    y_max = 0.0
    for slot, ax in zip(SLOTS, axes):
        di_rhos = [results["desc_inst"][(slot, K)][0] for K in KS]
        rs_rhos = [results["responses"][(slot, K)][0] for K in KS]
        y_max = max(y_max, max(di_rhos), max(rs_rhos))
        for ki in range(n_K):
            offset = (ki - (n_K - 1) / 2) * bar_width
            ax.bar(x_pos[0] + offset, di_rhos[ki], bar_width,
                   color=K_COLORS_DI[ki], edgecolor="black", linewidth=0.5)
            ax.bar(x_pos[1] + offset, rs_rhos[ki], bar_width,
                   color=K_COLORS_RS[ki], edgecolor="black", linewidth=0.5)
            ax.text(x_pos[0] + offset, di_rhos[ki] + 0.005,
                    f"{di_rhos[ki]:.3f}", ha="center", va="bottom",
                    fontsize=6, rotation=90)
            ax.text(x_pos[1] + offset, rs_rhos[ki] + 0.005,
                    f"{rs_rhos[ki]:.3f}", ha="center", va="bottom",
                    fontsize=6, rotation=90)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(SLOT_LABELS[slot], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    # Hide any unused panels (e.g. n_slots=5 in a 2x4 grid).
    for ax in axes[n_slots:]:
        ax.set_visible(False)
    for ax in axes[:n_slots]:
        ax.set_ylim(0, y_max * 1.25)
        # Faded dotted line at the global max ρ across all (slot, mode, K)
        # cells.  Lets the eye see at a glance which bars in the other
        # panels reach (or fall short of) the project-best operating
        # point.  Drawn AFTER set_ylim so it sits at the right height,
        # but with low zorder so it's behind the bars.
        ax.axhline(y_max, color="black", linestyle=":", alpha=0.35,
                   lw=1.0, zorder=0)

    # First axis in each row gets the y-axis label.
    for row_first in axes[::n_cols]:
        if row_first.get_visible():
            row_first.set_ylabel("Mean per-axis Spearman ρ", fontsize=10)

    legend_handles = []
    legend_labels = []
    for ki, K_label in enumerate(K_LABELS):
        legend_handles.append(Patch(facecolor=K_COLORS_DI[ki], edgecolor="black"))
        legend_labels.append(f"desc+inst, {K_label}")
        legend_handles.append(Patch(facecolor=K_COLORS_RS[ki], edgecolor="black"))
        legend_labels.append(f"responses, {K_label}")
    fig.legend(legend_handles, legend_labels,
               loc="upper center", ncol=n_K, fontsize=8,
               bbox_to_anchor=(0.5, 0.96), frameon=True, framealpha=0.9,
               handlelength=1.4, columnspacing=1.2)
    title_line = (f"Mean per-axis ρ by slot, source, and whitening level "
                  f"(layer={LAYER}; desc+inst = {len(pairs_di)} axes, "
                  f"GPT+Sonnet 4-way mean; "
                  f"responses = {len(pairs_resp)} axes, GPT-only)")
    fig.suptitle(title_line, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(rect=(0, 0, 1, 0.88))

    # ---- Provenance inputs ----------------------------------------------
    # ``inputs`` was populated above by load_and_register at every
    # cache-read site (subtree deps + pair lists + per-axis × per-judge
    # × per-mode score caches that were actually consumed).  The
    # previous duplicate post-load loop here registered files even for
    # axes that the read pass would have skipped on FileNotFoundError;
    # dropping it makes "what we recorded" exactly equal "what we
    # consumed".

    out = experiment_dir / args.plot
    plt.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
