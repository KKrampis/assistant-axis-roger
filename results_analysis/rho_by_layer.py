#!/usr/bin/env python3
"""``n_slots`` × 2 panel plot: per-layer mean Spearman ρ vs transformer layer for
selected slots × two judge sources, with raw + soft-K whitening overlaid.

Rows are token slots (defaults include body mean, historical ``\\n`` default,
``</think>``, and ``\\n\\n`` post; see ``SLOTS``).  Columns are the
two judge sources used elsewhere in the project:

- **desc+inst** -- desc+instr cohort (``pair_list_di.json``), 4-way mean of
  ``GPT_d, GPT_i, Son_d, Son_i`` blended at the project-canonical
  ``DEFAULT_GPT_SONNET_DI_WEIGHT`` and ``DEFAULT_DI_WEIGHTS`` (see
  :mod:`assistant_axis.judge_score_combine`);
- **responses** -- responses cohort (``pair_list_responses.json``),
  GPT+Haiku response-mode ensemble blended at the canonical
  ``DEFAULT_GPT_HAIKU_Q9_WEIGHT`` (was GPT-only until 2026-05-12,
  before Haiku response data existed at the project's canonical
  batch size; switched to the v2 ``load_response_scores`` loader
  with per-entity B=7-t3 / B=10-q9 fallback for Haiku and B=7 full
  volume for GPT, mirroring the operating point of
  :mod:`response_di_weight_sweep` and
  :mod:`gpt_anthropic_response_weight_sweep --rubric v2`).

For each (slot, layer, K) we compute mean per-axis Spearman ρ between
the soft-K-whitened activation projection at ``(slot, layer)`` and the
score for each axis.  K=0 = raw (no whitening); K>0 = soft-K
whitening fit on the augmented held-out pool from
:mod:`results_analysis.canonical_angles.data` (held-out = the 2
endpoints of the axis pair).  This is the same convention as
``rho_by_slot_and_K.py`` and ``whitening_k_sweep.py``.

Default Ks are ``[0, 1, 2, 3, 4]`` (K=5 and K=6 omitted for readability)
plotted as black (raw) + rainbow spread across K=1 … K=4.

Performance note: the SVD that powers soft-K whitening is computed
*once* per (slot, layer, leave-out-set) and shared across all K
values, so adding another K to the same fit costs almost nothing.
The dominant cost is N_layers × len(SLOTS) × N_unique_pairs SVDs
(order ~10k for the default 64-layer × 4-slot × up-to-45-pair sweep:
desc+inst-cohort axes + responses-cohort axes, deduped where the same
pair appears in both lists).

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
        gpt_responses_{roles,traits}_b{N}/scores_responses.json
        haiku_responses_{roles,traits}_b7_t3/scores_responses.json
        haiku_responses_{roles,traits}_b10_q9/scores_responses.json
        # ``{N}`` = assistant_axis.judge_batch.RESPONSE_BATCH_SIZE
        # Haiku uses per-entity B fallback (b7_t3 surgical → b10_q9
        # legacy) via assistant_axis.judge_loaders.load_response_scores.

Plus the standard activation-vectors directory passed via ``--data_dir``.

Outputs (to ``--experiment_dir``)
---------------------------------

- ``rho_by_layer.png`` -- the multi-panel slot × source plot.
- ``rho_by_layer.json`` -- per-(slot, layer, K, source) mean ρ table,
  written alongside the PNG so the plot can be re-rendered cheaply
  via ``--replot_from_json``.

Examples
--------

::

    # Default: all layers, slots 0+3+6+7, Ks = [0,1,2,3,4]
    uv run python results_analysis/rho_by_layer.py

    # Faster: a coarse layer subset
    uv run python results_analysis/rho_by_layer.py --layers 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60

    # Cheap: re-render the plot from the cached JSON sidecar
    # (does NOT recompute; useful for tweaking colors / gridlines).
    uv run python results_analysis/rho_by_layer.py --replot_from_json

    # Incremental: after extending ``SLOTS`` in source, reuse existing cells
    # and compute only missing (slot, layer, K, source) tuples.
    uv run python results_analysis/rho_by_layer.py --reuse_json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from assistant_axis import (
    entity_id,
    json_metadata,
    pair_type_of,
    png_metadata,
)
from assistant_axis.judge_loaders import (
    load_response_scores,
    migrate_v1_static_scores,
)
from assistant_axis.judge_score_combine import (
    DEFAULT_GPT_HAIKU_Q9_WEIGHT,
    add_di_weights_arg,
    combine_desc_inst_two_judges,
    declare_constants_dependency,
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
    build_goal_nogoal_subspaces,
)
from results_analysis.canonical_angles.whitening import (
    WhiteningBasis,
    fit_shear,
)
from results_analysis.pair_list_guard import (  # noqa: E402
    STATIC_DI_CACHES, check_pair_list, loader_response_caches,
)


def _load_rho_json(
    path: Path,
    *,
    inputs: list[InputSpec] | None = None,
    dep_key: str = "rho_json",
) -> dict:
    """Read the rho_by_layer.json sidecar, transparently unwrapping the
    provenance envelope when present (so --reuse_json / --replot_from_json
    work against both legacy bare-JSON and post-Phase-6 wrapped files).

    When ``inputs`` is supplied, also registers ``path`` as a
    dependency under ``dep_key`` via load_and_register, so the read
    and the InputSpec record happen together.
    """
    obj, _spec, _check = load_and_register(
        path, dep_key=dep_key, inputs=inputs, policy="warn",
    )
    return obj


DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / (
    "roger/axis_judge_experiments"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / (
    "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
)
# Slots 6 (</think>) and 7 (\\n\\n post) added May 2026.  Slot 6 is the
# new winner by judge ρ on the 8-slot rejudge
# (see roger/axis_judge_experiments/rho_by_slot_and_K.png); slot 7 is
# a close second and the latest position before the model's response.
# Slot 0 (body-mean) and slot 3 (\\n historical default) kept as
# baselines for comparison.
SLOTS = [0, 3, 6, 7]
SLOT_LABELS = {
    0: "Slot 0 (body mean)",
    3: "Slot 3 (\\n)",
    6: "Slot 6 (</think>)",
    7: "Slot 7 (\\n\\n post)",
}
# Defaults bumped to include K=5 in 2026-05-12 -- K=5 turns out to be
# the discrete grid peak on the 35-axis desc+inst cohort at slot 3
# layer 25, and the additional curve costs almost nothing (the SVD
# is shared across all K values per (slot, layer, leave-out-set)).
DEFAULT_KS = [0, 1, 2, 3, 4, 5]
# Soft-shear default L grid for --regime soft_shear.  Same shape as
# K -- raw + 6 non-raw curves -- with L=3 = current project default
# (see assistant_axis.canonical_angles.whitening.DEFAULT_SOFT_SHEAR_L).
DEFAULT_LS = [0, 1, 2, 3, 4, 5, 6]


def _slot_index(slot: int) -> int:
    """Index of ``slot`` in :data:`SLOTS` (row index into sliced tensors)."""
    return SLOTS.index(slot)


def _load_full_tensor(path: Path) -> np.ndarray:
    """Load and slice an entity tensor down to (n_kept_slots, n_layers, hidden)."""
    t = _load_vector_file(path).float().numpy()  # (n_slots, n_layers, hidden)
    return t[SLOTS]  # (len(SLOTS), n_layers, hidden)


def _build_entity_cache(
    data_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, int, dict[str, set]]:
    """Pre-load (default-centered) (n_slots, n_layers, hidden) per entity.

    Returns ``(entity_vecs, default_array, n_layers, kinds_for_name)``.
    ``entity_vecs`` keys are disambiguated ``entity_id`` form (``name|R``
    / ``name|T``) for both traits and roles (excluding default).
    ``kinds_for_name`` maps each bare entity stem to the set of kinds it
    appears in -- used downstream by
    :func:`assistant_axis.judge_loaders.migrate_v1_static_scores` to lift
    v1 (bare-name) Sonnet desc/inst caches up to v2 (entity_id) keys
    so the GPT v2 + Sonnet v1 four-way intersection doesn't silently
    collapse to ``{}``.
    """
    default_path = data_dir / "traits" / "vectors" / "default.pt"
    default_arr = _load_full_tensor(default_path)
    n_layers = default_arr.shape[1]
    vecs: dict[str, np.ndarray] = {}
    kinds_for_name: dict[str, set] = {}
    for et in ("traits", "roles"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            try:
                arr = _load_full_tensor(fp)
            except Exception:  # pragma: no cover -- skip unreadable files
                continue
            vecs[entity_id(fp.stem, et)] = arr - default_arr
            kinds_for_name.setdefault(fp.stem, set()).add(et)
    return vecs, default_arr, n_layers, kinds_for_name


def _build_pool_cache(
    data_dir: Path, default_arr: np.ndarray
) -> dict[tuple[str, str], np.ndarray]:
    """Cache pool-entry tensors at the kept slots/layers, NOT default-centered.

    Pool entries come from :func:`build_augmented_whitening_pool` and may
    include ``("traits", "default")`` as a separate row.
    """
    cache: dict[tuple[str, str], np.ndarray] = {
        ("traits", "default"): default_arr,
    }
    for et in ("traits", "roles"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            try:
                cache[(et, fp.stem)] = _load_full_tensor(fp)
            except Exception:  # pragma: no cover
                continue
    return cache


def _all_K_bases(pool: np.ndarray,
                 ks: list[int]) -> dict[int, WhiteningBasis]:
    """Compute one SVD on ``pool`` and derive a soft-K basis per K.

    Returns ``{K: WhiteningBasis}``; K=0 (or K >= rank) yields a
    ``method="raw"`` basis (identity).  Sharing one SVD across all K
    values is the performance optimisation that lets this script
    sweep many K values cheaply -- the dominant cost is the SVD per
    (slot, layer, leave-out-set), so adding another K above is
    nearly free.
    """
    out: dict[int, WhiteningBasis] = {}
    if not any(k > 0 for k in ks):
        for K in ks:
            out[K] = WhiteningBasis(method="raw", K=K)
        return out
    centered = pool - pool.mean(axis=0, keepdims=True)
    _U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    for K in ks:
        if K <= 0 or K >= len(S):
            out[K] = WhiteningBasis(method="raw", K=K)
            continue
        target_sigma = float(S[K])
        scales = target_sigma / np.maximum(S[:K], 1e-12)
        out[K] = WhiteningBasis(
            method="soft_K", K=K,
            Vt=Vt[:K].astype(np.float64),
            scales=scales.astype(np.float64),
        )
    return out


def _all_L_bases(
    A_goal: np.ndarray, A_nogoal: np.ndarray, ls: list[int],
) -> dict[int, WhiteningBasis]:
    """Soft-shear bases keyed by truncation depth L.

    Mirror of :func:`_all_K_bases` for the shear regime: returns
    ``{L: WhiteningBasis}``.  L=0 (or L > min(n_g, n_n), where the
    CA-pair budget is exhausted) yields a ``method="raw"`` basis.
    Independent of any per-pair leave-out -- the shear is corpus-wide,
    so this is called once per (slot, layer) and shared across all
    axes in the per-pair loop.
    """
    n_min = min(A_goal.shape[1], A_nogoal.shape[1])
    out: dict[int, WhiteningBasis] = {}
    for L in ls:
        if L <= 0 or L > n_min:
            out[L] = WhiteningBasis(method="raw", L=L)
        else:
            out[L] = fit_shear(A_goal, A_nogoal, L=L)
    return out


def _project(
    entity_vecs: dict[str, np.ndarray],
    basis: WhiteningBasis,
    slot_i: int,
    layer: int,
    axis_unit: np.ndarray,
    names: list[str],
) -> dict[str, float]:
    """Compute ``{name: projection onto basis-whitened axis unit}``.

    Generic over the whitening regime via ``basis.apply``; ``raw``
    short-circuits to the un-whitened dot product so the inner loop
    avoids matrix multiplications when whitening is the identity.
    """
    if basis.method == "raw":
        return {n: float(np.dot(entity_vecs[n][slot_i, layer], axis_unit))
                for n in names if n in entity_vecs}
    axis_w = basis.apply(axis_unit[None, :])[0]
    axis_n = float(np.linalg.norm(axis_w))
    out: dict[str, float] = {}
    for n in names:
        v = entity_vecs.get(n)
        if v is None:
            continue
        vw = basis.apply(v[slot_i, layer][None, :])[0]
        out[n] = float(np.dot(vw, axis_w)) / axis_n
    return out


def _bases_for_pair(
    pool_cache: dict,
    pool_entries: list[tuple[str, str]],
    slot_i: int,
    layer: int,
    ks: list[int],
) -> dict[int, WhiteningBasis]:
    """Build the (slot, layer)-specific pool from pool_entries and
    return one basis per K (soft-K regime)."""
    rows = []
    for entry in pool_entries:
        arr = pool_cache.get(entry)
        if arr is None:
            continue
        rows.append(arr[slot_i, layer])
    pool = np.stack(rows, axis=0)
    return _all_K_bases(pool, ks)


def _bases_for_slot_layer_shear(
    data_dir: Path, slot: int, layer: int,
    ls: list[int], *, kind: str = "combined",
) -> dict[int, WhiteningBasis]:
    """Build the soft-shear bases at (slot, layer) for every L in ``ls``.

    The shear is fit on goal/no-goal CA subspaces, which are corpus-
    wide -- so a single (slot, layer) call covers every axis at that
    (slot, layer).  Mirror of :func:`_bases_for_pair` for the shear
    regime; called once per (slot, layer) instead of once per
    (slot, layer, pair).
    """
    A_g, A_n = build_goal_nogoal_subspaces(
        data_dir, slot=slot, layer=layer, kind=kind)
    return _all_L_bases(A_g, A_n, ls)


def _axis_unit(
    entity_vecs: dict[str, np.ndarray],
    pos_eid: str, neg_eid: str, slot_i: int, layer: int,
) -> np.ndarray:
    """Unit vector pointing pos − neg, computed from default-centered
    entity vectors (= pos.pt − neg.pt since the +/− default cancel).

    ``pos_eid`` / ``neg_eid`` are disambiguated entity ids (see
    :func:`assistant_axis.entity_id.entity_id`) — a bare ``"patient"``
    is ambiguous since both kinds carry that name.
    """
    p = entity_vecs[pos_eid][slot_i, layer]
    n = entity_vecs[neg_eid][slot_i, layer]
    d = p - n
    return d / np.linalg.norm(d)


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
                   help="Pair-list JSON for the response-ensemble source "
                        "(default: pair_list_responses.json -- axes that "
                        "have both GPT and Haiku response-mode judging; "
                        "the column shows the canonical GPT+Haiku blend "
                        "via DEFAULT_GPT_HAIKU_Q9_WEIGHT).")
    p.add_argument("--regime", choices=("soft_K", "soft_shear"),
                   default="soft_K",
                   help="Whitening family to sweep.  'soft_K' (default) "
                        "uses fit_whitening('soft_K', augmented-pool, K) "
                        "with the leave-out-the-axis-endpoints pool; "
                        "'soft_shear' uses fit_shear on the corpus-wide "
                        "goal/no-goal CA subspaces (independent of any "
                        "axis pair).  See --ks / --ls for the per-regime "
                        "grids and --ca_kind for the shear subspace "
                        "selection.  Output filename + plot title adjust "
                        "automatically (rho_by_layer_K.* vs "
                        "rho_by_layer_L.*).")
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS,
                   help=f"K values to plot in --regime soft_K; 0 = raw, "
                        f"K>0 = soft-K whitening "
                        f"(default: {DEFAULT_KS}).")
    p.add_argument("--ls", type=int, nargs="+", default=DEFAULT_LS,
                   help=f"L values to plot in --regime soft_shear; 0 = "
                        f"raw, L>0 = soft-shear truncation depth "
                        f"(default: {DEFAULT_LS}).")
    # Choices = the kinds build_goal_nogoal_subspaces accepts (the old
    # "traits"/"roles" were rejected by it; review 2026-10-12).
    p.add_argument("--ca_kind", default="combined",
                   choices=("combined", "r_only", "t_only"),
                   help="Goal/no-goal subspaces for --regime soft_shear "
                        "(passed to ``build_goal_nogoal_subspaces``).  "
                        "Ignored in --regime soft_K.")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Transformer layers to sweep (default: all "
                        "layers in the activation tensor).")
    p.add_argument("--plot", default=None,
                   help="Output plot filename "
                        "(default: rho_by_layer_K.png in --regime "
                        "soft_K, rho_by_layer_L.png in --regime "
                        "soft_shear).")
    p.add_argument("--rhos_json", default=None,
                   help="Sidecar JSON filename "
                        "(default: rho_by_layer_K.json or "
                        "rho_by_layer_L.json depending on --regime).")
    p.add_argument("--replot_from_json", action="store_true",
                   help="Skip all computation and re-render the plot "
                        "directly from --rhos_json.  Useful for tweaking "
                        "colors / gridlines without paying the SVD cost.")
    p.add_argument("--reuse_json", action="store_true",
                   help="Reuse cached (slot, layer, K, source) ρ values "
                        "from --rhos_json (filtered to the current "
                        "SLOTS, --ks, and --layers), and only compute "
                        "the missing entries.  Use this when extending "
                        "to a new slot or adding/removing K values.  The "
                        "merged result is rewritten to --rhos_json.")
    add_di_weights_arg(p)
    args = p.parse_args()
    # Qwen3's slot numbers and names, used on every run (multi-model plan § 5.6)
    from assistant_axis.axis import require_tree_header_layout
    require_tree_header_layout(
        args.data_dir, where="rho_by_layer.py",
        hint="its SLOTS (0, 3, 6, 7) and SLOT_LABELS are Qwen3's 8-slot layout")
    di_weights = parse_di_weights_arg(args.di_weights)

    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    # ``regime`` decides which CLI grid we pull from and which inner
    # loop we use; the rest of the script (score loading, cohort
    # means, plotting layout, JSON envelope schema) is shared.  The
    # variable is still called ``KS`` downstream for compatibility
    # with the JSON sidecar schema -- it just carries L values in
    # the shear regime.
    regime: str = args.regime
    KS: list[int] = list(args.ks if regime == "soft_K" else args.ls)
    regime_letter = "K" if regime == "soft_K" else "L"
    # Resolve default filenames here (before the --replot branch, which
    # also needs them to point at the right sidecar).
    if args.plot is None:
        args.plot = f"rho_by_layer_{regime_letter}.png"
    if args.rhos_json is None:
        args.rhos_json = f"rho_by_layer_{regime_letter}.json"

    if args.replot_from_json:
        json_path = experiment_dir / args.rhos_json
        print(f"Loading cached results from {json_path}", flush=True)
        # Replot's PNG depends only on the JSON sidecar; the JSON's own
        # envelope captures the upstream judge-cache chain, and audit's
        # transitive propagation will mark this PNG stale via the JSON.
        # load_and_register reads + registers in one call.
        replot_inputs: list[InputSpec] = []
        cached = _load_rho_json(json_path, inputs=replot_inputs)
        layers: list[int] = list(cached["layers"])
        # Read the regime back from the cached envelope; default to
        # "soft_K" / "K" for pre-2026-05-12 sidecars that predate the
        # regime field.
        cached_regime = str(cached.get("regime", "soft_K"))
        cached_letter = str(cached.get("regime_letter",
                                       "K" if cached_regime == "soft_K"
                                       else "L"))
        # Respect ``--ks``/``--ls`` (depending on the cached regime) so
        # replots can drop/add curves without regenerating JSON.  Falls
        # back to the cached grid if the CLI default wasn't overridden.
        cli_grid = (args.ks if cached_regime == "soft_K" else args.ls)
        default_grid = (DEFAULT_KS if cached_regime == "soft_K"
                        else DEFAULT_LS)
        KS = list(cli_grid if list(cli_grid) != default_grid
                  else cached["ks"])
        n_di = int(cached["n_axes_di"])
        n_rs = int(cached["n_axes_resp"])
        rho_by_di = {(int(slot), int(L), int(K)): float(v)
                     for slot, L, K, v in cached["desc_inst"]}
        rho_by_rs = {(int(slot), int(L), int(K)): float(v)
                     for slot, L, K, v in cached["responses"]}
        slots_replot = list(cached.get("slots", SLOTS))
        _render_plot(experiment_dir, args.plot, layers, KS, slots_replot,
                     rho_by_di, rho_by_rs, n_di, n_rs,
                     regime_letter=cached_letter,
                     inputs=replot_inputs)
        return 0

    # Provenance accumulator: every cache read below threads through
    # load_and_register so the read + register happen together.
    inputs: list[InputSpec] = [
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors"),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors"),
    ]
    # We inherit DEFAULT_GPT_HAIKU_Q9_WEIGHT (response blend) and
    # DEFAULT_GPT_SONNET_DI_WEIGHT / DEFAULT_DI_WEIGHTS (desc+inst
    # blend) from judge_score_combine; declare the file as a dependency
    # so audit_caches.py flags this plot stale on any of those
    # constants getting retuned.
    declare_constants_dependency(inputs)
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
    # Walk-through A2: every listed axis needs its pole vectors and the
    # caches read below; one missing used to drop the axis silently.
    check_pair_list(pairs_di, pair_list=experiment_dir / args.pairs_di,
                    script="rho_by_layer.py", data_dir=data_dir,
                    experiment_dir=experiment_dir, caches_all=STATIC_DI_CACHES)
    check_pair_list(pairs_resp, pair_list=experiment_dir / args.pairs_resp,
                    script="rho_by_layer.py", data_dir=data_dir,
                    experiment_dir=experiment_dir,
                    caches_any=[("GPT response cache", loader_response_caches("gpt")),
                                ("Haiku response cache", loader_response_caches("haiku"))])

    print("Loading entity tensors (all kept slots, all layers)...", flush=True)
    entity_vecs, default_arr, n_layers, kinds_for_name = _build_entity_cache(data_dir)
    pool_cache = _build_pool_cache(data_dir, default_arr)
    print(f"  cached {len(entity_vecs)} entities, n_layers={n_layers}")

    layers: list[int] = list(args.layers) if args.layers is not None \
        else list(range(n_layers))
    print(f"Sweeping {len(layers)} layers × {len(SLOTS)} slots "
          f"× {len(KS)} {regime_letter}s  (regime={regime})", flush=True)

    # Pre-cache pool entries per leave-out set (a frozenset of stems).
    # Both desc+inst pairs and responses pairs use leave_out = {pos, neg}.
    # Only used in regime=soft_K; the shear regime doesn't take a
    # per-pair leave-out (the CA subspaces are corpus-wide), so we
    # skip the pool build to save time + memory.
    leave_out_sets: dict[tuple[str, str], list[tuple[str, str]]] = {}
    if regime == "soft_K":
        for it in pairs_di + pairs_resp:
            pos, neg = it["pos"], it["neg"]
            if (pos, neg) in leave_out_sets:
                continue
            leave_out = set()
            for n in (pos, neg):
                leave_out.add(("traits", n))
                leave_out.add(("roles", n))
            leave_out_sets[(pos, neg)] = build_augmented_whitening_pool(
                data_dir, leave_out=leave_out, scope="roles+traits")

    # Index: rho[(slot, layer, K, source)] = mean ρ across axes.
    rho_by_di: dict[tuple[int, int, int], float] = {}
    rho_by_rs: dict[tuple[int, int, int], float] = {}

    if args.reuse_json:
        json_path = experiment_dir / args.rhos_json
        if json_path.exists():
            # The reused-cached path also registers as a dep so the
            # output JSON's _provenance.inputs accurately captures
            # the partial-rebuild lineage.
            cached = _load_rho_json(
                json_path, inputs=inputs, dep_key="rho_json_reuse",
            )
            keep = set((s, L, K) for s in SLOTS for L in layers for K in KS)
            for slot, L, K, v in cached.get("desc_inst", []):
                key = (int(slot), int(L), int(K))
                if key in keep:
                    rho_by_di[key] = float(v)
            for slot, L, K, v in cached.get("responses", []):
                key = (int(slot), int(L), int(K))
                if key in keep:
                    rho_by_rs[key] = float(v)
            n_cached = len(rho_by_di) + len(rho_by_rs)
            n_total = 2 * len(SLOTS) * len(layers) * len(KS)
            print(f"Reusing {n_cached}/{n_total} cached (slot, layer, K, "
                  f"source) entries from {json_path}", flush=True)
        else:
            print(f"--reuse_json set but {json_path} does not exist; "
                  "computing everything from scratch.", flush=True)

    # Pre-load per-axis score arrays (don't reread on every layer).
    # Each load_and_register call appends to ``inputs`` so the recorded
    # provenance covers exactly the caches consumed.
    print("Loading per-axis scores...", flush=True)
    # Track each pair's kind so we can resolve the pos/neg endpoints to
    # disambiguated entity_id keys when looking them up in entity_vecs.
    pair_kinds: dict[tuple[str, str], str] = {}
    axis_scores_di: dict[tuple[str, str], dict[str, float]] = {}
    for it in pairs_di:
        pos, neg = it["pos"], it["neg"]
        axis_id = f"{pos}_vs_{neg}"
        axis_dir = experiment_dir / axis_id
        try:
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
                axis_dir / "sonnet" / "scores_descriptions.json",
                dep_key=f"judge_{axis_id}_descriptions_sonnet",
                inputs=inputs, policy="warn",
            )
            s_i, _, _ = load_and_register(
                axis_dir / "sonnet" / "scores_instructions.json",
                dep_key=f"judge_{axis_id}_instructions_sonnet",
                inputs=inputs, policy="warn",
            )
        except FileNotFoundError:
            continue
        # Lift v1 (bare-name) caches to v2 (entity_id) keys in-memory
        # before combining.  Without this, GPT v2 ∩ Sonnet v1 keys = {},
        # the 4-way combine silently returns ``{}``, and the desc+inst
        # rho curves go to NaN for every (slot, layer, K) cell.  Same
        # fix pattern as the other consumer scripts; migration is a
        # no-op on already-v2 input.
        g_d = migrate_v1_static_scores(g_d, kinds_for_name)
        g_i = migrate_v1_static_scores(g_i, kinds_for_name)
        s_d = migrate_v1_static_scores(s_d, kinds_for_name)
        s_i = migrate_v1_static_scores(s_i, kinds_for_name)
        axis_scores_di[(pos, neg)] = combine_desc_inst_two_judges(
            g_d, g_i, s_d, s_i, weights=di_weights,
        )
        pair_kinds[(pos, neg)] = pair_type_of(it)
    # Responses cohort: GPT + Haiku ensemble (was GPT-only until
    # 2026-05-12; see module docstring for the switchover note).
    #
    # ``load_response_scores`` handles per-entity B fallback:
    #   * judge="gpt"   rubric="v2" -> Phase-5c B=7 full-volume cache
    #     for every entity (no fallback needed).
    #   * judge="haiku" rubric="v2" -> Phase-5d B=7-t3 surgical cache
    #     for entities that were rejudged at tiered-M=3; B=10-q9
    #     fallback for the rest.
    # Roles + traits sides are merged via ``entity_id`` so each entity
    # is keyed by its disambiguated ``"name|R"`` / ``"name|T"`` id,
    # matching the desc+inst pipeline above.
    w_gpt = DEFAULT_GPT_HAIKU_Q9_WEIGHT
    w_haiku = 1.0 - w_gpt
    axis_scores_rs: dict[tuple[str, str], dict[str, float]] = {}
    for it in pairs_resp:
        pos, neg = it["pos"], it["neg"]
        axis_id = f"{pos}_vs_{neg}"
        gpt_resp: dict[str, float] = {}
        haiku_resp: dict[str, float] = {}
        for mode in ("traits", "roles"):
            for judge, dest in (("gpt", gpt_resp), ("haiku", haiku_resp)):
                scores_by_name, _src = load_response_scores(
                    axis=axis_id, kind=mode, judge=judge, rubric="v2",
                    experiments_root=experiment_dir,
                    inputs=inputs, policy="warn",
                )
                for n, info in scores_by_name.items():
                    ms = info.get("mean_score") if isinstance(info, dict) else None
                    if ms is not None:
                        dest[entity_id(n, mode)] = float(ms)
        # Per-entity blend on the GPT∩Haiku intersection (mirrors the
        # response_di_weight_sweep and gpt_anthropic_response_weight_sweep
        # conventions).  Entities present in only one judge's cache are
        # dropped from the response score for this axis -- the
        # downstream Spearman ρ over the GPT∩Haiku∩entity_vecs
        # intersection wouldn't have seen them anyway.
        common_resp = set(gpt_resp) & set(haiku_resp)
        scores: dict[str, float] = {
            n: w_gpt * gpt_resp[n] + w_haiku * haiku_resp[n]
            for n in common_resp
        }
        if scores:
            axis_scores_rs[(pos, neg)] = scores
            pair_kinds.setdefault((pos, neg), pair_type_of(it))

    for slot in SLOTS:
        slot_i = _slot_index(slot)
        for li, layer in enumerate(layers):
            ks_missing_di = [K for K in KS
                             if (slot, layer, K) not in rho_by_di]
            ks_missing_rs = [K for K in KS
                             if (slot, layer, K) not in rho_by_rs]
            ks_to_fit = sorted(set(ks_missing_di) | set(ks_missing_rs))
            if not ks_to_fit:
                print(f"  slot={slot} layer={layer} cached "
                      f"({li + 1}/{len(layers)})", flush=True)
                continue

            # Build the per-(slot, layer) basis bank.  Two regimes:
            #
            # * soft_K: one SVD per (slot, layer, pair) on the leave-out
            #   augmented pool; all Ks share the SVD so a wider K grid
            #   costs almost nothing.  The basis is per-pair.
            # * soft_shear: one shear fit per (slot, layer) on the
            #   corpus-wide goal/no-goal CA subspaces; the basis is
            #   shared across all pairs.
            #
            # ``get_basis(pair, K_or_L) -> WhiteningBasis`` papers over
            # the difference so the inner Spearman loop is shared.
            if regime == "soft_K":
                bases_per_pair: dict[tuple[str, str],
                                     dict[int, WhiteningBasis]] = {}
                for pair, pool_entries in leave_out_sets.items():
                    bases_per_pair[pair] = _bases_for_pair(
                        pool_cache, pool_entries, slot_i, layer, ks_to_fit)
                def get_basis(pair, k):  # noqa: E306 (closure on loop var)
                    return bases_per_pair[pair][k]
            else:  # soft_shear
                shear_bases = _bases_for_slot_layer_shear(
                    data_dir, slot, layer, ks_to_fit,
                    kind=args.ca_kind)
                def get_basis(pair, k):  # noqa: E306, ARG001 (pair unused)
                    return shear_bases[k]

            for K in ks_to_fit:
                if K in ks_missing_di:
                    rhos_di: list[float] = []
                    for pair, scores in axis_scores_di.items():
                        pos, neg = pair
                        ptype = pair_kinds[pair]
                        au = _axis_unit(entity_vecs,
                                        entity_id(pos, ptype),
                                        entity_id(neg, ptype),
                                        slot_i, layer)
                        proj = _project(entity_vecs, get_basis(pair, K),
                                        slot_i, layer, au, list(scores))
                        names = sorted(set(scores) & set(proj))
                        if len(names) < 3:
                            continue
                        x = np.array([scores[n] for n in names])
                        y = np.array([proj[n] for n in names])
                        rhos_di.append(spearmanr(x, y).correlation)
                    rho_by_di[(slot, layer, K)] = float(np.mean(rhos_di))

                if K in ks_missing_rs:
                    rhos_rs: list[float] = []
                    for pair, scores in axis_scores_rs.items():
                        pos, neg = pair
                        ptype = pair_kinds[pair]
                        au = _axis_unit(entity_vecs,
                                        entity_id(pos, ptype),
                                        entity_id(neg, ptype),
                                        slot_i, layer)
                        proj = _project(entity_vecs, get_basis(pair, K),
                                        slot_i, layer, au, list(scores))
                        names = sorted(set(scores) & set(proj))
                        if len(names) < 3:
                            continue
                        x = np.array([scores[n] for n in names])
                        y = np.array([proj[n] for n in names])
                        rhos_rs.append(spearmanr(x, y).correlation)
                    rho_by_rs[(slot, layer, K)] = float(np.mean(rhos_rs))

            print(f"  slot={slot} layer={layer} done "
                  f"({li + 1}/{len(layers)}, "
                  f"fit {regime_letter}s={ks_to_fit})", flush=True)

    # ------------------------------------------------------------------
    # Provenance inputs (shared by JSON sidecar + PNG)
    # ------------------------------------------------------------------
    # ``inputs`` was populated above by load_and_register at every read
    # site (subtree deps + pair lists + per-axis × per-judge × per-mode
    # judge caches).  Pre-retrofit a duplicate post-load loop here
    # registered all axes regardless of whether their caches were
    # actually consumed (axes with FileNotFoundError were silently
    # skipped by the read pass but recorded as deps anyway).

    # ------------------------------------------------------------------
    # JSON sidecar (so future replots are cheap)
    # ------------------------------------------------------------------
    json_out = {
        "layers": list(layers),
        "regime": regime,
        "regime_letter": regime_letter,
        "ks": list(KS),
        "slots": list(SLOTS),
        "n_axes_di": len(axis_scores_di),
        "n_axes_resp": len(axis_scores_rs),
        "desc_inst": [
            [int(slot), int(L), int(K), float(rho_by_di[(slot, L, K)])]
            for slot in SLOTS for L in layers for K in KS
        ],
        "responses": [
            [int(slot), int(L), int(K), float(rho_by_rs[(slot, L, K)])]
            for slot in SLOTS for L in layers for K in KS
        ],
    }
    json_path = experiment_dir / args.rhos_json
    envelope = json_metadata(
        json_out,
        inputs=inputs,
        title=f"rho_by_layer regime={regime} slots={SLOTS} "
              f"{regime_letter}s={KS}",
    )
    json.dump(envelope, open(json_path, "w"), indent=2)
    print(f"Wrote {json_path}")

    _render_plot(experiment_dir, args.plot, layers, KS, SLOTS,
                 rho_by_di, rho_by_rs,
                 len(axis_scores_di), len(axis_scores_rs),
                 regime_letter=regime_letter,
                 inputs=inputs)
    return 0


def _render_plot(experiment_dir: Path, plot_name: str,
                 layers: list[int], KS: list[int], slots: list[int],
                 rho_by_di: dict, rho_by_rs: dict,
                 n_axes_di: int, n_axes_resp: int,
                 *,
                 regime_letter: str = "K",
                 inputs: list[InputSpec] | None = None) -> None:
    """Pure plotting from precomputed ρ tables.  Shared by the main
    pipeline and the ``--replot_from_json`` fast path.

    ``inputs`` (when provided) is embedded in the PNG's ``Inputs``
    chunk so audit_pngs.py can validate freshness against the recorded
    judge caches / dataset subtrees.  The main pipeline passes the
    full upstream input list; the ``--replot_from_json`` fast path
    passes a single-element list pointing to the JSON sidecar (whose
    own envelope captures the upstream chain transitively).
    """
    nonraw = [k for k in KS if k != 0]
    # Spread hues across len(nonraw) so K=5 (or L=6) widens the
    # rainbow band rather than overflowing it.  Auto-adapts to any
    # K / L grid the caller passes.
    rainbow = (plt.cm.rainbow(np.linspace(0.0, 1.0, len(nonraw)))
               if nonraw else [])

    def k_color(K: int):
        if K == 0:
            return "black"
        return rainbow[nonraw.index(K)]

    def k_label(K: int) -> str:
        return "raw" if K == 0 else f"{regime_letter}={K}"

    # Draw high-K/L whitening first, low-K/L later, raw last so lower
    # K/L and raw sit visually on top.  In the default grids
    # ([0,1,2,3,4,5] or [0,1,2,3,4,5,6]) this draws K=5/L=6 first, K=1/L=1
    # last among the non-raw curves, then raw on the very top.
    ks_plot_order = sorted(nonraw, reverse=True) + ([0] if 0 in KS else [])

    n_slots = len(slots)
    # Layout: rows = slots, cols = sources (2).  Height scales with the
    # slot count so each panel keeps its ~4.75-in vertical room.
    fig, axes = plt.subplots(n_slots, 2, figsize=(13, 4.75 * n_slots),
                             sharex=True, sharey=True, squeeze=False)
    sources = [
        ("desc_inst", rho_by_di, f"desc+inst ({n_axes_di} axes)"),
        ("responses", rho_by_rs,
         f"responses (GPT+Haiku, {n_axes_resp} axes)"),
    ]
    L_min, L_max = min(layers), max(layers)
    even_layers = list(range(L_min - L_min % 2, L_max + 1, 2))
    # Column-wise "best" reference lines: drawn at the highest ρ in
    # the entire (slot × layer × K-or-L) cube for that column.  Same
    # value on every panel of a column; sharey=True so the line is
    # guaranteed to sit within view.  Helps the eye anchor on "how
    # close is this slot×layer×K to the cohort-wide best ρ?".  Only
    # the top-left panel gets the matplotlib legend entry (the rest
    # show the same line without a label); the column header makes
    # which value applies clear.
    best_by_col = [
        max(rho_dict.values()) if rho_dict else float("nan")
        for _src_id, rho_dict, _src_title in sources
    ]

    for row, slot in enumerate(slots):
        for col, (_src_id, rho_dict, src_title) in enumerate(sources):
            ax = axes[row, col]
            for K in ks_plot_order:
                ys = [rho_dict[(slot, L, K)] for L in layers]
                ax.plot(layers, ys, color=k_color(K),
                        lw=1.0 if K == 0 else 0.7,
                        marker="o", markersize=1.5, alpha=0.95,
                        label=k_label(K))
            best_val = best_by_col[col]
            if not np.isnan(best_val):
                # Label the line only on the panel that hosts the
                # legend; the same line on the other panels stays
                # legend-less (matplotlib treats label=None as "skip").
                hline_label = ("best" if (row == 0 and col == 0)
                               else None)
                ax.axhline(best_val, color="gray", linestyle=":",
                           lw=1.0, alpha=0.45, zorder=0,
                           label=hline_label)
            # Faint vertical guideline every 2 layers (minor grid),
            # plus the standard major grid for orientation.
            ax.set_xticks(even_layers, minor=True)
            ax.grid(which="major", alpha=0.3)
            ax.grid(which="minor", axis="x", alpha=0.12, lw=0.5)
            ax.set_title(f"{SLOT_LABELS[slot]} -- {src_title}", fontsize=10)
            if row == n_slots - 1:
                ax.set_xlabel("Transformer layer", fontsize=10)
            if col == 0:
                ax.set_ylabel("Mean per-axis Spearman ρ", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, ncol=2, loc="best",
                          framealpha=0.9, handlelength=1.6)

    regime_name = "soft-K whitening" if regime_letter == "K" else "soft-shear"
    title_line = (f"Mean per-axis ρ vs transformer layer, by slot × source "
                  f"× {regime_name} level "
                  f"(desc+inst = {n_axes_di} axes, GPT+Sonnet 4-way mean; "
                  f"responses = {n_axes_resp} axes, GPT+Haiku ensemble)")
    fig.suptitle(title_line, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    out = experiment_dir / plot_name
    plt.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
