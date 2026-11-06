#!/usr/bin/env python3
"""Adaptive (L, K, M) sweep for PC round-trip ρ.

DEPRECATED (May 2026) — see deprecation note below.  The (L, K, M, cell)
optimization machinery in this file is retained so the legacy plots can
still be regenerated, but it is no longer the headline analysis.  The
default ``plot_direction_cosines.py`` view skips this sweep entirely
(canonical-only: principled α-coefficient transport at L=K=0, evaluated
at the canonical cell and the 5 other cells, against shuffled-judge nulls
also at L=K=0).

Why deprecated
--------------
At high N the (L, K, cell) search reliably surfaces correlations that
match the LLM judge description but are NOT actually aligned with the
original canonical Nth PC direction.  Concretely: increasing K mixes
into the transported direction the structure of the first K target-cell
PCs, and once that mixture matches some sample-noise ranking that
loosely resembles the description, ρ goes up.  The peak ρ tells us
something about the description × low-rank target structure, not about
faithful round-trip recovery of the canonical PC.  Holding (L, K, cell)
rigidly fixed at the canonical point is the only honest read.

This file's primitives (``compute_M_done``, ``compute_pc_directions``,
``rho_for_direction``, ``load_combined_scores``, etc.) are still used
by the canonical-only plot path and are NOT deprecated.  Only the
``stage1_*`` / ``stage2_*`` (L, K, M)-search functions and the CLI
sweep-orchestration logic are deprecated.

Reads the cached judge scores produced by ``launch_judge_runs.py`` (which
runs the auto-described axis through ``axis_judge_correlation.py``), and
for each (PC, style) cell finds the best round-trip Spearman ρ across:

- ``slot, layer`` ∈ a small set (default: the three configs that dominate
  the per-PC winners — (3, 25), (0, 26), (0, 49)),
- ``L`` -- soft-shear truncation depth (default 0..5),
- ``K`` -- soft-K whitening depth, swept on a coarse log-spaced grid then
  refined with bracket-and-bisect around the peak,
- ``M`` -- optional truncation of M_done into the top-M PCs of the pool
  (default {64, 128, 256, 512, None=∞}).

Two stages:

1. **Stage 1 (M sweep)** — explore the M dimension at every coarse
   ``(L, K)`` grid point; cells where ``M < PC`` are dropped (the axis
   would be erased).

2. **Stage 2 (K refinement at M=∞)** — for each ``(slot, layer, L)`` and
   each ``(PC, style)``, identify the peak K on the coarse grid, bracket
   ``[K_left, K_right]``, insert two new K values evenly inside, re-evaluate,
   and recurse.  Stops when bracket width ≤ ``K_REFINE_MIN_BRACKET`` or the
   ρ-improvement falls below ``K_REFINE_TIE_THRESH``.

The discrete bracket-and-bisect ("ternary insertion") is the discrete
cousin of golden-section search; with an SE ≈ 1/√n ≈ 0.04 noise floor we
treat ρ-differences below ~0.005 as ties.

K_COARSE goes up to 512.  At very high K the soft-K-whitened matrix is
near-rank-deficient and the default LAPACK gesdd driver can fail to
converge -- the script falls back to scipy's gesvd driver (slower but
robust) and skips configs where even gesvd fails.

Output
------

A JSON file at ``--cache_path`` with two top-level dicts keyed by
``"pc{NNN}_{style}"`` -- one for stage 1 (best at each (L, K, M)) and one
for stage 2 (refined K at M=∞).  Each entry records the slot, layer, L,
K, (M for stage 1), and ρ for the winning cell.

This script does **no** API calls -- it just reads the cached judge
scores and runs CPU.  Re-running is cheap (~4 minutes on a workstation).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

from assistant_axis import entity_id
from assistant_axis.judge_score_combine import (
    DI_WEIGHT_CHOICES, combine_desc_inst_two_judges,
)
from assistant_axis.plot_metadata import json_metadata
from assistant_axis.provenance import (
    InputSpec, current_data_subtree_input, load_and_register,
)
from results_analysis.axis_judge_correlation import _load_vector_file
from results_analysis.canonical_angles.data import (
    DEFAULT_DATA_DIR,
    build_augmented_whitening_pool,
    build_goal_nogoal_subspaces,
    load_vector,
)
from results_analysis.canonical_angles.whitening import fit_shear, fit_whitening

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------

# 7-cell default for the round-trip experiment.  Order matters — the plot's
# n-cell aggregates take prefixes of this list (with n ∈ {1, 2, 4, 6, 7}).
# Canonical sits at the head; slot 6 and slot 0 layers come next; slot 3 is
# included at the tail as the "old canonical" diagnostic point.
#
# May 2026 migration: the canonical was moved from slot 3 (4-slot dataset,
# L=2 shear) to slot 7 (8-slot dataset, raw / L=0).  See AGENT_NOTES "PC
# round-trip experiment — May 2026 migration to 8-slot, slot 7, L=0 raw"
# for context.
DEFAULT_CONFIGS: list[tuple[int, int]] = [
    (7, 25), (7, 49), (6, 25), (6, 49), (0, 26), (0, 49), (3, 25),
]
# L grid: extends past the historical {0..5} now that fixed_principled L
# winners spread across the full range with no clear preference.  Both grids
# include the no-shear/no-whiten baselines (L=0, K=0) — empirically those
# are the most common winners for fixed_principled.
DEFAULT_L_VALUES: list[int] = [0, 1, 2, 4]
DEFAULT_K_COARSE: list[int] = [0, 1, 2, 4]
DEFAULT_M_VALUES: list[Optional[int]] = [64, 128, 256, 512, None]
DEFAULT_PCS: list[int] = [
    1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56,
    64, 80, 96, 128, 192, 256, 384, 512,
]
DEFAULT_STYLES: list[str] = ["glossary", "inline"]

K_REFINE_MAX_ITER = 5
K_REFINE_MIN_BRACKET = 3      # Stop when bracket width <= this (integer K spacing)
K_REFINE_TIE_THRESH = 0.005   # ρ-difference below this counts as a tie

# Note: the K-near-N dense-neighbourhood expansion (``expanded_k_grid``,
# ``K_NEAR_N_WIDTH``, ``k_neighborhood_for_pc``) was removed 2026-05-06.
# It existed to honestly evaluate the K=N spike for the ``nth_pc`` variant.
# The plot is now fixed_principled-only and that variant doesn't show K=N
# spike behaviour — its K winners cluster at the bottom of the grid (mostly
# K ≤ 13 in the 3-cell sweep).  The bracket-and-bisect refinement on the
# coarse grid is sufficient.


DEFAULT_SWEEP_DIR = "roger/pc_axis_describer_sweep"
DEFAULT_CACHE_PATH = "roger/pc_round_trip_klm_results.json"


# ---------------------------------------------------------------------------
# Setup / data loading
# ---------------------------------------------------------------------------

def setup_at(data_dir: Path, slot: int, layer: int):
    default_v = np.asarray(
        load_vector(data_dir, "traits", "default"), dtype=np.float32
    )[slot, layer]
    names, rows = [], []
    for et in ("roles", "traits"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            names.append(entity_id(fp.stem, et))
            v = _load_vector_file(fp).float().numpy()[slot, layer]
            rows.append(v - default_v)
    M_raw = np.stack(rows, axis=0).astype(np.float32)

    A_g, A_n = build_goal_nogoal_subspaces(data_dir, slot=slot, layer=layer,
                                            kind="combined")

    pool_entries = build_augmented_whitening_pool(data_dir, leave_out=frozenset())
    pool_rows = []
    for et, name in pool_entries:
        v = np.asarray(load_vector(data_dir, et, name), dtype=np.float32)[slot, layer]
        pool_rows.append(v - default_v)
    pool = np.stack(pool_rows, axis=0).astype(np.float32)

    return names, M_raw, A_g, A_n, pool


def _load_score_json(
    path: Path,
    *,
    dep_key: str | None = None,
    inputs: list[InputSpec] | None = None,
) -> dict[str, float]:
    """Load a judge-score JSON, transparently unwrapping the
    ``{"_provenance": ..., "result": ...}`` envelope written by recent
    ``axis_judge_correlation.py`` runs.  Older score files (pre-envelope)
    are bare ``{name: score}`` dicts and are returned unchanged.

    When an ``inputs`` accumulator is supplied, the file is also
    registered as a dependency under ``dep_key`` (defaulting to the
    file's basename).  Threads through ``load_and_register`` so the
    read and the InputSpec record are built together.
    """
    data, _spec, _check = load_and_register(
        path,
        dep_key=dep_key or f"score_json:{path.name}",
        inputs=inputs,
        policy="warn",
    )
    if not isinstance(data, dict):
        raise ValueError(f"unexpected score JSON shape at {path}: {type(data).__name__}")
    return data


def load_combined_scores(
    cell: Path,
    *,
    inputs: list[InputSpec] | None = None,
    cell_id: str | None = None,
) -> dict[str, float]:
    """Combine GPT + Sonnet × descriptions + instructions scores using the
    project default ``inst_tie`` weights (≈ equal weight, fall back to
    instructions on ties).

    When ``inputs`` is supplied, each of the four score caches consumed
    is registered as a dependency under
    ``judge_<cell_id>_<mode>_<judge>``; ``cell_id`` defaults to the
    cell directory's basename.
    """
    label = cell_id or cell.name
    g_d = _load_score_json(
        cell / "gpt" / "scores_descriptions.json",
        dep_key=f"judge_{label}_descriptions_gpt", inputs=inputs)
    g_i = _load_score_json(
        cell / "gpt" / "scores_instructions.json",
        dep_key=f"judge_{label}_instructions_gpt", inputs=inputs)
    s_d = _load_score_json(
        cell / "sonnet" / "scores_descriptions.json",
        dep_key=f"judge_{label}_descriptions_sonnet", inputs=inputs)
    s_i = _load_score_json(
        cell / "sonnet" / "scores_instructions.json",
        dep_key=f"judge_{label}_instructions_sonnet", inputs=inputs)
    return combine_desc_inst_two_judges(g_d, g_i, s_d, s_i,
                                         weights=DI_WEIGHT_CHOICES["inst_tie"])


# ---------------------------------------------------------------------------
# Core ρ computation primitives
# ---------------------------------------------------------------------------

def compute_M_done(M_raw, A_g, A_n, pool, L, K, shear_cache, whiten_cache):
    """Apply L-shear then K-soft-K to entity matrix.  Returns (M_done, pool_shear)."""
    if L == 0:
        M_shear = M_raw
        pool_shear = pool
    else:
        if L not in shear_cache:
            shear_cache[L] = fit_shear(A_g, A_n, L=L)
        sh = shear_cache[L]
        M_shear = sh.apply(M_raw)
        pool_shear = sh.apply(pool)

    if K == 0:
        M_done = M_shear
    else:
        wkey = (L, K)
        if wkey not in whiten_cache:
            whiten_cache[wkey] = fit_whitening("soft_K", pool_shear, K=K)
        M_done = whiten_cache[wkey].apply(M_shear)
    return M_done, pool_shear


def _safe_svd(centered, label=""):
    """``np.linalg.svd`` with fallback to scipy's ``gesvd`` driver on
    convergence failure.  At very high K the soft-K-whitened matrix is
    near-rank-deficient and the default LAPACK ``gesdd`` driver can fail to
    converge; ``gesvd`` is slower but more robust."""
    try:
        return np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        try:
            from scipy.linalg import svd as scipy_svd
            return scipy_svd(centered, full_matrices=False, lapack_driver="gesvd")
        except Exception as e:                              # noqa: BLE001
            print(f"  WARNING: SVD failed for {label}: {e}; skipping config")
            return None


def compute_pc_directions(M_done, max_pc):
    """SVD on centered M_done; return Vt (top max_pc rows), or None if SVD
    cannot converge (very high K can leave the matrix near-rank-deficient)."""
    centered = M_done - M_done.mean(axis=0, keepdims=True)
    res = _safe_svd(centered, label=f"compute_pc_directions max_pc={max_pc}")
    if res is None:
        return None
    _U, _S, Vt = res
    return Vt[:max_pc]


def compute_truncated_pc_directions(M_done, pool_shear, M_top, max_pc):
    """Truncate M_done into top-M_top of pool_shear's PC basis, re-SVD,
    return Vt mapped back into the original D-dim space."""
    centered_pool = pool_shear - pool_shear.mean(axis=0, keepdims=True)
    res_pool = _safe_svd(centered_pool, label=f"truncated pool M_top={M_top}")
    if res_pool is None:
        return None
    _Up, _Sp, Vt_pool = res_pool
    M_trunc = M_done @ Vt_pool[:M_top].T          # (n_entities, M_top)
    centered = M_trunc - M_trunc.mean(axis=0, keepdims=True)
    res = _safe_svd(centered, label=f"truncated M_top={M_top}")
    if res is None:
        return None
    _U, _S, Vt = res
    # Map back to original D-dim space: direction = Vt[k] @ Vt_pool[:M_top].
    return Vt[:max_pc] @ Vt_pool[:M_top]          # (max_pc, D)


def rho_for_direction(M_done, names, direction, scores) -> Optional[float]:
    proj = M_done @ direction
    common = [n for n in names if n in scores]
    if len(common) < 3:
        return None
    n2i = {n: i for i, n in enumerate(names)}
    x = np.array([scores[n] for n in common])
    y = proj[[n2i[n] for n in common]]
    r = spearmanr(x, y).correlation
    return None if np.isnan(r) else float(r)


# ---------------------------------------------------------------------------
# Stage 1: M sweep at coarse (L, K) grid
# ---------------------------------------------------------------------------

def stage1_m_sweep(scores_by_cell, *, data_dir, configs, l_values, k_coarse,
                    m_values, pcs, styles):
    """DEPRECATED (May 2026): part of the (L, K, M, cell) optimization
    sweep.  See file-level docstring for context."""
    print(f"\n{'='*92}")
    print("STAGE 1: M-truncation sweep at coarse (L, K) grid")
    print(f"{'='*92}")

    best_per_cell: dict = {}  # (pc, style) -> (rho, slot, layer, L, K, M)

    for slot, layer in configs:
        print(f"\n  ({slot},{layer:>2}): loading...")
        names, M_raw, A_g, A_n, pool = setup_at(data_dir, slot, layer)
        shear_cache: dict = {}
        whiten_cache: dict = {}

        for L in l_values:
            for K in k_coarse:
                M_done, pool_shear = compute_M_done(
                    M_raw, A_g, A_n, pool, L, K, shear_cache, whiten_cache
                )

                # M = None (no truncation): SVD on M_done
                Vt_full = compute_pc_directions(M_done, max(pcs))
                if Vt_full is None:
                    print(f"  SKIP K={K} L={L} slot={slot} layer={layer}: SVD failed")
                    continue

                # M = finite: truncate via pool_shear's top-M
                Vt_by_M: dict[Optional[int], np.ndarray] = {None: Vt_full}
                for M_top in m_values:
                    if M_top is None:
                        continue
                    if M_top > min(M_done.shape):
                        continue
                    Vt_trunc = compute_truncated_pc_directions(
                        M_done, pool_shear, M_top, max(pcs)
                    )
                    if Vt_trunc is None:
                        continue
                    Vt_by_M[M_top] = Vt_trunc

                # For each (PC, style, M): compute ρ
                for pc in pcs:
                    for style in styles:
                        scores = scores_by_cell[(pc, style)]
                        for M_top, Vt in Vt_by_M.items():
                            if M_top is not None and M_top < pc:
                                continue
                            if pc - 1 >= Vt.shape[0]:
                                continue
                            direction = Vt[pc - 1]
                            rho = rho_for_direction(M_done, names, direction, scores)
                            if rho is None:
                                continue
                            cur = best_per_cell.get((pc, style))
                            if cur is None or rho > cur[0]:
                                best_per_cell[(pc, style)] = (
                                    rho, slot, layer, L, K, M_top
                                )
        print("    done")

    return best_per_cell


# ---------------------------------------------------------------------------
# Stage 2: iterative K refinement at M=None
# ---------------------------------------------------------------------------

def stage2_k_refinement(scores_by_cell, *, data_dir, configs, l_values,
                         k_coarse, pcs, styles):
    """DEPRECATED (May 2026): part of the (L, K, cell) optimization
    sweep.  See file-level docstring for context."""
    print(f"\n{'='*92}")
    print("STAGE 2: iterative K refinement at M=None")
    print(f"{'='*92}")

    refined_results: dict = {}  # (pc, style) -> (rho, slot, layer, L, K, "refined")
    eval_count = 0

    for slot, layer in configs:
        print(f"\n  ({slot},{layer:>2}): loading...")
        names, M_raw, A_g, A_n, pool = setup_at(data_dir, slot, layer)
        shear_cache: dict = {}
        whiten_cache: dict = {}

        Vt_at_K_per_L: dict[tuple[int, int], np.ndarray] = {}
        M_done_at_K_per_L: dict[tuple[int, int], np.ndarray] = {}

        def evaluate_K(L: int, K: int) -> dict[tuple[int, str], float]:
            """Return {(pc, style): rho} for this K, computing SVD if needed.
            If SVD fails (very-high-K degeneracy), returns empty dict."""
            nonlocal eval_count
            key = (L, K)
            if key not in Vt_at_K_per_L:
                M_done, _pool_shear = compute_M_done(
                    M_raw, A_g, A_n, pool, L, K, shear_cache, whiten_cache
                )
                Vt = compute_pc_directions(M_done, max(pcs))
                Vt_at_K_per_L[key] = Vt   # may be None
                M_done_at_K_per_L[key] = M_done
                eval_count += 1
            Vt = Vt_at_K_per_L[key]
            M_done = M_done_at_K_per_L[key]
            if Vt is None:
                return {}
            out: dict[tuple[int, str], float] = {}
            for pc in pcs:
                for style in styles:
                    if pc - 1 >= Vt.shape[0]:
                        continue
                    rho = rho_for_direction(M_done, names, Vt[pc - 1],
                                             scores_by_cell[(pc, style)])
                    if rho is not None:
                        out[(pc, style)] = rho
            return out

        # K grid for stage-2 init = the coarse grid only; bracket-and-
        # bisect handles the rest.  (The K-near-N expansion that used to
        # live here was for the nth_pc variant's K=N spike; gone now.)
        k_grid_init = sorted(set(k_coarse))
        for L in l_values:
            # Memory hygiene: evict (L', K) cache entries for L' != L.
            # L's are processed in order so we never revisit.
            for cache in (Vt_at_K_per_L, M_done_at_K_per_L):
                for k in [k for k in cache if k[0] != L]:
                    del cache[k]
            for k in [k for k in whiten_cache if k[0] != L]:
                del whiten_cache[k]
            for k in [k for k in shear_cache if k != L]:
                del shear_cache[k]

            grid: dict[int, dict[tuple[int, str], float]] = {
                K: evaluate_K(L, K) for K in k_grid_init
            }

            for pc in pcs:
                for style in styles:
                    target = (pc, style)
                    K_evaluated = sorted(grid.keys())

                    def best_K_so_far():
                        best_rho = -2.0
                        best_K = None
                        for K in K_evaluated:
                            if target in grid[K]:
                                r = grid[K][target]
                                if r > best_rho:
                                    best_rho = r
                                    best_K = K
                        return best_K, best_rho

                    for _ in range(K_REFINE_MAX_ITER):
                        K_evaluated = sorted(grid.keys())
                        K_star, rho_star = best_K_so_far()
                        if K_star is None:
                            break
                        idx = K_evaluated.index(K_star)
                        K_left = K_evaluated[max(0, idx - 1)]
                        K_right = K_evaluated[min(len(K_evaluated) - 1, idx + 1)]
                        if K_right - K_left <= K_REFINE_MIN_BRACKET:
                            break

                        new_Ks = []
                        for frac in (1/3, 2/3):
                            new_K = int(round(K_left + frac * (K_right - K_left)))
                            # Allow refinement anywhere within the K_COARSE range.
                            if new_K not in grid and 0 <= new_K <= max(k_coarse):
                                new_Ks.append(new_K)
                        if not new_Ks:
                            break
                        for K in new_Ks:
                            grid[K] = evaluate_K(L, K)
                        K_evaluated = sorted(grid.keys())
                        new_K_star, new_rho_star = best_K_so_far()
                        if (new_rho_star <= rho_star + K_REFINE_TIE_THRESH
                                and new_K_star == K_star):
                            break

                    K_star, rho_star = best_K_so_far()
                    cur = refined_results.get(target)
                    if cur is None or (rho_star is not None and rho_star > cur[0]):
                        refined_results[target] = (
                            rho_star, slot, layer, L, K_star, "refined"
                        )

        print(f"    done; {len(Vt_at_K_per_L)} unique Ks evaluated, "
              f"{sum(1 for k in Vt_at_K_per_L if k[1] not in k_coarse)} new beyond coarse")

    print(f"\n  Total {eval_count} unique (slot, layer, L, K) SVDs computed.")
    return refined_results


# ---------------------------------------------------------------------------
# Output / report
# ---------------------------------------------------------------------------

def _build_subtree_inputs(*, data_dir: Path) -> list[InputSpec]:
    """Up-front subtree InputSpecs for klm_sweep / permutation_null.

    Per-(pc, style, judge, mode) judge-score cache InputSpecs are
    appended at read time inside ``load_combined_scores`` →
    ``load_and_register`` so the recorded provenance is built in
    lockstep with the actual reads (no ``_build_inputs`` post-pass
    that could drift away from what was consumed).
    """
    return [
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors"),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors"),
        current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/r_goal",
            dep_key="combos_r_goal"),
        current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/r_nogoal",
            dep_key="combos_r_nogoal"),
        current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/t_goal",
            dep_key="combos_t_goal"),
        current_data_subtree_input(
            data_dir, "combinations/vectors/derived/marginals/t_nogoal",
            dep_key="combos_t_nogoal"),
    ]


def report(stage1, stage2, *, pcs, styles, cache_path: Path,
           inputs: list[InputSpec] | None = None,
           data_dir: Path | None = None,
           sweep_dir: Path | None = None):
    print(f"\n\n{'='*92}")
    print("SUMMARY: best round-trip ρ per (PC, style) — stage 1 (with M) vs stage 2 (refined K, M=None)")
    print(f"{'='*92}")

    headers = (f"  {'PC':>4}  {'style':<9}  {'best+M ρ':>10}  "
               f"{'@(L,K,M)':<14}  {'refined ρ':>11}  {'@(L,K)':<10}")
    print(f"\n{headers}")
    print("-" * 92)
    for pc in pcs:
        for style in styles:
            tgt = (pc, style)
            s1 = stage1.get(tgt)
            s2 = stage2.get(tgt)
            s1_str = "(none)"
            s2_str = "(none)"
            if s1:
                rho1, _slot1, _lay1, L1, K1, M1 = s1
                M_str = "inf" if M1 is None else str(M1)
                s1_str = f"  {rho1:+10.4f}  L={L1},K={K1:>3},M={M_str:<3} "
            if s2:
                rho2, _slot2, _lay2, L2, K2, _ = s2
                s2_str = f"  {rho2:+11.4f}  L={L2},K={K2:>3}  "
            print(f"  {pc:>4}  {style:<9}{s1_str}{s2_str}")

    # Per-PC: avg across glossary+inline, separate stage1 and stage2.
    print(f"\n{'='*92}")
    print("Per-PC: avg across glossary+inline (for histogram)")
    print(f"{'='*92}")
    print(f"\n  {'PC':>4}  {'stage1 best+M':>15}  {'stage2 refined':>16}")
    avg_s1: dict[int, float] = {}
    avg_s2: dict[int, float] = {}
    for pc in pcs:
        s1_rs = [stage1[(pc, s)][0] for s in styles if (pc, s) in stage1]
        s2_rs = [stage2[(pc, s)][0] for s in styles if (pc, s) in stage2]
        if s1_rs:
            avg_s1[pc] = float(np.mean(s1_rs))
        if s2_rs:
            avg_s2[pc] = float(np.mean(s2_rs))
        print(f"  {pc:>4}  {avg_s1.get(pc, float('nan')):>+15.4f}  "
              f"{avg_s2.get(pc, float('nan')):>+16.4f}")

    serialisable = {
        "stage1_M_sweep": {f"pc{pc:03d}_{s}": {
            "rho": stage1[(pc, s)][0],
            "slot": stage1[(pc, s)][1], "layer": stage1[(pc, s)][2],
            "L": stage1[(pc, s)][3], "K": stage1[(pc, s)][4], "M": stage1[(pc, s)][5],
        } for pc in pcs for s in styles if (pc, s) in stage1},
        "stage2_K_refinement_M_inf": {f"pc{pc:03d}_{s}": {
            "rho": stage2[(pc, s)][0],
            "slot": stage2[(pc, s)][1], "layer": stage2[(pc, s)][2],
            "L": stage2[(pc, s)][3], "K": stage2[(pc, s)][4],
        } for pc in pcs for s in styles if (pc, s) in stage2},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if inputs is None:
        # Backwards-compat: callers that don't pass inputs (none in tree)
        # still get a bare-JSON write so behaviour doesn't regress.
        cache_path.write_text(json.dumps(serialisable, indent=2))
    else:
        envelope = json_metadata(
            serialisable,
            inputs=inputs,
            title="pc_round_trip_klm_sweep",
        )
        cache_path.write_text(json.dumps(envelope, indent=2))
    print(f"\nWrote {cache_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                   help="Base data directory containing roles/vectors and "
                        "traits/vectors with the entity .pt files.  "
                        "Default = the project's 8-slot DEFAULT_DATA_DIR "
                        "(post May-2026 pc_round_trip migration).")
    p.add_argument("--sweep_dir", type=str, default=DEFAULT_SWEEP_DIR,
                   help="Directory containing pcNNN_<style>/<provider>/scores_*.json "
                        "files produced by launch_judge_runs.py.")
    p.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH,
                   help="Where to write the JSON cache of best round-trip ρ "
                        "per (PC, style) for stage 1 and stage 2.")
    p.add_argument("--configs", type=str,
                   default=",".join(f"{s}:{l}" for s, l in DEFAULT_CONFIGS),
                   help="Comma-separated list of slot:layer pairs to sweep. "
                        f"Default: {DEFAULT_CONFIGS}")
    p.add_argument("--pcs", type=str,
                   default=",".join(str(pc) for pc in DEFAULT_PCS),
                   help="Comma-separated PC indices to evaluate. "
                        f"Default: {DEFAULT_PCS}")
    p.add_argument("--styles", type=str,
                   default=",".join(DEFAULT_STYLES),
                   help=f"Comma-separated description styles. Default: {DEFAULT_STYLES}")
    p.add_argument("--l_values", type=str,
                   default=",".join(str(l) for l in DEFAULT_L_VALUES),
                   help=f"Comma-separated soft-shear L values. Default: {DEFAULT_L_VALUES}")
    p.add_argument("--k_coarse", type=str,
                   default=",".join(str(k) for k in DEFAULT_K_COARSE),
                   help=f"Comma-separated coarse K grid. Default: {DEFAULT_K_COARSE}")
    p.add_argument("--m_values", type=str,
                   default=",".join("inf" if m is None else str(m)
                                     for m in DEFAULT_M_VALUES),
                   help="Comma-separated M values; use 'inf' for no "
                        f"truncation. Default: {DEFAULT_M_VALUES}")
    return p.parse_args()


def _parse_configs(s: str) -> list[tuple[int, int]]:
    out = []
    for tok in s.split(","):
        slot_str, layer_str = tok.split(":")
        out.append((int(slot_str), int(layer_str)))
    return out


def _parse_m_values(s: str) -> list[Optional[int]]:
    out: list[Optional[int]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.lower() in ("inf", "none", ""):
            out.append(None)
        else:
            out.append(int(tok))
    return out


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    sweep_dir = Path(args.sweep_dir)
    cache_path = Path(args.cache_path)
    configs = _parse_configs(args.configs)
    pcs = [int(x) for x in args.pcs.split(",")]
    styles = [s.strip() for s in args.styles.split(",")]
    l_values = [int(x) for x in args.l_values.split(",")]
    k_coarse = [int(x) for x in args.k_coarse.split(",")]
    m_values = _parse_m_values(args.m_values)

    print(f"Loading judge scores from {sweep_dir} ...")
    # Provenance accumulator: subtree deps up front, per-cell judge
    # cache deps appended at read time inside load_combined_scores.
    inputs: list[InputSpec] = _build_subtree_inputs(data_dir=data_dir)
    scores_by_cell = {}
    for pc in pcs:
        for style in styles:
            cell_id = f"pc{pc:03d}_{style}"
            scores_by_cell[(pc, style)] = load_combined_scores(
                sweep_dir / cell_id,
                inputs=inputs, cell_id=cell_id,
            )

    t0 = time.time()
    stage1 = stage1_m_sweep(
        scores_by_cell, data_dir=data_dir, configs=configs, l_values=l_values,
        k_coarse=k_coarse, m_values=m_values, pcs=pcs, styles=styles,
    )
    print(f"\nStage 1 elapsed: {time.time() - t0:.1f}s")

    t0 = time.time()
    stage2 = stage2_k_refinement(
        scores_by_cell, data_dir=data_dir, configs=configs, l_values=l_values,
        k_coarse=k_coarse, pcs=pcs, styles=styles,
    )
    print(f"\nStage 2 elapsed: {time.time() - t0:.1f}s")

    # ``inputs`` was populated above (subtree deps + per-cell judge
    # cache deps via load_and_register inside load_combined_scores).
    report(stage1, stage2, pcs=pcs, styles=styles, cache_path=cache_path,
           inputs=inputs, data_dir=data_dir, sweep_dir=sweep_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
