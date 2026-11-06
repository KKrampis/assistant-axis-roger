#!/usr/bin/env python3
"""Permutation-null sweep for PC round-trip ρ — fixed_principled variant.

DEPRECATED (May 2026) — this null is the bias band that accompanies the
deprecated (L, K, cell)-optimization view in ``plot_direction_cosines.py``.
The default canonical-only plot does NOT consume this cache; it computes
its own inline shuffled-judge null at L=K=0 (no optimization) directly in
the plot script.  This file is retained so the legacy optimization plots
can still be regenerated; do not extend it for new analyses.

For each (PC, style) cell, generate ``--n_perms`` random permutations of
the existing combined desc+inst judge scores (preserving the marginal
score distribution including ties), then run a (K, L) sweep at M=∞ to
find the best ρ each permutation can achieve.  The mean across
permutations gives the empirical noise floor of the K/L search procedure
when there is no real signal in the target ranking.

This variant is **fixed_principled**: the trial direction at each
``(slot, layer, L, K)`` is the canonical PC's α-coefficients applied to
the local centred entity matrix and forward-transported through the
trial ``(L, K)`` transforms — the same direction-construction that
``plot_direction_cosines.py`` uses on the actual data.  The previous
``nth_pc`` permutation null (re-PCA at every cell) is preserved in
``*.bak.before-fixed-principled-null-*.json`` snapshots.

Stages
------

- **Stage 1** -- coarse-grid sweep over ``k_coarse × l_values``, M=∞
  only.  Vectorised over all permutations per (slot, layer, L, K) cell.
- **Stage 2** -- per-(pc, style, perm_idx) bracket-and-bisect K
  refinement at M=∞, matching the actual experiment's stage 2.  Run on
  the first ``--n_stage2`` permutations.

Cache-resumable
---------------

Stage 1 results are saved as soon as each (slot, layer) finishes.
Stage 2 results are saved per (pc, style, perm_idx) cell as they
complete, so re-running with a larger ``--n_stage2`` extends the cache
without recomputing already-done cells.

CLI
---

::

    # Stage 1 for all 100, stage 2 for first 10:
    uv run python results_analysis/pc_round_trip/permutation_null.py \\
        --n_perms 100 --n_stage2 10

    # Later: extend stage 2 to all 100 (incremental, reuses cache):
    uv run python results_analysis/pc_round_trip/permutation_null.py \\
        --n_perms 100 --n_stage2 100

Output
------

A JSON file at ``--cache_path`` with two top-level dicts (``stage1`` and
``stage2``) keyed by ``"pcNNN_{style}_p{KKK}"``, each containing
``{rho, slot, layer, L, K}`` for the winning cell.  Plus ``_meta`` with
seed, n_perms, etc.
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import rankdata

from assistant_axis.plot_metadata import json_metadata
from assistant_axis.provenance import InputSpec
from results_analysis.pc_round_trip.klm_sweep import (
    setup_at, compute_M_done,
    DEFAULT_CONFIGS, DEFAULT_L_VALUES, DEFAULT_K_COARSE,
    DEFAULT_PCS, DEFAULT_STYLES,
    K_REFINE_MAX_ITER, K_REFINE_MIN_BRACKET, K_REFINE_TIE_THRESH,
    load_combined_scores,
    DEFAULT_SWEEP_DIR,
    _build_subtree_inputs,
)
from results_analysis.canonical_angles.data import DEFAULT_DATA_DIR
from results_analysis.canonical_angles.whitening import fit_shear

DEFAULT_CACHE_PATH = "roger/pc_round_trip_null_klm_results.json"
DEFAULT_N_PERMS = 100
DEFAULT_N_STAGE2 = 10
DEFAULT_SEED = 42

# Canonical reference geometry: where the α-coefficients are computed.
# May 2026 migration: 8-slot dataset, slot 7, layer 25, raw (L=0).
CANONICAL_SLOT = 7
CANONICAL_LAYER = 25
CANONICAL_L = 0


def compute_alpha_per_pc(
    data_dir: Path, pcs: list[int]
) -> tuple[list[str], dict[int, np.ndarray]]:
    """Return (canonical entity-name list, ``{pc: α[pc-1]}``) where α[N-1]
    = U[:, N-1] / σ_{N-1} from the SVD of the canonical entity matrix
    (post-shear at ``CANONICAL_L``; identity if L=0 / raw).  Applying these
    weights to a different cell's centred entity matrix gives the
    principled cross-cell transport of canonical PC N.
    """
    names, M_raw, A_g, A_n, _pool = setup_at(
        data_dir, CANONICAL_SLOT, CANONICAL_LAYER
    )
    sh = fit_shear(A_g, A_n, L=CANONICAL_L)
    M_canonical = sh.apply(M_raw)
    M_centered = M_canonical - M_canonical.mean(axis=0, keepdims=True)
    U, S, _Vt = np.linalg.svd(M_centered, full_matrices=False)
    alpha: dict[int, np.ndarray] = {}
    for pc in pcs:
        if pc - 1 < S.shape[0] and S[pc - 1] > 0:
            alpha[pc] = (U[:, pc - 1] / S[pc - 1]).astype(np.float32)
    return names, alpha


def principled_directions_in_raw(
    M_raw: np.ndarray, alpha_per_pc: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """For the cell whose entity matrix is ``M_raw``, return the principled
    direction in raw R^D for each PC: ``α[pc] @ M_centered`` (one row per
    PC).  Independent of (L, K) — those transforms are applied later.
    """
    M_centered = M_raw - M_raw.mean(axis=0, keepdims=True)
    return {pc: (a @ M_centered).astype(np.float32)
            for pc, a in alpha_per_pc.items()}


def transport_direction(d_raw: np.ndarray, L: int, K: int,
                         shear_basis, whiten_basis) -> np.ndarray:
    """Forward-apply L-shear then K-soft-K to a direction vector."""
    x = d_raw[None, :]
    if L > 0:
        x = shear_basis.apply(x)
    if K > 0:
        x = whiten_basis.apply(x)
    return x[0]


# ---------------------------------------------------------------------------
# Permutation setup
# ---------------------------------------------------------------------------

def build_permutation_score_arrays(
    actual_scores_per_cell: dict[tuple[int, str], dict[str, float]],
    names: list[str],
    n_perms: int,
    seed: int,
) -> tuple[
    dict[tuple[int, str], np.ndarray],   # cell -> (n_perms, n_valid) of rank values
    dict[tuple[int, str], np.ndarray],   # cell -> (n_entities,) bool valid mask
    dict[tuple[int, str], np.ndarray],   # cell -> (n_perms, n_valid) centred ranks
    dict[tuple[int, str], np.ndarray],   # cell -> (n_perms,) sum of squares of centred ranks
]:
    """For each cell, build n_perms permutations of the cell's actual judge
    score values, aligned to the canonical ``names`` order.  Pre-compute
    the ranks (vectorised for fast Spearman later) and centring/normsq
    so the inner ρ loop is just two matrix products.

    Each cell's permutations are independent (drawn from a per-cell rng
    seeded as ``seed * 10_000 + pc * 10 + style_idx``) so that
    ``--n_stage2 K`` gives a deterministic prefix of the ``n_perms``
    permutations for that cell across runs.
    """
    valid_mask_per_cell: dict[tuple[int, str], np.ndarray] = {}
    perm_ranks_per_cell: dict[tuple[int, str], np.ndarray] = {}
    perm_ranks_centred_per_cell: dict[tuple[int, str], np.ndarray] = {}
    perm_ranks_normsq_per_cell: dict[tuple[int, str], np.ndarray] = {}

    style_order = list(DEFAULT_STYLES)
    for (pc, style), scores in actual_scores_per_cell.items():
        rng = np.random.default_rng(
            seed * 10_000 + pc * 10 + style_order.index(style)
        )
        score_values = np.array([scores.get(n, np.nan) for n in names])
        valid = ~np.isnan(score_values)
        valid_values = score_values[valid]
        n_valid = len(valid_values)

        perms = np.empty((n_perms, n_valid), dtype=np.float64)
        for i in range(n_perms):
            perms[i] = rng.permutation(valid_values)
        ranks = np.apply_along_axis(rankdata, 1, perms)
        ranks_c = ranks - ranks.mean(axis=1, keepdims=True)
        normsq = (ranks_c ** 2).sum(axis=1)

        valid_mask_per_cell[(pc, style)] = valid
        perm_ranks_per_cell[(pc, style)] = ranks
        perm_ranks_centred_per_cell[(pc, style)] = ranks_c
        perm_ranks_normsq_per_cell[(pc, style)] = normsq

    return (
        perm_ranks_per_cell,
        valid_mask_per_cell,
        perm_ranks_centred_per_cell,
        perm_ranks_normsq_per_cell,
    )


# ---------------------------------------------------------------------------
# Vectorised Spearman: one projection vs all perms of a cell
# ---------------------------------------------------------------------------

def spearman_proj_vs_perms(
    proj: np.ndarray,
    valid: np.ndarray,
    perm_ranks_centred: np.ndarray,
    perm_ranks_normsq: np.ndarray,
) -> np.ndarray:
    """Spearman ρ between ``proj`` and each row of ``perm_ranks_centred``.

    Equivalent to: ``[scipy.stats.spearmanr(proj_valid, perms[i]).correlation
    for i in range(n_perms)]`` -- but vectorised via "Spearman = Pearson on
    ranks".  Returns a ``(n_perms,)`` array.
    """
    proj_valid = proj[valid]
    proj_ranks = rankdata(proj_valid)
    proj_ranks_c = proj_ranks - proj_ranks.mean()
    proj_normsq = float((proj_ranks_c ** 2).sum())
    if proj_normsq <= 0:
        return np.zeros(perm_ranks_centred.shape[0])
    num = perm_ranks_centred @ proj_ranks_c           # (n_perms,)
    denom = np.sqrt(proj_normsq * perm_ranks_normsq)  # (n_perms,)
    return num / np.maximum(denom, 1e-12)


# ---------------------------------------------------------------------------
# Stage 1: coarse-grid sweep, all permutations
# ---------------------------------------------------------------------------

def _apply_cutoff(direction: np.ndarray, Vt_raw_t: Optional[np.ndarray],
                   pc: int, cutoff_fraction: float) -> np.ndarray:
    """If a cutoff is active, project out the first ``floor(pc * fraction)``
    raw target-cell PCs from ``direction``.  When ``Vt_raw_t`` is None or
    cutoff is 0, this is a no-op."""
    if Vt_raw_t is None or cutoff_fraction <= 0:
        return direction
    n_cut = min(int(pc * cutoff_fraction), Vt_raw_t.shape[0])
    if n_cut <= 0:
        return direction
    coefs = Vt_raw_t[:n_cut] @ direction
    return direction - Vt_raw_t[:n_cut].T @ coefs


def stage1_coarse(
    # DEPRECATED (May 2026): coarse (L, K) grid sweep over shuffled judge
    # scores; only relevant to the deprecated optimization view.  See
    # file-level docstring.
    actual_scores_per_cell, names, n_perms, *,
    data_dir, configs, l_values, k_coarse, pcs, alpha_per_pc,
    perm_ranks_centred_per_cell, valid_mask_per_cell, perm_ranks_normsq_per_cell,
    cache_path: Path,
    save_callback,
    cutoff_fraction: float = 0.0,
):
    """Stage 1 sweep at M=∞, vectorised over permutations, fixed_principled
    variant.

    For each cell ``(slot, layer)`` we compute one direction-in-raw per PC
    (``α[pc] @ M_centered``).  Per ``(L, K)`` we forward-transport that
    direction and project ``M_done @ d`` to get the score-correlate; ρ
    against the permuted score arrays is then a vectorised dot-product.

    Returns ``(best, best_per_cell)`` keyed by ``(pc, style, perm_idx)``
    and ``(pc, style, perm_idx, slot, layer)`` respectively.
    """
    print(f"\n{'='*92}")
    print(f"STAGE 1: coarse (L, K) sweep, fixed_principled, n_perms={n_perms}")
    print(f"{'='*92}")

    best: dict[tuple[int, str, int], dict] = {}
    best_per_cell: dict[tuple[int, str, int, int, int], dict] = {}

    n_configs_total = len(configs) * len(l_values) * len(k_coarse)
    counter = 0
    t0 = time.time()

    for slot, layer in configs:
        print(f"\n  ({slot},{layer:>2}): loading...")
        names_check, M_raw, A_g, A_n, pool = setup_at(data_dir, slot, layer)
        if names_check != names:
            raise RuntimeError(
                f"Names mismatch at config ({slot}, {layer}): "
                f"{len(names_check)} vs {len(names)}"
            )
        shear_cache: dict = {}
        whiten_cache: dict = {}
        d_raw_per_pc = principled_directions_in_raw(M_raw, alpha_per_pc)
        # Pre-compute raw target Vt for cutoff (only when cutoff is active).
        # One SVD per cell, reused across all (L, K, PC) combinations.
        Vt_raw_t: Optional[np.ndarray] = None
        if cutoff_fraction > 0:
            M_centered_t = M_raw - M_raw.mean(axis=0, keepdims=True)
            _, _, Vt_raw_t = np.linalg.svd(M_centered_t, full_matrices=False)

        for L in l_values:
            for K in k_coarse:
                counter += 1
                M_done, _pool_shear = compute_M_done(
                    M_raw, A_g, A_n, pool, L, K, shear_cache, whiten_cache
                )
                # The shear/whiten bases are now in the caches.  Look them up
                # for the direction transport (fall through for L=0/K=0).
                sh = shear_cache.get(L) if L > 0 else None
                wh = whiten_cache.get((L, K)) if K > 0 else None

                for pc, d_raw in d_raw_per_pc.items():
                    direction = d_raw
                    if L > 0:
                        direction = sh.apply(direction[None, :])[0]
                    if K > 0:
                        direction = wh.apply(direction[None, :])[0]
                    direction = _apply_cutoff(direction, Vt_raw_t, pc,
                                                cutoff_fraction)
                    proj = M_done @ direction       # (n_entities,)

                    for style in DEFAULT_STYLES:
                        if (pc, style) not in actual_scores_per_cell:
                            continue
                        valid = valid_mask_per_cell[(pc, style)]
                        rhos = spearman_proj_vs_perms(
                            proj, valid,
                            perm_ranks_centred_per_cell[(pc, style)],
                            perm_ranks_normsq_per_cell[(pc, style)],
                        )
                        for perm_idx in range(n_perms):
                            r = float(rhos[perm_idx])
                            if not np.isfinite(r):
                                continue
                            pc_key = (pc, style, perm_idx, slot, layer)
                            cur_pc = best_per_cell.get(pc_key)
                            if cur_pc is None or r > cur_pc["rho"]:
                                best_per_cell[pc_key] = {
                                    "rho": r, "L": L, "K": K,
                                }
                            cur = best.get((pc, style, perm_idx))
                            if cur is None or r > cur["rho"]:
                                best[(pc, style, perm_idx)] = {
                                    "rho": r,
                                    "slot": slot, "layer": layer,
                                    "L": L, "K": K,
                                }
        elapsed = time.time() - t0
        eta_total = elapsed / (counter / n_configs_total) if counter > 0 else 0
        print(f"    ({slot},{layer:>2}): done at counter={counter}/{n_configs_total} "
              f"(elapsed {elapsed:.0f}s, projected total {eta_total:.0f}s)")
        save_callback(best, best_per_cell, cache_path, "stage1_partial")

    save_callback(best, best_per_cell, cache_path, "stage1_final")
    print(f"\nStage 1 done in {time.time()-t0:.1f}s, "
          f"best entries: {len(best)} / {len(pcs)*len(DEFAULT_STYLES)*n_perms}, "
          f"per-cell entries: {len(best_per_cell)}")
    return best, best_per_cell


# ---------------------------------------------------------------------------
# Stage 2: bracket-and-bisect K refinement at M=∞ for selected perms
# ---------------------------------------------------------------------------

def stage2_refine(
    # DEPRECATED (May 2026): per-(pc, style, perm) bracket-and-bisect K
    # refinement on shuffled judge scores; only relevant to the deprecated
    # optimization view.  See file-level docstring.
    actual_scores_per_cell, names, *,
    data_dir, configs, l_values, k_coarse, pcs, alpha_per_pc,
    perm_ranks_centred_per_cell, valid_mask_per_cell, perm_ranks_normsq_per_cell,
    perm_indices: list[int],
    cache_path: Path,
    existing_stage2: dict,
    existing_stage2_per_cell: dict,
    save_callback,
    cutoff_fraction: float = 0.0,
):
    """Stage 2 K refinement at M=∞ for the specified perm indices.

    Tracks both the global-winner result (across all configured cells) AND
    the per-cell winner.  The per-cell view enables n-shot noise-floor
    analysis where you compare actual data against a band derived from
    the same number of (slot, layer) cells the actual data search used.

    Skips per-(pc, style, perm_idx, slot, layer) cells already in
    ``existing_stage2_per_cell`` (so resume is lossless).  Returns
    ``(refined, refined_per_cell)``.
    """
    n_to_do = (len(pcs) * len(DEFAULT_STYLES) * len(perm_indices)
               * len(configs)
               - sum(1 for k in existing_stage2_per_cell
                     if int(k.split("_p")[1].split("_s")[0]) in perm_indices))
    print(f"\n{'='*92}")
    print(f"STAGE 2: K refinement at M=∞ for perm_indices={perm_indices[:5]}"
          f"{'...' if len(perm_indices) > 5 else ''}  ({len(perm_indices)} perms)")
    print(f"  per-cell evaluations to do: {n_to_do} (skipping "
          f"{len(existing_stage2_per_cell)} cached)")
    print(f"{'='*92}")

    refined: dict[tuple[int, str, int], dict] = {}
    refined_per_cell: dict[tuple[int, str, int, int, int], dict] = {}
    already_cached_per_cell: set[tuple[int, str, int, int, int]] = set()
    # Bring forward any existing global-winner stage2 results.
    for k_str, v in existing_stage2.items():
        pc = int(k_str[2:5])
        rest = k_str[6:]                            # "{style}_p{KKK}"
        style, p_str = rest.rsplit("_p", 1)
        target = (pc, style, int(p_str))
        refined[target] = v
    # Bring forward per-cell results (from possibly-larger key set).
    for k_str, v in existing_stage2_per_cell.items():
        # key format: "pc{NNN}_{style}_p{KKK}_s{S}_l{L}"
        pc = int(k_str[2:5])
        rest = k_str[6:]
        style_perm, sl = rest.rsplit("_s", 1)
        slot_str, layer_str = sl.split("_l", 1)
        slot_, layer_ = int(slot_str), int(layer_str)
        style, p_str = style_perm.rsplit("_p", 1)
        target_pc = (pc, style, int(p_str), slot_, layer_)
        refined_per_cell[target_pc] = v
        already_cached_per_cell.add(target_pc)

    eval_count = 0
    t0 = time.time()

    for slot, layer in configs:
        print(f"\n  ({slot},{layer:>2}): loading...")
        names_check, M_raw, A_g, A_n, pool = setup_at(data_dir, slot, layer)
        if names_check != names:
            raise RuntimeError("Names mismatch")
        shear_cache: dict = {}
        whiten_cache: dict = {}
        d_raw_per_pc = principled_directions_in_raw(M_raw, alpha_per_pc)
        # Pre-compute raw target Vt for cutoff (only when cutoff is active).
        Vt_raw_t: Optional[np.ndarray] = None
        if cutoff_fraction > 0:
            M_centered_t = M_raw - M_raw.mean(axis=0, keepdims=True)
            _, _, Vt_raw_t = np.linalg.svd(M_centered_t, full_matrices=False)

        # Cache: (L, K) -> M_done for this (slot, layer).  No Vt cache —
        # the principled variant doesn't SVD M_done.
        M_done_at_K_per_L: dict[tuple[int, int], np.ndarray] = {}

        def evaluate_K(L: int, K: int, pcs_to_eval: list[int]
                       ) -> dict[tuple[int, str, int], float]:
            """Return ``{(pc, style, perm_idx): rho}`` for this (L, K).
            fixed_principled: forward-transport ``α @ M_centered`` through
            (L, K), then project ``M_done @ d`` and Spearman vs perms."""
            nonlocal eval_count
            key = (L, K)
            if key not in M_done_at_K_per_L:
                M_done, _ = compute_M_done(
                    M_raw, A_g, A_n, pool, L, K, shear_cache, whiten_cache
                )
                M_done_at_K_per_L[key] = M_done
                eval_count += 1
            M_done = M_done_at_K_per_L[key]
            sh = shear_cache.get(L) if L > 0 else None
            wh = whiten_cache.get((L, K)) if K > 0 else None
            out: dict[tuple[int, str, int], float] = {}
            for pc in pcs_to_eval:
                d_raw = d_raw_per_pc.get(pc)
                if d_raw is None:
                    continue
                direction = d_raw
                if L > 0:
                    direction = sh.apply(direction[None, :])[0]
                if K > 0:
                    direction = wh.apply(direction[None, :])[0]
                direction = _apply_cutoff(direction, Vt_raw_t, pc,
                                            cutoff_fraction)
                proj = M_done @ direction
                for style in DEFAULT_STYLES:
                    if (pc, style) not in actual_scores_per_cell:
                        continue
                    valid = valid_mask_per_cell[(pc, style)]
                    rhos = spearman_proj_vs_perms(
                        proj, valid,
                        perm_ranks_centred_per_cell[(pc, style)],
                        perm_ranks_normsq_per_cell[(pc, style)],
                    )
                    for perm_idx in perm_indices:
                        r = float(rhos[perm_idx])
                        if np.isfinite(r):
                            out[(pc, style, perm_idx)] = r
            return out

        # Coarse grid only — bracket-and-bisect handles refinement.
        k_grid_init = sorted(set(k_coarse))
        for L in l_values:
            # Memory hygiene: evict (L', K) cache entries for L' != L.
            for k in [k for k in M_done_at_K_per_L if k[0] != L]:
                del M_done_at_K_per_L[k]
            for k in [k for k in whiten_cache if k[0] != L]:
                del whiten_cache[k]
            for k in [k for k in shear_cache if k != L]:
                del shear_cache[k]

            grid: dict[int, dict[tuple[int, str, int], float]] = {
                K: evaluate_K(L, K, pcs) for K in k_grid_init
            }

            # For each (pc, style, perm_idx) cell, refine K independently
            for pc in pcs:
                for style in DEFAULT_STYLES:
                    if (pc, style) not in actual_scores_per_cell:
                        continue
                    for perm_idx in perm_indices:
                        target = (pc, style, perm_idx)
                        target_pc = (pc, style, perm_idx, slot, layer)
                        if target_pc in already_cached_per_cell:
                            continue  # already refined at this cell

                        K_evaluated = sorted(grid.keys())

                        def best_K_so_far():
                            best_rho = -2.0
                            best_K_local: Optional[int] = None
                            for K_e in K_evaluated:
                                if target in grid[K_e]:
                                    r = grid[K_e][target]
                                    if r > best_rho:
                                        best_rho = r
                                        best_K_local = K_e
                            return best_K_local, best_rho

                        for _ in range(K_REFINE_MAX_ITER):
                            K_evaluated = sorted(grid.keys())
                            K_star, rho_star = best_K_so_far()
                            if K_star is None:
                                break
                            idx = K_evaluated.index(K_star)
                            K_left = K_evaluated[idx-1] if idx > 0 else None
                            K_right = K_evaluated[idx+1] if idx + 1 < len(K_evaluated) else None
                            if K_left is None and K_right is None:
                                break
                            bracket = ((K_right or K_star) - (K_left or K_star))
                            if bracket <= K_REFINE_MIN_BRACKET:
                                break
                            new_Ks: list[int] = []
                            if K_left is not None and (K_star - K_left) > K_REFINE_MIN_BRACKET:
                                new_Ks.append((K_left + K_star) // 2)
                            if K_right is not None and (K_right - K_star) > K_REFINE_MIN_BRACKET:
                                new_Ks.append((K_star + K_right) // 2)
                            new_Ks = [K for K in new_Ks if K not in grid and K >= 0]
                            if not new_Ks:
                                break
                            for K_new in new_Ks:
                                grid[K_new] = evaluate_K(L, K_new, pcs)
                            _, new_rho_star = best_K_so_far()
                            if new_rho_star - rho_star < K_REFINE_TIE_THRESH:
                                break

                        # Final winner across the (now refined) grid for this cell
                        K_evaluated = sorted(grid.keys())
                        K_star, rho_star = best_K_so_far()
                        if K_star is None:
                            continue
                        # Per-cell winner (maxed over L for this (slot, layer)).
                        cur_pc = refined_per_cell.get(target_pc)
                        if cur_pc is None or rho_star > cur_pc["rho"]:
                            refined_per_cell[target_pc] = {
                                "rho": float(rho_star),
                                "L": L, "K": K_star,
                            }
                        # Global winner across all configured cells.
                        cur = refined.get(target)
                        if cur is None or rho_star > cur["rho"]:
                            refined[target] = {
                                "rho": float(rho_star),
                                "slot": slot, "layer": layer,
                                "L": L, "K": K_star,
                            }

            elapsed = time.time() - t0
            print(f"    ({slot},{layer:>2}) L={L}: SVDs so far {eval_count} "
                  f"(elapsed {elapsed:.0f}s)")
            save_callback(refined, refined_per_cell, cache_path,
                          "stage2_partial")

    save_callback(refined, refined_per_cell, cache_path, "stage2_final")
    print(f"\nStage 2 done in {time.time()-t0:.1f}s, "
          f"refined entries: {len(refined)}, "
          f"per-cell entries: {len(refined_per_cell)}")
    return refined, refined_per_cell


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _key_str(pc: int, style: str, perm_idx: int) -> str:
    return f"pc{pc:03d}_{style}_p{perm_idx:03d}"


def _key_str_per_cell(pc: int, style: str, perm_idx: int,
                       slot: int, layer: int) -> str:
    return f"pc{pc:03d}_{style}_p{perm_idx:03d}_s{slot}_l{layer}"


def save_stage1(best, best_per_cell, path, label,
                inputs: list[InputSpec] | None = None):
    out = {
        "stage1": {
            _key_str(pc, style, p): v for (pc, style, p), v in best.items()
        },
        "stage1_per_cell": {
            _key_str_per_cell(pc, style, p, s, l): v
            for (pc, style, p, s, l), v in best_per_cell.items()
        },
    }
    _merge_save(out, path, label, inputs=inputs)


def save_stage2(refined, refined_per_cell, path, label,
                inputs: list[InputSpec] | None = None):
    out = {
        "stage2": {
            _key_str(pc, style, p): v for (pc, style, p), v in refined.items()
        },
        "stage2_per_cell": {
            _key_str_per_cell(pc, style, p, s, l): v
            for (pc, style, p, s, l), v in refined_per_cell.items()
        },
    }
    _merge_save(out, path, label, inputs=inputs)


def _merge_save(partial: dict, path: Path, label: str,
                inputs: list[InputSpec] | None = None) -> None:
    """Merge ``partial`` into the existing JSON at ``path`` (if any) and
    write it back atomically.  Preserves the OTHER stage's data.

    Provenance integration (May 2026): when ``inputs`` is provided the
    on-disk file is a ``{"_provenance": ..., "result": ...}`` envelope and
    all stage data + ``_meta`` live inside ``result``.  ``_merge_save``
    transparently unwraps any prior envelope (via ``load_existing_cache``),
    merges the stage data, and re-wraps with a fresh provenance block on
    every write so audits see the most recent input fingerprints.  The
    single-producer-multi-call pattern is fine because every save uses the
    same input set.
    """
    existing: dict = load_existing_cache(path)
    for k, v in partial.items():
        if isinstance(v, dict) and k in existing and isinstance(existing[k], dict):
            existing[k].update(v)
        else:
            existing[k] = v
    if "_meta" in existing:
        existing["_meta"]["last_save_label"] = label
        existing["_meta"]["last_save_ts"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime()
        )
    payload: dict
    if inputs is None:
        payload = existing
    else:
        payload = json_metadata(
            existing,
            inputs=inputs,
            title="pc_round_trip_permutation_null",
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def load_existing_cache(path: Path) -> dict:
    """Return the cached stage-1/stage-2 result dict, transparently
    unwrapping an envelope if present."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    if isinstance(data, dict) and "_provenance" in data and "result" in data:
        return data["result"]
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--sweep_dir", type=str, default=DEFAULT_SWEEP_DIR,
                   help="Directory with pcNNN_<style>/ cells (used to load actual judge scores).")
    p.add_argument("--cache_path", type=str, default=DEFAULT_CACHE_PATH)
    p.add_argument("--n_perms", type=int, default=DEFAULT_N_PERMS,
                   help="Number of random permutations per (PC, style) cell.")
    p.add_argument("--n_stage2", type=int, default=DEFAULT_N_STAGE2,
                   help="Number of permutations to refine in stage 2 "
                        "(prefix of n_perms; deterministic per cell via "
                        "per-cell rng seed).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--pcs", type=int, nargs="*", default=DEFAULT_PCS,
                   help="PC indices to evaluate.  Default = full klm_sweep "
                        "set (1..256).")
    p.add_argument("--configs", type=str, default=None,
                   help="Override the (slot, layer) configs as comma-"
                        "separated 'slot:layer' pairs (e.g. '3:25,0:26,0:49').")
    p.add_argument("--skip_stage1", action="store_true",
                   help="Reuse stage 1 from cache; only run stage 2.")
    p.add_argument("--skip_stage2", action="store_true",
                   help="Run stage 1 only.")
    cutoff_grp = p.add_mutually_exclusive_group()
    cutoff_grp.add_argument(
        "--cutoff_n_over_2", action="store_true",
        help="Apply N/2 cutoff to the transported direction (project out "
             "the first floor(N/2) raw target-cell PCs) before the "
             "spearman computation.  Cache path auto-suffixed with '_cutoff'.")
    cutoff_grp.add_argument(
        "--cutoff_2n_over_3", action="store_true",
        help="Apply 2N/3 cutoff to the transported direction (project out "
             "the first floor(2*N/3) raw target-cell PCs).  Cache path "
             "auto-suffixed with '_cutoff_2of3'.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    sweep_dir = Path(args.sweep_dir)
    # Derive cutoff configuration (parallels plot_direction_cosines).  When
    # active, append a suffix to the cache path so we don't clobber the
    # no-cutoff null cache.
    if args.cutoff_n_over_2:
        cutoff_fraction = 0.5
        cutoff_suffix = "_cutoff"
    elif args.cutoff_2n_over_3:
        cutoff_fraction = 2.0 / 3.0
        cutoff_suffix = "_cutoff_2of3"
    else:
        cutoff_fraction = 0.0
        cutoff_suffix = ""
    cache_path = Path(args.cache_path)
    if cutoff_suffix and cache_path.stem.endswith(cutoff_suffix) is False:
        # If the user passed an explicit --cache_path that already has the
        # suffix, don't double-suffix; otherwise auto-append.
        cache_path = cache_path.with_name(cache_path.stem + cutoff_suffix
                                           + cache_path.suffix)
    if cutoff_fraction > 0:
        print(f"  cutoff active (fraction={cutoff_fraction:.3f}, "
              f"suffix='{cutoff_suffix}'): cache at {cache_path}")
    pcs = list(args.pcs)
    if args.configs:
        configs = [tuple(int(x) for x in pair.split(":"))
                   for pair in args.configs.split(",")]
    else:
        configs = list(DEFAULT_CONFIGS)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load actual scores per cell --------------------------------------
    print(f"Loading actual judge scores from {sweep_dir}...")
    # Provenance accumulator: subtree deps up front (via klm_sweep's
    # helper), per-cell judge cache deps appended at read time inside
    # load_combined_scores → load_and_register.
    inputs: list[InputSpec] = _build_subtree_inputs(data_dir=data_dir)
    actual_scores_per_cell: dict[tuple[int, str], dict[str, float]] = {}
    missing_cells = []
    for pc in pcs:
        for style in DEFAULT_STYLES:
            cell_id = f"pc{pc:03d}_{style}"
            cell = sweep_dir / cell_id
            try:
                actual_scores_per_cell[(pc, style)] = load_combined_scores(
                    cell, inputs=inputs, cell_id=cell_id,
                )
            except FileNotFoundError:
                missing_cells.append((pc, style))
    if missing_cells:
        print(f"  WARNING: missing {len(missing_cells)} cells, skipping: "
              f"{missing_cells[:5]}...")
    print(f"  Loaded {len(actual_scores_per_cell)} cells.")

    # ---- Canonical α-coefficients + names list ----------------------------
    print(f"\nComputing canonical α-coefficients at "
          f"(slot={CANONICAL_SLOT}, layer={CANONICAL_LAYER}, L={CANONICAL_L}) ...")
    names, alpha_per_pc = compute_alpha_per_pc(data_dir, pcs)
    print(f"  Entity matrix: {len(names)} entities (roles + traits); "
          f"α available for {len(alpha_per_pc)}/{len(pcs)} PCs.")

    # ---- Build permutations + pre-compute ranks ---------------------------
    print(f"\nGenerating {args.n_perms} permutations per cell "
          f"(seed={args.seed})...")
    (_perm_ranks_per_cell, valid_mask_per_cell,
     perm_ranks_centred_per_cell, perm_ranks_normsq_per_cell
     ) = build_permutation_score_arrays(
        actual_scores_per_cell, names, args.n_perms, args.seed
    )

    # Initialise / update _meta
    meta = {
        "variant": "fixed_principled",
        "n_perms": args.n_perms,
        "n_stage2_requested": args.n_stage2,
        "seed": args.seed,
        "pcs": pcs,
        "configs": [list(c) for c in configs],
        "l_values": list(DEFAULT_L_VALUES),
        "k_coarse": list(DEFAULT_K_COARSE),
        "canonical_slot_layer_L": [CANONICAL_SLOT, CANONICAL_LAYER, CANONICAL_L],
        "M": "infinity",
        "cutoff_fraction": cutoff_fraction,
        "cutoff_suffix": cutoff_suffix,
    }
    existing = load_existing_cache(cache_path)
    prior_variant = (existing.get("_meta") or {}).get("variant", "nth_pc")
    if existing and prior_variant != "fixed_principled":
        # Cache is from the old nth_pc null run; discard stage data so we
        # don't silently mix variants.  (Snapshot was taken before this
        # script was switched over — see AGENT_NOTES.)
        print(f"  WARNING: existing cache variant={prior_variant!r} != "
              f"fixed_principled; wiping stage1/stage2 data and restarting.")
        for key in ("stage1", "stage1_per_cell", "stage2", "stage2_per_cell"):
            existing.pop(key, None)
    existing["_meta"] = meta
    # ``inputs`` was populated above by _build_subtree_inputs +
    # per-cell load_and_register calls inside load_combined_scores.
    envelope = json_metadata(
        existing, inputs=inputs, title="pc_round_trip_permutation_null",
    )
    cache_path.write_text(json.dumps(envelope, indent=2))

    save_stage1_with_inputs = functools.partial(save_stage1, inputs=inputs)
    save_stage2_with_inputs = functools.partial(save_stage2, inputs=inputs)

    # ---- Stage 1 ----------------------------------------------------------
    if args.skip_stage1:
        print("\nSkipping stage 1 (--skip_stage1).")
    else:
        stage1_coarse(
            actual_scores_per_cell, names, args.n_perms,
            data_dir=data_dir, configs=configs,
            l_values=DEFAULT_L_VALUES, k_coarse=DEFAULT_K_COARSE,
            pcs=pcs, alpha_per_pc=alpha_per_pc,
            perm_ranks_centred_per_cell=perm_ranks_centred_per_cell,
            valid_mask_per_cell=valid_mask_per_cell,
            perm_ranks_normsq_per_cell=perm_ranks_normsq_per_cell,
            cache_path=cache_path,
            save_callback=save_stage1_with_inputs,
            cutoff_fraction=cutoff_fraction,
        )

    # ---- Stage 2 ----------------------------------------------------------
    if args.skip_stage2:
        print("\nSkipping stage 2 (--skip_stage2).")
    else:
        existing = load_existing_cache(cache_path)
        existing_stage2 = existing.get("stage2", {})
        existing_stage2_per_cell = existing.get("stage2_per_cell", {})
        perm_indices = list(range(args.n_stage2))
        stage2_refine(
            actual_scores_per_cell, names,
            data_dir=data_dir, configs=configs,
            l_values=DEFAULT_L_VALUES, k_coarse=DEFAULT_K_COARSE,
            pcs=pcs, alpha_per_pc=alpha_per_pc,
            perm_ranks_centred_per_cell=perm_ranks_centred_per_cell,
            valid_mask_per_cell=valid_mask_per_cell,
            perm_ranks_normsq_per_cell=perm_ranks_normsq_per_cell,
            perm_indices=perm_indices,
            cache_path=cache_path,
            existing_stage2=existing_stage2,
            existing_stage2_per_cell=existing_stage2_per_cell,
            save_callback=save_stage2_with_inputs,
            cutoff_fraction=cutoff_fraction,
        )

    print(f"\nDone.  Cache at {cache_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
