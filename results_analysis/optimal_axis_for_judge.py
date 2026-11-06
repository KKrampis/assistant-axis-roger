#!/usr/bin/env python3
"""Find the unit direction in a fixed metric (slot, layer, whitening) that
maximises Spearman ρ between entity projections and a judge-score dict.

Algorithm: project entity vectors into the top-M PC subspace, then run
coordinate ascent on the M PC coefficients of the unknown direction ``d``,
solving each 1-D sub-problem exactly via the crossing-points trick (a
generalisation of the per-PC β optimum in ``coordinate_ascent.py``).
Multiple structured restarts handle local optima; a cosine-clustering pass
diagnoses how confident we are that ``d`` is well-determined.

See ``results_analysis/README.md`` (section ``optimal_axis_for_judge.py``)
for the headline algorithm sketch and use cases.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

# Numba JIT for the inner crossing-sweep loop.  Compiled lazily on first call
# (~1 s overhead) but then runs at C-like speed; without it the per-coord
# Python loop over O(N²) crossings is the dominant runtime cost.
try:
    from numba import njit                                       # type: ignore
    _NUMBA_OK = True
except ImportError:                                              # pragma: no cover
    _NUMBA_OK = False
    def njit(*_args, **_kwargs):                                 # noqa: D401
        """Fallback no-op decorator if numba isn't available."""
        def deco(fn):
            return fn
        if _args and callable(_args[0]):
            return _args[0]
        return deco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assistant_axis import entity_id, json_metadata, response_subdir
from assistant_axis.entity_id import normalize_to_file_name
from assistant_axis.judge_loaders import migrate_v1_static_scores
from assistant_axis.judge_score_combine import (
    DI_WEIGHT_CHOICES, combine_desc_inst_one_judge,
    combine_desc_inst_two_judges,
)
from assistant_axis.plot_metadata import png_metadata, suptitle_with_specs
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
    current_file_input,
    load_and_register,
)
from results_analysis.axis_judge_correlation import _load_vector_file
from results_analysis.canonical_angles.data import (
    DEFAULT_DATA_DIR, build_augmented_whitening_pool, load_vector,
)
from results_analysis.canonical_angles.whitening import fit_whitening


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_M = 30                # tuned 2026-05-01: M-sweep on di_combined
                                # (truthful_vs_deceitful) showed M=30 maximises
                                # ρ_cv (0.815 vs 0.787 at M=50, 0.783 at M=100)
                                # with the smallest train/CV gap (0.022).
                                # Above M=50 overfitting kicks in; above M=75
                                # the top cluster also fragments.
DEFAULT_N_RESTARTS = 8         # tuned 2026-05-01: at M=30 all 8 restarts
                                # converge to the same cluster (cos > 0.99
                                # median).  16 was overkill.  Reducing 16 → 8
                                # roughly halves wall-clock (and CV cost).
DEFAULT_MAX_PASSES = 20
DEFAULT_CONV_EPS = 1e-3
DEFAULT_CV_FOLDS = 5
DEFAULT_K_SEED = 4             # number of axis-aligned-signed seeds (PCs 0..K_seed-1)
DEFAULT_CLUSTER_COS_THRESHOLD = 0.95
DEFAULT_TIE_THRESH = 5e-4
DEFAULT_SLOT = 6  # New default after May 2026 rejudge run: slot 6 (</think>)
                  # beats slot 3 (\\n) on judge ρ.  Pass --slot 3 or 7 to
                  # compare.  --output_dir auto-derives from the cache key
                  # (pair, score_source, slot, layer, whitening, M, data_dir)
                  # via :func:`derive_output_dir`, so changing any default
                  # below produces a distinct on-disk dir without callers
                  # having to remember a manual labelling convention.
DEFAULT_LAYER = 25
DEFAULT_WHITENING = "soft_K=3"
DEFAULT_OUTPUT_BASE = Path("roger/optimal_axis")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RestartResult:
    seed_label: str
    final_rho_train: float
    final_d_pcs: np.ndarray            # (M,) unit vector in PC basis
    n_passes: int
    cluster_id: int = -1               # filled in after clustering


@dataclass
class FitResult:
    """Headline output of :func:`find_optimal_direction`."""
    direction: np.ndarray              # (D,) unit vector in original hidden space
    direction_pcs: np.ndarray          # (M,) coords in the working PC basis
    pc_basis: np.ndarray               # (M, D) PC basis used (for callers
                                        # that want to express other vectors in
                                        # the same coords)
    rho_train: float
    rho_cv: Optional[float]
    cluster_summary: dict
    restarts: list[RestartResult] = field(default_factory=list)
    metric: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 1: geometry setup
# ---------------------------------------------------------------------------

def parse_whitening_spec(spec: str) -> tuple[str, Optional[int]]:
    """``raw`` -> (raw, None); ``soft_K=3`` -> (soft_K, 3)."""
    s = spec.strip()
    if s == "raw":
        return "raw", None
    if s.startswith("soft_K="):
        try:
            K = int(s.split("=", 1)[1])
        except ValueError as e:
            raise ValueError(f"bad whitening spec {spec!r}") from e
        return "soft_K", K
    raise ValueError(f"unsupported whitening spec {spec!r}; "
                     "supported: 'raw' or 'soft_K=N'")


def load_entity_matrix(
    data_dir: Path, slot: int, layer: int,
    exclude: set[str] = frozenset(),
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load all standalone roles + traits at (slot, layer), default-centred.
    Returns (names, M_centered, default_v) where M_centered shape is (N, D).
    """
    default_v = _load_vector_file(
        data_dir / "traits" / "vectors" / "default.pt"
    ).float().numpy()[slot, layer]
    names: list[str] = []
    rows: list[np.ndarray] = []
    for et in ("roles", "traits"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default" or fp.stem in exclude:
                continue
            try:
                v = _load_vector_file(fp).float().numpy()[slot, layer]
            except Exception:                                  # noqa: BLE001
                continue
            names.append(entity_id(fp.stem, et))
            rows.append(v - default_v)
    return names, np.stack(rows, axis=0).astype(np.float64), default_v


def build_pool(
    data_dir: Path, slot: int, layer: int,
    exclude: set[str] = frozenset(),
) -> np.ndarray:
    """Augmented pool (held-out standalones + default) at (slot, layer),
    default-centred."""
    default_v = np.asarray(load_vector(data_dir, "traits", "default"),
                            dtype=np.float64)[slot, layer]
    leave = {(et, n) for et in ("roles", "traits") for n in exclude}
    pool_entries = build_augmented_whitening_pool(data_dir, leave_out=leave)
    rows = []
    for et, name in pool_entries:
        v = np.asarray(load_vector(data_dir, et, name),
                        dtype=np.float64)[slot, layer]
        rows.append(v - default_v)
    return np.stack(rows, axis=0)


def setup_geometry(
    data_dir: Path, *, slot: int, layer: int, whitening: str,
    M: int, exclude: set[str] = frozenset(),
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Whiten + centre entities, fit M-PC basis on the entity pool.

    Returns
    -------
    names    : list[str], length N
    alpha    : (N, M) entity loadings on the M PCs (post-whitening)
    Vt       : (M, D) PC basis rows
    sigma    : (M,) singular values (for reference / variance budget)
    M_full   : (N, D) whitened, centred entity matrix (for quick rho-eval
                with already-fitted directions)
    """
    method, K = parse_whitening_spec(whitening)

    # Pool for whitener fit (held-out standalones + default).
    pool = build_pool(data_dir, slot=slot, layer=layer, exclude=exclude)
    whitener = fit_whitening(method, pool, K=K)

    names, M_centered, _default = load_entity_matrix(
        data_dir, slot=slot, layer=layer, exclude=exclude,
    )
    M_white = whitener.apply(M_centered)
    M_white = M_white - M_white.mean(axis=0, keepdims=True)

    # SVD on the (N, D) whitened entity matrix.  Keep top M PCs.
    U, S, Vt = np.linalg.svd(M_white, full_matrices=False)
    # Effective rank is bounded by N-1 after centring.
    M_eff = min(M, len(S))
    Vt = Vt[:M_eff]                 # (M_eff, D)
    sigma = S[:M_eff]               # (M_eff,)
    alpha = M_white @ Vt.T          # (N, M_eff)
    return names, alpha, Vt, sigma, M_white


def load_axis_unit_at(
    data_dir: Path, pos: str, neg: str, slot: int, layer: int,
) -> Optional[np.ndarray]:
    """Project (vec[pos] - vec[neg]) at (slot, layer) and unit-normalise.
    Returns None if either file is missing / unloadable."""
    try:
        vp = _load_vector_file(
            data_dir / "traits" / "vectors" / f"{pos}.pt"
        ).float().numpy()[slot, layer]
        vn = _load_vector_file(
            data_dir / "traits" / "vectors" / f"{neg}.pt"
        ).float().numpy()[slot, layer]
    except Exception:                                          # noqa: BLE001
        return None
    d = (vp - vn).astype(np.float64)
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        return None
    return d / nrm


# ---------------------------------------------------------------------------
# Step 2: 1-D crossing-points solver (unbounded d_k in R)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sweep_crossings_jit(
    i_v: np.ndarray, j_v: np.ndarray,
    rank_of: np.ndarray,
    judge_avg_ranks: np.ndarray,
    sum_d2_init: float,
) -> tuple[int, float]:
    """Inner crossing-sweep: walk through M sorted crossings (i_v[m], j_v[m]),
    swap rank_of[i_v[m]] and rank_of[j_v[m]] at each step, track running Σd²,
    return (best_idx, best_sum_d2).  ``best_idx == -1`` means the initial
    interval (-inf, beta_cross[0]) is the optimum.

    rank_of is mutated in place.
    """
    M = i_v.shape[0]
    current_d2 = sum_d2_init
    best_d2 = sum_d2_init
    best_idx = -1
    for m in range(M):
        i = i_v[m]
        j = j_v[m]
        ri = rank_of[i]
        rj = rank_of[j]
        # Δ(Σd²) = 2·(ri − rj)·(jr_i − jr_j) at a rank-swap.
        current_d2 += 2.0 * (ri - rj) * (judge_avg_ranks[i] - judge_avg_ranks[j])
        rank_of[i] = rj
        rank_of[j] = ri
        if current_d2 < best_d2 - 1e-9:
            best_d2 = current_d2
            best_idx = m
    return best_idx, best_d2


def optimize_dk(
    baseline: np.ndarray,      # (N,) score contribution from all j != k
    a_k: np.ndarray,           # (N,) entity loading on PC k
    judge_avg_ranks: np.ndarray,
    *,
    i_idx: Optional[np.ndarray] = None,
    j_idx: Optional[np.ndarray] = None,
) -> tuple[float, float, int]:
    """Find d_k in R minimising Σ(rank(score_i) - judge_avg_rank_i)² where
    score_i(d_k) = baseline_i + d_k * a_k[i].

    Returns (d_k_best, sum_d2_best, n_crossings).  ρ-argmax = Σd²-argmin
    when projection has no exact ties at the chosen interior d_k (we use
    each interval's midpoint).  At unbounded tails, we pick a representative
    d_k well outside the crossing range.

    Pass cached ``i_idx, j_idx = np.triu_indices(N, k=1)`` to avoid the
    O(N²) reallocation each per-coord call.
    """
    N = len(baseline)
    if N < 2:
        return 0.0, 0.0, 0

    if i_idx is None or j_idx is None:
        i_idx, j_idx = np.triu_indices(N, k=1)
    num = baseline[j_idx] - baseline[i_idx]
    den = a_k[i_idx] - a_k[j_idx]                 # solving b_i + d*a_i == b_j + d*a_j
    nz = den != 0.0
    if not nz.any():
        # No crossings -- ρ is constant in d_k; return 0 (any value is fine).
        ranks = rankdata(baseline, method="ordinal").astype(np.float64)
        d = ranks - judge_avg_ranks
        return 0.0, float(d @ d), 0
    beta_cross = num[nz] / den[nz]
    i_v = i_idx[nz]
    j_v = j_idx[nz]

    # Sort crossings.
    order_c = np.argsort(beta_cross, kind="stable")
    beta_cross = beta_cross[order_c]
    i_v = np.ascontiguousarray(i_v[order_c], dtype=np.int64)
    j_v = np.ascontiguousarray(j_v[order_c], dtype=np.int64)
    M = len(beta_cross)

    # Initial state: d_k = beta_cross[0] - 1.
    d_left = float(beta_cross[0] - 1.0)
    score_left = baseline + d_left * a_k
    order0 = np.argsort(score_left, kind="stable")
    rank_of = np.empty(N, dtype=np.int64)
    rank_of[order0] = np.arange(1, N + 1)
    d_init = rank_of.astype(np.float64) - judge_avg_ranks
    sum_d2 = float(d_init @ d_init)

    # Hot inner loop, JIT'd.
    best_idx, best_d2 = _sweep_crossings_jit(
        i_v, j_v, rank_of,
        np.ascontiguousarray(judge_avg_ranks, dtype=np.float64),
        sum_d2,
    )

    # Resolve interval bounds and representative d_k.
    if best_idx < 0:
        best_lo = -math.inf
        best_hi = float(beta_cross[0])
    else:
        best_lo = float(beta_cross[best_idx])
        best_hi = (float(beta_cross[best_idx + 1])
                    if best_idx + 1 < M else math.inf)

    if math.isinf(best_lo) and math.isinf(best_hi):
        d_best = 0.0
    elif math.isinf(best_lo):
        span = float(beta_cross[-1] - beta_cross[0])
        d_best = best_hi - max(1.0, span * 0.25)
    elif math.isinf(best_hi):
        span = float(beta_cross[-1] - beta_cross[0])
        d_best = best_lo + max(1.0, span * 0.25)
    else:
        d_best = 0.5 * (best_lo + best_hi)
    return d_best, best_d2, M


def evaluate_rho_pcs(d: np.ndarray, alpha: np.ndarray,
                      judge_scores: np.ndarray) -> float:
    """True (tie-corrected) Spearman ρ for direction d in PC basis."""
    score = alpha @ d
    return float(spearmanr(score, judge_scores).correlation)


# ---------------------------------------------------------------------------
# Step 3: coordinate ascent
# ---------------------------------------------------------------------------

def coordinate_ascent(
    init_d: np.ndarray,
    alpha: np.ndarray,                # (N, M)
    judge_scores: np.ndarray,         # (N,)
    judge_avg_ranks: np.ndarray,      # (N,)
    *,
    max_passes: int = DEFAULT_MAX_PASSES,
    conv_eps: float = DEFAULT_CONV_EPS,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Round-robin coordinate ascent over the M PCs.  Returns (d_unit, n_passes).

    The direction is normalised to unit length before each pass to keep the
    coordinate magnitudes from drifting -- per-coord ρ-argmax is
    scale-invariant in d, so this is equivalent and numerically cleaner.
    """
    d = init_d.astype(np.float64).copy()
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        # Degenerate seed; perturb.
        d = rng.standard_normal(alpha.shape[1])
        nrm = float(np.linalg.norm(d))
    d = d / nrm

    M = alpha.shape[1]
    N = alpha.shape[0]
    # Cache the (N choose 2) index arrays for the per-coord crossing solver.
    i_idx, j_idx = np.triu_indices(N, k=1)
    last_pass = 0
    for pass_idx in range(max_passes):
        last_pass = pass_idx + 1
        max_move = 0.0
        order = rng.permutation(M)
        for k in order:
            full = alpha @ d                              # (N,)
            baseline = full - alpha[:, k] * d[k]
            a_k = alpha[:, k]
            d_new, _d2, _ncross = optimize_dk(
                baseline, a_k, judge_avg_ranks,
                i_idx=i_idx, j_idx=j_idx,
            )
            old_dk = float(d[k])
            d[k] = d_new
            # Renormalise after each coord update -- ρ is scale-invariant in d,
            # so this just keeps coord magnitudes from drifting and makes the
            # convergence criterion meaningful in absolute units.
            nrm = float(np.linalg.norm(d))
            if nrm > 1e-12:
                d = d / nrm
            move = abs(float(d[k]) - old_dk)
            if move > max_move:
                max_move = move
        if max_move < conv_eps:
            break

    # Sign-canonicalise: ρ(-d) = -ρ(d), and we want max ρ.  Coordinate
    # ascent finds a Σd²-min within whichever sign-hemisphere it started in;
    # flipping is a free upgrade if it lands negative.
    rho = evaluate_rho_pcs(d, alpha, judge_scores)
    if rho < 0:
        d = -d
    return d, last_pass


# ---------------------------------------------------------------------------
# Step 4: structured seed restarts
# ---------------------------------------------------------------------------

def fisher_lda_seed(alpha: np.ndarray, scores: np.ndarray) -> Optional[np.ndarray]:
    """Tercile-bin entities by score and return the Fisher discriminant
    direction in PC basis.  Returns None on degenerate input."""
    N, M = alpha.shape
    if N < 6:
        return None
    qs = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0])
    low = scores <= qs[0]
    high = scores >= qs[1]
    mid = ~(low | high)
    groups = [g for g in (low, mid, high) if int(g.sum()) >= 2]
    if len(groups) < 2:
        return None
    means = [alpha[g].mean(axis=0) for g in groups]
    overall = alpha.mean(axis=0)
    Sb = np.zeros((M, M))
    Sw = np.zeros((M, M))
    for g, mu in zip(groups, means):
        diff = (mu - overall).reshape(-1, 1)
        Sb += int(g.sum()) * diff @ diff.T
        Xc = alpha[g] - mu
        Sw += Xc.T @ Xc
    # Regularise.
    Sw_reg = Sw + 1e-3 * np.trace(Sw) / M * np.eye(M)
    try:
        sol = np.linalg.solve(Sw_reg, Sb)
        # Top eigvec.
        eigvals, eigvecs = np.linalg.eig(sol)
        d = eigvecs[:, np.argmax(eigvals.real)].real
    except np.linalg.LinAlgError:
        return None
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return None
    return d / nrm


def random_unit(M: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(M)
    return v / np.linalg.norm(v)


def project_into_pc_basis(d_full: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """Project a (D,) direction into the (M,) PC basis row-space.
    The residual orthogonal to the basis is dropped (= zero for the
    judge-prediction problem since entity loadings have no component there).
    """
    return Vt @ d_full


def build_seed_initials(
    *, alpha: np.ndarray, judge_scores: np.ndarray,
    Vt: np.ndarray,
    pole_diff_full: Optional[np.ndarray],
    n_restarts: int, K_seed: int, rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """Return list of (label, init_d_pcs) of length n_restarts.

    Order: pole_diff (if present), score_weighted_mean, fisher_lda,
    K_seed axis-aligned-signed, random fill.
    """
    out: list[tuple[str, np.ndarray]] = []
    M = alpha.shape[1]

    # 1. pole_diff
    if pole_diff_full is not None:
        d = project_into_pc_basis(pole_diff_full, Vt)
        nrm = float(np.linalg.norm(d))
        if nrm > 1e-12:
            out.append(("pole_diff", d / nrm))

    # 2. score-weighted mean
    centered = judge_scores - judge_scores.mean()
    d_swm = centered @ alpha
    nrm = float(np.linalg.norm(d_swm))
    if nrm > 1e-12:
        out.append(("score_weighted_mean", d_swm / nrm))

    # 3. fisher LDA
    d_fld = fisher_lda_seed(alpha, judge_scores)
    if d_fld is not None:
        out.append(("fisher_lda", d_fld))

    # 4. axis-aligned-signed
    K_seed = min(K_seed, M)
    for k in range(K_seed):
        a = alpha[:, k]
        sd = a.std()
        if sd < 1e-12:
            continue
        # Sign by Pearson correlation with scores; ranks would also do.
        c = float(np.corrcoef(a, judge_scores)[0, 1])
        sign = 1.0 if c >= 0 else -1.0
        e = np.zeros(M)
        e[k] = sign
        out.append((f"axis_aligned_pc{k}", e))

    # 5. random fill
    while len(out) < n_restarts:
        out.append((f"random_{len(out)}", random_unit(M, rng)))

    # Truncate if we overshot.
    return out[:n_restarts]


# ---------------------------------------------------------------------------
# Step 5: cosine clustering and consensus
# ---------------------------------------------------------------------------

def _sign_align_to_reference(directions: np.ndarray,
                               ref: np.ndarray) -> np.ndarray:
    """Flip each row's sign so it has positive cosine with ``ref``."""
    aligned = directions.copy()
    for i, d in enumerate(directions):
        if float(d @ ref) < 0:
            aligned[i] = -d
    return aligned


def single_link_cluster(directions: np.ndarray,
                         threshold: float = DEFAULT_CLUSTER_COS_THRESHOLD,
                         ) -> list[list[int]]:
    """Single-link agglomerative clustering on |cosine| similarity.
    Returns a list of clusters; each cluster is a list of row indices."""
    n = len(directions)
    cos = directions @ directions.T
    np.fill_diagonal(cos, 1.0)
    abs_cos = np.abs(cos)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if abs_cos[i, j] >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def cluster_summary(
    restarts: list[RestartResult],
    *, threshold: float = DEFAULT_CLUSTER_COS_THRESHOLD,
    rho_cv_per_restart: Optional[list[Optional[float]]] = None,
) -> tuple[dict, list[RestartResult]]:
    """Build cluster summary and assign cluster ids to restarts.  Returns
    (summary_dict, restarts_with_cluster_ids).

    Note: directions are sign-aligned to the highest-rho restart, since
    Spearman ρ is sign-invariant.
    """
    n = len(restarts)
    if n == 0:
        return {"n_restarts": 0, "top_cluster": None, "secondary_clusters": []}, restarts

    D = np.stack([r.final_d_pcs for r in restarts], axis=0)
    # Ensure all unit-norm (defensive).
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    D = D / np.maximum(norms, 1e-12)

    rhos = np.array([r.final_rho_train for r in restarts])
    best_idx = int(np.argmax(rhos))
    D_aligned = _sign_align_to_reference(D, D[best_idx])

    clusters = single_link_cluster(D_aligned, threshold=threshold)
    # Sort clusters by mean rho_train descending; assign ids.
    cluster_meta = []
    for cid, members in enumerate(clusters):
        m_arr = np.array(members)
        rho_train_mean = float(rhos[m_arr].mean())
        if rho_cv_per_restart is not None:
            cv_vals = [rho_cv_per_restart[i] for i in members
                        if rho_cv_per_restart[i] is not None]
            rho_cv_mean = float(np.mean(cv_vals)) if cv_vals else None
        else:
            rho_cv_mean = None
        # Intra-cluster cosine stats.
        if len(members) >= 2:
            sub = D_aligned[m_arr]
            cos = sub @ sub.T
            iu = np.triu_indices(len(members), k=1)
            pair = cos[iu]
            min_cos = float(pair.min())
            med_cos = float(np.median(pair))
        else:
            min_cos = 1.0
            med_cos = 1.0
        centroid = D_aligned[m_arr].mean(axis=0)
        cnorm = float(np.linalg.norm(centroid))
        if cnorm > 1e-12:
            centroid = centroid / cnorm
        cluster_meta.append({
            "members": members,
            "rho_train_mean": rho_train_mean,
            "rho_cv_mean": rho_cv_mean,
            "min_pairwise_cos": min_cos,
            "median_pairwise_cos": med_cos,
            "centroid": centroid,
        })
    cluster_meta.sort(key=lambda c: c["rho_train_mean"], reverse=True)
    for new_cid, c in enumerate(cluster_meta):
        for idx in c["members"]:
            restarts[idx].cluster_id = new_cid

    top = cluster_meta[0]
    summary = {
        "n_restarts": n,
        "n_clusters": len(cluster_meta),
        "top_cluster": {
            "n_members": len(top["members"]),
            "rho_train_mean": top["rho_train_mean"],
            "rho_cv_mean": top["rho_cv_mean"],
            "min_pairwise_cos": top["min_pairwise_cos"],
            "median_pairwise_cos": top["median_pairwise_cos"],
            "member_seed_labels": [restarts[i].seed_label for i in top["members"]],
        },
        "secondary_clusters": [
            {
                "n_members": len(c["members"]),
                "rho_train_mean": c["rho_train_mean"],
                "rho_cv_mean": c["rho_cv_mean"],
                "min_pairwise_cos": c["min_pairwise_cos"],
                "median_pairwise_cos": c["median_pairwise_cos"],
                "centroid_cos_to_top": float(np.abs(c["centroid"] @ top["centroid"])),
                "member_seed_labels": [restarts[i].seed_label for i in c["members"]],
            }
            for c in cluster_meta[1:]
        ],
    }
    return summary, restarts


def consensus_seed(top_cluster_centroid: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(top_cluster_centroid))
    if n < 1e-12:
        return top_cluster_centroid
    return top_cluster_centroid / n


# ---------------------------------------------------------------------------
# Step 6: cross-validation
# ---------------------------------------------------------------------------

def kfold_indices(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return [np.sort(f) for f in folds]


# ---------------------------------------------------------------------------
# Top-level fit orchestrator
# ---------------------------------------------------------------------------

def _fit_single_split(
    *,
    alpha_train: np.ndarray,
    scores_train: np.ndarray,
    avg_ranks_train: np.ndarray,
    Vt: np.ndarray,
    pole_diff_full: Optional[np.ndarray],
    n_restarts: int,
    K_seed: int,
    max_passes: int,
    conv_eps: float,
    rng: np.random.Generator,
    cluster_threshold: float = DEFAULT_CLUSTER_COS_THRESHOLD,
) -> tuple[list[RestartResult], np.ndarray]:
    """Run multi-restart coordinate ascent on a single (alpha, scores) split.
    Returns (restarts, consensus_d_pcs).  consensus_d is the final adopted
    direction after the consensus-re-converge step."""
    seeds = build_seed_initials(
        alpha=alpha_train, judge_scores=scores_train, Vt=Vt,
        pole_diff_full=pole_diff_full,
        n_restarts=n_restarts, K_seed=K_seed, rng=rng,
    )

    restarts: list[RestartResult] = []
    for label, init_d in seeds:
        d, n_passes = coordinate_ascent(
            init_d=init_d, alpha=alpha_train,
            judge_scores=scores_train, judge_avg_ranks=avg_ranks_train,
            max_passes=max_passes, conv_eps=conv_eps, rng=rng,
        )
        rho = evaluate_rho_pcs(d, alpha_train, scores_train)
        # Unit-normalise.
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-12:
            continue
        d = d / nrm
        restarts.append(RestartResult(
            seed_label=label, final_rho_train=rho,
            final_d_pcs=d, n_passes=n_passes,
        ))
    if not restarts:
        raise RuntimeError("All restarts failed; no usable directions.")

    # Cluster, consensus seed, re-converge.
    summary, restarts = cluster_summary(restarts, threshold=cluster_threshold)

    # Build consensus init from top cluster centroid (mean of members'
    # sign-aligned directions).
    best = max(restarts, key=lambda r: r.final_rho_train)
    top_members = [r for r in restarts if r.cluster_id == best.cluster_id]
    aligned = np.stack([
        r.final_d_pcs if float(r.final_d_pcs @ best.final_d_pcs) >= 0
        else -r.final_d_pcs
        for r in top_members
    ], axis=0)
    centroid = aligned.mean(axis=0)
    consensus_init = consensus_seed(centroid)

    d_cons, _ = coordinate_ascent(
        init_d=consensus_init, alpha=alpha_train,
        judge_scores=scores_train, judge_avg_ranks=avg_ranks_train,
        max_passes=max_passes, conv_eps=conv_eps, rng=rng,
    )
    rho_cons = evaluate_rho_pcs(d_cons, alpha_train, scores_train)
    nrm = float(np.linalg.norm(d_cons))
    if nrm > 1e-12:
        d_cons = d_cons / nrm

    if rho_cons > best.final_rho_train + DEFAULT_TIE_THRESH:
        # Adopt the consensus direction as the answer.
        # Append as an extra "consensus" restart entry.
        cons_rec = RestartResult(
            seed_label="consensus", final_rho_train=rho_cons,
            final_d_pcs=d_cons, n_passes=0, cluster_id=best.cluster_id,
        )
        restarts.append(cons_rec)
        adopted = d_cons
    else:
        adopted = best.final_d_pcs

    return restarts, adopted


def find_optimal_direction(
    *,
    judge_scores: dict[str, float],
    data_dir: Path,
    slot: int = DEFAULT_SLOT,
    layer: int = DEFAULT_LAYER,
    whitening: str = DEFAULT_WHITENING,
    M: int = DEFAULT_M,
    n_restarts: int = DEFAULT_N_RESTARTS,
    max_passes: int = DEFAULT_MAX_PASSES,
    conv_eps: float = DEFAULT_CONV_EPS,
    K_seed: int = DEFAULT_K_SEED,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed_pair: Optional[tuple[str, str]] = None,
    exclude_names: set[str] = frozenset(),
    rng_seed: int = 0,
    cluster_threshold: float = DEFAULT_CLUSTER_COS_THRESHOLD,
) -> FitResult:
    """End-to-end fit.  See module docstring."""
    rng = np.random.default_rng(rng_seed)

    # Geometry.
    names, alpha_full, Vt, sigma, _M_full = setup_geometry(
        data_dir, slot=slot, layer=layer, whitening=whitening,
        M=M, exclude=exclude_names,
    )
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Pole-diff seed (optional).
    pole_diff_full = None
    if seed_pair is not None:
        pole_diff_full = load_axis_unit_at(
            data_dir, seed_pair[0], seed_pair[1], slot, layer)

    # Restrict to entities the judge has scored.
    common = sorted(set(judge_scores) & set(name_to_idx))
    if len(common) < 5:
        raise ValueError(
            f"Need >=5 common entities; got {len(common)} "
            f"(judge has {len(judge_scores)}, geometry has {len(names)}).")
    common_idx = np.array([name_to_idx[n] for n in common])
    alpha_c = alpha_full[common_idx]                    # (n_common, M)
    scores_c = np.array([judge_scores[n] for n in common], dtype=np.float64)

    # Headline (full-data) fit.
    avg_ranks_c = rankdata(scores_c, method="average").astype(np.float64)
    restarts, adopted_pcs = _fit_single_split(
        alpha_train=alpha_c,
        scores_train=scores_c,
        avg_ranks_train=avg_ranks_c,
        Vt=Vt,
        pole_diff_full=pole_diff_full,
        n_restarts=n_restarts, K_seed=K_seed,
        max_passes=max_passes, conv_eps=conv_eps,
        rng=rng,
        cluster_threshold=cluster_threshold,
    )
    rho_train = evaluate_rho_pcs(adopted_pcs, alpha_c, scores_c)

    # CV.
    rho_cv: Optional[float] = None
    if cv_folds and cv_folds >= 2:
        n = len(common)
        folds = kfold_indices(n, cv_folds, rng)
        oof = np.full(n, np.nan)
        for f_i, test_idx in enumerate(folds):
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            scores_tr = scores_c[train_mask]
            alpha_tr = alpha_c[train_mask]
            ar_tr = rankdata(scores_tr, method="average").astype(np.float64)
            try:
                _r, d_fold = _fit_single_split(
                    alpha_train=alpha_tr,
                    scores_train=scores_tr,
                    avg_ranks_train=ar_tr,
                    Vt=Vt,
                    pole_diff_full=pole_diff_full,
                    n_restarts=max(4, n_restarts // 2),
                    K_seed=K_seed,
                    max_passes=max_passes, conv_eps=conv_eps,
                    rng=np.random.default_rng(rng_seed + 100 + f_i),
                    cluster_threshold=cluster_threshold,
                )
                oof[test_idx] = alpha_c[test_idx] @ d_fold
            except Exception as e:                                # noqa: BLE001
                print(f"  [cv] fold {f_i} failed: {e}; skipping", file=sys.stderr)
        valid = ~np.isnan(oof)
        if valid.sum() >= 5:
            rho_cv = float(spearmanr(oof[valid], scores_c[valid]).correlation)

    # Cluster summary (with cv info absent at cluster level for simplicity --
    # cv is per-fold not per-restart).
    summary, restarts = cluster_summary(restarts, threshold=cluster_threshold)
    summary["rho_train_best"] = float(max(r.final_rho_train for r in restarts))
    summary["rho_cv"] = rho_cv

    # Baseline cosines and ρ.
    direction_full = adopted_pcs @ Vt                  # (D,)
    nrm = float(np.linalg.norm(direction_full))
    if nrm > 1e-12:
        direction_full = direction_full / nrm

    if pole_diff_full is not None:
        d_pole_pcs = project_into_pc_basis(pole_diff_full, Vt)
        n2 = float(np.linalg.norm(d_pole_pcs))
        if n2 > 1e-12:
            d_pole_pcs = d_pole_pcs / n2
            summary["cos_to_pole_diff"] = float(abs(d_pole_pcs @ adopted_pcs))
            summary["rho_pole_diff"] = evaluate_rho_pcs(d_pole_pcs, alpha_c, scores_c)
        else:
            summary["cos_to_pole_diff"] = None
            summary["rho_pole_diff"] = None
    else:
        summary["cos_to_pole_diff"] = None
        summary["rho_pole_diff"] = None

    centered = scores_c - scores_c.mean()
    d_swm = centered @ alpha_c
    n3 = float(np.linalg.norm(d_swm))
    if n3 > 1e-12:
        d_swm_unit = d_swm / n3
        summary["cos_to_score_weighted_mean"] = float(abs(d_swm_unit @ adopted_pcs))
        summary["rho_score_weighted_mean"] = evaluate_rho_pcs(
            d_swm_unit, alpha_c, scores_c)
    else:
        summary["cos_to_score_weighted_mean"] = None
        summary["rho_score_weighted_mean"] = None

    metric = {
        "slot": slot, "layer": layer, "whitening": whitening,
        "M": int(alpha_full.shape[1]),
        "M_requested": int(M),
        "variance_top_M": float((sigma ** 2).sum() /
                                max(1e-12, ((sigma ** 2).sum()))),  # always 1
        "sigma_top": [float(s) for s in sigma[:5]],
        "n_entities_in_geometry": len(names),
        "n_entities_in_fit": len(common),
        "exclude_names": sorted(exclude_names),
        "seed_pair": list(seed_pair) if seed_pair else None,
    }

    return FitResult(
        direction=direction_full,
        direction_pcs=adopted_pcs,
        pc_basis=Vt,
        rho_train=rho_train,
        rho_cv=rho_cv,
        cluster_summary=summary,
        restarts=restarts,
        metric=metric,
    )


# ---------------------------------------------------------------------------
# Score loading helpers (CLI mode B)
# ---------------------------------------------------------------------------

SCORE_SOURCE_CHOICES = ["gpt", "sonnet", "haiku", "di_combined", "responses"]


def _flatten_static_scores(raw: dict) -> dict[str, float]:
    """Normalise either {name: int} or {name: {score: ..., scores: [...]}}."""
    out: dict[str, float] = {}
    for name, val in raw.items():
        if isinstance(val, (int, float)):
            out[name] = float(val)
        elif isinstance(val, dict):
            if "score" in val and isinstance(val["score"], (int, float)):
                out[name] = float(val["score"])
            elif "scores" in val and isinstance(val["scores"], list):
                nums = [s for s in val["scores"] if isinstance(s, (int, float))]
                if nums:
                    out[name] = float(np.mean(nums))
    return out


def _flatten_response_scores(raw: dict) -> dict[str, float]:
    """Mean-score per entity from an axis_judge_correlation responses cache."""
    out: dict[str, float] = {}
    for name, info in raw.items():
        if isinstance(info, dict):
            ms = info.get("mean_score")
            if isinstance(ms, (int, float)):
                out[name] = float(ms)
    return out


def score_source_paths(experiment_dir: Path, source: str) -> list[Path]:
    """Return the on-disk paths a given ``--score_source`` reads.

    Used by the provenance machinery to fingerprint the judge caches
    that fed an :func:`optimal_axis_for_judge` run.  Mirrors the
    branches in :func:`load_scores_from_experiment_dir`.  Non-existent
    paths are still returned (the caller filters); ``responses``
    gracefully skips missing roles/traits.
    """
    if source in ("gpt", "sonnet", "haiku"):
        d = experiment_dir / source
        return [d / "scores_descriptions.json",
                d / "scores_instructions.json"]
    if source == "di_combined":
        out: list[Path] = []
        for sub in ("gpt", "sonnet"):
            for fname in ("scores_descriptions.json",
                          "scores_instructions.json"):
                out.append(experiment_dir / sub / fname)
        return out
    if source == "responses":
        return [experiment_dir / response_subdir("gpt", mode) / "scores_responses.json"
                for mode in ("roles", "traits")]
    raise ValueError(f"Unknown source {source!r}; "
                     f"choose from {SCORE_SOURCE_CHOICES}")


def load_scores_from_experiment_dir(
    experiment_dir: Path, source: str,
    *,
    di_weights: tuple[float, float] = DI_WEIGHT_CHOICES["inst_tie"],
    inputs: list[InputSpec] | None = None,
    kinds_for_name: dict[str, set] | None = None,
) -> dict[str, float]:
    """Load a per-entity score series from an axis-judge-correlation
    experiment directory.

    Threads ``inputs`` through every cache read via
    ``load_and_register`` so the caller's recorded provenance lists
    exactly the files we consumed.

    sources:
      gpt           : combine_desc_inst_one_judge(gpt/desc, gpt/inst)
      sonnet        : combine_desc_inst_one_judge(sonnet/desc, sonnet/inst)
      haiku         : combine_desc_inst_one_judge(haiku/desc, haiku/inst)
      di_combined   : combine_desc_inst_two_judges(gpt, sonnet)  -- default 4-way
      responses     : average gpt_responses_{roles,traits}_b{N}/scores_responses.json
                       per-entity mean_score, where ``b{N}`` is
                       :data:`assistant_axis.judge_batch.RESPONSE_BATCH_SIZE`.
    """
    axis_id = experiment_dir.name

    def _load(path: Path, dep_key: str) -> dict:
        payload, _spec, _check = load_and_register(
            path, dep_key=dep_key, inputs=inputs, policy="warn",
        )
        return payload

    if source in ("gpt", "sonnet", "haiku"):
        d = experiment_dir / source
        sd = _flatten_static_scores(_load(
            d / "scores_descriptions.json",
            f"judge_{axis_id}_descriptions_{source}"))
        si = _flatten_static_scores(_load(
            d / "scores_instructions.json",
            f"judge_{axis_id}_instructions_{source}"))
        return combine_desc_inst_one_judge(sd, si, weights=di_weights)
    if source == "di_combined":
        gd = experiment_dir / "gpt"
        sd_p = experiment_dir / "sonnet"
        g_d = _flatten_static_scores(_load(
            gd / "scores_descriptions.json",
            f"judge_{axis_id}_descriptions_gpt"))
        g_i = _flatten_static_scores(_load(
            gd / "scores_instructions.json",
            f"judge_{axis_id}_instructions_gpt"))
        s_d = _flatten_static_scores(_load(
            sd_p / "scores_descriptions.json",
            f"judge_{axis_id}_descriptions_sonnet"))
        s_i = _flatten_static_scores(_load(
            sd_p / "scores_instructions.json",
            f"judge_{axis_id}_instructions_sonnet"))
        # When kinds_for_name is supplied, lift any v1 (bare-name) caches
        # to v2 (entity_id) keys in-memory so the 4-way intersection in
        # combine_desc_inst_two_judges actually has any overlap when one
        # judge is post-Phase-5 (v2) and the other hasn't been rejudged
        # yet (v1).  No-op on already-v2 input, so safe to apply blindly.
        if kinds_for_name is not None:
            g_d = migrate_v1_static_scores(g_d, kinds_for_name)
            g_i = migrate_v1_static_scores(g_i, kinds_for_name)
            s_d = migrate_v1_static_scores(s_d, kinds_for_name)
            s_i = migrate_v1_static_scores(s_i, kinds_for_name)
        return combine_desc_inst_two_judges(g_d, g_i, s_d, s_i, weights=di_weights)
    if source == "responses":
        # Path-suffix tracks the canonical project-wide response batch
        # size (assistant_axis.judge_batch.RESPONSE_BATCH_SIZE).
        out: dict[str, float] = {}
        for mode in ("roles", "traits"):
            sub = response_subdir("gpt", mode)
            path = experiment_dir / sub / "scores_responses.json"
            if path.exists():
                bare = _flatten_response_scores(_load(
                    path, f"judge_{axis_id}_gpt_responses_{mode}"))
                for n, score in bare.items():
                    out[entity_id(n, mode)] = score
        return out
    raise ValueError(f"Unknown --score_source {source!r}; "
                     f"choose from {SCORE_SOURCE_CHOICES}")


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
        ).decode().strip()
    except Exception:                                          # noqa: BLE001
        return "unknown"


def write_outputs(
    output_dir: Path, result: FitResult, *, run_label: str,
    argv: list[str], wallclock_s: float,
    inputs: list[InputSpec] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # direction.pt
    payload = {
        "direction": torch.from_numpy(result.direction.astype(np.float32)),
        "direction_pcs": torch.from_numpy(result.direction_pcs.astype(np.float32)),
        "pc_basis_Vt": torch.from_numpy(result.pc_basis.astype(np.float32)),
        "metric": result.metric,
    }
    torch.save(payload, output_dir / "direction.pt")

    # diagnostics.json (without restart d_pcs -- those go in restart_dirs.pt
    # since they're numeric arrays).  Wrapped in a provenance envelope
    # so callers can validate freshness against current inputs.  The
    # ``.pt`` siblings are part of the same atomic write bundle and
    # inherit the diagnostics.json's provenance transitively (they
    # don't need their own envelopes since torch.save isn't a JSON
    # target).
    diag = {
        "metric": result.metric,
        "rho_train": result.rho_train,
        "rho_cv": result.rho_cv,
        "cluster_summary": result.cluster_summary,
        "restarts": [
            {
                "seed_label": r.seed_label,
                "final_rho_train": r.final_rho_train,
                "n_passes": r.n_passes,
                "cluster_id": r.cluster_id,
            } for r in result.restarts
        ],
        "git_sha": _git_sha(),
        "argv": " ".join(argv),
        "wallclock_s": wallclock_s,
    }
    diag_envelope = json_metadata(
        diag, inputs=inputs, title=f"optimal_axis_for_judge: {run_label}")
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diag_envelope, indent=2))

    # restart_dirs.pt: per-restart final directions in PC basis, for
    # downstream cluster analysis or cross-source consistency studies.
    # Sign-aligned to the highest-rho restart.
    if result.restarts:
        D = np.stack([r.final_d_pcs for r in result.restarts], axis=0)
        norms = np.linalg.norm(D, axis=1, keepdims=True)
        D = D / np.maximum(norms, 1e-12)
        ref_idx = int(np.argmax([r.final_rho_train for r in result.restarts]))
        # Flip rows that are anti-aligned with reference.
        for i in range(len(D)):
            if float(D[i] @ D[ref_idx]) < 0:
                D[i] = -D[i]
        torch.save({
            "restart_d_pcs": torch.from_numpy(D.astype(np.float32)),
            "labels": [r.seed_label for r in result.restarts],
            "rhos_train": [r.final_rho_train for r in result.restarts],
            "cluster_ids": [r.cluster_id for r in result.restarts],
            "metric": result.metric,
        }, output_dir / "restart_dirs.pt")

    # restarts.png
    write_restarts_plot(output_dir / "restarts.png", result, run_label,
                        inputs=inputs)


def write_restarts_plot(out_path: Path, result: FitResult, run_label: str,
                          *, inputs: list[InputSpec] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    rs = result.restarts
    rhos = [r.final_rho_train for r in rs]
    cids = [r.cluster_id for r in rs]
    cmap = plt.colormaps["tab10"]
    colors = [cmap(c % 10) for c in cids]
    xs = np.arange(len(rs))
    ax.scatter(xs, rhos, c=colors, s=80, edgecolor="black", linewidth=0.6, zorder=5)
    for x, r in zip(xs, rs):
        ax.text(x, r.final_rho_train + 0.005, r.seed_label,
                ha="center", va="bottom", fontsize=7, rotation=45)

    cs = result.cluster_summary
    if cs.get("rho_pole_diff") is not None:
        ax.axhline(cs["rho_pole_diff"], color="grey", linestyle=":",
                    linewidth=1.4, label=f"pole_diff baseline ρ={cs['rho_pole_diff']:.3f}")
    if cs.get("rho_score_weighted_mean") is not None:
        ax.axhline(cs["rho_score_weighted_mean"], color="firebrick", linestyle=":",
                    linewidth=1.4, label=f"score_weighted_mean ρ={cs['rho_score_weighted_mean']:.3f}")
    ax.axhline(result.rho_train, color="black", linestyle="-", linewidth=1.0,
                label=f"adopted ρ={result.rho_train:.3f}"
                + (f"  (cv {result.rho_cv:.3f})" if result.rho_cv is not None else ""))

    ax.set_xlabel("restart index (coloured by cluster)")
    ax.set_ylabel("Spearman ρ on training scores")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs], fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

    title = f"Optimal-judge-axis restarts: {run_label}"
    metric = result.metric
    tc_n = cs["top_cluster"]["n_members"]
    tc_med = cs["top_cluster"]["median_pairwise_cos"]
    spec = (f"slot={metric['slot']}, layer={metric['layer']}, "
            f"whitening={metric['whitening']}, M={metric['M']}; "
            f"n_entities_fit={metric['n_entities_in_fit']}; "
            f"top cluster: {tc_n}/{cs['n_restarts']} restarts "
            f"(intra med cos {tc_med:.3f})")
    _, top_rect = suptitle_with_specs(fig, title, spec)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title,
                                       source_text=Path(__file__).read_text(),
                                       inputs=inputs))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _slug_whitening(spec: str) -> str:
    """Filesystem-safe slug for a whitening spec.

    Strips ``=`` and ``_`` (which appear in canonical specs like
    ``soft_K=3`` and ``soft_shear=2``) so the result is a single token.

    >>> _slug_whitening("soft_K=3")
    'softK3'
    >>> _slug_whitening("soft_shear=2")
    'softshear2'
    >>> _slug_whitening("raw")
    'raw'
    """
    return spec.replace("=", "").replace("_", "")


def _slug_data_dir(data_dir: str | Path) -> str:
    """Filesystem-safe short slug for a data_dir.

    Uses the basename only (the rest is just where the dataset lives on
    this machine, not part of its identity) and replaces spaces with
    hyphens.

    >>> _slug_data_dir("runpod_workspace/qwen/qwen-3-32b Roger 8slot")
    'qwen-3-32b-Roger-8slot'
    """
    name = Path(str(data_dir)).name
    return name.replace(" ", "-") or "datadir"


def _derive_pair_label(args: argparse.Namespace) -> str:
    """Pull a stable pair-label out of the argparse Namespace.

    Priority: ``--seed_pair POS NEG`` -> ``f"{POS}_vs_{NEG}"``;
    else ``--scores_file foo.json`` -> ``Path(foo).stem``.  At least one
    must be set (the script already requires either Mode A or Mode B).
    """
    if args.seed_pair:
        pos, neg = args.seed_pair
        return f"{pos}_vs_{neg}"
    if args.scores_file:
        return Path(args.scores_file).stem
    raise ValueError(
        "Cannot derive --output_dir without --seed_pair or --scores_file; "
        "pass one or set --output_dir explicitly.")


def derive_output_dir(args: argparse.Namespace,
                      base: Path = DEFAULT_OUTPUT_BASE) -> Path:
    """Auto-derive ``--output_dir`` from the full cache key.

    The dir name encodes every dimension that a different invocation of
    this script could vary -- pair, score source, slot, layer, whitening
    spec, working-subspace dimension M, and data_dir basename -- so two
    runs with different metric or input choices land in two different
    directories *even when one or more dimensions are at their current
    default*.  Defaults change over time (slot was 3 pre-May 2026, then
    6; data_dir was 4-slot, now 8-slot); encoding them explicitly stops
    a future default flip from silently shadowing prior outputs.

    Examples
    --------
    Default invocation with ``--seed_pair truthful deceitful``,
    ``--score_source gpt``, ``--slot 6``::

        roger/optimal_axis/truthful_vs_deceitful_gpt_slot6_layer25_softK3_M30_qwen-3-32b-Roger-8slot/

    Explicit ``--output_dir`` overrides this entirely (use it for one-off
    or comparison runs where you want a hand-picked dir name).
    """
    pair = _derive_pair_label(args)
    src = args.score_source if args.experiment_dir else "raw_scores"
    name = (
        f"{pair}_{src}"
        f"_slot{args.slot}"
        f"_layer{args.layer}"
        f"_{_slug_whitening(args.whitening)}"
        f"_M{args.M}"
        f"_{_slug_data_dir(args.data_dir)}"
    )
    return base / name


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("score source (mode A or B)")
    src.add_argument("--scores_file", type=str, default=None,
                     help="Mode A: JSON file mapping entity name -> float score.")
    src.add_argument("--experiment_dir", type=str, default=None,
                     help="Mode B: per-axis dir produced by axis_judge_correlation.")
    src.add_argument("--score_source", type=str, default="di_combined",
                     choices=SCORE_SOURCE_CHOICES,
                     help="(Mode B) which scores to load from --experiment_dir. "
                          "'di_combined' = GPT+Sonnet 4-way combined desc+inst.")

    geom = p.add_argument_group("geometry / metric")
    geom.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    geom.add_argument("--slot", type=int, default=DEFAULT_SLOT)
    geom.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    geom.add_argument("--whitening", type=str, default=DEFAULT_WHITENING,
                      help="'raw' or 'soft_K=N' (e.g. 'soft_K=3').")
    geom.add_argument("--M", type=int, default=DEFAULT_M,
                      help="Number of top PCs to use as the working subspace.")
    geom.add_argument("--exclude_names", type=str, default="",
                      help="Comma-separated entity names to omit from both pool "
                           "fit and PCA (e.g. the pole-defining traits).")
    geom.add_argument("--seed_pair", nargs=2, default=None, metavar=("POS", "NEG"),
                      help="Optional pair to seed the pole-difference restart.")

    opt = p.add_argument_group("optimiser")
    opt.add_argument("--n_restarts", type=int, default=DEFAULT_N_RESTARTS)
    opt.add_argument("--max_passes", type=int, default=DEFAULT_MAX_PASSES)
    opt.add_argument("--conv_eps", type=float, default=DEFAULT_CONV_EPS)
    opt.add_argument("--K_seed", type=int, default=DEFAULT_K_SEED)
    opt.add_argument("--cv_folds", type=int, default=DEFAULT_CV_FOLDS)
    opt.add_argument("--cluster_cos_threshold", type=float,
                     default=DEFAULT_CLUSTER_COS_THRESHOLD,
                     help="|cos| threshold for single-link clustering of "
                          "restart directions.  Default 0.95 (strict). At "
                          "M >= 75 a looser threshold (0.90) often merges "
                          "near-duplicate clusters.")
    opt.add_argument("--rng_seed", type=int, default=0)

    out = p.add_argument_group("output")
    out.add_argument("--output_dir", type=str, default=None,
                     help="Where to write direction.pt, diagnostics.json, "
                          "restarts.png.  When omitted, auto-derives from the "
                          "full cache key (pair, score_source, slot, layer, "
                          "whitening, M, data_dir basename) under "
                          f"{DEFAULT_OUTPUT_BASE}/ so every distinct invocation "
                          "lands in its own dir.  See derive_output_dir() for "
                          "the format.")
    out.add_argument("--run_label", type=str, default=None,
                     help="Short label for the plot/title (defaults to the "
                          "output dir basename).")
    return p.parse_args(argv)


def _normalize_scores_file_keys(
    scores: dict[str, float],
    *,
    src: Path,
) -> dict[str, float]:
    """Coerce display-form keys in a user-supplied --scores_file dict
    to canonical file-name form.

    Project convention is file-name form everywhere except display
    sites (see AGENT_NOTES "File-name vs display-name convention"),
    but external JSONs sometimes leak the display form -- spaces,
    hyphens, capitals, apostrophes -- which silently misses on every
    multi-word entity at the downstream
    ``set(scores) & set(geometry_names)`` intersection (~3 % of
    names: ``aligned_artificial_intelligence``, ``systems_thinker``,
    ``devils_advocate``, ...).

    For each key:
    * Pass through ``name|R``/``name|T`` disambiguated ids
      unchanged (the suffix tag is already canonical; rewriting the
      bare-name half would risk lower-casing a kind tag).
    * Otherwise apply :func:`normalize_to_file_name`.

    If the conversion is one-to-one we return the converted dict and
    print a WARNING listing up to 5 conversions, so the user notices
    their input was massaged.  If the conversion would collide (e.g.
    the file contains both ``Patient`` and ``patient`` mapping to
    the same key, or both ``systems_thinker`` and ``systems-thinker``)
    we raise ``SystemExit`` rather than silently drop one.
    """
    raw_to_norm: list[tuple[str, str, float]] = []
    for k, v in scores.items():
        # Pre-disambiguated id (``name|R``, ``name|T``) — pass through.
        if "|" in k and k.rsplit("|", 1)[1] in ("R", "T"):
            raw_to_norm.append((k, k, v))
        else:
            raw_to_norm.append((k, normalize_to_file_name(k), v))

    by_norm: dict[str, list[tuple[str, float]]] = {}
    for raw, norm, val in raw_to_norm:
        by_norm.setdefault(norm, []).append((raw, val))

    collisions = {n: [r for r, _ in raws]
                  for n, raws in by_norm.items() if len(raws) > 1}
    if collisions:
        head = "; ".join(
            f"{n!r} <- {raws!r}"
            for n, raws in list(collisions.items())[:3]
        )
        raise SystemExit(
            f"--scores_file {src}: {len(collisions)} entity name(s) "
            f"collide after display->file normalisation ({head}).  "
            f"This usually means the file mixes display and file "
            f"conventions for the same entity.  Pick one form per "
            f"entity; see AGENT_NOTES 'File-name vs display-name "
            f"convention'."
        )

    coerced = [(raw, norm) for raw, norm, _v in raw_to_norm if raw != norm]
    if coerced:
        head = "; ".join(
            f"{r!r} -> {n!r}" for r, n in coerced[:5]
        ) + ("..." if len(coerced) > 5 else "")
        # No module logger here; print to stderr so the warning is
        # impossible to miss in a regular run.
        print(
            f"WARNING: --scores_file {src}: coerced "
            f"{len(coerced)} display-form key(s) to file-name form "
            f"({head}).  Future inputs should use file-name form "
            f"directly; see AGENT_NOTES 'File-name vs display-name "
            f"convention'.",
            file=sys.stderr,
        )
    return {norm: v for _raw, norm, v in raw_to_norm}


def _load_scores_from_args(
    args: argparse.Namespace,
    inputs: list[InputSpec] | None = None,
) -> dict[str, float]:
    """Load judge scores per --scores_file or --experiment_dir mode.

    Threads ``inputs`` through every cache read via load_and_register
    (or current_file_input for the scores_file's pre-known path), so
    the caller's recorded provenance lists exactly the files we
    consumed.  Pre-retrofit this script ran a separate _build_inputs
    pass after the load that registered files via score_source_paths
    -- which could disagree with the actual reads inside
    load_scores_from_experiment_dir.

    External ``--scores_file`` JSON keys are passed through
    :func:`_normalize_scores_file_keys` (boundary input hardening:
    coerce display-form names to file-name form, with a WARNING).
    The ``--experiment_dir`` branch reads our own caches, which are
    already in file-name form by construction, so it skips this.
    """
    if args.scores_file and args.experiment_dir:
        raise SystemExit("Pass exactly one of --scores_file or --experiment_dir.")
    if args.scores_file:
        src = Path(args.scores_file)
        raw, _spec, _check = load_and_register(
            src,
            dep_key="scores_file",
            inputs=inputs, policy="warn",
        )
        # Accept both {name: float} and {name: {score: float}} forms.
        if all(isinstance(v, (int, float)) for v in raw.values()):
            scores = {str(k): float(v) for k, v in raw.items()}
        else:
            scores = _flatten_static_scores(raw)
        return _normalize_scores_file_keys(scores, src=src)
    if args.experiment_dir:
        # Build kinds_for_name from the corpus so v1 (bare-name) judge
        # caches can be lifted to v2 (entity_id) keys in-memory.  Only
        # ``di_combined`` actually intersects multiple judges (which is
        # where v1/v2 silent-drop happens); we still build the map for
        # all sources so a future v1 cache on a single-judge source
        # would error loudly rather than silently dropping collision
        # entities.
        kinds_for_name: dict[str, set] = {}
        data_dir = Path(args.data_dir)
        for et in ("traits", "roles"):
            vec_dir = data_dir / et / "vectors"
            if not vec_dir.is_dir():
                continue
            for fp in sorted(vec_dir.glob("*.pt")):
                if fp.stem == "default":
                    continue
                kinds_for_name.setdefault(fp.stem, set()).add(et)
        return load_scores_from_experiment_dir(
            Path(args.experiment_dir), args.score_source,
            inputs=inputs,
            kinds_for_name=kinds_for_name,
        )
    raise SystemExit("Must pass either --scores_file or --experiment_dir.")


def _build_inputs(args: argparse.Namespace) -> list[InputSpec]:
    """Construct the up-front InputSpec list for this run -- the two
    dataset subtrees that don't go through load_and_register.

    Per-cache judge file InputSpecs are appended at read time inside
    ``_load_scores_from_args`` (via load_and_register), so this
    function intentionally returns only the subtree deps; merging the
    two streams happens at the call site.  Pre-retrofit this function
    duplicated the per-cache registrations as a post-load pass, which
    risked falling out of sync with what was actually consumed.
    """
    data_dir = Path(args.data_dir)
    inputs: list[InputSpec] = [
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras={"slot": str(args.slot), "layer": str(args.layer)}),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras={"slot": str(args.slot), "layer": str(args.layer)}),
    ]
    return inputs


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = (Path(args.output_dir) if args.output_dir
               else derive_output_dir(args))
    run_label = args.run_label or out_dir.name

    # Build subtree deps first; per-cache judge file deps get appended
    # by ``_load_scores_from_args`` at read time via load_and_register.
    inputs = _build_inputs(args)
    judge_scores = _load_scores_from_args(args, inputs=inputs)
    if not judge_scores:
        raise SystemExit("Empty judge_scores after loading; check --score_source.")
    print(f"Loaded {len(judge_scores)} judge scores")

    exclude = set([s for s in args.exclude_names.split(",") if s.strip()])
    seed_pair = tuple(args.seed_pair) if args.seed_pair else None

    t0 = time.time()
    result = find_optimal_direction(
        judge_scores=judge_scores,
        data_dir=Path(args.data_dir),
        slot=args.slot, layer=args.layer, whitening=args.whitening,
        M=args.M,
        n_restarts=args.n_restarts, max_passes=args.max_passes,
        conv_eps=args.conv_eps, K_seed=args.K_seed,
        cv_folds=args.cv_folds, seed_pair=seed_pair,
        exclude_names=exclude, rng_seed=args.rng_seed,
        cluster_threshold=args.cluster_cos_threshold,
    )
    wallclock = time.time() - t0

    print(f"\n=== Result for {run_label} ===")
    print(f"  rho_train: {result.rho_train:+.4f}")
    if result.rho_cv is not None:
        print(f"  rho_cv:    {result.rho_cv:+.4f}")
    cs = result.cluster_summary
    if cs.get("rho_pole_diff") is not None:
        print(f"  vs pole_diff: ρ={cs['rho_pole_diff']:+.4f}, "
              f"cos={cs['cos_to_pole_diff']:.3f}")
    if cs.get("rho_score_weighted_mean") is not None:
        print(f"  vs score_weighted_mean: ρ={cs['rho_score_weighted_mean']:+.4f}, "
              f"cos={cs['cos_to_score_weighted_mean']:.3f}")
    tc = cs["top_cluster"]
    print(f"  top cluster: {tc['n_members']}/{cs['n_restarts']} restarts, "
          f"cosine min={tc['min_pairwise_cos']:.3f} median={tc['median_pairwise_cos']:.3f}")

    write_outputs(out_dir, result, run_label=run_label,
                   argv=sys.argv if argv is None else argv,
                   wallclock_s=wallclock, inputs=inputs)
    print(f"\nWrote outputs to {out_dir}/ in {wallclock:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
