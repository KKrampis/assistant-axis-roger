#!/usr/bin/env python3
"""Learn a per-axis whitening-K policy with a random forest.

Reframes the manual cascading colour-rule (red/yellow/green/blue) as a
supervised "lowest K needed" problem and asks whether a random forest
of axis-aligned threshold rules can outperform constant-K baselines
under cross-validation.

Problem statement
-----------------

Given the post-whitening cosine sequence
``x_k = |cos(axis_w_{K=k-1}, PC_k)|``  for k = 1..4
and the observed ρ@K for K ∈ {0, 1, 2, 3}, pick a K per axis to
maximise the cohort mean ρ.

The "lowest K needed" prior labels each axis with::

    K*(axis) = min { K  :  ρ@K(axis) ≥ max_K' ρ@K'(axis) − ε }

so an axis that gets within ε of its individual best at K=0 is
labelled ``0``, even if ρ@K=2 is technically slightly higher.  This
penalises gratuitous escalation to higher K (which throws away more
variance) and matches the structure of the manual cascading rules.

Models compared
---------------

* **All @ K=k**  (constants for k=0..3): trivial baselines.
* **Oracle**   : per-axis argmax K; upper bound, no training needed.
* **Hand cascade**: the manual threshold rule
  (:func:`_categorize_row` + ``red→K=0, yellow→K=1, green→K=2, else
  K=3``) — what we've been hand-tuning.
* **RF multiclass**: ``RandomForestClassifier`` on K* (one of 4
  classes) with features ``(x_1..x_4, x_2/x_1, x_3/x_2, x_4/x_3)``.
* **RF ordinal** : three cascading binary RF classifiers ("need K≥1?",
  "need K≥2?", "need K≥3?"), exactly mimicking the structure of the
  hand cascade.
* **RF regression**: predict ρ@K for each K (multi-output), then pick
  smallest K within ε of the predicted max.
* **Random K**   : sanity baseline (uniform random over {0,1,2,3}).

Cross-validation
----------------

6-fold split with **exactly 2 primary axes per fold**.  The 23
desc+instr-only "extras" are distributed round-robin across the same
folds (4, 4, 4, 4, 4, 3).  A deterministic seed (``--seed``)
controls within-cohort shuffling so re-runs are stable.

For each fold, the policy is fit on the other 5 folds and the cohort
mean ρ is evaluated on the held-out fold using the true (cached)
ρ@K values.  Final report is mean ± std of these per-fold means.

Outputs
-------

A bar chart comparing CV mean ρ per method with per-fold error bars,
plus a console table of per-fold scores and feature importances.

Defaults to ``slot 6, layer 25`` and the full 35-axis pair list.

Examples
--------

::

    uv run python results_analysis/axis_pc_k_policy_rf.py

    # Tune ε:
    uv run python results_analysis/axis_pc_k_policy_rf.py \\
        --epsilon_sweep 0.005 0.010 0.020 0.030
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from assistant_axis import (
    cohort_from_pairs,
    png_metadata,
    suptitle_with_specs,
)
from assistant_axis.provenance import (
    CACHE_POLICIES,
    InputSpec,
    current_data_subtree_input,
    load_and_register,
)
from results_analysis.axis_pc_alignment_vs_peak_K import (
    DEFAULT_DATA_DIR,
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_PAIR_LIST,
    DEFAULT_PRIMARY_PAIR_LIST,
    DEFAULT_SLOT,
    LAYER,
    N_PCS,
    _CATEGORY_ORDER,
    _categorize_row,
    _index_fits,
    _index_sweep,
    _per_axis_record,
)


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

# Features used by all RF models.  Order matters for the importance
# report at the end.
FEATURE_NAMES = (
    "x_1", "x_2", "x_3", "x_4",
    "x_2/x_1", "x_3/x_2", "x_4/x_3",
)


def _row_features(row: dict) -> np.ndarray:
    """Return a 1-D feature vector for one axis row."""
    cw = np.asarray(row.get("cos_post_whitening", []), dtype=float)
    if cw.size < 4:
        return np.full(len(FEATURE_NAMES), np.nan)
    x1, x2, x3, x4 = float(cw[0]), float(cw[1]), float(cw[2]), float(cw[3])
    eps = 1e-6
    return np.array([
        x1, x2, x3, x4,
        x2 / max(x1, eps),
        x3 / max(x2, eps),
        x4 / max(x3, eps),
    ], dtype=float)


# ---------------------------------------------------------------------
# Targets and labels
# ---------------------------------------------------------------------

def _rho_vec(
    row: dict, Ks: tuple[int, ...] = (0, 1, 2, 3),
) -> np.ndarray:
    """Return ρ@K vector (NaN-tolerant) for ``row`` over the given Ks."""
    rho_k = row.get("rho_at_K", {})
    return np.array([float(rho_k.get(K, float("nan"))) for K in Ks])


def _lowest_K_label(
    rho: np.ndarray, epsilon: float,
    K_values: tuple[int, ...] = (0, 1, 2, 3),
) -> int:
    """Return the K (drawn from ``K_values``) at the lowest index of
    ``K_values`` whose ρ is within ε of the max ρ over ``K_values``.

    ``rho`` must be ordered to match ``K_values`` (same length).
    """
    if not np.any(np.isfinite(rho)):
        return int(K_values[0])
    rho_max = float(np.nanmax(rho))
    for i in range(len(rho)):
        if np.isfinite(rho[i]) and rho[i] >= rho_max - epsilon:
            return int(K_values[i])
    return int(K_values[int(np.nanargmax(rho))])


def _snap_to_K_values(K: int, K_values: tuple[int, ...]) -> int:
    """Map an arbitrary integer K to the nearest value in ``K_values``."""
    K_arr = np.asarray(K_values)
    return int(K_arr[int(np.argmin(np.abs(K_arr - K)))])


# ---------------------------------------------------------------------
# Cross-validation splits
# ---------------------------------------------------------------------

def _build_folds(
    rows: list[dict],
    *,
    n_folds: int = 6,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return list of length ``n_folds`` of arrays of row indices.

    Primary rows are shuffled (seeded) and dealt round-robin into the
    folds, then extras are shuffled and dealt round-robin into the
    same folds.  With 12 primary and 6 folds, this produces exactly 2
    primary per fold; extras distribute 4/4/4/4/4/3 over 23.
    """
    rng = np.random.default_rng(seed)
    primary_idx = np.array([i for i, r in enumerate(rows) if r["primary"]])
    extra_idx = np.array([i for i, r in enumerate(rows) if not r["primary"]])
    rng.shuffle(primary_idx)
    rng.shuffle(extra_idx)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for k, idx in enumerate(primary_idx):
        folds[k % n_folds].append(int(idx))
    for k, idx in enumerate(extra_idx):
        folds[k % n_folds].append(int(idx))
    return [np.asarray(sorted(f)) for f in folds]


# ---------------------------------------------------------------------
# Policies (each returns chosen K per row)
# ---------------------------------------------------------------------

def _const_K(rows: list[dict], K: int) -> np.ndarray:
    return np.full(len(rows), K, dtype=int)


def _oracle_K(
    rows: list[dict], epsilon: float,
    K_values: tuple[int, ...] = (0, 1, 2, 3),
) -> np.ndarray:
    return np.array(
        [_lowest_K_label(_rho_vec(r, K_values), epsilon, K_values)
         for r in rows],
        dtype=int,
    )


# Hand cascade: red→K=0, yellow→K=1, green→K=2, blue→K=3.
_HAND_CASCADE_K = {"red": 0, "yellow": 1, "green": 2, "blue": 3, "gray": 1}


def _hand_cascade_K(
    rows: list[dict],
    K_values: tuple[int, ...] = (0, 1, 2, 3),
) -> np.ndarray:
    """Hand cascade clipped/snapped to ``K_values``."""
    out = np.empty(len(rows), dtype=int)
    for i, r in enumerate(rows):
        cat, _ = _categorize_row(r)
        K_raw = _HAND_CASCADE_K.get(cat, 1)
        out[i] = _snap_to_K_values(K_raw, K_values)
    return out


def _random_K(
    n: int, seed: int,
    K_values: tuple[int, ...] = (0, 1, 2, 3),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(rng.choice(K_values, size=n), dtype=int)


# ---------------------------------------------------------------------
# RF policies
# ---------------------------------------------------------------------

def _fit_rf_multiclass(
    X_train: np.ndarray, y_train: np.ndarray,
    n_estimators: int, max_depth: int | None,
    min_samples_leaf: int, seed: int,
    sample_weight: np.ndarray | None = None,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=1,
    ).fit(X_train, y_train, sample_weight=sample_weight)


def _fit_rf_ordinal(
    X_train: np.ndarray, y_train: np.ndarray,
    n_estimators: int, max_depth: int | None,
    min_samples_leaf: int, seed: int,
    K_values: tuple[int, ...] = (0, 1, 2, 3),
    sample_weight: np.ndarray | None = None,
) -> list[RandomForestClassifier]:
    """One binary RF per threshold in ``K_values[1:]`` — each asks
    "should we escalate past this K?".

    For ``K_values = (0, 1, 2, 3)`` this gives 3 binary classifiers
    (y ≥ 1?, y ≥ 2?, y ≥ 3?).  For ``K_values = (0, 1)`` it
    degenerates to a single binary classifier (y ≥ 1?).  Predictions
    accumulate via :func:`_predict_rf_ordinal`.
    """
    classifiers = []
    for threshold in K_values[1:]:
        y_bin = (y_train >= threshold).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            classifiers.append(("const", int(y_bin[0]) if len(y_bin) else 0))
            continue
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed + int(threshold),
            n_jobs=1,
        ).fit(X_train, y_bin, sample_weight=sample_weight)
        classifiers.append(("rf", clf))
    return classifiers


def _predict_rf_ordinal(
    classifiers: list, X_test: np.ndarray,
    K_values: tuple[int, ...] = (0, 1, 2, 3),
) -> np.ndarray:
    """Sum binary "should escalate?" predictions to an index into
    ``K_values``; return ``K_values[index]``."""
    n_escalations = np.zeros(X_test.shape[0], dtype=int)
    for kind, model in classifiers:
        if kind == "const":
            n_escalations += int(model)
        else:
            n_escalations += model.predict(X_test).astype(int)
    idx = np.clip(n_escalations, 0, len(K_values) - 1)
    return np.asarray(K_values)[idx].astype(int)


def _fit_rf_regression(
    X_train: np.ndarray, Y_rho_train: np.ndarray,
    n_estimators: int, max_depth: int | None,
    min_samples_leaf: int, seed: int,
    sample_weight: np.ndarray | None = None,
) -> RandomForestRegressor:
    """Multi-output regression on ρ@K (4 outputs)."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=1,
    ).fit(X_train, Y_rho_train, sample_weight=sample_weight)


def _axis_regret_weight(
    rho: np.ndarray, *,
    kind: str = "mean",
    eps_floor: float = 1e-3,
) -> float:
    """Per-axis "decision-relevance" weight.

    ``kind`` controls which summary of the ρ@K distribution drives the
    weight:

    * ``"second_best"``  -- ``max(ρ) − 2nd-max(ρ)``.  Only reflects the
      cost of picking the next-best K; misses how bad the worst K
      would be.  Adequate for binary K∈{0,1} (where 2nd-max = min) but
      *under*-weights multi-K axes whose 3rd/4th choices collapse.
    * ``"mean"`` (default) -- ``max(ρ) − mean(ρ)``.  The average regret
      of picking K uniformly at random.  Uses the full distribution
      and reduces to ``max − 2nd-max`` in the binary case (since
      ``mean({a, b}) = (a+b)/2 = a − (a−b)/2`` so ``max − mean = (max
      − 2nd)/2``, a positive monotone of the binary weight).
    * ``"min_max"``      -- ``max(ρ) − min(ρ)``.  The worst-case
      misprediction cost; gives the largest weight to axes with one
      good K and several bad ones.

    All variants floor at ``eps_floor`` so every axis still
    participates in training (ties barely contribute to splits).
    """
    r = rho[np.isfinite(rho)]
    if r.size < 2:
        return float(eps_floor)
    if kind == "second_best":
        sorted_r = np.sort(r)[::-1]
        w = float(sorted_r[0] - sorted_r[1])
    elif kind == "min_max":
        w = float(r.max() - r.min())
    elif kind == "mean":
        w = float(r.max() - r.mean())
    else:
        raise ValueError(f"Unknown regret-weight kind: {kind!r}")
    return max(w, float(eps_floor))


def _fit_rf_regret_weighted(
    X_train: np.ndarray, y_train: np.ndarray,
    Y_rho_train: np.ndarray,
    n_estimators: int, max_depth: int | None,
    min_samples_leaf: int, seed: int,
    sample_weight: np.ndarray | None = None,
    regret_weight_kind: str = "mean",
    eps_floor: float = 1e-3,
) -> RandomForestClassifier:
    """Multiclass RF with sample weights = primary_weight × per-axis
    regret-relevance.

    ``regret_weight_kind`` selects how the per-axis weight summarises
    the ρ@K distribution (see :func:`_axis_regret_weight`):

    * ``"mean"`` (default): ``max(ρ) − mean(ρ)`` -- uses the full
      distribution, properly penalises axes whose 3rd/4th K choices
      collapse.
    * ``"min_max"``: ``max(ρ) − min(ρ)`` -- worst-case regret.
    * ``"second_best"``: ``max(ρ) − 2nd-max(ρ)`` -- only the closest
      alternative; adequate for binary K but underweights multi-K
      axes with bad tail choices.
    """
    regret_w = np.array(
        [_axis_regret_weight(Y_rho_train[i],
                             kind=regret_weight_kind,
                             eps_floor=eps_floor)
         for i in range(len(y_train))],
        dtype=float,
    )
    if sample_weight is None:
        sample_weight = np.ones_like(regret_w)
    full_w = sample_weight * regret_w
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=1,
    ).fit(X_train, y_train, sample_weight=full_w)


def _fit_rf_regret_regression(
    X_train: np.ndarray, Y_rho_train: np.ndarray,
    n_estimators: int, max_depth: int | None,
    min_samples_leaf: int, seed: int,
    sample_weight: np.ndarray | None = None,
) -> RandomForestRegressor:
    """Multi-output regression where each target is ``regret@K =
    max(ρ) − ρ@K`` instead of raw ρ@K.

    Lets the regressor focus on relative differences between Ks
    rather than absolute ρ levels.  At inference, pick K with minimum
    predicted regret (see :func:`_predict_rf_regret_regression`).
    """
    Y_regret = Y_rho_train.max(axis=1, keepdims=True) - Y_rho_train
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=1,
    ).fit(X_train, Y_regret, sample_weight=sample_weight)


def _predict_rf_regret_regression(
    model: RandomForestRegressor, X_test: np.ndarray,
    K_values: tuple[int, ...] = (0, 1, 2, 3),
    epsilon: float = 0.0,
) -> np.ndarray:
    """Pick lowest K (in K_values order) whose predicted regret is
    within ``epsilon`` of the minimum predicted regret."""
    pred = model.predict(X_test)  # (n_test, len(K_values))
    if pred.ndim == 1:
        pred = pred[:, None]
    out = np.empty(pred.shape[0], dtype=int)
    for i in range(pred.shape[0]):
        # We want LOWEST K with regret close to MIN regret, so use
        # the same "_lowest_K_label" trick on -regret (since min regret
        # corresponds to max -regret).
        out[i] = _lowest_K_label(-pred[i], epsilon, K_values)
    return out


def _predict_rf_regression(
    model: RandomForestRegressor, X_test: np.ndarray, epsilon: float,
    K_values: tuple[int, ...] = (0, 1, 2, 3),
) -> np.ndarray:
    pred = model.predict(X_test)  # (n_test, len(K_values))
    if pred.ndim == 1:
        # Single-output collapse (sklearn returns 1-D when n_outputs=1).
        pred = pred[:, None]
    out = np.empty(pred.shape[0], dtype=int)
    for i in range(pred.shape[0]):
        out[i] = _lowest_K_label(pred[i], epsilon, K_values)
    return out


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------

def _score_policy(
    rows: list[dict], K_per_row: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted-mean ρ@K(chosen) over ``rows`` (skipping NaN ρ rows).

    ``weights`` is an array of per-row sample weights (1.0 for extras,
    ``primary_weight`` for primary by default).  If ``None``, defaults
    to uniform weights.
    """
    if len(rows) == 0:
        return float("nan")
    if weights is None:
        weights = np.ones(len(rows), dtype=float)
    rhos: list[float] = []
    ws: list[float] = []
    for r, K, w in zip(rows, K_per_row, weights):
        rho = r["rho_at_K"].get(int(K), float("nan"))
        if np.isfinite(rho):
            rhos.append(float(rho))
            ws.append(float(w))
    if not rhos:
        return float("nan")
    return float(np.sum(np.array(rhos) * np.array(ws)) / np.sum(ws))


# ---------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------

def run_cv(
    rows: list[dict],
    *,
    epsilon: float,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    n_folds: int,
    seed: int,
    primary_weight: float = 1.0,
    K_values: tuple[int, ...] = (0, 1, 2, 3),
) -> dict:
    """Run one full cross-validation pass; return result dict.

    ``primary_weight`` (default 1.0) is the per-sample weight given to
    primary-cohort axes; extras always get weight 1.0.  Setting
    ``primary_weight = 5.0`` reflects "each axis with response judging
    is worth ~5x an axis without", on the rationale that response-mode
    ρ carries ~80% of the final-blend signal vs ~20% for desc+instr.

    The weights are passed both to (a) the sklearn ``.fit(...)``
    ``sample_weight`` kwarg, and (b) the cohort-mean ρ aggregation in
    ``_score_policy``, so they are applied consistently in training
    and evaluation.
    """
    folds = _build_folds(rows, n_folds=n_folds, seed=seed)
    n_rows = len(rows)
    X = np.array([_row_features(r) for r in rows])
    y = np.array([_lowest_K_label(_rho_vec(r, K_values), epsilon, K_values)
                  for r in rows])
    Y_rho = np.array([_rho_vec(r, K_values) for r in rows])
    sample_w = np.array(
        [primary_weight if r["primary"] else 1.0 for r in rows],
        dtype=float,
    )

    methods = [f"All @ K={K}" for K in K_values]
    methods += [
        "Random K",
        "Hand cascade",
        "RF multiclass",
        "RF ordinal",
        "RF regression",
        "RF regret-wt(2nd)",
        "RF regret-wt(mean)",
        "RF regret-wt(min-max)",
        "RF regret-reg",
        "Oracle (ε)",
    ]
    per_fold: dict[str, list[float]] = {m: [] for m in methods}
    importances_multi: list[np.ndarray] = []
    importances_reg: list[np.ndarray] = []

    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.array(sorted(set(range(n_rows)) - set(test_idx.tolist())))
        rows_test = [rows[i] for i in test_idx]
        X_test, X_train = X[test_idx], X[train_idx]
        y_train = y[train_idx]
        Y_rho_train = Y_rho[train_idx]
        w_train = sample_w[train_idx]
        w_test = sample_w[test_idx]

        # Constants (one per K in K_values)
        for K_const in K_values:
            per_fold[f"All @ K={K_const}"].append(
                _score_policy(rows_test, _const_K(rows_test, K_const),
                              weights=w_test))

        # Random K (uniform over K_values)
        K_rand = _random_K(len(rows_test), seed=seed + fold_idx,
                           K_values=K_values)
        per_fold["Random K"].append(
            _score_policy(rows_test, K_rand, weights=w_test))

        # Hand cascade (snapped to K_values)
        K_hand = _hand_cascade_K(rows_test, K_values=K_values)
        per_fold["Hand cascade"].append(
            _score_policy(rows_test, K_hand, weights=w_test))

        # RF multiclass
        rf_mc = _fit_rf_multiclass(
            X_train, y_train, n_estimators, max_depth,
            min_samples_leaf, seed + fold_idx,
            sample_weight=w_train)
        K_mc = rf_mc.predict(X_test).astype(int)
        per_fold["RF multiclass"].append(
            _score_policy(rows_test, K_mc, weights=w_test))
        importances_multi.append(rf_mc.feature_importances_)

        # RF ordinal (one binary RF per threshold in K_values[1:])
        rf_ord = _fit_rf_ordinal(
            X_train, y_train, n_estimators, max_depth,
            min_samples_leaf, seed + fold_idx,
            K_values=K_values, sample_weight=w_train)
        K_ord = _predict_rf_ordinal(rf_ord, X_test, K_values=K_values)
        per_fold["RF ordinal"].append(
            _score_policy(rows_test, K_ord, weights=w_test))

        # RF regression on ρ@K  (len(K_values) outputs)
        rf_reg = _fit_rf_regression(
            X_train, Y_rho_train, n_estimators, max_depth,
            min_samples_leaf, seed + fold_idx,
            sample_weight=w_train)
        K_reg = _predict_rf_regression(rf_reg, X_test, epsilon,
                                       K_values=K_values)
        per_fold["RF regression"].append(
            _score_policy(rows_test, K_reg, weights=w_test))
        importances_reg.append(rf_reg.feature_importances_)

        # RF regret-weighted: three variants of the "decision-relevance"
        # weight, side-by-side, to expose how the choice of summary
        # statistic for the ρ@K distribution interacts with the
        # multi-K problem.
        for kind, label in (("second_best",  "RF regret-wt(2nd)"),
                            ("mean",         "RF regret-wt(mean)"),
                            ("min_max",      "RF regret-wt(min-max)")):
            rf_rw = _fit_rf_regret_weighted(
                X_train, y_train, Y_rho_train, n_estimators, max_depth,
                min_samples_leaf, seed + fold_idx,
                sample_weight=w_train,
                regret_weight_kind=kind,
            )
            K_rw = rf_rw.predict(X_test).astype(int)
            per_fold[label].append(
                _score_policy(rows_test, K_rw, weights=w_test))

        # RF regret-regression (multi-output regression on regret@K
        # instead of ρ@K).
        rf_rr = _fit_rf_regret_regression(
            X_train, Y_rho_train, n_estimators, max_depth,
            min_samples_leaf, seed + fold_idx,
            sample_weight=w_train)
        K_rr = _predict_rf_regret_regression(rf_rr, X_test,
                                             K_values=K_values,
                                             epsilon=epsilon)
        per_fold["RF regret-reg"].append(
            _score_policy(rows_test, K_rr, weights=w_test))

        # Oracle on this fold (computed from true rho)
        K_oracle = _oracle_K(rows_test, epsilon, K_values=K_values)
        per_fold["Oracle (ε)"].append(
            _score_policy(rows_test, K_oracle, weights=w_test))

    means = {m: float(np.mean(per_fold[m])) for m in methods}
    stds = {m: float(np.std(per_fold[m], ddof=1)) if len(per_fold[m]) > 1
            else 0.0 for m in methods}
    mean_imp_mc = np.mean(np.array(importances_multi), axis=0) \
        if importances_multi else None
    mean_imp_reg = np.mean(np.array(importances_reg), axis=0) \
        if importances_reg else None
    return {
        "methods": methods,
        "per_fold": per_fold,
        "means": means,
        "stds": stds,
        "y_labels": y,
        "importances_multi": mean_imp_mc,
        "importances_reg": mean_imp_reg,
        "folds": folds,
        "n_rows": n_rows,
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def _print_report(cv: dict, *, epsilon: float, header: str) -> None:
    print(f"\n=== {header} (ε = {epsilon:.3f}) ===")
    K_values = cv.get("K_values", (0, 1, 2, 3))
    print(f"Label distribution (K* under ε = {epsilon:.3f}):")
    total = len(cv["y_labels"])
    for K in K_values:
        n = int(np.sum(np.asarray(cv["y_labels"]) == K))
        pct = 100.0 * n / max(total, 1)
        print(f"  K* = {K}:  {n:3d}  ({pct:5.1f}%)")
    print()

    print(f"{'method':20s}  {'mean ρ':>9s}  {'± std':>8s}   "
          + "  ".join(f"f{i}" for i in range(len(cv['folds']))))
    print("-" * 72)
    order = sorted(cv["methods"],
                   key=lambda m: cv["means"][m], reverse=True)
    for m in order:
        per = "  ".join(f"{v:+.3f}" for v in cv["per_fold"][m])
        print(f"{m:20s}  {cv['means'][m]:+9.4f}  {cv['stds'][m]:8.4f}   "
              f"{per}")

    if cv["importances_multi"] is not None:
        print("\nFeature importance (RF multiclass, mean over folds):")
        for name, imp in zip(FEATURE_NAMES, cv["importances_multi"]):
            print(f"  {name:>8s}  {imp:.3f}")


def make_plot(
    cv: dict,
    *,
    epsilon: float,
    out_path: Path,
    inputs: list[InputSpec],
    title_extra: str,
) -> None:
    methods = cv["methods"]
    means = np.array([cv["means"][m] for m in methods])
    stds = np.array([cv["stds"][m] for m in methods])

    fig, ax = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
    x = np.arange(len(methods))
    colors = []
    for m in methods:
        if m.startswith("All @"):
            colors.append("#4C72B0")  # blue
        elif m == "Random K":
            colors.append("#999999")  # gray
        elif m == "Hand cascade":
            colors.append("#DDAA33")  # gold
        elif m.startswith("RF"):
            colors.append("#CC3311")  # red
        elif m.startswith("Oracle"):
            colors.append("#117733")  # green
        else:
            colors.append("#777777")
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.5)

    best_const_K1 = cv["means"]["All @ K=1"]
    ax.axhline(best_const_K1, ls="--", color="#4C72B0", alpha=0.5,
               linewidth=1.0, zorder=0)

    best_idx = int(np.argmax(means))
    for i, b in enumerate(bars):
        weight = "bold" if i == best_idx else "normal"
        ax.text(x[i], means[i] + stds[i] + 0.003, f"{means[i]:+.3f}",
                ha="center", va="bottom", fontsize=9, fontweight=weight)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("CV mean ρ  (±1 std across 6 folds)", fontsize=11)
    ax.set_ylim(min(0.0, float(means.min() - stds.max()) - 0.02),
                float(max(means.max(), 1.0) + stds.max()) * 1.0 + 0.04)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=0.5)

    title = "Per-axis K-policy: random forest vs hand cascade vs constants"
    pw = cv.get("primary_weight", 1.0)
    Ks = cv.get("K_values", (0, 1, 2, 3))
    weight_blurb = (f"primary axes weighted {pw:g}x (extras = 1x)"
                    if pw != 1.0 else "uniform weights")
    K_blurb = f"K ∈ {{{', '.join(str(k) for k in Ks)}}}"
    spec = (f"6-fold CV (2 primary/fold, 4-3 extras/fold).  "
            f"{K_blurb}.  "
            f"Label = min K with ρ@K ≥ max−ε  (ε = {epsilon:.3f}).  "
            f"{weight_blurb}.  {title_extra}")
    suptitle_with_specs(fig, title=title, specs=spec)

    fig.savefig(out_path, dpi=140, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    plt.close(fig)
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--experiment_dir", type=Path,
                   default=DEFAULT_EXPERIMENT_DIR)
    p.add_argument("--pairs", default=DEFAULT_PAIR_LIST)
    p.add_argument("--pairs_primary", default=DEFAULT_PRIMARY_PAIR_LIST)
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT)
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--epsilon", type=float, default=0.010,
                   help="Tolerance for 'lowest K that hurts by ≤ ε'.")
    p.add_argument("--epsilon_sweep", type=float, nargs="*",
                   help="Optional list of ε values to evaluate.")
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--min_samples_leaf", type=int, default=2)
    p.add_argument("--n_folds", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K_values", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="Set of K values the policy is allowed to choose "
                        "from (default 0 1 2 3).  Pass '0 1' to restrict "
                        "to the binary K=0/K=1 problem (loses some oracle "
                        "headroom but is far easier to learn).")
    p.add_argument("--primary_weight", type=float, default=5.0,
                   help="Sample weight for primary-cohort axes (axes with "
                        "response judging).  Extras (desc+instr only) always "
                        "get weight 1.0.  Default 5.0 reflects the ~5x "
                        "leverage primary axes have in the final-blend "
                        "signal (~80%% response vs ~20%% desc+instr).  Use "
                        "1.0 for an unweighted comparison.")
    p.add_argument("--out", default=None)
    p.add_argument("--cache_policy", default="strict",
                   choices=tuple(CACHE_POLICIES))
    args = p.parse_args(argv)

    data_dir = args.data_dir
    experiment_dir = args.experiment_dir

    cohort = "di"  # full cohort uses desc+instr-blended ρ
    fit_filename = f"whitening_k_peak_fit_{cohort}_slot{args.slot}.json"
    sweep_filename = f"whitening_k_sweep_{cohort}_slot{args.slot}.json"

    inputs: list[InputSpec] = []
    pairs, _, _ = load_and_register(
        experiment_dir / args.pairs, dep_key="pairs_json",
        inputs=inputs, policy=args.cache_policy)
    pairs_primary, _, _ = load_and_register(
        experiment_dir / args.pairs_primary, dep_key="pairs_primary_json",
        inputs=inputs, policy=args.cache_policy)
    fits, _, _ = load_and_register(
        experiment_dir / fit_filename, dep_key="peak_fit_json",
        inputs=inputs, policy=args.cache_policy)
    sweep, _, _ = load_and_register(
        experiment_dir / sweep_filename, dep_key="k_sweep_json",
        inputs=inputs, policy=args.cache_policy)
    inputs.extend([
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras={"slot": str(args.slot), "layer": str(args.layer)}),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras={"slot": str(args.slot), "layer": str(args.layer)}),
    ])

    primary_set = {(it["pos"], it["neg"]) for it in pairs_primary}
    fits_by_pair = _index_fits(fits)
    sweep_by_pair = _index_sweep(sweep)

    bottom_panel_Ks = (0, 1, 2, 3)
    rows: list[dict] = []
    for pair in pairs:
        rows.append(_per_axis_record(
            pair, fits_by_pair=fits_by_pair,
            sweep_by_pair=sweep_by_pair,
            primary_set=primary_set,
            data_dir=data_dir, slot=args.slot, layer=args.layer,
            n_pcs=N_PCS, bottom_panel_Ks=bottom_panel_Ks,
        ))
    # Drop rows with no valid post-whitening cos or ρ vector.
    rows = [r for r in rows if np.all(np.isfinite(_row_features(r)))
            and np.any(np.isfinite(_rho_vec(r)))]
    n_primary = sum(1 for r in rows if r["primary"])
    print(f"Loaded {len(rows)} usable axes "
          f"({n_primary} primary + {len(rows) - n_primary} extras).")
    print(f"RF config: n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, "
          f"min_samples_leaf={args.min_samples_leaf}, "
          f"folds={args.n_folds}, seed={args.seed}")

    K_values = tuple(sorted(set(int(k) for k in args.K_values)))
    eps_list = args.epsilon_sweep or [args.epsilon]
    eps_results: dict[float, dict] = {}
    for eps in eps_list:
        cv = run_cv(
            rows,
            epsilon=eps,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_folds=args.n_folds,
            seed=args.seed,
            primary_weight=args.primary_weight,
            K_values=K_values,
        )
        cv["primary_weight"] = args.primary_weight
        cv["K_values"] = K_values
        _print_report(cv, epsilon=eps, header=f"ε = {eps:.3f}")
        eps_results[eps] = cv

    # Pick the headline ε for the plot: --epsilon (or the single value
    # in --epsilon_sweep if length-1).
    headline_eps = args.epsilon if args.epsilon in eps_results \
        else eps_list[0]
    cv = eps_results[headline_eps]

    cohort_label = cohort_from_pairs(args.pairs)
    K_suffix = "K" + "".join(str(k) for k in K_values) \
        if K_values != (0, 1, 2, 3) else ""
    out_filename = args.out or (
        f"axis_pc_k_policy_rf_{cohort_label}_slot{args.slot}"
        f"{('_' + K_suffix) if K_suffix else ''}.png"
    )
    out_path = experiment_dir / out_filename
    title_extra = (f"slot {args.slot}, layer {args.layer}, "
                   f"n={cv['n_rows']} axes")
    make_plot(cv, epsilon=headline_eps, out_path=out_path,
              inputs=inputs, title_extra=title_extra)

    # If we swept ε, show the headline comparison table.
    if len(eps_results) > 1:
        print("\n=== ε sweep summary (mean ρ across 6 folds) ===")
        ms = list(eps_results[eps_list[0]]["methods"])
        header = "method              " + "  ".join(f"ε={e:.3f}"
                                                    for e in eps_list)
        print(header)
        print("-" * len(header))
        for m in ms:
            cells = "  ".join(f"{eps_results[e]['means'][m]:+.4f}"
                              for e in eps_list)
            print(f"{m:20s}  {cells}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
