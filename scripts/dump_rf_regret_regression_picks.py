#!/usr/bin/env python3
"""Dump per-axis K picks for the RF regret-regression policy at ε=0.005.

That policy was the nominal "winner-by-a-hair" in the K-policy CV (mean
ρ ≈ 0.6939 vs K=1 baseline ≈ 0.6928, primary 5x weights).  This script
re-runs the same CV folds and prints which K it actually selected for
each axis -- mostly K=1?  A handful of K=0?  Anything else?

Output: per-axis table sorted (primary first, then alphabetical), with
the predicted K, the true K* (ε=0.005), and ρ at predicted vs ρ at K=1
to show whether the swap helped or hurt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis.provenance import (
    InputSpec, current_data_subtree_input, load_and_register,
)
from results_analysis.axis_pc_k_policy_rf import (  # noqa: E402
    DEFAULT_DATA_DIR, DEFAULT_EXPERIMENT_DIR, DEFAULT_PAIR_LIST,
    DEFAULT_PRIMARY_PAIR_LIST, DEFAULT_SLOT, LAYER, N_PCS,
    _build_folds, _fit_rf_regret_regression, _index_fits,
    _index_sweep, _lowest_K_label, _per_axis_record,
    _predict_rf_regret_regression, _rho_vec, _row_features,
)


def main() -> int:
    epsilon = 0.005
    K_values = (0, 1, 2, 3)
    primary_weight = 5.0
    n_estimators = 300
    max_depth = 4
    min_samples_leaf = 2
    n_folds = 6
    seed = 0
    slot = DEFAULT_SLOT
    layer = LAYER

    experiment_dir = DEFAULT_EXPERIMENT_DIR
    data_dir = DEFAULT_DATA_DIR
    cohort = "di"
    fit_filename = f"whitening_k_peak_fit_{cohort}_slot{slot}.json"
    sweep_filename = f"whitening_k_sweep_{cohort}_slot{slot}.json"

    inputs: list[InputSpec] = []
    pairs, _, _ = load_and_register(
        experiment_dir / DEFAULT_PAIR_LIST, dep_key="pairs_json",
        inputs=inputs, policy="warn")
    pairs_primary, _, _ = load_and_register(
        experiment_dir / DEFAULT_PRIMARY_PAIR_LIST,
        dep_key="pairs_primary_json", inputs=inputs, policy="warn")
    fits, _, _ = load_and_register(
        experiment_dir / fit_filename, dep_key="peak_fit_json",
        inputs=inputs, policy="warn")
    sweep, _, _ = load_and_register(
        experiment_dir / sweep_filename, dep_key="k_sweep_json",
        inputs=inputs, policy="warn")
    inputs.append(current_data_subtree_input(
        data_dir, "traits/vectors", dep_key="traits_vectors",
        extras={"slot": str(slot), "layer": str(layer)}))
    inputs.append(current_data_subtree_input(
        data_dir, "roles/vectors", dep_key="roles_vectors",
        extras={"slot": str(slot), "layer": str(layer)}))

    primary_set = {(it["pos"], it["neg"]) for it in pairs_primary}
    fits_by_pair = _index_fits(fits)
    sweep_by_pair = _index_sweep(sweep)

    rows = []
    for pair in pairs:
        rows.append(_per_axis_record(
            pair, fits_by_pair=fits_by_pair,
            sweep_by_pair=sweep_by_pair,
            primary_set=primary_set,
            data_dir=data_dir, slot=slot, layer=layer,
            n_pcs=N_PCS, bottom_panel_Ks=K_values,
        ))
    rows = [r for r in rows if np.all(np.isfinite(_row_features(r)))
            and np.any(np.isfinite(_rho_vec(r)))]

    folds = _build_folds(rows, n_folds=n_folds, seed=seed)
    X = np.array([_row_features(r) for r in rows])
    y_label = np.array(
        [_lowest_K_label(_rho_vec(r, K_values), epsilon, K_values)
         for r in rows])
    Y_rho = np.array([_rho_vec(r, K_values) for r in rows])
    sample_w = np.array(
        [primary_weight if r["primary"] else 1.0 for r in rows],
        dtype=float)

    # Run the same CV procedure as run_cv() but only RF regret-regression.
    n_rows = len(rows)
    predicted_K = np.full(n_rows, -1, dtype=int)

    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.array(sorted(
            set(range(n_rows)) - set(test_idx.tolist())))
        X_test, X_train = X[test_idx], X[train_idx]
        Y_rho_train = Y_rho[train_idx]
        w_train = sample_w[train_idx]
        rf_rr = _fit_rf_regret_regression(
            X_train, Y_rho_train, n_estimators, max_depth,
            min_samples_leaf, seed + fold_idx,
            sample_weight=w_train)
        K_rr = _predict_rf_regret_regression(
            rf_rr, X_test, K_values=K_values, epsilon=epsilon)
        predicted_K[test_idx] = K_rr

    primary_idx = [i for i, r in enumerate(rows) if r["primary"]]
    extras_idx = [i for i, r in enumerate(rows) if not r["primary"]]

    def _print_section(label: str, idxs: list[int]) -> None:
        print(f"\n--- {label} ({len(idxs)} axes) ---")
        print(f"{'axis':40s}  pred  K*  rho_pred   rho@K=1   delta_K1   delta_oracle")
        idxs_sorted = sorted(idxs, key=lambda i: rows[i]["pos"])
        for i in idxs_sorted:
            r = rows[i]
            name = f"{r['pos']} vs {r['neg']}"
            kp = int(predicted_K[i])
            kstar = int(y_label[i])
            rho_p = r["rho_at_K"].get(kp, float("nan"))
            rho_k1 = r["rho_at_K"].get(1, float("nan"))
            rho_max = max(
                [r["rho_at_K"].get(K, -np.inf) for K in K_values])
            d_k1 = rho_p - rho_k1
            d_or = rho_p - rho_max
            print(f"{name:40s}  {kp:>3d}  {kstar:>3d}  "
                  f"{rho_p:+.4f}   {rho_k1:+.4f}   "
                  f"{d_k1:+.4f}    {d_or:+.4f}")

    print(f"=== RF regret-regression CV picks at ε={epsilon} "
          f"(primary {primary_weight:g}x) ===")
    print(f"Counts: predicted K distribution = "
          + ", ".join(f"K={K}:{int(np.sum(predicted_K == K))}"
                      for K in K_values))
    print(f"        label    K* distribution = "
          + ", ".join(f"K={K}:{int(np.sum(y_label == K))}"
                      for K in K_values))

    _print_section("PRIMARY", primary_idx)
    _print_section("EXTRAS", extras_idx)

    # Summary deltas
    rho_pred = np.array(
        [rows[i]["rho_at_K"].get(int(predicted_K[i]), float("nan"))
         for i in range(n_rows)])
    rho_k1 = np.array(
        [rows[i]["rho_at_K"].get(1, float("nan")) for i in range(n_rows)])
    rho_oracle = np.array(
        [max(rows[i]["rho_at_K"].get(K, -np.inf) for K in K_values)
         for i in range(n_rows)])
    w = sample_w
    valid = np.isfinite(rho_pred) & np.isfinite(rho_k1)
    m_pred = float(np.sum(rho_pred[valid] * w[valid]) / np.sum(w[valid]))
    m_k1 = float(np.sum(rho_k1[valid] * w[valid]) / np.sum(w[valid]))
    m_or = float(np.sum(rho_oracle[valid] * w[valid]) / np.sum(w[valid]))
    print(f"\nWeighted cohort means (primary {primary_weight}x):")
    print(f"  RF regret-reg @ε={epsilon}:  {m_pred:+.4f}")
    print(f"  All @ K=1            :  {m_k1:+.4f}  "
          f"(Δ = {m_pred - m_k1:+.4f})")
    print(f"  Oracle (ε=0)         :  {m_or:+.4f}  "
          f"(Δ = {m_pred - m_or:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
