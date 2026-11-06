#!/usr/bin/env python3
"""Compute cosine between two cross-layer/slot mappings of canonical Nth PC:

  (a) NAIVE: treat d_canonical (computed at the canonical cell) as a vector
      in R^D and use it unchanged at the target cell.  This is what
      ``plot_direction_cosines.py`` does in the fixed_direction variant.

  (b) PRINCIPLED: express d_canonical as a linear combination of pool-entity
      vectors at the canonical cell (coefficients α = U[:, N-1] / σ_{N-1}
      from PCA), then apply the same coefficients to the entities at the
      target cell.  This keeps the inter-entity geometric structure intact
      across the cell change.

The output is a table of cos(d_naive_at_target, d_principled_at_target)
per PC × per target (slot, layer).  Cosines computed in raw R^D (no shear).

May 2026 migration: canonical = 8-slot, slot 7, layer 25, raw (L=0).
Targets cover same-slot ±1 layer drift, far layer drift, and cross-slot
moves into the cells used by the round-trip sweep (slot 6, slot 0,
slot 3 = old canonical).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from results_analysis.canonical_angles.data import DEFAULT_DATA_DIR
from results_analysis.canonical_angles.whitening import fit_shear
from results_analysis.pc_round_trip.klm_sweep import (
    setup_at, compute_pc_directions,
)
from results_analysis.pc_round_trip.plot_direction_cosines import (
    woodbury_inverse_shear_apply,
)


SLOT_C, LAYER_C = 7, 25
# Decompose slot-only vs layer-only vs both contributions.
TARGETS = [
    (7, 26),   # same slot, layer +1 (pure layer drift, minimal)
    (7, 24),   # same slot, layer -1
    (7, 49),   # same slot, far layer (used in 2-cell sweep)
    (6, 25),   # different slot (slot-only drift)
    (6, 49),   # different slot, far layer (4-cell sweep)
    (0, 26),   # different slot, used in 6-cell sweep
    (0, 49),   # different slot, far layer (6-cell sweep)
    (3, 25),   # old canonical — diagnostic point (7-cell sweep)
]
CANONICAL_L = 0
PCS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    data_dir = Path(DEFAULT_DATA_DIR)

    # ---- Canonical setup at (SLOT_C, LAYER_C) -----------------------------
    names_c, M_raw_c, A_g_c, A_n_c, _ = setup_at(data_dir, SLOT_C, LAYER_C)
    sh_c = fit_shear(A_g_c, A_n_c, L=CANONICAL_L)
    M_canonical = sh_c.apply(M_raw_c)
    M_canon_mean = M_canonical.mean(axis=0, keepdims=True)
    M_canon_centered = M_canonical - M_canon_mean

    # SVD: M_canon_centered = U S V^T.  V^T[k] is the (k+1)-th PC direction.
    U, S, Vt = np.linalg.svd(M_canon_centered, full_matrices=False)
    # Coefficients for each PC: α_k = U[:, k] / S[k]   (length n_entities)
    # Verify: V^T[k] = α_k @ M_canon_centered.

    # Vector-view inverse of canonical shear: direction transforms like a
    # data point.  Reuse the helper from plot_direction_cosines.py for
    # consistency.
    inv_sh_c = woodbury_inverse_shear_apply(sh_c)

    rows = []
    for pc in PCS:
        if pc - 1 >= Vt.shape[0]:
            continue
        # d_canonical in canonical (L=2) shear space:
        d_canon_in_L2 = Vt[pc - 1]
        # In raw R^D (vector view: undo the shear):
        d_canon_in_raw = inv_sh_c(d_canon_in_L2[None, :])[0]
        # Coefficients for the PC as a linear combo of centered canonical
        # entities:
        alpha = U[:, pc - 1] / S[pc - 1]      # (n_entities,)
        for slot_t, layer_t in TARGETS:
            names_t, M_raw_t, _, _, _ = setup_at(data_dir, slot_t, layer_t)
            if names_t != names_c:
                # Should always match — entities are the same set.
                raise RuntimeError(
                    f"name mismatch at ({slot_t},{layer_t})")
            # PRINCIPLED layer-target direction: alpha @ centered M_raw at
            # the target layer.  Use raw (no shear) at target since we want
            # an apples-to-apples raw-space cosine vs d_canon_in_raw.
            M_t_centered = M_raw_t - M_raw_t.mean(axis=0, keepdims=True)
            d_principled_raw = alpha @ M_t_centered  # (D,)
            cos_naive_vs_principled = cos(d_canon_in_raw, d_principled_raw)
            rows.append((pc, slot_t, layer_t, cos_naive_vs_principled))

    print(f"\n{'PC':>4}  {'target':>10}  cos(d_canon@raw, d_principled@target)")
    print("-" * 60)
    for pc, s, l, c in rows:
        target = f"({s},{l})"
        print(f"{pc:>4}  {target:>10}  {c:>+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
