#!/usr/bin/env python3
"""Decompose nth_pc winner directions in the canonical PC basis.

For each PC index N, for each n-cell winner × {glossary, inline} style,
take the nth_pc winner direction (a unit vector in some (slot, layer,
L, K) post-whitened space), map it back to canonical raw space at
``(SLOT_C, LAYER_C)``, and decompose its squared norm across the
canonical PC basis.

Mapping back from a winner at (slot, layer, L_w, K_w):
  1. Vt[N-1] at the winner cell — direction in post-whitened space.
  2. Inverse K-whiten (vector view) → (slot, layer, L_w) sheared space.
  3. Inverse L-shear at (slot, layer) → raw R^D at (slot, layer).
  4. (Raw space) least-squares-project onto target's centered M_raw row
     span: α_inv = d @ pinv(M_raw_target_centered).  Coefficients in
     entity space.
  5. (Raw space) reconstruct at canonical:
     d_canon_raw = α_inv @ M_raw_canonical_centered.
  6. Apply canonical shear (forward, vector view) — identity if L=0.
  7. Decompose: c_k = d_canon @ Vt_canonical[k] for k = 0..n_canon-1.
  8. Squared coefficients (normalised to sum to 1).

May 2026 migration: canonical = 8-slot, slot 7, layer 25, L=0 raw, so
step 6 is identity and steps 3-5 produce a direction directly comparable
to the canonical raw-space PCA.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np

from assistant_axis.provenance import (
    CACHE_POLICIES, InputSpec, current_data_subtree_input,
    load_and_register,
)
from results_analysis.canonical_angles.data import DEFAULT_DATA_DIR
from results_analysis.canonical_angles.whitening import (
    fit_shear, fit_whitening, WhiteningBasis,
)
from results_analysis.pc_round_trip.klm_sweep import (
    setup_at, compute_M_done, compute_pc_directions,
    DEFAULT_PCS, DEFAULT_STYLES,
)
from results_analysis.pc_round_trip.plot_direction_cosines import (
    woodbury_inverse_shear_apply, woodbury_inverse_whiten_apply,
)


# May 2026 migration: 8-slot dataset, slot 7, layer 25, raw (L=0).
SLOT_C, LAYER_C = 7, 25
CANONICAL_L = 0
DEFAULT_CELL_ORDER = [(SLOT_C, LAYER_C), (7, 49), (6, 25), (6, 49),
                       (0, 26), (0, 49), (3, 25)]


def load_per_cell_winners(per_cell_path: Path,
                           policy: str = "warn",
                           inputs: list[InputSpec] | None = None,
                           dep_key: str = "per_cell_winners_json",
                           ) -> dict[tuple[int, str, int, int], dict]:
    """Load per-cell winners from a JSON cache produced by
    ``plot_direction_cosines.py --include_optimization``.

    Transparently unwraps the ``{"_provenance": ..., "result": ...}``
    envelope written by post-May-2026 runs; older bare JSONs load
    unchanged.  When an ``inputs`` accumulator is supplied, the cache
    file is also recorded as a dependency under ``dep_key`` (matching
    the read+register pattern documented in AGENT_NOTES.md so callers
    can't accidentally consume a cache without declaring it)."""
    raw, _spec, _check = load_and_register(
        per_cell_path,
        dep_key=dep_key,
        inputs=inputs,
        policy=policy,
    )
    out: dict[tuple[int, str, int, int], dict] = {}
    for k, v in raw.items():
        # key format: pc{NNN}_{style}_s{S}_l{L}
        pc = int(k[2:5])
        rest = k[6:]
        style_part, sl = rest.rsplit("_s", 1)
        slot_str, layer_str = sl.split("_l", 1)
        out[(pc, style_part, int(slot_str), int(layer_str))] = v
    return out


def n_cell_winner(per_cell: dict[tuple[int, str, int, int], dict],
                   pc: int, style: str, n: int,
                   cell_order: list[tuple[int, int]]) -> Optional[dict]:
    """Return the (rho, L, K, slot, layer) of the best cell over the
    first n cells of cell_order for (pc, style)."""
    cells = cell_order[:n]
    best = None
    for slot, layer in cells:
        v = per_cell.get((pc, style, slot, layer))
        if v is None:
            continue
        if best is None or v["rho"] > best["rho"]:
            best = dict(v) | {"slot": slot, "layer": layer}
    return best


def setup_canonical(data_dir: Path):
    """Compute canonical setup at (3, 25, L=2, K=0).  Returns names,
    M_raw_centered, sh_canonical, M_canonical_centered, Vt_canonical."""
    names, M_raw, A_g, A_n, _pool = setup_at(data_dir, SLOT_C, LAYER_C)
    M_raw_centered = M_raw - M_raw.mean(axis=0, keepdims=True)
    sh_canonical = fit_shear(A_g, A_n, L=CANONICAL_L)
    M_canonical = sh_canonical.apply(M_raw)
    M_canon_centered = M_canonical - M_canonical.mean(axis=0, keepdims=True)
    _U, _S, Vt_canonical = np.linalg.svd(M_canon_centered, full_matrices=False)
    return names, M_raw_centered, sh_canonical, M_canon_centered, Vt_canonical


def winner_direction_in_canonical_L2(
    pc: int, slot: int, layer: int, L_w: int, K_w: int,
    *,
    data_dir: Path,
    sh_canonical,
    M_raw_canonical_centered: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute the nth_pc winner direction at (slot, layer, L_w, K_w),
    map it back to canonical L=2 sheared space at (3, 25).

    Returns a (D,) array, or None if the winner cell SVD fails."""
    # Build the winner cell's M_done and the nth PC direction.
    names_t, M_raw_t, A_g_t, A_n_t, pool_t = setup_at(data_dir, slot, layer)
    sh_cache: dict = {}
    wh_cache: dict = {}
    M_done_t, _ = compute_M_done(
        M_raw_t, A_g_t, A_n_t, pool_t, L_w, K_w, sh_cache, wh_cache
    )
    Vt_t = compute_pc_directions(M_done_t, pc)
    if Vt_t is None or pc - 1 >= Vt_t.shape[0]:
        return None
    d_post = Vt_t[pc - 1]                        # (D,)

    # Step 2: inverse K-whitening (vector view).
    if K_w > 0:
        sh_t = sh_cache[L_w] if L_w > 0 else None
        wh_t = wh_cache[(L_w, K_w)]
        d = woodbury_inverse_whiten_apply(wh_t)(d_post[None, :])[0]
    else:
        d = d_post.copy()

    # Step 3: inverse L-shear at (slot, layer).
    if L_w > 0:
        sh_t = sh_cache[L_w]
        d = woodbury_inverse_shear_apply(sh_t)(d[None, :])[0]
    # Now d is in raw R^D at (slot, layer).

    # Steps 4–5: raw-space inverse mapping target→canonical.
    if (slot, layer) == (SLOT_C, LAYER_C):
        d_canon_raw = d
    else:
        # Least-squares-project onto target's M_raw_centered row span.
        M_raw_t_centered = M_raw_t - M_raw_t.mean(axis=0, keepdims=True)
        # alpha_inv such that alpha_inv @ M_raw_t_centered ≈ d.
        # alpha_inv = d @ pinv(M_raw_t_centered)  shape (n_entities,)
        alpha_inv = d @ np.linalg.pinv(M_raw_t_centered)
        # Reconstruct in canonical raw space.
        d_canon_raw = alpha_inv @ M_raw_canonical_centered

    # Step 6: forward canonical L=2 shear.
    d_canon_L2 = sh_canonical.apply(d_canon_raw[None, :])[0]
    return d_canon_L2.astype(np.float32)


def compute_decomposition_matrix(
    per_cell: dict[tuple[int, str, int, int], dict],
    *,
    data_dir: Path,
    pcs: list[int],
    cell_order: list[tuple[int, int]] = DEFAULT_CELL_ORDER,
    n_max: int = 3,
    n_canon_pcs: int = 0,
    normalise_columns: bool = True,
) -> tuple[np.ndarray, list[tuple[int, str, int]], dict]:
    """Build the (n_canon_pcs, n_pcs * len(STYLES) * n_max) heatmap matrix.

    Column order: outer = pc, then style, then n.  So for pcs P and
    styles [glossary, inline] and n in 1..n_max:
      cols = [(P[0], "glossary", 1), (P[0], "glossary", 2), ..., 
              (P[0], "inline", 3), (P[1], ...), ...]

    Returns (matrix, column_keys, meta)."""
    names, M_raw_canon_centered, sh_canonical, M_canon_centered, Vt_canonical = \
        setup_canonical(data_dir)
    if n_canon_pcs <= 0:
        n_canon_pcs = Vt_canonical.shape[0]
    Vt_used = Vt_canonical[:n_canon_pcs]            # (n_canon_pcs, D)

    column_keys: list[tuple[int, str, int]] = []
    columns: list[np.ndarray] = []                  # each (n_canon_pcs,)
    meta_rows: list[dict] = []

    for pc in pcs:
        for style in DEFAULT_STYLES:
            for n in range(1, n_max + 1):
                winner = n_cell_winner(per_cell, pc, style, n, cell_order)
                if winner is None:
                    column_keys.append((pc, style, n))
                    columns.append(np.zeros(n_canon_pcs, dtype=np.float32))
                    meta_rows.append({
                        "pc": pc, "style": style, "n": n,
                        "rho": None, "slot": None, "layer": None,
                        "L": None, "K": None,
                    })
                    continue
                d_canon_L2 = winner_direction_in_canonical_L2(
                    pc, winner["slot"], winner["layer"],
                    winner["L"], winner["K"],
                    data_dir=data_dir, sh_canonical=sh_canonical,
                    M_raw_canonical_centered=M_raw_canon_centered,
                )
                if d_canon_L2 is None:
                    coeffs = np.zeros(n_canon_pcs, dtype=np.float32)
                else:
                    # Project onto canonical Vt: c_k = d @ Vt[k].
                    coeffs_full = Vt_used @ d_canon_L2          # (n_canon_pcs,)
                    coeffs = (coeffs_full ** 2).astype(np.float32)
                    if normalise_columns and coeffs.sum() > 0:
                        coeffs = coeffs / coeffs.sum()
                column_keys.append((pc, style, n))
                columns.append(coeffs)
                meta_rows.append({
                    "pc": pc, "style": style, "n": n,
                    "rho": float(winner["rho"]),
                    "slot": int(winner["slot"]),
                    "layer": int(winner["layer"]),
                    "L": int(winner["L"]),
                    "K": int(winner["K"]),
                })
                print(f"  pc={pc:>3} {style:<8} n={n}  "
                      f"winner=({winner['slot']},{winner['layer']}) "
                      f"L={winner['L']} K={winner['K']}  "
                      f"ρ={winner['rho']:+.3f}  "
                      f"top-3 canon-PCs: "
                      f"{', '.join(str(int(i)) for i in np.argsort(coeffs)[::-1][:3])}")

    matrix = np.stack(columns, axis=1)              # (n_canon_pcs, n_cols)
    meta = {
        "n_canon_pcs": n_canon_pcs,
        "pcs": list(pcs),
        "cell_order": [list(c) for c in cell_order],
        "n_max": n_max,
        "normalise_columns": normalise_columns,
        "rows": meta_rows,
    }
    return matrix, column_keys, meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    # Default points at the canonical 7-cell fixed_principled per-cell
    # winners produced by ``plot_direction_cosines.py --include_optimization``
    # (May 2026 naming).  The pre-May-2026 ``nth_pc`` 3-cell file was
    # archived; this default reflects the current canonical pipeline.
    p.add_argument(
        "--per_cell_path",
        default=("roger/pc_round_trip_fixed_principled_per_cell_winners_"
                 "7-25_7-49_6-25_6-49_0-26_0-49_3-25.json"),
    )
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output_npz",
                   default="roger/pc_round_trip_nth_pc_winner_decomposition.npz")
    p.add_argument("--pcs", type=int, nargs="*", default=DEFAULT_PCS)
    p.add_argument("--n_canon_pcs", type=int, default=0,
                   help="0 = use all available canonical PCs.")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to handle stale or unrecognized inputs JSON envelopes.")
    args = p.parse_args()

    per_cell_path = Path(args.per_cell_path)
    inputs: list[InputSpec] = []
    per_cell = load_per_cell_winners(
        per_cell_path, policy=args.cache_policy,
        inputs=inputs, dep_key="per_cell_winners_json",
    )
    print(f"Loaded {len(per_cell)} per-cell winners from {args.per_cell_path}")

    matrix, keys, meta = compute_decomposition_matrix(
        per_cell, data_dir=Path(args.data_dir), pcs=args.pcs,
        n_canon_pcs=args.n_canon_pcs,
    )

    # ``np.savez`` doesn't support a JSON envelope, so we record the
    # InputSpec fingerprints inside ``meta`` -- the npz becomes
    # self-describing for ``plot_winner_decomposition.py`` to declare via
    # ``current_file_input(npz)`` while audits validate the upstream
    # per_cell JSON via its own envelope.
    data_dir = Path(args.data_dir)
    # ``per_cell_winners_json`` was already appended to ``inputs`` by
    # load_per_cell_winners → load_and_register at read time.
    inputs.extend([
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
    ])
    meta["_inputs"] = [dataclasses.asdict(s) for s in inputs]

    print(f"\nMatrix shape: {matrix.shape}")
    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        matrix=matrix,
        column_keys=np.array(keys, dtype=object),
        meta=json.dumps(meta),
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
