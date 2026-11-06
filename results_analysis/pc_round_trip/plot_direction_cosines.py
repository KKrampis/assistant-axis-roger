#!/usr/bin/env python3
"""PC round-trip ρ plot: principled α-coefficient cross-cell transport of
the canonical Nth PC, with no token / layer / whitening / cell optimization.

Default single-panel view: Spearman ρ vs PC index.
- dark blue ``×``: canonical cell (slot=7, layer=25, L=0, K=0) ρ.
- pale blue ``○``: same principled transport applied to each of the 5
  non-canonical cells, no shear, no whitening.
- dotted dark blue ``±2/√(2n)``: theoretical chance ρ at H₀.
- pink shaded band: shuffled-judge null at L=K=0 (per-cell, mean of
  gloss + inline scores), ±2 σ.

DEPRECATED legacy view (``--include_optimization``): adds a red ``◆`` line
showing the best fixed_principled ρ found by sweeping (L, K) and aggregating
max-over-cells, plus a matched K/L-search permutation-null band from
``permutation_null.py``.  This was disabled by default in May 2026 because
optimizing ρ against a fixed judge ranking at high N surfaces correlations
that match the description but do not actually align with the original
canonical PC direction; see comments at the optimization-related code blocks
and at the top of ``klm_sweep.py`` for context.

Default output: ``roger/pc_round_trip_direction_cosines_fixed.png``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assistant_axis.plot_metadata import png_metadata, suptitle_with_specs
from assistant_axis.provenance import (
    CACHE_POLICIES,
    InputSpec,
    current_data_subtree_input,
    current_file_input,
    load_and_register,
)
from results_analysis.canonical_angles.data import DEFAULT_DATA_DIR
from results_analysis.canonical_angles.whitening import fit_shear, fit_whitening
from assistant_axis.plot_metadata import json_metadata
from results_analysis.pc_round_trip.klm_sweep import (
    setup_at, compute_M_done, compute_pc_directions, rho_for_direction,
    DEFAULT_L_VALUES, DEFAULT_K_COARSE, DEFAULT_PCS, DEFAULT_STYLES,
    load_combined_scores, DEFAULT_SWEEP_DIR,
    K_REFINE_MAX_ITER, K_REFINE_MIN_BRACKET, K_REFINE_TIE_THRESH,
    _build_subtree_inputs as _klm_build_subtree_inputs,
)


# Canonical reference geometry (post May-2026 migration): 8-slot dataset,
# slot 7, layer 25, raw (L=0, K=0).
SLOT = 7
LAYER = 25
CANONICAL_L = 0
DEFAULT_OUTPUT = "roger/pc_round_trip_direction_cosines_fixed.png"


def woodbury_inverse_shear_apply(shear_basis):
    """Return a callable that applies T_shear^{-1} from the right to a row
    matrix, via Sherman-Morrison-Woodbury (since T_shear = I + E @ diag(f) @ E.T
    is symmetric and the rank-2L update has a tiny inverse).

    For row x: x @ T^{-1} = x - (x @ E) @ M @ E.T,  where
    M = (diag(1/f) + E.T @ E)^{-1}.  E is (D, 2L), M is (2L, 2L).
    """
    if shear_basis.method == "raw":
        # L=0 shear is identity; its inverse is also identity.
        return lambda X: X.copy()
    E = shear_basis.shear_basis            # (D, 2L)
    f = shear_basis.shear_factors          # (2L,)
    F_inv = np.diag(1.0 / f)
    M = np.linalg.inv(F_inv + E.T @ E)     # (2L, 2L)

    def apply(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return apply(X[None, :])[0]
        # X @ T^{-1} = X - (X @ E) @ M @ E.T
        return X - ((X @ E) @ M) @ E.T

    return apply


def woodbury_inverse_whiten_apply(whiten_basis):
    """Return a callable that applies T_whiten^{-1} from the right to a row
    matrix, via Sherman-Morrison-Woodbury.

    Soft-K whitening writes T_whiten = I + V.T @ diag(scales - 1) @ V where
    V is (K, D) and scales is (K,).  So
    T_whiten^{-1} = I + V.T @ diag(1/scales - 1) @ V
    (a closed-form inverse since the update is along the orthonormal V).
    For row x: x @ T_whiten^{-1} = x + (x @ V.T) * (1/scales - 1) @ V.
    """
    if whiten_basis.method == "raw":
        return lambda X: X.copy()
    if whiten_basis.method != "soft_K":
        raise NotImplementedError(
            f"Inverse for {whiten_basis.method!r} not implemented")
    Vt = whiten_basis.Vt                    # (K, D)
    scales = whiten_basis.scales            # (K,)
    inv_scales_minus_1 = 1.0 / scales - 1.0

    def apply(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return apply(X[None, :])[0]
        coefs = X @ Vt.T                                    # (n, K)
        return X + (coefs * inv_scales_minus_1) @ Vt        # (n, D)

    return apply


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--sweep_dir", default=DEFAULT_SWEEP_DIR)
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Output PNG path.  Default: {DEFAULT_OUTPUT}")
    p.add_argument("--pcs", type=int, nargs="*", default=DEFAULT_PCS)
    p.add_argument(
        "--restricted_cells",
        default="7:25,7:49,6:25,6:49,0:26,0:49,3:25",
        help="Comma-separated slot:layer pairs over which to run the "
             "in-script global sweep for fixed_direction and fixed_principled "
             "with coarse K grid + bracket-and-bisect refinement.  "
             "Default = the 7-cell migration set in canonical-first order.  "
             "Subset with e.g. '7:25,7:49' or pass the empty string to skip "
             "the sweep (use cached per-cell JSON).",
    )
    p.add_argument(
        "--exchanged", action="store_true",
        help="Pair-exchange control: pair PCs as 2^i ↔ 0.75·2^i (with "
             "PC 1 ↔ PC 2 as the special low-end pair), then swap each "
             "pair's judge scores between members.  This is 'structured "
             "nonsense' -- a real description with the wrong PC's entity "
             "ordering -- and should produce ρ values that collapse into "
             "the shuffled-judge null bands.  If they don't, something is "
             "wrong (descriptions are too generic, or there's a bug).  "
             "Output filename gets an '_exchanged' suffix when set.",
    )
    cutoff_grp = p.add_mutually_exclusive_group()
    cutoff_grp.add_argument(
        "--cutoff_n_over_2", action="store_true",
        help="Bake an N/2-PC cutoff into principled transport: after "
             "computing the transported direction d = α_N @ M_done at any "
             "cell × (L, K), zero out d's projection onto the FIRST "
             "floor(N/2) raw target-cell PCs.  This generalises the "
             "canonical-cell property 'the Nth PC is orthogonal to PCs "
             "1..N-1' to other cells, removing low-rank target-structure "
             "contamination from the transported direction.  At the "
             "canonical cell with L=K=0, the cutoff is a no-op (the "
             "canonical Nth PC is already orthogonal).  Output filename "
             "gets a '_cutoff' suffix when set.",
    )
    cutoff_grp.add_argument(
        "--cutoff_2n_over_3", action="store_true",
        help="Same as --cutoff_n_over_2 but with a more aggressive "
             "floor(2*N/3) cutoff.  Output filename gets a "
             "'_cutoff_2of3' suffix.",
    )
    p.add_argument(
        "--include_optimization", action="store_true",
        help="DEPRECATED.  Restore the legacy view: in addition to the "
             "default canonical-only plot, draw the red ◆ line showing the "
             "best fixed_principled ρ found by sweeping (L, K) per cell and "
             "aggregating max-over-cells, plus the matched K/L-search "
             "permutation-null band from permutation_null.py.  Disabled by "
             "default since May 2026 because at high N the K/L/cell search "
             "surfaces correlations that match the judge description but "
             "are not actually aligned with the original canonical PC "
             "direction (see klm_sweep.py header for context).  Output "
             "filename gets a '_with_optimization' suffix when set.",
    )
    # Back-compat: --canonical_only used to be the opt-in flag for the
    # default-now behaviour.  Accept it as a no-op alias so older pipelines
    # / scripts don't error out, but prefer dropping it from new callers.
    p.add_argument("--canonical_only", action="store_true",
                   help="Deprecated no-op alias (canonical-only is now the "
                        "default).  Kept for backward compatibility.")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to react to drift in the null-cache JSON's "
                        "recorded provenance: strict / warn (default) / "
                        "rebuild / off.  (Intermediate winners JSONs are "
                        "produced/consumed within a single run and are not "
                        "policy-validated.)")
    args = p.parse_args()
    # Derive a single cutoff configuration used throughout.
    # cutoff_active: bool — whether to apply any cutoff at all
    # cutoff_fraction: float — pc_cutoff(N) = floor(N * cutoff_fraction)
    # cutoff_suffix: str — appended to PNG and cached JSON filenames
    # cutoff_tag_label: optional title-tag string
    if args.cutoff_n_over_2:
        cutoff_fraction = 0.5
        cutoff_suffix = "_cutoff"
        cutoff_tag_label = "N/2 CUTOFF"
    elif args.cutoff_2n_over_3:
        cutoff_fraction = 2.0 / 3.0
        cutoff_suffix = "_cutoff_2of3"
        cutoff_tag_label = "2N/3 CUTOFF"
    else:
        cutoff_fraction = 0.0
        cutoff_suffix = ""
        cutoff_tag_label = None
    cutoff_active = cutoff_fraction > 0
    def pc_cutoff(n: int) -> int:
        return int(n * cutoff_fraction)
    restricted_cells: list[tuple[int, int]] = []
    if args.restricted_cells.strip():
        for tok in args.restricted_cells.split(","):
            s, l = tok.strip().split(":")
            restricted_cells.append((int(s), int(l)))

    data_dir = Path(args.data_dir)
    sweep_dir = Path(args.sweep_dir)
    pcs = list(args.pcs)
    # Provenance accumulator -- threaded through every load_and_register
    # call so the resulting PNG's _provenance.inputs reflects exactly
    # the cache files this run consumed (see AGENT_NOTES.md
    # "Reader+registrar pattern").
    inputs: list[InputSpec] = []
    # ---- Load actual judge scores -----------------------------------------
    print(f"Loading judge scores from {sweep_dir} ...")
    actual_scores: dict[tuple[int, str], dict[str, float]] = {}
    for pc in pcs:
        for style in DEFAULT_STYLES:
            cell_id = f"pc{pc:03d}_{style}"
            cell = sweep_dir / cell_id
            try:
                actual_scores[(pc, style)] = load_combined_scores(
                    cell, inputs=inputs, cell_id=cell_id,
                )
            except FileNotFoundError:
                pass
    print(f"  loaded {len(actual_scores)} cells")

    # ---- Pair-exchange control (--exchanged) -----------------------------
    # Pair PCs as 2^i ↔ 0.75·2^i (with the low-end PC 1 ↔ PC 2 as a special
    # "leftover" pair since PC 1.5 isn't in the data).  Swap each pair's
    # judge scores between members.  Both per-style scores are exchanged
    # per pair: score(pc_a, style) ⇄ score(pc_b, style), per style.
    #
    # The output is "structured nonsense": a real description with the
    # entity ordering from a related-but-different PC.  If the canonical /
    # optimized ρ machinery is correct, exchanged ρ should collapse into
    # the shuffled-judge null band (since the description doesn't match
    # the actual PC's entity ordering at all).  ρ values that stay high
    # in the exchanged setup would suggest description over-genericity or
    # a bug.
    PAIRS = [
        (1, 2), (4, 3), (8, 6), (16, 12), (32, 24),
        (64, 48), (128, 96), (256, 192), (512, 384),
    ]
    # TODO: PCs 20, 28, 36, 40, 56, 80 (added later for finer-grained N
    # coverage) don't fit the 2^i ↔ 0.75·2^i pairing scheme.  Currently
    # they have NO exchange partner, so the --exchanged path leaves them
    # un-shuffled (degenerates to no-op, see "no partner" branch below).
    # Their points on the exchanged plot are therefore identical to the
    # real plot's values for those PCs and should NOT be interpreted as
    # control values.  Decide on a sensible pairing (e.g. nearest-neighbour
    # pairing in log space) and update PAIRS / pair-exchange logic
    # accordingly.  Lower priority now that the optimization-based view
    # is deprecated and --exchanged is mostly a diagnostic for that view.
    if args.exchanged:
        partner_of: dict[int, int] = {}
        for a, b in PAIRS:
            partner_of[a] = b
            partner_of[b] = a
        exchanged: dict[tuple[int, str], dict[str, float]] = {}
        for (pc, style), scores in actual_scores.items():
            partner = partner_of.get(pc)
            if partner is not None and (partner, style) in actual_scores:
                exchanged[(pc, style)] = actual_scores[(partner, style)]
            else:
                # No partner in data -- keep self (degenerates to no-op).
                exchanged[(pc, style)] = scores
        actual_scores = exchanged
        print(f"  --exchanged: applied pair-exchange across "
              f"{len(PAIRS)} pairs ({len(exchanged)} cells)")
        # Auto-rename the output file so the un-exchanged plot isn't
        # overwritten.  Pattern: <stem>_exchanged.png
        out_p = Path(args.output)
        args.output = str(out_p.with_name(out_p.stem + "_exchanged" + out_p.suffix))
    if cutoff_active:
        # Auto-rename the output file with the appropriate cutoff suffix so
        # the no-cutoff plot isn't overwritten.  Combines with --exchanged.
        out_p = Path(args.output)
        args.output = str(out_p.with_name(out_p.stem + cutoff_suffix + out_p.suffix))
        print(f"  cutoff active (fraction={cutoff_fraction:.3f}, "
              f"suffix='{cutoff_suffix}'): writing to {args.output}")
    # canonical-only is the default view post May-2026; the legacy
    # K/L/cell-optimization view is gated behind --include_optimization
    # and gets a distinguishing filename suffix so the default plot
    # isn't overwritten by an opt-in legacy run.
    if args.include_optimization:
        out_p = Path(args.output)
        args.output = str(out_p.with_name(
            out_p.stem + "_with_optimization" + out_p.suffix))
        print(f"  --include_optimization (DEPRECATED): writing to "
              f"{args.output}")
    if args.canonical_only:
        # Deprecated alias, default is canonical-only already.
        print(f"  --canonical_only is now a no-op (default behavior); "
              f"flag kept for backward compatibility.")

    # ---- Setup at (slot=3, layer=25) --------------------------------------
    names, M_raw, A_g, A_n, pool = setup_at(data_dir, SLOT, LAYER)
    print(f"Entity matrix at ({SLOT}, {LAYER}): {M_raw.shape}")

    # ---- Canonical: L=2, K=0 ---------------------------------------------
    sh_canonical = fit_shear(A_g, A_n, L=CANONICAL_L)
    M_canonical = sh_canonical.apply(M_raw)
    # Full SVD so we can also extract α-coefficients (per-entity weights of
    # each canonical PC), needed for the principled cross-cell mapping
    # (fixed_principled variant).
    M_canon_centered = M_canonical - M_canonical.mean(axis=0, keepdims=True)
    U_canon, S_canon, Vt_canonical = np.linalg.svd(M_canon_centered,
                                                     full_matrices=False)
    Vt_canonical = Vt_canonical[:max(pcs)]
    print(f"Canonical Vt at (L={CANONICAL_L}, K=0): {Vt_canonical.shape}")
    inv_sh_canonical_apply = woodbury_inverse_shear_apply(sh_canonical)
    # α[N-1] = U[:, N-1] / S[N-1]: the linear-combination coefficients of
    # canonical-shear-space pool entities that equal the Nth canonical PC.
    # Using these at a different cell's centered M_raw gives the principled
    # cross-cell mapping (preserves inter-entity geometric structure).
    alpha_per_pc: dict[int, np.ndarray] = {}
    for pc_idx in pcs:
        if pc_idx - 1 < S_canon.shape[0] and S_canon[pc_idx - 1] > 0:
            alpha_per_pc[pc_idx] = (U_canon[:, pc_idx - 1]
                                      / S_canon[pc_idx - 1])

    # ====================================================================
    # DEPRECATED (gated behind --include_optimization, default off):
    # everything from here through the cached-winners fallback loaders is
    # the (L, K, cell)-optimization machinery.  We deprecated it in
    # May 2026 because at high N the search reliably surfaces correlations
    # that match the judge description but are not actually aligned with
    # the original canonical PC direction (it picks up structure in the
    # judge ranking shared with sample-noise directions in the activation
    # space, not the PC itself).  The default canonical-only view shows
    # the principled α-coefficient transport at L=K=0 against shuffled-
    # judge nulls, also at L=K=0, which gives an honest read.  The
    # optimization view is retained so we can still rebuild the legacy
    # plots if needed; it is no longer the headline result.
    # ====================================================================
    # ``inputs`` was declared earlier (right before the load_combined_scores
    # loop).  Pre-retrofit this script silently consumed three optional
    # cache files (fixed_direction_restricted_winners,
    # fixed_direction_per_cell_winners, fixed_principled_per_cell_winners)
    # without recording them as deps; the read+register split made it
    # easy to forget the second half.  load_and_register now collapses
    # both into one call at every read site.

    best_global_fixed_restricted: dict[tuple[int, str], dict] = {}
    best_global_principled_restricted: dict[tuple[int, str], dict] = {}
    per_cell_fixed: dict[tuple[int, str, int, int], dict] = {}
    per_cell_principled: dict[tuple[int, str, int, int], dict] = {}
    refinement_eval_count = 0
    Vt_per_LK: dict[tuple[int, int], np.ndarray] = {}
    shear_cache: dict = {}
    whiten_cache: dict = {}
    cached_winners_loaded = False
    null_per_cell: dict[tuple[int, str, int], dict[tuple[int, int], float]] = \
        {}
    null_per_pc: dict[int, list[float]] = {pc: [] for pc in pcs}

    if not args.include_optimization:
        print("\nSkipping (L, K, cell)-optimization sweep "
              "(canonical-only mode is the default; pass "
              "--include_optimization to restore the legacy view).")

    if args.include_optimization:
        # ---- Pre-warm SVD / shear / whitening caches at canonical cell ---
        # DEPRECATED (May 2026): part of the (L, K, cell)-optimization view.
        # Stage 2 (below) reuses these caches, so we walk the coarse + K=N
        # grid at the canonical cell once up front.  ρ is *not* accumulated
        # here -- stage 2 owns all per-(pc, style, variant) ρ bookkeeping so
        # seeding doesn't sneak in values from a different variant.
        print(f"\nPre-warming caches at (slot={SLOT}, layer={LAYER}) ...")

        K_max = max(DEFAULT_K_COARSE)
        # Coarse K grid only — bracket-and-bisect handles the rest.
        K_grid_top = sorted(set(DEFAULT_K_COARSE))
        K_grid_init = sorted(set(DEFAULT_K_COARSE))
        print(f"  pre-warm K grid: {len(K_grid_top)} values (coarse grid)")

        for L in DEFAULT_L_VALUES:
            for K in K_grid_top:
                M_done, _pool_shear = compute_M_done(
                    M_raw, A_g, A_n, pool, L, K, shear_cache, whiten_cache
                )
                Vt = compute_pc_directions(M_done, max(pcs))
                if Vt is None:
                    continue
                Vt_per_LK[(L, K)] = Vt

    # ---- Restricted global sweep: fixed_direction + fixed_principled over
    #      the cells in --restricted_cells.  Mirrors
    #      klm_sweep.stage2_k_refinement: pre-populate the expanded K grid
    #      per (slot, layer, L), then bracket-and-bisect around each
    #      (pc, style, variant) winner.
    # DEPRECATED (May 2026): see the deprecation banner above.  Only runs
    # when --include_optimization is set; otherwise per_cell_fixed and
    # per_cell_principled stay empty and downstream code falls through to
    # the canonical-only blue series.
    if args.include_optimization and restricted_cells:
        # Per-cell caches: setup data, shear/whitening basis caches, and
        # M_done / Vt caches keyed by (L, K).  Shared across both variants
        # and all (pc, style) targets.
        cell_caches: dict[tuple[int, int], dict] = {}

        def get_cell(slot: int, layer: int) -> dict:
            if (slot, layer) not in cell_caches:
                if (slot, layer) == (SLOT, LAYER):
                    # Reuse the (3,25) caches we already built above.
                    cell_caches[(slot, layer)] = {
                        "names": names, "M_raw": M_raw,
                        "A_g": A_g, "A_n": A_n, "pool": pool,
                        "sh_cache": shear_cache,
                        "wh_cache": whiten_cache,
                        "M_done_at_LK": {},
                        "Vt_at_LK": dict(Vt_per_LK),
                    }
                    # Recompute M_done at (3,25) lazily as we need it.
                    # Vt_per_LK is already populated from the (3,25) loop.
                else:
                    n_o, M_o, Ag_o, An_o, pl_o = \
                        setup_at(data_dir, slot, layer)
                    cell_caches[(slot, layer)] = {
                        "names": n_o, "M_raw": M_o,
                        "A_g": Ag_o, "A_n": An_o, "pool": pl_o,
                        "sh_cache": {}, "wh_cache": {},
                        "M_done_at_LK": {}, "Vt_at_LK": {},
                    }
            return cell_caches[(slot, layer)]

        def evaluate_cell(slot: int, layer: int, L: int, K: int):
            """Return (M_done, Vt, names) for (slot, layer, L, K), populating
            cache as needed.  Vt may be None if SVD fails."""
            c = get_cell(slot, layer)
            if (L, K) not in c["M_done_at_LK"]:
                M_done, _ = compute_M_done(
                    c["M_raw"], c["A_g"], c["A_n"], c["pool"],
                    L, K, c["sh_cache"], c["wh_cache"])
                c["M_done_at_LK"][(L, K)] = M_done
                if (L, K) not in c["Vt_at_LK"]:
                    c["Vt_at_LK"][(L, K)] = compute_pc_directions(
                        M_done, max(pcs))
            return (c["M_done_at_LK"][(L, K)],
                    c["Vt_at_LK"].get((L, K)),
                    c["names"])

        # Cache for principled per-cell raw direction (one per (slot, layer,
        # pc) — same across (L, K) since it lives in raw R^D, then we
        # forward-transport).
        principled_raw_cache: dict[tuple[int, int, int], np.ndarray] = {}

        def get_principled_raw(slot: int, layer: int, pc: int):
            """Return d_principled in raw R^D at (slot, layer) for PC index
            pc, or None if α isn't available for that PC."""
            key = (slot, layer, pc)
            if key in principled_raw_cache:
                return principled_raw_cache[key]
            if pc not in alpha_per_pc:
                principled_raw_cache[key] = None
                return None
            c = cell_caches[(slot, layer)]
            M_t = c["M_raw"]
            centered = M_t - M_t.mean(axis=0, keepdims=True)
            d = (alpha_per_pc[pc] @ centered).astype(np.float32)
            principled_raw_cache[key] = d
            return d

        def rho_at(slot: int, layer: int, L: int, K: int,
                   pc: int, style: str, variant: str) -> float | None:
            M_done_, Vt_, names_ = evaluate_cell(slot, layer, L, K)
            if Vt_ is None or pc - 1 >= Vt_.shape[0]:
                return None
            if variant == "fixed_direction":
                # "Same" / naive cross-cell transport: treat the canonical
                # direction as a vector in R^D and reuse it at any cell.
                # Only correct when target cell is geometrically close to
                # canonical (e.g. same slot, adjacent layer).
                if pc - 1 >= Vt_canonical.shape[0]:
                    return None
                c = cell_caches[(slot, layer)]
                d_canon_ = Vt_canonical[pc - 1]
                d_in_raw = inv_sh_canonical_apply(d_canon_[None, :])[0]
                x = d_in_raw[None, :]
                if L > 0:
                    x = c["sh_cache"][L].apply(x)
                if K > 0:
                    x = c["wh_cache"][(L, K)].apply(x)
                d_used = x[0]
            elif variant == "fixed_principled":
                # Principled cross-cell transport: keep α-coefficients
                # (entity weights from canonical PCA) and apply them to the
                # target cell's centered entities.  Preserves inter-entity
                # geometric structure across cell change.
                d_in_raw = get_principled_raw(slot, layer, pc)
                if d_in_raw is None:
                    return None
                c = cell_caches[(slot, layer)]
                x = d_in_raw[None, :]
                if L > 0:
                    x = c["sh_cache"][L].apply(x)
                if K > 0:
                    x = c["wh_cache"][(L, K)].apply(x)
                d_used = x[0]
            else:
                raise ValueError(f"unknown variant {variant!r}")
            # ---- Cutoff (when --cutoff_n_over_2 or --cutoff_2n_over_3):
            #      zero out the transported direction's projection onto
            #      the first pc_cutoff(N) RAW target-cell PCs (Vt at
            #      L=0, K=0).
            if cutoff_active:
                cutoff = pc_cutoff(pc)
                if cutoff > 0:
                    c_cut = cell_caches[(slot, layer)]
                    Vt_raw = c_cut["Vt_at_LK"].get((0, 0))
                    if Vt_raw is None:
                        # Lazy-populate raw Vt for this cell.
                        evaluate_cell(slot, layer, 0, 0)
                        Vt_raw = c_cut["Vt_at_LK"][(0, 0)]
                    n_cut = min(cutoff, Vt_raw.shape[0])
                    if n_cut > 0:
                        coefs = Vt_raw[:n_cut] @ d_used
                        d_used = d_used - Vt_raw[:n_cut].T @ coefs
            return rho_for_direction(M_done_, names_, d_used,
                                       actual_scores[(pc, style)])

        def update_best(target_dict: dict, key: tuple[int, str], rho: float,
                        L: int, K: int, slot: int, layer: int) -> None:
            cur = target_dict.get(key)
            if cur is None or rho > cur["rho"]:
                target_dict[key] = {
                    "rho": rho, "L": L, "K": K, "slot": slot, "layer": layer,
                }

        def update_best_per_cell(target_dict: dict,
                                   key: tuple[int, str, int, int],
                                   rho: float, L: int, K: int) -> None:
            cur = target_dict.get(key)
            if cur is None or rho > cur["rho"]:
                target_dict[key] = {"rho": rho, "L": L, "K": K}

        # ---- Stage 2 setup: register all cells so the SVD/shear/whiten
        # caches exist before refinement starts.  The (3,25) cell reuses
        # the upper-sweep caches (Vt_per_LK, shear_cache, whiten_cache);
        # other cells get freshly loaded matrices.  No "seeding" of the
        # accumulators is done -- stage 2 evaluates ρ for every cell ×
        # variant from scratch via populate_K, so seeding from a different
        # variant (which is only equal at canonical L=2, K=0) would risk
        # locking in a value that update_best_per_cell can't lower.
        if (SLOT, LAYER) in restricted_cells:
            get_cell(SLOT, LAYER)
        # Register non-(3,25) cells so stage 2 has them in cell_caches.  The
        # cell's setup data is loaded lazily on first access.
        for slot, layer in restricted_cells:
            if (slot, layer) == (SLOT, LAYER):
                continue
            get_cell(slot, layer)

        # ---- Stage 2: pre-populate expanded grid + K refinement ------------
        # For each (slot, layer, L), bracket-and-bisect K within
        # [0, max(DEFAULT_K_COARSE)] for each (pc, style, variant) target.
        # Refinement is per-(slot, layer, L); both variants benefit when one
        # adds a new K to the cache.
        K_MAX = max(DEFAULT_K_COARSE)
        # Count Vt evaluations (which also covers (3,25) cache pre-populated
        # in the upper loop).  Stage-2 newly-evaluated cells = after - before.
        n_LK_before = sum(len(c["Vt_at_LK"])
                           for c in cell_caches.values())

        for slot, layer in restricted_cells:
            print(f"  Stage 2: refining K at (slot={slot}, layer={layer}) "
                  f"(expanded K grid + bracket-bisect)...")
            for L in DEFAULT_L_VALUES:
                # Memory hygiene: evict (L', *) entries for L' != L from the
                # caches.  Within one L iteration, only this L's caches are
                # active; L's are processed in order so we never revisit.
                c = cell_caches[(slot, layer)]
                for cache in (c["Vt_at_LK"], c["M_done_at_LK"],
                              c["wh_cache"]):
                    for k in [k for k in cache if k[0] != L]:
                        del cache[k]
                for k in [k for k in c["sh_cache"] if k != L]:
                    del c["sh_cache"][k]

                # Pre-populate this L's K grid with the expanded grid (coarse
                # + K-near-N dense scan).  This guarantees K=N and a window
                # of width ceil(N/8) around it is evaluated for every PC,
                # before bracket-and-bisect is asked to refine.  Also pre-
                # compute ρ for every (variant, pc, style, K) so the
                # bracket-bisect loop is just dict lookups.
                rho_grid: dict[tuple[str, int, str, int], float] = {}

                def populate_K(K: int) -> None:
                    """Evaluate (slot, layer, L, K) and cache ρ for all
                    variants × all (pc, style) targets."""
                    evaluate_cell(slot, layer, L, K)
                    for pc in pcs:
                        for style in DEFAULT_STYLES:
                            if (pc, style) not in actual_scores:
                                continue
                            for variant in ("fixed_direction", "fixed_principled"):
                                if (variant, pc, style, K) in rho_grid:
                                    continue
                                r = rho_at(slot, layer, L, K, pc, style,
                                           variant)
                                if r is not None:
                                    rho_grid[(variant, pc, style, K)] = r

                for K in K_grid_init:
                    populate_K(K)

                # Set of K's evaluated at this (slot, layer, L) — grows during
                # refinement.  We key on Vt_at_LK because the (3,25) cell's
                # cache is pre-populated there from the upper sweep loop;
                # M_done_at_LK starts empty there and gets backfilled lazily
                # by evaluate_cell.
                def K_set() -> list[int]:
                    return sorted(K for (L_, K)
                                  in cell_caches[(slot, layer)][
                                      "Vt_at_LK"]
                                  if L_ == L)
                # Process every (pc, style, variant) target.
                for pc in pcs:
                    for style in DEFAULT_STYLES:
                        if (pc, style) not in actual_scores:
                            continue
                        for variant, store, pc_store in (
                            ("fixed_direction",
                             best_global_fixed_restricted, per_cell_fixed),
                            ("fixed_principled",
                             best_global_principled_restricted,
                             per_cell_principled),
                        ):
                            def best_K_in_grid() -> tuple[int | None, float]:
                                best_rho = -2.0
                                best_K: int | None = None
                                for K in K_set():
                                    r = rho_grid.get((variant, pc, style, K))
                                    if r is not None and r > best_rho:
                                        best_rho = r
                                        best_K = K
                                return best_K, best_rho

                            for _ in range(K_REFINE_MAX_ITER):
                                K_evaluated = K_set()
                                K_star, rho_star = best_K_in_grid()
                                if K_star is None:
                                    break
                                idx = K_evaluated.index(K_star)
                                K_left = K_evaluated[max(0, idx - 1)]
                                K_right = K_evaluated[
                                    min(len(K_evaluated) - 1, idx + 1)]
                                if K_right - K_left <= K_REFINE_MIN_BRACKET:
                                    break
                                added = False
                                for frac in (1/3, 2/3):
                                    new_K = int(round(K_left + frac
                                                       * (K_right - K_left)))
                                    if (new_K not in K_evaluated
                                            and 0 <= new_K <= K_MAX):
                                        populate_K(new_K)
                                        added = True
                                if not added:
                                    break
                                new_K_star, new_rho_star = best_K_in_grid()
                                if (new_rho_star
                                        <= rho_star + K_REFINE_TIE_THRESH
                                        and new_K_star == K_star):
                                    break
                            K_star, rho_star = best_K_in_grid()
                            if K_star is not None:
                                update_best(store, (pc, style), rho_star,
                                             L, K_star, slot, layer)
                                update_best_per_cell(
                                    pc_store,
                                    (pc, style, slot, layer),
                                    rho_star, L, K_star)

        n_LK_after = sum(len(c["Vt_at_LK"])
                          for c in cell_caches.values())
        refinement_eval_count = n_LK_after - n_LK_before
        print(f"  restricted sweep done over {len(restricted_cells)} cells; "
              f"stage 2 added {refinement_eval_count} new (L, K) SVDs; "
              f"fixed_principled global winners: "
              f"{len(best_global_principled_restricted)}")

        # Persist all variant winners JSONs for audit/replay (global maxes).
        cells_tag = "_".join(f"{s}-{l}" for s, l in restricted_cells)
        # Add suffixes to cached winner JSONs when running with the
        # --exchanged control or any cutoff flag, so we don't clobber
        # the canonical winners.  Combine them when both flags are set.
        if args.exchanged:
            cells_tag = cells_tag + "_exchanged"
        if cutoff_active:
            cells_tag = cells_tag + cutoff_suffix
        out_dir = Path(args.output).parent
        # Build the side-JSON inputs list.  Should match the PNG's
        # dependency set: the per-(pc, style, judge, mode) judge-cache
        # fingerprints already accumulated in ``inputs`` by the
        # ``load_combined_scores`` calls above, plus the up-front data
        # subtrees from the klm_sweep helper.  The null cache is NOT an
        # input here (these JSONs are *upstream* of the null overlay
        # path, not downstream).
        side_json_inputs = list(inputs) + _klm_build_subtree_inputs(
            data_dir=data_dir,
        )
        for tag, payload in [
            ("fixed_direction", best_global_fixed_restricted),
            ("fixed_principled", best_global_principled_restricted),
        ]:
            winners_path = out_dir / (
                f"pc_round_trip_{tag}_restricted_winners_{cells_tag}.json")
            winners_payload = {
                f"pc{pc:03d}_{style}": {
                    "rho": float(v["rho"]),
                    "L": int(v["L"]),
                    "K": int(v["K"]),
                    "slot": int(v["slot"]),
                    "layer": int(v["layer"]),
                }
                for (pc, style), v in payload.items()
            }
            envelope = json_metadata(
                winners_payload, inputs=side_json_inputs,
                title=f"pc_round_trip_{tag}_restricted_winners",
            )
            winners_path.write_text(json.dumps(envelope, indent=2, sort_keys=True))
            print(f"  wrote {winners_path}")

        # Persist per-cell winners (per (pc, style, slot, layer) — used to
        # derive 1-cell / 2-cell / 3-cell aggregates post-hoc).
        for tag, pc_payload in [
            ("fixed_direction", per_cell_fixed),
            ("fixed_principled", per_cell_principled),
        ]:
            per_cell_path = out_dir / (
                f"pc_round_trip_{tag}_per_cell_winners_{cells_tag}.json")
            per_cell_payload = {
                f"pc{pc:03d}_{style}_s{slot}_l{layer}": {
                    "rho": float(v["rho"]),
                    "L": int(v["L"]),
                    "K": int(v["K"]),
                }
                for (pc, style, slot, layer), v in pc_payload.items()
            }
            envelope = json_metadata(
                per_cell_payload, inputs=side_json_inputs,
                title=f"pc_round_trip_{tag}_per_cell_winners",
            )
            per_cell_path.write_text(json.dumps(envelope, indent=2, sort_keys=True))
            print(f"  wrote {per_cell_path}")

    # ---- Fallback: load cached-winner JSONs from a previous run --------
    # DEPRECATED (May 2026): only the optimization view consumes these,
    # so they're loaded conditionally.  When running with --exchanged,
    # prefer files whose name contains "_exchanged"; when running without,
    # exclude such files so we don't accidentally load exchanged-control
    # data into the canonical plot.
    def _filter_for_exchange(paths):
        # Filter cached winner paths to match the active variant flags so
        # we don't pick up an exchanged or cutoff cache when running
        # without those flags (or vice versa).
        out = list(paths)
        if args.exchanged:
            out = [p for p in out if "_exchanged" in p.name]
        else:
            out = [p for p in out if "_exchanged" not in p.name]
        if cutoff_active:
            out = [p for p in out if cutoff_suffix in p.name]
        else:
            out = [p for p in out if "_cutoff" not in p.name]
        return out
    if args.include_optimization and not best_global_fixed_restricted:
        out_dir_cache = Path(args.output).parent
        candidates_f = sorted(
            _filter_for_exchange(out_dir_cache.glob(
                "pc_round_trip_fixed_direction_restricted_winners_*.json")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates_f and not best_global_fixed_restricted:
            cached, _spec, _check = load_and_register(
                candidates_f[0],
                dep_key="fixed_direction_restricted_winners",
                inputs=inputs, policy=args.cache_policy,
            )
            for k, v in cached.items():
                pc = int(k[2:5])
                style = k.split("_", 2)[1]
                best_global_fixed_restricted[(pc, style)] = v
            print(f"  loaded cached fixed_direction winners from "
                  f"{candidates_f[0].name} ({len(cached)} entries)")
            cached_winners_loaded = True

    # Per-cell fallback loader (fixed_principled ρ lines × n-cell aggregates).
    # DEPRECATED (May 2026): only used by the optimization view.
    if args.include_optimization and not per_cell_fixed:
        candidates_pc = sorted(
            _filter_for_exchange(Path(args.output).parent.glob(
                "pc_round_trip_fixed_direction_per_cell_winners_*.json")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates_pc:
            cached, _spec, _check = load_and_register(
                candidates_pc[0],
                dep_key="fixed_direction_per_cell_winners",
                inputs=inputs, policy=args.cache_policy,
            )
            for k, v in cached.items():
                pc = int(k[2:5])
                rest = k[6:]
                style_part, sl = rest.rsplit("_s", 1)
                slot_str, layer_str = sl.split("_l", 1)
                per_cell_fixed[(pc, style_part, int(slot_str),
                                int(layer_str))] = v
            print(f"  loaded cached fixed_direction per-cell from "
                  f"{candidates_pc[0].name} ({len(cached)} entries)")

    # DEPRECATED (May 2026): only used by the optimization view.
    if args.include_optimization and not per_cell_principled:
        candidates_pc = sorted(
            _filter_for_exchange(Path(args.output).parent.glob(
                "pc_round_trip_fixed_principled_per_cell_winners_*.json")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates_pc:
            cached, _spec, _check = load_and_register(
                candidates_pc[0],
                dep_key="fixed_principled_per_cell_winners",
                inputs=inputs, policy=args.cache_policy,
            )
            for k, v in cached.items():
                pc = int(k[2:5])
                rest = k[6:]
                style_part, sl = rest.rsplit("_s", 1)
                slot_str, layer_str = sl.split("_l", 1)
                per_cell_principled[(pc, style_part, int(slot_str),
                                       int(layer_str))] = v
            print(f"  loaded cached fixed_principled per-cell from "
                  f"{candidates_pc[0].name} ({len(cached)} entries)")
    # Update cells_label downstream from the cached winners' content if we
    # loaded from cache (otherwise restricted_cells already drives it).
    if cached_winners_loaded and not restricted_cells:
        loaded_cells = sorted({
            (v["slot"], v["layer"])
            for v in {**best_global_fixed_restricted,
                      **best_global_principled_restricted}.values()
        })
        # Stash on a module-level container so the plot section can pick up.
        # (Keeping restricted_cells empty keeps the in-script sweep skipped.)
        # Set restricted_cells to the loaded set so the plot label is
        # accurate; the in-script sweep has already run/been skipped.
        restricted_cells = loaded_cells

    # ---- Null cache (K/L-search floor from permutation_null.py) ----------
    # DEPRECATED (May 2026): this is the matched-search bias band that
    # accompanies the optimization view's red ◆ markers.  Only loaded when
    # --include_optimization is set; the canonical-only default plot uses
    # the inline shuffled-judge null at L=K=0 computed below instead.
    if args.include_optimization:
        if cutoff_active:
            null_cache_path = Path(
                f"roger/pc_round_trip_null_klm_results{cutoff_suffix}.json")
        else:
            null_cache_path = Path(
                "roger/pc_round_trip_null_klm_results.json")
        try:
            null_full, _spec, _check = load_and_register(
                null_cache_path,
                dep_key="null_cache_json",
                inputs=inputs, policy=args.cache_policy,
            )
        except FileNotFoundError:
            print(f"  null cache not found at {null_cache_path}; "
                  "plotting without noise bands")
            null_full = {}
        null_data = null_full.get("stage2", {})
        null_per_cell_raw = null_full.get("stage2_per_cell", {})

        for k, v in null_data.items():
            pc = int(k[2:5])
            if pc in null_per_pc and v.get("rho") is not None:
                null_per_pc[pc].append(v["rho"])

        for k, v in null_per_cell_raw.items():
            try:
                pc = int(k[2:5])
                rest = k[6:]
                style_perm, sl = rest.rsplit("_s", 1)
                slot_str, layer_str = sl.split("_l", 1)
                style, p_str = style_perm.rsplit("_p", 1)
                target = (pc, style, int(p_str))
                cell = (int(slot_str), int(layer_str))
                null_per_cell.setdefault(target, {})[cell] = v["rho"]
            except Exception:
                continue

    # Canonical (no optimisation): ρ at (slot=7, layer=25, L=0, K=0).
    # When --cutoff_n_over_2 is set, also project out the first N/2 raw
    # canonical PCs from d.  This is mathematically a no-op at the
    # canonical cell because Vt_canonical[N-1] is by construction
    # orthogonal to Vt_canonical[:N-1] ⊇ Vt_canonical[:N/2], but we apply
    # it for symmetry with the other-cells loop.
    rho_canonical_per_pc: dict[int, list[float]] = {pc: [] for pc in pcs}
    for pc in pcs:
        if pc - 1 >= Vt_canonical.shape[0]:
            continue
        d = Vt_canonical[pc - 1]
        if cutoff_active:
            cutoff = pc_cutoff(pc)
            if cutoff > 0:
                n_cut = min(cutoff, Vt_canonical.shape[0])
                coefs = Vt_canonical[:n_cut] @ d
                d = d - Vt_canonical[:n_cut].T @ coefs
        for style in DEFAULT_STYLES:
            if (pc, style) not in actual_scores:
                continue
            rho = rho_for_direction(M_canonical, names, d,
                                      actual_scores[(pc, style)])
            if rho is not None:
                rho_canonical_per_pc[pc].append(rho)

    # Same-formula canonical (no optimisation) ρ at the OTHER 5 cells:
    # ``fixed_principled`` transport at L=K=0 (canonical α-coefficients
    # applied to the target cell's centred raw entity matrix, no shear,
    # no whitening).  At the canonical cell this is identical to the
    # ``M_canonical @ Vt[pc-1]`` projection above (sanity check holds);
    # at the other 5 cells it shows whether the canonical PC direction
    # transports geometrically without any optimisation help.
    SIX_CELLS_CANONICAL = [
        (7, 25), (7, 49), (6, 25), (6, 49), (0, 26), (0, 49),
    ]
    # Per-cell, per-PC, list of per-style ρ values (mean across styles
    # used for plotting).  At the canonical cell this overlaps with
    # rho_canonical_per_pc above (kept separate for clarity).
    rho_canonical_per_cell_per_pc: dict[
        tuple[int, int], dict[int, list[float]]
    ] = {cell: {pc: [] for pc in pcs} for cell in SIX_CELLS_CANONICAL}
    # Cache target-cell M_centered + names + raw Vt so we don't repeatedly
    # rebuild them in the per-PC loop.  Vt_raw is needed for the optional
    # cutoff (--cutoff_n_over_2 / --cutoff_2n_over_3): we project the
    # transported direction onto Vt_raw[:pc_cutoff(N)] and subtract.
    _cell_geom_l0_k0: dict[
        tuple[int, int], tuple[list[str], np.ndarray, np.ndarray | None]
    ] = {}
    for slot, layer in SIX_CELLS_CANONICAL:
        if (slot, layer) == (SLOT, LAYER):
            M_t_centered = M_canonical - M_canonical.mean(axis=0, keepdims=True)
            n_t = names
        else:
            n_t, M_t, _, _, _ = setup_at(data_dir, slot, layer)
            M_t_centered = M_t - M_t.mean(axis=0, keepdims=True)
        # Compute raw Vt only when cutoff is enabled (avoids unnecessary
        # SVD when not needed; the K/L sweep block computes its own copy).
        Vt_raw_t = None
        if cutoff_active:
            _U, _S, Vt_raw_t = np.linalg.svd(M_t_centered, full_matrices=False)
        _cell_geom_l0_k0[(slot, layer)] = (n_t, M_t_centered, Vt_raw_t)
    for pc in pcs:
        if pc not in alpha_per_pc:
            continue
        a = alpha_per_pc[pc]
        cutoff = pc_cutoff(pc) if cutoff_active else 0
        for slot, layer in SIX_CELLS_CANONICAL:
            names_t, M_centered_t, Vt_raw_t = _cell_geom_l0_k0[(slot, layer)]
            d_t = a @ M_centered_t   # canonical α applied to target cell
            if cutoff > 0 and Vt_raw_t is not None:
                n_cut = min(cutoff, Vt_raw_t.shape[0])
                if n_cut > 0:
                    coefs = Vt_raw_t[:n_cut] @ d_t
                    d_t = d_t - Vt_raw_t[:n_cut].T @ coefs
            proj = M_centered_t @ d_t
            n2i = {n: i for i, n in enumerate(names_t)}
            for style in DEFAULT_STYLES:
                sc = actual_scores.get((pc, style))
                if sc is None:
                    continue
                common = [n for n in names_t if n in sc]
                if len(common) < 3:
                    continue
                x = np.array([sc[n] for n in common])
                y = proj[[n2i[n] for n in common]]
                from scipy.stats import spearmanr as _sp
                r = _sp(x, y).correlation
                if not np.isnan(r):
                    rho_canonical_per_cell_per_pc[(slot, layer)][pc].append(
                        float(r))

    # ---- Inline shuffled-judge null at L=K=0 (with optional cutoff) ------
    # Like the K/L-search null bands but with NO optimization: the
    # transported direction at each cell is fixed at L=K=0 (+ cutoff),
    # and we permute judge scores instead.  Per (PC, style, perm), max ρ
    # over the 6 cells; aggregate to mean ± 2σ across (style × perm).
    # This null accounts for the actual judge-score structure (heavy ties
    # at high N, default-score concentration) -- which the theoretical
    # ±2/√(2n) chance line does not.
    NULL_PERMS = 100
    NULL_SEED = 42
    no_opt_null_per_pc: dict[int, list[float]] = {pc: [] for pc in pcs}
    rng = np.random.default_rng(NULL_SEED)
    # Pre-compute per-cell projections (one per PC, fixed across all perms).
    no_opt_projs_per_pc: dict[int, dict[tuple[int, int], np.ndarray]] = {}
    no_opt_names_per_cell: dict[tuple[int, int], list[str]] = {}
    for slot, layer in SIX_CELLS_CANONICAL:
        no_opt_names_per_cell[(slot, layer)] = (
            _cell_geom_l0_k0[(slot, layer)][0])
    for pc in pcs:
        if pc not in alpha_per_pc:
            continue
        a = alpha_per_pc[pc]
        cutoff = pc_cutoff(pc) if cutoff_active else 0
        cell_projs: dict[tuple[int, int], np.ndarray] = {}
        for slot, layer in SIX_CELLS_CANONICAL:
            _, M_centered_t, Vt_raw_t = _cell_geom_l0_k0[(slot, layer)]
            d_t = a @ M_centered_t
            if cutoff > 0 and Vt_raw_t is not None:
                n_cut = min(cutoff, Vt_raw_t.shape[0])
                if n_cut > 0:
                    coefs = Vt_raw_t[:n_cut] @ d_t
                    d_t = d_t - Vt_raw_t[:n_cut].T @ coefs
            cell_projs[(slot, layer)] = M_centered_t @ d_t
        no_opt_projs_per_pc[pc] = cell_projs
    # Run permutations.
    # ``Same shuffle applied to both gloss and inline'' (per perm).  Drawing
    # ONE entity permutation π per perm and re-indexing both styles' score
    # arrays by π preserves the natural gloss×inline correlation structure
    # on the actual data: both styles see the same shuffled entity-ordering.
    # The mean-over-styles ρ then has the SAME variance as single-style ρ
    # (no √2 reduction), since the two style ρ's are nearly perfectly
    # correlated in this regime.
    from scipy.stats import spearmanr as _sp_null
    for pc in pcs:
        cell_projs = no_opt_projs_per_pc.get(pc)
        if cell_projs is None:
            continue
        # Collect entity sets that have BOTH (a) a score in every style and
        # (b) presence in the canonical cell's name list.  Use this common
        # ordering for everything (style arrays + per-cell projection y).
        canonical_names = no_opt_names_per_cell[(SLOT, LAYER)]
        score_sets = []
        styles_present: list[str] = []
        for style in DEFAULT_STYLES:
            sc = actual_scores.get((pc, style))
            if sc is None:
                continue
            score_sets.append(set(sc.keys()))
            styles_present.append(style)
        if not score_sets:
            continue
        common_scored = set.intersection(*score_sets)
        common_keys = [n for n in canonical_names if n in common_scored]
        if len(common_keys) < 3:
            continue
        per_style_arrays: dict[str, np.ndarray] = {}
        for style in styles_present:
            sc = actual_scores[(pc, style)]
            per_style_arrays[style] = np.asarray(
                [sc[k] for k in common_keys], dtype=float)
        cell_y: dict[tuple[int, int], np.ndarray] = {}
        for slot, layer in SIX_CELLS_CANONICAL:
            names_t = no_opt_names_per_cell[(slot, layer)]
            n2i = {n: i for i, n in enumerate(names_t)}
            if all(k in n2i for k in common_keys):
                idx = [n2i[k] for k in common_keys]
                cell_y[(slot, layer)] = cell_projs[(slot, layer)][idx]
        if not cell_y or not per_style_arrays:
            continue
        n_common = len(common_keys)
        for perm_idx in range(NULL_PERMS):
            perm = np.arange(n_common)
            rng.shuffle(perm)
            shuffled_per_style = {
                style: arr[perm] for style, arr in per_style_arrays.items()
            }
            for slot, layer in cell_y:
                y_proj = cell_y[(slot, layer)]
                style_rhos: list[float] = []
                for style, arr in shuffled_per_style.items():
                    r = _sp_null(arr, y_proj).correlation
                    if not np.isnan(r):
                        style_rhos.append(float(r))
                if style_rhos:
                    no_opt_null_per_pc[pc].append(float(np.mean(style_rhos)))

    pcs_plot = sorted(
        pc for pc in pcs
        if rho_canonical_per_pc[pc]
        or any(t[0] == pc for t in per_cell_principled)
    )
    if not pcs_plot:
        pcs_plot = sorted(
            pc for pc in pcs
            if any((pc, s) in actual_scores for s in DEFAULT_STYLES)
        )
    pcs_with_data = pcs_plot

    rho_canonical_means = [
        float(np.mean(rho_canonical_per_pc[pc])) if rho_canonical_per_pc[pc]
        else float("nan")
        for pc in pcs_with_data
    ]
    rho_canonical_min = [
        float(np.min(rho_canonical_per_pc[pc])) if rho_canonical_per_pc[pc]
        else float("nan") for pc in pcs_with_data
    ]
    rho_canonical_max = [
        float(np.max(rho_canonical_per_pc[pc])) if rho_canonical_per_pc[pc]
        else float("nan") for pc in pcs_with_data
    ]

    # Per-PC chance-level CI for the canonical-baseline ρ (drawn as dotted
    # lines around y=0).  We plot the MEAN of per-style canonical ρ values
    # at each PC, so the appropriate null SE is the SE of that mean, not
    # the per-style SE.  Concretely:
    #
    #   - Per-style Spearman ρ has SE ≈ 1/√(n−1) under H₀ (no signal),
    #     where n = entities scored by that style's judges.
    #   - Averaging across S styles reduces the SE by 1/√S, so the mean's
    #     SE is roughly 1/√(S·n) = 1/√N_total where N_total = Σ_styles n_s.
    #   - 95%-ish chance-level CI: ±2/√N_total.
    #
    # In the typical 2-style case this is ±2/√(2n) (the user's "sqrt(2n)"
    # convention).  When only one style has data (e.g. during partial
    # judging), N_total = n and the band widens automatically.
    name_set = set(names)
    n_total_canonical_per_pc: list[int] = []
    for pc in pcs_with_data:
        ns_pc: list[int] = []
        for style in DEFAULT_STYLES:
            sc = actual_scores.get((pc, style))
            if sc:
                ns_pc.append(len(name_set & set(sc.keys())))
        n_total_canonical_per_pc.append(int(sum(ns_pc)) if ns_pc else 0)
    ctrl_canonical_pos = [
        +2.0 / np.sqrt(N) if N >= 4 else float("nan")
        for N in n_total_canonical_per_pc
    ]
    ctrl_canonical_neg = [-x if not np.isnan(x) else float("nan")
                          for x in ctrl_canonical_pos]

    fig, ax_rho = plt.subplots(1, 1, figsize=(11, 7.5))

    # ---- No-opt shuffled-judge null band ---------------------------------
    # Per-PC mean ± 2σ across (style × perm), where each value is the
    # max-over-6-cells ρ at L=K=0 (+ cutoff) on shuffled judge scores.
    # Drawn first so it sits behind everything.
    NULL_BAND_COLOR = "#cc7777"
    no_opt_null_means = [
        float(np.mean(no_opt_null_per_pc[pc])) if no_opt_null_per_pc[pc]
        else float("nan") for pc in pcs_with_data
    ]
    no_opt_null_sds = [
        float(np.std(no_opt_null_per_pc[pc], ddof=1))
        if len(no_opt_null_per_pc[pc]) > 1 else 0.0
        for pc in pcs_with_data
    ]
    no_opt_null_lo = [m - 2 * s for m, s in zip(no_opt_null_means,
                                                  no_opt_null_sds)]
    no_opt_null_hi = [m + 2 * s for m, s in zip(no_opt_null_means,
                                                  no_opt_null_sds)]
    ax_rho.fill_between(pcs_with_data, no_opt_null_lo, no_opt_null_hi,
                        color=NULL_BAND_COLOR, alpha=0.18, linewidth=0,
                        label="shuffled-judge null at L=K=0 (per-cell, "
                              "mean of gloss+inline) ± 2 σ")
    ax_rho.plot(pcs_with_data, no_opt_null_means,
                linestyle="--", linewidth=1.0,
                color=NULL_BAND_COLOR, alpha=0.7)

    # ---- Canonical PC ρ via principled α-coefficient transport ----------
    # Plotted before the control lines so its legend entry comes first.
    # ``zorder`` keeps the × markers on top of any later-drawn band fills.
    # NB: at every cell this is the L=K=0 principled transport; no
    # (L, K, cell) optimization is applied -- the direction is rigidly
    # the canonical Nth PC, transported via canonical α-coefficients.
    CANONICAL_COLOR = "#1f4e79"           # dark blue (canonical cell)
    OTHER_CELL_COLOR = "#a8c3dc"           # paler blue (other 5 cells)
    # First: 5 paler-blue lines for the non-canonical cells (L=K=0
    # principled transport).  Plotted BEFORE the canonical-cell line
    # so the dark blue × markers sit on top.
    other_cells = [c for c in SIX_CELLS_CANONICAL if c != (SLOT, LAYER)]
    other_label_used = False
    for slot, layer in other_cells:
        per_pc = rho_canonical_per_cell_per_pc[(slot, layer)]
        means = [
            float(np.mean(per_pc[pc])) if per_pc[pc] else float("nan")
            for pc in pcs_with_data
        ]
        ax_rho.plot(
            pcs_with_data, means, marker="o", markersize=3.0,
            linewidth=0.9, color=OTHER_CELL_COLOR, alpha=0.85, zorder=4,
            label=("canonical PC at non-canonical cells (5 cells)"
                   if not other_label_used else None),
        )
        other_label_used = True

    ax_rho.plot(pcs_with_data, rho_canonical_means, marker="x",
                markersize=10, linewidth=1.4, color=CANONICAL_COLOR,
                markeredgewidth=1.6, alpha=0.85, zorder=10,
                label="canonical PC at canonical cell")
    ax_rho.fill_between(pcs_with_data, rho_canonical_min, rho_canonical_max,
                        color=CANONICAL_COLOR, alpha=0.10, linewidth=0)
    for pc, val in zip(pcs_with_data, rho_canonical_means):
        if not np.isnan(val):
            ax_rho.text(pc, val + 0.025, f"{val:+.2f}", ha="center",
                        va="bottom", fontsize=7.5, color=CANONICAL_COLOR)

    # ---- Canonical chance-level (±2/√(2n)) reference --------------------
    # The dotted band represents the SE of the *mean* canonical ρ across
    # styles, which is what the × markers plot.  See the n_total_canonical_per_pc
    # construction above for the exact formula.
    ax_rho.plot(pcs_with_data, ctrl_canonical_pos,
                linestyle=":", color=CANONICAL_COLOR, linewidth=1.0, alpha=0.7,
                label="canonical control: ±2/√(2n) (95% chance ρ for null)")
    ax_rho.plot(pcs_with_data, ctrl_canonical_neg,
                linestyle=":", color=CANONICAL_COLOR, linewidth=1.0, alpha=0.7)

    # ---- Fixed_principled optima × n-cell aggregates + matched null -----
    # Cell order (post May-2026 migration): canonical (7, 25), then (7, 49)
    # to fill out slot 7, then slot 6 both layers, slot 0 both layers, and
    # finally the old canonical (3, 25) at the tail as a diagnostic point.
    # n-cell aggregates plot max ρ over the first N cells for N in
    # {1, 2, 4, 6, 7} — whole-slot increments only.
    cell_order = [(SLOT, LAYER), (7, 49),
                  (6, 25), (6, 49),
                  (0, 26), (0, 49),
                  (3, 25)]
    cells_available = sorted({
        (s, l)
        for (_, _, s, l) in (
            list(per_cell_fixed) + list(per_cell_principled))
    })
    n_available = sum(1 for c in cell_order if c in cells_available)
    # Only plot the 6-cell aggregate (the one we argue from in the writeup);
    # 1/2/4 added clutter without changing the read, and 7 is essentially
    # 6 + a single cell so adds no qualitative info.
    n_cell_levels = [n for n in (6,) if n <= n_available]

    # Single-level palette: red for the 6-cell aggregate.
    CELL_COUNT_COLORS = {
        6: "#c44e52",  # red
    }

    def n_cell_aggregate(per_cell_dict: dict, n: int) -> dict[
        tuple[int, str], float]:
        """Per (pc, style), max ρ over the first n cells of cell_order."""
        cells = set(cell_order[:n])
        out: dict[tuple[int, str], float] = {}
        for (pc, style, slot, layer), v in per_cell_dict.items():
            if (slot, layer) in cells:
                cur = out.get((pc, style), float("-inf"))
                if v["rho"] > cur:
                    out[(pc, style)] = v["rho"]
        return out

    def per_pc_stats(agg: dict[tuple[int, str], float]
                     ) -> tuple[list[float], list[float], list[float]]:
        per_pc: dict[int, list[float]] = {pc: [] for pc in pcs_with_data}
        for (pc, style), rho in agg.items():
            if pc in per_pc:
                per_pc[pc].append(rho)
        means = [float(np.mean(per_pc[pc])) if per_pc[pc] else float("nan")
                 for pc in pcs_with_data]
        mins = [float(np.min(per_pc[pc])) if per_pc[pc] else float("nan")
                for pc in pcs_with_data]
        maxs = [float(np.max(per_pc[pc])) if per_pc[pc] else float("nan")
                for pc in pcs_with_data]
        return means, mins, maxs

    # ---- DEPRECATED K/L-search permutation-null bands -------------------
    # Plot null bands FIRST (so they're behind the lines).
    # n-cell null = per (pc, style, perm), max ρ over the first n cells.
    # Aggregate per PC: mean and SD across perms × styles.
    # DEPRECATED (May 2026): only drawn when --include_optimization is
    # set; the canonical-only default uses the inline shuffled-judge
    # null at L=K=0 (drawn earlier as the pink band).
    if not args.include_optimization:
        pass
    elif null_per_cell:
        for n in n_cell_levels:
            color = CELL_COUNT_COLORS[n]
            cells_n = set(cell_order[:n])
            per_pc_n: dict[int, list[float]] = {pc: [] for pc in pcs_with_data}
            for (pc, style, perm_idx), cell_rhos in null_per_cell.items():
                if pc not in per_pc_n:
                    continue
                rhos_in = [r for c, r in cell_rhos.items() if c in cells_n]
                if rhos_in:
                    per_pc_n[pc].append(max(rhos_in))
            null_means_n = [float(np.mean(per_pc_n[pc]))
                             if per_pc_n[pc] else float("nan")
                             for pc in pcs_with_data]
            null_sds_n = [float(np.std(per_pc_n[pc], ddof=1))
                           if len(per_pc_n[pc]) > 1 else 0.0
                           for pc in pcs_with_data]
            null_lo_2 = [m - 2 * s for m, s in zip(null_means_n, null_sds_n)]
            null_hi_2 = [m + 2 * s for m, s in zip(null_means_n, null_sds_n)]
            null_lo_3 = [m - 3 * s for m, s in zip(null_means_n, null_sds_n)]
            null_hi_3 = [m + 3 * s for m, s in zip(null_means_n, null_sds_n)]
            ax_rho.fill_between(pcs_with_data, null_lo_2, null_hi_2,
                                color=color, alpha=0.12, linewidth=0,
                                label="range optimized against shuffled data "
                                      "(control) ± 2 σ")
            ax_rho.plot(pcs_with_data, null_means_n,
                        linestyle="--", linewidth=1.0,
                        color=color, alpha=0.7,
                        label=None)
            # ± 3 σ outer envelope (dotted) — visualises how unusual a real
            # ρ point is against the shuffled-data control's tail.
            ax_rho.plot(pcs_with_data, null_lo_3,
                        linestyle=":", linewidth=1.0,
                        color=color, alpha=0.6,
                        label="control ± 3 σ")
            ax_rho.plot(pcs_with_data, null_hi_3,
                        linestyle=":", linewidth=1.0,
                        color=color, alpha=0.6,
                        label=None)
    elif null_per_pc:
        # Fallback: only global null available (older cache).  Plot one
        # band as a generic noise-floor reference.
        null_means = [float(np.mean(null_per_pc[pc]))
                       if null_per_pc[pc] else 0.0
                       for pc in pcs_with_data]
        null_sds = [float(np.std(null_per_pc[pc], ddof=1))
                     if len(null_per_pc[pc]) > 1 else 0.0
                     for pc in pcs_with_data]
        null_lo_2 = [m - 2 * s for m, s in zip(null_means, null_sds)]
        null_hi_2 = [m + 2 * s for m, s in zip(null_means, null_sds)]
        null_lo_3 = [m - 3 * s for m, s in zip(null_means, null_sds)]
        null_hi_3 = [m + 3 * s for m, s in zip(null_means, null_sds)]
        ax_rho.fill_between(pcs_with_data, null_lo_2, null_hi_2,
                            color="#888888", alpha=0.18, linewidth=0,
                            label="range optimized against shuffled data "
                                  "(control, global) ± 2 σ")
        ax_rho.plot(pcs_with_data, null_means,
                    linestyle="--", linewidth=1.0, color="#888888",
                    alpha=0.7)
        ax_rho.plot(pcs_with_data, null_lo_3,
                    linestyle=":", linewidth=1.0, color="#888888",
                    alpha=0.6, label="control ± 3 σ (global)")
        ax_rho.plot(pcs_with_data, null_hi_3,
                    linestyle=":", linewidth=1.0, color="#888888",
                    alpha=0.6)

    # ---- DEPRECATED optimized ◆ red line -------------------------------
    # Only drawn when --include_optimization is set.  See deprecation
    # banner higher up for context.
    family_specs = [
        ("fixed_principled", per_cell_principled, "D",
         "optimized token, layer, and whitening (DEPRECATED)"),
    ] if args.include_optimization else []

    # Principled ρ: ◆ markers; one line per n in n_cell_levels.
    n_top = n_cell_levels[-1] if n_cell_levels else 0
    for _variant_name, per_cell_dict, marker, base_label in family_specs:
        if not per_cell_dict:
            continue
        for n in n_cell_levels:
            color = CELL_COUNT_COLORS[n]
            agg = n_cell_aggregate(per_cell_dict, n)
            if not agg:
                continue
            means, mins, maxs = per_pc_stats(agg)
            # When only one cell-count level is plotted, the user-facing
            # label has no suffix; otherwise tag it with the cell count
            # so multiple aggregates can be distinguished.
            label = base_label if len(n_cell_levels) == 1 else (
                f"{base_label} ({n}c)"
            )
            ax_rho.plot(pcs_with_data, means, marker=marker, markersize=8,
                        linewidth=1.6, color=color, alpha=1.0,
                        markerfacecolor=color,
                        markeredgecolor="black", markeredgewidth=0.5,
                        label=label)
            ax_rho.fill_between(pcs_with_data, mins, maxs,
                                color=color, alpha=0.10, linewidth=0)
            # Annotate ρ for the largest cell-count line.
            if n == n_top:
                for pc, v in zip(pcs_with_data, means):
                    if not np.isnan(v):
                        ax_rho.text(pc, v + 0.025, f"{v:+.2f}",
                                    ha="center", va="bottom", fontsize=7.5,
                                    color=color)

    ax_rho.axhline(0, color="black", linewidth=0.4)

    ax_rho.set_xscale("log")
    ax_rho.set_xticks(pcs_with_data)
    ax_rho.set_xticklabels([str(pc) for pc in pcs_with_data])
    ax_rho.minorticks_off()
    ax_rho.set_xlabel("PC index (log scale)")
    ax_rho.set_ylabel("Best round-trip Spearman ρ")
    ax_rho.grid(True, which="major", linestyle="-", alpha=0.25)
    # Legend outside the plot to the right — keeps all data visible.
    # Explicit ordering: canonical group (data + control) first, then the
    # optimized data + its two controls.  Matches the conceptual hierarchy
    # rather than the plot-call order (which has the bands plotted before
    # the markers so the markers sit on top visually).
    handles_, labels_ = ax_rho.get_legend_handles_labels()
    desired_order = [
        "canonical PC at canonical cell",
        "canonical PC at non-canonical cells (5 cells)",
        "canonical control: ±2/√(2n) (95% chance ρ for null)",
        "shuffled-judge null at L=K=0 (per-cell, mean of gloss+inline) ± 2 σ",
        "optimized token, layer, and whitening (DEPRECATED)",
        "range optimized against shuffled data (control) ± 2 σ",
        "control ± 3 σ",
    ]
    label_to_handle = dict(zip(labels_, handles_))
    ordered_handles = [label_to_handle[l] for l in desired_order
                       if l in label_to_handle]
    ordered_labels = [l for l in desired_order if l in label_to_handle]
    # Append any other labels we didn't account for (defensive).
    for h, l in zip(handles_, labels_):
        if l in ordered_labels:
            continue
        ordered_handles.append(h)
        ordered_labels.append(l)
    # Place the legend in figure coordinates so its position is decoupled
    # from any axes squeezing tight_layout does -- otherwise the legend
    # ends up overlapping the x-axis label when tight_layout shrinks the
    # axes to make room for the bottom margin.
    ax_rho.legend(ordered_handles, ordered_labels,
                   loc="upper center", bbox_to_anchor=(0.5, 0.17),
                   bbox_transform=fig.transFigure,
                   ncol=1, handletextpad=0.6,
                   framealpha=0.92, fontsize=8.5)
    levels_str = "/".join(str(n) for n in n_cell_levels)
    cells_spec = ", ".join(f"({s},{l})"
                            for s, l in cell_order[:n_top]) if n_top else "(none)"
    tag_parts = []
    if args.exchanged:
        tag_parts.append("PAIR-EXCHANGED CONTROL")
    if cutoff_tag_label:
        tag_parts.append(cutoff_tag_label)
    if args.include_optimization:
        tag_parts.append("LEGACY OPTIMIZATION VIEW (DEPRECATED)")
    extra_tag = ("  [" + ", ".join(tag_parts) + "]") if tag_parts else ""
    if not args.include_optimization:
        title = (f"PC round-trip ρ — principled cross-cell transport, "
                 f"6-cell view{extra_tag}")
        spec_lines = [
            "× = canonical PC at the canonical cell (slot=7, layer=25, "
            "L=0, K=0).  Dotted band: ±2/√(2n) chance ρ at H₀.",
            "Pale lines: same canonical PC transported to each of the 5 "
            "other (slot, layer) cells via canonical α-coefficients "
            "(L=K=0).",
            "Pink: shuffled-judge null at L=K=0 (per-cell, mean of "
            "gloss + inline) ± 2 σ.",
        ]
    else:
        title = (f"PC round-trip ρ — principled transport vs DEPRECATED "
                 f"(token, layer, whitening)-optimized; "
                 f"{levels_str}-cell aggregate{extra_tag}")
        spec_lines = [
            "× = canonical PC at canonical cell (slot=7, layer=25, "
            "L=0, K=0).  Dotted band: ±2/√(2n) chance ρ at H₀.",
            "◆ = (token, layer, whitening)-optimized canonical PC, max ρ "
            f"over {n_top}-cell prefix of [{cells_spec}].  DEPRECATED.",
            "Pink: shuffled-judge null at L=K=0 (per-cell, mean of "
            "gloss + inline) ± 2 σ.  Red band: matched K/L-search "
            "permutation null ± 2 σ (DEPRECATED, dotted = ±3 σ).",
        ]
    # ``line_height`` controls both the bold-title↔first-spec gap and the
    # spacing between consecutive spec lines.  Bumped from 0.026 to 0.034
    # to give the title some breathing room.  Then we override the
    # function's default ``bottom - 0.015`` plot-top with a tighter
    # ``bottom - 0.003`` so the spec block sits closer to the plot.
    bottom, _ = suptitle_with_specs(fig, title, spec_lines, line_height=0.034)
    # 0.18 bottom margin gives the 6-row 1-column legend room below the axes.
    fig.tight_layout(rect=(0, 0.18, 1.0, bottom - 0.003))

    # --- Provenance inputs ---
    # The sources this plot reads:
    #   * three optional refinement caches (already appended above by
    #     load_and_register when --include_optimization is set);
    #   * vector subtrees + four canonical-angles derived marginals
    #     (setup_at + build_goal_nogoal_subspaces fan-out);
    #   * per-(PC, style, judge, mode) judge-score caches under sweep_dir
    #     (up to PCs × 2 styles × 2 judges × 2 modes = up to 192 files at
    #     the current 24-PC default), each as its own InputSpec so audit
    #     reports can pinpoint which (PC, style) cell was rejudged;
    #   * the permutation-null cache (already appended above by
    #     load_and_register; the duplicate-path branch below preserves
    #     the legacy behaviour that recorded the dep even when the
    #     file did not exist at script start, but is now a no-op since
    #     we already attempted the load).
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
    # Per-(PC, style, judge, mode) judge cache InputSpecs were
    # already appended above by load_combined_scores → load_and_register
    # at read time, so the previous duplicate-append loop here is
    # gone.  Note: ``extras={"exchanged": str(args.exchanged)}`` no
    # longer rides on these specs -- the exchanged-pair semantics are
    # captured by the script's own producer_script fingerprint
    # (extras are advisory, never affect drift comparisons), so
    # nothing material is lost.
    # ``null_cache_json`` was already appended above by load_and_register
    # at the point of consumption (when --include_optimization is set
    # and the file exists).  The previous duplicate-append site here
    # would also register the dep when load_and_register had skipped
    # the read on FileNotFoundError -- but recording a dep we never
    # read is the exact failure mode the load_and_register retrofit
    # is designed to prevent, so the duplicate is dropped.

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
