#!/usr/bin/env python3
"""Render the nth_pc winner decomposition heatmap.

Loads the (n_canon_pcs, n_strips * 6) matrix from
``winner_decomposition.py``'s npz output and renders it as a heatmap
where:
  - Each column is a probability distribution over canonical PCs
    (squared coefficients of the winner direction projected onto canonical
    Vt, normalised to sum to 1).
  - The 16 PC indices have one strip each, side-by-side.
  - Within a strip: 6 columns = 3 cell counts × 2 styles.
    Order: glo-1c, glo-2c, glo-3c, inl-1c, inl-2c, inl-3c.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from assistant_axis.plot_metadata import png_metadata, suptitle_with_specs
from assistant_axis.provenance import (
    CACHE_POLICIES, InputSpec, load_and_register_npz,
)


DEFAULT_NPZ = "roger/pc_round_trip_nth_pc_winner_decomposition.npz"
DEFAULT_OUTPUT = "roger/pc_round_trip_nth_pc_winner_decomposition.png"


def render_heatmap(ax, matrix: np.ndarray, column_keys: list,
                    *, max_canon_pc: int = 0,
                    log_color: bool = True,
                    title: str = "") -> None:
    """Render heatmap on the given axis.

    matrix: (n_canon_pcs, n_cols).
    column_keys: list of (pc, style, n) tuples, len == n_cols.
    """
    if max_canon_pc <= 0:
        max_canon_pc = matrix.shape[0]
    M = matrix[:max_canon_pc, :].copy()
    n_canon, n_cols = M.shape

    # Replace zeros with a small floor for log color scale.
    if log_color:
        floor = 1e-5
        M_disp = np.clip(M, floor, 1.0)
        norm = mcolors.LogNorm(vmin=floor, vmax=1.0)
    else:
        M_disp = M
        norm = None

    # pcolormesh with log y.
    # x edges: 0, 1, 2, ..., n_cols
    # y edges: 0.5, 1.5, ..., n_canon + 0.5  (so canonical PC k is at y=k)
    x_edges = np.arange(n_cols + 1)
    y_edges = np.arange(n_canon + 1) + 0.5

    mesh = ax.pcolormesh(x_edges, y_edges, M_disp,
                         cmap="viridis", norm=norm, shading="flat",
                         rasterized=True)
    # Y axis: canonical PC 1 at TOP, large PCs at BOTTOM.  Log scale.
    ax.set_ylim(n_canon + 0.5, 0.5)        # inverted: 1 at top
    ax.set_yscale("log")
    ax.set_ylabel("Canonical PC index k\n(rows: contribution to decomposition)")

    # Draw PC strip boundaries (every 6 cols).
    pcs_in_order = []
    for pc, _, _ in column_keys:
        if pc not in pcs_in_order:
            pcs_in_order.append(pc)
    for i in range(1, len(pcs_in_order)):
        ax.axvline(i * 6, color="black", linewidth=0.6, alpha=0.4)
    # Within each strip, light divider after column 3 (separates glo and inl).
    for i in range(len(pcs_in_order)):
        ax.axvline(i * 6 + 3, color="white", linewidth=0.4, alpha=0.5)

    # X axis: strip centers labeled with PC index.
    strip_centers = [i * 6 + 3 for i in range(len(pcs_in_order))]
    ax.set_xticks(strip_centers)
    ax.set_xticklabels([str(pc) for pc in pcs_in_order])
    ax.set_xlabel("PC index N (each strip = 6 cols: glo-1c|glo-2c|glo-3c | "
                  "inl-1c|inl-2c|inl-3c)")
    ax.set_xlim(0, n_cols)
    if title:
        ax.set_title(title, fontsize=11)
    return mesh


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--npz", default=DEFAULT_NPZ)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--max_canon_pc", type=int, default=0,
                   help="Limit y-axis range to top-N canonical PCs "
                        "(default 0 = use all rows in matrix).")
    p.add_argument("--linear_color", action="store_true",
                   help="Use linear color scale (default: log).")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to handle drift between the npz's recorded "
                        "inputs (winner_decomposition.py's upstream JSON + "
                        "data subtrees) and current state.  Default 'warn' "
                        "prints a stderr summary but renders anyway.")
    args = p.parse_args()

    inputs: list[InputSpec] = []
    data, meta, _spec, _check = load_and_register_npz(
        Path(args.npz),
        dep_key="winner_decomposition_npz",
        policy=args.cache_policy,
        inputs=inputs,
    )
    matrix = data["matrix"]
    column_keys = list(data["column_keys"])

    fig, ax = plt.subplots(figsize=(15, 7))
    mesh = render_heatmap(
        ax, matrix, column_keys,
        max_canon_pc=args.max_canon_pc,
        log_color=not args.linear_color,
        title="nth_pc winner decomposition in canonical PC basis",
    )
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("squared coefficient (column sums to 1)")

    title = ("PC round-trip: nth_pc winner decomposition in canonical PC basis "
             "(L=2 sheared space)")
    spec = (
        "Each column shows the projected squared coefficients of an nth_pc "
        "winner direction onto canonical Vt[k] (normalised to sum to 1).\n"
        "Path: post-whitened winner → inverse whiten/shear at target → "
        "raw R^D at target → least-squares α onto target M_raw → reconstruct "
        "α @ M_canon_raw → forward canonical L=2 shear → decompose vs Vt_canon."
    )
    _, top_rect = suptitle_with_specs(fig, title, spec, line_height=0.024)
    fig.tight_layout(rect=(0, 0, 1, top_rect))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The npz can't carry a JSON envelope, so the transitive
    # provenance chain (back to per-cell-winners JSON + data
    # subtrees) lives inside its ``meta`` block; load_and_register_npz
    # validated that block above and registered the npz itself as our
    # one declared dependency in ``inputs``.  PNG metadata records
    # that single dep_key; downstream auditors recover the chain by
    # opening the npz and reading meta["_inputs"].
    fig.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
