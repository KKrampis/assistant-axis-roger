#!/usr/bin/env python3
"""Per-slot growth of role-pair difference norms across layers.

For a sample of random role pairs ``(A, B)``, plot
``‖A[s, L] − B[s, L]‖`` (log scale) as a function of layer ``L``,
one panel per slot ``s``.  Shows how *separable* roles are at each
token position, layer-by-layer.

Promoted from a one-off chat-inline plot (April 2026) that used the
old 4-slot ``qwen-3-32b Christina headers`` data.  Recreated here as
a tracked tool that auto-scales the panel grid to whatever number of
slots is present in the loaded vectors:

- 4 slots -> 2x2 grid
- 8 slots -> 2x4 grid (the new Qwen-3 non-thinking layout default)

Companion to ``token_position_noise_analysis.py``: where that script
asks "are slot-pair *direction-of-difference* signals consistent
across slots?" (Pearson of norms + cosine of diff-vectors), this one
asks "how does the *magnitude* of a role-pair difference grow per
slot as you walk up the layers?".  The two views together let you
spot slots where roles are well-separated AND consistent, vs slots
where they're either crowded or noisy.

Reading the plot: a tight band climbing from ``~10^0`` at layer 0 to
``~10^3`` at layer 60 means roles are progressively more separated
at that slot as you go deeper -- typical of header slots where the
model has finished accumulating role-specific context.  A plateau in
the middle layers (visible at slot 0 / body-mean in some checkpoints)
means role-pair separation isn't growing -- the body-mean
representation isn't accumulating much role-specific differentiation
at those depths.

Usage::

    # Default: 8-slot Roger data, 400 random pairs, all 64 layers
    uv run python results_analysis/role_pair_diff_norms.py

    # Run on traits instead of roles
    uv run python results_analysis/role_pair_diff_norms.py \\
        --vectors_subdir traits/vectors

    # Reproduce the original 4-slot version on Christina headers
    uv run python results_analysis/role_pair_diff_norms.py \\
        --data_dir 'runpod_workspace/qwen/qwen-3-32b Christina headers'

The 8-slot run with 400 pairs takes ~30 s end-to-end (most of it is
``torch.load`` for ~280 vector files); the plot itself is ~2 s.
"""
from __future__ import annotations

import argparse
import random
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from assistant_axis import png_metadata, suptitle_with_specs  # noqa: E402
from assistant_axis.provenance import (  # noqa: E402
    InputSpec,
    current_data_subtree_input,
)


# Same canonical 8-slot layout as token_position_noise_analysis.py and
# all_roles_pairwise_slots.py.  The script slices SLOT_NAMES[:S] based
# on the loaded tensor's S so older 4-slot data still works.
SLOT_NAMES = [
    "body-mean", "<|im_start|>", "assistant", r"\n",
    "<think>", r"\n\n (in)", "</think>", r"\n\n (post)",
]


# ---------------------------------------------------------------------------
# Subplot grid sizing
# ---------------------------------------------------------------------------

def _grid_shape(n_slots: int) -> tuple[int, int]:
    """Choose a (rows, cols) grid that's not too tall or too wide.

    Hard-coded for the small set of slot counts we actually use:
    1->(1,1), 2->(1,2), 3->(1,3), 4->(2,2), 5..6->(2,3), 7..8->(2,4),
    9..12->(3,4), >12 falls back to a square-ish layout.  Keeps panel
    aspect ratios close to landscape (~1.7:1) so log-y plots remain
    readable.
    """
    if n_slots <= 1:
        return (1, 1)
    if n_slots <= 3:
        return (1, n_slots)
    if n_slots <= 4:
        return (2, 2)
    if n_slots <= 6:
        return (2, 3)
    if n_slots <= 8:
        return (2, 4)
    if n_slots <= 12:
        return (3, 4)
    cols = int(np.ceil(np.sqrt(n_slots)))
    rows = int(np.ceil(n_slots / cols))
    return (rows, cols)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_vectors(vectors_dir: Path) -> tuple[dict[str, torch.Tensor], int, int]:
    """Load all entity vectors, skipping ``default.pt``.

    Returns ``(vecs, n_slots, n_layers)``.  Each value in ``vecs`` is
    ``(S, L, D)`` float32.
    """
    files = sorted(vectors_dir.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files in {vectors_dir}")
    vecs: dict[str, torch.Tensor] = {}
    s_l_d: tuple[int, int, int] | None = None
    for f in files:
        if f.stem == "default":
            continue
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:  # pylint: disable=broad-except
            print(f"  skip {f.name}: {e}")
            continue
        v = data["vector"] if isinstance(data, dict) and "vector" in data else data
        v = v.float()
        if s_l_d is None:
            s_l_d = (v.shape[0], v.shape[1], v.shape[2])
        vecs[f.stem] = v
    if not vecs:
        raise SystemExit(f"No usable vectors in {vectors_dir}")
    assert s_l_d is not None
    return vecs, s_l_d[0], s_l_d[1]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(
    vecs: dict[str, torch.Tensor],
    *,
    n_slots: int,
    n_layers: int,
    n_pairs: int,
    seed: int,
    output_path: Path,
    spec_extra: str,
    inputs: list[InputSpec] | None = None,
) -> Path:
    """Render the per-slot pair-diff-norm plot.

    Picks ``n_pairs`` random distinct (A, B) pairs from ``vecs`` (seed
    ``seed``), then for each slot computes ``‖A − B‖`` along all
    layers and draws one log-scale ribbon per pair.
    """
    names = list(vecs.keys())
    random.seed(seed)
    all_pairs = list(combinations(names, 2))
    if n_pairs > len(all_pairs):
        print(f"  only {len(all_pairs)} pairs available; using all of them")
        n_pairs = len(all_pairs)
    pairs = random.sample(all_pairs, n_pairs)
    print(f"  Sampled {len(pairs)} pairs from {len(names)} entities "
          f"({len(all_pairs)} possible)")

    # Pre-stack the chosen pairs into two tensors so we can compute
    # all per-slot norms in a single vectorised pass.  At 400 pairs *
    # 8 slots * 64 layers * 5120 dim, both tensors fit easily in
    # memory (~6 GB float32 * 2 = manageable).
    print("  Computing pair-diff norms...")
    A = torch.stack([vecs[a] for a, _ in pairs])  # (P, S, L, D)
    B = torch.stack([vecs[b] for _, b in pairs])
    diffs = (A - B).norm(dim=-1)                  # (P, S, L) -- log later
    diffs_np = diffs.numpy()                      # for plotting

    rows, cols = _grid_shape(n_slots)
    panel_w = 5.5
    panel_h = 3.6
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * panel_w, rows * panel_h))
    axes_flat = np.array(axes).reshape(-1) if rows * cols > 1 else np.array([axes])
    layers = np.arange(n_layers)

    for s in range(n_slots):
        ax = axes_flat[s]
        # alpha=0.15 + linewidth=0.5 keeps the ribbon dense but
        # individual outliers visible.  steelblue chosen for parity
        # with the original 4-slot one-off (matches the report figure
        # cosmetically; the panel content is the substantive thing).
        for p_idx in range(diffs_np.shape[0]):
            ax.semilogy(layers, diffs_np[p_idx, s], alpha=0.15,
                        linewidth=0.5, color="steelblue")
        slot_label = SLOT_NAMES[s] if s < len(SLOT_NAMES) else f"slot {s}"
        ax.set_title(f"Slot {s}: {slot_label}", fontsize=12)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel(r"$\|A - B\|$ (log scale)", fontsize=11)
        ax.set_xlim(0, n_layers - 1)
        # Major ticks/grid every 10 layers (with labels), minor every 2.
        # Matches the convention used by all_roles_pairwise_slots.py and
        # token_position_noise_analysis.py.
        ax.set_xticks(range(0, n_layers, 10))
        ax.set_xticks(range(0, n_layers, 2), minor=True)
        ax.grid(True, which="major", alpha=0.4, linewidth=0.8)
        ax.grid(True, which="minor", axis="x", alpha=0.25, linewidth=0.5)

    # Hide any unused panels (e.g. n_slots=5 in a 2x3 grid).
    for s in range(n_slots, len(axes_flat)):
        axes_flat[s].set_visible(False)

    suptitle = (f"Role-pair diff norms across layers "
                f"({len(pairs)} random pairs, log scale)")
    spec_line = (f"{len(names)} entities x {n_slots} slots x {n_layers} layers"
                 f"; seed={seed}{spec_extra}")
    _, top_rect = suptitle_with_specs(fig, suptitle, spec_line)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=suptitle, inputs=inputs))
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_dir", type=str,
        default="runpod_workspace/qwen/qwen-3-32b Roger 8slot",
        help="Base data dir; vectors loaded from "
             "<data_dir>/<vectors_subdir>/*.pt.",
    )
    p.add_argument(
        "--vectors_subdir", type=str, default="roles/vectors",
        help="Sub-path within --data_dir.  Use 'traits/vectors' for "
             "traits.",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="roger/role_pair_diff_norms_out",
        help="Where to write the PNG.  Gitignored under roger/ by "
             "default.",
    )
    p.add_argument("--n_pairs", type=int, default=400,
                   help="Number of random distinct (A, B) pairs to plot.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    data_dir = Path(args.data_dir)
    vectors_dir = data_dir / args.vectors_subdir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading vectors...")
    vecs, n_slots, n_layers = load_vectors(vectors_dir)
    print(f"  Loaded {len(vecs)} entities at "
          f"({n_slots} slots, {n_layers} layers)")

    entity_kind = Path(args.vectors_subdir).parts[0]
    out = output_dir / f"role_pair_diff_norms_{entity_kind}.png"
    print(f"Step 2: Plotting {n_slots}-panel grid -> {out}")
    inputs: list[InputSpec] = [
        current_data_subtree_input(
            data_dir, args.vectors_subdir, dep_key="vectors_subtree"),
    ]
    make_plot(
        vecs, n_slots=n_slots, n_layers=n_layers,
        n_pairs=args.n_pairs, seed=args.seed,
        output_path=out,
        spec_extra=f"; data: {data_dir.name}",
        inputs=inputs,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
