#!/usr/bin/env python3
"""All-slots PCA scree plots: variance explained per principal component.

Promoted from the ``Compare Scree plots`` cell of
``notebooks/pca.ipynb`` (April 2026 work).  For a fixed target layer,
fit a PCA per slot on the role-vector matrix, then overlay the
per-component variance-explained curves for all slots in a 1x3 panel:

- **Linear**     -- bars (individual VE %) + lines (cumulative VE %).
                    The bars are interleaved across slots with
                    width = 0.8 / n_slots so they always fit; the
                    lines let you see at a glance how many PCs each
                    slot needs to hit a given variance threshold.
- **Log-linear** -- per-component VE on log y, linear x.  Highlights
                    differences in the tail (small PCs).
- **Log-log**    -- log y, log x.  Reveals power-law-like structure
                    in the spectrum if it's there.

Auto-scales to whatever ``num_slots`` is on disk: for the older
4-slot Christina-headers data it produces 4 overlaid curves with the
original ``tab10``-style colours; for the 8-slot Qwen-3 non-thinking
header data it switches to the ``plasma`` palette so slots 0..7 stay
visually distinct (the 4-colour palette is too cramped at 8 slots).

Usage::

    # Default: 8-slot Roger data, layer 24, all roles
    uv run python results_analysis/pca_scree_plots.py

    # Pin to a different layer
    uv run python results_analysis/pca_scree_plots.py --layer 25

    # Run on traits instead of roles
    uv run python results_analysis/pca_scree_plots.py \\
        --vectors_subdir traits/vectors

    # Reproduce the original 4-slot notebook figure on Christina-headers
    uv run python results_analysis/pca_scree_plots.py \\
        --data_dir 'runpod_workspace/qwen/qwen-3-32b Christina headers'

Output is a single PNG per run: ``pca_scree_all_slots_<kind>_L<N>.png``
in ``--output_dir`` (default ``roger/pca_scree_plots_out/``).  The
plot embeds full provenance via ``png_metadata`` so the figure is
reproducible from the saved PNG alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from assistant_axis import (  # noqa: E402
    MeanScaler,
    compute_pca,
    load_role_vector,
    png_metadata,
    slot_colors_8,
    suptitle_with_specs,
)
from assistant_axis.provenance import (  # noqa: E402
    InputSpec,
    current_data_subtree_input,
)


# Disambiguated slot names (slots 5 and 7 are both ``\n\n`` text).
# Used purely for legend display when 8 slots are loaded; for 4 slots
# we use slot_labels() from metadata for the legend.
SLOT_NAMES_8 = [
    "body-mean", "<|im_start|>", "assistant", r"\n",
    "<think>", r"\n\n (in)", "</think>", r"\n\n (post)",
]


def _slot_legend_labels(num_slots: int, metadata: dict | None) -> list[str]:
    """Pick legend labels: prefer the disambiguated 8-slot constants
    when ``num_slots == 8``, otherwise fall back to whatever
    ``slot_labels`` derives from the loaded metadata.
    """
    if num_slots == 8:
        return list(SLOT_NAMES_8)
    # Fall back to metadata-derived labels (4-slot Christina-headers
    # data, etc.).  Reuse the project helper so the strings match
    # other plotting scripts.
    from assistant_axis import slot_labels
    labels = slot_labels(metadata)
    if len(labels) < num_slots:
        labels = labels + [f"slot {i}" for i in range(len(labels), num_slots)]
    # Display-escape any literal newlines.
    return [lbl.replace("\n", r"\n") for lbl in labels[:num_slots]]


def _slot_colors(num_slots: int) -> list:
    """Project-canonical 8-slot palette (see
    ``assistant_axis.plot_palette``):
      slot 0 (body-mean)              -> black
      slots 2/3/5/7 (regular text)    -> plasma, slot 7 dark-purple end
      slots 1/4/6 (special markers)   -> winter, slot 6 blue end

    For ``num_slots == 4`` (Christina-headers data) we fall back to
    the original ``tab10`` colours so the historical 4-slot figure
    still reproduces byte-for-byte.  For ``num_slots == 8`` we use
    the canonical 8-slot palette unchanged.  Other slot counts get
    the canonical palette truncated/extended via ``slot_colors``.
    """
    if num_slots == 4:
        return ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    if num_slots == 8:
        return slot_colors_8()
    # General case: use the truncate/extend helper.
    from assistant_axis import slot_colors  # type: ignore  # noqa: E402
    return slot_colors(num_slots)


# ---------------------------------------------------------------------------
# Data loading + PCA
# ---------------------------------------------------------------------------

def load_vectors_at_layer(
    vectors_dir: Path, layer: int,
) -> tuple[torch.Tensor, int, dict | None]:
    """Load all role vectors at ``layer``, stacked into ``(N, S, D)``.

    Skips ``default.pt``.  Returns ``(stacked, num_slots, first_meta)``.
    """
    files = sorted(vectors_dir.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files in {vectors_dir}")
    stacks: list[torch.Tensor] = []
    first_meta: dict | None = None
    num_slots: int | None = None
    for f in files:
        if f.stem == "default":
            continue
        try:
            vec, meta = load_role_vector(str(f))
        except Exception as e:  # pylint: disable=broad-except
            print(f"  skip {f.name}: {e}")
            continue
        if vec.ndim == 2:
            vec = vec.unsqueeze(0)
        if num_slots is None:
            num_slots = vec.shape[0]
        if first_meta is None and meta:
            first_meta = meta
        stacks.append(vec[:, layer].float())  # (S, D) at this layer
    if not stacks:
        raise SystemExit(f"No usable vectors in {vectors_dir}")
    assert num_slots is not None
    stacked = torch.stack(stacks, dim=0)  # (N, S, D)
    return stacked, num_slots, first_meta


def fit_pca_per_slot(
    stacked: torch.Tensor,  # (N, S, D)
) -> list[np.ndarray]:
    """Fit a PCA per slot at the chosen layer.

    Returns the per-slot ``variance_explained`` arrays (each is a
    1-D numpy array of length min(N, D)).
    """
    num_slots = stacked.shape[1]
    out: list[np.ndarray] = []
    for s in range(num_slots):
        vecs_s = stacked[:, s].float()
        scaler = MeanScaler()
        _, variance_explained, _, _, _ = compute_pca(
            vecs_s, layer=None, scaler=scaler, verbose=False,
        )
        out.append(np.asarray(variance_explained, dtype=float))
        print(f"  Slot {s}: PCA fitted, "
              f"{len(variance_explained)} components, "
              f"top PC variance = {variance_explained[0] * 100:.2f} %")
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(
    variance_per_slot: list[np.ndarray],
    *,
    legend_labels: list[str],
    colors: list,
    layer: int,
    model_title: str,
    output_path: Path,
    spec_extra: str,
    max_components: int = 100,
    inputs: list[InputSpec] | None = None,
) -> Path:
    """Render the 1x3 overlay panel (Linear / Log-linear / Log-log)."""
    num_slots = len(variance_per_slot)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    n_total = max(len(v) for v in variance_per_slot)
    n_show = min(n_total, max_components)
    components = np.arange(1, n_show + 1)

    bar_width = 0.8 / num_slots
    for s, ve in enumerate(variance_per_slot):
        ve = ve[:n_show]
        cumulative = np.cumsum(ve)
        color = colors[s]
        lbl = legend_labels[s]

        # Linear panel: interleaved bars (per-PC %) + cumulative line
        offset = (s - (num_slots - 1) / 2) * bar_width
        axes[0].bar(components + offset, ve * 100, width=bar_width,
                    color=color, alpha=0.6, edgecolor="none")
        axes[0].plot(components, cumulative * 100,
                     color=color, linewidth=1.2, label=lbl)

        # Log-linear: per-PC % on log y
        axes[1].plot(components, ve * 100, color=color, linewidth=1.2,
                     marker=".", markersize=3, label=lbl)
        # Log-log
        axes[2].plot(components, ve * 100, color=color, linewidth=1.2,
                     marker=".", markersize=3, label=lbl)

    axes[0].set_xlim(0, n_show + 1)
    axes[0].set_ylim(0, None)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_title(f"Linear -- All slots ({model_title})", fontsize=11)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    legend_fontsize = 8 if num_slots <= 4 else 7
    axes[0].legend(fontsize=legend_fontsize, loc="lower right")

    for ax, title_prefix, log_x in [
        (axes[1], "Log-linear", False),
        (axes[2], "Log-log", True),
    ]:
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance Explained (%)")
        ax.set_title(f"{title_prefix} -- All slots ({model_title})",
                     fontsize=11)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=legend_fontsize, loc="upper right")

    suptitle = (f"Per-slot PCA variance explained at layer {layer} "
                f"({model_title})")
    spec_line = (f"{num_slots} slots; max {n_show} PCs shown"
                 f"{spec_extra}")
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
             "the trait corpus.",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="roger/pca_scree_plots_out",
        help="Where to write the PNG.",
    )
    p.add_argument(
        "--layer", type=int, default=24,
        help="Target layer to fit PCA at; the original notebook used "
             "layer 24.  Each layer needs its own PNG.",
    )
    p.add_argument(
        "--max_components", type=int, default=100,
        help="Cap the number of PCs plotted; full spectrum can be "
             "long-tailed and the interesting regime is the head.",
    )
    p.add_argument(
        "--model_title", type=str, default="Qwen 3 32B",
        help="Display string for plot titles.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    vectors_dir = data_dir / args.vectors_subdir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Step 1: Loading vectors from {vectors_dir}, layer {args.layer}...")
    stacked, num_slots, first_meta = load_vectors_at_layer(
        vectors_dir, args.layer,
    )
    print(f"  Loaded {stacked.shape[0]} entities × {num_slots} slots × "
          f"{stacked.shape[-1]} hidden_dim")

    print(f"Step 2: Fitting PCA per slot at layer {args.layer}...")
    variance_per_slot = fit_pca_per_slot(stacked)

    legend_labels = _slot_legend_labels(num_slots, first_meta)
    colors = _slot_colors(num_slots)

    entity_kind = Path(args.vectors_subdir).parts[0]
    out = output_dir / (
        f"pca_scree_all_slots_{entity_kind}_L{args.layer}.png"
    )
    print(f"Step 3: Plotting -> {out}")
    inputs: list[InputSpec] = [
        current_data_subtree_input(
            data_dir, args.vectors_subdir, dep_key="vectors_subtree",
            extras={"layer": str(args.layer)}),
    ]
    make_plot(
        variance_per_slot,
        legend_labels=legend_labels,
        colors=colors,
        layer=args.layer,
        model_title=args.model_title,
        output_path=out,
        spec_extra=f"; data: {data_dir.name}",
        max_components=args.max_components,
        inputs=inputs,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
