#!/usr/bin/env python3
"""Token-position noise analysis: which token slot is the noisiest?

Promoted from ``roger/token_position_noise_analysis.py`` (April 2026 work)
to a tracked tool.  Compares body-mean (slot 0) vs the Qwen-3 chat
template's header tokens (slot 1 onwards) by examining the consistency
of role-pair difference vectors across random triples, per layer.

In the original 4-slot setup the comparison was body-mean vs the first
three header tokens (``<|im_start|>``, ``assistant``, ``\\n``).  In the
8-slot setup it includes the full non-thinking header
(``<|im_start|>``, ``assistant``, ``\\n``, ``<think>``, ``\\n\\n (in)``,
``</think>``, ``\\n\\n (post)``), giving 7 header positions to compare
against body-mean.  The script auto-detects ``S`` from the loaded
vectors; the only thing that needs editing for a new slot count is
``SLOT_NAMES`` below.

For each random triple ``(A, B, C)`` and each token slot s, the script
computes six metrics measuring how similar ``B-A`` is to ``B-C`` at slot
s, in three distance flavours (cosine, norm, squared-norm), each in
``sub`` (difference) and ``lograt`` (log-ratio) form. The "odd-one-out"
slot per triple/metric is the one whose value deviates most from the
other three; slots that are odd-one-out >25 % of the time are noisier
than the others.

Plus a pairwise-slot analysis (per layer, pooling over **all unordered
role pairs** `(A, B)` drawn from the entity set): rows/columns are token
slots.

- **Norm matrix** -- `norm_corr[i, j]` is the Pearson correlation across
  those `C(N_roles, 2)` pairs between `‖A_i − B_i‖` and `‖A_j − B_j‖`
  (same pair at two slots). High values mean the two slots agree on which
  pairs are "far apart" in magnitude.
- **Cosine matrix** -- `cos_between[i, j]` is the mean (over the same
  role pairs) of `cos(A_i − B_i, A_j − B_j)`. High values mean the two
  slots agree on the *direction* of each pair's difference vector.

Outputs five PNGs:

- ``ooo_heatmaps.png`` -- odd-one-out fraction by layer × slot, per metric.
- ``ooo_summary_bars.png`` -- slot odd-one-out rate per metric (avg layers).
- ``ooo_by_layer.png`` -- per-layer slot odd-one-out, averaged across metrics.
- ``pairwise_corr_matrices.png`` -- ``S×S`` heatmaps for the two summaries
  above (norm correlation + cosine agreement), averaged across layers.
- ``pairwise_cosine_by_layer.png`` -- one curve per unordered slot pair
  ``(s₁, s₂)`` (`C(S, 2)` lines at ``S``=8): at each layer, mean cosine
  ``cos((A−B)@s₁, (A−B)@s₂)`` over all unordered entity pairs `(A,B)`.

Original 4-slot finding (April 2026 run on the
``qwen-3-32b Christina headers`` data, 280 roles, 64 layers, 1000
triples): **slot 3 (``\\n``) was the least noisy** -- odd-one-out
~21-23 % of the time across metrics (slightly below the 25 % chance
floor) and strongest **aggregated** diff-direction cosine signal vs partner
slots in the 4-slot summary tables. Slot 0 (body-mean) was the noisiest
by all six metrics.  This
informed the project-wide convention of using slot 3 as the default
working slot.

Pending re-analysis on the 8-slot data (``qwen-3-32b Roger 8slot``,
which adds slots 4-7 = ``<think>``, ``\\n\\n (in)``, ``</think>``,
``\\n\\n (post)``): with seven header positions instead of three, the
chance floor for "odd-one-out" is now 12.5 % rather than 25 %, so the
absolute numbers shift even when the relative ranking doesn't.  The
default data dir below now points at the 8-slot data; pass
``--data_dir 'runpod_workspace/qwen/qwen-3-32b Christina headers'``
to reproduce the original report figure.

Usage::

    uv run python results_analysis/token_position_noise_analysis.py
    uv run python results_analysis/token_position_noise_analysis.py \\
        --data_dir 'runpod_workspace/qwen/qwen-3-32b Roger' \\
        --vectors_subdir traits/vectors \\
        --output_dir roger/token_position_noise_traits

The defaults reproduce a Roger-data run; override ``--data_dir`` to point
at the older ``qwen-3-32b Christina headers`` checkpoint to reproduce the
original report figures.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from assistant_axis import (  # noqa: E402
    png_metadata,
    slot_colors,
    suptitle_with_specs,
)
from assistant_axis.provenance import (  # noqa: E402
    InputSpec,
    current_data_subtree_input,
)


# Names for the 8-slot Qwen-3 non-thinking layout.  Slots 5 and 7 are
# both ``\\n\\n`` text but at different chat-template positions (inside
# vs after ``<think>...</think>``); the suffixes disambiguate them.
# For the older 4-slot data, only the first 4 entries are used (the
# script slices ``SLOT_NAMES[:S]`` based on the loaded tensor's S).
SLOT_NAMES = [
    "body-mean", "<|im_start|>", "assistant", "\\n",
    "<think>", r"\n\n (in)", "</think>", r"\n\n (post)",
]


def token_axis_label(slot_index: int) -> str:
    """Reader-facing token axis / legend shorthand: ``body-mean``, ``t1``, ``t2``, …."""
    if slot_index == 0:
        return "body-mean"
    return f"t{slot_index}"


def token_axis_labels(n_slots: int) -> List[str]:
    return [token_axis_label(i) for i in range(n_slots)]


SLOT_LABEL = "Token"
N_METRICS = 6
METRIC_NAMES = [
    "cosine_sub", "cosine_lograt",
    "norm_sub", "norm_lograt",
    "sqnorm_sub", "sqnorm_lograt",
]


# ---------------------------------------------------------------------------
# Compute helpers (algorithm preserved verbatim from the original script)
# ---------------------------------------------------------------------------

def load_all_vectors(vectors_dir: Path):
    """Load all role vectors into a single bf16 tensor.
    Returns ``(tensor, role_names)``."""
    pt_files = sorted(vectors_dir.glob("*.pt"))
    if not pt_files:
        raise SystemExit(f"No .pt files found in {vectors_dir}")

    vecs = []
    names = []
    for f in pt_files:
        if f.stem == "default":
            continue
        data = torch.load(f, map_location="cpu", weights_only=False)
        vec = data["vector"] if isinstance(data, dict) and "vector" in data else data
        vecs.append(vec.to(torch.bfloat16))
        names.append(f.stem)

    all_vecs = torch.stack(vecs)  # (N, S, L, D)
    print(f"Loaded {len(names)} entities, shape {tuple(all_vecs.shape)}, "
          f"dtype {all_vecs.dtype}")
    return all_vecs, names


def compute_metrics_for_layer(
    layer_vecs: torch.Tensor,    # (N, S, D) float32
    default_vec: torch.Tensor,   # (S, D) float32
    triples: list,
) -> torch.Tensor:
    """Six metrics for all triples at one layer; returns (N_triples, 6, S)."""
    N = len(triples)
    S = layer_vecs.shape[1]
    results = torch.zeros(N, N_METRICS, S)

    idx_A = torch.tensor([t[0] for t in triples])
    idx_B = torch.tensor([t[1] for t in triples])
    idx_C = torch.tensor([t[2] for t in triples])

    vA = layer_vecs[idx_A]
    vB = layer_vecs[idx_B]
    vC = layer_vecs[idx_C]
    vD = default_vec.unsqueeze(0)

    d_BA = vB - vA
    d_BC = vB - vC

    r_B = vB - vD
    r_A = vA - vD
    r_C = vC - vD

    cos_BA = F.cosine_similarity(r_B, r_A, dim=-1).clamp(-1, 1)
    cos_BC = F.cosine_similarity(r_B, r_C, dim=-1).clamp(-1, 1)

    cdist_BA = 1.0 - cos_BA
    cdist_BC = 1.0 - cos_BC

    norm_BA = d_BA.norm(dim=-1)
    norm_BC = d_BC.norm(dim=-1)
    sqnorm_BA = norm_BA.pow(2)
    sqnorm_BC = norm_BC.pow(2)

    eps = 1e-8

    results[:, 0, :] = cdist_BA - cdist_BC
    results[:, 1, :] = (cdist_BA / (cdist_BC + eps) + eps).abs().log()
    results[:, 2, :] = norm_BA - norm_BC
    results[:, 3, :] = (norm_BA / (norm_BC + eps) + eps).abs().log()
    results[:, 4, :] = sqnorm_BA - sqnorm_BC
    results[:, 5, :] = (sqnorm_BA / (sqnorm_BC + eps) + eps).abs().log()

    return results


def normalize_sub_metrics(all_metrics: torch.Tensor) -> None:
    """Normalize ``_sub`` metrics by per-slot-per-layer mean abs (in-place)."""
    for m_idx in (0, 2, 4):
        vals = all_metrics[:, m_idx, :, :]
        mean_abs = vals.abs().mean(dim=0, keepdim=True).clamp(min=1e-10)
        all_metrics[:, m_idx, :, :] = vals / mean_abs


def odd_one_out(values: torch.Tensor) -> torch.Tensor:
    """For each row of shape (..., S), index of the most anomalous slot
    (largest |slot_k − mean(other S−1)|)."""
    abs_vals = values.abs()
    total = abs_vals.sum(dim=-1, keepdim=True)
    leave_one_out_mean = (total - abs_vals) / (values.shape[-1] - 1)
    deviation = (abs_vals - leave_one_out_mean).abs()
    return deviation.argmax(dim=-1)


def compute_pairwise_slot_correlation(
    layer_vecs: torch.Tensor,  # (N, S, D) float32
):
    """Returns ``(norm_corr, cos_between)``, both ``(S, S)`` ndarrays.

    ``norm_corr[s1, s2]`` is the Pearson correlation across all role pairs
    of the diff norms ``‖A_s1 − B_s1‖`` and ``‖A_s2 − B_s2‖`` -- do the
    two slots agree on which role pairs are far apart?

    ``cos_between[s1, s2]`` is the mean (over all role pairs) cosine of
    the diff *direction vectors* ``(A_s1 − B_s1, A_s2 − B_s2)`` -- do the
    two slots agree on the *direction* of each role pair's difference?
    """
    N, S, _D = layer_vecs.shape
    pairs_i, pairs_j = zip(*itertools.combinations(range(N), 2))
    pairs_i = torch.tensor(pairs_i)
    pairs_j = torch.tensor(pairs_j)

    CHUNK = 4096
    n_pairs = len(pairs_i)

    norms_all = torch.zeros(n_pairs, S)
    for start in range(0, n_pairs, CHUNK):
        end = min(start + CHUNK, n_pairs)
        pi = pairs_i[start:end]
        pj = pairs_j[start:end]
        diffs = layer_vecs[pi] - layer_vecs[pj]
        norms_all[start:end] = diffs.norm(dim=-1)

    cos_between = np.zeros((S, S))
    for s1 in range(S):
        for s2 in range(s1, S):
            cos_vals = torch.zeros(n_pairs)
            for start in range(0, n_pairs, CHUNK):
                end = min(start + CHUNK, n_pairs)
                pi = pairs_i[start:end]
                pj = pairs_j[start:end]
                d1 = layer_vecs[pi, s1] - layer_vecs[pj, s1]
                d2 = layer_vecs[pi, s2] - layer_vecs[pj, s2]
                cos_vals[start:end] = F.cosine_similarity(d1, d2, dim=-1)
            cos_between[s1, s2] = cos_vals.mean().item()
            cos_between[s2, s1] = cos_between[s1, s2]

    norm_corr = np.corrcoef(norms_all.numpy().T)
    return norm_corr, cos_between


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_ooo_heatmaps(ooo_frac: torch.Tensor, output_dir: Path,
                       spec_line: str,
                       inputs: list[InputSpec] | None = None) -> Path:
    heatmap_order = [
        [0, 2, 4],  # cosine_sub, norm_sub, sqnorm_sub
        [1, 3, 5],  # cosine_lograt, norm_lograt, sqnorm_lograt
    ]
    S = ooo_frac.shape[-1]
    # Original figsize=(20, 9) was sized for S=4.  At S=8 the y-axis
    # has twice as many cells; bump fig height so each cell stays
    # ~comfortably tall.  Linear in S above 4, floored at the
    # original 9 inches.
    fig_h = max(9.0, 6.0 + 0.6 * S)
    fig, axes = plt.subplots(2, 3, figsize=(20, fig_h))
    yt_fontsize = 11 if S <= 4 else 10 if S <= 6 else 9
    for r, row_indices in enumerate(heatmap_order):
        for c, m in enumerate(row_indices):
            ax = axes[r, c]
            data = ooo_frac[m].numpy()
            im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.6)
            ax.set_title(METRIC_NAMES[m], fontsize=14)
            ax.set_xlabel("Layer", fontsize=13)
            ax.set_ylabel(SLOT_LABEL, fontsize=13)
            ax.set_yticks(range(S))
            # Short reader-facing names; slot order matches ``SLOT_NAMES``.
            ax.set_yticklabels(token_axis_labels(S), fontsize=yt_fontsize)
            ax.tick_params(axis="x", labelsize=11)
    title = "Odd-one-out fraction by layer and token position"
    _, top_rect = suptitle_with_specs(fig, title, spec_line)
    fig.tight_layout(rect=(0, 0, 0.93, top_rect))
    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Fraction odd-one-out")
    cbar_ax.yaxis.label.set_size(13)
    cbar_ax.tick_params(labelsize=11)
    out = output_dir / "ooo_heatmaps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    plt.close()
    return out


def _plot_ooo_summary_bars(ooo_frac: torch.Tensor, output_dir: Path,
                           spec_line: str,
                           inputs: list[InputSpec] | None = None) -> Path:
    S = ooo_frac.shape[-1]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(N_METRICS)
    # Bar width scales with S so the S-bar group always fits in 0.8
    # of a unit (leaving 0.2 of gap between groups), regardless of S.
    width = 0.8 / S
    # Per-slot palette per the canonical 8-slot convention (see
    # ``assistant_axis.plot_palette``):
    #   slot 0 (body-mean)              -> black
    #   slots 2/3/5/7 (regular text)    -> plasma, slot 7 dark-purple end
    #   slots 1/4/6 (special markers)   -> winter, slot 6 blue end
    colors = slot_colors(S)
    for s in range(S):
        vals = [ooo_frac[m].mean(dim=0)[s].item() for m in range(N_METRICS)]
        ax.bar(x + s * width, vals, width,
               label=token_axis_label(s), color=colors[s])
    # Centre the x-tick under the middle of the S-bar group.
    ax.set_xticks(x + (S - 1) / 2 * width)
    ax.set_xticklabels(METRIC_NAMES, rotation=45, ha="right", fontsize=13)
    ax.set_ylabel("Fraction odd-one-out (avg across layers)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    chance = 1.0 / S
    ax.axhline(chance, color="gray", linestyle="--", alpha=0.5,
               label=f"chance (1/{S} = {chance:.3f})")
    ax.set_xlim(-0.5, N_METRICS + 0.5)
    ax.legend(fontsize=11 if S <= 4 else 10, loc="upper right")
    title = "Which token position is most often the odd one out?"
    _, top_rect = suptitle_with_specs(fig, title, spec_line)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    out = output_dir / "ooo_summary_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    plt.close()
    return out


def _plot_ooo_by_layer(ooo_frac: torch.Tensor, L: int, output_dir: Path,
                       spec_line: str,
                       inputs: list[InputSpec] | None = None) -> Path:
    S = ooo_frac.shape[-1]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    avg_across_metrics = ooo_frac.mean(dim=0).numpy()  # (L, S)
    # Per-slot palette per ``assistant_axis.plot_palette`` (matches the
    # bar plot above and the pairwise-cosine plot below).
    colors = slot_colors(S)
    for s in range(S):
        ax.plot(range(L), avg_across_metrics[:, s],
                label=token_axis_label(s), alpha=0.85, color=colors[s],
                linewidth=1.5)
    chance = 1.0 / S
    ax.axhline(chance, color="gray", linestyle="--", alpha=0.5,
               label=f"chance (1/{S} = {chance:.3f})")
    for x in range(0, L, 2):
        ax.axvline(x, color="gray", linewidth=0.3, alpha=0.3)
    for x in range(0, L, 10):
        ax.axvline(x, color="gray", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Fraction odd-one-out (avg across 6 metrics)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11 if S <= 4 else 10, loc="upper right")
    title = "Odd-one-out fraction by layer (averaged across metrics)"
    _, top_rect = suptitle_with_specs(fig, title, spec_line)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    out = output_dir / "ooo_by_layer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    plt.close()
    return out


def _plot_pairwise_matrices(norm_corr_layers: np.ndarray,
                            cos_between_layers: np.ndarray,
                            output_dir: Path, spec_line: str,
                            inputs: list[InputSpec] | None = None) -> Path:
    S = norm_corr_layers.shape[-1]
    # Scale figsize and font with slot count so the 8-slot heatmap
    # doesn't squash labels.  Empirically (14, 5.8) was sized for S=4;
    # at S=8 the cell count quadruples per heatmap so we want roughly
    # 1.5x the linear panel size.
    panel_w = max(7.0, 1.5 + 0.7 * S)
    fig_w = 2 * panel_w + 1.2
    fig_h = max(5.8, 1.3 + 0.72 * S)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    cell_label_fontsize = 12 if S <= 4 else 10 if S <= 6 else 8
    tick_fontsize = 12 if S <= 4 else 10
    panels = [
        (axes[0], norm_corr_layers.mean(axis=0),
         "Pearson r of ‖A−B‖ across unordered role pairs"),
        (axes[1], cos_between_layers.mean(axis=0),
         "Mean cos(diff at row token, diff at col token), same pairs"),
    ]
    im = None
    title_fs = min(11, tick_fontsize + 1)
    for ax, mat, title in panels:
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(S))
        ax.set_xticklabels(token_axis_labels(S), fontsize=tick_fontsize,
                           rotation=45, ha="right")
        ax.set_yticks(range(S))
        ax.set_yticklabels(token_axis_labels(S), fontsize=tick_fontsize)
        ax.set_title(title, fontsize=title_fs, pad=12)
        ax.set_xlabel("Token (column)", fontsize=tick_fontsize)
        ax.set_ylabel("Token (row)", fontsize=tick_fontsize)
        for i in range(S):
            for j in range(S):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=cell_label_fontsize,
                        color="white" if mat[i, j] < 0.3 else "black")
    suptitle_text = (
        "Per-token matrices: unordered role pairs (A, B), mean over layers"
    )
    _, top_rect = suptitle_with_specs(fig, suptitle_text, spec_line)
    # Reserve extra headroom below the figure title/spec so subplot titles do
    # not collide with ``fig.text`` / suptitle (tight_layout's top_rect is
    # often optimistic for dual-column + long tick labels).
    top_rect_body = max(0.68, top_rect - 0.07)
    fig.tight_layout(rect=(0, 0, 0.92, top_rect_body))
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="agreement (0–1 scale)")
    cbar_ax.tick_params(labelsize=11)
    out = output_dir / "pairwise_corr_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=suptitle_text, inputs=inputs))
    plt.close()
    return out


def _annotate_pairwise_top_curves_at_layers(
    ax,
    cos_between_layers: np.ndarray,
    *,
    slot_pairs: List[tuple[int, int]],
    colors: np.ndarray,
    L: int,
    layers: tuple[int, ...] = (24, 49),
    top_k: int = 3,
    dx_left: float = 2.8,
) -> None:
    """Label the top-*top_k* slot pairs by cosine at each layer in *layers*.

    Text sits above and slightly left of each anchor; 1st/2nd/3rd stack with
    1st highest. Straight arrows run from the right edge of each label
    (``relpos``) to the curve point.
    """
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin + 1e-9
    ann_fs = max(6.0, 9.0 - 0.04 * float(len(slot_pairs)))
    # Vertical gap between stacked labels (data coords, scales with axis).
    min_gap = max(0.022 * y_span, 0.018)
    min_above_pt = max(0.010 * y_span, 0.008)
    ordinals = ("1st", "2nd", "3rd")

    for layer_i in layers:
        if layer_i < 0 or layer_i >= L:
            continue
        ranked = []
        for idx, (s1, s2) in enumerate(slot_pairs):
            yv = float(cos_between_layers[layer_i, s1, s2])
            ranked.append((idx, yv, s1, s2))
        ranked.sort(key=lambda t: (-t[1], t[2], t[3]))
        pick = ranked[:top_k]
        y_vals = [float(t[1]) for t in pick]
        # pick is sorted by cosine descending ⇒ y_vals[0] >= y_vals[1] >= ....
        # Stack text with fixed steps so 1st is topmost and each label sits
        # above its anchor: ty[j+1] <= ty[j] - min_gap.
        k = len(pick)
        ty = [
            y_vals[j] + min_above_pt + (k - 1 - j) * min_gap
            for j in range(k)
        ]
        ty = [
            float(np.clip(t, ymin + 0.02 * y_span, ymax - 0.02 * y_span))
            for t in ty
        ]

        text_x = float(layer_i) - dx_left
        text_x = max(ax.get_xlim()[0] + 0.5, text_x)

        for j, (idx, y_val, s_low, s_high) in enumerate(pick):
            ord_lbl = ordinals[j] if j < len(ordinals) else f"{j + 1}th"
            lbl = (
                f"{ord_lbl} {token_axis_label(s_low)} ↔ "
                f"{token_axis_label(s_high)}"
            )
            rgb = tuple(float(x) for x in colors[idx][:3])
            ap_rgba = rgb + (0.85,)

            ax.annotate(
                lbl,
                xy=(layer_i, y_val),
                xytext=(text_x, ty[j]),
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ap_rgba,
                    lw=1.05,
                    shrinkA=2,
                    shrinkB=3,
                    mutation_scale=9,
                    connectionstyle="arc3,rad=0",
                    # Tail at right centre of label bbox (default relpos is 0.5,0.5).
                    relpos=(1.0, 0.5),
                ),
                ha="right",
                va="center",
                fontsize=ann_fs,
                fontweight="semibold",
                color=tuple(colors[idx]),
                zorder=6,
            )


def _plot_pairwise_cosine_by_layer(cos_between_layers: np.ndarray, L: int,
                                   output_dir: Path, spec_line: str,
                                   inputs: list[InputSpec] | None = None
                                   ) -> Path:
    """One line per unordered slot pair (s1, s2); mean cos over role pairs."""
    S = cos_between_layers.shape[-1]
    slot_pairs = list(itertools.combinations(range(S), 2))
    n_curve = len(slot_pairs)
    # Wide figure + room on the right for a multi-column legend.
    fig_w = max(16.0, 11.0 + 0.22 * float(n_curve))
    fig_h = max(7.0, 5.8 + 0.05 * float(n_curve))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    colors = plt.cm.turbo(np.linspace(0.02, 0.98, max(n_curve, 2)))

    lw = 1.1 if n_curve <= 15 else 0.95
    for idx, (s1, s2) in enumerate(slot_pairs):
        ys = cos_between_layers[:L, s1, s2]
        label = f"{token_axis_label(s1)} ↔ {token_axis_label(s2)}"
        ax.plot(range(L), ys, color=colors[idx], linewidth=lw,
                alpha=0.88, label=label)

    for x in range(0, L, 2):
        ax.axvline(x, color="gray", linewidth=0.3, alpha=0.3)
    for x in range(0, L, 10):
        ax.axvline(x, color="gray", linewidth=0.7, alpha=0.6)
    _annotate_pairwise_top_curves_at_layers(
        ax,
        cos_between_layers,
        slot_pairs=slot_pairs,
        colors=colors,
        L=L,
        layers=(24, 49),
        top_k=3,
        dx_left=2.8,
    )
    for lz in (24, 49):
        if 0 <= lz < L:
            ax.axvline(
                lz, color="#4c1d95", linestyle=":", linewidth=1.85,
                alpha=0.42, zorder=1,
            )
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel(
        "Mean cos of (v_A−v_B) at one token vs (v_A−v_B) at another\n"
        "averaged over unordered role pairs (A, B); raw activations; "
        "pair names in legend (body-mean, t1, …)",
        fontsize=11,
    )
    ax.tick_params(labelsize=12)

    ncol = 3 if n_curve > 20 else (2 if n_curve > 8 else 1)
    leg_fs = max(5, int(11 - n_curve ** 0.45))
    ax.legend(
        fontsize=leg_fs,
        ncol=ncol,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.94,
        handlelength=1.3,
        borderpad=0.35,
        labelspacing=0.35,
        columnspacing=0.9,
    )
    title = f"Mean diff-direction cosine by layer — all {n_curve} token pairs"
    _, top_rect = suptitle_with_specs(
        fig, title, spec_line, line_height=0.030,
    )
    top_rect_body = max(0.74, top_rect - 0.055)
    fig.tight_layout(rect=(0, 0, 0.69, top_rect_body))
    out = output_dir / "pairwise_cosine_by_layer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_dir", type=str,
        default="runpod_workspace/qwen/qwen-3-32b Roger 8slot",
        help="Base data directory (vectors will be loaded from "
             "<data_dir>/<vectors_subdir>/*.pt). The original April 2026 "
             "report figures used 'runpod_workspace/qwen/qwen-3-32b "
             "Christina headers' (4 slots); the current default is the "
             "8-slot Roger data.",
    )
    p.add_argument(
        "--vectors_subdir", type=str, default="roles/vectors",
        help="Sub-path within --data_dir containing entity .pt files. "
             "Use 'traits/vectors' to run on traits instead of roles.",
    )
    p.add_argument(
        "--output_dir", type=str, default="roger/token_position_noise_out",
        help="Where to write the five PNGs. Gitignored under roger/ by "
             "default; pass a different path to keep results elsewhere.",
    )
    p.add_argument(
        "--n_triples", type=int, default=1000,
        help="Number of random (A, B, C) triples sampled from the entity "
             "set (with replacement of triples; entities within a triple "
             "are distinct).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for triple sampling.",
    )
    p.add_argument(
        "--max_layers", type=int, default=None,
        help="Optional cap on the number of layers analysed (for quick "
             "smoke tests). None = all layers.",
    )
    p.add_argument(
        "--no_cache", action="store_true",
        help="Force a fresh recompute even if a matching "
             "<output_dir>/intermediates.pt cache exists.  Without this "
             "flag, a cache hit on (data_dir, vectors_subdir, n_triples, "
             "seed, max_layers) loads the heavy ooo_frac / pairwise "
             "matrices from disk and skips straight to plotting -- "
             "useful for iterating on plot styling without recomputing "
             "the ~10-minute 1000-triple analysis.",
    )
    p.add_argument(
        "--from_cache", action="store_true",
        help="Refuse to recompute: load intermediates from cache or fail.  "
             "Use this when you've changed plotting code only and want "
             "to be sure you're not silently re-running compute.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

CACHE_FILENAME = "intermediates.pt"


def _cache_key(args: argparse.Namespace) -> dict:
    """Cache invalidation key.  Two runs share intermediates iff every
    field below matches.  ``data_dir`` is stored as the absolute path
    so a relative-vs-absolute --data_dir argument doesn't double-miss.
    """
    return {
        "data_dir": str(Path(args.data_dir).resolve()),
        "vectors_subdir": args.vectors_subdir,
        "n_triples": args.n_triples,
        "seed": args.seed,
        "max_layers": args.max_layers,
    }


def _load_cache(cache_path: Path, want_key: dict) -> dict | None:
    """Return cached intermediates if the on-disk key matches; else None."""
    if not cache_path.exists():
        return None
    try:
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as e:  # pylint: disable=broad-except
        print(f"  cache: failed to load {cache_path}: {e}")
        return None
    have_key = blob.get("key", {})
    if have_key != want_key:
        diff = {k: (have_key.get(k), want_key.get(k))
                for k in set(have_key) | set(want_key)
                if have_key.get(k) != want_key.get(k)}
        print(f"  cache: stale (key mismatch on {list(diff.keys())}); "
              f"will recompute.  Diff: {diff}")
        return None
    return blob


def _save_cache(cache_path: Path, key: dict, *,
                ooo_frac: torch.Tensor,
                norm_corr_layers: np.ndarray,
                cos_between_layers: np.ndarray,
                S: int, L: int, N_roles: int) -> None:
    """Write a fresh cache file next to the plot outputs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "key": key,
        "ooo_frac": ooo_frac,
        "norm_corr_layers": norm_corr_layers,
        "cos_between_layers": cos_between_layers,
        "S": int(S),
        "L": int(L),
        "N_roles": int(N_roles),
    }, cache_path)
    print(f"  cache: wrote {cache_path}")


def _compute_intermediates(args: argparse.Namespace) -> dict:
    """Run the heavy stats: load vectors, sample triples, compute
    per-layer metrics + pairwise slot correlations, and reduce to
    ooo_frac + the two layered (S, S) matrices.

    Returns a dict matching the shape of the cache blob (minus the
    "key" field).
    """
    data_dir = Path(args.data_dir)
    vectors_dir = data_dir / args.vectors_subdir

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Step 1: Loading vectors...")
    all_vecs, _role_names = load_all_vectors(vectors_dir)
    N_roles, S, L_full, D = all_vecs.shape
    L = min(L_full, args.max_layers) if args.max_layers else L_full
    print(f"  {N_roles} entities, {S} slots, {L_full} layers (using {L}), "
          f"hidden_dim={D}")

    default_vec = all_vecs.float().mean(dim=0)  # (S, L_full, D)
    default_vec_bf16 = default_vec.to(torch.bfloat16)
    del default_vec

    print(f"Step 2: Sampling {args.n_triples} triples...")
    role_indices = list(range(N_roles))
    triples: List[tuple] = []
    for _ in range(args.n_triples):
        triples.append(tuple(random.sample(role_indices, 3)))

    print("Step 2-3: Computing metrics per layer...")
    all_metrics = torch.zeros(args.n_triples, N_METRICS, S, L)
    norm_corr_layers = np.zeros((L, S, S))
    cos_between_layers = np.zeros((L, S, S))

    for layer in range(L):
        if layer % 8 == 0:
            print(f"  Layer {layer}/{L}...")
        layer_vecs = all_vecs[:, :, layer, :].float()
        default_layer = default_vec_bf16[:, layer, :].float()

        all_metrics[:, :, :, layer] = compute_metrics_for_layer(
            layer_vecs, default_layer, triples,
        )
        norm_corr, cos_between = compute_pairwise_slot_correlation(layer_vecs)
        norm_corr_layers[layer] = norm_corr
        cos_between_layers[layer] = cos_between
        del layer_vecs, default_layer

    print("Normalizing subtraction metrics...")
    normalize_sub_metrics(all_metrics)

    print("Step 3: Odd-one-out scoring...")
    flat = all_metrics.permute(0, 1, 3, 2).reshape(-1, S)
    winners = odd_one_out(flat).reshape(args.n_triples, N_METRICS, L)

    ooo_counts = torch.zeros(N_METRICS, L, S)
    for s in range(S):
        ooo_counts[:, :, s] = (winners == s).float().sum(dim=0)
    ooo_frac = ooo_counts / args.n_triples
    return {
        "ooo_frac": ooo_frac,
        "norm_corr_layers": norm_corr_layers,
        "cos_between_layers": cos_between_layers,
        "S": S,
        "L": L,
        "N_roles": N_roles,
    }


def main() -> int:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / CACHE_FILENAME

    # ---- Load from cache or recompute ------------------------------
    key = _cache_key(args)
    cached = None if args.no_cache else _load_cache(cache_path, key)
    if cached is not None:
        print(f"Cache hit: loading intermediates from {cache_path}")
        ooo_frac = cached["ooo_frac"]
        norm_corr_layers = cached["norm_corr_layers"]
        cos_between_layers = cached["cos_between_layers"]
        S = cached["S"]
        L = cached["L"]
        N_roles = cached["N_roles"]
    else:
        if args.from_cache:
            raise SystemExit(
                f"--from_cache requested but no usable cache at {cache_path}; "
                f"either drop --from_cache to recompute or run once "
                f"without --from_cache to populate the cache."
            )
        intermediates = _compute_intermediates(args)
        ooo_frac = intermediates["ooo_frac"]
        norm_corr_layers = intermediates["norm_corr_layers"]
        cos_between_layers = intermediates["cos_between_layers"]
        S = intermediates["S"]
        L = intermediates["L"]
        N_roles = intermediates["N_roles"]
        _save_cache(
            cache_path, key,
            ooo_frac=ooo_frac,
            norm_corr_layers=norm_corr_layers,
            cos_between_layers=cos_between_layers,
            S=S, L=L, N_roles=N_roles,
        )

    # Console summary.
    print("\n" + "=" * 80)
    print("RESULTS: Odd-one-out fraction (averaged across layers)")
    print("=" * 80)
    print(f"{'metric':<20s}", end="")
    for sn in token_axis_labels(S):
        print(f"{sn:>15s}", end="")
    print(f"{'  most odd':>12s}")
    print("-" * 80)
    for m in range(N_METRICS):
        avg = ooo_frac[m].mean(dim=0)
        print(f"{METRIC_NAMES[m]:<20s}", end="")
        for s in range(S):
            print(f"{avg[s].item():>15.3f}", end="")
        print(f"{'  ' + token_axis_label(int(avg.argmax().item())):>12s}")
    overall = ooo_frac.mean(dim=0).mean(dim=0)
    print("-" * 80)
    print(f"{'OVERALL':<20s}", end="")
    for s in range(S):
        print(f"{overall[s].item():>15.3f}", end="")
    print(f"{'  ' + token_axis_label(int(overall.argmax().item())):>12s}")

    print("\n" + "=" * 80)
    print("PAIRWISE TOKEN NORM CORRELATION (averaged across layers)")
    print("=" * 80)
    avg_norm_corr = norm_corr_layers.mean(axis=0)
    for s1 in range(S):
        print(f"  {token_axis_label(s1):>15s}", end="")
        for s2 in range(S):
            print(f"  {avg_norm_corr[s1, s2]:>8.4f}", end="")
        print()
    print("\n  Mean off-diagonal correlation per token:")
    for s in range(S):
        offdiag = [avg_norm_corr[s, s2] for s2 in range(S) if s2 != s]
        print(f"    {token_axis_label(s):>15s}: {np.mean(offdiag):.4f}")

    print("\n" + "=" * 80)
    print("PAIRWISE TOKEN DIFF-DIRECTION AGREEMENT (avg cosine, all layers)")
    print("=" * 80)
    avg_cos_between = cos_between_layers.mean(axis=0)
    for s1 in range(S):
        print(f"  {token_axis_label(s1):>15s}", end="")
        for s2 in range(S):
            print(f"  {avg_cos_between[s1, s2]:>8.4f}", end="")
        print()
    print("\n  Mean off-diagonal agreement per token:")
    for s in range(S):
        offdiag = [avg_cos_between[s, s2] for s2 in range(S) if s2 != s]
        print(f"    {token_axis_label(s):>15s}: {np.mean(offdiag):.4f}")

    print(f"\nSaving plots to {output_dir}/...")
    spec_line = (f"{N_roles} {Path(args.vectors_subdir).parts[0]} "
                 f"× {S} token positions × {L} layers; "
                 f"{args.n_triples} random triples; seed={args.seed}\n"
                 f"data: {data_dir.name}")

    inputs: list[InputSpec] = [
        current_data_subtree_input(
            data_dir, args.vectors_subdir, dep_key="vectors_subtree",
            extras={"n_triples": str(args.n_triples),
                    "seed": str(args.seed)}),
    ]
    paths = [
        _plot_ooo_heatmaps(ooo_frac, output_dir, spec_line, inputs=inputs),
        _plot_ooo_summary_bars(ooo_frac, output_dir, spec_line, inputs=inputs),
        _plot_ooo_by_layer(ooo_frac, L, output_dir, spec_line, inputs=inputs),
        _plot_pairwise_matrices(norm_corr_layers, cos_between_layers,
                                output_dir, spec_line, inputs=inputs),
        _plot_pairwise_cosine_by_layer(cos_between_layers, L, output_dir,
                                       spec_line, inputs=inputs),
    ]
    for p in paths:
        print(f"  wrote {p}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
