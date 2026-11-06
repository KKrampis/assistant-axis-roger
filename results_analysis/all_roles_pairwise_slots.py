#!/usr/bin/env python3
"""Per-role pairwise slot statistics across layers, as transparent ribbons.

Promoted from a one-off chat-inline producer (April 2026 work) that was
used in a fortnightly report to visualise how each entity's four token
positions relate to one another. Companion to
``token_position_noise_analysis.py`` -- where that script summarises
across-role *pair-difference* statistics into scalars per
``(slot, slot, layer)``, this one shows the **full per-entity
distribution** as ribbons of transparent lines, one per entity.

For each entity ``r`` (default-centred to ``δ = r - default``), each
pair of slots ``(s1, s2)``, and each layer ``L``:

- **Top panel**: ``cos(δ[s1, L], δ[s2, L])`` -- do the two slots agree
  on the *direction* this entity differs from baseline?
- **Bottom panel** (log y): ``‖δ[s2, L]‖ / ‖δ[s1, L]‖`` -- how does
  the *magnitude* of the deviation differ between slots?

Eighteen slot-pairs across three visual groups (plus the ``D`` frame
that shows all at once), encoded by colormap so group membership is
obvious:

- ``GROUP_BODY_HEADER`` -- 7 pairs ``(0, k)`` for ``k ∈ {1..7}``.
  body-mean vs each header slot.  **plasma**.
- ``GROUP_HEADER_TO_SLOT6`` -- 5 pairs ``(k, 6)`` for ``k ∈ {1..5}``.
  each of header slots 1–5 vs slot 6 (``</think>``).
  **winter**.
- ``GROUP_HEADER_TO_SLOT7`` -- 6 pairs ``(k, 7)`` for ``k ∈ {1..6}``.
  each of header slots 1–6 vs slot 7 (post-``\n\n`` before response).
  **viridis** (distinct from plasma + winter).

The full 8-choose-2 = 28-pair set is still too dense; this 18-pair
subset keeps the legend readable.

Four frame variants (default: emit all four side-by-side):

- ``A`` -- body-vs-header pairs at full alpha; other groups dimmed.
- ``B`` -- header 1–5 vs slot 6 at full alpha; others dimmed.
- ``C`` -- header 1–6 vs slot 7 at full alpha; others dimmed.
- ``D`` -- all 18 pairs at equal alpha (what used to be frame ``C``).

The legend lists all 18 pairs in every frame (non-active dimmed) so
stacking frames in a deck does not shift the legend.

Whitening (``--whitening``)
---------------------------

- ``raw`` (default): operate on raw activation differences.
- ``soft_K=N``: per-(slot, layer) soft-K whitening fitted on the
  augmented production pool (all standalone roles + traits + the corpus
  default). **No leave-one-out** -- the pool is fitted once per
  ``(slot, layer)`` and used for every entity. Per-entity LOO would
  require ~280 extra SVDs per layer, swamping the actual plot work; for
  pool size ~580 the bias from including the entity itself in its own
  whitening basis is ~1/580 -- negligible. The whitener is applied to
  ``δ = r - default`` *before* computing cosines and norms, so both
  panels show the whitened metric.

Usage
-----

::

    # Default: all four frames (A–D), raw, all 280 roles on Roger 8slot data.
    uv run python results_analysis/all_roles_pairwise_slots.py

    # Whitened version (project soft-K default), written next to the raw frames.
    uv run python results_analysis/all_roles_pairwise_slots.py \\
      --whitening soft_K=2

    # Run on traits instead.
    uv run python results_analysis/all_roles_pairwise_slots.py \\
      --vectors_subdir traits/vectors

    # Just the D frame (all pairs, no slide-deck stacking) for a single PNG.
    uv run python results_analysis/all_roles_pairwise_slots.py --variant D
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from assistant_axis import png_metadata, suptitle_with_specs  # noqa: E402
from assistant_axis.provenance import (  # noqa: E402
    InputSpec,
    current_data_subtree_input,
)
from results_analysis.axis_judge_correlation import _load_vector_file  # noqa: E402
from results_analysis.canonical_angles.data import (  # noqa: E402
    build_augmented_whitening_pool,
)
from results_analysis.canonical_angles.whitening import (  # noqa: E402
    fit_whitening,
    parse_whitening_spec,
)


# Qwen-3 chat-template slot layout used in the 8-slot data:
#   0  body-mean  -- aggregate over all body (user) tokens
#   1  <|im_start|>      \
#   2  assistant          |
#   3  \n                 |
#   4  <think>            |- 7 header tokens (the "non-thinking" header
#   5  \n\n  (in <think>) |  in Qwen-3 still includes empty <think>...</think>
#   6  </think>           |  scaffolding; slot 5 is between <think>/</think>,
#   7  \n\n  (post)      /   slot 7 is after </think> before the response).
#
# Slots 5 and 7 are both ``\n\n`` text; we disambiguate them in the
# legend by suffixing "(in)" / "(post)" so the plot is readable.
SLOT_NAMES = [
    "body-mean", "<|im_start|>", "assistant", r"\n",
    "<think>", r"\n\n (in)", "</think>", r"\n\n (post)",
]

# 8-choose-2 = 28 pairs is too dense to read; use three focused groups:
#   - GROUP_BODY_HEADER:      (0, k) for k in 1..7
#   - GROUP_HEADER_TO_SLOT6:  (k, 6) for k in 1..5
#   - GROUP_HEADER_TO_SLOT7:  (k, 7) for k in 1..6
# Total = 18 pairs.
GROUP_BODY_HEADER = [(0, k) for k in range(1, 8)]
GROUP_HEADER_TO_SLOT6 = [(k, 6) for k in range(1, 6)]
GROUP_HEADER_TO_SLOT7 = [(k, 7) for k in range(1, 7)]
ALL_PAIRS = (
    GROUP_BODY_HEADER + GROUP_HEADER_TO_SLOT6 + GROUP_HEADER_TO_SLOT7
)

# Three colormaps so the three groups stay visually separable at
# α=0.10 ribbon transparency (plasma / winter / viridis).
_w_body = plt.cm.plasma(np.linspace(0.05, 0.90, len(GROUP_BODY_HEADER)))
_w_to6 = plt.cm.winter(np.linspace(0.00, 1.00, len(GROUP_HEADER_TO_SLOT6)))
_w_to7 = plt.cm.viridis(np.linspace(0.05, 0.95, len(GROUP_HEADER_TO_SLOT7)))
PAIR_COLORS = {
    **{p: tuple(c) for p, c in zip(GROUP_BODY_HEADER, _w_body)},
    **{p: tuple(c) for p, c in zip(GROUP_HEADER_TO_SLOT6, _w_to6)},
    **{p: tuple(c) for p, c in zip(GROUP_HEADER_TO_SLOT7, _w_to7)},
}
PAIR_LABELS = {p: f"{SLOT_NAMES[p[0]]} vs {SLOT_NAMES[p[1]]}"
               for p in PAIR_COLORS}

VARIANTS = {
    "A": (GROUP_BODY_HEADER, "body-mean vs each header slot (7 pairs)"),
    "B": (GROUP_HEADER_TO_SLOT6,
          "header slots 1–5 vs slot 6 (</think>) (5 pairs)"),
    "C": (GROUP_HEADER_TO_SLOT7,
          "header slots 1–6 vs slot 7 (6 pairs)"),
    "D": (ALL_PAIRS,
          "all 18 pairs (body + vs-6 + vs-7)"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_entity_vectors(vectors_dir: Path
                        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Load all entity vectors plus the ``default.pt`` baseline.

    Returns ``(vecs, default)`` where ``vecs`` maps name → (S, L, D)
    float32 tensor and ``default`` is the (S, L, D) baseline. Skips
    files that fail to load with a warning.
    """
    files = sorted(vectors_dir.glob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files in {vectors_dir}")
    vecs: Dict[str, torch.Tensor] = {}
    default: torch.Tensor | None = None
    for f in files:
        try:
            v = _load_vector_file(f).float()
        except Exception as e:  # pylint: disable=broad-except
            print(f"  skip {f.name}: {e}")
            continue
        if f.stem == "default":
            default = v
        else:
            vecs[f.stem] = v
    if default is None:
        raise SystemExit(f"No default.pt in {vectors_dir}")
    return vecs, default


def _path_for(data_dir: Path, etype: str, name: str) -> Path:
    if etype == "combinations":
        return data_dir / "combinations" / "vectors" / f"{name}.pt"
    return data_dir / etype / "vectors" / f"{name}.pt"


def load_whitening_pool_tensor(data_dir: Path) -> torch.Tensor:
    """Load the full augmented pool (all roles + traits + corpus default)
    as a single ``(N_pool, S, L, D)`` float32 tensor.

    No leave-one-out: per the script's design, the pool is fit once per
    ``(slot, layer)`` and reused for all entities. See module docstring.
    """
    entries = build_augmented_whitening_pool(data_dir, leave_out=set(),
                                             scope="roles+traits")
    rows = []
    for et, n in entries:
        try:
            rows.append(_load_vector_file(_path_for(data_dir, et, n)).float())
        except Exception as e:  # pylint: disable=broad-except
            print(f"  skip pool entry ({et}, {n}): {e}")
    pool = torch.stack(rows, dim=0)
    print(f"  whitening pool: {pool.shape[0]} entries, "
          f"shape {tuple(pool.shape)}")
    return pool


# ---------------------------------------------------------------------------
# Pair stats
# ---------------------------------------------------------------------------

def compute_pair_stats(
    vecs: Dict[str, torch.Tensor],
    default: torch.Tensor,
    *,
    whitening: str = "raw",
    K: int | None = None,
    pool: torch.Tensor | None = None,
    max_layers: int | None = None,
) -> Tuple[Dict[Tuple[int, int], List[np.ndarray]],
           Dict[Tuple[int, int], List[np.ndarray]]]:
    """Compute ``cos_data[pair][role_idx]`` and ``ratio_data[pair][role_idx]``,
    each of length ``L`` (layer count).

    For ``whitening='soft_K'`` we fit a separate whitener per
    ``(slot, layer)`` from ``pool``, then apply it to ``(r - default)``
    at that ``(slot, layer)`` before computing cosine and norm ratio.
    """
    names = list(vecs.keys())
    sample = next(iter(vecs.values()))
    S, L_full, _ = sample.shape
    L = min(L_full, max_layers) if max_layers else L_full

    cos_data: Dict[Tuple[int, int], List[np.ndarray]] = {p: [] for p in ALL_PAIRS}
    ratio_data: Dict[Tuple[int, int], List[np.ndarray]] = {p: [] for p in ALL_PAIRS}

    # Pre-fit whiteners per (slot, layer) when whitening is on.
    bases = None
    if whitening == "soft_K":
        if pool is None:
            raise ValueError("pool required for soft_K whitening")
        bases = [[None for _ in range(L)] for _ in range(S)]
        print("  Fitting per-(slot, layer) whiteners...")
        for s in range(S):
            for L_idx in range(L):
                pool_sl = pool[:, s, L_idx, :].numpy()
                bases[s][L_idx] = fit_whitening("soft_K", pool_sl, K=K)
            print(f"    slot {s}: fit {L} whiteners")

    # Stack δ tensors per layer for vectorised norm/cosine.
    print(f"  Computing pair stats for {len(names)} entities × {L} layers...")
    for r in names:
        diff = vecs[r] - default  # (S, L_full, D)
        # Collect whitened δ per slot per layer (or just use raw).
        if bases is None:
            for p in ALL_PAIRS:
                a = diff[p[0], :L]                  # (L, D)
                b = diff[p[1], :L]
                an = a.norm(dim=-1).numpy()
                bn = b.norm(dim=-1).numpy()
                cos = torch.nn.functional.cosine_similarity(
                    a, b, dim=-1).numpy()
                cos_data[p].append(cos)
                ratio_data[p].append(bn / (an + 1e-8))
        else:
            # Apply per-(slot, layer) whitener to δ at that (slot, layer).
            # Resulting whitened tensor: (S, L, D)
            w_diff = np.zeros((S, L, diff.shape[-1]), dtype=np.float32)
            for s in range(S):
                for L_idx in range(L):
                    w_diff[s, L_idx] = bases[s][L_idx].apply(
                        diff[s, L_idx].numpy()[None, :])[0]
            for p in ALL_PAIRS:
                a = w_diff[p[0]]                    # (L, D)
                b = w_diff[p[1]]
                an = np.linalg.norm(a, axis=-1)
                bn = np.linalg.norm(b, axis=-1)
                # Cosine via dot products.
                num = (a * b).sum(axis=-1)
                cos = num / (an * bn + 1e-8)
                cos_data[p].append(cos)
                ratio_data[p].append(bn / (an + 1e-8))
    return cos_data, ratio_data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_frame(
    cos_data, ratio_data, *,
    variant: str,
    n_entities: int,
    L: int,
    entity_kind: str,
    whitening_label: str,
    spec_extra: str,
    output_path: Path,
    inputs: list[InputSpec] | None = None,
) -> Path:
    show_pairs, _ = VARIANTS[variant]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    layers = np.arange(L)

    for p in show_pairs:
        color = PAIR_COLORS[p]
        for cos_line in cos_data[p]:
            ax1.plot(layers, cos_line, alpha=0.10, linewidth=0.5, color=color)
        for ratio_line in ratio_data[p]:
            ax2.plot(layers, ratio_line, alpha=0.10, linewidth=1.0, color=color)

    # Every pair in ALL_PAIRS in the legend; non-active dimmed (same in
    # every frame so the legend doesn't shift when frames are stacked as a
    # slide-deck click-to-appear animation).
    for p in ALL_PAIRS:
        a_leg = 1.0 if p in show_pairs else 0.25
        ax1.plot([], [], color=PAIR_COLORS[p], linewidth=2, alpha=a_leg,
                 label=PAIR_LABELS[p])
        ax2.plot([], [], color=PAIR_COLORS[p], linewidth=2, alpha=a_leg,
                 label=PAIR_LABELS[p])

    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for ax in (ax1, ax2):
        ax.set_xlim(0, L - 1)
        ax.set_xticks(range(0, L, 10))
        ax.grid(True, which="major", alpha=0.4, linewidth=0.8)
        ax.set_xticks(range(0, L, 2), minor=True)
        ax.grid(True, which="minor", axis="x", alpha=0.25, linewidth=0.5)

    ax1.set_ylabel("Cosine Similarity", fontsize=13)
    ax1.set_ylim(-0.8, 1.0)
    ax1.set_title(f"All {n_entities} {entity_kind} (entity − default) "
                  f"— Pairwise Slot Cosine by Layer", fontsize=12)
    ax1.legend(loc="upper left", fontsize=8)

    ax2.set_yscale("log")
    ax2.set_ylim(0.1, 20)
    s10 = 10 ** 0.5
    ratio_ticks = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, s10, 5.0, 7.0, 10.0]
    ratio_tick_labels = ["0.1", "0.2", "0.3", "0.5", "0.7", "1.0", "1.5",
                         "2.0", "√10", "5.0", "7.0", "10.0"]
    ax2.set_yticks(ratio_ticks)
    ax2.set_yticklabels(ratio_tick_labels)
    ax2.yaxis.set_minor_locator(mticker.NullLocator())
    ax2.set_ylabel("Norm Ratio (slot B / slot A, log)", fontsize=13)
    ax2.set_xlabel("Layer", fontsize=13)
    ax2.set_title(f"All {n_entities} {entity_kind} (entity − default) "
                  f"— Pairwise Slot Norm Ratio by Layer", fontsize=12)
    ax2.legend(loc="upper left", fontsize=8)

    suptitle = f"Pairwise slot statistics across all {n_entities} {entity_kind}"
    spec_line = (f"frame {variant}: {VARIANTS[variant][1]}; whitening: "
                 f"{whitening_label}{spec_extra}")
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
    p.add_argument("--data_dir", type=str,
                   default="runpod_workspace/qwen/qwen-3-32b Roger 8slot",
                   help="Base data directory (vectors loaded from "
                        "<data_dir>/<vectors_subdir>/*.pt; whitening pool "
                        "loaded from across roles+traits+default).  Default "
                        "points at the 8-slot data; pass the older "
                        "'... Roger' (without 8slot) to reproduce the "
                        "original 4-slot 6-pair plot, but the SLOT_NAMES "
                        "list assumes 8 slots so anything ≤4 will be "
                        "labelled with names that don't match.")
    p.add_argument("--vectors_subdir", type=str, default="roles/vectors",
                   help="Sub-path within --data_dir for the entity .pt "
                        "files. Use 'traits/vectors' for the trait side.")
    p.add_argument("--output_dir", type=str,
                   default="roger/all_roles_pairwise_slots_out",
                   help="Where to write the PNG(s).")
    p.add_argument("--whitening", type=str, default="raw",
                   help="Whitening spec: 'raw' or 'soft_K=N' (e.g. "
                        "'soft_K=2', the project default). Whitening is "
                        "applied to (entity - default) per-(slot, layer) "
                        "before stats are computed; pool = full augmented "
                        "pool, no LOO.")
    p.add_argument("--variant", type=str, default="all",
                   choices=["A", "B", "C", "D", "all"],
                   help="Frame variant. 'all' (default) emits A–D side by "
                        "side -- intended as click-to-appear stacked layers "
                        "in a slide.")
    p.add_argument("--max_layers", type=int, default=None,
                   help="Optional cap on the number of layers (for fast "
                        "smoke tests). None = all layers.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    data_dir = Path(args.data_dir)
    vectors_dir = data_dir / args.vectors_subdir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method, K = parse_whitening_spec(args.whitening)
    if method not in ("raw", "soft_K"):
        raise SystemExit(
            f"--whitening must be 'raw' or 'soft_K=N' (got {args.whitening!r})")

    print("Step 1: Loading entity vectors...")
    vecs, default = load_entity_vectors(vectors_dir)
    n_entities = len(vecs)
    sample = next(iter(vecs.values()))
    S, L_full, _ = sample.shape
    L = min(L_full, args.max_layers) if args.max_layers else L_full
    entity_kind = Path(args.vectors_subdir).parts[0]
    print(f"  Loaded {n_entities} {entity_kind}, "
          f"shape (S={S}, L={L_full}, D={sample.shape[-1]}), using L={L}")

    pool: torch.Tensor | None = None
    if method == "soft_K":
        print(f"Step 2: Loading whitening pool for soft_K={K}...")
        pool = load_whitening_pool_tensor(data_dir)
        whitening_label = f"soft_K={K} (pool = roles+traits+default; no LOO)"
    else:
        whitening_label = "none (raw)"

    print("Step 3: Computing per-entity pair stats...")
    cos_data, ratio_data = compute_pair_stats(
        vecs, default,
        whitening=method, K=K, pool=pool, max_layers=args.max_layers,
    )

    if args.variant == "all":
        variants = ["A", "B", "C", "D"]
    else:
        variants = [args.variant]

    print(f"Step 4: Writing {len(variants)} frame(s) to {output_dir}/...")
    suffix = ""
    if method == "soft_K":
        suffix = f"_K={K}"

    spec_extra = f"; data: {data_dir.name}"
    # Provenance inputs.  When whitening is on, the soft_K basis is fit
    # on the augmented pool (roles+traits+default), so both subtrees are
    # real dependencies.  When raw, only the entity subtree matters.
    extras = {"whitening": args.whitening}
    inputs: list[InputSpec] = []
    if method == "soft_K":
        inputs.append(current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras=extras))
        inputs.append(current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras=extras))
    else:
        inputs.append(current_data_subtree_input(
            data_dir, args.vectors_subdir, dep_key="vectors_subtree",
            extras=extras))
    paths: List[Path] = []
    for v in variants:
        out = output_dir / f"all_{entity_kind}_pairwise_slots{suffix}_{v}.png"
        p = _plot_frame(
            cos_data, ratio_data,
            variant=v,
            n_entities=n_entities,
            L=L,
            entity_kind=entity_kind,
            whitening_label=whitening_label,
            spec_extra=spec_extra,
            output_path=out,
            inputs=inputs,
        )
        paths.append(p)
        print(f"  wrote {p}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
