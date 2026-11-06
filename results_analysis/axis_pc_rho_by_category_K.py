#!/usr/bin/env python3
"""Histogram of average observed ρ under seven K-assignment strategies.

For each axis at the configured (slot, layer), classify it by the
threshold rule on the post-whitening cosine sequence
``x_k = |cos(axis_w_{K=k-1}, PC_k)|`` (same rule as
:func:`results_analysis.axis_pc_alignment_vs_peak_K._categorize_row`):

    * red    -- x₁ > 0.33 ∧ x₂ < x₁           (PC1-dominant after K=0)
    * yellow -- x₂ > 0.25 ∧ x₃ < x₂           (PC2-dominant after K=1)
    * green  -- x₃ > 0.20 ∧ x₄ < x₃           (PC3-dominant after K=2)
    * blue   -- else                          (PC4+ dominant or weak)
    * gray   -- post-whitening cos unavailable

Then look up the observed ρ for that axis at K ∈ {0, 1, 2, 3} (blended
0.80 responses + 0.20 desc+instr for primary-cohort axes; desc+instr
alone for the desc+instr-only "extra" cohort).

Plot one bar per K-assignment policy:

    1) every axis at K=0
    2) every axis at K=1
    3) every axis at K=2
    4) every axis at K=3
    5) red → K=0, else K=1
    6) red → K=0, yellow → K=1, else K=2
    7) red → K=0, yellow → K=1, green → K=2, else K=3

Each bar is a *stacked* bar broken into the per-category
contributions ``(category_sum_of_ρ) / N_total``; the total height is
the cohort-mean ρ under that policy.  This shows both which policy
wins *and* which category drove the win.

Defaults to the full ~35-axis cohort (``pair_list_di.json``); pass
``--primary_only`` to restrict to the 12 axes with responses scoring
(consistent blend definition across rows).

Outputs
-------

``axis_pc_rho_by_category_K_<cohort>_slot{N}.png``.

Examples
--------

::

    uv run python results_analysis/axis_pc_rho_by_category_K.py

    # Primary cohort only (consistent 0.80/0.20 blend across rows):
    uv run python results_analysis/axis_pc_rho_by_category_K.py \\
        --primary_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow `uv run python results_analysis/axis_pc_rho_by_category_K.py`
# from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    _CATEGORY_CMAPS,
    _CATEGORY_ORDER,
    _categorize_row,
    _index_fits,
    _index_sweep,
    _per_axis_record,
)


# Bars 1-4: every axis evaluated at this single K.
_FIXED_K_POLICIES: list[tuple[str, int]] = [
    ("All @ K=0", 0),
    ("All @ K=1", 1),
    ("All @ K=2", 2),
    ("All @ K=3", 3),
]

# Bars 5-7: K is picked per-category by the cascading rule.  Each
# entry maps category code -> K.  Categories not explicitly named
# fall through to a default in `_policy_contributions`.
_MIXED_POLICIES: list[tuple[str, dict[str, int]]] = [
    ("red→K=0\nelse K=1",
     {"red": 0, "yellow": 1, "green": 1, "blue": 1, "gray": 1}),
    ("red→K=0, yellow→K=1\nelse K=2",
     {"red": 0, "yellow": 1, "green": 2, "blue": 2, "gray": 2}),
    ("red→K=0, yellow→K=1\ngreen→K=2, blue→K=3",
     {"red": 0, "yellow": 1, "green": 2, "blue": 3, "gray": 3}),
]


def _policy_contributions(
    rows: list[dict],
    policy: dict[str, int],
) -> dict[str, float]:
    """Per-category sum of ρ under ``policy`` divided by ``len(rows)``.

    Returned dict has one entry per category in
    :data:`_CATEGORY_ORDER`; summing them gives the cohort-mean ρ
    under that policy.
    """
    n_total = len(rows)
    out = {c: 0.0 for c in _CATEGORY_ORDER}
    if n_total == 0:
        return out
    for r in rows:
        cat, _ = _categorize_row(r)
        K = policy.get(cat, 1)
        rho = r["rho_at_K"].get(K, float("nan"))
        if not np.isfinite(rho):
            continue
        out[cat] += float(rho) / n_total
    return out


def _category_counts(rows: list[dict]) -> dict[str, int]:
    counts = {c: 0 for c in _CATEGORY_ORDER}
    for r in rows:
        cat, _ = _categorize_row(r)
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _print_console_summary(rows: list[dict], label: str) -> None:
    counts = _category_counts(rows)
    n = len(rows)
    parts = ", ".join(f"{c}={counts[c]}" for c in _CATEGORY_ORDER)
    print(f"\n=== {label}: n={n}  ({parts}) ===")
    headers = "  ".join(f"{c:>8s}" for c in _CATEGORY_ORDER)
    print(f"{'policy':45s}  {headers}  {'TOTAL':>8s}")
    print("-" * (47 + 10 * len(_CATEGORY_ORDER) + 10))
    for name, K in _FIXED_K_POLICIES:
        cb = _policy_contributions(rows, {c: K for c in _CATEGORY_ORDER})
        tot = sum(cb.values())
        cells = "  ".join(f"{cb[c]:+8.4f}" for c in _CATEGORY_ORDER)
        print(f"{name:45s}  {cells}  {tot:+8.4f}")
    for name, pol in _MIXED_POLICIES:
        cb = _policy_contributions(rows, pol)
        tot = sum(cb.values())
        cells = "  ".join(f"{cb[c]:+8.4f}" for c in _CATEGORY_ORDER)
        print(f"{name.replace(chr(10), ' '):45s}  {cells}  {tot:+8.4f}")


def _stack_color(cat: str) -> tuple[float, float, float, float]:
    """Mid-saturation shade of the category colormap for the stacked
    bar segment."""
    return _CATEGORY_CMAPS[cat](0.65)


def make_plot(
    rows: list[dict],
    *,
    out_path: Path,
    title_extra: str,
    inputs: list[InputSpec],
    primary_only: bool,
) -> None:
    """Render the 7-policy stacked-bar histogram."""
    policies: list[tuple[str, dict[str, int]]] = []
    for name, K in _FIXED_K_POLICIES:
        policies.append((name, {c: K for c in _CATEGORY_ORDER}))
    for name, pol in _MIXED_POLICIES:
        policies.append((name, pol))

    fig, ax = plt.subplots(figsize=(12.5, 6.4), constrained_layout=True)

    x = np.arange(len(policies), dtype=float)
    bar_width = 0.65

    # Stack categories bottom-to-top: red (largest contribution)
    # first, then yellow, then green, then blue, then gray on top.
    # Matches the threshold-rule order used elsewhere.
    cat_order = _CATEGORY_ORDER
    bottoms = np.zeros(len(policies), dtype=float)
    totals = np.zeros(len(policies), dtype=float)
    bar_handles: dict[str, object] = {}
    for cat in cat_order:
        heights = np.array(
            [_policy_contributions(rows, pol)[cat] for _, pol in policies],
            dtype=float,
        )
        if np.all(heights == 0.0):
            continue
        rect = ax.bar(
            x, heights, bottom=bottoms, width=bar_width,
            color=_stack_color(cat),
            edgecolor="black", linewidth=0.5,
            label=f"{cat} ({_category_counts(rows)[cat]} axes)",
        )
        bar_handles[cat] = rect
        bottoms += heights
        totals += heights

    # Total-label on top of each bar; bold + black for the overall winner.
    best_idx = int(np.argmax(totals))
    for xi, (tot_xi, tot) in enumerate(zip(x, totals)):
        label_weight = "bold" if xi == best_idx else "normal"
        ax.text(tot_xi, tot + 0.005, f"{tot:+.3f}",
                ha="center", va="bottom", fontsize=10,
                fontweight=label_weight)

    # "Winning sections": for EACH category, find which policy bars
    # use that category's best-performing K (i.e. give the largest
    # contribution for that category) and bold those segments' outline.
    # Ties bold all winners.
    for cat in cat_order:
        if cat not in bar_handles:
            continue
        cat_heights = np.array(
            [_policy_contributions(rows, pol)[cat] for _, pol in policies],
            dtype=float,
        )
        cat_max = float(cat_heights.max())
        if cat_max <= 0.0:
            continue
        for i, patch in enumerate(bar_handles[cat]):
            if cat_heights[i] >= cat_max - 1e-9:
                patch.set_edgecolor("black")
                patch.set_linewidth(2.0)

    # Also outline the overall-winning bar's *total* with a thicker
    # gray frame around all its segments, so the eye can find the
    # highest-totalling policy at a glance.
    for cat in cat_order:
        if cat not in bar_handles:
            continue
        patch = bar_handles[cat][best_idx]
        if patch.get_linewidth() < 1.5:
            patch.set_edgecolor("#444444")
            patch.set_linewidth(1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in policies], fontsize=8)
    ax.set_ylabel("Average ρ (stacked by post-whitening cos category)",
                  fontsize=11)
    ax.set_ylim(0.0, max(totals.max() * 1.12, 0.05))
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.0, color="black", linewidth=0.5)

    ax.legend(loc="lower right", fontsize=9, framealpha=0.85,
              title="Category (stack segment)")

    cohort_label = "primary cohort (responses-blended ρ)" if primary_only \
        else "full cohort (mixed: blended ρ where available, " \
             "desc+instr alone otherwise)"
    title_line = "Average ρ vs whitening-K policy"
    spec_line = (
        f"{cohort_label}.  "
        f"Stacked bars show each category's contribution "
        f"(category-sum-of-ρ ÷ n).  "
        f"Bold black outlines = per-category K-winner; "
        f"bold total label = overall winner.  "
        f"{title_extra}."
    )
    suptitle_with_specs(fig, title_line, spec_line)

    fig.savefig(out_path, dpi=140, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)
    print(f"\nWrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR),
                   help=f"Directory holding pair-list + peak-fit JSONs and "
                        f"output PNG (default: {DEFAULT_EXPERIMENT_DIR}).")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                   help=f"Vector tree (default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--pairs", default=DEFAULT_PAIR_LIST,
                   help=f"Pair-list JSON filename for the FULL cohort "
                        f"(default: {DEFAULT_PAIR_LIST}).")
    p.add_argument("--pairs_primary", default=DEFAULT_PRIMARY_PAIR_LIST,
                   help=f"Pair-list JSON filename for the PRIMARY cohort "
                        f"(default: {DEFAULT_PRIMARY_PAIR_LIST}).")
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT,
                   help=f"Token slot (default: {DEFAULT_SLOT}).")
    p.add_argument("--layer", type=int, default=LAYER,
                   help=f"Hidden-state layer (default: {LAYER}).")
    p.add_argument("--primary_only", action="store_true",
                   help="Restrict the histogram to the primary cohort "
                        "(axes with responses scoring; consistent "
                        "blended-ρ definition across all rows).")
    p.add_argument("--out", default=None,
                   help="Output PNG filename (default: "
                        "axis_pc_rho_by_category_K_<cohort>_slot{N}.png).")
    p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
                   help="How to react to drift in the peak-fit JSON's "
                        "recorded provenance (default: warn).")
    args = p.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"--data_dir does not exist: {data_dir}")

    cohort = cohort_from_pairs(args.pairs)
    fit_filename = f"whitening_k_peak_fit_{cohort}_slot{args.slot}.json"
    sweep_filename = f"whitening_k_sweep_{cohort}_slot{args.slot}.json"
    suffix = "_primary" if args.primary_only else ""
    out_filename = args.out or (
        f"axis_pc_rho_by_category_K_{cohort}_slot{args.slot}{suffix}.png"
    )

    inputs: list[InputSpec] = []

    pairs, _, _ = load_and_register(
        experiment_dir / args.pairs, dep_key="pairs_json",
        inputs=inputs, policy=args.cache_policy,
    )
    pairs_primary, _, _ = load_and_register(
        experiment_dir / args.pairs_primary, dep_key="pairs_primary_json",
        inputs=inputs, policy=args.cache_policy,
    )
    fits, _, _ = load_and_register(
        experiment_dir / fit_filename, dep_key="peak_fit_json",
        inputs=inputs, policy=args.cache_policy,
    )
    sweep, _, _ = load_and_register(
        experiment_dir / sweep_filename, dep_key="k_sweep_json",
        inputs=inputs, policy=args.cache_policy,
    )
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
    rows_all: list[dict] = []
    for pair in pairs:
        rows_all.append(_per_axis_record(
            pair,
            fits_by_pair=fits_by_pair, sweep_by_pair=sweep_by_pair,
            primary_set=primary_set,
            data_dir=data_dir, slot=args.slot, layer=args.layer,
            n_pcs=N_PCS, bottom_panel_Ks=bottom_panel_Ks,
        ))

    rows_primary = [r for r in rows_all if r["primary"]]
    rows = rows_primary if args.primary_only else rows_all

    n_primary = len(rows_primary)
    n_extra = len(rows_all) - n_primary
    print(f"Loaded {len(rows_all)} axes total "
          f"({n_primary} primary + {n_extra} desc+instr-only extras)")

    _print_console_summary(rows_all, "ALL axes (full cohort)")
    _print_console_summary(rows_primary, "PRIMARY cohort only")

    title_extra = (
        f"slot {args.slot}, layer {args.layer}, "
        f"n={len(rows)} axes"
    )
    out_path = experiment_dir / out_filename
    make_plot(rows, out_path=out_path, title_extra=title_extra,
              inputs=inputs, primary_only=args.primary_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
