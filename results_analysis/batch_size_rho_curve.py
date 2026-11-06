#!/usr/bin/env python3
"""GPT responses-mode ρ vs target batch size, swept across multiple
(slot, layer) configs and axes.

For each cell ``(axis, batch_size, slot, layer)`` the tool:

1. Loads the per-entity mean judge score from
   ``<experiment_dir>/<axis>/gpt_responses_{roles,traits}<suffix>/scores_responses.json``
   (suffix = ``""`` for B=15, ``"_b<N>"`` otherwise).
2. Loads entity vectors at ``(slot, layer)`` from ``--data_dir``.
3. Computes the axis direction ``unit(vec[pos] − vec[neg])`` at that
   ``(slot, layer)``.
4. Sweeps ``(L, K)`` with bracket-and-bisect refinement on K, applying
   L-shear then K-soft-K-whitening to both the entity matrix *and* the
   axis direction; takes the maximum Spearman ρ as the cell's score.

Output:

- ``<output_dir>/batch_size_curve_rho.json`` -- the full numerical result
  (per-axis cells + grand mean per batch + cost model + B integers).
- ``<output_dir>/batch_size_curve_rho.png`` -- two-panel figure: grouped
  histogram (3 (slot, layer) groups × N batch bars) + ρ-vs-B line plot
  with one line per (slot, layer) config plus a grand-mean overlay.
- ``<output_dir>/batch_size_cost_vs_rho.png`` -- single-panel cost-vs-ρ
  scatter (linear ρ axis).

Companion plotter
-----------------

``plot_batch_size_quality_vs_cost.py`` reads the JSON above and produces
``<output_dir>/batch_size_cost_vs_quality.png`` -- the same data on a
``1 / (1 − ρ)`` "quality" scale, which makes the diminishing-returns
shape of small-batch responses-mode judging much more visible than a
linear ρ axis (since at ρ ≈ 0.77, ``d(1/(1−ρ))/dρ ≈ 18`` so each
+0.01 ρ ≈ +4 % effective signal).

Library
-------

``compute_batch_size_curve(...) -> dict`` returns the same dict that gets
written to JSON, in case callers want to fold the data into a larger
analysis without re-running the L/K sweep.

Defaults
--------

The CLI defaults match the May 2026 cost-curve pilot:
``--axes`` is the 3 axes that already have GPT responses at all of
B = {5, 7, 10, 15} (``truthful_vs_deceitful``,
``progressive_vs_conservative``, ``improvisational_vs_methodical``);
``--batch_sizes`` is ``5,7,10,15``; ``--configs`` is
``0:26,0:49,6:25,6:49,7:25,7:49`` (the 6-cell set used for cross-cell
ρ averaging; matches the slot 0/6/7 cells in the May-2026
pc_round_trip migration).

CLI
---

::

    # Default: 3 axes × 4 batch sizes × 3 (slot, layer) configs.
    uv run python results_analysis/batch_size_rho_curve.py

    # Single axis, B=10 vs B=15 only:
    uv run python results_analysis/batch_size_rho_curve.py \\
        --axes truthful_vs_deceitful \\
        --batch_sizes 10,15 \\
        --output_dir roger/b_curve_truthful/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from assistant_axis import display_label, entity_id, json_metadata
from assistant_axis.plot_metadata import png_metadata, suptitle_with_specs
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
    load_and_register,
)
from results_analysis.axis_judge_correlation import _load_vector_file
from results_analysis.canonical_angles.data import (
    DEFAULT_DATA_DIR, build_augmented_whitening_pool,
    build_goal_nogoal_subspaces, load_vector,
)
from results_analysis.canonical_angles.whitening import fit_shear, fit_whitening


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_AXES: list[str] = [
    "truthful_vs_deceitful",
    "progressive_vs_conservative",
    "improvisational_vs_methodical",
]
DEFAULT_BATCH_SIZES: list[int] = [5, 7, 10, 15]
DEFAULT_CONFIGS: list[tuple[int, int]] = [
    (0, 26), (0, 49),
    (6, 25), (6, 49),
    (7, 25), (7, 49),
]
DEFAULT_EXPERIMENT_DIR = "roger/axis_judge_experiments"
DEFAULT_OUTPUT_DIR = "roger"

# Per-axis cost in USD (both cohorts combined) for GPT-4.1-mini at the
# given B, computed from the empirical token model in
# results_analysis/README.md "Judging cost model":
#   per-batch input  ≈ H + B*R  (H≈320 header tokens, R≈438 tokens/item)
#   per-batch output ≈ 80 tokens
#   N_items / axis  ≈ 246k (both cohorts), so n_batches = N_items/B
#   in_M  = (n_batches*H + N_items*R) / 1e6
#   out_M = (n_batches*80) / 1e6
# GPT-4.1-mini rates @ $0.40 in / $1.60 out per 1M tokens.
# These supersede the previous (16.50, 12.77, 9.98, 7.81, 6.72, 5.63)
# numbers, which were ~5x too low because they used a header-only
# token model that ignored the per-item response content (the bulk
# of every prompt).  Recomputed 2026-05-09 from a dry-run of 240 real
# B=10 prompts via tiktoken; see plot_batch_size_quality_vs_cost.py
# B10_GPT_INPUT_M_TOK comment for derivation.  The B-curve is much
# flatter than the old one: at B=30 we still pay ~$45/axis because
# item tokens dominate; only header overhead shrinks with B.
COST_PER_AXIS_USD: dict[int, float] = {
    5: 55.69, 7: 52.10, 10: 49.40, 15: 47.30, 20: 46.25, 30: 45.20,
}

# (L, K) sweep params for the per-cell ρ optimisation.
K_COARSE = [0, 1, 2, 4, 8, 16]
L_COARSE = [0, 1, 2, 3, 5, 8, 16]
K_REFINE_MAX_ITER = 4
K_REFINE_MIN_BRACKET = 2
TIE_THRESH = 0.005


# ---------------------------------------------------------------------------
# Score / vector loading
# ---------------------------------------------------------------------------

def _b_suffix(B: int) -> str:
    """B=15 used to be the un-suffixed "default" GPT responses output dir;
    other batch sizes get a `_b<N>` suffix.  Match the existing on-disk
    convention."""
    return "" if B == 15 else f"_b{B}"


def load_scores(
    experiment_dir: Path, axis: str, B: int,
    *,
    scores_filename: str = "scores_responses.json",
    inputs: list[InputSpec] | None = None,
) -> dict[str, float]:
    """Combine per-entity mean_score across roles + traits responses dirs.

    ``scores_filename`` defaults to the canonical cache name; override
    with ``scores_responses__rubric_v1.json`` to read v1 snapshots
    after a rubric version bump.  When an ``inputs`` accumulator is
    supplied, each successfully-read cache is appended via
    ``load_and_register`` so the read and the dependency record stay
    in lockstep (see AGENT_NOTES.md "Reader+registrar pattern").
    """
    suffix = _b_suffix(B)
    out: dict[str, float] = {}
    for side in ("roles", "traits"):
        path = (experiment_dir / axis
                / f"gpt_responses_{side}{suffix}" / scores_filename)
        if not path.exists():
            continue
        scores, _spec, _check = load_and_register(
            path,
            dep_key=f"judge_{axis}_responses_{side}_b{B}",
            extras={"axis": axis, "side": side, "B": str(B)},
            policy="warn",
            inputs=inputs,
        )
        for name, info in scores.items():
            ms = info.get("mean_score")
            if ms is not None:
                out[entity_id(name, side)] = float(ms)
    return out


def load_entity_matrix(data_dir: Path, slot: int,
                        layer: int) -> tuple[list[str], np.ndarray]:
    default_v = _load_vector_file(
        data_dir / "traits" / "vectors" / "default.pt"
    ).float().numpy()[slot, layer]
    names: list[str] = []
    rows = []
    for et in ("roles", "traits"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            names.append(entity_id(fp.stem, et))
            v = _load_vector_file(fp).float().numpy()[slot, layer]
            rows.append(v - default_v)
    return names, np.stack(rows, axis=0).astype(np.float32)


def axis_direction(data_dir: Path, pos_name: str, neg_name: str,
                    slot: int, layer: int) -> np.ndarray:
    p = _load_vector_file(data_dir / "traits" / "vectors" / f"{pos_name}.pt"
                          ).float().numpy()[slot, layer]
    n = _load_vector_file(data_dir / "traits" / "vectors" / f"{neg_name}.pt"
                          ).float().numpy()[slot, layer]
    d = p - n
    nrm = float(np.linalg.norm(d))
    return (d / nrm).astype(np.float32) if nrm > 0 else d.astype(np.float32)


def setup_at(data_dir: Path, slot: int, layer: int):
    """Load (entity matrix, A_g/A_n CA subspaces, augmented pool) for one
    (slot, layer) config.  Cache externally for re-use across axes."""
    names, M_raw = load_entity_matrix(data_dir, slot, layer)
    A_g, A_n = build_goal_nogoal_subspaces(data_dir, slot=slot, layer=layer,
                                            kind="combined")
    pool_entries = build_augmented_whitening_pool(data_dir,
                                                    leave_out=frozenset())
    default_v = _load_vector_file(
        data_dir / "traits" / "vectors" / "default.pt"
    ).float().numpy()[slot, layer]
    pool_rows = []
    for et, name in pool_entries:
        v = np.asarray(load_vector(data_dir, et, name),
                        dtype=np.float32)[slot, layer]
        pool_rows.append(v - default_v)
    pool = np.stack(pool_rows, axis=0).astype(np.float32)
    return names, M_raw, A_g, A_n, pool


# ---------------------------------------------------------------------------
# Per-cell (L, K)-sweep -> best ρ
# ---------------------------------------------------------------------------

def apply_LK(M_raw, axis_dir, A_g, A_n, pool, L: int, K: int,
              shear_cache, whiten_cache):
    """Apply L-shear + K-soft-K-whitening to both M_raw and axis_dir.
    Returns (M_done, axis_done) or None if either fit fails."""
    # fit_shear / fit_whitening can raise ValueError (bad L/K) or
    # LinAlgError (SVD didn't converge); both mean "this (L, K) cell is
    # unusable, skip it and return None upward".
    _FIT_FAIL = (ValueError, IndexError, np.linalg.LinAlgError)
    if L == 0:
        M_shear = M_raw
        pool_shear = pool
        axis_shear = axis_dir
    else:
        if L not in shear_cache:
            try:
                shear_cache[L] = fit_shear(A_g, A_n, L=L)
            except _FIT_FAIL:
                shear_cache[L] = None
        sh = shear_cache[L]
        if sh is None:
            return None
        M_shear = sh.apply(M_raw)
        pool_shear = sh.apply(pool)
        axis_shear = sh.apply(axis_dir[None, :])[0]

    if K == 0:
        return M_shear, axis_shear

    wkey = (L, K)
    if wkey not in whiten_cache:
        try:
            whiten_cache[wkey] = fit_whitening("soft_K", pool_shear, K=K)
        except _FIT_FAIL:
            whiten_cache[wkey] = None
    w = whiten_cache[wkey]
    if w is None:
        return None
    M_done = w.apply(M_shear)
    axis_done = w.apply(axis_shear[None, :])[0]
    return M_done, axis_done


def rho_at_LK(M_done, axis_done, names, scores) -> float | None:
    nrm = float(np.linalg.norm(axis_done))
    if nrm < 1e-12:
        return None
    proj = (M_done @ axis_done) / nrm
    n2i = {n: i for i, n in enumerate(names)}
    common = [n for n in names if n in scores]
    if len(common) < 5:
        return None
    x = np.array([scores[n] for n in common])
    y = proj[[n2i[n] for n in common]]
    r = spearmanr(x, y).correlation
    return None if np.isnan(r) else float(r)


def best_rho(M_raw, axis_dir, A_g, A_n, pool, names, scores) -> float:
    """Sweep (L, K) with refinement on K; return max ρ found."""
    shear_cache, whiten_cache = {}, {}

    def eval_LK(L, K):
        out = apply_LK(M_raw, axis_dir, A_g, A_n, pool, L, K,
                        shear_cache, whiten_cache)
        if out is None:
            return None
        return rho_at_LK(out[0], out[1], names, scores)

    grid: dict[tuple[int, int], float] = {}
    for L in L_COARSE:
        for K in K_COARSE:
            r = eval_LK(L, K)
            if r is not None:
                grid[(L, K)] = r

    if not grid:
        return float("nan")

    Ls = sorted({L for L, _ in grid})
    for L in Ls:
        Ks_done = sorted({K for (LL, K) in grid if LL == L})
        if not Ks_done:
            continue
        for _ in range(K_REFINE_MAX_ITER):
            Ks_done = sorted({K for (LL, K) in grid if LL == L})
            best_K, best_r = max(
                ((K, grid[(L, K)]) for K in Ks_done),
                key=lambda t: t[1],
            )
            idx = Ks_done.index(best_K)
            K_l = Ks_done[max(0, idx - 1)]
            K_r = Ks_done[min(len(Ks_done) - 1, idx + 1)]
            if K_r - K_l <= K_REFINE_MIN_BRACKET:
                break
            new_Ks = []
            for frac in (1/3, 2/3):
                nk = int(round(K_l + frac * (K_r - K_l)))
                if (L, nk) not in grid and 0 <= nk <= max(K_COARSE):
                    new_Ks.append(nk)
            if not new_Ks:
                break
            for nk in new_Ks:
                r = eval_LK(L, nk)
                if r is not None:
                    grid[(L, nk)] = r
            new_best_K, new_best_r = max(
                ((K, grid[(L, K)]) for K in
                  sorted({K for (LL, K) in grid if LL == L})),
                key=lambda t: t[1],
            )
            if (new_best_r <= best_r + TIE_THRESH
                    and new_best_K == best_K):
                break

    return max(grid.values())


# ---------------------------------------------------------------------------
# Public library entry point
# ---------------------------------------------------------------------------

def compute_batch_size_curve(
    *,
    experiment_dir: Path,
    data_dir: Path,
    axes: list[tuple[str, str, str]],
    configs: list[tuple[int, int]],
    batch_sizes: list[int],
    scores_filename: str = "scores_responses.json",
    inputs: list[InputSpec] | None = None,
) -> dict:
    """Run the full sweep and return the result dict (same as the JSON output).

    ``axes`` is a list of ``(axis_dir_name, pos_name, neg_name)`` triples.

    Returns
    -------
    {
      "axes":             [(name, pos, neg), ...],
      "batch_sizes":      [5, 7, 10, 15, ...],
      "configs":          [[3, 25], [0, 26], [0, 49], ...],
      "per_axis":         {"axis|b<N>|s<S>_l<L>": rho, ...},
      "avg_over_axes":    {"b<N>|s<S>_l<L>": rho, ...},
      "grand_mean_per_b": {"b<N>": rho, ...},
      "cost_per_axis_usd": {"b<N>": float, ...},
    }
    """
    geom: dict[tuple[int, int], tuple] = {}
    for slot, layer in configs:
        geom[(slot, layer)] = setup_at(data_dir, slot, layer)

    scores_cache: dict[tuple[str, int], dict] = {}
    for axis, _, _ in axes:
        for B in batch_sizes:
            s = load_scores(experiment_dir, axis, B,
                            scores_filename=scores_filename,
                            inputs=inputs)
            if not s:
                print(f"  WARNING: no scores for {axis} at B={B} "
                      f"(suffix={_b_suffix(B)!r})")
            scores_cache[(axis, B)] = s

    rho_per: dict[tuple[str, int, int, int], float] = {}
    for axis, pos, neg in axes:
        for slot, layer in configs:
            names, M_raw, A_g, A_n, pool = geom[(slot, layer)]
            ax_dir = axis_direction(data_dir, pos, neg, slot, layer)
            for B in batch_sizes:
                scores = scores_cache[(axis, B)]
                # Hold out the pole-defining traits from the scoring
                # set.  ``scores`` is entity_id-keyed (e.g. "patient|T")
                # but pos/neg are bare; compare on the bare name so
                # both kinds of any pole-name collision are held out
                # (the pair-typed script is trait-only via
                # ``axis_direction``, so excluding the role homonym
                # too is harmless and safer).
                scores_held = {n: v for n, v in scores.items()
                                if display_label(n) not in (pos, neg)}
                if not scores_held:
                    rho_per[(axis, B, slot, layer)] = float("nan")
                    continue
                r = best_rho(M_raw, ax_dir, A_g, A_n, pool, names, scores_held)
                rho_per[(axis, B, slot, layer)] = r
                print(f"  {axis} / B={B} / ({slot},{layer:>2}): "
                      f"best ρ = {r:+.4f}")

    avg_per: dict[tuple[int, int, int], float] = {}
    for B in batch_sizes:
        for slot, layer in configs:
            rs = [rho_per[(axis, B, slot, layer)] for axis, _, _ in axes
                  if not np.isnan(rho_per[(axis, B, slot, layer)])]
            avg_per[(B, slot, layer)] = float(np.mean(rs)) if rs else float("nan")

    grand_per_b: dict[int, float] = {}
    for B in batch_sizes:
        rs = [avg_per[(B, s, l)] for s, l in configs
              if not np.isnan(avg_per[(B, s, l)])]
        grand_per_b[B] = float(np.mean(rs)) if rs else float("nan")

    return {
        "axes": [list(a) for a in axes],
        "batch_sizes": list(batch_sizes),
        "configs": [list(c) for c in configs],
        "per_axis": {f"{a}|b{B}|s{s}_l{l}": v
                      for (a, B, s, l), v in rho_per.items()},
        "avg_over_axes": {f"b{B}|s{s}_l{l}": v
                          for (B, s, l), v in avg_per.items()},
        "grand_mean_per_b": {f"b{B}": v for B, v in grand_per_b.items()},
        "cost_per_axis_usd": {f"b{B}": COST_PER_AXIS_USD.get(B, float("nan"))
                                for B in batch_sizes},
        "B_integer": {f"b{B}": B for B in batch_sizes},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_BATCH_COLORS = {
    5:  "#4c72b0",
    7:  "#55a868",
    10: "#dd8452",
    15: "#9b59b6",
    20: "#e67e22",
    30: "#7f7f7f",
}


def plot_curve(result: dict, output_dir: Path, *,
                inputs: list[InputSpec] | None = None,
                stem: str = "batch_size_curve_rho",
                ) -> tuple[Path, Path]:
    """Two-panel histogram + line plot, plus a separate cost-vs-ρ scatter.
    Returns (curve_png, scatter_png).

    ``inputs`` (when provided) is embedded into the PNG metadata for
    provenance tracking.  See :mod:`assistant_axis.provenance`.

    ``stem`` controls the output filenames (``<stem>.png`` and
    ``batch_size_cost_vs_rho<stem-suffix>.png``); pass a non-default
    value when running v1- and v2-scope sweeps side-by-side so they
    don't clobber each other.
    """
    axes = result["axes"]
    batch_sizes = result["batch_sizes"]
    configs = [tuple(c) for c in result["configs"]]
    avg = {tuple(int(p[1:]) if p.startswith("b") else int(p[1:].split("_")[0])
                  for p in k.split("|"))[:1] +
           tuple(int(x) for x in k.split("|")[1].split("_")[0][1:].split())[:1]:
           v for k, v in result["avg_over_axes"].items()}
    # Simpler: re-parse via the keys directly.
    avg = {}
    for k, v in result["avg_over_axes"].items():
        # k = "b<B>|s<S>_l<L>"
        b_part, sl_part = k.split("|")
        B = int(b_part[1:])
        s_str, l_str = sl_part.split("_")
        slot = int(s_str[1:])
        layer = int(l_str[1:])
        avg[(B, slot, layer)] = v
    grand = {int(k[1:]): v for k, v in result["grand_mean_per_b"].items()}
    cost = {int(k[1:]): v for k, v in result["cost_per_axis_usd"].items()}
    axes_str = ", ".join(a[0] for a in axes)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Two-panel: histogram + line plot ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.0, 5.6),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    n_groups = len(configs)
    n_batches = len(batch_sizes)
    width = 0.8 / max(n_batches, 1)
    xs = np.arange(n_groups)
    for j, B in enumerate(batch_sizes):
        ys = [avg.get((B, s, l), float("nan")) for s, l in configs]
        offset = (j - (n_batches - 1) / 2) * width
        color = _BATCH_COLORS.get(B, "#888888")
        ax1.bar(xs + offset, ys, width=width,
                color=color, edgecolor="black", linewidth=0.6,
                label=f"B={B}  (${cost[B]:.2f}/axis)")
        for xi, val in zip(xs + offset, ys):
            if not np.isnan(val):
                ax1.text(xi, val + 0.004, f"{val:+.3f}",
                          ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"slot {s}\nlayer {l}" for s, l in configs],
                        fontsize=10)
    ax1.set_ylabel(f"Avg best ρ across {len(axes)} axis"
                   + ("es" if len(axes) != 1 else ""))
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(axis="y", linestyle="-", alpha=0.25)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)
    valid = [v for v in avg.values() if not np.isnan(v)]
    if valid:
        ax1.set_ylim(min(valid) - 0.03, max(valid) + 0.03)
    ax1.set_title("(a) per (slot, layer) × batch-size", fontsize=11)

    # Line plot panel.
    Bs = list(batch_sizes)
    for s, l in configs:
        ys = [avg.get((B, s, l), float("nan")) for B in Bs]
        ax2.plot(Bs, ys, marker="o", markersize=8, linewidth=1.8,
                  label=f"slot {s}, layer {l}")
    grand_ys = [grand[B] for B in Bs]
    ax2.plot(Bs, grand_ys, marker="s", markersize=10, linewidth=2.5,
              color="black", label=f"grand mean ({len(configs)} configs)")
    for x, y in zip(Bs, grand_ys):
        if not np.isnan(y):
            ax2.text(x, y + 0.005, f"{y:+.3f}",
                      ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Target batch size B")
    ax2.set_ylabel("Avg best ρ across axes")
    ax2.set_xticks(Bs)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax2.set_title("(b) ρ vs target batch size", fontsize=11)
    if max(Bs) > min(Bs):
        ax2.invert_xaxis()  # smaller B (more $) on the right

    title = f"GPT responses-mode ρ vs batch size ({len(axes)} axes)"
    spec = (f"axes: {axes_str};\n"
            f"best across L ∈ {L_COARSE} × K ∈ [0..{max(K_COARSE)}] "
            "(bracket-and-bisect refined on K)")
    _, top_rect = suptitle_with_specs(fig, title, spec)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    curve_path = output_dir / f"{stem}.png"
    src_text = Path(__file__).read_text(encoding="utf-8")
    fig.savefig(curve_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, source_text=src_text,
                                       inputs=inputs))
    plt.close(fig)
    print(f"Wrote {curve_path}")

    # --- Cost-vs-ρ scatter ---
    fig2, ax3 = plt.subplots(figsize=(7.0, 5.0))
    costs = [cost[B] for B in Bs]
    grand_ys = [grand[B] for B in Bs]
    ax3.plot(costs, grand_ys, marker="s", markersize=11, linewidth=2.0,
              color="#333333")
    for B, c, y in zip(Bs, costs, grand_ys):
        if not np.isnan(y):
            ax3.annotate(f"B={B}\n${c:.2f}/axis\nρ={y:+.3f}",
                          xy=(c, y), xytext=(8, 8),
                          textcoords="offset points", fontsize=9)
    ax3.set_xlabel("Cost per axis (USD, gpt-4.1-mini)")
    ax3.set_ylabel(f"Grand-mean ρ across {len(configs)} configs × "
                   f"{len(axes)} axes")
    ax3.grid(True, alpha=0.25)
    title2 = "Cost vs quality: GPT responses-mode batch size"
    spec2 = (f"ρ averaged over {len(axes)} axes × {len(configs)} (slot, layer) "
             "configs;\ncost from README's per-batch token model")
    _, top_rect2 = suptitle_with_specs(fig2, title2, spec2)
    fig2.tight_layout(rect=(0, 0, 1, top_rect2))
    # Mirror the curve's stem for the scatter name (replace
    # ``batch_size_curve_rho`` -> ``batch_size_cost_vs_rho`` so a
    # ``_v1`` / ``_v2`` suffix carries through):
    scatter_stem = stem.replace("batch_size_curve_rho", "batch_size_cost_vs_rho")
    scatter_path = output_dir / f"{scatter_stem}.png"
    fig2.savefig(scatter_path, dpi=150, bbox_inches="tight",
                  metadata=png_metadata(title=title2, source_text=src_text,
                                         inputs=inputs))
    plt.close(fig2)
    print(f"Wrote {scatter_path}")
    return curve_path, scatter_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Hardcoded pole pairs for the few axes we know about; CLI users can pass
# any axis name and the tool will infer pos/neg from the directory name
# (assumes ``<pos>_vs_<neg>`` convention).

def _infer_pos_neg(axis_name: str) -> tuple[str, str]:
    if "_vs_" not in axis_name:
        raise ValueError(
            f"Axis name {axis_name!r} does not match <pos>_vs_<neg>; "
            "pass --axes_with_poles to override."
        )
    pos, neg = axis_name.split("_vs_", 1)
    return pos, neg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", type=str, default=DEFAULT_EXPERIMENT_DIR,
                   help="Directory holding per-axis subdirs with score files "
                        f"(default: {DEFAULT_EXPERIMENT_DIR}).")
    p.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                   help="Activation vectors directory.")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where to write JSON + PNGs (default: {DEFAULT_OUTPUT_DIR}).")
    p.add_argument("--axes", type=str,
                   default=",".join(DEFAULT_AXES),
                   help="Comma-separated axis names ('<pos>_vs_<neg>'). "
                        f"Default: {','.join(DEFAULT_AXES)}.")
    p.add_argument("--batch_sizes", type=str,
                   default=",".join(str(B) for B in DEFAULT_BATCH_SIZES),
                   help="Comma-separated target batch sizes (e.g. '5,7,10,15').")
    p.add_argument("--configs", type=str,
                   default=",".join(f"{s}:{l}" for s, l in DEFAULT_CONFIGS),
                   help="Comma-separated 'slot:layer' pairs (e.g. '3:25,0:26').")
    p.add_argument("--json_name", type=str, default="batch_size_curve_rho.json",
                   help="Filename for the JSON cache inside --output_dir.")
    p.add_argument("--scores_filename", type=str,
                   default="scores_responses__rubric_v1.json",
                   help="Per-cell GPT-responses score cache filename "
                        "(default: scores_responses__rubric_v1.json -- the "
                        "v1 archive).  The B-curve is intrinsically a v1 "
                        "archive comparison: B={5,15} only exist in the "
                        "v1 snapshot (no v2 rejudge), and the v2 b10 "
                        "scores_responses.json is the broken 1/3-subsampled "
                        "Bug-B cache that's been deferred.  So the default "
                        "now reads v1 everywhere; this keeps all 4 B "
                        "values comparable, was previously silently "
                        "dropping B=5 and B=15 to no-data and reading the "
                        "broken v2 at B=10.  Pass scores_responses.json "
                        "explicitly if you want the (deferred-broken) v2 "
                        "view at the limited B values that have a v2 cache.")
    return p.parse_args()


def _parse_configs(s: str) -> list[tuple[int, int]]:
    out = []
    for tok in s.split(","):
        slot_str, layer_str = tok.split(":")
        out.append((int(slot_str), int(layer_str)))
    return out


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    axes_names = [a.strip() for a in args.axes.split(",") if a.strip()]
    axes = [(a, *_infer_pos_neg(a)) for a in axes_names]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    configs = _parse_configs(args.configs)

    print(f"axes ({len(axes)}): " + ", ".join(a[0] for a in axes))
    print(f"batch_sizes: {batch_sizes}")
    print(f"configs: {configs}")
    print(f"experiment_dir: {experiment_dir}")
    print(f"data_dir: {data_dir}")
    print(f"output_dir: {output_dir}")
    print()

    # --- Provenance inputs (used by JSON + both PNG writes) ---
    # Up-front deps: vector subtrees + the four derived combo-marginal
    # subtrees that build_goal_nogoal_subspaces reads.  Per-(axis, B,
    # side) judge cache InputSpecs are appended inside
    # ``compute_batch_size_curve`` via ``load_scores`` →
    # ``load_and_register``, so a cache that's missing on disk is also
    # absent from the recorded inputs (matches actual consumption).
    inputs: list[InputSpec] = [
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
    ]

    result = compute_batch_size_curve(
        experiment_dir=experiment_dir, data_dir=data_dir,
        axes=axes, configs=configs, batch_sizes=batch_sizes,
        scores_filename=args.scores_filename,
        inputs=inputs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / args.json_name
    envelope = json_metadata(
        result, inputs=inputs,
        title=f"batch_size_rho_curve axes={len(axes)} "
              f"Bs={','.join(str(B) for B in batch_sizes)}")
    json_path.write_text(json.dumps(envelope, indent=2))
    print(f"\nWrote {json_path}")

    # Use the JSON's stem as the PNG stem so v1 / v2 / custom
    # invocations produce non-clashing PNG filenames.
    png_stem = Path(args.json_name).stem
    plot_curve(result, output_dir, inputs=inputs, stem=png_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
