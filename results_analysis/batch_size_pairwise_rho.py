#!/usr/bin/env python3
"""Pairwise Spearman ρ between response-judging batch sizes.

The batch-size sweep in ``batch_size_rho_curve.py`` measures how well
each B-cohort tracks the *axis projection* (a vector-space ground
truth).  This script asks a complementary, lighter-weight question:
**how well do two judging batch sizes agree with each other on the
per-trait / per-role mean score?**

For every (axis, cohort_kind ∈ {traits, roles}, B ∈ {5, 7, 10, 15})
we already have ``scores_responses__rubric_v1.json`` -- a dict mapping
each trait/role to its mean rubric-v1 score over the cohort's
batched-judge calls.  Spearman ρ between the score vectors at two
batch sizes tells us how much the per-entity rank ordering shifts as
B changes, *without* invoking the axis or any L/K whitening sweep.

Coverage (May 2026): only GPT has all four B values, and only on
3 axes (TVD / PVC / IVM).  Sonnet and Haiku exist at B=10 only, so
this script is GPT-only by construction.

We compute three views per axis:

* ``traits``   -- ρ on trait entities alone (~302 names).
* ``roles``    -- ρ on role entities alone (~281 names).
* ``combined`` -- ρ on the joined roles+traits list (~583 entries),
  keyed by ``entity_id(name, kind)`` so the 9 names that are both
  a trait and a role (``patient``, ``critic``, ...) stay separate
  rather than colliding to a single key.  The on-disk cohort files
  are kind-pure (one file per kind), so disambiguation by
  source-file is exact -- the v1 ``scores_descriptions.json``-style
  collision contamination does NOT apply here.  Combining the two
  cohorts roughly doubles the n behind each ρ and is the more
  statistically powerful comparison; the per-kind panels remain
  for diagnostic/sanity-check purposes.

Outputs (default ``--output_dir`` = ``roger/axis_judge_experiments/batch_size_curve_8slot``):

- ``batch_size_pairwise_rho.json`` -- per-axis, per-cohort 4×4 ρ
  matrices (``traits``, ``roles``, ``combined``) plus the
  across-axis mean matrix (always with 1.0 on the diagonal).  All
  three views are kept in the JSON for diagnostic / sanity-check
  purposes.
- ``batch_size_pairwise_rho.png`` -- single-panel heatmap of the
  ``combined`` (roles ∪ traits, n ≈ 580) across-axis mean 4×4
  matrix with cell annotations.  This is the headline view: it
  uses the union of role + trait entities for ~2× the statistical
  power of either kind alone, which is what we typically want
  when comparing batch sizes.  The per-kind matrices are still
  computed (and visible in the JSON) but not plotted by default.

Both outputs carry a full ``_provenance`` envelope listing every
``scores_responses__rubric_v1.json`` consumed (one InputSpec per
(axis, cohort, B) cell -- 24 inputs at the default settings).

Usage::

    uv run python results_analysis/batch_size_pairwise_rho.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from assistant_axis import (  # noqa: E402
    entity_id,
    json_metadata,
    png_metadata,
    suptitle_with_specs,
)
from assistant_axis.provenance import InputSpec, load_and_register  # noqa: E402


DEFAULT_AXES = (
    "truthful_vs_deceitful",
    "progressive_vs_conservative",
    "improvisational_vs_methodical",
)
DEFAULT_BS = (5, 7, 10, 15)
DEFAULT_KINDS = ("traits", "roles")
COMBINED_KIND = "combined"
JUDGE = "gpt"  # only judge with full B coverage


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _cohort_dir(judge: str, kind: str, B: int) -> str:
    """``B == 15`` is the unsuffixed v1 cohort (legacy naming)."""
    return f"{judge}_responses_{kind}" + ("" if B == 15 else f"_b{B}")


def load_mean_scores(
    experiment_root: Path, axis: str, kind: str, B: int,
    *, inputs: list[InputSpec],
) -> dict[str, float]:
    """Load ``scores_responses__rubric_v1.json`` for one (axis, kind, B)
    cell and register it as an input dependency.  Returns
    ``{entity_name: mean_score}``.
    """
    sub = _cohort_dir(JUDGE, kind, B)
    path = experiment_root / axis / sub / "scores_responses__rubric_v1.json"
    payload, _spec, _check = load_and_register(
        path,
        dep_key=f"scores_responses__rubric_v1[{axis}/{sub}]",
        extras={"axis": axis, "kind": kind, "B": str(B), "judge": JUDGE},
        policy="warn",
        inputs=inputs,
    )
    return {
        name: entry["mean_score"]
        for name, entry in payload.items()
        if isinstance(entry, dict) and "mean_score" in entry
    }


def pairwise_rho_matrix(
    score_by_B: dict[int, dict[str, float]],
    bs: list[int],
) -> tuple[np.ndarray, int]:
    """Compute the |bs|×|bs| Spearman ρ matrix.  Diagonal forced to 1.0.

    Uses the intersection of entity keys across all four B values so
    every off-diagonal cell is computed on the same n.
    """
    common = sorted(set.intersection(*(set(score_by_B[B]) for B in bs)))
    arrs = {B: np.array([score_by_B[B][k] for k in common], dtype=float)
            for B in bs}
    n = len(bs)
    M = np.eye(n, dtype=float)
    for i, B1 in enumerate(bs):
        for j, B2 in enumerate(bs):
            if i >= j:
                continue
            rho, _ = spearmanr(arrs[B1], arrs[B2])
            M[i, j] = M[j, i] = float(rho)
    return M, len(common)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_heatmap_pair(
    mean_by_kind: dict[str, np.ndarray],
    n_by_kind: dict[str, int],
    *,
    bs: list[int],
    axes_used: list[str],
    output_path: Path,
    inputs: list[InputSpec],
    panel_kinds: tuple[str, ...] = ("traits", "roles", COMBINED_KIND),
) -> Path:
    """Render a 1×N heatmap panel (default 1×3: traits | roles |
    combined).  Each panel is a 4×4 ρ matrix with diagonal pinned at
    1.0 and off-diagonals = mean ρ across axes.
    """
    n_panels = len(panel_kinds)
    # Width scales linearly with panel count; ~5.3 in/panel feels
    # readable at the default 4-cell grid.  Height is 6.4 in -- this
    # leaves enough vertical room above the panels for the bold
    # suptitle + the smaller spec line + the per-panel title without
    # them stacking onto each other or onto the heatmap.
    #
    # Layout note: constrained_layout=True respects colorbars
    # correctly (tight_layout does not, and stuffs the colorbar on
    # top of the heatmap).  But constrained_layout only sees the
    # ``fig.suptitle`` headline -- not the ``fig.text`` spec line
    # rendered just below it.  Net effect: with constrained_layout
    # we have to manually create vertical breathing room, which is
    # what the larger figure height + the post-hoc
    # ``subplots_adjust(top=top_rect)`` below do.
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.3 * n_panels, 6.4),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = np.array([axes])
    cmap = plt.get_cmap("viridis")
    # Shared colour limits so all panels are directly comparable.
    off_diag_vals = np.concatenate([
        mean_by_kind[k][np.triu_indices_from(mean_by_kind[k], k=1)]
        for k in panel_kinds
    ])
    vmin = float(off_diag_vals.min())
    vmax = 1.0
    labels = [f"B={B}" for B in bs]

    for ax, kind in zip(axes, panel_kinds):
        M = mean_by_kind[kind]
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(bs)))
        ax.set_yticks(range(len(bs)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(len(bs)):
            for j in range(len(bs)):
                value = M[i, j]
                norm = (value - vmin) / max(vmax - vmin, 1e-9)
                text_color = "white" if norm < 0.55 else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                        color=text_color, fontsize=9)
        ax.set_title(
            f"{kind}  (mean over {len(axes_used)} axes; "
            f"n = {n_by_kind[kind]})",
            fontsize=11,
        )
        ax.set_xlabel("batch size")
        ax.set_ylabel("batch size")
    # shrink=0.7 (rather than the matplotlib default 1.0 or our
    # earlier 0.85) keeps the colorbar's top tick label well below
    # the spec line.  constrained_layout ignores
    # ``subplots_adjust(top=...)`` once a colorbar is in the
    # figure, so we control the colorbar height directly here.
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7,
                 label="Spearman ρ (rubric v1, GPT)")

    suptitle = (f"Pairwise Spearman ρ between Judging Batch Sizes "
                f"(per-entity mean score, GPT, rubric v1)")
    # Spec line is short on purpose: long single-line specs collide
    # with the colorbar at narrow figure widths (the spec is rendered
    # at fig-center, but the heatmap centre sits to the LEFT of
    # fig-centre because the colorbar reserves the right ~12-15% of
    # the canvas).  The axis count + n live in the panel title; full
    # axis list lives in the JSON _provenance + PNG metadata.
    spec_line = "diagonal pinned at 1.0; off-diagonals are the mean ρ across axes"
    _, top_rect = suptitle_with_specs(fig, suptitle, spec_line)
    # constrained_layout doesn't see the fig.text spec line, so
    # carve out the reserved top region manually.  This pushes the
    # heatmap + colorbar down out of the title block.
    fig.subplots_adjust(top=top_rect)
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
        "--experiment_root", type=str,
        default="roger/axis_judge_experiments",
        help="Where to find <axis>/<judge>_responses_<kind>[_bN]/ dirs.",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="roger/axis_judge_experiments/batch_size_curve_8slot",
        help="Where to write the JSON and PNG outputs.",
    )
    p.add_argument(
        "--axes", type=str, default=",".join(DEFAULT_AXES),
        help="Comma-separated axis directory names; defaults to the 3 "
             "axes that have GPT data at all four batch sizes.",
    )
    p.add_argument(
        "--batch_sizes", type=str,
        default=",".join(str(b) for b in DEFAULT_BS),
        help="Comma-separated batch sizes to compare.  B=15 maps to the "
             "unsuffixed v1 cohort directory (legacy naming).",
    )
    p.add_argument(
        "--kinds", type=str, default=",".join(DEFAULT_KINDS),
        help="Comma-separated cohort kinds (traits and/or roles).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    axes_used = [s.strip() for s in args.axes.split(",") if s.strip()]
    bs = sorted({int(s) for s in args.batch_sizes.split(",") if s.strip()})
    kinds = [s.strip() for s in args.kinds.split(",") if s.strip()]

    inputs: list[InputSpec] = []

    # Per-(axis, kind) ρ matrices, plus across-axis mean.  We also
    # compute a "combined" view that joins roles+traits per axis,
    # keyed by entity_id(name, kind) so the 9 collision names stay
    # disambiguated; this is reported as a third entry in
    # `per_cell` / `mean_across_axes` whenever both 'traits' and
    # 'roles' are in --kinds.
    per_cell: dict[str, dict[str, dict]] = {}
    mean_by_kind: dict[str, np.ndarray] = {}
    n_by_kind: dict[str, int] = {}

    # Cache loaded per-kind dicts so the combined view doesn't pay
    # the I/O (and provenance-registration) cost twice.
    score_by_axis_kind_B: dict[tuple[str, str, int], dict[str, float]] = {}

    for kind in kinds:
        per_cell[kind] = {}
        Ms = []
        ns = []
        for axis in axes_used:
            score_by_B = {
                B: load_mean_scores(experiment_root, axis, kind, B,
                                    inputs=inputs)
                for B in bs
            }
            for B, scores in score_by_B.items():
                score_by_axis_kind_B[(axis, kind, B)] = scores
            M, n = pairwise_rho_matrix(score_by_B, bs)
            per_cell[kind][axis] = {
                "matrix": M.tolist(),
                "n_common_entities": n,
            }
            Ms.append(M)
            ns.append(n)
            print(f"  {axis}/{kind}: n={n}, "
                  f"min off-diag ρ = {M[np.triu_indices_from(M, k=1)].min():.3f}")
        mean_M = np.mean(np.stack(Ms, axis=0), axis=0)
        # Re-pin diagonal at 1.0 (mean of 1.0s is already 1.0, but be explicit).
        np.fill_diagonal(mean_M, 1.0)
        mean_by_kind[kind] = mean_M
        n_by_kind[kind] = int(round(float(np.mean(ns))))

    # ------------------------------------------------------------------
    # Combined view (traits ∪ roles) keyed by entity_id(name, kind).
    # Per-kind cohort files are kind-pure on disk, so disambiguation
    # by source-file is exact and the 9 collision names are kept
    # separate (no v1-style cache contamination at this layer).
    # ------------------------------------------------------------------
    have_combined = {"traits", "roles"}.issubset(kinds)
    if have_combined:
        per_cell[COMBINED_KIND] = {}
        Ms = []
        ns = []
        for axis in axes_used:
            score_by_B: dict[int, dict[str, float]] = {}
            for B in bs:
                merged: dict[str, float] = {}
                for kind in ("traits", "roles"):
                    src = score_by_axis_kind_B[(axis, kind, B)]
                    etype = "T" if kind == "traits" else "R"
                    for name, score in src.items():
                        merged[entity_id(name, etype)] = score
                score_by_B[B] = merged
            M, n = pairwise_rho_matrix(score_by_B, bs)
            per_cell[COMBINED_KIND][axis] = {
                "matrix": M.tolist(),
                "n_common_entities": n,
            }
            Ms.append(M)
            ns.append(n)
            print(f"  {axis}/{COMBINED_KIND}: n={n}, "
                  f"min off-diag ρ = {M[np.triu_indices_from(M, k=1)].min():.3f}")
        mean_M = np.mean(np.stack(Ms, axis=0), axis=0)
        np.fill_diagonal(mean_M, 1.0)
        mean_by_kind[COMBINED_KIND] = mean_M
        n_by_kind[COMBINED_KIND] = int(round(float(np.mean(ns))))

    # JSON output
    payload_kinds = list(kinds) + ([COMBINED_KIND] if have_combined else [])
    payload = {
        "judge": JUDGE,
        "rubric": "v1",
        "axes": axes_used,
        "batch_sizes": bs,
        "kinds": payload_kinds,
        "per_cell": per_cell,
        "mean_across_axes": {
            k: mean_by_kind[k].tolist() for k in payload_kinds
        },
        "mean_across_axes_n": {
            k: n_by_kind[k] for k in payload_kinds
        },
    }
    json_path = output_dir / "batch_size_pairwise_rho.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            json_metadata(
                payload,
                title="Pairwise Spearman ρ between Judging Batch Sizes",
                inputs=inputs,
            ),
            f, indent=2,
        )
    print(f"Wrote {json_path}")

    png_path = output_dir / "batch_size_pairwise_rho.png"
    if have_combined:
        make_heatmap_pair(
            mean_by_kind, n_by_kind,
            bs=bs, axes_used=axes_used,
            output_path=png_path, inputs=inputs,
            panel_kinds=(COMBINED_KIND,),
        )
        print(f"Wrote {png_path}")
    else:
        print(f"Skipping PNG -- need both 'traits' and 'roles' kinds; "
              f"got {kinds}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
