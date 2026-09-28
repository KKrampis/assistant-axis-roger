#!/usr/bin/env python3
"""Seriated cosine heatmap of clean axes in the L=3 soft-shear frame.

Pairwise |cos| between every axis in ``pair_list_clean.json`` (or any
cohort passed via ``--pairs``) after applying soft-shear at the project
default ``L = DEFAULT_SOFT_SHEAR_L``; axes are reordered via optimal
leaf ordering (OLO) on a hierarchical clustering of the
``(1 − |cos|)`` dissimilarity, so highly-related axes appear adjacent
in the matrix and clusters show up as block-diagonal regions.

The seriation problem is "find a permutation that maximises diagonal
mass in a similarity matrix"; it's a 1-D embedding flavour of TSP.
General optimum is NP-hard but OLO (Bar-Joseph 2001) finds the
EXACT optimal leaf order given a dendrogram in O(n³ + n²·m) time --
``scipy.cluster.hierarchy.optimal_leaf_ordering`` implements it.

Outputs (default ``--experiment_dir`` = ``roger/axis_judge_experiments/``)
-------------------------------------------------------------------------

For ``--pairs pair_list_clean.json``, ``--L 3`` (defaults):

* ``axis_cosine_heatmap_clean_softshear3.png`` — seriated heatmap
  with cluster-bracket overlays at the threshold (default 0.5) and
  semantic-block ontology bands on the y/x axes (see below).
* ``axis_cosine_heatmap_clean_softshear3.json`` — datastore holding:

  * ``axis_labels`` (list, orig-index order) and ``abs_cos_matrix``
    (n×n) — full pairwise |cos| matrix in the chosen whitening frame,
    so downstream consumers don't have to re-run the (~30 s) build
    step.
  * ``linkage_matrix`` ((n−1)×4) — the scipy.cluster.hierarchy
    optimal-leaf-ordered linkage; pass to ``dendrogram`` /
    ``fcluster`` to recover the hierarchical structure.
  * ``seriated_order`` (list of dicts) — the OLO permutation
    (orig_index → seriated_index) plus pos/neg/label for each axis.
  * ``clusters_by_threshold`` — cluster memberships at thresholds
    0.7 / 0.6 / 0.5 / 0.4 / 0.3 (distance = 1 − |cos|).
  * ``ontology`` — the resolved semantic-block run structure when
    the cohort has a hand-curated ontology (see below).
  * ``pairwise_abs_cos`` — summary stats (min/max/mean/median) on
    the off-diagonal entries.

  Both raw (``--L 0``) and ``soft_shear=3`` (default) variants are
  cached as separate files so cross-frame comparison is one ``json.load``
  away.  This is the canonical datastore for axis-to-axis geometry;
  use it instead of recomputing pairwise cosines anywhere downstream.

With ``--L 0`` the basename swaps the ``softshear3`` tag for ``raw``
(emits ``axis_cosine_heatmap_clean_raw.{png,json}``) so the raw and
sheared frames can be compared side-by-side without filename
collisions.

Ontology overlay
----------------

When the chosen cohort has an entry in ``COHORT_ONTOLOGIES`` (currently
``clean`` and ``goalnongoal``), the plot grows two outer coloured bands
-- one on the left axis (labelled), one on the bottom axis (colour-only
since the same info is on the y-band) -- annotating each axis with its
semantic block.  The 8-block ontology for the clean 60-pair cohort is
hand-curated (May 2026):

  * Power & certainty       (5 axes; predator/prey, fragile/resilient, ...)
  * Epistemic style         (12 axes; descriptive/prescriptive, ...)
  * Affect                  (3 axes; passionate/dispassionate, ...)
  * Worldview & norms       (10 axes; religious/secular, ...)
  * Diligence & openness    (5 axes; impatient/patient, ...)
  * Honesty                 (3 axes; honest/dishonest, ...)
  * Alignment & prosocial   (13 axes; harmless/harmful, ...)
  * Social stance           (9 axes; cooperative/competitive, ...)

The ontology is purely annotational -- it doesn't feed into the
clustering or seriation.  The interesting empirical finding (May 2026)
is that the OLO seriation on the *canonical* L=3 soft-sheared frame
recovers the human-curated ontology exactly (8 runs / 8 blocks, 0
splits) whereas the *raw* frame fragments it heavily (21 runs / 8
blocks, 7 split).  Soft-shear isn't manufacturing the semantic
structure, but it IS aligning the data-driven geometry with it.

Pass ``--ontology none`` to disable the overlay; the bands disappear
and the JSON sidecar reports ``ontology.applied=False``.

The matrix is symmetric and uses |cos|, so axes with their poles
named in opposite conventions (e.g. ``careless/conscientious``
relative to ``trustworthy/untrustworthy``) cluster together as if
they were both pointing the same way.  Sign information is lost --
the heatmap measures "axis alignment irrespective of pole
convention".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.cluster.hierarchy import (
    dendrogram, fcluster, linkage, optimal_leaf_ordering,
)
from scipy.spatial.distance import squareform

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from assistant_axis import json_metadata, png_metadata  # noqa: E402
from assistant_axis.pair_list_cohort import (  # noqa: E402
    cohort_from_pairs, pair_type_of,
)
from assistant_axis.provenance import (  # noqa: E402
    InputSpec, current_data_subtree_input, current_file_input,
)
from results_analysis.axis_judge_correlation import (  # noqa: E402
    _load_vector_file,
)
from results_analysis.canonical_angles.data import (  # noqa: E402
    build_goal_nogoal_subspaces,
)
from results_analysis.canonical_angles.whitening import (  # noqa: E402
    DEFAULT_SOFT_SHEAR_L, fit_shear,
)


DEFAULT_EXPERIMENT_DIR = REPO / "roger" / "axis_judge_experiments"
DEFAULT_DATA_DIR = REPO / "runpod_workspace" / "qwen" / "qwen-3-32b Roger 8slot"
DEFAULT_SLOT, DEFAULT_LAYER = 6, 25


# ---------------------------------------------------------------------------
# Semantic-block ontology for the clean 60-pair cohort
# ---------------------------------------------------------------------------
# Hand-curated grouping of every axis in ``pair_list_clean.json`` into one
# of eight semantic blocks (May 2026 analysis).  Used purely for plot
# annotation -- the ontology has no role in the clustering or the seriation
# itself; it just provides human-readable group labels overlaid on the
# heatmap's axes.
#
# Derivation (read this if the cohort changes or you're adding new axes):
# -----------------------------------------------------------------------
# 1. Run the seriation on the canonical L=3 soft-shear frame with default
#    (average) linkage.  The OLO output places semantically-related axes
#    adjacent in the order -- this is the *data-driven* skeleton.
# 2. Walk the seriated list top-to-bottom and group axes into contiguous
#    "runs" where the SEMANTIC story is uniform.  The L=3 frame happens
#    to make this very easy on the clean 60-pair cohort -- the runs are
#    visually obvious in the heatmap and correspond to recognised
#    psychological / philosophical categories.
# 3. Name each run with a crisp single-phrase label (4-25 chars to fit
#    in the band).  Stash a one-line gloss in
#    ``CLEAN_COHORT_ONTOLOGY_DESCRIPTIONS`` so the names are auditable.
# 4. Write the per-axis lookup table below.  The cohort run prints
#    "Ontology: K/B blocks present across R runs; S split" at the end --
#    on the L=3 average-linkage seriation it should print
#    "8/8 blocks present across 8 runs; 0 split"; if it doesn't, the
#    ontology and the data have drifted apart and one of them is wrong.
#
# Why not just auto-cluster from the dendrogram and call those "blocks"?
# ---------------------------------------------------------------------
# Considered, doesn't work.  ``scipy.cluster.hierarchy.fcluster`` with
# either ``criterion='distance'`` (threshold-cut) or ``criterion='maxclust'``
# (top-K cut) produces flat clusters that ARE contiguous in the OLO
# order -- the cluster-integrity guarantee of hierarchical clustering --
# but the cluster boundaries land in the WRONG places for human
# semantics.  Empirical check on the clean 60-pair cohort at L=3
# average-linkage:
#
#   * ``fcluster(Z, t=8, criterion='maxclust')`` over-splits Epistemic
#     style into three separate clusters at high cut-levels (because
#     the dendrogram puts ``descriptive/prescriptive`` slightly apart
#     from the rest), while UNDER-splitting the lower half: it merges
#     Diligence + Honesty + Alignment + Social stance into one giant
#     cluster of 29 axes.
#   * ``fcluster(Z, t=0.5, criterion='distance')`` (i.e. |cos|>=0.5)
#     gives 31 clusters, most of them 2-3 axes -- way too fine.
#   * ``fcluster(Z, t=0.3, criterion='distance')`` (i.e. |cos|>=0.3)
#     gives 11 clusters and one massive 23-axis blob -- still wrong.
#
# The human ontology is a *non-uniform* cut through the dendrogram: some
# branches need a tight cut (Honesty deserves to stay as 3 axes, not be
# merged into a bigger ethics-related cluster), others a loose one
# (Epistemic style spans 12 axes that are tighter to each other than
# they are to the rest of the matrix but looser internally than
# Honesty is).  No single ``fcluster`` threshold captures this.  Could
# you do variable-cut-per-branch?  Probably -- you'd need a criterion
# like "merge children if size < K AND gloss-similarity > threshold",
# which itself requires semantic input.  Not worth the complexity for
# a one-off curated ontology.
#
# Practical regeneration workflow when ``pair_list_clean.json`` changes:
# --------------------------------------------------------------------
# 1. Run ``axis_cosine_seriation.py`` with the new pair list.  If new
#    axes are present that aren't in this table, the script prints a
#    warning listing them (``Ontology: N axes unmapped: ...``).
# 2. For each unmapped axis, look at where it lands in the seriated
#    order and which block it sits inside.  Usually obvious from the
#    label and the surrounding pairs.
# 3. Add the missing entries below, keyed by "pos/neg".  Run again to
#    confirm "0 axes unmapped, 8/8 blocks present, 0 split".
# 4. If the seriation places a new axis in a way that BRIDGES two
#    blocks or splits an existing one, that's a real signal -- either
#    the new axis is genuinely betwixt-and-between (consider renaming
#    or splitting an existing block) or your linkage / L choice has
#    drifted (re-run with the canonical L=3 average-linkage defaults).
#
# To extend to a new cohort entirely, add a sibling dict and key it via
# ``COHORT_ONTOLOGIES``; the plot code looks up by ``cohort_from_pairs``
# and silently disables the outer column when no ontology is registered.
CLEAN_COHORT_ONTOLOGY_BLOCKS: tuple[str, ...] = (
    "Power & certainty",
    "Epistemic style",
    "Affect",
    "Worldview & norms",
    "Diligence & openness",
    "Honesty",
    "Alignment & prosocial",
    "Social stance",
)

CLEAN_COHORT_ONTOLOGY: dict[str, str] = {
    # Power & certainty -- agency, fragility, resoluteness
    "predator/prey": "Power & certainty",
    "fragile/resilient": "Power & certainty",
    "dominant/submissive": "Power & certainty",
    "confident/uncertain": "Power & certainty",
    "decisive/indecisive": "Power & certainty",
    # Epistemic style -- how-they-think + communicative register
    "descriptive/prescriptive": "Epistemic style",
    "accessible/esoteric": "Epistemic style",
    "practical/theoretical": "Epistemic style",
    "idealistic/pragmatic": "Epistemic style",
    "ethereal/grounded": "Epistemic style",
    "concise/verbose": "Epistemic style",
    "convergent/divergent": "Epistemic style",
    "reductionist/holistic": "Epistemic style",
    "quantitative/qualitative": "Epistemic style",
    "precise/vague": "Epistemic style",
    "improvisational/methodical": "Epistemic style",
    "systems_thinker/analytical": "Epistemic style",
    # Affect -- emotional engagement
    "passionate/dispassionate": "Affect",
    "detached/empathetic": "Affect",
    "materialistic/spiritual": "Affect",
    # Worldview & norms -- ideology, tradition, seriousness
    "religious/secular": "Worldview & norms",
    "guardian/destroyer": "Worldview & norms",
    "progressive/conservative": "Worldview & norms",
    "innovative/traditional": "Worldview & norms",
    "obedient/rebellious": "Worldview & norms",
    "irreverent/reverent": "Worldview & norms",
    "casual/formal": "Worldview & norms",
    "playful/serious": "Worldview & norms",
    "relativist/absolutist": "Worldview & norms",
    "constructivist/essentialist": "Worldview & norms",
    # Diligence & openness -- work ethic, candour-as-information
    "impatient/patient": "Diligence & openness",
    "proactive/reactive": "Diligence & openness",
    "careless/conscientious": "Diligence & openness",
    "superficial/thorough": "Diligence & openness",
    "opaque/transparent": "Diligence & openness",
    # Honesty -- truthfulness axes
    "guileless/scheming": "Honesty",
    "honest/dishonest": "Honesty",
    "truthful/deceitful": "Honesty",
    # Alignment & prosocial -- safety / helpfulness / beneficence
    "aligned_artificial_intelligence/paperclip_maximizer": "Alignment & prosocial",
    # the same axis after the 2026-09-28 rename of its positive pole
    "instrumentally_aligned_ai/paperclip_maximizer": "Alignment & prosocial",
    "harmless/harmful": "Alignment & prosocial",
    "curious/incurious": "Alignment & prosocial",
    "helpful/unhelpful": "Alignment & prosocial",
    "generous/stingy": "Alignment & prosocial",
    "symbiont/parasite": "Alignment & prosocial",
    "optimistic/pessimistic": "Alignment & prosocial",
    "constructive/destructive": "Alignment & prosocial",
    "benign/malicious": "Alignment & prosocial",
    "angel/demon": "Alignment & prosocial",
    "trustworthy/untrustworthy": "Alignment & prosocial",
    "dependable/undependable": "Alignment & prosocial",
    "earnest/sardonic": "Alignment & prosocial",
    # Social stance -- interpersonal orientation
    "ecocentric/anthropocentric": "Social stance",
    "egalitarian/elitist": "Social stance",
    "arrogant/humble": "Social stance",
    "cooperative/competitive": "Social stance",
    "forgiving/unforgiving": "Social stance",
    "conciliatory/confrontational": "Social stance",
    "blunt/tactful": "Social stance",
    "individualistic/collectivistic": "Social stance",
    "introverted/extroverted": "Social stance",
}

CLEAN_COHORT_ONTOLOGY_DESCRIPTIONS: dict[str, str] = {
    "Power & certainty": "agency, fragility, resoluteness",
    "Epistemic style": "cognitive style + communicative register",
    "Affect": "emotional engagement",
    "Worldview & norms": "ideology, tradition, seriousness",
    "Diligence & openness": "work ethic, candour-as-information",
    "Honesty": "truthfulness axes",
    "Alignment & prosocial": "safety, helpfulness, beneficence",
    "Social stance": "interpersonal orientation",
}

# Map cohort name (from ``cohort_from_pairs``) → (block_order, label_dict).
# Add new cohorts here as their ontologies stabilise.
COHORT_ONTOLOGIES: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    "clean": (CLEAN_COHORT_ONTOLOGY_BLOCKS, CLEAN_COHORT_ONTOLOGY),
    # ``goalnongoal`` is the 38-axis subset of ``clean``; the same per-axis
    # block labels apply, so we reuse the lookup table.  The subset of
    # blocks that actually appears is determined at runtime.
    "goalnongoal": (
        CLEAN_COHORT_ONTOLOGY_BLOCKS, CLEAN_COHORT_ONTOLOGY,
    ),
}


def axis_unit(data_dir: Path, pos: str, neg: str, kind: str,
              slot: int, layer: int) -> np.ndarray:
    vp = _load_vector_file(data_dir / kind / "vectors" / f"{pos}.pt").float()
    vn = _load_vector_file(data_dir / kind / "vectors" / f"{neg}.pt").float()
    d = (vp[slot, layer] - vn[slot, layer]).numpy()
    return d / np.linalg.norm(d)


def _compute_block_runs(
    seriated_labels: list[str], ontology_map: dict[str, str],
) -> list[tuple[str | None, int, int]]:
    """Walk the seriated order; group adjacent axes by ontology block.

    Returns a list of ``(block_name, start_idx, end_idx_exclusive)``
    triples covering the full ``range(n)``.  Axes with no ontology
    entry get block_name=None (drawn as an "unlabelled" gap).  If a
    block's axes are not contiguous in the seriated order (i.e. the
    seriation has interleaved them with another block), this produces
    multiple runs for the same block -- a visual flag that the
    chosen linkage didn't preserve block structure.
    """
    runs: list[tuple[str | None, int, int]] = []
    n = len(seriated_labels)
    if n == 0:
        return runs
    blocks = [ontology_map.get(lbl) for lbl in seriated_labels]
    start = 0
    for i in range(1, n + 1):
        if i == n or blocks[i] != blocks[start]:
            runs.append((blocks[start], start, i))
            start = i
    return runs


def _ontology_block_colors(
    block_order: tuple[str, ...],
) -> dict[str, tuple[float, float, float, float]]:
    """Map block name → RGBA tuple.

    Uses matplotlib's qualitative ``Set3`` palette -- 12 uniformly
    pastel colours designed for categorical use, chosen here for
    high text-contrast (all light enough that black ``fontweight=
    medium`` labels read clearly on top).  The first 8 entries
    cover the 8-block ``CLEAN_COHORT_ONTOLOGY``.  Falls back to
    grey for unknown blocks (shouldn't happen if ontology is
    well-formed but the JSON sidecar might surface mismatches).

    Set3 ordering (for reference): light teal, light yellow, light
    lavender, light coral, light blue, light orange, light green,
    light pink, light grey, light mauve, light mint, light cream.
    """
    palette = plt.get_cmap("Set3").colors
    return {
        name: palette[i % len(palette)]
        for i, name in enumerate(block_order)
    }


def _wrap_block_label(name: str, max_chars_per_line: int = 11) -> str:
    """Two-line wrap for ontology block labels in a narrow horizontal band.

    Strategy:
    * If the label contains ``" & "`` (the canonical "X & Y" form
      our ontology uses for compound block names), break there:
      ``"Power & certainty"`` → ``"Power &\\ncertainty"``.  The
      ``&`` stays on the first line so the layout reads as a
      compound noun rather than a stranded conjunction.
    * Else if it's two whitespace-separated words and the longer
      word exceeds ``max_chars_per_line``, split into one word per
      line.  ``"Epistemic style"`` stays on one line at
      max_chars_per_line=11, ``"Social stance"`` also.
    * Otherwise leave unchanged (single-word labels like
      ``"Honesty"``, ``"Affect"`` stay one-line by construction).

    The default ``max_chars_per_line=11`` was chosen so the four
    ``"X & Y"`` labels in CLEAN_COHORT_ONTOLOGY (lengths 16-21)
    wrap and the two two-word labels (``"Epistemic style"`` 15,
    ``"Social stance"`` 13) don't -- the latter render fine inline
    in their multi-axis blocks (12 and 9 axes respectively).
    """
    if " & " in name:
        return name.replace(" & ", " &\n", 1)
    parts = name.split()
    if len(parts) == 2 and max(map(len, parts)) > max_chars_per_line:
        return "\n".join(parts)
    return name


def _draw_ontology_bands(
    fig, ax, runs: list[tuple[str | None, int, int]],
    block_colors: dict[str, tuple[float, float, float]],
    n: int, seriated_labels: list[str] | None = None,
    band_size_pct: str = "6%", tick_fontsize: int = 7,
    pad_buffer_inches: float = 0.10,
) -> tuple:
    """Append coloured strips on the left + bottom of the heatmap
    annotating each block run.  The left strip carries the block
    name as a horizontal label, anchored at the run's midpoint;
    the bottom strip is colour-only (block name redundant with
    the left strip in a square heatmap).

    Uses ``make_axes_locatable`` so the bands are proper sibling
    axes that share the heatmap's data extent -- this means the
    y-tick labels on the heatmap still render BETWEEN the band and
    the matrix, not over the band.

    The pad between the band and the main axes is sized to clear
    the longest tick label: at 7pt font, ~0.06 in/char in width and
    ~0.10 in/char in height (rotated 90deg on the x-axis).  We add
    a small buffer (default 0.10 in) so labels don't kiss the band.
    """
    if seriated_labels is None:
        seriated_labels = []
    longest_label_chars = max((len(s) for s in seriated_labels), default=0)
    # 7pt monospace ~ 0.058 in/char width; proportional fonts a bit
    # less.  Use 0.060 as a conservative monospace-ish estimate.
    char_w_in = tick_fontsize * 0.060 / 7.0
    label_width_in = longest_label_chars * char_w_in
    band_pad = label_width_in + pad_buffer_inches
    # Don't use sharey/sharex with ``append_axes`` -- it auto-suppresses
    # the inner-axis tick labels (so our long pair-name labels disappear
    # from the heatmap).  Instead, we set the band's data limits to
    # match the heatmap's after construction.
    div = make_axes_locatable(ax)
    ax_y = div.append_axes("left", size=band_size_pct, pad=band_pad)
    ax_x = div.append_axes("bottom", size=band_size_pct, pad=band_pad)
    grey = (0.85, 0.85, 0.85)

    # Pick exactly one labelled run per block: the longest one.  This
    # avoids stacked / overlapping labels when the seriation fragments
    # a block into many short runs (the raw-frame story).  Ties are
    # broken by earliest start index.
    labelled_idx: set[int] = set()
    by_block: dict[str, list[tuple[int, int]]] = {}  # block → [(len, idx), ...]
    for idx, (name, start, end) in enumerate(runs):
        if name is None:
            continue
        by_block.setdefault(name, []).append((end - start, idx))
    for block_runs in by_block.values():
        block_runs.sort(reverse=True)  # longest first
        labelled_idx.add(block_runs[0][1])

    for idx, (name, start, end) in enumerate(runs):
        colour = block_colors.get(name, grey) if name else grey
        # Left strip (colour-only; labels now live on the bottom).
        ax_y.add_patch(Rectangle(
            (0, start - 0.5), 1, end - start, facecolor=colour,
            edgecolor="none",
        ))
        # Bottom strip (carries the block-name labels, horizontal so
        # they're directly readable -- previous vertical labels in
        # the y-band were harder to scan).
        ax_x.add_patch(Rectangle(
            (start - 0.5, 0), end - start, 1, facecolor=colour,
            edgecolor="none",
        ))
        # Horizontal label on the bottom strip ONLY on the longest run
        # per block.  Skip degenerately-short runs (n=1 axis) where
        # there's no room to render the label legibly.  The pastel
        # palette is light enough that black medium-weight text reads
        # well on top of any block colour.  Long ``X & Y`` labels are
        # wrapped to two lines so they don't bleed past their run
        # into neighbouring blocks (see ``_wrap_block_label``).
        if name and idx in labelled_idx and (end - start) >= 2:
            ax_x.text(
                (start + end - 1) / 2.0, 0.5, _wrap_block_label(name),
                ha="center", va="center", rotation=0,
                fontsize=8, fontweight="medium",
                color="black", linespacing=0.95,
            )
    ax_y.set_xlim(0, 1)
    ax_y.set_ylim(ax.get_ylim())  # match heatmap (origin='upper')
    ax_x.set_ylim(0, 1)
    ax_x.set_xlim(ax.get_xlim())
    for a in (ax_y, ax_x):
        a.set_xticks([])
        a.set_yticks([])
        for spine in a.spines.values():
            spine.set_visible(False)
    return ax_y, ax_x


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pairs", default="pair_list_clean.json",
                   help="Pair list to seriate (default: pair_list_clean.json).")
    p.add_argument("--experiment_dir", default=str(DEFAULT_EXPERIMENT_DIR))
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT)
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    p.add_argument("--L", type=int, default=DEFAULT_SOFT_SHEAR_L,
                   help=f"Soft-shear truncation depth (default: "
                        f"DEFAULT_SOFT_SHEAR_L = {DEFAULT_SOFT_SHEAR_L}).  "
                        f"Pass L=0 for the raw-projection heatmap.")
    p.add_argument("--ca_kind", default="combined",
                   choices=("combined", "traits", "roles"))
    p.add_argument("--linkage", default="average",
                   choices=("average", "complete", "single", "ward"),
                   help="Hierarchical-clustering linkage method "
                        "(default: average).  Average and complete tend to "
                        "give the cleanest block structure for seriation.")
    p.add_argument("--cluster_threshold", type=float, default=0.5,
                   help="|cos| threshold for the cluster-bracket overlay "
                        "on the heatmap (default: 0.5).  The JSON sidecar "
                        "also writes cluster memberships at 0.7 / 0.6 / "
                        "0.5 / 0.4 for reference.")
    p.add_argument("--ontology", default="auto",
                   choices=("auto", "none"),
                   help="Overlay the semantic-block ontology as outer "
                        "coloured strips on the y/x axes (default: "
                        "``auto`` — uses the cohort-keyed ontology in "
                        "``COHORT_ONTOLOGIES`` if one is registered for "
                        "the current cohort, otherwise silently disables; "
                        "pass ``none`` to disable explicitly).")
    p.add_argument("--out_basename", default=None,
                   help="Output basename (default: axis_cosine_heatmap_<cohort>"
                        "_softshear<L>).")
    args = p.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    slot, layer, L = int(args.slot), int(args.layer), int(args.L)
    cohort = cohort_from_pairs(args.pairs)
    if args.out_basename is None:
        L_tag = f"softshear{L}" if L > 0 else "raw"
        args.out_basename = f"axis_cosine_heatmap_{cohort}_{L_tag}"

    pairs = json.loads((experiment_dir / args.pairs).read_text())
    print(f"Loaded {len(pairs)} pairs from {args.pairs}")

    # --- Build axis directions in the chosen frame --------------------
    if L > 0:
        A_g, A_n = build_goal_nogoal_subspaces(
            data_dir, slot=slot, layer=layer, kind=args.ca_kind)
        basis = fit_shear(A_g, A_n, L=L)
    else:
        basis = None

    labels, units = [], []
    for it in pairs:
        kind = pair_type_of(it)
        a = axis_unit(data_dir, it["pos"], it["neg"], kind, slot, layer)
        if basis is not None:
            a = basis.apply(a[None, :])[0]
            a = a / np.linalg.norm(a)
        labels.append(f'{it["pos"]}/{it["neg"]}')
        units.append(a)
    U = np.stack(units)
    M = np.abs(U @ U.T)
    n = M.shape[0]
    print(f"Built {n} axis units in "
          f"{'L=' + str(L) + ' sheared' if L > 0 else 'raw'} frame; "
          f"|cos|: mean={M[np.triu_indices(n, 1)].mean():.3f}, "
          f"median={np.median(M[np.triu_indices(n, 1)]):.3f}")

    # --- Seriate via optimal leaf ordering ----------------------------
    # Dissimilarity = 1 - |cos|; clip to [0, 1] (handles tiny numerical drift).
    D = np.clip(1.0 - M, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=args.linkage)
    Z_ordered = optimal_leaf_ordering(Z, condensed)
    # ``dendrogram`` returns the leaf order under the (possibly OLO-flipped)
    # linkage; using ``no_plot=True`` to extract just the ordering.
    leaf_order = dendrogram(Z_ordered, no_plot=True, distance_sort=False,
                            count_sort=False)["leaves"]
    M_seriated = M[leaf_order][:, leaf_order]
    seriated_labels = [labels[i] for i in leaf_order]

    # Cluster memberships at a few thresholds (distance = 1 - |cos|).
    cluster_thresholds = (0.7, 0.6, 0.5, 0.4, 0.3)
    clusterings: dict[float, list[int]] = {}
    for t in cluster_thresholds:
        clusterings[float(t)] = fcluster(
            Z_ordered, t=1 - t, criterion="distance"
        ).tolist()

    # --- Plot heatmap -------------------------------------------------
    fig_size = max(10, 0.20 * n + 4)  # scale with n
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    # ``magma`` (NOT ``magma_r``) so low |cos| = dark, high |cos| = light.
    # This is the perceptually-natural mapping for "near-orthogonal pairs
    # are uninteresting/empty space; highly-correlated pairs jump out".
    im = ax.imshow(M_seriated, cmap="magma", vmin=0, vmax=1,
                   aspect="equal", origin="upper")
    fig.colorbar(im, ax=ax, label="|cos|", shrink=0.7)

    # Cluster brackets at the user-specified threshold.  Walk the seriated
    # order and bracket adjacent runs whose cluster ID matches.
    cluster_ids_at_t = [
        clusterings[args.cluster_threshold][i] for i in leaf_order
    ]
    run_start = 0
    for i in range(1, n + 1):
        if i == n or cluster_ids_at_t[i] != cluster_ids_at_t[run_start]:
            if i - run_start >= 2:
                # Bracket: rectangle from (run_start, run_start) of size (run_len, run_len).
                ax.add_patch(plt.Rectangle(
                    (run_start - 0.5, run_start - 0.5),
                    i - run_start, i - run_start,
                    fill=False, edgecolor="#3a3a3a", lw=1.2,
                ))
            run_start = i

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(seriated_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(seriated_labels, fontsize=7)
    ax.tick_params(top=False, right=False, bottom=True, left=True)

    # --- Ontology bands (outer coloured strips on y/x axes) -----------
    ontology_runs: list[tuple[str | None, int, int]] = []
    ontology_block_order: tuple[str, ...] = ()
    ontology_map: dict[str, str] = {}
    if args.ontology != "none" and cohort in COHORT_ONTOLOGIES:
        ontology_block_order, ontology_map = COHORT_ONTOLOGIES[cohort]
        ontology_runs = _compute_block_runs(seriated_labels, ontology_map)
        block_colors = _ontology_block_colors(ontology_block_order)
        _draw_ontology_bands(
            fig, ax, ontology_runs, block_colors, n,
            seriated_labels=seriated_labels,
        )
        # Report block coverage to stdout for at-a-glance sanity.
        non_null_runs = [r for r in ontology_runs if r[0] is not None]
        unmapped = [lbl for lbl in seriated_labels
                    if lbl not in ontology_map]
        n_blocks_present = len({r[0] for r in non_null_runs})
        split = sum(
            1 for blk in ontology_block_order
            if sum(1 for r in non_null_runs if r[0] == blk) > 1
        )
        print(f"Ontology: {n_blocks_present}/{len(ontology_block_order)} "
              f"blocks present across {len(non_null_runs)} runs; "
              f"{split} block(s) split by seriation order")
        if unmapped:
            # Listing unmapped axes makes it trivial to extend the
            # ontology when ``pair_list_clean.json`` grows -- see the
            # "Practical regeneration workflow" block in the
            # ``CLEAN_COHORT_ONTOLOGY`` derivation notes.  Print the
            # seriated index too so it's easy to locate them in the
            # heatmap and read which block they sit inside.
            print(f"Ontology: {len(unmapped)} axes UNMAPPED -- add "
                  f"entries to CLEAN_COHORT_ONTOLOGY in "
                  f"axis_cosine_seriation.py for these:")
            for lbl in unmapped:
                idx = seriated_labels.index(lbl)
                neighbours = []
                if idx > 0:
                    prev = seriated_labels[idx - 1]
                    neighbours.append(
                        f"after {prev!r} "
                        f"(block={ontology_map.get(prev, '???')})"
                    )
                if idx < n - 1:
                    nxt = seriated_labels[idx + 1]
                    neighbours.append(
                        f"before {nxt!r} "
                        f"(block={ontology_map.get(nxt, '???')})"
                    )
                neigh = "; ".join(neighbours) if neighbours else "(boundary)"
                print(f"  [{idx:>2}] {lbl}: {neigh}")

    title = (f"Pairwise |cos| of clean-pair axes in "
             f"{'L=' + str(L) + ' soft-sheared' if L > 0 else 'raw'} space "
             f"(slot {slot}, layer {layer})")
    sub = (f"seriation via optimal leaf ordering, {args.linkage}-linkage; "
           f"brackets show clusters at |cos|≥{args.cluster_threshold}")
    fig.suptitle(title + "\n" + sub, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    # --- Provenance + write -------------------------------------------
    inputs: list[InputSpec] = [
        current_file_input(
            dep_key="producer_script",
            path=Path(__file__).resolve(),
            extras={
                "slot": str(slot), "layer": str(layer), "L": str(L),
                "linkage": args.linkage, "ca_kind": args.ca_kind,
                "pairs": args.pairs,
            },
        ),
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras={"slot": str(slot), "layer": str(layer)}),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras={"slot": str(slot), "layer": str(layer)}),
    ]
    plot_path = experiment_dir / f"{args.out_basename}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(title=title, inputs=inputs))
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # JSON sidecar with ordering + cluster memberships + the full
    # pairwise matrix and linkage so downstream consumers don't have
    # to recompute the (~30 sec) shear-and-multiply step.  The two
    # extra fields are small (60x60 matrix ~28 KB; 59x4 linkage
    # ~3 KB at default precision).
    #
    # The matrix is in orig-index order (== axis_labels order); the
    # seriated_order field provides the OLO permutation.  Reconstruct
    # the seriated matrix as M[leaf_order][:, leaf_order] where
    # leaf_order = [e["orig_index"] for e in seriated_order].
    json_payload = {
        "n_axes": n, "slot": slot, "layer": layer, "L": L,
        "ca_kind": args.ca_kind, "linkage": args.linkage,
        "axis_labels": list(labels),     # orig-index order
        "abs_cos_matrix": M.tolist(),    # n x n, float64; indexable by axis_labels
        "linkage_matrix": Z_ordered.tolist(),  # (n-1) x 4 scipy linkage
        "seriated_order": [
            {"pos": labels[i].split("/")[0], "neg": labels[i].split("/")[1],
             "label": labels[i], "orig_index": int(i),
             "seriated_index": int(rank)}
            for rank, i in enumerate(leaf_order)
        ],
        "clusters_by_threshold": {
            f"{t:.1f}": [
                {"label": labels[i], "cluster": int(clusterings[t][i])}
                for i in range(n)
            ] for t in cluster_thresholds
        },
        "pairwise_abs_cos": {
            "matrix_size": n,
            "min": float(M[np.triu_indices(n, 1)].min()),
            "max": float(M[np.triu_indices(n, 1)].max()),
            "mean": float(M[np.triu_indices(n, 1)].mean()),
            "median": float(np.median(M[np.triu_indices(n, 1)])),
        },
        "ontology": {
            "applied": bool(ontology_runs),
            "block_order": list(ontology_block_order),
            "runs": [
                {"block": blk, "start": int(s), "end": int(e),
                 "n_axes": int(e - s)}
                for blk, s, e in ontology_runs
            ],
            "block_descriptions": (
                CLEAN_COHORT_ONTOLOGY_DESCRIPTIONS
                if ontology_map is CLEAN_COHORT_ONTOLOGY else {}
            ),
            "unmapped_axes": [
                {"label": lbl, "seriated_index": seriated_labels.index(lbl)}
                for lbl in seriated_labels
                if lbl not in ontology_map
            ] if ontology_map else [],
        },
    }
    envelope = json_metadata(
        json_payload, inputs=inputs,
        title=f"axis_cosine_seriation {cohort} L={L} slot={slot}",
    )
    json_path = experiment_dir / f"{args.out_basename}.json"
    json_path.write_text(json.dumps(envelope, indent=2))
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
