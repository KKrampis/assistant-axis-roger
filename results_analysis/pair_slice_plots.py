#!/usr/bin/env python3
"""2D "slice" plots for a list of axis pairs.

For each ``(pos, neg)`` pair we define an oriented 2D plane:

- **origin**   = the trait mean (in the chosen whitening regime).
- **y-axis**   = the unit vector ``(pos − neg) / |pos − neg|`` -- pos at top.
- **x-axis**   = the orthogonal projection of ``(midpoint − trait_mean)`` into
                 the plane, then unitised.  This is the "common-mode"
                 direction shared by both poles relative to the trait mean.

Every trait and role in the corpus is then projected into the plane,
labelled with a greedy collision-free annotation algorithm (priority is
distance from origin plus a hand-curated semantic bonus, with the pair's
own poles always shown).  Each plot saves a PNG to ``<out_dir>/`` named
``{pos}_vs_{neg}_slice_K{K}.png`` and prints a compact semantic summary
(``d_pos``, ``d_neg``, ``d_mid``, ``||m||/||diff||``) to stdout.

The curated pairs in :data:`DEFAULT_PAIRS` are the 13 axis pairs
(out of the 33 with judging data) for which::

    |midpoint - trait_mean| / |pos - neg|  >=  0.5

at slot 3, layer 25, K=3 soft whitening -- i.e. axes whose +/- pole
pair sits noticeably *off* the trait-mean origin, indicating a
substantial common-mode component shared by both poles relative to
the corpus.  Sorted by that ratio descending.  See the project README
for the full ranked list and the headline ``13 / 33`` count.

Two of the 13 carry an explicit ``y_flag`` describing where the
y-direction's actual meaning departs from the pair name -- typically
because the trait description leaned into a hyperbolic / over-loaded
framing of one pole (e.g. "anthropocentric" written as
"sociopathic/dishonest" rather than "humans-as-moral-focus";
"unforgiving" written as "zealous/cruel/evil" rather than "holds
grudges").

Whitening
---------

Uses the standard ``canonical_angles.data.build_augmented_whitening_pool``
(roles + traits + ``default.pt``, **no** leave-out -- these are
visualisation plots, not held-out statistics) followed by
``fit_whitening("soft_K", pool, K=K)``.  This matches the K-sweep
tooling, ``rho_by_slot_and_K.py``, ``rho_by_layer.py``, etc., so soft-K
geometry is consistent across the whole project.

Output
------

For each pair, a PNG to ``--out_dir`` (default
``roger/axis_judge_experiments/pair_slices``).

Examples
--------

::

    # Default: all curated near-miss pairs at slot 3, layer 25, K=3
    uv run python results_analysis/pair_slice_plots.py

    # One specific pair at non-default whitening
    uv run python results_analysis/pair_slice_plots.py \\
        --pairs progressive,conservative --K 4 \\
        --out_dir /tmp/slices_K4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from assistant_axis.entity_id import corpus_display_name
from assistant_axis import (
    kind_color,
    kind_marker,
    kind_text_style,
    png_metadata,
    suptitle_with_specs,
)
from assistant_axis.provenance import (
    InputSpec,
    current_data_subtree_input,
)
from results_analysis.axis_judge_correlation import _load_vector_file
from results_analysis.canonical_angles.data import (
    build_augmented_whitening_pool,
)
from results_analysis.canonical_angles.whitening import (
    DEFAULT_SOFT_K,
    fit_whitening,
)


# Override the canonical-angles 4-slot default with the 8-slot Roger
# dataset; the 8-slot data is what ships slot 6 (</think>) and slot 7
# (\\n\\n post), which the new --slot=6 default needs.  Pass --data_dir to
# point at the older 4-slot Christina-headers data for back-compat.
LOCAL_DEFAULT_DATA_DIR = (Path(__file__).resolve().parent.parent
                          / "runpod_workspace/qwen/qwen-3-32b Roger 8slot")
DEFAULT_DATA_DIR = LOCAL_DEFAULT_DATA_DIR
DEFAULT_OUT_DIR = (Path(__file__).resolve().parent.parent
                   / "roger/axis_judge_experiments/pair_slices")


# Curated pair list.  Each entry: (pos, neg, x_label, y_label, y_flag).
#   x_label  -- semantic name of the x (common-mode) direction
#               (format: "x : <-x extreme>  ↔  <+x extreme>").
#   y_label  -- semantic name of the y direction
#               (format: "y : <-y / neg pole>  ↔  <+y / pos pole>";
#               should match the pair, but...).
#   y_flag   -- if non-empty, the y-direction does NOT cleanly match
#               the pair name; the string describes what the axis
#               actually measures.
#
# This curated set is exactly the 13 axis pairs (out of the 33 with
# judging data) for which ``|midpoint - trait_mean| / |pos - neg| >= 0.5``
# at slot 3, layer 25, K=3 soft whitening -- the pairs whose +/- pole
# pair sits noticeably off-axis from the trait mean, indicating a
# significant common-mode component in the joint pole geometry.  Sorted
# by that ratio descending.  Labels were originally inferred at L24/K4
# (existing 7 pairs) and L25/K3 (new 6 pairs) by reading the +/-x and
# +/-y extreme entities; re-run the analysis (see
# ``--pairs <single pair>``'s stdout summary) if you switch defaults.
DEFAULT_PAIRS: list[tuple[str, str, str, str, str]] = [
    ("systems_thinker", "analytical",
     "x : social-affective manner  ↔  cognitive-frame entity",
     "y : literal / methodical / robotic  ↔  gestalt / paradoxical / chaotic",
     ""),
    ("relativist", "absolutist",
     "x : playful / casual affect  ↔  firm philosophical worldview",
     "y : absolutist / prescriptive  ↔  relativist / fluid / open",
     ""),
    ("ecocentric", "anthropocentric",
     "x : casual entertainment style  ↔  moral-circle scope",
     "y : sociopathic / narcissistic / dishonest  ↔  ecocentric / spiritual / forgiving",
     "moral-circle-scope intrinsically correlates with selfish-vs-altruistic; "
     "compounded by dominionist tilt in anthropocentric.json.  "
     "Rewrite worthwhile -- see TRAITS_TO_ADD.md."),
    ("individualistic", "collectivistic",
     "x : performative / theatrical manner  ↔  stance about social organization",
     "y : collectivistic / cooperative  ↔  contrarian / cynical / sardonic",
     "warmth-asymmetric pole descriptions: collectivistic side carries "
     "cooperation/harmony vocabulary that individualistic.json (cleanly "
     "self-direction-focused) lacks.  See TRAITS_TO_ADD.md."),
    ("reductionist", "holistic",
     "x : social-affective manner  ↔  intellectual frame",
     "y : holistic / flexible / open-ended  ↔  reductive / literal / prescriptive",
     ""),
    ("progressive", "conservative",
     "x : a-political (style/affect, no political stance)  ↔  political-social stance",
     "y : conservative / traditional / dutiful  ↔  progressive / visionary / animated",
     ""),
    ("egalitarian", "elitist",
     "x : playful / casual / a-political affect  ↔  stance about social hierarchy / in-group scope",
     "y : narcissistic / arrogant / petty  ↔  egalitarian / inclusive / humanistic",
     ""),
    ("convergent", "divergent",
     "x : evil / arrogant / antisocial character  ↔  cognitive-style / problem-solving frame",
     "y : exploratory / open-ended / poetic  ↔  convergent / focused / efficient",
     ""),
    ("casual", "formal",
     "x : deep moral / philosophical / ideological stance (substance)  ↔  communicative style / register / craft (form)",
     "y : formal / scholarly / meticulous / restrained  ↔  casual / playful / silly / spontaneous",
     ""),
    ("forgiving", "unforgiving",
     "x : casual / unstructured affect  ↔  stance about handling conflict / wrongdoing",
     "y : zealous / vindictive / cruel  ↔  forgiving / empathetic / compassionate",
     "axis legitimately projects onto a prosocial / antisocial dimension "
     "(forgiveness is a Care-foundation moral disposition); compounded by "
     "\"un-forgiving\" linguistic asymmetry.  Not artifact -- keep as-is."),
    ("practical", "theoretical",
     "x : social-affective character  ↔  cognitive frame / mode of thought",
     "y : theoretical / abstract / philosophical  ↔  practical / efficient / grounded",
     ""),
    ("improvisational", "methodical",
     "x : social-arrogant / antisocial character  ↔  cognitive / problem-solving style",
     "y : methodical / precise / formalist  ↔  improvisational / flexible / risk-taking",
     ""),
    ("concise", "verbose",
     "x : emotional / affective character traits  ↔  communicative-precision / register",
     "y : verbose / erudite / scholarly  ↔  concise / brief / plain-spoken",
     ""),
]


# Semantic priority bonus for collision-free labelling.  Higher = more
# likely to survive collision resolution and appear on the plot.  Pole
# names get a per-plot dynamic bonus on top of this.
SEMANTIC_BONUS: dict[str, float] = {
    "helpful": 1e6, "unhelpful": 1e6,
    "compassionate": 60, "callous": 60, "malicious": 60,
    "harmful": 55, "harmless": 55, "constructive": 40, "destructive": 40,
    "sociopathic": 45, "cruel": 40, "hostile": 35, "malevolent": 40,
    "benign": 40, "supportive": 45, "empathetic": 40, "benevolent": 45,
    "nurturing": 30, "inspirational": 30, "educational": 30,
    "honest": 35, "dishonest": 35, "trustworthy": 30, "untrustworthy": 30,
    "cooperative": 30, "competitive": 25, "guileless": 30, "scheming": 30,
    "selfish": 30, "philanthropic": 25, "humanitarian": 25,
    "chaotic": 35, "paradoxical": 30, "ascetic": 30, "existentialist": 30,
    "materialist": 25, "deterministic": 25, "ethereal": 30, "mystical": 30,
    "fatalistic": 25, "nihilistic": 35, "absolutist": 25, "clannish": 20,
    "incurious": 35, "flippant": 25, "opaque": 25, "vague": 20, "careless": 25,
    "sycophantic": 35, "obedient": 25, "rebellious": 25,
    "passionate": 25, "dispassionate": 25,
    "progressive": 30, "conservative": 30,
    "individualistic": 30, "collectivistic": 30,
    "relativist": 30, "casual": 25, "formal": 25,
    "reductionist": 25, "holistic": 25, "analytical": 25, "systems_thinker": 25,
    "ecocentric": 25, "anthropocentric": 25,
    # roles
    "assistant": 100, "tutor": 50, "mentor": 45, "teacher": 40,
    "therapist": 40, "counselor": 35, "coach": 30, "guide": 35,
    "caregiver": 45, "doctor": 30, "instructor": 30,
    "angel": 70, "saint": 70, "bodhisattva": 70, "demon": 60,
    "villain": 45, "hero": 45,
    "paperclip_maximizer": 80, "aligned_artificial_intelligence": 80,
    # aligned_artificial_intelligence was renamed 2026-09-28; both stems
    # are listed so plots of older extractions keep their priority
    "instrumentally_aligned_ai": 80, "virtue_aligned_ai": 80,
    "scientist": 40, "philosopher": 40, "engineer": 30, "scholar": 30,
    "priest": 30, "judge": 25,
    "soldier": 20, "warrior": 20, "rogue": 20, "trickster": 20,
    "clown": 20, "jester": 20,
    "chimera": 45, "aberration": 40, "exile": 35, "crystalline": 30,
    "poet": 30, "absurdist": 30, "loner": 35, "purist": 25,
    "wraith": 35, "eldritch": 35, "void": 40,
    "hedonist": 30, "dreamer": 30, "prey": 25, "mystic": 35,
    "monk": 35, "hermit": 35, "nihilist": 35, "anarchist": 30,
    "rebel": 25, "revolutionary": 25, "reactionary": 25,
    "traditionalist": 25, "libertarian": 25, "authoritarian": 30,
    "activist": 25, "bureaucrat": 20,
    "mathematician": 25, "researcher": 25,
}


def _load_at(path: Path, slot: int, layer: int) -> np.ndarray:
    return _load_vector_file(path).float()[slot, layer].numpy()


def load_entities(data_dir: Path, slot: int, layer: int, K: int):
    """Load all (default-centered) trait+role vectors and apply soft-K
    whitening.  Returns ``(names, vecs, trait_mean, cent_fn, w_fn)``::

        names      -- {"trait": [...], "role": [...]}
        vecs       -- {"trait": (N_t, D), "role": (N_r, D)} after whitening
        trait_mean -- mean of the whitened trait vectors (the slice origin)
        cent_fn    -- helper: cent(etype, name) -> default-centered vector
        w_fn       -- helper: w(v) -> whitened vector (1-D)
    """
    default_v = _load_at(data_dir / "traits" / "vectors" / "default.pt",
                         slot, layer)

    def cent(etype: str, name: str) -> np.ndarray:
        return _load_at(data_dir / etype / "vectors" / f"{name}.pt",
                        slot, layer) - default_v

    pool_entries = build_augmented_whitening_pool(
        data_dir, leave_out=set(), scope="roles+traits")
    pool_rows = []
    for (etype, name) in pool_entries:
        sub = "combinations" if etype == "combinations" else etype
        try:
            pool_rows.append(_load_at(
                data_dir / sub / "vectors" / f"{name}.pt", slot, layer))
        except FileNotFoundError:  # pragma: no cover -- skip missing
            continue
    pool = np.stack(pool_rows, axis=0)
    basis = fit_whitening("soft_K", pool, K=K)

    def w(v: np.ndarray) -> np.ndarray:
        return basis.apply(v[None, :])[0]

    names: dict[str, list[str]] = {"trait": [], "role": []}
    vecs: dict[str, list[np.ndarray]] = {"trait": [], "role": []}
    for etype, kind in (("traits", "trait"), ("roles", "role")):
        for fp in sorted((data_dir / etype / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            names[kind].append(fp.stem)
            vecs[kind].append(w(_load_at(fp, slot, layer) - default_v))
    vecs_arr: dict[str, np.ndarray] = {
        "trait": np.stack(vecs["trait"], axis=0),
        "role": np.stack(vecs["role"], axis=0),
    }
    trait_mean = vecs_arr["trait"].mean(axis=0)
    return names, vecs_arr, trait_mean, cent, w


def plot_pair(pos_name: str, neg_name: str,
              x_label: str, y_label: str, y_flag: str,
              names: dict, vecs: dict, trait_mean: np.ndarray,
              cent_fn, w_fn,
              *,
              slot: int, layer: int, K: int,
              out_dir: Path,
              inputs: list[InputSpec] | None = None) -> Path:
    """Render one pair slice and save it.  Returns the output path."""
    pos_w = w_fn(cent_fn("traits", pos_name))
    neg_w = w_fn(cent_fn("traits", neg_name))
    y_hat = (pos_w - neg_w) / np.linalg.norm(pos_w - neg_w)
    mid = 0.5 * (pos_w + neg_w)
    cm = mid - trait_mean
    x_raw = cm - (cm @ y_hat) * y_hat
    x_hat = x_raw / np.linalg.norm(x_raw)

    t_xs = (vecs["trait"] - trait_mean) @ x_hat
    t_ys = (vecs["trait"] - trait_mean) @ y_hat
    r_xs = (vecs["role"] - trait_mean) @ x_hat
    r_ys = (vecs["role"] - trait_mean) @ y_hat

    hx, hy = (pos_w - trait_mean) @ x_hat, (pos_w - trait_mean) @ y_hat
    ux, uy = (neg_w - trait_mean) @ x_hat, (neg_w - trait_mean) @ y_hat
    mx, my = (mid - trait_mean) @ x_hat, (mid - trait_mean) @ y_hat

    d_pos = float(np.linalg.norm(pos_w - trait_mean))
    d_neg = float(np.linalg.norm(neg_w - trait_mean))
    d_mid = float(np.linalg.norm(mid - trait_mean))
    diff = float(np.linalg.norm(pos_w - neg_w))
    midpoint_ratio = (2 * d_mid / diff) if diff > 0 else float("nan")

    # ----- Build candidate label list with priority ordering. -----
    candidates: list[tuple[str, str, float, float, str, float]] = []
    for n, x, y in zip(names["trait"], t_xs, t_ys):
        side = "L" if x < 0 else "R"
        bonus = SEMANTIC_BONUS.get(n, 0.0)
        if n in (pos_name, neg_name):
            bonus = 5e5  # poles, rank just below the helpful/unhelpful 1e6 anchors
        prio = 5 + bonus + np.sqrt(x * x + y * y) / 5
        candidates.append(("trait", n, float(x), float(y), side, float(prio)))
    for n, x, y in zip(names["role"], r_xs, r_ys):
        side = "L" if x < 0 else "R"
        bonus = SEMANTIC_BONUS.get(n, 0.0)
        prio = 5 + bonus + np.sqrt(x * x + y * y) / 5
        candidates.append(("role", n, float(x), float(y), side, float(prio)))
    candidates.sort(key=lambda c: -c[5])
    candidates = candidates[:200]

    # ----- Figure setup. -----
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(t_xs, t_ys, s=25, color=kind_color("traits"),
               zorder=2, edgecolor="none", marker=kind_marker("traits"))
    ax.scatter(r_xs, r_ys, s=30, color=kind_color("roles"),
               zorder=3, edgecolor="none", marker=kind_marker("roles"))
    ax.scatter([hx], [hy], s=220, color="green",
               edgecolor="black", zorder=6, marker="o")
    ax.scatter([ux], [uy], s=220, color="red",
               edgecolor="black", zorder=6, marker="o")
    ax.scatter([0], [0], s=300, color="gold",
               edgecolor="black", zorder=7, marker="*")
    ax.scatter([mx], [my], s=150, color="orange",
               edgecolor="black", zorder=6, marker="X")
    ax.axhline(0, color="grey", lw=0.4, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.4, alpha=0.5)
    ax.plot([hx, ux], [hy, uy], "k--", alpha=0.35, lw=1, zorder=1)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)

    # Token-label convention (matches assistant_axis.axis.slot_labels):
    # slot 0 = body-mean, slot 1 = <|im_start|>, slot 2 = assistant,
    # slot 3 = \n (the post-assistant newline -- the canonical
    # analysis position for this project).
    slot_label = {0: "body-mean", 1: "<|im_start|>",
                  2: "assistant", 3: r"\n"}.get(slot, f"slot {slot}")
    title_line = (f"{pos_name} vs {neg_name} slice "
                  f"(origin = trait mean; K={K} soft, {slot_label}, L{layer})")
    spec_lines = [
        f"d_pos={d_pos:.1f}  d_neg={d_neg:.1f}  "
        f"d_mid={d_mid:.1f}  ||m||/||diff||={midpoint_ratio:.2f}",
    ]
    if y_flag:
        spec_lines.append(f"⚠ y-axis does NOT cleanly match pair name: "
                          f"{y_flag}")
    _, top_rect = suptitle_with_specs(fig, title_line, spec_lines)

    legend_items = [
        Line2D([], [], marker="o", color="green", markersize=11,
               markeredgecolor="black", linestyle="",
               label=f"{pos_name} (+pole)"),
        Line2D([], [], marker="o", color="red", markersize=11,
               markeredgecolor="black", linestyle="",
               label=f"{neg_name} (−pole)"),
        Line2D([], [], marker="*", color="gold", markersize=14,
               markeredgecolor="black", linestyle="",
               label="trait mean (origin)"),
        Line2D([], [], marker="X", color="orange", markersize=10,
               markeredgecolor="black", linestyle="", label="midpoint"),
        Line2D([], [], marker=kind_marker("traits"),
               color=kind_color("traits"), markersize=6,
               markeredgecolor="none", linestyle="", label="trait"),
        Line2D([], [], marker=kind_marker("roles"),
               color=kind_color("roles"), markersize=6,
               markeredgecolor="none", linestyle="", label="role"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    fig.canvas.draw()

    def _side_offset(side: str):
        return ((-5, 5), "right") if side == "L" else ((5, 3), "left")

    def add_label(kind: str, name: str, x: float, y: float, side: str,
                  is_pole: bool = False):
        offset, ha = _side_offset(side)
        # Project-wide kind→{color,fontstyle} convention from
        # assistant_axis.plot_palette.  Labels use the corpus display
        # form (trait ``positive_label`` / role override) via
        # ``corpus_display_name``; never the file-form stem.
        kind_long_name = "roles" if kind == "role" else "traits"
        disp = corpus_display_name(name, kind_long_name)
        if is_pole:
            return ax.annotate(disp, (x, y), fontsize=11, fontweight="bold",
                               xytext=(7, 5), textcoords="offset points")
        style = kind_text_style(kind_long_name)
        return ax.annotate(disp, (x, y), fontsize=8, xytext=offset,
                           textcoords="offset points", ha=ha, **style)

    kept_bboxes = []
    for nm, xx, yy in [(pos_name, hx, hy), (neg_name, ux, uy)]:
        ann = add_label("trait", nm, xx, yy, "R", is_pole=True)
        fig.canvas.draw()
        kept_bboxes.append(ann.get_window_extent().padded(1))

    kept_count = 2
    for (kind, name, x, y, side, _prio) in candidates:
        if name in (pos_name, neg_name):
            continue
        ann = add_label(kind, name, x, y, side)
        fig.canvas.draw()
        bb = ann.get_window_extent().padded(1)
        if any(bb.overlaps(kb) for kb in kept_bboxes):
            ann.remove()
        else:
            kept_bboxes.append(bb)
            kept_count += 1

    out = out_dir / f"{pos_name}_vs_{neg_name}_slice_K{K}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=inputs))
    plt.close(fig)

    # Compact semantic summary for stdout.
    print(f"\n=== {pos_name} / {neg_name} ===")
    print(f"  d_pos={d_pos:.1f}  d_neg={d_neg:.1f}  d_mid={d_mid:.1f}  "
          f"||m||/||diff||={midpoint_ratio:.2f}")
    print(f"  Kept {kept_count} labels")
    combined = ([(nm, x, y, "R") for nm, x, y in zip(names["role"], r_xs, r_ys)]
                + [(nm, x, y, "T") for nm, x, y in zip(names["trait"], t_xs, t_ys)])
    print("  Top 6 +x (common-mode, shared with pair): "
          + ", ".join(f"{n}({k})"
                      for n, x, y, k in sorted(combined, key=lambda e: -e[1])[:6]))
    print("  Top 6 -x : "
          + ", ".join(f"{n}({k})"
                      for n, x, y, k in sorted(combined, key=lambda e: e[1])[:6]))
    print(f"  Top 6 +y ({pos_name}-ward): "
          + ", ".join(f"{n}({k})"
                      for n, x, y, k in sorted(combined, key=lambda e: -e[2])[:6]))
    print(f"  Top 6 -y ({neg_name}-ward): "
          + ", ".join(f"{n}({k})"
                      for n, x, y, k in sorted(combined, key=lambda e: e[2])[:6]))
    print(f"  Wrote {out}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                   help=f"Activation vectors directory "
                        f"(default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--out_dir", default=None,
                   help=f"Output directory (default: {DEFAULT_OUT_DIR}_slot{{N}} "
                        f"-- the slot suffix matches --slot so multi-slot runs "
                        f"don't overwrite each other).")
    p.add_argument("--slot", type=int, default=6,
                   help="Token-slot index (default: 6 = </think>; new judge-ρ "
                        "winner from May 2026 rejudge).  Pass --slot 3 (\\n) or "
                        "7 (\\n\\n post) to compare.")
    p.add_argument("--layer", type=int, default=25,
                   help="Transformer layer (default: 25 -- Qwen-3-32B "
                        "optimum from rho_by_layer.py).")
    p.add_argument("--K", type=int, default=DEFAULT_SOFT_K,
                   help=f"Soft-K whitening (default: {DEFAULT_SOFT_K} -- "
                        f"the project soft-K default; see "
                        f"whitening.DEFAULT_SOFT_K).")
    p.add_argument("--pairs", default=None,
                   help="Optional comma-separated pair name list to filter "
                        "DEFAULT_PAIRS, e.g. 'progressive,conservative' to "
                        "pick a single pair.  When the supplied set has 2 "
                        "names, they're matched as a single (pos, neg) "
                        "pair; with more, any pair whose pos AND neg are "
                        "in the set is included.")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if args.out_dir is None:
        args.out_dir = f"{DEFAULT_OUT_DIR}_slot{args.slot}"
    out_dir = Path(args.out_dir).resolve()

    pairs_to_run: list[tuple[str, str, str, str, str]]
    if args.pairs:
        wanted = {s.strip() for s in args.pairs.split(",") if s.strip()}
        if len(wanted) == 2:
            pairs_to_run = [t for t in DEFAULT_PAIRS
                            if {t[0], t[1]} == wanted]
        else:
            pairs_to_run = [t for t in DEFAULT_PAIRS
                            if t[0] in wanted and t[1] in wanted]
        if not pairs_to_run:
            raise SystemExit(f"No pair in DEFAULT_PAIRS matches {sorted(wanted)}")
    else:
        pairs_to_run = list(DEFAULT_PAIRS)

    print(f"Loading entities at slot={args.slot} layer={args.layer} K={args.K}",
          flush=True)
    names, vecs, trait_mean, cent_fn, w_fn = load_entities(
        data_dir, args.slot, args.layer, args.K)
    print(f"Loaded {len(names['trait'])} traits and {len(names['role'])} roles")

    # Per-pair PNGs all share the same upstream subtrees: traits + roles
    # vectors at the chosen (slot, layer), whitened via soft_K=K on the
    # augmented pool (which itself is roles+traits+default).
    extras = {"slot": str(args.slot), "layer": str(args.layer),
              "K": str(args.K)}
    inputs: list[InputSpec] = [
        current_data_subtree_input(
            data_dir, "traits/vectors", dep_key="traits_vectors",
            extras=extras),
        current_data_subtree_input(
            data_dir, "roles/vectors", dep_key="roles_vectors",
            extras=extras),
    ]
    for (pos, neg, x_label, y_label, y_flag) in pairs_to_run:
        plot_pair(pos, neg, x_label, y_label, y_flag,
                  names, vecs, trait_mean, cent_fn, w_fn,
                  slot=args.slot, layer=args.layer, K=args.K,
                  out_dir=out_dir, inputs=inputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
