"""Instruction judging under the 12 projection variants, in the layout of judge_tier_robustness.py (a
heatmap of mean rho above the 12 Pareto panels).  A judge in an arm judges every one of an entity's five
instructions, each separately in the batched rounds (Roger, 2026-11-03: each instruction judged at least
once by every model chosen), and Nx is how many times it judges each one: N passes over the five, each
pass in its own batch orders, to average out sampling noise (Roger: "Nx is for reducing with T=1 noise,
not inter-instruction variation").  A pass is five rounds of judge_instruction_grid.py --design rounds:
round r shows instruction r mod 5 in the entity's own order, batched by shuffle seed r+1, so pass p is
rounds 5p to 5p+4.  The arms: every set of judges (each alone, each pair, the trio) at 1x and 2x, Luna
alone also at 3x and 4x (EXTRA_PASSES), and the MIXES, Luna 2x with Haiku, Gemini or both at 1x, and
Luna 2x + Haiku 2x + Gemini 1x.

An entity's score is each judge's mean over all its judgements, the judges weighted Luna 4 to another
judge's 3 (judge_score_combine.TIER_JUDGE_WEIGHTS; Roger, 2026-11-03; equal before); a mix's weights are
fitted to the most mean rho at the default variant (grid of WEIGHT_STEP), with a split-half and a
leave-one-axis-out check of the fit against the default weights.  The passes a judge has (PASSES) are
exchangeable, so an arm's rho and cost are averaged over every choice of its passes, made for each judge
apart (Luna 2x: 6 choices of its 4 passes).  The rho of one choice is judge_tier_robustness.cell_rho
(compute_correlations, poles excluded) on its projections; a choice's cost is its rounds' usage.json at
--price_basis, so Nx costs N full passes.

An earlier version (2026-11-03) read Nx as N of the five instructions judged once; its figure is kept in
<output_dir>/coverage_reading/.

Jev (TypeSafe AI, 2026-11-04, Roger: "show Jev, Jev 2x, adding Jev to the trio, and replacing Gemini with Jev"):
the arms Jev 1x and 2x, the trio + Jev and Luna + Haiku + Jev (Gemini replaced) at 1x and 2x, Jev weighted 3 as
each other judge.  Its cells are scripts/judge_jev_entities.py's best arm, a Noul question asking whether each
entity is nearer the axis's second end than most of the other entities, 160 entities a request (JEV_CELLS), in
the axes' own order like every other cell here; ``--without_jev`` draws the figure without them.

Directions (Roger's rule of 2026-11-04, ``--directions balanced``, the default since that day): every cell is a
balanced one, each judge's passes alternating each axis's direction, so a 2x is one pass each way and a 1x half
the axes each way (descriptions) or the instructions alternating (instructions, Roger's table); an arm's choices of
passes take only those with as many passes each way (Luna 2x of 4: 4 choices, not 6).  Descriptions:
<judge>@b10n20d<seed> (judge_descriptions.py), Jev's @jd_noul_relative_n160d<seed>; instructions:
<judge>@i1_p0n20d<round> (judge_instruction_grid.py), Jev's @ji_noul_relative_n160d<round>.  Luna's instruction
passes 3 and 4 are its own-order rounds 10-19 (Roger, 2026-11-04: "Don't bother getting this exactly correct for
3x and 4x").  ``--directions own`` draws the figure of the cells in the axes' own order, as before.

``--mode descriptions`` draws the same arms for description judging (Roger, 2026-11-03: "confirm that this
also works well for description judging"): a pass is one cell of judge_descriptions.py's method
(``<judge>@b10n20s<seed>jl``, the entity's description once, N=20 a call, the judge display form of the
names), every judge with two passes and Luna four, so the arms are every set at 1x and 2x, Luna alone to 4x,
and the mixes.

Writes <stem>.json and <stem>.png to --output_dir.  No API calls.

Usage:
    uv run python scripts/judge_instruction_12way.py \\
        --output_dir roger/axis_judge_experiments/instruction_judging_12way
    uv run python scripts/judge_instruction_12way.py --mode descriptions \\
        --output_dir roger/axis_judge_experiments/instruction_judging_12way
"""
import argparse, itertools, json, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_axis import json_metadata
from assistant_axis.judge_score_combine import (TIER_JUDGE_DEFAULT_WEIGHT, TIER_JUDGE_WEIGHTS, combine_tier_judges,
                                                tier_judge_weights)
from assistant_axis.provenance import CACHE_POLICIES, current_file_input, current_files_input, load_and_register
from results_analysis import judge_tier_robustness as rob
from results_analysis.judge_tier_cost_eval import cell_scores_path, cell_usage_path
from results_analysis.judge_tier_cost_plot_pooled import DEFAULT_PRICE_BASIS, PRICE_BASES, load_measured_cost

JUDGES = (("Luna", "gpt-5.6-luna"), ("Haiku", "claude-haiku-5-5"), ("Gemini", "gemini-3.1-flash-lite"))
JEV = "jev-1.14.0"
NAMES = {**{m: n for n, m in JUDGES}, JEV: "Jev"}
#: Jev's cells by mode, (round or pass) -> cell directory name: relative Noul, 160 entities a request, own order
#: (scripts/judge_jev_entities.py); two passes in each mode (instruction rounds 0-9, description seeds 1 and 2).
JEV_CELLS = {"instructions": lambda r: f"{JEV}@ji_noul_relative_n160_r{r}",
             "descriptions": lambda r: f"{JEV}@jd_noul_relative_n160s{r + 1}"}
JEV_PASSES = 2
#: The balanced cells (--directions balanced): Jev's, and the other judges' by mode.
JEV_CELLS_BALANCED = {"instructions": lambda r: f"{JEV}@ji_noul_relative_n160d{r}",
                      "descriptions": lambda r: f"{JEV}@jd_noul_relative_n160d{r + 1}"}
BALANCED = True
#: The mode being drawn, for the judges' weights (judge_score_combine.TIER_JUDGE_WEIGHTS_BY_MODE); set in main.
MODE_NAME = None
#: The Jev arms: the judges of each, drawn at 1x and 2x.
JEV_SETS = ((("Jev", JEV),), (*JUDGES, ("Jev", JEV)), (JUDGES[0], JUDGES[1], ("Jev", JEV)))
WITH_JEV = True
CELL_ARM = "i1_p0n20"          # one instruction per entity per call, plain wrapper, N=20
ROUNDS_PER_PASS = 5            # an entity's five instructions, one per round
#: Passes judged (Roger, 2026-11-03: "take Luna to 4x, Haiku and Gemini to 2x")
PASSES = {"gpt-5.6-luna": 4, "claude-haiku-5-5": 2, "gemini-3.1-flash-lite": 2}
SET_PASSES = (1, 2)            # every set of judges
#: Passes beyond SET_PASSES for one judge alone, to extend the frontier (Luna; the others' further passes
#: "are just clutter")
EXTRA_PASSES = {"gpt-5.6-luna": (3, 4)}
#: Arms whose judges take different numbers of passes, each with judge weights fitted to maximise rho
#: (Roger, 2026-11-03: "Optimize the blend for each individually")
MIXES = ({"gpt-5.6-luna": 2, "claude-haiku-5-5": 1},
         {"gpt-5.6-luna": 2, "gemini-3.1-flash-lite": 1},
         {"gpt-5.6-luna": 2, "claude-haiku-5-5": 1, "gemini-3.1-flash-lite": 1},
         {"gpt-5.6-luna": 2, "claude-haiku-5-5": 2, "gemini-3.1-flash-lite": 1})
WEIGHT_STEP = 0.05
STEM = "judge_instruction_12way"
X_TICKS = (0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7)     # 0.02 for Jev 1x (2026-11-04)
#: Description judging: the cells of judge_descriptions.py's method under the judge display form (the
#: ``jl`` siblings, scripts/judge_tier_reask.py --mode names, or judged under that form); Luna's cell name
#: carries its reasoning-off token.
DESC_CELL_TOKEN = {"gpt-5.6-luna": "@none"}
DESC_CELL_ARM = "b10n20"
DESC_PASSES = {"gpt-5.6-luna": 4, "claude-haiku-5-5": 2, "gemini-3.1-flash-lite": 2}
DESC_EXTRA_PASSES = {"gpt-5.6-luna": (3, 4)}     # Roger, 2026-11-03: Luna 3x and 4x on the description plot too


@dataclass(frozen=True)
class Mode:
    """What a pass is, and which arms are drawn, for one kind of judging."""
    name: str
    cell: Callable[[str, int], str]       # (model, round) -> the cell directory name
    rounds_per_pass: int
    passes: Dict[str, int]
    set_passes: Tuple[int, ...]
    extra_passes: Dict[str, Tuple[int, ...]]
    mixes: Tuple[Dict[str, int], ...]
    arm_token: str                        # in the arm keys
    stem: str
    x_ticks: Tuple[float, ...]
    text: str                             # what a judge judges, for the title


MODES = {
    "instructions": Mode("instructions", lambda m, r: f"{m}@{CELL_ARM}r{r}", ROUNDS_PER_PASS, PASSES, SET_PASSES,
                         EXTRA_PASSES, MIXES, CELL_ARM, STEM, X_TICKS, "each of an entity's five instructions"),
    "descriptions": Mode("descriptions", lambda m, r: f"{m}{DESC_CELL_TOKEN.get(m, '')}@{DESC_CELL_ARM}s{r + 1}jl", 1,
                         DESC_PASSES, SET_PASSES, DESC_EXTRA_PASSES, MIXES, f"{DESC_CELL_ARM}jl", "judge_description_12way",
                         (0.003, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3),
                         "an entity's description"),
}
INSTRUCTIONS = MODES["instructions"]


def mix_key(spec: Dict[str, int], mode: Mode = INSTRUCTIONS) -> str:
    return "+".join(spec) + f"@{mode.arm_token}m" + "-".join(str(k) for k in spec.values())


def pass_rounds(p: int, rounds_per_pass: int = ROUNDS_PER_PASS) -> range:
    """The rounds of pass ``p``: each of the five instructions once (one round for a description)."""
    return range(rounds_per_pass * p, rounds_per_pass * (p + 1))


def cell_of(mode: "Mode", m: str, r: int) -> str:
    """The cell directory of judge ``m``'s round (or pass) ``r`` in ``mode``: the balanced cells unless
    --directions own (Luna's instruction rounds past 9 are its own-order ones either way)."""
    if not BALANCED:
        return JEV_CELLS[mode.name](r) if m == JEV else mode.cell(m, r)
    if m == JEV:
        return JEV_CELLS_BALANCED[mode.name](r)
    if mode.name == "instructions":
        return f"{m}@{CELL_ARM}d{r}" if r < 2 * ROUNDS_PER_PASS else mode.cell(m, r)
    return f"{m}@{DESC_CELL_ARM}d{r + 1}"


def passes_of(mode: "Mode") -> Dict[str, int]:
    """Each judge's passes in ``mode``, Jev's included when it is drawn."""
    return {**mode.passes, **({JEV: JEV_PASSES} if WITH_JEV else {})}


def arms(mode: Mode = INSTRUCTIONS) -> Dict[str, Tuple[str, Dict[str, int]]]:
    """``{arm key: (label, {model: passes})}``: singles, pairs, the trio, each at every k of the mode's
    set_passes; a judge alone at its extra_passes; then the mixes; then, unless --without_jev, the Jev sets
    (JEV_SETS) at 1x and 2x."""
    out = {}
    for size in (1, 2, 3):
        for combo in itertools.combinations(JUDGES, size):
            extra = mode.extra_passes.get(combo[0][1], ()) if size == 1 else ()
            for k in mode.set_passes + extra:
                key = "+".join(m for _, m in combo) + f"@{mode.arm_token}x{k}"
                out[key] = (" + ".join(n for n, _ in combo) + f" {k}x", {m: k for _, m in combo})
    for spec in mode.mixes:
        out[mix_key(spec, mode)] = (" + ".join(f"{NAMES[m]} {k}x" for m, k in spec.items()), dict(spec))
    if WITH_JEV:
        for combo in JEV_SETS:
            for k in (1, 2):
                key = "+".join(m for _, m in combo) + f"@{mode.arm_token}x{k}"
                out[key] = (" + ".join(n for n, _ in combo) + f" {k}x", {m: k for _, m in combo})
    return out


def series(mode: Mode = INSTRUCTIONS) -> List[List[str]]:
    """One arrow per judge set, from its fewest passes to its most (the mixes stand alone)."""
    by_set: Dict[str, List[str]] = {}
    for key, (_, spec) in arms(mode).items():
        if len(set(spec.values())) == 1:
            by_set.setdefault(key.rsplit("x", 1)[0], []).append(key)
    return list(by_set.values())


def replicates(spec: Dict[str, int], passes: Dict[str, int] = PASSES,
               rounds_per_pass: int = ROUNDS_PER_PASS) -> List[Dict[str, Tuple[int, ...]]]:
    """Every choice of passes for an arm, as each judge's rounds: each judge's k of its ``passes``,
    chosen apart from the other judges' (every pass holds every instruction, so there is nothing to
    pair).  Luna 2x of 4 passes: 6 choices; Luna 2x + Haiku 1x of 2: 12."""
    def balanced(c) -> bool:
        """With balanced directions, an even number of passes takes as many each way (pass p's direction flips
        with p's parity); an odd number cannot be balanced and takes any."""
        return not BALANCED or len(c) % 2 or sum(p % 2 for p in c) * 2 == len(c)
    per = [[tuple(r for p in c for r in pass_rounds(p, rounds_per_pass)) for c in itertools.combinations(range(passes[m]), k)
            if balanced(c)] for m, k in spec.items()]
    return [dict(zip(spec, choice)) for choice in itertools.product(*per)]


def judge_means(per_judge: Dict[str, Sequence[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Each judge's mean over its rounds (a round without the entity is left out of its mean)."""
    means = {}
    for m, rounds in per_judge.items():
        keys = set().union(*rounds)
        means[m] = {k: float(np.mean([r[k] for r in rounds if k in r])) for k in keys}
    return means


def combine(per_judge: Dict[str, Sequence[Dict[str, float]]],
            weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """An entity's score: each judge's mean over its rounds, then the judges weighted by ``weights``, or
    by judge_score_combine.tier_judge_weights (Luna 4, another judge 3; Roger, 2026-11-03).  A judge without
    the entity is left out of that entity's mean."""
    return combine_tier_judges(judge_means(per_judge), weights, MODE_NAME)


def weight_grid(n: int, step: float = WEIGHT_STEP) -> np.ndarray:
    """Every weight vector over ``n`` judges on a grid of ``step``, summing to 1."""
    m = int(round(1 / step))
    return np.array([w for w in itertools.product(range(m + 1), repeat=n) if sum(w) == m], float) / m


def fit_rho(spec: Dict[str, int], axes: Sequence["rob.AxisDef"], scores: Dict[Tuple[str, str, int], Dict[str, float]],
            projection: Dict[str, Dict[str, float]], grid: np.ndarray, passes: Dict[str, int] = PASSES,
            rounds_per_pass: int = ROUNDS_PER_PASS) -> np.ndarray:
    """``[weights, axis]`` Spearman rho at one projection (``{axis: {entity: value}}``), averaged over the
    arm's choices of passes.  For speed the entities are those every judge scored with a finite
    projection, poles out: the weights are chosen on this, the plotted rho is arm_table's."""
    models = list(spec)
    out = np.zeros((len(grid), len(axes)))
    reps = replicates(spec, passes, rounds_per_pass)
    for j, ax in enumerate(axes):
        proj, poles = projection[ax.axis_dir], {ax.pos, ax.neg}
        for rep in reps:
            means = judge_means({m: [scores[ax.axis_dir, m, r] for r in rep[m]] for m in models})
            names = [n for n, p in proj.items() if np.isfinite(p) and rob._bare(n) not in poles
                     and all(n in means[m] for m in models)]
            P = rankdata([proj[n] for n in names])
            P = (P - P.mean()) / P.std()
            R = rankdata(grid @ np.array([[means[m][n] for n in names] for m in models]), axis=1)
            R = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
            out[:, j] += (R * P).mean(1)
        out[:, j] /= len(reps)
    return out


def fit_weights(G: np.ndarray, grid: np.ndarray, dev: np.ndarray, models: Sequence[str]) -> dict:
    """The weights with the best mean rho over all axes, and how they fare out of sample against the
    default weights (tier_judge_weights): fitted on the development axes and scored on the held-out
    ones and the reverse, and each axis scored with weights fitted on the others."""
    default = np.array([tier_judge_weights(models, MODE_NAME)[m] for m in models])
    G_def = G[int(np.argmin(np.abs(grid - default).sum(1)))]   # the default, to the nearest grid point
    best = int(np.argmax(G.mean(1)))
    fit_dev, fit_held = int(np.argmax(G[:, dev].mean(1))), int(np.argmax(G[:, ~dev].mean(1)))
    split = np.where(dev, G[fit_held], G[fit_dev])
    tot = G.sum(1)
    loo = np.array([G[int(np.argmax(tot - G[:, j])), j] for j in range(G.shape[1])])
    return {"weights": dict(zip(models, map(float, grid[best]))),
            "default_weights_on_grid": dict(zip(models, map(float, grid[int(np.argmin(np.abs(grid - default).sum(1)))]))),
            "in_sample_gain": float(G[best].mean() - G_def.mean()),
            "split_half_gain": float((split - G_def).mean()),
            "split_half_weights": {"dev": dict(zip(models, map(float, grid[fit_dev]))),
                                   "held": dict(zip(models, map(float, grid[fit_held])))},
            "leave_one_axis_out_gain": float((loo - G_def).mean())}


def arm_table(arm_specs: Dict[str, Dict[str, int]], axes: Sequence["rob.AxisDef"],
              scores: Dict[Tuple[str, str, int], Dict[str, float]], costs: Dict[Tuple[str, str, int], float],
              projections: Dict[str, Dict[str, Dict[str, float]]], vkeys: Sequence[str],
              passes: Dict[str, int] = PASSES, weights: Optional[Dict[str, Dict[str, float]]] = None,
              rounds_per_pass: int = ROUNDS_PER_PASS) -> Tuple[dict, Dict[str, float], Dict[str, List[float]]]:
    """``(rho[arm][axis][variant], cost[arm], per-choice mean rho at each variant [arm][variant])``.
    ``scores`` and ``costs`` are keyed ``(axis, model, round)``; rho and cost are means over the arm's
    choices of passes, a choice's cost summed over its judges' rounds and averaged over axes.
    ``weights``: judge weights for some arms (the others take tier_judge_weights)."""
    rho, cost, per_choice = {}, {}, {}
    for arm, spec in arm_specs.items():
        reps = replicates(spec, passes, rounds_per_pass)
        r_axes = np.full((len(reps), len(axes), len(vkeys)), np.nan)
        c_axes = np.zeros((len(reps), len(axes)))
        w = (weights or {}).get(arm)
        for i, rep in enumerate(reps):
            for j, ax in enumerate(axes):
                s = combine({m: [scores[ax.axis_dir, m, r] for r in rounds] for m, rounds in rep.items()}, w)
                c_axes[i, j] = sum(costs[ax.axis_dir, m, r] for m, rounds in rep.items() for r in rounds)
                for v, vk in enumerate(vkeys):
                    r_axes[i, j, v] = rob.cell_rho(s, projections[ax.axis_dir][vk], (ax.pos, ax.neg))[0]
        rho[arm] = {ax.axis_dir: {vk: float(r_axes[:, j, v].mean()) for v, vk in enumerate(vkeys)}
                    for j, ax in enumerate(axes)}
        cost[arm] = float(c_axes.mean())
        per_choice[arm] = {vk: [float(x) for x in r_axes[:, :, v].mean(1)] for v, vk in enumerate(vkeys)}
    return rho, cost, per_choice


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(rob.DEFAULT_DATA_DIR))
    ap.add_argument("--experiments_dir", default=str(rob.DEFAULT_EXPERIMENTS_DIR))
    ap.add_argument("--axes_file", default="scripts/tier_eval_axes.tsv")
    ap.add_argument("--cache_dir", default=str(rob.DEFAULT_EXPERIMENTS_DIR / "_tier_eval_scratch" / "robustness_cache"),
                    help="judge_tier_robustness.py's slice cache (rebuilt there when the vectors changed)")
    ap.add_argument("--price_basis", choices=PRICE_BASES, default=DEFAULT_PRICE_BASIS)
    ap.add_argument("--mode", choices=list(MODES), default="instructions",
                    help="instruction judging (the i1_p0n20 rounds) or description judging (the b10n20 jl cells)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stem", default=None, help="output file names (default: the mode's)")
    ap.add_argument("--axes_label", default="61 axes")
    ap.add_argument("--dev_axes_file", default="scripts/tier_eval_axes_dev30.tsv",
                    help="the development half, for the split-half check of the mixes' fitted weights")
    ap.add_argument("--without_jev", action="store_true", help="leave the Jev arms out (the figure before 2026-11-04)")
    ap.add_argument("--directions", choices=("balanced", "own"), default="balanced",
                    help="balanced (default since 2026-11-04): the cells with each judge's passes alternating each axis's "
                         "direction, a 2x one pass each way; own: the cells in the axes' own order")
    ap.add_argument("--cache-policy", dest="cache_policy", choices=CACHE_POLICIES, default="warn",
                    help="what to do when a score file's recorded inputs have changed: strict | warn (default) | rebuild | off")
    args = ap.parse_args(argv)
    global WITH_JEV, BALANCED, MODE_NAME
    WITH_JEV = not args.without_jev
    BALANCED = args.directions == "balanced"
    t0 = time.time()
    mode = MODES[args.mode]
    MODE_NAME = mode.name
    stem = args.stem or mode.stem
    data_dir, E, out = Path(args.data_dir).resolve(), Path(args.experiments_dir), Path(args.output_dir)
    variants = rob.build_variants(rob.DEFAULT_SLOTS, rob.DEFAULT_LAYERS, list(rob.REGIMES),
                                  K=rob.DEFAULT_SOFT_K, L=rob.DEFAULT_SOFT_SHEAR_L)
    vkeys, default = [v.key for v in variants], rob.DEFAULT_VARIANT
    axes = rob.read_axes_tsv(Path(args.axes_file))
    kinds = rob.vector_kinds(data_dir)
    inputs = [rob._subtree_input(data_dir, "traits/vectors", "traits_vectors"),
              rob._subtree_input(data_dir, "roles/vectors", "roles_vectors")]
    for sub in ("r_goal", "r_nogoal", "t_goal", "t_nogoal"):
        inputs.append(rob._subtree_input(data_dir, f"combinations/vectors/derived/marginals/{sub}", f"marginals_{sub}"))
    inputs.append(rob._subtree_input(data_dir, "combinations/vectors/derived/axis", "theatricality_axis"))
    inputs.append(current_file_input("axes_file", Path(args.axes_file)))

    # Every round of every judge on every axis
    specs = {k: spec for k, (_, spec) in arms(mode).items()}
    models = sorted({m for spec in specs.values() for m in spec})
    scores, costs, score_paths, usage_paths, missing = {}, {}, [], [], []
    passes = passes_of(mode)
    rounds_of = {m: range(mode.rounds_per_pass * passes[m]) for m in models}
    for ax in axes:
        for m in models:
            for r in rounds_of[m]:
                d = E / f"{ax.axis_dir}{rob.TIER_EVAL_SUFFIX}" / cell_of(mode, m, r)
                up = cell_usage_path(d) if cell_scores_path(d).exists() else d / "usage.json"
                if not up.exists():
                    missing.append(str(d))
                    continue
                payload, _spec, _check = load_and_register(cell_scores_path(d), dep_key=f"scores_{ax.axis_dir}_{m}_r{r}",
                                                           inputs=None, policy=args.cache_policy)
                scores[ax.axis_dir, m, r] = rob.lift_scores(rob.numeric_scores(rob._unwrap(payload)), kinds)
                costs[ax.axis_dir, m, r] = load_measured_cost(up, args.price_basis)
                score_paths.append(cell_scores_path(d))
                usage_paths.append(up)
    if missing:
        raise SystemExit(f"{len(missing)} cells not finished, e.g. {missing[:3]}")
    inputs.append(current_files_input(dep_key=f"{mode.name[:-1]}_scores", paths=score_paths))
    inputs.append(current_files_input(dep_key=f"{mode.name[:-1]}_usage", paths=usage_paths,
                                      extras={"price_basis": args.price_basis}))
    print(f"[cells] {len(scores)} round cells (" + ", ".join(f"{NAMES[m]} {passes[m]} passes" for m in models)
          + f"; {len(axes)} axes), {time.time() - t0:.1f}s", file=sys.stderr)

    # Projections under the 12 variants
    meta = {"version": rob.CACHE_VERSION, "fingerprint": rob.data_fingerprint(data_dir), "ca_kind": "combined"}
    slices = rob.load_slices(Path(args.cache_dir), rob.DEFAULT_SLOTS, rob.DEFAULT_LAYERS, meta, True)
    if slices is None:
        print(f"[slices] building from {data_dir} (reads every vector file once)", file=sys.stderr)
        slices = rob.build_slices(data_dir, rob.DEFAULT_SLOTS, rob.DEFAULT_LAYERS, with_shear=True, ca_kind="combined")
        rob.save_slices(Path(args.cache_dir), slices, meta)
    names_by_axis = {ax.axis_dir: set().union(*(scores[ax.axis_dir, m, r] for m in models for r in rounds_of[m]))
                     for ax in axes}
    projections = rob.project_variants(slices, axes, variants, names_by_axis, data_dir)
    print(f"[projections] {time.time() - t0:.1f}s", file=sys.stderr)

    # The mixes' judge weights, fitted at the default variant
    dev_names = {l.split()[0] for l in open(args.dev_axes_file) if l.strip() and not l.startswith("#")}
    dev = np.array([ax.axis_dir in dev_names for ax in axes])
    labels = {k: lab for k, (lab, _) in arms(mode).items()}
    fits, weights = {}, {}
    for key, spec in specs.items():
        if len(set(spec.values())) > 1:
            grid = weight_grid(len(spec))
            G = fit_rho(spec, axes, scores, {a: p[default] for a, p in projections.items()}, grid, passes,
                        mode.rounds_per_pass)
            fits[key] = fit_weights(G, grid, dev, list(spec))
            weights[key] = fits[key]["weights"]
            labels[key] += " (" + ":".join(f"{w:.2f}" for w in weights[key].values()) + ")"
            f = fits[key]
            print(f"[fit] {labels[key]}: gain over Luna 4 : 3 others {f['in_sample_gain']:+.4f} in sample, "
                  f"{f['split_half_gain']:+.4f} split-half, {f['leave_one_axis_out_gain']:+.4f} leaving one axis out",
                  file=sys.stderr)
    print(f"[fit] {time.time() - t0:.1f}s", file=sys.stderr)

    rho, cost, per_choice = arm_table(specs, axes, scores, costs, projections, vkeys, passes, weights,
                                      mode.rounds_per_pass)
    print(f"[rho] {time.time() - t0:.1f}s", file=sys.stderr)
    keys = list(specs)
    per_variant, frontiers = {}, {}
    for vk in vkeys:
        per_variant[vk], frontiers[vk] = rob.summarise_variant(rho, cost, keys, vk)
    order = rob.ranking(per_variant[default], keys)
    counts = rob.frontier_counts(frontiers, keys)

    w = max(len(labels[k]) for k in keys)
    print(f"\n{'arm':{w}s} {'rho ' + default:>16s} {'choices sd':>10s} {'$/axis':>8s}  frontier of {len(vkeys)}")
    for k in order:
        ch = per_choice[k][default]
        print(f"{labels[k]:{w}s} {per_variant[default][k]['mean_rho']:16.4f} {np.std(ch):10.4f} {cost[k]:8.4f}  {counts[k]:2d}")

    out.mkdir(parents=True, exist_ok=True)
    extra = ", ".join(f"{NAMES[m]} alone to {max(k)}x" for m, k in mode.extra_passes.items())
    result = {"config": {"variants": vkeys, "default_variant": default, "price_basis": args.price_basis,
                         "mode": mode.name, "passes": passes, "rounds_per_pass": mode.rounds_per_pass,
                         "cell_arm": mode.arm_token, "axes_file": args.axes_file, "data_dir": str(data_dir),
                         "nx": f"{mode.text} judged N times, in N passes (pass p = rounds {mode.rounds_per_pass}p..)",
                         "passes_rule": "every choice of each judge's k passes, made for each judge apart",
                         "score_rule": "each judge's mean over all its judgements, judges weighted by tier_judge_weights, "
                                       "or for a mix by weights fitted at the default variant",
                         "judge_weights": {**TIER_JUDGE_WEIGHTS, "others": TIER_JUDGE_DEFAULT_WEIGHT},
                         "weight_step": WEIGHT_STEP, "dev_axes_file": args.dev_axes_file},
              "axes": [ax.axis_dir for ax in axes],
              "arms": {k: {"label": labels[k], "judges": specs[k],
                           "n_choices": len(replicates(specs[k], passes, mode.rounds_per_pass)),
                           "mean_cost": cost[k], "mean_rho": {vk: per_variant[vk][k]["mean_rho"] for vk in vkeys},
                           "on_frontier_of_12": counts[k], "per_choice_mean_rho": per_choice[k],
                           **({"fitted_weights": fits[k]} if k in fits else {})} for k in order},
              "frontier_sets": frontiers, "rho": rho}
    (out / f"{stem}.json").write_text(json.dumps(json_metadata(
        result, title=f"{mode.name.capitalize()[:-1]} judging by passes, {len(vkeys)} projection variants",
        inputs=inputs), indent=1))
    headline = (f"{mode.name.capitalize()[:-1]} judging, every set of judges at 1x and 2x"
                + (f" ({extra})" if extra else "")
                + ("; Jev alone, beside the trio and in Gemini's place" if WITH_JEV else "")
                + f", under {len(variants)} projection variants: {args.axes_label} (default {default})")
    jw = tier_judge_weights([m for _, m in JUDGES] + ([JEV] if WITH_JEV else []), MODE_NAME)
    lw = jw[JUDGES[0][1]]
    weights_text = " : ".join(f"{NAMES[m]} {w / lw * 4:g}" for m, w in jw.items())
    sub = (f"Nx = every judge judges {mode.text} N times (N passes, each in its own batch orders, N=20 a call"
           + ("; each judge's passes alternate each axis's direction, a 2x one pass each way" if BALANCED else
              "; every pass in the axes' own order") + f"); judges weighted {weights_text}; arrows run from 1x up\n"
           f"each point averaged over every choice of its passes (passes judged: " + ", ".join(
               f"{NAMES[m]} {n}" for m, n in passes.items()) + f"); mixes: judge weights (in parentheses) fitted to "
           f"the most rho at {default}\n"
           f"cost at {rob.PRICE_BASIS_LABEL[args.price_basis]}"
           + ("; Jev (relative Noul, 160 a call) at its one price, $0.042 per 1M input, output free" if WITH_JEV else "")
           + f"; soft_K: K={rob.DEFAULT_SOFT_K}; "
           f"soft_shear: L={rob.DEFAULT_SOFT_SHEAR_L}")
    rob.render_figure(out / f"{stem}.png", order, variants, per_variant, frontiers, {k: True for k in keys}, default,
                      len(axes), sub, inputs,
                      rob.FigureOptions(headline=headline, cell_labels=labels, series=series(mode), own_y=True, x_log=True,
                                        x_ticks=mode.x_ticks, x_log_pad=1.5, panel_w=4.6, panel_h=3.8))
    print(f"\nwrote {out / stem}.json and .png ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
