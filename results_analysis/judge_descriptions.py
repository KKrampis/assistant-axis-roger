#!/usr/bin/env python3
"""Judge axes from the entities' descriptions by the settled method.

The method, from the judge-tier evaluation of November 2026 (reports/judge_pareto/judge_tier_update_2026-11-01.md;
Roger, 2026-11-02, the judges revised 2026-11-03 and 2026-11-04): the batched percentile rubric (b10, the judge
places each entity by percentile among all of them on the axis), 20 entities per call, two passes, one each way,
no reasoning, each judge's passes averaged, for GPT-5.6 Luna, Haiku 5.5 and Gemini 3.1 Flash-Lite; and since
2026-11-04 Jev (TypeSafe AI) beside them by its own method, two passes, one each way (Roger: "the new standard is
(Luna + Haiku + Gemini + Jev) x2, with the reversal in the x2"; JEV_* below, scripts/judge_jev_entities.py).  The
four are blended Luna 4 : Haiku 3 : Gemini 3 : Jev 24 (judge_score_combine.TIER_JUDGE_WEIGHTS_BY_MODE, the Jev
weight fitted on the 12 projection variants: reports/judge_pareto/jev_judging.md).  On the 61 axes, balanced
cells, slot 6, layer 25, raw (rho against the activation projections, cost per axis at the real price, Flex and
batch; scripts/judge_instruction_12way.py --mode descriptions):

    gpt-5.6-luna alone, 1 pass                                  0.709   $0.011
    the three, 2 passes each (the default from 2026-11-03)      0.751   $0.076
    Jev alone, 2 passes                                         0.758   $0.008    frontier in 12 of 12
    Luna + Haiku + Jev, 2 passes each                           0.766   $0.050    frontier in 12 of 12
    the four, 2 passes each (the default)                       0.766   $0.084    frontier in 5 of 12

Roger's rule (2026-11-03), chosen over picking among twenty arms a few thousandths apart: every model we have,
since the cross-model ensemble helps and varies by axis, and as many passes as the one text per entity needs
(a second pass of one judge adds 0.02 here, against 0.005 on instructions, which have five texts).  Luna's third
and fourth passes add 0.007 and 0.004 for the price of a second judge, which adds more.  Reasoning adds nothing to
any judge, and N=5 and N=7 are never on a judge's frontier.  The defaults below are that decision: change them
here, in one place.  Instruction judging has its own settled method, scripts/judge_instruction_grid.py.

Directions (Roger, 2026-11-04): a judge's two passes show each axis once in its own order and once with its
poles swapped (a coin per axis says which pass is swapped; the swapped pass's scores are mapped back), the
default since that day.  The pole-order test (reports/judge_pareto/response_judging_plan.md) found that the
order shifts the judges' placements and that a judge does better with the morally good pole at the high
number; a fixed order would carry that valence asymmetry into every result, so the directions are balanced
instead (and on axes without a good pole, one pass each way also beats two in one order).  --directions own
judges every pass in the axis's own order, under the old labels.

Each pass is a cell of judge_tier_cost_eval.py, <axis>_tier_eval/<judge>@b10n<N>d<seed>/ (s<seed> with
--directions own), whose every answer is saved as it arrives, so a stopped or failed run is simply run again.
Then each judge's passes are averaged into <judge>@b10n<N>dx<passes> (x<passes>), and the judges into
<judge>+<judge>@b10n<N>dx<passes>.  Projections come from
<experiments_dir>/<axis>/gpt/projections.json; for new data write them with

    uv run python results_analysis/axis_judge_correlation.py --projections_only --pair POS NEG --pair_type traits \\
        --data_dir DATA --output_dir <experiments_dir>/POS_vs_NEG/gpt

Check the cost first (--dry_run), then, for example:

    uv run python results_analysis/judge_descriptions.py --axes_file scripts/tier_eval_axes.tsv --budget_usd 20

Exit status: 0 done, 1 a pass failed or a source was refused for averaging, and otherwise the failing pass's own
status (judge_tier_cost_eval.py): 2 a budget stop, 3 a stall, 4 calls left unanswered: run again.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant_axis.judge import provider_for_model  # noqa: E402
from results_analysis import judge_tier_average as avg  # noqa: E402
from results_analysis import judge_tier_cost_eval as ev  # noqa: E402
from results_analysis.judge_tier_rubrics import DEFAULT_NAME_FORMAT, NAME_FORMATS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#: The settled method (Roger, 2026-11-02).
DESCRIPTION_RUBRIC = "b10"
DESCRIPTION_ENTITIES_PER_CALL = 20
DESCRIPTION_PASSES = 2
#: Jev (TypeSafe AI) as a fourth judge (Roger, 2026-11-04: "the new standard is (Luna + Haiku + Gemini + Jev) x2,
#: with the reversal in the x2"): its own method, not b10 (scripts/judge_jev_entities.py: a Noul question per
#: entity, "is it nearer the axis's second end than most of the other entries?", 160 entities a request, the
#: value Jev's probability of yes), its passes balanced the same way as the others', its weight in
#: judge_score_combine.TIER_JUDGE_WEIGHTS.  Its key is TYPESAFE_API_KEY.
JEV_MODEL = "jev-1.14.0"
#: The name requested: TypeSafe's API serves only jev-latest and jev-preview (GET /v1/models, 2026-11-04), so the
#: drivers ask for jev-latest and require answers from JEV_MODEL (judge_jev_entities.py --expect_model).
JEV_REQUEST_MODEL = "jev-latest"
JEV_TOOL = Path(__file__).resolve().parents[1] / "scripts" / "judge_jev_entities.py"
JEV_ENTITIES_PER_CALL = 160
JEV_METHOD = ("--question", "noul", "--wording", "relative")
#: The judges (Roger, 2026-11-03: the three; 2026-11-04: with Jev).
DESCRIPTION_JUDGES = ("gpt-5.6-luna", "claude-haiku-5-5", "gemini-3.1-flash-lite", JEV_MODEL)
#: Supported and measured, not in the default: pass them as --judge_model.  GPT-6 Luna alone for the cheap end.
OPTIONAL_JUDGES = ("gpt-6-luna",)
#: The reply cap per entity.  No reply reached these in the evaluation; Haiku 5.5 writes about twice as much.
MAX_TOKENS_PER_ENTITY = 60
MAX_TOKENS_PER_ENTITY_BY_PROVIDER = {"anthropic": 150}
FIRST_SEED = 1
#: Balanced directions (Roger, 2026-11-04): "balanced", a judge's passes alternating the axis's direction
#: (judge_tier_cost_eval.py --directions descriptions); "own", every pass in the axis's own order.
DESCRIPTION_DIRECTIONS = "balanced"
#: Judges run side by side and a judge's passes one after another, so that at most this many processes hit the
#: providers at once (eight Flex processes dropped every connection on 2026-11-01; four are fine).
MAX_CONCURRENT_JUDGES = 4

TOOL = Path(__file__).with_name("judge_tier_cost_eval.py")


def max_tokens_for(model: str, entities_per_call: int) -> int:
    return entities_per_call * MAX_TOKENS_PER_ENTITY_BY_PROVIDER.get(provider_for_model(model), MAX_TOKENS_PER_ENTITY)


def is_jev(model: str) -> bool:
    return model.startswith("jev")


def jev_pass_label(seed: int, directions: str = DESCRIPTION_DIRECTIONS) -> str:
    """A Jev pass's cell label (judge_jev_entities.py's arm name; the 61-axis cells of 2026-11-04 carry it)."""
    return f"@jd_noul_relative_n{JEV_ENTITIES_PER_CALL}{'d' if directions == 'balanced' else 's'}{seed}"


def jev_averaged_label(passes: int, directions: str = DESCRIPTION_DIRECTIONS) -> str:
    return f"@jd_noul_relative_n{JEV_ENTITIES_PER_CALL}{'d' if directions == 'balanced' else ''}x{passes}"


def judge_pass_label(model: str, entities_per_call: int, seed: int, directions: str = DESCRIPTION_DIRECTIONS) -> str:
    return jev_pass_label(seed, directions) if is_jev(model) else pass_label(entities_per_call, seed, directions)


def pass_label(entities_per_call: int, seed: int, directions: str = DESCRIPTION_DIRECTIONS) -> str:
    """A pass's cell label: ``d<seed>`` for balanced directions, ``s<seed>`` for the axis's own order, so that a
    cell of one kind is never taken for the other."""
    return f"@{DESCRIPTION_RUBRIC}n{entities_per_call}{'d' if directions == 'balanced' else 's'}{seed}"


def averaged_label(entities_per_call: int, passes: int, directions: str = DESCRIPTION_DIRECTIONS) -> str:
    return f"@{DESCRIPTION_RUBRIC}n{entities_per_call}{'d' if directions == 'balanced' else ''}x{passes}"


def pass_argv(args: argparse.Namespace, model: str, seed: int) -> List[str]:
    """The tool's arguments for one judge's one pass."""
    argv = ["--judge_model", model, "--rubric", DESCRIPTION_RUBRIC, "--entities_per_call", str(args.entities_per_call),
            "--shuffle_seed", str(seed), "--cell_label", pass_label(args.entities_per_call, seed, args.directions),
            "--max_tokens", str(max_tokens_for(model, args.entities_per_call)), "--experiments_dir", args.experiments_dir,
            "--instructions_dir", args.instructions_dir, "--name_format", args.name_format, "--transport", args.transport]
    if args.directions == "balanced":
        argv += ["--directions", "descriptions", "--direction_pass", str(seed - args.first_seed)]
    if args.axes_file:
        argv += ["--axes_file", args.axes_file]
    else:
        pos, neg = args.pair
        axis = f"{pos}_vs_{neg}"
        argv += ["--pair", pos, neg, "--pair_type", args.pair_type,
                 "--projections_file", str(Path(args.experiments_dir) / axis / "gpt" / "projections.json"),
                 "--output_dir", str(Path(args.experiments_dir) / f"{axis}{ev.TIER_EVAL_SUFFIX}")]
    for flag, value in (("--budget_usd", args.budget_usd), ("--limit", args.limit),
                        ("--requests_per_second", args.requests_per_second), ("--max_in_flight", args.max_in_flight)):
        if value is not None:
            argv += [flag, str(value)]
    if args.dry_run:
        argv.append("--dry_run")
    return argv


def jev_pass_argv(args: argparse.Namespace, model: str, seed: int) -> List[str]:
    """judge_jev_entities.py's arguments for one Jev pass: the same axes, seed and direction pass as the other
    judges' pass of that seed (so its batches are its own, 160 a request, and its directions the same coin's)."""
    argv = ["--mode", "descriptions", "--model", JEV_REQUEST_MODEL, "--expect_model", model, *JEV_METHOD, "--entities_per_call", str(JEV_ENTITIES_PER_CALL),
            "--shuffle_seed", str(seed), "--cell_label", jev_pass_label(seed, args.directions),
            "--experiments_dir", args.experiments_dir, "--instructions_dir", args.instructions_dir]
    if args.directions == "balanced":
        argv += ["--directions", "descriptions", "--direction_pass", str(seed - args.first_seed)]
    if args.axes_file:
        argv += ["--axes_file", args.axes_file]
    else:
        argv += ["--pair", *args.pair, "--pair_type", args.pair_type]
    for flag, value in (("--budget_usd", args.budget_usd), ("--limit", args.limit), ("--max_in_flight", args.max_in_flight)):
        if value is not None:
            argv += [flag, str(value)]
    if args.dry_run:
        argv.append("--dry_run")
    return argv


def run_tool(argv: Sequence[str], tool: Path = TOOL) -> int:
    """One pass, as its own process: the tool keeps a run's settings in module state, and restarts itself after a
    stall (os.execv)."""
    return subprocess.run([sys.executable, str(tool), *argv], check=False).returncode


def run_judge(args: argparse.Namespace, model: str) -> int:
    """A judge's passes, one after another; the first failure ends them."""
    for seed in range(args.first_seed, args.first_seed + args.passes):
        code = (run_tool(jev_pass_argv(args, model, seed), JEV_TOOL) if is_jev(model)
                else run_tool(pass_argv(args, model, seed)))
        if code:
            logger.error(f"{model}{judge_pass_label(model, args.entities_per_call, seed, args.directions)}: "
                         f"exit status {code}")
            return code
    return 0


def jobs_of(args: argparse.Namespace) -> List["ev.AxisJob"]:
    E = Path(args.experiments_dir)
    if args.axes_file:
        return ev.jobs_from_axes_file(Path(args.axes_file), E)
    pos, neg = args.pair
    axis = f"{pos}_vs_{neg}"
    return [ev.AxisJob(axis, pos, neg, args.pair_type, E / axis / "gpt" / "projections.json",
                       E / f"{axis}{ev.TIER_EVAL_SUFFIX}")]


def average_all(args: argparse.Namespace, models: Sequence[str]) -> List[str]:
    """Each judge's passes into one cell, and the judges into a blend.  Returns the refusals."""
    N, K = args.entities_per_call, args.passes
    seeds = range(args.first_seed, args.first_seed + K)
    D = args.directions
    passes: Dict[str, List[str]] = {m: [f"{m}{judge_pass_label(m, N, s, D)}" for s in seeds] for m in models}
    E, data, jobs = Path(args.experiments_dir), Path(args.instructions_dir), jobs_of(args)
    refused: List[str] = []
    recipes = [(f"{m}{jev_averaged_label(K, D) if is_jev(m) else averaged_label(N, K, D)}", passes[m])
               for m in models if K > 1]
    if len(models) > 1:
        recipes.append((f"{'+'.join(models)}{averaged_label(N, K, D)}", [c for m in models for c in passes[m]]))
    for out, sources in recipes:
        r = avg.average_cells(out, sources, jobs, E, data, limit=args.limit)
        refused += [f"{out}: {line}" for line in r.refused]
        print(f"{out}: {r.made} cells derived, {r.kept} up to date, {len(r.refused)} refused")
    return refused


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axes_file", help="'axis_dir pos neg pair_type' rows (scripts/tier_eval_axes.tsv); or --pair")
    p.add_argument("--pair", nargs=2, metavar=("POS", "NEG"))
    p.add_argument("--pair_type", choices=["roles", "traits"])
    p.add_argument("--experiments_dir", default=ev.DEFAULT_EXPERIMENTS_DIR)
    p.add_argument("--instructions_dir", default="data", help="the corpus being judged (default: data)")
    p.add_argument("--judge_model", action="append", default=None,
                   help=f"Repeatable (default: {', '.join(DESCRIPTION_JUDGES)}; also measured: {', '.join(OPTIONAL_JUDGES)})")
    p.add_argument("--entities_per_call", type=int, default=DESCRIPTION_ENTITIES_PER_CALL,
                   help=f"default {DESCRIPTION_ENTITIES_PER_CALL}")
    p.add_argument("--passes", type=int, default=DESCRIPTION_PASSES, help=f"random orders per judge (default {DESCRIPTION_PASSES})")
    p.add_argument("--first_seed", type=int, default=FIRST_SEED, help=f"shuffle seed of the first pass (default {FIRST_SEED})")
    p.add_argument("--directions", choices=("balanced", "own"), default=DESCRIPTION_DIRECTIONS,
                   help=f"balanced (default): a judge's passes alternate each axis's direction, so with two passes "
                        f"every axis is judged once each way by every judge (an odd count leaves one pass over, "
                        f"its direction a coin per axis); own: every pass in the axis's own order")
    p.add_argument("--name_format", choices=list(NAME_FORMATS), default=DEFAULT_NAME_FORMAT,
                   help="how prompts show entity names (judge_tier_cost_eval.py)")
    p.add_argument("--transport", choices=list(ev.tt.TRANSPORT_CHOICES), default="auto",
                   help="default auto: GPT-5.6 Luna by Flex, Haiku 5.5 by Message Batches, Gemini by its Batch API "
                        "at half price (since 2026-11-03; realtime for its generateContent calls)")
    p.add_argument("--budget_usd", type=float, default=None,
                   help="per pass of a judge: stop issuing live and Flex requests past this measured spend (a batch "
                        "is submitted whole)")
    p.add_argument("--limit", type=int, default=None, help="only the first N entities of each axis, to try a judge cheaply")
    p.add_argument("--requests_per_second", type=int, default=None)
    p.add_argument("--max_in_flight", type=int, default=None)
    p.add_argument("--dry_run", action="store_true", help="the cost estimate of each pass; no calls, nothing written")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.axes_file) == bool(args.pair):
        parser.error("give --axes_file or --pair, one of them")
    if args.pair and not args.pair_type:
        parser.error("--pair needs --pair_type")
    if args.passes < 1 or args.entities_per_call < 2:
        parser.error("--passes must be at least 1 and --entities_per_call at least 2")
    models = list(dict.fromkeys(args.judge_model or DESCRIPTION_JUDGES))
    if any(is_jev(m) for m in models) and args.name_format != DEFAULT_NAME_FORMAT:
        parser.error(f"Jev's passes show names as {DEFAULT_NAME_FORMAT} only: drop --name_format, or Jev "
                     f"(--judge_model for the others)")
    with ThreadPoolExecutor(max_workers=min(len(models), MAX_CONCURRENT_JUDGES)) as pool:
        codes = list(pool.map(lambda m: run_judge(args, m), models))
    failed = {m: c for m, c in zip(models, codes) if c}
    if failed:
        logger.error(f"not averaged, since passes did not finish: {failed}; run again, and finished work is kept")
        return next(iter(failed.values()))
    if args.dry_run:
        return 0
    refused = average_all(args, models)
    for line in refused:
        logger.error(f"REFUSED {line}")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
