"""Run a grid of instruction-judging cells through judge_tier_cost_eval.py, each cell as its own process,
as judge_descriptions.py runs the passes of description judging.  A cell already finished is skipped by the
tool's resume rule, so a rerun only judges what is missing.

--design sets: each --passes of each batch size N is the cell <judge>@is_<arm>n<N>s<seed>.
--design rounds: one pass of all --rounds, the cells <judge>@i1_<arm>n<N>r<round>, each round in its own
shuffle (seed round + 1); round r shows instruction r mod 5, so rounds 0-4 are one pass over an entity's
five instructions and rounds 5-9 a second.

**The settled method for instruction judging (Roger, 2026-11-04: "(Luna + Haiku + Gemini + Jev) x2, and just do
both directions in the x2")** is this script's defaults: two passes (rounds 0-9) of each of GPT-5.6 Luna, Haiku 5.5
and Gemini 3.1 Flash-Lite, the plain wrapper (p0) and 20 entities a call, and of Jev (TypeSafe AI) by its own
method (scripts/judge_jev_entities.py: a Noul question per entity, "is it nearer the second end than most of the
other entries?", 160 entities a request; its cells jev-1.14.0@ji_noul_relative_n160d<round>), and, once every
cell is in, the blend <judges>@i1_p0n20dx2, each judge's mean over its ten rounds weighted by
judge_score_combine.TIER_JUDGE_WEIGHTS (judge_tier_average.py; the scores in scores_instructions.json,
correlations.json beside them).  Before 2026-11-04 the method was one pass of the three (Roger, 2026-11-03).
Account: results_analysis/README.md, "Instruction judging: the settled method".

**Directions (Roger, 2026-11-04), the default of the rounds design since that day:** each round is judged by
each judge in the direction of Roger's table (judge_tier_cost_eval.INSTRUCTION_DIRECTIONS, --directions
instructions): each instruction both ways across the judges (2:1), each judge 3:2, by weight 27:23 or 23:27,
per axis a coin for which way is which and a permutation of the instruction positions, a second pass the
first reversed; swapped calls' scores mapped back.  The pole-order test found a judge does better with the
morally good pole at the high number, and a fixed order would carry that asymmetry into every result.  The
cells are <judge>@i1_<arm>n<N>d<round> and the blend <judges>@i1_<arm>n<N>dx<passes>; --directions own gives
the old r<round> and x<passes> cells, all in the axis's own order.  Jev's directions are its own column of the
table (judge_jev_entities.JEV_INSTRUCTION_COLUMN, B A B A d): with the three columns, each of the first four
instruction positions is judged 2:2, and with two passes every judge sees every instruction once each way.

Usage: uv run python scripts/judge_instruction_grid.py --design rounds --axes_file AXES.tsv --instructions_dir DATA --dry_run
       uv run python scripts/judge_instruction_grid.py --design sets --arms p0,p1,q0,q1,t0,t1 --ns 20 --passes 2 \\
           --axes_file scripts/tier_eval_axes_dev30.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data --dry_run"""
import argparse, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from assistant_axis.judge import provider_for_model
from results_analysis import judge_tier_average as avg, judge_tier_cost_eval as ev
from results_analysis.judge_descriptions import max_tokens_for

TOOL = REPO / "results_analysis" / "judge_tier_cost_eval.py"
JEV_TOOL = REPO / "scripts" / "judge_jev_entities.py"
#: Requested by this versioned ID, pinned, and every answer required to come from it (judge_descriptions.JEV_MODEL).
JEV_MODEL = "jev-1.14.0"
JEV_ENTITIES_PER_CALL = 160
JEV_METHOD = ("--question", "noul", "--wording", "relative")
#: The settled instruction judges (Roger, 2026-11-03: the three; 2026-11-04: with Jev, two passes each)
INSTRUCTION_JUDGES = ("gpt-5.6-luna", "claude-haiku-5-5", "gemini-3.1-flash-lite", JEV_MODEL)
ROUNDS_PER_PASS = 5
DEFAULT_ROUNDS = "0,1,2,3,4,5,6,7,8,9"


def is_jev(model: str) -> bool:
    return model.startswith("jev")


def jev_round_label(r: int, directions: str = "own") -> str:
    """A Jev round's cell label (judge_jev_entities.py's arm name; the 61-axis cells of 2026-11-04 carry it)."""
    return f"@ji_noul_relative_n{JEV_ENTITIES_PER_CALL}{'d' if directions == 'balanced' else '_r'}{r}"


def jev_command(model: str, r: int, args) -> list:
    """judge_jev_entities.py's command for Jev's round ``r``: instruction r mod 5, seed r + 1, its own batches of
    160, its column of Roger's direction table."""
    argv = [sys.executable, str(JEV_TOOL), "--mode", "instructions", "--instruction_round", str(r), "--model", model,
            "--expect_model", model,
            *JEV_METHOD, "--entities_per_call", str(JEV_ENTITIES_PER_CALL), "--shuffle_seed", str(r + 1),
            "--cell_label", jev_round_label(r, args.directions), "--axes_file", args.axes_file,
            "--experiments_dir", args.experiments_dir, "--instructions_dir", args.instructions_dir,
            "--max_in_flight", str(min(args.max_in_flight, 16))]
    if args.directions == "balanced":
        argv += ["--directions", "instructions"]
    if args.budget_usd_per_cell is not None:
        argv += ["--budget_usd", str(args.budget_usd_per_cell)]
    if args.dry_run:
        argv += ["--dry_run"]
    return argv


def round_label(arm: str, n: int, r: int, directions: str = "own") -> str:
    """A round's cell label: ``d<round>`` with balanced directions, ``r<round>`` in the axis's own order, so that
    a cell of one kind is never taken for the other."""
    return f"@i1_{arm}n{n}{'d' if directions == 'balanced' else 'r'}{r}"


def blend_label(arm: str, n: int, rounds, directions: str = "own") -> str:
    """The blend cell of a rounds design, ``@i1_<arm>n<N>x<passes>`` (``dx`` with balanced directions); None
    unless the rounds are whole passes from round 0 (rounds 5-9 alone are a second pass, not a cell called x1)."""
    if len(rounds) % ROUNDS_PER_PASS or sorted(rounds) != list(range(len(rounds))):
        return None
    return f"@i1_{arm}n{n}{'d' if directions == 'balanced' else ''}x{len(rounds) // ROUNDS_PER_PASS}"


def cells(design: str, arms, ns, passes: int, rounds, directions: str = "own"):
    """``(rubric, label, N, seed, round or None)`` for every cell of one judge."""
    for arm in arms:
        for n in ns:
            if design == "sets":
                for seed in range(1, passes + 1):
                    yield f"is_{arm}", f"@is_{arm}n{n}s{seed}", n, seed, None
            else:
                for r in rounds:
                    yield f"i1_{arm}", round_label(arm, n, r, directions), n, r + 1, r


def command(model: str, cell, args) -> list:
    rubric, label, n, seed, round_ = cell
    argv = [sys.executable, str(TOOL), "--judge_model", model, "--rubric", rubric, "--cell_label", label,
            "--entities_per_call", str(n), "--shuffle_seed", str(seed), "--max_tokens", str(max_tokens_for(model, n)),
            "--axes_file", args.axes_file, "--experiments_dir", args.experiments_dir, "--instructions_dir", args.instructions_dir]
    if round_ is not None:
        argv += ["--instruction_round", str(round_)]
        if getattr(args, "directions", "own") == "balanced":
            argv += ["--directions", "instructions"]
    if args.budget_usd_per_cell is not None:
        argv += ["--budget_usd", str(args.budget_usd_per_cell)]
    argv += ["--max_in_flight", str(args.max_in_flight), "--transport", args.transport]
    if args.dry_run:
        argv += ["--dry_run"]
    return argv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=("sets", "rounds"), required=True)
    ap.add_argument("--arms", default="p0", help="comma-separated wrapper and sentence codes: p0,p1,q0,q1,t0,t1")
    ap.add_argument("--ns", default="20", help="comma-separated batch sizes")
    ap.add_argument("--passes", type=int, default=2, help="--design sets: shuffles per batch size")
    ap.add_argument("--rounds", default=DEFAULT_ROUNDS, help=f"--design rounds: rounds to judge (default {DEFAULT_ROUNDS}, "
                                                              f"two passes)")
    ap.add_argument("--judges", default=",".join(INSTRUCTION_JUDGES),
                    help=f"comma-separated (default the settled four, {','.join(INSTRUCTION_JUDGES)})")
    ap.add_argument("--no_blend", action="store_true",
                    help="--design rounds: do not derive the blend cell <judges>@i1_<arm>n<N>x<passes> (Luna 4 : others "
                         "3, judge_tier_average.py) after the cells finish")
    ap.add_argument("--directions", choices=("balanced", "own"), default=None,
                    help="--design rounds: balanced (the default since 2026-11-04, Roger's table) or own (every round in "
                         "the axis's own order); --design sets: own only")
    ap.add_argument("--axes_file", required=True)
    ap.add_argument("--experiments_dir", default="roger/axis_judge_experiments")
    ap.add_argument("--instructions_dir", required=True, help="the corpus to judge (the May pin for the May axes)")
    ap.add_argument("--budget_usd_per_cell", type=float, default=None, help="--budget_usd of each Flex or real-time cell")
    ap.add_argument("--parallel", type=int, default=4, help="cells running at once, per judge")
    ap.add_argument("--max_in_flight", type=int, default=50,
                    help="Flex and real-time requests in flight per cell (8 cells at the tool's 200 gave 1,500 connection "
                         "errors a cell and lost Anthropic batch uploads, 2026-11-02)")
    ap.add_argument("--transport", default="auto", choices=("auto", "flex", "batch", "realtime"),
                    help="each cell's --transport (auto: Luna by Flex, Haiku by Message Batches, Gemini by its "
                         "Batch API at half price; realtime for Gemini's generateContent calls)")
    ap.add_argument("--logs", default=None, help="a directory for each cell's log (default: inherit stdout and stderr)")
    ap.add_argument("--dry_run", action="store_true", help="each cell's dry run: nothing sent, nothing written")
    args = ap.parse_args(argv)
    if args.directions is None:
        args.directions = "balanced" if args.design == "rounds" else "own"
    if args.directions == "balanced":
        if args.design != "rounds":
            ap.error("--directions balanced goes with --design rounds (a set shows all five instructions in one item)")
        unknown = [m for m in args.judges.split(",") if m not in ev.INSTRUCTION_DIRECTION_JUDGES and not is_jev(m)]
        if unknown:
            ap.error(f"--directions balanced: Roger's table has no column for {', '.join(unknown)} "
                     f"({', '.join(ev.INSTRUCTION_DIRECTION_JUDGES)}); use --directions own")
    if args.design == "sets" and any(is_jev(m) for m in args.judges.split(",")):
        ap.error("--design sets: Jev judges rounds only (pass --judges without it)")
    arms, ns = args.arms.split(","), [int(x) for x in args.ns.split(",")]
    rounds = [int(x) for x in args.rounds.split(",")]
    # Anthropic's first, one judge at a time: a batch upload is lost when Flex cells hold the connection
    judges = sorted(args.judges.split(","), key=lambda m: (True, True) if is_jev(m) else (False, provider_for_model(m) != "anthropic"))
    jobs = [(m, c) for m in judges if not is_jev(m) for c in cells(args.design, arms, ns, args.passes, rounds, args.directions)]
    jobs += [(m, ("jev", jev_round_label(r, args.directions), JEV_ENTITIES_PER_CALL, r + 1, r))
             for m in judges if is_jev(m) for r in rounds]
    print(f"{len(jobs)} cells: {args.design}, arms {arms}, batch sizes {ns}, judges {args.judges}, axes {args.axes_file}")
    logs = Path(args.logs) if args.logs else None
    if logs:
        logs.mkdir(parents=True, exist_ok=True)

    def run(job) -> int:
        model, cell = job
        cmd = jev_command(model, cell[4], args) if is_jev(model) else command(model, cell, args)
        if not logs:
            return subprocess.run(cmd, cwd=REPO).returncode
        with open(logs / f"{model}{cell[1]}.log", "a") as f:
            return subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode

    codes = []
    for judge in judges:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            codes += list(pool.map(run, [j for j in jobs if j[0] == judge]))
    bad = [(f"{m}{c[1]}", code) for (m, c), code in zip(jobs, codes) if code]
    for name, code in bad:
        print(f"{name}: exit status {code}", file=sys.stderr)
    print(f"{len(jobs) - len(bad)} of {len(jobs)} cells finished")
    if bad or args.dry_run or args.design != "rounds" or args.no_blend:
        return 1 if bad else 0
    # The blend of the judges' rounds, one cell per arm and batch size (the judges in the order given)
    refused = []
    given = args.judges.split(",")
    for arm in arms:
        for n in ns:
            label = blend_label(arm, n, rounds, args.directions)
            if label is None:
                print(f"no blend for rounds {rounds}: not whole passes from round 0", file=sys.stderr)
                continue
            out = "+".join(given) + label
            sources = [f"{m}{jev_round_label(r, args.directions) if is_jev(m) else round_label(arm, n, r, args.directions)}"
                       for m in given for r in rounds]
            E = Path(args.experiments_dir)
            r = avg.average_cells(out, sources, ev.jobs_from_axes_file(Path(args.axes_file), E), E, Path(args.instructions_dir))
            refused += [f"{out}: {line}" for line in r.refused]
            print(f"{out}: {r.made} cells derived, {r.kept} up to date, {len(r.refused)} refused")
    for line in refused:
        print(f"REFUSED {line}", file=sys.stderr)
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
