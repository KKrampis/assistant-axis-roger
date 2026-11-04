"""Description and instruction judging with TypeSafe AI's Jev, a batch of entities per request (Roger, 2026-11-04:
"Jev, like the other models, would likely benefit from seeing a batch of descriptions or instructions.  We can ask it
multiple questions, so we can get it to return multiple scores ... see if you can get Jev to somewhere cost/quality
competitive, at least as a fourth element to add to our ensemble").

Jev answers typed questions about one ``state`` per request: every question sees the whole state and is answered on
its own, the state is billed once per request, output is free ($0.042 per million input tokens; up to 64k tokens a
request and 32k a question; docs.typesafe.ai, read 2026-11-04).  So one request judges a batch: the state holds the
axis (its two ends and their descriptions) and N entities, numbered, each with its name, kind and text (its
description, or one of its instructions, one a round, as judge_tier_cost_eval.py's ``i1_*`` rounds); the questions
are one per entity, each placing that entity on the axis.  The batches are judge_tier_cost_eval.py's own
(``axis_groups``: the axis's entities in a seeded shuffle cut into groups of N), so a cell of seed s holds exactly
the groups the other judges' cells of seed s were judged in.

A question is (``--question``):

``score``  a Score question of ``--levels`` ordered levels (2 to 10), each read on its own, so each names both ends;
           the value is Jev's probability-weighted level.
``choice`` a Choice question of ``--levels`` options "0".."L-1" (up to 255), each described like a level; the value
           is the probability-weighted option number, computed here.
``noul``   a yes/no question ("is the entity nearer the second end than the first?"); the value is Jev's
           probability of yes.
``pairs``  a yes/no question for each pair of entries in the batch ("is entry j nearer the second end than entry
           k?"; all pairs, or with ``--pair_ring d`` each entry against its d neighbours either side); an entry's
           value is its expected share of wins, which, the batch-mates being a random sample of the axis's
           entities, estimates its percentile among them.

Every value is scaled to 0..100, the axis's first end at 0.  Directions (``--directions``, Roger's standing rule of
2026-11-04: never judge in one fixed pole order, balance it; AGENT_NOTES.md "Axis direction in judge prompts"): a
swapped call shows the axis with its ends exchanged and its values are mapped back (100 - v), and every answer
record says which way its call went.  ``descriptions``: a judge's passes alternate, by judge_tier_cost_eval's own
coin per axis (``--direction_pass`` 0 for the first pass, 1 for the second), as the trio's passes do.
``instructions``: Jev's column of Roger's table (judge_tier_cost_eval.INSTRUCTION_DIRECTIONS), JEV_INSTRUCTION_COLUMN,
which with the trio's columns judges each of the first four instruction positions twice each way.  ``all``: every
call swapped (to measure the order).  ``own``: every call in the axis's own order (the runs before the rule).  The wording of the state and the questions is
``--wording`` (WORDINGS below).  Every reply is kept whole in the answer log
(``<experiments_dir>/_tier_eval_jev/<label>/answers.jsonl``), probabilities included, so other ways of reading a
score out of the same answers cost nothing; a stopped run resumes from it, and a run under other settings is
refused.  The cells are judge_tier_cost_eval.py's: ``<axis>_tier_eval/<model><label>/`` with
``scores_<mode>.json``, ``correlations.json`` and ``usage.json`` (the axis's own requests).  The key is
TYPESAFE_API_KEY, from the environment or a .env, never printed or written.

Usage:
    uv run python scripts/judge_jev_entities.py --mode descriptions --axes_file scripts/tier_eval_axes_dev30.tsv \\
        --instructions_dir /tmp/claude-501/corpus_2026-05-24/data --question score --levels 5 \\
        --entities_per_call 20 --shuffle_seed 1 --cell_label @jd_s5n20s1 --dry_run
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_axis.atomic_io import atomic_write_text  # noqa: E402
from assistant_axis.judge import warn_if_low_parse_rate  # noqa: E402
from assistant_axis.judge_pricing import MultiModelUsage, price_for_model  # noqa: E402
from results_analysis import judge_tier_cost_eval as ev  # noqa: E402
from results_analysis import judge_tier_rubrics as jr  # noqa: E402
from results_analysis.axis_judge_correlation import compute_correlations  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_responses_jev as jrj  # noqa: E402  -- the client, retries and log reading

logger = logging.getLogger("judge_jev_entities")
#: A versioned ID, pinned: TypeSafe accepts one in the request's ``model`` field "whether or not they appear in the
#: list" (docs.typesafe.ai/models.md; GET /v1/models lists only the aliases jev-latest and jev-preview), and an alias
#: "moves when a new release ships, so its answers can change" (checked live 2026-11-04: jev-1.14.0 and jev-1.13.0
#: each answer as themselves, a malformed or unknown ID is an HTTP 400).
DEFAULT_MODEL = "jev-1.14.0"
#: A versioned model ID, as against an alias; a run that requests one requires every answer to come from it.
VERSIONED_ID = re.compile(r"jev-\d+\.\d+\.\d+")
JEV_DIRNAME = "_tier_eval_jev"
CHARS_PER_TOKEN = 4
MODES = {"descriptions": "b10", "instructions": "i1_p0"}
#: Jev's column of Roger's instruction-direction table (judge_tier_cost_eval.INSTRUCTION_DIRECTIONS, whose columns
#: are Luna, Haiku and Gemini): with theirs, instruction positions 1-4 are judged 2:2 by the four judges, the fifth
#: 3:1, and Jev 3:2 or 2:3 per axis, as each of the others.
JEV_INSTRUCTION_COLUMN = ("B", "A", "B", "A", "d")
DIRECTION_MODES = ("own", "descriptions", "instructions", "all")


def call_swapped(directions: str, axis: str, direction_pass: int, instruction_round: Optional[int]) -> bool:
    """Whether a call on ``axis`` shows the axis with its ends exchanged (a pure function of the run's settings and
    the axis: every call of an axis in a pass goes one way, as judge_tier_cost_eval's description and instruction
    designs do)."""
    if directions == "own":
        return False
    if directions == "all":
        return True
    if directions == "descriptions":
        return bool((direction_pass + ev._coin("descriptions", axis)) % 2)
    if directions == "instructions":
        n = len(ev.INSTRUCTION_DIRECTIONS)
        r = instruction_round or 0
        letter = JEV_INSTRUCTION_COLUMN[ev.instruction_positions(axis)[r % n]]
        d = ev._coin("instructions", "d", axis)
        is_b = {"A": 0, "B": 1, "d": d, "-d": 1 - d}[letter]
        return bool(is_b ^ ev._coin("instructions", "A", axis) ^ ((r // n) % 2))
    raise ValueError(f"unknown --directions {directions!r}")


#: --pair_ring: with --question pairs, each entry against this many neighbours either side (None: every pair).
PAIR_RING: Optional[int] = None


def pair_list(n: int, ring: Optional[int]) -> List[Tuple[int, int]]:
    """The (a, b) index pairs, a < b, asked about in a batch of ``n``: every pair, or each entry against its
    ``ring`` neighbours on either side, round the batch."""
    if ring is None or 2 * ring >= n - 1:
        return [(a, b) for a in range(n) for b in range(a + 1, n)]
    return sorted({tuple(sorted((a, (a + d) % n))) for a in range(n) for d in range(1, ring + 1)})


def pair_wins(answers: Dict[str, Any], n: int) -> List[Optional[float]]:
    """Each entry's expected share of wins over the entries it was compared with, 0..100: a pair's answer is the
    probability that its first entry is nearer the second end; None for an entry with no usable comparison."""
    wins = [[] for _ in range(n)]
    for qid, ans in answers.items():
        if not qid.startswith("p") or not isinstance(ans, dict):
            continue
        v = ans.get("noul")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1:
            continue
        a, b = (int(x) - 1 for x in qid[1:].split("_"))
        if 0 <= a < n and 0 <= b < n:
            wins[a].append(v)
            wins[b].append(1 - v)
    return [round(100 * sum(w) / len(w), 3) if w else None for w in wins]
#: The degrees on each side of a level ladder, by how many each side has.
SIDE_DEGREES = {1: ["Clearly"], 2: ["Clearly", "Slightly"], 3: ["Strongly", "Clearly", "Slightly"],
                4: ["Strongly", "Clearly", "Somewhat", "Slightly"],
                5: ["Extremely", "Strongly", "Clearly", "Somewhat", "Slightly"]}
DEGREE_TEXT = {"Extremely": "as {a} as anyone could be, with nothing at all of {b}",
               "Strongly": "markedly {a}, with almost nothing of {b}",
               "Clearly": "plainly more {a} than {b}",
               "Somewhat": "leaning toward {a} more than toward {b}",
               "Slightly": "barely on the {a} side of the middle between {a} and {b}"}


def ladder(neg: str, pos: str, levels: int, who: str) -> List[str]:
    """``levels`` self-contained level descriptions from the ``neg`` end to the ``pos`` end, each naming both
    ends (Jev reads a level on its own); a middle level when the count is odd.  ``who``: "the person
    described" or the like.  Past ten levels (Choice only) the ladder is percentile points instead."""
    if levels > 11:
        return [f"{round(100 * i / (levels - 1))}: {who} is further toward {pos} than about "
                f"{round(100 * i / (levels - 1))}% of all roles and traits, and further toward {neg} than the rest"
                for i in range(levels)]
    per_side = levels // 2
    if per_side not in SIDE_DEGREES:
        raise ValueError(f"{levels} levels: 2 to 11 (or more, as percentile points, for a Choice question)")
    side = SIDE_DEGREES[per_side]
    middle = [f"Neither: {who} shows neither {neg} nor {pos}, or both about equally"] if levels % 2 else []
    return ([f"{d} {neg}: {who} is " + DEGREE_TEXT[d].format(a=neg, b=pos) for d in side] + middle
            + [f"{d} {pos}: {who} is " + DEGREE_TEXT[d].format(a=pos, b=neg) for d in reversed(side)])


def axis_state(spec: Any, pair_type: str, show) -> Tuple[str, str, Dict[str, Any]]:
    neg, pos = show(spec.neg_examples[0], pair_type), show(spec.pos_examples[0], pair_type)
    return neg, pos, {"from": neg, "to": pos, "ends": {neg: spec.neg_pole, pos: spec.pos_pole}}


def entry_of(etype: str, name: str, text: str, mode: str, show) -> Dict[str, str]:
    kind = jr._entity_words(etype)[0]
    field = "description" if mode == "descriptions" else "instruction used to ask a model to take it on"
    return {"name": show(name, etype), "kind": kind, field: text}


# ---------------------------------------------------------------------------
# Wordings: (state, {question id: question}) for one batch
# ---------------------------------------------------------------------------

def _how(mode: str) -> Tuple[str, str]:
    """(what to consider, who is placed) for the mode."""
    if mode == "descriptions":
        return ("how the person described typically behaves, what they value and what they are associated with",
                "the person described")
    return ("how someone following that instruction would typically behave, what they would value and what they "
            "would be associated with", "someone following that instruction")


def wording_base(spec, pair_type, group, mode, show, question, levels):
    """The axis and the entries in the state; each question names its entry and asks where it lies."""
    neg, pos, axis = axis_state(spec, pair_type, show)
    state = {"task": "Roles and traits to be placed along one semantic axis.", "axis": axis,
             "entries": {str(k): entry_of(etype, n, text, mode, show) for k, (etype, n, text) in enumerate(group, 1)}}
    consider, who = _how(mode)
    qs = {}
    if question == "pairs":
        heads = [f"entry {k}, {show(n, etype)} ({jr._entity_words(etype)[0]})" for k, (etype, n, _) in enumerate(group, 1)]
        for a, b in pair_list(len(group), PAIR_RING):
            qs[f"p{a + 1}_{b + 1}"] = {
                "type": "noul",
                "instructions": f"Of {heads[a]}, and {heads[b]}: consider {consider} for each.  Is entry {a + 1} "
                                f"nearer the {pos} end of the axis than entry {b + 1}?",
                "criteria": {"true": f"entry {a + 1} is nearer {pos} than entry {b + 1} is",
                             "false": f"entry {b + 1} is nearer {pos} than entry {a + 1} is"}}
        return state, qs
    for k, (etype, n, _) in enumerate(group, 1):
        head = f"Entry {k}, {show(n, etype)} ({jr._entity_words(etype)[0]})"
        if question == "noul":
            qs[f"e{k}"] = {"type": "noul",
                           "instructions": f"{head}: consider {consider}.  Is it nearer the {pos} end of the axis "
                                           f"than the {neg} end?",
                           "criteria": {"true": f"nearer {pos} than {neg}", "false": f"nearer {neg} than {pos}"}}
            continue
        instr = f"{head}: consider {consider}.  Where does it lie on the axis, between {neg} and {pos}?"
        crit = ladder(neg, pos, levels, who)
        qs[f"e{k}"] = ({"type": "score", "instructions": instr, "criteria": crit} if question == "score" else
                       {"type": "choice", "instructions": instr + f"  The options are the points 0 to {levels - 1} "
                        f"of a scale from {neg} (0) to {pos} ({levels - 1}).",
                        "criteria": {str(i): c for i, c in enumerate(crit)}})
    return state, qs


def wording_rank(spec, pair_type, group, mode, show, question, levels):
    """The base wording, the question asking for the entry's place among all roles and traits and against the
    other entries, as the percentile rubrics do: most relate to an axis only partly, so place each by its lean."""
    state, qs = wording_base(spec, pair_type, group, mode, show, question, levels)
    neg, pos, _ = axis_state(spec, pair_type, show)
    extra = (f"  Judge it against all roles and traits, the other entries among them: most relate to this axis only "
             f"partly or indirectly, so place it by which way it leans, and only in the middle if it is genuinely "
             f"balanced between {neg} and {pos}.")
    for q in qs.values():
        q["instructions"] = q["instructions"] + extra
    return state, qs


def wording_strong(spec, pair_type, group, mode, show, question, levels):
    """Noul only: the base question (nearer the second end than the first?) and two more per entry, whether it is
    strongly at the second end and whether strongly at the first, combined by strong_value."""
    if question != "noul":
        raise SystemExit("--wording strong goes with --question noul")
    state, qs = wording_base(spec, pair_type, group, mode, show, question, levels)
    neg, pos, _ = axis_state(spec, pair_type, show)
    consider, _ = _how(mode)
    for k, (etype, n, _) in enumerate(group, 1):
        head = f"Entry {k}, {show(n, etype)} ({jr._entity_words(etype)[0]})"
        for tag, end, other in (("s", pos, neg), ("n", neg, pos)):
            qs[f"e{k}{tag}"] = {"type": "noul",
                                "instructions": f"{head}: consider {consider}.  Is it strongly at the {end} end of "
                                                f"the axis, clearly more {end} than most roles and traits?",
                                "criteria": {"true": f"strongly {end}", "false": f"not strongly {end}: in the middle "
                                                                                 f"or toward {other}"}}
    return state, qs


def wording_relative(spec, pair_type, group, mode, show, question, levels):
    """Noul only: is the entry nearer the second end than most of the other entries in the state?"""
    if question != "noul":
        raise SystemExit("--wording relative goes with --question noul")
    state, qs = wording_base(spec, pair_type, group, mode, show, question, levels)
    neg, pos, _ = axis_state(spec, pair_type, show)
    consider, _ = _how(mode)
    for k, (etype, n, _) in enumerate(group, 1):
        head = f"Entry {k}, {show(n, etype)} ({jr._entity_words(etype)[0]})"
        qs[f"e{k}"] = {"type": "noul",
                       "instructions": f"{head}: consider {consider}, and the same for the other entries in the "
                                       f"state.  Is entry {k} nearer the {pos} end of the axis than most of the other "
                                       f"entries are?",
                       "criteria": {"true": f"nearer {pos} than most of the other entries",
                                    "false": f"nearer {neg} than most of the other entries"}}
    return state, qs


def strong_value(answers: Dict[str, Any], k: int) -> Optional[float]:
    """0..100 from an entry's three Nouls (wording strong): the mean of P(leans to the second end), P(strongly at
    the second end) and 1 - P(strongly at the first)."""
    vals = []
    for qid, flip in ((f"e{k}", False), (f"e{k}s", False), (f"e{k}n", True)):
        a = answers.get(qid)
        v = a.get("noul") if isinstance(a, dict) else None
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1:
            return None
        vals.append(1 - v if flip else v)
    return round(100.0 * sum(vals) / 3, 3)


WORDINGS: Dict[str, Callable] = {"base": wording_base, "rank": wording_rank, "strong": wording_strong,
                                 "relative": wording_relative}


# ---------------------------------------------------------------------------
# Reading a value out of an answer
# ---------------------------------------------------------------------------

def answer_value(answer: Any, question: str, levels: int) -> Optional[float]:
    """0..100 from one raw answer (None when it is not a well-formed answer of the asked type)."""
    if not isinstance(answer, dict):
        return None
    if question == "noul":
        v = answer.get("noul")
        return round(100.0 * v, 3) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1 else None
    if question == "score":
        v = answer.get("score")
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= levels - 1
        return round(100.0 * v / (levels - 1), 3) if ok else None
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return None
    try:
        pairs = [(int(k), float(p)) for k, p in probs.items()]
    except (TypeError, ValueError):
        return None
    total = sum(p for _, p in pairs)
    if any(not 0 <= k < levels or p < 0 for k, p in pairs) or abs(total - 1.0) > 0.02:
        return None
    return round(100.0 * sum(k * p for k, p in pairs) / total / (levels - 1), 3)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def same_settings(found: Dict[str, Any], header: Dict[str, Any], answers: Optional[Dict[str, Dict[str, Any]]] = None,
                  expect_model: Optional[str] = None) -> bool:
    """Whether a log's header and this run's agree, the sample request aside: one label spans several axes files
    (the development and the held-out axes), whose first requests differ, and its records are keyed by axis.  The
    name requested may differ (the balanced logs of 2026-11-04 asked for jev-latest, a run now for jev-1.14.0) when
    this run requires a version and every answer the log holds came from it."""
    strip = lambda h: {k: v for k, v in h.items() if k not in ("digest", "sample", "model")}
    if jrj.settings_digest(strip(found)) != jrj.settings_digest(strip(header)):
        return False
    if found.get("model") == header.get("model"):
        return True
    return expect_model is not None and all(r.get("model") == expect_model for r in (answers or {}).values())


def group_key(axis: str, names: Sequence[str]) -> str:
    return f"{axis}|" + hashlib.sha1("\x1f".join(names).encode()).hexdigest()[:12]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES), required=True)
    p.add_argument("--instruction_round", type=int, default=None,
                   help="with --mode instructions: which of an entity's instructions (round r: instruction r mod 5)")
    p.add_argument("--axes_file", default=None, help="'axis_dir pos neg pair_type' rows; or --pair")
    p.add_argument("--pair", nargs=2, metavar=("POS", "NEG"), help="one axis, POS_vs_NEG, its projections under "
                   "<experiments_dir>/POS_vs_NEG/gpt/ (judge_descriptions.py's single-pair form)")
    p.add_argument("--pair_type", choices=["roles", "traits"])
    p.add_argument("--limit", type=int, default=None, help="only the first N entities of each axis, to try it cheaply")
    p.add_argument("--experiments_dir", default=ev.DEFAULT_EXPERIMENTS_DIR)
    p.add_argument("--instructions_dir", required=True, help="the corpus (the May pin for the May axes)")
    p.add_argument("--question", choices=("score", "choice", "noul", "pairs"), default="score")
    p.add_argument("--pair_ring", type=int, default=None,
                   help="with --question pairs: compare each entry with this many neighbours either side (default all)")
    p.add_argument("--levels", type=int, default=5, help="Score: 2 to 10; Choice: 2 to 255 (past 11 as percentile "
                                                          "points); ignored for noul")
    p.add_argument("--wording", choices=list(WORDINGS), default="base")
    p.add_argument("--directions", choices=DIRECTION_MODES, default="own",
                   help="which way the calls show each axis (see the module docstring); own is the axis's own order")
    p.add_argument("--direction_pass", type=int, default=0, help="with --directions descriptions: 0 or 1")
    p.add_argument("--entities_per_call", type=int, default=20)
    p.add_argument("--shuffle_seed", type=int, default=1)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="the name requested: a versioned ID (the default, pinned; TypeSafe accepts one although GET "
                        "/v1/models lists only the aliases) or an alias (jev-latest, jev-preview), whose answers "
                        "change when it moves")
    p.add_argument("--expect_model", default=None,
                   help="the version the answers must come from (default: --model when that is a versioned ID): an "
                        "answer from another version stops the run before any cell is written")
    p.add_argument("--cell_label", required=True, help="suffix of the cells, starting with @")
    p.add_argument("--max_in_flight", type=int, default=24)
    p.add_argument("--budget_usd", type=float, default=None, help="stop issuing requests past this spend")
    p.add_argument("--limit_axes", type=int, default=None, help="only the first N axes, for a smoke test")
    p.add_argument("--dry_run", action="store_true", help="the requests and the estimated cost; nothing sent")
    return p


def check_args(args) -> None:
    if not args.cell_label.startswith("@"):
        raise SystemExit("--cell_label starts with @")
    if (args.mode == "instructions") != (args.instruction_round is not None):
        raise SystemExit("--instruction_round goes with --mode instructions, and that mode needs it")
    if args.question == "score" and not 2 <= args.levels <= 10:
        raise SystemExit("a Score question takes 2 to 10 levels")
    if args.question == "choice" and not 2 <= args.levels <= 255:
        raise SystemExit("a Choice question takes 2 to 255 options")
    if args.entities_per_call < 1:
        raise SystemExit("--entities_per_call: 1 or more")
    if bool(args.axes_file) == bool(args.pair) or (args.pair and not args.pair_type):
        raise SystemExit("give --axes_file, or --pair with --pair_type")


async def main_async(argv=None) -> int:
    args = build_parser().parse_args(argv)
    check_args(args)
    if args.expect_model is None and VERSIONED_ID.fullmatch(args.model):
        args.expect_model = args.model
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    levels = args.levels if args.question not in ("noul", "pairs") else 2
    global PAIR_RING
    PAIR_RING = args.pair_ring
    E = Path(args.experiments_dir)
    ev.RUBRIC = jr.get(MODES[args.mode])
    ev.INSTRUCTION_ROUND = args.instruction_round or 0
    ev.AXIS_INDEX = None
    ev.SWAP_POLES = False
    ev.DIRECTIONS, ev.DIRECTION_PASS = "own", 0      # this script swaps its own calls (call_swapped)
    if args.axes_file:
        jobs = ev.jobs_from_axes_file(Path(args.axes_file), E)
    else:
        axis = f"{args.pair[0]}_vs_{args.pair[1]}"
        jobs = [ev.AxisJob(axis, args.pair[0], args.pair[1], args.pair_type, E / axis / "gpt" / "projections.json",
                           E / f"{axis}{ev.TIER_EVAL_SUFFIX}")]
    if args.limit_axes:
        jobs = jobs[: args.limit_axes]
    show = jr.name_shower(jr.DEFAULT_NAME_FORMAT, Path(args.instructions_dir))
    ctx = SimpleNamespace(shuffle_seed=args.shuffle_seed, entities_per_call=args.entities_per_call, batch_design="shuffle")
    wording = WORDINGS[args.wording]
    requests: List[Tuple[Any, str, List[str], Dict[str, Any]]] = []
    skipped = []
    for job in jobs:
        if not Path(job.projections_file).exists():
            skipped.append(job.axis_dir)
            continue
        ev.load_axis(job, Path(args.instructions_dir))
        if args.limit:
            job.entities = job.entities[: args.limit]
        swapped = call_swapped(args.directions, job.axis_dir, args.direction_pass, args.instruction_round)
        spec = jr.swap_axis(job.axis_spec) if swapped else job.axis_spec
        for group in ev.axis_groups(job, ctx):
            state, qs = wording(spec, job.pair_type, group, args.mode, show, args.question, levels)
            names = [n for _, n, _ in group]
            requests.append((job, group_key(job.axis_dir, names), names,
                             {"model": args.model, "state": state, "questions": qs}, swapped))
    if skipped:
        logger.warning(f"{len(skipped)} axes without projections left out: {', '.join(skipped[:5])}")
    header = {"kind": "header", "model": args.model, "mode": args.mode, "instruction_round": args.instruction_round,
              "question": args.question, "levels": levels, "wording": args.wording,
              "entities_per_call": args.entities_per_call, "shuffle_seed": args.shuffle_seed,
              "instructions_dir": str(Path(args.instructions_dir).resolve()),
              "sample": requests[0][3] if requests else None}
    if args.pair_ring is not None:      # only then, so that the logs written before the flag still resume
        header["pair_ring"] = args.pair_ring
    if args.directions != "own":        # likewise
        header["directions"], header["direction_pass"] = args.directions, args.direction_pass
    if args.limit:                      # a trial on the first N entities never shares a log with a full run
        header["limit"] = args.limit
    work = E / JEV_DIRNAME / args.cell_label.lstrip("@")
    log_path = work / "answers.jsonl"
    found, answers = jrj.read_log(log_path)
    if found is not None and not same_settings(found, header, answers, args.expect_model):
        raise SystemExit(f"{log_path} was written under other settings: use another --cell_label, or move it aside")
    todo = [r for r in requests if r[1] not in answers]
    rate_in, _ = price_for_model(args.model)
    est = sum(len(json.dumps(r[3])) / CHARS_PER_TOKEN for r in todo) * rate_in / 1e6
    n_entities = sum(len(r[2]) for r in requests)
    logger.info(f"{len(jobs) - len(skipped)} axes, {n_entities} entity judgements in {len(requests)} requests "
                f"({len(requests) - len(todo)} answered before); estimated ${est:.3f} at ${rate_in}/1M input")
    if args.dry_run:
        print(json.dumps(requests[0][3], indent=1)[:6000] if requests else "no requests")
        return 0

    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    client = jrj.make_client()
    work.mkdir(parents=True, exist_ok=True)
    if found is None:
        with open(log_path, "a") as f:
            f.write(json.dumps({**header, "digest": jrj.settings_digest(header)}) + "\n")
    usage = MultiModelUsage()
    for rec in answers.values():
        usage.charge(rec.get("model") or args.model, rec.get("input_tokens", 0), rec.get("output_tokens", 0))
    spent0 = usage.total_cost_usd
    sem = asyncio.Semaphore(max(1, args.max_in_flight))
    fatal: List[BaseException] = []
    failed = 0
    lock = asyncio.Lock()

    async def one(req) -> None:
        nonlocal failed
        job, key, names, body, swapped = req
        if fatal or (args.budget_usd is not None and usage.total_cost_usd - spent0 > args.budget_usd):
            return
        async with sem:
            try:
                reply = await jrj.ask(client, body)
            except jrj.JevRequestError as e:
                fatal.append(e)
                return
            except RuntimeError as e:
                failed += 1
                logger.warning(f"{key}: {e}")
                return
        u = reply.get("usage") or {}
        rec = {"kind": "answer", "key": key, "axis": job.axis_dir, "names": names, "swapped": swapped,
               "model": reply.get("model") or args.model, "answers": reply.get("answers") or {},
               "input_tokens": int(u.get("input_tokens") or 0), "output_tokens": int(u.get("output_tokens") or 0),
               "at": time.time()}
        async with lock:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            answers[key] = rec
            usage.charge(rec["model"], rec["input_tokens"], rec["output_tokens"])

    try:
        await asyncio.gather(*(one(r) for r in todo))
    finally:
        await client.aclose()
    if fatal:
        logger.error(f"Jev refused a request: {fatal[0]}")
        return 1
    atomic_write_text(json.dumps(usage.as_dict(), indent=2, sort_keys=True), work / "usage.json", logger_obj=logger)
    logger.info(usage.log_line())
    missing = [r for r in requests if r[1] not in answers]
    if missing:
        logger.error(f"{len(missing)} requests unanswered ({failed} failed): rerun to ask them; cells not written")
        return 4
    requested = [found["model"], args.model] if found is not None and found.get("model") != args.model else args.model
    return write_cells(jobs, requests, answers, args, levels, log_path, requested)


def write_cells(jobs, requests, answers, args, levels: int, log_path: Path, requested=None) -> int:
    """The cells, from every answer of the log.  ``requested``: the name or names the log's requests asked for
    (two when its first run asked for an alias and a later one for the version, as the balanced logs of 2026-11-04)."""
    by_axis: Dict[str, List[Tuple[List[str], Dict[str, Any]]]] = {}
    for job, key, names, _, _ in requests:
        by_axis.setdefault(job.axis_dir, []).append((names, answers[key]))
    resolved = sorted({r.get("model") for r in answers.values() if r.get("model")})
    if args.expect_model and resolved != [args.expect_model]:
        logger.error(f"Jev answered as {', '.join(resolved) or 'no named version'}, not {args.expect_model}: the "
                     f"cells' names and Jev's fitted weights are for {args.expect_model}, so no cell is written.  "
                     f"Requested as {args.model}: if that is the version, TypeSafe did not honour it (ask them); if an "
                     f"alias, it has moved.  A new version is a change of judge: JEV_MODEL in the drivers "
                     f"(judge_descriptions.py, judge_instruction_grid.py), and Jev's weight refitted")
        return 5
    model = resolved[0] if len(resolved) == 1 else args.model
    mode = args.mode
    n_ok = n_all = 0
    for job in jobs:
        if job.axis_dir not in by_axis:
            continue
        result: Dict[str, float] = {}
        usage = MultiModelUsage()
        for names, rec in by_axis[job.axis_dir]:
            usage.charge(rec.get("model") or args.model, rec.get("input_tokens", 0), rec.get("output_tokens", 0))
            pw = pair_wins(rec["answers"], len(names)) if args.question == "pairs" else None
            for k, name in enumerate(names, 1):
                if pw is not None:
                    v = pw[k - 1]
                elif args.wording == "strong":
                    v = strong_value(rec["answers"], k)
                else:
                    v = answer_value(rec["answers"].get(f"e{k}"), args.question, levels)
                if v is not None and rec.get("swapped"):
                    v = round(100.0 - v, 3)     # the call showed the axis the other way round: mapped back
                n_all += 1
                if v is not None:
                    n_ok += 1
                    if not ev.is_pole(name, job.axis_spec):
                        result[name] = v
        d = Path(job.output_dir) / f"{model}{args.cell_label}"
        d.mkdir(parents=True, exist_ok=True)
        prov = {"judge_model": model, "model_requested": requested or args.model, "rubric": f"jev_{args.question}{levels}_{args.wording}",
                "text_source": "descriptions" if mode == "descriptions" else "instruction_rounds",
                "instruction_round": args.instruction_round, "question": args.question, "levels": levels,
                "wording": args.wording, "pair_ring": args.pair_ring, "directions": args.directions,
                "direction_pass": args.direction_pass,
                "n_calls_swapped": sum(1 for _, rec in by_axis[job.axis_dir] if rec.get("swapped")),
                "n_calls": len(by_axis[job.axis_dir]), "entities_per_call": args.entities_per_call,
                "shuffle_seed": args.shuffle_seed,
                "batch_design": "shuffle", "name_format": jr.DEFAULT_NAME_FORMAT, "cell_label": args.cell_label,
                "projections_file": str(job.projections_file), "answers_log": str(log_path),
                "n_entities": len([e for e in job.entities if not ev.is_pole(e[1], job.axis_spec)]),
                "n_scored": len(result)}
        atomic_write_text(json.dumps({"result": result, "_provenance": prov}, indent=2),
                          d / f"scores_{ev.RUBRIC.mode}.json", logger_obj=logger)
        corr = compute_correlations(scores_by_mode={ev.RUBRIC.mode: result}, projections=job.projections,
                                    slots=list(job.projections.keys()), excluded_set=job.axis_spec.pole_entity_ids())
        atomic_write_text(json.dumps(corr, indent=2), d / "correlations.json", logger_obj=logger)
        atomic_write_text(json.dumps(usage.as_dict(), indent=2, sort_keys=True), d / "usage.json", logger_obj=logger)
        raw = corr[ev.RUBRIC.mode].get("6", next(iter(corr[ev.RUBRIC.mode].values())))["raw"]
        logger.info(f"[{model}] {job.axis_dir}: scored {len(result)}; slot 6 raw rho={raw['rho']:.4f}")
    warn_if_low_parse_rate(label=f"judge_jev_entities:{model}", n_ok=n_ok, n_total=n_all, logger_obj=logger)
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
