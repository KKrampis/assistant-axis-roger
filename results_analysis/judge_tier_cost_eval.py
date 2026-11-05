#!/usr/bin/env python3
"""
Score a new judge model against an EXISTING, already-computed axis, without
recomputing activations, projections, or touching a GPU.  The Pareto investigation's
tool: every rubric, batch size, transport and request setting can be varied.  To
judge axes by the settled method, run judge_descriptions.py, which runs this.

Motivation: `axis_judge_correlation.py` unconditionally recomputes
projections from raw activation vectors (`--data_dir` is required), even
though those projections don't change when you just want to try a new judge
model -- they're a pure function of the (fixed) axis direction and the
(fixed) activation vectors, neither of which a new judge model affects.

Every axis under roger/axis_judge_experiments/ with a cached
`<axis>/gpt/projections.json` (61 of them; per-entity, per-slot, both-metric
projections, about 571-580 entities x 8 slots x {raw, whitened}) can be
re-judged this way.  scripts/tier_eval_axes.tsv lists all 61 with their
poles and pair type.

This script: loads a cached projections.json, judges the SAME entities'
descriptions with one or more NEW judge models (any --judge_model
axis_judge_correlation.py itself accepts), correlates the new scores
against the existing projections via the exact same compute_correlations()
/ Spearman machinery, records real measured cost in usage.json, and
writes output in the same shape axis_judge_correlation.py itself writes
(so it's a drop-in for downstream tooling expecting that layout).

Transports (--transport, default auto; added 2026-11-01):
  * flex      OpenAI Flex processing (service_tier="flex"): batch prices,
              answered like real-time calls.  All requests for all axes go at
              once through the rate limiter, on a client with a 900 s timeout
              and 8 retries; a request Flex keeps failing for 10 minutes is
              sent once to the standard tier.  Usage under <model>:flex, the
              fallbacks under <model>.
  * batch     The provider's batch API (OpenAI Batch API, Anthropic Message
              Batches; half price, results within 24 hours, usually within
              one).  One submission per model for all the axes being run;
              each cell is written as soon as its axis is complete; requests a
              batch never answered are sent in real time on the standard tier
              (up to --batch_fallback_max).  Work and state under
              <experiments_dir>/_tier_eval_batches/<model>/; a rerun collects
              batches still in flight before submitting anything.  Usage under
              <model>:batch, the real-time remainder under <model>.
  * realtime  Every request in real time through the rate limiter (the
              behaviour before 2026-11-01).
  auto picks batch for an Anthropic model, real time for a Gemini model; for an
  OpenAI model, flex when assistant_axis/judge.py's MODEL_CAPABILITIES says OpenAI
  takes it, batch when it says OpenAI refuses it, and when it is untested,
  flex with a probe: the first request alone on Flex, and a service-tier
  HTTP 400 switches the model to batch for the run (that request goes into
  the batch).  The choice is logged and recorded in each cell's
  scores_descriptions.json _provenance (transport, request_settings,
  max_tokens).

Request settings per model (reasoning effort, thinking) come from
assistant_axis/judge.py's JUDGE_REQUEST_SETTINGS, the requests pipeline step 3
sends.  --request_setting [MODEL_FRAGMENT:]key=<json> overrides the table for one
run.  Cells judged before the port of 2026-11-02 went by an adapted table, so to
extend one: 'gpt-5.6:reasoning_effort=null' for the cells of GPT-5.6 with no @none
in their label, and 'opus-5:output_config={"effort":"low"}' and the like for those
of Claude 5.

A batched reply that leaves members unread (a stop after the first lines, a form the parser cannot
read) is followed by the same prompt again, up to PARSE_RETRIES times, on every transport; each member
keeps the first reply that gives it a line, and what the last leaves unread goes unscored.  A refusal is
split instead, and a reply cut at the cap has its repair.  scripts/judge_tier_reask.py does the same for
cells already written.

Entity names in prompts (--name_format): judge_label, the project's judge display
form; "mechanical" reproduces the prompts of the cells judged before 2026-11-02.

Reply cap (--max_tokens, default 300 since 2026-11-01, Roger: "200 is clearly
too low for safety"; 200 before): a reply cut at the cap (Anthropic
stop_reason "max_tokens", OpenAI finish_reason "length") usually lacks its
SCORE line, and parse_signed_score's fallback would score it from a stray
number in its reasoning, so it is scored as missing without being parsed,
counted in the run summary ("truncated N") and as a failure in the parse rate,
and listed in the cell's provenance ("truncated", sorted entity keys).

Repair (--repair_truncated, 2026-11-01): for each finished cell of the given
models and axes, re-ask exactly the entities whose reply was cut at the cap,
in real time on the standard tier with the model's request settings, at
--max_tokens (equivalent to rerunning at that cap: a cap changes only the
replies that reach it), and patch the cell: those scores (a reply cut again
scores as missing), correlations.json, usage.json (the re-asks under the plain
model name) and a "repairs" record in the provenance.  The cut entities come
from the provenance "truncated" list; a cell without one is reported and left
alone.

Resume rule: an axis whose <output_dir>/<model>/usage.json exists is skipped
for that model (in both modes), so a rerun only judges what is missing.

Saved answers (2026-11-01, after a Flex run held nine hours of answers only in
memory): every billed live or Flex answer is appended, as it arrives, to its
cell's answers.partial.jsonl (a header with the model, cell label, rubric
version, request settings and reply cap, then one line per answer: entity,
reply text, cut at the cap or not, charges, a digest of the exact prompt).  A
rerun, by any transport, takes every saved answer to the same prompt instead
of asking again (and its charges into the cell's usage.json, not into this
run's budget), and refuses a log written under other settings.  Calls that
failed without a charge are not saved and are asked again.  When the cell is
written, the log is kept compressed as answers.jsonl.gz and removed.  So a run
can be stopped at any moment and rerun losing at most the requests in flight.

Instruction judging (2026-11-02; the is_* and i1_* rubrics): the same run with the entities' positive
instructions for text, the scores in scores_instructions.json.  An i1_* cell is one round
(--instruction_round R): one instruction per entity, each entity going through its instructions in its
own seeded order, so that no batch holds two of one entity's.

A batched call that comes back with nothing (a refusal, a content flag, an empty reply) has lost all
its members' scores, usually to one of them, so its two halves are asked, and so on down, on any
transport; the entity left alone without a score is listed under "refused" in the provenance, and each
such call is a "refusal" line in the answer log.  Batch API replies are kept in
batch_replies.jsonl.gz, since a batch's refusals are otherwise on record only as totals.

Live and Flex calls go axis by axis, at most --max_in_flight (200) at a time; a
progress line per model every five minutes; and when no answer arrives for
--stall_minutes (30) while requests remain, the run stops and starts itself
again (--stall_restarts, 3), resuming from the saved answers.

Usage:
    # All 61 axes, one or more models, preview cost (zero API calls, nothing written):
    uv run python results_analysis/judge_tier_cost_eval.py \\
        --axes_file scripts/tier_eval_axes.tsv \\
        --instructions_dir /tmp/corpus_2026-05-24/data \\
        --judge_model gpt-5.6-luna --judge_model claude-haiku-5-5 \\
        --dry_run

    # One axis (the original interface):
    uv run python results_analysis/judge_tier_cost_eval.py \\
        --pair angel demon --pair_type roles \\
        --projections_file roger/axis_judge_experiments/angel_vs_demon/gpt/projections.json \\
        --judge_model claude-opus-5 --judge_model gpt-5.6-luna \\
        --output_dir roger/axis_judge_experiments/angel_vs_demon_tier_eval \\
        --dry_run

    # Re-ask the replies a run's cells lost at the cap, at 300 tokens (preview first):
    uv run python results_analysis/judge_tier_cost_eval.py \\
        --axes_file scripts/tier_eval_axes.tsv \\
        --instructions_dir /tmp/corpus_2026-05-24/data \\
        --judge_model claude-sonnet-5 --repair_truncated --max_tokens 300 --dry_run

Drop --dry_run to actually run it.  scripts/run_tier_eval.sh wraps the
axes-file form with the corpus check.  Each --judge_model may be repeated;
models run concurrently.  Exit status: 0 done, 1 a model failed (or a key is
missing), 2 --budget_usd stopped the run, 3 it stalled and no restart was left,
4 calls stayed unanswered after their re-asks (rerun: cells with gaps have no
usage.json, and the saved answers are kept).
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse the exact rubric, prompt-building, parsing, and correlation
# machinery axis_judge_correlation.py already has -- not reimplemented here,
# so this script can never drift out of sync with the canonical rubric.
from results_analysis.axis_judge_correlation import (  # noqa: E402
    AxisSpec,
    RUBRIC_VERSION,
    build_static_prompt,
    compute_correlations,
    parse_signed_score,
)
from results_analysis import judge_tier_transport as tt  # noqa: E402
from results_analysis.judge_tier_transport import provider_for_model  # noqa: E402
from results_analysis import judge_tier_rubrics as jr  # noqa: E402
from assistant_axis.judge import RateLimiter, warn_if_low_parse_rate  # noqa: E402
from assistant_axis.judge_pricing import (  # noqa: E402
    BATCH_SUFFIX, MultiModelUsage, TokenUsage, anthropic_usage, gemini_usage, openai_usage, price_for_model,
)
from assistant_axis.entity_id import entity_id, parse_entity_id  # noqa: E402
from assistant_axis.atomic_io import atomic_write_bytes, atomic_write_text  # noqa: E402
from assistant_axis.openai_batch import OpenAIBatchRunner  # noqa: E402
from assistant_axis.anthropic_batch import AnthropicBatchRunner  # noqa: E402
from assistant_axis.gemini_batch import GeminiBatchRunner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENTS_DIR = "roger/axis_judge_experiments"
TIER_EVAL_SUFFIX = "_tier_eval"
#: Batch work and state, per model, under the experiments dir: never inside a
#: *_tier_eval directory (the pooled plotter globs those).
BATCH_DIRNAME = "_tier_eval_batches"
#: Custom ids are ``a<axis index>|<entity name>``, within both providers' limits
#: (the Anthropic runner maps them through its chunk files anyway).
CUSTOM_ID_RE = re.compile(r"[A-Za-z0-9_|-]{1,64}")
# Dry-run token model: this repo's measured desc+inst per-item model (README
# "Judging cost model"): ~320 header + ~150 per short description, ~80 output.
# Coarse, for planning only; the real run records ACTUAL tokens in usage.json.
EST_IN_PER_ENTITY = 320 + 150
EST_OUT_PER_ENTITY = 80
#: With batched judging (the percentile rubric, measured November 2026): the shared prompt, then per entity.
#: Output per entity is about 22 tokens on the Lunas, 30 on Gemini 3.1, 60 on Haiku 5.5.
EST_BATCH_HEADER = 320
EST_BATCH_IN_PER_ENTITY = 40
EST_BATCH_OUT_PER_ENTITY = 30
#: Input per entity by what the entity's text is (measured on the May corpus, 2026-11-02).
EST_BATCH_IN_BY_SOURCE = {"descriptions": EST_BATCH_IN_PER_ENTITY, "instruction_sets": 140, "instruction_rounds": 35,
                          "responses": 410}     # a kept response is about 392 tokens (May corpus, 2026-11-03)
DEFAULT_BATCH_FALLBACK_MAX = 2000
#: The judge's reply cap (--max_tokens); 200 until 2026-11-01 (Roger, 2026-11-01:
#: "200 is clearly too low for safety").
DEFAULT_MAX_TOKENS = 300
#: --batch_chunk_requests: requests per batch submission (the runners' own default).
DEFAULT_BATCH_CHUNK_REQUESTS = 10_000
#: Live and Flex calls: at most this many requests in flight per model
#: (--max_in_flight).  Before 2026-11-01 every request of every axis was handed
#: to the HTTP client at once; with 35,214 per model the client's connection
#: pool spent all its time on bookkeeping (each event walks the whole queue
#: against every connection), the process sat at 100% of a core for nine hours,
#: and almost nothing was sent.
DEFAULT_MAX_IN_FLIGHT = 200
#: A progress line per model this often, while live or Flex requests remain.
PROGRESS_EVERY_S = 300.0
#: No answer for this long while live or Flex requests remain: stop and restart
#: (--stall_minutes, --stall_restarts).  Every answer is on disk by then.
DEFAULT_STALL_MINUTES = 30.0
DEFAULT_STALL_RESTARTS = 3
EXIT_STALLED = 3
EXIT_INCOMPLETE = 4
#: A call that got no answer (a network or provider failure, not a refusal) leaves its entities
#: unscored and its cell open; live and Flex runs ask such calls again this many times, this far
#: apart, before leaving them for a rerun.
REASK_ROUNDS = 3
REASK_SLEEP_S = 60.0
#: How many times a batched reply that left members unread is asked again, the same prompt, before they go
#: unscored: these judges run at temperature 1, so a re-ask is a new sample (Roger, 2026-11-03: "at least a
#: couple of times"; Luna stopped right after a score in words in 7 of 6,960 replies, Haiku wrote no colon in 3).
PARSE_RETRIES = 3
#: The environment variable that counts restarts after a stall across execs.
RESTART_ENV = "JUDGE_TIER_EVAL_STALL_RESTARTS"
#: Per-cell answer log (see "Saved answers" in the docstring).
ANSWERS_PARTIAL = "answers.partial.jsonl"
ANSWERS_ARCHIVE = "answers.jsonl.gz"
#: Batch API replies as they are collected, kept apart from the answer log, which a rerun resumes from.
BATCH_REPLIES = "batch_replies.partial.jsonl"
BATCH_REPLIES_ARCHIVE = "batch_replies.jsonl.gz"
CHECKPOINT_VERSION = 1
#: Header fields an answer log must share with the run that resumes it.
CHECKPOINT_KEYS = ("version", "judge_model", "cell_label", "rubric_version", "request_settings", "max_tokens",
                   "name_format", "swap_poles", "directions", "direction_pass", "stack", "stack_seed",
                   "response_design")
#: What an answer log's header means by leaving a key out: it predates the key.
CHECKPOINT_DEFAULTS = {"name_format": "mechanical", "swap_poles": False, "directions": "own", "direction_pass": 0,
                       "stack": 0, "stack_seed": 1, "response_design": "question"}


def load_pole_description(instructions_dir: Path, pair_type: str, name: str) -> str:
    path = instructions_dir / pair_type / "instructions" / f"{name}.json"
    with open(path) as f:
        return json.load(f)["description"]


def load_entity_description(instructions_dir: Path, etype: str, name: str) -> Optional[str]:
    path = instructions_dir / etype / "instructions" / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("description")


def load_entity_instructions(instructions_dir: Path, etype: str, name: str) -> Optional[Tuple[str, ...]]:
    """An entity's positive instructions, in file order; None when it has no file or none."""
    path = instructions_dir / etype / "instructions" / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return tuple(x["pos"] for x in json.load(f).get("instruction") or [] if x.get("pos")) or None


def instruction_of_round(instructions: Sequence[str], etype: str, name: str, round_: int) -> str:
    """The instruction an entity shows in a round: rounds 0 to n-1 go through all n in an order seeded
    by the entity, so that one round's instructions are not all the first of their files, and every
    arm and axis shows the same one."""
    order = list(range(len(instructions)))
    random.Random(f"instruction order|{etype}|{name}").shuffle(order)
    return instructions[order[round_ % len(order)]]


def cell_usage_path(d: Path) -> Path:
    """The usage file that says what a cell's design costs: its own, or, for a cell re-asked in part from
    another (``reasked_from``, scripts/judge_tier_reask.py), whose usage.json holds only the re-asks, that one's."""
    d = Path(d)
    source = json.loads(cell_scores_path(d).read_text()).get("_provenance", {}).get("reasked_from")
    return d.parent / source / "usage.json" if source else d / "usage.json"


#: Response judging (an r1_* rubric; set in main_async): the pipeline tree holding
#: ``<kind>/responses/<entity>.jsonl`` and ``<kind>/scores/<entity>.json`` (--responses_dir); which part of
#: an entity's responses this cell judges (--response_split M --response_part K, cut as RESPONSE_DESIGN says;
#: part None for all); and a cut on the answer text in characters (--response_chars; None for whole).
RESPONSES_DIR: Optional[Path] = None
#: With --axes_per_call: judge only these members of each group (--axis_index), one axis per call with r1_pn,
#: several with rk_pn, so that every run sends the same batches and a cached prefix is read back.
AXIS_INDEX: Optional[Tuple[int, ...]] = None
RESPONSE_SPLIT = 3
RESPONSE_PART: Optional[int] = None
#: How a part is cut (--response_design; Roger, 2026-11-04).  "golden": a response's place on the unit circle is
#: frac((5 Q + p) / phi), Q the absolute question number (the questions file's id, not the position after step 1's
#: every-third selection or step 3's filter), p the instruction (0 to 4), phi the golden ratio, computed exactly as
#: Fibonacci hashing (golden_part); part K of M is the interval [K/M, (K+1)/M).  It depends on the question and the
#: instruction only (no entity rank), has no period for step 1's stride to alias with (5Q + p mod 3 is p mod 3 when
#: every Q is a multiple of 3, as in both trees, which gave thirds of 40/40/20), keeps the parts within 1% of equal at
#: thirds and 3% at ninths on our trees (a plain hash: 8% and 28%), and splits every question's five instructions
#: 1/2/2 over the thirds.  Parts nest as intervals: a ninth's parts 0, 1 and 2 lie inside the third's part 0.  Not
#: for fifths (Roger: avoid cutting by five).  response_judging_plan.md section 4b has the measurements.  "question":
#: the questions with (q + entity rank) % M == K, q the position in the tree, every cell before 2026-11-04; a ninth's
#: parts 0, 3 and 6 lie inside the third's part 0.
RESPONSE_DESIGN = "golden"
#: Fibonacci hashing's constant, round(2**64 / phi): (n * GOLDEN_64) mod 2**64 is frac(n / phi) in 64-bit fixed point.
GOLDEN_64 = 0x9E3779B97F4A7C15
#: Instructions an entity has; the golden design numbers a question's responses 5 Q + p.
INSTRUCTIONS_PER_ENTITY = 5
#: Where a response record without ``question_id`` (trees made before 2026-10-12) finds its absolute question
#: number: its question's text in this file, whose ids are the questions' places in it.
QUESTIONS_FILE = Path(__file__).resolve().parents[1] / "data" / "extraction_questions.jsonl"
#: Question text to absolute number: from the records' ``question_id`` as they are read, else QUESTIONS_FILE.
QUESTION_IDS: Dict[str, int] = {}
RESPONSE_CHARS: Optional[int] = None
RESPONSE_CUT_MARK = " […]"


def load_entity_responses(responses_dir: Path, etype: str, name: str) -> Optional[List[Tuple[int, int, str, str]]]:
    """An entity's kept responses, ``(instruction p, question q, question text, answer)`` for every ``pos``
    item step 3 scored 3, in file order; None when the entity has no response file or nothing kept."""
    path = responses_dir / etype / "responses" / f"{name}.jsonl"
    scores_path = responses_dir / etype / "scores" / f"{name}.json"
    if not path.exists() or not scores_path.exists():
        return None
    with open(scores_path) as f:
        kept = json.load(f)
    out: List[Tuple[int, int, str, str]] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("label") != "pos":
                continue
            p, q = int(rec["prompt_index"]), int(rec["question_index"])
            if kept.get(f"pos_p{p}_q{q}") != 3:
                continue
            if rec.get("question_id") is not None:
                note_question_id(rec["question"], int(rec["question_id"]), path)
            answer = rec["conversation"][-1]["content"]
            if RESPONSE_CHARS is not None and len(answer) > RESPONSE_CHARS:
                answer = answer[:RESPONSE_CHARS].rstrip() + RESPONSE_CUT_MARK
            out.append((p, q, rec["question"], answer))
    return out or None


def response_item_key(name: str, p: int, q: int) -> str:
    """The key of one response in a cell's scores: the entity's projection key, its instruction and question."""
    return f"{name}|p{p}q{q}"


def response_item_entity(key: str) -> str:
    return key.rsplit("|", 1)[0]


def response_question(key: str) -> int:
    return int(key.rsplit("q", 1)[1])


def entity_text_loader() -> Any:
    """``load(instructions_dir, etype, name)``: the text this run's rubric judges an entity by."""
    if RUBRIC.text_source == "descriptions":
        return load_entity_description
    if RUBRIC.text_source == "instruction_sets":
        return load_entity_instructions
    if RUBRIC.text_source == "responses":
        return lambda _dir, etype, name: load_entity_responses(RESPONSES_DIR, etype, name)

    def load(instructions_dir: Path, etype: str, name: str) -> Optional[str]:
        instructions = load_entity_instructions(instructions_dir, etype, name)
        return instruction_of_round(instructions, etype, name, INSTRUCTION_ROUND) if instructions else None
    return load


def note_question_id(text: str, qid: int, where: Any = None) -> None:
    """Record a question's absolute number from a response record; two numbers for one text is an error."""
    if QUESTION_IDS.setdefault(text, qid) != qid:
        raise ValueError(f"{where}: question {text[:60]!r} has question_id {qid} here and {QUESTION_IDS[text]} elsewhere")


def absolute_question(text: str) -> int:
    """A question's absolute number: its records' ``question_id``, or its place in QUESTIONS_FILE (the May tree's
    records have none; its texts all sit at 3 x question_index there, step 1 having taken every third)."""
    if text not in QUESTION_IDS:
        if not getattr(absolute_question, "loaded", None) == QUESTIONS_FILE and QUESTIONS_FILE.exists():
            with open(QUESTIONS_FILE) as f:
                for i, line in enumerate(x for x in f if x.strip()):
                    rec = json.loads(line)
                    QUESTION_IDS.setdefault(rec["question"], int(rec.get("id", i)))
            absolute_question.loaded = QUESTIONS_FILE
        if text not in QUESTION_IDS:
            raise ValueError(f"question {text[:60]!r}: its records carry no question_id and it is not in "
                             f"{QUESTIONS_FILE}, so its absolute number is unknown (--response_design question "
                             f"does without)")
    return QUESTION_IDS[text]


def golden_point(question: int, instruction: int) -> int:
    """A response's place on the unit circle in 64-bit fixed point: frac((5 question + instruction) / phi)."""
    if not 0 <= instruction < INSTRUCTIONS_PER_ENTITY:
        raise ValueError(f"instruction {instruction}: the golden design numbers 5 per question")
    return ((INSTRUCTIONS_PER_ENTITY * question + instruction) * GOLDEN_64) % (1 << 64)


def golden_part(question: int, instruction: int, split: int) -> int:
    """The part (0 to split - 1) of the golden design a response falls in: floor(point x split)."""
    return (golden_point(question, instruction) * split) >> 64


def response_items(entities: Sequence[Tuple[str, str, Any]], part: Optional[int],
                   split: int = RESPONSE_SPLIT, design: Optional[str] = None) -> List[Tuple[str, str, Any]]:
    """The response items of ``entities`` (each ``(etype, name, [(p, q, question, answer), ...])``), one
    ``(etype, key, (question, answer))`` per kept response: all of them, or, with ``part``, the part ``design``
    (default RESPONSE_DESIGN) cuts.  "golden": ``golden_part(Q, p, split) == part``, Q the absolute question
    number, every entity the same cells, so that the judges' parts each hold every entity (Roger's round-robin
    coverage, 2026-11-03: the parts together judge every response once); split 9's parts 3K to 3K + 2 lie inside
    split 3's part K.  "question": ``(q + rank) % split == part``, q the position in the tree and ``rank`` the
    entity's place in the sorted names (the cells before 2026-11-04); split 9's parts 0, 3 and 6 lie inside split
    3's part 0.  Either way a ninth is a subsample of a third."""
    design = design or RESPONSE_DESIGN
    if design not in ("golden", "question"):
        raise ValueError(f"response design {design!r}: golden or question")
    ranks = {name: r for r, name in enumerate(sorted(name for _, name, _ in entities))}
    out: List[Tuple[str, str, Any]] = []
    for etype, name, items in entities:
        for p, q, question, answer in items:
            if part is not None:
                at = golden_part(absolute_question(question), p, split) if design == "golden" else (q + ranks[name]) % split
                if at != part:
                    continue
            out.append((etype, response_item_key(name, p, q), (question, answer)))
    return out


def response_design_in_effect() -> str:
    """The design that cut this run's responses: RESPONSE_DESIGN when a response rubric judges a part, else
    "question", which is also what a cell or log that predates the setting means (a run that cuts nothing
    matches either)."""
    if getattr(RUBRIC, "text_source", None) == "responses" and RESPONSE_PART is not None:
        return RESPONSE_DESIGN
    return "question"


def entity_means(scores: Dict[str, float]) -> Dict[str, float]:
    """Per-entity mean of response-item scores (``name|p..q..`` keys to ``name``)."""
    sums: Dict[str, List[float]] = {}
    for key, v in scores.items():
        sums.setdefault(response_item_entity(key), []).append(v)
    return {name: sum(v) / len(v) for name, v in sorted(sums.items())}


def cell_scores_path(d: Path) -> Path:
    """A finished cell's scores file: ``scores_instructions.json`` for instruction judging,
    ``scores_responses.json`` for response judging, ``scores_descriptions.json`` otherwise."""
    for mode in ("instructions", "responses"):
        p = Path(d) / f"scores_{mode}.json"
        if p.exists():
            return p
    return Path(d) / "scores_descriptions.json"


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

@dataclass
class AxisJob:
    """One axis: where its projections come from and where its cells go."""
    axis_dir: str
    pos: str
    neg: str
    pair_type: str
    projections_file: Path
    output_dir: Path
    # filled by load_axis
    projections: Dict[int, Dict[str, Dict[str, float]]] = field(default_factory=dict, repr=False)
    entities: List[Tuple[str, str, str]] = field(default_factory=list, repr=False)
    missing: List[str] = field(default_factory=list, repr=False)
    axis_spec: Optional[Any] = field(default=None, repr=False)
    #: A multi-axis job (--axes_per_call, the rk_* rubrics): the axes judged together in each call; the
    #: job's own cell holds the answer logs and every item's k scores, each member gets its own cell.
    members: List["AxisJob"] = field(default_factory=list, repr=False)

    def map_entry(self) -> Dict[str, str]:
        """How the batch axis map records this axis (absolute paths, so a rerun
        from another directory finds the same entry)."""
        return {"axis_dir": self.axis_dir, "pos": self.pos, "neg": self.neg, "pair_type": self.pair_type,
                "projections_file": str(Path(self.projections_file).resolve()),
                "output_dir": str(Path(self.output_dir).resolve())}

    @classmethod
    def from_map_entry(cls, entry: Dict[str, str]) -> "AxisJob":
        return cls(entry["axis_dir"], entry["pos"], entry["neg"], entry["pair_type"],
                   Path(entry["projections_file"]), Path(entry["output_dir"]))


def read_axes_file(path: Path) -> List[Tuple[str, str, str, str]]:
    """Rows ``axis_dir pos neg pair_type`` (tabs or spaces), ``#`` comments allowed."""
    rows: List[Tuple[str, str, str, str]] = []
    seen: Set[str] = set()
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"{path}:{lineno}: expected 'axis_dir pos neg pair_type', got {line!r}")
        axis_dir, pos, neg, pair_type = parts
        if pair_type not in ("roles", "traits"):
            raise ValueError(f"{path}:{lineno}: pair_type {pair_type!r} is not 'roles' or 'traits'")
        if axis_dir in seen:
            raise ValueError(f"{path}:{lineno}: axis {axis_dir!r} listed twice")
        seen.add(axis_dir)
        rows.append((axis_dir, pos, neg, pair_type))
    return rows


def jobs_from_axes_file(path: Path, experiments_dir: Path) -> List[AxisJob]:
    return [AxisJob(axis_dir, pos, neg, pair_type,
                    experiments_dir / axis_dir / "gpt" / "projections.json",
                    experiments_dir / f"{axis_dir}{TIER_EVAL_SUFFIX}")
            for axis_dir, pos, neg, pair_type in read_axes_file(path)]


MULTI_DIRNAME = "_tier_eval_multi"


def axis_indices(text: str) -> Tuple[int, ...]:
    """``--axis_index``: distinct member indices, comma-separated (``0`` or ``0,3``)."""
    try:
        out = tuple(int(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: comma-separated integers") from None
    if len(set(out)) != len(out):
        raise argparse.ArgumentTypeError(f"{text!r}: an index repeated")
    return out


#: The longest group directory name kept as the members joined; a longer one is shortened (twelve axes joined
#: passed the file system's 255, 2026-11-04).
GROUP_DIRNAME_MAX = 200


def group_dirname(axis_dirs: Sequence[str]) -> str:
    """A group's directory name: its axes joined by ``+``, or, past :data:`GROUP_DIRNAME_MAX` characters, the
    first axis, the count and a digest of the whole list (``a+11more-<sha1:10>``), so that every group keeps
    one stable name."""
    joined = "+".join(axis_dirs)
    if len(joined) <= GROUP_DIRNAME_MAX:
        return joined
    return f"{axis_dirs[0]}+{len(axis_dirs) - 1}more-{hashlib.sha1(joined.encode()).hexdigest()[:10]}"


def group_jobs(jobs: Sequence[AxisJob], k: int, experiments_dir: Path) -> List[AxisJob]:
    """Consecutive groups of ``k`` axes as multi-axis jobs (the last may be shorter): the group's own
    cells go under ``<experiments_dir>/_tier_eval_multi/<axis>+<axis>.../``."""
    out: List[AxisJob] = []
    for i in range(0, len(jobs), k):
        members = list(jobs[i:i + k])
        gid = group_dirname([m.axis_dir for m in members])
        out.append(AxisJob(gid, "", "", "mixed", members[0].projections_file,
                           Path(experiments_dir) / MULTI_DIRNAME / gid, members=members))
    return out


def load_projections(path: Path, slots: str) -> Dict[int, Dict[str, Dict[str, float]]]:
    with open(path) as f:
        proj_data = json.load(f)
    # Two schema variants in the wild:
    #   new (schema_v2+): {"result": {"6": {name: {raw, whitened}}}, "_provenance": ...}
    #   old (pre-provenance): {"0": {name: {raw, whitened}}, "1": ...} — slots at top level
    projections_raw = proj_data.get("result", proj_data)
    projections = {int(s): v for s, v in projections_raw.items() if s.lstrip("-").isdigit()}
    if slots != "all":
        wanted = {int(s) for s in slots.split(",")}
        projections = {s: v for s, v in projections.items() if s in wanted}
    return projections


def resolve_entities(entity_names: Sequence[str], instructions_dir: Path,
                     load: Any = load_entity_description) -> Tuple[List[Tuple[str, str, Any]], List[str]]:
    """``(etype, name, text)`` per entity (``load``'s text, by default the description), and the names
    with no file.

    We do NOT assume the axis's pair_type applies to every entity -- the corpus
    is mixed.  Projections may use disambiguated keys like "patient|R" or
    "patient|T": the kind suffix gives the kind; otherwise roles are tried,
    then traits.  Anything that has vanished from the instructions dir since
    the projections were computed is reported (silently dropping it would
    just shrink the correlation's n)."""
    entities: List[Tuple[str, str, str]] = []
    missing: List[str] = []
    for name in entity_names:
        try:
            eid = parse_entity_id(name)
            bare, kind_hint = eid.name, eid.kind
        except Exception:
            bare, kind_hint = name, None
        if kind_hint is not None:
            desc, etype = load(instructions_dir, kind_hint, bare), kind_hint
        else:
            desc, etype = load(instructions_dir, "roles", bare), "roles"
            if desc is None:
                desc, etype = load(instructions_dir, "traits", bare), "traits"
        if desc is None:
            missing.append(name)
            continue
        entities.append((etype, name, desc))
    return entities, missing


def is_pole(name: str, spec: AxisSpec) -> bool:
    """Whether the projection key ``name`` is one of the axis's poles: by entity id, so that on a
    traits axis the role of a pole's name is judged like any other entity (production's rule,
    ``AxisSpec.pole_entity_ids``).  A bare key's kind is unknown, so it goes by its name."""
    ids = spec.pole_entity_ids()
    return name in ids if "|" in name else name in spec.pole_pair_names


def build_axis_spec(pos: str, neg: str, pair_type: str, instructions_dir: Path,
                    projections_file: Path) -> AxisSpec:
    spec = _axis_spec(pos, neg, pair_type, instructions_dir, projections_file)
    return jr.swap_axis(spec) if SWAP_POLES else spec


def _axis_spec(pos: str, neg: str, pair_type: str, instructions_dir: Path, projections_file: Path) -> AxisSpec:
    return AxisSpec(
        axis_name=f"{pos} (+) vs {neg} (-) [{pair_type}]",
        neg_pole=load_pole_description(instructions_dir, pair_type, neg),
        pos_pole=load_pole_description(instructions_dir, pair_type, pos),
        neg_examples=[neg],
        pos_examples=[pos],
        axis_by_slot={},  # not needed -- we never recompute the axis direction
        source_description=f"reused from {projections_file}",
        exclusions=[pos, neg],
        pole_pair_names=[pos, neg],
        pole_kind=pair_type,
    )


def load_axis(job: AxisJob, instructions_dir: Path, slots: str = "all", limit: Optional[int] = None) -> AxisJob:
    if job.members:
        # a multi-axis job: every entity of every member (the members' poles included; each member's
        # cell leaves its own poles out), each with its response items
        for m in job.members:
            m.projections = load_projections(m.projections_file, slots)
            m.axis_spec = build_axis_spec(m.pos, m.neg, m.pair_type, instructions_dir, m.projections_file)
        job.projections = job.members[0].projections
        job.axis_spec = jr.MultiAxisSpec(tuple((m.axis_spec, m.pair_type) for m in job.members))
        if AXIS_INDEX is not None:
            job.axis_spec = jr.MultiAxisSpec(tuple(job.axis_spec.members[i] for i in AXIS_INDEX))
        # the members' projections key entities as bare names (the May caches) or as name|R / name|T:
        # one canonical id per entity here, each member's cell written back in its own form
        names = sorted(set().union(*(set(next(iter(m.projections.values()))) for m in job.members)))
        resolved, job.missing = resolve_entities(names, instructions_dir, entity_text_loader())
        by_id: Dict[str, Tuple[str, str, Any]] = {}
        for etype, name, items in resolved:
            cid = name if "|" in name else entity_id(name, etype)
            if cid not in by_id or "|" in name:
                by_id[cid] = (etype, cid, items)
        entities = [by_id[k] for k in sorted(by_id)]
        if limit is not None:
            entities = entities[:limit]
        job.entities = response_items(entities, RESPONSE_PART, RESPONSE_SPLIT)
        if STACK:       # one copy of every item per slot (--stack)
            job.entities = [(e, f"{n}{SLOT_MARK}{j}", t) for e, n, t in job.entities for j in range(stack_slots(job))]
        return job
    job.projections = load_projections(job.projections_file, slots)
    any_slot = next(iter(job.projections.values()))
    job.axis_spec = build_axis_spec(job.pos, job.neg, job.pair_type, instructions_dir, job.projections_file)
    names = sorted(n for n in any_slot if not is_pole(n, job.axis_spec))
    if limit is not None:
        names = names[:limit]
    job.entities, job.missing = resolve_entities(names, instructions_dir, entity_text_loader())
    if job.missing:
        logger.warning(f"{job.axis_dir}: {len(job.missing)} entities from the cached projections no longer "
                       f"have an instructions file under {instructions_dir} -- excluded: {job.missing[:10]}"
                       f"{'...' if len(job.missing) > 10 else ''}")
    if RUBRIC.text_source == "responses":
        job.entities = response_items(job.entities, RESPONSE_PART, RESPONSE_SPLIT)
    return job


#: Suffix of every cell directory (and batch work directory) of this run,
#: from --cell_label: ``claude-opus-5@nothink`` holds Opus 5 judged with thinking
#: off beside ``claude-opus-5`` (low effort), so cells made under different
#: settings never overwrite or resume into each other.  Set in main_async.
CELL_LABEL = ""
#: --swap_poles: every axis judged with its ends exchanged (the first-named end, at the low number, is the
#: positive pole), the scores mapped back (judge_tier_rubrics.with_poles_swapped); set in main_async.
SWAP_POLES = False
#: --directions: which way each call presents its axis (Roger, 2026-11-04, after the pole-order test: the order
#: shifts the judges' placements, and a judge does better with the morally good pole at the high number, so a
#: fixed order would put a valence asymmetry into every result; balance the directions instead).  "own": every
#: call in the axis's own order.  "descriptions": a judge's passes alternate (--direction_pass, a coin per axis
#: deciding which pass is swapped).  "instructions": INSTRUCTION_DIRECTIONS, per axis, judge and instruction.
#: "responses": a coin per call (axis, judge and the call's members), so every entity and judge see both.
#: A swapped call shows the axis with its poles exchanged and its scores are mapped back (build_group,
#: parse_group); every answer record says which way its call went.  Set in main_async.
DIRECTIONS = "own"
DIRECTION_MODES = ("own", "descriptions", "instructions", "responses")
DIRECTION_PASS = 0
#: --stack k (Roger, 2026-11-04: stacked axes leak into one another, "if we shuffle it enough across different
#: responses, we can make it into just noise"): a multi-axis group (--axes_per_call M, an rk_* rubric) judged k
#: axes a call, each batch's axes partitioned afresh at random (STACK_SEED and the batch's members) into groups
#: of k in a random letter order, each axis's direction a coin per call; every batch is sent once per group (a
#: slot), its items named <key>#s<slot> so that the slots' scores stand apart.  0: off.  Set in main_async.
STACK = 0
STACK_SEED = 1
SLOT_MARK = "#s"
#: Roger's instruction design (2026-11-04): for instruction positions 1-5 (rows) the direction of each judge
#: (Luna, Haiku, Gemini): A or B, and on the fifth a coin per axis, d.  Each instruction is judged both ways
#: across the judges (2:1), each judge 3:2, and by weight (4:3:3) 27:23 or 23:27.  Per axis (seeded by its
#: name) a coin decides which of A and B is the axis's own order, and the instruction positions are permuted; a
#: later pass (rounds 5-9, ...) is the first reversed.
INSTRUCTION_DIRECTIONS = (("B", "A", "A"), ("A", "B", "B"), ("A", "B", "A"), ("B", "A", "B"), ("d", "d", "-d"))
INSTRUCTION_DIRECTION_JUDGES = ("gpt-5.6-luna", "claude-haiku-5-5", "gemini-3.1-flash-lite")
#: The rubric of this run (--rubric, judge_tier_rubrics.RUBRICS); v3 unless set.  Set in main_async.
RUBRIC: "jr.RubricSpec" = jr.RUBRICS["v3"]
#: The round of an i1_* rubric's cell (--instruction_round); set in main_async.
INSTRUCTION_ROUND = 0
#: Batched judging of this run (--entities_per_call, --shuffle_seed); set in main_async.
BATCHING = {"n": 1, "seed": 0, "design": "shuffle"}
#: How this run's prompts show entity names (--name_format, judge_tier_rubrics.name_shower); set in main_async.
NAME_FORMAT = jr.DEFAULT_NAME_FORMAT
NAME_SHOW = jr.name_shower()


def grid_batches(names: List[Any], n: int, design: str) -> List[List[Any]]:
    """Batches of up to ``n`` entities.  ``"shuffle"``: consecutive groups of the
    list as given.  ``"rows"`` / ``"cols"`` (Roger, 2026-11-01): the list is cut
    into squares of n x n (the last one smaller, side ceil(sqrt(k))); one pass
    judges the rows, the other the columns, so an entity's two batches share
    no other member (two independent shuffles share one in about one case in
    seven)."""
    if design == "shuffle":
        return [names[k:k + n] for k in range(0, len(names), n)]
    out: List[List[Any]] = []
    for start in range(0, len(names), n * n):
        square = names[start:start + n * n]
        side = n if len(square) == n * n else max(1, math.ceil(math.sqrt(len(square))))
        if design == "rows":
            out += [square[k:k + side] for k in range(0, len(square), side)]
        elif design == "cols":
            out += [square[c::side] for c in range(side) if square[c::side]]
        else:
            raise ValueError(f"unknown batch design {design!r}")
    return out


def _coin(*parts: Any) -> int:
    """A fair, reproducible bit from ``parts`` (SHA-256 of their text)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) & 1


def instruction_positions(axis_dir: str) -> List[int]:
    """The axis's permutation of the instruction positions: instruction i takes row ``[i]`` of
    INSTRUCTION_DIRECTIONS."""
    order = list(range(len(INSTRUCTION_DIRECTIONS)))
    random.Random(f"directions|positions|{axis_dir}").shuffle(order)
    return order


def group_swapped(job: "AxisJob", names: Sequence[str], model: str) -> bool:
    """Whether the call of ``names`` (an axis's entities or response items) by ``model`` shows the axis
    with its poles swapped, under --directions; a pure function of its arguments and the run's settings,
    so that every path that builds or reads a call agrees."""
    if DIRECTIONS == "own":
        return False
    axis = job.axis_dir
    if getattr(job, "members", None) and AXIS_INDEX is not None and len(AXIS_INDEX) == 1:
        axis = job.members[AXIS_INDEX[0]].axis_dir      # the one axis judged on the group's shared batches
    if DIRECTIONS == "descriptions":
        return bool((DIRECTION_PASS + _coin("descriptions", axis)) % 2)
    if DIRECTIONS == "instructions":
        n = len(INSTRUCTION_DIRECTIONS)
        row = INSTRUCTION_DIRECTIONS[instruction_positions(axis)[INSTRUCTION_ROUND % n]]
        d = _coin("instructions", "d", axis)
        letter = row[INSTRUCTION_DIRECTION_JUDGES.index(model)]
        is_b = {"A": 0, "B": 1, "d": d, "-d": 1 - d}[letter]
        return bool(is_b ^ _coin("instructions", "A", axis) ^ ((INSTRUCTION_ROUND // n) % 2))
    if DIRECTIONS == "responses":
        return bool(_coin("responses", axis, model, *sorted(names)))
    raise ValueError(f"unknown --directions {DIRECTIONS!r}")


def build_group(job: "AxisJob", group: Sequence[Tuple[str, str, Any]], model: str) -> Any:
    """The prompt of one batched call, its axis swapped when :func:`group_swapped` says so."""
    if STACK:
        members, swapped = _group_plan(job, [g[1] for g in group], model)
        spec = jr.MultiAxisSpec(tuple(((jr.swap_axis(job.members[m].axis_spec) if sw else job.members[m].axis_spec),
                                       job.members[m].pair_type) for m, sw in zip(members, swapped)))
        return RUBRIC.build_batch(spec, "mixed", group, show=NAME_SHOW)
    spec = job.axis_spec
    if group_swapped(job, [g[1] for g in group], model):
        spec = (jr.MultiAxisSpec(tuple((jr.swap_axis(a), pt) for a, pt in spec.members))
                if isinstance(spec, jr.MultiAxisSpec) else jr.swap_axis(spec))
    return RUBRIC.build_batch(spec, job.pair_type, group, show=NAME_SHOW)


def parse_group(job: "AxisJob", names: Sequence[str], model: str, text: Optional[str],
                swapped: Optional[bool] = None) -> Dict[int, Optional[float]]:
    """A batched reply's scores in the axis's own direction: mapped back (``lo + hi - v``) when the call
    was swapped (``swapped``, from its answer record, or else :func:`group_swapped`)."""
    parsed = RUBRIC.parse_batch(text, len(names))
    lo, hi = RUBRIC.lo, RUBRIC.hi
    if STACK:
        # each item's scores, in letter order, onto its members' indices, each in its axis's own direction
        members, ways = _group_plan(job, names, model)
        out: Dict[int, Any] = {}
        for k, v in parsed.items():
            if v is not None and not isinstance(v, (list, tuple)):
                v = [v]     # --stack 1: a one-axis rubric's scalar
            ok = isinstance(v, (list, tuple)) and len(v) == len(members) and all(x is not None for x in v)
            out[k] = {m: (lo + hi - x if sw else x) for m, sw, x in zip(members, ways, v)} if ok else None
        return out
    if not (group_swapped(job, names, model) if swapped is None else swapped):
        return parsed
    return {k: None if v is None else lo + hi - v for k, v in parsed.items()}


def stack_real(name: str) -> str:
    """A stacked item's own key, without its slot."""
    return name.split(SLOT_MARK, 1)[0]


def stack_slots(job: "AxisJob") -> int:
    return -(-len(job.members) // STACK) if STACK else 1


def stack_plan(job: "AxisJob", real_names: Sequence[str], slot: int, model: str) -> Tuple[List[int], List[bool]]:
    """The members (indices into ``job.members``, in letter order) of one batch's slot under --stack, and each
    one's direction (True: swapped): the batch's axes in a random order fixed by STACK_SEED and its members, cut
    into groups of STACK; a coin per axis, call and judge."""
    order = list(range(len(job.members)))
    random.Random("|".join(["stack", job.axis_dir, str(STACK_SEED), *sorted(real_names)])).shuffle(order)
    chunk = order[slot * STACK:(slot + 1) * STACK]
    return chunk, [bool(_coin("stack", job.members[m].axis_dir, model, slot, STACK_SEED, *sorted(real_names)))
                   for m in chunk]


def _group_plan(job: "AxisJob", names: Sequence[str], model: str) -> Tuple[List[int], List[bool]]:
    return stack_plan(job, [stack_real(n) for n in names], int(names[0].split(SLOT_MARK, 1)[1]), model)


def build_prompt(job: "AxisJob", etype: str, name: str, desc: str) -> str:
    """The judge prompt of this run's rubric for one entity of ``job``."""
    return RUBRIC.build(job.axis_spec, job.pair_type, etype, name, desc, show=NAME_SHOW)


def parse_score(text: Optional[str]) -> Optional[float]:
    return RUBRIC.parse(text)


def cell_name(model: str) -> str:
    """The cell directory name of ``model`` in this run (the model plus --cell_label)."""
    return f"{model}{CELL_LABEL}"


def cell_dir(job: AxisJob, model: str) -> Path:
    return Path(job.output_dir) / cell_name(model)


def cell_done(job: AxisJob, model: str) -> bool:
    """The resume rule: a model's cell counts as done once its usage.json exists."""
    return (cell_dir(job, model) / "usage.json").exists()


# ---------------------------------------------------------------------------
# Cells and per-model run state
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One (axis, model) cell being filled: every attempted entity's score (None
    when unparseable, failed or cut at the cap), the entities whose reply was
    cut at max_tokens, and the cell's own usage."""
    idx: int
    job: AxisJob
    expected: Set[str]
    scores: Dict[str, Optional[int]] = field(default_factory=dict)
    truncated: Set[str] = field(default_factory=set)
    refused: Set[str] = field(default_factory=set)      # no reply at all for the entity alone or in its group
    parse_retries: int = 0                              # re-asks of a group whose reply left members unread
    usage: MultiModelUsage = field(default_factory=MultiModelUsage)
    written: bool = False
    swapped: Dict[str, bool] = field(default_factory=dict)  # --directions: the direction each name was judged in

    @property
    def complete(self) -> bool:
        return self.expected <= set(self.scores)

    @property
    def finished(self) -> bool:
        """Every entity has its outcome and at least one has a score: the cell may be written
        as done.  A call that got no answer leaves its entities out of ``scores``."""
        return self.complete and any(s is not None for s in self.scores.values())

    def parsed_scores(self) -> Dict[str, int]:
        return {n: s for n, s in sorted(self.scores.items()) if s is not None}


@dataclass
class ModelRun:
    model: str
    settings: Dict[str, Any]
    choice: tt.TransportChoice
    transport: str = ""
    transport_reason: str = ""
    stats: tt.CallStats = field(default_factory=tt.CallStats)
    usage: MultiModelUsage = field(default_factory=MultiModelUsage)
    attempted: int = 0
    parsed: int = 0
    cells_written: int = 0
    cells_partial: int = 0
    cells_incomplete: int = 0
    left_for_rerun: int = 0
    resumed: int = 0        # answers taken from the answer logs of earlier runs
    parse_retries: int = 0  # re-asks of a group whose reply left members unread
    to_judge: int = 0       # live or Flex requests this run set out to send
    in_flight: int = 0
    error: Optional[BaseException] = None
    live_cells: List[Cell] = field(default_factory=list)   # the last live round's, for the rerun record

    def __post_init__(self) -> None:
        self.transport = self.transport or self.choice.transport
        self.transport_reason = self.transport_reason or self.choice.reason


class _Charge:
    """Charges one call to several trackers (the cell's, the model's, the run's)
    and keeps the charges, ``[key, prompt_tokens, completion_tokens]`` and the
    cache-written and cache-read tokens when there are any, for the answer log."""

    def __init__(self, *trackers: MultiModelUsage):
        self.trackers = trackers
        self.charges: List[List[Any]] = []

    def charge_usage(self, key: str, usage: TokenUsage) -> None:
        cached = [usage.cache_write_tokens, usage.cache_read_tokens] if usage.cache_write_tokens or usage.cache_read_tokens else []
        self.charges.append([key, usage.prompt_tokens, usage.completion_tokens, *cached])
        for t in self.trackers:
            t.charge_usage(key, usage)

    def charge(self, key: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.charge_usage(key, TokenUsage(int(prompt_tokens), int(completion_tokens)))


@dataclass
class RunContext:
    instructions_dir: Path
    batch_root: Path
    max_tokens: int
    requests_per_second: int
    budget_usd: Optional[float] = None
    batch_poll_seconds: float = 60.0
    batch_fallback_max: int = DEFAULT_BATCH_FALLBACK_MAX
    slots: str = "all"
    limit: Optional[int] = None
    run_usage: MultiModelUsage = field(default_factory=MultiModelUsage)
    budget_hit: bool = False
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT
    stall_s: float = DEFAULT_STALL_MINUTES * 60
    batch_chunk_requests: int = DEFAULT_BATCH_CHUNK_REQUESTS
    entities_per_call: int = 1      # batched judging (--entities_per_call); live and Flex only
    shuffle_seed: int = 0
    batch_design: str = "shuffle"   # "shuffle": consecutive groups of a seeded shuffle; "rows" / "cols": grid passes
    live_pending: int = 0           # live or Flex requests not yet finished, all models
    last_progress: float = field(default_factory=time.monotonic)
    stalled: bool = False

    def add_pending(self, n: int) -> None:
        if n > 0 and self.live_pending == 0:
            self.last_progress = time.monotonic()   # the stall clock starts with the work
        self.live_pending += n

    def note_answer(self) -> None:
        self.last_progress = time.monotonic()

    def over_budget(self) -> bool:
        if self.budget_usd is None or self.run_usage.total_cost_usd <= self.budget_usd:
            return False
        if not self.budget_hit:
            self.budget_hit = True
            logger.error(f"--budget_usd {self.budget_usd:.2f} reached (${self.run_usage.total_cost_usd:.2f} "
                         f"spent this run): no further requests are issued; requests in flight finish")
        return True


_UNSET = object()


def record(mrun: ModelRun, cell: Cell, name: str, reply: Any, score: Any = _UNSET,
           swapped: Optional[bool] = None) -> None:
    """Record one reply: its text, None (a failed call or a refusal), or a
    :class:`judge_tier_transport.Truncated` (cut at max_tokens), which scores
    None without being parsed and is listed in the cell's provenance.  With
    ``score`` given (batched judging), the reply is not parsed again."""
    mrun.attempted += 1
    if swapped is not None:
        cell.swapped[name] = swapped
    if isinstance(reply, tt.Truncated):
        cell.scores[name] = None
        cell.truncated.add(name)
        mrun.stats.truncated += 1
        logger.warning(f"{mrun.model}: reply for {name!r} ({cell.job.axis_dir}) cut at max_tokens: "
                       f"scored as missing")
        return
    cell.truncated.discard(name)
    if reply is None:
        cell.refused.add(name)
    score = (parse_score(reply) if reply else None) if score is _UNSET else score
    cell.scores[name] = score
    if score is None:
        logger.warning(f"{mrun.model}: unparseable/empty response for {name!r} ({cell.job.axis_dir})")
    else:
        mrun.parsed += 1


def write_cell(cell: Cell, mrun: ModelRun, max_tokens: int) -> None:
    """Write the cell's scores_<mode>.json, correlations.json and, last,
    usage.json (the file the resume rule looks for)."""
    job, model = cell.job, mrun.model
    d = cell_dir(job, model)
    d.mkdir(parents=True, exist_ok=True)
    scores = cell.parsed_scores()
    provenance = {"judge_model": model, "rubric_version": RUBRIC.version, "rubric": RUBRIC.name,
                  "projections_file": str(job.projections_file), "swap_poles": SWAP_POLES,
                  "transport": mrun.transport, "transport_reason": mrun.transport_reason,
                  "request_settings": mrun.settings, "max_tokens": max_tokens,
                  "cell_label": CELL_LABEL, "truncated": sorted(cell.truncated),
                  "entities_per_call": BATCHING["n"], "shuffle_seed": BATCHING["seed"],
                  "batch_design": BATCHING["design"], "name_format": NAME_FORMAT,
                  "n_entities": len(cell.expected), "n_scored": len(scores)}
    if DIRECTIONS != "own":
        provenance.update(direction_summary(cell))
    if cell.refused:
        provenance["refused"] = sorted(cell.refused)
    if cell.parse_retries:
        provenance["parse_retries"] = cell.parse_retries
    if RUBRIC.text_source != "descriptions":
        provenance["text_source"] = RUBRIC.text_source
    if RUBRIC.text_source == "instruction_rounds":
        provenance["instruction_round"] = INSTRUCTION_ROUND
    payload: Dict[str, Any] = {"result": scores, "_provenance": provenance}
    correlations: Dict[str, Any] = {}
    if RUBRIC.text_source == "responses":
        provenance.update({"responses_dir": str(RESPONSES_DIR), "response_split": RESPONSE_SPLIT,
                           "response_part": RESPONSE_PART, "response_chars": RESPONSE_CHARS,
                           "response_design": RESPONSE_DESIGN if RESPONSE_PART is not None else None,
                           "n_items": len(cell.expected), "n_items_scored": len(scores)})
        provenance["n_entities"] = len({response_item_entity(k) for k in cell.expected})
        if job.members:
            # the group's cell keeps every item's k scores; each member's cell is an ordinary response cell
            provenance["multi_axis"] = [m.axis_dir for m in job.members]
            provenance["axis_indices"] = list(AXIS_INDEX) if AXIS_INDEX is not None else None
            payload = {"result": {}, "items": scores, "_provenance": provenance}
            provenance["n_scored"] = 0
            write_member_cells(cell, mrun, max_tokens, scores, provenance)
        else:
            # result: the per-entity means, what every consumer reads; items: each response's own score
            payload = {"result": entity_means(scores), "items": scores, "_provenance": provenance}
            provenance["n_scored"] = len(payload["result"])
            scores = payload["result"]
    atomic_write_text(json.dumps(payload, indent=2), d / f"scores_{RUBRIC.mode}.json", logger_obj=logger)
    if not job.members:
        correlations = compute_correlations(
            scores_by_mode={RUBRIC.mode: scores},
            projections=job.projections,
            slots=list(job.projections.keys()),
            excluded_set=job.axis_spec.pole_entity_ids(),
        )
        atomic_write_text(json.dumps(correlations, indent=2), d / "correlations.json", logger_obj=logger)
    log = d / ANSWERS_PARTIAL
    replies = d / BATCH_REPLIES
    if log.exists():
        # the raw replies stay, compressed; the log itself goes once usage.json is written
        atomic_write_bytes(gzip.compress(log.read_bytes()), d / ANSWERS_ARCHIVE, logger_obj=logger)
    if replies.exists():
        atomic_write_bytes(gzip.compress(replies.read_bytes()), d / BATCH_REPLIES_ARCHIVE, logger_obj=logger)
    atomic_write_text(json.dumps(cell.usage.as_dict(), indent=2, sort_keys=True), d / "usage.json",
                      logger_obj=logger)
    log.unlink(missing_ok=True)
    replies.unlink(missing_ok=True)
    for stale in (f"scores_{RUBRIC.mode}.partial.json", "usage.partial.json"):
        (d / stale).unlink(missing_ok=True)
    cell.written = True
    mrun.cells_written += 1
    logger.info(f"[{model}] {job.axis_dir}: scored {len(cell.parsed_scores())}/{len(cell.expected)}"
                f"{' responses' if RUBRIC.text_source == 'responses' else ''}"
                f"{f', {len(cell.truncated)} cut at max_tokens' if cell.truncated else ''}; "
                f"cost ${cell.usage.total_cost_usd:.4f} (see {d / 'usage.json'})")
    for slot, per_metric in correlations.get(RUBRIC.mode, {}).items():
        raw = per_metric["raw"]
        logger.info(f"[{model}] {job.axis_dir} slot={slot} raw: rho={raw['rho']:.4f} n={raw['n']}")


def scale_usage(usage: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """A usage record's counts and costs times ``factor`` (a multi-axis call's cost shared by its axes)."""
    def scale(v: Any) -> Any:
        if isinstance(v, bool) or isinstance(v, str):
            return v
        if isinstance(v, int):
            return int(round(v * factor))
        if isinstance(v, float):
            return v * factor
        if isinstance(v, dict):
            return {k: scale(x) for k, x in v.items()}
        return v
    return scale(usage)


def write_member_cells(cell: Cell, mrun: ModelRun, max_tokens: int, items: Dict[str, Any],
                       provenance: Dict[str, Any]) -> None:
    """Each axis of a multi-axis cell as its own response cell, ``<axis>_tier_eval/<model><label>/``: its
    scores of the items (the axis's own poles left out), the per-entity means, correlations.json and a
    usage.json of the call's cost shared equally (so every consumer reads it as a cell judged alone)."""
    job, model = cell.job, mrun.model
    members = [(i, job.members[i]) for i in AXIS_INDEX] if AXIS_INDEX is not None else list(enumerate(job.members))
    k = len(members)
    usage = cell.usage.as_dict()
    if STACK:
        # --stack: an item's slot copies hold {member: score} for the axes of their calls; each member takes its
        # one score of every item, under the item's own key
        stacked = {key: v for key, v in items.items() if isinstance(v, dict)}
        items = {}
    else:
        items = {key: (v if isinstance(v, (list, tuple)) else [v]) for key, v in items.items() if v is not None}
        bad = sum(1 for v in items.values() if len(v) != k)
        if bad:
            logger.warning(f"[{model}] {job.axis_dir}: {bad} items without exactly {k} scores, left unscored")
    for pos, (i, m) in enumerate(members):
        pnames = set(next(iter(m.projections.values())))

        def own_key(key: str) -> str:
            """The item key in this member's form: the canonical id where its projections have it, else
            the bare name (the May caches)."""
            cid, rest = key.rsplit("|", 1)
            return key if cid in pnames or parse_entity_id(cid).name not in pnames else f"{parse_entity_id(cid).name}|{rest}"

        if STACK:
            own = {own_key(stack_real(key)): int(v[i]) for key, v in stacked.items()
                   if v.get(i) is not None and not is_pole(response_item_entity(own_key(stack_real(key))), m.axis_spec)}
            expected = sorted({own_key(stack_real(key)) for key in cell.expected
                               if not is_pole(response_item_entity(own_key(stack_real(key))), m.axis_spec)})
            ways = [w[i] for key, w in cell.swapped.items() if isinstance(w, dict) and i in w and key in stacked]
            way = {"stack": STACK, "stack_seed": STACK_SEED,
                   "direction_counts": {"own": sum(not w for w in ways), "swapped": sum(bool(w) for w in ways)}}
        else:
            own = {own_key(key): int(v[pos]) for key, v in items.items()
                   if len(v) == k and v[pos] is not None and not is_pole(response_item_entity(own_key(key)), m.axis_spec)}
            expected = [own_key(key) for key in cell.expected if not is_pole(response_item_entity(own_key(key)), m.axis_spec)]
            way = {}
        result = entity_means(own)
        prov = {**provenance, **way, "axis_index": i, "projections_file": str(m.projections_file),
                "axis_letter": jr.AXIS_LETTERS[pos] if k > 1 and not STACK else None,
                "n_items": len(expected), "n_items_scored": len(own),
                "n_entities": len({response_item_entity(key) for key in expected}), "n_scored": len(result)}
        dm = Path(m.output_dir) / cell_name(model)
        dm.mkdir(parents=True, exist_ok=True)
        atomic_write_text(json.dumps({"result": result, "items": dict(sorted(own.items())), "_provenance": prov}, indent=2),
                          dm / f"scores_{RUBRIC.mode}.json", logger_obj=logger)
        corr = compute_correlations(scores_by_mode={RUBRIC.mode: result}, projections=m.projections,
                                    slots=list(m.projections.keys()), excluded_set=m.axis_spec.pole_entity_ids())
        atomic_write_text(json.dumps(corr, indent=2), dm / "correlations.json", logger_obj=logger)
        atomic_write_text(json.dumps(scale_usage(usage, 1.0 / k), indent=2, sort_keys=True), dm / "usage.json",
                          logger_obj=logger)
        raw = corr[RUBRIC.mode].get("6", next(iter(corr[RUBRIC.mode].values())))["raw"]
        logger.info(f"[{model}] {m.axis_dir} (axis {i} of {job.axis_dir}{'' if STACK else ', letter ' + jr.AXIS_LETTERS[pos]}): scored {len(own)}/{len(expected)} "
                    f"responses, {len(result)} entities; slot 6 raw rho={raw['rho']:.4f}")


def write_partial_cell(cell: Cell, mrun: ModelRun, max_tokens: int) -> None:
    """A cell left incomplete (a budget stop, or calls with no answer): its scores and spend so far
    under ``*.partial.json`` (no usage.json, so a rerun judges the axis again).  The cell's usage
    already holds what earlier runs spent, from the answer log, so the file is replaced, not added to."""
    job, model = cell.job, mrun.model
    d = cell_dir(job, model)
    d.mkdir(parents=True, exist_ok=True)
    provenance = {"judge_model": model, "rubric_version": RUBRIC.version, "rubric": RUBRIC.name,
                  "projections_file": str(job.projections_file), "transport": mrun.transport,
                  "request_settings": mrun.settings, "max_tokens": max_tokens, "complete": False,
                  "n_attempted": len(cell.scores), "n_entities": len(cell.expected),
                  "truncated": sorted(cell.truncated)}
    atomic_write_text(json.dumps({"result": cell.parsed_scores(), "_provenance": provenance}, indent=2),
                      d / f"scores_{RUBRIC.mode}.partial.json", logger_obj=logger)
    atomic_write_text(json.dumps(cell.usage.as_dict(), indent=2, sort_keys=True), d / "usage.partial.json",
                      logger_obj=logger)
    mrun.cells_partial += 1
    logger.warning(f"[{model}] {job.axis_dir}: incomplete ({len(cell.scores)}/{len(cell.expected)} attempted); "
                   f"written as {d / f'scores_{RUBRIC.mode}.partial.json'}, no usage.json")


# ---------------------------------------------------------------------------
# Saved answers: the per-cell answer log (2026-11-01)
# ---------------------------------------------------------------------------

class CheckpointMismatchError(RuntimeError):
    """An answer log written under other settings than this run's."""


def checkpoint_header(mrun: "ModelRun", max_tokens: int) -> Dict[str, Any]:
    return {"kind": "header", "version": CHECKPOINT_VERSION, "judge_model": mrun.model, "cell_label": CELL_LABEL,
            "rubric_version": RUBRIC.version, "request_settings": mrun.settings, "max_tokens": max_tokens,
            "name_format": NAME_FORMAT, "swap_poles": SWAP_POLES, "directions": DIRECTIONS,
            "direction_pass": DIRECTION_PASS, "stack": STACK, "stack_seed": STACK_SEED,
            "response_design": response_design_in_effect()}


def prompt_digest(prompt: str) -> str:
    """Identifies the exact prompt an answer replied to (corpus text and rubric included; a two-part prompt
    by its parts)."""
    text = prompt if isinstance(prompt, str) else "\x1e".join(prompt)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_answer_log(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """``(header, answers by entity)`` of an answer log; an unreadable line (a
    run stopped while writing it) is skipped, and a later answer for an entity
    replaces an earlier one."""
    header: Optional[Dict[str, Any]] = None
    answers: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return None, answers
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                logger.warning(f"{path}:{n}: unreadable line skipped (a run stopped while writing it?)")
                continue
            if rec.get("kind") == "header":
                header = header or rec
            elif rec.get("kind") == "answer" and isinstance(rec.get("name"), str):
                answers[rec["name"]] = rec
    return header, answers


def read_reasks(path: Path) -> List[Dict[str, Any]]:
    """The ``reask`` lines of an answer log: each a reply that left members unread and was asked again,
    with that reply's charges, which no answer line carries."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") == "reask":
                out.append(rec)
    return out


def resume_cell(cell: Cell, mrun: "ModelRun", header: Dict[str, Any]) -> int:
    """Fill ``cell`` from its answer log: every saved answer to the very prompt
    this run would send, with its charges in the cell's usage (not this run's,
    which the budget counts).  Entities already scored (batch results) are
    left alone.  Returns the number taken."""
    path = cell_dir(cell.job, mrun.model) / ANSWERS_PARTIAL
    if not path.exists():
        return 0
    last = b"\n"
    with open(path, "rb") as f:
        if f.seek(0, os.SEEK_END):
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
    if last != b"\n":
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")   # a torn last line must not swallow the next answer
    found, answers = read_answer_log(path)
    reasks = read_reasks(path)
    if not answers and not reasks:
        return 0
    if found is None:
        raise CheckpointMismatchError(f"{path}: no header line, so its settings are unknown: move it aside to "
                                      f"judge {cell.job.axis_dir} afresh")
    diffs = {k: (found.get(k, CHECKPOINT_DEFAULTS.get(k)), header.get(k)) for k in CHECKPOINT_KEYS
             if found.get(k, CHECKPOINT_DEFAULTS.get(k)) != header.get(k)}
    if diffs:
        raise CheckpointMismatchError(
            f"{path} was written under other settings than this run's ("
            + "; ".join(f"{k}: {a!r} there, {b!r} now" for k, (a, b) in diffs.items())
            + "): rerun with those settings to finish it, use another --cell_label, or move the file aside "
              f"to judge {cell.job.axis_dir} afresh")
    for rec in reasks:
        cell.parse_retries += 1
        for key, *counts in rec.get("charges") or []:
            cell.usage.charge(key, *counts)
    descs = {name: (etype, desc) for etype, name, desc in cell.job.entities}
    taken = stale = 0
    for name, rec in answers.items():
        if name not in cell.expected:
            continue
        if name in cell.scores:
            # scored by a batch result, but this answer was billed all the same (and a group's charges
            # are on its first member's record, whoever else the batch scored)
            for key, *counts in rec.get("charges") or []:
                cell.usage.charge(key, *counts)
            continue
        etype, desc = descs[name]
        if not rec.get("batch") and rec.get("prompt") != prompt_digest(build_prompt(cell.job, etype, name, desc)):
            stale += 1
            continue
        if rec.get("batch"):
            # a batched reply: this entity's line, by its position in the batch
            batch = rec["batch"]
            sw = rec.get("swapped")
            if sw is None:
                sw = group_swapped(cell.job, batch, mrun.model)
            parsed = ({} if rec.get("truncated") or not rec.get("reply") or RUBRIC.parse_batch is None
                      else parse_group(cell.job, batch, mrun.model, rec["reply"], swapped=sw))
            restore_score(cell, name, parsed.get(batch.index(name) + 1) if parsed and name in batch else None,
                          bool(rec.get("truncated")))
            if DIRECTIONS != "own":
                cell.swapped[name] = bool(sw)
            if STACK and rec.get("stack"):
                cell.swapped[name] = dict(zip(rec["stack"]["members"], rec["stack"]["swapped"]))
        else:
            restore(cell, name, rec.get("reply"), bool(rec.get("truncated")))
        if rec.get("reply") is None:
            cell.refused.add(name)
        for key, *counts in rec.get("charges") or []:
            cell.usage.charge(key, *counts)
        taken += 1
    if stale:
        logger.warning(f"{path}: {stale} saved answers replied to another prompt (corpus or rubric text "
                       f"changed): judged again")
    mrun.resumed += taken
    return taken


def restore_score(cell: Cell, name: str, score: Optional[float], truncated: bool) -> None:
    """A saved batched answer back into its cell, already parsed."""
    if truncated:
        cell.scores[name] = None
        cell.truncated.add(name)
        return
    cell.truncated.discard(name)
    cell.scores[name] = score


def restore(cell: Cell, name: str, text: Optional[str], truncated: bool) -> None:
    """A saved answer back into its cell, scored as :func:`record` scores it."""
    if truncated:
        cell.scores[name] = None
        cell.truncated.add(name)
        return
    cell.truncated.discard(name)
    cell.scores[name] = parse_score(text) if text else None


def append_answer(cell: Cell, model: str, header: Dict[str, Any], name: str, reply: Any,
                  charges: List[List[Any]], prompt: str, batch: Optional[List[str]] = None) -> None:
    """Save one billed answer to the cell's answer log as soon as it arrives
    (the header first, when the log is new).  Calls that failed without a
    charge are not saved, so a rerun asks them again."""
    truncated = isinstance(reply, tt.Truncated)
    way = ({"swapped": group_swapped(cell.job, batch, model)} if batch and DIRECTIONS != "own" else {})
    if STACK and batch:
        members, ways = _group_plan(cell.job, batch, model)
        way = {"stack": {"members": members, "swapped": ways}}
        cell.swapped[name] = dict(zip(members, ways))
    _append_log(cell, model, header, {
        "kind": "answer", "name": name, "reply": reply.text if truncated else reply, "truncated": truncated,
        "charges": charges, "prompt": prompt_digest(prompt), **({"batch": batch} if batch else {}), **way,
        "at": _utc_now()})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _append_log(cell: Cell, model: str, header: Dict[str, Any], rec: Dict[str, Any]) -> None:
    d = cell_dir(cell.job, model)
    d.mkdir(parents=True, exist_ok=True)
    path = d / ANSWERS_PARTIAL
    lines = [] if path.exists() else [json.dumps(header, sort_keys=True)]
    lines.append(json.dumps(rec, sort_keys=True))
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()


def append_batch_reply(cell: Cell, model: str, cid: str, members: List[str], text: Optional[str],
                       finish: Optional[str], refused: bool) -> None:
    """Keep one Batch API reply as it is collected, for the refusal counts and for reading afterwards."""
    d = cell_dir(cell.job, model)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / BATCH_REPLIES, "a", encoding="utf-8") as f:
        f.write(json.dumps({"custom_id": cid, "members": members, "reply": text, "finish": finish,
                            "refused": refused, "at": _utc_now()}, sort_keys=True) + "\n")


def record_group_reply(mrun: "ModelRun", cell: Cell, header: Dict[str, Any], group: Sequence[Tuple[str, str, Any]],
                       text: Any, charges: List[List[Any]], prompt: str) -> None:
    """One batched reply into its cell: each member's line, the call's charges on the first member's record."""
    names = [g[1] for g in group]
    sw = group_swapped(cell.job, names, mrun.model) if DIRECTIONS != "own" else None
    scores = ({} if isinstance(text, tt.Truncated) or not text else parse_group(cell.job, names, mrun.model, text))
    for k, gname in enumerate(names, 1):
        if charges:
            append_answer(cell, mrun.model, header, gname, text, charges if k == 1 else [], prompt, batch=names)
        record(mrun, cell, gname, text, score=scores.get(k) if scores else None, swapped=sw)


async def read_group_reply(ctx: "RunContext", mrun: "ModelRun", cell: Cell, header: Dict[str, Any],
                           group: Sequence[Tuple[str, str, Any]], text: Any, charges: List[List[Any]], prompt: str,
                           call: Any) -> None:
    """A batched reply into its cell, asking the same prompt again while replies leave members unread (up to
    PARSE_RETRIES times).  Each member keeps the first reply that gives it a line; members a batch result has
    already scored are left alone.  A failed re-ask leaves the unread open, and the cell with them.  Every
    re-asked reply goes on a ``reask`` line with its charges, so a rerun still counts them."""
    if text is None or isinstance(text, tt.Truncated):
        record_group_reply(mrun, cell, header, group, text, charges, prompt)
        return
    names = [g[1] for g in group]
    sw = group_swapped(cell.job, names, mrun.model) if DIRECTIONS != "own" else None
    unread = [n for n in names if n not in cell.scores]
    for attempt in range(PARSE_RETRIES + 1):
        parsed = parse_group(cell.job, names, mrun.model, text) if text else {}
        read = [n for n in unread if parsed.get(names.index(n) + 1) is not None]
        unread = [n for n in unread if n not in read]
        last = not unread or attempt == PARSE_RETRIES
        for n in read:
            append_answer(cell, mrun.model, header, n, text, charges if last else [], prompt, batch=names)
            if last:
                charges = []
            record(mrun, cell, n, text, score=parsed[names.index(n) + 1], swapped=sw)
        if last:
            break
        _append_log(cell, mrun.model, header, {
            "kind": "reask", "batch": names, "unread": unread, "reply": text, "charges": charges,
            "prompt": prompt_digest(prompt), "at": _utc_now()})
        charge = _Charge(cell.usage, mrun.usage, ctx.run_usage)
        mrun.to_judge += 1
        ctx.add_pending(1)
        mrun.in_flight += 1
        cell.parse_retries += 1
        mrun.parse_retries += 1
        try:
            text = await call(prompt, charge)
        finally:
            mrun.in_flight -= 1
            ctx.live_pending -= 1
        ctx.note_answer()
        if text is tt.SKIPPED or text is tt.FAILED:
            return      # no outcome for the unread: the cell stays open and they are asked again
        charges = charge.charges
        text = "" if text is None else text.text if isinstance(text, tt.Truncated) else text
    for n in unread:    # the last reply left them unread: their outcome, on its record
        append_answer(cell, mrun.model, header, n, text, charges, prompt, batch=names)
        charges = []
        record(mrun, cell, n, text, score=None, swapped=sw)


async def judge_split_group(ctx: "RunContext", mrun: "ModelRun", cell: Cell, header: Dict[str, Any],
                            group: Sequence[Tuple[str, str, Any]], call: Any, carried: Sequence[List[Any]] = ()) -> None:
    """A group whose call came back with nothing (a refusal, a content flag, an empty reply) has lost
    every member's score, usually to one of them: ask its two halves, and so on down, so that only the
    entity that refuses goes without.  ``call(prompt, charge)`` sends a request as the caller's path
    does.  ``carried``: the charges of the calls that came back with nothing, which go on the next
    answer's record, so that a rerun still counts them."""
    _append_log(cell, mrun.model, header, {
        "kind": "refusal", "batch": [g[1] for g in group], "at": _utc_now(),
        "prompt": prompt_digest(build_group(cell.job, group, mrun.model))})
    pending = list(carried)
    mid = len(group) // 2
    for part in (group[:mid], group[mid:]):
        prompt = build_group(cell.job, part, mrun.model)
        charge = _Charge(cell.usage, mrun.usage, ctx.run_usage)
        mrun.to_judge += 1
        ctx.add_pending(1)
        mrun.in_flight += 1
        try:
            text = await call(prompt, charge)
        finally:
            mrun.in_flight -= 1
            ctx.live_pending -= 1
        if text is tt.SKIPPED or text is tt.FAILED:
            continue        # no outcome for these: the cell stays open and they are asked again
        pending += charge.charges
        if text is None and len(part) > 1:
            await judge_split_group(ctx, mrun, cell, header, part, call, pending)
        else:
            await read_group_reply(ctx, mrun, cell, header, part, text, pending, prompt, call)
        pending = []
        ctx.note_answer()


def saved_answers(job: AxisJob, model: str) -> int:
    """How many entities of ``job`` the cell's answer log holds an answer for
    (for the dry run; the run itself also checks settings and prompts)."""
    _, answers = read_answer_log(cell_dir(job, model) / ANSWERS_PARTIAL)
    names = {name for _, name, _ in job.entities}
    return sum(1 for name in answers if name in names)


async def run_bounded(worker, items: Sequence[Any], limit: int) -> None:
    """``await worker(item)`` for every item, at most ``limit`` at a time, in
    order.  The first exception stops the rest and is raised as it is."""
    if not items:
        return
    it = iter(items)

    async def lane() -> None:
        for item in it:   # a shared iterator: each lane takes the next item
            await worker(item)

    try:
        async with asyncio.TaskGroup() as tg:
            for _ in range(max(1, min(limit, len(items)))):
                tg.create_task(lane())
    except BaseExceptionGroup as eg:
        raise eg.exceptions[0]


# ---------------------------------------------------------------------------
# Clients (module-level so tests can replace them)
# ---------------------------------------------------------------------------

def make_openai_client(service_tier: Optional[str] = None):
    import openai
    if service_tier == "flex":
        # Flex calls can wait in a queue and be refused with an uncharged 429:
        # a longer timeout and more SDK retries, as judge.judge_client.
        return openai.AsyncOpenAI(timeout=tt.FLEX_TIMEOUT_S, max_retries=tt.FLEX_MAX_RETRIES)
    return openai.AsyncOpenAI()


def make_anthropic_client():
    import anthropic
    return anthropic.AsyncAnthropic()


def key_name(model: str) -> str:
    """The environment variable holding the key ``model`` needs (only its presence is checked)."""
    return {"anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY"}.get(provider_for_model(model), "OPENAI_API_KEY")


def make_gemini_client():
    return tt.make_gemini_client(os.environ["GEMINI_API_KEY"])


def standard_client(model: str):
    """The client for ``model``'s standard tier: its batch API, and real time."""
    provider = provider_for_model(model)
    if provider == "anthropic":
        return make_anthropic_client()
    if provider == "google":
        return make_gemini_client()
    return make_openai_client(None)


# ---------------------------------------------------------------------------
# Real time and Flex
# ---------------------------------------------------------------------------

async def run_live(ctx: RunContext, mrun: ModelRun, todo: List[AxisJob], *, tier: Optional[str],
                   probe: bool = False) -> bool:
    """Judge every entity of ``todo`` in real time (``tier`` None) or on Flex,
    axis by axis with at most ``ctx.max_in_flight`` requests in flight,
    writing each cell as soon as its axis is complete.  Every billed answer is
    saved to its cell's answer log as it arrives, and answers an earlier run
    saved are taken from there instead of asked again.  With ``probe``, the
    first request goes alone and a service-tier refusal returns False before
    anything else is sent."""
    model = mrun.model
    rate_limiter = RateLimiter(ctx.requests_per_second)
    if provider_for_model(model) == "anthropic":
        client, fallback = make_anthropic_client(), None
    elif provider_for_model(model) == "google":
        client, fallback = make_gemini_client(), None
    else:
        client = make_openai_client(tier)
        fallback = make_openai_client(None) if tier == "flex" else None
    header = checkpoint_header(mrun, ctx.max_tokens)
    cells = {i: Cell(i, job, expected={name for _, name, _ in job.entities}) for i, job in enumerate(todo)}
    mrun.live_cells = list(cells.values())
    for cell in cells.values():
        resume_cell(cell, mrun, header)
        if cell.finished and not cell.written:
            write_cell(cell, mrun, ctx.max_tokens)
    items = [(i, etype, name, desc) for i, job in enumerate(todo) for etype, name, desc in job.entities
             if not cells[i].written and name not in cells[i].scores]
    n_per = ctx.entities_per_call
    if n_per > 1:
        # groups of n_per entities of one axis, in a seeded shuffle: the batch-mates are random, and
        # another --shuffle_seed gives every entity different ones
        groups: List[Tuple[int, List[Tuple[str, str, str]]]] = []
        for i in sorted({it[0] for it in items}):
            # groups come from the axis's FULL entity list (so rows and cols of one seed agree, and the
            # Batch API path makes the same groups), minus what is already scored
            todo_names = {it[2] for it in items if it[0] == i}
            batches = axis_groups(cells[i].job, ctx)
            groups += [(i, g) for g in ([[e for e in b if e[1] in todo_names] for b in batches]) if g]
        if mrun.resumed:
            logger.info(f"[{model}] {mrun.resumed} answers taken from the answer logs of earlier runs")
        logger.info(f"[{model}] batched judging ({ctx.batch_design}): {len(items)} entities in {len(groups)} calls of "
                    f"up to {n_per}, shuffle seed {ctx.shuffle_seed}")
        items = groups
    elif mrun.resumed:
        logger.info(f"[{model}] {mrun.resumed} answers taken from the answer logs of earlier runs; "
                    f"{len(items)} requests left")
    mrun.to_judge += len(items)
    ctx.add_pending(len(items))
    unfinished = len(items)
    fatal: List[BaseException] = []

    def gate() -> bool:
        return bool(fatal) or ctx.over_budget()

    def call(prompt: str, charge: _Charge) -> Any:
        return tt.call_model(model, prompt, ctx.max_tokens, rate_limiter, charge, client=client,
                             settings=mrun.settings, service_tier=tier, fallback_client=fallback,
                             stats=mrun.stats, gate=gate)

    async def one(item: Tuple[int, Any, Any, Any], *, probing: bool = False) -> None:
        nonlocal unfinished
        if n_per > 1:
            i, group = item[0], item[1]
            cell = cells[i]
            prompt = build_group(cell.job, group, mrun.model)
        else:
            i, etype, name, desc = item
            cell = cells[i]
            prompt = build_prompt(cell.job, etype, name, desc)
        charge = _Charge(cell.usage, mrun.usage, ctx.run_usage)
        mrun.in_flight += 1
        try:
            text = await call(prompt, charge)
        except tt.FlexRefusedError as e:
            if probing:
                raise
            fatal.append(e)
            return
        except tt.JudgeRequestError as e:
            fatal.append(e)
            return
        finally:
            mrun.in_flight -= 1
            unfinished -= 1
            ctx.live_pending -= 1
        if text is tt.SKIPPED or text is tt.FAILED:
            return      # no outcome for these entities: the cell stays open and they are asked again
        if n_per > 1:
            try:
                if text is None and len(group) > 1:
                    await judge_split_group(ctx, mrun, cell, header, group, call, charge.charges)
                else:
                    await read_group_reply(ctx, mrun, cell, header, group, text, charge.charges, prompt, call)
            except tt.JudgeRequestError as e:
                fatal.append(e)
                return
        else:
            if charge.charges:
                append_answer(cell, model, header, name, text, charge.charges, prompt)
            record(mrun, cell, name, text)
        ctx.note_answer()
        if cell.finished and not cell.written:
            write_cell(cell, mrun, ctx.max_tokens)

    try:
        if probe and items:
            first, items = items[0], items[1:]
            try:
                await one(first, probing=True)
            except tt.FlexRefusedError as e:
                logger.warning(f"[{model}] probe: OpenAI refused Flex processing ({e}); judging {model} through "
                               f"the Batch API for this run.  Add {model!r}: {{'flex': False}} to "
                               f"MODEL_CAPABILITIES in assistant_axis/judge.py")
                return False
            if not fatal:
                mrun.transport_reason += "; the probe was answered on Flex"
                logger.info(f"[{model}] probe: Flex accepted; add {model!r}: {{'flex': True}} to "
                            f"MODEL_CAPABILITIES in assistant_axis/judge.py")
        await run_bounded(one, items, ctx.max_in_flight)
    finally:
        ctx.live_pending -= unfinished   # requests never sent (a refused probe, a cancelled run)
    if fatal:
        raise fatal[0]
    return True


def close_live_cells(mrun: ModelRun, max_tokens: int) -> None:
    """After the last live round: a cell not finished goes on record as partial (no usage.json, so
    a rerun judges it, taking its saved answers), and its unanswered entities as left for a rerun."""
    for cell in mrun.live_cells:
        if cell.written:
            continue
        mrun.left_for_rerun += len(cell.expected - set(cell.scores))
        if cell.scores:
            write_partial_cell(cell, mrun, max_tokens)
        else:
            mrun.cells_incomplete += 1


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class ModelDriftError(RuntimeError):
    """Batches in flight (or results collected) under a model or request
    settings other than this run's."""


class AxisMap:
    """``<batch work dir>/axis_map.json``: the axis index in a custom id
    (``a<index>|<entity>``) -> the axis, cumulative across runs (an index is
    never reused), plus the settings the batches were requested with."""
    VERSION = 1

    def __init__(self, path: Path):
        self.path = Path(path)
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        if data and data.get("version") != self.VERSION:
            raise RuntimeError(f"{self.path}: unknown version {data.get('version')!r}")
        self.settings: Optional[Dict[str, Any]] = data.get("settings")
        self.axes: Dict[int, Dict[str, str]] = {int(k): v for k, v in (data.get("axes") or {}).items()}

    def index_for(self, job: AxisJob) -> int:
        entry = job.map_entry()
        for idx, e in self.axes.items():
            if e == entry:
                return idx
        idx = max(self.axes, default=-1) + 1
        self.axes[idx] = entry
        return idx

    def save(self) -> None:
        atomic_write_text(json.dumps({"version": self.VERSION, "settings": self.settings,
                                      "axes": {str(k): v for k, v in sorted(self.axes.items())}},
                                     indent=2, sort_keys=True) + "\n", self.path, logger_obj=logger)


def custom_id(idx: int, name: str) -> str:
    cid = f"a{idx}|{name}"
    if not CUSTOM_ID_RE.fullmatch(cid):
        raise ValueError(f"custom id {cid!r} is not 1-64 letters, digits, '_', '-' or '|'")
    return cid


def split_custom_id(cid: str) -> Tuple[int, str]:
    head, name = cid.split("|", 1)
    if not head.startswith("a"):
        raise ValueError(f"custom id {cid!r} does not start with a<axis index>|")
    return int(head[1:]), name


def batch_work_dir(ctx: RunContext, model: str) -> Path:
    return ctx.batch_root / cell_name(model)


def inflight_batches(work: Path) -> int:
    """Batches recorded in ``work``'s state file and not collected."""
    state = work / "batch_state.json"
    if not state.exists():
        return 0
    try:
        chunks = json.loads(state.read_text()).get("chunks", [])
    except (OSError, ValueError):
        return 0
    return sum(1 for r in chunks if r.get("batch_id") and not r.get("collected"))


def batch_settings(ctx: RunContext, mrun: ModelRun) -> Dict[str, Any]:
    out = {"model": mrun.model, "max_tokens": ctx.max_tokens, "request_settings": mrun.settings,
           "rubric_version": RUBRIC.version, "instructions_dir": str(Path(ctx.instructions_dir).resolve()),
           "name_format": NAME_FORMAT}
    if ctx.entities_per_call > 1:
        out.update(entities_per_call=ctx.entities_per_call, shuffle_seed=ctx.shuffle_seed,
                   batch_design=ctx.batch_design)
    return out


def axis_groups(job: AxisJob, ctx: RunContext) -> List[List[Tuple[str, str, str]]]:
    """The batched-judging groups of an axis's full entity list, from the seed
    and design (the same in every run with the same settings).  Under --stack, the
    groups of the items themselves, each sent once per slot, a group's slots in turn."""
    if STACK:
        real = [(e, stack_real(n), t) for e, n, t in job.entities if n.endswith(f"{SLOT_MARK}0")]
        return [[(e, f"{n}{SLOT_MARK}{j}", t) for e, n, t in g]
                for g in item_groups(real, ctx) for j in range(stack_slots(job))]
    return item_groups(list(job.entities), ctx)


def item_groups(full: List[Tuple[str, str, Any]], ctx: RunContext) -> List[List[Tuple[str, str, Any]]]:
    if RUBRIC.text_source == "responses":
        # a call's answers share their question: shuffle within each question, then cut into groups
        by_q: Dict[int, List[Tuple[str, str, Any]]] = {}
        for item in full:
            by_q.setdefault(response_question(item[1]), []).append(item)
        groups: List[List[Tuple[str, str, Any]]] = []
        for q in sorted(by_q):
            items = by_q[q]
            random.Random(ctx.shuffle_seed * 1_000_003 + q + 1).shuffle(items)
            groups += grid_batches(items, ctx.entities_per_call, "shuffle")
        return groups
    random.Random(ctx.shuffle_seed * 1_000_003 + 0).shuffle(full)
    return grid_batches(full, ctx.entities_per_call, ctx.batch_design)


async def run_batch(ctx: RunContext, mrun: ModelRun, jobs: List[AxisJob]) -> None:
    """Judge every entity of the not-done axes of ``jobs`` through the
    provider's batch API: one submission for all axes, results collected into
    per-axis cells, each written as soon as its axis is complete."""
    model = mrun.model
    provider = provider_for_model(model)
    work = batch_work_dir(ctx, model)
    collected_dir = work / "collected"
    if not (work / "batch_state.json").exists() and all(cell_done(job, model) for job in jobs):
        logger.info(f"[{model}] batch: nothing to submit")
        return
    client = standard_client(model)
    if provider == "anthropic":
        # sibling cells submit batches of exactly this size, so a lost submission is never adopted
        runner = AnthropicBatchRunner(client, work / "chunks", work / "batch_state.json",
                                      poll_seconds=ctx.batch_poll_seconds, adopt_lost_batches=False, logger=logger)
    elif provider == "google":
        runner = GeminiBatchRunner(client, work / "chunks", work / "batch_state.json", model=model,
                                   poll_seconds=ctx.batch_poll_seconds, logger=logger)
    else:
        runner = OpenAIBatchRunner(client, work / "chunks", work / "batch_state.json",
                                   poll_seconds=ctx.batch_poll_seconds,
                                   metadata={"source": "results_analysis/judge_tier_cost_eval.py",
                                             "judge_model": model},
                                   logger=logger)
    amap = AxisMap(work / "axis_map.json")
    settings_now = batch_settings(ctx, mrun)
    batch_key = f"{model}{BATCH_SUFFIX}"

    inflight = await runner.resume()
    # Model drift (as step 3): a batch in flight requested with another model
    # would be collected and charged under this one's name.
    other = {m: names for m, names in runner.inflight_models().items() if m != model}
    if other:
        raise ModelDriftError(
            f"{work / 'batch_state.json'} records batches in flight requested with "
            + "; ".join(f"{m} ({len(n)} chunk(s))" for m, n in sorted(other.items()))
            + f", not {model}: delete the state file to abandon them")
    collected_files = sorted(collected_dir.glob("a*.json")) if collected_dir.exists() else []
    if (inflight or collected_files) and amap.settings is not None and amap.settings != settings_now:
        raise ModelDriftError(
            f"{work}: batches in flight or results collected under other settings ({amap.settings}) than "
            f"this run's ({settings_now}): rerun with the recorded settings to finish them, or delete {work} "
            f"to abandon them")

    def collected_path(idx: int) -> Path:
        return collected_dir / f"a{idx}.json"

    cells: Dict[int, Cell] = {}
    dead: Set[int] = set()   # axes already done: results for them are dropped
    header = checkpoint_header(mrun, ctx.max_tokens)

    def new_cell(idx: int, job: AxisJob) -> Cell:
        cell = Cell(idx, job, expected={name for _, name, _ in job.entities})
        p = collected_path(idx)
        if p.exists():
            data = json.loads(p.read_text())
            cell.scores = {k: (None if v is None else int(v)) for k, v in data.get("scores", {}).items()}
            cell.truncated = set(data.get("truncated") or [])
            cell.usage = MultiModelUsage.from_dict(data.get("usage") or {})
        # answers a live or Flex run saved are not asked again (they are not in collected/, and a
        # result for the same entity from a batch already counts)
        resume_cell(cell, mrun, header)
        return cell

    current: List[int] = []
    for job in jobs:
        idx = amap.index_for(job)
        if cell_done(job, model):
            dead.add(idx)
            collected_path(idx).unlink(missing_ok=True)
            continue
        cells[idx] = new_cell(idx, job)
        current.append(idx)

    def cell_for(idx: int) -> Optional[Cell]:
        if idx in cells:
            return cells[idx]
        if idx in dead or idx not in amap.axes:
            return None
        job = AxisJob.from_map_entry(amap.axes[idx])
        if cell_done(job, model):
            dead.add(idx)
            return None
        logger.info(f"[{model}] collecting results for {job.axis_dir}, an axis of an earlier run")
        cells[idx] = new_cell(idx, load_axis(job, ctx.instructions_dir, ctx.slots, ctx.limit))
        return cells[idx]

    touched: Set[int] = set()

    groups_path = work / "groups.json"
    groups: Dict[str, List[str]] = json.loads(groups_path.read_text()) if groups_path.exists() else {}
    n_per = ctx.entities_per_call

    def group_members(cid: str) -> Optional[List[str]]:
        """The member names of a group custom id (``a<idx>|g<k>``), from groups.json."""
        return groups.get(cid)

    to_split: List[Tuple[Cell, List[str]]] = []     # groups that came back with nothing
    to_reask: List[Tuple[Cell, List[str], str]] = []    # groups whose reply left members unread

    def record_group(cell: Cell, cid: str, reply: Any) -> None:
        members = group_members(cid)
        if members is None:
            logger.warning(f"[{model}] group {cid!r} is not in {groups_path}; its reply is dropped")
            return
        if reply is None and len(members) > 1:
            to_split.append((cell, members))
            return
        parsed = ({} if isinstance(reply, tt.Truncated) or not reply else parse_group(cell.job, members, mrun.model, reply))
        sw = group_swapped(cell.job, members, mrun.model) if DIRECTIONS != "own" else None
        retry = bool(reply) and not isinstance(reply, tt.Truncated) and any(parsed.get(k) is None for k in range(1, len(members) + 1))
        for k, gname in enumerate(members, 1):
            if not retry or parsed.get(k) is not None:
                record(mrun, cell, gname, reply, score=parsed.get(k) if parsed else None, swapped=sw)
        if retry:
            to_reask.append((cell, members, reply))

    def on_result(cid: str, body: Optional[dict], error: Optional[str]) -> None:
        if body is None:
            return   # unanswered: comes back from runner.run and is sent in real time
        idx, name = split_custom_id(cid)
        refused = False
        if provider == "anthropic":
            text, finish, _n_in, _n_out, refused = tt.anthropic_reply(body)
            tokens = anthropic_usage(body)
            if refused:
                details = body.get("stop_details") if isinstance(body, dict) else None
                mrun.stats.note_refusal(model, (details or {}).get("category")
                                        if isinstance(details, dict) else None)
                text = None
        elif provider == "google":
            text, finish, n_in, n_out = tt.gemini_reply(body)
            tokens = gemini_usage(body)
            finish = tt.CUT_AT_CAP if finish == "MAX_TOKENS" else finish
            if text is None:        # a blocked prompt, as judge.call_gemini_judge counts it live
                mrun.stats.content_flagged += 1
        else:
            text, finish, _n_in, _n_out = tt.openai_reply(body)
            tokens = openai_usage(body)
        reply: Any = tt.Truncated(text) if finish == tt.CUT_AT_CAP and not refused else text
        cell = cell_for(idx)
        if cell is None:
            mrun.usage.charge_usage(batch_key, tokens)
            ctx.run_usage.charge_usage(batch_key, tokens)
            logger.warning(f"[{model}] result {cid!r} is for an axis already done or unknown; dropped "
                           f"(its tokens are in this run's total only)")
            return
        _Charge(cell.usage, mrun.usage, ctx.run_usage).charge_usage(batch_key, tokens)
        is_group = name.startswith("g") and name[1:].isdigit() and n_per > 1
        append_batch_reply(cell, model, cid, (group_members(cid) or []) if is_group else [name], text, finish, refused)
        if is_group:
            record_group(cell, cid, reply)
        else:
            record(mrun, cell, name, reply)
        touched.add(idx)

    def flush() -> None:
        """Called after each batch's results, before the runner marks it
        collected: complete cells are written, the others saved under
        collected/ so a rerun does not ask for them again."""
        for idx in sorted(touched):
            cell = cells[idx]
            if cell.written:
                continue
            if cell.finished:
                write_cell(cell, mrun, ctx.max_tokens)
                collected_path(idx).unlink(missing_ok=True)
            else:
                atomic_write_text(json.dumps({"axis": amap.axes.get(idx), "settings": settings_now,
                                              "scores": cell.scores, "truncated": sorted(cell.truncated),
                                              "usage": cell.usage.as_dict()},
                                             indent=2, sort_keys=True) + "\n",
                                  collected_path(idx), logger_obj=logger)
        touched.clear()

    # A cell already complete from collected results (a run stopped between
    # saving them and writing the cell) is written now.
    for idx in current:
        if cells[idx].finished:
            touched.add(idx)
    flush()

    # 1. Batches still in flight are collected before anything is submitted.
    unanswered: Dict[str, Dict[str, Any]] = {}
    if inflight:
        logger.info(f"[{model}] collecting {len(inflight)} requests in batches already submitted, "
                    f"before submitting anything")
        unanswered.update(await runner.run([], on_result, flush))

    # 2. One submission for everything still missing.
    def pending():
        """``(cid, idx, prompt)`` for every request still to submit: one per entity, or with
        batched judging one per group with any member unscored."""
        for idx in current:
            cell = cells[idx]
            if cell.written:
                continue
            if n_per > 1:
                for k, group in enumerate(axis_groups(cell.job, ctx)):
                    cid = custom_id(idx, f"g{k}")
                    if cid in unanswered or cid in inflight or all(name in cell.scores for _, name, _ in group):
                        continue
                    groups[cid] = [name for _, name, _ in group]
                    yield cid, idx, build_group(cell.job, group, mrun.model)
                continue
            for etype, name, desc in cell.job.entities:
                if name in cell.scores:
                    continue
                cid = custom_id(idx, name)
                if cid in unanswered or cid in inflight:
                    continue
                yield cid, idx, build_prompt(cell.job, etype, name, desc)

    n_pending = sum(1 for _ in pending())
    if n_pending:
        amap.settings = settings_now
        amap.save()   # before the submission, so a rerun can route what it collects
        if n_per > 1:
            atomic_write_text(json.dumps(groups, indent=0, sort_keys=True) + "\n", groups_path, logger_obj=logger)
        chunks = runner.write_chunks(
            ((cid, tt.request_for(model, prompt, ctx.max_tokens, mrun.settings))
             for cid, idx, prompt in pending()),
            # the runner's own default unless --batch_chunk_requests says otherwise
            **({"max_requests": ctx.batch_chunk_requests} if ctx.batch_chunk_requests != DEFAULT_BATCH_CHUNK_REQUESTS
               else {}))
        logger.info(f"[{model}] batch: {n_pending} requests in {len(chunks)} chunk(s) for "
                    f"{sum(1 for i in current if not cells[i].written)} axes")
        unanswered.update(await runner.run(chunks, on_result, flush))
    elif not inflight:
        logger.info(f"[{model}] batch: nothing to submit")

    # 3. What no batch answered goes in real time, on the standard tier.
    if unanswered and len(unanswered) <= ctx.batch_fallback_max:
        logger.info(f"[{model}] judging {len(unanswered)} requests no batch answered in real time")
        rate_limiter = RateLimiter(ctx.requests_per_second)
        remainder = sorted(unanswered.items())
        mrun.to_judge += len(remainder)
        ctx.add_pending(len(remainder))
        unfinished = len(remainder)

        async def one(entry: Tuple[str, Dict[str, Any]]) -> None:
            nonlocal unfinished
            cid, request = entry
            idx, name = split_custom_id(cid)
            is_group = n_per > 1 and name.startswith("g") and name[1:].isdigit()
            try:
                cell = cell_for(idx)
                if cell is None or cell.written or (not is_group and name in cell.scores):
                    return
                prompt = tt.prompt_of_request(request)
                charge = _Charge(cell.usage, mrun.usage, ctx.run_usage)
                mrun.in_flight += 1
                try:
                    text = await tt.call_model(model, prompt, ctx.max_tokens, rate_limiter, charge,
                                               client=client, settings=mrun.settings, stats=mrun.stats,
                                               gate=ctx.over_budget)
                finally:
                    mrun.in_flight -= 1
            finally:
                unfinished -= 1
                ctx.live_pending -= 1
            if text is tt.SKIPPED or text is tt.FAILED:
                return      # no outcome: left out of the cell, and a rerun submits it again
            if is_group:
                members = group_members(cid) or []
                if text is not None or len(members) < 2:
                    for k, gname in enumerate(members, 1):
                        if charge.charges:
                            append_answer(cell, model, header, gname, text, charge.charges if k == 1 else [], prompt,
                                          batch=members)
                record_group(cell, cid, text)
            else:
                if charge.charges:
                    append_answer(cell, model, header, name, text, charge.charges, prompt)
                record(mrun, cell, name, text)
            ctx.note_answer()
            touched.add(idx)

        try:
            await run_bounded(one, remainder, ctx.max_in_flight)
        finally:
            ctx.live_pending -= unfinished
        flush()
    elif unanswered:
        logger.error(f"[{model}] {len(unanswered)} requests were not answered by their batches, more than "
                     f"--batch_fallback_max ({ctx.batch_fallback_max}) to send in real time; they stay "
                     f"unscored, and a rerun submits them again")

    # 4. A group that came back with nothing: its halves in real time, on the standard tier; and a
    # group whose reply left members unread: the same prompt again, the same way.
    if to_split or to_reask:
        logger.warning(f"[{model}] {len(to_split)} groups came back with nothing, asking their halves in real time; "
                       f"{len(to_reask)} replies left members unread, asking them again in real time")
        rate_limiter = RateLimiter(ctx.requests_per_second)

        def call(prompt: str, charge: _Charge) -> Any:
            return tt.call_model(model, prompt, ctx.max_tokens, rate_limiter, charge, client=client,
                                 settings=mrun.settings, stats=mrun.stats, gate=ctx.over_budget)

        for cell, members in to_split:
            by_name = {e[1]: e for e in cell.job.entities}
            await judge_split_group(ctx, mrun, cell, header, [by_name[n] for n in members if n in by_name], call)
            touched.add(cell.idx)
        for cell, members, reply in to_reask:
            by_name = {e[1]: e for e in cell.job.entities}
            group = [by_name[n] for n in members]
            prompt = build_group(cell.job, group, mrun.model)
            await read_group_reply(ctx, mrun, cell, header, group, reply, [], prompt, call)
            touched.add(cell.idx)
        flush()
    for idx in current:
        cell = cells[idx]
        if not cell.written:
            mrun.cells_incomplete += 1
            mrun.left_for_rerun += len(cell.expected - set(cell.scores))
            if cell.scores:
                touched.add(idx)
    flush()   # incomplete cells' results stay under collected/ for the rerun


# ---------------------------------------------------------------------------
# Per model
# ---------------------------------------------------------------------------

def direction_summary(cell: Cell) -> Dict[str, Any]:
    """A cell's directions for its provenance (--directions): the mode, the pass, how many of its scored names
    each way went, and, for response items, the spread over entities of the share judged swapped."""
    scored = [n for n, v in cell.scores.items() if v is not None]
    ways = [cell.swapped.get(n) for n in scored]
    out: Dict[str, Any] = {"directions": DIRECTIONS, "direction_pass": DIRECTION_PASS,
                           "direction_counts": {"own": sum(w is False for w in ways), "swapped": sum(w is True for w in ways),
                                                "unknown": sum(w is None for w in ways)}}
    if RUBRIC.text_source == "responses":
        per: Dict[str, List[bool]] = {}
        for n in scored:
            if n in cell.swapped:
                per.setdefault(response_item_entity(n), []).append(cell.swapped[n])
        shares = sorted(sum(v) / len(v) for v in per.values() if len(v) >= 10)
        if shares:
            out["direction_entity_share_swapped"] = {"n_entities": len(shares), "min": shares[0],
                                                     "median": shares[len(shares) // 2], "max": shares[-1]}
    return out


def refuse_done_cells_judged_otherwise(jobs: Sequence[AxisJob], model: str) -> None:
    """A finished cell is skipped, so one judged under other --directions than this run's would pass for this
    run's: stop instead, naming it (Roger, 2026-11-04: an unnoticed direction asymmetry could invalidate results).
    Likewise a response cell cut under another --response_design (the default became golden on 2026-11-04, so a
    rerun of an older label would otherwise report its by-question cells as this run's)."""
    for job in jobs:
        if not cell_done(job, model):
            continue
        sp = cell_scores_path(cell_dir(job, model))
        try:
            prov = json.loads(sp.read_text()).get("_provenance", {}) if sp.exists() else {}
        except (OSError, ValueError):
            prov = {}
        found = (prov.get("directions", "own"), int(prov.get("direction_pass", 0)))
        if found != (DIRECTIONS, DIRECTION_PASS if DIRECTIONS == "descriptions" else 0):
            raise SystemExit(f"{cell_dir(job, model)} was judged under --directions {found[0]} (pass {found[1]}), not this "
                             f"run's {DIRECTIONS} (pass {DIRECTION_PASS}): use another --cell_label")
        design = (prov.get("response_design") or "question") if prov.get("response_part") is not None else "question"
        if design != response_design_in_effect():
            raise SystemExit(f"{cell_dir(job, model)} was cut under --response_design {design}, not this run's "
                             f"{response_design_in_effect()}: pass --response_design {design} to resume or reread it, "
                             f"or use another --cell_label")


async def run_model(ctx: RunContext, mrun: ModelRun, jobs: List[AxisJob]) -> None:
    model = mrun.model
    refuse_done_cells_judged_otherwise(jobs, model)
    todo = [job for job in jobs if not cell_done(job, model)]
    for job in jobs:
        if cell_done(job, model):
            logger.info(f"[{model}] {job.axis_dir}: SKIP (already done: {cell_dir(job, model) / 'usage.json'})")
    logger.info(f"[{model}] transport {mrun.choice.describe()}; request settings {json.dumps(mrun.settings)}; "
                f"{len(todo)}/{len(jobs)} axes to judge")
    if mrun.choice.transport == "batch":
        await run_batch(ctx, mrun, jobs)
        return
    work = batch_work_dir(ctx, model)
    if inflight_batches(work):
        raise ModelDriftError(f"{work / 'batch_state.json'} records {inflight_batches(work)} batch(es) of "
                              f"{model} still in flight: rerun with --transport batch to collect them first "
                              f"(or delete {work} to abandon them)")
    if (work / "collected").exists() and any((work / "collected").glob("a*.json")):
        logger.warning(f"[{model}] {work / 'collected'} holds results of an earlier batch run; "
                       f"--transport {mrun.choice.transport} judges those axes afresh (the files are kept)")
    if not todo:
        return
    tier = "flex" if mrun.choice.transport == "flex" else None
    for round_ in range(REASK_ROUNDS + 1):
        failed_before, resumed_before = mrun.stats.errors, mrun.resumed
        accepted = await run_live(ctx, mrun, todo, tier=tier, probe=mrun.choice.probe and round_ == 0)
        if not accepted:
            mrun.transport = "batch"
            mrun.transport_reason = (f"{mrun.choice.reason}; OpenAI refused Flex on the probe (HTTP 400), "
                                     f"so the run switched to the Batch API")
            await run_batch(ctx, mrun, jobs)
            return
        if round_:
            mrun.resumed = resumed_before    # the same saved answers, not more of them
        todo = [job for job in todo if not cell_done(job, model)]
        if mrun.stats.errors == failed_before or not todo or round_ == REASK_ROUNDS or ctx.budget_hit:
            break
        logger.warning(f"[{model}] {mrun.stats.errors - failed_before} calls got no answer; asking them again in "
                       f"{REASK_SLEEP_S:.0f}s (round {round_ + 1} of {REASK_ROUNDS})")
        await asyncio.sleep(REASK_SLEEP_S)
    close_live_cells(mrun, ctx.max_tokens)


def summarise(mrun: ModelRun) -> str:
    s = mrun.stats
    parts = [f"{mrun.model}: transport={mrun.transport}", f"parsed {mrun.parsed}/{mrun.attempted}",
             f"truncated {s.truncated}", f"cells written {mrun.cells_written}"]
    if mrun.resumed:
        parts.append(f"resumed {mrun.resumed}")
    if mrun.parse_retries:
        parts.append(f"parse retries {mrun.parse_retries}")
    if mrun.cells_partial:
        parts.append(f"partial {mrun.cells_partial}")
    if mrun.cells_incomplete:
        parts.append(f"incomplete {mrun.cells_incomplete}")
    if mrun.transport == "flex" or s.flex_fallbacks:
        parts.append(f"flex fallbacks {s.flex_fallbacks}")
    for label, n in (("refused", s.refused), ("content-flagged", s.content_flagged), ("failed calls", s.errors),
                     ("not issued", s.skipped), ("left for a rerun", mrun.left_for_rerun)):
        if n:
            parts.append(f"{label} {n}")
    parts.append(f"cost ${mrun.usage.total_cost_usd:.4f}")
    return "  ".join(parts)


async def report_progress(ctx: RunContext, runs: List[ModelRun], every: float) -> None:
    """A progress line per model with live or Flex work, every ``every`` seconds."""
    while True:
        await asyncio.sleep(every)
        for m in runs:
            if not m.to_judge:
                continue
            logger.info(f"[{m.model}] progress: {m.attempted}/{m.to_judge} judged this run"
                        + (f" ({m.resumed} more from the answer logs)" if m.resumed else "")
                        + f"; in flight {m.in_flight}; Flex fallbacks {m.stats.flex_fallbacks}; "
                          f"cells written {m.cells_written}; this run's cost ${m.usage.total_cost_usd:.2f}")


async def watch_for_stall(ctx: RunContext, work: "asyncio.Future[Any]") -> None:
    """Cancel ``work`` when no answer has arrived for ``ctx.stall_s`` while live
    or Flex requests remain (batch waits do not count).  Every answer is in an
    answer log by then, so a restart loses nothing."""
    while not work.done():
        await asyncio.sleep(min(60.0, max(0.05, ctx.stall_s / 10)))
        if ctx.live_pending > 0 and time.monotonic() - ctx.last_progress > ctx.stall_s:
            ctx.stalled = True
            logger.error(f"no answer for {ctx.stall_s / 60:.1f} min with {ctx.live_pending} requests still to "
                         f"judge: stopping (every answer so far is saved in the cells' {ANSWERS_PARTIAL})")
            work.cancel()
            return


#: --stall_restarts of this run; read by main() after a stall.
STALL_RESTARTS = DEFAULT_STALL_RESTARTS


def restart_after_stall() -> None:
    """Start this script again with the same arguments (os.execv: same process,
    same log), at most STALL_RESTARTS times in a row (counted in RESTART_ENV);
    returns only when no restart is left."""
    used = int(os.environ.get(RESTART_ENV) or 0)
    if used >= STALL_RESTARTS:
        logger.error(f"stalled after {used} restart(s) (--stall_restarts {STALL_RESTARTS}): giving up; a rerun "
                     f"resumes from the saved answers")
        return
    os.environ[RESTART_ENV] = str(used + 1)
    logger.warning(f"restarting after the stall ({used + 1} of {STALL_RESTARTS}); the saved answers are resumed")
    for h in logging.getLogger().handlers:
        h.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, *sys.argv])


# ---------------------------------------------------------------------------
# Repair: re-ask the replies cut at the cap (--repair_truncated, 2026-11-01)
# ---------------------------------------------------------------------------

SOURCE_PROVENANCE = "provenance"


@dataclass
class RepairCell:
    job: AxisJob
    dir: Path
    data: Dict[str, Any]                  # scores_descriptions.json as read
    source: str = ""
    original: Set[str] = field(default_factory=set)   # cut in the cell's own run
    todo: List[str] = field(default_factory=list)     # to re-ask now
    est_in: Dict[str, int] = field(default_factory=dict)
    slots: List[int] = field(default_factory=list)     # as in the cell's correlations.json
    answers: Dict[str, Any] = field(default_factory=dict)   # entity -> int, None or Truncated
    usage: MultiModelUsage = field(default_factory=MultiModelUsage)

    @property
    def prov(self) -> Dict[str, Any]:
        return self.data.get("_provenance") or {}


@dataclass
class RepairStats:
    cells_with_todo: int = 0
    cells_repaired: int = 0
    reasked: int = 0
    now_scored: int = 0
    unscored: int = 0          # answered without a score (unparseable or refused)
    still_truncated: int = 0
    failed: int = 0            # no answer (call failed): left as they were, a rerun tries again
    est_cost: float = 0.0


def entities_to_reask(original: Set[str], repairs: Sequence[Dict[str, Any]], cell_cap: int,
                      cap: int) -> Tuple[List[str], List[str]]:
    """``(to re-ask, already re-asked at >= cap)`` from the entities cut in the
    cell's own run and its earlier repair records: an entity is re-asked when
    it is still cut and ``cap`` is above the cap it was last asked at."""
    current = set(original)
    last_cap = {n: cell_cap for n in original}
    for rec in repairs:
        for n in rec.get("reasked") or []:
            current.discard(n)
            last_cap[n] = int(rec.get("max_tokens") or 0)
        current.update(rec.get("still_truncated") or [])
    todo = sorted(n for n in current if cap > last_cap.get(n, cell_cap))
    return todo, sorted(current - set(todo))


def _same_correlations(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """The same entities, scores and n per slot and metric, and rho and p equal
    to 1e-12 (or both NaN)."""
    import math
    try:
        if set(a) != set(b):
            return False
        for mode in a:
            if set(a[mode]) != set(b[mode]):
                return False
            for slot in a[mode]:
                for metric in ("raw", "whitened"):
                    x, y = a[mode][slot][metric], b[mode][slot][metric]
                    if (x["names"], x["scores"], x["n"]) != (y["names"], y["scores"], y["n"]):
                        return False
                    for k in ("rho", "p"):
                        u, v = float(x[k]), float(y[k])
                        if not ((math.isnan(u) and math.isnan(v)) or abs(u - v) <= 1e-12):
                            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def merge_usage_dict(existing: Dict[str, Any], add: MultiModelUsage) -> Dict[str, Any]:
    """``usage.json`` with ``add`` merged in: entries ``add`` does not touch are
    kept as they were, touched ones summed (cost from the summed tokens), and
    the totals recomputed."""
    merged = MultiModelUsage.from_dict(existing)
    merged.merge_from(add)
    out = merged.as_dict()
    for key, entry in (existing.get("per_model") or {}).items():
        if key not in add.per_model:
            out["per_model"][key] = entry
    return out


def patch_cell(rc: RepairCell, mrun: ModelRun, max_tokens: int) -> Tuple[List[str], List[str]]:
    """Write the re-asked entities' new scores into the cell, recompute its
    correlations and merge the re-ask usage into usage.json (written last).
    Returns ``(re-asked, still cut)``."""
    job, d = rc.job, rc.dir
    result = dict(rc.data.get("result") or {})
    reasked = sorted(rc.answers)
    still = sorted(n for n, a in rc.answers.items() if isinstance(a, tt.Truncated))
    for name, answer in rc.answers.items():
        if isinstance(answer, int) and not isinstance(answer, bool):
            result[name] = answer
        else:
            result.pop(name, None)
    result = {k: result[k] for k in sorted(result)}
    prov = dict(rc.prov)
    prov["repairs"] = list(prov.get("repairs") or []) + [{
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "max_tokens": max_tokens,
        "reasked": reasked, "still_truncated": still, "source": rc.source}]
    new_data = dict(rc.data)
    new_data["result"] = result
    new_data["_provenance"] = prov
    correlations = compute_correlations(
        scores_by_mode={"descriptions": result},
        projections=job.projections,
        slots=rc.slots,
        excluded_set=job.axis_spec.pole_entity_ids(),
    )
    usage = merge_usage_dict(json.loads((d / "usage.json").read_text()), rc.usage)
    atomic_write_text(json.dumps(new_data, indent=2), d / "scores_descriptions.json", logger_obj=logger)
    atomic_write_text(json.dumps(correlations, indent=2), d / "correlations.json", logger_obj=logger)
    atomic_write_text(json.dumps(usage, indent=2, sort_keys=True), d / "usage.json", logger_obj=logger)
    logger.info(f"[{mrun.model}] {job.axis_dir}: repaired {len(reasked)} ({len(reasked) - len(still)} no longer "
                f"cut, {len(still)} cut again at {max_tokens}); cost ${rc.usage.total_cost_usd:.4f}")
    return reasked, still


def prepare_repair_cell(rc: RepairCell, mrun: ModelRun, ctx: RunContext, notes: List[str]) -> bool:
    """Load the cell's axis, check the cell can be patched as its original run
    wrote it (model, request settings, projections, correlations), and work out
    what to re-ask.  False (with a note) when it cannot."""
    job, prov, model = rc.job, rc.prov, mrun.model
    if prov.get("judge_model") not in (None, model):
        notes.append(f"{job.axis_dir}: judged by {prov.get('judge_model')}, not {model}; left alone")
        return False
    if prov.get("request_settings") != mrun.settings:
        notes.append(f"{job.axis_dir}: recorded request settings {json.dumps(prov.get('request_settings'))} "
                     f"differ from this run's {json.dumps(mrun.settings)} (pass the run's --request_setting "
                     f"and --cell_label); left alone")
        return False
    if int(prov.get("entities_per_call") or 1) > 1:
        notes.append(f"{job.axis_dir}: judged {prov['entities_per_call']} entities per call, and a repair re-asks one "
                     f"at a time, a different prompt; left alone")
        return False
    if prov.get("rubric", "v3") != RUBRIC.name:
        notes.append(f"{job.axis_dir}: judged under rubric {prov.get('rubric', 'v3')}, not this run's {RUBRIC.name} "
                     f"(pass the run's --rubric); left alone")
        return False
    recorded_names = prov.get("name_format", CHECKPOINT_DEFAULTS["name_format"])
    if recorded_names != NAME_FORMAT:
        notes.append(f"{job.axis_dir}: its prompts show names as {recorded_names}, this run's as {NAME_FORMAT} "
                     f"(pass the run's --name_format); left alone")
        return False
    cell_cap = int(prov.get("max_tokens") or 0)
    rc.todo, already = entities_to_reask(rc.original, prov.get("repairs") or [], cell_cap, ctx.max_tokens)
    if already:
        notes.append(f"{job.axis_dir}: {len(already)} cut entit{'y' if len(already) == 1 else 'ies'} already "
                     f"re-asked at max_tokens >= {ctx.max_tokens} ({', '.join(already[:5])})")
    if not rc.todo:
        return False
    if not job.entities:
        load_axis(job, ctx.instructions_dir)
    known = {name for _, name, _ in job.entities}
    gone = [n for n in rc.todo if n not in known]
    if gone:
        notes.append(f"{job.axis_dir}: {len(gone)} cut entities have no description under "
                     f"{ctx.instructions_dir} and are not re-asked: {gone[:5]}")
        rc.todo = [n for n in rc.todo if n in known]
    # The correlations are recomputed exactly as the original write did: the
    # same projections, slots and pole exclusion.  Recomputing them from the
    # unchanged scores must give the file on disk.
    try:
        on_disk = json.loads((rc.dir / "correlations.json").read_text())
        rc.slots = [int(s) for s in on_disk["descriptions"]]
    except (OSError, ValueError, KeyError) as e:
        notes.append(f"{job.axis_dir}: correlations.json unreadable ({e}); left alone")
        return False
    all_proj = load_projections(job.projections_file, "all")
    if any(s not in all_proj for s in rc.slots):
        notes.append(f"{job.axis_dir}: correlations.json has slots {rc.slots} that {job.projections_file} "
                     f"lacks; left alone")
        return False
    job.projections = {s: all_proj[s] for s in rc.slots}
    again = compute_correlations(scores_by_mode={"descriptions": rc.data.get("result") or {}},
                                 projections=job.projections, slots=rc.slots,
                                 excluded_set=job.axis_spec.pole_entity_ids())
    if not _same_correlations(json.loads(json.dumps(again)), on_disk):
        notes.append(f"{job.axis_dir}: the correlations recomputed from the cell's scores and "
                     f"{job.projections_file} are not those on disk; left alone")
        return False
    return bool(rc.todo)


async def repair_model(ctx: RunContext, mrun: ModelRun, jobs: List[AxisJob], *, dry_run: bool) -> RepairStats:
    """--repair_truncated for one model: find each finished cell's entities whose
    reply was cut at the cap, re-ask them in real time at ``ctx.max_tokens``
    and patch the cell (dry run: report them and the estimated cost)."""
    model = mrun.model
    stats = RepairStats()
    notes: List[str] = []
    cells: List[RepairCell] = []
    for job in jobs:
        d = cell_dir(job, model)
        if not cell_done(job, model):
            notes.append(f"{job.axis_dir}: no finished cell at {d}; skipped")
            continue
        data = json.loads((d / "scores_descriptions.json").read_text())
        rc = RepairCell(job, d, data)
        prov = rc.prov
        if "truncated" in prov:
            rc.source, rc.original = SOURCE_PROVENANCE, set(prov.get("truncated") or [])
            cells.append(rc)
        else:
            notes.append(f"{job.axis_dir}: no truncation record (no truncation list in its provenance, as in "
                         f"cells made before 2026-11-01); left alone")

    work_cells = [rc for rc in cells if prepare_repair_cell(rc, mrun, ctx, notes)]
    rate_in, rate_out = price_for_model(model)
    for rc in work_cells:
        stats.cells_with_todo += 1
        stats.est_cost += sum(rc.est_in.get(n, EST_IN_PER_ENTITY) * rate_in + ctx.max_tokens * rate_out
                              for n in rc.todo) / 1e6
    n_todo = sum(len(rc.todo) for rc in work_cells)

    print(f"\n{model}{CELL_LABEL}: {len(work_cells)} cell(s) with replies cut at the cap to re-ask, {n_todo} "
          f"entities, at max_tokens {ctx.max_tokens}; est. cost ${stats.est_cost:.4f} (real time at "
          f"${rate_in:.2f}/${rate_out:.2f} per 1M; input as measured where known, output at the cap)")
    print(f"  {len(cells)} finished cell(s) with a truncation record, "
          f"{sum(len(rc.original) for rc in cells)} replies cut in their own runs")
    for rc in work_cells:
        print(f"  {rc.job.axis_dir:55s} {len(rc.todo):4d}  (source: {rc.source}; cell cap "
              f"{rc.prov.get('max_tokens')})  {', '.join(rc.todo[:8])}{' ...' if len(rc.todo) > 8 else ''}")
    for note in notes:
        print(f"  note: {note}")
    if dry_run or not work_cells:
        return stats

    client = standard_client(model)
    rate_limiter = RateLimiter(ctx.requests_per_second)
    fatal: List[BaseException] = []

    def gate() -> bool:
        return bool(fatal) or ctx.over_budget()

    async def one(rc: RepairCell, name: str) -> None:
        etype, desc = next((t, ds) for t, n, ds in rc.job.entities if n == name)
        prompt = build_prompt(rc.job, etype, name, desc)
        this_call = MultiModelUsage()
        try:
            reply = await tt.call_model(model, prompt, ctx.max_tokens, rate_limiter,
                                        _Charge(rc.usage, mrun.usage, ctx.run_usage, this_call), client=client,
                                        settings=mrun.settings, stats=mrun.stats, gate=gate)
        except tt.JudgeRequestError as e:
            fatal.append(e)
            return
        if reply is tt.SKIPPED:
            return
        if reply is tt.FAILED or this_call.n_calls == 0:      # no answer: left as it was
            stats.failed += 1
            return
        mrun.attempted += 1
        if isinstance(reply, tt.Truncated):
            mrun.stats.truncated += 1
            rc.answers[name] = reply
            logger.warning(f"[{model}] {rc.job.axis_dir}: {name!r} cut again at max_tokens {ctx.max_tokens}")
            return
        score = parse_score(reply) if reply else None
        rc.answers[name] = score
        if score is not None:
            mrun.parsed += 1

    await asyncio.gather(*(one(rc, name) for rc in work_cells for name in rc.todo))
    for rc in work_cells:
        if not rc.answers:
            continue
        reasked, still = patch_cell(rc, mrun, ctx.max_tokens)
        stats.cells_repaired += 1
        stats.reasked += len(reasked)
        stats.still_truncated += len(still)
        stats.now_scored += sum(1 for a in rc.answers.values() if isinstance(a, int) and not isinstance(a, bool))
    stats.unscored = stats.reasked - stats.now_scored - stats.still_truncated
    if fatal:
        raise fatal[0]
    return stats


async def repair_main(args: argparse.Namespace, jobs: List[AxisJob], runs: List[ModelRun]) -> int:
    ctx = RunContext(instructions_dir=Path(args.instructions_dir),
                     batch_root=Path(args.experiments_dir) / BATCH_DIRNAME,
                     max_tokens=args.max_tokens, requests_per_second=args.requests_per_second,
                     budget_usd=args.budget_usd)
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    if not args.dry_run:
        missing_keys = sorted({key_name(m.model) for m in runs if not os.getenv(key_name(m.model))})
        if missing_keys:
            logger.error(f"{', '.join(missing_keys)} not set (environment or .env): nothing sent")
            return 1
    failed = False
    results: List[Tuple[ModelRun, Optional[RepairStats]]] = []
    for mrun in runs:
        try:
            st = await repair_model(ctx, mrun, jobs, dry_run=args.dry_run)
        except Exception as e:  # noqa: BLE001 -- reported per model
            logger.error(f"[{mrun.model}] repair failed: {type(e).__name__}: {e}")
            mrun.error, st, failed = e, None, True
        results.append((mrun, st))
    if args.dry_run:
        total = sum(st.est_cost for _, st in results if st is not None)
        print(f"\nTotal estimated cost: ${total:.4f}")
        print("Dry run only -- no judge calls made, nothing written.")
        return 1 if failed else 0
    print("\nRepair summary:")
    for mrun, st in results:
        if st is None:
            print(f"  {mrun.model}{CELL_LABEL}: FAILED: {type(mrun.error).__name__}: {mrun.error}")
            continue
        print(f"  {mrun.model}{CELL_LABEL}: cells repaired {st.cells_repaired}  entities re-asked {st.reasked}  "
              f"now scored {st.now_scored}  still truncated {st.still_truncated}"
              + (f"  answered without a score {st.unscored}" if st.unscored else "")
              + (f"  failed calls {st.failed} (left as they were; rerun to retry)" if st.failed else "")
              + (f"  not issued {mrun.stats.skipped}" if mrun.stats.skipped else "")
              + f"  cost ${mrun.usage.total_cost_usd:.4f}")
        if st.reasked or st.failed:
            warn_if_low_parse_rate(label=f"judge_tier_cost_eval:repair:{mrun.model}{CELL_LABEL}",
                                   n_ok=st.now_scored, n_total=st.reasked + st.failed, logger_obj=logger)
    logger.info(ctx.run_usage.log_line())
    if failed:
        return 1
    return 2 if ctx.budget_hit else 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axes_file", type=str, default=None,
                        help="TSV (or whitespace-separated) file of 'axis_dir pos neg pair_type' rows, one "
                             "axis per line, '#' comments allowed (scripts/tier_eval_axes.tsv lists all 61).  "
                             "Each axis reads <experiments_dir>/<axis_dir>/gpt/projections.json and writes "
                             "<experiments_dir>/<axis_dir>_tier_eval/<model>/.  Excludes --pair, --pair_type, "
                             "--projections_file and --output_dir.")
    parser.add_argument("--experiments_dir", type=str, default=DEFAULT_EXPERIMENTS_DIR,
                        help="With --axes_file, where the axes live; in both modes, where batch work and "
                             f"state go ({BATCH_DIRNAME}/<model>/).  Default: {DEFAULT_EXPERIMENTS_DIR}")
    parser.add_argument("--pair", nargs=2, metavar=("POS", "NEG"))
    parser.add_argument("--pair_type", choices=["roles", "traits"])
    parser.add_argument("--projections_file", type=str,
                        help="Existing projections.json to correlate new scores against "
                             "(e.g. roger/axis_judge_experiments/angel_vs_demon/gpt/projections.json)")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--instructions_dir", type=str, default="data")
    parser.add_argument("--judge_model", action="append", required=True,
                        help="Repeatable.  gpt-/o1-/o3-/o4- -> OpenAI, claude- -> Anthropic, gemini- -> Google.")
    parser.add_argument("--transport", choices=list(tt.TRANSPORT_CHOICES), default="auto",
                        help="auto (default): Anthropic -> batch; OpenAI -> flex when MODEL_CAPABILITIES says "
                             "OpenAI takes Flex for the model, batch when it says OpenAI refuses it, flex with "
                             "a probe when untested.  flex / batch / realtime force one for every model.")
    parser.add_argument("--request_setting", action="append", default=[], metavar="[MODEL:]KEY=JSON",
                        help="Repeatable.  A request field set on top of assistant_axis/judge.py's "
                             "JUDGE_REQUEST_SETTINGS, for every model or only those matching MODEL "
                             "(e.g. 'gpt-6:reasoning_effort=\"low\"'); a null value removes a field.")
    parser.add_argument("--slots", type=str, default="all",
                        help="Comma-separated slot indices, or 'all' (default; matches every "
                             "slot present in the cached projections.json).")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"The judge's reply cap (default {DEFAULT_MAX_TOKENS}; 200 before 2026-11-01).  "
                             "A reply cut at the cap is scored as missing, never parsed, and listed under "
                             "'truncated' in the cell's provenance; --repair_truncated re-asks such replies "
                             "at a higher cap.")
    parser.add_argument("--rubric", choices=list(jr.RUBRICS), default="v3",
                        help="The judge prompt (results_analysis/judge_tier_rubrics.py): v3 (default, the "
                             "2026-04-20 descriptions rubric), v3fix (v3 with the poles named by their own type), "
                             "c1-c4 (contemporary rubrics: the score's purpose stated, 0 only for balanced "
                             "entities, -3..+3 or -10..+10, one or two sentences of reasoning).  Give a "
                             "non-default rubric its own --cell_label; the rubric is recorded in each cell.")
    parser.add_argument("--name_format", choices=list(jr.NAME_FORMATS), default=jr.DEFAULT_NAME_FORMAT,
                        help="How prompts show entity names: judge_label (default), the project's judge display "
                             "form; mechanical, underscores to spaces, reproduces the prompts of the cells judged "
                             "before 2026-11-02, to extend one of them.  Not the v3 rubrics, which are "
                             "build_static_prompt's.")
    parser.add_argument("--cell_label", type=str, default="",
                        help="Suffix for this run's cell directories, e.g. @t0 or @nothink: cells go to "
                             "<axis>_tier_eval/<model><label>/ (and batches to _tier_eval_batches/<model><label>/), "
                             "so a rerun under other settings sits beside the earlier cells instead of "
                             "overwriting them or being skipped as done.")
    parser.add_argument("--requests_per_second", type=int, default=10)
    parser.add_argument("--max_in_flight", type=int, default=DEFAULT_MAX_IN_FLIGHT,
                        help=f"Live and Flex calls: at most this many requests in flight per model (default "
                             f"{DEFAULT_MAX_IN_FLIGHT}), sent axis by axis.  Handing the HTTP client every request at "
                             "once stalled a 35,000-request run for nine hours on 2026-11-01.")
    parser.add_argument("--stall_minutes", type=float, default=DEFAULT_STALL_MINUTES,
                        help=f"No answer for this many minutes while live or Flex requests remain (default "
                             f"{DEFAULT_STALL_MINUTES:g}; batch waits do not count): stop, and restart up to "
                             "--stall_restarts times; every answer is already saved, so nothing is lost.  0 turns "
                             f"the check off.  Exit status {EXIT_STALLED} when no restart is left.")
    parser.add_argument("--stall_restarts", type=int, default=DEFAULT_STALL_RESTARTS,
                        help=f"Restarts after a stall before giving up (default {DEFAULT_STALL_RESTARTS}; 0 stops "
                             f"at the first stall)")
    parser.add_argument("--budget_usd", type=float, default=None,
                        help="Real-time and Flex calls (a batch's real-time remainder included): stop issuing "
                             "new requests once this run's measured spend passes this.  A batch is submitted "
                             "whole; --dry_run is its control.  Cells left incomplete are written as "
                             "*.partial.json (no usage.json), and the run exits 2.")
    parser.add_argument("--batch_chunk_requests", type=int, default=DEFAULT_BATCH_CHUNK_REQUESTS,
                        help="Requests per batch submission (default 10,000).  A 10,000-request Anthropic "
                             "submission (about 15 MB in one call) hung for 20 minutes and dropped three times on "
                             "2026-11-01 while Flex runs saturated the connection; 2,500 is safer then.")
    parser.add_argument("--entities_per_call", type=int, default=1,
                        help="Batched judging: this many entities of one axis per call, in a seeded shuffle "
                             "(--shuffle_seed), with a rubric that has a batch form (b5, b10), through any "
                             "transport (one Batch API request per group since 2026-11-01).  Default 1: one "
                             "entity per call.")
    parser.add_argument("--batch_design", choices=("shuffle", "rows", "cols"), default="shuffle",
                        help="How --entities_per_call groups an axis's entities: 'shuffle' (default), consecutive "
                             "groups of a seeded shuffle; 'rows' / 'cols', the rows or the columns of N x N squares "
                             "of the same seeded shuffle, so that the two passes share no batch-mate (judge both, "
                             "under two labels, and average them).")
    parser.add_argument("--instruction_round", type=int, default=None,
                        help="With an i1_* rubric (required): which round of one instruction per entity this cell "
                             "judges, from 0; each entity goes through its instructions in its own seeded order.")
    parser.add_argument("--responses_dir", type=str, default=None,
                        help="With an r1_* rubric (required): the pipeline tree with <kind>/responses/<entity>.jsonl "
                             "and <kind>/scores/<entity>.json (e.g. 'runpod_workspace/qwen/qwen-3-32b Roger 8slot'); "
                             "the entity's pos responses step 3 scored 3 are the items, one score each, the cell's "
                             "result their per-entity means.  A call holds --entities_per_call answers to one question.")
    parser.add_argument("--response_split", type=int, default=3,
                        help="With an r1_* rubric: the number of parts each entity's questions are split into "
                             "(default 3, the thirds of the round-robin coverage; 9 for a ninth, whose parts 0, 3 "
                             "and 6 are the thirds of third 0).")
    parser.add_argument("--response_part", type=int, default=None,
                        help="With an r1_* rubric: judge only this part of the responses, cut as --response_design "
                             "says (the round-robin coverage: the judges' parts together judge every response "
                             "once).  Default: every kept response.")
    parser.add_argument("--response_design", choices=("golden", "question"), default="golden",
                        help="How --response_part cuts: golden (default since 2026-11-04), the responses whose point "
                             "frac((5 x absolute question number + instruction) / golden ratio) lies in the part's "
                             "interval, every entity the same cells (RESPONSE_DESIGN says why; avoid a split of 5); "
                             "question, the questions with (question + entity rank) %% --response_split equal to "
                             "the part, as every cell before 2026-11-04 (pass it to resume or reread those).")
    parser.add_argument("--axes_per_call", type=int, default=1,
                        help="With an rk_* rubric (required) and --axes_file: judge this many consecutive axes of the "
                             "file in each call (the response text paid once for all of them); each axis still gets "
                             "its own cell, and the group's answer log lives under _tier_eval_multi/.  Live and Flex only.")
    parser.add_argument("--directions", choices=DIRECTION_MODES, default="own",
                        help="Which way each call presents its axis (Roger, 2026-11-04: balance the directions so that "
                             "no valence asymmetry enters a result).  own (default): the axis's order.  descriptions: "
                             "a judge's passes alternate (--direction_pass).  instructions: Roger's table per judge and "
                             "instruction (an i1_* rubric; the three tier judges).  responses: a coin per call.  "
                             "Swapped calls' scores are mapped back; every answer record and the cell say which way.  "
                             "Batched rubrics only; not with --swap_poles or several axes per call.")
    parser.add_argument("--stack", type=int, default=0,
                        help="With --axes_per_call M and an rk_* rubric: judge k axes a call (or --stack 1 with r1_pn: "
                             "one axis a call, a batch's M calls one after another, so its prefix stays cached), each batch's M axes "
                             "partitioned afresh at random into groups of k (a random letter order, each axis's "
                             "direction a coin per call), so that leakage between stacked axes is noise, not a fixed "
                             "bias; every batch sent once per group.  Live and Flex only.")
    parser.add_argument("--stack_seed", type=int, default=1, help="With --stack: the seed of the partitions.")
    parser.add_argument("--direction_pass", type=int, default=0,
                        help="With --directions descriptions: which of the judge's passes this run is, from 0.")
    parser.add_argument("--swap_poles", action="store_true",
                        help="Judge every axis with its ends exchanged: the positive pole named first and at the low "
                             "number.  Scores are mapped back to the axis's own direction, so the cells read like any "
                             "other; for testing what the order does.  Needs --cell_label.")
    parser.add_argument("--axis_index", type=axis_indices, default=None,
                        help="With --axes_per_call and a cached-prefix response rubric: judge only these members of "
                             "each group, from 0, comma-separated: one with r1_pn (one axis per call), two or more "
                             "with rk_pn (judged together).  Every run on the group then sends the same batches, so "
                             "a cached prefix (the question and the answers) is read back by the next run.")
    parser.add_argument("--response_chars", type=int, default=None,
                        help="With an r1_* rubric: cut each answer to this many characters (about 4 a token) with "
                             "a mark.  Default: whole.")
    parser.add_argument("--shuffle_seed", type=int, default=0,
                        help="Seed of the batch shuffle (--entities_per_call); another seed gives every entity "
                             "different batch-mates, which measures the crosstalk.")
    parser.add_argument("--batch_poll_seconds", type=float, default=60.0,
                        help="Seconds between status checks of batches in flight")
    parser.add_argument("--batch_fallback_max", type=int, default=DEFAULT_BATCH_FALLBACK_MAX,
                        help="Requests no batch answered are sent in real time up to this many; past it they "
                             "are left for a rerun")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print per axis the entity count, and per model the transport, the number of "
                             "calls and the estimated cost at that transport's price; make zero API calls "
                             "and write nothing.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N entities (alphabetical) of each axis. For validating a "
                             "new judge model cheaply before committing to the full corpus.")
    parser.add_argument("--repair_truncated", action="store_true",
                        help="Instead of judging, repair finished cells: re-ask in real time (standard tier, the "
                             "model's request settings as in a normal run, --max_tokens) exactly the entities "
                             "whose reply was cut at the cap, and patch each cell (those scores, its "
                             "correlations.json, usage.json, and a 'repairs' record in its provenance).  The "
                             "cut entities come from the cell's provenance 'truncated' list; a cell without one "
                             "is left alone.  With --dry_run: report the cells, counts and estimated cost (no "
                             "judge calls).")
    return parser


def build_jobs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[AxisJob]:
    single = {"--pair": args.pair, "--pair_type": args.pair_type, "--projections_file": args.projections_file,
              "--output_dir": args.output_dir}
    if args.axes_file:
        given = [k for k, v in single.items() if v is not None]
        if given:
            parser.error(f"--axes_file excludes {', '.join(given)}")
        jobs = jobs_from_axes_file(Path(args.axes_file), Path(args.experiments_dir))
        if args.axes_per_call == 1:
            return jobs
        groups = group_jobs(jobs, args.axes_per_call, Path(args.experiments_dir))
        short = [g.axis_dir for g in groups if args.axis_index and max(args.axis_index) >= len(g.members)]
        if short:
            parser.error(f"--axis_index {','.join(map(str, args.axis_index))}: the group {short[0]} has fewer members "
                         f"(the axes file's rows come in groups of --axes_per_call; the last may be shorter)")
        return groups
    missing = [k for k, v in single.items() if v is None]
    if missing:
        parser.error(f"without --axes_file, {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} required")
    pos, neg = args.pair
    output_dir = Path(args.output_dir)
    name = output_dir.name[: -len(TIER_EVAL_SUFFIX)] if output_dir.name.endswith(TIER_EVAL_SUFFIX) \
        else f"{pos}_vs_{neg}"
    return [AxisJob(name, pos, neg, args.pair_type, Path(args.projections_file), output_dir)]


def dry_run_report(jobs: List[AxisJob], runs: List[ModelRun], skipped_axes: List[AxisJob]) -> None:
    print(f"\nRubric: {RUBRIC.name} ({RUBRIC.summary}); cell label {CELL_LABEL or '(none)'}"
          + (f"; {BATCHING['n']} entities per call, shuffle seed {BATCHING['seed']}" if BATCHING["n"] > 1 else ""))
    print(f"\n{len(jobs)} axes" + (f" ({len(skipped_axes)} more skipped: no projections.json)"
                                  if skipped_axes else ""))
    for i, job in enumerate(jobs):
        extra = f"  ({len(job.missing)} without a description, excluded)" if job.missing else ""
        print(f"  [{i:2d}] {job.axis_dir:55s} {job.pair_type:6s} {len(job.entities):4d} entities{extra}")
    print("\nPer model:")
    total = 0.0
    for mrun in runs:
        model, choice = mrun.model, mrun.choice
        todo = [job for job in jobs if not cell_done(job, model)]
        n_saved = sum(saved_answers(job, model) for job in todo)
        n_calls = sum(len(job.entities) for job in todo) - n_saved
        per_call = BATCHING["n"]
        if per_call > 1:
            n_calls = sum(math.ceil((len(job.entities) - saved_answers(job, model)) / per_call) for job in todo)
        rate_in, rate_out = price_for_model(tt.price_key(model, choice.transport))
        if per_call > 1:
            est = n_calls * ((EST_BATCH_HEADER + EST_BATCH_IN_BY_SOURCE[RUBRIC.text_source] * per_call) * rate_in
                             + EST_BATCH_OUT_PER_ENTITY * per_call * rate_out) / 1e6
        else:
            est = n_calls * (EST_IN_PER_ENTITY * rate_in + EST_OUT_PER_ENTITY * rate_out) / 1e6
        total += est
        print(f"{model:22s} provider={provider_for_model(model):9s} transport={choice.transport}"
              f"{' (probe)' if choice.probe else '':8s} rate=${rate_in:.3f}/${rate_out:.3f} per 1M  "
              f"axes {len(todo)}/{len(jobs)}  ~{n_calls:,} calls  est.cost=${est:.2f}")
        print(f"{'':22s} why: {choice.reason}")
        if n_saved:
            print(f"{'':22s} {n_saved:,} answers saved by earlier runs are resumed (not counted above)")
        print(f"{'':22s} request settings: {json.dumps(mrun.settings)}")
        if choice.transport == "flex":
            print(f"{'':22s} (Flex at batch prices; a request sent to the standard tier after "
                  f"{tt.FLEX_FALLBACK_AFTER_S:.0f} s of Flex failures costs the full rate)")
    tokens = (f"{EST_BATCH_HEADER} + {EST_BATCH_IN_BY_SOURCE[RUBRIC.text_source]} per entity in, "
              f"{EST_BATCH_OUT_PER_ENTITY} per entity out"
              if BATCHING["n"] > 1 else f"{EST_IN_PER_ENTITY} in / {EST_OUT_PER_ENTITY} out per entity")
    print(f"\nTotal estimated cost: ${total:.2f}  (token model: {tokens}, coarse)")
    print("Dry run only -- no API calls made, no cost incurred, nothing written.")


async def main_async(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    global CELL_LABEL, RUBRIC, NAME_FORMAT, NAME_SHOW, INSTRUCTION_ROUND, RESPONSES_DIR, RESPONSE_SPLIT, RESPONSE_PART, \
        RESPONSE_CHARS, AXIS_INDEX, SWAP_POLES, DIRECTIONS, DIRECTION_PASS, STACK, STACK_SEED, RESPONSE_DESIGN
    CELL_LABEL = args.cell_label or ""
    SWAP_POLES = bool(args.swap_poles)
    DIRECTIONS, DIRECTION_PASS = args.directions, args.direction_pass
    STACK, STACK_SEED = args.stack, args.stack_seed
    if STACK:
        rubric = jr.get(args.rubric)
        if rubric.text_source != "responses" or (STACK == 1) == rubric.multi_axis:
            parser.error("--stack goes with a multi-axis response rubric (rk_pn, rk_pl, rk_pa), or --stack 1 with a "
                         "one-axis one (r1_pn: every axis of the group a call of its own, a batch's calls in turn)")
        if not 1 <= STACK <= args.axes_per_call:
            parser.error("--stack k needs 1 <= k <= --axes_per_call")
        if args.axis_index is not None or DIRECTIONS != "own" or args.swap_poles:
            parser.error("--stack makes its own groups and directions: not with --axis_index, --directions or --swap_poles")
        if not args.cell_label:
            parser.error("--stack needs a --cell_label")
    if DIRECTIONS != "own":
        need = {"descriptions": "descriptions", "instructions": "instruction_rounds", "responses": "responses"}[DIRECTIONS]
        rubric = jr.get(args.rubric)
        if SWAP_POLES:
            parser.error("--directions and --swap_poles exclude each other")
        if rubric.text_source != need:
            parser.error(f"--directions {DIRECTIONS} goes with a rubric judging {need}, not {rubric.name}")
        if rubric.build_batch is None or args.entities_per_call < 2 or rubric.multi_axis:
            parser.error(f"--directions {DIRECTIONS}: a batched one-axis rubric with --entities_per_call 2 or more")
        if DIRECTIONS == "instructions":
            unknown = [m for m in args.judge_model if m not in INSTRUCTION_DIRECTION_JUDGES]
            if unknown:
                parser.error(f"--directions instructions: no column for {', '.join(unknown)} in INSTRUCTION_DIRECTIONS "
                             f"({', '.join(INSTRUCTION_DIRECTION_JUDGES)})")
        if not CELL_LABEL:
            parser.error(f"--directions {DIRECTIONS} needs a --cell_label, so that its cells stand apart")
    if DIRECTION_PASS and DIRECTIONS != "descriptions":
        parser.error("--direction_pass goes with --directions descriptions")
    RUBRIC = jr.with_poles_swapped(jr.get(args.rubric)) if SWAP_POLES else jr.get(args.rubric)
    if SWAP_POLES and not CELL_LABEL:
        parser.error("--swap_poles needs a --cell_label, so that its cells stand apart from the axis's own")
    if (RUBRIC.text_source == "instruction_rounds") != (args.instruction_round is not None):
        parser.error("--instruction_round goes with an i1_* rubric, and an i1_* rubric needs it")
    INSTRUCTION_ROUND = args.instruction_round or 0
    if (RUBRIC.text_source == "responses") != (args.responses_dir is not None):
        parser.error("--responses_dir goes with an r1_* rubric, and an r1_* rubric needs it")
    if RUBRIC.text_source != "responses" and (args.response_part is not None or args.response_chars is not None):
        parser.error("--response_part and --response_chars go with an r1_* rubric")
    if RUBRIC.text_source == "responses" and args.entities_per_call < 2:
        parser.error("an r1_* rubric judges several answers to one question per call: give --entities_per_call")
    if args.response_split < 1 or (args.response_part is not None and not 0 <= args.response_part < args.response_split):
        parser.error("--response_part must be from 0 to --response_split - 1")
    if RUBRIC.multi_axis and args.axes_per_call < 2:
        parser.error("an rk_* rubric needs --axes_per_call (2 or more)")
    if args.axes_per_call > 1 and not (RUBRIC.multi_axis or args.axis_index is not None or args.stack == 1):
        parser.error("--axes_per_call goes with an rk_* rubric, or with --axis_index or --stack 1 and a one-axis "
                     "response rubric")
    if args.axis_index is not None:
        if (RUBRIC.text_source != "responses" or args.axes_per_call < 2
                or not all(0 <= i < args.axes_per_call for i in args.axis_index)):
            parser.error("--axis_index needs --axes_per_call, a response rubric, and 0 <= each index < axes per call")
        if RUBRIC.multi_axis and len(args.axis_index) < 2:
            parser.error(f"--axis_index with {RUBRIC.name}: two or more members (one axis per call is r1_pn)")
        if not RUBRIC.multi_axis and len(args.axis_index) != 1:
            parser.error(f"--axis_index with {RUBRIC.name}: one member (several judged together is rk_pn)")
    if args.axes_per_call > 1 and not args.axes_file:
        parser.error("--axes_per_call needs --axes_file (the axes judged together are consecutive rows)")
    AXIS_INDEX = args.axis_index
    RESPONSES_DIR = Path(args.responses_dir) if args.responses_dir else None
    RESPONSE_SPLIT, RESPONSE_PART, RESPONSE_CHARS = args.response_split, args.response_part, args.response_chars
    RESPONSE_DESIGN = args.response_design
    if args.repair_truncated and RUBRIC.text_source != "descriptions":
        parser.error("--repair_truncated repairs description cells only")
    NAME_FORMAT = args.name_format
    NAME_SHOW = jr.name_shower(NAME_FORMAT, Path(args.instructions_dir))
    BATCHING["n"], BATCHING["seed"], BATCHING["design"] = max(1, args.entities_per_call), args.shuffle_seed, args.batch_design
    if args.entities_per_call > 1:
        if RUBRIC.build_batch is None:
            parser.error(f"--entities_per_call needs a rubric with a batch form (b5, b10), not {args.rubric}")
        if not CELL_LABEL:
            parser.error("--entities_per_call needs a --cell_label (e.g. @b5s1)")
    if args.rubric != "v3" and not CELL_LABEL:
        parser.error(f"--rubric {args.rubric} needs a --cell_label (e.g. @{args.rubric}), so its cells do not "
                     f"mix with the v3 cells")
    if CELL_LABEL and not re.fullmatch(r"[@A-Za-z0-9._-]+", CELL_LABEL):
        parser.error(f"--cell_label {CELL_LABEL!r}: letters, digits and @ . _ - only")
    jobs = build_jobs(args, parser)
    instructions_dir = Path(args.instructions_dir)
    try:
        overrides = [tt.parse_request_setting(s) for s in args.request_setting]
    except ValueError as e:
        parser.error(str(e))

    runs: List[ModelRun] = []
    for model in dict.fromkeys(args.judge_model):
        if args.repair_truncated:
            choice = tt.TransportChoice("realtime", False, "--repair_truncated: real time, standard tier")
        else:
            try:
                choice = tt.choose_transport(model, args.transport)
            except (tt.JudgeRequestError, ValueError) as e:
                parser.error(str(e))
        if args.axes_per_call > 1 and choice.transport == "batch":
            parser.error(f"{model}: --axes_per_call is not wired for the Batch API yet; --transport flex or realtime")
        runs.append(ModelRun(model=model, settings=tt.judge_request_settings(model, overrides), choice=choice))

    if args.repair_truncated:
        for flag, value in (("--limit", args.limit), ("--slots", None if args.slots == "all" else args.slots),
                            ("--transport", None if args.transport == "auto" else args.transport)):
            if value is not None:
                parser.error(f"--repair_truncated excludes {flag} (a repair re-asks in real time and keeps "
                             f"the cell's entities and slots)")
        for mrun in runs:
            try:
                price_for_model(mrun.model)
            except KeyError as e:
                parser.error(e.args[0] if e.args else f"no price for {mrun.model!r}")

    skipped_axes = [job for job in jobs if not Path(job.projections_file).exists()]
    for job in skipped_axes:
        logger.warning(f"SKIP (no projections.json): {job.axis_dir} ({job.projections_file})")
    jobs = [job for job in jobs if Path(job.projections_file).exists()]
    if args.repair_truncated:
        return await repair_main(args, jobs, runs)
    for job in jobs:
        load_axis(job, instructions_dir, args.slots, args.limit)
    if args.limit is not None:
        logger.warning(f"--limit {args.limit}: only scoring the first {args.limit} entities of each axis "
                       f"(correlations below are NOT meaningful at this n, validation only)")
    logger.info(f"{len(jobs)} axes, {sum(len(j.entities) for j in jobs)} (axis, entity) pairs scorable "
                f"({RUBRIC.mode} mode)")

    # Every usage key a run may charge must be priced before anything is sent.
    for mrun in runs:
        for key in {tt.price_key(mrun.model, mrun.choice.transport), mrun.model,
                    *([tt.price_key(mrun.model, "batch")] if mrun.choice.probe else [])}:
            try:
                price_for_model(key)
            except KeyError as e:
                parser.error(e.args[0] if e.args else f"no price for {key!r}")

    if args.dry_run:
        dry_run_report(jobs, runs, skipped_axes)
        return 0

    # Keys come from the environment or from a .env found from the working
    # directory upward (the project's convention: load_dotenv, never sourced
    # by a shell script, never printed).  usecwd=True so that a run started
    # from the repository root finds the root's .env even when this script
    # lives elsewhere.
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    missing_keys = sorted({key_name(m.model) for m in runs if not os.getenv(key_name(m.model))})
    if missing_keys:
        logger.error(f"{', '.join(missing_keys)} not set (environment or .env): nothing sent")
        return 1

    ctx = RunContext(instructions_dir=instructions_dir,
                     batch_root=Path(args.experiments_dir) / BATCH_DIRNAME,
                     max_tokens=args.max_tokens, requests_per_second=args.requests_per_second,
                     budget_usd=args.budget_usd, batch_poll_seconds=args.batch_poll_seconds,
                     batch_fallback_max=args.batch_fallback_max, slots=args.slots, limit=args.limit,
                     max_in_flight=max(1, args.max_in_flight), stall_s=max(0.0, args.stall_minutes) * 60,
                     batch_chunk_requests=max(1, args.batch_chunk_requests),
                     entities_per_call=max(1, args.entities_per_call), shuffle_seed=args.shuffle_seed,
                     batch_design=args.batch_design)
    global STALL_RESTARTS
    STALL_RESTARTS = max(0, args.stall_restarts)
    if os.environ.get(RESTART_ENV):
        logger.warning(f"restart {os.environ[RESTART_ENV]} of {STALL_RESTARTS} after a stall: resuming from the "
                       f"saved answers")
    if args.budget_usd is not None and any(m.choice.transport == "batch" for m in runs):
        logger.info("--budget_usd does not stop a batch submission (it is submitted whole); it applies to "
                    "real-time and Flex calls, a batch's real-time remainder included")
    work = asyncio.ensure_future(asyncio.gather(*(run_model(ctx, mrun, jobs) for mrun in runs),
                                                return_exceptions=True))
    helpers = [asyncio.create_task(report_progress(ctx, runs, PROGRESS_EVERY_S))]
    if ctx.stall_s > 0:
        helpers.append(asyncio.create_task(watch_for_stall(ctx, work)))
    try:
        results = await work
    except asyncio.CancelledError:
        if not ctx.stalled:
            raise
        results = [None] * len(runs)
    finally:
        for h in helpers:
            h.cancel()
    for mrun, res in zip(runs, results):
        if isinstance(res, BaseException):
            mrun.error = res
            logger.error(f"[{mrun.model}] failed: {type(res).__name__}: {res}")

    print("\nSummary:")
    for mrun in runs:
        print("  " + summarise(mrun) + (f"  FAILED: {type(mrun.error).__name__}" if mrun.error else ""))
        # requests left for a rerun count against the parse rate, as in step 3
        warn_if_low_parse_rate(label=f"judge_tier_cost_eval:{mrun.model}", n_ok=mrun.parsed,
                               n_total=mrun.attempted + mrun.left_for_rerun, logger_obj=logger)
    logger.info(ctx.run_usage.log_line())
    if ctx.stalled:
        return EXIT_STALLED
    if any(m.error for m in runs):
        return 1
    if ctx.budget_hit:
        return 2
    if any(m.left_for_rerun or m.cells_incomplete or m.cells_partial for m in runs):
        logger.error("calls stayed unanswered: rerun to ask them (cells with gaps have no usage.json, and the "
                     "saved answers are kept)")
        return EXIT_INCOMPLETE
    return 0


def main() -> None:
    code = asyncio.run(main_async())
    if code == EXIT_STALLED:
        restart_after_stall()   # returns only when no restart is left
    sys.exit(code)


if __name__ == "__main__":
    main()
