#!/usr/bin/env python3
"""
Score a new judge model against an EXISTING, already-computed axis, without
recomputing activations, projections, or touching a GPU.

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
  auto picks batch for an Anthropic model; for an OpenAI model, flex when
  results_analysis/judge_tier_transport.py's MODEL_CAPABILITIES says OpenAI
  takes it, batch when it says OpenAI refuses it, and when it is untested,
  flex with a probe: the first request alone on Flex, and a service-tier
  HTTP 400 switches the model to batch for the run (that request goes into
  the batch).  The choice is logged and recorded in each cell's
  scores_descriptions.json _provenance (transport, request_settings,
  max_tokens).

Request settings per model (reasoning effort, thinking, effort) come from
judge_tier_transport.JUDGE_REQUEST_SETTINGS; for the models earlier cells used
they reproduce the requests this tree's assistant_axis/judge.py sent, so new
cells extend old ones.  --request_setting [MODEL_FRAGMENT:]key=<json> overrides
the table for one run.

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
from the provenance "truncated" list, or for batch cells made before that list
existed from the batches' results; a cell with neither is reported and left
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
missing), 2 --budget_usd stopped the run, 3 it stalled and no restart was left.
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
from results_analysis import judge_tier_rubrics as jr  # noqa: E402
from assistant_axis.judge import RateLimiter, provider_for_model, warn_if_low_parse_rate  # noqa: E402
from assistant_axis.judge_pricing import BATCH_SUFFIX, MultiModelUsage, price_for_model  # noqa: E402
from assistant_axis.entity_id import parse_entity_id  # noqa: E402
from assistant_axis.atomic_io import atomic_write_bytes, atomic_write_text  # noqa: E402
from assistant_axis.openai_batch import OpenAIBatchRunner  # noqa: E402
from assistant_axis.anthropic_batch import AnthropicBatchRunner  # noqa: E402

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
#: The environment variable that counts restarts after a stall across execs.
RESTART_ENV = "JUDGE_TIER_EVAL_STALL_RESTARTS"
#: Per-cell answer log (see "Saved answers" in the docstring).
ANSWERS_PARTIAL = "answers.partial.jsonl"
ANSWERS_ARCHIVE = "answers.jsonl.gz"
CHECKPOINT_VERSION = 1
#: Header fields an answer log must share with the run that resumes it.
CHECKPOINT_KEYS = ("version", "judge_model", "cell_label", "rubric_version", "request_settings", "max_tokens")


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
    axis_spec: Optional[AxisSpec] = field(default=None, repr=False)

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


def resolve_entities(entity_names: Sequence[str],
                     instructions_dir: Path) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """``(etype, name, description)`` per entity, and the names with no file.

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
            desc, etype = load_entity_description(instructions_dir, kind_hint, bare), kind_hint
        else:
            desc, etype = load_entity_description(instructions_dir, "roles", bare), "roles"
            if desc is None:
                desc, etype = load_entity_description(instructions_dir, "traits", bare), "traits"
        if desc is None:
            missing.append(name)
            continue
        entities.append((etype, name, desc))
    return entities, missing


def build_axis_spec(pos: str, neg: str, pair_type: str, instructions_dir: Path,
                    projections_file: Path) -> AxisSpec:
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
    )


def load_axis(job: AxisJob, instructions_dir: Path, slots: str = "all", limit: Optional[int] = None) -> AxisJob:
    job.projections = load_projections(job.projections_file, slots)
    any_slot = next(iter(job.projections.values()))
    names = sorted(any_slot.keys())
    if limit is not None:
        names = names[:limit]
    job.entities, job.missing = resolve_entities(names, instructions_dir)
    if job.missing:
        logger.warning(f"{job.axis_dir}: {len(job.missing)} entities from the cached projections no longer "
                       f"have an instructions file under {instructions_dir} -- excluded: {job.missing[:10]}"
                       f"{'...' if len(job.missing) > 10 else ''}")
    job.axis_spec = build_axis_spec(job.pos, job.neg, job.pair_type, instructions_dir, job.projections_file)
    return job


#: Suffix of every cell directory (and batch work directory) of this run,
#: from --cell_label: ``claude-opus-5@nothink`` holds Opus 5 judged with thinking
#: off beside ``claude-opus-5`` (low effort), so cells made under different
#: settings never overwrite or resume into each other.  Set in main_async.
CELL_LABEL = ""
#: The rubric of this run (--rubric, judge_tier_rubrics.RUBRICS); v3 unless set.  Set in main_async.
RUBRIC: "jr.RubricSpec" = jr.RUBRICS["v3"]
#: Batched judging of this run (--entities_per_call, --shuffle_seed); set in main_async.
BATCHING = {"n": 1, "seed": 0, "design": "shuffle"}


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


def build_prompt(job: "AxisJob", etype: str, name: str, desc: str) -> str:
    """The judge prompt of this run's rubric for one entity of ``job``."""
    return RUBRIC.build(job.axis_spec, job.pair_type, etype, name, desc)


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
    usage: MultiModelUsage = field(default_factory=MultiModelUsage)
    written: bool = False

    @property
    def complete(self) -> bool:
        return self.expected <= set(self.scores)

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
    to_judge: int = 0       # live or Flex requests this run set out to send
    in_flight: int = 0
    error: Optional[BaseException] = None

    def __post_init__(self) -> None:
        self.transport = self.transport or self.choice.transport
        self.transport_reason = self.transport_reason or self.choice.reason


class _Charge:
    """Charges one call to several trackers (the cell's, the model's, the run's)
    and keeps the charges, ``[key, prompt_tokens, completion_tokens]``, for the
    answer log."""

    def __init__(self, *trackers: MultiModelUsage):
        self.trackers = trackers
        self.charges: List[List[Any]] = []

    def charge(self, key: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.charges.append([key, int(prompt_tokens), int(completion_tokens)])
        for t in self.trackers:
            t.charge(key, prompt_tokens, completion_tokens)


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


def record(mrun: ModelRun, cell: Cell, name: str, reply: Any, score: Any = _UNSET) -> None:
    """Record one reply: its text, None (a failed call or a refusal), or a
    :class:`judge_tier_transport.Truncated` (cut at max_tokens), which scores
    None without being parsed and is listed in the cell's provenance.  With
    ``score`` given (batched judging), the reply is not parsed again."""
    mrun.attempted += 1
    if isinstance(reply, tt.Truncated):
        cell.scores[name] = None
        cell.truncated.add(name)
        mrun.stats.truncated += 1
        logger.warning(f"{mrun.model}: reply for {name!r} ({cell.job.axis_dir}) cut at max_tokens: "
                       f"scored as missing")
        return
    cell.truncated.discard(name)
    score = (parse_score(reply) if reply else None) if score is _UNSET else score
    cell.scores[name] = score
    if score is None:
        logger.warning(f"{mrun.model}: unparseable/empty response for {name!r} ({cell.job.axis_dir})")
    else:
        mrun.parsed += 1


def write_cell(cell: Cell, mrun: ModelRun, max_tokens: int) -> None:
    """Write the cell's scores_descriptions.json, correlations.json and, last,
    usage.json (the file the resume rule looks for)."""
    job, model = cell.job, mrun.model
    d = cell_dir(job, model)
    d.mkdir(parents=True, exist_ok=True)
    scores = cell.parsed_scores()
    provenance = {"judge_model": model, "rubric_version": RUBRIC.version, "rubric": RUBRIC.name,
                  "projections_file": str(job.projections_file),
                  "transport": mrun.transport, "transport_reason": mrun.transport_reason,
                  "request_settings": mrun.settings, "max_tokens": max_tokens,
                  "cell_label": CELL_LABEL, "truncated": sorted(cell.truncated),
                  "entities_per_call": BATCHING["n"], "shuffle_seed": BATCHING["seed"],
                  "batch_design": BATCHING["design"]}
    atomic_write_text(json.dumps({"result": scores, "_provenance": provenance}, indent=2),
                      d / "scores_descriptions.json", logger_obj=logger)
    correlations = compute_correlations(
        scores_by_mode={"descriptions": scores},
        projections=job.projections,
        slots=list(job.projections.keys()),
        excluded_set=set(job.axis_spec.exclusions),
    )
    atomic_write_text(json.dumps(correlations, indent=2), d / "correlations.json", logger_obj=logger)
    log = d / ANSWERS_PARTIAL
    if log.exists():
        # the raw replies stay, compressed; the log itself goes once usage.json is written
        atomic_write_bytes(gzip.compress(log.read_bytes()), d / ANSWERS_ARCHIVE, logger_obj=logger)
    atomic_write_text(json.dumps(cell.usage.as_dict(), indent=2, sort_keys=True), d / "usage.json",
                      logger_obj=logger)
    log.unlink(missing_ok=True)
    cell.written = True
    mrun.cells_written += 1
    logger.info(f"[{model}] {job.axis_dir}: scored {len(scores)}/{len(cell.expected)}"
                f"{f', {len(cell.truncated)} cut at max_tokens' if cell.truncated else ''}; "
                f"cost ${cell.usage.total_cost_usd:.4f} (see {d / 'usage.json'})")
    for slot, per_metric in correlations["descriptions"].items():
        raw = per_metric["raw"]
        logger.info(f"[{model}] {job.axis_dir} slot={slot} raw: rho={raw['rho']:.4f} n={raw['n']}")


def write_partial_cell(cell: Cell, mrun: ModelRun, max_tokens: int) -> None:
    """A cell the budget stop left incomplete: its scores and spend under
    ``*.partial.json`` (no usage.json, so a rerun judges the axis again; the
    partial spend accumulates across runs)."""
    job, model = cell.job, mrun.model
    d = cell_dir(job, model)
    d.mkdir(parents=True, exist_ok=True)
    provenance = {"judge_model": model, "rubric_version": RUBRIC.version, "rubric": RUBRIC.name,
                  "projections_file": str(job.projections_file), "transport": mrun.transport,
                  "request_settings": mrun.settings, "max_tokens": max_tokens, "complete": False,
                  "n_attempted": len(cell.scores), "n_entities": len(cell.expected),
                  "truncated": sorted(cell.truncated)}
    atomic_write_text(json.dumps({"result": cell.parsed_scores(), "_provenance": provenance}, indent=2),
                      d / "scores_descriptions.partial.json", logger_obj=logger)
    acc = MultiModelUsage.load_or_create(d / "usage.partial.json")
    acc.merge_from(cell.usage)
    atomic_write_text(json.dumps(acc.as_dict(), indent=2, sort_keys=True), d / "usage.partial.json",
                      logger_obj=logger)
    mrun.cells_partial += 1
    logger.warning(f"[{model}] {job.axis_dir}: incomplete ({len(cell.scores)}/{len(cell.expected)} attempted); "
                   f"written as {d / 'scores_descriptions.partial.json'}, no usage.json")


# ---------------------------------------------------------------------------
# Saved answers: the per-cell answer log (2026-11-01)
# ---------------------------------------------------------------------------

class CheckpointMismatchError(RuntimeError):
    """An answer log written under other settings than this run's."""


def checkpoint_header(mrun: "ModelRun", max_tokens: int) -> Dict[str, Any]:
    return {"kind": "header", "version": CHECKPOINT_VERSION, "judge_model": mrun.model, "cell_label": CELL_LABEL,
            "rubric_version": RUBRIC.version, "request_settings": mrun.settings, "max_tokens": max_tokens}


def prompt_digest(prompt: str) -> str:
    """Identifies the exact prompt an answer replied to (corpus text and rubric included)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


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
    if not answers:
        return 0
    if found is None:
        raise CheckpointMismatchError(f"{path}: no header line, so its settings are unknown: move it aside to "
                                      f"judge {cell.job.axis_dir} afresh")
    diffs = {k: (found.get(k), header.get(k)) for k in CHECKPOINT_KEYS if found.get(k) != header.get(k)}
    if diffs:
        raise CheckpointMismatchError(
            f"{path} was written under other settings than this run's ("
            + "; ".join(f"{k}: {a!r} there, {b!r} now" for k, (a, b) in diffs.items())
            + "): rerun with those settings to finish it, use another --cell_label, or move the file aside "
              f"to judge {cell.job.axis_dir} afresh")
    descs = {name: (etype, desc) for etype, name, desc in cell.job.entities}
    taken = stale = 0
    for name, rec in answers.items():
        if name not in cell.expected or name in cell.scores:
            continue
        etype, desc = descs[name]
        if not rec.get("batch") and rec.get("prompt") != prompt_digest(build_prompt(cell.job, etype, name, desc)):
            stale += 1
            continue
        if rec.get("batch"):
            # a batched reply: this entity's line, by its position in the batch
            batch = rec["batch"]
            parsed = ({} if rec.get("truncated") or not rec.get("reply") or RUBRIC.parse_batch is None
                      else RUBRIC.parse_batch(rec["reply"], len(batch)))
            restore_score(cell, name, parsed.get(batch.index(name) + 1) if parsed and name in batch else None,
                          bool(rec.get("truncated")))
        else:
            restore(cell, name, rec.get("reply"), bool(rec.get("truncated")))
        for key, pt, ct in rec.get("charges") or []:
            cell.usage.charge(key, pt, ct)
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
    d = cell_dir(cell.job, model)
    d.mkdir(parents=True, exist_ok=True)
    path = d / ANSWERS_PARTIAL
    truncated = isinstance(reply, tt.Truncated)
    rec = {"kind": "answer", "name": name, "reply": reply.text if truncated else reply, "truncated": truncated,
           "charges": charges, "prompt": prompt_digest(prompt), **({"batch": batch} if batch else {}),
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")}
    lines = [] if path.exists() else [json.dumps(header, sort_keys=True)]
    lines.append(json.dumps(rec, sort_keys=True))
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()


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
    # Diagnosed 2026-09-23: a mismatched httpx2/Brotli version pairing
    # in the (ad-hoc, not uv.lock-pinned) environment this was first
    # run in raised "TypeError: process() takes no keyword arguments"
    # deep in response decompression for gpt-5.6-* (larger/brotli-
    # encoded responses than the older gpt-4.1-mini ever triggered).
    # Disabling response compression sidesteps it regardless of root
    # cause; harmless here since judge payloads are a short prompt +
    # a few hundred tokens, not documents worth compressing.
    kwargs: Dict[str, Any] = {"default_headers": {"Accept-Encoding": "identity"}}
    if service_tier == "flex":
        # Flex calls can wait in a queue and be refused with an uncharged 429:
        # a longer timeout and more SDK retries, as step 3's judge_client.
        kwargs.update(timeout=tt.FLEX_TIMEOUT_S, max_retries=tt.FLEX_MAX_RETRIES)
    return openai.AsyncOpenAI(**kwargs)


def make_anthropic_client():
    import anthropic
    return anthropic.AsyncAnthropic()


def key_name(model: str) -> str:
    """The environment variable holding the key ``model`` needs (only its presence is checked)."""
    return "ANTHROPIC_API_KEY" if provider_for_model(model) == "anthropic" else "OPENAI_API_KEY"


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
    else:
        client = make_openai_client(tier)
        fallback = make_openai_client(None) if tier == "flex" else None
    header = checkpoint_header(mrun, ctx.max_tokens)
    cells = {i: Cell(i, job, expected={name for _, name, _ in job.entities}) for i, job in enumerate(todo)}
    for cell in cells.values():
        resume_cell(cell, mrun, header)
        if cell.complete and not cell.written:
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

    async def one(item: Tuple[int, Any, Any, Any], *, probing: bool = False) -> None:
        nonlocal unfinished
        if n_per > 1:
            i, group = item[0], item[1]
            cell = cells[i]
            prompt = RUBRIC.build_batch(cell.job.axis_spec, cell.job.pair_type, group)
        else:
            i, etype, name, desc = item
            cell = cells[i]
            prompt = build_prompt(cell.job, etype, name, desc)
        charge = _Charge(cell.usage, mrun.usage, ctx.run_usage)
        mrun.in_flight += 1
        try:
            text = await tt.call_model(model, prompt, ctx.max_tokens, rate_limiter, charge, client=client,
                                       settings=mrun.settings, service_tier=tier, fallback_client=fallback,
                                       stats=mrun.stats, gate=gate)
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
        if text is tt.SKIPPED:
            return
        if n_per > 1:
            names = [g[1] for g in group]
            scores = ({} if isinstance(text, tt.Truncated) or not text
                      else RUBRIC.parse_batch(text, len(group)))
            for k, (_etype, gname, _desc) in enumerate(group, 1):
                if charge.charges:
                    append_answer(cell, model, header, gname, text, charge.charges if k == 1 else [], prompt,
                                  batch=names)
                record(mrun, cell, gname, text, score=scores.get(k) if scores else None)
        else:
            if charge.charges:
                append_answer(cell, model, header, name, text, charge.charges, prompt)
            record(mrun, cell, name, text)
        ctx.note_answer()
        if cell.complete and not cell.written:
            write_cell(cell, mrun, ctx.max_tokens)

    try:
        if probe and items:
            first, items = items[0], items[1:]
            try:
                await one(first, probing=True)
            except tt.FlexRefusedError as e:
                logger.warning(f"[{model}] probe: OpenAI refused Flex processing ({e}); judging {model} through "
                               f"the Batch API for this run.  Add {model!r}: {{'flex': False}} to "
                               f"MODEL_CAPABILITIES in results_analysis/judge_tier_transport.py")
                return False
            if not fatal:
                mrun.transport_reason += "; the probe was answered on Flex"
                logger.info(f"[{model}] probe: Flex accepted; add {model!r}: {{'flex': True}} to "
                            f"MODEL_CAPABILITIES in results_analysis/judge_tier_transport.py")
        await run_bounded(one, items, ctx.max_in_flight)
    finally:
        ctx.live_pending -= unfinished   # requests never sent (a refused probe, a cancelled run)
    if fatal:
        raise fatal[0]
    for cell in cells.values():
        if not cell.written:
            if cell.scores:
                write_partial_cell(cell, mrun, ctx.max_tokens)
            else:
                mrun.cells_incomplete += 1
    return True


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
           "rubric_version": RUBRIC.version, "instructions_dir": str(Path(ctx.instructions_dir).resolve())}
    if ctx.entities_per_call > 1:
        out.update(entities_per_call=ctx.entities_per_call, shuffle_seed=ctx.shuffle_seed,
                   batch_design=ctx.batch_design)
    return out


def axis_groups(job: AxisJob, ctx: RunContext) -> List[List[Tuple[str, str, str]]]:
    """The batched-judging groups of an axis's full entity list, from the seed
    and design (the same in every run with the same settings)."""
    full = list(job.entities)
    random.Random(ctx.shuffle_seed * 1_000_003 + 0).shuffle(full)
    return grid_batches(full, ctx.entities_per_call, ctx.batch_design)


async def run_batch(ctx: RunContext, mrun: ModelRun, jobs: List[AxisJob]) -> None:
    """Judge every entity of the not-done axes of ``jobs`` through the
    provider's batch API: one submission for all axes, results collected into
    per-axis cells, each written as soon as its axis is complete."""
    model = mrun.model
    anthropic = provider_for_model(model) == "anthropic"
    work = batch_work_dir(ctx, model)
    collected_dir = work / "collected"
    if not (work / "batch_state.json").exists() and all(cell_done(job, model) for job in jobs):
        logger.info(f"[{model}] batch: nothing to submit")
        return
    client = make_anthropic_client() if anthropic else make_openai_client(None)
    if anthropic:
        runner = AnthropicBatchRunner(client, work / "chunks", work / "batch_state.json",
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

    def record_group(cell: Cell, cid: str, reply: Any) -> None:
        members = group_members(cid)
        if members is None:
            logger.warning(f"[{model}] group {cid!r} is not in {groups_path}; its reply is dropped")
            return
        parsed = ({} if isinstance(reply, tt.Truncated) or not reply else RUBRIC.parse_batch(reply, len(members)))
        for k, gname in enumerate(members, 1):
            record(mrun, cell, gname, reply, score=parsed.get(k) if parsed else None)

    def on_result(cid: str, body: Optional[dict], error: Optional[str]) -> None:
        if body is None:
            return   # unanswered: comes back from runner.run and is sent in real time
        idx, name = split_custom_id(cid)
        if anthropic:
            text, finish, n_in, n_out, refused = tt.anthropic_reply(body)
            if refused:
                details = body.get("stop_details") if isinstance(body, dict) else None
                mrun.stats.note_refusal(model, (details or {}).get("category")
                                        if isinstance(details, dict) else None)
                text = None
        else:
            text, finish, n_in, n_out = tt.openai_reply(body)
            refused = False
        reply: Any = tt.Truncated(text) if finish == tt.CUT_AT_CAP and not refused else text
        cell = cell_for(idx)
        if cell is None:
            mrun.usage.charge(batch_key, n_in, n_out)
            ctx.run_usage.charge(batch_key, n_in, n_out)
            logger.warning(f"[{model}] result {cid!r} is for an axis already done or unknown; dropped "
                           f"(its tokens are in this run's total only)")
            return
        _Charge(cell.usage, mrun.usage, ctx.run_usage).charge(batch_key, n_in, n_out)
        if name.startswith("g") and name[1:].isdigit() and n_per > 1:
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
            if cell.complete:
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
        if cells[idx].complete:
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
                    yield cid, idx, RUBRIC.build_batch(cell.job.axis_spec, cell.job.pair_type, group)
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
            if text is tt.SKIPPED:
                return
            if is_group:
                members = group_members(cid) or []
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
        mrun.left_for_rerun += len(unanswered)
        logger.error(f"[{model}] {len(unanswered)} requests were not answered by their batches, more than "
                     f"--batch_fallback_max ({ctx.batch_fallback_max}) to send in real time; they stay "
                     f"unscored, and a rerun submits them again")
    for idx in current:
        cell = cells[idx]
        if not cell.written:
            mrun.cells_incomplete += 1
            if cell.scores:
                touched.add(idx)
    flush()   # incomplete cells' results stay under collected/ for the rerun


# ---------------------------------------------------------------------------
# Per model
# ---------------------------------------------------------------------------

async def run_model(ctx: RunContext, mrun: ModelRun, jobs: List[AxisJob]) -> None:
    model = mrun.model
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
    accepted = await run_live(ctx, mrun, todo, tier=tier, probe=mrun.choice.probe)
    if not accepted:
        mrun.transport = "batch"
        mrun.transport_reason = (f"{mrun.choice.reason}; OpenAI refused Flex on the probe (HTTP 400), "
                                 f"so the run switched to the Batch API")
        await run_batch(ctx, mrun, jobs)


def summarise(mrun: ModelRun) -> str:
    s = mrun.stats
    parts = [f"{mrun.model}: transport={mrun.transport}", f"parsed {mrun.parsed}/{mrun.attempted}",
             f"truncated {s.truncated}", f"cells written {mrun.cells_written}"]
    if mrun.resumed:
        parts.append(f"resumed {mrun.resumed}")
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
SOURCE_BATCH = "batch results"
_SHORT_ID_RE = re.compile(r"r(\d{6})")


class BatchRecordError(RuntimeError):
    """A model's batch results cannot be tied to its cells (a batch or output
    file unreadable, or short ids that cannot be mapped back): its cells with
    no truncation list in their provenance are left alone."""


@dataclass
class BatchReply:
    """One succeeded batch result, as the repair needs it."""
    text: Optional[str]
    finish: Optional[str]
    refused: bool
    n_in: int

    @property
    def cut(self) -> bool:
        return self.finish == tt.CUT_AT_CAP and not self.refused

    def legacy_score(self) -> Optional[int]:
        """The score this script recorded for the reply before 2026-11-01
        (a cut reply parsed like any other)."""
        return None if self.refused or not self.text else parse_score(self.text)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


async def _items(x: Any):
    """Items of a sync or async iterable, or of an awaitable giving one (the
    SDKs' batch results decoder)."""
    if asyncio.iscoroutine(x) or isinstance(x, asyncio.Future):
        x = await x
    if hasattr(x, "__aiter__"):
        async for item in x:
            yield item
    else:
        for item in x:
            yield item


def _content_text(content: Any) -> str:
    """The text of an OpenAI ``files.content`` response."""
    text = _get(content, "text")
    if callable(text):
        text = text()
    if text is None and hasattr(content, "read"):
        raw = content.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return text or ""


async def _anthropic_batch_replies(client: Any, batch_id: str) -> Tuple[Dict[str, BatchReply], Set[str]]:
    """``(short id -> reply for the succeeded results, every short id returned)``."""
    replies: Dict[str, BatchReply] = {}
    seen: Set[str] = set()
    async for res in _items(client.messages.batches.results(batch_id)):
        bcid = str(_get(res, "custom_id"))
        seen.add(bcid)
        result = _get(res, "result")
        if _get(result, "type") != "succeeded":
            continue
        text, finish, n_in, _n_out, refused = tt.anthropic_reply(_get(result, "message"))
        replies[bcid] = BatchReply(text, finish, refused, n_in)
    return replies, seen


async def _openai_batch_replies(client: Any, batch_id: str) -> Dict[str, BatchReply]:
    """custom id -> reply for the succeeded lines of an OpenAI batch's output file."""
    batch = await client.batches.retrieve(batch_id)
    fid = _get(batch, "output_file_id")
    if not fid:
        raise BatchRecordError(f"batch {batch_id} has no output file")
    try:
        text = _content_text(await client.files.content(fid))
    except Exception as e:  # noqa: BLE001 -- the runner deletes the file once collected
        raise BatchRecordError(f"batch {batch_id}: output file {fid} unreadable ({type(e).__name__}: {e}; "
                               f"the batch runner deletes it once the batch is collected)") from e
    replies: Dict[str, BatchReply] = {}
    for line in text.split("\n"):
        if not line.strip():
            continue
        item = json.loads(line)
        response = item.get("response") or {}
        if item.get("error") is None and response.get("status_code") == 200:
            t, finish, pt, _ct = tt.openai_reply(response.get("body") or {})
            replies[str(item.get("custom_id"))] = BatchReply(t, finish, False, pt)
    return replies


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


def _earlier_reasked(cell_data: Dict[str, Any]) -> Set[str]:
    return {n for rec in (cell_data.get("_provenance") or {}).get("repairs") or []
            for n in rec.get("reasked") or []}


def _read_cell(job: AxisJob, model: str) -> Optional[Dict[str, Any]]:
    p = cell_dir(job, model) / "scores_descriptions.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


async def read_batch_record(ctx: RunContext, mrun: ModelRun, jobs: List[AxisJob], client: Any,
                            notes: List[str]) -> Dict[int, Dict[str, BatchReply]]:
    """The model's batch results by axis index and entity, for every batch its
    ``batch_state.json`` records, checked against the cells.

    Custom ids are ``a<axis index>|<entity>``.  OpenAI results carry them; the
    Anthropic runner sent short ids (``r000123``) and maps them back through
    its chunk files, which it deletes once a batch is collected.  Without a
    chunk file the short ids are rebuilt from the order the requests were
    written in: the axes judged in batch in axis-map order, each axis's
    entities in load order, chunks in their recorded order, as a run (or a
    rerun that collected the batches in flight first and submitted the rest)
    writes them.  Either way every result must agree with the cell it maps to
    (its score as this script recorded it, a cut reply's either that or none),
    or :class:`BatchRecordError` is raised."""
    model = mrun.model
    work = batch_work_dir(ctx, model)
    state_path = work / "batch_state.json"
    if not state_path.exists():
        raise BatchRecordError(f"no {state_path}")
    chunks = [r for r in json.loads(state_path.read_text()).get("chunks", []) if r.get("batch_id")]
    if not chunks:
        raise BatchRecordError(f"{state_path} records no batch")
    if any(not r.get("collected") for r in chunks):
        raise BatchRecordError(f"{state_path} records batches not yet collected: finish the run first")
    amap = AxisMap(work / "axis_map.json")
    rec_settings = amap.settings or {}
    if rec_settings.get("model") not in (None, model):
        raise BatchRecordError(f"{work / 'axis_map.json'} was requested with {rec_settings.get('model')}, "
                               f"not {model}")
    recorded_dir = rec_settings.get("instructions_dir")
    if recorded_dir and Path(recorded_dir).resolve() != Path(ctx.instructions_dir).resolve():
        notes.append(f"the batches were requested with --instructions_dir {recorded_dir}, this run reads "
                     f"{ctx.instructions_dir}")

    # The axis map's axes: the repair's own job for an axis it names, else the map's entry.
    by_name = {(j.axis_dir, j.pos, j.neg, j.pair_type): j for j in jobs}
    axis_jobs: Dict[int, AxisJob] = {}
    for idx, entry in amap.axes.items():
        key = (entry["axis_dir"], entry["pos"], entry["neg"], entry["pair_type"])
        axis_jobs[idx] = by_name.get(key) or AxisJob.from_map_entry(entry)
    cells_data = {idx: _read_cell(job, model) for idx, job in axis_jobs.items()}
    batch_axes = sorted(idx for idx, data in cells_data.items()
                        if data is not None and (data.get("_provenance") or {}).get("transport") == "batch")
    loaded: Dict[int, AxisJob] = {}

    def entities_of(idx: int) -> List[str]:
        if idx not in loaded:
            job = axis_jobs[idx]
            loaded[idx] = job if job.entities else load_axis(job, ctx.instructions_dir)
        return [name for _, name, _ in loaded[idx].entities]

    anthropic = provider_for_model(model) == "anthropic"
    by_cid: Dict[str, BatchReply] = {}
    if anthropic:
        per_chunk = []
        for rec in chunks:
            try:
                replies, seen = await _anthropic_batch_replies(client, rec["batch_id"])
            except Exception as e:  # noqa: BLE001
                raise BatchRecordError(f"batch {rec['batch_id']}: results unreadable "
                                       f"({type(e).__name__}: {e})") from e
            per_chunk.append((rec, replies, seen))
        if all(Path(rec["path"]).exists() for rec, _, _ in per_chunk):
            how = "through the chunk files"
            for rec, replies, _seen in per_chunk:
                with open(rec["path"], encoding="utf-8") as f:
                    lines = [json.loads(x) for x in f if x.strip()]
                to_cid = {line["batch_custom_id"]: line["custom_id"] for line in lines}
                for bcid, reply in replies.items():
                    if bcid not in to_cid:
                        raise BatchRecordError(f"batch {rec['batch_id']}: result {bcid} is not in {rec['path']}")
                    by_cid[to_cid[bcid]] = reply
        else:
            how = "by rebuilding the order the requests were written in (chunk files deleted at collection)"
            sequence = [custom_id(idx, name) for idx in batch_axes for name in entities_of(idx)]
            total = sum(int(rec.get("n_requests") or 0) for rec in chunks)
            if total != len(sequence):
                raise BatchRecordError(
                    f"the chunk files are gone and the {total} requests the batches recorded are not the "
                    f"{len(sequence)} entities of the {len(batch_axes)} axes judged in batch, so the short ids "
                    f"cannot be mapped back")
            offset = 0
            for rec, replies, seen in per_chunk:
                n = int(rec.get("n_requests") or 0)
                ids = sequence[offset:offset + n]
                if Path(rec["path"]).exists():
                    with open(rec["path"], encoding="utf-8") as f:
                        if [json.loads(x)["custom_id"] for x in f if x.strip()] != ids:
                            raise BatchRecordError(f"{rec['path']} does not hold the requests the rebuilt "
                                                   f"order gives it")
                for bcid in seen:
                    m = _SHORT_ID_RE.fullmatch(bcid)
                    if m is None or int(m.group(1)) >= n:
                        raise BatchRecordError(f"batch {rec['batch_id']}: short id {bcid!r} is outside "
                                               f"r000000..r{n - 1:06d}")
                for bcid, reply in replies.items():
                    by_cid[ids[int(bcid[1:])]] = reply
                offset += n
    else:
        how = "by their custom ids"
        for rec in chunks:
            by_cid.update(await _openai_batch_replies(client, rec["batch_id"]))

    out: Dict[int, Dict[str, BatchReply]] = {}
    for cid, reply in by_cid.items():
        idx, name = split_custom_id(cid)
        out.setdefault(idx, {})[name] = reply

    # Every result must agree with the cell it maps to.
    checked = 0
    mismatches: List[str] = []
    for idx, replies in out.items():
        data = cells_data.get(idx)
        if idx not in batch_axes or data is None:
            continue
        result = data.get("result") or {}
        skip = _earlier_reasked(data)
        for name, reply in replies.items():
            if name in skip:
                continue
            checked += 1
            have = result.get(name)
            ok = have in (reply.legacy_score(), None) if reply.cut else have == reply.legacy_score()
            if not ok:
                mismatches.append(f"{axis_jobs[idx].axis_dir}/{name}: cell {have}, batch "
                                  f"{reply.legacy_score()}")
    if mismatches:
        raise BatchRecordError(f"{len(mismatches)} of {checked} batch results, mapped {how}, disagree with "
                               f"the cells' scores (e.g. {'; '.join(mismatches[:5])}): the record does not "
                               f"describe these cells")
    n_cut = sum(r.cut for replies in out.values() for r in replies.values())
    logger.info(f"[{model}] batch record: {len(by_cid)} results from {len(chunks)} batch(es), mapped {how}; "
                f"{checked} checked against the cells, all agree; {n_cut} cut at max_tokens")
    return out


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
        excluded_set=set(job.axis_spec.exclusions),
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
                                 excluded_set=set(job.axis_spec.exclusions))
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
    need_batch: List[RepairCell] = []
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
        elif prov.get("transport") == "batch":
            rc.source = SOURCE_BATCH
            need_batch.append(rc)
        else:
            notes.append(f"{job.axis_dir}: no truncation record (made "
                         f"{'by ' + prov['transport'] if prov.get('transport') else 'by an earlier version of this script'}, "
                         f"no truncation list in its provenance); left alone")

    client = None
    if need_batch or not dry_run:
        client = make_anthropic_client() if provider_for_model(model) == "anthropic" else make_openai_client(None)
    if need_batch:
        try:
            record_ = await read_batch_record(ctx, mrun, jobs, client, notes)
        except BatchRecordError as e:
            logger.error(f"[{model}] batch record unusable: {e}")
            for rc in need_batch:
                notes.append(f"{rc.job.axis_dir}: no usable truncation record ({e}); left alone")
            record_ = None
        if record_ is not None:
            amap = AxisMap(batch_work_dir(ctx, model) / "axis_map.json")
            requested = amap.settings or {}
            for rc in need_batch:
                if (requested.get("max_tokens"), requested.get("request_settings")) != (
                        rc.prov.get("max_tokens"), rc.prov.get("request_settings")):
                    notes.append(f"{rc.job.axis_dir}: the batches were requested with max_tokens "
                                 f"{requested.get('max_tokens')} and {json.dumps(requested.get('request_settings'))}, "
                                 f"the cell records {rc.prov.get('max_tokens')} and "
                                 f"{json.dumps(rc.prov.get('request_settings'))}; left alone")
                    continue
                entry = rc.job.map_entry()
                idx = next((i for i, e in amap.axes.items() if e == entry), None)
                if idx is None:
                    idx = next((i for i, e in amap.axes.items()
                                if (e["axis_dir"], e["pos"], e["neg"], e["pair_type"])
                                == (rc.job.axis_dir, rc.job.pos, rc.job.neg, rc.job.pair_type)), None)
                if idx is None or idx not in record_:
                    notes.append(f"{rc.job.axis_dir}: not in the batch record; left alone")
                    continue
                replies = record_[idx]
                rc.original = {n for n, r in replies.items() if r.cut}
                rc.est_in = {n: r.n_in for n, r in replies.items() if r.cut}
                cells.append(rc)

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
    print(f"  {len(cells)} finished cell(s) with a truncation record "
          f"({sum(rc.source == SOURCE_PROVENANCE for rc in cells)} from their provenance, "
          f"{sum(rc.source == SOURCE_BATCH for rc in cells)} from batch results), "
          f"{sum(len(rc.original) for rc in cells)} replies cut in their own runs")
    for rc in work_cells:
        print(f"  {rc.job.axis_dir:55s} {len(rc.todo):4d}  (source: {rc.source}; cell cap "
              f"{rc.prov.get('max_tokens')})  {', '.join(rc.todo[:8])}{' ...' if len(rc.todo) > 8 else ''}")
    for note in notes:
        print(f"  note: {note}")
    if dry_run or not work_cells:
        return stats

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
        if this_call.n_calls == 0:      # no answer (a failed call): left as it was
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
        print("Dry run only -- no judge calls made, nothing written (batch results were read).")
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
                        help="Repeatable. Any model axis_judge_correlation.py itself accepts "
                             "(gpt-/o1-/o3-/o4- -> OpenAI, claude- -> Anthropic).")
    parser.add_argument("--transport", choices=list(tt.TRANSPORT_CHOICES), default="auto",
                        help="auto (default): Anthropic -> batch; OpenAI -> flex when MODEL_CAPABILITIES says "
                             "OpenAI takes Flex for the model, batch when it says OpenAI refuses it, flex with "
                             "a probe when untested.  flex / batch / realtime force one for every model.")
    parser.add_argument("--request_setting", action="append", default=[], metavar="[MODEL:]KEY=JSON",
                        help="Repeatable.  A request field set on top of judge_tier_transport's "
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
                             "cut entities come from the cell's provenance 'truncated' list, or, for a batch "
                             "cell made before 2026-11-01, from its batches' results "
                             f"({BATCH_DIRNAME}/<model><label>/batch_state.json); a cell with neither is left "
                             "alone.  With --dry_run: report the cells, counts and estimated cost (batch results "
                             "are read; no judge calls).")
    return parser


def build_jobs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> List[AxisJob]:
    single = {"--pair": args.pair, "--pair_type": args.pair_type, "--projections_file": args.projections_file,
              "--output_dir": args.output_dir}
    if args.axes_file:
        given = [k for k, v in single.items() if v is not None]
        if given:
            parser.error(f"--axes_file excludes {', '.join(given)}")
        return jobs_from_axes_file(Path(args.axes_file), Path(args.experiments_dir))
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
        est = n_calls * ((EST_IN_PER_ENTITY + 150 * (per_call - 1)) * rate_in
                         + EST_OUT_PER_ENTITY * per_call * rate_out) / 1e6
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
    print(f"\nTotal estimated cost: ${total:.2f}  (token model: {EST_IN_PER_ENTITY} in / {EST_OUT_PER_ENTITY} "
          f"out per entity, coarse)")
    print("Dry run only -- no API calls made, no cost incurred, nothing written.")


async def main_async(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    global CELL_LABEL, RUBRIC
    CELL_LABEL = args.cell_label or ""
    RUBRIC = jr.get(args.rubric)
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
                f"(description mode)")

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
    return 0


def main() -> None:
    code = asyncio.run(main_async())
    if code == EXIT_STALLED:
        restart_after_stall()   # returns only when no restart is left
    sys.exit(code)


if __name__ == "__main__":
    main()
