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

Resume rule: an axis whose <output_dir>/<model>/usage.json exists is skipped
for that model (in both modes), so a rerun only judges what is missing.

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

Drop --dry_run to actually run it.  scripts/run_tier_eval.sh wraps the
axes-file form with the corpus check.  Each --judge_model may be repeated;
models run concurrently.  Exit status: 0 done, 1 a model failed (or a key is
missing), 2 --budget_usd stopped the run.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
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
from assistant_axis.judge import RateLimiter, provider_for_model, warn_if_low_parse_rate  # noqa: E402
from assistant_axis.judge_pricing import BATCH_SUFFIX, MultiModelUsage, price_for_model  # noqa: E402
from assistant_axis.entity_id import parse_entity_id  # noqa: E402
from assistant_axis.atomic_io import atomic_write_text  # noqa: E402
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


def cell_dir(job: AxisJob, model: str) -> Path:
    return Path(job.output_dir) / model


def cell_done(job: AxisJob, model: str) -> bool:
    """The resume rule: a model's cell counts as done once its usage.json exists."""
    return (cell_dir(job, model) / "usage.json").exists()


# ---------------------------------------------------------------------------
# Cells and per-model run state
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One (axis, model) cell being filled: every attempted entity's score (None
    when unparseable or failed) and the cell's own usage."""
    idx: int
    job: AxisJob
    expected: Set[str]
    scores: Dict[str, Optional[int]] = field(default_factory=dict)
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
    error: Optional[BaseException] = None

    def __post_init__(self) -> None:
        self.transport = self.transport or self.choice.transport
        self.transport_reason = self.transport_reason or self.choice.reason


class _Charge:
    """Charges one call to several trackers (the cell's, the model's, the run's)."""

    def __init__(self, *trackers: MultiModelUsage):
        self.trackers = trackers

    def charge(self, key: str, prompt_tokens: int, completion_tokens: int) -> None:
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

    def over_budget(self) -> bool:
        if self.budget_usd is None or self.run_usage.total_cost_usd <= self.budget_usd:
            return False
        if not self.budget_hit:
            self.budget_hit = True
            logger.error(f"--budget_usd {self.budget_usd:.2f} reached (${self.run_usage.total_cost_usd:.2f} "
                         f"spent this run): no further requests are issued; requests in flight finish")
        return True


def record(mrun: ModelRun, cell: Cell, name: str, text: Optional[str]) -> None:
    score = parse_signed_score(text) if text else None
    cell.scores[name] = score
    mrun.attempted += 1
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
    provenance = {"judge_model": model, "rubric_version": RUBRIC_VERSION,
                  "projections_file": str(job.projections_file),
                  "transport": mrun.transport, "transport_reason": mrun.transport_reason,
                  "request_settings": mrun.settings, "max_tokens": max_tokens}
    atomic_write_text(json.dumps({"result": scores, "_provenance": provenance}, indent=2),
                      d / "scores_descriptions.json", logger_obj=logger)
    correlations = compute_correlations(
        scores_by_mode={"descriptions": scores},
        projections=job.projections,
        slots=list(job.projections.keys()),
        excluded_set=set(job.axis_spec.exclusions),
    )
    atomic_write_text(json.dumps(correlations, indent=2), d / "correlations.json", logger_obj=logger)
    atomic_write_text(json.dumps(cell.usage.as_dict(), indent=2, sort_keys=True), d / "usage.json",
                      logger_obj=logger)
    cell.written = True
    mrun.cells_written += 1
    logger.info(f"[{model}] {job.axis_dir}: scored {len(scores)}/{len(cell.expected)}; "
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
    provenance = {"judge_model": model, "rubric_version": RUBRIC_VERSION,
                  "projections_file": str(job.projections_file), "transport": mrun.transport,
                  "request_settings": mrun.settings, "max_tokens": max_tokens, "complete": False,
                  "n_attempted": len(cell.scores), "n_entities": len(cell.expected)}
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
    all axes at once through the rate limiter, writing each cell as soon as
    its axis is complete.  With ``probe``, the first request goes alone and a
    service-tier refusal returns False before anything else is sent."""
    model = mrun.model
    rate_limiter = RateLimiter(ctx.requests_per_second)
    if provider_for_model(model) == "anthropic":
        client, fallback = make_anthropic_client(), None
    else:
        client = make_openai_client(tier)
        fallback = make_openai_client(None) if tier == "flex" else None
    cells = {i: Cell(i, job, expected={name for _, name, _ in job.entities}) for i, job in enumerate(todo)}
    items = [(i, etype, name, desc) for i, job in enumerate(todo) for etype, name, desc in job.entities]
    fatal: List[BaseException] = []

    def gate() -> bool:
        return bool(fatal) or ctx.over_budget()

    async def one(item: Tuple[int, str, str, str], *, probing: bool = False) -> None:
        i, etype, name, desc = item
        cell = cells[i]
        prompt = build_static_prompt(cell.job.axis_spec, etype, name, desc)
        try:
            text = await tt.call_model(model, prompt, ctx.max_tokens, rate_limiter,
                                       _Charge(cell.usage, mrun.usage, ctx.run_usage), client=client,
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
        if text is tt.SKIPPED:
            return
        record(mrun, cell, name, text)
        if cell.complete and not cell.written:
            write_cell(cell, mrun, ctx.max_tokens)

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
            logger.info(f"[{model}] probe: Flex accepted; add {model!r}: {{'flex': True}} to MODEL_CAPABILITIES "
                        f"in results_analysis/judge_tier_transport.py")
    await asyncio.gather(*(one(it) for it in items))
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
    return ctx.batch_root / model


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
    return {"model": mrun.model, "max_tokens": ctx.max_tokens, "request_settings": mrun.settings,
            "rubric_version": RUBRIC_VERSION, "instructions_dir": str(Path(ctx.instructions_dir).resolve())}


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

    def new_cell(idx: int, job: AxisJob) -> Cell:
        cell = Cell(idx, job, expected={name for _, name, _ in job.entities})
        p = collected_path(idx)
        if p.exists():
            data = json.loads(p.read_text())
            cell.scores = {k: (None if v is None else int(v)) for k, v in data.get("scores", {}).items()}
            cell.usage = MultiModelUsage.from_dict(data.get("usage") or {})
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

    def on_result(cid: str, body: Optional[dict], error: Optional[str]) -> None:
        if body is None:
            return   # unanswered: comes back from runner.run and is sent in real time
        idx, name = split_custom_id(cid)
        if anthropic:
            text, _finish, n_in, n_out, refused = tt.anthropic_reply(body)
            if refused:
                details = body.get("stop_details") if isinstance(body, dict) else None
                mrun.stats.note_refusal(model, (details or {}).get("category")
                                        if isinstance(details, dict) else None)
                text = None
        else:
            text, _finish, n_in, n_out = tt.openai_reply(body)
        cell = cell_for(idx)
        if cell is None:
            mrun.usage.charge(batch_key, n_in, n_out)
            ctx.run_usage.charge(batch_key, n_in, n_out)
            logger.warning(f"[{model}] result {cid!r} is for an axis already done or unknown; dropped "
                           f"(its tokens are in this run's total only)")
            return
        _Charge(cell.usage, mrun.usage, ctx.run_usage).charge(batch_key, n_in, n_out)
        record(mrun, cell, name, text)
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
                                              "scores": cell.scores, "usage": cell.usage.as_dict()},
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
        for idx in current:
            cell = cells[idx]
            if cell.written:
                continue
            for etype, name, desc in cell.job.entities:
                if name in cell.scores:
                    continue
                cid = custom_id(idx, name)
                if cid in unanswered or cid in inflight:
                    continue
                yield cid, idx, etype, name, desc

    n_pending = sum(1 for _ in pending())
    if n_pending:
        amap.settings = settings_now
        amap.save()   # before the submission, so a rerun can route what it collects
        chunks = runner.write_chunks(
            (cid, tt.request_for(model, build_static_prompt(cells[idx].job.axis_spec, etype, name, desc),
                                 ctx.max_tokens, mrun.settings))
            for cid, idx, etype, name, desc in pending())
        logger.info(f"[{model}] batch: {n_pending} requests in {len(chunks)} chunk(s) for "
                    f"{sum(1 for i in current if not cells[i].written)} axes")
        unanswered.update(await runner.run(chunks, on_result, flush))
    elif not inflight:
        logger.info(f"[{model}] batch: nothing to submit")

    # 3. What no batch answered goes in real time, on the standard tier.
    if unanswered and len(unanswered) <= ctx.batch_fallback_max:
        logger.info(f"[{model}] judging {len(unanswered)} requests no batch answered in real time")
        rate_limiter = RateLimiter(ctx.requests_per_second)

        async def one(cid: str, request: Dict[str, Any]) -> None:
            idx, name = split_custom_id(cid)
            cell = cell_for(idx)
            if cell is None or cell.written:
                return
            text = await tt.call_model(model, tt.prompt_of_request(request), ctx.max_tokens, rate_limiter,
                                       _Charge(cell.usage, mrun.usage, ctx.run_usage), client=client,
                                       settings=mrun.settings, stats=mrun.stats, gate=ctx.over_budget)
            if text is tt.SKIPPED:
                return
            record(mrun, cell, name, text)
            touched.add(idx)

        await asyncio.gather(*(one(cid, req) for cid, req in sorted(unanswered.items())))
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
             f"cells written {mrun.cells_written}"]
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
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--requests_per_second", type=int, default=10)
    parser.add_argument("--budget_usd", type=float, default=None,
                        help="Real-time and Flex calls (a batch's real-time remainder included): stop issuing "
                             "new requests once this run's measured spend passes this.  A batch is submitted "
                             "whole; --dry_run is its control.  Cells left incomplete are written as "
                             "*.partial.json (no usage.json), and the run exits 2.")
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
        n_calls = sum(len(job.entities) for job in todo)
        rate_in, rate_out = price_for_model(tt.price_key(model, choice.transport))
        est = n_calls * (EST_IN_PER_ENTITY * rate_in + EST_OUT_PER_ENTITY * rate_out) / 1e6
        total += est
        print(f"{model:22s} provider={provider_for_model(model):9s} transport={choice.transport}"
              f"{' (probe)' if choice.probe else '':8s} rate=${rate_in:.3f}/${rate_out:.3f} per 1M  "
              f"axes {len(todo)}/{len(jobs)}  ~{n_calls:,} calls  est.cost=${est:.2f}")
        print(f"{'':22s} why: {choice.reason}")
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
    jobs = build_jobs(args, parser)
    instructions_dir = Path(args.instructions_dir)
    try:
        overrides = [tt.parse_request_setting(s) for s in args.request_setting]
    except ValueError as e:
        parser.error(str(e))

    runs: List[ModelRun] = []
    for model in dict.fromkeys(args.judge_model):
        try:
            choice = tt.choose_transport(model, args.transport)
        except (tt.JudgeRequestError, ValueError) as e:
            parser.error(str(e))
        runs.append(ModelRun(model=model, settings=tt.judge_request_settings(model, overrides), choice=choice))

    skipped_axes = [job for job in jobs if not Path(job.projections_file).exists()]
    for job in skipped_axes:
        logger.warning(f"SKIP (no projections.json): {job.axis_dir} ({job.projections_file})")
    jobs = [job for job in jobs if Path(job.projections_file).exists()]
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
                     batch_fallback_max=args.batch_fallback_max, slots=args.slots, limit=args.limit)
    if args.budget_usd is not None and any(m.choice.transport == "batch" for m in runs):
        logger.info("--budget_usd does not stop a batch submission (it is submitted whole); it applies to "
                    "real-time and Flex calls, a batch's real-time remainder included")
    results = await asyncio.gather(*(run_model(ctx, mrun, jobs) for mrun in runs), return_exceptions=True)
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
    if any(m.error for m in runs):
        return 1
    if ctx.budget_hit:
        return 2
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
