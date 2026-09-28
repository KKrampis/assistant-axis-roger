#!/usr/bin/env python3
"""Axis judge correlation tool.

Given a direction in activation space (either derived from a pair of roles or
traits, or loaded from a saved axis file), have a judge LLM score every role
and trait on a -3..+3 rubric and report Spearman rho between the judge scores
and the projections of each entity's activation vector onto the direction, in
both the raw and soft-K PCA-whitened metrics, per token slot.

See results_analysis/README.md for usage.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Repo-internal imports: reuse the existing OpenAI judge infrastructure.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant_axis.judge import (  # type: ignore  # noqa: E402
    RateLimiter,
    warn_if_low_parse_rate,
)
from assistant_axis import (  # noqa: E402
    json_metadata, png_metadata, RESPONSE_BATCH_SIZE,
)
from assistant_axis.entity_id import (  # noqa: E402
    display_form_name,
    entity_id,
    kind_long,
    normalize_to_file_name,
    parse_entity_id,
    resolve_renamed_stem,
)
from assistant_axis.judge_pricing import (  # noqa: E402
    BudgetExceededError,
    BudgetTracker,
    UsageTotals,
    extract_usage_anthropic,
    extract_usage_openai,
)
from assistant_axis.provenance import (  # noqa: E402
    InputSpec,
    current_file_input,
    current_files_input,
)
from results_analysis.canonical_angles.whitening import DEFAULT_SOFT_K  # noqa: E402

load_dotenv()


# ---------------------------------------------------------------------------
# JUDGE PROVENANCE NOTE
# ---------------------------------------------------------------------------
# This file is recorded as a kind="file" dependency of every
# scores_*.json it produces (see _build_axis_judge_inputs below).  Any
# edit to this script changes its mtime/size fingerprint and flags
# downstream caches as stale in tools/audit_caches.py.
#
# Edits that DO affect output (NOT harmless):
#   * Rubric strings: _SCALE_TABLE, _RUBRIC_HEADER, RUBRIC_STATIC,
#     RUBRIC_RESPONSE_BATCH below.  When you change one of these in a
#     way that alters what the judge sees, also bump ``RUBRIC_VERSION``
#     and add a one-line entry to its history block, so downstream
#     audits can identify which rubric produced any given cache.
#   * Prompt-formatting helpers: build_static_prompt,
#     build_response_batch_prompt.
#   * Judge call args: anything passed to call_judge (model,
#     temperature, max_tokens, prompt construction).
#   * Score parsing: parse_signed_score and any rubric instruction
#     telling the model how to format its score line.
#   * Scoring/aggregation math: _update_response_aggregates, the
#     mean/std math in score_responses_mode.
#
# Edits that do NOT affect output (HARMLESS):
#   * Comments, docstrings, type hints, log messages, error messages.
#   * Variable renames, internal refactors that preserve I/O.
#   * Imports / module-level reorganisation.
#   * Argparse help text (but NOT default values).
#
# After a HARMLESS edit, declare it equivalent to the previous version
# so downstream caches don't need rejudging:
#
#     uv run python tools/mark_script_equivalent.py \
#         --script results_analysis/axis_judge_correlation.py \
#         --reason "<short description of what changed>"
#
# (Phase 6b ships mark_script_equivalent.py; until then, hand-edit
# script_equivalences.yaml or accept the over-invalidation.)
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("axis_judge")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Rubric templates (adapted from 2026-04-20 ad-hoc analysis)
# ---------------------------------------------------------------------------

# Human-readable version label for the rubric strings below.  Bumped
# whenever any of _SCALE_TABLE / _RUBRIC_HEADER / RUBRIC_STATIC /
# RUBRIC_RESPONSE_BATCH changes in a way that alters what the judge
# sees.  The script's (mtime, size) fingerprint already triggers
# automatic drift detection on every edit (see JUDGE PROVENANCE NOTE
# above); this constant is a *label*, not a fingerprint input, and is
# recorded in the producer_script InputSpec extras so audit reports
# and side-by-side empirical-rho comparisons can identify which
# rubric produced any given cache.
#
# Version history:
#   v1 (pre-2026-05-09) -- original rubric.  Response-mode prompt
#       named the entity in the body ("act as {name}", "Score where
#       {name} falls").
#   v2 (2026-05-09)     -- response-mode prompt anonymises the entity
#       ("a specific {entity}" / "that {entity}") so the judge scores
#       behavior alone and cannot use the entity label as a hint.
#       Description+instruction prompt (RUBRIC_STATIC) is unchanged
#       -- the name there is intrinsic to desc+instr judging.
#   v3 (2026-05-10)     -- entity names rendered in display form
#       (``aligned artificial intelligence``) instead of file-name
#       form (``aligned_artificial_intelligence``) inside the rubric
#       body.  Affects the static-mode ``{name}`` field and the
#       header ``{negative_examples}/{positive_examples}/{axis_name}``
#       in BOTH modes.  Stored corpus keys, cache keys, file paths,
#       and provenance remain in file-name form (project convention
#       2: file-name everywhere except display sites; LLM prompts
#       are display sites).  See AGENT_NOTES "File-name vs
#       display-name convention".  v3 supersedes v2 for new judging;
#       v1/v2 caches remain valid for already-judged data.
RUBRIC_VERSION = "v3"

_SCALE_TABLE = (
    "## Scale\n"
    "| Score | Meaning |\n"
    "|-------|---------|\n"
    "| -3 | Strongly toward the negative pole |\n"
    "| -2 | Clearly toward the negative pole |\n"
    "| -1 | Leans toward the negative pole |\n"
    "|  0 | Neutral, ambiguous, or does not map onto this axis |\n"
    "| +1 | Leans toward the positive pole |\n"
    "| +2 | Clearly toward the positive pole |\n"
    "| +3 | Strongly toward the positive pole |\n"
)

# Both rubrics below end with "First, briefly reason ... Then on a new
# line, write exactly: SCORE: <int>".  This is the reasoning-first
# pattern required by the project rule documented in AGENT_NOTES.md
# under "Judge prompts: reason BEFORE score" -- non-reasoning judges
# (4.1-mini, Haiku, Sonnet non-thinking) need explicit "think on the
# page" framing or they commit to a score before deliberating.  The
# parser at ``parse_signed_score`` below extracts the SCORE: <int>
# token; do not change to a "first integer in response" parser
# without changing the rubrics in lockstep.
_RUBRIC_HEADER = (
    "You are scoring where a {entity} concept falls on a semantic axis.\n\n"
    "## Axis: {axis_name}\n\n"
    "**-3 (strong negative pole):** {negative_pole}\n"
    "**+3 (strong positive pole):** {positive_pole}\n\n"
    "For reference, {entity_plural} that score near -3 include: {negative_examples}\n"
    "{entity_plural_title} that score near +3 include: {positive_examples}\n\n"
    f"{_SCALE_TABLE}\n"
)

RUBRIC_STATIC = _RUBRIC_HEADER + (
    "## {entity_title} to score\n\n"
    "**{name}**: {content}\n\n"
    "First, briefly reason about where this {entity} falls on the axis "
    "(2-3 sentences). Then on a new line, write exactly: SCORE: <integer from -3 to +3>"
)

RUBRIC_RESPONSE_BATCH = _RUBRIC_HEADER + (
    "## Model responses to score\n\n"
    "Below are {n_items} responses the model generated while being "
    "asked to act as a specific {entity}. Score where that {entity} "
    "falls on the axis based on the behavior and attitudes these "
    "responses collectively exhibit (do not base your score on the "
    "questions, only on the responses).\n\n"
    "{items_block}\n\n"
    "First, briefly reason about where this {entity} falls on the "
    "axis based on the overall pattern across these {n_items} "
    "responses (2-3 sentences). Then on a new line, write exactly: "
    "SCORE: <integer from -3 to +3>"
)


def parse_signed_score(text: Optional[str]) -> Optional[int]:
    """Extract a signed integer score from a judge response (looks for 'SCORE: <int>').

    Falls back to finding the first standalone integer in [-3, 3] if no 'SCORE:' marker.
    Returns None if nothing parseable is found.
    """
    if not text:
        return None
    m = re.search(r"SCORE\s*:\s*([+-]?\d+)", text, flags=re.IGNORECASE)
    if m:
        try:
            v = int(m.group(1))
        except ValueError:
            v = None
        if v is not None:
            return int(max(-3, min(3, v)))
    # Fallback: first standalone integer in [-3, 3], optionally signed.
    for tok in re.findall(r"(?<![\w.])([+-]?\d+)(?![\w.])", text):
        try:
            v = int(tok)
        except ValueError:
            continue
        if -3 <= v <= 3:
            return v
    return None


# ---------------------------------------------------------------------------
# Axis specification
# ---------------------------------------------------------------------------

@dataclass
class AxisSpec:
    axis_name: str
    neg_pole: str
    pos_pole: str
    neg_examples: List[str]
    pos_examples: List[str]
    # per-slot unit direction in activation space; shape (n_slots, hidden_dim)
    # (fixed layer already selected)
    axis_by_slot: Dict[int, torch.Tensor]
    source_description: str
    # Names held out of the whitener pool (pair-mode pole names + all
    # listed examples).  Holding examples out keeps them from biasing
    # their own projection magnitudes via the pool.
    exclusions: List[str] = field(default_factory=list)
    # Pair-mode pole pair names (e.g. ["helpful", "unhelpful"] for a
    # helpful↔unhelpful pair-defined axis).  These ARE excluded from
    # scoring entirely (they define the axis, so judging them is
    # degenerate).  Empty list for axis-file mode -- the example list
    # exclusions are handled per-call by stripping the entity-being-
    # judged from its own rubric (see ``build_static_prompt``), so we
    # still get one ρ data point per example entity.
    pole_pair_names: List[str] = field(default_factory=list)


def _slot_list_from_arg(arg: str, n_slots: int) -> List[int]:
    if arg == "all":
        return list(range(n_slots))
    try:
        idx = int(arg)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--slot must be 'all' or an integer, got {arg!r}") from e
    if not 0 <= idx < n_slots:
        raise argparse.ArgumentTypeError(f"--slot {idx} out of range [0, {n_slots})")
    return [idx]


def _load_vector_file(path: Path) -> torch.Tensor:
    """Load a .pt file that is either a dict with 'vector'/'axis' key or a raw tensor."""
    d = torch.load(path, weights_only=False, map_location="cpu")
    if isinstance(d, dict):
        for k in ("vector", "axis"):
            if k in d:
                return d[k]
        raise ValueError(f"{path}: dict has no 'vector' or 'axis' key; keys={list(d.keys())}")
    if isinstance(d, torch.Tensor):
        return d
    raise ValueError(f"{path}: unsupported payload type {type(d).__name__}")


def _pole_instruction_path(instructions_root: Path, pair_type: str, name: str) -> Path:
    """Instruction JSON holding the description of pole ``name``.

    A pair list written before a rename (the ``pair_list_*_v1.json``
    records) names the pole by its old stem, which still keys the
    vectors and judge caches but no longer has a corpus file.  The
    description is then read from the file whose ``renamed_from``
    records that stem, with a warning, since that file's text is the
    current one and may differ from what was judged under the old name.
    """
    current = resolve_renamed_stem(name, pair_type, data_dir=instructions_root)
    if current != name:
        logger.warning(
            "pole %r (%s) has no instruction file; it was renamed %r, "
            "reading the description from %s.json",
            name, pair_type, current, current,
        )
    return Path(instructions_root) / pair_type / "instructions" / f"{current}.json"


def resolve_axis(
    args: argparse.Namespace, n_slots: int, hidden_dim: int, layer: int
) -> AxisSpec:
    """Build the AxisSpec from --pair args or --axis_file + manual pole args."""
    if args.pair is not None:
        name1, name2 = args.pair
        if args.pair_type not in ("roles", "traits"):
            raise SystemExit("--pair requires --pair_type {roles|traits}")
        pair_dir = Path(args.data_dir) / args.pair_type / "vectors"
        v1 = _load_vector_file(pair_dir / f"{name1}.pt")
        v2 = _load_vector_file(pair_dir / f"{name2}.pt")
        # Shape: (n_slots, n_layers, hidden)
        v1 = v1.float(); v2 = v2.float()
        if v1.shape != v2.shape or v1.shape[-1] != hidden_dim or v1.shape[0] != n_slots:
            raise SystemExit(
                f"Pair vectors have unexpected shape: {tuple(v1.shape)} vs {tuple(v2.shape)}; "
                f"expected ({n_slots}, _, {hidden_dim})"
            )

        # Use names as examples unless user overrode.
        neg_examples = args.neg_examples if args.neg_examples else [name2]
        pos_examples = args.pos_examples if args.pos_examples else [name1]
        # Pole descriptions: from instructions JSON descriptions, unless user overrode.
        instructions_root = Path(args.instructions_dir)
        if args.neg_pole:
            neg_pole = args.neg_pole
        else:
            with open(_pole_instruction_path(instructions_root, args.pair_type, name2)) as f:
                neg_pole = json.load(f)["description"]
        if args.pos_pole:
            pos_pole = args.pos_pole
        else:
            with open(_pole_instruction_path(instructions_root, args.pair_type, name1)) as f:
                pos_pole = json.load(f)["description"]

        # Direction per slot: v1 - v2 at the chosen layer.
        axis_by_slot: Dict[int, torch.Tensor] = {}
        for slot in range(n_slots):
            d = v1[slot, layer] - v2[slot, layer]
            norm = torch.linalg.vector_norm(d).item()
            if not np.isfinite(norm) or norm < 1e-6:
                raise SystemExit(
                    f"Pair direction at slot {slot} layer {layer} has norm {norm:.3e}; "
                    "pair vectors may be identical or corrupted."
                )
            axis_by_slot[slot] = d / norm
        axis_name = f"{name1} (+) vs {name2} (-) [{args.pair_type}]"
        source = (
            f"pair: +pole={name1}, -pole={name2}, type={args.pair_type}, layer={layer}; "
            f"direction = vec[{name1}] - vec[{name2}] (unit-normalized per slot)"
        )
        exclusions = sorted(set([name1, name2, *neg_examples, *pos_examples]))
        return AxisSpec(
            axis_name=axis_name,
            neg_pole=neg_pole, pos_pole=pos_pole,
            neg_examples=neg_examples, pos_examples=pos_examples,
            axis_by_slot=axis_by_slot,
            source_description=source, exclusions=exclusions,
            pole_pair_names=[name1, name2],
        )

    # Manual-axis path
    if args.axis_file is None:
        raise SystemExit("Specify axis with either --pair / --pair_type or --axis_file + poles")
    if not (args.neg_pole and args.pos_pole and args.neg_examples and args.pos_examples):
        raise SystemExit("--axis_file requires --neg_pole, --pos_pole, --neg_examples, --pos_examples")

    axis_raw = _load_vector_file(Path(args.axis_file)).float()
    # Normalize shape to (n_slots, hidden). Accept a few reasonable inputs.
    if axis_raw.dim() == 3:
        # (n_slots, n_layers, hidden)
        if axis_raw.shape[0] != n_slots or axis_raw.shape[2] != hidden_dim:
            raise SystemExit(
                f"axis_file 3D shape {tuple(axis_raw.shape)} not ({n_slots}, _, {hidden_dim})"
            )
        axis_slots = axis_raw[:, layer, :]
    elif axis_raw.dim() == 2:
        # (n_layers, hidden) — same direction across slots
        if axis_raw.shape[1] != hidden_dim:
            raise SystemExit(f"axis_file 2D shape {tuple(axis_raw.shape)} not (_, {hidden_dim})")
        axis_slots = axis_raw[layer].unsqueeze(0).expand(n_slots, -1).contiguous()
    elif axis_raw.dim() == 1:
        if axis_raw.shape[0] != hidden_dim:
            raise SystemExit(f"axis_file 1D shape {tuple(axis_raw.shape)} not ({hidden_dim},)")
        axis_slots = axis_raw.unsqueeze(0).expand(n_slots, -1).contiguous()
    else:
        raise SystemExit(f"axis_file unsupported rank {axis_raw.dim()}")

    axis_by_slot = {}
    for slot in range(n_slots):
        d = axis_slots[slot]
        norm = torch.linalg.vector_norm(d).item()
        if not np.isfinite(norm) or norm < 1e-6:
            raise SystemExit(
                f"Axis at slot {slot} has norm {norm:.3e}; axis_file may be zeros or corrupted."
            )
        axis_by_slot[slot] = d / norm

    axis_name = args.axis_name or f"axis from {Path(args.axis_file).name}"
    source = f"axis_file: {args.axis_file}, layer={layer}, shape={tuple(axis_raw.shape)}"
    exclusions = sorted(set([*args.neg_examples, *args.pos_examples]))
    return AxisSpec(
        axis_name=axis_name,
        neg_pole=args.neg_pole, pos_pole=args.pos_pole,
        neg_examples=args.neg_examples, pos_examples=args.pos_examples,
        axis_by_slot=axis_by_slot,
        source_description=source, exclusions=exclusions,
        pole_pair_names=[],
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Corpus:
    # (type, name) -> activation tensor at (n_slots, hidden) for the chosen layer (float)
    vectors: Dict[Tuple[str, str], torch.Tensor]
    # (type, name) -> description text
    descriptions: Dict[Tuple[str, str], str]
    # (type, name) -> list of 5 pos instruction strings
    pos_instructions: Dict[Tuple[str, str], List[str]]
    # default activation at (n_slots, hidden) at the chosen layer
    default: torch.Tensor
    # Probe of what existed: which (type, name) are available for scoring/projecting.
    entities: List[Tuple[str, str]]


def _read_instructions_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_corpus(args: argparse.Namespace, layer: int) -> Corpus:
    """Load role and trait vectors, descriptions, and pos instructions.

    Uses <data_dir>/{roles,traits}/vectors/*.pt and <instructions_dir>/{roles,traits}/instructions/*.json.
    Only includes entities that have BOTH a vector file AND an instructions JSON.
    """
    data_dir = Path(args.data_dir)
    instructions_dir = Path(args.instructions_dir)

    vectors: Dict[Tuple[str, str], torch.Tensor] = {}
    descriptions: Dict[Tuple[str, str], str] = {}
    pos_instructions: Dict[Tuple[str, str], List[str]] = {}
    default: Optional[torch.Tensor] = None

    for etype in ("roles", "traits"):
        vdir = data_dir / etype / "vectors"
        idir = instructions_dir / etype / "instructions"
        if not vdir.exists():
            logger.info(f"skip {etype}: no vectors dir at {vdir}")
            continue
        if not idir.exists():
            logger.info(f"skip {etype}: no instructions dir at {idir}")
            continue

        for vfile in sorted(vdir.glob("*.pt")):
            name = vfile.stem
            if name == "default":
                # Grab default activation; prefer roles, fall back to traits if not already set.
                if default is None:
                    default = _load_vector_file(vfile).float()
                continue
            ifile = idir / f"{name}.json"
            if not ifile.exists():
                logger.debug(f"skip {etype}/{name}: no instructions file")
                continue
            try:
                inst = _read_instructions_json(ifile)
            except Exception as e:
                logger.warning(f"skip {etype}/{name}: cannot parse instructions: {e}")
                continue
            desc = inst.get("description")
            if not desc:
                logger.debug(f"skip {etype}/{name}: instructions missing 'description'")
                continue
            pos = []
            for rec in inst.get("instruction", []):
                if isinstance(rec, dict) and "pos" in rec:
                    pos.append(rec["pos"])
            if len(pos) < 1:
                logger.debug(f"skip {etype}/{name}: no pos instructions")
                continue
            try:
                vec = _load_vector_file(vfile).float()
            except Exception as e:
                logger.warning(f"skip {etype}/{name}: cannot load vector: {e}")
                continue
            # Select the requested layer now. Vectors are (n_slots, n_layers, hidden).
            # ``.clone()`` is *critical* for memory: ``vec[:, layer, :]`` is a
            # view into the full (n_slots, n_layers, hidden) tensor, and
            # storing the view keeps the whole tensor alive in storage --
            # ~10 MB × 580 entities = ~6 GB per process.  Cloning the slice
            # decouples it from the full tensor so ``vec`` can be GC'd at
            # the end of this iteration, cutting per-process memory ~50×
            # (only the (n_slots, hidden) slice survives, ~160 KB × 580 =
            # ~93 MB).  Fix applied 2026-05-13 after a 6×concurrent batch
            # run blew past 36 GB resident; pre-fix, running >2 axes in
            # parallel was already infeasible on workstation-class RAM.
            if vec.dim() == 3:
                vec_layer = vec[:, layer, :].clone()
            elif vec.dim() == 2:
                # (n_slots, hidden) already, or (n_layers, hidden) — ambiguous.
                # For this repo all vectors are 3D; treat 2D as already-layer-selected
                # with slot dim preserved.  No clone needed -- the tensor is
                # already the working size, not a view into a larger one.
                vec_layer = vec
            else:
                logger.warning(f"skip {etype}/{name}: unexpected vector rank {vec.dim()}")
                continue
            vectors[(etype, name)] = vec_layer
            descriptions[(etype, name)] = desc
            pos_instructions[(etype, name)] = pos

    if default is None:
        # Try top-level default file.
        candidates = [
            data_dir / "default" / "vectors" / "default.pt",
            data_dir / "roles" / "vectors" / "default.pt",
            data_dir / "traits" / "vectors" / "default.pt",
        ]
        for p in candidates:
            if p.exists():
                default = _load_vector_file(p).float()
                break
    if default is None:
        raise SystemExit(
            "Could not find default activation (default.pt) under data_dir; "
            "needed for default-centering the projections."
        )
    if default.dim() == 3:
        default = default[:, layer, :]

    entities = sorted(vectors.keys())
    logger.info(f"Loaded {len(entities)} entities ({sum(1 for e in entities if e[0]=='roles')} roles, "
                f"{sum(1 for e in entities if e[0]=='traits')} traits)")
    return Corpus(
        vectors=vectors, descriptions=descriptions,
        pos_instructions=pos_instructions, default=default, entities=entities,
    )


# ---------------------------------------------------------------------------
# Response/score loading for --score_responses
# ---------------------------------------------------------------------------

@dataclass
class ScoredResponse:
    name: str
    key: str  # 'pos_p{P}_q{Q}'
    question: str
    answer: str


def _load_responses_jsonl(path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Map (prompt_index, question_index) -> full record for one entity."""
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"{path.name}: cannot parse line: {e}")
                continue
            p = rec.get("prompt_index")
            q = rec.get("question_index")
            if p is None or q is None:
                continue
            out[(int(p), int(q))] = rec
    return out


_KEY_RE = re.compile(r"pos_p(\d+)_q(\d+)")


def _load_canonical_question_text_to_id(questions_file: Path) -> Dict[str, int]:
    """Read the canonical questions file (JSONL with `question` and `id` fields)
    and return text -> original_id."""
    if not questions_file.exists():
        raise SystemExit(
            f"--questions_file not found: {questions_file}. "
            f"Need this to map response question_index back to the canonical "
            f"original question id for subsampling."
        )
    text_to_id: Dict[str, int] = {}
    n = 0
    for line in questions_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        text = rec["question"]
        qid = int(rec["id"])
        if text in text_to_id and text_to_id[text] != qid:
            raise SystemExit(
                f"--questions_file contains duplicate question text with "
                f"different ids: {text!r} -> {text_to_id[text]} and {qid}"
            )
        text_to_id[text] = qid
        n += 1
    logger.info(f"[subsample] loaded {n} canonical questions from {questions_file}")
    return text_to_id


def apply_question_subsample(
    score3: Dict[str, List[ScoredResponse]],
    responses_dir: Path,
    questions_file: Path,
    modulo: int,
    *,
    abort_on_mismatch: bool = True,
) -> Dict[str, List[ScoredResponse]]:
    """Filter score3 to responses whose original question id (in `questions_file`)
    is divisible by `modulo`.

    The response data's `question_index` is 0-based into the *already-reduced*
    list of questions used at generation time, not into the canonical questions
    file. We backreference via question text (which is preserved verbatim in the
    response records).

    Sanity check: we expect to retain exactly
        N_kept_canonical = |{ q in questions_file : q.id % modulo == 0 }|
    distinct question_index values across the response data per entity (modulo
    score==3 filter, which can drop more). If the canonical retained count
    doesn't match, abort -- given the cost of these runs, we'd rather fail loud
    than silently ship bad subsampling.
    """
    if modulo <= 0:
        return score3

    text_to_id = _load_canonical_question_text_to_id(questions_file)

    # Build the canonical retained-id set (over the entire questions file).
    retained_orig_ids = {qid for qid in text_to_id.values() if qid % modulo == 0}
    expected_canonical_kept = len(retained_orig_ids)

    # Build q_idx -> original_id by reading the FIRST entity's response file
    # (the response pipeline uses the same question list across all entities).
    # We use score3 keys to find a corresponding response file via responses_dir.
    if not score3:
        logger.warning("[subsample] no entities with score==3 responses; nothing to filter")
        return score3

    sample_entity = next(iter(score3.keys()))
    sample_path = responses_dir / f"{sample_entity}.jsonl"
    if not sample_path.exists():
        raise SystemExit(
            f"[subsample] cannot find responses file for {sample_entity} "
            f"at {sample_path} (needed to build q_idx->original_id map)"
        )
    qidx_to_orig: Dict[int, int] = {}
    unknown_qidx: Dict[int, str] = {}  # q_idx -> unmatched text (first occurrence)
    for line in sample_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qi = rec.get("question_index")
        qt = rec.get("question")
        if qi is None or qt is None:
            continue
        qi = int(qi)
        if qi in qidx_to_orig or qi in unknown_qidx:
            continue
        oid = text_to_id.get(qt)
        if oid is None:
            unknown_qidx[qi] = qt
            continue
        qidx_to_orig[qi] = oid

    if unknown_qidx:
        first_qi = sorted(unknown_qidx)[0]
        sample = unknown_qidx[first_qi][:120]
        raise SystemExit(
            f"[subsample] {len(unknown_qidx)} distinct response question_index "
            f"values have text not present in the canonical questions file "
            f"({sample_entity}.jsonl). "
            f"First unmatched: q_idx={first_qi}, text={sample!r}. "
            f"Aborting -- cost of misaligned subsampling is too high."
        )

    retained_qidx = {qi for qi, oid in qidx_to_orig.items() if oid % modulo == 0}
    actual_kept = len(retained_qidx)
    n_qidx_total = len(qidx_to_orig)

    # Sanity check: detect the regular stride of the response pipeline (Roger's
    # current setup uses stride=3 -- original ids {0, 3, 6, ..., 297}) and check
    # that `actual_kept` matches what that stride implies for `modulo`. The
    # closed form is: |{x in {0, s, 2s, ..., (k-1)s} : x % m == 0}|
    #               = ceil(k * gcd(s, m) / m).
    # If the response data isn't a regular arithmetic progression in original-id
    # space, fall back to a warning but don't compute an expected count.
    import math
    orig_ids_sorted = sorted(qidx_to_orig.values())
    diffs = {
        orig_ids_sorted[i + 1] - orig_ids_sorted[i]
        for i in range(len(orig_ids_sorted) - 1)
    }
    expected_actual_kept: Optional[int]
    if len(diffs) == 1 and orig_ids_sorted and orig_ids_sorted[0] == 0:
        stride = diffs.pop()
        g = math.gcd(stride, modulo)
        expected_actual_kept = math.ceil(n_qidx_total * g / modulo)
        stride_msg = f"stride={stride} detected"
    else:
        expected_actual_kept = None
        stride_msg = (
            "non-regular stride; sanity check downgraded to a tautology "
            f"(min diff: {min(diffs) if diffs else 'n/a'})"
        )

    base_msg = (
        f"[subsample] modulo={modulo}: response data uses "
        f"{n_qidx_total}/{len(text_to_id)} canonical questions ({stride_msg}); "
        f"keeping {actual_kept} q_idx values "
        f"({100*actual_kept/max(1,n_qidx_total):.1f}% of response questions, "
        f"{100*actual_kept/len(text_to_id):.1f}% of canonical)"
    )
    if expected_actual_kept is not None:
        base_msg += f"; expected exactly {expected_actual_kept}"
    logger.info(base_msg)

    if expected_actual_kept is not None and actual_kept != expected_actual_kept:
        err = (
            f"[subsample] SANITY CHECK FAILED: kept {actual_kept} q_idx values, "
            f"expected exactly {expected_actual_kept} given stride={stride}, "
            f"modulo={modulo}. This usually means the question_index -> "
            f"original_id backreference is broken (mismatched question text). "
            f"Aborting before any judge calls."
        )
        if abort_on_mismatch:
            raise SystemExit(err)
        logger.error(err)

    # Filter each entity's score==3 items.
    out: Dict[str, List[ScoredResponse]] = {}
    n_in = n_kept = 0
    for name, items in score3.items():
        kept: List[ScoredResponse] = []
        for it in items:
            m = _KEY_RE.match(it.key)
            if not m:
                continue
            q = int(m.group(2))
            if q in retained_qidx:
                kept.append(it)
        n_in += len(items)
        n_kept += len(kept)
        if kept:
            out[name] = kept
    logger.info(
        f"[subsample] kept {n_kept}/{n_in} score==3 responses "
        f"({100*n_kept/max(1,n_in):.1f}%) across {len(out)}/{len(score3)} entities"
    )
    return out


def _build_qidx_to_orig_map(
    responses_dir: Path,
    questions_file: Path,
    sample_entity: str,
    *,
    abort_on_mismatch: bool = True,
) -> Dict[int, int]:
    """Build q_idx -> original-canonical-id map from one entity's response file.

    The mapping is global (set at generation time and identical across entities,
    modulo per-entity RP filtering that drops some q_idx values), so any one
    file suffices to back-reference q_idx to the canonical question id.
    """
    text_to_id = _load_canonical_question_text_to_id(questions_file)
    sample_path = responses_dir / f"{sample_entity}.jsonl"
    if not sample_path.exists():
        raise SystemExit(
            f"[subsample] cannot find responses file for {sample_entity} at "
            f"{sample_path} (needed to build q_idx->original_id map)"
        )
    qidx_to_orig: Dict[int, int] = {}
    unknown_qidx: Dict[int, str] = {}
    for line in sample_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qi = rec.get("question_index")
        qt = rec.get("question")
        if qi is None or qt is None:
            continue
        qi = int(qi)
        if qi in qidx_to_orig or qi in unknown_qidx:
            continue
        oid = text_to_id.get(qt)
        if oid is None:
            unknown_qidx[qi] = qt
            continue
        qidx_to_orig[qi] = oid
    if unknown_qidx:
        first_qi = sorted(unknown_qidx)[0]
        sample = unknown_qidx[first_qi][:120]
        msg = (
            f"[subsample] {len(unknown_qidx)} distinct response question_index "
            f"values have text not present in the canonical questions file "
            f"({sample_entity}.jsonl). First unmatched: q_idx={first_qi}, "
            f"text={sample!r}."
        )
        if abort_on_mismatch:
            raise SystemExit(msg + " Aborting -- cost of misaligned subsampling is too high.")
        logger.warning(msg)
    return qidx_to_orig


def _count_default_persona_items(
    responses_dir: Path,
    *,
    default_name: str = "default",
) -> Optional[int]:
    """Total response items (lines) in the default persona's response file.

    The default persona is generated against the full canonical question list
    with no RP filtering, so its line count is the natural "expected upper
    bound" for a fully-populated entity (questions x passes, e.g. 500 ≈
    100 questions x 5 passes in the current Roger pipeline).

    Returns ``None`` if the file is missing (caller can fall back).
    """
    p = responses_dir / f"{default_name}.jsonl"
    if not p.exists():
        return None
    n = 0
    for line in p.read_text().splitlines():
        if line.strip():
            n += 1
    return n if n > 0 else None


def apply_tiered_question_subsample(
    score3: Dict[str, List[ScoredResponse]],
    responses_dir: Path,
    questions_file: Path,
    *,
    modulo_per_chunk: int = 3,
    n_default: Optional[int] = None,
    abort_on_mismatch: bool = True,
) -> Dict[str, List[ScoredResponse]]:
    """Per-entity tiered question subsampling for response-mode judging.

    Three tiers, indexed by the dense response-pipeline ``q_idx`` modulo
    ``M = modulo_per_chunk`` (default 3, i.e. 1/3 chunks). ``q_idx`` is
    chosen over the canonical orig_id because it's stride-agnostic: the
    response pipeline currently uses ``orig_id = 3 * q_idx`` (reduce=3 from
    the 300-question canonical), so ``orig_id % 3 == 0`` would be vacuously
    true for every item, while ``q_idx % 3 == 0`` cleanly partitions the
    100 generated questions into thirds.

    * **Tier 1** (1/3 of the canonical pool): keep items with ``q_idx % M == 0``.
    * **Tier 2** (2/3 of the canonical pool): keep items with ``q_idx % M in {0, 1}``.
    * **Tier 3** (full): keep all items.

    With ``N_default`` = the default persona's total response-item count
    (questions x passes; auto-detected from ``responses_dir/default.jsonl``)
    we set ``expected_t1 = N_default / M`` -- the items-per-entity we expect
    when nothing is RP-filtered. Per-entity tier choice (apply, in order):

      * If items passing tier-1 filter ``>= expected_t1 / 2``  -> use tier 1.
      * Else if items passing tier-2 filter ``>= 2 * expected_t1 / 3`` -> tier 2.
      * Else -> tier 3 (no subsampling).

    With the current pipeline (``N_default = 500``, ``M = 3``) this gives
    thresholds of ``~83`` and ``~111`` items, capping the typical per-entity
    workload at ``~expected_t1 = 167`` items while gracefully widening the
    sample when RP filtering depletes an entity to < 1/2 of expected.

    The ``questions_file`` argument is used only for a sanity check (do the
    response data's q_idx values back-reference cleanly to canonical question
    text?); the chunking itself is q_idx-based and ignores orig_id.
    """
    if not score3:
        return score3

    if n_default is None:
        n_default = _count_default_persona_items(responses_dir)
    if n_default is None or n_default <= 0:
        text_to_id = _load_canonical_question_text_to_id(questions_file)
        n_default = max(1, len(text_to_id)) * 5  # ~5 passes/question heuristic
        logger.warning(
            f"[tiered subsample] default persona response file missing or empty "
            f"under {responses_dir}; falling back to canonical question count x 5 "
            f"passes = {n_default} as N_default proxy"
        )

    expected_t1 = n_default / modulo_per_chunk
    threshold_to_t2 = expected_t1 / 2.0
    threshold_to_t3 = 2.0 * expected_t1 / 3.0

    # Sanity check the q_idx -> canonical mapping (also catches text drift
    # between the response pipeline and the questions file). Result isn't
    # used for chunking but failures here mean something deeper is broken.
    # Try score3 keys in order, then fall back to default.jsonl.
    sample_entity: Optional[str] = None
    for cand in list(score3.keys()) + ["default"]:
        if (responses_dir / f"{cand}.jsonl").exists():
            sample_entity = cand
            break
    if sample_entity is not None:
        _ = _build_qidx_to_orig_map(
            responses_dir, questions_file, sample_entity,
            abort_on_mismatch=abort_on_mismatch,
        )
    else:
        logger.warning(
            f"[tiered subsample] no response file found in {responses_dir} for "
            f"any of {list(score3.keys())[:3]} or default.jsonl; skipping the "
            f"q_idx -> canonical sanity check (chunking still works)."
        )

    def _qidx_of(it: "ScoredResponse") -> Optional[int]:
        m = _KEY_RE.match(it.key)
        return int(m.group(2)) if m else None

    out: Dict[str, List[ScoredResponse]] = {}
    tier_entities = {1: 0, 2: 0, 3: 0}
    tier_items = {1: 0, 2: 0, 3: 0}
    n_in = 0
    n_kept = 0
    for name, items in score3.items():
        n_in += len(items)
        c1 = c2 = 0
        for it in items:
            qi = _qidx_of(it)
            if qi is None:
                continue
            mod = qi % modulo_per_chunk
            if mod == 0:
                c1 += 1
            if mod in (0, 1):
                c2 += 1

        if c1 >= threshold_to_t2:
            tier = 1
        elif c2 >= threshold_to_t3:
            tier = 2
        else:
            tier = 3

        kept: List[ScoredResponse] = []
        for it in items:
            qi = _qidx_of(it)
            if qi is None:
                continue
            mod = qi % modulo_per_chunk
            if tier == 1 and mod == 0:
                kept.append(it)
            elif tier == 2 and mod in (0, 1):
                kept.append(it)
            elif tier == 3:
                kept.append(it)

        tier_entities[tier] += 1
        tier_items[tier] += len(kept)
        n_kept += len(kept)
        if kept:
            out[name] = kept

    logger.info(
        f"[tiered subsample] N_default={n_default} (M={modulo_per_chunk}); "
        f"thresholds: drop->t2 if t1-count<{threshold_to_t2:.0f}, "
        f"drop->t3 if t2-count<{threshold_to_t3:.0f}; "
        f"per-tier entities/items: "
        f"t1={tier_entities[1]}/{tier_items[1]}, "
        f"t2={tier_entities[2]}/{tier_items[2]}, "
        f"t3={tier_entities[3]}/{tier_items[3]}; "
        f"kept {n_kept}/{n_in} ({100*n_kept/max(1,n_in):.1f}%) across "
        f"{len(out)}/{len(score3)} entities"
    )
    return out


def load_score3_responses(
    entity_names: Sequence[str],
    scores_dir: Path,
    responses_dir: Path,
) -> Dict[str, List[ScoredResponse]]:
    """For each entity, gather (question, answer) pairs whose response scored 3.

    Performs a sanity check that every score==3 key maps to a response record.
    Missing matches are logged and dropped.
    """
    out: Dict[str, List[ScoredResponse]] = {}
    for name in entity_names:
        scores_path = scores_dir / f"{name}.json"
        responses_path = responses_dir / f"{name}.jsonl"
        if not scores_path.exists():
            logger.warning(f"{name}: no scores file at {scores_path}; skipping response-mode")
            continue
        if not responses_path.exists():
            logger.warning(f"{name}: no responses file at {responses_path}; skipping response-mode")
            continue
        try:
            scores = json.loads(scores_path.read_text())
        except Exception as e:
            logger.warning(f"{name}: cannot parse scores: {e}")
            continue
        response_map = _load_responses_jsonl(responses_path)

        score3_keys = [k for k, v in scores.items() if v == 3]
        matched: List[ScoredResponse] = []
        missing = 0
        for k in score3_keys:
            m = _KEY_RE.match(k)
            if not m:
                continue
            p, q = int(m.group(1)), int(m.group(2))
            rec = response_map.get((p, q))
            if rec is None:
                missing += 1
                continue
            q_text = rec.get("question", "")
            # The final assistant reply is the last assistant message in the conversation.
            conv = rec.get("conversation", [])
            answer = ""
            for turn in reversed(conv):
                if isinstance(turn, dict) and turn.get("role") == "assistant":
                    answer = turn.get("content", "")
                    break
            if not q_text or not answer:
                continue
            matched.append(ScoredResponse(name=name, key=k, question=q_text, answer=answer))
        if missing:
            logger.warning(
                f"{name}: {missing}/{len(score3_keys)} score==3 keys had no matching response record"
            )
        if matched:
            out[name] = matched
        else:
            logger.info(f"{name}: no usable score==3 responses")
    return out


# ---------------------------------------------------------------------------
# Whitener
# ---------------------------------------------------------------------------

@dataclass
class SoftKWhitener:
    """Soft-K PCA whitener: scales top-K PC directions of the fit distribution
    down so they match sigma_{K+1}, leaves all other directions untouched.
    Transform is linear and in-place on row-vectors.
    """
    Vt: torch.Tensor        # (n_pcs, hidden)
    scale: torch.Tensor     # (n_pcs,)
    mu: torch.Tensor        # (hidden,)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., hidden) -> whitened (..., hidden). Does NOT subtract mu;
        the whitener is a pure linear rescaling of directions."""
        # For a vector v, W(v) = v + sum_k (scale_k - 1) * (v @ v_k) * v_k
        proj = x @ self.Vt.T                     # (..., n_pcs)
        delta = (proj * (self.scale - 1.0)) @ self.Vt  # (..., hidden)
        return x + delta


def fit_soft_k_whitener(pool: torch.Tensor, K: int) -> SoftKWhitener:
    """pool: (N, hidden). Fit on mean-centered SVD of the pool."""
    mu = pool.mean(dim=0)
    X = pool - mu
    # full_matrices=False SVD -> Vt is (min(N, hidden), hidden)
    _U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    if K < 0:
        raise ValueError("K must be >= 0")
    K_eff = min(K, S.numel() - 1) if S.numel() >= 2 else S.numel()
    scale = torch.ones_like(S)
    if K_eff > 0:
        sigma_floor = S[K_eff]
        # Guard against zero singular values (degenerate pool)
        top_sigma = torch.clamp(S[:K_eff], min=torch.finfo(S.dtype).eps)
        scale[:K_eff] = sigma_floor / top_sigma
    return SoftKWhitener(Vt=Vt, scale=scale, mu=mu)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def compute_projections(
    corpus: Corpus,
    axis_spec: AxisSpec,
    whiten_K: int,
    slots: Sequence[int],
    pole_pair_names: Sequence[str],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Return ``{slot: {entity_id: {'raw': float, 'whitened': float}}}``.

    The per-slot inner dict is keyed by the **disambiguated** entity
    id (e.g. ``"patient|R"`` / ``"patient|T"``) — see
    :mod:`assistant_axis.entity_id` and AGENT_NOTES "Trait/role name
    collisions".  This is the May 2026 schema_version=2 fix for the
    bare-name-overwrite bug at the original
    ``{name: ... for (_etype, name), vec in corpus.vectors.items()}``
    site, which silently dropped one side of every collision name (9
    names per axis: ``ascetic, contrarian, cosmopolitan, generalist,
    pacifist, patient, perfectionist, romantic, stoic``).

    Per-entity leave-one-out pool: when computing the projection of entity
    X, the whitener pool is built from every other entity in the corpus
    except X (and except the pair-mode pole pair names, which are the axis-
    defining vectors and would be degenerate to include).

    This avoids the "deflate variance along the axis you're judging" failure
    mode of holding all listed-example entities out together: extremes along
    the canonical PC stay in the pool while we judge any single entity, so
    the top-K direction ordering matches the canonical-side computation.
    For the entity being judged, its own contribution is removed (the "treat
    as a new sample" framing).

    At ``whiten_K == 0`` the soft-K whitener is identity, so we skip the
    per-entity SVD entirely and ``raw == whitened`` per entity.
    """
    pole_skip = set(pole_pair_names)
    result: Dict[int, Dict[str, Dict[str, float]]] = {}
    for slot in slots:
        default_vec = corpus.default[slot]  # (hidden,)
        # Pre-compute centered (vec - default) for all entities at this
        # slot.  Keys are disambiguated entity_ids so collision names
        # don't silently overwrite each other.
        all_centered: Dict[str, torch.Tensor] = {
            entity_id(name, etype): (vec[slot] - default_vec)
            for (etype, name), vec in corpus.vectors.items()
        }
        axis_unit_raw = axis_spec.axis_by_slot[slot]  # (hidden,)

        per_slot: Dict[str, Dict[str, float]] = {}

        if whiten_K == 0:
            # Identity whitener (regardless of pool).  Skip SVDs.
            for eid, x in all_centered.items():
                raw_proj = float((x @ axis_unit_raw).item())
                per_slot[eid] = {"raw": raw_proj, "whitened": raw_proj}
        else:
            # Per-entity LOO whitener.  This is O(N) SVDs per slot; expect
            # ~0.15 s per SVD on a 553x5120 pool, so ~85 s per slot per
            # cell.  Multiplied across slots / cells / providers this can
            # reach hours -- consider SMW rank-1 updates for K>0 in
            # high-throughput contexts.
            for target_eid, target_x in all_centered.items():
                target_name = parse_entity_id(target_eid).name
                if target_name in pole_skip:
                    continue
                # LOO pool: everyone except target_eid and pole-pair names
                # (regardless of kind, since the pole pair is axis-defining).
                pool_rows = [
                    x for eid_other, x in all_centered.items()
                    if eid_other != target_eid
                    and parse_entity_id(eid_other).name not in pole_skip
                ]
                if len(pool_rows) < 2:
                    raise SystemExit(
                        f"slot {slot}: LOO pool for {target_eid} has "
                        f"<2 entries; cannot fit whitener"
                    )
                pool = torch.stack(pool_rows, dim=0)  # (N-1, hidden)
                whitener = fit_soft_k_whitener(pool, K=whiten_K)
                # Whitened axis (cell-specific because the whitener is
                # cell-specific now).
                axis_w = whitener.apply(
                    axis_unit_raw.unsqueeze(0)).squeeze(0)
                axis_w_norm = torch.linalg.vector_norm(axis_w).item()

                raw_proj = float((target_x @ axis_unit_raw).item())
                if axis_w_norm < 1e-12:
                    w_proj = float("nan")
                else:
                    xw = whitener.apply(target_x.unsqueeze(0)).squeeze(0)
                    w_proj = (float((xw @ axis_w).item())
                              / axis_w_norm)
                per_slot[target_eid] = {"raw": raw_proj, "whitened": w_proj}
        result[slot] = per_slot
    return result


# ---------------------------------------------------------------------------
# Scoring: prompt construction
# ---------------------------------------------------------------------------

def _entity_words(etype: str) -> Tuple[str, str, str, str]:
    """(singular, plural, singular_title, plural_title)"""
    if etype == "roles":
        return ("role", "roles", "Role", "Roles")
    return ("trait", "traits", "Trait", "Traits")


def build_static_prompt(axis_spec: AxisSpec, etype: str, name: str, content: str) -> str:
    # Per-call leakage prevention: strip the entity being judged from its
    # own rubric example list, so a 10-example axis becomes a 9-example
    # rubric for that one call when the entity is itself an example.
    # Other entities still see the full 10-example rubric.  Comparison
    # is done in file-name form (canonical key) before either side is
    # converted to display form for rubric injection.
    sing, plur, sing_t, plur_t = _entity_words(etype)
    neg_ex = [e for e in axis_spec.neg_examples if e != name]
    pos_ex = [e for e in axis_spec.pos_examples if e != name]
    # RUBRIC_VERSION v3: entity names rendered in display form
    # (``aligned artificial intelligence``) for human / LLM
    # readability; canonical file-name form is preserved everywhere
    # else (cache keys, dict keys, provenance, exclusions).
    return RUBRIC_STATIC.format(
        entity=sing, entity_plural=plur,
        entity_title=sing_t, entity_plural_title=plur_t,
        axis_name=display_form_name(axis_spec.axis_name),
        negative_pole=axis_spec.neg_pole, positive_pole=axis_spec.pos_pole,
        negative_examples=", ".join(display_form_name(e) for e in neg_ex)
            or "(none)",
        positive_examples=", ".join(display_form_name(e) for e in pos_ex)
            or "(none)",
        name=display_form_name(name), content=content,
    )


def build_response_batch_prompt(
    axis_spec: AxisSpec, etype: str, name: str, items: Sequence["ScoredResponse"]
) -> str:
    # Same per-call leakage prevention as build_static_prompt: the entity
    # whose responses are being judged is stripped from the example list
    # (file-name comparison; display-name injection -- see v3 notes
    # above).
    sing, plur, sing_t, plur_t = _entity_words(etype)
    n = len(items)
    blocks = []
    for i, it in enumerate(items, 1):
        blocks.append(
            f"--- Response {i} of {n} ---\n"
            f"[QUESTION]\n{it.question}\n[/QUESTION]\n"
            f"[RESPONSE]\n{it.answer}\n[/RESPONSE]"
        )
    items_block = "\n\n".join(blocks)
    neg_ex = [e for e in axis_spec.neg_examples if e != name]
    pos_ex = [e for e in axis_spec.pos_examples if e != name]
    return RUBRIC_RESPONSE_BATCH.format(
        entity=sing, entity_plural=plur,
        entity_title=sing_t, entity_plural_title=plur_t,
        axis_name=display_form_name(axis_spec.axis_name),
        negative_pole=axis_spec.neg_pole, positive_pole=axis_spec.pos_pole,
        negative_examples=", ".join(display_form_name(e) for e in neg_ex)
            or "(none)",
        positive_examples=", ".join(display_form_name(e) for e in pos_ex)
            or "(none)",
        name=display_form_name(name), n_items=n, items_block=items_block,
    )


def plan_response_batches(keys: Sequence[str], target_batch_size: int) -> List[List[str]]:
    """Partition `keys` into batches closest in average size to `target_batch_size`.

    n_batches = max(1, round(N / target)). Sizes differ by at most 1: the first
    `N % n_batches` batches get ceil(N/n_batches), the rest floor(N/n_batches).
    """
    n = len(keys)
    if n == 0:
        return []
    n_batches = max(1, round(n / target_batch_size))
    base = n // n_batches
    extra = n % n_batches
    batches: List[List[str]] = []
    idx = 0
    for b in range(n_batches):
        size = base + (1 if b < extra else 0)
        batches.append(list(keys[idx:idx + size]))
        idx += size
    return batches


def _sort_keys_by_question_then_prompt(keys: Sequence[str]) -> List[str]:
    """Sort 'pos_p{P}_q{Q}' keys by (Q, P) so batches naturally interleave
    across the 5 system-prompt variants rather than clumping by prompt."""
    def key_fn(k: str) -> Tuple[int, int]:
        m = _KEY_RE.match(k)
        if not m:
            return (10**9, 10**9)
        return (int(m.group(2)), int(m.group(1)))
    return sorted(keys, key=key_fn)


# ---------------------------------------------------------------------------
# Judge: provider-agnostic async calls
# ---------------------------------------------------------------------------

# Retry policy: attempt the call up to (1 + len(RETRY_DELAYS)) times.
# Cumulative waits ≈ 5 + 20 + 60 + 180 = 265 s, comfortably surviving the
# kind of ~3-minute network/provider blip we observed on 2026-04-25 03:40 UTC
# without leaving silent gaps in the score caches. Tunable via
# axis_judge_correlation_set_retry_delays() if a caller wants different.
RETRY_DELAYS: List[float] = [5.0, 20.0, 60.0, 180.0]


def _classify_judge_error(e: BaseException) -> Tuple[str, float | None]:
    """Bucket a provider exception into 'retry' / 'fatal' / 'parse_only'.

    Returns ``(action, suggested_wait)`` where ``action`` is one of:

    - ``"retry"``: transient -- worth re-attempting (timeouts, connection
      errors, 5xx server errors, 429 rate-limit-not-quota).
    - ``"fatal"``: permanent for this run -- don't retry (insufficient
      quota, auth errors, 4xx parameter errors).
    - ``"parse_only"``: API succeeded but content couldn't be parsed; this
      shouldn't reach us through exceptions, returned for completeness.

    ``suggested_wait`` is the server-suggested retry delay (from a 429
    Retry-After header) when available, else ``None`` (caller falls back to
    its own backoff schedule).

    The classifier is provider-agnostic and works by string-matching the
    exception class name + message; this avoids importing both SDKs at the
    module top level.
    """
    cls = type(e).__name__
    msg = str(e).lower()

    # insufficient_quota is OpenAI's "out of credits" signal -- never retry.
    if "insufficient_quota" in msg or "you exceeded your current quota" in msg:
        return "fatal", None

    # Auth / parameter / not-found errors -- never retry.
    fatal_classes = {
        "AuthenticationError", "PermissionDeniedError", "BadRequestError",
        "NotFoundError", "UnprocessableEntityError", "InvalidRequestError",
    }
    if cls in fatal_classes:
        return "fatal", None

    # Explicit "fatal" status codes from any APIStatusError-ish exception.
    code = getattr(e, "status_code", None) or getattr(e, "status", None)
    if isinstance(code, int) and 400 <= code < 500 and code != 429:
        return "fatal", None

    # Retryable: timeouts, connection errors, server errors (5xx), 429.
    retry_classes = {
        "APITimeoutError", "APIConnectionError", "InternalServerError",
        "ServiceUnavailableError", "RateLimitError", "TimeoutError",
        "ConnectionError", "ReadTimeout", "ConnectError",
    }
    if cls in retry_classes:
        # Try to honour Retry-After if present.
        wait = None
        for attr in ("retry_after", "_retry_after"):
            v = getattr(e, attr, None)
            if isinstance(v, (int, float)) and v > 0:
                wait = float(v)
                break
        return "retry", wait

    # Status-code-based fallbacks.
    if isinstance(code, int):
        if code == 429 or 500 <= code < 600:
            return "retry", None

    # Heuristic message-matching for SDK wrappers that lose the class name.
    if any(s in msg for s in (
        "timed out", "timeout", "connection", "request was interrupted",
        "overloaded", "internal server error", "service unavailable",
        "bad gateway", "rate_limit",
    )):
        return "retry", None

    # Unknown -- treat as retryable but only once (caller may differ).
    return "retry", None


async def _call_with_retry(
    one_attempt: "callable",
    *,
    provider_name: str,
    delays: Sequence[float] = RETRY_DELAYS,
) -> Optional[str]:
    """Call ``one_attempt()`` (an awaitable returning ``Optional[str]``) with
    classify+backoff retries. Returns the call result, or ``None`` if all
    attempts failed (logged at WARNING with full classification).

    :class:`BudgetExceededError` (Phase 4c, May 2026) is RE-RAISED
    unconditionally — it indicates the cost cap was crossed and the
    run must abort immediately, NOT retry.  All other exceptions are
    classified normally.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1 + len(delays)):
        try:
            return await one_attempt()
        except BudgetExceededError:
            # Hard escape: bypass all retry/swallow logic.  The
            # orchestrator's try/except in run() catches this and
            # exits with code 2.
            raise
        except Exception as e:  # noqa: BLE001 (we intentionally classify)
            action, server_wait = _classify_judge_error(e)
            last_exc = e
            if action == "fatal":
                logger.error(
                    f"{provider_name} fatal error (no retry): "
                    f"{type(e).__name__}: {e}"
                )
                return None
            if attempt == len(delays):
                logger.warning(
                    f"{provider_name} call failed after {attempt + 1} attempts: "
                    f"{type(e).__name__}: {e}"
                )
                return None
            wait = server_wait if server_wait is not None else delays[attempt]
            logger.info(
                f"{provider_name} call transient error "
                f"({type(e).__name__}); retrying in {wait:.0f}s "
                f"(attempt {attempt + 1}/{len(delays) + 1})"
            )
            await asyncio.sleep(wait)
    # Defensive; the loop should always return.
    logger.warning(f"{provider_name} call exhausted retries: {last_exc!r}")
    return None


async def _call_anthropic_batch(
    prompts: Sequence[str],
    model: str,
    max_tokens: int,
    temperature: float,
    rate_limiter: RateLimiter,
    batch_size: int,
    tracker: Optional[BudgetTracker] = None,
) -> List[Optional[str]]:
    import anthropic  # local import so OpenAI-only runs don't need it
    client = anthropic.AsyncAnthropic()

    async def one(prompt: str) -> Optional[str]:
        async def attempt() -> Optional[str]:
            await rate_limiter.acquire()
            resp = await client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            # Tick the budget tracker BEFORE returning text so a
            # cap-crossing call still gets accounted for in n_calls
            # and prompt/completion tokens (the orchestrator wants
            # exact counts in its summary even when it stopped on a
            # cap).  charge() raises BudgetExceededError if the
            # running total now exceeds the cap.
            if tracker is not None:
                pt, ct = extract_usage_anthropic(resp)
                tracker.charge(pt, ct)
            parts = []
            for block in resp.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts) if parts else None

        return await _call_with_retry(attempt, provider_name="anthropic")

    results: List[Optional[str]] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        tasks = [one(p) for p in chunk]
        r = await asyncio.gather(*tasks, return_exceptions=True)
        for x in r:
            if isinstance(x, BudgetExceededError):
                # Phase 4c: cap exceeded — abort immediately rather
                # than swallowing as None (which would let the loop
                # continue spending on the remaining chunks).
                raise x
            if isinstance(x, Exception):
                logger.error(f"anthropic gather: {x}")
                results.append(None)
            else:
                results.append(x)
    return results


async def _call_openai_batch(
    prompts: Sequence[str],
    model: str,
    max_tokens: int,
    temperature: float,
    rate_limiter: RateLimiter,
    batch_size: int,
    tracker: Optional[BudgetTracker] = None,
) -> List[Optional[str]]:
    """Local OpenAI async path; parallels _call_anthropic_batch so we can set
    temperature and token budget freely (the shared assistant_axis/judge.py
    hardcodes temperature=1).
    """
    import openai
    client = openai.AsyncOpenAI()

    async def one(prompt: str) -> Optional[str]:
        async def attempt() -> Optional[str]:
            await rate_limiter.acquire()
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            if tracker is not None:
                pt, ct = extract_usage_openai(resp)
                tracker.charge(pt, ct)
            return resp.choices[0].message.content

        return await _call_with_retry(attempt, provider_name="openai")

    results: List[Optional[str]] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        tasks = [one(p) for p in chunk]
        r = await asyncio.gather(*tasks, return_exceptions=True)
        for x in r:
            if isinstance(x, BudgetExceededError):
                # Phase 4c: cap exceeded — abort immediately.
                raise x
            if isinstance(x, Exception):
                logger.error(f"openai gather: {x}")
                results.append(None)
            else:
                results.append(x)
    return results


async def call_judge(
    prompts: Sequence[str],
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    rps: float,
    batch_size: int,
    tracker: Optional[BudgetTracker] = None,
) -> List[Optional[str]]:
    rate_limiter = RateLimiter(rps)
    if provider == "anthropic":
        return await _call_anthropic_batch(
            prompts, model, max_tokens, temperature, rate_limiter, batch_size,
            tracker=tracker,
        )
    if provider == "openai":
        return await _call_openai_batch(
            prompts, model, max_tokens, temperature, rate_limiter, batch_size,
            tracker=tracker,
        )
    raise SystemExit(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Scoring orchestration (with incremental cache)
# ---------------------------------------------------------------------------

def _default_model_for_provider(provider: str) -> str:
    if provider == "anthropic":
        # Sonnet 4 retired Sep 2026; 4.6 keeps the same sampling semantics.
        return "claude-sonnet-4-6"
    if provider == "openai":
        return "gpt-4.1-mini"
    raise SystemExit(f"Unknown provider: {provider}")


def _judge_family(provider: str, model: Optional[str]) -> str:
    """Classify a (provider, model) into one of ``"gpt"`` /
    ``"haiku"`` / ``"sonnet"`` for judge-aware defaults.

    The classification drives Bug-B-aware subsampling defaults: GPT is
    cheap → full volume by default; Anthropic-haiku is medium → tiered
    1/3 by default; Anthropic-sonnet is expensive → tiered 1/3
    *only*, with no escape hatch.

    ``model`` may be ``None`` (means ``--judge_model`` wasn't passed
    and the provider default applies); we resolve to the provider
    default in that case so the family classification is always
    well-defined.
    """
    resolved_model = model or _default_model_for_provider(provider)
    m = resolved_model.lower()
    if provider == "openai":
        return "gpt"
    if provider == "anthropic":
        if "haiku" in m:
            return "haiku"
        if "sonnet" in m:
            return "sonnet"
        # Future Opus / unknown Anthropic models default to the most
        # expensive bucket (no --no_subsample escape hatch) — fail safe.
        return "sonnet"
    raise SystemExit(f"Unknown provider: {provider}")


def _apply_judge_aware_subsample_defaults(args: argparse.Namespace) -> None:
    """Resolve the tri-state ``--no_subsample`` flag against the
    judge family.

    Rules (May 2026 — fixes Bug B):

    * If the user explicitly passed ``--no_subsample`` and the judge
      is Sonnet, hard-fail.  Sonnet is too expensive at full volume
      and the project never intended to support it there.
    * If ``--no_subsample`` was not specified (``None``):
        - GPT judge: default to ``True`` (full volume — GPT is cheap
          enough that subsampling is unnecessary, and pre-May-2026
          accidental subsampling produced the v2 GPT b10 caches that
          we're regenerating in Phase 5c).
        - Haiku/Sonnet judges: default to ``False`` so the existing
          tiered-1/3 path runs (was always the intended behaviour).
    * If the user explicitly set ``--no_subsample`` (True) for GPT
      or Haiku, honour it.

    Mutates ``args.no_subsample`` in place.  Called from
    :func:`run` after :func:`parse_args` so the resolved value
    appears in ``config.json`` and in the ``judge_extras`` provenance.
    """
    family = _judge_family(args.provider, args.judge_model)
    user_set = args.no_subsample is True
    if family == "sonnet" and user_set:
        raise SystemExit(
            "--no_subsample is not supported for the Sonnet judge "
            "(too expensive at full volume; the project never "
            "intended to support it). Run with the default tiered "
            "1/3 subsampling, or pick a cheaper judge."
        )
    if args.no_subsample is None:
        args.no_subsample = (family == "gpt")
    logger.info(
        f"[subsample] judge_family={family} no_subsample={args.no_subsample} "
        f"(question_subsample_modulo={args.question_subsample_modulo}, "
        f"tiered_modulo_per_chunk={args.tiered_modulo_per_chunk})"
    )


def _load_json_or_empty(path: Path) -> Dict[str, Any]:
    """Read a cache file written by ``_save_json``.

    Transparently unwraps the ``{"result": ..., "_provenance": ...,
    "schema_version": <int>}`` envelope produced when ``_save_json``
    is given an ``inputs`` list, so resume/refill paths don't need to
    know whether the on-disk cache is wrapped or bare.

    Note: this loader is permissive about ``schema_version``;
    consumers that need loud-reject behaviour on stale schemas should
    use :func:`assistant_axis.judge_loaders.load_static_scores` (or
    siblings) instead.  The producer's resume path is intentionally
    permissive: a stale-schema cache will simply be overwritten with
    fresh-schema data on the next save.
    """
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "_provenance" in data and "result" in data:
                data = data["result"]
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"{path}: cannot parse existing cache ({e}); starting fresh")
    return {}


def _peek_rubric_version(path: Path) -> Optional[str]:
    """Extract the recorded ``rubric_version`` label from a judge
    cache's provenance envelope.

    Returns the string stamped into the ``producer_script``
    :class:`InputSpec`'s ``extras`` at write time (e.g. ``"v3"``), or
    ``None`` if the file is missing, is not envelope-wrapped, or
    doesn't record a ``rubric_version`` (legacy pre-Phase-6 cache).

    Used by :func:`score_static_mode` and :func:`score_responses_mode`
    at cache-resume time to detect rubric drift: if the cache was
    written under an older rubric (e.g. ``v2``) and the current code
    is ``RUBRIC_VERSION = "v3"``, the cache scores are scientifically
    stale even though their schema_version / mtime fingerprints might
    still match.  The producer-script ``(mtime, size)`` fingerprint
    catches this too (RUBRIC_VERSION lives in this file), but the
    label-based check is more human-readable in logs and survives
    script-equivalence remarkings where the fingerprint changes for
    cosmetic reasons.
    """
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    prov = obj.get("_provenance")
    if not isinstance(prov, dict):
        return None
    for inp in prov.get("inputs", []) or []:
        if not isinstance(inp, dict):
            continue
        if inp.get("dep_key") != "producer_script":
            continue
        extras = inp.get("extras") or {}
        v = extras.get("rubric_version")
        if isinstance(v, str):
            return v
    return None


def _axis_from_cache_path(cache_path: Path) -> Optional[str]:
    """Infer the axis name from a judge-cache path.

    Cache paths in this project always live under
    ``<...>/<axis>/<judge_or_cohort>/scores_<mode>.json``, so the
    axis is exactly two ``parent`` hops up.  Returns ``None`` when
    the path doesn't fit that shape (e.g. an integration test
    running against a temp dir whose grandparent isn't an axis).
    Used by the rubric-equivalence lookup to scope the registry
    query to the right axis cell; ``None`` is treated as the
    axis wildcard.
    """
    try:
        return cache_path.parent.parent.name or None
    except Exception:  # pragma: no cover -- defensive
        return None


def _peek_per_entity_rubric_versions(path: Path) -> Dict[str, str]:
    """Read the per-entity ``rubric_version`` map stamped into a
    cache's ``_provenance.notes.per_entity_rubric_versions`` block.

    Returns an empty dict when the file is missing, isn't envelope-
    wrapped, or doesn't carry per-entity stamps (truly legacy caches
    written before per-entity stamping landed).  In that case the
    caller falls back to the cohort-level
    :func:`_peek_rubric_version` stamp as the effective version
    for every entry, matching the pre-stamping interpretation.

    Keys in the returned map are whatever the producer keys the
    main cache by (disambiguated ``eid`` for static-mode caches,
    bare ``name`` for the kind-pure response-mode caches).  See
    :func:`_check_rubric_version_on_resume` for how the map is
    interpreted.
    """
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    if not isinstance(obj, dict):
        return {}
    prov = obj.get("_provenance")
    if not isinstance(prov, dict):
        return {}
    notes = prov.get("notes") or {}
    if not isinstance(notes, dict):
        return {}
    raw = notes.get("per_entity_rubric_versions")
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def _peek_judge_model(path: Path) -> Optional[str]:
    """Extract the recorded ``judge_model`` from a judge cache's
    provenance envelope (the ``producer_script`` InputSpec ``extras``
    written by :func:`_build_axis_judge_inputs`), or ``None`` for a
    missing / bare / pre-Phase-6 cache.
    """
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    prov = obj.get("_provenance")
    if not isinstance(prov, dict):
        return None
    for inp in prov.get("inputs", []) or []:
        if isinstance(inp, dict) and inp.get("dep_key") == "producer_script":
            v = (inp.get("extras") or {}).get("judge_model")
            if isinstance(v, str) and v:
                return v
    return None


def _check_judge_model_on_resume(
    cache_path: Path,
    cache: Dict[str, Any],
    *,
    mode: str,
    current_model: str,
    strict: bool = False,
    drop_unrecorded: bool = False,
) -> Dict[str, Any]:
    """Judge-model drift check on resume (Sep 2026).

    Judge caches are keyed by judge *family* (``sonnet/``, ``gpt/``,
    ``haiku_responses_traits_b7_t3/``), not by model id, so a change
    of the default model -- Sonnet 4 -> 4.5 -> 4.6 -- would otherwise
    append new-model scores into an old-model cohort with nothing but
    the (rewritten) cohort-level provenance to show for it.  This
    gate keeps every cache file single-model:

    * recorded ``judge_model`` == ``current_model``: keep everything.
    * recorded and **different**: drop every entry (the whole file is
      rejudged under the current model), with a WARNING naming both
      models; at ``strict=True`` raise :class:`SystemExit` instead.
    * **unrecorded** (legacy pre-Phase-6 cache, 100+ GPT and Haiku
      files): keep with a WARNING by default -- we cannot tell which
      model wrote it and those judges have not changed.  Pass
      ``drop_unrecorded=True`` (CLI ``--drop_unrecorded_judge_model``)
      when migrating a family whose model *did* change (the Sonnet
      4 / 4.5 -> 4.6 migration), which drops the file for a full
      rejudge; at ``strict=True`` that combination aborts instead.

    Returns the (possibly emptied) cache.  Empty caches pass through.
    """
    if not cache:
        return cache
    recorded = _peek_judge_model(cache_path)
    n = len(cache)
    if recorded == current_model:
        return cache
    if recorded is None:
        if not drop_unrecorded:
            logger.warning(
                f"[{mode}] {cache_path.name}: cache records no judge_model "
                f"(legacy envelope); keeping its {n} entries and appending "
                f"{current_model!r} scores.  Pass --drop_unrecorded_judge_model "
                f"to rejudge the whole file instead."
            )
            return cache
        if strict:
            raise SystemExit(
                f"[{mode}] {cache_path.name}: cache records no judge_model and "
                f"--drop_unrecorded_judge_model is set; --strict_judge_model "
                f"refuses to silently rejudge its {n} entries under "
                f"{current_model!r}.  Drop one of the two flags."
            )
        logger.warning(
            f"[{mode}] {cache_path.name}: cache records no judge_model; "
            f"--drop_unrecorded_judge_model set, dropping all {n} entries for a "
            f"full rejudge under {current_model!r}."
        )
        return {}
    # Recorded and different.
    if strict:
        raise SystemExit(
            f"[{mode}] judge_model drift at {cache_path.name}: cache was "
            f"written by {recorded!r}, current judge is {current_model!r}.  "
            f"--strict_judge_model refuses to mix judges in one cohort.  "
            f"Either re-run with --judge_model {recorded} to extend the "
            f"existing cohort, or drop --strict_judge_model to rejudge all "
            f"{n} entries under {current_model!r}."
        )
    logger.warning(
        f"[{mode}] judge_model drift at {cache_path.name}: cache was written "
        f"by {recorded!r}, current judge is {current_model!r}; dropping all "
        f"{n} entries so the file is rejudged as a single-model cohort.  "
        f"Pass --strict_judge_model to abort instead, or --judge_model "
        f"{recorded} to keep extending the old cohort."
    )
    return {}


def _check_rubric_version_on_resume(
    cache_path: Path,
    cache: Dict[str, Any],
    *,
    mode: str,
    strict: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Per-entry rubric-version drift check on resume.

    Walks every entry in ``cache``, finds its effective rubric
    version (per-entry stamp if present in
    ``_provenance.notes.per_entity_rubric_versions``, else falls
    back to the cohort-level stamp from
    :func:`_peek_rubric_version`), and decides per entry:

    * **Matches current** :data:`RUBRIC_VERSION`: keep.
    * **Differs but declared equivalent** in
      :mod:`assistant_axis.rubric_equivalence` for this (axis, mode,
      entity_id) cell: keep, with an ``INFO`` log citing the registry
      edge.
    * **Differs and not equivalent**: drop the entry.  At ``strict=True``
      this becomes :class:`SystemExit` instead, listing the offending
      entries and pointing at ``tools/mark_rubric_equivalent.py``.

    Returns ``(kept_cache, kept_stamps)`` -- a filtered cache (entries
    that survived the check) and a stamps map covering every survivor
    (every kept entry gets a stamp: its prior per-entity stamp,
    promoted from cohort-level when missing).  The producer should
    pass ``kept_stamps`` forward as the starting per-entity map for
    further writes; new judging stamps the current ``RUBRIC_VERSION``
    on each newly-written entry.

    Empty caches pass through unchanged.  Legacy caches that lack
    both per-entity and cohort-level stamps are passed through as
    well -- the older ``schema_version`` / fingerprint checks
    elsewhere are responsible for those.
    """
    if not cache:
        return {}, {}
    cohort_rv = _peek_rubric_version(cache_path)
    per_entity_rv = _peek_per_entity_rubric_versions(cache_path)
    # No stamping information of any kind -> legacy.  Trust the
    # older schema_version / mtime fingerprints.
    if cohort_rv is None and not per_entity_rv:
        return cache, {}

    from assistant_axis import rubric_equivalence as _eq
    axis = _axis_from_cache_path(cache_path)
    registry = _eq.load_registry()

    kept: Dict[str, Any] = {}
    kept_stamps: Dict[str, str] = {}
    dropped_by_version: Dict[str, list] = {}
    equivalent_by_version: Dict[str, int] = {}
    for key, value in cache.items():
        eid = key  # static mode: disambiguated eid; responses: bare name
        rv = per_entity_rv.get(key, cohort_rv)
        if rv is None or rv == RUBRIC_VERSION:
            kept[key] = value
            if rv is not None:
                kept_stamps[key] = rv
            continue
        if _eq.is_equivalent(
            rv, RUBRIC_VERSION,
            axis=axis, mode=mode, entity_id=eid,
            registry=registry,
        ):
            kept[key] = value
            kept_stamps[key] = rv
            equivalent_by_version[rv] = equivalent_by_version.get(rv, 0) + 1
            continue
        dropped_by_version.setdefault(rv, []).append(key)

    n_total = len(cache)
    n_kept = len(kept)
    n_dropped = n_total - n_kept

    if equivalent_by_version:
        summary = ", ".join(
            f"{n} from {v!r}" for v, n in sorted(equivalent_by_version.items())
        )
        logger.info(
            f"[{mode}] rubric_equivalence: kept {summary} (out of {n_total}) "
            f"at {cache_path.name} (axis={axis!r}) via declared edges to "
            f"current RUBRIC_VERSION={RUBRIC_VERSION!r}; see "
            f"rubric_equivalences.yaml."
        )

    if not dropped_by_version:
        return kept, kept_stamps

    if strict:
        examples = []
        for v, eids in sorted(dropped_by_version.items()):
            head = ", ".join(eids[:5]) + ("..." if len(eids) > 5 else "")
            examples.append(f"{len(eids)} from {v!r}: {head}")
        examples_str = "; ".join(examples)
        # Suggest the most common version as the from-rubric template in
        # the example command.  Don't try to be clever about entity-id
        # vs axes; just print the bones.
        biggest_v = max(dropped_by_version, key=lambda k: len(dropped_by_version[k]))
        eid_hint = dropped_by_version[biggest_v][0]
        raise SystemExit(
            f"[{mode}] rubric_version drift at {cache_path.name} "
            f"(axis={axis!r}): {n_dropped}/{n_total} entries differ from "
            f"current RUBRIC_VERSION={RUBRIC_VERSION!r} with no declared "
            f"equivalence ({examples_str}).  --strict_rubric_version refuses "
            f"to silently rejudge.  Options: (1) accept the rejudge by "
            f"re-running without --strict_rubric_version; or (2) if the "
            f"rubric bump is prompt-preserving for these cells, declare "
            f"equivalence via:\n"
            f"    uv run python tools/mark_rubric_equivalent.py \\\n"
            f"        --from {biggest_v} --to {RUBRIC_VERSION} \\\n"
            f"        --modes {mode} --axes {axis} \\\n"
            f"        --entity-ids {eid_hint} '<other entity ids...>' \\\n"
            f"        --reason '<why this bump didn't change the rendered prompt>'"
        )

    summary_bits = []
    for v, eids in sorted(dropped_by_version.items()):
        summary_bits.append(f"{len(eids)} from rubric_version={v!r}")
    logger.warning(
        f"[{mode}] rubric_version drift at {cache_path.name} "
        f"(axis={axis!r}): dropped {n_dropped}/{n_total} entries "
        f"({', '.join(summary_bits)}); current RUBRIC_VERSION="
        f"{RUBRIC_VERSION!r}, no equivalence declared.  Affected "
        f"entries will be rejudged from scratch.  Pass "
        f"--strict_rubric_version to abort here instead, or run "
        f"tools/mark_rubric_equivalent.py to declare these bumps "
        f"prompt-preserving.  Affected entity ids per source rubric:\n"
        + "\n".join(
            f"  {v!r}: {', '.join(eids[:8])}"
            + ("..." if len(eids) > 8 else "")
            for v, eids in sorted(dropped_by_version.items())
        )
    )
    return kept, kept_stamps


def _save_json(
    path: Path,
    obj: Any,
    *,
    inputs: Optional[Sequence[InputSpec]] = None,
    title: Optional[str] = None,
    schema_version: Optional[int] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``obj`` to ``path``.

    When ``inputs`` is provided, wraps ``obj`` in a ``_provenance``
    envelope via :func:`assistant_axis.json_metadata` so downstream
    consumers can validate freshness via
    :func:`assistant_axis.provenance.load_validated_json`.

    When ``inputs`` is ``None`` the legacy bare-JSON shape is written
    (used for operational files like ``gaps.json`` and ``config.json``
    that don't have meaningful data dependencies).

    Args:
        schema_version: Optional payload-schema version stamp written
            at the top level of the envelope (alongside ``result`` and
            ``_provenance``).  Used by Option-C loud-reject readers
            (e.g. :class:`assistant_axis.judge_loaders.StaleSchemaError`).
            Pass ``2`` for the AT-RISK mixed-kind cache families
            (``scores_descriptions``, ``scores_instructions``,
            ``projections``, ``correlations``) after the May 2026
            disambiguated-key fix.
        notes: Optional dict merged into ``_provenance.notes`` (e.g.
            ``{"rejudge_names": [...]}`` for surgical rejudges, or
            ``{"usage": ...}`` for budget tracking).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if inputs is not None:
        envelope = json_metadata(obj, inputs=list(inputs), title=title)
        if schema_version is not None:
            envelope["schema_version"] = int(schema_version)
        if notes:
            existing = envelope["_provenance"].get("notes") or {}
            if not isinstance(existing, dict):
                existing = {"notes": existing}
            merged = {**existing, **notes}
            envelope["_provenance"]["notes"] = merged
        path.write_text(json.dumps(envelope, indent=2, sort_keys=True))
    else:
        path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _flush_budget_artifacts(
    args: argparse.Namespace,
    tracker: BudgetTracker,
) -> None:
    """End-of-run budget reporting: log one summary line and write a
    ``usage.json`` side-car next to the other outputs.

    Called from both the clean-finish path and the
    BudgetExceededError-cleanup path in :func:`run`.

    The side-car location is:

    * ``args.usage_json`` if explicitly set (operator override)
    * ``<output_dir>/usage.json`` otherwise

    Set ``--usage_json /dev/null`` to disable the side-car (useful in
    tests).
    """
    logger.info(tracker.log_line())
    side_car = getattr(args, "usage_json", None)
    if side_car is None and args.output_dir:
        side_car = str(Path(args.output_dir) / "usage.json")
    if not side_car or side_car == "/dev/null":
        return
    p = Path(side_car)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tracker.as_dict(), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Provenance: declare the inputs every output JSON depends on.
# See JUDGE PROVENANCE NOTE near the top for the rules on what counts
# as a meaningful change to the producer script.
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve()


def _build_axis_judge_inputs(
    args: argparse.Namespace,
    *,
    mode: str,
) -> List[InputSpec]:
    """Build the InputSpec list for one ``axis_judge_correlation.py`` output.

    ``mode`` is one of:
      * ``"descriptions"`` / ``"instructions"`` -- judge caches
        produced by :func:`score_static_mode`.
      * ``"responses"`` -- the judge cache produced by
        :func:`score_responses_mode`.
      * ``"projections"`` -- the activation-projection cache
        ``projections.json`` (does NOT depend on the judge).
      * ``"correlations"`` -- the per-mode correlation summary
        ``correlations.json`` (depends on every score cache that was
        produced this run, plus the projections).

    Granularity choice: we record full directory listings as
    ``current_files_input`` over every per-entity file the script can
    plausibly read, even when the judge call for one entity doesn't
    actually consume the others.  This over-invalidates a little
    (e.g. editing trait *X*'s description flags scores for trait *Y*
    as stale even though *Y*'s judge prompt didn't include *X*) but
    is dramatically simpler than per-entity selectivity and matches
    the project's stated tradeoff (regenerating plots is cheap; only
    rejudging is expensive, and Phase 6b's script-equivalence
    registry plus the deferred-rejudge registry handle the cost
    cases).
    """
    inputs: List[InputSpec] = []

    # Producer script (rubric + prompt code lives here).  Judge config
    # extras (model, temperature, etc.) are recorded as ``extras`` for
    # human inspection; they don't affect fingerprint comparisons but
    # are visible in audit reports / mark_script_equivalent.py.  The
    # rubric_version label lets a glance at any cache identify which
    # rubric produced it (mtime/size fingerprint already enforces drift
    # detection automatically; this is human-readable provenance).
    judge_extras: Dict[str, str] = {
        "rubric_version": RUBRIC_VERSION,
        "provider": args.provider,
        "judge_model": args.judge_model,
        "temperature": f"{args.temperature}",
        "max_tokens": f"{args.max_tokens}",
    }
    if mode == "responses":
        judge_extras["response_target_batch_size"] = f"{args.response_target_batch_size}"
        judge_extras["tiered_modulo_per_chunk"] = f"{args.tiered_modulo_per_chunk}"
        judge_extras["question_subsample_modulo"] = f"{args.question_subsample_modulo}"
        judge_extras["no_subsample"] = str(args.no_subsample)
    inputs.append(current_file_input(
        dep_key="producer_script",
        path=_SCRIPT_PATH,
        extras=judge_extras,
    ))

    # Axis spec source.
    if getattr(args, "axis_file", None):
        inputs.append(current_file_input(
            dep_key="axis_file",
            path=Path(args.axis_file),
        ))
    elif getattr(args, "pair", None):
        name1, name2 = args.pair
        pair_dir = Path(args.data_dir) / args.pair_type / "vectors"
        pair_paths = [pair_dir / f"{name1}.pt", pair_dir / f"{name2}.pt"]
        if all(p.exists() for p in pair_paths):
            inputs.append(current_files_input(
                dep_key="pair_vectors",
                paths=pair_paths,
                extras={"pair_type": args.pair_type, "pos_name": name1, "neg_name": name2},
            ))
        # Pole text source: only included when the user didn't override
        # via --pos_pole / --neg_pole (overridden poles come straight from
        # CLI args and don't read any file).
        pole_paths: List[Path] = []
        if not args.pos_pole:
            pole_paths.append(
                _pole_instruction_path(Path(args.instructions_dir), args.pair_type, name1)
            )
        if not args.neg_pole:
            pole_paths.append(
                _pole_instruction_path(Path(args.instructions_dir), args.pair_type, name2)
            )
        pole_paths = [p for p in pole_paths if p.exists()]
        if pole_paths:
            inputs.append(current_files_input(
                dep_key="pole_instructions", paths=pole_paths,
            ))

    # Corpus (entity descriptions / instructions) -- read by load_corpus
    # in every mode; affects the set of scorable entities, and for
    # descriptions/instructions modes is the actual content judged.
    if mode in ("descriptions", "instructions", "responses", "correlations"):
        instr_root = Path(args.instructions_dir)
        instr_paths: List[Path] = []
        for etype in ("roles", "traits"):
            idir = instr_root / etype / "instructions"
            if idir.exists():
                instr_paths.extend(sorted(idir.glob("*.json")))
        if instr_paths:
            inputs.append(current_files_input(
                dep_key="corpus_instructions", paths=instr_paths,
            ))

    # Corpus vectors -- needed by projections (always loaded; defines
    # the scorable entity set for every mode).
    if mode in ("projections", "correlations", "descriptions", "instructions", "responses"):
        data_dir = Path(args.data_dir)
        vec_paths: List[Path] = []
        for etype in ("roles", "traits"):
            vdir = data_dir / etype / "vectors"
            if vdir.exists():
                vec_paths.extend(sorted(vdir.glob("*.pt")))
        if vec_paths:
            inputs.append(current_files_input(
                dep_key="corpus_vectors",
                paths=vec_paths,
                extras={"layer": str(args.layer)},
            ))

    # Mode-specific: response judging reads per-entity score files,
    # response files, the default-persona response file (for tiered
    # subsampling threshold), and the canonical questions file.
    if mode == "responses" or mode == "correlations":
        if getattr(args, "responses_dir", None):
            responses_dir = Path(args.responses_dir)
            response_files = [
                p for p in sorted(responses_dir.glob("*.jsonl"))
                if p.name != "default.jsonl"
            ]
            if response_files:
                inputs.append(current_files_input(
                    dep_key="response_files", paths=response_files,
                ))
            default_jsonl = responses_dir / "default.jsonl"
            if default_jsonl.exists():
                inputs.append(current_file_input(
                    dep_key="default_responses", path=default_jsonl,
                ))
        if getattr(args, "scores_dir", None):
            scores_dir = Path(args.scores_dir)
            score_files = sorted(scores_dir.glob("*.json"))
            if score_files:
                inputs.append(current_files_input(
                    dep_key="response_score_files", paths=score_files,
                ))
        questions = Path(getattr(args, "questions_file", "data/extraction_questions.jsonl"))
        if questions.exists():
            inputs.append(current_file_input(
                dep_key="questions_file", path=questions,
            ))

    return inputs


# ---------------------------------------------------------------------------
# Gap-detection: post-run reporting and persistent gaps.json
# ---------------------------------------------------------------------------

def _update_gaps_file(
    output_dir: Path,
    *,
    mode: str,
    gaps: Any,
) -> Path:
    """Merge a per-mode gap report into ``<output_dir>/gaps.json``.

    The merged file has the shape::

        {
          "descriptions": [...names...],
          "instructions": [...names...],
          "responses": {
            "<name>": [<batch_indices_with_no_score>, ...],
            ...
          }
        }

    Modes write empty containers when they finish cleanly, so the file is
    always present and an empty value means "this mode has no gaps right
    now". Callers (notably ``refill_judge_gaps.py``) read this file to
    decide whether re-invocation is worthwhile.
    """
    gaps_path = output_dir / "gaps.json"
    existing: Dict[str, Any] = {}
    if gaps_path.exists():
        try:
            existing = json.loads(gaps_path.read_text())
        except Exception:
            existing = {}
    existing[mode] = gaps
    _save_json(gaps_path, existing)
    return gaps_path


def _warn_with_banner(message: str) -> None:
    """Multi-line WARNING banner that's hard to miss in run.log tails."""
    bar = "!" * 78
    logger.warning(bar)
    for line in message.splitlines():
        logger.warning(line)
    logger.warning(bar)


async def score_static_mode(
    mode: str,  # 'descriptions' or 'instructions'
    corpus: Corpus,
    axis_spec: AxisSpec,
    scorable: Sequence[Tuple[str, str]],
    args: argparse.Namespace,
    tracker: Optional[BudgetTracker] = None,
) -> Dict[str, int]:
    """Static-mode (descriptions / instructions) scoring.

    Cache key is the disambiguated :func:`entity_id` (e.g. ``"patient|R"``
    / ``"patient|T"``) so collision names don't silently overwrite each
    other (May 2026 ``schema_version: 2`` fix).  Resume: an entity is
    skipped if its disambiguated id is already in the cache.

    When ``args.rejudge_names`` is set, only listed (kind, name) pairs
    are scored, and any pre-existing cache entries for those ids are
    overwritten in place (other entries stay untouched).
    """
    cache_path = Path(args.output_dir) / f"scores_{mode}.json"
    cache: Dict[str, Any] = {} if args.no_cache else _load_json_or_empty(cache_path)
    # Rubric-version drift check (May 2026): per-entity granularity.
    # Entries whose recorded ``rubric_version`` differs from the current
    # :data:`RUBRIC_VERSION` AND that aren't covered by a declared
    # equivalence edge in :mod:`assistant_axis.rubric_equivalence` get
    # dropped here so the producer re-judges them under the current
    # rubric.  Survivors keep their existing stamps in ``per_entity_rv``;
    # newly-judged entries below stamp the current ``RUBRIC_VERSION``.
    # ``--strict_rubric_version`` promotes drop to SystemExit.
    cache, per_entity_rv = _check_rubric_version_on_resume(
        cache_path, cache, mode=mode,
        strict=bool(getattr(args, "strict_rubric_version", False)),
    )
    # Judge-model drift check (Sep 2026): caches are keyed by judge
    # family, so a model change (Sonnet 4/4.5 -> 4.6) must not append
    # into an old-model cohort.  See _check_judge_model_on_resume.
    cache = _check_judge_model_on_resume(
        cache_path, cache, mode=mode,
        current_model=args.judge_model or _default_model_for_provider(args.provider),
        strict=bool(getattr(args, "strict_judge_model", False)),
        drop_unrecorded=bool(getattr(args, "drop_unrecorded_judge_model", False)),
    )
    per_entity_rv = {k: v for k, v in per_entity_rv.items() if k in cache}
    # Phase 4 → 5 migration (May 2026): a v1 (bare-name) cache is
    # silently relabelled to v2 (entity_id keys) on resume.  The
    # corpus's ``descriptions`` map is the authoritative source for
    # what kind each bare name belongs to.  Three cases:
    #
    # (a) bare name is unique to ONE kind (~571/580 entries) →
    #     relabel as ``entity_id(name, kind)``; data preserved.
    # (b) bare name is a known collision (one of the 9 names in BOTH
    #     traits AND roles) → DROP the entry.  Bug A means we don't
    #     know which kind's score we have, so it's unsafe to keep.
    #     Phase 5b rejudges these surgically.
    # (c) bare name is in NEITHER corpus (orphan from a corpus
    #     edit or external import) → DROP, no recovery possible.
    #
    # Already-disambiguated keys pass through untouched, so v2 caches
    # produced by this script (or by a future migration) are
    # idempotent under the same load step.
    # NB: ``entity_id`` and ``is_entity_id`` MUST be in scope before the
    # ``if cache:`` block, otherwise Python's function-scope analysis
    # binds them as local-vars-via-import inside the block and the
    # outer reference at line ~2260 (``eid = entity_id(name, etype)``)
    # raises UnboundLocalError on first-time judging (no cache to
    # migrate, branch skipped, import never ran).  Bug fixed 2026-05-13
    # after the di-extension batch failed on all 26 new axes with this
    # exact symptom; previously hidden because Phase 5b/5c always
    # rejudged from a populated cache and the branch always fired.
    from assistant_axis.entity_id import entity_id, is_entity_id  # noqa: F401, E402
    if cache:
        kinds_for_name: Dict[str, set] = {}
        for (et, n) in corpus.descriptions.keys():
            kinds_for_name.setdefault(n, set()).add(et)
        migrated: Dict[str, Any] = {}
        n_relabelled = 0
        n_dropped_collision = 0
        n_dropped_orphan = 0
        for k, v in cache.items():
            if is_entity_id(k):
                migrated[k] = v
                continue
            kinds = kinds_for_name.get(k, set())
            if len(kinds) == 1:
                only_kind = next(iter(kinds))
                migrated[entity_id(k, only_kind)] = v
                n_relabelled += 1
            elif len(kinds) >= 2:
                n_dropped_collision += 1
            else:
                n_dropped_orphan += 1
        if n_relabelled or n_dropped_collision or n_dropped_orphan:
            logger.warning(
                f"[{mode}] migrated {cache_path.name}: "
                f"{n_relabelled} bare→eid relabelled, "
                f"{n_dropped_collision} collision-name drops "
                f"(Phase 5b rejudges these), "
                f"{n_dropped_orphan} orphan drops"
            )
        cache = migrated
    # Inputs are stable across the run; build once and reuse on every
    # incremental save inside the loop below.
    cache_inputs = _build_axis_judge_inputs(args, mode=mode)
    cache_title = f"axis_judge_correlation:{mode}:{axis_spec.axis_name}"

    rejudge_set: Optional[set] = (
        set(args.rejudge_names) if getattr(args, "rejudge_names", None) else None
    )

    prompts: List[str] = []
    names: List[str] = []
    etypes: List[str] = []
    eids: List[str] = []
    for (etype, name) in scorable:
        eid = entity_id(name, etype)
        if rejudge_set is not None and (etype, name) not in rejudge_set:
            # Entity not in the surgical rejudge list -> skip entirely
            # (don't score, don't touch cache).
            continue
        if rejudge_set is None and eid in cache:
            # Default behaviour: skip already-cached entities.
            continue
        # rejudge_set listed this entity: score it even if already cached
        # (cache entry will be overwritten on the next save).
        if mode == "descriptions":
            content = corpus.descriptions[(etype, name)]
        elif mode == "instructions":
            # Leading newline so the numbered list starts below `**name**:`
            # rather than running on the same line as the name.
            content = "\n" + "\n".join(
                f"{i+1}. {s}" for i, s in enumerate(corpus.pos_instructions[(etype, name)])
            )
        else:
            raise ValueError(mode)
        prompts.append(build_static_prompt(axis_spec, etype, name, content))
        names.append(name)
        etypes.append(etype)
        eids.append(eid)
    if not prompts:
        logger.info(f"[{mode}] all {len(scorable)} entities already cached at {cache_path}")
        # Still emit a clean gaps.json entry so downstream tools can rely on
        # its presence (and absence-of-mode = "this run didn't touch it").
        _update_gaps_file(Path(args.output_dir), mode=mode, gaps=[])
        # Resume path: existing cache may be a stale schema-v1 (bare-name)
        # blob; surface the int-valued entries as-is.  Loud-reject reads
        # happen in :func:`assistant_axis.judge_loaders.load_static_scores`,
        # not here.
        return {k: v for k, v in cache.items() if isinstance(v, int)}

    logger.info(
        f"[{mode}] scoring {len(prompts)} entities "
        f"({len(scorable) - len(prompts)} already cached) via {args.provider}:{args.judge_model}"
    )

    # Score in smaller sub-batches so the cache is saved frequently for crash-safety.
    written_cache = dict(cache)
    rejudge_notes: Optional[Dict[str, Any]] = None
    if rejudge_set is not None:
        rejudge_notes = {
            "rejudge_names": sorted(
                f"{kind_long(et)[0].upper()}:{n}" for (et, n) in rejudge_set
            ),
            "rejudge_mode": "static",
        }

    def _build_static_notes() -> Dict[str, Any]:
        """Per-save snapshot: rejudge metadata + budget tracker state +
        per-entity rubric_version stamps.

        The budget snapshot is taken at save time (each save_every
        chunk) so a partial cache that survives a BudgetExceededError
        still reflects the actual cost up to the abort point — which
        is the operator's primary forensic question.

        ``per_entity_rubric_versions`` is the (eid -> rubric_version)
        map persisted next to the cache so the resume path
        (:func:`_check_rubric_version_on_resume`) can do per-entry
        drift filtering instead of dropping the whole cohort when
        one entry's rubric drifts.  Each newly-judged entry below
        stamps the current ``RUBRIC_VERSION``; survivors of the
        resume check retain their previously-recorded stamp.
        """
        notes: Dict[str, Any] = {}
        if rejudge_notes:
            notes.update(rejudge_notes)
        if tracker is not None:
            notes["usage"] = tracker.as_dict()
        if per_entity_rv:
            notes["per_entity_rubric_versions"] = dict(per_entity_rv)
        return notes

    save_every = max(1, args.save_every)
    n_call_attempted = 0  # calls actually issued in this run
    n_call_parsed = 0  # of those, how many produced an int score
    for i in tqdm(range(0, len(prompts), save_every), desc=f"{mode}"):
        chunk_prompts = prompts[i:i + save_every]
        chunk_names = names[i:i + save_every]
        chunk_eids = eids[i:i + save_every]
        raw_texts = await call_judge(
            chunk_prompts, provider=args.provider, model=args.judge_model,
            max_tokens=args.max_tokens, temperature=args.temperature,
            rps=args.rps, batch_size=args.batch_size,
            tracker=tracker,
        )
        if tracker is not None:
            tracker.maybe_log(logger)
        for name, eid, text in zip(chunk_names, chunk_eids, raw_texts):
            n_call_attempted += 1
            score = parse_signed_score(text)
            if score is None:
                logger.warning(f"[{mode}] {name}: could not parse score from: {text!r}")
                continue
            n_call_parsed += 1
            # Disambiguated-id key: collision names don't overwrite.
            written_cache[eid] = score
            per_entity_rv[eid] = RUBRIC_VERSION
        _save_json(
            cache_path, written_cache,
            inputs=cache_inputs, title=cache_title,
            schema_version=2, notes=_build_static_notes(),
        )

    # Loud warning if parse rate this run dropped below 99%.  Note this
    # is *this run only* (not historical) so resumes that re-attempt
    # only previously-failed entries can still surface a healthy rate
    # if today's calls succeed.
    warn_if_low_parse_rate(
        label=f"axis_judge_correlation:{mode}:{args.provider}:{args.judge_model}",
        n_ok=n_call_parsed,
        n_total=n_call_attempted,
        logger_obj=logger,
        extra=f"this run only; see {Path(args.output_dir) / 'gaps.json'} for any persisting gaps",
    )

    # End-of-mode gap audit: which scorable entities still aren't in the cache?
    cached_eids = {k for k, v in written_cache.items() if isinstance(v, int)}
    missing = [
        entity_id(n, et) for (et, n) in scorable
        if entity_id(n, et) not in cached_eids
        and (rejudge_set is None or (et, n) in rejudge_set)
    ]
    _update_gaps_file(Path(args.output_dir), mode=mode, gaps=missing)
    if missing:
        # Producer-side automatic refusal fallback (2026-05-13):
        # Partition the missing entities into (allowlisted) vs
        # (unexpected) via data/judge_refusal_allowlist.json.  For
        # allowlisted entries we attempt an inline backfill from the
        # configured fallback judge (typically Haiku) -- no separate
        # tool invocation needed.  For unexpected entries we emit the
        # loud banner so they get investigated.  See
        # assistant_axis/judge_refusal_fallback.py for the shared
        # implementation (also used by tools/fill_judge_refusal_gaps.py
        # for post-hoc audits of caches produced before this code path
        # existed).
        from assistant_axis.judge_refusal_fallback import (
            fill_allowlisted_gaps_inline, load_allowlist,
            partition_gaps_by_allowlist,
        )
        allowlist = load_allowlist()
        # First do a partition-only pass so we can emit the loud-banner
        # warning for unexpected gaps BEFORE we make any fallback API
        # calls -- gives the user maximum visibility into surprises.
        _allowlisted_preview, unexpected_misses = partition_gaps_by_allowlist(
            missing, judge_model=args.judge_model, mode=mode,
            allowlist=allowlist,
        )
        if unexpected_misses:
            head = ", ".join(unexpected_misses[:8]) + (
                "..." if len(unexpected_misses) > 8 else "")
            _warn_with_banner(
                f"[{mode}] {len(unexpected_misses)} of {len(scorable)} "
                f"scorable entities have no cached score (call/parse "
                f"failed across all retries) AND are NOT on the judge-"
                f"refusal allowlist for {args.judge_model!r}.\n"
                f"  unexpected: {head}\n"
                f"  full list: {Path(args.output_dir) / 'gaps.json'}\n"
                f"  options:\n"
                f"   1. Retry the run (transient failure).\n"
                f"   2. If reproducible, confirm a fallback judge accepts\n"
                f"      the prompt and add (judge_model, entity_id) to\n"
                f"      data/judge_refusal_allowlist.json, then re-run\n"
                f"      this script (allowlisted gaps will be auto-filled\n"
                f"      next time around).\n"
                f"   3. If the prompt itself is the problem, fix it."
            )
        # Inline fallback for allowlisted gaps.  Builds the same prompt
        # via build_static_prompt + the in-scope axis_spec, so the
        # fallback judge sees byte-identical context to what the
        # primary judge refused.  Patches written_cache + the on-disk
        # scores file with notes.fallback_fills annotation so the
        # substitution is auditable.
        if _allowlisted_preview:
            def _producer_prompt_builder(eid: str, _mode: str) -> str:
                # _mode is always == mode here (single-mode call site)
                # but kept in signature for symmetry with the tool's
                # cross-mode caller.
                from assistant_axis.judge_refusal_fallback import (
                    kind_of_entity_id, name_of_entity_id,
                    build_content_for_mode,
                )
                kind = kind_of_entity_id(eid)
                name = name_of_entity_id(eid)
                content = build_content_for_mode(
                    kind, name, _mode,
                    instructions_root=Path(args.instructions_dir),
                )
                return build_static_prompt(axis_spec, kind, name, content)

            fill_results, _ = await fill_allowlisted_gaps_inline(
                cache_path=cache_path,
                mode=mode,
                missing_eids=missing,
                judge_model=args.judge_model,
                prompt_builder=_producer_prompt_builder,
                parse_score=parse_signed_score,
                allowlist=allowlist,
                concurrency=3,
                logger_obj=logger,
            )
            # Patch our in-memory written_cache so the return value of
            # this function reflects the fallback-filled entities.
            for r in fill_results:
                if r["status"] == "filled":
                    written_cache[r["entity_id"]] = int(r["score"])

            # Recompute gaps and refresh gaps.json with the *post*-fill
            # state so the file no longer points at entities we just
            # successfully filled.
            cached_eids = {
                k for k, v in written_cache.items() if isinstance(v, int)
            }
            missing = [
                entity_id(n, et) for (et, n) in scorable
                if entity_id(n, et) not in cached_eids
                and (rejudge_set is None or (et, n) in rejudge_set)
            ]
            _update_gaps_file(Path(args.output_dir), mode=mode, gaps=missing)
    else:
        logger.info(f"[{mode}] all {len(scorable)} scorable entities cached cleanly.")

    # Return only int-valued entries (keyed by disambiguated entity id).
    return {k: v for k, v in written_cache.items() if isinstance(v, int)}


async def score_responses_mode(
    axis_spec: AxisSpec,
    scorable: Sequence[Tuple[str, str]],
    args: argparse.Namespace,
    tracker: Optional[BudgetTracker] = None,
) -> Dict[str, Dict[str, Any]]:
    """Score ALL score==3 responses per entity, partitioned into roughly equal-sized
    batches (targeting `--response_target_batch_size`, default 10). Each batch is one
    judge call producing one -3..+3 score; the entity's final score is the mean of
    those batch scores.

    Cache layout per entity:
        {
          'target_batch_size': int,
          'n_total_items': int,
          'n_batches': int,
          'per_batch': [{'keys': [...], 'score': int|None, 'text': str|None}, ...],
          'batch_scores': [int, ...],     # non-None batch scores
          'mean_score': float|None,
          'std_score': float|None,
        }

    Resume: if the cached plan (target_batch_size + sorted key list per batch) matches
    what we'd compute from the current score==3 pool AND every batch has a non-None
    score, the entity is skipped.
    """
    if not args.scores_dir or not args.responses_dir:
        raise SystemExit("--score_responses requires --scores_dir and --responses_dir")

    cache_path = Path(args.output_dir) / "scores_responses.json"
    cache: Dict[str, Any] = {} if args.no_cache else _load_json_or_empty(cache_path)
    # Rubric-version drift check (May 2026): see score_static_mode.
    cache, per_entity_rv = _check_rubric_version_on_resume(
        cache_path, cache, mode="responses",
        strict=bool(getattr(args, "strict_rubric_version", False)),
    )
    cache = _check_judge_model_on_resume(
        cache_path, cache, mode="responses",
        current_model=args.judge_model or _default_model_for_provider(args.provider),
        strict=bool(getattr(args, "strict_judge_model", False)),
        drop_unrecorded=bool(getattr(args, "drop_unrecorded_judge_model", False)),
    )
    per_entity_rv = {k: v for k, v in per_entity_rv.items() if k in cache}
    cache_inputs = _build_axis_judge_inputs(args, mode="responses")
    cache_title = f"axis_judge_correlation:responses:{axis_spec.axis_name}"

    # Response mode reads from a kind-pure ``responses_dir`` (e.g.
    # ``.../traits/responses/`` or ``.../roles/responses/``).  Derive
    # the canonical kind from the path and filter ``scorable`` so
    # collision names get the right prompt label (Bug A residual fix:
    # the old ``ent_type_map = {n: et for (et, n) in scorable}`` keyed
    # by bare name silently overwrote one side of every collision name,
    # so 9 names per axis got the wrong "trait/role" label in the
    # rubric prompt).
    responses_kind = Path(args.responses_dir).parent.name
    if responses_kind not in ("roles", "traits"):
        raise SystemExit(
            f"--responses_dir parent dir must be 'roles' or 'traits' "
            f"(got {Path(args.responses_dir).parent}); response-mode "
            f"requires a kind-pure cohort root.  responses_dir="
            f"{args.responses_dir!r}"
        )
    rejudge_set: Optional[set] = (
        set(args.rejudge_names) if getattr(args, "rejudge_names", None) else None
    )
    scorable_kind_pure = [
        (et, n) for (et, n) in scorable
        if et == responses_kind
        and (rejudge_set is None or (et, n) in rejudge_set)
    ]
    if not scorable_kind_pure:
        if rejudge_set is not None:
            logger.warning(
                f"--rejudge_names filter left no {responses_kind} entities "
                f"to score in this axis; emitting empty cache."
            )
        else:
            logger.warning(
                f"No {responses_kind} entities in scorable; emitting empty cache."
            )
    scorable = scorable_kind_pure
    names_only = [n for (_, n) in scorable]
    # Now collision-free because scorable is kind-pure.
    ent_type_map = {n: et for (et, n) in scorable}

    score3 = load_score3_responses(
        names_only, Path(args.scores_dir), Path(args.responses_dir)
    )
    logger.info(f"[responses] {len(score3)} entities have at least one score==3 response")

    if args.question_subsample_modulo and args.question_subsample_modulo > 0:
        # Legacy uniform mode: every entity filtered identically by orig_id % M == 0.
        # Kept for backward compatibility with existing q9 runs.
        score3 = apply_question_subsample(
            score3,
            responses_dir=Path(args.responses_dir),
            questions_file=Path(args.questions_file),
            modulo=int(args.question_subsample_modulo),
            abort_on_mismatch=True,
        )
    elif not args.no_subsample:
        # Default: per-entity tiered subsampling. RP-light entities use 1/3 of
        # the canonical pool; entities depleted by RP filtering fall back to
        # 2/3 or full to keep the sample large enough for stable judging.
        score3 = apply_tiered_question_subsample(
            score3,
            responses_dir=Path(args.responses_dir),
            questions_file=Path(args.questions_file),
            modulo_per_chunk=int(args.tiered_modulo_per_chunk),
            abort_on_mismatch=True,
        )

    target = args.response_target_batch_size

    # Plan batches for each entity (deterministic from the sorted score==3 keys).
    plans: Dict[str, Dict[str, Any]] = {}  # name -> {items, key_batches, key_to_item}
    for name, items in score3.items():
        sorted_keys = _sort_keys_by_question_then_prompt([it.key for it in items])
        key_to_item = {it.key: it for it in items}
        key_batches = plan_response_batches(sorted_keys, target)
        plans[name] = {
            "items": items,
            "key_batches": key_batches,
            "key_to_item": key_to_item,
        }

    # Build the actual work list: (name, etype, batch_index, [items in batch]).
    to_call: List[Tuple[str, str, int, List[ScoredResponse]]] = []
    written: Dict[str, Any] = dict(cache)
    for name, plan in plans.items():
        etype = ent_type_map.get(name, "roles")
        key_batches: List[List[str]] = plan["key_batches"]
        key_to_item: Dict[str, ScoredResponse] = plan["key_to_item"]
        existing = cache.get(name, {})
        cached_pb = existing.get("per_batch") or []
        cache_matches = (
            existing.get("target_batch_size") == target
            and len(cached_pb) == len(key_batches)
            and all(cached_pb[i].get("keys") == key_batches[i] for i in range(len(key_batches)))
        )
        if cache_matches and not args.no_cache:
            per_batch = cached_pb
        else:
            # Stale or missing -> rebuild per_batch skeleton, drop stale scores.
            per_batch = [{"keys": kb, "score": None, "text": None} for kb in key_batches]

        # Determine which batches need calls.
        for bi, entry in enumerate(per_batch):
            if entry.get("score") is None:
                batch_items = [key_to_item[k] for k in entry["keys"]]
                to_call.append((name, etype, bi, batch_items))

        # Write the up-to-date skeleton into the cache shell.
        written[name] = {
            "target_batch_size": target,
            "n_total_items": len(plan["items"]),
            "n_batches": len(key_batches),
            "per_batch": per_batch,
            "batch_scores": [e["score"] for e in per_batch if isinstance(e.get("score"), int)],
            "mean_score": None,
            "std_score": None,
        }
        _update_response_aggregates(written[name])

    logger.info(
        f"[responses] target_batch_size={target}; "
        f"{len(to_call)} batch calls to run "
        f"({sum(len(p['key_batches']) for p in plans.values())} batches total, "
        f"{sum(len(p['items']) for p in plans.values())} items across all entities)"
    )
    if to_call:
        batch_sizes = [len(items) for (_, _, _, items) in to_call]
        logger.info(
            f"[responses] batch sizes in pending calls: "
            f"min={min(batch_sizes)}, max={max(batch_sizes)}, "
            f"mean={np.mean(batch_sizes):.2f}"
        )

    def _response_notes() -> Optional[Dict[str, Any]]:
        """Per-save snapshot: budget tracker state + per-entity
        rubric_version stamps.  Returns ``None`` only when neither is
        populated (matches legacy bare-cache callers).

        ``per_entity_rubric_versions`` is the (name -> rubric_version)
        map persisted next to the cache so the resume path
        (:func:`_check_rubric_version_on_resume`) can do per-entry
        drift filtering.  Response-mode caches are kind-pure, so
        keys here are bare names (not disambiguated eids).
        """
        notes: Dict[str, Any] = {}
        if tracker is not None:
            notes["usage"] = tracker.as_dict()
        if per_entity_rv:
            notes["per_entity_rubric_versions"] = dict(per_entity_rv)
        return notes or None

    # Save initial skeleton so a crash here still leaves a coherent cache.
    _save_json(
        cache_path, written, inputs=cache_inputs, title=cache_title,
        notes=_response_notes(),
    )

    save_every = max(1, args.save_every)
    n_batch_attempted = 0  # batch calls actually issued this run
    n_batch_parsed = 0  # of those, how many produced an int score
    for i in tqdm(range(0, len(to_call), save_every), desc="responses"):
        chunk = to_call[i:i + save_every]
        prompts = [
            build_response_batch_prompt(axis_spec, etype, name, items)
            for (name, etype, _bi, items) in chunk
        ]
        raw_texts = await call_judge(
            prompts, provider=args.provider, model=args.judge_model,
            max_tokens=args.max_tokens, temperature=args.temperature,
            rps=args.rps, batch_size=args.batch_size,
            tracker=tracker,
        )
        if tracker is not None:
            tracker.maybe_log(logger)
        for (name, _etype, bi, _items), text in zip(chunk, raw_texts):
            n_batch_attempted += 1
            score = parse_signed_score(text)
            if score is not None:
                n_batch_parsed += 1
            written[name]["per_batch"][bi]["score"] = score
            written[name]["per_batch"][bi]["text"] = text
            # Stamp the per-entity rubric_version on every entry we
            # touch this run.  An entity whose batches span two resume
            # generations will end up stamped under the last-touching
            # rubric -- which is the conservative ("most recent
            # rubric") choice; the producer's resume contract is that
            # any per_batch with a None score gets re-issued, so the
            # final stamp reflects the rubric all surviving batches
            # were under once the cache stabilises.
            per_entity_rv[name] = RUBRIC_VERSION

        for name in written:
            _update_response_aggregates(written[name])
        _save_json(
            cache_path, written, inputs=cache_inputs, title=cache_title,
            notes=_response_notes(),
        )

    # Loud warning if per-batch parse rate this run dropped below 99%.
    # Each "call" here is one ~target_batch_size-item batch; an UNPARSEABLE
    # batch loses *all* of its items, so the threshold is item-conservative.
    warn_if_low_parse_rate(
        label=f"axis_judge_correlation:responses:{args.provider}:{args.judge_model}",
        n_ok=n_batch_parsed,
        n_total=n_batch_attempted,
        logger_obj=logger,
        extra=f"this run only; see {Path(args.output_dir) / 'gaps.json'} for any persisting gaps",
    )

    for name in written:
        _update_response_aggregates(written[name])
    _save_json(
        cache_path, written, inputs=cache_inputs, title=cache_title,
        notes=_response_notes(),
    )

    # End-of-mode gap audit: which entities still have None batches?
    gaps_by_entity: Dict[str, List[int]] = {}
    n_none_total = 0
    n_entities_no_score = 0
    for name, entry in written.items():
        if not isinstance(entry, dict):
            continue
        per_batch = entry.get("per_batch", []) or []
        none_idxs = [i for i, b in enumerate(per_batch) if b.get("score") is None]
        if none_idxs:
            gaps_by_entity[name] = none_idxs
            n_none_total += len(none_idxs)
        if entry.get("mean_score") is None and per_batch:
            n_entities_no_score += 1
    _update_gaps_file(Path(args.output_dir), mode="responses", gaps=gaps_by_entity)
    if gaps_by_entity:
        head = ", ".join(list(gaps_by_entity.keys())[:8]) + (
            "..." if len(gaps_by_entity) > 8 else ""
        )
        _warn_with_banner(
            f"[responses] {n_none_total} batches across {len(gaps_by_entity)} entities "
            f"still have no score (call/parse failed across all retries); "
            f"{n_entities_no_score} entities have *no* valid batch score at all "
            f"and will be excluded from ρ.\n"
            f"  examples: {head}\n"
            f"  full list: {Path(args.output_dir) / 'gaps.json'}\n"
            f"  re-run the same command to retry only the missing batches "
            f"(response-mode resume re-issues calls for any per_batch entry "
            f"with score=None)."
        )
    else:
        logger.info(
            f"[responses] all {len(written)} entities have complete batch coverage."
        )
    return {k: v for k, v in written.items() if isinstance(v, dict)}


def _update_response_aggregates(entry: Dict[str, Any]) -> None:
    """Recompute batch_scores/mean_score/std_score from entry['per_batch']."""
    per_batch = entry.get("per_batch", [])
    ints = [b["score"] for b in per_batch if isinstance(b.get("score"), int)]
    entry["batch_scores"] = ints
    if ints:
        arr = np.asarray(ints, dtype=float)
        entry["mean_score"] = float(arr.mean())
        entry["std_score"] = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0
    else:
        entry["mean_score"] = None
        entry["std_score"] = None


# ---------------------------------------------------------------------------
# Correlations and plot
# ---------------------------------------------------------------------------

def _spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spearman rho and a two-sided p-value (via scipy if available, else manual)."""
    try:
        from scipy.stats import spearmanr  # type: ignore
        res = spearmanr(x, y)
        return float(res.correlation), float(res.pvalue)
    except Exception:
        # Fallback: rank-Pearson without p-value.
        def _rankdata(v: np.ndarray) -> np.ndarray:
            order = np.argsort(v, kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(v) + 1)
            # Tied ranks: replace tied runs with their mean
            sv = v[order]
            i = 0
            while i < len(sv):
                j = i + 1
                while j < len(sv) and sv[j] == sv[i]:
                    j += 1
                if j > i + 1:
                    mean_rank = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
                    for k in range(i, j):
                        ranks[order[k]] = mean_rank
                i = j
            return ranks
        rx, ry = _rankdata(x), _rankdata(y)
        if len(x) < 3:
            return float("nan"), float("nan")
        rx_c, ry_c = rx - rx.mean(), ry - ry.mean()
        denom = np.sqrt((rx_c**2).sum() * (ry_c**2).sum())
        rho = float((rx_c * ry_c).sum() / denom) if denom > 0 else float("nan")
        return rho, float("nan")


def compute_correlations(
    scores_by_mode: Dict[str, Dict[str, float]],
    projections: Dict[int, Dict[str, Dict[str, float]]],
    slots: Sequence[int],
    excluded_set: set,
) -> Dict[str, Any]:
    """Returns ``{mode: {slot: {raw|whitened: {rho, p, n, names, scores, projections}}}}``.

    ``score_map`` and ``per_slot`` are keyed by **disambiguated entity
    ids** (e.g. ``"patient|R"``) for the ``descriptions`` /
    ``instructions`` static modes; the response mode is kind-pure (one
    cohort dir per kind) so its keys are bare names.  Both shapes are
    accepted: the helper looks up the bare-name in ``excluded_set``
    (the pole-pair set is bare-name) via :func:`parse_entity_id` when
    the key carries a kind suffix.

    The emitted ``names`` array preserves whichever key shape the
    inputs used (so a v2 static-mode correlation stamps disambiguated
    ids; a response-mode correlation stamps bare names).
    """
    out: Dict[str, Any] = {}

    def _bare_name(key: str) -> str:
        # Tolerate both ``"patient|R"`` and ``"patient"`` so we don't
        # crash on response-mode (kind-pure, bare-name) inputs.
        try:
            return parse_entity_id(key).name
        except ValueError:
            return key

    for mode, score_map in scores_by_mode.items():
        out[mode] = {}
        for slot in slots:
            out[mode][str(slot)] = {}
            per_slot = projections.get(slot, {})
            for metric in ("raw", "whitened"):
                names = []
                svals = []
                pvals = []
                for key, s in sorted(score_map.items()):
                    bare = _bare_name(key)
                    if bare in excluded_set:
                        continue
                    if key not in per_slot:
                        continue
                    p = per_slot[key].get(metric)
                    if p is None or not np.isfinite(p):
                        continue
                    names.append(key)
                    svals.append(s)
                    pvals.append(p)
                if len(svals) >= 3:
                    rho, pval = _spearmanr(np.asarray(svals, dtype=float),
                                           np.asarray(pvals, dtype=float))
                else:
                    rho, pval = float("nan"), float("nan")
                out[mode][str(slot)][metric] = {
                    "rho": rho, "p": pval, "n": len(svals),
                    "names": names, "scores": svals, "projections": pvals,
                }
    return out


def make_plot(
    correlations: Dict[str, Any],
    slots: Sequence[int],
    axis_spec: AxisSpec,
    output_path: Path,
    slot_labels: Optional[Dict[int, str]] = None,
    inputs: Optional[Sequence[InputSpec]] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from assistant_axis import is_entity_id, kind_color
    from assistant_axis.entity_id import parse_entity_id

    modes = list(correlations.keys())
    if not modes:
        logger.warning("no correlation data to plot")
        return
    n_modes = len(modes)
    metrics = ("raw", "whitened")
    n_cols = len(slots) * len(metrics)
    n_rows = n_modes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows),
                             squeeze=False, sharex=False, sharey=False)
    title_line = f"Axis judge correlation: {axis_spec.axis_name}"
    fig.suptitle(
        title_line + "\n"
        f"neg: {axis_spec.neg_pole[:90]}...\npos: {axis_spec.pos_pole[:90]}...",
        fontsize=10,
    )

    sl = slot_labels or {i: f"slot {i}" for i in slots}
    for ri, mode in enumerate(modes):
        for ci, (slot, metric) in enumerate(
            [(s, m) for s in slots for m in metrics]
        ):
            ax = axes[ri, ci]
            entry = correlations[mode][str(slot)][metric]
            s = np.asarray(entry["scores"], dtype=float)
            p = np.asarray(entry["projections"], dtype=float)
            rho = entry["rho"]
            n = entry["n"]
            # Phase 3.5: when names carry kind suffixes (entity_ids,
            # static mode), colour-code so the operator can see at a
            # glance whether trait or role outliers are driving rho.
            # Kind-pure runs (response mode, bare names) get the
            # default single colour.
            names = entry.get("names") or []
            kinds = [
                parse_entity_id(n_).kind if is_entity_id(n_) else None
                for n_ in names
            ]
            if any(k is not None for k in kinds):
                colors = [kind_color(k) if k else "#777777" for k in kinds]
                ax.scatter(s, p, s=10, alpha=0.65, edgecolor="none",
                           c=colors)
            else:
                ax.scatter(s, p, s=10, alpha=0.55, edgecolor="none")
            if n >= 3 and np.isfinite(rho) and np.isfinite(p).all() and p.std() > 0 and s.std() > 0:
                # Simple OLS best-fit overlay
                slope, intercept = np.polyfit(s, p, 1)
                xs = np.array([s.min() - 0.2, s.max() + 0.2])
                ax.plot(xs, slope * xs + intercept, color="black", lw=0.8, alpha=0.6)
            ax.set_title(
                f"{mode} | {sl[slot]} | {metric}\n"
                + (f"ρ={rho:.3f}, n={n}" if np.isfinite(rho) else f"n={n}"),
                fontsize=9,
            )
            ax.set_xlabel("judge score", fontsize=8)
            ax.set_ylabel("projection", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight",
                metadata=png_metadata(title=title_line, inputs=list(inputs) if inputs else None))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score roles/traits on a semantic axis with a judge LLM "
                    "and correlate with activation-space projections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Axis specification (choose one of two paths).
    axis = p.add_argument_group("axis specification")
    axis.add_argument("--pair", nargs=2, metavar=("POS_NAME", "NEG_NAME"),
                      help="Pair of role or trait names. Positive pole listed first. "
                           "Their descriptions become pole text (unless overridden), their "
                           "names become example lists, and their vector difference is the "
                           "direction.")
    axis.add_argument("--pair_type", choices=["roles", "traits"],
                      help="Required with --pair.")
    axis.add_argument("--axis_file", type=str,
                      help="Path to a .pt file (dict with 'axis' or 'vector' key, or raw tensor). "
                           "Accepted shapes: (n_slots, n_layers, hidden), (n_layers, hidden), "
                           "or (hidden,).")
    axis.add_argument("--axis_name", type=str,
                      help="Display name for the axis. Defaults to pair / file name.")
    axis.add_argument("--neg_pole", type=str,
                      help="Text description of the negative pole. The "
                           "convention is to start with 'This means [verb-ing]...' "
                           "(behavioral description). When using a spec from "
                           "infer_axis_description.py, prefer the "
                           "'neg_pole_standardized' field over 'neg_pole' (the "
                           "former is the Sonnet-rephrased version produced by "
                           "standardize_axis_spec.py).")
    axis.add_argument("--pos_pole", type=str,
                      help="Text description of the positive pole. See --neg_pole "
                           "for the convention.")
    axis.add_argument("--neg_examples", type=_csv_or_space_list, default=[],
                      help="Comma-separated names of negative-pole exemplars.")
    axis.add_argument("--pos_examples", type=_csv_or_space_list, default=[],
                      help="Comma-separated names of positive-pole exemplars.")

    # Data.
    data = p.add_argument_group("data")
    data.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing roles/vectors/*.pt and/or traits/vectors/*.pt.")
    data.add_argument("--instructions_dir", type=str, default="data",
                      help="Top-level directory with roles/instructions/*.json and "
                           "traits/instructions/*.json.")
    data.add_argument("--scores_dir", type=str,
                      help="Directory of per-entity scores JSONs; required for --score_responses.")
    data.add_argument("--responses_dir", type=str,
                      help="Directory of per-entity responses .jsonl files; required for "
                           "--score_responses.")
    data.add_argument("--layer", type=int, default=25)  # Qwen-3-32B; tuned via rho_by_layer.py.
    data.add_argument("--slot", type=str, default="all",
                      help="Token-slot index, or 'all'.")

    # Whitening.
    w = p.add_argument_group("whitening")
    w.add_argument("--whiten_K", type=int, default=DEFAULT_SOFT_K,
                   help=f"Number of top PCs to soft-scale down to "
                        f"sigma_{{K+1}}.  Default K={DEFAULT_SOFT_K} is the "
                        f"current project default (see "
                        f"results_analysis.canonical_angles.whitening."
                        f"DEFAULT_SOFT_K).  K was set to 3 historically; "
                        f"K=2 was adopted Apr 2026 after the LKM grid sweep "
                        f"showed K=2 wins consistently on no-shear baselines "
                        f"and L=2 soft-shear with K=0 is the best overall "
                        f"primary regime.")

    # Scoring modes.
    sm = p.add_argument_group("scoring modes")
    sm.add_argument("--score_descriptions", action="store_true",
                    help="Score each entity's description text.")
    sm.add_argument("--score_instructions", action="store_true",
                    help="Score each entity's concatenated 5 pos instructions.")
    sm.add_argument("--score_responses", action="store_true",
                    help="Score all score==3 model responses per entity, partitioned into "
                         "roughly equal-sized batches (one -3..+3 score per batch; "
                         "entity score = mean of batch scores).")
    sm.add_argument("--response_target_batch_size", type=int,
                    default=RESPONSE_BATCH_SIZE,
                    help="Target responses per batch for --score_responses. Actual batch "
                         "sizes will differ by at most 1 to cover all items (N items -> "
                         "max(1, round(N/target)) equal-sized batches). Default "
                         f"{RESPONSE_BATCH_SIZE} (canonical project-wide value, see "
                         "``assistant_axis.judge_batch.RESPONSE_BATCH_SIZE``); matches the "
                         "steering effect judge's default so the two score distributions "
                         "are apples-to-apples in the same judging regime; was 15 "
                         "historically, see steering judge Phase-2 plan for the rationale.")
    sm.add_argument("--all", action="store_true",
                    help="Equivalent to --score_descriptions --score_instructions --score_responses.")
    sm.add_argument("--questions_file", type=str,
                    default="data/extraction_questions.jsonl",
                    help="Canonical questions JSONL (one {question, id} per line); used "
                         "to backreference response question_index to original id "
                         "for --question_subsample_modulo.")
    sm.add_argument("--question_subsample_modulo", type=int, default=0,
                    help="LEGACY (uniform) mode. If >0, restrict --score_responses to "
                         "responses whose original question id (looked up in "
                         "--questions_file) is divisible by this number, applied "
                         "identically to every entity. With the historic Roger "
                         "response pipeline (reduce=3 from a 300-question canonical), "
                         "modulo=9 gave 1/3 of the currently-judged questions "
                         "(34 / 100). When 0 (the default), per-entity tiered "
                         "subsampling kicks in instead (see --no_subsample / "
                         "--tiered_modulo_per_chunk).")
    sm.add_argument("--no_subsample", action="store_true", default=None,
                    help="Disable all subsampling. Use every score==3 response.  "
                         "JUDGE-AWARE DEFAULT (May 2026): if not specified, defaults "
                         "to True for the OpenAI/GPT judge (cheap → always full "
                         "volume) and False for Anthropic/Haiku/Sonnet judges "
                         "(tiered 1/3 by default).  --no_subsample is FORBIDDEN "
                         "for the Sonnet judge — it's too expensive at full volume "
                         "and the project never intended to support it there.  "
                         "Pre-May-2026 default was False for everything (Bug B).")
    sm.add_argument("--tiered_modulo_per_chunk", type=int, default=3,
                    help="Chunk size M for per-entity tiered subsampling (default 3 "
                         "= 1/3 chunks). Tier 1 keeps orig_id %% M == 0, tier 2 "
                         "keeps orig_id %% M in {0,1}, tier 3 keeps everything. "
                         "Per-entity tier choice uses thresholds derived from the "
                         "default persona's response-item count N_default: drop "
                         "to tier 2 if tier-1 count < N_default/(2*M); drop to "
                         "tier 3 if tier-2 count < 2*N_default/(3*M). Ignored when "
                         "--question_subsample_modulo > 0 or --no_subsample is set.")

    # Judge.
    j = p.add_argument_group("judge")
    j.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    j.add_argument("--judge_model", type=str, default=None,
                   help="Defaults: claude-sonnet-4-6 (anthropic) / gpt-4.1-mini (openai).")
    j.add_argument("--max_tokens", type=int, default=1024,
                   help="Token budget for the judge response (must fit reasoning + SCORE line).")
    j.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 for greedy/deterministic rubric grading.")
    j.add_argument("--rps", type=float, default=10.0, help="Requests per second (token bucket).")
    j.add_argument("--batch_size", type=int, default=20, help="Concurrent in-flight calls.")
    j.add_argument("--save_every", type=int, default=40,
                   help="How many judge calls between cache flushes.")
    j.add_argument("--no_cache", action="store_true", help="Rescore everything from scratch.")

    # Output.
    o = p.add_argument_group("output")
    o.add_argument("--output_dir", type=str, required=True)
    o.add_argument("--max_entities", type=int, default=None,
                   help="Limit total entities scored (for dry runs).")

    # Rubric-version drift policy.
    rd = p.add_argument_group("rubric-version drift policy")
    rd.add_argument(
        "--strict_judge_model", action="store_true",
        help=(
            "On cache resume, if the cache's recorded judge_model differs "
            "from the current --judge_model, abort instead of dropping the "
            "whole file for a single-model rejudge.  Caches are keyed by "
            "judge family (sonnet/, gpt/, ...), not model id, so this is "
            "the only thing keeping a cohort single-model."
        ),
    )
    rd.add_argument(
        "--drop_unrecorded_judge_model", action="store_true",
        help=(
            "Treat caches with no recorded judge_model (legacy pre-Phase-6 "
            "envelopes) as drifted and rejudge them whole.  Default keeps "
            "them with a warning, because 100+ GPT and Haiku legacy caches "
            "exist and those judges have not changed.  Use when migrating a "
            "family whose model did change (Sonnet 4/4.5 -> 4.6)."
        ),
    )
    rd.add_argument(
        "--strict_rubric_version", action="store_true",
        help=(
            "On cache resume, if any cached entry's recorded "
            "rubric_version differs from the current RUBRIC_VERSION "
            "AND the (from, to) pair isn't declared equivalent for "
            "this (axis, mode, entity_id) cell in "
            "rubric_equivalences.yaml, abort with SystemExit instead "
            "of silently dropping the affected entries and rejudging "
            "them.  Use when you want the run to halt on undeclared "
            "rubric drift so you can either accept the rejudge "
            "explicitly (drop this flag) or declare equivalence via "
            "tools/mark_rubric_equivalent.py.  Default (off) is to "
            "drop-and-rejudge with a WARNING; this is safer for "
            "long-running judging sweeps that shouldn't abort "
            "mid-flight on a rubric bump."
        ),
    )

    # Surgical rejudge restriction.
    rj = p.add_argument_group("surgical rejudge restriction")
    rj.add_argument(
        "--rejudge_names", type=str, default=None,
        help=(
            "Comma- or whitespace-separated list of '<KIND>:<name>' pairs "
            "to restrict scoring to (e.g. 'ROLE:patient,TRAIT:patient'). "
            "<KIND> is R/ROLE/ROLES or T/TRAIT/TRAITS (case-insensitive). "
            "In static modes, listed entities are scored even if already "
            "in the cache (overwriting); other entries stay untouched. "
            "In response mode, scoring is also restricted to listed names. "
            "Used by Phases 5b, 5d-i, 5d-ii of the trait/role disambiguation "
            "plan."
        ),
    )

    bg = p.add_argument_group("budget tracking (Phase 4c)")
    bg.add_argument(
        "--budget_usd", type=float, default=None,
        help=(
            "Hard cost cap in USD.  When set, the run aborts cleanly "
            "(exit code 2; partial caches flushed) the first time the "
            "running cost crosses this cap.  Recommended (post-May "
            "2026 calibration on 5c.1/5d.1 full sweeps): ~**1.25× "
            "expected cost + $5 floor** as a safety net for "
            "miscalculation.  Empirical residual prediction error "
            "after the new per-call scales + roles/traits factor is "
            "~3-5%%, so 1.25× leaves ~20pt headroom for "
            "model-drift or stupid-mistake catch.  For "
            "first-of-kind judges/prompts (no canary yet), keep "
            "~1.5× + $10 until calibrated.  Default = no cap "
            "(legacy behaviour)."
        ),
    )
    bg.add_argument(
        "--expected_cost_usd", type=float, default=None,
        help=(
            "Expected cost in USD.  Advisory only — used for the "
            "actual/expected ratio in [budget] log lines and the "
            "end-of-run summary, and for canary GREEN/AMBER/RED "
            "gating.  Doesn't affect cap enforcement."
        ),
    )
    bg.add_argument(
        "--usage_json", type=str, default=None,
        help=(
            "Optional path to write a usage.json side-car at end-of-run "
            "(token totals + USD cost + cap + expected, in the "
            "BudgetTracker.as_dict() schema).  Defaults to "
            "<output_dir>/usage.json when --output_dir is set; pass "
            "this flag to override (or set to '/dev/null' to disable)."
        ),
    )

    args = p.parse_args()
    if args.all:
        args.score_descriptions = True
        args.score_instructions = True
        args.score_responses = True
    if not (args.score_descriptions or args.score_instructions or args.score_responses):
        p.error("Specify at least one of --score_descriptions / --score_instructions / "
                "--score_responses, or use --all.")
    if args.judge_model is None:
        args.judge_model = _default_model_for_provider(args.provider)
    args.rejudge_names = _parse_rejudge_names(args.rejudge_names)
    return args


def _parse_rejudge_names(arg: Optional[str]) -> Optional[List[Tuple[str, str]]]:
    """Parse the --rejudge_names CLI value into a list of (kind, name) pairs.

    Format: comma- or whitespace-separated list of ``<KIND>:<name>``
    entries, where ``<KIND>`` is any spelling
    :func:`assistant_axis.entity_id.kind_long` accepts (``R``, ``T``,
    ``role``, ``roles``, ``trait``, ``traits``; case-insensitive).

    Returns ``None`` when the arg is unset (no filtering); otherwise a
    list of ``(kind_long, name)`` tuples ready for membership checks.

    Names are run through :func:`normalize_to_file_name` so that
    display-form input (hyphens, apostrophes, capitals, e.g.
    ``R:devil's-advocate``) is silently coerced to file-name form
    (``devils_advocate``); a WARNING is logged whenever a conversion
    happens so users notice their input was massaged.  Spaces are not
    possible inside an entry because the comma/whitespace splitter
    above tokenises on them.

    Raises:
        SystemExit: On any malformed entry, with a clear example.
    """
    if arg is None:
        return None
    raw = re.split(r"[,\s]+", arg.strip())
    raw = [p for p in raw if p]
    if not raw:
        return None
    out: List[Tuple[str, str]] = []
    coerced: List[Tuple[str, str]] = []  # (raw_name, normalized_name)
    for p in raw:
        if ":" not in p:
            raise SystemExit(
                f"--rejudge_names: malformed entry {p!r}; expected "
                f"'<KIND>:<name>' (e.g. 'ROLE:patient' or 'T:stoic')."
            )
        k_raw, name = p.split(":", 1)
        try:
            kind = kind_long(k_raw)
        except ValueError as e:
            raise SystemExit(
                f"--rejudge_names: {p!r}: {e}"
            ) from e
        if not name:
            raise SystemExit(
                f"--rejudge_names: malformed entry {p!r}; missing entity name."
            )
        normalized = normalize_to_file_name(name)
        if normalized != name:
            coerced.append((name, normalized))
        out.append((kind, normalized))
    if coerced:
        head = "; ".join(
            f"{raw_n!r} -> {norm_n!r}" for raw_n, norm_n in coerced[:5]
        ) + ("..." if len(coerced) > 5 else "")
        logger.warning(
            f"--rejudge_names: coerced {len(coerced)} display-form entries "
            f"to file-name form ({head}).  Future invocations should pass "
            f"file-name form directly; see AGENT_NOTES "
            f"'File-name vs display-name convention'."
        )
    return out


def _resolve_rejudge_names_against_scorable(
    rejudge_names: Sequence[Tuple[str, str]],
    *,
    scorable: Sequence[Tuple[str, str]],
    corpus_entities: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Validate ``--rejudge_names`` entries against ``corpus`` and ``scorable``.

    Returns the subset of ``rejudge_names`` that remains scorable on this
    axis: entries are kept iff they are in ``scorable``; entries that
    are in the full ``corpus_entities`` but pole-skipped from
    ``scorable`` are silently dropped (with an INFO log line).

    Two cases are distinguished:

    1. Entry is in the corpus but pole-skipped on this axis — log and
       silently drop.  Expected when a 5d-ii RP-depleted trait happens
       to be the pole pair of one of the 12 v2 axes (e.g.
       ``T:harmful`` on ``harmless_vs_harmful``); the same
       axis-invariant rejudge list runs on every axis, so 3 of 11
       axes hit one such conflict each and we just skip.

    2. Entry is NOT in the corpus at all — fatal, almost certainly a
       typo or stale corpus reference.

    Raises:
        SystemExit: When at least one ``rejudge_names`` entry is not
            present in ``corpus_entities`` (a likely typo).
    """
    scorable_set = set(scorable)
    corpus_set = set(corpus_entities)
    not_in_corpus = [pair for pair in rejudge_names if pair not in corpus_set]
    if not_in_corpus:
        head = ", ".join(
            f"{kind_long(et)[0].upper()}:{n}" for et, n in not_in_corpus[:8]
        )
        raise SystemExit(
            f"--rejudge_names: {len(not_in_corpus)} entries not in the "
            f"corpus at all (typo or stale reference): {head}"
            + ("..." if len(not_in_corpus) > 8 else "")
            + f".  corpus has {len(corpus_set)} entries; "
            "first few: "
            + ", ".join(f"{et}:{n}" for et, n in sorted(corpus_set)[:6])
            + ".  --rejudge_names is the new flag from Phase 4 of the "
            "trait/role disambiguation plan."
        )
    pole_skipped = [pair for pair in rejudge_names if pair not in scorable_set]
    if pole_skipped:
        logger.info(
            f"--rejudge_names: {len(pole_skipped)} entries silently "
            "dropped because they are pole-pair excluded on this axis: "
            + ", ".join(
                f"{kind_long(et)[0].upper()}:{n}" for et, n in pole_skipped
            )
        )
    kept = [pair for pair in rejudge_names if pair in scorable_set]
    logger.info(
        f"--rejudge_names: scoring restricted to {len(kept)} "
        f"entities: " + ", ".join(
            f"{kind_long(et)[0].upper()}:{n}" for et, n in kept[:10]
        ) + ("..." if len(kept) > 10 else "")
    )
    return kept


def _csv_or_space_list(s: str) -> List[str]:
    if not s:
        return []
    # Accept comma- or whitespace-separated.
    parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
    return parts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    # Bug B (May 2026): resolve judge-aware subsampling defaults
    # *before* anything else so the resolved no_subsample value is
    # visible to provenance (judge_extras, config.json, _provenance.notes)
    # and downstream branches.
    _apply_judge_aware_subsample_defaults(args)

    # Phase 4c (May 2026): build a BudgetTracker that ticks on every
    # judge call and aborts the run cleanly the moment the running
    # cost crosses --budget_usd.  Construct unconditionally — without
    # --budget_usd it acts as a free token-counter for the
    # end-of-run summary + usage.json side-car.
    tracker = BudgetTracker(
        totals=UsageTotals(model=args.judge_model),
        budget_usd=args.budget_usd,
        expected_cost_usd=args.expected_cost_usd,
    )
    if args.budget_usd is not None:
        logger.info(
            f"[budget] cap=${args.budget_usd:.2f}; "
            f"expected=${args.expected_cost_usd or 0:.2f}; "
            f"model={args.judge_model}"
        )

    # Peek at one vector file to figure out n_slots / hidden_dim.
    data_dir = Path(args.data_dir)
    probe_paths = list((data_dir / "roles" / "vectors").glob("*.pt")) \
        + list((data_dir / "traits" / "vectors").glob("*.pt"))
    probe_paths = [p for p in probe_paths if p.name != "default.pt"]
    if not probe_paths:
        raise SystemExit(f"No vectors found under {data_dir}/(roles|traits)/vectors/")
    probe = _load_vector_file(probe_paths[0]).float()
    if probe.dim() != 3:
        raise SystemExit(f"Expected (n_slots, n_layers, hidden) vectors; got shape {tuple(probe.shape)}")
    n_slots, n_layers, hidden_dim = probe.shape
    logger.info(f"n_slots={n_slots} n_layers={n_layers} hidden={hidden_dim}; "
                f"using layer={args.layer}")
    if not 0 <= args.layer < n_layers:
        raise SystemExit(f"--layer {args.layer} out of range [0, {n_layers})")
    slots = _slot_list_from_arg(args.slot, n_slots)

    # Build the axis spec and load corpus.
    axis_spec = resolve_axis(args, n_slots, hidden_dim, args.layer)
    corpus = load_corpus(args, args.layer)

    # Which entities are scorable?  All corpus entities minus the **pole
    # pair** names (axis-defining; degenerate to judge).  Listed examples
    # ARE scorable -- their per-call rubric strips them from the example
    # list (see build_static_prompt), so judging an example entity is no
    # longer self-referential.
    exclusion_set = set(axis_spec.exclusions)            # full set: still used to hold examples out of the whitener pool below
    pole_skip_set = set(axis_spec.pole_pair_names)        # names actually skipped from scoring + ρ
    scorable: List[Tuple[str, str]] = [
        (et, n) for (et, n) in corpus.entities if n not in pole_skip_set
    ]
    if args.max_entities is not None:
        scorable = scorable[:args.max_entities]

    if args.rejudge_names:
        args.rejudge_names = _resolve_rejudge_names_against_scorable(
            args.rejudge_names,
            scorable=scorable,
            corpus_entities=list(corpus.entities),
        )

    logger.info(
        f"Scorable entities: {len(scorable)} "
        f"(skipped {len(corpus.entities) - len(scorable)} pole-pair names "
        f"or by --max_entities; example entities ARE scored, with their "
        f"name stripped from each per-call rubric)"
    )

    # Projections first (no API calls) so we can save them even if scoring is slow.
    # Per-entity LOO pool: each entity's projection uses a pool built from
    # every other corpus entity except itself (and the pole-pair names,
    # which are axis-defining and degenerate to include).  At whiten_K=0
    # the whitener is identity and the per-entity SVD is short-circuited.
    projections = compute_projections(
        corpus=corpus, axis_spec=axis_spec, whiten_K=args.whiten_K,
        slots=slots, pole_pair_names=axis_spec.pole_pair_names,
    )
    proj_out = {str(slot): per_slot for slot, per_slot in projections.items()}
    # Projections is computed from the corpus, no API calls — usage
    # is always zero here, but stamp it for schema consistency.
    _save_json(
        Path(args.output_dir) / "projections.json",
        proj_out,
        inputs=_build_axis_judge_inputs(args, mode="projections"),
        title=f"axis_judge_correlation:projections:{axis_spec.axis_name}",
        schema_version=2,
        notes={"usage": tracker.as_dict()},
    )

    # Save config up-front.  ``cohort_kind`` records the kind-pure
    # cohort the responses_dir points at (when in response mode);
    # for static-mode runs (no responses_dir) it's None.  This fixes
    # the historical "config.json's args.pair_type always reads as
    # the axis pair_type, regardless of which kind is being judged"
    # bug — the new ``cohort_kind`` field is unambiguous.
    cohort_kind: Optional[str] = None
    if args.responses_dir:
        cand = Path(args.responses_dir).parent.name
        if cand in ("roles", "traits"):
            cohort_kind = cand
    rejudge_for_config = (
        sorted(f"{kind_long(et)[0].upper()}:{n}" for et, n in args.rejudge_names)
        if args.rejudge_names else None
    )
    config = {
        "args": {k: v for k, v in vars(args).items()},
        "n_slots": n_slots, "n_layers": n_layers, "hidden_dim": hidden_dim,
        "slots": slots, "layer": args.layer, "whiten_K": args.whiten_K,
        "axis_name": axis_spec.axis_name,
        "axis_source": axis_spec.source_description,
        "neg_pole": axis_spec.neg_pole, "pos_pole": axis_spec.pos_pole,
        "neg_examples": axis_spec.neg_examples, "pos_examples": axis_spec.pos_examples,
        "exclusions": axis_spec.exclusions,
        "n_scorable": len(scorable),
        "cohort_kind": cohort_kind,
        "rejudge_names": rejudge_for_config,
    }
    _save_json(Path(args.output_dir) / "config.json", config)

    # Run each requested scoring mode.
    scores_for_corr: Dict[str, Dict[str, float]] = {}
    try:
        if args.score_descriptions:
            d = await score_static_mode(
                "descriptions", corpus, axis_spec, scorable, args,
                tracker=tracker,
            )
            scores_for_corr["descriptions"] = {k: float(v) for k, v in d.items()}
        if args.score_instructions:
            d = await score_static_mode(
                "instructions", corpus, axis_spec, scorable, args,
                tracker=tracker,
            )
            scores_for_corr["instructions"] = {k: float(v) for k, v in d.items()}
        if args.score_responses:
            resp_cache = await score_responses_mode(
                axis_spec, scorable, args, tracker=tracker,
            )
    except BudgetExceededError as e:
        # Cleanup path: partial caches were already flushed by
        # score_*_mode's save_every loop the moment the cap-crossing
        # call returned.  Log a final budget summary and exit with
        # code 2 (= cap exceeded; distinct from generic CLI error).
        logger.error(str(e))
        _flush_budget_artifacts(args, tracker)
        raise SystemExit(2) from e
    else:
        # On clean completion, also write the side-car / log a
        # one-line final summary so the operator sees the bill.
        _flush_budget_artifacts(args, tracker)
    if args.score_responses:
        # Response cache keys are bare names (kind-pure cohort dir).
        # For mixed-kind correlation against per_slot (which is keyed
        # by disambiguated entity_id), promote the bare names with the
        # cohort_kind discriminator so the lookups in compute_correlations
        # match.
        if cohort_kind:
            means = {
                entity_id(n, cohort_kind): float(d["mean_score"])
                for n, d in resp_cache.items()
                if d.get("mean_score") is not None
            }
        else:
            means = {
                n: float(d["mean_score"]) for n, d in resp_cache.items()
                if d.get("mean_score") is not None
            }
        scores_for_corr["responses"] = means

    # Compute correlations and output.
    # ρ calc skips only pole-pair names (which weren't judged anyway);
    # listed examples DO contribute ρ data points now.
    correlations = compute_correlations(
        scores_by_mode=scores_for_corr, projections=projections,
        slots=slots, excluded_set=pole_skip_set,
    )
    _save_json(
        Path(args.output_dir) / "correlations.json",
        correlations,
        inputs=_build_axis_judge_inputs(args, mode="correlations"),
        title=f"axis_judge_correlation:correlations:{axis_spec.axis_name}",
        schema_version=2,
        notes={"usage": tracker.as_dict()},
    )

    # Make plot.
    make_plot(
        correlations=correlations, slots=slots, axis_spec=axis_spec,
        output_path=Path(args.output_dir) / "correlation_plot.png",
        inputs=_build_axis_judge_inputs(args, mode="correlations"),
    )

    # Print a brief summary.
    print("\n=== Spearman rho summary ===")
    print(f"axis: {axis_spec.axis_name}")
    for mode in correlations:
        print(f"\n{mode}:")
        for slot in slots:
            r = correlations[mode][str(slot)]
            print(
                f"  slot {slot}: "
                f"raw rho={r['raw']['rho']:+.3f} (n={r['raw']['n']}), "
                f"wht rho={r['whitened']['rho']:+.3f} (n={r['whitened']['n']})"
            )


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
