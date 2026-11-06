"""Single-cell steering sweep runner.

One call to :func:`run_steering_cell` runs the sweep for a single
``(slot, layer, sign)`` work item: builds a directional sequence of
steering strengths (geometric: ±weakest, ±weakest*mult, ..., up to
±max), generates batched responses to a fixed question list at each
strength, writes records to ``records.jsonl``, and consults a pluggable
:class:`JudgeDispatcher` for both per-record judging side-effects and an
optional incoherence-based early-termination signal.

Phase 1 of the steering sweep plan ships with
:class:`NoOpJudgeDispatcher` as the default dispatcher: records land on
disk with their ``judges`` field left null, and
``should_stop_at(strength)`` always returns False so the sweep runs to
its configured max.  Phase 2 will provide a real Sonnet-backed
dispatcher implementing the same protocol.

Baselines (strength=0 responses) are computed once per
``(persona, question)`` via :func:`compute_baselines` and shared across
all cells via the shared ``baselines/records.jsonl`` file at the
experiment root.  Cell sweeps therefore start at ±weakest, not 0.

Restart semantics
-----------------

On entry, :func:`run_steering_cell` reads any existing
``records.jsonl`` and skips ``(strength, question_idx)`` pairs already
present.  If ``summary.json`` exists with a non-null
``stopped_at_strength``, the sweep is considered complete and the
function returns immediately with a no-op (subsequent judge passes
remain a Phase 2 concern).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch

from .atomic_io import (
    atomic_write_text, read_jsonl_with_retry, read_text_with_retry, write_jsonl,
)
from .steering import ActivationSteering
from .steering_judges import (
    DEFAULT_SKIP_THRESHOLD,
    JudgeDispatcher,
    NoOpJudgeDispatcher,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sweep schedule
# ---------------------------------------------------------------------------

def directional_schedule(
    sign: int,
    *,
    weakest: float,
    max_strength: float,
    multiplier: float,
) -> List[float]:
    """Geometric schedule of strengths along one sign of the sweep.

    Used by ``scan_mode="legacy_unidirectional"`` only.  The new
    default (``scan_mode="bidirectional"``) uses
    :class:`BidirectionalCursor` instead.

    Returns absolute values in increasing magnitude order; the caller
    multiplies by ``sign`` when applying the steering coefficient.
    Per Roger's notebook usage, the schedule is geometric:

        [weakest, weakest * multiplier, weakest * multiplier**2, ...,
         up to and including the largest <= max_strength]

    With weakest=1.0, multiplier=1.189 (≈ 2**(1/4)), max=64 we get
    30 strengths spaced ~quarter-octave apart, matching the notebook.

    `sign` is taken as informational here (the function doesn't use it
    directly) but is required so the caller can't accidentally lose
    it; we return abs values to keep the schedule symmetric across signs.
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    if weakest <= 0 or max_strength <= 0 or multiplier <= 1:
        raise ValueError(
            f"need weakest>0, max_strength>0, multiplier>1; "
            f"got weakest={weakest}, max_strength={max_strength}, multiplier={multiplier}"
        )
    if max_strength < weakest:
        raise ValueError(f"max_strength ({max_strength}) < weakest ({weakest})")

    schedule: List[float] = []
    s = weakest
    # Round to a stable precision so JSON keys / equality checks behave
    # well across runs, matching the notebook's truncated decimals
    # (1.189, 1.414, ...).  3 decimal places is plenty.
    while s <= max_strength * (1.0 + 1e-9):
        schedule.append(round(s, 6))
        s *= multiplier
    return schedule


# ---------------------------------------------------------------------------
# Bidirectional schedule cursor (new default scan_mode)
# ---------------------------------------------------------------------------


class BidirectionalCursor:
    """Lazily-extensible bidirectional geometric strength cursor.

    Used by ``scan_mode="bidirectional"`` (the new default 2026-05-14).
    The legacy ``directional_schedule`` above remains for
    ``scan_mode="legacy_unidirectional"``.

    The cursor produces strengths in two directions from a centre anchor
    ``s_init = weakest * multiplier**start_steps_up``.  Per-cell starts
    come from :mod:`assistant_axis.sweep_start_heuristics` (May 2026:
    replaces the historical flat ``start_steps_up=2`` with a per-
    ``(positions_mode, slot, layer)`` lookup) so each cell starts in
    its empirically-determined productive band.  ``start_steps_up`` may
    be negative -- the real safety floor for ``s_init`` is
    ``min_strength``, not zero.

    The UP direction terminates when the next step would exceed
    ``max_strength`` (default 64.0).  Empirically the 14 May-2026
    production experiments never exceeded ~16, so the cap is mainly
    a safety bound -- incoherence in the judging loop is the real
    UP stop signal.

    The DOWN direction terminates when the next step would fall below
    ``min_strength`` (default 0.125, i.e. 3 octaves below the historical
    weakest=1.0 of the unidirectional schedule).  The eff-stop on the
    judging loop is the real DOWN stop signal; ``min_strength`` is the
    safety floor for very-strong-steering cells where ``|effect|``
    stays above the eff-threshold all the way down.

    Cursor state is INTERNAL to the instance.  For restart support
    after a crash, the runner re-emits next_up()/next_down() calls and
    skips strengths already present in records.jsonl -- the cursor is
    deterministic, so re-emitting the same sequence on resume is safe.
    Geometric rounding to 6 decimal places matches the legacy schedule
    and keeps JSON dict keys / set membership stable.
    """

    def __init__(
        self,
        *,
        weakest: float,
        max_strength: float,
        min_strength: float,
        multiplier: float,
        start_steps_up: int = 2,
    ) -> None:
        if weakest <= 0 or max_strength <= 0 or min_strength <= 0:
            raise ValueError(
                f"need weakest>0, max_strength>0, min_strength>0; "
                f"got weakest={weakest}, max_strength={max_strength}, "
                f"min_strength={min_strength}"
            )
        if multiplier <= 1:
            raise ValueError(f"need multiplier>1; got multiplier={multiplier}")
        if max_strength < weakest:
            raise ValueError(
                f"max_strength ({max_strength}) < weakest ({weakest})"
            )
        if min_strength > weakest:
            raise ValueError(
                f"min_strength ({min_strength}) > weakest ({weakest}); "
                f"min_strength must be at or below the historical lower "
                f"floor"
            )
        self._mult = float(multiplier)
        self._max = float(max_strength)
        self._min = float(min_strength)
        # Centre anchor.  Round to match the schedule's stable-precision
        # convention so equality checks across restart sessions work.
        #
        # Negative ``start_steps_up`` is allowed and means
        # ``s_init = weakest * multiplier ** -N`` (i.e. starting BELOW
        # ``weakest``).  This is required by
        # ``sweep_start_heuristics.OVERRIDES`` for cells where the
        # coherence cliff is close to ``weakest`` itself (e.g. all-mode
        # slot=7 layer=49 on Qwen-3-32B: coh tripped at s≈1.0 in ~25%
        # of axes pre-fix).  The actual lower safety bound is
        # ``min_strength`` -- enforced just below.
        self.s_init: float = round(
            float(weakest) * (multiplier ** start_steps_up), 6
        )
        if self.s_init < min_strength:
            raise ValueError(
                f"resolved s_init={self.s_init} < min_strength={min_strength}; "
                f"need (weakest * multiplier ** start_steps_up) >= min_strength.  "
                f"Got weakest={weakest}, multiplier={multiplier}, "
                f"start_steps_up={start_steps_up}.  Lower ``min_strength`` if "
                f"a sub-min start is intended."
            )
        # Cursors track the LAST strength emitted in each direction
        # (or s_init if nothing has been emitted yet on that side).
        self._last_up: float = self.s_init
        self._last_down: float = self.s_init
        # Once a direction returns None it stays exhausted forever
        # (mirrors the "blocked is sticky" semantics of the stop
        # conditions in the runner's state machine).
        self._up_exhausted: bool = False
        self._down_exhausted: bool = False

    def next_up(self) -> Optional[float]:
        """Return the next strength above the last UP cursor position,
        or None if the cursor would exceed ``max_strength``.

        Idempotent in the failure case -- once exhausted, subsequent
        calls keep returning None without advancing state.
        """
        if self._up_exhausted:
            return None
        nxt = round(self._last_up * self._mult, 6)
        if nxt > self._max * (1.0 + 1e-9):
            self._up_exhausted = True
            return None
        self._last_up = nxt
        return nxt

    def next_down(self) -> Optional[float]:
        """Return the next strength below the last DOWN cursor position,
        or None if the cursor would fall below ``min_strength``.

        Idempotent in the failure case -- once exhausted, subsequent
        calls keep returning None without advancing state.
        """
        if self._down_exhausted:
            return None
        nxt = round(self._last_down / self._mult, 6)
        if nxt < self._min * (1.0 - 1e-9):
            self._down_exhausted = True
            return None
        self._last_down = nxt
        return nxt

    @property
    def up_exhausted(self) -> bool:
        return self._up_exhausted

    @property
    def down_exhausted(self) -> bool:
        return self._down_exhausted


# ---------------------------------------------------------------------------
# Position-mode kwargs
# ---------------------------------------------------------------------------

_SUPPORTED_POSITION_MODES_V1 = {"all", "prefill_only"}
_DEFERRED_POSITION_MODES = {
    "header_matched", "system_only", "user_only", "system_user_only", "last",
}


def _build_position_kwargs(positions_mode: str,
                           input_ids: Optional[torch.Tensor] = None,
                           **kwargs) -> Dict[str, Any]:
    """Build the ActivationSteering positions/region kwargs for v1.

    V1 supports `"all"` and `"prefill_only"`.  Other modes raise so the
    runner doesn't silently fall through to the wrong behaviour; the
    expected modes for v2 are listed in the error message.
    """
    if positions_mode in _SUPPORTED_POSITION_MODES_V1:
        return {"positions": positions_mode}
    if positions_mode in _DEFERRED_POSITION_MODES:
        raise NotImplementedError(
            f"positions_mode={positions_mode!r} is deferred to a later "
            f"plan; v1 supports {_SUPPORTED_POSITION_MODES_V1}."
        )
    raise ValueError(
        f"unknown positions_mode={positions_mode!r}; "
        f"v1 supports {_SUPPORTED_POSITION_MODES_V1}"
    )


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _build_conversations(
    persona_system_prompt: str,
    questions: Sequence[str],
) -> List[List[Dict[str, str]]]:
    return [
        [
            {"role": "system", "content": persona_system_prompt},
            {"role": "user", "content": q},
        ]
        for q in questions
    ]


def _tokenize_batch(
    tokenizer,
    conversations: Sequence[Sequence[Dict[str, str]]],
    *,
    model_name: str = "",
):
    """Apply chat template + tokenize a list of conversations as a left-padded batch.

    Left-padding is required for `model.generate(...)`: with right-padding,
    generated tokens would start at different absolute positions for
    each batch element, producing garbled output.
    """
    chat_template_kwargs: Dict[str, Any] = {}
    if "qwen" in model_name.lower() or "qwen" in getattr(tokenizer, "name_or_path", "").lower():
        chat_template_kwargs["enable_thinking"] = False

    prompts = [
        tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs
        )
        for conv in conversations
    ]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = prev_padding_side
    return inputs


# ---------------------------------------------------------------------------
# Truncation detection + decode helper
# ---------------------------------------------------------------------------


def _decode_with_truncation_marker(
    gen_ids,  # torch.Tensor of shape (max_new_tokens,)
    tokenizer,
) -> Tuple[str, int, bool]:
    """Decode the generated ids and detect whether generation was truncated
    by ``max_new_tokens`` (vs naturally stopped at a stop token).

    Returns ``(response, n_tokens, truncated)``:

    - ``response``: decoded text with ``skip_special_tokens=True``.  When
      truncated, a trailing " …" sentinel is appended so both effect-judges
      and human readers see at a glance that the model was still mid-output
      when cut off.  The sentinel is one char of context, two glyphs at the
      end (`` … ``), which is robust to tokenizer round-trips and unlikely
      to collide with content the model would itself emit.
    - ``n_tokens``: count of non-pad tokens in the generated slice.
    - ``truncated``: True iff zero pad tokens are present in the generated
      slice -- meaning the model never reached any stop token and the
      ``max_new_tokens`` cap was hit.

    The detection rests on the HuggingFace ``model.generate`` convention
    that once a sequence in a batch emits a stop token (per
    ``eos_token_id`` in the generation config, which on Qwen3 maps to
    ``<|im_end|>`` for chat completions), the remaining positions in that
    sequence's row of the output tensor are filled with ``pad_token_id``.
    A generated slice containing at least one pad therefore indicates the
    model naturally stopped; a slice with zero pad tokens indicates the
    cap was hit.

    Edge case: if the very last token generated happened to be a stop
    token (no pad fill needed because we were at exactly ``max_new_tokens``
    already), this heuristic would mis-classify as truncated.  This is
    rare and the "truncated" label is arguably correct -- the model was
    out of budget either way.  May 2026 review of architect_ecocentric_v2
    + chef_helpful_v2 found that nearly all responses that hit
    ``max_new_tokens=256`` were genuinely mid-sentence at the cap.
    """
    pad = tokenizer.pad_token_id
    n_pad = int((gen_ids == pad).sum().item())
    n_tok = int(gen_ids.shape[0] - n_pad)
    truncated = (n_pad == 0)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if truncated:
        text = text + " …"
    return text, n_tok, truncated


# ---------------------------------------------------------------------------
# Records I/O
# ---------------------------------------------------------------------------

def _read_existing_records(records_path: Path) -> List[Dict[str, Any]]:
    """Read records.jsonl with NFS-flake retries.  Empty list if file absent."""
    if not records_path.exists():
        return []
    return read_jsonl_with_retry(records_path, logger_obj=logger)


def _read_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    """Read summary.json with NFS retries; return None if absent or corrupt.

    A corrupt summary.json is treated as "no summary" rather than fatal:
    the runner will rebuild from records.jsonl on next pass.
    """
    if not summary_path.exists():
        return None
    try:
        text = read_text_with_retry(summary_path, logger_obj=logger)
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"summary.json: corrupt at {summary_path}: {e}; "
                       f"treating as missing")
        return None
    except OSError as e:
        # Read retries already exhausted by read_text_with_retry; this
        # path is effectively unreachable since it would have re-raised.
        logger.error(f"summary.json: read failed at {summary_path}: {e}")
        return None


def _write_summary(summary_path: Path, summary: Dict[str, Any]) -> None:
    atomic_write_text(json.dumps(summary, indent=2) + "\n", summary_path,
                      logger_obj=logger)


def _strength_records_complete(
    recs: List[Dict[str, Any]],
    expected_K: int,
    *,
    skip_threshold: float = 1.5,
) -> bool:
    """Return True iff a strength's records on disk are fully judged.

    A "complete" strength has all ``expected_K`` questions generated
    AND each record carries fully-resolved judges:

    * ``coherence.score`` is non-None (coherence is the gating
      dimension and is always judged, never skipped).
    * ``persona.score`` is non-None OR
      (``persona.skipped_due_to_strength_mean_coh`` is True AND
      the stamped ``strength_mean_coh >= skip_threshold``).
    * ``effect.combined`` is non-None OR (same skipped-condition for
      effect).

    The ``skip_threshold`` check is what lets the resume gap-fill
    detect "previously skipped under the OLD skip_threshold but
    should have been judged under the current threshold" -- e.g.
    pre-2026-05-17 sweeps skipped rp+effect at ``mean_coh > 1.0``;
    after the runtime bump to 1.5 a back-fill restart should
    re-judge strengths whose ``strength_mean_coh`` lands in the
    band ``[1.0, 1.5)``.  The completeness check flags those as
    incomplete so the gap-fill regenerates + re-dispatches.

    Used by the bidirectional resume path to find kill-induced gaps
    (partial generation OR judging-never-ran).  Mid-strength kills
    can leave a record with ``judges = {"coherence": None, "persona":
    None, "effect": None}`` (the empty placeholder filled in by
    ``_generate_and_dispatch_strength`` after response generation but
    before judging dispatches); those records flag the strength as
    incomplete and the resume gap-fill regenerates them.
    """
    if len(recs) != expected_K:
        return False
    for r in recs:
        j = r.get("judges")
        if not isinstance(j, dict):
            return False
        coh = j.get("coherence")
        if not isinstance(coh, dict) or coh.get("score") is None:
            return False
        # Pull the strength's recorded mean coh (stamped by the
        # coherence-future writeback).  Used to decide whether the
        # "skipped" flag was correctly set under the CURRENT
        # skip_threshold or whether the strength should be re-judged.
        smc = j.get("strength_mean_coh")
        try:
            smc_f = float(smc) if smc is not None else None
        except (TypeError, ValueError):
            smc_f = None

        persona = j.get("persona") or {}
        if not isinstance(persona, dict):
            return False
        if persona.get("score") is None:
            # Either skipped or judging failed.  Must be explicitly
            # skipped AND the mean_coh stamp must justify skipping
            # under the current threshold.
            if not persona.get("skipped_due_to_strength_mean_coh"):
                return False
            if smc_f is None or smc_f < skip_threshold:
                # Stamp says this strength SHOULD be judged under
                # the current threshold; was skipped under an older
                # (lower) threshold.  Treat as incomplete so the
                # gap-fill regenerates + rejudges.
                return False

        effect = j.get("effect") or {}
        if not isinstance(effect, dict):
            return False
        if effect.get("combined") is None:
            if not effect.get("skipped_due_to_strength_mean_coh"):
                return False
            if smc_f is None or smc_f < skip_threshold:
                return False
    return True


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def compute_baselines(
    model,
    tokenizer,
    *,
    persona_system_prompt: str,
    questions: Sequence[str],
    output_dir: Union[str, Path],
    batch_size: int = 8,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.0,
    model_name: str = "",
) -> List[Dict[str, Any]]:
    """Generate baseline (no-steering) responses for every question.

    Stored once per ``(persona, question)`` at
    ``output_dir/baselines/records.jsonl``; cell sweeps consume them by
    ``question_idx`` when assembling judge prompts in Phase 2 (and as a
    diff reference for cherrypicking).

    Idempotent: existing baseline records.jsonl entries are skipped.
    """
    output_dir = Path(output_dir)
    baselines_dir = output_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    records_path = baselines_dir / "records.jsonl"

    existing = _read_existing_records(records_path)
    have_idx = {r["question_idx"] for r in existing}

    # Sentinel file that other GPU workers running cell items poll on
    # before constructing their dispatchers.  Touched both here (the
    # "everything already on disk" fast path) and after the generation
    # loop completes; cell-side wait makes the multi-GPU race
    # (worker A still computing baselines while worker B picks up a
    # cell for the same experiment) safe.
    sentinel_path = baselines_dir / ".complete"

    todo = [(i, q) for i, q in enumerate(questions) if i not in have_idx]
    if not todo:
        logger.info(f"[baselines] all {len(questions)} already present at {records_path}")
        sentinel_path.touch(exist_ok=True)
        return existing

    logger.info(
        f"[baselines] computing {len(todo)} of {len(questions)} baseline responses"
    )

    new_records: List[Dict[str, Any]] = []
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start:batch_start + batch_size]
        batch_questions = [q for _, q in batch]
        batch_idx = [i for i, _ in batch]
        conversations = _build_conversations(persona_system_prompt, batch_questions)
        inputs = _tokenize_batch(tokenizer, conversations, model_name=model_name)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - t0

        for j, (q_idx, q) in enumerate(zip(batch_idx, batch_questions)):
            gen_ids = outputs[j, prompt_len:]
            response, n_tok, truncated = _decode_with_truncation_marker(
                gen_ids, tokenizer
            )
            rec = {
                "strength": 0.0,
                "sign": 0,
                "question_idx": q_idx,
                "question": q,
                "response": response,
                "n_tokens": n_tok,
                "truncated": truncated,
                "judges": {"coherence": None, "persona": None, "effect": None},
                "timing": {"gen_s": elapsed / max(1, len(batch))},
            }
            existing.append(rec)
            new_records.append(rec)

        # Flush after each batch so a crash doesn't lose more than a batch
        write_jsonl(existing, records_path, logger_obj=logger)

    sentinel_path.touch(exist_ok=True)
    logger.info(
        f"[baselines] wrote {len(new_records)} new records to {records_path}; "
        f"touched sentinel {sentinel_path.name}"
    )
    return existing


# ---------------------------------------------------------------------------
# The main per-cell runner
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    """Outcome of running a single (slot, layer, sign) cell."""
    stopped_at_strength: Optional[float] = None
    reason: str = "completed"     # "completed" | "incoherent" | "already_done"
    n_records: int = 0
    n_strengths_swept: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)


def run_steering_cell(
    model,
    tokenizer,
    *,
    axis_vector: torch.Tensor,
    slot: int,
    layer: int,
    sign: int,
    strengths: Optional[Sequence[float]] = None,
    persona_system_prompt: str,
    questions: Sequence[str],
    output_dir: Union[str, Path],
    batch_size: int = 8,
    max_new_tokens: int = 512,
    positions_mode: str = "all",
    do_sample: bool = False,
    temperature: float = 0.0,
    judge_dispatcher: Optional[JudgeDispatcher] = None,
    model_name: str = "",
    coh_stop_threshold: float = 1.5,
    coh_stop_consecutive: int = 2,
    scan_mode: Literal["bidirectional", "legacy_unidirectional"] = "bidirectional",
    eff_stop_threshold: float = 1.0 / 3.0,
    eff_stop_consecutive: int = 2,
    weakest_strength: float = 1.0,
    max_strength: float = 64.0,
    multiplier: float = 1.189,
    min_strength: float = 0.125,
    start_strength_multiplier_steps: int = 2,
) -> CellResult:
    """Run the steering sweep for one (slot, layer, sign) cell.

    Parameters
    ----------
    model, tokenizer
        Already-loaded HuggingFace model + tokenizer on the target GPU.
    axis_vector
        1-D steering direction, shape (hidden,).  Caller pre-extracts
        from ``axis_raw[slot, layer]``.
    slot, layer, sign
        Cell coordinates.  Recorded in each output record so cross-cell
        analysis can locate every record's origin.
    strengths
        LEGACY ONLY (used iff ``scan_mode="legacy_unidirectional"``):
        absolute-value strength schedule in increasing order (caller
        multiplies by ``sign`` internally; pass output of
        :func:`directional_schedule`).  In bidirectional mode (the
        2026-05-14 default) the schedule is built lazily from
        ``weakest_strength`` / ``max_strength`` / ``min_strength`` /
        ``multiplier`` / ``start_strength_multiplier_steps`` and
        ``strengths`` is ignored.
    persona_system_prompt
        System prompt fixed for this experiment (one persona per run in
        v1).
    questions
        Frozen question list for the experiment.
    output_dir
        Cell-specific directory, e.g.
        ``/workspace/.../{experiment}/s{slot}_l{layer}_{sign}``.
    batch_size, max_new_tokens, do_sample, temperature
        Generation knobs.  Default greedy with 256 max tokens to keep
        screening sweeps fast.
    positions_mode
        ``"all"`` or ``"prefill_only"``; other modes raise NotImplementedError
        (deferred to v2).  Persisted to summary.json for audit.
    judge_dispatcher
        If None, a :class:`NoOpJudgeDispatcher` is used (Phase 1
        default).  Phase 2 callers will pass a real dispatcher.
    model_name
        Used to detect Qwen-family chat-template kwargs (enable_thinking).
        Optional.
    scan_mode
        ``"bidirectional"`` (the new 2026-05-14 default) drives a
        middle-out, two-stop-condition scan starting at
        ``weakest * multiplier ** start_strength_multiplier_steps``
        (default ~1.414): UP stops on ``coh_stop_consecutive`` consecutive
        strengths with mean coherence >= ``coh_stop_threshold``; DOWN
        stops on ``eff_stop_consecutive`` consecutive strengths with
        ``mean(|effect.combined|) < eff_stop_threshold``.  Either side
        also stops on cursor exhaustion (UP cap = ``max_strength``,
        DOWN floor = ``min_strength``).
        ``"legacy_unidirectional"`` runs the pre-2026-05-14 bottom-up
        sweep over the explicit ``strengths`` list and only uses the
        coh-stop tail check.
    eff_stop_threshold, eff_stop_consecutive
        Bidirectional-only DOWN-side stop knobs.  Default 0.25 over 2
        consecutive strengths -- if two consecutive sub-``s_init``
        strengths have ``mean(|effect.combined|) < 0.25`` we stop
        scanning down on the assumption further-down strengths will be
        even weaker.
    weakest_strength, max_strength, multiplier, min_strength,
    start_strength_multiplier_steps
        Bidirectional schedule knobs.  ``s_init = weakest_strength *
        multiplier ** start_strength_multiplier_steps`` (default
        ``1.0 * 1.189**2 ≈ 1.414``).  See :class:`BidirectionalCursor`
        for the full geometry.

    Returns
    -------
    CellResult describing whether the sweep completed or stopped early.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"

    dispatcher = judge_dispatcher or NoOpJudgeDispatcher()

    # Resume shortcut: if summary already records a stop, we're done with
    # generation and just need Phase 2 (in v1, nothing more to do).
    #
    # NOTE (2026-05-17): we still scan records.jsonl for incomplete
    # strengths (partial-K, judges=None) even if the cell has a final
    # summary -- those gaps are artifacts of prior kill-and-resume cycles
    # and should be filled in.  If any are found, we DROP the early-skip
    # and fall through to the normal sweep loop where Stage 2.5 gap-fill
    # will regenerate them.  Common case (no gaps): summary present,
    # all strengths complete, skip as before.
    existing_summary = _read_summary(summary_path)
    if existing_summary and existing_summary.get("stopped_at_strength") is not None:
        pre_skip_records = _read_existing_records(records_path)
        K_questions = len(questions)
        # Group existing records by strength (skip baseline at strength=0)
        pre_skip_by_s: Dict[float, List[Dict[str, Any]]] = {}
        for rec in pre_skip_records:
            s = float(rec.get("strength", 0.0))
            if s == 0.0:
                continue
            pre_skip_by_s.setdefault(s, []).append(rec)
        _eff_skip_threshold = getattr(
            dispatcher, "skip_threshold", DEFAULT_SKIP_THRESHOLD,
        )
        pre_skip_incomplete = [
            s for s, recs in pre_skip_by_s.items()
            if not _strength_records_complete(
                recs, K_questions, skip_threshold=_eff_skip_threshold,
            )
        ]
        # 2026-05-21: cap-extension fall-through.  If the cell previously
        # stopped because it hit the *configured* max_strength (UP cap)
        # rather than the physical coherence cliff, and the caller has
        # now raised the cap, fall through and let the bidirectional
        # state machine extend the UP-walk past the old cap.  Existing
        # records are preserved (done_pairs gates regeneration), so
        # only the new strengths (> old cap) get generated and judged.
        # Same for the DOWN side if min_strength was lowered.
        old_max = float(existing_summary.get("max_strength", max_strength))
        old_min = float(existing_summary.get("min_strength", min_strength))
        up_reason = existing_summary.get("up_blocked_reason")
        down_reason = existing_summary.get("down_blocked_reason")
        cap_extend_up = (
            up_reason == "max_strength_reached"
            and max_strength > old_max * (1.0 + 1e-9)
        )
        cap_extend_down = (
            down_reason == "min_strength_reached"
            and min_strength < old_min * (1.0 - 1e-9)
        )

        if pre_skip_incomplete:
            logger.info(
                f"[cell s{slot}_l{layer}_{sign:+d}] resume: summary exists but "
                f"{len(pre_skip_incomplete)} strength(s) incomplete "
                f"{sorted(round(abs(s), 4) for s in pre_skip_incomplete)} -- "
                f"falling through to gap-fill"
            )
        elif cap_extend_up or cap_extend_down:
            parts = []
            if cap_extend_up:
                parts.append(f"UP cap {old_max} -> {max_strength}")
            if cap_extend_down:
                parts.append(f"DOWN floor {old_min} -> {min_strength}")
            logger.info(
                f"[cell s{slot}_l{layer}_{sign:+d}] resume: previously stopped "
                f"at config cap ({', '.join(parts)}); falling through to "
                f"extend bidirectional walk past prior cap.  Existing records "
                f"preserved; only new strengths beyond the old cap will be "
                f"generated and judged."
            )
        else:
            logger.info(
                f"[cell s{slot}_l{layer}_{sign}] already stopped at "
                f"strength={existing_summary['stopped_at_strength']}; skipping"
            )
            return CellResult(
                stopped_at_strength=existing_summary.get("stopped_at_strength"),
                reason=existing_summary.get("reason", "incoherent"),
                n_records=existing_summary.get("n_records", 0),
                n_strengths_swept=existing_summary.get("n_strengths_swept", 0),
                summary=existing_summary,
            )

    # Resolve the effective scan_mode.  If summary.json from a prior
    # session pinned a mode, honour it: a mid-run mode switch would
    # corrupt the schedule (e.g. existing UP/DOWN strengths around
    # s_init would be reinterpreted as below-weakest legacy-mode noise).
    effective_scan_mode = scan_mode
    if existing_summary and "scan_mode" in existing_summary:
        persisted = existing_summary["scan_mode"]
        if persisted in ("bidirectional", "legacy_unidirectional"):
            if persisted != scan_mode:
                logger.warning(
                    f"[cell s{slot}_l{layer}_{sign}] honouring persisted "
                    f"scan_mode={persisted!r} from summary.json (caller "
                    f"requested {scan_mode!r}); cross-mode resume is "
                    f"unsupported"
                )
            effective_scan_mode = persisted

    existing_records = _read_existing_records(records_path)
    done_pairs = {(r["strength"], r["question_idx"]) for r in existing_records}
    if existing_records:
        logger.info(
            f"[cell s{slot}_l{layer}_{sign}] resuming with "
            f"{len(existing_records)} existing records "
            f"(scan_mode={effective_scan_mode!r})"
        )

    # Validate positions_mode early so we don't get most of the way
    # through the sweep before discovering misconfiguration.
    _build_position_kwargs(positions_mode)

    # Pre-build conversations once (same across strengths).
    conversations = _build_conversations(persona_system_prompt, questions)

    # Bidirectional branch: middle-out scan with two stop conditions.
    if effective_scan_mode == "bidirectional":
        return _run_bidirectional_cell(
            model=model, tokenizer=tokenizer,
            axis_vector=axis_vector,
            slot=slot, layer=layer, sign=sign,
            persona_system_prompt=persona_system_prompt,
            questions=questions, conversations=conversations,
            output_dir=output_dir,
            records_path=records_path, summary_path=summary_path,
            existing_records=existing_records, done_pairs=done_pairs,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            positions_mode=positions_mode,
            do_sample=do_sample, temperature=temperature,
            dispatcher=dispatcher, model_name=model_name,
            coh_stop_threshold=coh_stop_threshold,
            coh_stop_consecutive=coh_stop_consecutive,
            eff_stop_threshold=eff_stop_threshold,
            eff_stop_consecutive=eff_stop_consecutive,
            weakest_strength=weakest_strength,
            max_strength=max_strength,
            multiplier=multiplier,
            min_strength=min_strength,
            start_strength_multiplier_steps=start_strength_multiplier_steps,
        )

    # Legacy unidirectional path -- preserves 2026-05-13-and-earlier
    # behaviour bit-for-bit.  Caller-built `strengths` schedule required.
    if strengths is None:
        raise ValueError(
            "scan_mode='legacy_unidirectional' requires `strengths` to be "
            "passed in (caller-built via directional_schedule); got None"
        )

    n_strengths_swept = 0
    last_strength_completed: Optional[float] = None

    def _flush_records():
        """Use the dispatcher's atomic-write if available (avoids races with
        async judge writes); otherwise fall back to plain write_jsonl."""
        if hasattr(dispatcher, "write_records_atomic"):
            dispatcher.write_records_atomic(existing_records)
        else:
            write_jsonl(existing_records, records_path, logger_obj=logger)

    # ------------------------------------------------------------------
    # Pipelined coherence + greenlight/yellow state machine
    # ------------------------------------------------------------------
    #
    # Greenlight: generate strength S; dispatch coh-judge for S in
    # background; immediately move on to S+1 without waiting.  Speeds up
    # the bulk of the sweep when the model is producing coherent answers
    # (most strengths) by overlapping API latency with GPU generation.
    #
    # Yellow (sticky once entered): a previous strength's mean coh has
    # crossed the threshold.  Wait for ALL outstanding judges before
    # generating the next strength, so we never get "two ahead" of a
    # decision.  Reset point only on cell exit.
    #
    # Stop: K consecutive strengths (K = coh_stop_consecutive, default 2)
    # at the tail of the generated sequence, all judged with mean coh
    # >= coh_stop_threshold.  Default K=2 means one row past the first
    # crossing must also cross before we exit -- false-alarm-tolerant
    # without spending compute on confirmed-incoherent regions.

    seen_crossing = False
    pending_judges: List[Tuple[float, "concurrent.futures.Future"]] = []
    mean_by_strength: Dict[float, float] = {}
    generated_in_order: List[float] = []

    # Bootstrap from existing records: any prior session's strengths
    # already have judges stamped on disk; lift them into our local
    # state so the consecutive-crossings tail check works on restart.
    if existing_records:
        for rec in existing_records:
            s = float(rec.get("strength", 0.0))
            if s == 0.0:  # skip baselines (sign=0)
                continue
            j = rec.get("judges") or {}
            smc = j.get("strength_mean_coh")
            if smc is not None and s not in mean_by_strength:
                mean_by_strength[s] = float(smc)
                if s not in generated_in_order:
                    generated_in_order.append(s)
        generated_in_order.sort()

    def _reap_done_futures():
        """Move any completed pending futures into mean_by_strength.
        Updates seen_crossing if any newly-known mean is >= threshold."""
        nonlocal seen_crossing, pending_judges
        still_pending: List[Tuple[float, "concurrent.futures.Future"]] = []
        for s_, fut_ in pending_judges:
            if fut_.done():
                try:
                    m = float(fut_.result())
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"[cell s{slot}_l{layer}_{sign}] coherence future "
                        f"raised for strength={s_}: {e}; treating as 0.0"
                    )
                    m = 0.0
                mean_by_strength[s_] = m
                if m >= coh_stop_threshold:
                    seen_crossing = True
            else:
                still_pending.append((s_, fut_))
        pending_judges[:] = still_pending

    def _await_all_pending():
        """Block until every still-pending future resolves; updates state."""
        nonlocal pending_judges, seen_crossing
        for s_, fut_ in pending_judges:
            try:
                m = float(fut_.result())
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"[cell s{slot}_l{layer}_{sign}] coherence future "
                    f"raised (await) for strength={s_}: {e}; treating as 0.0"
                )
                m = 0.0
            mean_by_strength[s_] = m
            if m >= coh_stop_threshold:
                seen_crossing = True
        pending_judges = []

    def _consecutive_crossings_at_tail() -> int:
        """How many strength-ordered tail strengths have judged mean >= threshold?
        Returns 0 if the most-recently-generated strength isn't fully judged."""
        n = 0
        for s_ in reversed(generated_in_order):
            if s_ not in mean_by_strength:
                return n  # latest strength still pending; can't conclude
            if mean_by_strength[s_] < coh_stop_threshold:
                return n
            n += 1
        return n

    stopped_strength: Optional[float] = None
    stop_reason: str = "completed"

    for strength in strengths:
        # Reap any judges that finished while we were generating; this is
        # what flips greenlight -> yellow when a crossing is observed.
        _reap_done_futures()

        if seen_crossing:
            # Yellow mode: wait for everything outstanding, then check stop.
            _await_all_pending()
            n_consec = _consecutive_crossings_at_tail()
            if n_consec >= coh_stop_consecutive:
                logger.info(
                    f"[cell s{slot}_l{layer}_{sign}] stop: "
                    f"{n_consec} consecutive strengths above threshold "
                    f"{coh_stop_threshold} (>= {coh_stop_consecutive}); "
                    f"last_generated={generated_in_order[-1]}"
                )
                stopped_strength = generated_in_order[-1]
                stop_reason = "incoherent"
                break

        eff_coeff = float(strength) * float(sign)
        logger.info(
            f"[cell s{slot}_l{layer}_{sign}] strength={strength:.4f} "
            f"sign={sign:+d} eff_coeff={eff_coeff:+.4f} "
            f"({'YELLOW' if seen_crossing else 'GREEN'})"
        )
        any_progress_this_strength = False
        # Tracks newly-generated records at this strength so we can
        # enqueue them as a single (cell, sign, strength) group after
        # the strength's batches finish.
        new_at_strength: List[Dict[str, Any]] = []

        for batch_start in range(0, len(questions), batch_size):
            # Build batch of (q_idx, question, conversation) for items
            # not yet in the records.
            batch_items = []
            for offset in range(batch_size):
                q_idx = batch_start + offset
                if q_idx >= len(questions):
                    break
                if (strength, q_idx) in done_pairs:
                    continue
                batch_items.append((q_idx, questions[q_idx], conversations[q_idx]))
            if not batch_items:
                continue
            any_progress_this_strength = True

            batch_convs = [c for _, _, c in batch_items]
            inputs = _tokenize_batch(tokenizer, batch_convs, model_name=model_name)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            position_kwargs = _build_position_kwargs(positions_mode)
            t0 = time.time()
            with ActivationSteering(
                model,
                steering_vectors=[axis_vector],
                coefficients=[eff_coeff],
                layer_indices=[layer],
                **position_kwargs,
            ):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            elapsed = time.time() - t0
            per_item_s = elapsed / max(1, len(batch_items))

            for j, (q_idx, q, _conv) in enumerate(batch_items):
                gen_ids = outputs[j, prompt_len:]
                response, n_tok, truncated = _decode_with_truncation_marker(
                    gen_ids, tokenizer
                )
                rec: Dict[str, Any] = {
                    "strength": float(strength),
                    "sign": int(sign),
                    "slot": int(slot),
                    "layer": int(layer),
                    "question_idx": int(q_idx),
                    "question": q,
                    "response": response,
                    "n_tokens": n_tok,
                    "truncated": truncated,
                    "judges": {"coherence": None, "persona": None, "effect": None},
                    "timing": {"gen_s": per_item_s},
                    "abandoned": False,
                }
                existing_records.append(rec)
                new_at_strength.append(rec)
                done_pairs.add((float(strength), int(q_idx)))

            # Flush per-batch (per Roger's "save after each question set")
            # so a crash mid-strength doesn't lose the just-generated
            # responses.  Coherence will be filled in async by the
            # dispatcher; the runner's atomic write here uses the
            # dispatcher's lock to avoid racing.
            _flush_records()

        if any_progress_this_strength:
            n_strengths_swept += 1
            last_strength_completed = float(strength)
            generated_in_order.append(float(strength))

        # Async coherence dispatch + chained RP/effect.  In greenlight
        # mode this returns immediately; the next strength will start
        # generating before this one's coh judges arrive.  In yellow
        # mode the loop's top will await.
        if new_at_strength:
            coh_future = dispatcher.judge_coherence_for_strength_async(
                new_at_strength
            )
            # If the future resolved synchronously (e.g. NoOpJudgeDispatcher),
            # the records now carry strength_mean_coh in memory but the
            # disk copy is still pre-judging.  Re-flush so the on-disk
            # state matches.  For RealJudgeDispatcher the future is still
            # pending; its coro persists records via _merge_records_to_disk
            # under the same lock so this re-flush is harmless redundancy.
            if coh_future.done():
                _flush_records()

            def _on_coh_done(_f, recs=new_at_strength, sn=float(strength)):
                # Once coherence is in, fire the RP/effect dispatch.
                # The dispatcher reads strength_mean_coh from the
                # records (stamped by the coh future itself) and
                # decides skip-or-judge.
                try:
                    dispatcher.enqueue_strength_group(
                        cell_dir=str(output_dir),
                        slot=int(slot), layer=int(layer),
                        sign=int(sign), strength=sn,
                        records=recs,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"[cell s{slot}_l{layer}_{sign}] enqueue_strength_group "
                        f"chain failed for strength={sn}: {e}"
                    )

            coh_future.add_done_callback(_on_coh_done)
            pending_judges.append((float(strength), coh_future))

    # End of sweep: drain any still-pending coh futures so records.jsonl
    # is fully populated before we write the summary.
    _await_all_pending()

    if stopped_strength is not None:
        summary = {
            "slot": slot, "layer": layer, "sign": sign,
            "stopped_at_strength": float(stopped_strength),
            "reason": stop_reason,
            "n_records": len(existing_records),
            "n_strengths_swept": n_strengths_swept,
            "last_strength_completed": last_strength_completed,
            "coh_stop_consecutive": int(coh_stop_consecutive),
            "coh_stop_threshold": float(coh_stop_threshold),
            "scan_mode": "legacy_unidirectional",
            "positions_mode": positions_mode,
        }
        _write_summary(summary_path, summary)
        dispatcher.drain()
        return CellResult(
            stopped_at_strength=float(stopped_strength),
            reason=stop_reason,
            n_records=len(existing_records),
            n_strengths_swept=n_strengths_swept,
            summary=summary,
        )

    # Sweep completed all strengths
    summary = {
        "slot": slot, "layer": layer, "sign": sign,
        "stopped_at_strength": None,
        "reason": "completed",
        "n_records": len(existing_records),
        "n_strengths_swept": n_strengths_swept,
        "last_strength_completed": last_strength_completed,
        "coh_stop_consecutive": int(coh_stop_consecutive),
        "coh_stop_threshold": float(coh_stop_threshold),
        "scan_mode": "legacy_unidirectional",
        "positions_mode": positions_mode,
    }
    _write_summary(summary_path, summary)
    dispatcher.drain()
    return CellResult(
        stopped_at_strength=None,
        reason="completed",
        n_records=len(existing_records),
        n_strengths_swept=n_strengths_swept,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Bidirectional scan helper (2026-05-14: new default scan_mode)
# ---------------------------------------------------------------------------


def _generate_and_dispatch_strength(
    *,
    strength: float,
    sign: int,
    model,
    tokenizer,
    axis_vector: torch.Tensor,
    slot: int,
    layer: int,
    questions: Sequence[str],
    conversations: List[List[Dict[str, str]]],
    output_dir: Path,
    done_pairs: set,
    existing_records: List[Dict[str, Any]],
    flush_records: "callable",
    dispatcher: JudgeDispatcher,
    batch_size: int,
    max_new_tokens: int,
    positions_mode: str,
    do_sample: bool,
    temperature: float,
    model_name: str,
) -> Tuple[bool, List[Dict[str, Any]], Optional["concurrent.futures.Future"]]:
    """Generate one strength's responses + fire async coh judging.

    Returns ``(any_progress, new_at_strength, coh_future)``.  If the
    strength was already fully covered by ``done_pairs`` (resume), the
    function returns ``(False, [], None)`` and does no work.  Otherwise
    it generates the missing batches, appends to ``existing_records``,
    flushes to disk, and returns the in-flight coherence future (which
    also chains into the dispatcher's effect/persona pipeline via an
    ``on_done`` callback).

    Used by both legacy and bidirectional cell runners.  Mirrors the
    per-strength block of the legacy main loop verbatim, factored out
    only so the bidirectional state machine doesn't have to duplicate it.
    """
    eff_coeff = float(strength) * float(sign)
    any_progress = False
    new_at_strength: List[Dict[str, Any]] = []

    for batch_start in range(0, len(questions), batch_size):
        batch_items = []
        for offset in range(batch_size):
            q_idx = batch_start + offset
            if q_idx >= len(questions):
                break
            if (strength, q_idx) in done_pairs:
                continue
            batch_items.append((q_idx, questions[q_idx], conversations[q_idx]))
        if not batch_items:
            continue
        any_progress = True

        batch_convs = [c for _, _, c in batch_items]
        inputs = _tokenize_batch(tokenizer, batch_convs, model_name=model_name)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        position_kwargs = _build_position_kwargs(positions_mode)
        t0 = time.time()
        with ActivationSteering(
            model,
            steering_vectors=[axis_vector],
            coefficients=[eff_coeff],
            layer_indices=[layer],
            **position_kwargs,
        ):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=tokenizer.pad_token_id,
                )
        elapsed = time.time() - t0
        per_item_s = elapsed / max(1, len(batch_items))

        for j, (q_idx, q, _conv) in enumerate(batch_items):
            gen_ids = outputs[j, prompt_len:]
            response, n_tok, truncated = _decode_with_truncation_marker(
                gen_ids, tokenizer
            )
            rec: Dict[str, Any] = {
                "strength": float(strength),
                "sign": int(sign),
                "slot": int(slot),
                "layer": int(layer),
                "question_idx": int(q_idx),
                "question": q,
                "response": response,
                "n_tokens": n_tok,
                "truncated": truncated,
                "judges": {"coherence": None, "persona": None, "effect": None},
                "timing": {"gen_s": per_item_s},
                "abandoned": False,
            }
            existing_records.append(rec)
            new_at_strength.append(rec)
            done_pairs.add((float(strength), int(q_idx)))

        flush_records()

    if not new_at_strength:
        return any_progress, [], None

    # Async coherence dispatch + chained RP/effect.
    coh_future = dispatcher.judge_coherence_for_strength_async(new_at_strength)
    if coh_future.done():
        flush_records()

    def _on_coh_done(_f, recs=new_at_strength, sn=float(strength)):
        try:
            dispatcher.enqueue_strength_group(
                cell_dir=str(output_dir),
                slot=int(slot), layer=int(layer),
                sign=int(sign), strength=sn,
                records=recs,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"[cell s{slot}_l{layer}_{sign}] enqueue_strength_group "
                f"chain failed for strength={sn}: {e}"
            )

    coh_future.add_done_callback(_on_coh_done)
    return any_progress, new_at_strength, coh_future


def _run_bidirectional_cell(
    *,
    model,
    tokenizer,
    axis_vector: torch.Tensor,
    slot: int,
    layer: int,
    sign: int,
    persona_system_prompt: str,
    questions: Sequence[str],
    conversations: List[List[Dict[str, str]]],
    output_dir: Path,
    records_path: Path,
    summary_path: Path,
    existing_records: List[Dict[str, Any]],
    done_pairs: set,
    batch_size: int,
    max_new_tokens: int,
    positions_mode: str,
    do_sample: bool,
    temperature: float,
    dispatcher: JudgeDispatcher,
    model_name: str,
    coh_stop_threshold: float,
    coh_stop_consecutive: int,
    eff_stop_threshold: float,
    eff_stop_consecutive: int,
    weakest_strength: float,
    max_strength: float,
    multiplier: float,
    min_strength: float,
    start_strength_multiplier_steps: int,
) -> CellResult:
    """Bidirectional middle-out scan with two-stop-condition state machine.

    See the design plan
    ``/Users/roger/.cursor/plans/bidirectional_steering_scan_e93878b3.plan.md``
    for the full state-machine spec.  In short:

    * Centre anchor: ``s_init = weakest * mult ** start_steps_up``
      (default 1.414, "2 multiplier-steps above the legacy weakest").
    * BothOpen: alternate UP, DOWN, UP, DOWN; judging is async so the
      next-direction generation overlaps with the previous-direction's
      judges.
    * UpBlocked / DownBlocked: only step the still-open direction;
      AWAIT that direction's judge between strengths so each
      stop-decision uses fully-resolved scores.
    * Stop conditions (each evaluated against direction-local tail
      window): UP via ``coh_stop_consecutive`` consecutive
      ``mean_coh >= coh_stop_threshold`` strengths; DOWN via
      ``eff_stop_consecutive`` consecutive ``mean_abs_eff <
      eff_stop_threshold`` strengths.  Either side also stops on
      cursor exhaustion (``max_strength`` / ``min_strength``).
    * NaN handling: a DOWN strength whose effect future resolved to NaN
      (judging skipped due to high coherence) does NOT count toward the
      eff-stop tail window -- we treat it as "unknown".
    """
    cursor = BidirectionalCursor(
        weakest=weakest_strength,
        max_strength=max_strength,
        min_strength=min_strength,
        multiplier=multiplier,
        start_steps_up=start_strength_multiplier_steps,
    )

    def _flush_records():
        if hasattr(dispatcher, "write_records_atomic"):
            dispatcher.write_records_atomic(existing_records)
        else:
            write_jsonl(existing_records, records_path, logger_obj=logger)

    # ------------------------------------------------------------------
    # Bootstrap state from existing records (restart path)
    # ------------------------------------------------------------------
    # Per-direction tail buffers.  We track strengths emitted on each
    # side in cursor order so the tail-window check is straightforward.
    up_strengths_in_order: List[float] = []      # strictly > s_init
    down_strengths_in_order: List[float] = []    # strictly < s_init
    mean_coh_by_strength: Dict[float, float] = {}
    # mean_abs_eff_by_strength: NaN sentinel = skipped/unknown; do not
    # count toward DOWN eff-stop tail window.
    mean_abs_eff_by_strength: Dict[float, float] = {}

    # ``seen_strengths`` is shared between Stage 1 (mean_coh / mean_abs_eff
    # bootstrap), Stage 2 (cursor walk), and the post-s_init gap-fill
    # below.  Empty dict when ``existing_records`` is empty (fresh run).
    seen_strengths: Dict[float, List[Dict[str, Any]]] = {}
    if existing_records:
        # Stage 1: collect strengths + their stamped mean_coh from
        # records.jsonl so the tail-window checks have history to chew on.
        for rec in existing_records:
            s = float(rec.get("strength", 0.0))
            if s == 0.0:
                continue
            seen_strengths.setdefault(s, []).append(rec)
        # mean_coh from any record's strength_mean_coh stamp
        for s, recs in seen_strengths.items():
            for r in recs:
                smc = (r.get("judges") or {}).get("strength_mean_coh")
                if smc is not None:
                    mean_coh_by_strength[s] = float(smc)
                    break
            # mean_abs_eff: compute fresh from effect.combined across
            # the strength's records (matches the runtime computation).
            #
            # NOTE (2026-05-17 fix): we compute ``|mean(eff_i)|``, NOT
            # ``mean(|eff_i|)``.  The pre-fix per-record-abs version
            # stayed at ~0.5-0.8 in pure noise (half-normal expected
            # value of |x| when x~N(0,1)), so the eff-stop predicate
            # almost never fired and the down-sweep collected to
            # min_strength on every cell with a weak signal.  Matches
            # the convention used by tools/test_effect_order_bias.py
            # from the start.
            vals = []
            any_skipped = False
            for r in recs:
                eff = (r.get("judges") or {}).get("effect") or {}
                if eff.get("skipped_due_to_strength_mean_coh"):
                    any_skipped = True
                v = eff.get("combined")
                if isinstance(v, (int, float)):
                    vals.append(float(v))  # keep sign
            if vals:
                mean_abs_eff_by_strength[s] = abs(sum(vals) / len(vals))
            elif any_skipped:
                mean_abs_eff_by_strength[s] = float("nan")
            # else: leave unset; tail-window check treats unset as
            # "still pending" and will not advance.

        # Stage 2: advance the cursors past every existing strength on
        # each side, so subsequent next_up/next_down emit FRESH values.
        # Cursor is deterministic, so re-walking it matches what was
        # emitted before.
        existing_above = sorted({s for s in seen_strengths if s > cursor.s_init})
        existing_below = sorted(
            {s for s in seen_strengths if s < cursor.s_init}, reverse=True,
        )
        for expected in existing_above:
            got = cursor.next_up()
            if got is None or abs(got - expected) > 1e-6:
                logger.warning(
                    f"[cell s{slot}_l{layer}_{sign}] bidirectional resume: "
                    f"cursor UP re-emitted {got} but expected {expected}; "
                    f"records may be from a different schedule -- "
                    f"continuing but tail history may be lossy"
                )
                break
            up_strengths_in_order.append(expected)
        for expected in existing_below:
            got = cursor.next_down()
            if got is None or abs(got - expected) > 1e-6:
                logger.warning(
                    f"[cell s{slot}_l{layer}_{sign}] bidirectional resume: "
                    f"cursor DOWN re-emitted {got} but expected {expected}; "
                    f"records may be from a different schedule -- "
                    f"continuing but tail history may be lossy"
                )
                break
            down_strengths_in_order.append(expected)
        # The s_init record itself (if any) -- it lives in neither tail
        # buffer; we treat it as "the centre" and count it only toward
        # n_strengths_swept below.

    # In-flight futures, keyed by direction so the state machine can
    # await per-direction without blocking on the other side.
    pending_up: List[Tuple[float, "concurrent.futures.Future", "concurrent.futures.Future"]] = []
    pending_down: List[Tuple[float, "concurrent.futures.Future", "concurrent.futures.Future"]] = []
    # (strength, coh_future, eff_future) tuples; eff_future may be the
    # NoOp instant 0.0 for dispatchers that don't surface effect.

    up_blocked = cursor.up_exhausted
    down_blocked = cursor.down_exhausted

    def _reap_direction(pending: list, mean_dict_coh: dict,
                        mean_dict_eff: dict) -> None:
        """Drain any completed futures in this direction's pending list
        into the corresponding mean_dict (coh + abs-eff)."""
        still: list = []
        for s_, coh_fut, eff_fut in pending:
            both_done = coh_fut.done() and eff_fut.done()
            if both_done:
                try:
                    m_coh = float(coh_fut.result())
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"[cell s{slot}_l{layer}_{sign}] coh future raised "
                        f"for strength={s_}: {e}; treating as 0.0"
                    )
                    m_coh = 0.0
                try:
                    m_eff = float(eff_fut.result())
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"[cell s{slot}_l{layer}_{sign}] eff future raised "
                        f"for strength={s_}: {e}; treating as NaN"
                    )
                    m_eff = float("nan")
                mean_dict_coh[s_] = m_coh
                mean_dict_eff[s_] = m_eff
            else:
                still.append((s_, coh_fut, eff_fut))
        pending[:] = still

    def _await_direction(pending: list, mean_dict_coh: dict,
                         mean_dict_eff: dict) -> None:
        """Block until every future in this direction's pending list
        resolves; update mean_dicts in lock-step."""
        for s_, coh_fut, eff_fut in pending:
            try:
                m_coh = float(coh_fut.result())
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"[cell s{slot}_l{layer}_{sign}] coh future raised "
                    f"(await) for strength={s_}: {e}; treating as 0.0"
                )
                m_coh = 0.0
            try:
                m_eff = float(eff_fut.result())
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"[cell s{slot}_l{layer}_{sign}] eff future raised "
                    f"(await) for strength={s_}: {e}; treating as NaN"
                )
                m_eff = float("nan")
            mean_dict_coh[s_] = m_coh
            mean_dict_eff[s_] = m_eff
        pending.clear()

    def _consecutive_above_coh_at_up_tail() -> int:
        """Count tail-end UP strengths with mean_coh >= coh_stop_threshold.

        Returns 0 if the newest UP strength isn't yet judged (can't
        conclude until the latest data point is in).
        """
        import math
        n = 0
        for s_ in reversed(up_strengths_in_order):
            if s_ not in mean_coh_by_strength:
                return n
            v = mean_coh_by_strength[s_]
            if math.isnan(v) or v < coh_stop_threshold:
                return n
            n += 1
        return n

    def _consecutive_below_eff_at_down_tail() -> int:
        """Count tail-end DOWN strengths with mean_abs_eff < eff_stop_threshold.

        NaN-eff strengths (skipped due to high coherence) do NOT count
        toward the tail window -- they short-circuit the count back to
        0 because we don't know whether the effect would have been
        below or above threshold.
        """
        import math
        n = 0
        for s_ in reversed(down_strengths_in_order):
            if s_ not in mean_abs_eff_by_strength:
                return n
            v = mean_abs_eff_by_strength[s_]
            if math.isnan(v):
                return 0  # skipped strength breaks the streak
            if v >= eff_stop_threshold:
                return n
            n += 1
        return n

    def _do_one_strength(strength: float, direction: str) -> bool:
        """Generate at one strength + dispatch judges + record futures.

        Returns True if any new records were created (False on restart
        where the strength is already fully done -- caller treats that
        as a no-op step and tries the next cursor value).
        """
        any_progress, new_recs, coh_fut = _generate_and_dispatch_strength(
            strength=strength, sign=sign,
            model=model, tokenizer=tokenizer,
            axis_vector=axis_vector,
            slot=slot, layer=layer,
            questions=questions, conversations=conversations,
            output_dir=output_dir,
            done_pairs=done_pairs,
            existing_records=existing_records,
            flush_records=_flush_records,
            dispatcher=dispatcher,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            positions_mode=positions_mode,
            do_sample=do_sample, temperature=temperature,
            model_name=model_name,
        )
        if not any_progress or coh_fut is None:
            return False
        eff_fut = dispatcher.judge_effect_for_strength_async(new_recs)
        if direction == "up":
            pending_up.append((float(strength), coh_fut, eff_fut))
            if strength not in up_strengths_in_order:
                up_strengths_in_order.append(float(strength))
        else:
            pending_down.append((float(strength), coh_fut, eff_fut))
            if strength not in down_strengths_in_order:
                down_strengths_in_order.append(float(strength))
        return True

    # ------------------------------------------------------------------
    # s_init: generate first (only if not already covered) so we have an
    # anchor record and the per-direction tail windows have correct
    # boundaries (s_init itself counts toward NEITHER tail; it's the
    # centre).  Restart path: if all s_init records are done, this is a
    # no-op.
    # ------------------------------------------------------------------
    s_init_progress, s_init_recs, s_init_coh = _generate_and_dispatch_strength(
        strength=cursor.s_init, sign=sign,
        model=model, tokenizer=tokenizer,
        axis_vector=axis_vector,
        slot=slot, layer=layer,
        questions=questions, conversations=conversations,
        output_dir=output_dir,
        done_pairs=done_pairs,
        existing_records=existing_records,
        flush_records=_flush_records,
        dispatcher=dispatcher,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        positions_mode=positions_mode,
        do_sample=do_sample, temperature=temperature,
        model_name=model_name,
    )
    s_init_pending: List[Tuple[float, "concurrent.futures.Future", "concurrent.futures.Future"]] = []
    if s_init_progress and s_init_coh is not None:
        s_init_eff = dispatcher.judge_effect_for_strength_async(s_init_recs)
        s_init_pending.append((float(cursor.s_init), s_init_coh, s_init_eff))

    # ------------------------------------------------------------------
    # Resume gap-fill (2026-05-17): if a previous run was killed mid-
    # strength, that strength's records on disk are PARTIAL -- some
    # questions generated but not all 14, and judging never ran.  The
    # cursor's Stage-2 walk above happily skipped past them (it only
    # checks which strengths are present, not whether they're complete).
    # Here we identify those incomplete strengths and re-emit them via
    # _generate_and_dispatch_strength.  The function regenerates any
    # missing (strength, q_idx) pairs and dispatches coh+rp+effect
    # judging for the freshly-generated rows.
    #
    # For strengths where >=1 record exists but it doesn't make the
    # completeness cut (judges=None, partial K, coh-judged but
    # rp/effect missing without skip flag), we WIPE the existing
    # partial records and regenerate from scratch.  This costs at
    # most K generations per gap; acceptable given the alternative
    # is silent interior gaps in every restarted cell's curve.
    # ------------------------------------------------------------------
    if existing_records:
        K_questions = len(questions)
        incomplete: List[float] = []
        for s, recs in seen_strengths.items():
            if abs(s - cursor.s_init) < 1e-6:
                continue  # s_init was just refreshed by the call above
            if not _strength_records_complete(
                recs, K_questions,
                skip_threshold=getattr(
                    dispatcher, "skip_threshold", DEFAULT_SKIP_THRESHOLD,
                ),
            ):
                incomplete.append(s)
        if incomplete:
            logger.info(
                f"[cell s{slot}_l{layer}_{sign:+d}] resume gap-fill: "
                f"{len(incomplete)} incomplete strength(s) "
                f"{sorted(abs(s) for s in incomplete)} -- regenerating"
            )
        for s in sorted(incomplete, key=abs):
            # Wipe partial records (in-memory + disk via flush below)
            existing_records[:] = [
                r for r in existing_records
                if abs(float(r.get("strength", 0.0)) - s) > 1e-6
            ]
            for q in range(K_questions):
                done_pairs.discard((s, q))
            # Clear stale stats; will be re-populated when the new
            # future resolves via _reap_direction in the main loop.
            mean_coh_by_strength.pop(s, None)
            mean_abs_eff_by_strength.pop(s, None)
            _flush_records()
            # Regenerate + dispatch judging for all K questions
            gp, gnew, gcoh = _generate_and_dispatch_strength(
                strength=s, sign=sign,
                model=model, tokenizer=tokenizer,
                axis_vector=axis_vector, slot=slot, layer=layer,
                questions=questions, conversations=conversations,
                output_dir=output_dir, done_pairs=done_pairs,
                existing_records=existing_records,
                flush_records=_flush_records,
                dispatcher=dispatcher,
                batch_size=batch_size, max_new_tokens=max_new_tokens,
                positions_mode=positions_mode,
                do_sample=do_sample, temperature=temperature,
                model_name=model_name,
            )
            if gp and gcoh is not None:
                geff = dispatcher.judge_effect_for_strength_async(gnew)
                if s > cursor.s_init:
                    pending_up.append((float(s), gcoh, geff))
                    # Stage 2's cursor walk already appended this
                    # strength to up_strengths_in_order, so don't
                    # double-add here.
                else:
                    pending_down.append((float(s), gcoh, geff))

    # ------------------------------------------------------------------
    # Main state-machine loop.  Alternate UP/DOWN while both open; only
    # step the open side while one is blocked.  After every step, reap
    # judging futures and re-check stop conditions.
    # ------------------------------------------------------------------
    last_direction: Literal["down", "up"] = "down"  # so next is "up"
    stopped_strength: Optional[float] = None
    stop_reason: str = "completed"
    up_stop_reason: Optional[str] = None
    down_stop_reason: Optional[str] = None

    while True:
        # Reap any judges that finished while we were generating.
        _reap_direction(pending_up, mean_coh_by_strength, mean_abs_eff_by_strength)
        _reap_direction(pending_down, mean_coh_by_strength, mean_abs_eff_by_strength)

        # Check stop conditions on both sides (sticky once set).
        if not up_blocked:
            if cursor.up_exhausted:
                up_blocked = True
                up_stop_reason = "max_strength_reached"
            elif _consecutive_above_coh_at_up_tail() >= coh_stop_consecutive:
                up_blocked = True
                up_stop_reason = "incoherent"
        if not down_blocked:
            if cursor.down_exhausted:
                down_blocked = True
                down_stop_reason = "min_strength_reached"
            elif _consecutive_below_eff_at_down_tail() >= eff_stop_consecutive:
                down_blocked = True
                down_stop_reason = "sub_threshold_effect"

        if up_blocked and down_blocked:
            # Both sides blocked.  Drain remaining futures so summary
            # captures final state, then exit.
            _await_direction(pending_up, mean_coh_by_strength, mean_abs_eff_by_strength)
            _await_direction(pending_down, mean_coh_by_strength, mean_abs_eff_by_strength)
            _await_direction(s_init_pending, mean_coh_by_strength, mean_abs_eff_by_strength)
            stop_reason = "bidirectional_done"
            # stopped_at_strength: max generated strength on the UP side
            # (matches the legacy "where did we stop scanning?" semantics
            # for downstream consumers that look at this field).
            if up_strengths_in_order:
                stopped_strength = up_strengths_in_order[-1]
            elif down_strengths_in_order:
                stopped_strength = down_strengths_in_order[-1]
            else:
                stopped_strength = float(cursor.s_init)
            break

        # Choose direction: alternate when both open; the open side when
        # one is blocked.  Yellow-mode AWAIT for the blocked-elsewhere
        # case so each stop-decision uses fully-resolved scores before
        # the next generation.
        if up_blocked:
            direction = "down"
            _await_direction(pending_down, mean_coh_by_strength, mean_abs_eff_by_strength)
            # Re-check after await; maybe this just blocked DOWN too.
            if _consecutive_below_eff_at_down_tail() >= eff_stop_consecutive:
                down_blocked = True
                down_stop_reason = "sub_threshold_effect"
                continue
        elif down_blocked:
            direction = "up"
            _await_direction(pending_up, mean_coh_by_strength, mean_abs_eff_by_strength)
            if _consecutive_above_coh_at_up_tail() >= coh_stop_consecutive:
                up_blocked = True
                up_stop_reason = "incoherent"
                continue
        else:
            direction = "up" if last_direction == "down" else "down"

        if direction == "up":
            nxt = cursor.next_up()
            if nxt is None:
                up_blocked = True
                up_stop_reason = "max_strength_reached"
                continue
            _do_one_strength(nxt, direction="up")
            last_direction = "up"
        else:
            nxt = cursor.next_down()
            if nxt is None:
                down_blocked = True
                down_stop_reason = "min_strength_reached"
                continue
            _do_one_strength(nxt, direction="down")
            last_direction = "down"

    # End of bidirectional scan: write summary + drain dispatcher.
    n_strengths_swept = (
        len(up_strengths_in_order)
        + len(down_strengths_in_order)
        + (1 if s_init_progress else 0)
    )
    last_strength_completed: Optional[float] = None
    if up_strengths_in_order:
        last_strength_completed = up_strengths_in_order[-1]
    elif s_init_progress:
        last_strength_completed = float(cursor.s_init)
    elif down_strengths_in_order:
        last_strength_completed = down_strengths_in_order[-1]

    summary = {
        "slot": slot, "layer": layer, "sign": sign,
        "stopped_at_strength": (
            float(stopped_strength) if stopped_strength is not None else None
        ),
        "reason": stop_reason,
        "n_records": len(existing_records),
        "n_strengths_swept": n_strengths_swept,
        "last_strength_completed": last_strength_completed,
        "coh_stop_consecutive": int(coh_stop_consecutive),
        "coh_stop_threshold": float(coh_stop_threshold),
        "scan_mode": "bidirectional",
        "positions_mode": positions_mode,
        "eff_stop_threshold": float(eff_stop_threshold),
        "eff_stop_consecutive": int(eff_stop_consecutive),
        "s_init": float(cursor.s_init),
        "min_strength": float(min_strength),
        "max_strength": float(max_strength),
        "multiplier": float(multiplier),
        "weakest_strength": float(weakest_strength),
        "start_strength_multiplier_steps": int(start_strength_multiplier_steps),
        "up_blocked_reason": up_stop_reason,
        "down_blocked_reason": down_stop_reason,
        "up_blocked_at_strength": (
            up_strengths_in_order[-1] if up_strengths_in_order else None
        ),
        "down_blocked_at_strength": (
            down_strengths_in_order[-1] if down_strengths_in_order else None
        ),
        "records_in_scan_order": True,
    }
    _write_summary(summary_path, summary)
    dispatcher.drain()
    return CellResult(
        stopped_at_strength=summary["stopped_at_strength"],
        reason=stop_reason,
        n_records=len(existing_records),
        n_strengths_swept=n_strengths_swept,
        summary=summary,
    )
