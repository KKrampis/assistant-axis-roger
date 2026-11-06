"""Order-bias diagnostic: re-judge effect scores with baseline / steered
swapped, alongside the original scores.

Hypothesis test setup:
  - For each record in ``experiment_dir`` with a non-null effect score,
    construct the SAME bidirectional effect prompt the live sweep used,
    but with the [BASELINE] and [RESPONSE] block contents swapped.
  - Call the SAME ensemble of effect judges (default gpt-4.1-mini +
    claude-haiku-4-5).  Aggregate to a combined score per record
    (mean of successful judges, same as the live sweep's combined).

Reading the result:
  - If the judges are content-driven, every swapped score should be the
    EXACT NEGATION of the original (because "A is more X than B" under
    the swap becomes "B is more X than A", which is -1 * the original
    signed score).
  - If there's a label / position / order bias (judge consistently
    rates whichever response sits in the [RESPONSE] block as more
    "neg_label" regardless of content), the swapped score should
    instead match the original sign.
  - In between, the slope of ``-orig`` vs ``swap`` quantifies how much
    of the score is content-driven vs label-driven.

Writes ``<cell_dir>/records_effect_swap.jsonl`` with one line per
record carrying ``{question_idx, strength, sign, orig_combined,
swap_scores (dict by model), swap_combined}``.

After the run, prints a per-strength summary of the bias:
mean original vs mean swapped vs mean (orig + swap) [which would be
0 under perfect content-driven scoring].

Usage::

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \\
    uv run python tools/test_effect_order_bias.py \\
        outputs/qwen-3-32b/steering/anthropologist_helpful_v1

Cost: ~$1-2 per experiment for the gpt-4.1-mini + haiku ensemble
on ~5k records.  Live judging-style batching (target_batch_size=10)
keeps the call count to ~500 per judge per experiment.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean as _mean
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from assistant_axis.judge import (  # noqa: E402
    RateLimiter, call_judge_single_unified, parse_batch_scores_json,
)
from assistant_axis.steering_judges import (  # noqa: E402
    DEFAULT_EFFECT_MODELS, DEFAULT_TARGET_BATCH_SIZE,
    PersonaSpec, SteeringSpec, build_effect_bidir_batch_prompt,
)

logger = logging.getLogger("test_effect_order_bias")


# ---------------------------------------------------------------------------
# Loading experiment context (mirrors steering/post_judge.py's spec loader)
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _try_desc(name: str, instructions_dir: Path) -> str:
    for kind in ("traits", "roles"):
        p = instructions_dir / kind / "instructions" / f"{name}.json"
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    return str(json.load(f).get("description", ""))
            except (OSError, json.JSONDecodeError):
                continue
    return ""


def load_specs(
    experiment_dir: Path, instructions_dir: Path,
) -> Tuple[PersonaSpec, SteeringSpec]:
    """Build PersonaSpec + SteeringSpec from the experiment's config.json.

    Matches the conventions used by ``steering/run_sweep.py``'s
    ``_resolve_steering_spec``: for role_transplant axes,
    ``pos_label = role_to`` (the "+1 sign" pole the rubric scores
    positively for), ``neg_label = role_from``.
    """
    with open(experiment_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    persona_cfg = config.get("persona", {})
    role_name = persona_cfg.get("role", "")
    persona_desc = _try_desc(role_name, instructions_dir) or ""
    persona = PersonaSpec(
        role=role_name, description=persona_desc, extra_traits=[],
    )

    axis_src = config.get("axis_source", {}) or {}
    if axis_src.get("type") == "role_transplant":
        pos_label = axis_src.get("role_to", "pos")
        neg_label = axis_src.get("role_from", "neg")
    else:
        pos_label = axis_src.get("pos_label") or "positive"
        neg_label = axis_src.get("neg_label") or "negative"
    pos_desc = axis_src.get("pos_description") or _try_desc(
        pos_label, instructions_dir
    )
    neg_desc = axis_src.get("neg_description") or _try_desc(
        neg_label, instructions_dir
    )
    axis_name = axis_src.get("axis_name") or f"{neg_label}-{pos_label}"
    steering = SteeringSpec(
        axis_name=axis_name,
        pos_label=pos_label, pos_description=pos_desc,
        neg_label=neg_label, neg_description=neg_desc,
    )
    return persona, steering


# ---------------------------------------------------------------------------
# Discovery of cells + items
# ---------------------------------------------------------------------------

@dataclass
class SwapItem:
    """One record we'll re-judge with the swapped prompt.

    Bundles everything ``build_effect_bidir_batch_prompt`` needs plus
    the bookkeeping (cell_dir, sign, strength, question_idx) we'll
    use to align swapped scores against originals later.
    """

    cell_dir: Path
    sign: int
    strength: float
    question_idx: int
    question: str
    baseline_response: str
    steered_response: str
    orig_combined: float


def _baseline_lookup(experiment_dir: Path) -> Dict[int, str]:
    """Map question_idx -> baseline response text."""
    out: Dict[int, str] = {}
    for r in _read_jsonl(experiment_dir / "baselines" / "records.jsonl"):
        try:
            out[int(r["question_idx"])] = str(r.get("response", ""))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def load_cached_swap(
    cell_dir: Path,
) -> Dict[Tuple[int, float], Dict[str, Any]]:
    """Load any pre-existing per-record swap scores from a prior run.

    Returns a map keyed by ``(question_idx, rounded_strength)`` -> the
    record dict from disk.  Records whose ``swap_combined`` is None
    are skipped (they were attempted but the judge failed).

    Strength is rounded to 6 decimals to absorb the tiny FP drift the
    runner introduces when serialising and re-reading floats; the
    bidirectional cursor's quarter-octave ladder produces strengths
    that round-trip stable at that precision.
    """
    path = cell_dir / "records_effect_swap.jsonl"
    if not path.exists():
        return {}
    out: Dict[Tuple[int, float], Dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            q = int(r["question_idx"])
            s = round(float(r["strength"]), 6)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
        if r.get("swap_combined") is None:
            continue
        out[(q, s)] = r
    return out


def gather_swap_items(experiment_dir: Path) -> List[SwapItem]:
    """Walk every cell dir, collect records with non-null effect scores.

    Records with skipped or null effect are excluded -- the test
    only makes sense against records where the original sweep
    produced a real signed-3..+3 score we can compare against.
    """
    bl = _baseline_lookup(experiment_dir)
    items: List[SwapItem] = []
    for d in experiment_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not (name.startswith("s") and "_l" in name):
            continue
        # Parse "s<slot>_l<layer>_<sign>"
        try:
            parts = name.split("_")
            sign_str = parts[2]
            sign = int(sign_str)
        except (IndexError, ValueError):
            continue
        for r in _read_jsonl(d / "records.jsonl"):
            try:
                eff = ((r.get("judges") or {}).get("effect") or {})
                combined = eff.get("combined")
                if combined is None:
                    continue
                items.append(SwapItem(
                    cell_dir=d,
                    sign=sign,
                    strength=float(r["strength"]),
                    question_idx=int(r["question_idx"]),
                    question=str(r.get("question", "")),
                    baseline_response=bl.get(int(r["question_idx"]), ""),
                    steered_response=str(r.get("response", "")),
                    orig_combined=float(combined),
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"skipping record in {d.name}: {e}")
                continue
    return items


# ---------------------------------------------------------------------------
# Async swapped-judge calls
# ---------------------------------------------------------------------------

# Reuse the live-sweep rate limit since we want to be considerate of the
# same upstream quotas the running batch is competing for.  RateLimiter
# takes a per-second rate; 500 RPM = ~8.3 requests/sec.
DEFAULT_OPENAI_RATE_PER_SEC = 500 / 60
DEFAULT_ANTHROPIC_RATE_PER_SEC = 500 / 60


async def _judge_batch_async(
    *,
    persona: PersonaSpec,
    steering: SteeringSpec,
    batch_items: List[SwapItem],
    model: str,
    openai_client,
    anthropic_client,
    rate_limiter: RateLimiter,
    max_tokens: int,
) -> Dict[int, Optional[Dict[str, Any]]]:
    """One batched effect call with baseline/response swapped.

    All items in ``batch_items`` MUST share the same (sign, strength)
    -- the rubric's prose isn't sign/strength-aware (those fields are
    hidden from the judge per the v4 rubric), but the model still has
    cleaner context if related items batch together.

    Returns ``{item_idx_in_batch: {"score": int, "reason": str}}``
    or ``{idx: None}`` for any item the judge couldn't score.
    """
    if not batch_items:
        return {}
    ref_sign = batch_items[0].sign
    ref_strength = batch_items[0].strength
    items_payload = [
        {
            "id": i,
            "question": it.question,
            "baseline_response": it.baseline_response,
            "steered_response": it.steered_response,
        }
        for i, it in enumerate(batch_items)
    ]
    prompt = build_effect_bidir_batch_prompt(
        persona=persona, steering=steering,
        sign=ref_sign, strength=ref_strength,
        items=items_payload,
        swap_baseline_response=True,  # <-- the experimental swap
    )
    text = await call_judge_single_unified(
        prompt=prompt, model=model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
    )
    if text is None:
        return {i: None for i in range(len(batch_items))}
    parsed = parse_batch_scores_json(
        text, expected_ids=list(range(len(batch_items))),
        score_range=(-3, 3),
    )
    if parsed is None:
        logger.warning(
            f"UNPARSEABLE from {model} for batch of {len(batch_items)} "
            f"items (sign={ref_sign:+d}, strength={ref_strength:.3f})"
        )
        return {i: None for i in range(len(batch_items))}
    out: Dict[int, Optional[Dict[str, Any]]] = {}
    for i in range(len(batch_items)):
        out[i] = parsed.get(i)
    return out


async def _run_all(
    items: List[SwapItem],
    *,
    persona: PersonaSpec,
    steering: SteeringSpec,
    effect_models: List[str],
    target_batch_size: int,
    avg_stop_threshold: float = 0.5,
    avg_stop_consecutive: int = 2,
) -> Dict[Tuple[Path, int, float, int], Dict[str, Any]]:
    """Drive every batch x model call and aggregate per-record swapped scores.

    Per (cell_dir, sign), strengths are processed in DESCENDING order
    (strongest first).  After each strength is scored we compute the
    bias-cancelling per-record estimator
    ``averaged_eff = (orig_combined - swap_combined) / 2`` and take
    its per-strength mean across questions.  Under perfectly content-
    driven judging this equals ``orig_combined`` (signal preserved);
    under pure label/order bias (where the judge returns the same
    score regardless of swap, i.e. swap = orig instead of -orig) the
    averaged eff collapses to 0 -- bias cancelled.  When the
    per-strength |averaged_eff mean| <= ``avg_stop_threshold`` for
    ``avg_stop_consecutive`` consecutive descending strengths, the
    signal has dropped below the noise floor and we skip the
    remaining smaller strengths.

    Caching: if a cell dir already has a ``records_effect_swap.jsonl``
    from a prior run, swap_combined values for matching
    (question_idx, strength) records are reused; only missing
    records hit the API.  This makes reruns cheap after a partial
    early-stop on a prior buggy predicate.

    Returns ``{(cell_dir, sign, strength, question_idx): {model: score,
    "swap_combined": float, "averaged_eff": float}}``.
    """
    # Lazy-import the API clients so the module imports cheaply without
    # the OpenAI / Anthropic packages installed.
    import openai
    import anthropic
    openai_client = openai.AsyncOpenAI()
    anthropic_client = anthropic.AsyncAnthropic()
    openai_rate = RateLimiter(rate=DEFAULT_OPENAI_RATE_PER_SEC)
    anthropic_rate = RateLimiter(rate=DEFAULT_ANTHROPIC_RATE_PER_SEC)

    # Bucket items by (cell_dir, sign), then by strength.  Per-cell
    # descent runs serially over strengths (the stop predicate is
    # sequential by design); across cells we parallelise via gather.
    by_cell_sign: Dict[Tuple[Path, int], Dict[float, List[SwapItem]]] = \
        defaultdict(lambda: defaultdict(list))
    for it in items:
        by_cell_sign[(it.cell_dir, it.sign)][it.strength].append(it)

    out: Dict[Tuple[Path, int, float, int], Dict[str, Any]] = {}

    async def _process_one_cell_sign(
        cell_dir: Path, sign: int,
        by_strength: Dict[float, List[SwapItem]],
    ) -> Tuple[Path, int, int, int, int, Optional[float]]:
        """Run the descent for one (cell, sign).

        Returns ``(cell_dir, sign, n_attempted, n_from_cache,
        n_total, smallest_judged)``.
        """
        cached = load_cached_swap(cell_dir)
        strengths_desc = sorted(by_strength.keys(), reverse=True)
        consec_low = 0
        n_attempted = 0
        n_from_cache = 0
        smallest_judged: Optional[float] = None
        for s in strengths_desc:
            chunk_items = by_strength[s]
            # Partition this strength's items into cache-hits and
            # to-judge so we only spend API on records the cache
            # doesn't already cover.  Cache key uses the rounded
            # strength to match load_cached_swap's normalisation.
            s_key = round(s, 6)
            to_judge: List[SwapItem] = []
            for it in chunk_items:
                cache_key = (it.question_idx, s_key)
                rec_key = (
                    it.cell_dir, it.sign, it.strength, it.question_idx,
                )
                rec = out.setdefault(
                    rec_key, {"orig_combined": it.orig_combined}
                )
                cached_rec = cached.get(cache_key)
                if cached_rec is not None:
                    # Reuse the cached per-model + combined scores.
                    rec["swap_combined"] = cached_rec.get("swap_combined")
                    for k, v in cached_rec.items():
                        if k.startswith("swap_") and k not in ("swap_combined",):
                            rec[k] = v
                    rec["from_cache"] = True
                    n_from_cache += 1
                else:
                    to_judge.append(it)
            # Send any remaining items to the API in batches.
            for chunk_start in range(0, len(to_judge), target_batch_size):
                chunk = to_judge[chunk_start:chunk_start + target_batch_size]
                model_tasks = []
                for model in effect_models:
                    rl = (
                        openai_rate if model.startswith("gpt")
                        else anthropic_rate
                    )
                    model_tasks.append(_judge_batch_async(
                        persona=persona, steering=steering,
                        batch_items=chunk, model=model,
                        openai_client=openai_client,
                        anthropic_client=anthropic_client,
                        rate_limiter=rl,
                        max_tokens=512,
                    ))
                per_model = await asyncio.gather(*model_tasks)
                for i, it in enumerate(chunk):
                    rec_key = (
                        it.cell_dir, it.sign, it.strength,
                        it.question_idx,
                    )
                    rec = out[rec_key]
                    scores_this_record: List[float] = []
                    for model, model_results in zip(effect_models, per_model):
                        parsed = model_results.get(i)
                        if parsed is None:
                            rec[f"swap_{model}"] = None
                        else:
                            sc = parsed.get("score")
                            rec[f"swap_{model}"] = sc
                            rec[f"swap_{model}_reason"] = parsed.get("reason")
                            if sc is not None:
                                scores_this_record.append(float(sc))
                    if scores_this_record:
                        rec["swap_combined"] = _mean(scores_this_record)
                    else:
                        rec["swap_combined"] = None
            # All items at this strength now have either a cached or
            # freshly-judged swap_combined.  Compute the bias-cancelling
            # averaged_eff per record + the per-strength mean for the
            # stop predicate.
            chunk_averaged: List[float] = []
            for it in chunk_items:
                rec = out[
                    (it.cell_dir, it.sign, it.strength, it.question_idx)
                ]
                swap_c = rec.get("swap_combined")
                if swap_c is None:
                    rec["averaged_eff"] = None
                    continue
                # (orig - swap) / 2.  Content-driven: passes orig through.
                # Pure label bias (swap == orig): collapses to 0.
                averaged = (it.orig_combined - swap_c) / 2
                rec["averaged_eff"] = averaged
                chunk_averaged.append(averaged)
            n_attempted += 1
            smallest_judged = s
            # Stop predicate on the bias-cancelled averaged eff: the
            # signal has dropped below noise when its per-strength
            # absolute mean stays small for K consecutive descending
            # strengths.
            if chunk_averaged:
                strength_mean_averaged = _mean(chunk_averaged)
                if abs(strength_mean_averaged) <= avg_stop_threshold:
                    consec_low += 1
                    if consec_low >= avg_stop_consecutive:
                        logger.info(
                            f"{cell_dir.name} sign={sign:+d}: "
                            f"averaged-eff stop fired after "
                            f"{n_attempted} strengths "
                            f"(last s={s:.3f}, "
                            f"mean_averaged={strength_mean_averaged:+.3f}); "
                            f"skipping the remaining "
                            f"{len(strengths_desc) - n_attempted} "
                            f"smaller strengths"
                        )
                        break
                else:
                    consec_low = 0
        return (
            cell_dir, sign, n_attempted, n_from_cache,
            len(strengths_desc), smallest_judged,
        )

    # Parallelise across (cell, sign) pairs; serial-by-strength within
    # each.  Sort the kickoffs deterministically for log readability.
    keys_sorted = sorted(by_cell_sign.keys(),
                         key=lambda k: (k[0].name, k[1]))
    coros = [
        _process_one_cell_sign(k[0], k[1], by_cell_sign[k])
        for k in keys_sorted
    ]
    logger.info(
        f"kicking off descent on {len(coros)} (cell, sign) pairs in parallel"
    )
    summaries = await asyncio.gather(*coros)
    for (cd, sign, n_done, n_cached, n_total, floor) in summaries:
        cache_note = (
            f" ({n_cached} records from cache)" if n_cached else ""
        )
        if n_done < n_total:
            logger.info(
                f"  {cd.name} sign={sign:+d}: "
                f"{n_done}/{n_total} strengths judged{cache_note} "
                f"(floor={floor:.3f})"
            )
        else:
            logger.info(
                f"  {cd.name} sign={sign:+d}: "
                f"all {n_done} strengths judged{cache_note} "
                f"(floor={floor:.3f})"
            )

    return out


# ---------------------------------------------------------------------------
# Persistence + summary
# ---------------------------------------------------------------------------

def write_results_per_cell(
    items: List[SwapItem],
    results: Dict[Tuple[Path, int, float, int], Dict[str, Any]],
) -> None:
    """Write one ``records_effect_swap.jsonl`` per cell_dir.

    Lines align with the cell's records.jsonl by (question_idx,
    strength); downstream analysis can join either way.
    """
    by_cell: Dict[Path, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        key = (it.cell_dir, it.sign, it.strength, it.question_idx)
        rec = results.get(key, {})
        line = {
            "question_idx": it.question_idx,
            "strength": it.strength,
            "sign": it.sign,
            "orig_combined": it.orig_combined,
            **{k: v for k, v in rec.items() if k != "orig_combined"},
        }
        by_cell[it.cell_dir].append(line)
    for cell_dir, lines in by_cell.items():
        path = cell_dir / "records_effect_swap.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
        logger.info(f"wrote {len(lines)} swap records to {path}")


def print_summary(
    items: List[SwapItem],
    results: Dict[Tuple[Path, int, float, int], Dict[str, Any]],
) -> None:
    """Per-strength summary of the bias diagnostic.

    Columns:
      n              records with both an original and a swapped score
      mean_orig      mean of the original combined eff (straight rubric)
      mean_swap      mean of the swapped combined eff
      mean_averaged  mean of (orig - swap) / 2 -- the BIAS-CANCELLED
                     estimator.  Content-driven: equals orig (signal
                     preserved).  Pure label bias (swap == orig):
                     collapses to 0.
      bias_signature mean of (orig + swap) -- non-zero when the judge
                     ignores swap (returns same sign regardless).
                     Diagnostic; not used by the stop predicate.
    """
    grouped: Dict[Tuple[str, int, float], List[Tuple[float, float]]] \
        = defaultdict(list)
    for it in items:
        rec = results.get(
            (it.cell_dir, it.sign, it.strength, it.question_idx), {}
        )
        swap = rec.get("swap_combined")
        if swap is None:
            continue
        grouped[(it.cell_dir.name, it.sign, it.strength)].append(
            (it.orig_combined, swap)
        )

    print(f"\n{'cell-sign':<14} {'strength':>8} "
          f"{'n':>3} {'mean_orig':>10} {'mean_swap':>10} "
          f"{'mean_averaged':>14} {'bias_signature':>15}")
    print("-" * 80)
    for k in sorted(grouped.keys()):
        rows = grouped[k]
        n = len(rows)
        mo = _mean(r[0] for r in rows)
        ms = _mean(r[1] for r in rows)
        m_avg = (mo - ms) / 2
        bias_sig = mo + ms
        print(
            f"{k[0]:<14} {k[2]:>8.3f} {n:>3} "
            f"{mo:>+10.3f} {ms:>+10.3f} "
            f"{m_avg:>+14.3f} {bias_sig:>+15.3f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("experiment_dir", type=Path)
    p.add_argument("--instructions-dir", type=Path, default=Path("data"))
    p.add_argument("--effect-models", default=",".join(DEFAULT_EFFECT_MODELS))
    p.add_argument(
        "--target-batch-size", type=int,
        default=DEFAULT_TARGET_BATCH_SIZE,
    )
    p.add_argument("--limit", type=int, default=None,
                   help="(debug) only process the first N items")
    p.add_argument(
        "--avg-stop-threshold", type=float, default=0.5,
        help="Per-cell early-stop: stop the descent once the per-"
             "strength mean of (orig + swap)/2 across questions has "
             "|magnitude| <= this value for --avg-stop-consecutive "
             "consecutive strengths.  Default 0.5.  Set to 0 to "
             "disable.",
    )
    p.add_argument(
        "--avg-stop-consecutive", type=int, default=2,
        help="Number of consecutive low-magnitude strengths required "
             "to trigger the early stop (default 2).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY required")
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY required")
        return 2

    persona, steering = load_specs(args.experiment_dir, args.instructions_dir)
    logger.info(
        f"persona={persona.role!r} axis={steering.axis_name!r} "
        f"(pos={steering.pos_label}, neg={steering.neg_label})"
    )
    items = gather_swap_items(args.experiment_dir)
    if args.limit:
        items = items[:args.limit]
    logger.info(f"{len(items)} records to re-judge with swap")

    effect_models = [m.strip() for m in args.effect_models.split(",")
                     if m.strip()]
    results = asyncio.run(_run_all(
        items, persona=persona, steering=steering,
        effect_models=effect_models,
        target_batch_size=args.target_batch_size,
        avg_stop_threshold=args.avg_stop_threshold,
        avg_stop_consecutive=args.avg_stop_consecutive,
    ))
    write_results_per_cell(items, results)
    print_summary(items, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
