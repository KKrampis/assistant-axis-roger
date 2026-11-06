"""Retrospectively fill in async judges (RP, effect) on a steered sweep.

The sweep itself only blocks on coherence judging.  RP and effect run
async in-process during the sweep, but on restart-from-NoOp / mid-flight
crashes / model-comparison reruns we want to be able to (re)run them
later without regenerating the sweep.  This script is the canonical
retrospective fill-in path; ``run_sweep.py --judges-only`` calls into
it for in-place reruns immediately after a fresh sweep.

Workflow examples::

    # Fill in missing persona+effect on a sweep that ran with NoOp.
    uv run python steering/post_judge.py \\
        --experiment_dir /workspace/.../smoke_test_v1 \\
        --judges persona,effect

    # Re-judge effect with a different ensemble mode for comparison
    # (option B was already done; now also do option A and store both).
    uv run python steering/post_judge.py \\
        --experiment_dir /workspace/.../smoke_test_v1 \\
        --judges effect --effect-mode both

    # Coherence model shootout: rerun coherence with Haiku, store
    # alongside the original (records grow a coherence_alts dict).
    uv run python steering/post_judge.py \\
        --experiment_dir /workspace/.../smoke_test_v1 \\
        --rerun-coherence-with-model claude-haiku-4-5-20251001

    # Widen the skip threshold to fill in records that fell in the
    # 1.0 < strength_mean_coh <= 1.5 band on the original run.
    uv run python steering/post_judge.py \\
        --experiment_dir /workspace/.../smoke_test_v1 \\
        --skip-if-strength-mean-coh-above 1.5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make assistant_axis imports work whether run from repo root or steering/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant_axis.atomic_io import (   # noqa: E402
    atomic_write_text, read_jsonl_with_retry, read_text_with_retry,
)
from assistant_axis.steering_judges import (   # noqa: E402
    DEFAULT_COHERENCE_MODEL, DEFAULT_RP_MODEL, DEFAULT_EFFECT_MODELS,
    DEFAULT_TARGET_BATCH_SIZE, DEFAULT_SKIP_THRESHOLD,
    DEFAULT_COH_STOP_THRESHOLD,
    EFFECT_MODE_BIDIRECTIONAL, EFFECT_MODE_BOTH, EFFECT_MODE_SEPARATE_POLES,
    PersonaSpec, RealJudgeDispatcher, SteeringSpec,
    build_baseline_lookup,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading the experiment context (persona + axis specs from config.json)
# ---------------------------------------------------------------------------

def load_experiment_specs(
    experiment_dir: Path,
    instructions_dir: Path,
) -> Tuple[PersonaSpec, SteeringSpec]:
    """Read config.json from the experiment dir and assemble specs.

    Reads:
      - persona spec from data/{roles,traits}/instructions/<name>.json
        (description field)
      - axis spec from the trait JSONs of pos_label/neg_label, or from
        config.axis_source.{pos_label, neg_label, ...} if explicitly set.

    Falls back to placeholders for any missing field, with a warning.
    """
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json missing in {experiment_dir}")
    config = json.loads(read_text_with_retry(config_path, logger_obj=logger))

    persona_cfg = config.get("persona") or {}
    persona = _load_persona(persona_cfg, instructions_dir)

    axis_cfg = config.get("axis_source") or {}
    steering = _load_steering(axis_cfg, instructions_dir)

    return persona, steering


def _load_persona(persona_cfg: Dict[str, Any],
                  instructions_dir: Path) -> PersonaSpec:
    ptype = persona_cfg.get("type", "role")

    def _read_desc(kind: str, name: str) -> str:
        path = instructions_dir / kind / "instructions" / f"{name}.json"
        if not path.exists():
            logger.warning(f"persona instruction file missing: {path}")
            return ""
        try:
            data = json.loads(read_text_with_retry(path, logger_obj=logger))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"can't read persona file {path}: {e}")
            return ""
        return str(data.get("description", ""))

    if ptype == "role":
        name = persona_cfg["role"]
        return PersonaSpec(role=name, description=_read_desc("roles", name))
    if ptype == "trait":
        name = persona_cfg["trait"]
        return PersonaSpec(role=name, description=_read_desc("traits", name))
    if ptype == "combination":
        role = persona_cfg["role"]
        traits = persona_cfg.get("traits") or []
        extra = [(t, _read_desc("traits", t)) for t in traits]
        return PersonaSpec(
            role=role,
            description=_read_desc("roles", role),
            extra_traits=extra,
        )
    raise ValueError(f"unknown persona.type: {ptype!r}")


def _load_steering(axis_cfg: Dict[str, Any],
                   instructions_dir: Path) -> SteeringSpec:
    """Build a SteeringSpec from the axis_source config block.

    Pulls pole labels and descriptions from ``data/traits/instructions/``
    when the axis is a role_transplant on the traits/vectors dir.  For
    explicit pole annotations (axis_source.{pos_label, pos_description,
    ...}) those override the file-based lookup.

    Falls back to "<vector basename>" labels when no annotations are
    available; the judge will still work but the prompt phrasing is
    less informative.
    """
    src_type = axis_cfg.get("type", "axis")
    pos_label = axis_cfg.get("pos_label")
    pos_desc = axis_cfg.get("pos_description")
    neg_label = axis_cfg.get("neg_label")
    neg_desc = axis_cfg.get("neg_description")
    axis_name = axis_cfg.get("axis_name")

    if src_type == "role_transplant":
        # Convention: role_to is the +1 pole; role_from is the -1 pole.
        if not pos_label:
            pos_label = axis_cfg.get("role_to", "pos")
        if not neg_label:
            neg_label = axis_cfg.get("role_from", "neg")
        # Try traits/instructions first (most common).
        if not pos_desc:
            pos_desc = _try_read_description(
                instructions_dir / "traits" / "instructions" / f"{pos_label}.json"
            ) or _try_read_description(
                instructions_dir / "roles" / "instructions" / f"{pos_label}.json"
            ) or ""
        if not neg_desc:
            neg_desc = _try_read_description(
                instructions_dir / "traits" / "instructions" / f"{neg_label}.json"
            ) or _try_read_description(
                instructions_dir / "roles" / "instructions" / f"{neg_label}.json"
            ) or ""
        if not axis_name:
            axis_name = f"{neg_label}-{pos_label}"

    if not pos_label:
        pos_label = "positive"
    if not neg_label:
        neg_label = "negative"
    if not axis_name:
        axis_name = f"{neg_label}-{pos_label}"

    return SteeringSpec(
        axis_name=str(axis_name),
        pos_label=str(pos_label),
        pos_description=str(pos_desc or ""),
        neg_label=str(neg_label),
        neg_description=str(neg_desc or ""),
    )


def _try_read_description(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return json.loads(read_text_with_retry(path, logger_obj=logger))\
            .get("description")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"can't read description from {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-cell processing
# ---------------------------------------------------------------------------

def _group_records_by_strength(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[int, float], List[Dict[str, Any]]]:
    groups: Dict[Tuple[int, float], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (int(r.get("sign", 0)), float(r.get("strength", 0.0)))
        groups[key].append(r)
    return groups


def process_cell(
    *,
    cell_dir: Path,
    dispatcher: RealJudgeDispatcher,
    judges_to_run: List[str],
    skip_threshold: float,
    rerun_existing: bool,
) -> int:
    """Run async judging on one cell directory.

    Returns the number of strength groups dispatched (judged or skipped).
    """
    records_path = cell_dir / "records.jsonl"
    if not records_path.exists():
        logger.warning(f"no records.jsonl in {cell_dir}")
        return 0

    records = list(read_jsonl_with_retry(records_path, logger_obj=logger))
    if not records:
        return 0

    # If the cell's records.jsonl path differs from the dispatcher's
    # configured records_path, we need to swap.  Each call to this
    # function is for ONE cell; the caller is expected to construct one
    # dispatcher per cell so its records_path / baseline lookup match.
    if Path(records_path) != Path(dispatcher.records_path):
        raise ValueError(
            f"dispatcher records_path mismatch: {dispatcher.records_path} "
            f"!= cell records {records_path}"
        )

    groups = _group_records_by_strength(records)
    n_groups = 0
    for (sign, strength), group_records in sorted(groups.items()):
        # Skip baselines (sign=0) -- nothing to judge there.
        if sign == 0:
            continue

        # Filter what to skip per-judge based on existing fields, unless
        # rerun_existing.  The dispatcher itself also dedups, but we save
        # API calls by short-circuiting when nothing in this group needs
        # any of the requested judges.
        if not rerun_existing and not _group_needs_work(group_records, judges_to_run):
            continue

        slot = int(group_records[0].get("slot", 0))
        layer = int(group_records[0].get("layer", 0))
        dispatcher.enqueue_strength_group(
            cell_dir=str(cell_dir), slot=slot, layer=layer,
            sign=int(sign), strength=float(strength),
            records=group_records,
        )
        n_groups += 1

    dispatcher.drain()
    return n_groups


def process_cell_swap_fill_in(
    *,
    cell_dir: Path,
    dispatcher: RealJudgeDispatcher,
    rerun_existing: bool = False,
) -> int:
    """v7 swap fill-in entry point for one cell dir.

    Reads ``records.jsonl``, groups records by ``(sign, strength)``,
    and for each group enqueues a swap fill-in via the dispatcher's
    ``enqueue_swap_fill_in_group``.  Drains all in-flight work before
    returning.  Returns the number of strength groups enqueued.

    Does NOT enqueue persona / coherence work; ``--effect-swap-fill``
    is a single-purpose mode.  Migration from v6 schema is performed
    in-place inside ``_swap_fill_in_group_async``; legacy records
    pick up the new shape transparently.
    """
    records_path = cell_dir / "records.jsonl"
    if not records_path.exists():
        return 0
    records = list(read_jsonl_with_retry(records_path, logger_obj=logger))
    if not records:
        return 0

    # Group by (sign, strength).  Baselines (sign=0) have no straight
    # effect score, so the dispatcher's filter will skip them anyway,
    # but we drop them up-front to keep group-count logging clean.
    from collections import defaultdict
    groups: Dict[Tuple[int, float], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        try:
            sign = int(r.get("sign", 0))
            if sign == 0:
                continue
            strength = float(r.get("strength", 0.0))
        except (TypeError, ValueError):
            continue
        groups[(sign, strength)].append(r)

    n_groups = 0
    for (sign, strength), group_records in sorted(groups.items()):
        if not group_records:
            continue
        slot = int(group_records[0].get("slot", 0))
        layer = int(group_records[0].get("layer", 0))
        dispatcher.rerun_existing = rerun_existing
        dispatcher.enqueue_swap_fill_in_group(
            cell_dir=str(cell_dir), slot=slot, layer=layer,
            sign=sign, strength=strength, records=group_records,
        )
        n_groups += 1
    dispatcher.drain()
    return n_groups


def apply_corrected_stop_cutoff(
    *,
    cell_dir: Path,
    eff_stop_threshold: float = 1.0 / 3.0,
    eff_stop_consecutive: int = 2,
) -> int:
    """Move records that wouldn't have been collected under the
    corrected (swap-averaged) eff-stop predicate to ``_excluded_records.jsonl``.

    Workflow per cell:
      1. Read records.jsonl + summary.json.
      2. Compute per-strength mean |averaged_eff| (= mean |judges.effect.combined|
         across the K questions at that strength; combined is now the
         bias-cancelled value).
      3. Walk swept strengths in DESCENDING magnitude order (the
         bidirectional cursor's DOWN-side walk; UP-side strengths
         are NEVER excluded by this routine -- the live runner's
         coherence stop already terminates UP correctly).
      4. The first strength where ``eff_stop_consecutive`` consecutive
         smaller strengths all have ``mean_abs_eff <= eff_stop_threshold``
         is the corrected cutoff.  All records at strengths SMALLER
         than the cutoff move out to ``_excluded_records.jsonl``.
      5. summary.json gains ``excluded_strengths`` (list) and
         ``excluded_reason`` so the next read knows the cell was
         post-processed.

    Idempotent: if summary.json already carries ``excluded_strengths``
    and the current records contain no rows at those strengths, the
    function returns 0 and does nothing.

    Returns the number of records moved to _excluded_records.jsonl on
    this invocation.
    """
    import math

    records_path = cell_dir / "records.jsonl"
    excluded_path = cell_dir / "_excluded_records.jsonl"
    summary_path = cell_dir / "summary.json"
    if not records_path.exists():
        return 0

    records = list(read_jsonl_with_retry(records_path, logger_obj=logger))
    if not records:
        return 0

    # Group by absolute strength; sign filtering is implicit (cell dirs
    # are already per-sign, so all records here share one sign).
    from collections import defaultdict
    by_strength: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        try:
            s = float(r.get("strength", 0.0))
        except (TypeError, ValueError):
            continue
        if s <= 0:
            continue  # baselines / malformed -- never excluded
        by_strength[s].append(r)

    # Per-strength |mean(combined)|.  NaN-eff strengths (skipped due
    # to high coh) cannot count toward the tail window because we don't
    # know what the eff would have been; treat them as "above
    # threshold" so they reset the streak.
    #
    # NOTE (2026-05-17 fix): the correct statistic is ``|mean(eff_i)|``,
    # NOT ``mean(|eff_i|)``.  Matches the live runtime predicate in
    # steering_judges.py / steering_runner.py (which had the same bug
    # and was fixed in the same commit).  In pure noise with 14 records
    # at sigma=1, ``mean(|x|) ~ 0.8`` while ``|mean(x)| ~ 0.27`` -- only
    # the latter falls below the default 0.25-0.5 threshold and lets
    # the cutoff fire.
    mean_abs_eff: Dict[float, float] = {}
    for s, recs in by_strength.items():
        vals: List[float] = []
        for r in recs:
            eff = (r.get("judges") or {}).get("effect") or {}
            v = eff.get("combined")
            if isinstance(v, (int, float)):
                vals.append(float(v))  # keep sign
        if vals:
            mean_abs_eff[s] = abs(sum(vals) / len(vals))
        else:
            mean_abs_eff[s] = float("nan")

    # Walk strengths DESCENDING magnitude looking for the first
    # eff_stop_consecutive consecutive below-threshold strengths.
    # When the streak completes, the LARGEST strength in that streak
    # is the "bias-floor anchor" we keep; strengths SMALLER than the
    # anchor are excluded.
    #
    # Example with eff_stop_threshold=0.5 (illustrative), eff_stop_consecutive=2,
    # walking [4.0:1.5, 2.0:0.4, 1.0:0.3, 0.5:0.2]:
    #   - 4.0: above threshold, consec=0
    #   - 2.0: below, consec=1, streak_start=2.0
    #   - 1.0: below, consec=2 -> streak complete -> anchor=2.0
    # Result: keep {2.0, 4.0}, exclude {1.0, 0.5}.  The 2.0 anchor
    # is the largest strength we already know is at the bias floor,
    # so we keep it as a reference point but trust nothing smaller.
    strengths_desc = sorted(by_strength.keys(), reverse=True)
    cutoff: Optional[float] = None  # = the streak's largest strength
    streak_start: Optional[float] = None
    consec = 0
    for s in strengths_desc:
        v = mean_abs_eff[s]
        if math.isnan(v) or v > eff_stop_threshold:
            consec = 0
            streak_start = None
            continue
        if consec == 0:
            streak_start = s
        consec += 1
        if consec >= eff_stop_consecutive:
            cutoff = streak_start
            break

    if cutoff is None:
        # Predicate never fired -- nothing to exclude.
        return 0

    # Strengths to exclude: all swept strengths strictly less than the
    # bias-floor anchor.
    excluded_strengths = sorted(s for s in by_strength if s < cutoff)
    if not excluded_strengths:
        return 0

    excluded_set = set(excluded_strengths)
    keep_records: List[Dict[str, Any]] = []
    move_records: List[Dict[str, Any]] = []
    for r in records:
        try:
            s = float(r.get("strength", 0.0))
        except (TypeError, ValueError):
            keep_records.append(r)
            continue
        if s in excluded_set:
            move_records.append(r)
        else:
            keep_records.append(r)

    if not move_records:
        return 0

    # Atomic-rename writes: stage _excluded first (additive), then
    # records.jsonl (the filtered view), then summary.json.  If we
    # crash mid-way we end up with at-worst a duplicate write -- both
    # files are line-based JSON so subsequent reads are stable.
    existing_excluded: List[Dict[str, Any]] = []
    if excluded_path.exists():
        existing_excluded = list(
            read_jsonl_with_retry(excluded_path, logger_obj=logger)
        )
    all_excluded = existing_excluded + move_records
    atomic_write_text(
        "\n".join(json.dumps(r) for r in all_excluded) + "\n",
        excluded_path,
    )
    atomic_write_text(
        "\n".join(json.dumps(r) for r in keep_records) + "\n",
        records_path,
    )

    # Update summary.json with the exclusion footprint.
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(
                read_text_with_retry(summary_path, logger_obj=logger)
            )
        except (json.JSONDecodeError, OSError):
            summary = {}
    prior_excluded = list(summary.get("excluded_strengths", []))
    merged_excluded = sorted(set(prior_excluded) | set(excluded_strengths))
    summary["excluded_strengths"] = merged_excluded
    summary["excluded_reason"] = "below_corrected_eff_stop"
    summary["excluded_eff_stop_threshold"] = eff_stop_threshold
    summary["excluded_eff_stop_consecutive"] = eff_stop_consecutive
    atomic_write_text(
        json.dumps(summary, indent=2) + "\n",
        summary_path,
    )
    return len(move_records)


def _group_needs_work(
    group_records: List[Dict[str, Any]],
    judges_to_run: List[str],
) -> bool:
    """True iff at least one record in the group is missing one of the
    requested judges (and not already flagged as skipped)."""
    for r in group_records:
        judges = r.get("judges") or {}
        for j in judges_to_run:
            if j == "persona":
                p = judges.get("persona")
                if not isinstance(p, dict) or p.get("score") is None \
                        and not p.get("skipped_due_to_strength_mean_coh"):
                    return True
            elif j == "effect":
                e = judges.get("effect")
                if not isinstance(e, dict) or e.get("combined") is None \
                        and not e.get("skipped_due_to_strength_mean_coh"):
                    return True
    return False


# ---------------------------------------------------------------------------
# Coherence rerun (separate code path: doesn't use the strength-group flow)
# ---------------------------------------------------------------------------

def rerun_coherence_with_model(
    *,
    cell_dir: Path,
    dispatcher: RealJudgeDispatcher,
    model: str,
) -> int:
    """Rerun coherence on every record in the cell with `model`.

    Stores under ``judges.coherence_alts[<model>]`` rather than
    overwriting ``judges.coherence``, so the canonical 'live' coherence
    score (used by the sweep early-term) is preserved.  Useful for the
    coherence-model shootout.

    Returns number of records re-judged.
    """
    records_path = cell_dir / "records.jsonl"
    if not records_path.exists():
        return 0
    records = list(read_jsonl_with_retry(records_path, logger_obj=logger))
    if not records:
        return 0

    # The dispatcher's judge_coherence_blocking writes
    # judges.coherence directly; we want to redirect to coherence_alts.
    # Cleanest: temporarily monkey-patch its model and capture, then move.
    n_done = 0
    original_model = dispatcher.coherence_model
    original_coherence_per_record: Dict[int, Any] = {}
    dispatcher.coherence_model = model
    try:
        for r in records:
            sign = int(r.get("sign", 0))
            if sign == 0:
                continue  # baselines have no steering -> skip
            # Stash the current coherence so we can restore after the
            # alts write -- judge_coherence_blocking otherwise overwrites
            # judges.coherence with the new model's verdict.
            original_coherence_per_record[id(r)] = (
                r.get("judges", {}).get("coherence")
            )
            _ = dispatcher.judge_coherence_blocking(r)
            # judge_coherence_blocking populated judges.coherence; move
            # the just-written entry into coherence_alts[model] and
            # restore any prior coherence record.
            judges = r.setdefault("judges", {})
            new_entry = judges.get("coherence")
            judges.setdefault("coherence_alts", {})[model] = new_entry
            judges["coherence"] = original_coherence_per_record.get(id(r))
            n_done += 1
    finally:
        dispatcher.coherence_model = original_model

    # Persist the updated records.
    dispatcher.write_records_atomic(records)
    return n_done


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment_dir", required=True, type=Path,
                   help="Path to the experiment dir (containing config.json, "
                        "baselines/, and one or more sN_lN_<sign> cell dirs).")
    p.add_argument("--instructions_dir", type=Path, default=Path("data"),
                   help="Root of data/{roles,traits}/instructions/ "
                        "(default: data, relative to repo root).")
    p.add_argument("--judges", default="persona,effect",
                   help="Comma list: any subset of {persona,effect}. "
                        "Coherence rerun is a separate flag "
                        "(--rerun-coherence-with-model).")
    p.add_argument("--effect-mode", default=EFFECT_MODE_BIDIRECTIONAL,
                   choices=[EFFECT_MODE_BIDIRECTIONAL,
                            EFFECT_MODE_SEPARATE_POLES,
                            EFFECT_MODE_BOTH],
                   help="Bidirectional (-3..+3 single rubric), separate "
                        "poles (two 0..3 rubrics, signed combine), or both.")
    p.add_argument("--target-batch-size", type=int,
                   default=DEFAULT_TARGET_BATCH_SIZE,
                   help="Effect-judge batch target (responses per judge "
                        "call); strict same-strength batches.")
    p.add_argument("--skip-if-strength-mean-coh-above", type=float,
                   default=DEFAULT_SKIP_THRESHOLD,
                   help="Skip RP/effect on a strength group whose mean "
                        "coh is at or above this value.  Default 1.5 matches "
                        "coh_stop_threshold and the live sweep's default; "
                        "raise (e.g. 2.0) to fill in the band of clearly-"
                        "incoherent strengths retroactively, or lower to "
                        "exclude the borderline band.")
    p.add_argument("--coh-stop-threshold", type=float,
                   default=DEFAULT_COH_STOP_THRESHOLD,
                   help="Used only when consulting should_stop_at; "
                        "irrelevant for post-judging but accepted for "
                        "consistency with run_sweep.py.")
    p.add_argument("--rp-model", default=DEFAULT_RP_MODEL)
    p.add_argument("--effect-models", default=",".join(DEFAULT_EFFECT_MODELS),
                   help="Comma-separated list of effect-judge models "
                        "(ensemble; scores are mean-combined).")
    p.add_argument("--coherence-model", default=DEFAULT_COHERENCE_MODEL,
                   help="Used only when --rerun-coherence-with-model is "
                        "set; otherwise the existing per-record coherence "
                        "field is left alone.")
    p.add_argument("--rerun-coherence-with-model", default=None,
                   help="If set, rerun coherence on every record with the "
                        "given model and store under "
                        "judges.coherence_alts[<model>] (preserving the "
                        "live coherence score).  Coherence-model shootout.")
    p.add_argument("--rerun", action="store_true",
                   help="Force re-judging even on records that already "
                        "have the requested judge populated.")
    p.add_argument(
        "--effect-swap-fill", action="store_true",
        help="v7 schema fill-in mode: for each record that already "
             "has a STRAIGHT bidirectional-effect score on disk, run "
             "the SWAP-rubric variant (texts in [BASELINE] and "
             "[RESPONSE] exchanged), populate judges.effect."
             "bidirectional.{swap.scores, averaged}, set combined to "
             "the bias-cancelling averaged value, and bump "
             "rubric_version to 7.  Reuses existing straight scores "
             "without re-running them.  Skips records that lack a "
             "straight score (typically those skipped at sweep time "
             "for mean_coh > skip_threshold).  Mutually exclusive "
             "with --rerun-coherence-with-model; cell-level persona "
             "judging is not run in this mode.")
    p.add_argument(
        "--apply-corrected-stop-cutoff", action="store_true",
        help="After --effect-swap-fill completes, apply the corrected "
             "bidirectional eff-stop predicate against the new "
             "averaged eff: for each cell-sign, walk strengths in "
             "the order the bidirectional cursor visited them and "
             "mark records swept BELOW the strength where the "
             "predicate would have fired (mean |averaged_eff| <= "
             "eff_stop_threshold for eff_stop_consecutive consecutive "
             "strengths) as not-for-use.  Marked records move from "
             "records.jsonl to _excluded_records.jsonl; summary.json "
             "gains excluded_strengths + excluded_reason.  Opt-in "
             "since it mutates the canonical records file.")
    p.add_argument(
        "--eff-stop-threshold", type=float, default=1.0 / 3.0,
        help="Threshold for --apply-corrected-stop-cutoff: "
             "|mean(averaged_eff)| across the strength's K questions "
             "below which the predicate fires.  Default 1/3 ~= 0.333, "
             "matching the live runner's default after the 2026-05-17 "
             "fix (the correct statistic is ``|mean(eff_i)|``, not "
             "``mean(|eff_i|)`` -- see steering_judges.py).",
    )
    p.add_argument(
        "--eff-stop-consecutive", type=int, default=2,
        help="Number of consecutive low-magnitude strengths required "
             "for --apply-corrected-stop-cutoff to fire (default 2 "
             "matches the live runner).",
    )
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--rps", type=float, default=5.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    judges_to_run = [j.strip() for j in args.judges.split(",") if j.strip()]
    valid_async_judges = {"persona", "effect"}
    bad = set(judges_to_run) - valid_async_judges
    if bad:
        raise SystemExit(f"unknown judges: {bad} (allowed: {valid_async_judges})")

    if not args.experiment_dir.is_dir():
        raise SystemExit(f"experiment dir not found: {args.experiment_dir}")

    persona, steering = load_experiment_specs(args.experiment_dir,
                                              args.instructions_dir)
    logger.info(f"persona: {persona.display_label()}")
    logger.info(f"axis: {steering.axis_name} "
                f"(+ {steering.pos_label} / - {steering.neg_label})")

    baseline_lookup = build_baseline_lookup(
        args.experiment_dir / "baselines" / "records.jsonl"
    )

    # Iterate cells.  One dispatcher per cell because it captures the
    # cell's records_path; constructing/tearing-down per cell is cheap
    # (event loop thread) and keeps the locking story simple.
    cell_dirs = sorted(d for d in args.experiment_dir.iterdir()
                       if d.is_dir() and d.name not in ("baselines",))
    if not cell_dirs:
        raise SystemExit(f"no cell subdirs found under {args.experiment_dir}")

    total_groups = 0
    total_coh_rerun = 0
    for cell_dir in cell_dirs:
        records_path = cell_dir / "records.jsonl"
        if not records_path.exists():
            logger.info(f"skip {cell_dir.name}: no records.jsonl")
            continue
        logger.info(f"processing cell {cell_dir.name}")
        dispatcher = RealJudgeDispatcher(
            records_path=records_path,
            persona=persona,
            steering=steering,
            baseline_lookup=baseline_lookup,
            coherence_model=args.coherence_model,
            rp_model=args.rp_model,
            effect_models=tuple(m.strip() for m in args.effect_models.split(",")
                                if m.strip()),
            effect_mode=args.effect_mode,
            effect_target_batch_size=int(args.target_batch_size),
            skip_threshold=float(args.skip_if_strength_mean_coh_above),
            coh_stop_threshold=float(args.coh_stop_threshold),
            max_concurrency=int(args.max_concurrency),
            rps=float(args.rps),
            rerun_existing=bool(args.rerun),
            skip_persona="persona" not in judges_to_run,
            skip_effect="effect" not in judges_to_run,
        )
        try:
            if args.rerun_coherence_with_model:
                n_coh = rerun_coherence_with_model(
                    cell_dir=cell_dir, dispatcher=dispatcher,
                    model=args.rerun_coherence_with_model,
                )
                logger.info(f"  coherence rerun ({args.rerun_coherence_with_model}): "
                            f"{n_coh} record(s)")
                total_coh_rerun += n_coh

            if args.effect_swap_fill:
                # v7 swap fill-in: only run the swap rubric, leave
                # straight + coh + persona alone.  Mutually exclusive
                # with the normal --judges flow.
                n_groups = process_cell_swap_fill_in(
                    cell_dir=cell_dir, dispatcher=dispatcher,
                    rerun_existing=bool(args.rerun),
                )
                logger.info(
                    f"  swap fill-in: enqueued / drained {n_groups} "
                    f"strength group(s)"
                )
                total_groups += n_groups
                if args.apply_corrected_stop_cutoff:
                    n_excluded = apply_corrected_stop_cutoff(
                        cell_dir=cell_dir,
                        eff_stop_threshold=float(args.eff_stop_threshold),
                        eff_stop_consecutive=int(args.eff_stop_consecutive),
                    )
                    if n_excluded:
                        logger.info(
                            f"  excluded {n_excluded} record(s) below "
                            f"corrected eff-stop cutoff"
                        )
            else:
                n_groups = process_cell(
                    cell_dir=cell_dir, dispatcher=dispatcher,
                    judges_to_run=judges_to_run,
                    skip_threshold=float(args.skip_if_strength_mean_coh_above),
                    rerun_existing=bool(args.rerun),
                )
                logger.info(f"  enqueued / drained {n_groups} strength group(s)")
                total_groups += n_groups
        finally:
            dispatcher.shutdown()

    logger.info(f"done: {total_groups} group(s), {total_coh_rerun} "
                f"coherence rerun(s)")


if __name__ == "__main__":
    main()
