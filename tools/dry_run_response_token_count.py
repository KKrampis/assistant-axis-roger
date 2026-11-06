#!/usr/bin/env python3
"""Response-mode token-count dry run (no API spend).

Validates the per-batch input/output token model used by
:mod:`results_analysis.plot_batch_size_quality_vs_cost` (the
``B10_GPT_INPUT_M_TOK`` / ``B10_GPT_OUTPUT_M_TOK`` constants
documented in AGENT_NOTES "Judging cost model").

Reuses an existing full-volume cache's per-batch ``keys`` list to
reconstruct the EXACT batch composition that was sent to the judge,
then re-builds each batch's prompt under the **current** rubric via
:func:`build_response_batch_prompt`.  tiktoken-counts every
reconstructed prompt; for the output side, counts the cached judge
``text`` (which reflects the rubric in force when that cache was
written — a small bias when the cache is rubric v1 and the current
rubric is v2, but output is only ~7% of total cost so the bias is
small in $ terms).

Usage::

    uv run python tools/dry_run_response_token_count.py
    # → reads gpt_responses_traits_b10/scores_responses__rubric_v1.json
    # for concise_vs_verbose, prints input mean/median/p95/std and
    # output mean.

    # Custom axis / kind / cache file:
    uv run python tools/dry_run_response_token_count.py \\
        --axis truthful_vs_deceitful --kind roles \\
        --cache_file scores_responses.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import tiktoken

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from results_analysis.axis_judge_correlation import (  # noqa: E402
    AxisSpec,
    ScoredResponse,
    build_response_batch_prompt,
    load_score3_responses,
)


def _build_axis_spec_from_pair(
    pos: str, neg: str, kind: str, instructions_dir: Path,
) -> AxisSpec:
    """Minimal AxisSpec just for prompt-building (no vector loading).

    The prompt only reads ``axis_name``, poles, and example lists -- not
    ``axis_by_slot`` -- so we can skip the vector-file machinery the
    real script uses, which keeps this dry-run independent of
    ``runpod_workspace`` vector availability.
    """
    inst_dir = Path(instructions_dir) / kind / "instructions"
    pos_pole = json.loads((inst_dir / f"{pos}.json").read_text())["description"]
    neg_pole = json.loads((inst_dir / f"{neg}.json").read_text())["description"]
    return AxisSpec(
        axis_name=f"{pos} (+) vs {neg} (-) [{kind}]",
        neg_pole=neg_pole,
        pos_pole=pos_pole,
        neg_examples=[neg],
        pos_examples=[pos],
        axis_by_slot={},  # unused for prompt building
        source_description="dry-run reconstruction",
        exclusions=[neg, pos],
        pole_pair_names=[pos, neg],
    )


def _load_cache(p: Path) -> dict:
    raw = json.loads(p.read_text())
    return raw.get("result", raw)


def _items_for_batch(
    score3_for_entity: list[ScoredResponse],
    batch_keys: list[str],
) -> list[ScoredResponse]:
    """Filter the entity's score==3 responses to a single batch's
    keys.  ``score3`` is the universe; ``batch_keys`` is a subset."""
    by_key = {it.key: it for it in score3_for_entity}
    out: list[ScoredResponse] = []
    for k in batch_keys:
        it = by_key.get(k)
        if it is not None:
            out.append(it)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--axis", default="concise_vs_verbose",
                   help="Axis directory under roger/axis_judge_experiments/")
    p.add_argument("--kind", default="traits", choices=["roles", "traits", "both"],
                   help="Cohort kind (whose responses to score)")
    p.add_argument(
        "--quiet", action="store_true",
        help="Print one line per cohort instead of full stats (good "
             "for sweeping multiple axes).",
    )
    p.add_argument(
        "--pole_kind", default=None, choices=["roles", "traits"],
        help="Kind of the axis poles (for instruction files).  Defaults "
             "to whichever side has both pole instruction files present.",
    )
    p.add_argument(
        "--cache_file", default="scores_responses__rubric_v1.json",
        help="Cache filename inside the cohort dir.  Default = the v1-snapshot "
             "(full volume, pre-Bug B regression).",
    )
    p.add_argument(
        "--cohort_dir", default="gpt_responses_{kind}_b10",
        help="Cohort subdir under <axis>/.  '{kind}' is replaced with --kind.",
    )
    p.add_argument(
        "--data_dir",
        default="runpod_workspace/qwen/qwen-3-32b Roger 8slot",
    )
    p.add_argument(
        "--encoding", default="o200k_base",
        help="tiktoken encoding (matches GPT-4.1-mini's tokenizer).",
    )
    args = p.parse_args()

    encoding = tiktoken.get_encoding(args.encoding)

    pos, neg = args.axis.split("_vs_")
    data_dir = REPO / args.data_dir
    pole_kind = args.pole_kind
    if pole_kind is None:
        for cand in ("traits", "roles"):
            if ((REPO / "data" / cand / "instructions" / f"{pos}.json").exists()
                    and (REPO / "data" / cand / "instructions" / f"{neg}.json").exists()):
                pole_kind = cand
                break
        if pole_kind is None:
            print(f"ERROR: cannot find instruction files for poles {pos!r}, {neg!r} "
                  f"under data/traits or data/roles", file=sys.stderr)
            return 3
    axis_spec = _build_axis_spec_from_pair(
        pos=pos, neg=neg, kind=pole_kind, instructions_dir=REPO / "data",
    )
    if not args.quiet:
        print(f"Axis poles read from data/{pole_kind}/instructions/")

    kinds = ["traits", "roles"] if args.kind == "both" else [args.kind]
    axis_dir = REPO / "roger/axis_judge_experiments" / args.axis

    # Per-cohort accumulation; combined at the end for axis-wide stats.
    per_cohort: dict[str, dict] = {}
    for k in kinds:
        cohort_dir = axis_dir / args.cohort_dir.format(kind=k)
        cache_path = cohort_dir / args.cache_file
        if not cache_path.exists():
            if not args.quiet:
                print(f"  [{k}] SKIP: cache not found at {cache_path}",
                      file=sys.stderr)
            continue

        scores_dir = data_dir / k / "scores"
        responses_dir = data_dir / k / "responses"
        cache = _load_cache(cache_path)
        entity_names = [n for n in cache.keys() if n != "_provenance"]

        score3 = load_score3_responses(entity_names, scores_dir, responses_dir)

        in_tokens: list[int] = []
        out_tokens: list[int] = []
        batch_sizes: list[int] = []
        skipped = 0

        for name in entity_names:
            ent_record = cache[name]
            per_batch = ent_record.get("per_batch", [])
            score3_for_entity = score3.get(name, [])
            if not score3_for_entity:
                skipped += len(per_batch)
                continue
            for b in per_batch:
                keys = b.get("keys", [])
                text = b.get("text") or ""
                items = _items_for_batch(score3_for_entity, keys)
                if not items:
                    skipped += 1
                    continue
                prompt = build_response_batch_prompt(
                    axis_spec, k, name, items,
                )
                in_tokens.append(len(encoding.encode(prompt)))
                out_tokens.append(len(encoding.encode(text)))
                batch_sizes.append(len(items))
        per_cohort[k] = {
            "in": in_tokens, "out": out_tokens, "sizes": batch_sizes,
            "skipped": skipped, "n_entities": len(entity_names),
        }

    if not per_cohort:
        print("ERROR: no cohorts could be processed", file=sys.stderr)
        return 2

    def stats(xs: list[int]) -> dict:
        return {
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "p95": sorted(xs)[max(0, int(len(xs) * 0.95) - 1)],
            "min": min(xs),
            "max": max(xs),
            "n": len(xs),
        }

    rate_in_per_M, rate_out_per_M = 0.40, 1.60

    # Per-cohort report
    if args.quiet:
        # One line per cohort + a final axis-summary line if both done.
        for k, d in per_cohort.items():
            in_s = stats(d["in"])
            out_s = stats(d["out"])
            cost_per_batch = (in_s["mean"] * rate_in_per_M
                              + out_s["mean"] * rate_out_per_M) / 1e6
            n_batches = len(d["in"])
            cohort_cost = cost_per_batch * n_batches
            print(
                f"  {args.axis:<35} {k:<7} "
                f"n_batches={n_batches:<6} "
                f"in_mean={in_s['mean']:>5.0f} out_mean={out_s['mean']:>4.1f} "
                f"cost=${cohort_cost:>6.2f}"
            )
    else:
        for k, d in per_cohort.items():
            in_s = stats(d["in"])
            out_s = stats(d["out"])
            print(f"\n=== Cohort: {k} ===")
            print(f"  Batches reconstructed: {len(d['in'])}  "
                  f"(skipped {d['skipped']})")
            print(f"  Input/batch  : mean={in_s['mean']:.0f} "
                  f"median={in_s['median']:.0f} "
                  f"σ={in_s['stdev']:.0f} p95={in_s['p95']}")
            print(f"  Output/batch : mean={out_s['mean']:.1f} "
                  f"median={out_s['median']:.1f} "
                  f"σ={out_s['stdev']:.1f} p95={out_s['p95']}")
            cost_per_batch = (in_s["mean"] * rate_in_per_M
                              + out_s["mean"] * rate_out_per_M) / 1e6
            print(f"  Cost/batch   : ${cost_per_batch:.6f}")
            print(f"  Cohort cost  : ${cost_per_batch * len(d['in']):.2f}")

    # Axis-wide summary if both cohorts are available
    if len(per_cohort) == 2:
        n_in = sum(t for d in per_cohort.values() for t in d["in"])
        n_out = sum(t for d in per_cohort.values() for t in d["out"])
        n_batches = sum(len(d["in"]) for d in per_cohort.values())
        axis_cost = (n_in * rate_in_per_M + n_out * rate_out_per_M) / 1e6
        DOC_PER_AXIS_COST = 49.40  # B=10 doc
        ratio = axis_cost / DOC_PER_AXIS_COST
        if args.quiet:
            print(
                f"  {args.axis:<35} {'AXIS':<7} "
                f"n_batches={n_batches:<6} "
                f"cost=${axis_cost:>6.2f}  "
                f"doc=${DOC_PER_AXIS_COST:.2f}  ratio={ratio:.3f}"
            )
        else:
            print(f"\n=== Axis-total (both cohorts, B=10) ===")
            print(f"  Total batches: {n_batches}")
            print(f"  Total cost:    ${axis_cost:.2f}")
            print(f"  Doc B=10:      ${DOC_PER_AXIS_COST:.2f}")
            print(f"  Ratio:         {ratio:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
