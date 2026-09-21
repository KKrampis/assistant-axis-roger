#!/usr/bin/env python3
"""
Score a new judge model against an EXISTING, already-computed axis, without
recomputing activations, projections, or touching a GPU.

Motivation: `axis_judge_correlation.py` unconditionally recomputes
projections from raw activation vectors (`--data_dir` is required), even
though those projections don't change when you just want to try a new judge
model -- they're a pure function of the (fixed) axis direction and the
(fixed) activation vectors, neither of which a new judge model affects.

For roger/axis_judge_experiments/{angel_vs_demon,decisive_vs_indecisive}/,
the per-entity, per-slot, both-metric projections ARE already committed
(projections.json, 571 entities x 8 slots x {raw, whitened}) -- these are
the only two axes in the repo with a full reusable projection cache; every
other axis in the 35-axis GPT-vs-Sonnet comparison only has its aggregate
summary committed (gpt_vs_sonnet_rhos_di.json), not the per-axis raw data
this script needs. See pipeline/SUBSET_RUNS.md and this repo's judge-
comparison work for the fuller picture.

This script: loads a cached projections.json, judges the SAME 571 entities'
descriptions with one or more NEW judge models (any --judge_model
axis_judge_correlation.py itself accepts), correlates the new scores
against the existing projections via the exact same compute_correlations()
/ Spearman machinery, tracks real measured cost via BudgetTracker, and
writes output in the same shape axis_judge_correlation.py itself writes
(so it's a drop-in for downstream tooling expecting that layout).

Usage:
    uv run python results_analysis/judge_tier_cost_eval.py \\
        --pair angel demon --pair_type roles \\
        --projections_file roger/axis_judge_experiments/angel_vs_demon/gpt/projections.json \\
        --judge_model claude-opus-5 claude-sonnet-5 claude-haiku-4-5 \\
        --judge_model gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna \\
        --output_dir roger/axis_judge_experiments/angel_vs_demon_tier_eval \\
        --dry_run   # preview cost, make zero API calls

Drop --dry_run to actually run it. Each --judge_model may be repeated.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
from assistant_axis.judge import RateLimiter, call_judge_single_unified, provider_for_model  # noqa: E402
from assistant_axis.judge_pricing import MultiModelUsage, price_for_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def build_axis_spec(args: argparse.Namespace, instructions_dir: Path) -> AxisSpec:
    name1, name2 = args.pair
    return AxisSpec(
        axis_name=f"{name1} (+) vs {name2} (-) [{args.pair_type}]",
        neg_pole=load_pole_description(instructions_dir, args.pair_type, name2),
        pos_pole=load_pole_description(instructions_dir, args.pair_type, name1),
        neg_examples=[name2],
        pos_examples=[name1],
        axis_by_slot={},  # not needed -- we never recompute the axis direction
        source_description=f"reused from {args.projections_file}",
        exclusions=[name1, name2],
        pole_pair_names=[name1, name2],
    )


async def score_one_judge(
    judge_model: str,
    axis_spec: AxisSpec,
    entities: List[tuple],  # (etype_singular_or_plural, name, description)
    rate_limiter: RateLimiter,
    usage: MultiModelUsage,
    openai_client,
    anthropic_client,
    max_tokens: int,
) -> Dict[str, int]:
    scores: Dict[str, int] = {}
    for etype, name, content in entities:
        prompt = build_static_prompt(axis_spec, etype, name, content)
        text = await call_judge_single_unified(
            prompt=prompt,
            model=judge_model,
            max_tokens=max_tokens,
            rate_limiter=rate_limiter,
            openai_client=openai_client,
            anthropic_client=anthropic_client,
            usage=usage,
        )
        score = parse_signed_score(text) if text else None
        if score is not None:
            scores[name] = score
        else:
            logger.warning(f"{judge_model}: unparseable/empty response for {name!r}")
    return scores


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, metavar=("POS", "NEG"), required=True)
    parser.add_argument("--pair_type", choices=["roles", "traits"], required=True)
    parser.add_argument("--projections_file", type=str, required=True,
                         help="Existing projections.json to correlate new scores against "
                              "(e.g. roger/axis_judge_experiments/angel_vs_demon/gpt/projections.json)")
    parser.add_argument("--instructions_dir", type=str, default="data")
    parser.add_argument("--judge_model", action="append", required=True,
                         help="Repeatable. Any model axis_judge_correlation.py itself accepts "
                              "(gpt-/o1-/o3-/o4- -> OpenAI, claude- -> Anthropic).")
    parser.add_argument("--slots", type=str, default="all",
                         help="Comma-separated slot indices, or 'all' (default; matches every "
                              "slot present in the cached projections.json).")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--requests_per_second", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true",
                         help="Print entity count + estimated cost per judge model; make zero API calls.")
    args = parser.parse_args()

    proj_path = Path(args.projections_file)
    with open(proj_path) as f:
        proj_data = json.load(f)
    projections_raw = proj_data["result"]  # {slot: {name: {raw, whitened}}}
    projections = {int(s): v for s, v in projections_raw.items()}
    if args.slots != "all":
        wanted = {int(s) for s in args.slots.split(",")}
        projections = {s: v for s, v in projections.items() if s in wanted}
    any_slot = next(iter(projections.values()))
    entity_names = sorted(any_slot.keys())
    logger.info(f"Loaded projections for {len(entity_names)} entities across {len(projections)} slots "
                f"from {proj_path}")

    instructions_dir = Path(args.instructions_dir)
    axis_spec = build_axis_spec(args, instructions_dir)

    # Resolve each entity's kind (roles vs traits) and description text.
    # We do NOT assume args.pair_type applies to every entity -- the corpus
    # is mixed. Try both instruction dirs; skip (with a warning) anything
    # that's vanished from data/ since the projections were computed (e.g.
    # a file renamed or removed) -- silently dropping it would just shrink
    # the correlation's n, so warn loudly instead.
    entities = []
    missing = []
    for name in entity_names:
        desc = load_entity_description(instructions_dir, "roles", name)
        etype = "roles"
        if desc is None:
            desc = load_entity_description(instructions_dir, "traits", name)
            etype = "traits"
        if desc is None:
            missing.append(name)
            continue
        entities.append((etype, name, desc))
    if missing:
        logger.warning(f"{len(missing)} entities from the cached projections no longer have an "
                        f"instructions file under {instructions_dir} -- excluded: {missing[:10]}"
                        f"{'...' if len(missing) > 10 else ''}")
    logger.info(f"{len(entities)} entities scorable (description mode)")

    output_dir = Path(args.output_dir)

    for judge_model in args.judge_model:
        provider = provider_for_model(judge_model)
        rate_in, rate_out = price_for_model(judge_model)

        if args.dry_run:
            # Rough estimate: reuse this repo's own measured desc+inst
            # per-item token model (README "Judging cost model") --
            # ~320 header + ~150 per short description, ~80 output.
            # This is a coarse estimate for planning purposes only; the
            # real run tracks ACTUAL tokens via BudgetTracker, not this.
            est_in_tokens = len(entities) * (320 + 150)
            est_out_tokens = len(entities) * 80
            est_cost = est_in_tokens * rate_in / 1e6 + est_out_tokens * rate_out / 1e6
            print(f"{judge_model:28s} provider={provider:9s} rate=${rate_in:.2f}/${rate_out:.2f} per 1M  "
                  f"~{len(entities)} calls  est.cost=${est_cost:.2f}")
            continue

        judge_output_dir = output_dir / judge_model
        judge_output_dir.mkdir(parents=True, exist_ok=True)

        usage = MultiModelUsage()
        rate_limiter = RateLimiter(args.requests_per_second)
        openai_client = None
        anthropic_client = None
        if provider == "openai":
            import openai
            openai_client = openai.AsyncOpenAI()
        else:
            import anthropic
            anthropic_client = anthropic.AsyncAnthropic()

        logger.info(f"Scoring {len(entities)} entities with {judge_model} ({provider})...")
        scores = await score_one_judge(
            judge_model=judge_model, axis_spec=axis_spec, entities=entities,
            rate_limiter=rate_limiter, usage=usage,
            openai_client=openai_client, anthropic_client=anthropic_client,
            max_tokens=args.max_tokens,
        )

        scores_path = judge_output_dir / "scores_descriptions.json"
        scores_path.write_text(json.dumps(
            {"result": scores, "_provenance": {"judge_model": judge_model, "rubric_version": RUBRIC_VERSION,
                                                "projections_file": str(proj_path)}},
            indent=2,
        ))
        usage_path = judge_output_dir / "usage.json"
        usage_path.write_text(json.dumps(usage.as_dict(), indent=2, sort_keys=True))
        logger.info(f"[{judge_model}] scored {len(scores)}/{len(entities)}; "
                    f"cost: ${usage.total_cost_usd:.4f} (see {usage_path})")

        correlations = compute_correlations(
            scores_by_mode={"descriptions": scores},
            projections=projections,
            slots=list(projections.keys()),
            excluded_set=set(axis_spec.exclusions),
        )
        (judge_output_dir / "correlations.json").write_text(json.dumps(correlations, indent=2))
        for slot, per_metric in correlations["descriptions"].items():
            raw = per_metric["raw"]
            logger.info(f"[{judge_model}] slot={slot} raw: rho={raw['rho']:.4f} n={raw['n']}")

    if args.dry_run:
        print("\nDry run only -- no API calls made, no cost incurred.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
