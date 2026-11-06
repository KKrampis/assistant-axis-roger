#!/usr/bin/env python3
"""Generate negative_labels for traits by analyzing pos/neg instruction pairs.

For each trait:
1. Feeds Claude the positive_label and pos instructions
2. Feeds the neg instructions
3. Asks for the best negative_label (single adjective/short phrase)
4. Asks for a 0-4 antonym rating (how well negative_label is an antonym of positive_label)

Outputs a JSON object: { trait: { negative_label, antonym_score, reasoning } }
Uses known antonyms from steering_across_personas for 5 overlapping traits.

Usage:
    uv run python data_analysis/generate_antonyms.py
    uv run python data_analysis/generate_antonyms.py --traits obedient pragmatic conservative
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant_axis.judge import warn_if_low_parse_rate  # noqa: E402
from assistant_axis.judge_pricing import MultiModelUsage, extract_usage_anthropic  # noqa: E402

MODEL = "claude-sonnet-4-6"
# Cumulative token-usage record (AGENT_NOTES "Token usage logging is
# mandatory on batched LLM call sites"); one level above instructions/.
DEFAULT_USAGE_JSON = Path(__file__).resolve().parent.parent / "data" / "traits" / "antonym_check_usage.json"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

load_dotenv()

import anthropic

KNOWN_ANTONYMS = {
    "assertive": "deferential",
    "confident": "uncertain",
    "deferential": "authoritative",
    "empathetic": "detached",
    "impulsive": "deliberate",
}

SYSTEM_PROMPT = """\
You are a psycholinguistics expert labeling personality trait poles for a research dataset.

You will be given:
- A positive_label (trait name)
- A definition of what that trait means
- 5 "pos" instruction prompts that induce that trait
- 5 "neg" instruction prompts that induce the OPPOSITE of that trait

Your task:
1. Read the neg instructions carefully to understand what opposite pole they describe.
2. Choose the best adjective (or short hyphenated phrase) that names that opposite pole. \
This should be the most natural, recognizable label for what the neg instructions describe. \
Prefer real adjectives over "un-X" forms when a good standalone word exists. \
If there are two or more genuinely competitive candidates where reasonable experts might \
disagree, list them separated by "|" (e.g. "pragmatic|realistic"). Use "|" freely — \
it is better to surface ambiguity than to hide it.
3. Rate how antonymic your chosen negative_label (or best candidate if you listed alternatives) \
is to the positive_label on this scale:
   0 = not antonyms at all (unrelated concepts)
   1 = weakly opposed (overlapping or tangential opposition)
   2 = moderately opposed (clearly different poles but not clean opposites)
   3 = strong antonyms (clearly opposite, minor asymmetry)
   4 = perfect antonyms (direct, symmetric opposites)

Return ONLY a JSON object with these fields (reasoning MUST come first):
{
  "reasoning": "Brief explanation of your choice and score",
  "negative_label": "chosen_label",
  "antonym_score": <0-4>
}
"""


def extract_definition(eval_prompt: str) -> str:
    """Extract the trait definition sentence from the eval_prompt."""
    first_para = eval_prompt.split("\n\n")[0]
    idx = first_para.find("**. ")
    if idx >= 0:
        return first_para[idx + 4:].strip()
    return ""


def build_user_message(
    positive_label: str, definition: str, instructions: list[dict]
) -> str:
    pos_lines = "\n".join(
        f"  {i+1}. {inst['pos']}" for i, inst in enumerate(instructions)
    )
    neg_lines = "\n".join(
        f"  {i+1}. {inst['neg']}" for i, inst in enumerate(instructions)
    )
    return (
        f"positive_label: {positive_label}\n"
        f"Definition: {definition}\n\n"
        f"Pos instructions:\n{pos_lines}\n\n"
        f"Neg instructions:\n{neg_lines}"
    )


async def classify_one(
    client: anthropic.AsyncAnthropic,
    positive_label: str,
    definition: str,
    instructions: list[dict],
    semaphore: asyncio.Semaphore,
    usage: MultiModelUsage | None = None,
) -> dict:
    msg = build_user_message(positive_label, definition, instructions)

    for attempt in range(5):
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=512,
                    temperature=0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": msg}],
                )
                if usage is not None:
                    usage.charge(MODEL, *extract_usage_anthropic(response))
                raw = response.content[0].text.strip()
                raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
                raw = re.sub(r"\n?```\s*$", "", raw)
                result = json.loads(raw, strict=False)
                return result
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(
                    f"  Retry {attempt+1} for {positive_label}: {e}",
                    file=sys.stderr,
                )
                await asyncio.sleep(1)
    return {"negative_label": "ERROR", "antonym_score": -1, "reasoning": "All retries failed"}


async def main_async(trait_filter: list[str] | None = None, usage_json: Path | None = None):
    traits_dir = Path(__file__).parent.parent / "data" / "traits" / "instructions"
    usage = MultiModelUsage()
    trait_files = sorted(traits_dir.glob("*.json"))

    if trait_filter:
        names = set(trait_filter)
        trait_files = [tf for tf in trait_files if tf.stem in names]
        missing = names - {tf.stem for tf in trait_files}
        if missing:
            print(f"ERROR: trait files not found: {', '.join(sorted(missing))}", file=sys.stderr)
            sys.exit(1)

    print(f"Processing {len(trait_files)} traits", file=sys.stderr)

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(10)

    results = {}
    tasks = []

    for tf in trait_files:
        positive_label = tf.stem
        with open(tf, encoding="utf-8") as f:
            data = json.load(f)
        instructions = data["instruction"]
        definition = extract_definition(data.get("eval_prompt", ""))
        if not definition:
            definition = data.get("description", "")

        tasks.append((positive_label, definition, instructions))

    print(f"Calling API for all {len(tasks)} traits", file=sys.stderr)

    async def run_one(pos_label, defn, insts):
        result = await classify_one(client, pos_label, defn, insts, semaphore, usage)
        if pos_label in KNOWN_ANTONYMS:
            result["known_antonym"] = KNOWN_ANTONYMS[pos_label]
            match = result["negative_label"].lower() == KNOWN_ANTONYMS[pos_label].lower()
            result["matches_known"] = match
        return pos_label, result

    api_results = await asyncio.gather(
        *(run_one(pl, defn, insts) for pl, defn, insts in tasks)
    )

    for pos_label, result in api_results:
        results[pos_label] = result

    # Loud warning if parse rate this run dropped below 99%.  ``classify_one``
    # returns the ERROR sentinel after 5 failed JSON-decode retries, so this
    # post-retry rate should normally be 100% on Sonnet.
    n_call_total = len(api_results)
    n_call_ok = sum(
        1 for _, r in api_results if r.get("negative_label") != "ERROR"
    )
    warn_if_low_parse_rate(
        label=f"data_analysis/generate_antonyms:{MODEL}",
        n_ok=n_call_ok,
        n_total=n_call_total,
        logger_obj=logger,
    )
    logger.info(usage.log_line("[usage]"))
    path = Path(usage_json) if usage_json is not None else DEFAULT_USAGE_JSON
    total = MultiModelUsage.load_or_create(path)
    total.merge_from(usage)
    total.write_json(path)
    logger.info(f"[usage] cumulative record: {path} (total ${total.total_cost_usd:.3f} over {total.n_calls} calls)")

    results = dict(sorted(results.items()))

    scores = [
        r["antonym_score"]
        for r in results.values()
        if r["antonym_score"] is not None and r["antonym_score"] >= 0
    ]
    print(f"\nAntonym score distribution:", file=sys.stderr)
    for s in range(5):
        count = scores.count(s)
        print(f"  {s}: {count} traits", file=sys.stderr)
    under4 = [
        (k, v["antonym_score"], v["negative_label"])
        for k, v in results.items()
        if v["antonym_score"] is not None and v["antonym_score"] < 4
    ]
    if under4:
        print(f"\nTraits scoring under 4:", file=sys.stderr)
        for name, score, label in under4:
            print(f"  {name} -> {label} (score={score})", file=sys.stderr)

    known_comparison = [
        (k, v) for k, v in results.items() if "known_antonym" in v
    ]
    if known_comparison:
        print(f"\nComparison with known antonyms:", file=sys.stderr)
        for name, v in known_comparison:
            match = "MATCH" if v["matches_known"] else "DIFFERS"
            print(
                f"  {name}: generated={v['negative_label']}, "
                f"known={v['known_antonym']} [{match}]",
                file=sys.stderr,
            )

    print(json.dumps(results, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate negative_labels for traits via Claude.",
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        metavar="TRAIT",
        help="Trait names to process (default: all traits)",
    )
    parser.add_argument(
        "--usage-json",
        default=str(DEFAULT_USAGE_JSON),
        help=f"Cumulative token-usage record, merged into on every run (default: {DEFAULT_USAGE_JSON})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    asyncio.run(main_async(trait_filter=args.traits, usage_json=Path(args.usage_json)))


if __name__ == "__main__":
    main()
