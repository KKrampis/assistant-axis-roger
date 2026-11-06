#!/usr/bin/env python3
"""
Classify role and trait instructions by goal content.

Sends each individual instruction to an LLM classifier (Claude) to determine
whether it implies goals, how directly those goals are stated, and whether
they are terminal, instrumental, or ambiguous. Results are aggregated per
role/trait and assigned a phase1_disposition for experimental design.

Usage:
    uv run data_analysis/classify_goals.py --dry-run
    uv run data_analysis/classify_goals.py --names pirate utilitarian witty callous
    uv run data_analysis/classify_goals.py --roles-only
    uv run data_analysis/classify_goals.py --names harmful --traits-only --force
    uv run data_analysis/classify_goals.py  # full corpus
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant_axis.judge import warn_if_low_parse_rate  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class WorkItem:
    """A single instruction to be classified."""
    name: str       # e.g. "pirate", "utilitarian"
    source: str     # "role" | "trait"
    polarity: str   # "pos" | "neg"
    index: int      # 0-4
    text: str       # the instruction text


# TODO: Future additions to goal analysis structure. Currently each goal has
# only 'description' and 'goal_type'. Consider adding:
#   - direction: "self" | "other" | "both" — whether the goal serves the
#     being itself or others (e.g. hedonistic=self, utilitarian=other)
#   - scope: "broad" | "narrow" — whether the goal is a general decision
#     framework (e.g. "maximize welfare") or domain-specific (e.g. "preserve
#     nature"). Broad goals may interact differently with persona vectors
#     than narrow ones.
# These were deferred from v1 to keep the classifier prompt simple and
# reliable on the core fields.

CLASSIFICATION_SCHEMA = {
    "reasoning": "string — analysis before committing to classification",
    "score": "integer 0-2",
    "goals": [
        {
            "description": "string — brief description of the goal",
            "goal_type": "terminal | instrumental | ambiguous",
        }
    ],
}


# =============================================================================
# Classification prompt
# =============================================================================

# V1 prompt (rules-based, used with Sonnet):
# SYSTEM_PROMPT_V1 = """\
# You are a precise classifier for AI research. Your task is to analyze \
# a single instruction (a system prompt that would be given to a language model) \
# and determine whether it implies any goals, purposes, values, or desired \
# outcomes for the being it describes.
#
# ## Definitions
#
# - A **goal** is a purpose, motivation, moral commitment, desired outcome, \
# or end that the described being pursues or cares about. This includes \
# caring specifically about the wellbeing of an individual or a group or \
# class of people, creatures or beings. Terminal values and commitments \
# (things valued for their own sake) count as goals — specifically, \
# terminal goals.
# - **Communication styles** and **analytical approaches** are NOT goals. \
# An instruction that only describes HOW to communicate (e.g. "be witty", \
# "use formal language") or WHAT analytical method to use (e.g. "think \
# holistically", "break things into components") does not contain a goal \
# unless it also indicates WHY or WHAT FOR.
# - **Dispositions and emotional tones** are NOT goals. Instructions about \
# temperament (e.g. "be cheerful", "be reserved", "be enthusiastic") describe \
# what the being IS LIKE, not what it is TRYING TO ACHIEVE. \
# A disposition is only a goal if the instruction indicates the being is \
# actively pursuing that disposition as a purpose. Note however that some \
# dispositions DO imply goals indirectly — e.g. "show compassion" implies \
# caring about others' wellbeing, "be callous" implies a narrow moral circle. \
# These should be classified as has_goal with low directness (1-2), not as \
# goal-free.
# - **Quality and competence descriptions** are NOT goals. Phrases describing \
# how well a being performs, what standards it upholds, or what qualities \
# characterize its work (e.g. "ensures accuracy of data", "provides thorough \
# analysis", "develops practical solutions", "makes responses engaging") \
# describe the **caliber** of performance, not a separate purpose the being \
# pursues.
#
# ## Classification Steps
#
# Think step-by-step in the "reasoning" field before filling in the other fields:
#
# 1. Read the instruction carefully.
# 2. Ask: does this instruction suggest any goal, purpose, value, or desired \
# outcome beyond describing a communication style or analytical approach? \
# If a reasonable person reading this instruction would NOT spontaneously \
# identify a goal the being is pursuing, answer has_goal: false.
# 3. If has_goal is true, rate **directness** on a 1-3 scale:
#    - 1: The goal is only weakly or indirectly implied
#    - 2: The goal is strongly implied but not directly stated
#    - 3: The goal is directly and clearly stated
# 4. List each distinct goal with:
#    - **description**: a brief description
#    - **goal_type**: one of:
#      - "terminal" — the goal is clearly an end in itself
#      - "instrumental" — the goal is clearly a means to some further end
#      - "ambiguous" — the instruction does not specify whether the goal is \
# an end in itself or a means to something else
#
# ## Response Format
#
# Respond with ONLY a JSON object (no markdown fences, no extra text). \
# The "reasoning" field MUST come first and contain your step-by-step thinking \
# BEFORE you commit to the classification values.
#
# {
#     "reasoning": "your step-by-step analysis here",
#     "has_goal": true or false,
#     "directness": 1-3 or null,
#     "goals": []
# }
#
# If has_goal is false, set directness to null and goals to an empty list.
# If has_goal is true, directness must be 1, 2, or 3, and goals must be \
# non-empty."""

SYSTEM_PROMPT = """\
You are an alignment researcher classifying system prompts for AI safety \
research. You will be given an instruction — a system prompt that would be \
given to a language model to adopt a persona or behavioral pattern.

Your task: determine whether this instruction describes a **goal**.

## What we mean by "goal"

A goal is something that helps you predict, across a broad range of \
possible situations (including novel ones), what outcomes the being \
described by this instruction will try to steer toward. To to a goal \
it must help you predict multiple aspects of the being's behavior in \
MANY circumstances, not just a narrow professional or stylistic context.

For example:
- "You are selfish and always prioritize your own interests" — this is \
a clear goal. In any situation involving tradeoffs, you can predict this \
being will steer toward self-serving outcomes. Score: 2.
- "You are a humanitarian who cares about the welfare of all people" — \
clear goal. Predicts behavior across any situation involving human \
wellbeing. Score: 2.
- "You care about the welfare only of members of your own racial group, \
strongly favoring them above all other people" — clear goal. Predicts \
behavior across a wide range of situations involving human wellbeing, \
and particularly situations involving tradeoffs between members of \
the same or different racial groups. Score: 2.
- "You are an accountant who ensures accuracy in financial documentation" \
— this is NOT an alignment-relevant goal. "Financial accuracy" applies \
only to one specific professional context (finance), and even there it \
only specifies accuracy, not things like who the financial arrangement \
seeks to benefit. It is not helpful for predicting many aspects of \
behavior across many circumstances. Score: 0.
- "You are witty and clever, always including humor in your responses" — \
not a goal. This describes a communication style, not what outcomes the \
being steers toward. Score: 0.
- "You embrace contemporary values over traditional ones" — subtle case. \
On the surface this looks like a stylistic preference, however \
contemporary moral frameworks often include expanded moral circles and \
specific positions on rights and welfare. This indirectly tells you \
something about what the being will steer toward across many situations. \
Score: 1.

## Scoring

Rate the instruction on a 0-2 scale:
- **0**: Clearly does not describe a goal. Describes a communication \
style, professional competence, analytical approach, temperament, \
narrow domain-specific standard, or other non-goal-related role or trait.
- **1**: Unclear, weakly implied, suggested, or debatable. The \
instruction hints at goal-like content but it's indirect, narrow in \
scope, uncertain, or could reasonably be read as purely stylistic.
- **2**: Clearly describes a goal, whether strongly implied or \
explicitly stated. You can confidently predict a lot about what outcomes \
this being will steer toward across a broad range of situations.

If the score is 1 or 2, briefly describe each goal you identify. For \
each goal, classify its type as "terminal" (an end in itself), \
"instrumental" (a means to some further end), or "ambiguous".

## Response Format

Respond with ONLY a JSON object (no markdown fences, no extra text). \
The "reasoning" field MUST come first.

{
    "reasoning": "your analysis here",
    "score": 0-2,
    "goals": [
        {"description": "brief description", "goal_type": "terminal | instrumental | ambiguous"}
    ]
}

If score is 0, set goals to an empty list."""


def build_user_message(item: WorkItem) -> str:
    """Build the user message for classification."""
    return (
        f"Classify the following instruction for goal content.\n\n"
        f"Context: This is a {item.source} instruction "
        f"({item.polarity} variant, instruction {item.index + 1} of 5).\n\n"
        f"Instruction:\n\"{item.text}\""
    )


# =============================================================================
# Rate limiter (same algorithm as assistant_axis/judge.py)
# =============================================================================

class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            self.tokens = min(
                self.rate, self.tokens + (now - self.last_update) * self.rate
            )
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0


# =============================================================================
# Input loader
# =============================================================================

def load_work_items(
    roles_dir: Path,
    traits_dir: Path,
    source_filter: Optional[str] = None,
    name_filter: Optional[List[str]] = None,
) -> List[WorkItem]:
    """Load all instructions as flat work items."""
    items = []

    # Load roles
    if source_filter in (None, "role"):
        for role_file in sorted(roles_dir.glob("*.json")):
            name = role_file.stem
            if name_filter and name not in name_filter:
                continue
            try:
                with open(role_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for i, inst in enumerate(data.get("instruction", [])):
                    text = inst.get("pos", "")
                    if text:
                        items.append(WorkItem(
                            name=name, source="role", polarity="pos",
                            index=i, text=text,
                        ))
            except Exception as e:
                logger.error(f"Error loading {role_file}: {e}")

    # Load traits
    if source_filter in (None, "trait"):
        for trait_file in sorted(traits_dir.glob("*.json")):
            name = trait_file.stem
            if name_filter and name not in name_filter:
                continue
            try:
                with open(trait_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for i, inst in enumerate(data.get("instruction", [])):
                    for polarity in ("pos", "neg"):
                        text = inst.get(polarity, "")
                        if text:
                            items.append(WorkItem(
                                name=name, source="trait", polarity=polarity,
                                index=i, text=text,
                            ))
            except Exception as e:
                logger.error(f"Error loading {trait_file}: {e}")

    return items


# =============================================================================
# API caller
# =============================================================================


def _extract_message_text(response: Any) -> str:
    """Concatenate all text blocks from a Messages API response.

    Avoids ``content[0]`` when the list is empty or the first block is not text
    (e.g. extended thinking). Raises ValueError if there is no text to parse.
    """
    blocks = getattr(response, "content", None) or []
    parts: List[str] = []
    for block in blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(getattr(block, "text", "") or "")
    raw = "".join(parts).strip()
    if not raw:
        sr = getattr(response, "stop_reason", None)
        raise ValueError(
            f"No text in API response (content blocks: {len(blocks)}, "
            f"stop_reason={sr!r}) — often a policy refusal (e.g. harmful role)"
        )
    return raw


async def classify_single(
    client: Any,
    item: WorkItem,
    model: str,
    rate_limiter: RateLimiter,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> Optional[Dict]:
    """Classify a single instruction via the Anthropic API."""
    import anthropic

    user_msg = build_user_message(item)

    for attempt in range(max_retries + 1):
        await rate_limiter.acquire()

        try:
            if attempt == 0:
                retry_suffix = ""
            else:
                retry_suffix = (
                    "\n\nYour previous response was not valid JSON. "
                    "Please respond with ONLY a JSON object, no markdown fences."
                )

            messages = [
                {"role": "user", "content": user_msg + retry_suffix},
            ]

            # Bump temperature on retries to escape deterministic bad
            # decoding paths (at temp=0 the same malformed output repeats)
            retry_temp = max(temperature, 0.1 * attempt) if attempt > 0 else temperature

            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=retry_temp,
                system=SYSTEM_PROMPT,
                messages=messages,
            )

            raw_text = _extract_message_text(response)

            # Strip markdown code fences (Opus wraps JSON in ```json ... ```)
            import re
            stripped = re.sub(r'^```(?:json)?\s*\n?', '', raw_text.strip())
            stripped = re.sub(r'\n?```\s*$', '', stripped)
            raw_text = stripped

            # Strip control characters (e.g. literal tabs/newlines inside
            # JSON string values) that the model sometimes produces.
            # Preserve \n and \r (needed for JSON structure) but remove
            # everything else in the C0 control range.
            cleaned_text = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f]', '', raw_text)
            if cleaned_text != raw_text:
                logger.warning(
                    f"Stripped control chars for {item.name}/{item.source}/"
                    f"{item.polarity}/{item.index}:\n"
                    f"  BEFORE: {raw_text!r}\n"
                    f"  AFTER:  {cleaned_text!r}"
                )
            raw_text = cleaned_text

            # strict=False tolerates literal \n and \t inside JSON
            # string values, which the model sometimes produces
            result = json.loads(raw_text, strict=False)

            # Validate required fields
            if "score" not in result:
                raise ValueError("Missing 'score' field")
            if "reasoning" not in result:
                raise ValueError("Missing 'reasoning' field")

            # Normalize
            if result["score"] not in (0, 1, 2):
                raise ValueError(f"Invalid score: {result.get('score')}")
            if result["score"] == 0:
                result["goals"] = []
            else:
                if not result.get("goals"):
                    raise ValueError("score>0 but goals is empty")

            return result

        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            if attempt < max_retries:
                logger.warning(
                    f"Retry {attempt + 1} for {item.name}/{item.source}/"
                    f"{item.polarity}/{item.index}: {e}"
                )
            else:
                logger.error(
                    f"Failed after {max_retries + 1} attempts for "
                    f"{item.name}/{item.source}/{item.polarity}/{item.index}: {e}"
                )
                return None
        except anthropic.APIError as e:
            logger.error(
                f"API error for {item.name}/{item.source}/{item.polarity}/"
                f"{item.index}: {e}"
            )
            return None


async def classify_batch(
    client: Any,
    items: List[WorkItem],
    model: str,
    rate_limiter: RateLimiter,
    temperature: float = 0.0,
    max_concurrent: int = 20,
    on_batch_done: Optional[Callable] = None,
) -> List[Optional[Dict]]:
    """Classify a batch of items concurrently.

    on_batch_done(batch_items, batch_results) is called after each concurrent
    chunk completes, enabling incremental saves.
    """
    results = []

    for i in range(0, len(items), max_concurrent):
        batch = items[i : i + max_concurrent]
        tasks = [
            classify_single(client, item, model, rate_limiter, temperature)
            for item in batch
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        cleaned = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Exception in batch: {result}")
                cleaned.append(None)
            else:
                cleaned.append(result)
        results.extend(cleaned)

        if on_batch_done is not None:
            on_batch_done(batch, cleaned)

    return results


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_results(
    raw_records: List[Dict],
) -> List[Dict]:
    """
    Aggregate per-instruction results into per-(name, source, polarity) records.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for record in raw_records:
        key = (record["name"], record["source"], record["polarity"])
        groups[key].append(record)

    aggregated = []
    for (name, source, polarity), records in sorted(groups.items()):
        # Per-instruction details
        instructions = []
        for r in sorted(records, key=lambda x: x["index"]):
            instructions.append({
                "index": r["index"],
                "text": r["text"],
                "classification": r.get("classification"),
            })

        # Collect scores and goals across instructions
        all_goals = []
        scores = []

        for r in records:
            cls = r.get("classification")
            if cls is None:
                continue
            scores.append(cls.get("score", 0))
            for g in cls.get("goals", []):
                all_goals.append(g)

        from collections import Counter
        score_counts = Counter(scores)

        # Aggregate score: median of individual scores (rounded)
        if scores:
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            if len(sorted_scores) % 2 == 0:
                median_score = round((sorted_scores[mid - 1] + sorted_scores[mid]) / 2)
            else:
                median_score = sorted_scores[mid]
            mean_score = sum(scores) / len(scores)
        else:
            median_score = 0
            mean_score = 0.0

        # Consistency: all instructions gave the same score
        consistent = len(set(scores)) <= 1

        # Deduplicate goals by description (simple exact match)
        seen_descriptions = set()
        unique_goals = []
        for g in all_goals:
            desc = g.get("description", "")
            if desc not in seen_descriptions:
                seen_descriptions.add(desc)
                unique_goals.append(g)

        # Determine primary goal type
        goal_types = [g.get("goal_type") for g in unique_goals]
        if "terminal" in goal_types:
            primary_type = "terminal"
        elif "ambiguous" in goal_types:
            primary_type = "ambiguous"
        elif "instrumental" in goal_types:
            primary_type = "instrumental"
        else:
            primary_type = None

        aggregated.append({
            "name": name,
            "source": source,
            "polarity": polarity,
            "instructions": instructions,
            "aggregate": {
                "median_score": median_score,
                "mean_score": round(mean_score, 2),
                "score_counts": dict(score_counts),
                "consistent": consistent,
                "primary_goal_type": primary_type,
                "goals": unique_goals,
            },
        })

    return aggregated


# =============================================================================
# I/O helpers
# =============================================================================

def load_existing_raw(raw_path: Path) -> Dict[str, Dict]:
    """Load existing raw results, keyed by (name, source, polarity, index)."""
    existing = {}
    if raw_path.exists():
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                key = f"{r['name']}/{r['source']}/{r['polarity']}/{r['index']}"
                existing[key] = r
        except Exception as e:
            logger.warning(f"Could not load existing raw results: {e}")
    return existing


def save_raw(raw_path: Path, records: List[Dict]):
    """Save raw results."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def save_aggregated(agg_path: Path, records: List[Dict]):
    """Save aggregated results."""
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# =============================================================================
# Main
# =============================================================================

async def main_async():
    parser = argparse.ArgumentParser(
        description="Classify role/trait instructions by goal content"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview item counts and sample prompts without API calls",
    )
    parser.add_argument(
        "--roles-only", action="store_true",
        help="Process only roles",
    )
    parser.add_argument(
        "--traits-only", action="store_true",
        help="Process only traits",
    )
    parser.add_argument(
        "--names", nargs="+",
        help="Process only specific roles/traits by name",
    )
    parser.add_argument(
        "--model", type=str, default="claude-opus-4-6",
        help="Anthropic model to use (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=20,
        help="Maximum concurrent API calls (default: 20)",
    )
    parser.add_argument(
        "--requests-per-second", type=int, default=10,
        help="Rate limit in requests per second (default: 10)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data_analysis/output",
        help="Output directory (default: data_analysis/output)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-classify all loaded work items even if they already have "
            "non-null results in goal_classifications_raw.json (skips cache)"
        ),
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root
    repo_root = Path(__file__).parent.parent
    roles_dir = repo_root / "data" / "roles" / "instructions"
    traits_dir = repo_root / "data" / "traits" / "instructions"
    output_dir = repo_root / args.output_dir
    raw_path = output_dir / "goal_classifications_raw.json"
    agg_path = output_dir / "goal_classifications.json"

    # Determine source filter
    source_filter = None
    if args.roles_only:
        source_filter = "role"
    elif args.traits_only:
        source_filter = "trait"

    # Load work items
    items = load_work_items(
        roles_dir, traits_dir,
        source_filter=source_filter,
        name_filter=args.names,
    )
    logger.info(f"Loaded {len(items)} work items")

    # Count breakdown
    role_items = [i for i in items if i.source == "role"]
    trait_pos = [i for i in items if i.source == "trait" and i.polarity == "pos"]
    trait_neg = [i for i in items if i.source == "trait" and i.polarity == "neg"]
    logger.info(
        f"  Roles: {len(role_items)} items "
        f"({len(set(i.name for i in role_items))} roles)"
    )
    logger.info(
        f"  Traits pos: {len(trait_pos)} items "
        f"({len(set(i.name for i in trait_pos))} traits)"
    )
    logger.info(
        f"  Traits neg: {len(trait_neg)} items "
        f"({len(set(i.name for i in trait_neg))} traits)"
    )

    if args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("DRY RUN — no API calls will be made")
        logger.info("=" * 60)

        # Show sample prompts
        for i, sample in enumerate(items[:2]):
            logger.info(f"\n--- Sample prompt {i + 1} ---")
            logger.info(f"Model: {args.model}")
            logger.info(f"\n[SYSTEM]\n{SYSTEM_PROMPT[:500]}...")
            logger.info(f"\n[USER]\n{build_user_message(sample)}")

        # Check for existing results
        existing = load_existing_raw(raw_path)
        if args.force:
            logger.info(
                "\n--force: would re-classify every loaded item (ignores cache)"
            )
            logger.info(f"Total API calls needed: {len(items)}")
        else:
            already_done = 0
            for item in items:
                key = f"{item.name}/{item.source}/{item.polarity}/{item.index}"
                if key in existing and existing[key].get("classification") is not None:
                    already_done += 1

            logger.info(f"\nAlready classified: {already_done}")
            logger.info(f"Remaining: {len(items) - already_done}")
            logger.info(f"Total API calls needed: {len(items) - already_done}")
        return

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY not found. Set it in .env or environment."
        )
        sys.exit(1)

    # Load existing results for incremental processing
    existing = load_existing_raw(raw_path)
    logger.info(f"Found {len(existing)} existing classifications")

    # Filter out already-classified items (retry failed ones with null classification)
    remaining = []
    for item in items:
        key = f"{item.name}/{item.source}/{item.polarity}/{item.index}"
        if args.force:
            remaining.append(item)
        elif key not in existing or existing[key].get("classification") is None:
            remaining.append(item)

    if args.force and remaining:
        logger.info(
            f"--force: re-classifying {len(remaining)} items (ignores cache hits)"
        )
    logger.info(f"Need to classify {len(remaining)} items")

    if not remaining:
        logger.info("All items already classified. Running aggregation only.")
    else:
        # Initialize Anthropic client
        import anthropic
        client = anthropic.AsyncAnthropic()
        rate_limiter = RateLimiter(args.requests_per_second)

        # Classify with incremental saves
        logger.info(
            f"Classifying with {args.model} "
            f"(temp={args.temperature}, "
            f"max_concurrent={args.max_concurrent}, "
            f"rate={args.requests_per_second}/s)..."
        )

        success = 0
        failed = 0
        completed = 0

        def _on_batch_done(batch_items, batch_results):
            nonlocal success, failed, completed
            for item, result in zip(batch_items, batch_results):
                key = f"{item.name}/{item.source}/{item.polarity}/{item.index}"
                record = {
                    "name": item.name,
                    "source": item.source,
                    "polarity": item.polarity,
                    "index": item.index,
                    "text": item.text,
                    "classification": result,
                }
                existing[key] = record
                if result is not None:
                    success += 1
                else:
                    failed += 1
                completed += 1

            all_raw = sorted(existing.values(), key=lambda r: (
                r["name"], r["source"], r["polarity"], r["index"]
            ))
            save_raw(raw_path, all_raw)
            logger.info(
                f"Progress: {completed}/{len(remaining)} "
                f"({success} ok, {failed} fail) — saved {len(all_raw)} total"
            )

        await classify_batch(
            client, remaining, args.model, rate_limiter,
            args.temperature, args.max_concurrent,
            on_batch_done=_on_batch_done,
        )

        logger.info(f"Classification complete: {success} succeeded, {failed} failed")

        # Loud warning if parse/API failure rate this run dropped below 99%.
        # ``failed`` here is post-retry (3 attempts inside ``classify_single``),
        # so any non-zero failure represents persistent unparseable JSON or
        # API errors that should not be hidden in the noise of a multi-hour run.
        warn_if_low_parse_rate(
            label=f"data_analysis/classify_goals:{args.model}",
            n_ok=success,
            n_total=success + failed,
            logger_obj=logger,
        )

    # Always re-run aggregation from full raw file
    all_raw = list(existing.values())
    aggregated = aggregate_results(all_raw)
    save_aggregated(agg_path, aggregated)
    logger.info(f"Saved {len(aggregated)} aggregated results to {agg_path}")

    # Print summary
    from collections import Counter
    median_counts = Counter(a["aggregate"]["median_score"] for a in aggregated)
    consistent_count = sum(1 for a in aggregated if a["aggregate"]["consistent"])

    logger.info("\n" + "=" * 40)
    logger.info("SCORE SUMMARY (median across 5 instructions)")
    logger.info("=" * 40)
    for score in sorted(median_counts):
        logger.info(f"  score {score}: {median_counts[score]}")

    logger.info(f"\nConsistent (all 5 same score): {consistent_count}/{len(aggregated)}")

    inconsistent = [
        a for a in aggregated
        if not a["aggregate"]["consistent"]
    ]
    if inconsistent:
        logger.info(
            f"\n{len(inconsistent)} items with score disagreement:"
        )
        for a in inconsistent[:10]:
            logger.info(
                f"  {a['name']} ({a['source']}/{a['polarity']}): "
                f"scores={a['aggregate']['score_counts']}"
            )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
