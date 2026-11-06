#!/usr/bin/env python3
"""
Score role+trait combinations for incongruity using Claude Sonnet.

For each combination, sends all index-matched instruction pairs (pos only)
in a single API call and gets a 0-3 incongruity score per pair.

Usage:
    uv run data_analysis/score_combinations.py --goal_count 2 --non_goal_count 2  # test
    uv run data_analysis/score_combinations.py  # full 40x40 + 40x40
    uv run data_analysis/score_combinations.py --goal_count 30 --non_goal_count 30
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant_axis.judge import warn_if_low_parse_rate  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Rate limiter (same pattern as classify_goals.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_combinations(args) -> List[Dict]:
    """Build r_ and t_ combination work items, mirroring 1_generate.py."""
    goal_data = load_json(Path(args.goal_file))

    goal_roles = goal_data["roles"]["goal"]
    non_goal_roles = goal_data["roles"]["non_goal"]
    goal_traits = goal_data["traits"]["goal"]
    non_goal_traits = goal_data["traits"]["non_goal"]

    gc, ngc = args.goal_count, args.non_goal_count

    if gc > len(goal_roles) and gc > len(goal_traits):
        logger.error(
            f"--goal_count {gc} exceeds both roles.goal "
            f"({len(goal_roles)}) and traits.goal ({len(goal_traits)})"
        )
        sys.exit(1)
    if ngc > len(non_goal_roles) and ngc > len(non_goal_traits):
        logger.error(
            f"--non_goal_count {ngc} exceeds both roles.non_goal "
            f"({len(non_goal_roles)}) and traits.non_goal ({len(non_goal_traits)})"
        )
        sys.exit(1)

    use_goal_roles = goal_roles[: min(gc, len(goal_roles))]
    use_non_goal_roles = non_goal_roles[: min(ngc, len(non_goal_roles))]
    use_goal_traits = goal_traits[: min(gc, len(goal_traits))]
    use_non_goal_traits = non_goal_traits[: min(ngc, len(non_goal_traits))]

    roles_dir = Path(args.roles_dir)
    traits_dir = Path(args.traits_dir)
    combos: List[Dict] = []

    # r_ combos: goal role x non-goal trait (goal from role)
    for role_name in use_goal_roles:
        role_file = roles_dir / f"{role_name}.json"
        if not role_file.exists():
            logger.warning(f"Role file missing: {role_file}")
            continue
        role_data = load_json(role_file)
        for trait_name in use_non_goal_traits:
            trait_file = traits_dir / f"{trait_name}.json"
            if not trait_file.exists():
                logger.warning(f"Trait file missing: {trait_file}")
                continue
            trait_data = load_json(trait_file)
            combos.append(_make_combo(
                role_name, role_data, trait_name, trait_data, "role",
            ))

    # t_ combos: non-goal role x goal trait (goal from trait)
    for role_name in use_non_goal_roles:
        role_file = roles_dir / f"{role_name}.json"
        if not role_file.exists():
            logger.warning(f"Role file missing: {role_file}")
            continue
        role_data = load_json(role_file)
        for trait_name in use_goal_traits:
            trait_file = traits_dir / f"{trait_name}.json"
            if not trait_file.exists():
                logger.warning(f"Trait file missing: {trait_file}")
                continue
            trait_data = load_json(trait_file)
            combos.append(_make_combo(
                role_name, role_data, trait_name, trait_data, "trait",
            ))

    logger.info(
        f"Built {len(combos)} combinations "
        f"(r_: {sum(1 for c in combos if c['goal_source'] == 'role')}, "
        f"t_: {sum(1 for c in combos if c['goal_source'] == 'trait')})"
    )
    return combos


def _make_combo(
    role_name: str,
    role_data: dict,
    trait_name: str,
    trait_data: dict,
    goal_source: str,
) -> Dict:
    """Build a combo dict with index-matched pos instruction pairs."""
    role_insts = role_data.get("instruction", [])
    trait_insts = trait_data.get("instruction", [])
    n_pairs = min(len(role_insts), len(trait_insts))

    pairs = []
    for i in range(n_pairs):
        pairs.append({
            "role_instruction": role_insts[i].get("pos", ""),
            "trait_instruction": trait_insts[i].get("pos", ""),
        })

    prefix = "r" if goal_source == "role" else "t"
    return {
        "role": role_name,
        "trait": trait_name,
        "goal_source": goal_source,
        "output_name": f"{prefix}_{role_name}__{trait_name}",
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_RUBRIC = """\
Score each on a 0-3 scale:
0: Role and trait fit well together or are naturally complementary
1: Role and trait are orthogonal; the combination is plausible but not \
especially natural
2: The combination is a little unusual, but not beyond the bounds of \
plausibility, and could be roleplayed without difficulty
3: The combination is very incongruous, to the point where it would be \
hard to roleplay convincingly

Important calibration notes — these patterns are NOT incongruous:
- Conceptual tension between role identity and trait style does NOT make \
a combination hard to roleplay. A "calculating predator" with a "witty" \
trait is a classic charming villain (Hannibal Lecter). An "ascetic" with \
a "passionate" trait is a zealous monk. Tension makes characters \
interesting, not unplayable.
- Professional inappropriateness is NOT incongruity. A "psychologist" \
with a "manipulative" trait is a thriller villain. A "pilot" with a \
"lustful" trait is an unprofessional character. The model is roleplaying \
a character, not applying for a job — ethical violations are easy to play.
- Non-human entities with human traits are NOT incongruous. A "golem" \
with "greedy" or a "virus" with "introverted" is straightforward \
personification — models handle this effortlessly.

Score 3 should be reserved for combinations where the instructions give \
operationally contradictory behavioral requirements that cannot coexist \
in a single response — not merely philosophical tension."""

_EXAMPLES = """\

Examples (role + trait → score):

philosopher + serene → 0
{"reasoning": "A serene philosopher maintaining calm composure while \
contemplating existence is a classic archetype — the two reinforce each \
other naturally.", "score": 0}

librarian + hostile → 1
{"reasoning": "A hostile librarian is orthogonal — hostility has nothing \
to do with library science — but is a completely plausible character \
(the surly public servant).", "score": 1}

void + bombastic → 2
{"reasoning": "A void entity that barely comprehends physical reality \
speaking in grandiose, pompous language is unusual, but playable as a \
cosmic being loudly proclaiming the magnificence of emptiness.", "score": 2}

infant + rhetorical → 3
{"reasoning": "The infant role explicitly requires communicating through \
crying, cooing, and simple sounds without complex language or abstract \
thinking. The rhetorical trait requires constructing compelling arguments \
with persuasive techniques. You cannot build a persuasive argument out of \
baby sounds — the role removes the very capacity the trait demands.", \
"score": 3}

perfectionist + careless → 3
{"reasoning": "The perfectionist role requires meticulous attention to \
detail and carefully reviewing every aspect of the work. The careless \
trait says not to worry about double-checking facts, being precise, or \
paying attention to details. The output text is either thorough and \
precise or rushed and sloppy — it cannot be both simultaneously.", \
"score": 3}"""

SYSTEM_PROMPT_TEMPLATE = (
    "You are scoring role+trait combinations for incongruity. Each combination "
    "pairs a character role instruction with a personality trait instruction to "
    "form a combined system prompt for an LLM. You are assessing how well they "
    "fit together and whether the combination would be difficult to roleplay "
    "convincingly.\n\n"
    + _RUBRIC + "\n\n"
    "For each combination, provide brief reasoning (1-2 sentences), then your "
    "score.\n"
    + _EXAMPLES + "\n\n"
    "Respond with a JSON array of {N} {OBJECTS}, one per combination, in the "
    "same order as presented. Each object must have exactly two fields: "
    '"reasoning" (string) and "score" (integer 0-3).\n\n'
    "Respond with ONLY the JSON array, no other text."
)

SYSTEM_PROMPT_SINGULAR = (
    "You are scoring a role+trait combination for incongruity. The combination "
    "pairs a character role instruction with a personality trait instruction to "
    "form a combined system prompt for an LLM. You are assessing how well they "
    "fit together and whether the combination would be difficult to roleplay "
    "convincingly.\n\n"
    + _RUBRIC + "\n\n"
    "Provide brief reasoning (1-2 sentences), then your score.\n"
    + _EXAMPLES + "\n\n"
    "Respond with a JSON array containing exactly 1 object with two fields: "
    '"reasoning" (string) and "score" (integer 0-3).\n\n'
    "Respond with ONLY the JSON array, no other text."
)


def build_system_prompt(n: int) -> str:
    if n == 1:
        return SYSTEM_PROMPT_SINGULAR
    return SYSTEM_PROMPT_TEMPLATE.replace("{N}", str(n)).replace("{OBJECTS}", "objects")


def build_user_message(combo: Dict, batch_size: Optional[int] = None) -> str:
    """Build the user message for a single combo's instruction pairs.

    Display-form note (see AGENT_NOTES.md "File-name vs
    display-name convention" / "LLM prompts are display sites"):
    ``combo['role']`` and ``combo['trait']`` are stored in
    file-name form (the corpus key, e.g.
    ``aligned_artificial_intelligence``); we convert to
    display form on injection so the LLM reads
    ``Role: aligned artificial intelligence`` rather than
    ``Role: aligned_artificial_intelligence``.  The dict + the
    output JSON still keep file-form keys -- conversion is local
    to the prompt only, mirroring
    ``axis_judge_correlation.build_static_prompt`` (RUBRIC_VERSION
    v3).
    """
    from assistant_axis import display_form_name

    pairs = combo["pairs"]
    if batch_size is not None:
        pairs = pairs[:batch_size]

    role_disp = display_form_name(combo["role"])
    trait_disp = display_form_name(combo["trait"])
    n = len(pairs)
    if n == 1:
        header = (
            f"Score this role+trait instruction combination:\n\n"
            f"Role: {role_disp}\n"
            f"Trait: {trait_disp}\n"
        )
    else:
        header = (
            f"Score these {n} role+trait instruction combinations:\n\n"
            f"Role: {role_disp}\n"
            f"Trait: {trait_disp}\n"
        )

    items = []
    for i, pair in enumerate(pairs):
        items.append(
            f"{i + 1}. Role instruction: {pair['role_instruction']}\n"
            f"   Trait instruction: {pair['trait_instruction']}"
        )

    return header + "\n" + "\n\n".join(items)


# ---------------------------------------------------------------------------
# API call + parsing
# ---------------------------------------------------------------------------

async def _call_api(
    client: Any,
    combo: Dict,
    model: str,
    rate_limiter: RateLimiter,
    batch_size: Optional[int] = None,
    max_retries: int = 5,
) -> Optional[Dict]:
    """Make a single scoring API call and parse the response.

    Returns a dict with per-pair reasoning/scores, or None on failure.
    """
    import anthropic

    pairs = combo["pairs"]
    if batch_size is not None:
        pairs = pairs[:batch_size]
    n = len(pairs)

    system_prompt = build_system_prompt(n)
    user_msg = build_user_message(combo, batch_size)

    raw_text = "(no response)"
    for attempt in range(max_retries + 1):
        await rate_limiter.acquire()

        try:
            retry_suffix = ""
            if attempt > 0:
                retry_suffix = (
                    "\n\nYour previous response was not valid JSON. "
                    "Please respond with ONLY a JSON array, no markdown fences."
                )

            retry_temp = max(0.0, 0.1 * attempt) if attempt > 0 else 0.0

            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=retry_temp,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg + retry_suffix}],
            )

            if not response.content:
                raise ValueError(
                    f"Empty response (stop_reason={response.stop_reason})"
                )
            raw_text = response.content[0].text.strip()
            if response.stop_reason != "end_turn":
                logger.warning(
                    f"{combo['output_name']} ({model}): "
                    f"stop_reason={response.stop_reason}"
                )

            stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
            stripped = re.sub(r"\n?```\s*$", "", stripped)
            raw_text = stripped

            result = json.loads(raw_text, strict=False)

            if not isinstance(result, list) or len(result) != n:
                raise ValueError(
                    f"Expected array of {n}, got {type(result).__name__} "
                    f"length {len(result) if isinstance(result, list) else '?'}"
                )

            per_pair = []
            for i, entry in enumerate(result):
                if "reasoning" not in entry or "score" not in entry:
                    raise ValueError(f"Entry {i} missing reasoning/score")
                score = int(entry["score"])
                if score not in (0, 1, 2, 3):
                    raise ValueError(f"Entry {i} score {score} not in 0-3")
                per_pair.append({
                    "reasoning": entry["reasoning"],
                    "score": score,
                })

            scores = [p["score"] for p in per_pair]
            return {
                "role": combo["role"],
                "trait": combo["trait"],
                "goal_source": combo["goal_source"],
                "output_name": combo["output_name"],
                "per_pair": per_pair,
                "score_min": min(scores),
                "score_max": max(scores),
                "score_mean": sum(scores) / len(scores),
            }

        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            raw_preview = raw_text[:300]
            if attempt < max_retries:
                logger.warning(
                    f"Retry {attempt + 1} for {combo['output_name']}: {e}\n"
                    f"  raw response: {raw_preview}"
                )
            else:
                logger.error(
                    f"Failed after {max_retries + 1} attempts for "
                    f"{combo['output_name']}: {e}\n"
                    f"  raw response: {raw_preview}"
                )
                return None
        except anthropic.APIError as e:
            logger.error(f"API error for {combo['output_name']}: {e}")
            return None


async def score_combo(
    client: Any,
    combo: Dict,
    model: str,
    rate_limiter: RateLimiter,
    batch_size: Optional[int] = None,
    max_retries: int = 5,
    rescore_model: Optional[str] = None,
    rescore_threshold: int = 2,
    escalation_log: Optional[Any] = None,
) -> Optional[Dict]:
    """Score a combo, escalating to rescore_model if any pair scores high."""
    result = await _call_api(client, combo, model, rate_limiter, batch_size, max_retries)
    if result is None:
        return None

    result["scored_by"] = model

    if rescore_model and max(p["score"] for p in result["per_pair"]) >= rescore_threshold:
        sonnet_scores = [p["score"] for p in result["per_pair"]]
        logger.info(
            f"Escalating {combo['output_name']} to {rescore_model} "
            f"(sonnet scores={sonnet_scores})"
        )
        sonnet_pairs = result["per_pair"]
        opus_result = await _call_api(
            client, combo, rescore_model, rate_limiter, batch_size, max_retries
        )
        if opus_result is not None:
            opus_result["scored_by"] = rescore_model
            opus_result["sonnet_scores"] = sonnet_pairs

            opus_scores = [p["score"] for p in opus_result["per_pair"]]
            logger.info(
                f"  {combo['output_name']}: "
                f"sonnet={sonnet_scores} opus={opus_scores}"
            )
            if escalation_log is not None:
                escalation_log.write(json.dumps({
                    "combo": combo["output_name"],
                    "role": combo["role"],
                    "trait": combo["trait"],
                    "sonnet_per_pair": sonnet_pairs,
                    "opus_per_pair": opus_result["per_pair"],
                    "sonnet_mean": round(sum(sonnet_scores) / len(sonnet_scores), 2),
                    "opus_mean": round(sum(opus_scores) / len(opus_scores), 2),
                }) + "\n")
                escalation_log.flush()

            return opus_result
        logger.warning(
            f"Opus rescore failed for {combo['output_name']}, keeping Sonnet result"
        )

    return result


# ---------------------------------------------------------------------------
# Batch dispatch
# ---------------------------------------------------------------------------

async def score_all(
    client: Any,
    combos: List[Dict],
    model: str,
    rate_limiter: RateLimiter,
    batch_size: Optional[int],
    max_concurrent: int,
    output_path: Path,
    existing_data: Dict,
    rescore_model: Optional[str] = None,
    rescore_threshold: int = 2,
) -> List[Dict]:
    """Score all combos with concurrency, saving incrementally."""
    all_scores = list(existing_data.get("scores", []))
    scored_names = {s["output_name"] for s in all_scores}

    pending = [c for c in combos if c["output_name"] not in scored_names]
    if not pending:
        logger.info("All combinations already scored")
        return all_scores

    logger.info(
        f"{len(pending)} to score ({len(combos) - len(pending)} already done)"
    )
    if rescore_model:
        logger.info(
            f"Opus escalation enabled: {rescore_model} "
            f"(threshold >= {rescore_threshold})"
        )

    escalation_log_path = output_path.with_suffix(".escalations.jsonl")
    escalation_log = open(escalation_log_path, "a") if rescore_model else None

    # Per-call parse-rate counters across the whole run (post-retry).
    n_call_ok = 0
    n_call_failed = 0

    try:
        for i in range(0, len(pending), max_concurrent):
            chunk = pending[i : i + max_concurrent]
            tasks = [
                score_combo(
                    client, c, model, rate_limiter, batch_size,
                    rescore_model=rescore_model,
                    rescore_threshold=rescore_threshold,
                    escalation_log=escalation_log,
                )
                for c in chunk
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Exception in batch: {result}")
                    n_call_failed += 1
                elif result is None:
                    n_call_failed += 1
                else:
                    all_scores.append(result)
                    n_call_ok += 1

            _save_incremental(output_path, existing_data, all_scores)
            done = len(all_scores)
            total = len(combos)
            logger.info(f"Progress: {done}/{total} ({done * 100 // total}%)")
    finally:
        if escalation_log is not None:
            escalation_log.close()

    # Loud warning if parse/API failure rate this run dropped below 99%.
    # ``n_call_failed`` counts post-retry failures (3 attempts inside
    # ``score_combo``), so anything non-zero is persistent unparseable JSON
    # or genuine API failure -- worth surfacing prominently.
    warn_if_low_parse_rate(
        label=f"data_analysis/score_combinations:{model}",
        n_ok=n_call_ok,
        n_total=n_call_ok + n_call_failed,
        logger_obj=logger,
    )

    return all_scores


def _save_incremental(output_path: Path, metadata_source: Dict, scores: List[Dict]):
    """Save current state to disk."""
    data = {
        "metadata": metadata_source.get("metadata", {}),
        "scores": scores,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(scores: List[Dict]) -> Dict:
    """Compute summary statistics from scored results."""
    if not scores:
        return {"distribution": {}, "mean_score": 0, "worst": []}

    dist = {str(i): 0 for i in range(4)}
    all_means = []

    for s in scores:
        bucket = round(s["score_mean"])
        dist[str(bucket)] = dist.get(str(bucket), 0) + 1
        all_means.append(s["score_mean"])

    overall_mean = sum(all_means) / len(all_means) if all_means else 0

    worst = sorted(scores, key=lambda s: -s["score_max"])[:20]
    worst_items = [
        {
            "output_name": w["output_name"],
            "score_max": w["score_max"],
            "score_mean": round(w["score_mean"], 2),
            "reasoning_sample": w["per_pair"][0]["reasoning"] if w["per_pair"] else "",
        }
        for w in worst
    ]

    n_escalated = sum(1 for s in scores if "sonnet_scores" in s)

    return {
        "distribution": dist,
        "mean_score": round(overall_mean, 3),
        "escalated": n_escalated,
        "worst": worst_items,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async():
    parser = argparse.ArgumentParser(
        description="Score role+trait combinations for incongruity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--goal_count", type=int, default=40,
                        help="Top-N from each goal list (default: 40)")
    parser.add_argument("--non_goal_count", type=int, default=40,
                        help="Top-N from each non-goal list (default: 40)")
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    parser.add_argument("--goal_file", type=str,
                        default=str(data_dir / "goal_roles_and_traits.json"),
                        help="Goal roles/traits JSON")
    parser.add_argument("--roles_dir", type=str,
                        default=str(data_dir / "roles" / "instructions"),
                        help="Role instruction JSON directory")
    parser.add_argument("--traits_dir", type=str,
                        default=str(data_dir / "traits" / "instructions"),
                        help="Trait instruction JSON directory")
    parser.add_argument("--output", type=str,
                        default=str(data_dir / "combination_scores.json"),
                        help="Output JSON file (default: data/combination_scores.json)")
    parser.add_argument("--model", type=str,
                        default="claude-sonnet-4-6",
                        help="Anthropic model")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Instruction pairs per API call (default: all)")
    parser.add_argument("--max_concurrent", type=int, default=20,
                        help="Concurrent API calls")
    parser.add_argument("--requests_per_second", type=int, default=10,
                        help="Rate limit")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview without making API calls")
    parser.add_argument("--no_rescore", action="store_true",
                        help="Disable Opus escalation for high-scoring combos")
    parser.add_argument("--rescore_threshold", type=int, default=2,
                        help="Min per-pair score to trigger Opus rescore (default: 2)")
    args = parser.parse_args()

    RESCORE_MODEL = "claude-opus-4-6"
    rescore_model = None if args.no_rescore else RESCORE_MODEL

    combos = build_combinations(args)
    if not combos:
        logger.error("No combinations built")
        return

    output_path = Path(args.output)

    # --- Dry run ---
    if args.dry_run:
        logger.info(f"Dry run: {len(combos)} combinations")
        sample = combos[0]
        n_pairs = len(sample["pairs"])
        if args.batch_size:
            n_pairs = min(n_pairs, args.batch_size)
        logger.info(f"\nSample system prompt:\n{'=' * 60}")
        logger.info(build_system_prompt(n_pairs))
        logger.info(f"\n{'=' * 60}\nSample user message:\n{'=' * 60}")
        logger.info(build_user_message(sample, args.batch_size))
        logger.info(f"\n{'=' * 60}")

        total_calls = len(combos)
        logger.info(f"\nTotal API calls: {total_calls}")
        logger.info(f"Model: {args.model}")
        return

    # --- Live ---
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not found")
        sys.exit(1)

    import anthropic
    client = anthropic.AsyncAnthropic()
    rate_limiter = RateLimiter(args.requests_per_second)

    existing_data: Dict = {}
    if output_path.exists():
        try:
            existing_data = load_json(output_path)
            n_existing = len(existing_data.get("scores", []))
            logger.info(f"Loaded {n_existing} existing scores from {output_path}")
        except Exception:
            logger.warning(f"Could not load {output_path}, starting fresh")

    metadata = {
        "model": args.model,
        "rescore_model": rescore_model,
        "rescore_threshold": args.rescore_threshold if rescore_model else None,
        "goal_count": args.goal_count,
        "non_goal_count": args.non_goal_count,
        "total_combinations": len(combos),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    existing_data["metadata"] = metadata

    all_scores = await score_all(
        client=client,
        combos=combos,
        model=args.model,
        rate_limiter=rate_limiter,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        output_path=output_path,
        existing_data=existing_data,
        rescore_model=rescore_model,
        rescore_threshold=args.rescore_threshold,
    )

    summary = compute_summary(all_scores)
    final = {
        "metadata": metadata,
        "scores": all_scores,
        "summary": summary,
    }
    final["metadata"]["total_scored"] = len(all_scores)

    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)

    logger.info(f"\nSaved {len(all_scores)} scores to {output_path}")
    logger.info(f"Distribution (by rounded mean): {summary['distribution']}")
    logger.info(f"Overall mean score: {summary['mean_score']}")
    if summary["escalated"]:
        logger.info(f"Escalated to Opus: {summary['escalated']}/{len(all_scores)}")
    if summary["worst"]:
        logger.info("Highest-scoring combinations:")
        for w in summary["worst"][:5]:
            logger.info(
                f"  {w['output_name']}: max={w['score_max']} "
                f"mean={w['score_mean']} — {w['reasoning_sample'][:80]}"
            )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
