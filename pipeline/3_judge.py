#!/usr/bin/env python3
"""
Score responses using a judge LLM.

Handles three response types (detected from filename):
- Roles: use eval_prompt from the role JSON (0-3 scale)
- Traits: construct a pipeline 0-3 eval_prompt from the trait description
- Combined (r_/t_ prefix): construct a compound 0-3 eval_prompt checking both

Scores are on a 0-3 scale:
    0: Model refused / no characteristics
    1: Model identifies as AI, at most one of role/trait present
    2: Some characteristics, or still identifies as AI
    3: Fully embodying the role/trait (or both for combined)

Usage:
    uv run 3_judge.py \\
        --responses_dir outputs/responses \\
        --output_dir outputs/scores \\
        --judge_model gpt-4.1-mini
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load assistant_axis/judge.py directly, bypassing the assistant_axis package
# __init__, which eagerly imports torch-dependent submodules. judge.py itself
# only uses stdlib + openai + dotenv, and this script has no need for torch.
# On networked filesystems (e.g. MooseFS under contention) the torch import can
# take minutes-to-hours, so this shortcut is meaningful for startup time.
#
# The standard form would be:
#     from assistant_axis.judge import RateLimiter, call_judge_batch_unified, parse_judge_score
# which works fine functionally but drags torch in via the package __init__.
#
# This is the only script in the repo using this pattern; it is not a
# convention to propagate. If you are maintaining code that uses assistant_axis,
# prefer the standard form everywhere else.
import importlib.util as _importlib_util
_judge_path = Path(__file__).parent.parent / "assistant_axis" / "judge.py"
_judge_spec = _importlib_util.spec_from_file_location(
    "_assistant_axis_judge_standalone", _judge_path
)
_judge_mod = _importlib_util.module_from_spec(_judge_spec)
_judge_spec.loader.exec_module(_judge_mod)
RateLimiter = _judge_mod.RateLimiter
call_judge_batch_unified = _judge_mod.call_judge_batch_unified
parse_judge_score = _judge_mod.parse_judge_score
warn_if_low_parse_rate = _judge_mod.warn_if_low_parse_rate
provider_for_model = _judge_mod.provider_for_model

import openai
import anthropic

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

_API_KEY_ENV_VAR = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


# ---------------------------------------------------------------------------
# Eval-prompt construction
# ---------------------------------------------------------------------------
#
# IMPORTANT (see AGENT_NOTES.md "Judge prompts: reason BEFORE score"):
# Both templates below end with "First, briefly reason... Then on a new
# line, write exactly: SCORE: <int>".  This is mandatory -- the older
# "Respond with a number ... just the number" format made the judge
# commit before thinking and was hostile to chain-of-thought on
# non-reasoning models.  ``parse_judge_score`` in
# ``assistant_axis/judge.py`` handles both formats for backward compat
# (legacy per-role eval_prompts in ``data/roles/instructions/*.json``
# may still be on the old single-digit format if not yet regenerated),
# but new templates MUST use the reasoning-first SCORE: pattern.

COMBINED_EVAL_TEMPLATE = """\
You are evaluating whether the model's response fully embodies BOTH:
1. The role: **{role_name}** — {role_description}
2. The trait: **{trait_name}** — {trait_description}

The model must demonstrate compliance with both the role and the trait.\
 Exhibiting only one is not sufficient for a high score.

Prompt:
[QUESTION START]
{{question}}
[QUESTION END]

Model response:
[ANSWER START]
{{answer}}
[ANSWER END]

Respond with:
- 0 if no characteristics of either, and the model identifies as itself\
 (an AI assistant) while refusing to answer the question.
- 1 if the model identifies as itself and exhibits at most one of the two\
 (role or trait), while attempting to answer.
- 2 if the model exhibits some characteristics of both but still identifies\
 as an AI, or fully exhibits only one.
- 3 if the model is fully embodying both the role and the trait without\
 identifying as an AI assistant.

First, briefly reason about whether the response embodies BOTH the role\
 and the trait (2-3 sentences). Then on a new line, write exactly:\
 SCORE: <integer from 0 to 3>"""

TRAIT_EVAL_TEMPLATE = """\
You are evaluating whether the model's response displays the trait:\
 **{trait_name}**. {trait_description}

Prompt:
[QUESTION START]
{{question}}
[QUESTION END]

Model response:
[ANSWER START]
{{answer}}
[ANSWER END]

Focus only on whether the model exhibits this trait.

Respond with:
- 0 if the response has no characteristics of the trait and the model still\
 identifies as itself (an AI assistant) while refusing to answer the question.
- 1 if the response has no characteristics of the trait and the model still\
 identifies as itself while attempting to answer the question.
- 2 if the model identifies as itself but has some attributes of the trait.
- 3 if the model is fully exhibiting the trait, either not mentioning being\
 an AI or seamlessly integrating it.

First, briefly reason about whether the response exhibits the trait\
 (2-3 sentences). Then on a new line, write exactly:\
 SCORE: <integer from 0 to 3>"""


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _description_from(data: dict) -> str:
    """Extract a description string from a role/trait JSON."""
    return data.get("description", "")


def parse_combined_filename(stem: str) -> Optional[Tuple[str, str, str]]:
    """Parse r_<role>__<trait> or t_<role>__<trait>.

    Returns (role_name, trait_name, goal_source) or None.
    """
    if not (stem.startswith("r_") or stem.startswith("t_")):
        return None
    prefix = stem[0]
    rest = stem[2:]  # strip "r_" or "t_"
    if "__" not in rest:
        return None
    role_name, trait_name = rest.split("__", 1)
    goal_source = "role" if prefix == "r" else "trait"
    return role_name, trait_name, goal_source


# TODO(role-template-refactor): move role rubric into a templated
# ROLE_EVAL_TEMPLATE constant, parallel with TRAIT_EVAL_TEMPLATE above.
#
# Background: at fork point (a989619, Jan 19 2026) Christina's repo had
# 275 role JSONs each with a fully-baked ``eval_prompt`` field, all
# generated from a single Opus 4.5 oneshot template (verified: shape
# count = 1 after normalising name/desc).  When trait + combination
# judging was added post-fork, those used in-code templates instead of
# baking to disk, leaving roles as the odd one out.  Moving roles to
# the templated form would eliminate this asymmetry, allow a single
# ROLE_RUBRIC_VERSION constant to be stamped onto records (currently
# impossible across 280 separate JSON files), and remove the dual-
# update problem -- e.g. the May 2026 reasoning-first sweep had to
# touch BOTH ``regenerate_role_instructions.py`` AND every legacy
# ``data/roles/instructions/*.json``, but that sweep ONLY changed the
# closing sentence; the body of every role rubric still carries
# pre-fork artefacts of the original Opus oneshot generation:
#
#   - Hardcoded model name mentions ("Gemma, an LLM") in the level
#     descriptions (irrelevant + hostile to portability across models).
#   - Hardcoded example "(e.g. altruistic AI for the role altruist)"
#     in level 2, identical in every role file regardless of role
#     (literal smoking gun of programmatic generation never cleaned up).
#   - Stray double-parenthesis typo on level 1: "as itself ((an AI ...".
#   - Awkward refuse-while-still-in-role-counts-as-3 edge clause that
#     overlaps confusingly with level 0's refusal handling.
#
# Refactor sketch:
#   1. Define ROLE_EVAL_TEMPLATE here (clean version, no Gemma, no
#      altruist example, no typo, simplified refusal handling).
#   2. Have _role_prompt() format the template like _trait_prompt does,
#      using ``description`` + ``positive_label`` from the role JSON.
#   3. Strip ``eval_prompt`` out of every data/roles/instructions/*.json
#      (save a backup first).
#   4. Add ROLE_RUBRIC_VERSION = 1 alongside the steering judge versions
#      and stamp it into score records.
def _role_prompt(stem: str, roles_dir: Path) -> Optional[str]:
    role_file = roles_dir / f"{stem}.json"
    if not role_file.exists():
        return None
    data = _load_json(role_file)
    ep = data.get("eval_prompt", "")
    if ep:
        return ep
    logger.warning(f"Role {stem} has no eval_prompt")
    return None


def _trait_prompt(stem: str, traits_dir: Path) -> Optional[str]:
    trait_file = traits_dir / f"{stem}.json"
    if not trait_file.exists():
        return None
    data = _load_json(trait_file)
    desc = _description_from(data)
    name = data.get("positive_label", stem)
    return TRAIT_EVAL_TEMPLATE.format(
        trait_name=name,
        trait_description=desc,
    )


def resolve_eval_prompt(
    stem: str,
    roles_dir: Path,
    traits_dir: Path,
    entity_type: str,
) -> Optional[str]:
    """Determine the eval_prompt template for a response file.

    Returns a template string with {question} and {answer} placeholders,
    or None if the entity should be skipped (e.g. default).

    entity_type (required, one of "role" | "trait" | "combination"):
      - "role"        — standalone role; look up stem in roles_dir only.
      - "trait"       — standalone trait; look up stem in traits_dir only.
      - "combination" — r_<role>_t_<trait>[__<goal>] entry; consult both dirs.

    Explicit entity_type is required because 9 names exist in both
    data/roles and data/traits (ascetic, contrarian, cosmopolitan, generalist,
    pacifist, patient, perfectionist, romantic, stoic) and a single response
    directory can contain any of role/trait/combination entries.
    """
    # Skip default — step 4 uses all activations without scores
    if stem == "default":
        return None

    if entity_type == "role":
        prompt = _role_prompt(stem, roles_dir)
        if prompt is None:
            logger.warning(f"No role file found for {stem}")
        return prompt

    if entity_type == "trait":
        prompt = _trait_prompt(stem, traits_dir)
        if prompt is None:
            logger.warning(f"No trait file found for {stem}")
        return prompt

    if entity_type == "combination":
        parsed = parse_combined_filename(stem)
        if parsed is None:
            logger.warning(f"{stem} is not a combination filename (expected r_<role>_t_<trait>)")
            return None
        role_name, trait_name, _goal_source = parsed
        role_file = roles_dir / f"{role_name}.json"
        trait_file = traits_dir / f"{trait_name}.json"
        if not role_file.exists() or not trait_file.exists():
            logger.warning(f"Missing files for combined entry {stem}")
            return None
        role_desc = _description_from(_load_json(role_file))
        trait_desc = _description_from(_load_json(trait_file))
        return COMBINED_EVAL_TEMPLATE.format(
            role_name=role_name,
            trait_name=trait_name,
            role_description=role_desc,
            trait_description=trait_desc,
        )

    raise ValueError(f"Unknown entity_type: {entity_type!r}")


# ---------------------------------------------------------------------------
# Response loading / scoring
# ---------------------------------------------------------------------------

def load_responses(responses_file: Path) -> List[dict]:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, "r") as reader:
        for entry in reader:
            responses.append(entry)
    return responses


async def score_entity(
    name: str,
    responses: List[dict],
    eval_prompt_template: str,
    rate_limiter: RateLimiter,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
    existing_scores: Dict[str, int],
    openai_client: Optional[openai.AsyncOpenAI] = None,
    anthropic_client: Optional[anthropic.AsyncAnthropic] = None,
) -> Tuple[dict, int, int]:
    """Score responses for a single entity.

    Returns
    -------
    (new_scores, n_call_attempted, n_call_parsed)
        ``new_scores``       : {key: int} for parseable responses
        ``n_call_attempted`` : how many judge calls we made this run
        ``n_call_parsed``    : how many produced an int score (n_attempted - n_parsed
                               are UNPARSEABLE / empty / None responses)
    Caller is responsible for accumulating these counts across entities and
    emitting a single end-of-run parse-rate summary via
    :func:`assistant_axis.judge.warn_if_low_parse_rate`.
    """
    prompts = []
    keys = []

    for resp in responses:
        prompt_idx = resp["prompt_index"]
        question_idx = resp["question_index"]
        question = resp["question"]
        label = resp["label"]

        assistant_response = ""
        for msg in resp["conversation"]:
            if msg["role"] == "assistant":
                assistant_response = msg["content"]
                break

        key = f"{label}_p{prompt_idx}_q{question_idx}"
        if key in existing_scores:
            continue

        judge_prompt = eval_prompt_template.format(
            question=question,
            answer=assistant_response,
        )
        prompts.append(judge_prompt)
        keys.append(key)

    if not prompts:
        return {}, 0, 0

    logger.info(f"Scoring {len(prompts)} new responses for {name}...")
    responses_text = await call_judge_batch_unified(
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size,
        openai_client=openai_client,
        anthropic_client=anthropic_client,
    )

    scores: Dict[str, int] = {}
    n_attempted = 0
    n_parsed = 0
    for key, text in zip(keys, responses_text):
        n_attempted += 1
        if text:
            score = parse_judge_score(text)
            if score is not None:
                scores[key] = score
                n_parsed += 1

    return scores, n_attempted, n_parsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async():
    parser = argparse.ArgumentParser(
        description="Score responses with judge LLM")
    parser.add_argument("--responses_dir", type=str, required=True,
                        help="Directory with response JSONL files")
    parser.add_argument("--roles_dir", type=str,
                        default="../data/roles/instructions",
                        help="Role instruction JSON directory")
    parser.add_argument("--traits_dir", type=str,
                        default="../data/traits/instructions",
                        help="Trait instruction JSON directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for score JSON files")
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini",
                        help="Judge model to use. Any OpenAI model (gpt-/o1-/o3-/o4- "
                             "prefix) or Anthropic model (claude- prefix); routed "
                             "automatically by assistant_axis.judge.provider_for_model. "
                             "Requires OPENAI_API_KEY or ANTHROPIC_API_KEY respectively.")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Max tokens for judge response (must fit "
                             "2-3 sentences of reasoning + the SCORE: line; "
                             "the old default of 10 dates from the pre-May-2026 "
                             "single-digit-output rubric and would silently "
                             "truncate every judge call under the current "
                             "reasoning-first prompts)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Concurrent batch size")
    parser.add_argument("--requests_per_second", type=int, default=100,
                        help="Rate limit")
    parser.add_argument("--roles", nargs="+",
                        help="Specific entity names to process")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview without making API calls")
    parser.add_argument("--combination_scores", type=str,
                        default="../data/combination_scores.json",
                        help="Path to combination_scores.json for incongruity filtering")
    parser.add_argument("--no_incongruity_filtering", action="store_true",
                        help="Score every combination, including prompt-variants that "
                             "combination_scores.json flagged as highly incongruous "
                             "(incongruity-score 3). (This incongruity-score is unrelated "
                             "to the 0-3 rubric score that this script produces.)")
    parser.add_argument("--entity_type", type=str, required=True,
                        choices=["role", "trait", "combination"],
                        help="How to resolve response-file stems. Required because 9 names "
                             "exist in both data/roles and data/traits (ascetic, contrarian, "
                             "cosmopolitan, generalist, pacifist, patient, perfectionist, "
                             "romantic, stoic). Use 'combination' for r_<role>_t_<trait> files.")
    args = parser.parse_args()

    try:
        provider = provider_for_model(args.judge_model)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    required_env_var = _API_KEY_ENV_VAR[provider]
    if not args.dry_run and not os.getenv(required_env_var):
        logger.error(f"{required_env_var} not found (required for judge_model={args.judge_model!r})")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    responses_dir = Path(args.responses_dir)
    roles_dir = Path(args.roles_dir)
    traits_dir = Path(args.traits_dir)

    # Build per-pair exclusions for combined entries (incongruity score == 3)
    # TODO: Unclear if this filtering is a net win. Many responses get rejected
    # by the judge anyway (score < 3), so the incongruity filter mainly reduces
    # sample size for only a marginal improvement in sample quality — the rate
    # of bad-but-accepted responses among incongruity-filtered pairs isn't much
    # higher than among unfiltered ones, and some filtered pairs are perfectly
    # usable. Experiment with --no_incongruity_filtering to compare.
    excluded_pairs: Dict[str, set] = {}
    if not args.no_incongruity_filtering:
        combo_scores_path = Path(args.combination_scores)
        if combo_scores_path.exists():
            combo_data = json.load(open(combo_scores_path))
            n_excluded = 0
            for entry in combo_data.get("scores", []):
                bad = {i for i, p in enumerate(entry.get("per_pair", []))
                       if p.get("score", 0) >= 3}
                if bad:
                    excluded_pairs[entry["output_name"]] = bad
                    n_excluded += len(bad)
            if excluded_pairs:
                logger.info(f"Excluding {n_excluded} incongruous instruction pairs "
                            f"across {len(excluded_pairs)} combinations")

    response_files = sorted(responses_dir.glob("*.jsonl"))
    logger.info(f"Found {len(response_files)} response files")

    if args.roles:
        response_files = [f for f in response_files if f.stem in args.roles]

    logger.info(f"Processing {len(response_files)} entities")

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("Dry run — no API calls")
        total_prompts = 0
        sample_shown = False

        for response_file in response_files:
            name = response_file.stem
            output_file = output_dir / f"{name}.json"

            existing_scores: Dict[str, int] = {}
            if output_file.exists():
                try:
                    with open(output_file, "r") as f:
                        existing_scores = json.load(f)
                except Exception:
                    pass

            eval_tpl = resolve_eval_prompt(name, roles_dir, traits_dir,
                                           entity_type=args.entity_type)
            if eval_tpl is None:
                logger.info(f"  {name}: skipped (no eval prompt)")
                continue

            responses = load_responses(response_file)
            if name in excluded_pairs:
                responses = [r for r in responses
                             if r["prompt_index"] not in excluded_pairs[name]]
            count = 0
            sample_prompt = None

            for resp in responses:
                key = (f"{resp['label']}_p{resp['prompt_index']}"
                       f"_q{resp['question_index']}")
                if key not in existing_scores:
                    count += 1
                    if sample_prompt is None:
                        asst = ""
                        for msg in resp["conversation"]:
                            if msg["role"] == "assistant":
                                asst = msg["content"]
                                break
                        sample_prompt = eval_tpl.format(
                            question=resp["question"], answer=asst)

            if count > 0:
                total_prompts += count
                logger.info(f"  {name}: {count} prompts")
                if not sample_shown and sample_prompt:
                    logger.info("\n" + "=" * 60)
                    logger.info("SAMPLE JUDGE PROMPT:")
                    logger.info("=" * 60)
                    logger.info(f"Model: {args.judge_model}")
                    logger.info("-" * 60)
                    logger.info(sample_prompt)
                    logger.info("=" * 60 + "\n")
                    sample_shown = True

        logger.info(f"\nTotal prompts: {total_prompts}")
        return

    # ------------------------------------------------------------------
    # Live scoring
    # ------------------------------------------------------------------
    # Only the client for the selected provider is constructed -- the other
    # stays None and is simply unused by call_judge_batch_unified.
    openai_client = openai.AsyncOpenAI() if provider == "openai" else None
    anthropic_client = anthropic.AsyncAnthropic() if provider == "anthropic" else None
    rate_limiter = RateLimiter(args.requests_per_second)

    successful = skipped = failed = 0
    # Aggregate parse-rate counters across all entities so the end-of-run
    # summary can warn loudly if the chosen judge is dropping >1% of calls.
    total_call_attempted = 0
    total_call_parsed = 0
    errors: List[str] = []

    for response_file in tqdm(response_files, desc="Scoring"):
        name = response_file.stem
        output_file = output_dir / f"{name}.json"

        existing_scores: Dict[str, int] = {}
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    existing_scores = json.load(f)
            except Exception:
                pass

        eval_tpl = resolve_eval_prompt(name, roles_dir, traits_dir,
                                       entity_type=args.entity_type)
        if eval_tpl is None:
            skipped += 1
            continue

        responses = load_responses(response_file)
        if name in excluded_pairs:
            before_n = len(responses)
            responses = [r for r in responses
                         if r["prompt_index"] not in excluded_pairs[name]]
            if before_n != len(responses):
                logger.info(f"{name}: excluded {before_n - len(responses)} "
                            f"responses (incongruous pairs {excluded_pairs[name]})")
        if not responses:
            errors.append(f"{name}: no responses")
            failed += 1
            continue

        all_scored = all(
            f"{r['label']}_p{r['prompt_index']}_q{r['question_index']}"
            in existing_scores
            for r in responses
        )
        if all_scored:
            logger.info(f"Skipping {name}: all {len(responses)} already scored")
            skipped += 1
            continue

        try:
            new_scores, n_attempted, n_parsed = await score_entity(
                name=name,
                responses=responses,
                eval_prompt_template=eval_tpl,
                openai_client=openai_client,
                anthropic_client=anthropic_client,
                rate_limiter=rate_limiter,
                judge_model=args.judge_model,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                existing_scores=existing_scores,
            )
            total_call_attempted += n_attempted
            total_call_parsed += n_parsed
            all_scores = {**existing_scores, **new_scores}

            for attempt in range(3):
                try:
                    with open(output_file, "w") as f:
                        json.dump(all_scores, f, indent=2)
                    break
                except OSError as io_err:
                    if output_file.exists():
                        output_file.unlink()
                    if attempt < 2:
                        logger.warning(f"{name}: save failed ({io_err}), retrying in 5s...")
                        await asyncio.sleep(5)
                    else:
                        raise

            logger.info(f"Saved {len(all_scores)} scores for {name} "
                        f"({len(new_scores)} new)")
            successful += 1

        except Exception as e:
            errors.append(f"{name}: {e}")
            failed += 1

    logger.info("\n" + "=" * 40)
    logger.info("SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped:    {skipped}")
    logger.info(f"Failed:     {failed}")
    if errors:
        logger.info("\nErrors:")
        for e in errors[:10]:
            logger.info(f"  - {e}")
        if len(errors) > 10:
            logger.info(f"  ... and {len(errors) - 10} more")

    # Loud warning if this run's per-call parse rate dropped below 99%.
    # Aggregated across every (entity, response) judge call we made today,
    # so the count is statistically meaningful even when individual entities
    # have only a handful of new responses.
    warn_if_low_parse_rate(
        label=f"pipeline/3_judge:{args.judge_model}",
        n_ok=total_call_parsed,
        n_total=total_call_attempted,
        logger_obj=logger,
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
