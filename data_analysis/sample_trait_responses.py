#!/usr/bin/env python3
"""Sample pos/neg response pairs for a trait to eyeball how the model responds.

Sends each instruction (pos and neg) as a system prompt with a sample of
questions, and prints the responses side by side.

Usage:
    uv run python data_analysis/sample_trait_responses.py techno_hierophantic
    uv run python data_analysis/sample_trait_responses.py techno_hierophantic --n-questions 3
    uv run python data_analysis/sample_trait_responses.py techno_hierophantic --pairs 0 2 4
    uv run python data_analysis/sample_trait_responses.py techno_hierophantic --model claude-sonnet-4-6
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic

TRAITS_DIR = Path(__file__).resolve().parent.parent / "data" / "traits" / "instructions"


async def run(args):
    client = anthropic.AsyncAnthropic()

    trait_path = TRAITS_DIR / f"{args.trait}.json"
    if not trait_path.exists():
        print(f"Trait file not found: {trait_path}", file=sys.stderr)
        sys.exit(1)

    with open(trait_path, encoding="utf-8") as f:
        data = json.load(f)

    instructions = data.get("instruction", [])
    questions = data.get("questions", [])

    if not instructions:
        print("No instructions found in trait file", file=sys.stderr)
        sys.exit(1)
    if not questions:
        print("No questions found in trait file", file=sys.stderr)
        sys.exit(1)

    if args.pairs:
        pair_indices = args.pairs
    else:
        pair_indices = list(range(len(instructions)))

    rng = random.Random(args.seed)
    sample_qs = rng.sample(questions, min(args.n_questions, len(questions)))

    total = len(pair_indices) * len(sample_qs) * 2
    print(
        f"Trait: {data.get('positive_label', args.trait)}\n"
        f"Pairs: {pair_indices}, Questions: {len(sample_qs)}, "
        f"Total API calls: {total}\n"
        f"Model: {args.model}\n",
        file=sys.stderr,
    )

    for pi in pair_indices:
        pair = instructions[pi]
        pos_sys = pair["pos"]
        neg_sys = pair["neg"]

        print(f"{'='*80}")
        print(f"PAIR {pi}")
        print(f"{'='*80}")
        print(f"POS system: {pos_sys}")
        print(f"NEG system: {neg_sys}")
        print()

        for qi, question in enumerate(sample_qs):
            print(f"--- Question {qi+1}: {question} ---")
            print()

            pos_resp, neg_resp = await asyncio.gather(
                client.messages.create(
                    model=args.model,
                    max_tokens=args.max_tokens,
                    system=pos_sys,
                    messages=[{"role": "user", "content": question}],
                ),
                client.messages.create(
                    model=args.model,
                    max_tokens=args.max_tokens,
                    system=neg_sys,
                    messages=[{"role": "user", "content": question}],
                ),
            )

            pos_text = pos_resp.content[0].text
            neg_text = neg_resp.content[0].text

            print(f"[POS] {pos_text}")
            print()
            print(f"[NEG] {neg_text}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Sample pos/neg responses for a trait")
    parser.add_argument("trait", help="Trait file stem (e.g. techno_hierophantic)")
    parser.add_argument("--n-questions", type=int, default=3, help="Number of questions to sample (default: 3)")
    parser.add_argument("--pairs", type=int, nargs="+", help="Instruction pair indices to use (default: all)")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question sampling")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
