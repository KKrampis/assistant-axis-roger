"""Diagnose UNPARSEABLE failures by replaying failed batches.

Run against a steering experiment dir.  For each strength group where
some/all records have `judges.effect.combined = None` (and not skipped
due to coh), this:

1. Reconstructs the EFFECT_BIDIR_BATCH_RUBRIC prompt verbatim.
2. Calls each effect-judge model once.
3. Logs:
   - finish_reason (whether truncated vs natural stop)
   - response length (chars)
   - raw response text
   - whether parse_batch_scores_json succeeds
4. Saves a detailed report.

Usage:
    uv run python tools/diagnose_unparseable.py \\
        --experiment_dir /workspace/.../architect_ecocentric_v2 \\
        --max_failed_strengths 3 \\
        --output /tmp/unparseable_diag.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant_axis.atomic_io import read_jsonl_with_retry  # noqa: E402
from assistant_axis.judge import (  # noqa: E402
    parse_batch_scores_json,
    provider_for_model,
)
from assistant_axis.steering_judges import (  # noqa: E402
    DEFAULT_EFFECT_MODELS,
    build_effect_bidir_batch_prompt,
)
from steering.post_judge import load_experiment_specs  # noqa: E402


async def call_one_with_finish_reason(
    *, model: str, prompt: str, max_tokens: int,
    openai_client, anthropic_client,
) -> Dict[str, Any]:
    """Call one judge, return raw text + finish_reason."""
    provider = provider_for_model(model)
    if provider == "openai":
        resp = await openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=1,
        )
        choice = resp.choices[0] if resp.choices else None
        return {
            "model": model,
            "provider": "openai",
            "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
            "text": choice.message.content if choice else None,
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
            },
        }
    elif provider == "anthropic":
        resp = await anthropic_client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=1,
        )
        text = "".join(blk.text for blk in resp.content if hasattr(blk, "text"))
        return {
            "model": model,
            "provider": "anthropic",
            "finish_reason": resp.stop_reason,
            "text": text,
            "usage": {
                "prompt_tokens": resp.usage.input_tokens if resp.usage else None,
                "completion_tokens": resp.usage.output_tokens if resp.usage else None,
            },
        }
    else:
        raise ValueError(f"unknown provider for {model!r}")


async def diagnose(
    *, experiment_dir: Path, instructions_dir: Path,
    effect_models: List[str], max_failed_strengths: int,
    max_tokens_per_item: int,
) -> Dict[str, Any]:
    persona_spec, steering_spec = load_experiment_specs(
        experiment_dir, instructions_dir
    )

    # Load baselines for the prompt builder.
    baselines_path = experiment_dir / "baselines" / "records.jsonl"
    baselines = list(read_jsonl_with_retry(baselines_path))
    baseline_by_q = {int(b["question_idx"]): b["response"] for b in baselines}

    import openai
    import anthropic
    openai_client = openai.AsyncOpenAI()
    anthropic_client = anthropic.AsyncAnthropic()

    cells = sorted(d for d in experiment_dir.iterdir()
                   if d.is_dir() and d.name.startswith("s")
                   and d.name not in ("baselines",))
    results: List[Dict[str, Any]] = []
    n_failed_seen = 0
    for cell in cells:
        recs_path = cell / "records.jsonl"
        if not recs_path.exists():
            continue
        recs = list(read_jsonl_with_retry(recs_path))
        # group by strength
        by_s: Dict[float, List[Dict[str, Any]]] = {}
        for r in recs:
            by_s.setdefault(r["strength"], []).append(r)
        for s in sorted(by_s):
            rs = by_s[s]
            # Find strengths where AT LEAST ONE judge model failed on >=
            # half the records (batch-level fail).
            fail_by_model: Dict[str, int] = {m: 0 for m in effect_models}
            skip = 0
            for r in rs:
                eff = (r.get("judges") or {}).get("effect") or {}
                if eff.get("skipped_due_to_strength_mean_coh"):
                    skip += 1
                    continue
                scores = (eff.get("bidirectional") or {}).get("scores") or {}
                for m in effect_models:
                    sc = scores.get(m, {})
                    if isinstance(sc, dict) and sc.get("score") is None:
                        fail_by_model[m] += 1
            if skip == len(rs):
                continue
            if not any(c > len(rs) // 2 for c in fail_by_model.values()):
                continue
            # Pick this strength as a diagnostic candidate.
            n_failed_seen += 1
            if n_failed_seen > max_failed_strengths:
                break
            sign = int(rs[0]["sign"])
            items = [
                {"id": r["question_idx"], "question": r["question"],
                 "baseline_response": baseline_by_q.get(int(r["question_idx"]), ""),
                 "steered_response": r["response"]}
                for r in rs
            ]
            prompt = build_effect_bidir_batch_prompt(
                persona=persona_spec, steering=steering_spec,
                sign=sign, strength=float(s), items=items,
            )
            max_tokens = max_tokens_per_item * len(items) + 200
            print(f"  retrying {cell.name} s={s} (orig fail counts: {fail_by_model})...",
                  file=sys.stderr)
            per_model_runs: List[Dict[str, Any]] = []
            for m in effect_models:
                try:
                    run = await call_one_with_finish_reason(
                        model=m, prompt=prompt, max_tokens=max_tokens,
                        openai_client=openai_client,
                        anthropic_client=anthropic_client,
                    )
                except Exception as e:
                    run = {"model": m, "error": str(e)}
                if "text" in run and run["text"]:
                    parsed = parse_batch_scores_json(
                        run["text"],
                        expected_ids=[r["question_idx"] for r in rs],
                        score_range=(-3, 3),
                    )
                    run["parsed_ok"] = parsed is not None
                    run["parsed_n_items"] = len(parsed) if parsed else 0
                    run["text_len"] = len(run["text"])
                    # Save first/last 400 chars so we can see where things go off
                    if not run["parsed_ok"] or len(run["text"]) > 800:
                        run["text_head"] = run["text"][:400]
                        run["text_tail"] = run["text"][-400:]
                    else:
                        run["text_head"] = run["text"]
                per_model_runs.append(run)
            results.append({
                "cell": cell.name, "sign": sign, "strength": s,
                "n_records": len(rs),
                "orig_fail_counts": fail_by_model,
                "max_tokens_requested": max_tokens,
                "retries": per_model_runs,
            })
        if n_failed_seen > max_failed_strengths:
            break
    return {"experiment_dir": str(experiment_dir), "n_diagnostics": len(results),
            "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", type=Path, required=True)
    ap.add_argument("--instructions_dir", type=Path, default=Path("data"))
    ap.add_argument("--max_failed_strengths", type=int, default=3)
    ap.add_argument("--max_tokens_per_item", type=int, default=200)
    ap.add_argument("--effect_models", default=",".join(DEFAULT_EFFECT_MODELS))
    ap.add_argument("--output", type=Path, default=Path("/tmp/unparseable_diag.json"))
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models = [m.strip() for m in args.effect_models.split(",") if m.strip()]
    result = asyncio.run(diagnose(
        experiment_dir=args.experiment_dir,
        instructions_dir=args.instructions_dir,
        effect_models=models,
        max_failed_strengths=args.max_failed_strengths,
        max_tokens_per_item=args.max_tokens_per_item,
    ))
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}", file=sys.stderr)

    # Print a short summary
    for r in result["results"]:
        print(f"\n=== {r['cell']} s={r['strength']} ===")
        for run in r["retries"]:
            m = run["model"]
            if "error" in run:
                print(f"  {m}: ERROR {run['error']}")
                continue
            ok = run.get("parsed_ok")
            n_items = run.get("parsed_n_items", 0)
            fr = run.get("finish_reason")
            tlen = run.get("text_len", 0)
            ct = (run.get("usage") or {}).get("completion_tokens")
            print(f"  {m}: parsed_ok={ok} n={n_items} finish={fr} text_len={tlen} completion_tokens={ct}")


if __name__ == "__main__":
    main()
