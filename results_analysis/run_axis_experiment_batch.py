#!/usr/bin/env python3
"""Driver for running axis_judge_correlation across a list of pairs.

Runs up to `--concurrency` pair invocations in parallel (as subprocesses),
writes each to `<output_root>/{pos}_vs_{neg}/{provider}/`, and records a
summary CSV+JSON at the root when all are done.

Usage:
    uv run python results_analysis/run_axis_experiment_batch.py \\
      --pair_list roger/axis_judge_experiments/pair_list.json \\
      --provider openai --judge_model gpt-4.1-mini \\
      --output_root roger/axis_judge_experiments \\
      --concurrency 3

Skips pairs whose `<output_root>/{pos}_vs_{neg}/{provider}/correlations.json`
already exists (lets you resume / add provider runs incrementally).
"""

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

from assistant_axis.pair_list_cohort import pair_type_of
from results_analysis.canonical_angles.whitening import DEFAULT_SOFT_K


async def run_one(pair, provider, judge_model, output_root, data_dir, instructions_dir,
                  layer, whiten_K, max_tokens, temperature, rps, batch_size, save_every,
                  score_modes, subdir_name, scores_dir, responses_dir,
                  response_target_batch_size, refill_gaps,
                  question_subsample_modulo=None, questions_file=None,
                  budget_usd=None):
    pos, neg = pair['pos'], pair['neg']
    pair_type = pair_type_of(pair)        # "traits" (default) or "roles"
    out_dir = output_root / f'{pos}_vs_{neg}' / subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_path = out_dir / 'correlations.json'
    if corr_path.exists() and not refill_gaps:
        return (pos, neg, 'SKIP (already done)', 0.0)
    if corr_path.exists() and refill_gaps:
        # If there's a gaps.json showing all modes empty, this run already
        # completed cleanly; skip even in refill mode.
        gaps_path = out_dir / 'gaps.json'
        if gaps_path.exists():
            try:
                gaps = json.loads(gaps_path.read_text())
                # An "empty gap" in any mode is either [] or {}.
                def _empty(v): return v == [] or v == {} or v is None
                if gaps and all(_empty(v) for v in gaps.values()):
                    return (pos, neg, 'SKIP (refill: no gaps)', 0.0)
            except Exception:
                pass  # fall through and re-run; gaps.json malformed

    cmd = [
        'uv', 'run', 'python', 'results_analysis/axis_judge_correlation.py',
        '--pair', pos, neg, '--pair_type', pair_type,
        '--data_dir', data_dir,
        '--instructions_dir', instructions_dir,
        '--layer', str(layer), '--whiten_K', str(whiten_K),
        '--provider', 'anthropic' if provider == 'sonnet' else 'openai',
        '--judge_model', judge_model,
        *score_modes,
        '--batch_size', str(batch_size), '--save_every', str(save_every),
        '--rps', str(rps), '--max_tokens', str(max_tokens), '--temperature', str(temperature),
        '--output_dir', str(out_dir),
    ]
    if scores_dir:
        cmd += ['--scores_dir', scores_dir]
    if responses_dir:
        cmd += ['--responses_dir', responses_dir]
    if response_target_batch_size is not None:
        cmd += ['--response_target_batch_size', str(response_target_batch_size)]
    if question_subsample_modulo is not None and int(question_subsample_modulo) > 0:
        cmd += ['--question_subsample_modulo', str(question_subsample_modulo)]
        if questions_file:
            cmd += ['--questions_file', str(questions_file)]
    if budget_usd is not None:
        cmd += ['--budget_usd', str(budget_usd)]
    log_path = out_dir / 'run.log'
    t0 = time.time()
    with open(log_path, 'wb') as logf:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=logf, stderr=asyncio.subprocess.STDOUT
        )
        rc = await proc.wait()
    dt = time.time() - t0
    status = f'OK (rc={rc})' if rc == 0 else f'FAIL (rc={rc}); see {log_path}'
    return (pos, neg, status, dt)


async def run_all(args):
    pairs = json.load(open(args.pair_list))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    score_modes = []
    if args.score_descriptions: score_modes.append('--score_descriptions')
    if args.score_instructions: score_modes.append('--score_instructions')
    if args.score_responses:    score_modes.append('--score_responses')
    if not score_modes:
        score_modes = ['--score_descriptions', '--score_instructions']

    sem = asyncio.Semaphore(args.concurrency)

    subdir_name = args.subdir or args.provider
    async def bounded(p):
        async with sem:
            return await run_one(
                p, args.provider, args.judge_model, output_root,
                args.data_dir, args.instructions_dir,
                args.layer, args.whiten_K, args.max_tokens, args.temperature,
                args.rps, args.batch_size, args.save_every, score_modes,
                subdir_name, args.scores_dir, args.responses_dir,
                args.response_target_batch_size, args.refill_gaps,
                question_subsample_modulo=args.question_subsample_modulo,
                questions_file=args.questions_file,
                budget_usd=args.budget_usd,
            )

    print(f'Launching {len(pairs)} pair runs (concurrency={args.concurrency}, provider={args.provider})')
    tasks = [asyncio.create_task(bounded(p)) for p in pairs]
    for fut in asyncio.as_completed(tasks):
        pos, neg, status, dt = await fut
        print(f'  [{dt:5.1f}s] {pos:16s} vs {neg:16s}  {status}')


def collect_summary(output_root: Path, providers):
    """Scan output_root/{pos_vs_neg}/{provider}/correlations.json files and build
    a long-format summary: (pos, neg, provider, mode, slot, metric, rho, p, n).
    """
    records = []
    for pair_dir in sorted(output_root.iterdir()):
        if not pair_dir.is_dir() or '_vs_' not in pair_dir.name:
            continue
        pos, neg = pair_dir.name.split('_vs_', 1)
        for prov in providers:
            cp = pair_dir / prov / 'correlations.json'
            if not cp.exists():
                continue
            try:
                corr = json.loads(cp.read_text())
            except json.JSONDecodeError:
                print(f'  [warn] {cp}: malformed JSON, skipping summary entry')
                continue
            # Defensive schema-shape checks.  Pre-May-2026 correlations.json
            # files (or partially-written ones from failed runs) may have
            # a non-dict at any level; skip with a warning rather than
            # crash the whole batch summary.  Expected schema:
            # corr = {mode: {slot: {metric: {rho, p, n}}}}.
            if not isinstance(corr, dict):
                print(f'  [warn] {cp}: top-level not a dict, skipping')
                continue
            for mode, slots in corr.items():
                if not isinstance(slots, dict):
                    print(f'  [warn] {cp}: corr[{mode!r}] not a dict, skipping')
                    continue
                for slot, metrics in slots.items():
                    if not isinstance(metrics, dict):
                        print(f'  [warn] {cp}: corr[{mode!r}][{slot!r}] '
                              f'not a dict (got {type(metrics).__name__}), '
                              f'skipping')
                        continue
                    for metric in ('raw', 'whitened'):
                        entry = metrics.get(metric)
                        if not entry or not isinstance(entry, dict):
                            continue
                        records.append({
                            'pos': pos, 'neg': neg, 'provider': prov,
                            'mode': mode, 'slot': int(slot), 'metric': metric,
                            'rho': entry.get('rho'), 'p': entry.get('p'), 'n': entry.get('n'),
                        })
    return records


def write_summary(output_root: Path, records):
    out_json = output_root / 'summary.json'
    out_csv = output_root / 'summary.csv'
    out_json.write_text(json.dumps(records, indent=2))
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pos','neg','provider','mode','slot','metric','rho','p','n'])
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f'\nWrote {len(records)} records to {out_json} and {out_csv}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pair_list', required=True)
    p.add_argument('--provider', required=True, choices=['openai', 'gpt', 'sonnet', 'anthropic'])
    p.add_argument('--judge_model', required=True)
    p.add_argument('--output_root', required=True)
    p.add_argument('--data_dir', default='runpod_workspace/qwen/qwen-3-32b Roger 8slot')
    p.add_argument('--instructions_dir', default='data')
    p.add_argument('--layer', type=int, default=25)  # Qwen-3-32B; tuned via rho_by_layer.py.
    p.add_argument('--whiten_K', type=int, default=DEFAULT_SOFT_K)
    # ^^ Project default for soft-K whitening (Apr 2026); was 3 historically,
    # 128 in the legacy K-sweep era.  See whitening.DEFAULT_SOFT_K.
    p.add_argument('--max_tokens', type=int, default=1024)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--rps', type=float, default=10.0)
    p.add_argument('--batch_size', type=int, default=20)
    p.add_argument('--save_every', type=int, default=40)
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--score_descriptions', action='store_true')
    p.add_argument('--score_instructions', action='store_true')
    p.add_argument('--score_responses', action='store_true')
    p.add_argument('--scores_dir', type=str, default=None,
                   help='Directory of per-entity score files (for --score_responses)')
    p.add_argument('--responses_dir', type=str, default=None,
                   help='Directory of per-entity response jsonl files (for --score_responses)')
    p.add_argument('--response_target_batch_size', type=int, default=None)
    p.add_argument('--question_subsample_modulo', type=int, default=None,
                   help='Pass-through to axis_judge_correlation.py: response-mode '
                        'sub-sampling (q_idx %% N == 0).')
    p.add_argument('--questions_file', type=str, default=None,
                   help='Pass-through to axis_judge_correlation.py: canonical '
                        'questions list for --question_subsample_modulo.')
    p.add_argument('--subdir', type=str, default=None,
                   help='Override the per-axis subdir name (default = provider). '
                        'Use e.g. "gpt_responses_traits" to keep response-mode outputs separate.')
    p.add_argument('--refill_gaps', action='store_true',
                   help='Re-run pairs whose correlations.json already exists, so the '
                        'inner resume logic can fill any None/missing entries in their '
                        'caches. Pairs whose gaps.json shows all modes empty are still '
                        'skipped.')
    p.add_argument('--budget_usd', type=float, default=None,
                   help='Per-axis budget cap in USD passed through to '
                        'axis_judge_correlation.py.  Each pair gets an '
                        'independent cap; total worst-case = budget_usd × '
                        'n_pairs.  Default: None (no cap).')
    p.add_argument('--only_summary', action='store_true', help='Skip runs, just (re)build the summary')
    args = p.parse_args()
    # Normalize provider name -> subdir name
    args.provider = {'openai': 'gpt', 'gpt': 'gpt', 'anthropic': 'sonnet', 'sonnet': 'sonnet'}[args.provider]

    if not args.only_summary:
        asyncio.run(run_all(args))

    records = collect_summary(Path(args.output_root), providers=('gpt', 'sonnet'))
    write_summary(Path(args.output_root), records)


if __name__ == '__main__':
    main()
