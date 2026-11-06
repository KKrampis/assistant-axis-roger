#!/bin/zsh
# The cached-prefix arm (Roger, 2026-11-04): r1_pn, the question and the answers first (cached), the axis after;
# one axis per run, the six runs in sequence on the same batches, so runs 1-5 read the prefix run 0 wrote.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
for i in 0 1 2 3 4 5; do
  label="@r1_pnn20_s9p0_a$i"
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py --judge_model gpt-5.6-luna --rubric r1_pn --axes_per_call 6 \
    --axis_index $i --entities_per_call 20 --shuffle_seed 1 --cell_label $label --max_tokens 300 \
    --axes_file scripts/tier_eval_axes_responses6.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data \
    --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0 \
    --budget_usd 4 --max_in_flight 50 > $L/$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
done
echo "=== done $(date -u +%FT%TZ)"
