#!/bin/zsh
# Haiku 5.5, 20 answers a call against 40, on all 61 axes through Message Batches (Roger, 2026-11-05: "all flex or
# batch as appropriate"; the batch path for --stack runs built that night, 590d5864): r1_pn, every axis its own call
# (--stack 1, a coin per axis and call), the ninth cut by question, seed 1, as Luna's cells.  Haiku's cache works in
# batches (one write per prefix, shared across concurrent batches; probe of 2026-11-05).  About $13 + $10.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
C=(--judge_model claude-haiku-5-5 --transport batch --rubric r1_pn --stack 1 --axes_per_call 61 --shuffle_seed 1
   --axes_file scripts/tier_eval_axes.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0
   --response_design question --batch_poll_seconds 60)
( .venv/bin/python results_analysis/judge_tier_cost_eval.py "${C[@]}" --entities_per_call 20 --max_tokens 300 \
      --cell_label @r1_pnn20_s9p0g61_st1 > $L/claude-haiku-5-5@r1_pnn20_s9p0g61_st1.log 2>&1
  echo "20: exit $? $(grep -A1 '^Summary:' $L/claude-haiku-5-5@r1_pnn20_s9p0g61_st1.log | tail -1)" ) &
( .venv/bin/python results_analysis/judge_tier_cost_eval.py "${C[@]}" --entities_per_call 40 --max_tokens 600 \
      --cell_label @r1_pnn40_s9p0g61_st1 > $L/claude-haiku-5-5@r1_pnn40_s9p0g61_st1.log 2>&1
  echo "40: exit $? $(grep -A1 '^Summary:' $L/claude-haiku-5-5@r1_pnn40_s9p0g61_st1.log | tail -1)" ) &
wait
echo "=== done $(date -u +%FT%TZ)"
