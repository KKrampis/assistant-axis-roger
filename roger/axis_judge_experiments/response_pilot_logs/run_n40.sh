#!/bin/zsh
# Batches of 40 answers against 20 (Roger, 2026-11-04: "main question is whether results go up or down ... include a
# range"): GPT-5.6 Luna one axis a call, a coin per call, the twelve axes (rho at N=20 from 0.32 to 0.91, spanning the
# 61's range), the same ninth cut by question and the same seed as the N=20 cells, 40 answers to a question a call.
# The first axis writes the new prefixes to the cache; the rest read them.  Run from the root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
C=(--judge_model gpt-5.6-luna --axes_per_call 12 --entities_per_call 40 --shuffle_seed 1
   --axes_file scripts/tier_eval_axes_responses12.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0
   --response_design question --max_in_flight 50 --rubric r1_pn --directions responses --max_tokens 600)
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
  label=@r1_pnn40_s9p0g12_a${i}c
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py "${C[@]}" --cell_label $label --axis_index $i \
      --budget_usd $([[ $i == 0 ]] && echo 4 || echo 1.5) > $L/$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
done
echo "=== done $(date -u +%FT%TZ)"
