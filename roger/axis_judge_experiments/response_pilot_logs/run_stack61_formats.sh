#!/bin/zsh
# The other two stacking formats on the 61 axes (Roger, 2026-11-05: "the full one", about $60): labelled
# "1. SCORES: A=18, B=42" (rk_pl) and axis by axis "A: 1=18, 2=32" (rk_pa) at k = 2, 4 and 6, as the plain runs of
# run_stack61.sh: GPT-5.6 Luna by Flex, the ninth cut by question, N=20, seed 1, all 61 axes of
# scripts/tier_eval_axes.tsv as one group sharing the cached prefix, each batch's axes regrouped at random.
# Three streams; the first calls of each write prefixes the cache no longer holds.  Run from the root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
C=(--judge_model gpt-5.6-luna --transport flex --axes_per_call 61 --entities_per_call 20 --shuffle_seed 1
   --axes_file scripts/tier_eval_axes.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0
   --response_design question --max_in_flight 50)
run() {   # format, k, budget
  local label=@$1n20_s9p0g61_k$2
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py "${C[@]}" --cell_label $label --rubric $1 --stack $2 \
      --max_tokens $(( 200 + 120 * $2 )) --budget_usd $3 > $L/$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
}
( run rk_pl 2 25; run rk_pa 6 16 ) &
( run rk_pa 2 23; run rk_pl 6 16 ) &
( run rk_pl 4 18; run rk_pa 4 18 ) &
wait
echo "=== done $(date -u +%FT%TZ)"
