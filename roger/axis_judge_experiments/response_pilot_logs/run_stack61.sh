#!/bin/zsh
# Stacking on the 61 axes (Roger, 2026-11-04: "take the stacking up to 61 axes for a broader sample"; approved,
# about $38): the ninth of the May responses cut by question (as the twelve-axis cells, so that those pool with
# these), N=20, seed 1, GPT-5.6 Luna by Flex, all 61 axes of scripts/tier_eval_axes.tsv as one group sharing the
# cached prefix.  First Luna one axis a call, a coin per call, on the 49 axes not among the twelve (the first run
# writes the prefix cache); then plain stacking at k = 2, 4 and 6 over all 61, each batch's axes regrouped at
# random (a batch's last call holds the one axis left over).  Run from the root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
C=(--judge_model gpt-5.6-luna --axes_per_call 61 --entities_per_call 20 --shuffle_seed 1
   --axes_file scripts/tier_eval_axes.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0
   --response_design question --max_in_flight 50)
run() {
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py "${C[@]}" --cell_label $label "$@" > $L/$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
}
# Phase 1: one axis alone, which writes every batch's prefix to the cache.
run @r1_pnn20_s9p0g61_a1c --rubric r1_pn --axis_index 1 --directions responses --max_tokens 300 --budget_usd 3
# Phase 2: three streams reading the warm cache at once (about 3 hours instead of 6.5 in a row).
(
  for i in 2 3 4 5 6 8 9 10 11 12 13 15 16 17 18 19 20 21 22 24 25 27 29 30 31 32 33 34 36 37 38 39 40 41 43 44 46 47 49 50 51 54 55 56 57 58 59 60; do
    run @r1_pnn20_s9p0g61_a${i}c --rubric r1_pn --axis_index $i --directions responses --max_tokens 300 --budget_usd 2
  done
) &
( run @rk_pnn20_s9p0g61_k2 --rubric rk_pn --stack 2 --max_tokens 440 --budget_usd 18 ) &
( run @rk_pnn20_s9p0g61_k4 --rubric rk_pn --stack 4 --max_tokens 680 --budget_usd 11
  run @rk_pnn20_s9p0g61_k6 --rubric rk_pn --stack 6 --max_tokens 920 --budget_usd 9 ) &
wait
echo "=== done $(date -u +%FT%TZ)"
