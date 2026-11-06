#!/bin/zsh
# 40 answers a call on the 49 axes outside the twelve (Roger, 2026-11-05: "Sorry, I meant do it at flex"; about $14):
# GPT-5.6 Luna by Flex, one axis a call, a coin per call, all 61 axes of scripts/tier_eval_axes.tsv as one group sharing
# the cached prefix, the ninth cut by question and seed 1 as the 20-answer cells (@r1_pnn20_s9p0g61_a<i>c).  The first
# axis writes the prefixes to the cache; then three streams read it.  Run from the root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
C=(--judge_model gpt-5.6-luna --transport flex --axes_per_call 61 --entities_per_call 40 --shuffle_seed 1
   --axes_file scripts/tier_eval_axes.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0
   --response_design question --max_in_flight 50 --rubric r1_pn --directions responses --max_tokens 600)
run() {
  local i=$1 b=$2 label=@r1_pnn40_s9p0g61_a$1c
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py "${C[@]}" --cell_label $label --axis_index $i --budget_usd $b > $L/$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
}
run 1 4
( for i in 2 3 4 5 6 8 9 10 11 12 13 15 16 17 18 19; do run $i 1.5; done ) &
( for i in 20 21 22 24 25 27 29 30 31 32 33 34 36 37 38 39; do run $i 1.5; done ) &
( for i in 40 41 43 44 46 47 49 50 51 54 55 56 57 58 59 60; do run $i 1.5; done ) &
wait
echo "=== done $(date -u +%FT%TZ)"
