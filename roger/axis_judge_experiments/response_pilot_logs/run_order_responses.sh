#!/bin/zsh
# Pole order on responses (Roger, 2026-11-04: does the order, and so which pole is the high number, make a
# difference, and is good-high better for rho?): the twelve axes of the stacking test judged alone on the cached
# prefix with their poles swapped (--swap_poles, the scores mapped back), twice, on the batches of the
# twelve-axis group, while that group's runs keep the prefix cached.  Run from the repository root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
for rep in w wr; do
  for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
    label="@r1_pnn20_s9p0g12_a${i}${rep}"
    echo "=== $label $(date -u +%FT%TZ)"
    .venv/bin/python results_analysis/judge_tier_cost_eval.py --judge_model gpt-5.6-luna --rubric r1_pn --axes_per_call 12 \
      --axis_index $i --entities_per_call 20 --shuffle_seed 1 --cell_label $label --max_tokens 300 --swap_poles \
      --axes_file scripts/tier_eval_axes_responses12.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data \
      --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0 \
      --budget_usd 4 --max_in_flight 50 > $L/$label.log 2>&1
    echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
  done
done
echo "=== done $(date -u +%FT%TZ)"
