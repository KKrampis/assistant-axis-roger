#!/bin/zsh
# The cached prefix and k axes stacked (Roger, 2026-11-04: "in theory the two should even stack ... avoid putting
# axes with high cosine angles with each other in the same batch"): rk_pn on the batches of the r1_pn runs
# (--axes_per_call 6, seed 1, N=20, a ninth), the members judged together given by --axis_index.  Groups from the
# axis-direction cosines (slot 6, layer 25): pairs with no |cosine| above 0.14 (0,2 1,5 3,4); the control, the
# pairs with the highest (2,3 0.46, 0,5 -0.41, 1,4 0.09); triples (0,1,2 3,4,5, the best possible, 0.44); all six.
# The first run writes the prefix cache, the rest read it.  Run from the repository root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
for idx in 0,2 1,5 3,4 2,3 0,5 1,4 0,1,2 3,4,5 0,1,2,3,4,5; do
  label="@rk_pnn20_s9p0_a${idx//,/}"
  k=$(( ${#idx} / 2 + 1 ))
  echo "=== $label (k=$k) $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py --judge_model gpt-5.6-luna --rubric rk_pn --axes_per_call 6 \
    --axis_index $idx --entities_per_call 20 --shuffle_seed 1 --cell_label $label --max_tokens $(( 200 + 120 * k )) \
    --axes_file scripts/tier_eval_axes_responses6.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data \
    --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0 \
    --budget_usd 4 --max_in_flight 50 > $L/$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
done
echo "=== done $(date -u +%FT%TZ)"
