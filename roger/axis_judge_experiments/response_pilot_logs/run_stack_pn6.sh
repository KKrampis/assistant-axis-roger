#!/bin/zsh
# The stacking arm run_stack_shuffled.sh skipped (Roger, 2026-11-04: "One stacking direction seems to be missing?"):
# plain "1, SCORES: 18, 42, ..." at k=6, on the same ninth (cut by question, as every stacked cell: the golden
# default would judge another ninth), seed, transport and settings.  First the one-axis warm-up the original driver
# ran, under its own label, so that every batch's prefix is cached and the k=6 run's measured cost is comparable
# with its siblings'.  Run from the root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
C=(--judge_model gpt-5.6-luna --axes_per_call 12 --entities_per_call 20 --shuffle_seed 1
   --axes_file scripts/tier_eval_axes_responses12.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0
   --response_design question --budget_usd 4 --max_in_flight 50)
run() {
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py "${C[@]}" --cell_label $label "$@" > $L/$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
}
run @r1_pnn20_s9p0g12_a0cwarm --rubric r1_pn --axis_index 0 --directions responses --max_tokens 300
run @rk_pnn20_s9p0g12_k6 --rubric rk_pn --stack 6 --max_tokens $(( 200 + 120 * 6 ))
echo "=== done $(date -u +%FT%TZ)"
