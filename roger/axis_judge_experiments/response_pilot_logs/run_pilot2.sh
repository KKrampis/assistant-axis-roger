#!/bin/zsh
# The pilot's second part (2026-11-04): the score-alone rubric r1_hn led the ninth arms (0.683 against the
# design's 0.662, 16% cheaper), so its own fraction curve (the third, then two axes in full) and the batch
# size above 20 (Roger: "Would N > 20 be better?").  Run from the repository root after run_pilot.sh.
set -u
E=roger/axis_judge_experiments
L=$E/response_pilot_logs
R="runpod_workspace/qwen/qwen-3-32b Roger 8slot"
C=/tmp/claude-501/corpus_2026-05-24/data
run() {   # label rubric N max_tokens budget axes_file [extra...]
  local label=$1 rubric=$2 n=$3 cap=$4 budget=$5 axes=$6; shift 6
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py --judge_model gpt-5.6-luna --rubric $rubric \
    --entities_per_call $n --shuffle_seed 1 --cell_label $label --max_tokens $cap --axes_file $axes \
    --instructions_dir $C --responses_dir "$R" --budget_usd $budget --max_in_flight 50 "$@" > $L/$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
}
A6=scripts/tier_eval_axes_responses6.tsv
A2=scripts/tier_eval_axes_responses2.tsv
run @r1_hnn40_s9p0 r1_hn 40 600 20 $A6 --response_split 9 --response_part 0     # N=40, the score alone
run @r1_hnn20_s3p0 r1_hn 20 300 30 $A6 --response_split 3 --response_part 0     # the third, for the fraction curve
run @r1_hnn20_all  r1_hn 20 300 60 $A2                                           # two axes in full
echo "=== done2 $(date -u +%FT%TZ)"
