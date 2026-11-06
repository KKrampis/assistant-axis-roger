#!/bin/zsh
# The response-judging pilot (reports/judge_pareto/response_judging_plan.md, approved 2026-11-03): GPT-5.6 Luna by
# Flex on the six axes of scripts/tier_eval_axes_responses6.tsv.  The arms on a ninth of each entity's kept
# responses (split 9, part 0), the base design on the third it lies in (split 3, part 0), and two axes in full.
# Each cell is one judge_tier_cost_eval.py run; a rerun resumes.  Run from the repository root.
set -u
E=roger/axis_judge_experiments
L=$E/response_pilot_logs
R="runpod_workspace/qwen/qwen-3-32b Roger 8slot"
C=/tmp/claude-501/corpus_2026-05-24/data
run() {   # rubric N max_tokens split part budget axes_file [extra...]
  local rubric=$1 n=$2 cap=$3 split=$4 part=$5 budget=$6 axes=$7; shift 7
  local label="@${rubric}n${n}_s${split}p${part}"
  [[ "$*" == *response_chars* ]] && label="@${rubric}n${n}c_s${split}p${part}"
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py --judge_model gpt-5.6-luna --rubric $rubric \
    --entities_per_call $n --shuffle_seed 1 --cell_label $label --max_tokens $cap --axes_file $axes \
    --instructions_dir $C --responses_dir "$R" --response_split $split --response_part $part \
    --budget_usd $budget --max_in_flight 50 "$@" > $L/$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
}
A6=scripts/tier_eval_axes_responses6.tsv
A2=scripts/tier_eval_axes_responses2.tsv
run r1_h  20 1500 9 0 20 $A6                          # the design: question once, N=20, a sentence of reasoning
run r1_q  20 1500 9 0 20 $A6                          # Q: the question beside each answer
run r1_h  10  800 9 0 20 $A6                          # N=10
run r1_h   5  500 9 0 25 $A6                          # N=5
run r1_hn 20  300 9 0 20 $A6                          # R: the score alone
run r1_h  20 1500 9 0 20 $A6 --response_chars 1100    # C: answers cut at about 256 tokens
run r1_h  20 1500 3 0 50 $A6                          # the design on the whole third (the fraction curve)
echo "=== full coverage, two axes"
.venv/bin/python results_analysis/judge_tier_cost_eval.py --judge_model gpt-5.6-luna --rubric r1_h \
  --entities_per_call 20 --shuffle_seed 1 --cell_label @r1_hn20_all --max_tokens 1500 --axes_file $A2 \
  --instructions_dir $C --responses_dir "$R" --budget_usd 60 --max_in_flight 50 > $L/@r1_hn20_all.log 2>&1
echo "    exit $? $(grep -A1 '^Summary:' $L/@r1_hn20_all.log | tail -1)"
echo "=== done $(date -u +%FT%TZ)"
