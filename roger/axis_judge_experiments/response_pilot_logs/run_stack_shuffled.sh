#!/bin/zsh
# Stacking with leakage turned into noise (Roger, 2026-11-04): the twelve axes of scripts/tier_eval_axes_responses12.tsv
# as one group sharing the cached prefix (a ninth, N=20, seed 1, GPT-5.6 Luna by Flex), each batch's axes partitioned
# afresh into groups of k (--stack k, a random letter order, each axis's direction a coin per call), in three output
# formats: rk_pn "1, SCORES: 18, 42"; rk_pl "1. SCORES: A=18, B=42"; rk_pa axis by axis "A: 1=18, 2=32, ...".  First a
# one-axis run with coins on axis 0, which writes the prefix cache for the rest.  The baseline: the one-axis runs
# already on disk, own order (@r1_pnn20_s9p0g12_a<i>) and swapped (..._a<i>w, ..._a<i>wr).  Run from the root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
C=(--judge_model gpt-5.6-luna --axes_per_call 12 --entities_per_call 20 --shuffle_seed 1
   --axes_file scripts/tier_eval_axes_responses12.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0
   --budget_usd 4 --max_in_flight 50)
run() {   # label, then its own flags
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py "${C[@]}" --cell_label $label "$@" > $L/$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$label.log | tail -1)"
}
run @r1_pnn20_s9p0g12_a0c --rubric r1_pn --axis_index 0 --directions responses --max_tokens 300
for k in 2 4 6; do
  for fmt in rk_pa rk_pl rk_pn; do
    [[ $fmt == rk_pn && $k == 6 ]] && continue
    run "@${fmt}n20_s9p0g12_k$k" --rubric $fmt --stack $k --max_tokens $(( 200 + 120 * k ))
  done
done
echo "=== done $(date -u +%FT%TZ)"
