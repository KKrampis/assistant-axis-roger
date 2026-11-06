#!/bin/zsh
# The stacking test on twelve axes (Roger, 2026-11-04: axes "too similar" by the rho between their scores by the
# judge itself, wanting it near zero): scripts/tier_eval_axes_responses12.tsv, one group of twelve sharing the
# cached prefix (--axes_per_call 12, seed 1, N=20, a ninth).  Each axis alone (r1_pn, the baseline) and in two
# pairings (rk_pn): low, every |rho| of GPT-5.6 Luna's instruction scores at 0.08 or below; high, every |rho| at
# 0.44 or above.  Interleaved so the prefix is read every few minutes.  Run from the repository root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
LOW=(5,10 4,8 0,6 1,9 2,7 3,11); HIGH=(3,5 0,4 1,8 7,9 2,11 6,10)
run() {   # rubric index-list label max_tokens
  echo "=== $3 $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py --judge_model gpt-5.6-luna --rubric $1 --axes_per_call 12 \
    --axis_index $2 --entities_per_call 20 --shuffle_seed 1 --cell_label $3 --max_tokens $4 \
    --axes_file scripts/tier_eval_axes_responses12.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data \
    --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0 \
    --budget_usd 4 --max_in_flight 50 > $L/$3.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/$3.log | tail -1)"
}
for j in 1 2 3 4 5 6; do
  for i in $(( 2 * j - 2 )) $(( 2 * j - 1 )); do run r1_pn $i "@r1_pnn20_s9p0g12_a$i" 300; done
  run rk_pn ${LOW[$j]} "@rk_pnn20_s9p0g12_a${LOW[$j]//,/-}" 440
  run rk_pn ${HIGH[$j]} "@rk_pnn20_s9p0g12_a${HIGH[$j]//,/-}" 440
done
echo "=== done $(date -u +%FT%TZ)"
