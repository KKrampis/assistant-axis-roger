#!/bin/zsh
# Pole order on descriptions and instructions (Roger, 2026-11-04: "does order (and thus which corresponds to the
# high number) make any difference: are two repeats in the same order more similar to each other than two in
# opposite order?  do we get a better rho if good is a high number?"): GPT-5.6 Luna on the 61 axes.
# Descriptions (b10, N=20): the axis's own order (o) and swapped (w, --swap_poles), seeds 1 and 2, each twice.
# Instructions (i1_p0, N=20, round r = instruction r with seed r+1): the first pass (rounds 0-4) repeated in the
# axis's order, and swapped twice; the existing @i1_p0n20r0..r4 cells are its first copy.  Run from the root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
C=(--judge_model gpt-5.6-luna --entities_per_call 20 --max_tokens 1200 --axes_file scripts/tier_eval_axes.tsv
   --instructions_dir /tmp/claude-501/corpus_2026-05-24/data --max_in_flight 50)
run() {   # label, then the cell's own flags
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py "${C[@]}" --cell_label $label "$@" > $L/order_$label.log 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' $L/order_$label.log | tail -1)"
}
for s in 1 2; do
  for o in o w; do
    for rep in "" r; do
      flags=(--rubric b10 --shuffle_seed $s --budget_usd 2); [[ $o == w ]] && flags+=(--swap_poles)
      run "@b10n20${o}_s${s}${rep}" "${flags[@]}"
    done
  done
done
for r in 0 1 2 3 4; do
  run "@i1_p0n20o_r${r}r" --rubric i1_p0 --instruction_round $r --shuffle_seed $(( r + 1 )) --budget_usd 1.5
  for rep in "" r; do
    run "@i1_p0n20w_r${r}${rep}" --rubric i1_p0 --instruction_round $r --shuffle_seed $(( r + 1 )) --budget_usd 1.5 --swap_poles
  done
done
echo "=== done $(date -u +%FT%TZ)"
