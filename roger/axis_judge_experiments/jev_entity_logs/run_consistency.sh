#!/bin/zsh
# Jev rerun consistency and second passes (Roger, 2026-11-04: "How consistent is it if you rerun it?  Is Jev 2x any
# better?  How about if the two use opposite axis directions?"), relative Noul, 160 a request, all 61 axes:
# an exact rerun of the own-order seed-1 description pass, an own-order seed-2 pass, a swapped seed-1 pass (the
# balanced d1/d2 cells exist); and the own-order second instruction pass (rounds 5-9, seed r+1), for the 12-way
# plot's Jev 2x.  About $2.  Run from the repository root.
set -u
L=roger/axis_judge_experiments/jev_entity_logs
A=(--axes_file scripts/tier_eval_axes.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data --question noul
   --wording relative --entities_per_call 160 --budget_usd 1.5 --max_in_flight 16)
run() {
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python scripts/judge_jev_entities.py "${A[@]}" --cell_label $label "$@" > $L/$label.log 2>&1
  echo "    exit $? $(grep -h 'usage\]' $L/$label.log | tail -1 | cut -c34-)"
}
run @jd_noul_relative_n160s1r --mode descriptions --shuffle_seed 1
run @jd_noul_relative_n160s2 --mode descriptions --shuffle_seed 2
run @jd_noul_relative_n160s1w --mode descriptions --shuffle_seed 1 --directions all
for r in 5 6 7 8 9; do
  run @ji_noul_relative_n160_r$r --mode instructions --instruction_round $r --shuffle_seed $(( r + 1 ))
done
echo "=== done $(date -u +%FT%TZ)"
