#!/bin/zsh
# Jev Noul confirmed on the 31 held-out axes (2026-11-04): the arm chosen on the development axes (Noul, base
# wording, 20 a request), in the axes' own order as the trio's existing cells are; descriptions two passes
# (seeds 1 and 2), instructions one pass (rounds 0-4, seed r+1).  Run from the repository root.
set -u
L=roger/axis_judge_experiments/jev_entity_logs
C=(--axes_file scripts/tier_eval_axes_holdout31.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --question noul --wording base --entities_per_call 20 --budget_usd 1 --max_in_flight 24)
run() {
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python scripts/judge_jev_entities.py "${C[@]}" --cell_label $label "$@" > $L/$label.log 2>&1
  echo "    exit $? $(grep -h 'usage\]' $L/$label.log | tail -1 | cut -c34-)"
}
run @jd_noul_base_n20s1 --mode descriptions --shuffle_seed 1
run @jd_noul_base_n20s2 --mode descriptions --shuffle_seed 2
for r in 0 1 2 3 4; do
  run @ji_noul_base_n20_r$r --mode instructions --instruction_round $r --shuffle_seed $(( r + 1 ))
done
echo "=== done $(date -u +%FT%TZ)"
