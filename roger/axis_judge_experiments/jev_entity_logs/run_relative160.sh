#!/bin/zsh
# Relative Noul, 160 entities a request (2026-11-04, the best arm on the development axes): confirmed on the 31
# held-out axes in the axes' own order (descriptions seed 1; instructions one pass), then with balanced directions
# on all 61 axes (descriptions two passes, instructions one pass, Jev's column).  About $2.50.  From the root.
set -u
L=roger/axis_judge_experiments/jev_entity_logs
A=(--instructions_dir /tmp/claude-501/corpus_2026-05-24/data --question noul --wording relative --entities_per_call 160
   --budget_usd 1.5 --max_in_flight 16)
run() {
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python scripts/judge_jev_entities.py "${A[@]}" --cell_label $label "$@" > $L/$label.$2.log 2>&1
  echo "    exit $? $(grep -h 'usage\]' $L/$label.$2.log | tail -1 | cut -c34-)"
}
H=(--axes_file scripts/tier_eval_axes_holdout31.tsv)
run @jd_noul_relative_n160s1 --mode descriptions "${H[@]}" --shuffle_seed 1
for r in 0 1 2 3 4; do
  run @ji_noul_relative_n160_r$r --mode instructions "${H[@]}" --instruction_round $r --shuffle_seed $(( r + 1 ))
done
F=(--axes_file scripts/tier_eval_axes.tsv)
for p in 0 1; do
  run @jd_noul_relative_n160d$(( p + 1 )) --mode descriptions "${F[@]}" --shuffle_seed $(( p + 1 )) --directions descriptions --direction_pass $p
done
for r in 0 1 2 3 4; do
  run @ji_noul_relative_n160d$r --mode instructions "${F[@]}" --instruction_round $r --shuffle_seed $(( r + 1 )) --directions instructions
done
echo "=== done $(date -u +%FT%TZ)"
