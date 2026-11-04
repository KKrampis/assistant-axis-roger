#!/bin/zsh
# Jev description arms, second round (2026-11-04): Noul led the first (0.701 alone on the 30 development axes, one
# pass, against Luna's one pass 0.684), so its second pass, its wording, its batch size, and the pairwise design.
# Waits for the first driver.  Run from the repository root.
set -u
L=roger/axis_judge_experiments/jev_entity_logs
until grep -q "^=== done" $L/driver_desc_dev.log 2>/dev/null; do sleep 20; done
C=(--mode descriptions --axes_file scripts/tier_eval_axes_dev30.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --budget_usd 1.5 --max_in_flight 24)
run() {
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python scripts/judge_jev_entities.py "${C[@]}" --cell_label $label "$@" > $L/$label.log 2>&1
  echo "    exit $? $(grep -h 'usage\]' $L/$label.log | tail -1 | cut -c34-)"
}
run @jd_noul_base_n20s2 --question noul --wording base --entities_per_call 20 --shuffle_seed 2
run @jd_noul_rank_n20s1 --question noul --wording rank --entities_per_call 20 --shuffle_seed 1
run @jd_noul_base_n10s1 --question noul --wording base --entities_per_call 10 --shuffle_seed 1
run @jd_noul_base_n40s1 --question noul --wording base --entities_per_call 40 --shuffle_seed 1
run @jd_pairs_ring3_n20s1 --question pairs --pair_ring 3 --wording base --entities_per_call 20 --shuffle_seed 1
run @jd_pairs_base_n20s1 --question pairs --wording base --entities_per_call 20 --shuffle_seed 1
run @jd_noul_base_n1s1 --question noul --wording base --entities_per_call 1 --shuffle_seed 1
echo "=== done $(date -u +%FT%TZ)"
