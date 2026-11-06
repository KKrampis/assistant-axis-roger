#!/bin/zsh
# Jev on description judging, the first arms (2026-11-04, Roger: "experiment with the rubric, batching, and
# approaches for extracting scores from Jev"): the 30 development axes, seed 1 (the trio's groups of 20), one cell
# per arm; each about $0.2 at $0.042 per million input tokens.  Run from the repository root.
set -u
L=roger/axis_judge_experiments/jev_entity_logs
C=(--mode descriptions --axes_file scripts/tier_eval_axes_dev30.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --shuffle_seed 1 --budget_usd 1 --max_in_flight 24)
run() {   # label, then the arm's flags
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python scripts/judge_jev_entities.py "${C[@]}" --cell_label $label "$@" > $L/$label.log 2>&1
  echo "    exit $? $(grep -h 'usage\]' $L/$label.log | tail -1 | cut -c34-)"
}
run @jd_score5_base_n20s1 --question score --levels 5 --wording base --entities_per_call 20
run @jd_score10_base_n20s1 --question score --levels 10 --wording base --entities_per_call 20
run @jd_choice11_base_n20s1 --question choice --levels 11 --wording base --entities_per_call 20
run @jd_choice21_base_n20s1 --question choice --levels 21 --wording base --entities_per_call 20
run @jd_noul_base_n20s1 --question noul --wording base --entities_per_call 20
run @jd_score5_rank_n20s1 --question score --levels 5 --wording rank --entities_per_call 20
run @jd_score5_base_n1s1 --question score --levels 5 --wording base --entities_per_call 1
run @jd_score5_base_n40s1 --question score --levels 5 --wording base --entities_per_call 40
echo "=== done $(date -u +%FT%TZ)"
