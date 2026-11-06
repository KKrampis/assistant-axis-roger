#!/bin/zsh
# Jev with balanced pole directions on all 61 axes (Roger's standing rule, 2026-11-04: never judge in one fixed
# pole order): the chosen arm's flags as the first argument (default Noul, base wording, 20 a request).
# Descriptions: two passes, seeds 1 and 2, direction_pass 0 and 1 (judge_tier_cost_eval's coin per axis decides
# which is swapped), labels @jd_<arm>d1 and d2.  Instructions: one pass, rounds 0-4 with seed r+1, Jev's column
# of Roger's direction table, labels @ji_<arm>d0 to d4.  The labels follow the trio's balanced cells
# (<judge>@b10n20d1/d2, <judge>@i1_p0n20d0..d4).  About $1.75.  Run from the repository root.
set -u
L=roger/axis_judge_experiments/jev_entity_logs
ARM=${1:-noul_n20}
FLAGS=(${=${2:---question noul --wording base --entities_per_call 20}})
C=(--axes_file scripts/tier_eval_axes.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data --budget_usd 2
   --max_in_flight 24 "${FLAGS[@]}")
run() {
  local label=$1; shift
  echo "=== $label $(date -u +%FT%TZ)"
  .venv/bin/python scripts/judge_jev_entities.py "${C[@]}" --cell_label $label "$@" > $L/$label.log 2>&1
  echo "    exit $? $(grep -h 'usage\]' $L/$label.log | tail -1 | cut -c34-)"
}
for p in 0 1; do
  run @jd_${ARM}d$(( p + 1 )) --mode descriptions --shuffle_seed $(( p + 1 )) --directions descriptions --direction_pass $p
done
for r in 0 1 2 3 4; do
  run @ji_${ARM}d$r --mode instructions --instruction_round $r --shuffle_seed $(( r + 1 )) --directions instructions
done
echo "=== done $(date -u +%FT%TZ)"
