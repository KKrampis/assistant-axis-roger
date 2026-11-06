#!/bin/zsh
# Jev on instruction judging (2026-11-04): the arms given as arguments, label:flags pairs, each a pass of five
# rounds (round r: instruction r with seed r+1, the trio's groups), on the 30 development axes; a round is about
# $0.2.  Usage: run_instr_dev.sh '@ji_score5_base_n20' '--question score --levels 5 --wording base' ...
set -u
L=roger/axis_judge_experiments/jev_entity_logs
C=(--mode instructions --axes_file scripts/tier_eval_axes_dev30.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --entities_per_call 20 --budget_usd 1 --max_in_flight 24)
while (( $# >= 2 )); do
  label=$1; flags=(${=2}); shift 2
  for r in 0 1 2 3 4; do
    echo "=== ${label}_r$r $(date -u +%FT%TZ)"
    .venv/bin/python scripts/judge_jev_entities.py "${C[@]}" --instruction_round $r --shuffle_seed $(( r + 1 )) \
      --cell_label ${label}_r$r "${flags[@]}" > $L/${label}_r$r.log 2>&1
    echo "    exit $? $(grep -h 'usage\]' $L/${label}_r$r.log | tail -1 | cut -c34-)"
  done
done
echo "=== done $(date -u +%FT%TZ)"
