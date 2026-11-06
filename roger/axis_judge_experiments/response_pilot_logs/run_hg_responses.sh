#!/bin/zsh
# Haiku 5.5 and Gemini 3.1 Flash-Lite on the twelve response axes (Roger, 2026-11-04: the Pareto plot with the
# Luna + Haiku and trio options): r1_pn on the twelve-axis group's batches (N=20, seed 1), one axis per run in turn
# so each batch's prefix is read while cached (Haiku's cache lives five minutes), a coin per call for the direction.
# Haiku: the ninth (split 9, part 0) in real time.  Gemini: its round-robin third of that ninth (split 27, part 0;
# (q + rank) % 27 == 0 lies inside (q + rank) % 9 == 0), real time, implicit caching.  Usage: run_hg_responses.sh
# haiku|gemini.  Run from the repository root.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
case $1 in
  haiku)  M=claude-haiku-5-5; SPLIT=9; LBL=s9p0; CAP=4 ;;
  gemini) M=gemini-3.1-flash-lite; SPLIT=27; LBL=s27p0; CAP=2 ;;
esac
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
  label="@r1_pnn20_${LBL}g12_a${i}c"
  echo "=== $M $label $(date -u +%FT%TZ)"
  .venv/bin/python results_analysis/judge_tier_cost_eval.py --judge_model $M --rubric r1_pn --axes_per_call 12 \
    --axis_index $i --directions responses --entities_per_call 20 --shuffle_seed 1 --cell_label $label --max_tokens 600 \
    --axes_file scripts/tier_eval_axes_responses12.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data \
    --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split $SPLIT --response_part 0 \
    --transport realtime --budget_usd $CAP --max_in_flight 50 > "$L/$M$label.log" 2>&1
  echo "    exit $? $(grep -A1 '^Summary:' "$L/$M$label.log" | tail -1)"
done
echo "=== done $(date -u +%FT%TZ)"
