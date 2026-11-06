#!/bin/zsh
# Haiku 5.5, the next step of the decision tree (Roger, 2026-11-05: "go ahead and get data for the next step"):
# plain stacking (rk_pn) at k = 2, 4 and 6 on the 61 axes, 20 answers a call (Haiku's 20 against 40 a tie, -0.0002;
# 20 kept, as Luna and Gemini), through Message Batches, the ninth cut by question, seed 1, as Luna's run_stack61.sh.
# Caps 300 + 150 k (Haiku writes a little more than Luna: 240 of 86,376 one-axis replies cut at 300).  About $18.
set -u
E=roger/axis_judge_experiments; L=$E/response_pilot_logs
C=(--judge_model claude-haiku-5-5 --transport batch --rubric rk_pn --axes_per_call 61 --entities_per_call 20 --shuffle_seed 1
   --axes_file scripts/tier_eval_axes.tsv --instructions_dir /tmp/claude-501/corpus_2026-05-24/data
   --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot" --response_split 9 --response_part 0
   --response_design question --batch_poll_seconds 60)
for k in 2 4 6; do
  ( label=@rk_pnn20_s9p0g61_k$k
    .venv/bin/python results_analysis/judge_tier_cost_eval.py "${C[@]}" --stack $k --max_tokens $(( 300 + 150 * k )) \
        --cell_label $label > "$L/claude-haiku-5-5$label.log" 2>&1
    echo "k=$k: exit $? $(grep -A1 '^Summary:' "$L/claude-haiku-5-5$label.log" | tail -1)" ) &
done
wait
echo "=== done $(date -u +%FT%TZ)"
