#!/bin/zsh
# Regenerate every judge-tier figure from the recorded cells (no API calls), from the repository root.  Each
# figure's PNG records its command, its git commit and its inputs.  Each list of judges is explicit: with no list a
# figure takes every cell on disk, and the grid made that a 233-megapixel heatmap.
# Usage: zsh scripts/judge_tier_make_figures.sh
set -e
E=roger/axis_judge_experiments
D=runpod_workspace/qwen/qwen-3-32b\ Roger\ 8slot
C=$E/_tier_eval_scratch/robustness_cache
PY=(uv run python)
rob() { $PY results_analysis/judge_tier_robustness.py --experiments_dir $E --data_dir "$D" --output_dir $E --cache_dir $C "$@"; }
pool() { $PY results_analysis/judge_tier_cost_plot_pooled.py --experiments_dir $E "$@"; }

OLD61=(gpt-4.1-mini claude-haiku-5-5 gpt-5.6-luna@none gpt-6-luna claude-haiku-4-5@t0 claude-sonnet-5 gpt-5.6-terra@none gpt-5.4-mini)
NEW61=(gpt-4.1-mini@b10x2 claude-haiku-5-5@b10x2 gpt-5.6-luna@none@b10x2 gpt-6-luna@b10x2 claude-haiku-4-5@t0@b10x2 claude-sonnet-5@b10x2 gpt-5.6-terra@none@b10x2 gpt-5.4-mini@b10x2)
LINKS61=(); for m in $OLD61; do LINKS61+=("$m:$m@b10x2"); done
OLD20=($OLD61 claude-opus-5@nothink gpt-5.6-sol@none)
NEW20=($NEW61 claude-opus-5@nothink@b10x2 gpt-5.6-sol@none@b10x2)
LINKS20=(); for m in $OLD20; do LINKS20+=("$m:$m@b10x2"); done
GRID61=($NEW61 gpt-5.6-luna@none@b10n20x2 gpt-6-luna@b10n20x2 claude-haiku-5-5@b10n20x2 gpt-5.6-luna+claude-haiku-5-5@b10n20x2)
FIRST=(claude-opus-5 gpt-4.1-mini claude-haiku-4-5@t0 claude-sonnet-5 gpt-5.6-terra@none gpt-5.6-luna@none claude-haiku-5-5 gpt-5.4-mini gpt-6-luna)
FIRST20=(claude-opus-5 claude-opus-5@nothink gpt-4.1-mini claude-haiku-4-5@t0 claude-sonnet-5 gpt-5.6-sol@none claude-haiku-5-5 gpt-5.6-luna@none gpt-5.6-terra@none gpt-5.4-mini gpt-6-luna)
EARLY=(claude-haiku-4-5 gpt-5.6-luna gpt-5.6-terra)

# The robustness tables and figures: every judge before tuning, then the rubrics, then the grid
rob --axes_file scripts/tier_eval_axes.tsv    --stem judge_tier_robustness_61axes --include_models $FIRST
rob --axes_file scripts/tier_eval_axes.tsv    --stem judge_tier_robustness_61axes_all --include_models $FIRST $EARLY
rob --axes_file scripts/tier_eval_axes_20.tsv --stem judge_tier_robustness_20axes --include_models $FIRST20
rob --axes_file scripts/tier_eval_axes.tsv    --stem judge_tier_robustness_61axes_rubrics --include_models $OLD61 $NEW61
rob --axes_file scripts/tier_eval_axes_20.tsv --stem judge_tier_robustness_20axes_rubrics --include_models $OLD20 $NEW20
rob --axes_file scripts/tier_eval_axes.tsv    --stem judge_tier_robustness_61axes_grid --include_models $GRID61
# Unfiltered vectors (scripts/judge_tier_build_unfiltered_vectors.py): the step-3 filter's coupling
$PY results_analysis/judge_tier_robustness.py --experiments_dir $E --data_dir "runpod_workspace/qwen/qwen-3-32b Roger 8slot unfiltered" \
  --output_dir $E --cache_dir $E/_tier_eval_scratch/robustness_cache_unfiltered --axes_file scripts/tier_eval_axes.tsv \
  --stem judge_tier_robustness_61axes_unfiltered --include_models $FIRST

# The pooled Pareto plots, arrows from the old rubric to the batched method
pool --robustness_json $E/judge_tier_robustness_61axes.json --include_models $FIRST --output $E/pareto_pooled_61axes.png
pool --axes_file scripts/tier_eval_axes_20.tsv --robustness_json $E/judge_tier_robustness_20axes.json --include_models $FIRST20 --output $E/pareto_pooled_20axes.png
pool --robustness_json $E/judge_tier_robustness_61axes_rubrics.json --include_models $OLD61 $NEW61 --links $LINKS61 --output $E/pareto_pooled_61axes_rubrics.png
pool --axes_file scripts/tier_eval_axes_20.tsv --robustness_json $E/judge_tier_robustness_20axes_rubrics.json --include_models $OLD20 $NEW20 --links $LINKS20 --output $E/pareto_pooled_20axes_rubrics.png
pool --robustness_json $E/judge_tier_robustness_61axes_grid.json --include_models $GRID61 \
  --links gpt-5.6-luna@none@b10x2:gpt-5.6-luna@none@b10n20x2 gpt-6-luna@b10x2:gpt-6-luna@b10n20x2 claude-haiku-5-5@b10x2:claude-haiku-5-5@b10n20x2 \
  --links_label 'same judge: N=10 → N=20' --output $E/pareto_pooled_61axes_grid.png

# The rubric tuning, on the development axes
DEV=scripts/tier_eval_axes_dev30.tsv
TUNE=(gpt-4.1-mini gpt-4.1-mini@c8 gpt-4.1-mini@rep2 gpt-4.1-mini@v3fix gpt-4.1-mini@b10x2)
for j in gpt-6-luna claude-haiku-5-5 gpt-5.6-luna@none; do
  TUNE+=($j $j@b10x2); for c in c1 c2 c3 c4 c5 c6 c7 c8 c9 c10; do TUNE+=($j@$c); done
done
rob --axes_file $DEV --stem judge_tier_robustness_dev30_rubric_tuning --include_models $TUNE
$PY scripts/judge_tier_tuning_panels.py --robustness_json $E/judge_tier_robustness_dev30_rubric_tuning.json --output $E/pareto_pooled_dev30_rubric_tuning.png

# The optimisation grid on the development axes
G=$E/judge_tier_grid_dev30
mkdir -p $G
$PY results_analysis/judge_tier_robustness.py --experiments_dir $E --data_dir "$D" --output_dir $G --cache_dir $C --axes_file $DEV \
  --stem judge_tier_grid_12way_rho --include_models $($PY scripts/judge_tier_grid_12way.py --print_cells)
$PY scripts/judge_tier_grid_analysis.py --experiments_dir $E --axes_file $DEV
$PY scripts/judge_tier_grid_12way.py --robustness_json $G/judge_tier_grid_12way_rho.json --output_dir $G
$PY scripts/judge_tier_grid_shapes.py --robustness_json $G/judge_tier_grid_12way_rho.json --output_dir $G

# Judge sets by passes, instruction and description judging, on the 61 axes (about 4 minutes each; the
# slice cache above)
I=$E/instruction_judging_12way
$PY scripts/judge_instruction_12way.py --experiments_dir $E --data_dir "$D" --cache_dir $C --output_dir $I
$PY scripts/judge_instruction_12way.py --experiments_dir $E --data_dir "$D" --cache_dir $C --output_dir $I --mode descriptions
