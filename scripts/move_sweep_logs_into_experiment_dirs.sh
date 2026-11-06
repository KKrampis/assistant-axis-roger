#!/bin/bash
# Retroactively move existing sweep logs into their experiment dirs.
#
# Today (pre 2026-05-14) sweep logs lived in /workspace/assistant-axis/
# (or /workspace/) wherever Roger's shell-redirect of stdout landed.
# After this change, run_sweep.py auto-attaches a FileHandler to
# {output_root}/sweep.log so future runs land in the right place.  This
# script is a one-shot cleanup for the historical 14 sweep logs.
#
# Idempotent: if a destination file already exists, skips (warns)
# rather than overwriting.  Safe to re-run.
#
# Usage on runpod:
#   bash /workspace/assistant-axis/scripts/move_sweep_logs_into_experiment_dirs.sh

set -euo pipefail

STEERING_ROOT="/workspace/outputs/qwen-3-32b/steering"
SRC_REPO="/workspace/assistant-axis"
SRC_WORKSPACE="/workspace"

# (source_path, dest_experiment_dir, dest_basename)
# Two-run experiments (journalist, merchant) get sweep_1.log / sweep_2.log
# so both partial-run logs are preserved.  Everything else lands as
# sweep.log (the canonical name future runs will use).
MOVES=(
    "${SRC_REPO}/sweep_architect_ecocentric_v1.log:architect_ecocentric_v1:sweep.log"
    "${SRC_REPO}/sweep_bartender_progressive_v1.log:bartender_progressive_v1:sweep.log"
    "${SRC_REPO}/sweep_doctor_honest_v1.log:doctor_honest_v1:sweep.log"
    "${SRC_REPO}/sweep_mediator_truthful_v1.log:mediator_truthful_v1:sweep.log"
    "${SRC_REPO}/sweep_parent_relativist_v1.log:parent_relativist_v1:sweep.log"
    "${SRC_REPO}/sweep_pharmacist_helpful_v1.log:pharmacist_helpful_v1:sweep.log"
    "${SRC_REPO}/sweep_pilot_improvisational_v1.log:pilot_improvisational_v1:sweep.log"
    "${SRC_REPO}/sweep_planner_systems_v1.log:planner_systems_v1:sweep.log"
    "${SRC_REPO}/sweep_soldier_concise_v1.log:soldier_concise_v1:sweep.log"
    "${SRC_REPO}/journalist_sweep.log:journalist_callous_v1:sweep_1.log"
    "${SRC_REPO}/journalist_sweep2.log:journalist_callous_v1:sweep_2.log"
    "${SRC_REPO}/merchant_sweep.log:merchant_guileless_v1:sweep_1.log"
    "${SRC_REPO}/merchant_sweep2.log:merchant_guileless_v1:sweep_2.log"
    "${SRC_REPO}/teacher_sweep.log:teacher_egalitarian_v1:sweep.log"
    "${SRC_WORKSPACE}/chef_helpful_v1.log:chef_helpful_v1:sweep.log"
)

moved=0
skipped=0
missing=0
for entry in "${MOVES[@]}"; do
    src="${entry%%:*}"
    rest="${entry#*:}"
    exp="${rest%%:*}"
    dest_name="${rest#*:}"
    dest_dir="${STEERING_ROOT}/${exp}"
    dest="${dest_dir}/${dest_name}"

    if [[ ! -e "$src" ]]; then
        echo "SKIP (missing source): $src"
        missing=$((missing + 1))
        continue
    fi
    if [[ ! -d "$dest_dir" ]]; then
        echo "SKIP (no experiment dir): $dest_dir"
        missing=$((missing + 1))
        continue
    fi
    if [[ -e "$dest" ]]; then
        echo "SKIP (dest exists): $dest"
        skipped=$((skipped + 1))
        continue
    fi
    echo "MOVE $src -> $dest"
    mv "$src" "$dest"
    moved=$((moved + 1))
done

echo
echo "Done: moved=$moved, skipped=$skipped, missing=$missing"
