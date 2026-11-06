#!/usr/bin/env bash
# Pull the latest steering sweep outputs from the runpod to the laptop.
#
# Per-axis loop so a flaky SSH connection only kills one axis and the
# rest still get pulled.  --partial + --append-verify means an
# interrupted axis resumes cleanly on the next pass rather than
# starting from scratch.  Safe to run while sweep #2 is still
# producing records.jsonl on the pod (rsync's atomic temp-file
# strategy avoids partial-line reads even with concurrent writes).
#
# Usage:
#   scripts/rsync_pod_sweeps.sh            # full set (sweep #1 + #2 + baselines)
#   scripts/rsync_pod_sweeps.sh second     # only sweep-#2 second-role personas
#   scripts/rsync_pod_sweeps.sh first      # only sweep-#1 first-role personas
set -u  # don't -e: we want to keep going past per-axis SSH failures.

POD_USER="root"
POD_HOST="154.54.102.50"
POD_PORT="17140"
POD_KEY="$HOME/.ssh/id_ed25519_runpod"
POD_PATH="/workspace/outputs/qwen-3-32b/steering"

LOCAL_ROOT="outputs/qwen-3-32b/steering"
LOG="/tmp/rsync_pod_sweeps_$(date +%Y%m%d_%H%M%S).log"

SSH_OPTS=(
    -p "$POD_PORT" -i "$POD_KEY"
    -o ConnectTimeout=15
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
    -o ControlMaster=no
    -o ControlPath=none
    -o StrictHostKeyChecking=no
)

mkdir -p "$LOCAL_ROOT"

# Personas grouped by sweep wave.
FIRST_ROLES=(
    # sweep #1: 12 first-role personas in all + prefill modes.
    # (Most already downloaded; rsync will skip up-to-date files.)
    novelist_honest_v1                 novelist_honest_v1_prefill
    publisher_truthful_v1              publisher_truthful_v1_prefill
    publisher_guileless_v1             publisher_guileless_v1_prefill
    cartographer_egalitarian_v1        cartographer_egalitarian_v1_prefill
    navigator_progressive_v1           navigator_progressive_v1_prefill
    anarchist_concise_v1               anarchist_concise_v1_prefill
    architect_ecocentric_v3            architect_ecocentric_v3_prefill
    competitor_improvisational_v1      competitor_improvisational_v1_prefill
    virtuoso_relativist_v1             virtuoso_relativist_v1_prefill
    rebel_systems_thinker_v1           rebel_systems_thinker_v1_prefill
    chef_helpful_v1                    chef_helpful_v1_prefill
    prodigy_harmless_v1                prodigy_harmless_v1_prefill
)

SECOND_ROLES=(
    # sweep #2: 12 second-role personas, all-mode only.
    vampire_helpful_v1
    saboteur_harmless_v1
    marketer_honest_v1
    spy_truthful_v1
    lawyer_guileless_v1
    fixer_egalitarian_v1
    curator_progressive_v1
    soldier_concise_v1
    chemist_ecocentric_v1
    artist_improvisational_v1
    publisher_relativist_v1
    vampire_systems_thinker_v1
)

BASELINES=( baselines )

case "${1:-all}" in
    first)  PERSONAS=("${FIRST_ROLES[@]}") ;;
    second) PERSONAS=("${SECOND_ROLES[@]}") ;;
    base*)  PERSONAS=("${BASELINES[@]}") ;;
    all|"") PERSONAS=("${FIRST_ROLES[@]}" "${SECOND_ROLES[@]}" "${BASELINES[@]}") ;;
    *) echo "usage: $0 [all|first|second|baselines]" >&2; exit 2 ;;
esac

echo "Rsync target: $LOCAL_ROOT" | tee -a "$LOG"
echo "Personas (${#PERSONAS[@]}): ${PERSONAS[*]}" | tee -a "$LOG"
echo "Log: $LOG"
echo "" | tee -a "$LOG"

n_ok=0; n_fail=0
for persona in "${PERSONAS[@]}"; do
    src="${POD_USER}@${POD_HOST}:${POD_PATH}/${persona}/"
    dst="${LOCAL_ROOT}/${persona}/"
    mkdir -p "$dst"
    # Up to 3 attempts; rsync's --partial + --append-verify resumes.
    for attempt in 1 2 3; do
        echo "[$(date +%H:%M:%S)] rsync $persona (attempt $attempt)..." | tee -a "$LOG"
        if rsync -aH --partial --append-verify --info=stats1 \
            -e "ssh ${SSH_OPTS[*]}" \
            "$src" "$dst" >>"$LOG" 2>&1; then
            echo "  OK" | tee -a "$LOG"
            n_ok=$((n_ok + 1))
            break
        fi
        echo "  FAILED (attempt $attempt)" | tee -a "$LOG"
        if [ "$attempt" -eq 3 ]; then
            n_fail=$((n_fail + 1))
        else
            sleep $((attempt * 10))  # back off 10s, then 20s
        fi
    done
done

echo "" | tee -a "$LOG"
echo "Done: $n_ok personas downloaded, $n_fail failed after 3 attempts." | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
