#!/usr/bin/env bash
# Incrementally pull completed axes from the pod to the laptop.
# An axis is "complete" when its dir has >=14 summary.json files.
# Run periodically (cron, manual, or sweep_49axes_rsync_loop.sh).
# Safe to run while sweep is ongoing; uses --partial + per-axis loop.
set -u

POD_USER=root
POD_HOST=154.54.102.50
POD_PORT=17140
POD_KEY="$HOME/.ssh/id_ed25519_runpod"
POD_OUTPUTS=/workspace/outputs/qwen-3-32b/steering

LOCAL_OUTPUTS=outputs/qwen-3-32b/steering
BATCH=data/steering/configs/multi_cell_batch_v1_remaining49.txt
LOG=/tmp/rsync_completed_axes.log

SSH_OPTS=(
    -p "$POD_PORT" -i "$POD_KEY"
    -o ConnectTimeout=15
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
    -o ControlMaster=no -o ControlPath=none
    -o StrictHostKeyChecking=no -o BatchMode=yes
)

cd "$(dirname "$0")/.."
mkdir -p "$LOCAL_OUTPUTS"

echo "[$(date)] rsync_completed_axes started" | tee -a "$LOG"

# Get the list of currently-complete axes from the pod (each has >=14 summary.json files).
remote_completed=$(
    ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" \
        "cd /workspace/assistant-axis && \
         while IFS= read -r cfg; do \
            [ -z \"\$cfg\" ] && continue; \
            name=\$(basename \"\$cfg\" .yaml); \
            n=\$(find $POD_OUTPUTS/\"\$name\" -name summary.json 2>/dev/null | wc -l); \
            if [ \"\$n\" -ge 14 ]; then echo \"\$name\"; fi; \
         done < $BATCH" 2>>"$LOG"
)

n_remote=$(echo "$remote_completed" | grep -c .)
echo "[$(date)] $n_remote axes complete on pod" | tee -a "$LOG"

n_pulled=0; n_skipped=0; n_failed=0
for name in $remote_completed; do
    # Skip if local dir already has summary.json for every cell.
    n_local=$(find "$LOCAL_OUTPUTS/$name" -name summary.json 2>/dev/null | wc -l)
    if [ "$n_local" -ge 14 ]; then
        n_skipped=$((n_skipped + 1))
        continue
    fi
    echo "[$(date)] pulling $name (local has $n_local)" | tee -a "$LOG"
    src="$POD_USER@$POD_HOST:$POD_OUTPUTS/$name/"
    dst="$LOCAL_OUTPUTS/$name/"
    mkdir -p "$dst"
    # macOS bundled rsync is old: avoid --info, --append-verify, etc.
    if rsync -aH --partial -q \
        -e "ssh ${SSH_OPTS[*]}" \
        "$src" "$dst" >>"$LOG" 2>&1; then
        n_pulled=$((n_pulled + 1))
    else
        echo "[$(date)] FAILED $name" | tee -a "$LOG"
        n_failed=$((n_failed + 1))
    fi
done

echo "[$(date)] done: pulled=$n_pulled, already-local=$n_skipped, failed=$n_failed" | tee -a "$LOG"
