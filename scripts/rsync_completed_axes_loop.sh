#!/usr/bin/env bash
# Run rsync_completed_axes.sh every 30 minutes until manually killed.
# Survives intermittent SSH failures by exiting non-fatally and waiting
# for the next cycle.  Safe to interrupt and restart any time.
set -u

cd "$(dirname "$0")/.."
LOG=/tmp/rsync_completed_axes_loop.log

echo "[$(date)] loop started, polling every 30 min; pid=$$" | tee -a "$LOG"
while true; do
    bash scripts/rsync_completed_axes.sh >> "$LOG" 2>&1 || true
    sleep 1800   # 30 min
done
