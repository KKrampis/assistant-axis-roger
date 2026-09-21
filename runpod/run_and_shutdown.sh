#!/bin/bash
#
# Run pipeline/run_pipeline.sh, then shut this pod down automatically --
# so a finished (or failed) run doesn't keep billing GPU-hours unattended.
#
# Run this ON THE POD, after bootstrap.sh, instead of calling
# run_pipeline.sh directly:
#
#   bash runpod/run_and_shutdown.sh --mode christina \
#       --model Qwen/Qwen3-32B --tensor_parallel_size 1 \
#       --reduce_questions 3 \
#       --roles_dir ../data/traits/instructions/_moral_circle \
#       --output_dir /workspace/outputs/qwen-3-32b/moral-circle
#
# Every argument is forwarded verbatim to pipeline/run_pipeline.sh.
#
# Uses RUNPOD_POD_ID and RUNPOD_API_KEY, which RunPod auto-injects into
# every pod's environment (a pod-scoped key, not your real account-wide
# RUNPOD_API_KEY -- that one never needs to be copied onto the pod at all).
# Verify both are present with: env | grep RUNPOD_
#
# Default action once the pipeline exits: STOP the pod (pauses GPU billing,
# the persistent volume -- and your results on it -- survive). Pass
# --terminate to delete the pod entirely instead (only do this after you've
# pulled results off; the volume is destroyed too). Pass
# --keep-alive-on-failure to skip shutdown specifically when
# run_pipeline.sh exits non-zero, so you can SSH back in and debug instead
# of losing the pod mid-investigation (a failed run still shuts down by
# default otherwise -- an unattended failure left running is a bigger cost
# risk than losing debug access).
#
# NOTE: this hasn't been exercised against a live RunPod pod yet (built and
# reviewed, not run) -- the API contract (POST /pods/{id}/stop,
# DELETE /pods/{id}, bearer auth) matches launch_pod.py's, which mirrors
# RunPod's documented REST API, but confirm the pod-scoped RUNPOD_API_KEY
# actually has permission to stop/delete its own pod on your first real run
# before relying on it unattended.

set -uo pipefail  # NOT -e: a failed run_pipeline.sh must not skip shutdown

TERMINATE=false
KEEP_ALIVE_ON_FAILURE=false
PIPELINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --terminate)
            TERMINATE=true; shift ;;
        --keep-alive-on-failure)
            KEEP_ALIVE_ON_FAILURE=true; shift ;;
        *)
            PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_POD_ID and/or RUNPOD_API_KEY not set." >&2
    echo "These should be auto-injected by RunPod into every pod's environment." >&2
    echo "Check with: env | grep RUNPOD_" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running pipeline (pod $RUNPOD_POD_ID will shut down when this finishes) ==="
"$SCRIPT_DIR/../pipeline/run_pipeline.sh" "${PIPELINE_ARGS[@]}"
PIPELINE_EXIT=$?

if [ "$PIPELINE_EXIT" -ne 0 ] && [ "$KEEP_ALIVE_ON_FAILURE" = "true" ]; then
    echo "=== Pipeline failed (exit $PIPELINE_EXIT); --keep-alive-on-failure set, NOT shutting down ===" >&2
    echo "Pod $RUNPOD_POD_ID left running -- stop it yourself when done debugging:" >&2
    echo "  python3 runpod/stop_pod.py $RUNPOD_POD_ID  (from your local machine)" >&2
    exit "$PIPELINE_EXIT"
fi

ACTION="stop"
METHOD="POST"
PATH_SUFFIX="/stop"
if [ "$TERMINATE" = "true" ]; then
    ACTION="terminate"
    METHOD="DELETE"
    PATH_SUFFIX=""
fi

echo "=== Pipeline exited $PIPELINE_EXIT. Shutting pod down now (action: $ACTION) ==="
curl -sS -X "$METHOD" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}${PATH_SUFFIX}"
echo ""
echo "Shutdown request sent. If this was the last thing keeping the SSH"
echo "session open, the connection may drop shortly."

exit "$PIPELINE_EXIT"
