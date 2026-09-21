#!/bin/bash
#
# Copy a pipeline run's small result files (vectors, axis.pt, scores) into
# the repo and push them, from the pod itself.
#
# Requires bootstrap.sh to have been run with GH_PUSH_TOKEN set (a GitHub
# fine-grained PAT scoped to just this repo) -- otherwise the remote is
# still the default read-only HTTPS clone and the final `git push` here
# will fail with an auth error. That's intentional: no silent no-op.
#
# Usage:
#   bash runpod/push_results.sh \
#       --src /workspace/outputs/qwen-3-32b/moral-circle \
#       --dest results/moral-circle-qwen-3-32b \
#       --message "Add moral-circle axis results (qwen-3-32b)"
#
# What gets copied: everything under --src EXCEPT the responses/ and
# activations/ subdirectories (large raw intermediates -- never meant for
# git, and already covered by the repo's outputs/ gitignore rule elsewhere).
# In practice that means vectors/, axis*.pt, and any scores*/ directories
# (including the scores_<model>/ split used when judging with multiple
# models -- see pipeline/SUBSET_RUNS.md).

set -euo pipefail

SRC=""
DEST=""
MESSAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src) SRC="$2"; shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        --message) MESSAGE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$SRC" ] || [ -z "$DEST" ]; then
    echo "Usage: push_results.sh --src <output_dir> --dest <path-in-repo> [--message <msg>]" >&2
    exit 1
fi
if [ ! -d "$SRC" ]; then
    echo "ERROR: --src directory does not exist: $SRC" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FULL_DEST="$REPO_DIR/$DEST"

echo "=== Copying results ==="
echo "  from: $SRC"
echo "  to:   $FULL_DEST"
mkdir -p "$FULL_DEST"
rsync -av --exclude 'responses/' --exclude 'activations/' "$SRC"/ "$FULL_DEST"/

cd "$REPO_DIR"

echo ""
echo "=== git status for $DEST ==="
git status --porcelain -- "$DEST"

if [ -z "$(git status --porcelain -- "$DEST")" ]; then
    echo "Nothing changed under $DEST -- nothing to commit."
    exit 0
fi

git add -- "$DEST"

if [ -z "$MESSAGE" ]; then
    MESSAGE="Add pipeline results: $DEST"
fi

git commit -m "$(cat <<EOF
$MESSAGE

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"

echo ""
echo "=== Pushing ==="
git push origin HEAD

echo ""
echo "Done. Pushed $DEST to $(git rev-parse --abbrev-ref HEAD)."
