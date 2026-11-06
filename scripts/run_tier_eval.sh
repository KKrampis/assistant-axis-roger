#!/usr/bin/env bash
# Run judge_tier_cost_eval.py over all 61 axes of scripts/tier_eval_axes.tsv
# for a list of judge models, in one invocation (2026-11-01; supersedes
# run_tier_eval_20axes.sh and run_tier_eval_41axes.sh).
#
# The script picks a transport per model (--transport auto): Anthropic models
# by Message Batches, OpenAI models by Flex where MODEL_CAPABILITIES says
# OpenAI takes it, else by the Batch API (an untested model probes Flex with
# its first request).  Models run concurrently; each cell is written as soon
# as its axis is complete.  Pass --transport realtime to call everything in
# real time instead.
#
# Prerequisites:
#   Corpus pinned to /tmp/corpus_2026-05-24 (run once):
#     mkdir -p /tmp/corpus_2026-05-24
#     git archive 1448a84c data/traits/instructions data/roles/instructions | tar -x -C /tmp/corpus_2026-05-24
#   (or point CORPUS at another copy: CORPUS=/path/to/data bash scripts/run_tier_eval.sh)
#
#   API keys: OPENAI_API_KEY / ANTHROPIC_API_KEY in the environment or in .env
#   at the repo root (gitignored).  The Python script loads .env itself
#   (load_dotenv) and refuses to start, naming the variable, when a key a
#   requested model needs is missing; this script never reads .env.
#
# Usage (from repo root):
#   bash scripts/run_tier_eval.sh --dry_run                       # preview: transports, calls, est. cost
#   bash scripts/run_tier_eval.sh                                 # default models (below)
#   MODELS="gpt-5.6-luna claude-haiku-5-5" bash scripts/run_tier_eval.sh --budget_usd 30
#   bash scripts/run_tier_eval.sh 2>&1 | tee /tmp/tier_eval.log
# Any extra arguments go to judge_tier_cost_eval.py.
#
# Resume: an axis whose <axis>_tier_eval/<model>/usage.json exists is skipped
# for that model; batches still in flight are collected before anything new
# is submitted (state under roger/axis_judge_experiments/_tier_eval_batches/).
# Exit status is the Python script's: 0 done, 1 a model failed, 2 --budget_usd hit.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CORPUS="${CORPUS:-/tmp/corpus_2026-05-24/data}"
EXPERIMENTS="${EXPERIMENTS:-$REPO_ROOT/roger/axis_judge_experiments}"
AXES_FILE="${AXES_FILE:-$REPO_ROOT/scripts/tier_eval_axes.tsv}"
read -r -a MODEL_LIST <<< "${MODELS:-gpt-5.6-luna gpt-5.6-terra claude-opus-5}"

# Verify corpus exists
if [[ ! -d "$CORPUS/traits/instructions" || ! -d "$CORPUS/roles/instructions" ]]; then
  echo "ERROR: corpus not found at $CORPUS -- run:" >&2
  echo "  mkdir -p /tmp/corpus_2026-05-24" >&2
  echo "  git archive 1448a84c data/traits/instructions data/roles/instructions | tar -x -C /tmp/corpus_2026-05-24" >&2
  exit 1
fi
if [[ ! -f "$AXES_FILE" ]]; then
  echo "ERROR: axes file not found: $AXES_FILE" >&2
  exit 1
fi

MODEL_ARGS=()
for MODEL in "${MODEL_LIST[@]}"; do
  MODEL_ARGS+=(--judge_model "$MODEL")
done

echo "Models: ${MODEL_LIST[*]}"
echo "Axes:   $AXES_FILE ($(grep -cv '^[[:space:]]*\(#\|$\)' "$AXES_FILE") axes)"
echo "Corpus: $CORPUS"

exec uv run python "$REPO_ROOT/results_analysis/judge_tier_cost_eval.py" \
  --axes_file "$AXES_FILE" \
  --experiments_dir "$EXPERIMENTS" \
  --instructions_dir "$CORPUS" \
  "${MODEL_ARGS[@]}" \
  "$@"
