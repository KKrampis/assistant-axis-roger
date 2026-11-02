#!/usr/bin/env bash
# Run judge_tier_cost_eval.py for 18 new axes x 3 models (luna, terra, opus).
# Angel_vs_demon and decisive_vs_indecisive are already done and skipped here.
#
# Superseded (2026-11-01) by scripts/run_tier_eval.sh, which runs all 61 axes of
# scripts/tier_eval_axes.tsv in one invocation with batch / Flex transports.  This
# script still works: it passes --transport realtime, so each (axis, model) call is
# judged in real time as before (the default, auto, would submit and wait for one
# batch per axis).
#
# Prerequisites:
#   Corpus pinned to /tmp/corpus_2026-05-24 (run once):
#     mkdir -p /tmp/corpus_2026-05-24
#     git archive 1448a84c data/traits/instructions data/roles/instructions | tar -x -C /tmp/corpus_2026-05-24
#
#   API keys: either export OPENAI_API_KEY/ANTHROPIC_API_KEY, or create .env
#   at the repo root with those two lines (gitignored; auto-loaded below).
#
# Usage (from repo root):
#   bash scripts/run_tier_eval_20axes.sh
#
# Resume: skips any model dir that already has a usage.json (safe to re-run).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Load API keys from .env at repo root if present (gitignored; never committed)
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi

CORPUS=/tmp/corpus_2026-05-24/data
EXPERIMENTS="$REPO_ROOT/roger/axis_judge_experiments"

# Verify corpus exists
if [[ ! -d "$CORPUS/traits/instructions" ]]; then
  echo "ERROR: corpus not found at $CORPUS -- run:" >&2
  echo "  mkdir -p /tmp/corpus_2026-05-24" >&2
  echo "  git archive 1448a84c data/traits/instructions data/roles/instructions | tar -x -C /tmp/corpus_2026-05-24" >&2
  exit 1
fi

# Verify keys
if [[ -z "${OPENAI_API_KEY:-}" || -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY and ANTHROPIC_API_KEY must be set in environment." >&2
  exit 1
fi

MODELS=(gpt-5.6-luna gpt-5.6-terra claude-opus-5)

# 18 axes as "axis_dir pos neg" lines -- bash 3.2 compatible (no associative arrays)
AXIS_SPECS=(
  "trustworthy_vs_untrustworthy        trustworthy      untrustworthy"
  "constructive_vs_destructive         constructive     destructive"
  "helpful_vs_unhelpful                helpful          unhelpful"
  "benign_vs_malicious                 benign           malicious"
  "harmless_vs_harmful                 harmless         harmful"
  "improvisational_vs_methodical       improvisational  methodical"
  "cooperative_vs_competitive          cooperative      competitive"
  "reductionist_vs_holistic            reductionist     holistic"
  "individualistic_vs_collectivistic   individualistic  collectivistic"
  "passionate_vs_dispassionate         passionate       dispassionate"
  "compassionate_vs_callous            compassionate    callous"
  "convergent_vs_divergent             convergent       divergent"
  "playful_vs_serious                  playful          serious"
  "truthful_vs_deceitful               truthful         deceitful"
  "practical_vs_theoretical            practical        theoretical"
  "casual_vs_formal                    casual           formal"
  "generous_vs_stingy                  generous         stingy"
  "introverted_vs_extroverted          introverted      extroverted"
)

TOTAL_AXES=${#AXIS_SPECS[@]}
AXIS_NUM=0

for SPEC in "${AXIS_SPECS[@]}"; do
  AXIS_DIR=$(echo "$SPEC" | awk '{print $1}')
  POS=$(echo "$SPEC" | awk '{print $2}')
  NEG=$(echo "$SPEC" | awk '{print $3}')
  PROJ="$EXPERIMENTS/$AXIS_DIR/gpt/projections.json"
  OUT_DIR="$EXPERIMENTS/${AXIS_DIR}_tier_eval"
  AXIS_NUM=$((AXIS_NUM + 1))

  if [[ ! -f "$PROJ" ]]; then
    echo "[$AXIS_NUM/$TOTAL_AXES] SKIP (no projections.json): $AXIS_DIR" >&2
    continue
  fi

  echo ""
  echo "=== [$AXIS_NUM/$TOTAL_AXES] $AXIS_DIR ==="

  for MODEL in "${MODELS[@]}"; do
    USAGE_FILE="$OUT_DIR/$MODEL/usage.json"
    if [[ -f "$USAGE_FILE" ]]; then
      echo "  SKIP (already done): $MODEL"
      continue
    fi

    echo "  Running: $MODEL ..."
    uv run python "$REPO_ROOT/results_analysis/judge_tier_cost_eval.py" \
      --pair "$POS" "$NEG" \
      --pair_type traits \
      --projections_file "$PROJ" \
      --instructions_dir "$CORPUS" \
      --judge_model "$MODEL" \
      --output_dir "$OUT_DIR" \
      --transport realtime
  done
done

echo ""
echo "=== All done ==="
