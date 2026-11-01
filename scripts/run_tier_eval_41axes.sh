#!/usr/bin/env bash
# Run judge_tier_cost_eval.py for the remaining 41 axes x 3 models (luna, terra, opus).
# The first 20 axes (angel/demon, decisive/indecisive + 18 from run_tier_eval_20axes.sh)
# are already done and skipped by the usage.json resume check below.
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
#   bash scripts/run_tier_eval_41axes.sh
#   bash scripts/run_tier_eval_41axes.sh 2>&1 | tee /tmp/tier_eval_41axes.log
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

# 41 axes as "axis_dir pos neg pair_type" lines -- bash 3.2 compatible (no associative arrays)
AXIS_SPECS=(
  "accessible_vs_esoteric                              accessible                              esoteric                       traits"
  "aligned_artificial_intelligence_vs_paperclip_maximizer  aligned_artificial_intelligence     paperclip_maximizer            roles"
  "arrogant_vs_humble                                  arrogant                                humble                         traits"
  "blunt_vs_tactful                                    blunt                                   tactful                        traits"
  "careless_vs_conscientious                           careless                                conscientious                  traits"
  "conciliatory_vs_confrontational                     conciliatory                            confrontational                traits"
  "concise_vs_verbose                                  concise                                 verbose                        traits"
  "confident_vs_uncertain                              confident                               uncertain                      traits"
  "constructivist_vs_essentialist                      constructivist                          essentialist                   traits"
  "curious_vs_incurious                                curious                                 incurious                      traits"
  "dependable_vs_undependable                          dependable                              undependable                   traits"
  "descriptive_vs_prescriptive                         descriptive                             prescriptive                   traits"
  "detached_vs_empathetic                              detached                                empathetic                     traits"
  "dominant_vs_submissive                              dominant                                submissive                     traits"
  "earnest_vs_sardonic                                 earnest                                 sardonic                       traits"
  "ecocentric_vs_anthropocentric                       ecocentric                              anthropocentric                traits"
  "egalitarian_vs_elitist                              egalitarian                             elitist                        traits"
  "ethereal_vs_grounded                                ethereal                                grounded                       traits"
  "forgiving_vs_unforgiving                            forgiving                               unforgiving                    traits"
  "fragile_vs_resilient                                fragile                                 resilient                      traits"
  "guardian_vs_destroyer                               guardian                                destroyer                      roles"
  "guileless_vs_scheming                               guileless                               scheming                       traits"
  "honest_vs_dishonest                                 honest                                  dishonest                      traits"
  "idealistic_vs_pragmatic                             idealistic                              pragmatic                      traits"
  "impatient_vs_patient                                impatient                               patient                        traits"
  "innovative_vs_traditional                           innovative                              traditional                    traits"
  "irreverent_vs_reverent                              irreverent                              reverent                       traits"
  "materialistic_vs_spiritual                          materialistic                           spiritual                      traits"
  "obedient_vs_rebellious                              obedient                                rebellious                     traits"
  "opaque_vs_transparent                               opaque                                  transparent                    traits"
  "optimistic_vs_pessimistic                           optimistic                              pessimistic                    traits"
  "precise_vs_vague                                    precise                                 vague                          traits"
  "predator_vs_prey                                    predator                                prey                           roles"
  "proactive_vs_reactive                               proactive                               reactive                       traits"
  "progressive_vs_conservative                         progressive                             conservative                   traits"
  "quantitative_vs_qualitative                         quantitative                            qualitative                    traits"
  "relativist_vs_absolutist                            relativist                              absolutist                     traits"
  "religious_vs_secular                                religious                               secular                        traits"
  "superficial_vs_thorough                             superficial                             thorough                       traits"
  "symbiont_vs_parasite                                symbiont                                parasite                       roles"
  "systems_thinker_vs_analytical                       systems_thinker                         analytical                     traits"
)

TOTAL_AXES=${#AXIS_SPECS[@]}
AXIS_NUM=0

for SPEC in "${AXIS_SPECS[@]}"; do
  AXIS_DIR=$(echo "$SPEC" | awk '{print $1}')
  POS=$(echo "$SPEC" | awk '{print $2}')
  NEG=$(echo "$SPEC" | awk '{print $3}')
  PTYPE=$(echo "$SPEC" | awk '{print $4}')
  PROJ="$EXPERIMENTS/$AXIS_DIR/gpt/projections.json"
  OUT_DIR="$EXPERIMENTS/${AXIS_DIR}_tier_eval"
  AXIS_NUM=$((AXIS_NUM + 1))

  if [[ ! -f "$PROJ" ]]; then
    echo "[$AXIS_NUM/$TOTAL_AXES] SKIP (no projections.json): $AXIS_DIR" >&2
    continue
  fi

  echo ""
  echo "=== [$AXIS_NUM/$TOTAL_AXES] $AXIS_DIR ($PTYPE) ==="

  for MODEL in "${MODELS[@]}"; do
    USAGE_FILE="$OUT_DIR/$MODEL/usage.json"
    if [[ -f "$USAGE_FILE" ]]; then
      echo "  SKIP (already done): $MODEL"
      continue
    fi

    echo "  Running: $MODEL ..."
    uv run python "$REPO_ROOT/results_analysis/judge_tier_cost_eval.py" \
      --pair "$POS" "$NEG" \
      --pair_type "$PTYPE" \
      --projections_file "$PROJ" \
      --instructions_dir "$CORPUS" \
      --judge_model "$MODEL" \
      --output_dir "$OUT_DIR" \
      --transport realtime
  done
done

echo ""
echo "=== All done ==="
