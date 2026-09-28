#!/bin/bash
# Rejudge axis correlations after the April–May 2026 trait edits.
#
# Sequencing:
#   1. gpt desc/instr (cheap, fast feedback)
#   2. gpt response cohort A (12 axes × {b15, b10}) for traits + roles
#   3. gpt response cohort B (3 axes × {b5, b7}) traits + roles
#   4. sonnet desc/instr
#   5. haiku desc/instr
#   6. haiku response cohort B ({b10, b10_q9})
#   7. sonnet response cohort B ({b10_q9})
#
# Each phase logs to /tmp/rejudge_<phase>_<timestamp>.log.
#
# This script is idempotent: every axis-judge invocation uses --refill_gaps,
# so already-judged entries are kept and only the gaps (created by the
# pre-script cache invalidation) get filled.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="roger/axis_judge_experiments"
DATA_DIR='runpod_workspace/qwen/qwen-3-32b Roger 8slot'
LOG_DIR="/tmp/rejudge_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"

PAIR_ALL="$ROOT/pair_list_di.json"          # desc+instr cohort — covers everything desc/instr touched
PAIR_A="$ROOT/pair_list_responses.json"     # responses cohort — gpt b15/b10 response judging
PAIR_B="$ROOT/pair_list_3_cohort_b.json"    # 3 axes — cohort B (multi-judge / multi-batch)

# Helper: orchestrator wrapper
run_orch() {
    local phase="$1"; shift
    echo
    echo "=== [$phase] $* ==="
    local log="$LOG_DIR/${phase}.log"
    if uv run python results_analysis/run_axis_experiment_batch.py "$@" \
        --data_dir "$DATA_DIR" --instructions_dir data \
        --concurrency 3 --refill_gaps \
        2>&1 | tee "$log"; then
        echo "[$phase] DONE"
    else
        echo "[$phase] FAILED — see $log"
        exit 1
    fi
}

# ---------------------------------------------------------------
# Phase 1: gpt desc/instr — covers all 33 axes (3 fully-affected
# axes get a full re-judge from empty cache; 30 unaffected axes
# only refill the 8 invalidated trait entries).
# ---------------------------------------------------------------
run_orch phase1_gpt_descinstr \
    --pair_list "$PAIR_ALL" \
    --provider gpt --judge_model gpt-4.1-mini \
    --output_root "$ROOT" \
    --score_descriptions --score_instructions

# ---------------------------------------------------------------
# Phase 2: gpt response, cohort A (b15 + b10).
# - Traits: 12 axes; 10 unaffected get 8 trait entries refilled,
#   2 fully-affected get full ~290-entity rejudge.
# - Roles: same 12 axes; only the 2 fully-affected axes need
#   re-judge (the 10 unaffected role caches are intact and
#   --refill_gaps will skip them with no gaps).
# ---------------------------------------------------------------
for bs in 15 10; do
    suffix=""; [ "$bs" = "10" ] && suffix="_b10"

    run_orch phase2_gpt_traits_b${bs} \
        --pair_list "$PAIR_A" \
        --provider gpt --judge_model gpt-4.1-mini \
        --output_root "$ROOT" \
        --score_responses \
        --scores_dir "$DATA_DIR/traits/scores" \
        --responses_dir "$DATA_DIR/traits/responses" \
        --subdir "gpt_responses_traits${suffix}" \
        --response_target_batch_size $bs

    run_orch phase2_gpt_roles_b${bs} \
        --pair_list "$PAIR_A" \
        --provider gpt --judge_model gpt-4.1-mini \
        --output_root "$ROOT" \
        --score_responses \
        --scores_dir "$DATA_DIR/roles/scores" \
        --responses_dir "$DATA_DIR/roles/responses" \
        --subdir "gpt_responses_roles${suffix}" \
        --response_target_batch_size $bs
done

# ---------------------------------------------------------------
# Phase 3: gpt response, cohort B extras (b5 + b7).
# - 3 axes only.  Traits + roles.  Only the 8 affected traits
#   need refill in the trait subdirs; role caches (no affected
#   entities) will be skipped via --refill_gaps no-gap path.
# ---------------------------------------------------------------
for bs in 5 7; do
    run_orch phase3_gpt_traits_b${bs} \
        --pair_list "$PAIR_B" \
        --provider gpt --judge_model gpt-4.1-mini \
        --output_root "$ROOT" \
        --score_responses \
        --scores_dir "$DATA_DIR/traits/scores" \
        --responses_dir "$DATA_DIR/traits/responses" \
        --subdir "gpt_responses_traits_b${bs}" \
        --response_target_batch_size $bs

    run_orch phase3_gpt_roles_b${bs} \
        --pair_list "$PAIR_B" \
        --provider gpt --judge_model gpt-4.1-mini \
        --output_root "$ROOT" \
        --score_responses \
        --scores_dir "$DATA_DIR/roles/scores" \
        --responses_dir "$DATA_DIR/roles/responses" \
        --subdir "gpt_responses_roles_b${bs}" \
        --response_target_batch_size $bs
done

# ---------------------------------------------------------------
# Phase 4: sonnet desc/instr — same scope as phase 1.
# ---------------------------------------------------------------
run_orch phase4_sonnet_descinstr \
    --pair_list "$PAIR_ALL" \
    --provider sonnet --judge_model claude-sonnet-4-6 \
    --output_root "$ROOT" \
    --score_descriptions --score_instructions

# ---------------------------------------------------------------
# Phase 5: haiku desc/instr — orchestrator's --provider doesn't
# directly support haiku, so we pass --provider anthropic (forces
# the inner script to use anthropic) and override the subdir to
# 'haiku' so cache lands in the existing layout.
# ---------------------------------------------------------------
run_orch phase5_haiku_descinstr \
    --pair_list "$PAIR_ALL" \
    --provider anthropic --judge_model claude-haiku-4-5-20251001 \
    --output_root "$ROOT" \
    --subdir haiku \
    --score_descriptions --score_instructions

# ---------------------------------------------------------------
# Phase 6: haiku response, cohort B (b10 and b10_q9).
# ---------------------------------------------------------------
run_orch phase6_haiku_traits_b10 \
    --pair_list "$PAIR_B" \
    --provider anthropic --judge_model claude-haiku-4-5-20251001 \
    --output_root "$ROOT" \
    --subdir haiku_responses_traits_b10 \
    --score_responses \
    --scores_dir "$DATA_DIR/traits/scores" \
    --responses_dir "$DATA_DIR/traits/responses" \
    --response_target_batch_size 10

run_orch phase6_haiku_roles_b10 \
    --pair_list "$PAIR_B" \
    --provider anthropic --judge_model claude-haiku-4-5-20251001 \
    --output_root "$ROOT" \
    --subdir haiku_responses_roles_b10 \
    --score_responses \
    --scores_dir "$DATA_DIR/roles/scores" \
    --responses_dir "$DATA_DIR/roles/responses" \
    --response_target_batch_size 10

run_orch phase6_haiku_traits_b10_q9 \
    --pair_list "$PAIR_B" \
    --provider anthropic --judge_model claude-haiku-4-5-20251001 \
    --output_root "$ROOT" \
    --subdir haiku_responses_traits_b10_q9 \
    --score_responses \
    --scores_dir "$DATA_DIR/traits/scores" \
    --responses_dir "$DATA_DIR/traits/responses" \
    --response_target_batch_size 10 \
    --question_subsample_modulo 9 \
    --questions_file data/extraction_questions.jsonl

run_orch phase6_haiku_roles_b10_q9 \
    --pair_list "$PAIR_B" \
    --provider anthropic --judge_model claude-haiku-4-5-20251001 \
    --output_root "$ROOT" \
    --subdir haiku_responses_roles_b10_q9 \
    --score_responses \
    --scores_dir "$DATA_DIR/roles/scores" \
    --responses_dir "$DATA_DIR/roles/responses" \
    --response_target_batch_size 10 \
    --question_subsample_modulo 9 \
    --questions_file data/extraction_questions.jsonl

# ---------------------------------------------------------------
# Phase 7: sonnet response, cohort B (b10_q9 only).
# ---------------------------------------------------------------
run_orch phase7_sonnet_traits_b10_q9 \
    --pair_list "$PAIR_B" \
    --provider sonnet --judge_model claude-sonnet-4-6 \
    --output_root "$ROOT" \
    --subdir sonnet_responses_traits_b10_q9 \
    --score_responses \
    --scores_dir "$DATA_DIR/traits/scores" \
    --responses_dir "$DATA_DIR/traits/responses" \
    --response_target_batch_size 10 \
    --question_subsample_modulo 9 \
    --questions_file data/extraction_questions.jsonl

run_orch phase7_sonnet_roles_b10_q9 \
    --pair_list "$PAIR_B" \
    --provider sonnet --judge_model claude-sonnet-4-6 \
    --output_root "$ROOT" \
    --subdir sonnet_responses_roles_b10_q9 \
    --score_responses \
    --scores_dir "$DATA_DIR/roles/scores" \
    --responses_dir "$DATA_DIR/roles/responses" \
    --response_target_batch_size 10 \
    --question_subsample_modulo 9 \
    --questions_file data/extraction_questions.jsonl

echo
echo "=== ALL PHASES COMPLETE ==="
echo "Logs in $LOG_DIR"
