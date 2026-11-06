#!/usr/bin/env bash
# Phase 5a (May 2026): Recompute all aggregate JSONs that became stale
# after the Phase 5b/5c.1/5d.1 judging sweeps rewrote the upstream
# scores_*.json + projections.json + correlations.json caches with
# disambiguated `name|R` / `name|T` keys.
#
# No judge calls; pure numerical / plot recomputation.  Each script
# reads from existing on-disk v2 caches and writes a new envelope-
# bearing output.  Audit tools will report each output as `current`
# after this script finishes.
#
# Tier 1 (parallel-safe): independent producers.
# Tier 2: consumers of `whitening_k_sweep_*` (the peak fits) and
#         of `gpt_sonnet_weight_sweep_*` (the scatter).
#
# Run from repo root.

# NOTE: -e is intentionally omitted so a single producer failure
# doesn't abort the remaining 25.  Each per-job failure is logged and
# tallied; final summary at exit lists everything that didn't finish
# clean.
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p /tmp/phase5a_logs
LOG_DIR="/tmp/phase5a_logs"
echo "Logs: $LOG_DIR"

PHASE5A_FAILED=()

run_one() {
    local name="$1"; shift
    local log="$LOG_DIR/$name.log"
    echo "[$(date +%H:%M:%S)] starting: $name"
    if "$@" >"$log" 2>&1; then
        echo "[$(date +%H:%M:%S)] OK:       $name"
    else
        echo "[$(date +%H:%M:%S)] FAILED:   $name (see $log)"
        tail -20 "$log" | sed 's/^/    /'
        PHASE5A_FAILED+=("$name")
    fi
}

# ----- TIER 1 ----------------------------------------------------------------
echo "=== Tier 1: independent producers (16 jobs) ==="

# whitening_k_sweep (6) -- inputs to tier 2 peak_fit/weighted_scatter
run_one wks_di_slot3       uv run python -m results_analysis.whitening_k_sweep --slot 3 --pairs pair_list_di.json
run_one wks_di_slot6       uv run python -m results_analysis.whitening_k_sweep --slot 6 --pairs pair_list_di.json
run_one wks_di_slot7       uv run python -m results_analysis.whitening_k_sweep --slot 7 --pairs pair_list_di.json
run_one wks_resp_slot3     uv run python -m results_analysis.whitening_k_sweep --slot 3 --pairs pair_list_responses.json
run_one wks_resp_slot6     uv run python -m results_analysis.whitening_k_sweep --slot 6 --pairs pair_list_responses.json
run_one wks_resp_slot7     uv run python -m results_analysis.whitening_k_sweep --slot 7 --pairs pair_list_responses.json

# gpt_sonnet_weight_sweep (3) -- inputs to gpt_vs_sonnet_scatter
run_one gss_slot3          uv run python -m results_analysis.gpt_sonnet_weight_sweep --slot 3
run_one gss_slot6          uv run python -m results_analysis.gpt_sonnet_weight_sweep --slot 6
run_one gss_slot7          uv run python -m results_analysis.gpt_sonnet_weight_sweep --slot 7

# gpt_haiku weight sweeps (3)
run_one ghs_di_slot6       uv run python results_analysis/gpt_sonnet_weight_sweep.py --slot 6 --second_judge haiku
run_one ghs_resp_full      uv run python -m results_analysis.gpt_anthropic_response_weight_sweep --anthropic_combo haiku_full --scores_filename scores_responses__rubric_v1.json
run_one ghs_resp_q9        uv run python -m results_analysis.gpt_anthropic_response_weight_sweep --anthropic_combo haiku_q9

# response_di_weight_sweep (2)
run_one rds_slot6          uv run python results_analysis/response_di_weight_sweep.py
run_one rds_slot6_no_eco   uv run python results_analysis/response_di_weight_sweep.py --exclude_axes ecocentric_vs_anthropocentric

# rho_by_layer -- already recomputed in foreground on 2026-05-11 (21 min, ~10k SVDs);
# uncomment to re-run.
# run_one rho_by_layer       uv run python -m results_analysis.rho_by_layer

# rubric_v1_v2_compare
run_one rubric_v1_v2       uv run python -m results_analysis.rubric_v1_v2_compare

# optimal_axis_for_judge diagnostics (3)
run_one oafj_di_combined   uv run python results_analysis/optimal_axis_for_judge.py --data_dir 'runpod_workspace/qwen/qwen-3-32b Roger 8slot' --experiment_dir roger/axis_judge_experiments/truthful_vs_deceitful --score_source di_combined --seed_pair truthful deceitful --exclude_names truthful,deceitful --slot 6 --layer 25 --cv_folds 5
run_one oafj_gpt           uv run python results_analysis/optimal_axis_for_judge.py --data_dir 'runpod_workspace/qwen/qwen-3-32b Roger 8slot' --experiment_dir roger/axis_judge_experiments/truthful_vs_deceitful --score_source gpt --seed_pair truthful deceitful --exclude_names truthful,deceitful --slot 6 --layer 25 --cv_folds 5
run_one oafj_haiku         uv run python results_analysis/optimal_axis_for_judge.py --data_dir 'runpod_workspace/qwen/qwen-3-32b Roger 8slot' --experiment_dir roger/axis_judge_experiments/truthful_vs_deceitful --score_source haiku --seed_pair truthful deceitful --exclude_names truthful,deceitful --slot 6 --layer 25 --cv_folds 5

# ----- TIER 2 ----------------------------------------------------------------
echo "=== Tier 2: consumers of tier 1 (10 jobs) ==="

run_one wkpf_di_slot3      uv run python -m results_analysis.whitening_k_peak_fit --slot 3 --pairs pair_list_di.json
run_one wkpf_di_slot6      uv run python -m results_analysis.whitening_k_peak_fit --slot 6 --pairs pair_list_di.json
run_one wkpf_di_slot7      uv run python -m results_analysis.whitening_k_peak_fit --slot 7 --pairs pair_list_di.json
run_one wkpf_resp_slot3    uv run python results_analysis/whitening_k_peak_fit.py --pairs pair_list_responses.json --slot 3
run_one wkpf_resp_slot6    uv run python results_analysis/whitening_k_peak_fit.py --pairs pair_list_responses.json --slot 6
run_one wkpf_resp_slot7    uv run python results_analysis/whitening_k_peak_fit.py --pairs pair_list_responses.json --slot 7
run_one wkws_slot3         uv run python results_analysis/whitening_k_weighted_scatter.py --slot 3
run_one wkws_slot6         uv run python results_analysis/whitening_k_weighted_scatter.py --slot 6
run_one wkws_slot7         uv run python results_analysis/whitening_k_weighted_scatter.py --slot 7

run_one gpt_vs_sonnet      uv run python -m results_analysis.gpt_vs_sonnet_scatter

echo "=== Phase 5a recompute summary ==="
if [ ${#PHASE5A_FAILED[@]} -eq 0 ]; then
    echo "All jobs OK."
    exit 0
fi
echo "Failures (${#PHASE5A_FAILED[@]}):"
for n in "${PHASE5A_FAILED[@]}"; do
    echo "  - $n  (see $LOG_DIR/$n.log)"
done
exit 1
