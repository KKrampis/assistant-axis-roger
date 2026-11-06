#!/bin/bash
#
# Pipeline for computing the Assistant Axis.
#
# Supports two modes:
#   roger (default) — combined role+trait instructions, standalone traits/roles, default
#     --types combinations roles traits  (any subset; default: all three)
#       Each type gets its own subdirectory under OUTPUT_DIR.
#       "default" is generated once and symlinked into each type's dirs.
#     Two extra subset types restrict combinations to one half of the grid:
#       r_combinations  (goal role x non-goal trait, file prefix "r_")
#       t_combinations  (non-goal role x goal trait, file prefix "t_")
#     Both write into the SAME $OUTPUT_DIR/combinations/ tree as the full
#     "combinations" type, so downstream tooling sees a single combinations
#     directory regardless of which subset was generated.  Mixing
#     "combinations" with either subset is redundant (the subset is implied).
#   christina — standalone roles/traits from --roles_dir.
#     --types is ignored in christina mode.
#
# Usage:
#   ./pipeline/run_pipeline.sh
#   ./pipeline/run_pipeline.sh --mode christina --output_dir /workspace/qwen-3-32b/roles
#   ./pipeline/run_pipeline.sh --mode christina --output_dir /workspace/qwen-3-32b/traits \
#       --roles_dir ../data/traits/instructions
#   ./pipeline/run_pipeline.sh --types combinations roles
#   ./pipeline/run_pipeline.sh --types r_combinations          # only r_ combos
#   ./pipeline/run_pipeline.sh --types r_combinations t_combinations roles traits
#   ./pipeline/run_pipeline.sh --skip-gpu-check                # bypass pre-flight
#   ./pipeline/run_pipeline.sh --steps23serial                 # disable parallel steps 2+3
#   ./pipeline/run_pipeline.sh --no-scan                       # skip post-pipeline audit
#   ./pipeline/run_pipeline.sh --mode christina --reduce_questions 3 \
#       --roles_dir ../data/traits/instructions/_moral_circle \
#       --output_dir /workspace/outputs/qwen-3-32b/moral-circle
#       # christina mode now forwards --reduce_questions to steps 1 and 4
#
# Post-pipeline scan (default ON): after step 5, runs
# pipeline/scan_missing_vectors.py to audit every (activation, vector) pair
# and flag any corrupt-on-write activations, missing scores, below-min-count
# entities, or unloadable vectors.  Cache-backed so re-scans are cheap.
# Pass --no-scan to skip (e.g. when you want to inspect intermediate state
# yourself, or when the pipeline is part of a larger automated job that
# runs the scanner separately).
#
# Step 2/3 parallelism (default ON): step 2 (GPU-bound activation extraction)
# and step 3 (judge-API-bound response scoring) share no resources beyond
# step 1's response files, so they run concurrently by default.  Step 2's
# stdout is prefixed with "[step2] " and step 3's with "[step3] " so the
# interleaved output is still readable; if you prefer clean per-step logs
# (e.g. for debugging) pass --steps23serial.  The script waits for both to
# finish (and surfaces failures of either) before moving on to step 4.
#
# Pre-flight check:
#   Before any GPU work, the script checks every visible GPU and aborts if
#   any has more than ${GPU_BUSY_THRESHOLD_MIB} MiB already in use -- this
#   catches orphaned vLLM workers from previous crashed runs (which would
#   otherwise cause silent partial step-2 output via per-worker OOMs).  Pass
#   --skip-gpu-check to bypass, or --gpu-busy-threshold-mib N to raise the
#   threshold (e.g. for shared boxes with known co-tenants).
#
# Requirements:
#   - OPENAI_API_KEY environment variable (for step 3)
#   - Sufficient GPU memory for the model

set -e

# ---- Configuration (defaults) ---------------------------------------------
MODEL="Qwen/Qwen3-32B"
MODE="roger"                    # roger | christina
ROLES_DIR="../data/roles/instructions"  # christina: directory of instruction JSONs
GOAL_COUNT=30                   # roger mode: top-N from goal lists
NON_GOAL_COUNT=30               # roger mode: top-N from non-goal lists
REDUCE_QUESTIONS=3              # roger mode: take every Nth question (1=all, 3=every 3rd)
MIN_COUNT=30                    # roger mode: min score=3 samples for vector (50 for christina)
OUTPUT_DIR="/workspace/outputs/qwen-3-32b"
# Tensor-parallel size per worker for steps 1 & 2.  The pipeline runs
# total_gpus / tensor_parallel_size workers in parallel.  Default 1 fits a
# 32B bf16 model on one 80 GB GPU and gives near-linear scaling on 4-GPU
# boxes.  Set to 2 for 40-48 GB GPUs that need two-way model splitting.
TENSOR_PARALLEL_SIZE=1

# Use /dev/shm (tmpfs, RAM-backed) for two purposes:
#   1. As HF_HOME so model weights are loaded from RAM rather than NFS.  Some
#      RunPod network volumes have a flaky NFS layer where concurrent mmap
#      faults from multiple worker processes hang the kernel
#      (folio_wait_bit_common in D-state).  RAM-backed tmpfs sidesteps this.
#   2. As TMPDIR so step 2's per-role staging files (~2.6 GB each) don't blow
#      out a small container /tmp when 4 workers write concurrently.
# Set to false if you don't have /dev/shm, are on a RAM-constrained box, or
# already have an HF_HOME setup you want preserved.
USE_TMPFS=true
TMPFS_HF_HOME=/dev/shm/hf-cache
# Where step 2's atomic-write staging files land.  Default /dev/shm
# (RAM-backed tmpfs).  Override via --tmpdir.  Unconditional override
# of $TMPDIR by design: pod-defaults like /workspace/tmp silently
# route every staging write through slow NFS.  Callers who genuinely
# want to preserve their existing $TMPDIR must pass --tmpdir "$TMPDIR".
TMPFS_TMPDIR=/dev/shm

# GPU pre-flight check: refuse to run if any visible GPU has >GPU_BUSY_THRESHOLD_MIB
# of memory in use, since that strongly suggests an orphaned vLLM worker from a
# previous (crashed) run still holding a CUDA context.  Such orphans cause OOMs
# on individual workers and silent partial outputs (other workers proceed,
# 2_activations.py exits 0 even when some workers died, the pipeline marches on
# to step 3 with an incomplete activations directory).  See the discussion in
# AGENT_NOTES.md / README.md.  Pass --skip-gpu-check to bypass.
GPU_BUSY_THRESHOLD_MIB=2048

# Run Step 2 (GPU-bound activation extraction) and Step 3 (judge-API-bound
# response scoring) concurrently.  They share no resources -- step 2 uses the
# GPU and step 3 talks to OpenAI/Anthropic over the network -- and only depend
# on step 1's responses, which are fully written before step 2 begins.
# Wall-clock savings can be substantial when step 3 is hours of API calls.
# Disable with --steps23serial if you want the historical sequential behaviour
# (cleaner per-step logs; useful for debugging).
STEPS23_PARALLEL=true

# Run pipeline/scan_missing_vectors.py at the end of the pipeline to verify
# every activation produced a healthy vector.  Cheap when nothing went wrong
# (cache hits dominate -- a re-scan after a clean run is typically a few
# minutes), genuinely useful when something silently failed during step 2 or
# step 4 (corrupt-on-write activation, ZFS/NFS short-read, missing scores
# file, ...).  Disable with --no-scan if you want to inspect intermediate
# state by hand or are on a tight wall-clock budget.
RUN_SCAN=true

# ---- Parse command line ---------------------------------------------------
TYPES=""
SKIP_GPU_CHECK=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --roles_dir)
            ROLES_DIR="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --tensor_parallel_size)
            TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --reduce_questions)
            REDUCE_QUESTIONS="$2"; shift 2 ;;
        --no-tmpfs)
            USE_TMPFS=false; shift ;;
        --tmpfs-dir)
            TMPFS_HF_HOME="$2"; USE_TMPFS=true; shift 2 ;;
        --tmpdir)
            TMPFS_TMPDIR="$2"; shift 2 ;;
        --skip-gpu-check)
            SKIP_GPU_CHECK=true; shift ;;
        --gpu-busy-threshold-mib)
            GPU_BUSY_THRESHOLD_MIB="$2"; shift 2 ;;
        --steps23serial)
            STEPS23_PARALLEL=false; shift ;;
        --no-scan)
            RUN_SCAN=false; shift ;;
        --types)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TYPES="$TYPES $1"
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done
TYPES="${TYPES# }"  # trim leading space
if [ -z "$TYPES" ]; then
    TYPES="combinations roles traits"
fi

# Validate mode
case "$MODE" in
    roger|christina) ;;
    *) echo "Invalid mode: $MODE (must be roger or christina)" >&2; exit 1 ;;
esac

# Validate types (roger mode only).  r_combinations / t_combinations are
# subsets that share the combinations/ output directory.
if [ "$MODE" = "roger" ]; then
    for t in $TYPES; do
        case "$t" in
            combinations|r_combinations|t_combinations|roles|traits) ;;
            *) echo "Invalid type: $t (must be one of combinations, r_combinations, t_combinations, roles, traits)" >&2; exit 1 ;;
        esac
    done
fi

# Helper: map a type name (combinations | r_combinations | t_combinations |
# roles | traits) to the on-disk subdirectory under $OUTPUT_DIR.  The two
# combination subsets share the combinations/ tree.
type_subdir() {
    case "$1" in
        r_combinations|t_combinations) echo "combinations" ;;
        *) echo "$1" ;;
    esac
}

# ---- Logging --------------------------------------------------------------
LOG_FILE="$OUTPUT_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Assistant Axis Pipeline ==="
echo "Log: $LOG_FILE"
echo "Model:  $MODEL"
echo "Mode:   $MODE"
echo "Output: $OUTPUT_DIR"
echo "TP size:$TENSOR_PARALLEL_SIZE"
echo "Tmpfs:  $USE_TMPFS"
echo "Steps 2+3: $([ "$STEPS23_PARALLEL" = "true" ] && echo "parallel (GPU+judge concurrent)" || echo "serial (--steps23serial)")"
echo "Scan:   $([ "$RUN_SCAN" = "true" ] && echo "yes (post-pipeline audit; --no-scan to skip)" || echo "no (--no-scan)")"
if [ "$MODE" = "christina" ]; then
    echo "Roles:  $ROLES_DIR"
else
    echo "Types:  $TYPES"
fi
echo ""

# ---- GPU pre-flight check --------------------------------------------------
# Refuses to run if any visible GPU is already holding more than
# GPU_BUSY_THRESHOLD_MIB of memory.  The most common cause is an orphaned
# vLLM worker from a previous (crashed) run -- the parent died but the worker
# child got adopted by init and kept its CUDA context, so its weights remain
# resident.  When the new run's worker tries to load the model on that same
# GPU, it OOMs partway through; the multi-worker pool joins all children and
# exits 0 even when some died, so the pipeline silently produces a partial
# step-2 output and the failure isn't noticed until step 4 (or later).
#
# This check runs *after* logging is set up so the diagnostic lands in the
# pipeline log, but before any GPU work starts.  Bypass with --skip-gpu-check.
gpu_preflight_check() {
    if [ "$SKIP_GPU_CHECK" = "true" ]; then
        echo "[gpu-check] skipped (--skip-gpu-check)"
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gpu-check] nvidia-smi not found; skipping"
        return
    fi

    # Restrict the check to GPUs the pipeline will actually use.
    local visible_arg=""
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        visible_arg="--id=$CUDA_VISIBLE_DEVICES"
        echo "[gpu-check] checking GPUs $CUDA_VISIBLE_DEVICES (CUDA_VISIBLE_DEVICES) for >${GPU_BUSY_THRESHOLD_MIB} MiB"
    else
        echo "[gpu-check] checking all visible GPUs for >${GPU_BUSY_THRESHOLD_MIB} MiB"
    fi

    # query: index, name, memory used in MiB.  Output is CSV without units/headers.
    local query
    query=$(nvidia-smi $visible_arg \
        --query-gpu=index,name,memory.used \
        --format=csv,noheader,nounits 2>/dev/null) || true
    if [ -z "$query" ]; then
        echo "[gpu-check] nvidia-smi returned no data; skipping"
        return
    fi

    local busy_lines=""
    while IFS=, read -r idx name used; do
        idx=$(echo "$idx" | xargs)       # trim whitespace
        used=$(echo "$used" | xargs)
        if [ -z "$used" ]; then continue; fi
        if [ "$used" -gt "$GPU_BUSY_THRESHOLD_MIB" ]; then
            busy_lines+="  GPU $idx ($name): ${used} MiB used"$'\n'
        fi
    done <<< "$query"

    if [ -n "$busy_lines" ]; then
        echo ""
        echo "[gpu-check] ERROR: one or more GPUs are already busy."
        echo "$busy_lines"
        echo "Holders (from nvidia-smi --query-compute-apps):"
        nvidia-smi $visible_arg \
            --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv 2>/dev/null \
            | sed 's/^/  /' || true
        echo ""
        echo "This usually means a previous pipeline crashed leaving an orphaned"
        echo "vLLM worker holding a CUDA context.  To clean up:"
        echo ""
        echo "  pkill -f 'assistant-axis/.venv/bin/python3'"
        echo "  sleep 3"
        echo "  nvidia-smi   # all GPUs should now show 0 MiB"
        echo ""
        echo "If a PID still appears in nvidia-smi but ps shows no such process,"
        echo "the driver has zombie memory.  Try:"
        echo ""
        echo "  fuser -k /dev/nvidia*"
        echo "  # or as last resort, restart the container"
        echo ""
        echo "If you know the existing GPU usage is fine (e.g. a co-tenant on"
        echo "a shared box), pass --skip-gpu-check to bypass this check, or"
        echo "set CUDA_VISIBLE_DEVICES to limit the pipeline to free GPUs."
        echo "You can also raise the threshold with --gpu-busy-threshold-mib N."
        exit 1
    fi
    echo "[gpu-check] all visible GPUs are clean"
}
gpu_preflight_check
echo ""

# ---- Tmpfs setup -----------------------------------------------------------
# Mirror the HF cache for $MODEL into /dev/shm so all workers load weights
# from RAM-backed tmpfs (avoids flaky NFS mmap behavior) and redirect TMPDIR
# similarly so step 2's ~2.6 GB per-role staging files don't fill a small
# container /tmp.  Idempotent: re-uses an existing tmpfs cache, skips the
# copy if disk space is short, and is fully bypassed by --no-tmpfs.
setup_tmpfs() {
    if [ "$USE_TMPFS" != "true" ]; then
        echo "[tmpfs] disabled (--no-tmpfs); using system HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} and TMPDIR=${TMPDIR:-/tmp}"
        return
    fi
    if [ ! -d /dev/shm ]; then
        echo "[tmpfs] /dev/shm not present; tmpfs setup skipped"
        return
    fi

    # Redirect TMPDIR to $TMPFS_TMPDIR (default /dev/shm) unconditionally.
    # Step 2 (2_activations.py) honours TMPDIR for its staging files; if
    # the pod sets a default like TMPDIR=/workspace/tmp (NFS-backed), every
    # 2.6 GB staging write would route through slow network storage.
    # Unconditional override avoids that silent footgun.  Callers who want
    # to preserve their existing $TMPDIR must pass --tmpdir "$TMPDIR".
    if [ ! -d "$TMPFS_TMPDIR" ]; then
        echo "[tmpfs] WARNING: --tmpdir target '$TMPFS_TMPDIR' not present; falling back to /tmp"
        export TMPDIR=/tmp
    elif [ -n "$TMPDIR" ] && [ "$TMPDIR" != "$TMPFS_TMPDIR" ]; then
        echo "[tmpfs] TMPDIR=$TMPFS_TMPDIR (overriding prior $TMPDIR; pass --tmpdir '$TMPDIR' to preserve)"
        export TMPDIR="$TMPFS_TMPDIR"
    else
        export TMPDIR="$TMPFS_TMPDIR"
        echo "[tmpfs] TMPDIR=$TMPDIR (avoid filling small container /tmp / NFS staging)"
    fi

    # Triton / torchinductor compile CUDA kernels into .so files at runtime
    # then dlopen() them.  RunPod (and most hardened containers) mount /dev/shm
    # noexec, so dlopen on a .so under /dev/shm/torchinductor_root/... fails
    # with "ImportError: ... failed to map segment from shared object".  Route
    # the compile cache to ~/.cache/{triton,torchinductor} (regular FS, exec
    # allowed) unless the user has overridden them.  Bonus: persists across
    # container restarts so the ~50 s torch.compile pass is amortised.
    local cache_root="${XDG_CACHE_HOME:-$HOME/.cache}"
    if [ -z "$TRITON_CACHE_DIR" ]; then
        export TRITON_CACHE_DIR="$cache_root/triton"
        mkdir -p "$TRITON_CACHE_DIR"
        echo "[tmpfs] TRITON_CACHE_DIR=$TRITON_CACHE_DIR (off /dev/shm noexec)"
    fi
    if [ -z "$TORCHINDUCTOR_CACHE_DIR" ]; then
        export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
        mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
        echo "[tmpfs] TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR (off /dev/shm noexec)"
    fi

    # Mirror HF cache for the requested $MODEL.  HF caches under
    # <HF_HOME>/hub/models--<owner>--<name>/.  We copy just the requested
    # model's subtree to keep the tmpfs footprint minimal.
    local source_hf="${HF_HOME:-$HOME/.cache/huggingface}"
    local model_subdir="models--${MODEL//\//--}"
    local source_model_dir="$source_hf/hub/$model_subdir"
    local tmpfs_model_dir="$TMPFS_HF_HOME/hub/$model_subdir"

    if [ ! -d "$source_model_dir" ]; then
        echo "[tmpfs] $MODEL not found at $source_model_dir; HF will download to $source_hf on first use (no tmpfs cache pre-populated)"
        return
    fi

    if [ -d "$tmpfs_model_dir" ]; then
        echo "[tmpfs] $MODEL already present at $tmpfs_model_dir; reusing"
        export HF_HOME="$TMPFS_HF_HOME"
        echo "[tmpfs] HF_HOME=$HF_HOME"
        return
    fi

    # Disk-space check: require 1.5× model size headroom in /dev/shm.
    # `cp -r` is sequential so its own peak usage is ~1× + a small buffer,
    # but TMPDIR=/dev/shm too and step 2 puts ~2.6 GB per-role staging
    # files there during extraction, so we want some slack on top of 1×.
    # 2× was overly conservative and caused silent fallbacks to NFS that
    # resulted in multi-hour cold mmap reads of model weights.
    local needed_kb
    needed_kb=$(du -sk "$source_model_dir" 2>/dev/null | awk '{print int($1 * 3 / 2)}')
    local avail_kb
    avail_kb=$(df --output=avail /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
    if [ -z "$needed_kb" ] || [ -z "$avail_kb" ] || [ "$avail_kb" -lt "$needed_kb" ]; then
        echo ""
        echo "[tmpfs] ============================================================"
        echo "[tmpfs]  WARNING: insufficient /dev/shm space — tmpfs cache DISABLED"
        echo "[tmpfs]  need ${needed_kb}kB, have ${avail_kb}kB"
        echo "[tmpfs]  Falling back to source HF cache: $source_hf"
        echo "[tmpfs]  If that is on NFS or other slow storage, expect step 2 to"
        echo "[tmpfs]  hang for HOURS on cold mmap reads of model weights."
        echo "[tmpfs]  Fix: increase --shm-size on the container, or free /dev/shm."
        echo "[tmpfs] ============================================================"
        echo ""
        return
    fi

    echo "[tmpfs] copying $MODEL ($((needed_kb / 2 / 1024)) MB) from $source_model_dir to $tmpfs_model_dir ..."
    mkdir -p "$TMPFS_HF_HOME/hub"
    if cp -r "$source_model_dir" "$TMPFS_HF_HOME/hub/"; then
        export HF_HOME="$TMPFS_HF_HOME"
        echo "[tmpfs] copy complete; HF_HOME=$HF_HOME"
    else
        echo "[tmpfs] copy failed; falling back to $source_hf"
    fi
}

setup_tmpfs
echo ""

# ---- Helper: symlink default files into a type's subdir -------------------
# Usage: symlink_default <subdir_name> <file_extension>
# e.g.  symlink_default responses jsonl
symlink_default() {
    local subdir="$1" ext="$2"
    local src="$OUTPUT_DIR/default/$subdir/default.$ext"
    local dst_dir="$OUTPUT_DIR/$type/$subdir"
    local rel="../../default/$subdir/default.$ext"
    mkdir -p "$dst_dir"
    if [ -f "$src" ] && [ ! -e "$dst_dir/default.$ext" ]; then
        ln -s "$rel" "$dst_dir/default.$ext"
        echo "  Symlinked default.$ext -> $type/$subdir/"
    fi
}

# ---- Christina mode -------------------------------------------------------
# NOTE: --reduce_questions is now forwarded to steps 1 and 4 (previously
# silently ignored in this branch, so christina-mode runs always used
# 1_generate.py's own default of reduce_questions=1 / question_count=240,
# regardless of the $REDUCE_QUESTIONS config above -- this matches how
# roger mode already threads $REDUCE_QUESTIONS through steps 1 and 4).
if [ "$MODE" = "christina" ]; then

    echo "=== Step 1: Generating responses ==="
    uv run 1_generate.py \
        --mode "$MODE" \
        --model "$MODEL" \
        --roles_dir "$ROLES_DIR" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --reduce_questions "$REDUCE_QUESTIONS" \
        --output_dir "$OUTPUT_DIR/responses"

    echo ""

    # Infer entity_type for step 3 from ROLES_DIR so the 9 collision names
    # (ascetic, contrarian, ... stoic) are scored against the right prompt.
    # Christina mode is standalone-only (no combinations).
    case "$ROLES_DIR" in
        *traits*) CHR_ENT_TYPE=trait ;;
        *roles*)  CHR_ENT_TYPE=role ;;
        *) echo "ERROR: cannot infer --entity_type from ROLES_DIR=$ROLES_DIR"; exit 1 ;;
    esac

    # -- Step 2 + Step 3 (concurrent by default; serial with --steps23serial)
    christina_step2() {
        set -e
        set -o pipefail
        echo "=== Step 2: Extracting activations ==="
        uv run 2_activations.py \
            --model "$MODEL" \
            --responses_dir "$OUTPUT_DIR/responses" \
            --output_dir "$OUTPUT_DIR/activations" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            --batch_size 8
    }
    christina_step3() {
        set -e
        set -o pipefail
        echo "=== Step 3: Scoring responses ==="
        uv run 3_judge.py \
            --entity_type "$CHR_ENT_TYPE" \
            --responses_dir "$OUTPUT_DIR/responses" \
            --output_dir "$OUTPUT_DIR/scores"
    }

    if [ "$STEPS23_PARALLEL" = "true" ]; then
        echo "Running Step 2 (GPU) and Step 3 (judge API) concurrently."
        echo "Disable with --steps23serial.  stdout is prefixed with [step2]/[step3]."
        echo ""
        ( christina_step2 2>&1 | sed -u 's/^/[step2] /' ) &
        PID2=$!
        ( christina_step3 2>&1 | sed -u 's/^/[step3] /' ) &
        PID3=$!
        fail=0
        wait $PID2 || { echo "ERROR: Step 2 failed (exit $?)"; fail=1; }
        wait $PID3 || { echo "ERROR: Step 3 failed (exit $?)"; fail=1; }
        if [ "$fail" -ne 0 ]; then
            echo "=== Pipeline aborted: parallel step 2/3 had failures ==="
            exit 1
        fi
    else
        christina_step2
        echo ""
        christina_step3
    fi
    echo ""

    echo "=== Step 4: Computing vectors ==="
    uv run 4_vectors.py \
        --activations_dir "$OUTPUT_DIR/activations" \
        --scores_dir "$OUTPUT_DIR/scores" \
        --output_dir "$OUTPUT_DIR/vectors" \
        --reduce_questions "$REDUCE_QUESTIONS"

    echo ""
    echo "=== Step 5: Computing axis ==="
    uv run 5_axis.py \
        --vectors_dir "$OUTPUT_DIR/vectors" \
        --output "$OUTPUT_DIR/axis.pt"

    echo ""
    echo "=== Pipeline complete ==="
    echo "Axis saved to: $OUTPUT_DIR/axis.pt"

# ---- Roger mode (types-aware) ---------------------------------------------
else

    # -- Step 0: Generate, extract, and vectorize default (once) ---------------
    echo "=== Step 0: Default role ==="
    DEFAULT_DIR="$OUTPUT_DIR/default"
    DEFAULT_RESP="$DEFAULT_DIR/responses"
    DEFAULT_ACT="$DEFAULT_DIR/activations"
    DEFAULT_VEC="$DEFAULT_DIR/vectors"
    mkdir -p "$DEFAULT_RESP" "$DEFAULT_ACT" "$DEFAULT_DIR/scores" "$DEFAULT_VEC"

    if [ ! -f "$DEFAULT_RESP/default.jsonl" ]; then
        echo "  Generating default responses..."
        uv run 1_generate.py \
            --mode roger \
            --model "$MODEL" \
            --roles_only \
            --roles default \
            --reduce_questions "$REDUCE_QUESTIONS" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            --output_dir "$DEFAULT_RESP"
    else
        echo "  default.jsonl already exists, skipping generation."
    fi

    if [ ! -f "$DEFAULT_ACT/default.pt" ]; then
        echo "  Extracting default activations..."
        uv run 2_activations.py \
            --model "$MODEL" \
            --responses_dir "$DEFAULT_RESP" \
            --output_dir "$DEFAULT_ACT" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            --batch_size 8
    else
        echo "  default.pt already exists, skipping extraction."
    fi

    if [ ! -f "$DEFAULT_VEC/default.pt" ]; then
        echo "  Computing default vector..."
        uv run 4_vectors.py \
            --activations_dir "$DEFAULT_ACT" \
            --scores_dir "$DEFAULT_DIR/scores" \
            --output_dir "$DEFAULT_VEC"
    else
        echo "  default vector already exists, skipping."
    fi
    echo ""

    # -- Step 1: Generate responses ------------------------------------------
    echo "=== Step 1: Generating responses ==="
    for type in $TYPES; do
        echo "--- Step 1 [$type] ---"
        type_dir_name=$(type_subdir "$type")
        TYPE_DIR="$OUTPUT_DIR/$type_dir_name"
        # symlink_default reads the local variable $type, but uses it only as
        # the destination prefix; for the combo subsets we want the symlink
        # placed under combinations/ so override $type for the helper call.
        type_for_symlink="$type"; type="$type_dir_name"
        symlink_default responses jsonl
        type="$type_for_symlink"

        case "$type" in
            combinations|r_combinations|t_combinations)
                # Subset flag (empty for full combinations; --combos_only_r /
                # --combos_only_t for the half-grid subsets).  Skip already-
                # generated halves are a no-op thanks to per-file skip logic.
                subset_flag=""
                case "$type" in
                    r_combinations) subset_flag="--combos_only_r" ;;
                    t_combinations) subset_flag="--combos_only_t" ;;
                esac
                uv run 1_generate.py \
                    --mode roger \
                    --model "$MODEL" \
                    --goal_count "$GOAL_COUNT" \
                    --non_goal_count "$NON_GOAL_COUNT" \
                    --reduce_questions "$REDUCE_QUESTIONS" \
                    --no_traits $subset_flag \
                    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
                    --output_dir "$TYPE_DIR/responses"
                ;;
            roles)
                uv run 1_generate.py \
                    --mode roger \
                    --model "$MODEL" \
                    --reduce_questions "$REDUCE_QUESTIONS" \
                    --roles_only \
                    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
                    --output_dir "$TYPE_DIR/responses"
                ;;
            traits)
                uv run 1_generate.py \
                    --mode roger \
                    --model "$MODEL" \
                    --reduce_questions "$REDUCE_QUESTIONS" \
                    --traits_only \
                    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
                    --output_dir "$TYPE_DIR/responses"
                ;;
        esac
    done
    echo ""

    # Step 2 honours the r_/t_ subset distinction so disk-limited runs can
    # extract only one half of the combination grid at a time (each .pt is
    # ~2.6 GB; 1800 combos = ~4.7 TB).  We iterate $TYPES rather than the
    # deduped list, passing --name_prefix r_ / t_ to filter response files;
    # per-file skip-if-exists handles any redundant listings.
    #
    # Steps 3-5 only need the deduped list (their work is per-response or
    # per-vector and inexpensive enough to do all at once).
    DEDUP_TYPES=""
    for type in $TYPES; do
        d=$(type_subdir "$type")
        case " $DEDUP_TYPES " in
            *" $d "*) ;;
            *) DEDUP_TYPES="$DEDUP_TYPES $d" ;;
        esac
    done
    DEDUP_TYPES="${DEDUP_TYPES# }"

    # -- Step 2 + Step 3 (concurrent by default; serial with --steps23serial) ----
    # Step 2 is GPU-bound (model forward pass + hidden-state hooks); step 3 is
    # judge-API-bound (network calls).  They share no resources beyond step 1's
    # response files (which are fully written by now), so we can run them in
    # parallel for substantial wall-clock savings.  Each runs in its own
    # subshell with stdout/stderr prefixed for readability.

    run_step2_loop() {
        set -e
        set -o pipefail
        echo "=== Step 2: Extracting activations ==="
        for type in $TYPES; do
            echo "--- Step 2 [$type] ---"
            type_dir_name=$(type_subdir "$type")
            TYPE_DIR="$OUTPUT_DIR/$type_dir_name"
            type_for_symlink="$type"; type="$type_dir_name"
            symlink_default activations pt
            type="$type_for_symlink"

            prefix_flag=""
            case "$type" in
                r_combinations) prefix_flag="--name_prefix r_" ;;
                t_combinations) prefix_flag="--name_prefix t_" ;;
            esac

            uv run 2_activations.py \
                --model "$MODEL" \
                --responses_dir "$TYPE_DIR/responses" \
                --output_dir "$TYPE_DIR/activations" \
                --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
                --batch_size 8 $prefix_flag
        done
    }

    run_step3_loop() {
        set -e
        set -o pipefail
        echo "=== Step 3: Scoring responses ==="
        for type in $DEDUP_TYPES; do
            echo "--- Step 3 [$type] ---"
            TYPE_DIR="$OUTPUT_DIR/$type"

            # entity_type disambiguates 9 names that exist in both data/roles and
            # data/traits, and tells the judge when to use the combined-eval prompt.
            case "$type" in
                roles)        ENT_TYPE=role ;;
                traits)       ENT_TYPE=trait ;;
                combinations) ENT_TYPE=combination ;;
            esac

            uv run 3_judge.py \
                --entity_type "$ENT_TYPE" \
                --responses_dir "$TYPE_DIR/responses" \
                --output_dir "$TYPE_DIR/scores"
        done
    }

    if [ "$STEPS23_PARALLEL" = "true" ]; then
        echo "Running Step 2 (GPU) and Step 3 (judge API) concurrently."
        echo "Disable with --steps23serial.  stdout is prefixed with [step2]/[step3]."
        echo ""
        ( run_step2_loop 2>&1 | sed -u 's/^/[step2] /' ) &
        PID2=$!
        ( run_step3_loop 2>&1 | sed -u 's/^/[step3] /' ) &
        PID3=$!
        fail=0
        wait $PID2 || { echo "ERROR: Step 2 failed (exit $?)"; fail=1; }
        wait $PID3 || { echo "ERROR: Step 3 failed (exit $?)"; fail=1; }
        if [ "$fail" -ne 0 ]; then
            echo "=== Pipeline aborted: parallel step 2/3 had failures ==="
            exit 1
        fi
    else
        run_step2_loop
        echo ""
        run_step3_loop
    fi
    echo ""

    # -- Step 4: Compute vectors ----------------------------------------------
    echo "=== Step 4: Computing vectors ==="
    for type in $DEDUP_TYPES; do
        echo "--- Step 4 [$type] ---"
        TYPE_DIR="$OUTPUT_DIR/$type"
        symlink_default vectors pt

        uv run 4_vectors.py \
            --activations_dir "$TYPE_DIR/activations" \
            --scores_dir "$TYPE_DIR/scores" \
            --output_dir "$TYPE_DIR/vectors" \
            --min_count "$MIN_COUNT"
    done
    echo ""

    # -- Step 5: Compute axis -------------------------------------------------
    echo "=== Step 5: Computing axis ==="
    for type in $DEDUP_TYPES; do
        echo "--- Step 5 [$type] ---"
        TYPE_DIR="$OUTPUT_DIR/$type"

        uv run 5_axis.py \
            --vectors_dir "$TYPE_DIR/vectors" \
            --output "$TYPE_DIR/axis.pt"
    done
    echo ""

    echo "=== Pipeline complete ==="
    for type in $DEDUP_TYPES; do
        echo "  $type axis: $OUTPUT_DIR/$type/axis.pt"
    done

fi

# ---- Post-pipeline scan ---------------------------------------------------
# Walks every activation/vector pair under $OUTPUT_DIR and classifies any
# misses or corruptions (corrupt-on-write activation .pt, missing scores,
# below-min-count, etc).  The scanner caches its torch.load outcomes at
# $OUTPUT_DIR/scan_cache.json so this is a fast no-op on the second run
# (cache hits skip the expensive deep_load).  Output:
#   - $OUTPUT_DIR/<type>/vectors/missing_vectors_audit.json (per-target)
#   - $OUTPUT_DIR/<type>/vectors/missing_vectors_rerun.txt (re-run candidates)
# A non-zero exit from the scanner is logged but does not fail the pipeline:
# the pipeline itself has already succeeded by this point, the scan is just
# a verification pass.  Skip with --no-scan.
if [ "$RUN_SCAN" = "true" ]; then
    echo ""
    echo "=== Post-pipeline scan ==="
    case "$MODE" in
        roger)
            # --root mode: scanner auto-discovers entity-type subdirs and
            # every vectors* sibling under $OUTPUT_DIR.  Use the same
            # MIN_COUNT the pipeline used so below_min_count classifications
            # match what step 4 would actually skip.
            uv run scan_missing_vectors.py \
                --root        "$OUTPUT_DIR" \
                --min_count   "$MIN_COUNT" \
                --deep_load \
                || echo "[scan] WARNING: scanner exited non-zero (pipeline itself succeeded; rerun the scan manually for details)"
            ;;
        christina)
            # Single-pair mode: christina mode writes one entity-type tree
            # directly under $OUTPUT_DIR, no entity-type subdir layer.
            uv run scan_missing_vectors.py \
                --activations_dir "$OUTPUT_DIR/activations" \
                --vectors_dir     "$OUTPUT_DIR/vectors" \
                --scores_dir      "$OUTPUT_DIR/scores" \
                --min_count       "$MIN_COUNT" \
                --deep_load \
                || echo "[scan] WARNING: scanner exited non-zero (pipeline itself succeeded; rerun the scan manually for details)"
            ;;
    esac
fi

echo "Log:  $LOG_FILE"

# ---- Summary of warnings/errors ------------------------------------------
ISSUES=$(grep -E ' - (WARNING|ERROR) - ' "$LOG_FILE" 2>/dev/null || true)
if [ -n "$ISSUES" ]; then
    COUNT=$(echo "$ISSUES" | wc -l | tr -d ' ')
    echo ""
    echo "*** $COUNT warning(s)/error(s) during pipeline run: ***"
    echo "$ISSUES"
fi
