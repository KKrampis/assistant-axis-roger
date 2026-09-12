#!/bin/bash
#
# One-time setup script to run ON a fresh RunPod pod before the pipeline.
# Idempotent: safe to re-run (e.g. after a pod restart) without redoing work.
#
# Usage (from an SSH session on the pod):
#   export OPENAI_API_KEY=sk-...
#   bash runpod/bootstrap.sh
#
# Or pass it as the pod's startup command (see runpod/launch_pod.py).

set -euo pipefail

REPO_URL="https://github.com/KKrampis/assistant-axis-roger.git"
BRANCH="anthropic-vllm-uv"
REPO_DIR="/workspace/assistant-axis-roger"

# --- Persist caches on the network volume, not container disk -------------
# Mirrors runpod_workspace/qwen/.env_pod's existing convention: weights and
# uv's package cache should survive a pod restart/re-provision, and shouldn't
# fill up the small container disk (containerDiskInGb) with a 64GB+ model.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv-cache}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. pipeline/3_judge.py requires it" >&2
    echo "(default --judge_model is gpt-4.1-mini). Set it and re-run." >&2
    exit 1
fi

# --- Install uv -------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Clone or update the repo -----------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning $REPO_URL ($BRANCH)..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo "Repo already present, updating..."
    git -C "$REPO_DIR" fetch origin "$BRANCH"
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
fi

cd "$REPO_DIR"

# --- Install dependencies ---------------------------------------------------
# pyproject.toml marks vllm as `sys_platform == 'linux'`-only, so this
# correctly pulls vLLM here (it would be skipped on macOS).
echo "Running uv sync (this downloads torch/vllm/etc, can take several minutes)..."
uv sync

# --- Sanity checks -----------------------------------------------------------
echo ""
echo "=== GPU check ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || {
    echo "WARNING: nvidia-smi not found or no GPU visible." >&2
}

echo ""
echo "Bootstrap complete."
echo "Repo:   $REPO_DIR (branch $BRANCH)"
echo "HF_HOME:       $HF_HOME"
echo "UV_CACHE_DIR:  $UV_CACHE_DIR"
echo ""
echo "Next, from $REPO_DIR/pipeline:"
echo "  ./run_pipeline.sh --mode christina --reduce_questions 3 \\"
echo "      --model Qwen/Qwen3-32B \\"
echo "      --tensor_parallel_size 1 \\"
echo "      --roles_dir ../data/traits/instructions/_moral_circle \\"
echo "      --output_dir /workspace/outputs/qwen-3-32b/moral-circle"
