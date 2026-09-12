# Running the pipeline on RunPod

Local development (Mac or otherwise) can drive stages 3–5 of the Assistant
Axis pipeline (judging, vectors, axis/PCA — all CPU-only), but stages 1–2
(response generation via vLLM, activation extraction) need a real NVIDIA
GPU and enough VRAM for the target model. `pyproject.toml` already marks
`vllm` as `sys_platform == 'linux'`-only, so `uv sync` correctly skips it
on macOS and installs it on the pod.

This directory has the scripts to provision that GPU pod, set it up, and
tear it down again.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `launch_pod.py` | Your machine | Creates the RunPod GPU pod via RunPod's REST API, polls until it's running |
| `bootstrap.sh` | On the pod | One-time setup: installs `uv`, clones this repo, `uv sync`, sanity-checks the GPU |
| `stop_pod.py` | Your machine | Stops (pause, keeps volume) or terminates (deletes everything) the pod |

## Requirements

- `RUNPOD_API_KEY` — your RunPod account API key
- `OPENAI_API_KEY` — forwarded into the pod's environment; `pipeline/3_judge.py` needs it for step 3 (default judge model: `gpt-4.1-mini`)

Both are read from your shell environment. Never hardcode them into these
scripts or commit them.

## Workflow

```bash
export RUNPOD_API_KEY=...
export OPENAI_API_KEY=...

# 1. Create the pod (defaults: 1x 80GB GPU — A100 80GB or H100, whichever
#    RunPod has available — 200GB persistent volume at /workspace).
python3 runpod/launch_pod.py
# Prints the pod id and connection details once it's RUNNING.

# 2. SSH in, then bootstrap the environment.
bash runpod/bootstrap.sh

# 3. Run whatever pipeline batch you need. Example — the moral-circle
#    trait batch (see data/traits/instructions/_moral_circle/README.md):
cd assistant-axis-roger/pipeline
./run_pipeline.sh --mode christina --reduce_questions 3 \
    --model Qwen/Qwen3-32B \
    --tensor_parallel_size 1 \
    --roles_dir ../data/traits/instructions/_moral_circle \
    --output_dir /workspace/outputs/qwen-3-32b/moral-circle

# 4. Pull results back to your machine (small files only — vectors and
#    the final axis; NOT the raw responses/activations, which are large
#    and already gitignored under outputs/).
rsync -avz pod:/workspace/outputs/qwen-3-32b/moral-circle/vectors ./local-results/
rsync -avz pod:/workspace/outputs/qwen-3-32b/moral-circle/axis.pt ./local-results/

# 5. Stop billing as soon as you're done.
python3 runpod/stop_pod.py <pod_id>              # pause, volume persists
python3 runpod/stop_pod.py <pod_id> --terminate  # delete everything
```

## Cost awareness

`AGENT_NOTES.md`'s hard rule: **stop and confirm parameters before any run
that could exceed $20 in judging/API spend or 13 GPU-hours of compute.**
This applies here. Concretely:

- `launch_pod.py` prints `costPerHr` while polling — note it, and watch
  wall-clock time once `run_pipeline.sh` starts.
- Judge cost scales with how many responses step 3 scores. `--reduce_questions N`
  (forwarded to steps 1 and 4 as of this branch's `run_pipeline.sh`) cuts
  generation volume by roughly `N`x — e.g. `--reduce_questions 3` on the
  12-trait moral-circle batch is ~4,800 generations instead of the full
  14,400 at default settings.
- Re-running `run_pipeline.sh` after a partial failure is safe — every
  stage skips files that already exist, so you only pay for what's missing.
- **Terminate, don't just stop**, once you've pulled results off and don't
  expect to resume — a stopped pod's persistent volume still costs storage.

## Notes on the pod config

- **GPU choice**: `launch_pod.py` requests `gpuTypeIds` as a priority list
  (A100 80GB PCIe → A100-SXM4-80GB → H100 80GB HBM3) and RunPod allocates
  whichever has capacity. Qwen3-32B in bf16 is ~64GB of weights alone, so
  don't go below an 80GB card without also changing `--tensor_parallel_size`
  and splitting across multiple smaller GPUs.
- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` —
  the "devel" (not "runtime") variant is used deliberately since vLLM's
  install step can need `nvcc`/CUDA headers.
- The REST API contract used here (`POST /pods`, `POST /pods/{id}/stop`,
  `DELETE /pods/{id}`, bearer-token auth) was verified against RunPod's
  live OpenAPI spec at `https://rest.runpod.io/v1/openapi.json` when these
  scripts were written. If pod creation starts failing, check that spec
  again before assuming the scripts are wrong — RunPod's available images
  and GPU inventory change over time.
