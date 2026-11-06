# Running the pipeline on RunPod

Local development (Mac or otherwise) can drive stages 3–5 of the Assistant
Axis pipeline (judging, vectors, axis/PCA — all CPU-only), but stages 1–2
(response generation via vLLM, activation extraction) need a real NVIDIA
GPU and enough VRAM for the target model. `pyproject.toml` already marks
`vllm` as `sys_platform == 'linux'`-only, so `uv sync` correctly skips it
on macOS and installs it on the pod.

This directory has the scripts to provision that GPU pod, set it up, run
the pipeline on it, and tear it down again — automatically, where possible.

## Which branch is this?

**`master` is this repo's default branch on GitHub** — a fresh `git clone`
checks it out automatically. The RunPod/subset-run/multi-judge work
described here lives on **`personas-konstantinos`**, a feature branch, not
the default. `bootstrap.sh` clones `personas-konstantinos` explicitly
(overridable — see below) precisely because the default branch alone
wouldn't have any of this.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `launch_pod.py` | Your machine | Creates the RunPod GPU pod via RunPod's REST API, refuses to launch a second one if one's already running, polls until it's up, prints the SSH command |
| `bootstrap.sh` | On the pod | One-time setup: installs `uv`, clones this repo, `uv sync`, configures push access if given a token, sanity-checks the GPU |
| `run_and_shutdown.sh` | On the pod | Wraps `pipeline/run_pipeline.sh` and shuts the pod down automatically once it finishes |
| `push_results.sh` | On the pod | Copies a run's small result files into the repo and pushes them, from the pod itself |
| `stop_pod.py` | Your machine | Stops (pause, keeps volume) or terminates (deletes everything) the pod, manually |

## Requirements

- `RUNPOD_API_KEY` — your RunPod account API key (your machine only — never
  copied onto the pod; the pod gets its own separate, auto-injected,
  pod-scoped key, see [Auto-shutdown](#auto-shutdown-after-the-run-finishes))
- `OPENAI_API_KEY` — forwarded into the pod's environment; `pipeline/3_judge.py`
  needs it to judge with the default `gpt-4.1-mini`
- `ANTHROPIC_API_KEY` — optional, forwarded if set locally; only needed if
  you're judging with a `claude-*` model (see `pipeline/SUBSET_RUNS.md`)
- `GH_PUSH_TOKEN` — optional, forwarded if set locally; only needed if you
  want the pod to push results itself (see
  [Pushing results from the pod](#pushing-results-from-the-pod))

All of these are read from your shell environment. Never hardcode them into
these scripts or commit them.

## The full local → pod → GitHub flow

```
Your machine                    RunPod pod                         GitHub
─────────────                   ──────────                         ──────
launch_pod.py  ──creates pod──▶
                                 bootstrap.sh
                                 (clone, uv sync,      ◀──git clone── repo
                                  configure push)
                                 run_and_shutdown.sh
                                 (runs run_pipeline.sh,
                                  uses the GPU, then
                                  shuts itself down)
                                 push_results.sh   ────git push─────▶ repo
                                 (optional, needs
                                  GH_PUSH_TOKEN)
rsync results  ◀──pulls back (if you didn't use push_results.sh)────
stop_pod.py    ──stop/terminate (if the pod didn't already
                 self-shutdown, or you used --keep-alive-on-failure)
```

**1. Provision the pod, from your machine:**

```bash
export RUNPOD_API_KEY=...
export OPENAI_API_KEY=...
python3 runpod/launch_pod.py
```

Creates a 1x 80GB-GPU pod (A100 80GB or H100, whichever RunPod has
capacity for) with a 200GB persistent volume at `/workspace`, polls every
10s, and once it's `RUNNING` prints the actual SSH command:

```
SSH in with:
  ssh root@<ip> -p <port>
```

**This refuses to launch if a pod from this project is already running**
(see [One pod at a time](#one-pod-at-a-time)) — no accidentally paying for
two.

**2. SSH in, then bootstrap:**

```bash
ssh root@<ip> -p <port>
export OPENAI_API_KEY=...
bash runpod/bootstrap.sh
```

Installs `uv`, clones the repo at `personas-konstantinos` into
`/workspace/assistant-axis-roger`, points `HF_HOME`/`UV_CACHE_DIR` at the
persistent volume, `uv sync`s (this is where `vllm` actually installs,
being the first Linux machine involved), and — if `GH_PUSH_TOKEN` was set —
configures the git remote for pushing.

**3. Run the pipeline, with automatic shutdown when it's done:**

```bash
cd assistant-axis-roger
bash runpod/run_and_shutdown.sh --mode christina --reduce_questions 3 \
    --model Qwen/Qwen3-32B \
    --tensor_parallel_size 1 \
    --roles_dir ../data/traits/instructions/_moral_circle \
    --output_dir /workspace/outputs/qwen-3-32b/moral-circle
```

Every flag is forwarded to `pipeline/run_pipeline.sh` unchanged (see
`pipeline/SUBSET_RUNS.md` for scoping which roles/traits actually run).
When it exits — success or failure — the pod stops itself. See
[Auto-shutdown](#auto-shutdown-after-the-run-finishes) for the flags that
change this behavior.

**4. Get your results out**, either automatically or manually:

```bash
# Automatic (from the pod, needs GH_PUSH_TOKEN configured in step 2):
bash runpod/push_results.sh \
    --src /workspace/outputs/qwen-3-32b/moral-circle \
    --dest results/moral-circle-qwen-3-32b

# Manual (from your machine, any time before the pod is terminated):
rsync -avz -e "ssh -p <port>" root@<ip>:/workspace/outputs/qwen-3-32b/moral-circle/vectors ./local-results/
rsync -avz -e "ssh -p <port>" root@<ip>:/workspace/outputs/qwen-3-32b/moral-circle/axis.pt ./local-results/
```

**5. Clean up, if the pod didn't already self-terminate:**

```bash
python3 runpod/stop_pod.py <pod_id>              # pause, volume persists
python3 runpod/stop_pod.py <pod_id> --terminate  # delete everything
```

## Auto-shutdown after the run finishes

`run_and_shutdown.sh` uses `RUNPOD_POD_ID` and `RUNPOD_API_KEY` —
environment variables **RunPod automatically injects into every pod**
(confirmed against RunPod's docs). The `RUNPOD_API_KEY` here is a
**pod-scoped key**, distinct from your real account-wide `RUNPOD_API_KEY`
on your local machine — your real key is never copied onto the pod at all,
which is the whole point: the pod can stop *itself* without holding
account-wide credentials.

Default behavior: `POST /pods/{id}/stop` once `run_pipeline.sh` exits,
whether it succeeded or failed — an unattended failed run left running is a
bigger cost risk than losing the chance to debug it interactively. Flags:

- `--terminate` — `DELETE /pods/{id}` instead of stopping (destroys the
  volume too). Only use this if `push_results.sh` or an `rsync` already
  ran successfully.
- `--keep-alive-on-failure` — skip shutdown specifically when
  `run_pipeline.sh` exits non-zero, so you can SSH back in and debug. Still
  shuts down normally on success.

This hasn't been exercised against a live pod yet (the API calls mirror
`launch_pod.py`'s already-verified contract, but confirm the pod-scoped key
actually has stop/delete permission on your own pod on a first real run,
before relying on it unattended).

## One pod at a time

`launch_pod.py` names every pod it creates with the prefix
`assistant-axis-` and checks `GET /pods` before creating a new one — if
anything with that prefix is already `RUNNING`, it refuses and tells you
the existing pod's id, rather than silently letting you pay for two.
Override with `--force` if you deliberately want two running concurrently.
This only looks at pods this project created — it won't touch or complain
about unrelated pods on the same RunPod account.

There's no cap today on GPU-hours or dollar spend *within* a single run —
that's still on you and `AGENT_NOTES.md`'s existing hard rule (see
[Cost awareness](#cost-awareness) below), not something these scripts
enforce automatically.

## Pushing results from the pod

`push_results.sh` lets the pod commit and push its own results, instead of
you having to `rsync` them back manually. This needs `GH_PUSH_TOKEN` set
*before* `bootstrap.sh` runs.

**Why not just copy your personal SSH key onto the pod?** That key almost
certainly grants push access to every repo you own (and possibly more —
other services, other accounts). Copying it onto a rented third-party GPU
pod means that broad credential leaves your control and sits on
infrastructure you don't own until the pod is terminated. A
repo-scoped token avoids all of that: if it leaks, the damage is contained
to this one repo, and you can revoke it independently at any time without
touching your real SSH key.

**Setting it up** (one-time, on GitHub):

1. GitHub → Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token.
2. **Repository access**: "Only select repositories" → this repo only.
3. **Permissions**: Repository permissions → Contents → **Read and write**.
   Nothing else needed.
4. Set an expiration you're comfortable with — short-lived is fine, this
   is meant to be regenerated per experiment, not a permanent credential.
5. Copy the token, then locally: `export GH_PUSH_TOKEN=github_pat_...`
   before `python3 runpod/launch_pod.py` (it gets forwarded into the pod's
   env), or export it directly in the SSH session before running
   `bootstrap.sh`.

**Using it**, from the pod, after a run finishes:

```bash
bash runpod/push_results.sh \
    --src /workspace/outputs/qwen-3-32b/moral-circle \
    --dest results/moral-circle-qwen-3-32b \
    --message "Add moral-circle axis results (qwen-3-32b)"
```

Copies everything from `--src` *except* `responses/` and `activations/`
(the large raw intermediates — never meant for git) into `<repo>/<dest>/`,
then commits and pushes just that path. Commits from the pod are authored
as `Claude (RunPod) <noreply@anthropic.com>`, set by `bootstrap.sh` when it
configures the token — same convention as commits made from local Claude
Code sessions in this repo.

Without `GH_PUSH_TOKEN`, the remote stays the default read-only HTTPS
clone URL and `push_results.sh`'s final `git push` fails with an auth
error — a loud failure, not a silent no-op.

## Cost awareness

`AGENT_NOTES.md`'s hard rule: **stop and confirm parameters before any run
that could exceed $20 in judging/API spend or 13 GPU-hours of compute.**
This applies here. Concretely:

- **Pricing** (Secure Cloud, `cloudType` used here; check
  [runpod.io/pricing](https://www.runpod.io/pricing) for current numbers —
  these change): A100 80GB PCIe/SXM ≈ **$1.59/hr**, H100 80GB ≈
  **$2.89–3.49/hr** depending on the exact SKU RunPod allocates.
  `launch_pod.py` also prints the actual `costPerHr` it got while polling.
- Judge cost scales with how many responses step 3 scores. `--reduce_questions N`
  (forwarded to steps 1 and 4 as of this branch's `run_pipeline.sh`) cuts
  generation volume by roughly `N`x — e.g. `--reduce_questions 3` on the
  12-trait moral-circle batch is ~4,800 generations instead of the full
  14,400 at default settings.
- Re-running `run_pipeline.sh` after a partial failure is safe — every
  stage skips files that already exist, so you only pay for what's missing.
- `run_and_shutdown.sh` (see above) removes the most common way this goes
  wrong in practice — forgetting to stop the pod after a run finishes.
- **Terminate, don't just stop**, once you've pulled/pushed results and
  don't expect to resume — a stopped pod's persistent volume still costs
  storage.

## Notes on the pod config

- **GPU choice**: `launch_pod.py` requests `gpuTypeIds` as a priority list
  (A100 80GB PCIe → A100-SXM4-80GB → H100 80GB HBM3) and RunPod allocates
  whichever has capacity. Qwen3-32B in bf16 is ~64GB of weights alone, so
  don't go below an 80GB card without also changing `--tensor_parallel_size`
  and splitting across multiple smaller GPUs.
- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` —
  the "devel" (not "runtime") variant is used deliberately since vLLM's
  install step can need `nvcc`/CUDA headers.
- The REST API contract used here (`POST /pods`, `GET /pods`,
  `GET /pods/{id}`, `POST /pods/{id}/stop`, `DELETE /pods/{id}`,
  bearer-token auth) was verified against RunPod's live OpenAPI spec at
  `https://rest.runpod.io/v1/openapi.json` when these scripts were written.
  If pod creation starts failing, check that spec again before assuming the
  scripts are wrong — RunPod's available images and GPU inventory change
  over time.
