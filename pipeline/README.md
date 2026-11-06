# Computing the Assistant Axis

This directory contains the 5-step pipeline for computing the Assistant Axis from scratch for any model.

## Requirements

- GPU with sufficient memory for your target model
- `OPENAI_API_KEY` environment variable (for the LLM judge in step 3)

## Pipeline Modes

### Roger mode (default)

Generates combined role+trait instructions, standalone traits, and the default baseline. Uses `data/goal_roles_and_traits.json` to select which roles/traits to combine.

- `r_{role}__{trait}.jsonl` — goal role x non-goal trait (goal from role)
- `t_{role}__{trait}.jsonl` — non-goal role x goal trait (goal from trait)
- `{trait}.jsonl` — standalone trait
- `default.jsonl` — default baseline

Default 30x30 produces 900 r_ + 900 t_ + ~245 traits + 1 default = ~2046 entities.

### Christina mode

Generates standalone roles (or traits) from `--roles_dir`. Run the pipeline separately for roles and traits using different `--roles_dir` and `--output_dir` to avoid filename collisions.

Neither mode has a built-in "just run these N entities" flag at the `run_pipeline.sh` level. See [`SUBSET_RUNS.md`](SUBSET_RUNS.md) for the two ways to scope a run to a specific subset of roles/traits — passing `--roles` to each script directly, or pointing `--roles_dir` at a small directory of symlinks (worked example: [`data/traits/instructions/_moral_circle/`](../data/traits/instructions/_moral_circle/README.md)).

`--reduce_questions` is now forwarded to steps 1 and 4 in christina mode (previously silently ignored there, so christina runs always used the un-reduced defaults regardless of the script's `$REDUCE_QUESTIONS` setting).

For running any of this on a rented GPU (e.g. because you're developing locally on a machine without a big enough NVIDIA card), see [`../runpod/README.md`](../runpod/README.md).

## Full pipeline tips

We recommend running scripts separately rather than the bash script.

- Run `1_generate.py` then `2_activations.py` in order with [task-spooler](https://github.com/justanhduc/task-spooler) or tmux
- `3_judge.py` can run in parallel with `2_activations.py` after `1_generate.py` completes
- Steps 4 and 5 run after 1, 2, 3 are all complete
- All scripts skip existing outputs on rerun
- `1_generate.py` and `2_activations.py` support multi-worker tensor parallelization

```bash
./run_pipeline.sh
```

Edit the script to configure model, mode, and output directory.

## Step-by-Step Instructions

### 1. Generate Responses

```bash
# Roger mode (combined + traits + default)
uv run 1_generate.py \
    --model Qwen/Qwen3-32B \
    --mode roger \
    --goal_count 30 --non_goal_count 30 \
    --output_dir outputs/roger/responses

# Christina mode (roles only)
uv run 1_generate.py \
    --model Qwen/Qwen3-32B \
    --mode christina \
    --output_dir outputs/roles/responses
```

Each entity produces up to 1500 responses (5 instruction variants × 300 questions in roger mode; 5 × 240 in christina mode).  With `--reduce_questions N` (default `3` for roger, `1` for christina) every Nth question is kept, so a typical roger run produces 5 × 100 = 500 responses per entity.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `roger` | `roger` or `christina` |
| `--goal_count` | 30 | Top-N from each goal list (roger mode) |
| `--non_goal_count` | 30 | Top-N from each non-goal list (roger mode) |
| `--traits_dir` | `../data/traits/instructions` | Trait JSON directory |
| `--goal_file` | `../data/goal_roles_and_traits.json` | Goal lists file (roger mode) |
| `--roles_dir` | `../data/roles/instructions` | Role JSON directory |
| `--questions_file` | `../data/extraction_questions.jsonl` | Questions JSONL |
| `--question_count` | 300 (roger) / 240 (christina) | Questions per entity |
| `--tensor_parallel_size` | auto | GPUs per worker |
| `--roles` | all | Specific role names (christina mode) |

### 2. Extract Activations

Extract mean response activations (unchanged from original):

```bash
uv run 2_activations.py \
    --model Qwen/Qwen3-32B \
    --responses_dir outputs/roger/responses \
    --output_dir outputs/roger/activations \
    --batch_size 8
```

Processes whatever JSONL files are in `--responses_dir`.

**Options:**
- `--batch_size`: Adjust for GPU memory
- `--layers`: Layers to extract (default: all)
- `--tensor_parallel_size`: GPUs per worker

### 3. Score Responses

Score role/trait adherence using an LLM judge:

```bash
uv run 3_judge.py \
    --responses_dir outputs/roger/responses \
    --output_dir outputs/roger/scores
```

The judge automatically detects response type from the filename:

| Filename pattern | Type | Eval prompt source |
|-----------------|------|-------------------|
| `r_<role>__<trait>` | Combined | Compound prompt from both descriptions (0-3) |
| `t_<role>__<trait>` | Combined | Compound prompt from both descriptions (0-3) |
| `default` | Default | Skipped (step 4 uses all activations) |
| Exists in `--roles_dir` | Role | `eval_prompt` from role JSON (0-3) |
| Exists in `--traits_dir` | Trait | Constructed from description (0-3) |

**Options:**
- `--traits_dir`: Trait JSON directory (default: `../data/traits/instructions`)
- `--roles_dir`: Role JSON directory (default: `../data/roles/instructions`)
- `--dry_run`: Preview prompts without API calls
- `--judge_model`: Judge model (default: `gpt-4.1-mini`)

**Score scale (all types):**

| Score | Meaning |
|-------|---------|
| 0 | No characteristics, model refused |
| 1 | No characteristics (or only one for combined), model identifies as AI |
| 2 | Some characteristics but still identifies as AI |
| 3 | Fully embodying the role/trait/both |

### 4. Compute Per-Entity Vectors

Compute mean activation vectors from high-scoring (score=3) responses:

```bash
uv run 4_vectors.py \
    --activations_dir outputs/roger/activations \
    --scores_dir outputs/roger/scores \
    --output_dir outputs/roger/vectors
```

Processes whatever files are in the input directories.

**Limitation — question-matched filtering for trait directions:**
Each entity's vector is the mean of its own score=3 activations, filtered independently.
For antonym pairs (e.g., helpful/unhelpful), the score=3 question sets can differ
substantially — "helpful" may keep 95%+ of questions while "unhelpful" keeps only ~50%.
Differencing these vectors to get a trait direction therefore mixes a true trait contrast
with a question-distribution artifact.

A future step 4b could fix this by loading both activation files for a known pair,
intersecting their score=3 question keys, and computing matched means from only the
shared questions before differencing. This must be a pipeline step (not an analysis
script) because the activation files are large.

### Auditing missing or corrupt vectors

After step 4, run `scan_missing_vectors.py` to verify every activation
ended up with a corresponding vector, classify any gaps, and detect
vectors that landed at full size but are unreadable (corrupt-on-write).

**Comprehensive mode (recommended)** — point at the output root and the
scanner auto-discovers every entity-type subdir (`default/`, `roles/`,
`traits/`, `combinations/`, ...) with an `activations/` subdir, plus
every sibling `vectors*` variant (`vectors/`, `vectors_unfiltered/`, ...)
that has real (non-symlinked) `.pt` files.  A `missing_vectors_audit.json`
and `missing_vectors_rerun.txt` are written next to each scanned
vectors dir:

```bash
uv run scan_missing_vectors.py --root outputs/qwen-3-32b/ --min_count 50
```

**Single-pair mode (back-compat)**:

```bash
uv run scan_missing_vectors.py \
    --activations_dir outputs/qwen-3-32b/roles/activations \
    --vectors_dir     outputs/qwen-3-32b/roles/vectors \
    --scores_dir      outputs/qwen-3-32b/roles/scores \
    --min_count       50 \
    --deep_load \
    --rerun_list      outputs/qwen-3-32b/roles/vectors_rerun.txt
```

Each entity is classified as one of:

| status                              | meaning                                                                              |
|-------------------------------------|--------------------------------------------------------------------------------------|
| `ok`                                | vector present, healthy size, **and `torch.load` succeeds**                          |
| `ok_zero_size`                      | vector exists but < 256 bytes (likely a stub from an aborted save)                   |
| `corrupt_vector`                    | vector full-size but `torch.load` raises (the corrupt-on-write failure mode)         |
| `missing_activation`                | no activation .pt at all                                                             |
| `corrupt_or_truncated_activation`   | activation .pt smaller than 50% of peer median, OR (with `--deep_load`) full-size but `torch.load` raises |
| `missing_scores`                    | filtered mode, no scores .json (judging never ran for this entity)                   |
| `below_min_count`                   | filtered mode, < `--min_count` score=3 entries (legitimate filter)                   |
| `all_nan_or_empty`                  | activations exist but contain no usable tensors                                      |
| `mysterious`                        | everything upstream looks fine but step 4 produced no vector — re-run candidate      |
| `orphan_vector`                     | vector .pt exists with no source activation                                          |

The re-run candidates are: `mysterious`, `corrupt_or_truncated_activation`,
`corrupt_vector`, `ok_zero_size`.  These come from two upstream failure
modes:

- **Transient NFS short-read at step-4 read time** — pre-retry step 4
  logged a warning and skipped the entity, leaving no vector.  Show up
  as `mysterious`.
- **Corrupt-on-write at step-4 save time** — full-size vector landed
  on disk but pickle/zip stream is malformed.  Caught now by the
  `torch_save_with_retry` post-write size check on new runs; the
  scanner's load-check finds historical damage.  Show up as
  `corrupt_vector`.

Re-running step 4 with `--overwrite missing_vectors_rerun.txt` (per
vectors variant) will recover all four re-runnable categories.

**Performance / accuracy knobs:**

- The per-vector `torch.load` check is **default-on**.  Pass
  `--no_load_check` to skip it (much faster, but won't detect
  `corrupt_vector`).
- `--deep_load`: for *missing* vectors, additionally `torch.load` the
  activation file and re-apply step 4's filter to confirm the
  classification.  Without it, anything that passes the cheap shallow
  checks is reported as `mysterious`.
- In comprehensive mode, `vectors_unfiltered/` (or any `vectors*`
  variant ending in `_unfiltered`) is auto-detected as unfiltered mode
  regardless of whether `scores/` exists.

### Robust file IO (NFS / MooseFS)

Steps 2, 4 and 5 use the project-standard transient-IO retry from
`assistant_axis/atomic_io.py`:

- 5 attempts with exponential backoff `[5, 20, 60, 180]` s
  (cumulative ~265 s, matches `axis_judge_correlation`'s API retry).
- Treats `OSError`, **`RuntimeError`** *and* `EOFError` as transient.
  The `RuntimeError` case is the failure mode that surfaced as
  `"storage has wrong byte size of dtype ..."` / `"PytorchStreamReader
  failed reading zip archive: ... unexpected EOF, expected N more
  bytes"` from `torch.load` on RunPod NFS — earlier code only retried
  `OSError` and missed it.
- All saves go through `torch_save_with_retry`, which stages to
  `TMPDIR` (point this at `/dev/shm` on RunPod), copies with retry/
  backoff, atomically renames into place, and runs three layers of
  post-copy integrity verification (each triggers another copy
  attempt on failure):

  1. **Size check** — destination must match staging byte-count.
     Catches outright short-writes.
  2. **SHA-256 byte-equality check** — destination must hash to the
     same digest as staging.  Catches silent byte-flips inside a
     correct-size file (e.g. corrupted NFS chunks that pass length
     checks).  This is the failure mode behind the apparently-healthy
     2.6 GB `r_guardian__casual.pt` that later refused to `torch.load`
     with `"storage has wrong byte size of dtype"`.
  3. **`torch.load` round-trip** — destination must deserialise
     end-to-end with `weights_only=False`.  Tests at the actual
     usage level; catches version-drift / pickle-format weirdness
     that byte-equality alone wouldn't.

  Cost on a ~2.6 GB activation file is roughly one extra read for
  SHA-256 plus one extra read for `torch.load` (both warm in the
  kernel page cache after the write, so the wall-clock impact is
  measurably less than 2× the bare write).  Pass
  `verify_sha256=False` and/or `verify_load=False` to opt out for
  hot-path writes where the cost outweighs the safety.

If you hit a final-failure ERROR after all retries, that's a real
non-transient corruption — clear the affected file and re-run that
step.

### 5. Compute Axis

Aggregate vectors into the final axis:

```bash
uv run 5_axis.py \
    --vectors_dir outputs/roger/vectors \
    --output outputs/roger/axis.pt
```

```
axis = mean(default_vectors) - mean(non_default_vectors)
```

## Data Format

### Roles

Role definitions in `data/roles/instructions/*.json`:

```json
{
  "description": "...",
  "instruction": [
    {"pos": "You are a pirate..."},
    {"pos": "Act as a swashbuckling pirate..."}
  ],
  "eval_prompt": "Rate how well the answer embodies a pirate..."
}
```

275 character roles plus `default.json` with neutral system prompts.

### Traits

Trait definitions in `data/traits/instructions/*.json`:

```json
{
  "positive_label": "greedy",
  "negative_label": "non-greedy",
  "description": "...",
  "instruction": [
    {"pos": "You are greedy...", "neg": "You are generous..."},
    ...
  ],
  "eval_prompt": "..."
}
```

The pipeline uses only `instruction[].pos`. The `eval_prompt` in trait files uses a 0-100 scale for analysis; the pipeline constructs its own 0-3 eval prompt from the `description`.

### Combined filename convention

`r_{role}__{trait}` — goal from role (goal role x non-goal trait)
`t_{role}__{trait}` — goal from trait (non-goal role x goal trait)

Double underscore `__` separates role and trait names.

### Questions

`data/extraction_questions.jsonl` (300 questions designed to elicit persona-specific responses; christina mode uses the first 240 by default).

## Tips

- **Parallelization**: Steps 2 (GPU-bound) and 3 (judge-API-bound) run
  concurrently by default — they share no resources beyond step 1's response
  files, so the wall-clock saving is roughly the smaller of the two step
  times.  Disable with `--steps23serial` if you want clean per-step logs
  (e.g. for debugging); errors from either parallel branch surface and abort
  the run before step 4.
- **Checkpointing**: All steps skip existing outputs — delete to regenerate
- **Task spooler**: For long-running jobs, use [task-spooler](https://github.com/justanhduc/task-spooler)

## Pre-computed Axes and Vectors

Pre-computed axes for Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B are available on [HuggingFace](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors).
