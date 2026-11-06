# Running the pipeline for a specific subset of roles/traits

Both pipeline modes process *everything* in their input directory by
default:

- **Roger mode** runs the full role×trait grid from `data/goal_roles_and_traits.json`.
- **Christina mode** runs every `.json` file found in `--roles_dir` — for
  `data/traits/instructions/` that's 300+ files, not just the handful you
  probably want.

Neither `run_pipeline.sh` nor the underlying scripts have a single flag
that means "just these N entities" end-to-end. This doc covers the two
ways to actually scope a run, when to use each, and a full walkthrough of
Option B.

Both options assume you already have a GPU available (locally, or via
[`../runpod/README.md`](../runpod/README.md)) and the judge API key set for
step 3 — `OPENAI_API_KEY` for the default `gpt-4.1-mini` judge, or
`ANTHROPIC_API_KEY` if you're using a Claude judge model (see
[Judging the same responses with a different model](#judging-the-same-responses-with-a-different-model)).

## Option A — `--roles` flag on each script

`1_generate.py`, `2_activations.py`, and `3_judge.py` each accept
`--roles <name> <name> ...` to filter to specific entity names (`.stem`
of the response/activation/score filename). `4_vectors.py` and `5_axis.py`
have no such flag and don't need one — they just process whatever `.pt`
files already exist in `--activations_dir`/`--vectors_dir`, so they're
automatically scoped by what steps 1–3 produced.

**`run_pipeline.sh` does not forward `--roles`** in christina mode (the
same gap `--reduce_questions` had before it was fixed — see
[`README.md`](README.md)), so Option A means calling the five scripts
directly instead of the wrapper. You lose the wrapper's conveniences (GPU
pre-flight check, tmpfs `HF_HOME`, steps 2+3 running concurrently,
post-pipeline `scan_missing_vectors.py` audit) and have to repeat the same
`--roles` list at each of steps 1–3, but nothing needs creating first.

```bash
cd pipeline

uv run 1_generate.py --mode christina --model Qwen/Qwen3-32B \
    --roles_dir ../data/traits/instructions \
    --roles selfish nationalist kind_to_animals default \
    --output_dir /workspace/outputs/qwen-3-32b/adhoc/responses

uv run 2_activations.py --model Qwen/Qwen3-32B \
    --responses_dir /workspace/outputs/qwen-3-32b/adhoc/responses \
    --output_dir /workspace/outputs/qwen-3-32b/adhoc/activations \
    --roles selfish nationalist kind_to_animals default

uv run 3_judge.py --entity_type trait \
    --responses_dir /workspace/outputs/qwen-3-32b/adhoc/responses \
    --output_dir /workspace/outputs/qwen-3-32b/adhoc/scores \
    --roles selfish nationalist kind_to_animals
    # note: no "default" here -- 3_judge.py skips it unconditionally
    # (step 4 uses all of default's activations without scores)

uv run 4_vectors.py \
    --activations_dir /workspace/outputs/qwen-3-32b/adhoc/activations \
    --scores_dir /workspace/outputs/qwen-3-32b/adhoc/scores \
    --output_dir /workspace/outputs/qwen-3-32b/adhoc/vectors

uv run 5_axis.py \
    --vectors_dir /workspace/outputs/qwen-3-32b/adhoc/vectors \
    --output /workspace/outputs/qwen-3-32b/adhoc/axis.pt
```

Don't forget `default` in the `--roles` list for steps 1 and 2 (but *not*
3) — stage 5 needs `default`'s activations as the baseline
(`axis = mean(default) − mean(trait_vectors)`), and it has no
`eval_prompt`/rubric of its own to be judged against.

**Use Option A when:** the selection is a one-off, you're iterating
quickly on which entities to include, or you don't want a new directory
left behind in the repo.

## Option B — a scoped symlink directory

Instead of filtering at the script level, point `--roles_dir` at a small
directory that *only contains* the entities you want, via symlinks to the
originals in `data/traits/instructions/` or `data/roles/instructions/`.
`run_pipeline.sh`'s christina mode needs no changes for this — it already
just globs `*.json` in whatever `--roles_dir` you give it, so a smaller
directory means a smaller run, automatically, at every stage including 4
and 5.

**Use Option B when:** the batch has a name and a reason (a themed
group of traits, a recurring experiment), you want the run driven by the
unmodified `run_pipeline.sh` wrapper (GPU pre-flight check, tmpfs, steps
2+3 concurrency, post-pipeline scan all included for free), or you expect
to rerun/extend the same batch later.

`data/traits/instructions/_moral_circle/` is a worked example of this
pattern already in the repo — see its own
[README](../data/traits/instructions/_moral_circle/README.md) for the
specific experiment it documents. What follows is the general recipe.

### Step-by-step

**1. Decide the entity list and where they live.**

Traits live in `data/traits/instructions/`, roles in
`data/roles/instructions/`. Don't mix the two in one directory — christina
mode infers `--entity_type` (`trait` vs `role`) for step 3 from whether
`--roles_dir` contains the substring `traits` or `roles`
(`run_pipeline.sh`'s `case "$ROLES_DIR" in *traits*) ... *roles*) ...`),
so **the scoped directory's path must itself contain `traits` or `roles`**
— put it *inside* `data/traits/instructions/` or
`data/roles/instructions/`, not next to them.

**2. Create the scoped directory.**

```bash
cd /Github/assistant-axis-roger   # repo root
mkdir -p data/traits/instructions/_my_batch
cd data/traits/instructions/_my_batch
```

**3. Symlink each entity you want into it.**

```bash
for t in trait_one trait_two trait_three; do
    ln -sf "../${t}.json" "${t}.json"
done
```

Relative symlinks (`../trait_one.json`, not an absolute path) so the
directory stays portable if the repo is cloned elsewhere.

**4. Add the `default` baseline.**

Stage 5 needs a `default` baseline in the *same* run, and
`data/traits/instructions/` has no `default.json` of its own (only
`data/roles/instructions/` does) — christina mode will otherwise generate
responses for your traits but have nothing to subtract them from at step 5.

```bash
ln -sf "../../../roles/instructions/default.json" "default.json"
```

(For a *roles* batch under `data/roles/instructions/_my_batch/`, this step
is unnecessary — `default.json` already lives alongside the other role
files at `../default.json`.)

**5. Verify every symlink resolves.**

```bash
for f in *.json; do [ -f "$f" ] && echo "OK $f" || echo "BROKEN $f"; done
```

A broken symlink here (typo'd trait name, wrong relative path depth) fails
loudly and immediately, rather than surfacing later as a confusing "no
eval_prompt" warning from step 3.

**6. Run the pipeline against the scoped directory.**

From `pipeline/`:

```bash
cd pipeline
./run_pipeline.sh --mode christina \
    --model Qwen/Qwen3-32B \
    --tensor_parallel_size 1 \
    --reduce_questions 3 \
    --roles_dir ../data/traits/instructions/_my_batch \
    --output_dir /workspace/outputs/qwen-3-32b/my_batch
```

- `--reduce_questions 3` (every 3rd question) is optional but recommended
  for a first run of a new batch — cuts generation+judging volume ~3x
  while still safely clearing `4_vectors.py`'s default `--min_count 50`
  threshold. Omit it (or pass `1`) for the full-scale run once the
  reduced run looks sane; `run_pipeline.sh` skips files that already
  exist, so this only fills in what's missing rather than redoing
  everything.
- `--skip-gpu-check` if you're on a shared box and the pre-flight check's
  "GPU already busy" error is a false positive from a known co-tenant.
- Everything else in [`README.md`](README.md) (step 2/3 concurrency,
  tmpfs, the post-pipeline scan) applies unchanged.

Steps 1–6 above are the whole setup-and-run sequence. What follows are
things you'll do after it finishes.

### After it finishes

**Check the output.**

```
/workspace/outputs/qwen-3-32b/my_batch/
├── responses/       # stage 1 -- one .jsonl per entity (+ default.jsonl)
├── activations/     # stage 2 -- one .pt per entity
├── scores/          # stage 3 -- one .json per entity (no default.json)
├── vectors/         # stage 4 -- one .pt per entity that cleared min_count
└── axis.pt          # stage 5 -- the final result
```

Only `vectors/` and `axis.pt` are small and worth keeping/committing;
`responses/` and `activations/` are large raw intermediates (already
covered by the repo's `outputs/` gitignore rule — leave them on the pod or
archive separately). `responses/` and `activations/` are also exactly what
you need if you want to re-judge with a different model — see
[Judging the same responses with a different model](#judging-the-same-responses-with-a-different-model)
below.

**Decide whether to commit the scoped directory.**

The symlink directory itself (`_my_batch/`) is tiny (a handful of symlinks
+ maybe a short README) and safe to commit if the batch is worth
documenting for others — that's what
[`_moral_circle/`](../data/traits/instructions/_moral_circle/README.md)
does. If it was truly a one-off, delete it locally instead:

```bash
rm -rf data/traits/instructions/_my_batch
```

(This only removes the symlinks, never the original trait/role files they
point to.)

## Judging the same responses with a different model

> **Already have a computed axis and just want to try a new judge tier?**
> If it's `angel_vs_demon` or `decisive_vs_indecisive`, you don't need any
> of this — `results_analysis/judge_tier_cost_eval.py` reuses the already-
> committed `projections.json` for those two axes directly, no GPU/pipeline
> run needed at all. See
> [`results_analysis/README.md`](../results_analysis/README.md#evaluating-a-new-judge-tier-without-recomputing-activations).
> Everything below is for entities that aren't one of those two.

This is the actual point of scoping a run to a couple of entities first —
generation (steps 1–2) is the expensive, GPU-bound part; judging (step 3)
is comparatively cheap. Once you have `responses/` and `activations/` for
your chosen entities (from either Option A or Option B), you can re-run
**just steps 3–5** as many times as you want, once per judge model, each
into its own output directories — reusing the same `responses/` and
`activations/` every time, since neither depends on which judge scored the
responses.

### `3_judge.py` supports both OpenAI and Anthropic judges

`--judge_model` is routed automatically by
`assistant_axis.judge.provider_for_model()` based on the model name prefix:

| Prefix | Provider | Required env var |
|---|---|---|
| `gpt-`, `o1-`, `o3-`, `o4-` | OpenAI | `OPENAI_API_KEY` |
| `claude-` | Anthropic | `ANTHROPIC_API_KEY` |

(As of this branch — `pipeline/3_judge.py` previously only supported
OpenAI models even though `assistant_axis/judge.py`'s Anthropic support
already existed underneath; it's now wired through
`call_judge_batch_unified()`, which dispatches by the same provider
detection. Set whichever env var(s) you need for the judges you're
comparing — both, if running both.)

### Example: comparing a GPT judge and a Claude judge

Continuing the `_my_batch` example from above:

```bash
cd pipeline

# Judge #1: gpt-4.1-mini
uv run 3_judge.py --entity_type trait \
    --responses_dir /workspace/outputs/qwen-3-32b/my_batch/responses \
    --output_dir /workspace/outputs/qwen-3-32b/my_batch/scores_gpt41mini \
    --judge_model gpt-4.1-mini

uv run 4_vectors.py \
    --activations_dir /workspace/outputs/qwen-3-32b/my_batch/activations \
    --scores_dir /workspace/outputs/qwen-3-32b/my_batch/scores_gpt41mini \
    --output_dir /workspace/outputs/qwen-3-32b/my_batch/vectors_gpt41mini

uv run 5_axis.py \
    --vectors_dir /workspace/outputs/qwen-3-32b/my_batch/vectors_gpt41mini \
    --output /workspace/outputs/qwen-3-32b/my_batch/axis_gpt41mini.pt

# Judge #2: a Claude model -- substitute whichever current Claude model id
# you intend to use as judge (check what's currently available/priced
# before picking one; not hardcoding a specific id here).
uv run 3_judge.py --entity_type trait \
    --responses_dir /workspace/outputs/qwen-3-32b/my_batch/responses \
    --output_dir /workspace/outputs/qwen-3-32b/my_batch/scores_claude \
    --judge_model claude-<model-id>

uv run 4_vectors.py \
    --activations_dir /workspace/outputs/qwen-3-32b/my_batch/activations \
    --scores_dir /workspace/outputs/qwen-3-32b/my_batch/scores_claude \
    --output_dir /workspace/outputs/qwen-3-32b/my_batch/vectors_claude

uv run 5_axis.py \
    --vectors_dir /workspace/outputs/qwen-3-32b/my_batch/vectors_claude \
    --output /workspace/outputs/qwen-3-32b/my_batch/axis_claude.pt
```

`axis_gpt41mini.pt` and `axis_claude.pt` are now directly comparable (e.g.
per-layer cosine similarity) since they're built from **identical**
responses and activations — the judge model is the only variable that
differs between them. This is the same shape of comparison already in the
repo at `roger/axis_judge_experiments/angel_vs_demon/{gpt,sonnet}/`
(`gpt_vs_sonnet_rhos_di.json`), just applied to whatever entities you
scoped in steps 1–6 instead of the `angel`/`demon` pair.

**Watch for:** the two judges can disagree on which responses count as
"fully role-playing" (score 3), so one judge's run may drop below
`4_vectors.py`'s `--min_count` threshold (default 50) for an entity where
the other judge's run doesn't — especially likely if you generated with
`--reduce_questions` reduced. If an entity is missing from one
`vectors_*/` directory but present in the other, that's almost always why;
check `scores_*/<entity>.json` for the score distribution before assuming
something's broken.

### Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ERROR: cannot infer --entity_type from ROLES_DIR=...` | The scoped directory's path doesn't contain `traits` or `roles` (step 1 of the recipe above) |
| Step 3 logs `Role <name> has no eval_prompt` / `No trait file found for <name>` | A broken or missing symlink — rerun the verification loop in step 5 |
| Step 4 silently produces fewer vectors than entities you generated | Some entity didn't clear `--min_count` (default 50) score=3 responses — check `scores/<name>.json`; a heavily `--reduce_questions`-reduced run is the usual cause |
| `axis.pt` looks wrong / missing the baseline | `default` wasn't included in the scoped directory (or wasn't generated in the same `--output_dir`) |

## Comparison

| | Option A (`--roles`) | Option B (scoped directory) |
|---|---|---|
| Setup | None | Create a directory + symlinks once |
| Uses `run_pipeline.sh` as-is | No — call the 5 scripts directly | Yes |
| Gets GPU pre-flight / tmpfs / step 2+3 concurrency / post-scan | No (unless you replicate them by hand) | Yes, automatically |
| Repeating the selection at each step | Yes, at steps 1–3 | No — encoded once in the directory |
| Leaves something behind in the repo | No | Yes (unless deleted afterward) |
| Best for | Quick one-off, still-deciding-what-to-include | Named, recurring, or documented batches |
