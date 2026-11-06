# Moral-circle spectrum batch

This directory is a curated subset of `data/traits/instructions/` — symlinks
to the 12 trait definitions below, plus `default.json` (symlinked from
`data/roles/instructions/default.json`, since `data/traits/instructions/`
has no `default.json` of its own and stage 5 needs a default baseline to
compute `axis = mean(default) − mean(trait_vectors)` against).

It exists because `pipeline/run_pipeline.sh`'s christina mode processes
*every* `.json` file it finds in `--roles_dir` — there's no per-run entity
filter — and `data/traits/instructions/` has 304 files. Pointing the
pipeline at this directory instead scopes a run to just this batch.

## Background

These 12 traits were added by Roger Dearnaley's fork (branch
`anthropic-vllm-uv`) as part of extending Christina Lu's Assistant Axis
method toward a "moral-circle size" spectrum — how wide a circle of
concern a persona extends, from purely self-interested to universal. Full
instruction/question/judge-rubric definitions are committed; **as of this
writing, none of the 5 pipeline stages have been run on them** (no
responses, activations, judge scores, vectors, or axis exist for any of
these 12). See `agents/agents-9-12-26/` for the fuller research-context
writeup and two visual artifacts (`assistant_axis_pipeline_progress.html`,
`persona_goal_subspace_map.html`) mapping this batch against the rest of
the fork's open work.

## The spectrum

Ordered self → all beings, per the source instruction files:

| Trait | Negative pole | Notes |
|---|---|---|
| `selfish` | unselfish | Self-most pole |
| `clannish` | non-clannish | Family / close in-group |
| `cliqueish` | non-cliqueish | Friend group / clique |
| `insular` | non-insular | Own local community |
| `parochial` | non-parochial | Narrow local-area concern (paired with `insular` to separate two shades of the same idea) |
| `regionalist` | non-regionalist | Multi-nation region |
| `nationalist` | non-nationalist | Nation-bounded |
| `patriotic` | non-patriotic | Nation-level, positive framing (companion to `nationalist`) |
| `sectarian` | non-sectarian | Religious/ideological group — cuts across the geographic spine above |
| `philanthropic` | non-philanthropic | Actively benefits people broadly |
| `humanitarian` | non-humanitarian | Widest human-scale pole |
| `kind_to_animals` | indifferent-to-animals | Moral circle extended beyond humanity |

Each trait's own JSON file (`../<trait>.json`, following the symlinks)
already contains both poles as paired `pos`/`neg` instruction variants —
no separate negative-pole files exist or are needed. Note the pipeline's
standard generation path (`RoleResponseGenerator`) only consumes the
`pos` field; the `neg` field is present as metadata/documentation of the
intended contrast but isn't currently wired into a separate negative-pole
generation step.

## Running it

See [`runpod/README.md`](../../../../runpod/README.md) for the full
provisioning workflow. The pipeline invocation for this batch:

```bash
cd pipeline
./run_pipeline.sh --mode christina --reduce_questions 3 \
    --model Qwen/Qwen3-32B \
    --tensor_parallel_size 1 \
    --roles_dir ../data/traits/instructions/_moral_circle \
    --output_dir /workspace/outputs/qwen-3-32b/moral-circle
```

`--reduce_questions 3` (every 3rd question) was chosen for a first real
run on this batch: ~4,800 generations total across the 12 traits rather
than the full-default 14,400, while still clearing `4_vectors.py`'s
`--min_count 50` threshold per trait. Once these look sane, rerun without
`--reduce_questions` (or with `1`) for the full-scale version — the
pipeline skips files that already exist, so this only regenerates what's
missing.

## Output

`$OUTPUT_DIR/axis.pt` and `$OUTPUT_DIR/vectors/*.pt` are the results worth
keeping (small; safe to commit or pull back to a local machine).
`$OUTPUT_DIR/responses/` and `$OUTPUT_DIR/activations/` are large raw
intermediates — leave them on the pod or archive them separately; they're
gitignored (`outputs/`) and not meant to be committed.

Note `5_axis.py` computes a **self-contained axis over just this batch's
12 traits + default** — it does not embed these into the same shared PCA
space as the fork's other 33 finished axes. Doing that (to compare this
spectrum against `angel`/`demon`, the RLHF triad, etc., in one persona
space) is a separate, manual step using `notebooks/pca.ipynb` over the
combined vector set once both are available.
