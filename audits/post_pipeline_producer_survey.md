# Post-pipeline producer survey (Phase 1.0a)

Goal: identify every script that writes files into the dataset tree
(`runpod_workspace/qwen/qwen-3-32b *`) **after** the main vector-extraction
pipeline finishes.  These post-pipeline files currently live alongside raw
pipeline outputs in `combinations/vectors/`, where their mtimes would be
mistaken for pipeline-output timestamps by any future provenance/manifest
machinery.  Phase 1.0 will move them into a dedicated `derived/` subtree.

## Decisive finding

**`results_analysis/compute_combo_marginals.py` is the only script that writes
post-pipeline outputs into the dataset tree.**

No other `results_analysis/**.py` script touches the dataset tree.  All other
analysis scripts write into `roger/`, `data/`, or an explicit
`--experiment_dir` / `--output_dir` / `--output` path — all outside the
dataset.

Spot-checked all `torch.save`, `np.save`, `np.savez`, `json.dump`,
`json.dumps`, and `.write_text(...)` call sites under `results_analysis/`:

| Script | Output target | In dataset tree? |
|---|---|---|
| `compute_combo_marginals.py` | `data_dir/combinations/vectors/...` | **YES** |
| `gpt_vs_sonnet_scatter.py`, `batch_size_rho_curve.py`, `pc_round_trip/*`, `rho_by_layer.py`, `gpt_sonnet_weight_sweep.py`, `token_position_noise_analysis.py`, `optimal_axis_for_judge.py`, `whitening_k_peak_fit.py`, `whitening_k_weighted_scatter.py`, `whitening_k_sweep.py`, `judge_ensemble_rho_curve.py`, `axis_judge_correlation.py`, `infer_axis_description.py`, `run_axis_experiment_batch.py`, `gpt_anthropic_response_weight_sweep.py`, `standardize_axis_spec.py`, `judge_ensemble_rho_curve.py` | `roger/...` or `experiment_dir/` | no |

## What `compute_combo_marginals.py` writes

Default invocation (`--output_root` defaults to `--data_dir`) writes the
following into `data_dir/combinations/vectors/`:

| Path | Producer call site | Description |
|---|---|---|
| `r_goal/<role>.pt` (per goal-supplying role) | `compute_marginals(... kind='r' side='goal')` (line 232) | residual or centroid role marginals (~30 files) |
| `r_nogoal/<trait>.pt` (per non-goal trait) | `compute_marginals(... kind='r' side='nogoal')` (line 232) | residual or centroid trait marginals |
| `t_goal/<trait>.pt` (per goal-supplying trait) | `compute_marginals(... kind='t' side='goal')` (line 232) | residual or centroid trait marginals |
| `t_nogoal/<role>.pt` (per non-goal role) | `compute_marginals(... kind='t' side='nogoal')` (line 232) | residual or centroid role marginals |
| `mean_r_combos.pt` | `compute_kind_centroid('r', ...)` (line 284) | cross-kind centroid; auxiliary, no longer used by canonical-angles tool |
| `mean_t_combos.pt` | `compute_kind_centroid('t', ...)` (line 284) | cross-kind centroid; auxiliary |
| `theatricality_axis.pt` | `compute_theatricality_axis(...)` (line 414) | per-(slot, layer) unit axis + default offset |

Plus, when invoked in **centroid mode with `--output_root` pointed at
`*_legacy_centroid/`**:

| Path | Description |
|---|---|
| `r_goal_legacy_centroid/<role>.pt` etc. | bit-exact snapshots of the pre-residual files, kept for historical reproduction |

## Pipeline-generated files in `combinations/vectors/` (**stay in place**)

These are touched by the pipeline itself, not by post-pipeline analysis
scripts:

| Path | Producer | Notes |
|---|---|---|
| `default.pt` | symlink set up by pipeline step 4 → `../../default/vectors/default.pt` | raw pipeline output (verified `lrwxr-xr-x` on Roger and Roger 8slot) |
| `r_<role>__<trait>.pt`, `t_<role>__<trait>.pt` (per-pair) | the activations pipeline | raw pipeline outputs (~3.6k files) |
| `missing_vectors_audit.json`, `missing_vectors_rerun.txt` | `pipeline/scan_missing_vectors.py` | pipeline-stage audit; runs at end of each pipeline run |

## Consumers that read the post-pipeline derived files

Path resolution for these files is centralized in `results_analysis/canonical_angles/data.py`.
The grep below shows every read site:

| Reader | Read sites | Notes |
|---|---|---|
| `results_analysis/canonical_angles/data.py` | line 121 (`theatricality_axis.pt`), line 141, 253–254 (per-etype root resolution), 437 (`legacy_centroid`), 782 (combined kind triple), 392 (existence check) | sole owner of post-pipeline-derived path resolution |
| `results_analysis/canonical_angles/core.py` | line 213 (etype dispatch), line 195–197 (theatricality docstring) | uses `data.py` resolver |
| `results_analysis/canonical_angles/plots/ca1_plane_pre_post_shear.py` | line 105 (`cv = data_dir / "combinations" / "vectors" / folder`) | hardcoded; needs an update when layout moves |
| `results_analysis/canonical_angles/plots/combos_vs_traits_roles_pooled.py`, `goal_vs_nogoal_mutual.py` | reference `_goal/_nogoal` etypes | go through `data.py` resolver |
| `results_analysis/variance_decomposition.py` | line 626 (`combos_dir = data_dir / "combinations" / "vectors"`) | only reads raw combos (the `r_*__*.pt` / `t_*__*.pt` files), **not** derived |
| `results_analysis/all_roles_pairwise_slots.py` | line 192 (`data_dir / "combinations" / "vectors" / f"{name}.pt"`) | only reads raw combos |
| `results_analysis/whitening_k_sweep.py` | line 168 (similar) | only reads raw combos |

## Inventory across the four datasets

```
runpod_workspace/qwen/qwen-3-32b Christina         -- no combinations/vectors/ at all (out of scope)
runpod_workspace/qwen/qwen-3-32b Christina headers -- no combinations/vectors/ at all (out of scope)
runpod_workspace/qwen/qwen-3-32b Roger             -- has the full post-pipeline set, including:
    r_goal/, r_nogoal/, t_goal/, t_nogoal/        (current residual marginals)
    r_goal_legacy_centroid/  + 3 siblings         (historical centroid snapshots)
    r_goal copy/  + 3 siblings                    (FINDER ACCIDENTS — not produced by any script;
                                                    safe to delete during migration)
    mean_r_combos.pt, mean_t_combos.pt
    theatricality_axis.pt
runpod_workspace/qwen/qwen-3-32b Roger 8slot       -- canonical current state:
    r_goal/, r_nogoal/, t_goal/, t_nogoal/
    mean_r_combos.pt, mean_t_combos.pt, theatricality_axis.pt
    missing_vectors_audit.json, missing_vectors_rerun.txt   (pipeline outputs; stay)
```

## Implications for Phase 1.0b/c

- Producer to update: **one** script (`compute_combo_marginals.py`) — change
  the default of `--output_root`, or its derived paths, so writes go under
  `combinations/vectors/derived/...` instead of directly under `vectors/`.
- Consumers to update: **one** path-resolution module (`canonical_angles/data.py`)
  + **one** plot script with a hardcoded path
  (`canonical_angles/plots/ca1_plane_pre_post_shear.py`).  Everything else
  goes through the resolver.
- Migration script (Phase 1.0d) only has to move 7 things per dataset:
  `r_goal/`, `r_nogoal/`, `t_goal/`, `t_nogoal/`, `mean_r_combos.pt`,
  `mean_t_combos.pt`, `theatricality_axis.pt`.  In Roger 4-slot it also has
  to move 4 `_legacy_centroid/` siblings and decide what to do with the
  4 ` copy/` Finder dupes (recommend: delete, with a `--prune-finder-copies`
  flag and a `git ls-files` check first).
- Pipeline-generated files (`default.pt` symlink, `*__*.pt` raw combos,
  `missing_vectors_*`) **stay in place at `combinations/vectors/`** — the
  raw subtree continues to be the natural unit of pipeline-output
  manifesting.
