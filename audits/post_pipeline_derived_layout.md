# Phase 1.0b: post-pipeline `derived/` layout (locked)

> **Status (May 2026):** implemented and live. All four datasets
> (`Roger`, `Roger 4slot`, `Roger 8slot`, `Christina v2`) have been
> migrated to this layout. Code in `results_analysis/canonical_angles/data.py`
> and `results_analysis/compute_combo_marginals.py` writes / reads the
> `derived/` subtree by default and falls back to the legacy flat
> layout only when `derived/` is absent. See
> [`../AGENT_NOTES.md` § When to regenerate manifests](../AGENT_NOTES.md)
> for the operational rules.

## On-disk layout

Each dataset's `combinations/vectors/` becomes a *raw-only* subtree (pipeline
outputs only).  All post-pipeline derivatives live under a new `derived/`
subdir, organized by category.

```
data_dir/
└── combinations/
    └── vectors/
        ├── r_<role>__<trait>.pt        # raw pipeline outputs (~3.6k files)
        ├── t_<role>__<trait>.pt        #   "
        ├── default.pt                  # symlink set by pipeline step 4
        ├── missing_vectors_audit.json  # pipeline scan output
        ├── missing_vectors_rerun.txt   #   "
        └── derived/                    # ALL post-pipeline files live here
            ├── marginals/
            │   ├── r_goal/<role>.pt        # was: vectors/r_goal/<role>.pt
            │   ├── r_nogoal/<trait>.pt     # was: vectors/r_nogoal/...
            │   ├── t_goal/<trait>.pt       # was: vectors/t_goal/...
            │   └── t_nogoal/<role>.pt      # was: vectors/t_nogoal/...
            ├── aggregates/
            │   ├── mean_r_combos.pt        # was: vectors/mean_r_combos.pt
            │   └── mean_t_combos.pt        # was: vectors/mean_t_combos.pt
            ├── axis/
            │   └── theatricality_axis.pt   # was: vectors/theatricality_axis.pt
            └── legacy_centroid/        # only present on Roger 4-slot
                ├── r_goal/<role>.pt        # was: vectors/r_goal_legacy_centroid/...
                ├── r_nogoal/...            # was: vectors/r_nogoal_legacy_centroid/...
                ├── t_goal/...              # was: vectors/t_goal_legacy_centroid/...
                └── t_nogoal/...            # was: vectors/t_nogoal_legacy_centroid/...
```

## Etype-to-path mapping

The internal etype names in `canonical_angles/data.py` stay the same; only
the on-disk path resolution changes.

| Etype | Old path (under `combinations/vectors/`) | New path (under `combinations/vectors/derived/`) |
|---|---|---|
| `r_goal`, `r_nogoal`, `t_goal`, `t_nogoal` | `<etype>/<name>.pt` | `marginals/<etype>/<name>.pt` |
| `r_goal_legacy_centroid` ... `t_nogoal_legacy_centroid` | `<etype>/<name>.pt` | `legacy_centroid/<base_etype>/<name>.pt` (suffix stripped) |
| `theatricality_axis.pt` (no etype) | `theatricality_axis.pt` | `axis/theatricality_axis.pt` |
| `mean_r_combos.pt`, `mean_t_combos.pt` (no etype) | `<name>` | `aggregates/<name>` |

Files outside the mapping (i.e., raw pipeline files: `r_*__*.pt`,
`t_*__*.pt`, `default.pt`, `missing_vectors_*`) **stay where they are**.

## Reader resolution: fallback strategy

Each path-resolving helper tries the new path first; if it doesn't exist,
falls back to the legacy path.  This decouples the code change from the
physical migration, so:

- Code change can land before any data moves.
- Each dataset can be migrated independently, and the code stays correct
  during the in-flight period.
- If a future agent regenerates a dataset in the legacy layout, it still
  works.

## Affected source files

| File | Lines (approx) | What changes |
|---|---|---|
| `results_analysis/canonical_angles/data.py` | 121, 141, 252-254, 391-392, 437, 782-808 | introduce `_derived_*` helpers; replace direct path constructors |
| `results_analysis/compute_combo_marginals.py` | 466-495 (main) | write to derived/{marginals,aggregates,axis}/ instead of flat vectors/ |
| `results_analysis/canonical_angles/plots/ca1_plane_pre_post_shear.py` | 105 (`cv = data_dir / "combinations" / "vectors" / folder`) | use the resolver instead of hardcoded path |

## Migration script (Phase 1.0d) responsibilities

For each dataset that already has post-pipeline derivatives:

1. Create `combinations/vectors/derived/{marginals,aggregates,axis}` (and
   `legacy_centroid/` if any `*_legacy_centroid/` siblings exist).
2. `os.replace()` each of the 7 (or 11 with legacy) items from old to new
   location.  Idempotent: if already moved, skip.
3. On Roger 4-slot only: diff each `* copy/` Finder dir against its
   canonical sibling; if byte-identical, delete; if any diff, halt and
   report.
4. Print a summary: per-dataset list of moved items, deleted Finder copies,
   and any items already in the new layout.
