# PNG provenance audit: `/Users/roger/Documents/GitHub/assistant-axis/runpod_workspace`

Total PNGs: **1**.  Status legend: current / stale / deferred / legacy / frozen (see ``tools/audit_pngs.py`` docstring).

## Summary by status

| Status | Count |
|---|---:|
| current | 0 |
| stale | 0 |
| deferred | 1 |
| legacy | 0 |
| frozen | 0 |

## By producing script

| Script | current | stale | deferred | legacy | frozen | total |
|---|---:|---:|---:|---:|---:|---:|
| `(unknown source)` | 0 | 0 | 1 | 0 | 0 | 1 |

## Deferred PNGs (1)

PNGs that would otherwise be ``stale`` but match an entry in ``deferred_rejudges.yaml``.  Use ``tools/defer_rejudge.py --remove`` to lift.

### `/Users/roger/Documents/GitHub/assistant-axis/runpod_workspace/qwen/assistant-axis/img/assistant_axis.png`
- **Was**: legacy
- **Deferred by**: `runpod_workspace/*` -- Dataset content under runpod_workspace/<dataset>/ is tracked via per-dataset MANIFEST.json rather than per-file _provenance envelopes (see AGENT_NOTES.md 'Manifest-tracked producers').  Freshness is asserted by re-running tools/regenerate_dataset_manifest.py and confirming a byte-identical MANIFEST.json (the manifest is stable on unchanged input).  Audit reports against the analysis tree (roger/) trace freshness through file-level dependencies into these subtrees automatically.  *(at 2026-05-09T04:04:36+00:00)*
- **Deferred by**: `runpod_workspace/qwen/assistant-axis/img/*.png` -- Static logo asset checked into the embedded assistant-axis dataset checkout under runpod_workspace.  Not a producer output.  *(at 2026-05-09T04:04:37+00:00)*
