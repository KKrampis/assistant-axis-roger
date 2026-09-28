---
paths:
- assistant_axis/deferral_registry.py
- assistant_axis/script_equivalence.py
- assistant_axis/rubric_equivalence.py
- tools/audit_*.py
- tools/defer_rejudge.py
- tools/diff_against_recorded.py
- tools/mark_*_equivalent.py
- tools/regenerate_dataset_manifest.py
- deferred_rejudges.yaml
- pipeline/3_judge.py
- assistant_axis/provenance.py
- results_analysis/axis_judge_correlation.py
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: provenance-judge-step

**When:** touching rubric or script equivalence, deferred rejudges, cache audits, or recovery of recorded outputs.  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=provenance-judge-step`; edit there, then run `uv run python tools/sync_agent_notes.py`.

### Judge-step provenance (rubrics, equivalence, deferral, recovery)

Judging is the single expensive non-deterministic step in the
pipeline, so it gets a heavier provenance regime than the rest of
the system.  The four cooperating layers:

#### 1. Producer dependencies (per output mode)

`axis_judge_correlation.py`'s
[`_build_axis_judge_inputs`](results_analysis/axis_judge_correlation.py)
builds an `InputSpec` list per output, one of `descriptions` /
`instructions` / `responses` / `projections` / `correlations`:

- **producer_script** (`kind="file"`): the source of
  `axis_judge_correlation.py` itself, with current judge config
  (provider, model, temperature, max_tokens, batch-size knobs)
  recorded in `extras`.  Editing rubric strings, prompt builders,
  or score-parsing logic flips this fingerprint.
- **axis_file** OR (**pair_vectors** + **pole_instructions**):
  whichever path defined the axis.  Pole text deps are skipped
  when `--pos_pole` / `--neg_pole` were passed on the CLI (the
  text came from args, not from a file).
- **corpus_instructions** (`kind="multi"`): every
  `instructions_dir/{roles,traits}/instructions/*.json` the script
  could read.  Editing one trait JSON's `description` or
  `instruction.pos[*]` field flips the whole multi fingerprint;
  this over-invalidates a little but keeps logic simple.
- **corpus_vectors** (`kind="multi"`): every
  `data_dir/{roles,traits}/vectors/*.pt` (used for projections;
  defines the scorable entity set in every mode).
- **responses-mode extras** (only for `responses` / `correlations`):
  `response_files`, `response_score_files`, `default_responses`,
  `questions_file`.

A `JUDGE PROVENANCE NOTE` comment block at the top of
`axis_judge_correlation.py` lists which kinds of edit are harmless
vs not, and shows the one-line `mark_script_equivalent.py`
invocation to declare the harmless ones.

#### 2. Per-axis consumer granularity

The 5 consumers that read judge caches (`whitening_k_sweep.py`,
`gpt_sonnet_weight_sweep.py`, `batch_size_rho_curve.py`,
`optimal_axis_for_judge.py`, `pc_round_trip/plot_direction_cosines.py`)
record one `current_file_input` *per (axis, mode, judge_source)*
rather than one composite `current_files_input` over the whole fan-
out.  Convention: `dep_key = f"judge_{axis_id}_{mode}_{source}"`
(e.g. `judge_q9_angel_vs_demon_descriptions_gpt`).  Result: an
`audit_caches.py` report on a downstream cache pinpoints exactly
which axis was rejudged rather than smearing the drift over the
whole bundle.

#### 3. Script equivalence (harmless code edits)

`script_equivalences.yaml` at repo root declares per-edit
"output-preserving" pairs for `kind="file"` producers.
[`assistant_axis.script_equivalence.is_equivalent`](assistant_axis/script_equivalence.py)
does a transitive BFS over the registered edges; when
`validate_recorded` sees `kind="file"` drift on a script and the
registry has a chain `recorded_fp → ... → current_fp`, the
status downgrades from `drift` to `equivalent` (counted as ok in
`ProvenanceCheck.ok`).

Workflow after a harmless edit (docstring, type hint, log message,
internal refactor that preserves I/O):

```sh
uv run python tools/mark_script_equivalent.py \
    --script results_analysis/axis_judge_correlation.py \
    --reason "Refactored prompt builder; identical I/O."
```

The CLI auto-detects `--from-fp` from the most recent envelope-
bearing cache that recorded the script (looking under `roger/` and
`runpod_workspace/`).  `--to-fp` defaults to the current on-disk
fingerprint.  `tools/mark_script_equivalent.py --list` shows all
declared edges; `--check` queries an arbitrary edge without
mutating the registry.

#### 4. Deferred rejudging (intentional staleness)

`deferred_rejudges.yaml` at repo root declares "I know this is
stale, don't bug me about it" entries: a `path_glob` (and
optional `dep_key` glob) plus a free-text `reason`.  Both
`audit_caches.py` and `audit_pngs.py` reclassify matched rows to
`deferred`, covering **two pre-deferred statuses**:

- **`stale_*` → `deferred`**: "this would otherwise need
  rerunning; defer the rerun instead".  The original use case —
  for example, a trait edit that drifted one judge cache and we
  don't want to re-judge yet.
- **`legacy` → `deferred`** (May 2026 backfill): "this never had
  an envelope and we've decided not to migrate the producer.
  Presume current as of mechanism introduction".  Used to mark
  pre-Phase-6 judge caches (`scores_*.json`, `projections.json`,
  `correlations.json`, `correlation_plot.png`, `gaps.json`)
  written by `axis_judge_correlation.py` before its Phase 6a
  wrap.  The deferral self-clears the moment the producer is
  re-run with the modern writer pattern (the cache then carries
  an envelope and reads as `current` directly; the registry
  entry becomes a no-op).

`current` and (PNG-only) `frozen` are never reclassified — the
registry only takes effect for rows that would otherwise read as
`stale_*` or `legacy`.  The deferred section in audit reports
shows `pre_deferred_status`, the matching registry entry, and
when it was declared.

Deferrals are *deferred-until-removed* — there's no `expires_at`,
they stay active until the maintainer deletes the entry.  Use
`tools/defer_rejudge.py --list` to inspect, `--remove` to lift.

```sh
# Defer all q9 description rejudges (stale → deferred):
uv run python tools/defer_rejudge.py \
    --path 'roger/axis_judge_experiments/q9_*/gpt/scores_descriptions.json' \
    --reason "Held until 4-model audit completes."

# Backfill: declare all pre-Phase-6 judge caches as legacy → deferred
# (the May 2026 standard reason template):
uv run python tools/defer_rejudge.py \
    --path '**/scores_descriptions.json' \
    --reason "Legacy judge cache pre-dating Phase 6 provenance mechanism (May 2026). Presumed current as of mechanism introduction; cost-prohibitive to re-judge. Auto-clears when re-judged."

# Lift later:
uv run python tools/defer_rejudge.py --remove \
    --path 'roger/axis_judge_experiments/q9_*/gpt/scores_descriptions.json'

# Skip the registry entirely in one run (debug):
uv run python tools/audit_caches.py --ignore-deferrals
```

Audit `--status` semantics:
- `--status stale` matches `stale_direct` + `stale_transitive`
  (does NOT include `deferred`, which is the explicit "skip me"
  bucket).
- `--status deferred` shows just the deferred rows.
- `--status legacy` shows just unmigrated/uncovered rows
  (i.e. legacy that *isn't* matched by a registry entry).
- No `--status` filter shows everything in their respective
  sections.

##### Deferral categories (schema v2, May 2026)

Each entry carries a structured `category` tag drawn from
`DeferralCategory` in
[`assistant_axis/deferral_registry.py`](assistant_axis/deferral_registry.py).
The category drives both audit-report grouping and category-specific
invariant checks in `tools/audit_deferrals.py`:

| Category | Semantics | Auto-clears? |
|---|---|---|
| `legacy_bare` | Pre-Phase-6 cache without an envelope. | When producer is re-run (envelope replaces bare file). |
| `frozen_snapshot` | Point-in-time capture *with a live twin*; comparison plots typically read both. Optional `compares_to` glob points at the live twin. | Never. |
| `archived` | Standalone output of a retired pipeline configuration; no live twin (e.g. pre-cohort-cutover backups, deprecated variants). | Never. |
| `orphan_no_producer` | Output whose producer script is not in the git tree (one-off `/tmp/_*.py`, removed/superseded producers). Optional `producer_script` records the historical path. | Manual; `audit_deferrals` ⚠️ alarms if a recorded `producer_script` reappears in `git ls-files`. |
| `superseded` | Replaced by a named newer artifact (`replaced_by` glob). | When `replaced_by` is removed (audit errors). |
| `operational` | Config / API-usage / tracker files written without an envelope by design. | Never. |
| `manifest_tracked` | Freshness asserted via a per-dataset `MANIFEST.json` rather than per-file envelopes. | Never. |
| `external_pipeline` | Output of a workflow outside the current provenance migration scope (coherence_eval, pc_axis_describer). | Scope decision. |
| `experimental_one_off` | Exploratory artifact with no plan to integrate. | Manual. |
| `hand_curated_input` | Hand-edited config consumed by producers (`pair_list*.json`, embedded logo PNGs). | Never (it's an input, not a producer output). |
| `uncategorized` | Schema-v1 entry that hasn't been backfilled. | The audit raises so backfill is forced. |

The `frozen_snapshot` vs `archived` distinction is intentional even
though both never auto-clear: `frozen_snapshot` implies a live twin
exists, so the audit can verify `compares_to` and warn when paired
comparison plots have lost half their provenance.  `archived` makes
no such claim — it's "this is from a retired pipeline configuration,
nothing to compare against."

When deferring something via the CLI, pass the category explicitly:

```sh
uv run python tools/defer_rejudge.py \
    --path 'roger/.../some_orphan_*.png' \
    --category orphan_no_producer \
    --producer-script /tmp/_old_canonical_angles.py \
    --reason "Pre-Phase-6 one-off; producer no longer in codebase."
```

Optional metadata fields (`--producer-script`, `--replaced-by`,
`--compares-to`) are emitted only when set; the on-disk YAML stays
terse for the common case.  Schema-v1 files (no `category` field)
are still readable; existing entries load as `UNCATEGORIZED` and the
file is upgraded to v2 in place on first write.

##### `tools/audit_deferrals.py` (registry coherence)

Companion to `audit_caches.py` / `audit_pngs.py` — those check
*on-disk artifacts*; this checks *the registry itself*.  Runs four
category-specific invariants:

* **`orphan_promoted`** (error) — an `orphan_no_producer` entry's
  `producer_script` is now tracked by git.  Either the producer was
  promoted (lift the deferral, re-run, let fresh envelopes land) or
  the path collision is coincidental (rename the recorded
  `producer_script` to disambiguate).
* **`superseded_replaced_by_missing`** (error) — `replaced_by` glob
  matches no file on disk.  Either the successor was deleted
  (remove the deferral entry too) or it never got produced
  (regenerate the successor).
* **`superseded_missing_replaced_by`** (warning) — `superseded`
  entry without a `replaced_by` field; consumers can't navigate to
  the successor.
* **`frozen_snapshot_twin_missing`** (warning) — `compares_to` glob
  matches no file; the live twin used by paired comparison plots is
  missing.  The snapshot may belong in `archived` instead.
* **`uncategorized`** (error) — a schema-v1 entry that needs
  backfilling.

```sh
# Markdown roll-up to stdout:
uv run python tools/audit_deferrals.py
# Save to file:
uv run python tools/audit_deferrals.py --output reports/audit_deferrals.md
# CI-friendly (exit non-zero on any error):
uv run python tools/audit_deferrals.py --strict
```

The audit deliberately does NOT inspect file contents — only the
registry plus a snapshot of `git ls-files` and the on-disk path
index.  Cheap to run, ~1s.

##### Apply deferrals BEFORE propagating transitive staleness

Subtle ordering invariant in `tools/audit_caches.py`'s `main()`:
`apply_deferrals(rows)` must run *before*
`propagate_transitive_stale(rows)`.  The propagation BFS only walks
edges from rows whose status is in
`STALE_STATUSES = ("stale_direct", "stale_transitive")`, so deferred
rows act as barriers — a deferred upstream cache no longer falsely
taints downstream consumers as `stale_transitive`.

This was a real bug fixed in May 2026 (Finding 1 of an audit pass):
when the order was swapped, deferring an upstream desc+inst cache
was tainting every downstream rho/whitening sweep transitively, even
though the consumers' recorded fingerprints of the deferred cache
were perfectly current.  Test:
`tools.tests.test_audit_caches.test_deferred_upstream_does_not_taint_downstream`.

#### 5. Recovery: `tools/diff_against_recorded.py`

When a cache reports `stale_direct` on `producer_script`, this
tool reconstructs the diff between the recorded version and the
current one, so the maintainer can decide
"harmless → mark equivalent" vs "real change → rejudge or defer":

```sh
uv run python tools/diff_against_recorded.py \
    roger/axis_judge_experiments/q9_angel_vs_demon/gpt/scores_descriptions.json
```

Resolution order:
1. **Git** (`git show <recorded_sha>:<script_rel>`): preferred
   when the envelope's `produced_by.git_sha` is reachable.  Skip
   with `--no-git`.
2. **Cursor Local History fallback**: walks
   `~/Library/Application Support/Cursor/User/History/<hash>/`,
   finds the snapshot whose `timestamp` is closest to the
   recorded `last_modified_at`, and diffs it against the current
   file.  Skip with `--no-history`.

The tool prints a ready-to-paste `mark_script_equivalent.py`
invocation pre-filled with the recorded `--from-fp`, so a
maintainer who confirms the diff is harmless can declare
equivalence in one paste.

**Cursor Local History tuning** — the default per-file entry cap
is small (~50).  For long-lived editing sessions on
`axis_judge_correlation.py`, bump it via
`workbench.localHistory.maxFileEntries: 500` in
`~/.cursor/settings.json` so the recovery tool can reach further
back.

### Deferred future work — Phase 7 (subtree content hashing)

Originally scoped as Phase 6, demoted to Phase 7, then **deferred**
in May 2026 after a cost/value review.  Today's subtree
fingerprints are metadata-only (sorted `(rel_path, mtime_ns_floored,
size)` triples, SHA-256'd); content hashing would make them
robust to `cp` / `scp` / non-`-t` rsync / `git checkout` / editor
no-op-rewrites and would catch silent bit-rot, at the cost of
making manifest regen 30-300x slower (still <1 minute on local
SSD for the full dataset).

The full write-up — what it solves, what it doesn't, the per-subtree
hashing-cost numbers, the implementation plan, and the decision
criteria for when to actually pick it up — lives in code as the
`TODO(phase-7-deferred)` block at the top of
[`assistant_axis/provenance.py`](assistant_axis/provenance.py),
with a coordinated manifest-regen-side checklist at the
`_summarize` function in
[`tools/regenerate_dataset_manifest.py`](tools/regenerate_dataset_manifest.py).
Pick up Phase 7 if we start moving datasets between machines via
something other than `rsync -at`, if false-positive drift from
no-op-rewriting tooling becomes a recurring nuisance in audit
reports, or if we need a defensible answer to "did this dataset get
corrupted on disk?".  Until then, metadata fingerprints are
strictly cheaper for equal-or-better real-world behaviour in
Roger's workflow.
