---
paths:
- steering/**
- assistant_axis/steering.py
- assistant_axis/steering_runner.py
- assistant_axis/sweep_start_heuristics.py
- data/steering/configs/**
- tools/analyse_start_strength.py
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: steering-runs

**When:** launching, configuring, or debugging steering sweeps (strength scans, start strengths, queue runner, multi-GPU, sweep logs).  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=steering-runs`; edit there, then run `uv run python tools/sync_agent_notes.py`.

### Bidirectional steering scan (May 2026, default)

The steering-strength sweep was switched from a bottom-up
unidirectional scan to a **bidirectional middle-out scan** with two
stop conditions.  Implemented in
[`assistant_axis/steering_runner.py`](assistant_axis/steering_runner.py)
(`BidirectionalCursor`, `_run_bidirectional_cell`), wired through
[`steering/run_sweep.py`](steering/run_sweep.py).  Design plan:
[`/Users/roger/.cursor/plans/bidirectional_steering_scan_e93878b3.plan.md`](.cursor/plans/bidirectional_steering_scan_e93878b3.plan.md).

**Motivation.**  Pre-May-2026 sweeps started at `weakest_strength=1.0`
and walked up in `multiplier=1.189` steps until 2 consecutive
strengths crossed the coherence threshold.  Empirical review of the
14 production experiments before this change:
- Median strength at which `|effect| >= 0.5` is **3-4 multiplier
  steps** above weakest (so the bottom 3 strengths typically carry
  no measurable signal).
- Top strength reached was 15.96 (`architect_ecocentric −1`) -- well
  below the historical `max_strength=64.0` cap.  The UP cap is
  effectively informational; coh-stop is the binding constraint.
- Several cells need 6+ steps up before a meaningful effect appears
  (`mediator_truthful +1`, `journalist_callous +1`); they spend
  similar effort on the low-signal tail that gets thrown away.

**New scan.**

- Centre anchor: `s_init = weakest * multiplier**start_steps_up`
  (default `1.0 * 1.189**2 ≈ 1.414`, i.e. "2 steps above the
  historical floor").
- UP direction: step up from `s_init` in `multiplier` ratios; stop on
  `coh_stop_consecutive` consecutive `mean_coh >= coh_stop_threshold`
  strengths (defaults `2` and `1.5`, unchanged from pre-May 2026).
  Hard cap at `max_strength=64.0`.
- DOWN direction: step down from `s_init` in `multiplier` ratios;
  stop on `eff_stop_consecutive` consecutive
  `mean(|effect.combined|) < eff_stop_threshold` strengths (defaults
  `2` and `0.25`).  Hard floor at `min_strength=0.125` (3 octaves
  below historical weakest=1.0).
- State machine: `BothOpen → {UpBlocked, DownBlocked} → Done`.
  Sticky: a blocked direction never unblocks.  Single-pipeline
  generation alternating UP/DOWN while both open; once one side
  blocks, only the open side steps and we await its judges between
  steps (yellow-mode per direction).
- NaN handling: a DOWN strength whose effect was skipped due to
  high coherence (the `_StrengthState.judging_skipped` path) resolves
  the effect-mean future to **NaN**; the eff-stop tail check ignores
  NaN strengths rather than counting them as zero-effect (which
  would falsely block DOWN early).

**Config.**  All bidirectional knobs surface as both `sweep_cfg`
keys in `steering/run_sweep.py` and kwargs on
`run_steering_cell()`:

| key | default | meaning |
|---|---|---|
| `scan_mode` | `"bidirectional"` | switch to `"legacy_unidirectional"` for old behaviour |
| `start_strength_multiplier_steps` | per-cell heuristic — see below | `s_init = weakest * mult ** N` |
| `min_strength` | `0.125` | DOWN cursor floor |
| `eff_stop_threshold` | `0.25` | per-strength `mean(|effect.combined|)` below which counts toward DOWN stop |
| `eff_stop_consecutive` | `2` | how many consecutive sub-threshold strengths to require |
| (existing `weakest_strength`, `max_strength`, `multiplier`, `coh_stop_threshold`, `coh_stop_consecutive` unchanged) |

**Legacy mode.**  Setting `scan_mode: legacy_unidirectional` in
`sweep_cfg` (or passing `scan_mode="legacy_unidirectional"` directly
to `run_steering_cell`) restores pre-2026-05-14 behaviour
bit-for-bit on the records.jsonl path.  The summary.json now also
records `scan_mode` and `positions_mode` -- additive fields only,
nothing pre-existing is removed.  Test
`TestLegacyModeParity` pins the legacy invariants
(visited-strengths order, completion semantics, summary fields).

**Restart semantics.**  `summary.json` now persists `scan_mode`; on
restart, the persisted mode overrides the caller's request (with a
warning) so mid-run mode switches don't corrupt the schedule.  For
mid-run crashes with `records.jsonl` present but no `summary.json`,
the bidirectional cell re-walks the cursor past existing strengths
on each side (cursor is deterministic, so re-emission matches
prior emission), bootstraps `mean_coh_by_strength` and
`mean_abs_eff_by_strength` from the on-disk records, then
re-evaluates both stop conditions against the resumed history
before generating any new strengths.  Test
`TestBidirectionalRestart` covers this path.

**summary.json schema** (bidirectional cells):

```json
{
  "scan_mode": "bidirectional",
  "positions_mode": "all",
  "s_init": 1.413721,
  "weakest_strength": 1.0,
  "max_strength": 64.0,
  "min_strength": 0.125,
  "multiplier": 1.189,
  "start_strength_multiplier_steps": 2,
  "eff_stop_threshold": 0.25,
  "eff_stop_consecutive": 2,
  "coh_stop_threshold": 1.5,
  "coh_stop_consecutive": 2,
  "up_blocked_reason": "incoherent" | "max_strength_reached" | null,
  "down_blocked_reason": "sub_threshold_effect" | "min_strength_reached" | null,
  "up_blocked_at_strength": 8.0,
  "down_blocked_at_strength": 0.25,
  "records_in_scan_order": true,
  ...legacy fields (slot, layer, sign, n_records, etc.)
}
```

Downstream `results_analysis/*` plot scripts already group records
by `record["strength"]` so disk order doesn't matter -- the
`records_in_scan_order: true` field is purely documentary.

### Per-cell start-strength heuristic (May 2026)

Before 2026-05-24 the bidirectional sweep used a flat
`start_strength_multiplier_steps=2` (so `s_init ≈ 1.41` at
`weakest=1.0, mult=1.189`) for every cell.  Empirically this is
sub-optimal: a 310-cell audit
([`tools/analyse_start_strength.py`](./tools/analyse_start_strength.py))
showed **49% of all-mode** cells started in the productive band but
**only 11% of prefill-mode** cells did, with 89% of prefill cells
starting TOO_LOW (the downward walk immediately eff-stopped at a
barren region) and ~25% of `(slot=7, layer=49, all-mode)` cells
starting TOO_HIGH (the upward walk immediately coh-stopped at an
already-incoherent region).

The fix is a per-cell lookup table in
[`assistant_axis/sweep_start_heuristics.py`](./assistant_axis/sweep_start_heuristics.py),
keyed on `(positions_mode, slot, layer)` with per-mode defaults:

| mode | slot | layer | start_steps | s_init | rationale |
|---|---:|---:|---:|---:|---|
| all | 0 | 25 | 5 | 2.38 | eff weak (~0.4) at default → bump |
| all | 0 | 31 | 4 | 2.00 | eff borderline → small bump |
| all | 0 | 49 | 0 | 1.00 | high-effect already; lower to cut TOO_HIGH risk |
| all | 6 | 25 | 6 | 2.83 | weakest all-mode cell (eff ~0.26) |
| all | 7 | 25 | 7 | 3.36 | weak (eff ~0.26) |
| all | 7 | 49 | **−3** | 0.59 | hottest cell — coh tripped at default in ~25% of axes; needs sub-weakest start (requires `start_steps_up < 0`, allowed 2026-05-24; floor is now `min_strength`) |
| all | any other | | 3 (default) | 1.68 | |
| prefill | 0 | 25 | 9 | 4.95 | eff ~0.2; need 3-5× higher |
| prefill | 0 | 49 | 5 | 2.38 | layer-49 prefill is closer to cliff; modest bump |
| prefill | 6 | 25 | 9 | 4.95 | weakest prefill cell |
| prefill | 6 | 49 | **1** | 1.19 | **layer 49 prefill is already near coh cliff** (median coh ~0.5 at default); needs LOWER start than default |
| prefill | 7 | 25 | 9 | 4.95 | weak (eff ~0.18) |
| prefill | 7 | 49 | **1** | 1.19 | same as (prefill, 6, 49): near-cliff already |
| prefill | any other | | 6 (default) | 2.67 | |

Counter-intuitive pattern: **layer-49 prefill cells start LOWER than
the prefill default**, because prefill steering at deep layers
already has coherence near the cliff at weak strengths (median coh
~0.5 at `s_init = 1.41`).  Pushing prefill higher to fix the
weak-effect cells would push deep-layer prefill over.

**Negative start_steps.**  The cursor previously asserted
`start_steps_up >= 0`; relaxed 2026-05-24 because the (all, 7, 49)
override needs `s_init < weakest`.  The actual safety floor is
`s_init >= min_strength` (cursor's `__init__` enforces this).

**Override semantics.**  If the YAML explicitly sets
`sweep.start_strength_multiplier_steps`, that value wins for every
cell (so one-off experiments can still pin a value).  Per-cell YAML
override is *not* supported -- the table is the per-cell knob.

**Re-tuning.**  Re-run
[`tools/analyse_start_strength.py`](./tools/analyse_start_strength.py)
after every meaningful sweep batch (e.g. when sweep #2 second-role
data lands and we have ~2× more cells per bucket).  Inspect the
`(slot, layer, sign, mode)` table at the end of the report; if a
bucket newly shifts away from GOOD-dominant, add or adjust an
`OVERRIDES` entry and ship a one-line table edit.  No code or
runner changes needed.

**What about scan_mode legacy?**  The lookup applies only to
`scan_mode="bidirectional"` (the default).  Legacy unidirectional
sweeps use the cell-level `start_strength_multiplier_steps` value
directly without consulting the heuristic.

### Sweep log location (May 2026)

`steering/run_sweep.py` now auto-attaches a `FileHandler` to
`{output_root}/sweep.log` on both the parent and every spawned
worker, so the sweep log lives **inside the experiment dir**
alongside `config.json` / `questions.json` /
`persona_system_prompt.txt`.  Pre-May-2026 sweeps wrote logs
wherever the shell-redirect landed
(`/workspace/assistant-axis/sweep_<exp>.log`); those were moved
into their experiment dirs by
[`scripts/move_sweep_logs_into_experiment_dirs.sh`](scripts/move_sweep_logs_into_experiment_dirs.sh)
(one-shot, idempotent).

Future runs no longer need shell-redirect to capture logs.
Recommended runpod invocation:

```bash
python -m steering.run_sweep --config data/steering/configs/foo_v1.yaml &
tail -F /workspace/outputs/qwen-3-32b/steering/foo_v1/sweep.log
```

The two-run journalist + merchant sweeps preserve both runs as
`sweep_1.log` and `sweep_2.log` in the experiment dir; one-run
experiments use the canonical `sweep.log` name.  Workers and
parent share the same `sweep.log` via `mode='a'` line-buffered
appends -- safe on Linux because steering log records are well
under PIPE_BUF (4096B), so no mid-line interleaving is possible.

### Multi-config queue runner (May 2026)

`steering/run_sweep.py --config` now accepts **multiple `--config`
flags** (repeatable) and a `--config-list FILE` (one path per line,
`#` comments OK).  When given more than one config, the runner
loads the model **once**, then drains a single combined work queue
across every queued experiment.  Saves the cold-import + 60+ GB
model-load cost per extra config -- a 12-experiment batch goes
from `12 * (cold imports + model load + run)` to
`(cold imports + model load) + 12 * run`.

All queued configs must share `model_name` (one model fits in GPU
memory at a time); mixed-model queues raise loudly at startup.

Per-experiment `sweep.log` files still partition cleanly: each
work item carries its own `sweep_log_dir` and the worker swaps
`FileHandler`s (`_detach_sweep_log_filehandler` +
`_attach_sweep_log_filehandler`) when transitioning between
experiments.  Per-experiment `config.json` / `questions.json` /
`persona_system_prompt.txt` are still atomically written into the
experiment dir during parent-side setup, before the work queue
starts draining.

Typical batch invocation:

```bash
python -m steering.run_sweep \
    --config-list data/steering/configs/multi_cell_batch_v1.txt \
    --gpus 4
```

where `multi_cell_batch_v1.txt` is just:

```
data/steering/configs/anthropologist_helpful_v1.yaml
data/steering/configs/prodigy_harmless_v1.yaml
data/steering/configs/architect_ecocentric_v3.yaml
# ... 9 more
```

### Multi-GPU safety: baselines sentinel (May 2026)

Cells depend on their config's `baselines/records.jsonl` existing
(read by `build_baseline_lookup` to fill `baseline_response` in
judge prompts).  With `--gpus N` workers draining a shared queue,
a naive ordering would let worker B pop a cell for config X while
worker A is still computing config X's baselines -- the cell would
silently judge with blank `baseline_response`s.

The runner now:

1. **Sorts all baselines to the head of the queue.**  With N
   workers, the first N items popped are baselines, parallelising
   baseline throughput.
2. **Touches `baselines/.complete` after `compute_baselines`
   succeeds** (both the generation path and the "all already on
   disk" resume fast path -- the latter so resumed runs that
   missed the sentinel in a prior session still produce it).
3. **Workers wait on the sentinel before dispatching a cell**:
   if the cell's `baselines/.complete` doesn't yet exist, the
   worker polls every 2s with a 20-minute hard timeout.  Worst
   case waste with 4 GPUs + 12 configs is ~90 GPU-seconds per
   config = ~18 GPU-minutes across the batch; effectively zero
   compared to the ~hours of generation+judging.

Pre-May-2026 single-GPU sweeps never hit this race (one worker,
sequential order).  Pre-existing multi-GPU sweeps would have
quietly corrupted judge prompts on the first cell of each
experiment; the sentinel fix is also a pre-existing-bug fix.
