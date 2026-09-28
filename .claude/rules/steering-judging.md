---
paths:
- assistant_axis/steering_judges.py
- assistant_axis/cherrypick.py
- assistant_axis/steering_runner.py
- steering/post_judge.py
- results_analysis/steering_response_curves.py
- tools/sheet_layout.py
- tools/test_effect_order_bias.py
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: steering-judging

**When:** judging steering outputs (effect and coherence judges, swap-averaging, response curves).  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=steering-judging`; edit there, then run `uv run python tools/sync_agent_notes.py`.

### Steering judges (Phase-2 architecture)

**Activation space (read this first).**  Steering vectors live in
**raw model-activation space** — the native frame the model
operates in.  `ActivationSteering` (in
[`assistant_axis/steering.py`](assistant_axis/steering.py)) and the
sweep runner ([`steering/run_sweep.py`](steering/run_sweep.py)) load
the unwhitened `axis.pt` directly and add it to the residual stream
at the chosen layer; **no whitening, no shear**.  The
whitening / soft-shear regimes documented in *"Whitening / soft-shear
defaults"* below are **analysis-side only** (they sharpen
judge-projection ρ) and explicitly do not apply here.  When you
load a judging-side axis vector for use as a steering vector, pass
it through `ActivationSteering` *as-is* — whitening or shearing
first would push it out of the model's native frame and degrade
both the steering and any safety-relevant invariants
(e.g. activation-capping thresholds calibrated against the
distribution of raw projections).  Same reason
`pc_round_trip/launch_judge_runs.py` keeps its own
`DEFAULT_SHEAR_L=0` — see the caveat at the end of the whitening
section.

Steering sweeps use a **two-tier judging** protocol baked into
[`assistant_axis/steering_judges.py`](assistant_axis/steering_judges.py)
and consumed by [`assistant_axis/steering_runner.py`](assistant_axis/steering_runner.py)
via the `JudgeDispatcher` protocol:

1. **Coherence (synchronous, blocking)** — per-record `judge_coherence_blocking()`
   call after each generation batch.  0-3 rubric.  Default model
   `gpt-4.1-mini`; configurable via `--coherence-model`.  Gates the
   `should_stop_at(strength)` early-termination decision.
2. **Persona / RP (asynchronous, post-batch)** — gpt-4.1-mini, unbatched,
   matches `pipeline/3_judge.py`'s rubric augmented with steering-direction
   awareness.  0-3 rubric.
3. **Effect (asynchronous, batched)** — GPT+Haiku ensemble, batched at
   `target_batch_size=10` (matches response judging's default), strict
   single-(cell, sign, strength) batches.  `--effect-mode` selects
   bidirectional (one ±3 prompt) vs separate-poles (two 0-3 prompts,
   signed combine = pos.mean - neg.mean) vs both.

**Skip rule (cost optimization)**: tiers 2 and 3 are **skipped at the
(cell, sign, strength) granularity** when `mean(coh) > skip_threshold`
(default 1.0).  Filtering on the strength-mean rather than per-record
coh reflects that incoherence is a strength-level property whereas
judge noise is per-question; averaging the K coh scores cancels noise
without losing the gross signal.  Skipped records get
`judges.persona.skipped_due_to_strength_mean_coh = True` (and same for
effect); `strength_mean_coh` is stamped on every record regardless.

**Records schema** (`{cell_dir}/records.jsonl`):

```json
{
  "strength": 1.0, "sign": +1, "question_idx": 7, ...,
  "judges": {
    "coherence": {"score": 1, "reason": "...", "model": "...", "ts": ..., "rubric_version": 1},
    "strength_mean_coh": 0.7,
    "persona": {"score": 2, "model": "...", "skipped_due_to_strength_mean_coh": false, ...},
    "effect": {
      "mode": "bidirectional",
      "bidirectional": {"scores": {"<model_a>": {"score": 2, "reason": "..."},
                                    "<model_b>": {"score": 3, "reason": "..."},
                                    "mean": 2.5}},
      "combined": 2.5,
      "skipped_due_to_strength_mean_coh": false, ...
    }
  }
}
```

**CLIs**:

- [`steering/run_sweep.py`](steering/run_sweep.py) — live sweep with
  judging.  `--no-live-judging` falls back to `NoOpJudgeDispatcher`
  (no API calls; useful when iterating on generation alone).
  `--judges-only` skips generation entirely and delegates to
  `post_judge.py` for retrospective fill-in.
- [`steering/post_judge.py`](steering/post_judge.py) — retrospective
  RP/effect fill-in on an existing experiment dir.  Same skip rule;
  `--rerun-coherence-with-model` populates `judges.coherence_alts[<model>]`
  for the coherence-model shootout (preserves the live coherence
  field).
- [`assistant_axis/cherrypick.py`](assistant_axis/cherrypick.py) — walks
  one or more experiment dirs, applies a default filter
  (`strength_mean_coh<=1.0 AND persona>=2 AND |effect.combined|>=2`)
  or a Python-expression filter, prints a summary table and pretty
  spotlights for the top N by |effect.combined|.

**Cost characteristic**: coherence dominates the bill because it's
unfiltered (every record gets a call).  RP+effect are skipped on
incoherent strengths, so they're typically <30% of total spend.
gpt-4.1-mini default is ~5x cheaper than Sonnet on the unfiltered hot
path; rationale documented in
[`/Users/roger/.cursor/plans/steering_judges_phase2_a54a28b6.plan.md`](.cursor/plans/steering_judges_phase2_a54a28b6.plan.md).

**Known judging artifacts:**

- **Concise-direction RP underscoring**: When the concise/verbose axis
  (or any axis that produces very short responses) is steered toward
  brevity, the RP judge gives low persona scores (e.g. 1/3) even when
  the response content is perfectly in-character.  The judge simply
  can't find enough textual evidence of persona in a 3-8 word response.
  This is a measurement ceiling, not a real persona loss.  Expect
  similar artifacts from any axis that drastically reduces output
  length (e.g. terse/elaborate, laconic/expansive if those ever exist).

**Sigma-rescaling (deferred)**: effect scores are stored on the raw
±3 scale.  A separate downstream tool will eventually consume
`records[*].judges.effect.combined` and rescale to standard deviations
of the response-judging distribution.  The default
`response_target_batch_size=10` (in
[`results_analysis/axis_judge_correlation.py`](results_analysis/axis_judge_correlation.py))
matches the steering effect default so the two distributions are
apples-to-apples in the same judging regime.

### Per-cell response curves vs steps-to-incoherence (May 2026)

[`results_analysis/steering_response_curves.py`](results_analysis/steering_response_curves.py)
plots one figure per steering experiment with 7 cells × 2 signs × 4
filter regimes per cell.  The X axis is "steps remaining to
incoherence cliff" (0 at the right edge), aligning every cell at its
own incoherence point so cells with different absolute cliff
strengths are visually comparable.

The four filter regimes per (cell, sign) probe whether the average
effect score at each strength is being dragged around by noisy
responses:

  - **(a) all responses** -- raw mean of effect scores at that
    strength.  Solid line, weight 1.6.
  - **(b) coh=0 only** -- mean over responses judged perfectly
    coherent.  Dashed (`(0, (5, 2))`).
  - **(c) rp=3 only** -- mean over responses strongly in-persona.
    Dotted (`(0, (1, 2))`).
  - **(d) coh=0 & rp=3** -- mean over responses that are both
    perfectly coherent AND strongly in-persona.  Dash-dot
    (`(0, (3, 2, 1, 2))`).

Diverging curves between (a) and (d) indicate strength regions where
some questions are giving misleading judge scores (incoherent or
off-persona responses contaminating the mean).  Convergence between
(a) and (d) is a sanity check that the population average reflects
the in-persona / coherent population.

Two side-by-side subplots:
  - **Left**: sign=+1 (steers toward `role_to` = negative pole).
  - **Right**: sign=-1 (steers toward `role_from` = positive pole).

Legend on the left subplot maps colour to `(slot, layer)`; legend on
the right maps linestyle to filter regime, so you don't have to
mentally cross-reference between the two.  See
[`results_analysis/tests/test_steering_response_curves.py`](results_analysis/tests/test_steering_response_curves.py)
for the aggregation-logic test suite.

Typical invocation:

```bash
uv run python results_analysis/steering_response_curves.py \
    outputs/qwen-3-32b/steering/anthropologist_helpful_v1
```

Output: `<experiment_dir>/response_curves.png` (override with
`--output PATH`).  PNG carries the standard
[plot-provenance metadata](#plot-provenance-metadata-mandatory-for-every-plot)
chunk.

### Swap-averaged effect judging (rubric v7, May 2026)

The bidirectional effect rubric has a strong label/order bias: the
judges treat whatever sits in the `[RESPONSE]` block as "more
thoroughly developed" relative to whatever sits in the `[BASELINE]`
block, even when the texts are swapped between those slots.
Empirically on anthropologist_helpful_v1 + prodigy_harmless_v1 the
bias dominated content-driven scoring at low/mid steering strengths
across both safety axes (helpful/unhelpful and harmless/harmful).
Diagnostic: [`tools/test_effect_order_bias.py`](tools/test_effect_order_bias.py)
swaps the [BASELINE] and [RESPONSE] contents in the rubric and
compares; a content-driven judge would produce `swap_score =
-straight_score`, a label-biased judge produces `swap_score =
straight_score`.

**Solution (rubric v7)**: every bidirectional effect judging call
fires BOTH variants and records the bias-cancelling estimator

```
averaged = (straight.mean - swap.mean) / 2
```

Under a content-driven judge `swap = -straight` so `averaged ==
straight` (signal preserved).  Under pure label bias `swap ==
straight` so `averaged = 0` (bias cancelled).  Real judges sit
between these extremes; the estimator subtracts off whatever
constant additive bias the judge has.

#### Schema (records.jsonl)

```json
"effect": {
  "bidirectional": {
    "straight": {"scores": {"<model>": {"score": int, "reason": str},
                            "mean": float, ...}},
    "swap":     {"scores": {"<model>": {"score": int, "reason": str},
                            "mean": float, ...}},
    "averaged": float
  },
  "mode": "bidirectional",
  "combined": float,           // = averaged (preferred) ->
                               //   straight.scores.mean (swap missing) ->
                               //   legacy bidirectional.scores.mean (v6)
  "rubric_version": 7,
  "skipped_due_to_strength_mean_coh": bool,
  "ts": float
}
```

Migration helper: [`assistant_axis.steering_judges.migrate_effect_dict_in_place`](assistant_axis/steering_judges.py).
Called automatically on every dispatcher write so legacy v6 records
end up in v7 shape the first time they're touched.  Idempotent:
already-v7 dicts return unchanged.

The bidirectional cursor's eff-stop predicate
([`assistant_axis/steering_runner.py`](assistant_axis/steering_runner.py))
reads `judges.effect.combined` to populate
`mean_abs_eff_by_strength`; no predicate-logic change was needed,
just the input semantics shifted from straight-only to
bias-cancelled.

#### Filling in older experiments

[`steering/post_judge.py`](steering/post_judge.py) gained two new
flags for the post-batch fill-in:

  - `--effect-swap-fill` runs ONLY the swap rubric on records with a
    straight half on disk, populates `bidirectional.swap` +
    `bidirectional.averaged`, sets `combined = averaged`, bumps
    `rubric_version` to 7.  Skips records lacking a straight half
    (skipped at sweep time due to high mean_coh).  Reuses existing
    straight scores -- no re-judging of straight.
  - `--apply-corrected-stop-cutoff` walks each cell's strengths in
    descending magnitude order and applies the live runner's
    bidirectional eff-stop predicate against the new averaged eff.
    The K-consecutive below-threshold streak (default K=2, threshold
    0.5) identifies the bias-floor anchor; strengths SMALLER than
    the anchor move from `records.jsonl` to `_excluded_records.jsonl`
    in the same cell dir.  `summary.json` gains `excluded_strengths`,
    `excluded_reason`, `excluded_eff_stop_threshold`,
    `excluded_eff_stop_consecutive`.  Idempotent on re-run.

Typical invocation:

```bash
uv run python steering/post_judge.py \
    --experiment_dir outputs/qwen-3-32b/steering/anthropologist_helpful_v1 \
    --effect-swap-fill \
    --apply-corrected-stop-cutoff
```

Cost reference: ~$10-15 API + ~3-10 min wall-clock per experiment
on the v6-anthropologist scale (14 cells × ~20 strengths).

#### Cost trade-off in the live pipeline

The dispatcher now fires 2× as many effect-judging calls per
strength (straight + swap × 2-model ensemble = 4 API calls instead
of 2).  Roughly cost-neutral with the buggy-predicate pre-v7
pipeline because the corrected stop fires earlier on the DOWN side
(observed ~36% strength-count reduction across anthropologist +
prodigy's swap descent), and that saving offsets the doubled
effect-judging cost.  See the original plan
[`swap_averaged_effect_judging_5af7c7a6.plan.md`](.cursor/plans/swap_averaged_effect_judging_5af7c7a6.plan.md)
for the cost band breakdown.

#### Analysis surfaces (no code changes needed)

Both [`tools/sheet_layout.py`](tools/sheet_layout.py) and
[`results_analysis/steering_response_curves.py`](results_analysis/steering_response_curves.py)
read `judges.effect.combined` for the eff value -- the same field
that now carries the averaged (bias-cancelled) estimator post-fill.
No analysis code changes were needed for v7 records; the change
percolates through the existing readers.
