# Results Analysis

Scripts that analyze outputs of the [`pipeline/`](../pipeline/) — vectors, scores,
responses, axes. Parallel in intent to [`data_analysis/`](../data_analysis/),
which _prepares_ the role/trait data for the pipeline.

## Convention: tuned mixing ratios for judge ensembles

Four families of weights govern how this project reduces multiple
judge × mode scores to a single per-entity scalar.  All four live
as named constants in
[`assistant_axis/judge_score_combine.py`](../assistant_axis/judge_score_combine.py)
(single source of truth; that module's docstring carries the full
empirical derivation for each value).  Import the constants instead
of re-declaring magic floats so future re-tunings propagate
everywhere with one edit.

```python
from assistant_axis.judge_score_combine import (
    DEFAULT_DI_WEIGHTS,             # 0.499 desc / 0.501 inst   (within one judge)
    DEFAULT_GPT_SONNET_DI_WEIGHT,   # 0.625 GPT / 0.375 Sonnet  (within desc+inst ensemble; v2 retune 2026-05-12)
    DEFAULT_GPT_HAIKU_Q9_WEIGHT,    # 0.525 GPT / 0.475 Haiku   (within response ensemble; 22-axis retune 2026-05-22)
    DEFAULT_RESPONSE_DI_WEIGHT,     # 0.80 response / 0.20 DI   (final blend)
    combine_desc_inst_two_judges,
)
```

| # | Constant | Default | What it weights | Tuning script | Operating cell |
|---|---|---|---|---|---|
| 1 | `DEFAULT_DI_WEIGHTS` | `(0.499, 0.501)` | desc / inst within one judge | (manual sweep, ablations encoded as the `inst_tie`/`equal`/`desc_tie` choices) | slot=(3,25) / (0,26) / (0,49) |
| 2 | `DEFAULT_GPT_SONNET_DI_WEIGHT` | `0.625` | GPT / Sonnet within the desc+inst ensemble (per mode) (was `0.50` until 2026-05-12; bumped to the soft_shear=3 discrete grid peak w=0.625) | `gpt_sonnet_weight_sweep.py --whitening soft_shear=3` | slot 6 / layer 25 |
| 3 | `DEFAULT_GPT_HAIKU_Q9_WEIGHT` | `0.525` | GPT / Haiku within the response-mode ensemble (was `0.60` pre-Phase-5d; `0.41` on the raw v2 sweep; `0.625` on the canonical-whitening v2 sweep at 12 axes; retuned to **`0.525`** on the 22-axis cohort 2026-05-22 once the Phase-1/2 axes landed) | `gpt_anthropic_response_weight_sweep.py --rubric v2 --whitening soft_shear=3` | slot 6 / layer 25 |
| 4 | `DEFAULT_RESPONSE_DI_WEIGHT` | `0.80` | response ensemble / desc+inst ensemble in the final per-entity score | `response_di_weight_sweep.py` | slot 6 / layer 25 |

### 1. Per-judge desc / inst (inst-tiebreak)

`(0.499, 0.501)` is essentially equal but **acts as a tiebreaker**
for entities where desc and inst disagree, leaning slightly toward
instructions.  Empirical rationale: at slot=(3,25) / (0,26) /
(0,49), inst-tiebreak gave consistently higher mean activation→
judge ρ than equal weighting (~+0.0005 ρ), and equal in turn beat
desc-tiebreak by a similar margin — the ordering is monotonic
across all (slot, layer) configurations tested.  Matches the
desc/inst judge audit that found instruction-mode judging more
reliable than description-mode (~99% defensible vs ~94%).

```python
scores = combine_desc_inst_two_judges(g_d, g_i, s_d, s_i)
# equivalent to: 0.499 * (g_d + s_d)/2 + 0.501 * (g_i + s_i)/2
```

CLI ablation on any consumer:

    --di_weights {inst_tie,equal,desc_tie}   # default: inst_tie

`equal` reproduces the historical 0.5/0.5 weighting; `desc_tie` is
the symmetric ablation.  Absolute ρ shift across the three is in
the third decimal — *direction* comparisons are insensitive to
the choice.

Consumers today: `rho_by_slot_and_K.py`, `rho_by_layer.py`,
`whitening_k_sweep.py`, `gpt_sonnet_weight_sweep.py`,
`response_di_weight_sweep.py`, `rubric_v1_v2_compare.py`,
`judge_ensemble_rho_curve.py`.

### 2. Within the desc+inst ensemble (`DEFAULT_GPT_SONNET_DI_WEIGHT`)

`0.625` weight on GPT, `0.375` on Sonnet, applied per-mode (desc and
inst separately) before the desc/inst tiebreak combination:

```python
di = combine_desc_inst_two_judges(g_d, g_i, s_d, s_i)
# Equivalent to:
#   desc_avg = 0.625 * g_d + 0.375 * s_d
#   inst_avg = 0.625 * g_i + 0.375 * s_i
#   score    = 0.499 * desc_avg + 0.501 * inst_avg
```

Selection history:

* **0.50** (Apr 2026, pre-canonical-whitening): 33-axis ``--rubric v1``
  raw-projection sweep at slot 3 / layer 25.  Parabolic interior peak
  at **w = 0.530**, mean ρ = 0.5949 — 0.0004 below the discrete 50/50
  value 0.5953.  The flat curve (±0.0004 over all of
  `w ∈ [0.1, 0.9]`) made 0.50 the natural round-number pick, with the
  bonus of reproducing the historical 4-way mean
  `(g_d + g_i + s_d + s_i) / 4` when paired with `DI_WEIGHTS_EQUAL`.

* **0.625** (2026-05-12, canonical-whitening retune): re-tuned on the
  35-axis ``--whitening soft_shear=3`` view at slot 6 / layer 25 (the
  current canonical operating point).  Discrete grid peak at
  **w = 0.625**, ρ ≈ 0.70198 (w_step = 0.025).  Parabolic interior
  fit peaks slightly higher at w = 0.700 (R²=0.987), but the upper
  plateau `w ∈ [0.575, 0.700]` is flat within 0.0003 ρ — entirely
  inside the ~±0.04 95% CI half-width (n=35 axes, per-axis stdev
  0.124).  The discrete peak and the parabolic fit are statistically
  tied across the upper half of the plateau, so we pick the literal
  discrete peak.  Under whitening the GPT-favoring axes outvote the
  Sonnet-favoring ones more strongly because soft-shear cleans away
  the goal/no-goal-overlap noise that was previously masking GPT's
  per-axis advantage.

The function takes an optional `gpt_sonnet_weight` parameter for
ablations; at `0.5` it's mathematically identical to the historical
hardcoded `(gpt + sonnet) / 2` averaging that pre-dated the
parameterisation.  No CLI knob today (the empirical case for
re-tuning is essentially nonexistent at the operating point); add
one if a future sweep with new judges or new axes shifts the peak
outside `[0.45, 0.55]`.

### 3. Within the response ensemble (`DEFAULT_GPT_HAIKU_Q9_WEIGHT`)

`0.525` weight on GPT-4.1-mini, `0.475` on Haiku for the
response-mode ensemble (as of 2026-05-22; was `0.625` on the
original 12-axis cohort, retuned after the Phase-1/2 expansion
to 22 axes):

```python
response = {
    n: DEFAULT_GPT_HAIKU_Q9_WEIGHT * gpt_resp[n]
       + (1 - DEFAULT_GPT_HAIKU_Q9_WEIGHT) * haiku_resp[n]
    for n in set(gpt_resp) & set(haiku_resp)
}
```

**Selection history.**  Bounced from `0.60` to `0.41` and back to
`0.625` over the May 2026 retunes:

- **0.60 (Apr 2026)** picked from the v1 sweep at slot 6 / layer 25
  on the legacy `gpt_responses_*_b10` / `haiku_responses_*_b10_q9`
  cohorts (`gpt_anthropic_response_weight_sweep.py --rubric v1`; plot
  at `gpt_haiku_q9_response_weight_sweep_slot6.png`).  12-axis
  parabolic peak at **w ≈ 0.609** → rounded to 0.60.

- **0.41 (May 11 2026, raw-projection v2)** re-tuned on the v2 view
  at the same (slot, layer) using `--rubric v2` (GPT at B=7 Phase-5c
  full-volume vs Haiku at B=7-t3 surgical with B=10-q9 per-entity
  fallback; plot at
  `gpt_haiku_q9_response_weight_sweep_slot6__rubric_v2.png`).
  Parabolic peak at **w ≈ 0.410** at *raw projection*.  The leftward
  shift reflects Haiku's improved data quality after Phase-5d
  escalated the ~34 RP-depleted entities per axis from q9-uniform to
  tiered-M=3.  Pure-Haiku ρ went up by +0.041 between regimes (v1
  pure-Haiku ρ=+0.690 → v2 ρ=+0.731) while pure-GPT stayed flat, so
  the ensemble wanted more Haiku.  Superseded by the next entry
  once whitening became the project canonical.

- **0.625 (May 12 2026, canonical-whitening v2, 12 axes)** re-tuned
  on the v2 view at the canonical whitening regime `--whitening
  soft_shear=3` on the original 12-axis cohort (plot at
  `gpt_haiku_q9_response_weight_sweep_slot6_softshear3__rubric_v2.png`
  in its pre-2026-05-22 form).
  Discrete grid peak at **w = 0.600** (ρ = 0.7524, w_step = 0.025);
  parabolic interior fit peaks slightly higher at **w = 0.624**.
  Plateau `w ∈ [0.50, 0.70]` flat within 0.0006 ρ — entirely
  inside the ~±0.055 95% CI half-width (n=12 axes, per-axis stdev
  0.097), so 0.600 / 0.624 / 0.625 are statistically tied.  Picked
  `0.625` for symmetry with `DEFAULT_GPT_SONNET_DI_WEIGHT` (which
  independently lands its discrete grid peak there): under
  canonical whitening every judge-pair weight sweep we've run lands
  its peak within a few percent of 0.625, reflecting that
  soft-shear cleans away goal/no-goal-overlap noise that previously
  dragged the optimum toward whichever judge handled that noise
  better.  Pure-Haiku ρ under whitening is +0.7267 (vs +0.731 at
  raw); pure-GPT is +0.7329; peak ρ at w=0.6 is +0.7524 — a real
  improvement on top of Phase-5d's gains, not just a re-weighting.

- **0.525 (May 22 2026, 22-axis retune)** re-tuned on the same v2 /
  soft_shear=3 view but on the **22-axis cohort** (the original 12
  + 10 Phase-1/2 personality-style axes — proactive, descriptive,
  religious, fragile, introverted, benign, accessible, precise,
  blunt, earnest — all now response-judged at the canonical B=7
  cohort).  Discrete grid peak at **w = 0.525** (ρ = 0.7521,
  w_step = 0.025); parabolic interior fit peaks at **w = 0.472**
  (ρ = 0.7515, R²=0.927).  Plateau `w ∈ [0.375, 0.575]` flat within
  0.0013 ρ.  Plot at the same path
  (`gpt_haiku_q9_response_weight_sweep_slot6_softshear3__rubric_v2.png`,
  now regenerated).
  The shift from 0.625 → 0.525 reflects **Haiku becoming the better
  single-judge baseline on the expanded cohort**: pure-Haiku ρ =
  0.7402, pure-GPT ρ = 0.7088 (inverted from the 12-axis ranking
  where GPT was slightly ahead).  This is sweep #2 of the
  ``Re-tuning checklist`` (below) and is **independent** of any
  desc+inst-vs-response-quality bias the new axes may carry — the
  sweep compares GPT_response vs Haiku_response only.

Haiku-q9 remained the operating-point winner on cost-per-quality
across the 4-Pareto-set view
(`batch_size_curve_8slot/batch_size_cost_vs_quality.png`) for the
v1 era; for v2 the operating-point geometry shifts and the Pareto
needs a refresh too (queued).

The constant's name retains `_Q9` for back-compat; the v2 operating
point is actually the mixed `_b7_t3 ⇢ _b10_q9` cohort.  Re-tune only
if a future sweep with substantially different cohorts shows the
peak shifting outside `[0.50, 0.75]`.

CLI override on `judge_ensemble_rho_curve.py`:

    --gpt_weight FLOAT   # default: DEFAULT_GPT_HAIKU_Q9_WEIGHT (= 0.625)

### 4. Response × desc+inst final blend (`DEFAULT_RESPONSE_DI_WEIGHT`)

`0.80` weight on the response ensemble, `0.20` on the desc+inst
ensemble in the final per-entity score:

```python
di = combine_desc_inst_two_judges(g_d, g_i, s_d, s_i)
final = {
    n: DEFAULT_RESPONSE_DI_WEIGHT * response[n]
       + (1 - DEFAULT_RESPONSE_DI_WEIGHT) * di[n]
    for n in set(response) & set(di)
}
```

Picked from the response × desc+inst weight sweep at slot 6 /
layer 25 (`response_di_weight_sweep.py`; plot at
`roger/axis_judge_experiments/response_di_weight_sweep_slot6.png`).
Two parabolic fits informed the round number:

- **12-axis** (all response-mode axes): peak at `w ≈ 0.84`,
  ρ ≈ 0.763.
- **11-axis** (excluding `ecocentric_vs_anthropocentric`, the
  outlier where Haiku-q9 is essentially uncorrelated on desc+inst
  at ρ ≈ 0.11; plot at
  `roger/axis_judge_experiments/response_di_weight_sweep_slot6_no_eco_anthro.png`):
  peak at `w ≈ 0.71`, ρ ≈ 0.772.

`0.80` lands in the flat plateau of *both* curves — 12-axis ρ at
w=0.8 is 0.762 (vs 0.763 at the peak); 11-axis ρ at w=0.8 is 0.770
(vs 0.772 at the peak) — so the choice is robust to whether
eco/anthro is treated as a regular member or a held-out anomaly.

The complementary view of *why* this matters lives at
`roger/axis_judge_experiments/rubric_v1_v2_compare_slot6.png`
(`rubric_v1_v2_compare.py`): under the v2 rubric the combined-mode
mean ρ delta is +0.019 across the
12 axes (vs only +0.006 for response alone), and per-axis losses
in response-only mode are largely recovered or fully reversed once
desc+inst is mixed in.

### Tuning history

Every tuning event re-runs the relevant sweep script and updates
the central constant.  Audit trail:

| Date | Constant changed | From | To | Rationale / driver |
|---|---|---|---|---|
| (legacy) | `DEFAULT_DI_WEIGHTS` | — | `(0.499, 0.501)` | Initial sweep at slot=(3,25)/(0,26)/(0,49); see module docstring. |
| (legacy) | hardcoded GPT/Sonnet `(g + s)/2` | — | (parameterised, default 0.50) | Pre-dated the constant catalog; promoted to `DEFAULT_GPT_SONNET_DI_WEIGHT = 0.50` on 2026-05-09 for symmetry with the other three. 33-axis sweep peaks at w=0.530 with mean ρ within 0.0004 of the round-number value. |
| 2026-05-12 | `DEFAULT_GPT_SONNET_DI_WEIGHT` | `0.50` | `0.625` | Canonical-whitening retune: 35-axis `gpt_sonnet_weight_sweep.py --whitening soft_shear=3` at slot 6 layer 25, **discrete grid peak w=0.625** (ρ=0.70198, w_step=0.025).  Parabolic peak is nominally w=0.700 but the plateau `w∈[0.575, 0.700]` is flat within 0.0003 ρ vs ~0.04 95% CI half-width, so we pick the literal discrete peak.  Δρ vs old 0.50 default is +0.0008 (~judge noise). |
| 2026-05-08 | `DEFAULT_GPT_HAIKU_Q9_WEIGHT` | (was 0.5 = "even split" at first) | `0.60` | 12-axis sweep, parabolic peak at w=0.609 rounded; Pareto-frontier check on cost-per-quality. |
| 2026-05-11 | `DEFAULT_GPT_HAIKU_Q9_WEIGHT` | `0.60` | `0.41` | v2 raw-projection re-tune, parabolic peak ~0.41 (Phase-5d Haiku data improved). |
| 2026-05-12 | `DEFAULT_GPT_HAIKU_Q9_WEIGHT` | `0.41` | `0.625` | Canonical-whitening (`--whitening soft_shear=3`) v2 retune at slot 6 / layer 25; 12-axis discrete grid peak w=0.600, parabolic w=0.624, rounded to 0.625 for symmetry with `DEFAULT_GPT_SONNET_DI_WEIGHT`. |
| 2026-05-22 | `DEFAULT_GPT_HAIKU_Q9_WEIGHT` | `0.625` | `0.525` | **22-axis retune** after Phase-1/2 expansion added 10 personality-style axes.  Discrete grid peak w=0.525 (ρ=0.7521), parabolic w=0.472, plateau [0.375, 0.575].  Haiku is now the better single-judge baseline (Haiku 0.740 vs GPT 0.709), inverting the 12-axis ranking. |
| 2026-05-09 | `DEFAULT_RESPONSE_DI_WEIGHT` | `0.50` (placeholder) | `0.80` | 12-axis and 11-axis-without-eco sweeps both flat-plateau at w=0.8 (12-axis peak 0.84, 11-axis peak 0.71). |

### Re-tuning checklist

When new judging data lands (more axes, a new judge, a re-batched
cohort, etc.), expect to revisit ratios 2 and 3.  The drill:

1. **Inventory what's new.** Which axes / judges / batch sizes /
   subsample modes have caches under `roger/axis_judge_experiments/`?
   The `tools/audit_caches.py` markdown report is the easiest way
   to enumerate.
2. **Re-run sweep 2 (within-response):**

   ```bash
   for combo in haiku_q9 haiku_full sonnet_q9; do
       uv run python -m results_analysis.gpt_anthropic_response_weight_sweep \
           --anthropic_combo "$combo"
   done
   ```

   Compare new vs old `gpt_haiku_q9_response_weight_sweep_slot6.json`
   parabolic peak `w`.  If the shift is non-trivial (>0.03 say), or
   the ρ improvement at the new peak vs `0.60` exceeds noise floor
   (~0.04 SE per axis, scales by `1/sqrt(n_axes)` for the mean), bump
   `DEFAULT_GPT_HAIKU_Q9_WEIGHT`.
3. **Re-run sweep 3 (final blend):**

   ```bash
   uv run python -m results_analysis.response_di_weight_sweep
   uv run python -m results_analysis.response_di_weight_sweep \
       --exclude_axes ecocentric_vs_anthropocentric
   ```

   Both 12-axis and outlier-excluded variants matter — the
   round-number plateau choice is best when the two peaks
   bracket it.  Re-tune `DEFAULT_RESPONSE_DI_WEIGHT` if the new
   round-number plateau midpoint moves by more than ~0.05.
4. **Update the central docstring + this README** with the new
   numbers and add a row to the tuning-history table above so the
   audit trail stays continuous.
5. **Snapshot before invalidating.**  Per the snapshot-before-
   invalidate principle (see `AGENT_NOTES.md`), if the constant
   change implies redoing some judging, take a `__rubric_vN.json`
   snapshot of every cache the change will overwrite *before*
   running the rejudge — the v1↔v2 comparison plots
   (`rubric_v1_v2_compare.py`) depend on those snapshots existing
   on disk.

## Convention: provenance for analysis scripts

Two related but distinct provenance layers apply here. The first
records *how* an artifact was made; the second records *what its
inputs were*, so we can detect when an artifact is out of date.

### Plot metadata (mandatory for tracked plots)

Every plot produced by a tracked script in this directory embeds
`Title / Author / Software / Creation Time / Source` text-chunks via
`assistant_axis.png_metadata` (see the helper at
[`assistant_axis/plot_metadata.py`](../assistant_axis/plot_metadata.py)
and the agent-facing convention in [`../AGENT_NOTES.md`](../AGENT_NOTES.md)).
Ad-hoc exploration scripts that produce plots also embed their full
source via `source_text=Path(__file__).read_text()`, so any PNG in the
tree carries enough provenance to reproduce.  Read with
`exiftool foo.png` or `PIL.Image.open(p).info["Source Code"]`.

### Input fingerprints (writer side)

Analysis scripts that emit JSON should wrap their payload in a
`_provenance` envelope, declaring all inputs they read:

```python
from assistant_axis import json_metadata, png_metadata
from assistant_axis.provenance import (
    InputSpec, current_data_subtree_input, current_file_input,
    current_files_input,
)

inputs: list[InputSpec] = [
    current_data_subtree_input(data_dir, "traits/vectors", dep_key="traits_vectors"),
    current_file_input(dep_key="pairs_json", path=experiment_dir / args.pairs),
    current_files_input(dep_key="judge_caches", paths=judge_cache_paths),
]
out_path.write_text(json.dumps(
    json_metadata(records, inputs=inputs, title="my_analysis"), indent=2))
fig.savefig(plot_path, dpi=150, bbox_inches="tight",
            metadata=png_metadata(title=title, inputs=inputs))
```

The same `inputs` list is reused for both PNG and JSON output so a
plot's freshness can be verified against the same dependency graph as
its source cache.

### `--cache-policy` (reader side)

When a script reads a JSON cache produced by another tracked script
(rather than reading raw datasets directly), it should validate the
cache's recorded inputs:

```python
from assistant_axis.provenance import CACHE_POLICIES, load_validated_json

p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn")
records, check = load_validated_json(sweep_path, policy=args.cache_policy)
```

Policies: `strict` (raise on drift), `warn` (default; print
diagnostics, return payload), `rebuild` (call back to regenerate),
`off` (skip validation). Legacy bare-JSON caches load transparently
without an envelope.

### Audit tools

To find stale artifacts repo-wide:

```bash
uv run python tools/audit_pngs.py   --status stale
uv run python tools/audit_caches.py --status stale
```

`audit_caches.py` also propagates staleness transitively through
inter-cache file dependencies, so a freshly-built consumer cache is
flagged `stale_transitive` when its upstream input is `stale_direct`.

Full reference and migrated-script roster:
[`../AGENT_NOTES.md` § End-to-End Data Provenance](../AGENT_NOTES.md).

## Scripts

### `axis_judge_correlation.py`

Tests how well a chosen semantic axis in activation space corresponds to
external judgement. Given an axis direction (either derived from a pair of
roles/traits, or loaded from a saved axis file) and pole descriptions plus
example lists, it:

1. Has an LLM judge score every role and trait in the corpus on a −3..+3
   rubric (excluding the pair itself and any example-list names).
2. Projects every role/trait's activation vector onto the axis, in both the
   raw activation metric and a soft-K=3 PCA-whitened metric (whitener fit
   on the held-out pool of roles + traits, with the excluded names removed
   so the same items don't define both the axis and the whitening basis).
3. Reports Spearman ρ between judge scores and projections, per token-slot
   and per metric, for up to three scoring modes:
    - the `description` field,
    - the 5 `pos` instructions,
    - score=3 model responses (question + answer).
4. Writes JSON results and a scatter-plot grid to `--output_dir`.

```bash
# Angel vs demon, three scoring modes, all slots
uv run python results_analysis/axis_judge_correlation.py \
  --pair angel demon --pair_type roles \
  --data_dir "runpod_workspace/qwen/qwen-3-32b Roger" \
  --scores_dir "runpod_workspace/qwen/qwen-3-32b Roger/roles/scores" \
  --responses_dir "runpod_workspace/qwen/qwen-3-32b Roger/roles/responses" \
  --layer 25 --whiten_K 3 --provider anthropic --all \
  --output_dir roger/axis_judge_angel_demon

# Manual axis from a saved axis.pt, scoring descriptions only
uv run python results_analysis/axis_judge_correlation.py \
  --axis_file "runpod_workspace/qwen/qwen-3-32b Roger/roles/axis.pt" \
  --neg_pole "mythical/unstructured entity" \
  --pos_pole "defined professional role with structured output" \
  --neg_examples demon,vampire,revenant,chimera \
  --pos_examples consultant,researcher,engineer,accountant,librarian \
  --data_dir "runpod_workspace/qwen/qwen-3-32b Roger" \
  --layer 25 --slot 1 --score_descriptions \
  --output_dir roger/axis_judge_pc1_roles
```

#### Cost model: GPT-4.1-mini responses-mode judging

The `responses` scoring mode is the dominant cost in this script (the
`description` and `instructions` modes each fit in a few hundred prompts
per axis). Numbers below were re-derived from scratch on **2026-05-09**
via a 240-batch dry-run sample: real B=10 prompts reconstructed from
v1-rubric cache batch keys + matching responses in
`runpod_workspace/.../{roles,traits}/responses/<entity>.jsonl`,
tokenized with `tiktoken` `o200k_base` (the GPT-4.1-mini encoding).
**These numbers supersede the previous cost table, which was ~5× too
low because it estimated per-item input at 35.3 tokens (a header-only
under-count) instead of the ~438 tokens that real Qwen-3-32B responses
actually consume.**

**2026-05-10 cross-axis validation** — re-ran the dry-run via
[`tools/dry_run_response_token_count.py`](../tools/dry_run_response_token_count.py)
on **all 12 v2 axes × both cohorts = 295,258 reconstructed B=10
batches**, prompt-rebuilt under the current (v2) rubric.  Per-axis
cost ratio (`measured / $49.40 doc`) came in at **mean 1.006 ± σ 0.013,
range [0.989, 1.034]**.  The 4,700/80/24,605 figures and the
`COST_PER_AXIS_USD` dict in
[`batch_size_rho_curve.py`](batch_size_rho_curve.py) are correct
within ±3% per axis and within 0.6% on average — no recalibration
applied.  See AGENT_NOTES "Judging cost model" for the per-axis
breakdown table.  Re-run the dry-run after any rubric change to
re-validate automatically.

**Per-batch token model** (240-batch sample, 12 v1 axes × both cohorts):

| Component | Tokens | Source |
|-----------|--------|--------|
| Fixed input overhead per batch | **320** | rubric header + scale table + axis name + pos/neg pole text + name field + closing instructions; tiktoken-measured on a single header (range across axes: 280-360) |
| Per-item input | **438** | `--- Response i of N ---` block + `[QUESTION] / [/QUESTION] / [RESPONSE] / [/RESPONSE]` markers + a real Qwen-3-32B response. Sampled mean per-batch input is **4,700 tokens** at B=10 (median 5,018, σ 1,065, min 1,307, max 6,381 across 240 batches): subtract the 320-token header, divide by 10. |
| Output per batch | **80** | median 78, mean 80, σ 14 — measured by tiktoken-encoding the cached judge text from the same 240 sampled batches |

So per batch: `input ≈ 320 + B × 438`, `output ≈ 80`.

**Pricing** (gpt-4.1-mini, 2026 rates): input **$0.40 / 1M tokens**,
output **$1.60 / 1M tokens**. Anthropic's `claude-haiku-4-5-20251001`
costs $1.00 / $5.00 per 1M; `claude-sonnet-4-20250514` is $3.00 /
$15.00 per 1M.

**Items per axis**: with the project's standard scoring corpus the
`responses` mode covers ~246k items per axis (280 roles × ~403
responses + 300 traits × ~445 responses; mean over 12 v1 axes). One
axis-worth of work, both sides combined.

**Per-axis cost curve** (`responses` mode, single axis, both sides
combined; GPT-4.1-mini at $0.40 in / $1.60 out per 1M):

| B  | Batches | Input M tok | Output M tok | $ in  | $ out | **$ total** | Relative |
|----|---------|-------------|--------------|-------|-------|-------------|----------|
|  5 | 49,200  | 123.5       | 3.94         | 49.40 |  6.30 | **$55.70**  | 1.18×    |
|  7 | 35,142  | 119.0       | 2.81         | 47.60 |  4.50 | **$52.10**  | 1.10×    |
| 10 | 24,600  | 115.6       | 1.97         | 46.25 |  3.15 | **$49.40**  | 1.04×    |
| 15 | 16,400  | 113.0       | 1.31         | 45.20 |  2.10 | **$47.30**  | **1.00×** (ref) |
| 20 | 12,300  | 111.7       | 0.98         | 44.68 |  1.57 | **$46.25**  | 0.98×    |
| 30 |  8,200  | 110.4       | 0.66         | 44.16 |  1.05 | **$45.20**  | 0.96×    |

**Why the curve is so flat**: per-item content (246k items × ~438
tokens = ~108M tokens) is fixed regardless of batch size, so total
input tokens shrink only via the (B-dependent) per-batch overhead.
The 320-token header gets paid `N_items / B` times, which is a small
fraction of the total budget — going from B=5 to B=30 saves only
~13M tokens (~$5/axis). Cost asymptotes to ~$44/axis (the items-only
floor) as B → ∞. The previous cost model erroneously made this curve
look 3× steeper because it under-counted item tokens.

**12-axis sweep cost** (the standard "all-axes" responses run we use
as a baseline): **B=15 ≈ $568**, B=10 ≈ $593, B=7 ≈ $625, B=5 ≈ $668.

**Quality curve** (measured 2026-05-01 on 3 axes ×
3 (slot, layer) cells × 4 batch sizes:
`truthful_vs_deceitful`, `progressive_vs_conservative`,
`improvisational_vs_methodical`; grand-mean ρ averaged over the
3 axes × 3 (slot, layer) cells; cost from the corrected model above):

| B  | grand-mean ρ | Δρ vs B=15 | $ / axis | extra $ vs B=15 | $ per +0.01 ρ |
|----|--------------|------------|----------|-----------------|---------------|
|  5 | +0.7842 | +0.0175 | $55.70 | +$8.40 | $4.80 |
|  7 | +0.7809 | +0.0143 | $52.10 | +$4.80 | $3.36 |
| 10 | +0.7767 | +0.0100 | $49.40 | +$2.10 | **$2.10** |
| 15 | +0.7667 | (ref)   | $47.30 | (ref)  | (ref) |

The curve is monotonic and **diminishing returns kick in below B=10**:
- B=15 → B=10 buys you the first +0.010 ρ for $2.10 extra (best value).
- B=10 → B=7 buys another +0.004 ρ for an additional $2.70 ($6.81/+0.01 ρ marginal).
- B=7 → B=5 buys another +0.003 ρ for $3.60 more ($11.92/+0.01 ρ marginal).

The relative *steps between B values* are essentially unchanged from
the old (under-counted) model because the per-item-rate piece cancels
out — only the absolute level shifts upward by ~5×.

Per-cell behaviour: the gain from going small is **smallest at slot=3
/ layer=25** (Δρ ≈ +0.009 across all batches) and **largest at
slot=0** (Δρ +0.020 to +0.024).  The slot=0 cells benefit most from
small batches, which is consistent with "the noise being removed by
small B is in the judge's per-batch variance, not in the activation
side".

(Full 12-axis sweep cost-per-Δρ: at $4.80/0.01 ρ for B=5 (worst-value
endpoint), the 12-axis sweep would cost $100 extra for +0.018 ρ, i.e.
**$555 per +0.01 ρ across the corpus**.)

A complementary view of the same data plots cost vs **quality
= 1/(1−ρ)**, which amplifies small ρ changes near the ceiling
(`d(1/(1−ρ))/dρ = 1/(1−ρ)² ≈ 18` at ρ=0.77, so each +0.01 ρ buys
~0.18 quality units, or **~4 % effective signal**).  The marginal
quality-per-dollar between adjacent batch sizes:

| step | Δquality | Δ$ | quality / $ |
|------|----------|-----|-------------|
| B=15 → B=10 | +0.192 | +$2.10 | **0.091** |
| B=10 → B=7  | +0.087 | +$2.70 | 0.032 |
| B=7  → B=5  | +0.068 | +$3.60 | 0.019 |

The first step (B=15 → B=10) is **2.8× more cost-efficient** than
B=10 → B=7 and **4.8× more** than B=7 → B=5: the curve is sharply
concave on the quality scale.

**Recommendation (project default for new responses-mode runs: B=10)**:
- **B=15**: high-throughput sweeps where the +0.01 ρ uplift is below
  the per-axis SE (~0.04).
- **B=10** *(picked as default 2026-05)*: best marginal cost-efficiency.
  +28 % cost over B=15, captures ~57 % of the available ρ-improvement
  and ~70 % of the quality-improvement before diminishing returns kick
  in.  See `roger/batch_size_cost_vs_quality.png`.
- **B=7 or B=5**: only worth the extra spend for paper figures or
  narrow follow-up comparisons where the absolute ρ ceiling matters
  more than throughput.

Plots:
- `roger/batch_size_curve_rho.png` -- per-cell grouped histogram + ρ-vs-B
  curves (one line per slot/layer).
- `roger/batch_size_cost_vs_rho.png` -- linear cost-vs-ρ scatter.
- `roger/batch_size_cost_vs_quality.png` -- cost-vs-quality scatter on the
  1/(1−ρ) scale; the diminishing-returns shape is most visible here.

The curve was measured on 3 axes × 3 (slot, layer) cells × 4 batch
sizes via [`batch_size_rho_curve.py`](#batch_size_rho_curvepy); the quality
view is regenerated from the cached JSON via
[`plot_batch_size_quality_vs_cost.py`](#plot_batch_size_quality_vs_costpy).

#### `batch_size_rho_curve.py`

Sweeps responses-mode ρ across multiple target batch sizes, axes, and
``(slot, layer)`` configs.  For each cell ``(axis, B, slot, layer)``:

1. Loads per-entity ``mean_score`` from
   ``<experiment_dir>/<axis>/gpt_responses_{roles,traits}<suffix>/scores_responses.json``
   (suffix = ``""`` for B=15, ``"_b<N>"`` otherwise).
2. Computes the axis direction ``unit(vec[pos] − vec[neg])`` at
   ``(slot, layer)`` from the standalone trait vectors.
3. Sweeps ``L ∈ [0, 1, 2, 3, 5, 8, 16] × K ∈ [0..16]`` with
   bracket-and-bisect refinement on K, applying the same
   L-shear + K-soft-K transform to both the entity matrix and the
   axis direction; takes max ρ as the cell's score.

Outputs (to ``--output_dir``):

- ``batch_size_curve_rho.json`` -- per-cell ρ + grand-mean per B + cost
  model + B integers (everything needed to re-plot without re-running).
- ``batch_size_curve_rho.png`` -- two-panel: grouped histogram (configs ×
  batch bars) + ρ-vs-B line plot (one line per config + grand mean).
- ``batch_size_cost_vs_rho.png`` -- linear cost-vs-ρ scatter.

```bash
# Default: 3 axes × 4 batch sizes × 3 configs.  Reproduces the May 2026
# pilot.
uv run python results_analysis/batch_size_rho_curve.py

# Single axis, B=10 vs B=15 only:
uv run python results_analysis/batch_size_rho_curve.py \
  --axes truthful_vs_deceitful \
  --batch_sizes 10,15 \
  --output_dir roger/b_curve_truthful/
```

Library: ``compute_batch_size_curve(...) -> dict`` returns the same dict
that's written to JSON, for callers that want to fold the data into a
larger analysis without re-running the L/K sweep.

#### `plot_batch_size_quality_vs_cost.py`

Regenerates the cost-vs-quality scatter
(``roger/batch_size_cost_vs_quality.png``) from a
``batch_size_curve_rho.json`` cache, on the ``1 / (1 − ρ)`` "quality"
scale.  Each marker is annotated with its ``B``, ρ, cost-per-axis, and
1/(1−ρ); a dotted reference line connects the cheapest-and-most-expensive
endpoints (concave curves lie above their endpoint chord -- so the
distance above the line is a visual measure of how concave the curve
is).  Side y-axis shows the corresponding ρ for each quality value.

Also prints the marginal Δquality / Δ$ between adjacent batch sizes,
which is the cleanest read of "where does the cost-efficiency curve
fall off?".

```bash
# Default: read roger/batch_size_curve_rho.json,
#          write roger/batch_size_cost_vs_quality.png.
uv run python results_analysis/plot_batch_size_quality_vs_cost.py
```

Library: ``plot_quality_vs_cost(input_path, output_path)`` and
``print_marginal_table(input_path)``.

### `infer_axis_description.py`

The conceptual inverse of [`axis_judge_correlation.py`](#axis_judge_correlationpy).
That tool *takes* an axis description and *produces* per-entity
projections-vs-rubric-score correlations. This tool *takes* a per-entity
projection ranking and *produces* an axis description (axis name, two pole
descriptions, example lists) -- ready to feed straight back into
`axis_judge_correlation.py` via its `--axis_name` / `--pos_pole` / `--neg_pole`
/ `--pos_examples` / `--neg_examples` flags.

The tool is geometry-agnostic: callers compute whatever direction they want
to characterize (a contrastive pair, a CA basis vector, a PCA component, a
shear axis, ...), project all roles and traits onto it, hand the resulting
list to this tool, and get back a structured axis spec. The only thing this
tool needs to know about each entity is its name, its type (role / trait),
and the scalar projection.

Internally the tool z-normalizes the projections to mean 0, std 1, rounds
to 0.1 precision, and asks Claude Opus (`claude-opus-4-6`, with a 10K
extended-thinking budget by default) to identify what semantic axis the
direction captures. Output is a JSON file with five fields, in display-label
form for the prose and filename form for the example lists, so it can be
piped straight into the forward tool.

```bash
# Caller computes scores (any geometric flavour) and writes them to a JSON file
# (or CSV); the format is just [{name, type, score}, ...].
uv run python results_analysis/infer_axis_description.py \
  --input scored.json \
  --output axis_spec.json \
  [--style glossary] \                 # glossary (default) | inline
  [--instructions_dir data] \
  [--model claude-opus-4-6] \
  [--thinking_budget 10000] \
  [--top_n 0]                          # 0 = include all entities; >0 = top + bottom N only
```

Input format (JSON example; CSV with columns `name,type,score` also accepted):

```json
[
  {"name": "helpful",             "type": "trait", "score":  3.21},
  {"name": "advocate",            "type": "role",  "score":  2.74},
  {"name": "paperclip_maximizer", "type": "R",     "score": -1.42}
]
```

* `name`: filename format with underscores (matching `data/{roles,traits}/instructions/*.json` filenames).
* `type`: any of `R`, `T`, `role`, `trait`, `roles`, `traits` (case-insensitive).
* `score`: any float -- the raw scalar projection. The tool z-normalizes internally.

Output format (matches `axis_judge_correlation.py`'s axis-spec flags exactly):

```json
{
  "axis_name": "prosocial vs antisocial",
  "pos_pole": "This pole represents concepts oriented toward...",
  "neg_pole": "This pole represents concepts oriented toward...",
  "pos_pole_standardized": "This means being oriented toward...",
  "neg_pole_standardized": "This means being oriented toward...",
  "pos_examples": ["helpful", "advocate", "guileless", "systems_thinker"],
  "neg_examples": ["unhelpful", "deceitful", "paperclip_maximizer"]
}
```

The `pos_pole` / `neg_pole` fields hold the raw Opus output (typically
opening with "This pole/end represents..."), while
`pos_pole_standardized` / `neg_pole_standardized` are auto-rephrased by
[`standardize_axis_spec.py`](#standardize_axis_specpy) (Sonnet) into the
project's standard "This means [verb-ing]..." form -- the form expected
by the desc+inst judge. **For judge runs, prefer the `*_standardized`
fields**; the raw fields are kept for inspection / provenance.

To skip the auto-rephrase, pass `--no_standardize`.

Library API:

```python
from results_analysis.infer_axis_description import summarize_axis

result = summarize_axis(
    scores=[{"name": "helpful", "type": "trait", "score": 2.34}, ...],
    style="glossary",
)
# -> {"axis_name": ..., "pos_pole": ..., "neg_pole": ...,
#     "pos_pole_standardized": ..., "neg_pole_standardized": ...,
#     "pos_examples": [...], "neg_examples": [...]}
```

**Layout (`--style`):** the prompt sent to Opus comes in two forms:

* `glossary` (default): a compact one-line-per-entity ranking (rank, z-score,
  R/T, display label) followed by an alphabetical glossary block listing
  every entity's description. Lets the model scan the gradient first and
  consult the glossary for unfamiliar names.
* `inline`: ranking with descriptions on the same line. No cross-referencing
  required; same total token count.

The two should give similar results in most cases. Default is `glossary`
based on intuition; A/B both on a known axis (e.g.
`truthful_vs_deceitful`) if you want to lock in a winner.

**Naming convention.** Two formats coexist in the codebase:

* *Filename* (underscores, no hyphens): `systems_thinker`, `paperclip_maximizer`.
  This is what the tool's input and output use, so the output spec can be
  passed straight back to `axis_judge_correlation.py`.
* *Display label* (more human-readable): `systems-thinker` (from the trait
  JSON's `positive_label` field), `paperclip maximizer` (`_` -> ` ` for
  roles, which by convention never use hyphens). This is what the prompt
  shows the model. The tool maintains a bidirectional map and converts the
  model's emitted example labels back to filename form on parse, with a
  small fuzzy-match fallback (lowercase + alphanumerics).

**Cost.** Per axis, with default thinking budget (10K tokens):

* Input: ~42K tokens (full 579-entity list with descriptions): ~$0.63 at
  Opus list pricing ($15/M).
* Output (incl. thinking): up to 10K tokens at $75/M = ~$0.75 worst-case;
  typical usage less.
* Total: ~$1.30-1.50 per axis.

Drop with `--top_n 50` (top + bottom 50 only) for cheap experimentation
(~$0.85/axis), or `--thinking_budget 0` to disable extended thinking
(~$0.10/axis).

**Reproducibility.** Anthropic's API forces `temperature=1` when extended
thinking is enabled, so re-runs are not bit-exactly deterministic; even at
temp 0 the API isn't guaranteed reproducible. Re-runs may produce slightly
different phrasings; the inferred axis itself should be stable on hard
tasks. We don't engineer around this (no caching layer); cost per re-run is
~$1, which is cheaper than implementation complexity.

### `standardize_axis_spec.py`

A small Sonnet-based shim that **adds standardized `"This means..."`-form
pole descriptions** to an axis spec without overwriting the originals.

Position in the pipeline (auto-invoked from
[`infer_axis_description.py`](#infer_axis_descriptionpy) by default):

```
infer_axis_description.py     standardize_axis_spec.py     axis_judge_correlation.py
   (Opus, with thinking)         (Sonnet rephrase)
[sorted projection list]   ->  [pos_pole, neg_pole       ->  [pole_standardized
                                + *_standardized fields]      drop straight into
                                                              --pos_pole / --neg_pole]
```

**Why it exists.** Auto-generated specs typically open their pole
descriptions with phrases like `"This pole represents..."` /
`"Someone who is..."`, which describe the pole as a *category* rather
than the *behavior of an entity at that pole*. The desc+inst judge
prompt template inserts pole text directly into a slot for an entity
description, so the project standard is to start each pole description
with `"This means [verb-ing/being]..."` for a behavioral framing.

The shim runs one Sonnet call per spec (~$0.002), preserving the
original `pos_pole` / `neg_pole` fields and adding new
`pos_pole_standardized` / `neg_pole_standardized` fields. All other
fields (`axis_name`, `pos_examples`, `neg_examples`, `_metadata`, etc.)
are passed through unchanged.

**Auto-invocation.** When `infer_axis_description.py` runs without
`--no_standardize`, this shim is called automatically; the resulting
spec already has both raw and standardized pole text.

**Idempotent.** If a spec already has `*_standardized` fields, the shim
skips the API call. Use `--force` to re-rephrase (e.g. after editing the
prompt's multi-shot examples).

**Standalone CLI.**

```bash
# Single file
uv run python results_analysis/standardize_axis_spec.py \
  --input raw_spec.json --output spec.json

# In-place batch update (idempotent)
uv run python results_analysis/standardize_axis_spec.py \
  --in_place roger/pc_axis_describer_sweep/*/spec.json
```

**Library API.**

```python
from results_analysis.standardize_axis_spec import standardize_axis_spec
extended = standardize_axis_spec(spec)   # adds *_standardized fields
```

**Multi-shot examples in the prompt** cover every opener pattern we've
seen in 16 auto-generated specs: `"This pole represents..."`,
`"This end represents..."`, `"Someone who..."`, plus a no-op example
showing an already-standard pole pass through unchanged.

### `refill_judge_gaps.py` and the gap-detection workflow

`axis_judge_correlation.py` calls can occasionally fail mid-run due to
transient provider issues (timeouts, 5xx, rate-limit blips). The script
ships with two layers of defense:

1. **Per-call retry-with-backoff** (`RETRY_DELAYS = [5s, 20s, 60s, 180s]`).
   Transient errors -- timeouts, connection errors, 5xx, 429-rate-limit --
   are retried up to four times with growing waits. Cumulative tolerance
   ~265 s, comfortably surviving ~3-min provider blips. Permanent errors
   (`insufficient_quota`, 4xx auth/parameter errors) are *not* retried.
2. **End-of-run gap audit.** When a run finishes, every scoring mode emits
   a clear WARNING banner if any entity / batch was left unscored, and
   writes a structured `gaps.json` to the output dir. Empty modes write
   empty containers (`[]` or `{}`) so downstream tools can rely on the
   file being present.

Even with retries, gaps can persist (e.g., a multi-hour outage). The
`refill_judge_gaps.py` tool is the systematic way to find and fix them:

```bash
# Audit-only: list dirs with gaps under the experiment root.
uv run python results_analysis/refill_judge_gaps.py \
  --glob 'roger/axis_judge_experiments/*/*' --scan

# Refill one specific dir.
uv run python results_analysis/refill_judge_gaps.py \
  --dir roger/axis_judge_experiments/truthful_vs_deceitful/gpt_responses_traits

# Refill all dirty GPT-response caches under the experiment root.
uv run python results_analysis/refill_judge_gaps.py \
  --glob 'roger/axis_judge_experiments/*/gpt_responses_*'
```

The tool reads each dir's `config.json` and re-invokes
`axis_judge_correlation.py` with the same args. The inner script's
**resume logic only re-issues calls for the gaps**: parse failures aren't
cached (so they're retried), and response-mode `per_batch` entries with
`score=None` get re-attempted. So a refill costs roughly the size of the
gap, not a full re-run.

For batch refills via `run_axis_experiment_batch.py`, pass `--refill_gaps`
to bypass the "skip if `correlations.json` exists" gate. Pairs with a
clean `gaps.json` (all modes empty) are still skipped, so refilling a
mostly-clean experiment root is cheap.

### `pc_round_trip/` — auto-described axis vs. original projection

End-to-end pipeline that asks: when we auto-describe the n-th principal
component of the post-shear entity space (using
[`infer_axis_description.py`](#infer_axis_descriptionpy)), then re-score
the resulting axis text against the entities (using
[`axis_judge_correlation.py`](#axis_judge_correlationpy)), how well does
the round-trip recover the original projection?

The pipeline lives in `results_analysis/pc_round_trip/` and has four
scripts:

```
infer_axis_description.py     launch_judge_runs.py     klm_sweep.py     plot_loglin.py
   (per-cell auto-describe        (28× judge calls          (no API,         (no API,
    -> spec.json)                  for desc+inst             ~4 min CPU)      instant)
   [also auto-runs                 over GPT+Sonnet)
    standardize_axis_spec.py]
```

#### `pc_round_trip/launch_judge_runs.py`

Computes PC directions in post-shear space (default `slot=3, layer=25,
shear_L=DEFAULT_SOFT_SHEAR_L`), saves per-entity projections, and
launches `axis_judge_correlation.py` for every (PC × {glossary, inline}
× {openai, anthropic}) cell. Each cell ends up with::

    pcNNN_{glossary,inline}/
        axis_postshear.pt          (PC direction, post-shear)
        post_shear_projection.json (entity projections onto that PC)
        prompt.txt + response.txt + thinking.txt + spec.json
            ^-- produced upstream by infer_axis_description.py
        gpt/                       (axis_judge_correlation.py output)
            scores_descriptions.json
            scores_instructions.json
            ...
        sonnet/

Per-cell `spec.json` must already exist; this script does **not** call
`infer_axis_description.py`. Run that (with auto-standardize) first for
each PC × style. The describer is the dominant cost — Claude Opus with
a 10K thinking budget runs ~$0.50 per cell; the judge sweep is the
dominant volume — ~hundreds of small batches per cell.

`--dry_run` prints the planned commands and exits, useful for sanity
checks. Combine with `--skip_setup` to skip the SVD setup too:

```bash
# Dry-run a single cell (no API, no SVD).
uv run python -m results_analysis.pc_round_trip.launch_judge_runs \
  --dry_run --skip_setup \
  --pcs 8 --styles inline --providers anthropic
```

`--max_parallel` defaults to 2 for ML-footprint headroom; bump on a
cleaner machine.

#### `pc_round_trip/klm_sweep.py`

Adaptive **(L, K, M)** sweep over the cached judge scores. For each
(PC, style) cell, finds the best round-trip Spearman ρ across:

- `slot, layer` ∈ `{(3, 25), (0, 26), (0, 49)}` by default — the three
  configs that dominate the per-PC winners (low PCs land on slot=3 /
  L=25; high PCs split between slot=0 / L=26 and slot=0 / L=49).
- `L` — soft-shear truncation depth (default `0..5`).
- `K` — soft-K whitening depth, swept on a coarse log-spaced grid then
  refined with bracket-and-bisect around the peak (default coarse
  `[0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512]`).
- `M` — optional truncation of the entity matrix into the top-M PCs of
  the pool (default `{64, 128, 256, 512, ∞}`).

Two stages: stage 1 explores the M dimension at every coarse
`(L, K)` grid point; stage 2 refines K at M=∞ via discrete
bracket-and-bisect, the integer cousin of golden-section search. Stops
when the bracket is ≤ 3 K-units wide or ρ-improvement < 0.005 (one-tenth
of the 1/√n ≈ 0.04 noise floor).

At very high K, the soft-K-whitened matrix becomes near-rank-deficient
and the default LAPACK `gesdd` driver can fail to converge; the script
falls back to scipy's `gesvd` (slower but robust) and skips configs
where even that fails.

This stage does **no** API calls — just reads cached judge scores and
runs CPU. Re-running is cheap (~4 min on a workstation).

```bash
# Default: full sweep over all 14 PCs × 2 styles, writing to
# roger/pc_round_trip_klm_results.json.
uv run python -m results_analysis.pc_round_trip.klm_sweep
```

#### `pc_round_trip/plot_loglin.py`, `plot_histogram.py`

Two PNG views of the cached sweep results:

- `plot_loglin.py` — log-x line+marker plot of average best ρ vs PC
  index. Headline figure for the experiment; reads cleanly across the
  full PC range. Default source is **stage 2** (refined K, M=∞);
  `--source max` takes per-cell max(stage 1, stage 2).
- `plot_histogram.py` — bar-chart companion that prints per-cell winner
  labels (`slot=…,ly=…,L=…,K=…,M=…`) for diagnostic inspection. Always
  uses per-cell max(stage 1, stage 2) so finite-M wins are visible.

Both scripts emit PNGs with embedded provenance metadata
(`png_metadata` + `suptitle_with_specs`); both are instant.

```bash
uv run python -m results_analysis.pc_round_trip.plot_loglin
uv run python -m results_analysis.pc_round_trip.plot_histogram
```

#### Key finding (April 2026)

The K-coarse grid initially capped at K=128; PC 192 and PC 256 were then
showing near-noise-floor ρ (~+0.05). Extending K to 512 lifted them to
+0.36 and +0.31 — the optimum K tracks the PC index very tightly
(`K* ≈ N` for `N ≥ 16`). Higher PCs need more aggressive whitening to
expose their semantic content; capping K too low collapses the signal.

Slot/layer dominance also shifts with PC index: PCs 1, 2, 4, 8 win at
slot=3 / L=25 (the canonical "best" cell), while everything from PC 32
onward wins at slot=0 / L∈{26, 49} — the body-mean representation at
the two "post-step" layers right after the major computation cliffs at
24 and 48.

### `token_position_noise_analysis.py`

Diagnostic that asks **which of the 4 token slots is the noisiest**, and
which agrees most with the others on the *direction* of role differences.
Promoted from a `roger/` script (April 2026 work) that informed the
project-wide convention of using slot 3 (`\n`) as the default working slot.

For each random triple `(A, B, C)` and each slot `s`, the script computes
six metrics measuring how similar `B−A` is to `B−C` at slot `s`:

- 3 distance flavours: cosine distance, Euclidean norm, squared norm.
- 2 cross-diff operators: `_sub` (subtraction, normalized) and `_lograt`
  (log-ratio).

For each (triple, metric, layer), the slot whose value deviates *most*
from the mean of the other three is the **odd one out**. Slots that are
odd-one-out >25 % of the time (the chance rate) are noisier than peers.

Plus pairwise-slot agreement (per layer, pooling **all unordered entity
pairs** `(A, B)` with `A ≠ B`, i.e. `C(N_roles, 2)` pairs), two ways:

- **Norm correlation**: Pearson correlation *across those pairs* between
  `‖A_s − B_s‖` at slot `s` and `‖A_s′ − B_s′‖` at slot `s′` -- do two
  slots agree on which pairs are far apart in magnitude?
- **Direction agreement**: mean cosine of `(A_s − B_s)` vs `(A_s′ − B_s′)`
  over the **same** role pairs -- do two slots agree on the *direction* of each pair's difference?

Outputs five PNGs (default to `roger/token_position_noise_out/`):

| file | content |
|---|---|
| `ooo_heatmaps.png` | Odd-one-out fraction by layer × token (`body-mean`, `t1`…), one panel per metric. |
| `ooo_summary_bars.png` | Token odd-one-out rate per metric (avg across layers); legend uses same labels. |
| `ooo_by_layer.png` | Per-layer token odd-one-out, averaged across the 6 metrics. |
| `pairwise_corr_matrices.png` | Two averaged `S×S` heatmaps: (left) Pearson r of `‖A−B‖` at row vs col token across all unordered pairs, then mean over layers; (right) mean cosine between diff vectors at the two tokens, same pooling. Axes: `body-mean`, `t1`… |
| `pairwise_cosine_by_layer.png` | **`C(S,2)`** curves (28 at *S*=8): for each unordered token pair, mean over unordered role pairs of cosine between `(A−B)` at those positions vs layer (**same statistic as heatmap entries, not ρ**). Turbo colormap legend keyed as `body-mean` ↔ `t1`, etc. |

```bash
# Default: 280 roles × 8 slots × all layers, 1000 triples, on Roger 8slot data.
uv run python results_analysis/token_position_noise_analysis.py

# Run on traits instead, write elsewhere.
uv run python results_analysis/token_position_noise_analysis.py \
  --vectors_subdir traits/vectors \
  --output_dir roger/token_position_noise_traits

# Reproduce the original 4-slot Christina-headers figure.
uv run python results_analysis/token_position_noise_analysis.py \
  --data_dir 'runpod_workspace/qwen/qwen-3-32b Christina headers'
```

**Headline finding** (April 2026 run on the older 4-slot `qwen-3-32b
Christina headers` data, 280 roles × 64 layers × 1000 triples): slot 3
(`\n`) is the *least* noisy slot -- odd-one-out ~21-23 % across metrics
(slightly below the 25 % chance floor) and the highest mean diff-direction
agreement with the other three slots. Slot 0 (body-mean) is the noisiest
by all six metrics; `<|im_start|>` and `assistant` are intermediate. This
empirical finding underwrote the original slot-3 default.

**Pending re-analysis** on 8-slot data: with seven header positions
(slots 1-7 = `<|im_start|>`, `assistant`, `\n`, `<think>`, `\n\n (in)`,
`</think>`, `\n\n (post)`) instead of three, the chance floor for "odd-
one-out" is 12.5 % rather than 25 %, so the absolute numbers shift even
when the relative ranking doesn't. Whether slot 3 remains the project
default depends on this re-run.

### `pca_scree_plots.py`

All-slots PCA scree plots: variance explained per principal component,
overlaid for each slot.  Promoted from the `Compare Scree plots` cell
of `notebooks/pca.ipynb` (April 2026 work).

For a fixed target layer (default 24), fits a separate PCA per slot
on the role-vector matrix at that layer, then renders a 1×3 panel:

- **Linear** -- bars (per-PC variance %) + lines (cumulative %), with
  bars interleaved across slots at width = 0.8 / n_slots so they
  always fit regardless of slot count.
- **Log-linear** -- per-component variance on log y, linear x.
  Highlights tail differences between slots.
- **Log-log** -- log y, log x.  Reveals power-law-like structure.

Auto-scales to whatever `num_slots` is on disk.  At `n_slots=4` it
uses the original `tab10`-style palette (so the figure matches the
notebook's historical colour scheme byte-for-byte); at `n_slots=8`
it switches to the `plasma` palette (matching `all_roles_pairwise_slots.py`,
`role_pair_diff_norms.py`, and the post-update plots in
`token_position_noise_analysis.py`).

```bash
# Default: 8-slot Roger data, layer 24, all roles
uv run python results_analysis/pca_scree_plots.py

# Different layer
uv run python results_analysis/pca_scree_plots.py --layer 25

# Run on traits
uv run python results_analysis/pca_scree_plots.py \
  --vectors_subdir traits/vectors

# Reproduce the original 4-slot figure on Christina headers
uv run python results_analysis/pca_scree_plots.py \
  --data_dir 'runpod_workspace/qwen/qwen-3-32b Christina headers'
```

Output: `roger/pca_scree_plots_out/pca_scree_all_slots_<kind>_L<N>.png`,
e.g. `pca_scree_all_slots_roles_L24.png`.  Runtime is ~15 s end-to-end
(most of it is the per-file `torch.load`; the 8 PCAs themselves take
<1 s combined).

**Headline finding** on the 8-slot Roger data at layer 24: body-mean
(slot 0) has a structurally different PCA spectrum from the 7 header
slots.  Top PC of body-mean explains ~43 % of the variance vs ~19–28 %
for header slots; on the log-linear and log-log views the body-mean
curve sits well below the seven header curves throughout the spectrum.
Header slots cluster tightly with one another -- consistent with the
"newline cluster" 3/5/7 finding from `token_position_noise_analysis.py`.

### `role_pair_diff_norms.py`

For a sample of random role pairs `(A, B)`, plot `‖A[s, L] − B[s, L]‖`
(log y) as a function of layer `L`, one panel per slot `s`. Promoted
from a one-off chat-inline plot (April 2026) that used the old 4-slot
`qwen-3-32b Christina headers` data. Auto-scales the panel grid to
whatever `n_slots` is on disk (4 → 2×2, 8 → 2×4, etc.).

Companion to `token_position_noise_analysis.py`: where that script asks
"are slot-pair *direction-of-difference* signals consistent across
slots?" (Pearson of norms + cosine of diff-vectors), this one asks "how
does the *magnitude* of role-pair difference grow per slot as you walk
up the layers?". The two views together let you spot slots where roles
are well-separated AND consistent, vs slots where they're either crowded
or noisy.

Reading the plot: a tight band climbing from `~10⁰` at layer 0 to `~10³`
at layer 60 means roles are progressively more separated at that slot
as you go deeper -- typical of header slots where the model has finished
accumulating role-specific context. A plateau in the middle layers
(visible at slot 0 / body-mean and at the `<think>`/`</think>` "marker"
slots) means role-pair separation isn't growing -- the representation
isn't accumulating much role-specific differentiation at those depths.

```bash
# Default: 8-slot Roger data, 400 random pairs, all 64 layers.
uv run python results_analysis/role_pair_diff_norms.py

# Run on traits instead.
uv run python results_analysis/role_pair_diff_norms.py \
  --vectors_subdir traits/vectors

# Reproduce the original 4-slot version on Christina headers (2×2 grid).
uv run python results_analysis/role_pair_diff_norms.py \
  --data_dir 'runpod_workspace/qwen/qwen-3-32b Christina headers'
```

Output: `roger/role_pair_diff_norms_out/role_pair_diff_norms_<kind>.png`,
e.g. `..._roles.png` or `..._traits.png`.  Runtime is ~25 s end-to-end
(most of it is `torch.load`).

### `axis_cosine_seriation.py`

Pairwise |cos| heatmap of every axis in the chosen pair-list cohort,
reordered so semantically-related axes sit adjacent to one another,
with an optional **semantic-block ontology overlay** on the y/x axes.
Promoted from a one-off geometry exploration (May 2026) used to
visualise the structure of the clean 60-pair axis space.

For each pair `(pos, neg)` in `pair_list_clean.json` (configurable via
`--pairs`), the script:

1. Loads the layer-25 / slot-6 difference vector `v_pos − v_neg` from
   `traits/vectors` or `roles/vectors` (auto-routed by `pair_type_of`)
   and unit-normalises.
2. Optionally applies soft-shear at depth `L = DEFAULT_SOFT_SHEAR_L = 3`
   (canonical analysis frame; pass `--L 0` for the raw-projection
   comparison).
3. Builds the symmetric `|cos|` matrix and runs **optimal leaf
   ordering** (Bar-Joseph 2001, via `scipy.cluster.hierarchy.
   optimal_leaf_ordering`) on the `(1 − |cos|)`-distance dendrogram.
   OLO is the EXACT optimum for the leaf-permutation problem given a
   dendrogram — clusters land block-diagonal, and within each block
   the most-similar pairs sit adjacent rather than at opposite ends.
4. Overlays `|cos| ≥ threshold` cluster brackets (default 0.5) on the
   heatmap; ships memberships at five thresholds (0.7 / 0.6 / 0.5 /
   0.4 / 0.3) in the JSON sidecar for downstream consumers.
5. Adds outer coloured bands annotating each axis with its semantic
   block from the hand-curated 8-block ontology (`Power & certainty`,
   `Epistemic style`, `Affect`, `Worldview & norms`, `Diligence &
   openness`, `Honesty`, `Alignment & prosocial`, `Social stance`).
   The ontology is annotation only — it doesn't feed into the
   clustering — but seeing the bands next to the seriated heatmap is
   the cleanest way to verify the data-driven geometry agrees with
   the human reading of which traits belong together.

The colormap is `magma` (low |cos| dark, high |cos| bright), chosen
so near-orthogonal pairs read as the visual background and clustered
pairs jump out.

```bash
# Default: clean 60-pair cohort, soft-shear L=3, slot 6 / layer 25.
uv run python results_analysis/axis_cosine_seriation.py

# Raw-frame comparison.
uv run python results_analysis/axis_cosine_seriation.py --L 0

# Goal/non-goal cohort only.
uv run python results_analysis/axis_cosine_seriation.py \
  --pairs pair_list_goalnongoal.json

# Tighter cluster brackets.
uv run python results_analysis/axis_cosine_seriation.py \
  --cluster_threshold 0.6

# Suppress the ontology overlay (for cohorts where it's not curated
# or you want a clean diagonal-only view).
uv run python results_analysis/axis_cosine_seriation.py --ontology none
```

Outputs (under `roger/axis_judge_experiments/`):
- `axis_cosine_heatmap_<cohort>_<frame>.png` (the seriated heatmap
  with ontology bands).
- `axis_cosine_heatmap_<cohort>_<frame>.json` (provenance envelope
  with OLO leaf order, cluster memberships at five thresholds,
  pairwise |cos| stats, and the resolved per-block run structure
  including any block that the seriation fragmented).

Where `<frame>` is `softshear<L>` (or `raw` for `--L 0`). Runtime is
~5 s end-to-end on 60 axes.

**Headline finding** on the clean 60-pair cohort at slot 6 / layer 25:
in the canonical **L=3 soft-shear** frame, the OLO seriation recovers
the hand-curated 8-block ontology **exactly** — 8 runs across 8
blocks, zero splits, blocks contiguous in the order shown above.  In
the **raw** frame the same seriation fragments the same ontology into
**21 runs across 8 blocks, 7 of which are split**: blocks like
`Epistemic style`, `Worldview & norms`, and `Social stance` appear in
4–5 separate stripes each, interleaved with other blocks.  Whitening
isn't manufacturing semantic structure — block-internal axes ARE the
ones the model represents as adjacent in raw space — but it IS
aligning the data-driven geometry with the human ontology, by
attenuating the dominant goal/non-goal direction that otherwise
dominates the cosine structure.  This is a strong empirical
validation of the "whitening sharpens, doesn't create" story
documented under "Whitening / soft-shear defaults" in
`AGENT_NOTES.md`.

Mean off-diagonal |cos|: 0.222 (median 0.188) in L=3 vs 0.245 (median
0.210) in raw — a ~10 % tighten in absolute |cos|, but the ontology-
contiguity story is the more compelling finding.

### `all_roles_pairwise_slots.py`

Companion to `token_position_noise_analysis.py`: shows the **full
per-entity distribution** of pairwise slot statistics across layers, as
transparent ribbons of thin lines (one per entity), rather than scalar
summaries. Promoted from a one-off chat-inline producer (April 2026)
that fed a fortnightly-report figure.

For each entity (default-centred so `δ = entity − default`), each pair
of slots `(s1, s2)`, and each layer `L`:

- **Top panel** -- `cos(δ[s1, L], δ[s2, L])`: do the two slots agree
  on the *direction* the entity differs from baseline?
- **Bottom panel** (log y) -- `‖δ[s2, L]‖ / ‖δ[s1, L]‖`: how does
  the *magnitude* of the deviation differ between slots?

Eighteen slot-pairs across three colour groups, all 280 entity ribbons
overlaid at α = 0.10.  The full 8-choose-2 = 28 subset is still too
dense; three focused groups keep the legend usable:

- **`GROUP_BODY_HEADER`** (7 pairs) -- `(0, k)` for `k ∈ {1..7}`:
  body-mean vs each header token. **`plasma`**.
- **`GROUP_HEADER_TO_SLOT6`** (5 pairs) -- `(k, 6)` for `k ∈ {1..5}`:
  header slots 1–5 vs slot 6 (`</think>`). **`winter`**.
- **`GROUP_HEADER_TO_SLOT7`** (6 pairs) -- `(k, 7)` for `k ∈ {1..6}`:
  header slots 1–6 vs slot 7 (`\n\n (post)` before the model's response).
  **`viridis`** (distinct from plasma + winter).

Four frame variants:

- **A** -- body-vs-header (7) at full alpha; vs-6 and vs-7 pairs dimmed.
- **B** -- vs slot 6 group (5) at full alpha; others dimmed.
- **C** -- vs slot 7 group (6) at full alpha; others dimmed.
- **D** -- all 18 pairs at equal alpha (composite / no deck stacking).

The legend always lists all 18 pairs (inactive pairs dimmed), so A–D
stack as click-to-appear layers without jitter.

The 8 slots are: `body-mean` (slot 0), `<|im_start|>` (1), `assistant`
(2), `\n` (3), `<think>` (4), `\n\n (in)` (5), `</think>` (6),
`\n\n (post)` (7).  Slots 5 and 7 are both `\n\n` text but at
different chat-template positions (inside vs after the `<think>...
</think>` block), and the legend disambiguates them with `(in)` /
`(post)` suffixes.

Whitening (`--whitening`):

- `raw` (default): operate on raw activation differences.
- `soft_K=N`: per-(slot, layer) soft-K whitening, fit on the full
  augmented production pool (roles + traits + corpus default). **No
  leave-one-out** -- the pool is fitted once per `(slot, layer)` and
  reused for every entity. Per-entity LOO would mean ~280 extra SVDs
  per layer per slot, swamping the actual plot work; for pool size
  ~580 the bias from including the entity in its own basis is ~1/580,
  negligible. The whitener is applied to `δ` *before* computing
  cosine and norm, so both panels reflect the whitened metric.

```bash
# Default: all 4 frames (A–D), raw, all 280 roles on Roger 8slot data.
uv run python results_analysis/all_roles_pairwise_slots.py

# Whitened K=3 version (companion frames, one per variant).
uv run python results_analysis/all_roles_pairwise_slots.py \
  --whitening soft_K=3

# Traits only; single PNG with every pair emphasized (composite / no deck).
uv run python results_analysis/all_roles_pairwise_slots.py \
  --vectors_subdir traits/vectors --variant D
```

Outputs default to `roger/all_roles_pairwise_slots_out/`. Filenames use
the entity-kind in the prefix and append `_K=N` when whitening is on:
`all_{roles,traits}_pairwise_slots[_K=N]_{A,B,C,D}.png`. The full layer
set (`--max_layers None`, the default) takes ~25–35 s for the four raw
frames on 8-slot data -- the per-(slot, layer)
SVD count for soft-K mode scales linearly with `n_slots`.

**When to prefer `--whitening soft_K=3`.** Inter-pair structure
(mean of each colour-band) is essentially unchanged by soft-K=3
whitening -- which makes sense: the slot-vs-slot relational signal
lives in the residual subspace, not the top-3 PCs of overall
activation variance. But the **per-role spread** (ribbon width) does
shrink visibly, because the top-3 PCs are where most of the
between-entity amplitude variation lives. Net effect: whitening makes
the all-pairs `D` frame substantially less muddy without erasing
any of the headline patterns. For figures that need to show all six
pairs at once (e.g. report stills without click-to-appear stacking),
the soft-K=3 variant is the readable one; the `A`/`B` frames are fine
either way since they only show three pairs at full alpha.

### `compute_combo_marginals.py`

Computes marginal-mean vectors from the role-by-trait combination grid that
the pipeline produces under `combinations/vectors/r_*__*.pt` and
`combinations/vectors/t_*__*.pt`. Output goes to four subdirectories indexing
one axis at a time:

| subdir | construction (default `--mode residual`) | meaning |
|---|---|---|
| `r_goal/<role>.pt` | mean over partner traits B of `(r_<role>__B - traits/B)` | "average net effect of role A *on top of* its partner trait's main effect". One per goal-supplying role. |
| `r_nogoal/<trait>.pt` | mean over partner roles A of `(r_A__<trait> - roles/A)` | one per non-goal trait |
| `t_goal/<trait>.pt` | mean over partner roles A of `(t_A__<trait> - roles/A)` | one per goal-supplying trait |
| `t_nogoal/<role>.pt` | mean over partner traits B of `(t_<role>__B - traits/B)` | one per non-goal role |

#### Why residual instead of centroid?

The earlier "centroid" reduction (just `mean over partners of combo(A,B)`,
no per-row partner subtraction) is biased whenever the role×trait grid is
incomplete or imbalanced: roles paired with only some of the traits get a
marginal that bakes in the mean of *those particular partners*, which leaks
into downstream subspaces (e.g. shrinks the canonical angle between the
goal and non-goal subspaces in a misleading way).

The residual reduction subtracts the partner's standalone main effect per
contribution before averaging, so each marginal is purely "what indexed
entity A contributes net of its partner". On a fully balanced grid the
residual and centroid subspaces are identical (up to a global shift); on
imbalanced or sparse grids they differ substantially -- in canonical-angle
analyses, by tens of degrees in the median (see
`results_analysis/canonical_angles/README.md`).

The original centroid versions are preserved at
`combinations/vectors/{r_goal,r_nogoal,t_goal,t_nogoal}_legacy_centroid/`
for bit-exact reproduction of the historical PNGs.

#### Storage format

Each `.pt` is a dict `{vector, type, role|trait, metadata}` where the
vector has shape `(n_slots, n_layers, hidden)` in `bf16`, and `metadata`
records `mode` (`'residual'` or `'centroid'`) and `n_partners_averaged`
(30 typically, occasionally 27-29 because some combinations were filtered
out earlier). The mean is computed in `float32` and cast back to `bf16`
for storage to avoid cumulative `bf16` rounding error.

#### Cross-kind combo centroids (auxiliary)

The script also writes two single-vector files at the top level of
`combinations/vectors/`:

- `mean_r_combos.pt` -- mean of all 898 ``r_*__*`` combinations.
- `mean_t_combos.pt` -- mean of all 894 ``t_*__*`` combinations.

These were originally added to the canonical-angles tool's
whitening pool to inject the theatricality direction.  Empirically the
augmentation was nearly inert (CA1 changed <0.1deg at the K values we
care about); the theatricality shift on the subspace side proved
cleaner and more effective, so the canonical-angles tool no longer
uses these files.  See `results_analysis/canonical_angles/README.md`
("Why we don't inject the theatricality direction into the pool") for
the experiment summary.

The files are still produced by default for ad-hoc analyses (they're
cheap).  Pass `--skip_kind_centroids` to skip writing them.

#### Theatricality axis

The script also writes ``combinations/vectors/theatricality_axis.pt``,
a multi-component file with shape ``(n_slots, n_layers, ...)`` carrying:

- ``v_theat`` -- per-(slot, layer) **unit** theatricality direction:

      v_theat ∝ avg over kinds of mean(combo - role - trait + μ_pool)

  i.e. the residual-after-additive-fit direction.  All four
  ``combo_{r,t}_{goal,nogoal}`` residual subspaces project heavily onto
  this single direction; it represents a common-mode "performative
  persona" offset that the additive model needs as a constant term.

- ``default_offset`` -- per-(slot, layer) signed scalar
  ``(default - μ_pool) · v_theat`` (the theatricality coordinate of the
  default vector relative to the held-out pool mean).

- ``shift`` -- the per-(slot, layer) vector ``default_offset × v_theat``.
  Adding this to a ``combo_residual`` marginal relocates it from the
  residual-mean origin to the additive model's true origin.  Used by the
  canonical-angles tool's ``combo_residual_theat_shifted`` aggregation
  (see `canonical_angles/README.md`).

Pass ``--skip_theatricality_axis`` to skip writing this file.

```bash
# Default: regenerate all four subdirs as residual marginals
uv run python results_analysis/compute_combo_marginals.py

# Reproduce the legacy centroid layout (used to live in r_goal/ etc.;
# now preserved at *_legacy_centroid/) -- including default.pt copies:
uv run python results_analysis/compute_combo_marginals.py \
    --mode centroid \
    --include_default r_goal r_nogoal \
    --output_root /tmp/centroid_check

# Generate only one kind/side
uv run python results_analysis/compute_combo_marginals.py --kinds r --sides goal
```

These marginals are the input to the canonical-angles analyses
(`canonical_angles_combos_*.png`, `canonical_angles_goal_vs_nogoal_*.png`)
which compare the goal subspace to the non-goal subspace under various
whitening regimes.

### `variance_decomposition.py`

Decomposes the variance of role+trait combination activations into a
sequence of nested explanatory models, separately for each
`(slot, raw|whitened)` panel.  At each step we report ``Δ R²`` per
dataset (r_ blue, t_ orange) -- the marginal increase in fraction of
variance explained as the next model component is added:

| step | model | what it captures |
|---|---|---|
| (a) | ``role + trait − pool_mean`` | the bare additive baseline |
| (b) | + ``X · v_theat`` (heuristic ``X = pool_mean · v_theat``) | the no-fit guess at the residual direction |
| (c) | + ``X · v_theat`` (optimised ``X``) | the best scalar shift along ``v_theat`` from the additive residuals |
| (d) | + 4-param ``(a, b, c, d) ∈ [0, 1]`` weighted fit | per-kind weight asymmetry between role and trait contributions |
| (rem) | ``1 − R²(d)`` | unexplained residual |

The 4-param model parameterises the predicted combo as
``(a+b)·role + (c+d)·trait`` for r_ and ``(a+d)·role + (b+c)·trait``
for t_, which lets the four "free" coefficients absorb any per-kind
difference in how role and trait contribute.  The optimisation is
shared across r_ and t_ but the reported ``Δ R²`` is computed
per-dataset.

The whitened columns apply soft-K PCA whitening to ``role``, ``trait``,
``combo``, ``default``, and ``v_theat`` before the decomposition.
Default is ``--K 2`` (a single K column plus a "raw" column at the
left -- an `n_metrics × n_slots` grid).  Pass several values for a
wider K-sweep.  The whitening basis is fit per-slot on the standard
augmented canonical-angles pool (held-out roles+traits standalones +
``default.pt``); pass ``--no-augment`` to drop the default augmentation.
``v_theat`` is computed inline per slot (per-row least-squares fit
``a·R + b·T ≈ combo + pool_mean``, mean residual, unit-normalised),
matching the original April 23 reconstruction.  At header slots this
matches the on-disk ``combinations/vectors/theatricality_axis.pt``
direction to cos ≈ 0.99; at slot 0 the inline LSQ direction differs
(cos ≈ 0.72) because the "theatricality offset" is genuinely a
header-slot phenomenon and the slot-0 residual lacks a clear
common-mode direction.

The script auto-detects the number of slots from the loaded data
(`num_slots = default.shape[0]`), so the same script handles 4-slot
Christina-headers data and 8-slot Roger-8slot data.  Per-slot panel
width compresses at higher slot counts (7 in/slot at S=4, 3.5 in/slot
at S=8) so the figure stays under ~32 inches wide.

#### Reading the plot (8-slot Roger data)

The pattern from the original 4-slot finding (header slots 1-3 show a
large step-(b) heuristic theatricality shift; body-mean does not)
extends cleanly to all 7 header slots:

| slot | (a) raw | (b) raw | interpretation |
|---|---|---|---|
| 0 (body-mean) | 68 % | +0.1 % | no theatricality offset (original) |
| 1 (`<\|im_start\|>`) | 42 % | +29 % | strong header offset |
| 2 (`assistant`) | 47 % | +20 % | strong |
| 3 (`\n`) | 51 % | +16 % | strong |
| 4 (`<think>`) | 48 % | +21 % | strong |
| 5 (`\n\n (in)`) | 45 % | +16 % | strong |
| 6 (`</think>`) | 67 % | +12 % | high a (close to body-mean), moderate b |
| 7 (`\n\n (post)`) | 51 % | +12 % | strong |

Slot 6 (`</think>`) is unusual: high additive R² (67 %, comparable to
body-mean) but still has a non-zero theatricality offset.  The other
six header slots cluster: additive R² 42-51 %, theatricality
contribution 12-29 %.  Step (c) and (d) remain small at every header
slot — the heuristic guess is good and the additive model is nearly
weight-symmetric.

At the K=2 column the ``Δ R²`` shifts modestly from ``role+trait`` (a)
into ``heuristic theat_offset`` (b) compared to raw: as whitening
shrinks the top-2 PCs, the additive baseline explains a little less
of the combo variance and the constant theatricality offset becomes
proportionally more important.  Pass a wider K range (e.g. ``--K 1 2
3 4 6 8``) to see the gradient build up monotonically as more PCs are
shrunk.

Slot 0 (body mean) does NOT show the same pattern -- the heuristic
shift is near zero (raw) or only weakly positive (whitened),
consistent with theatricality being absent from the body-mean
activation.  See `canonical_angles/README.md` for further discussion.

```bash
# Default (8 slots × 2 metrics: raw, K=2 wht; layer 25, augmented pool, Roger 8slot data)
uv run python results_analysis/variance_decomposition.py \
    --output roger/variance_decomp_bars.png

# Pies version (each cell is r_ + t_ pie pair)
uv run python results_analysis/variance_decomposition.py \
    --style pies --output roger/variance_decomp_pies.png

# Wider K-sweep at the analysis layer
uv run python results_analysis/variance_decomposition.py \
    --K 1 2 3 4 6 8 \
    --output /tmp/var_decomp_Ksweep.png

# Reproduce the original 4-slot figure on Christina headers
uv run python results_analysis/variance_decomposition.py \
    --data_dir 'runpod_workspace/qwen/qwen-3-32b Christina headers' \
    --output /tmp/var_decomp_4slot.png
```

The reconstructed default plot is at
[`roger/variance_decomp_bars.png`](../roger/variance_decomp_bars.png);
the original April 23 4-slot version (with a single K=128 wht column) is
preserved at
[`roger/variance_decomp_bars_apr23.png`](../roger/variance_decomp_bars_apr23.png)
for direct visual comparison.  Differences vs Apr 23 are small (within
~1-2% on raw panels; the wht panels reshuffle Δ R² between steps (b)
and (c) but their *sum* matches), driven by the augmented pool's
inclusion of ``default.pt`` in the pool-mean computation.

### `whitening_k_sweep.py` and `whitening_k_peak_fit.py`

Per-axis ρ-vs-K analysis tools for the axis-judging-correlation
pipeline.  Use these together when you want to see how soft-K
whitening shifts the agreement between an LLM judge and the
projection onto a chosen axis, and (optionally) localise a "sweet
spot" K via parabolic fits.

**Inputs (per axis pair):** the cached judge scores produced by
[`axis_judge_correlation.py`](#axis_judge_correlationpy), one
subdirectory per axis under `--experiment_dir` (default
`roger/axis_judge_experiments/`):

```
<experiment_dir>/
  pair_list_responses.json         # or any other pair list
  <pos>_vs_<neg>/
    gpt/scores_descriptions.json
    gpt/scores_instructions.json
    sonnet/scores_descriptions.json
    sonnet/scores_instructions.json
    gpt_responses_traits/scores_responses.json
    gpt_responses_roles/scores_responses.json
```

The default `pair_list_responses.json` is the **responses cohort** — every
axis that currently has all six score files cached (desc+instr from both
GPT and Sonnet, plus GPT response-mode judging on roles + traits).
**``pair_list_di.json``** is the wider **desc+instr cohort** — every axis
with desc+instr judging from both providers, regardless of whether
response judging exists.  Many `pair_list_di.json` axes do **not** yet
have GPT response scores, so the `responses` ρ curve is often all-NaN
there (the peak-fit script still plots **desc+instr** and leaves a short
note on empty panels).  Both cohorts grow as more axes get judged; the
filenames stay stable because they encode definition, not count.  Add
new pairs by running the judge pipeline and editing/creating a new pair
list JSON.

#### `whitening_k_sweep.py`

For each axis pair and source ∈ {`desc_inst`, `responses`}, fit a
soft-K whitener on the held-out pool (excluding the two pair
endpoints) at K ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128}, project all
entity vectors onto the (whitened) axis, and compute Spearman ρ
between judge scores and projections.  The `desc_inst` source combines
GPT and Sonnet × descriptions and instructions per entity using
``assistant_axis.judge_score_combine.combine_desc_inst_two_judges``
(default: inst-tiebreak weighting `0.499*desc + 0.501*inst`; see
"Combining desc/inst scores" below); the `responses` source merges GPT
response-mode mean scores from the roles + traits runs.

Outputs to `--experiment_dir`:

- `whitening_k_sweep.json` -- N × 2 × 9 records of `{pos, neg, source, K, rho}`.
- `rho_vs_whitening_K.png` -- one solid (responses) and one dotted
  (desc+inst) line per axis, colored by axis, with K on the x-axis.

```bash
# Default: responses cohort (writes whitening_k_sweep_responses_slot6.json
# + rho_vs_whitening_K_responses_slot6.png)
uv run python results_analysis/whitening_k_sweep.py

# Desc+instr cohort at slot 7 (writes whitening_k_sweep_di_slot7.json
# + rho_vs_whitening_K_di_slot7.png; auto-derive embeds the cohort token
# from --pairs so cohorts can never silently overwrite each other)
uv run python results_analysis/whitening_k_sweep.py \
  --pairs pair_list_di.json --slot 7
```

#### `whitening_k_peak_fit.py`

Reads a sweep JSON (auto-derived to `whitening_k_sweep_<cohort>_slot{N}.json`)
and fits a parabola to each ρ-vs-K curve over K ∈ [0, 32] in two parametrisations:

- linear-K:  ``ρ(K) ≈ a + b·K + c·K²``
- log₂(K+1): ``ρ(K) ≈ a + b·log₂(K+1) + c·log₂(K+1)²``

**Defaults:** `--pairs pair_list_di.json` so the mosaic has one panel per
desc+instr-cohort axis (currently 33, in a **6×6** grid with three cells
blank).  Output basenames auto-derive to
`whitening_k_peak_fit_<cohort>_slot{N}.json` /
`rho_vs_K_parabolic_fits_<cohort>_slot{N}.png` / sweep input
`whitening_k_sweep_<cohort>_slot{N}.json` -- where `<cohort>` comes from
`--pairs` (`di` or `responses`) and `{N}` from `--slot` (default 6).

The fitter skips NaN ρ points (usual on **`responses`** when GPT response
scores are missing for that axis).  Peak-K agreement (**desc+inst** vs
**responses**) is only printed when both peaks are finite (typically the
responses cohort).  Empirically the log parametrisation fits better on
fully-scored axes.

Outputs to `--experiment_dir`:

- `whitening_k_peak_fit_<cohort>_slot{N}.json` -- per-curve fit records
  (R² per parametrisation, fitted peak K, y-hats at observed K).
- `rho_vs_K_parabolic_fits_<cohort>_slot{N}.png` -- one panel per
  **`--pairs` entry**, in order.

```bash
# Desc+inst cohort at slots 3/6/7
uv run python results_analysis/whitening_k_peak_fit.py --slot 3
uv run python results_analysis/whitening_k_peak_fit.py --slot 6
uv run python results_analysis/whitening_k_peak_fit.py --slot 7

# Responses cohort at slot 6 (smaller mosaic)
uv run python results_analysis/whitening_k_peak_fit.py \\
    --pairs pair_list_responses.json --slot 6
```

#### `gpt_vs_sonnet_scatter.py`

Visualises GPT-vs-Sonnet judge agreement at the per-entity level.
For each axis, computes ``both_mean = (descriptions + instructions) / 2``
per provider and produces two scatter plots:

- ``gpt_vs_sonnet_scatter_pooled.png`` -- one figure with all axes'
  points overlaid (colored by axis), showing the pooled Spearman ρ
  across the ``(entity, axis)`` set.
- ``gpt_vs_sonnet_scatter_grid.png`` -- N-panel grid (one panel per
  axis, layout auto-sizes to ⌈N/4⌉ rows of 4) with each panel's own ρ.

Small Gaussian jitter (σ=0.08, deterministic seed) is added to both
axes since judge scores are integer-valued in [-3, 3] and many points
overlap exactly without it.

Outputs to ``--experiment_dir``:

- the two PNGs (auto-named ``gpt_vs_sonnet_scatter_{pooled,grid}_<cohort>.png``;
  paths overridable via ``--pooled`` / ``--grid``).
- ``gpt_vs_sonnet_rhos_<cohort>.json`` -- per-axis + pooled Spearman ρ.

```bash
# Default: desc+instr cohort (every axis with desc+inst from both GPT
# and Sonnet cached on disk).  Auto-writes
# gpt_vs_sonnet_{rhos,scatter_pooled,scatter_grid}_di.{json,png}.
uv run python results_analysis/gpt_vs_sonnet_scatter.py
```

#### `gpt_sonnet_weight_sweep.py`

Sweeps the GPT/Sonnet weight in the desc+inst score average and plots
the mean per-axis projection-ρ as a function of the blend.  For each
weight ``w ∈ [0, 1]`` and each axis we compute, per entity::

    score(w) = w · ((GPT_d + GPT_i) / 2) + (1 - w) · ((Son_d + Son_i) / 2)

then take Spearman ρ vs the raw activation projection at slot 3,
layer 25, and average across all axes in the pair list.  ``w = 0.5``
corresponds to the 4-way mean of ``{GPT_d, GPT_i, Son_d, Son_i}`` we
already use across the canonical-angles + axis-judge tooling.

**Empirical finding (33 axes):**

| weight on GPT | mean ρ |
|---:|---:|
| 0.0 (pure Sonnet) | 0.5693 |
| 0.5 (50/50) | **0.5953** |
| 1.0 (pure GPT) | 0.5758 |

Mean ρ is essentially flat over ``w ∈ [0.1, 0.9]``: a parabola fit
restricted to that interior interval places the peak at ``w = 0.530``
(R² = 0.971) with ρ ≈ 0.5949, which is *0.0004 below* the discrete
50/50 value.  The endpoint kinks at ``w = 0`` and ``w = 1`` reflect a
categorical regime change ("averaging two judges" vs "single judge"),
not a smooth blend, and are excluded from the parabola fit.

**Decision.**  Continue to score desc+inst with both GPT-4.1-mini and
Sonnet-4 by default.  When reducing to a single score per entity, the
helper `assistant_axis.judge_score_combine.combine_desc_inst_two_judges`
takes the GPT/Sonnet mean separately for desc and inst, then combines
them using the **inst-tiebreak weighting** `0.499*desc + 0.501*inst`
(see "Combining desc/inst scores" section below for the empirical
rationale).  Cost: ~6× a single-provider run for ~+0.019 ρ
improvement over GPT alone (the cheaper provider) and ~+0.026 over
Sonnet alone.  Negligible per-axis sensitivity to the exact mix
(within ±0.01 of peak across ``w ∈ [0.1, 0.9]``).

**GPT-vs-Haiku as a sister sweep** (May 2026, same 33 axes; pass
``--second_judge haiku`` to ``gpt_sonnet_weight_sweep.py``):

| weight on GPT | GPT+Sonnet ρ | GPT+Haiku ρ | Δ (Sonnet − Haiku) |
|---:|---:|---:|---:|
| 0.0 (pure other-judge) | +0.5929 | +0.5700 | +0.0229 |
| 0.5 (50/50) | **+0.6222** | **+0.6147** | +0.0076 |
| 1.0 (pure GPT) | +0.6020 | +0.6020 | 0 (sanity) |
| parabolic peak (interior fit) | w=0.55, ρ=+0.622 | w=0.70, ρ=+0.617 | +0.005 |

Two notable points:
1. **Haiku alone is meaningfully worse than Sonnet alone** (Δ = -0.023 ρ)
   -- Haiku 4.5 trades quality for ~3× lower cost than Sonnet 4.
2. **The Haiku weight sweep peaks at w*=0.70 vs Sonnet's w*=0.55** --
   when blending with Haiku, the optimum puts more weight on GPT, which
   is consistent with "Haiku contributes less per unit weight".  The
   peak ρ values themselves differ by only +0.005 ρ.

Plot at ``roger/axis_judge_experiments/gpt_haiku_weight_sweep.png``
(sister to the existing ``gpt_sonnet_weight_sweep.png``).

**Haiku as second judge: cost-efficient alternative to Sonnet.**
Across **6 (slot, layer) × metric cells** (``(3, 25)``, ``(0, 26)``,
``(0, 49)`` × ``{raw, soft_K=3}``), the three headline questions had
strikingly consistent answers (May 2026):

| Question | Mean Δρ across 6 cells | Per-axis wins |
|---|---:|---|
| Q1: ρ(Haiku alone) − ρ(Sonnet alone)         | **−0.020** | Haiku wins 12-17 / 33 axes |
| Q2: ρ(GPT+Haiku) − ρ(GPT alone)              | **+0.013** | GPT+Haiku wins 24-26 / 33 |
| Q3: ρ(GPT+Sonnet) − ρ(GPT+Haiku)             | **+0.008** | GPT+Sonnet wins 16-20 / 33 |

The findings are robust across (slot, layer) and across raw vs soft-K=3
whitening -- the per-cell Δρ values lie in tight bands around these
means.  Translation:

- **Sonnet alone is a better single judge than Haiku alone** (Δ ≈ +0.02 ρ).
- **Adding Haiku to GPT helps** (~75-80% of axes improve, mean +0.013 ρ).
- **Sonnet is a slightly better diversity partner than Haiku** for a
  GPT+J ensemble (Δ ≈ +0.008 ρ at peak), but not by much.

Cost-quality with Haiku 4.5 vs Sonnet 4 (roughly 3× cheaper):

|   | GPT alone | GPT + Haiku | GPT + Sonnet |
|---|:---:|:---:|:---:|
| desc+inst $ / axis | ~$0.95 | ~$2.85 | ~$8.05 |
| Δρ vs GPT alone    | (ref)  | +0.013  | +0.020 |
| $ per +0.01 ρ      | (ref)  | **~$1.46**  | **~$3.55** |

**Haiku is ~2.4× more cost-efficient as a second judge than Sonnet**;
for full corpus sweeps where the +0.007 ρ delta isn't critical,
``GPT + Haiku`` is the cost-quality winner.  Sonnet remains the right
second judge for paper-grade comparisons or specific axes where
Sonnet's value-laden-axis advantage matters (see "GPT-better /
Sonnet-better" tables in the previous subsection).

The multi-config result was generated by ``/tmp/_judge_provider_compare_multi.py``
(saved JSON: ``roger/judge_provider_compare_multi.json``).  If this
becomes a recurring pattern, promote it to
``results_analysis/judge_provider_compare.py`` -- for now it's a
one-off.

The plot also colors each per-axis curve by its interior slope
``Δρ = ρ(w=0.9) − ρ(w=0.1)`` (blue → Sonnet-better, red → GPT-better)
and orders the legend the same way, so you can read off which axes
each provider is stronger at.  Headline pattern: GPT tends to win on
*cognitive-style / register* axes (systems_thinker, improvisational,
casual, passionate, confident, playful, convergent) while Sonnet
tends to win on *moral / value-laden* axes (progressive, ethereal,
harmless, individualistic, generous, idealistic, reductionist,
practical).

```bash
# Default: 33 axes
uv run python results_analysis/gpt_sonnet_weight_sweep.py
```

#### TODO: per-axis judge-mix optimisation

The current default uses a single 50/50 mix for every axis.  For
axes where one provider is clearly stronger (per the "GPT-better /
Sonnet-better" tables above; effect sizes 0.02–0.09 ρ), a per-axis
optimisation of the mix should give a small but real improvement on
those axes.  A simple test: for each axis, compute ρ at five
candidate mixes ``w ∈ {0, 0.1, 0.5, 0.9, 1.0}`` and pick the best;
project the gain (vs uniform 50/50) across the 33-axis set.  If the
gain is meaningful (>~0.01 mean ρ across axes), make the choice
data-driven; otherwise document and stick with 50/50 as the
universal default.  Should also include "weighted-by-fit-quality"
versions analogous to ``whitening_k_weighted_scatter.py``'s confidence
scheme.

Pooled ρ across the 33 axes is **+0.862 (n=18632)**, slightly higher
than the older 7-axis number (+0.816, n=3976) and the 12-axis number
(+0.817, n=6816) -- adding clean morality / register / cognitive-style
pairs raises the average agreement.  Per-axis ρ ranges from 0.681
(systems_thinker/analytical -- the messiest) to 0.923 (helpful/unhelpful)
and 0.922 (trustworthy/untrustworthy).  This is the empirical basis
for the README's ["GPT vs Sonnet agreement"](#gpt-vs-sonnet-agreement)
finding: GPT-4.1-mini is statistically equivalent to Sonnet 4 on
desc+inst judging at ~5× lower cost.

The desc+instr cohort pair list ``pair_list_di.json`` is auto-discovered
from on-disk score files: every ``<pos>_vs_<neg>/`` directory under
``--experiment_dir`` that has all four
``{gpt,sonnet}/scores_{descriptions,instructions}.json`` files is
included.  Add new pairs by running the judge pipeline; the next
re-build of ``pair_list_di.json`` will pick them up.

### `optimal_axis_for_judge.py` -- direction-fitting from judge scores

Given a fixed metric ``M = (slot, layer, whitening)`` and a judge-score
dict ``{entity_name: float}``, this tool returns the unit direction
``d`` in ``M``-space that **maximises Spearman ρ** between the judge
scores and entity projections ``<entity_i, d>``.  Output is a unit
vector (in both the original hidden space and a working PC basis), plus
diagnostics that report how confident we are that ``d`` is
well-determined (cosine clustering across multi-restart).

#### Why

Almost every existing axis-analysis tool fixes the *direction* (typically
``vec[pos] - vec[neg]`` in hidden space) and varies geometry around it
(whitening K, soft-shear L, slot, layer).  This tool inverts the
question: given the judge has expressed a preference, what *direction*
in the chosen metric best captures it?  The recovered ``d`` then enables
- cosines with other directions (e.g. how aligned is the judge-implied
  direction with the pole-difference axis?  with PC1?  with another
  axis's optimal direction?);
- 2-D plotting planes built from two learned directions;
- round-tripping the direction back through ``infer_axis_description.py``
  to ask the LLM what semantic axis it just learned;
- diagnosing cross-judge / cross-source agreement geometrically.

#### Algorithm

ρ is piecewise-constant in ``d`` (changing only when two entity
projections swap order), so it has zero gradient almost everywhere.
The trick is **exact 1-D optimisation per coordinate via crossing
points**: holding all other coordinates fixed, the score
``score_i(d_k) = b_i + d_k · α[i, k]`` is linear in ``d_k``, so rank
order changes only at ``O(N²)`` discrete crossings
``d_k* = (b_j - b_i) / (α[i, k] - α[j, k])``.  Sort the crossings, sweep,
incrementally update ``Σd² = Σ(rank_proj − rank_judge)²`` with
``Δ = 2·(rank_i − rank_j)·(judge_rank_i − judge_rank_j)`` per swap, and
the interval (or unbounded tail) with smallest ``Σd²`` is the global
1-D optimum.  This is a generalisation of the per-PC-β optimum in the
original ``coordinate_ascent.py`` (in ``roger/axis_judge_experiments/``,
where it was used for whitening-scale optimisation along a fixed
direction); the inner sweep is JIT-compiled with numba so a per-coord
update on N≈580 entities takes a few ms instead of ~150 ms in pure
Python.

The full algorithm:

```
1. Project entities to top-M PC basis (default M=100):
     alpha[i, k] = <whitened, centred entity_i, PC_k>
2. Coordinate ascent over the M PC coefficients of d:
     round-robin permuted PC order; exact 1-D optimum per coord;
     renormalise after each update; stop on max |Δd_k| < ε.
3. Multi-restart with structured seeds (default n_restarts = 16):
     - pole-difference axis (if --seed_pair given) -- the "baseline-to-beat"
     - score-weighted mean Σ_i (s_i − μ) · α[i, :] -- the linear-regression-y baseline
     - Fisher LDA from tercile-binned scores
     - K_seed top-PC-aligned with score-correlated sign
     - random unit vectors fill to n_restarts
4. Cosine cluster the converged directions:
     |cos|-cluster at threshold 0.95; rank by mean ρ;
     re-run coord ascent from the top-cluster centroid
     (consensus seed); adopt if ρ improves.
5. Optional: 5-fold CV on the judge scores -- per-fold fit on train,
     evaluate ρ on held-out; aggregate to one rho_cv scalar.
6. Compute headline cosines vs pole-diff axis and vs score-weighted mean
     (which one of these the optimum looks most like is informative).
```

The M-PC restriction (step 1) is the main anti-overfitting move: with
N ≈ 580 entities and D = 5120 raw, fitting in 5120-D would memorise
judge noise.  M = 100 covers ≥ 99% of pool variance for the standard
``slot=3, layer=25, soft_K=3`` setup, with N/M ≈ 6× more samples than
free parameters.  Override with ``--M``.

#### Output

`direction.pt`:

```python
{"direction":     torch.Tensor (D,)  unit, in original hidden space,
 "direction_pcs": torch.Tensor (M,)  same direction in PC basis,
 "pc_basis_Vt":   torch.Tensor (M, D)  PC basis used (for callers that
                                       want to express other vectors in
                                       the same coords),
 "metric":        {slot, layer, whitening, M, ...}}
```

`diagnostics.json` includes:
- `rho_train`, `rho_cv` (paired training and cross-validated ρ)
- `cluster_summary.top_cluster` -- size, mean ρ, intra-cluster cosine
  spread (min / median).  Top-cluster `min_pairwise_cos ≥ 0.95` =
  high confidence; `≤ 0.5` = multiple distinct directions giving
  comparable ρ, the corpus isn't constraining enough to pin one down.
- `cos_to_pole_diff`, `rho_pole_diff` -- the canonical baseline.  ≥ 0.95
  cosine = optimum is just the pole axis; 0.7-0.95 = corpus is gently
  steering off the pole-diff line; ≤ 0.5 = surprising, possibly
  informative or possibly low-SNR.
- `cos_to_score_weighted_mean`, `rho_score_weighted_mean` -- linear-
  regression-y baseline; if cos > 0.97, gradient descent isn't doing
  anything beyond what a one-line numpy expression would.
- `restarts` -- per-restart seed label, final ρ, and cluster id.

`restarts.png`: scatter of (restart index, final ρ_train) coloured by
cluster, with horizontal lines for the two baselines, for at-a-glance
multi-modality check.  Embeds full source as PNG metadata.

#### CLI

Two modes: pass either a raw scores JSON, or point at an existing
axis-judge-correlation experiment dir with a `--score_source` selector
(``gpt`` / ``sonnet`` / ``haiku`` / ``di_combined`` (GPT+Sonnet 4-way) /
``responses``).

**Output directory layout:** runs are treated as a "fill-on-demand"
cache.  When `--output_dir` is omitted, it auto-derives from the full
cache key:

```
roger/optimal_axis/<pair>_<src>_slot{N}_layer{L}_<whitening>_M{M}_<data_dir_slug>/
```

Every dimension that could change between invocations is encoded
explicitly so a future default flip (e.g. slot=3 → slot=6 in May 2026,
or `data_dir` swap from 4-slot to 8-slot) lands in a distinct directory
without silently shadowing prior outputs.  Pass `--output_dir` to
override entirely (use it for one-off comparison runs you want to
hand-name).

```bash
# Mode A: raw scores file ({name: float} or
#         {name: {scores: [list of floats]}} -- both accepted).
# Auto-derives to roger/optimal_axis/my_label_raw_scores_slot6_layer25_softK3_M30_<data>/
uv run python results_analysis/optimal_axis_for_judge.py \
  --scores_file my_label.json \
  --slot 6 --layer 25 --whitening soft_K=3 --M 30

# Mode B: existing axis-judge-correlation experiment dir.
# Auto-derives to roger/optimal_axis/truthful_vs_deceitful_di_combined_slot6_layer25_softK3_M30_<data>/
uv run python results_analysis/optimal_axis_for_judge.py \
  --experiment_dir roger/axis_judge_experiments/truthful_vs_deceitful \
  --score_source di_combined \
  --seed_pair truthful deceitful \
  --exclude_names truthful,deceitful \
  --slot 6 --layer 25 --whitening soft_K=3
```

Library:

```python
from results_analysis.optimal_axis_for_judge import find_optimal_direction

result = find_optimal_direction(
    judge_scores={"angel": 2.5, "demon": -2.8, ...},
    data_dir=Path("..."),
    slot=3, layer=25, whitening="soft_K=3",
    M=100, n_restarts=16, cv_folds=5,
    seed_pair=("truthful", "deceitful"),
)
# result.direction       : np.ndarray (D,) unit vector in original metric
# result.direction_pcs   : np.ndarray (M,) coords in PC basis
# result.rho_train       : float
# result.rho_cv          : float | None
# result.cluster_summary : dict
# result.restarts        : list[RestartResult]
```

#### Cost / scope

CPU-only after the existing scores files are in.  With numba-JIT, a
single-source run at ``N ≈ 580, M = 100, n_restarts = 16, no CV`` is
~2 min wall-clock; adding 5-fold CV roughly 6× that (~10 min).  Trivial
parallelism per source.  Zero new API spend.

#### Default tuning (May 2026)

The defaults `M=30, n_restarts=8` were chosen from a sweep on
`truthful_vs_deceitful / di_combined` at the canonical
`slot=3, layer=25, soft_K=3` setup:

| M  | ρ_train | ρ_cv  | gap   | top cluster | top med cos | wallclock |
|----|---------|-------|-------|-------------|-------------|-----------|
| 20 | +0.813  | +0.787| 0.026 | 8/8         | 0.995       | 29 s      |
| **30** | **+0.837** | **+0.815** | **0.022** | 8/8 | 0.992 | **40 s** |
| 50 | +0.853  | +0.787| 0.066 | 8/8         | 0.969       | 71 s      |
| 75 | +0.864  | +0.795| 0.069 | 1/8         | 1.000       | 94 s      |
| 100| +0.885  | +0.783| 0.102 | 1/8         | 1.000       | 135 s     |

`M=30` maximises ρ_cv with the smallest train/CV gap.  `M >= 50`
overfits (ρ_train climbs while ρ_cv drops); `M >= 75` fragments the top
cluster (per-restart directions diverge under the strict `cos >= 0.95`
threshold).  Below M=30 the working subspace is too small to capture
the available signal.

`n_restarts=8` covers the four structured seeds (pole-diff,
score-weighted mean, Fisher LDA, top-K-axis-aligned) plus 4 random
fills.  At M=30 all 8 routinely converge to the same cluster
(intra-cluster median cos > 0.99), so 8 is comfortably enough; 16 was
the original cautious default and proved overkill.

#### Pilot finding (May 2026, ``truthful_vs_deceitful``, 5 sources)

Run with `slot=3, layer=25, soft_K=3, M=30, n_restarts=8, cv_folds=5`
across the 5 judge sources currently available for
`truthful_vs_deceitful` (GPT desc+inst, Sonnet desc+inst, Haiku
desc+inst, GPT+Sonnet 4-way, GPT responses-mode):

| source       | ρ_train | ρ_cv  | gap   | cos→pole | ρ_pole | cos→swm | ρ_swm | top cluster |
|--------------|---------|-------|-------|----------|--------|---------|-------|-------------|
| gpt          | +0.808  | +0.772| 0.035 | 0.626    | +0.558 | 0.702   | +0.714| 8/8         |
| sonnet       | +0.823  | +0.790| 0.032 | 0.603    | +0.569 | 0.693   | +0.724| 8/8         |
| haiku        | +0.835  | +0.798| 0.037 | 0.691    | +0.571 | 0.730   | +0.744| 7/7         |
| di_combined  | +0.837  | +0.815| 0.022 | 0.654    | +0.588 | 0.720   | +0.742| 8/8         |
| responses    | +0.890  | +0.871| 0.018 | 0.804    | +0.751 | 0.809   | +0.781| 8/8         |

Cross-source `|cos|` of the recovered directions (in PC basis,
M=30; random-baseline `E[|cos|] ≈ 0.146`):

|             | gpt   | sonnet| haiku | di_comb | resp  |
|-------------|-------|-------|-------|---------|-------|
| gpt         |   -   | 0.927 | 0.912 | 0.974   | 0.600 |
| sonnet      | 0.927 |   -   | 0.932 | 0.974   | 0.624 |
| haiku       | 0.912 | 0.932 |   -   | 0.938   | 0.708 |
| di_combined | 0.974 | 0.974 | 0.938 |   -     | 0.644 |
| responses   | 0.600 | 0.624 | 0.708 | 0.644   |   -   |

**Findings:**

- **The optimisation works**: all sources clear the canonical
  pole-difference baseline by **+0.10 to +0.21 ρ_cv** and the
  score-weighted-mean linear-regression baseline by **+0.06 to +0.13**.
  The crossing-points coord ascent is doing real work neither baseline
  catches.
- **Convergence is rock-solid at M=30**: 8/8 restarts in the top cluster
  for every source, with intra-cluster median cosine `>= 0.98`.  The
  multi-modality concern doesn't materialise at this dimensionality.
- **Train/CV gap is small (~0.03)**: the M=30 fit generalises well;
  the optimisation isn't memorising judge noise.  At M=100 the gap was
  ~0.10 -- confirming M=30 is the right operating point.
- **Cross-source consistency within desc+inst is very high**
  (cos 0.91 - 0.97).  GPT, Sonnet, Haiku, and the GPT+Sonnet combo all
  find essentially the same direction in PC space -- judge-model choice
  has small geometric impact when fitting a direction.
- **Responses-mode finds a related but distinct direction**
  (cos 0.60 - 0.71 with desc+inst).  Same axis, ~40-50 degrees of
  separation in M=30-D PC space -- a real signed difference between
  "judge a description of how the entity behaves" and "judge actual
  responses the model produced under the role".
- **The optimum is meaningfully off the pole-difference axis**
  (cos 0.60 - 0.80, ρ-lift of +0.15 - +0.25 over pole-diff): the
  corpus is voting for a direction that's clearly related to but not
  the same as `vec[truthful] - vec[deceitful]`.  Worth running
  `infer_axis_description.py` on the recovered direction to see what
  semantic axis the judge data is actually picking out -- a natural
  Phase 2 follow-up.

(For comparison: the original M=100 run -- before the May 2026 retune
-- gave `avg ρ_cv = 0.794, avg gap = 0.088, cross-source cos within
desc+inst = 0.67-0.91, cos to responses = 0.19-0.32`.  At that M
overfitting was inflating ρ_train and pushing apart the per-source
directions; the M=30 numbers are the correct headline.)

### `whitening_k_weighted_scatter.py`

Cross-axis summary plot: per-axis fitted peak K from desc+inst (x)
vs responses (y), in `log₂(K+1)` space, marker area ∝ per-axis fit
quality.  Asks "does the whitening sweet spot transfer across judge
sources, or is it source-specific?".

The plot reads `whitening_k_sweep.json` and refits the log-space
parabolas (the same fit used by `whitening_k_peak_fit.py`) so the
extra (a, b, c, R², |c|) intermediates are available locally; it then
combines them into a per-axis confidence weight using a few moderately
principled heuristics:

```
confidence  = peak_ρ × R² × √|c|         # per-curve
              ↑          ↑     ↑
              signal     fit   curvature (sharp vertex)
              strength   qual. → narrow peak-K uncertainty

weight      = √( confidence_di
                 × min(1, peak_ρ_di / peak_ρ_rs)   # cross-source rebalance:
                 × confidence_rs )                  # weak d+i can't
                                                    # outvote strong rs
```

Peaks falling below K=0 or above K=32 (i.e. the parabola's vertex
extrapolates outside the measured range) are clipped to the boundary
for display: a negative K is meaningless (raw is already K=0; you
can't whiten "negative" PCs), so the principled stand-in is "data
never rose above raw".

Reports unweighted Pearson + Spearman correlations and the weighted
Pearson, plus a weighted-OLS line drawn alongside the y=x reference.

Outputs to `--experiment_dir`:

- `rho_peak_K_weighted_scatter.png`
- `whitening_k_peak_fit_weighted.json` -- per-(axis, source) fit
  records including the (a, b, c) coefficients, R², peak_log,
  peak_rho, |c|, and confidence; plus the per-axis weight.

```bash
# Default: 12 axes
uv run python results_analysis/whitening_k_weighted_scatter.py
```

#### What the curves show

Across the 12-axis set, three qualitative patterns emerge:

- **Strongly aligned axes** (helpful, harmless, honest, truthful):
  ρ is highest at K=0 and declines monotonically as K grows -- the
  axis lives close to the dominant variance directions, and shrinking
  the top PCs only erodes the signal.
- **Mid-K peakers** (progressive, concise, ecocentric, systems_thinker):
  ρ peaks at K∈[2, 8], then declines.  These axes have a meaningful
  alignment with mid-rank PCs, and aggressive whitening helps reveal
  it.
- **Flat curves** (relativist, improvisational): ρ is roughly constant
  across K -- the axis-judge agreement is largely insensitive to the
  whitening regime.

The desc+inst-vs-responses peak agreement (Spearman ρ across the 12
axes, log space, clipping non-finite peaks) sits around **0.35** --
modest agreement: clean morality axes tend to peak at K=0 in both
sources, while structural axes show substantial spread between
sources, suggesting the "right" K is partially source-specific.

#### `rho_by_slot_and_K.py`

4-panel grouped histogram: per-slot mean per-axis Spearman ρ at
several whitening levels, for two judge sources side-by-side.  Lets
you eyeball how ρ varies with slot and K simultaneously, across the
full 33-axis desc+inst panel and the 12-axis GPT-responses panel,
without picking a particular axis.

For each slot ∈ {0, 1, 2, 3} and each ``K ∈ KS`` (default
``[0, 1, 2, 3, 4, 5]``, where K=0 = raw, K>0 = soft-K whitening on
the augmented held-out pool with only the 2 axis endpoints removed)
we compute mean per-axis Spearman ρ between the projection at
``(slot, layer=25)`` and:

- **desc+inst** -- 33 axes, score per entity is
  `combine_desc_inst_two_judges(GPT_d, GPT_i, Son_d, Son_i)` --
  inst-tiebreak weighting (`0.499*desc + 0.501*inst`) by default;
- **responses** -- 12 axes, GPT-only mean over response-mode evals.

Output (to ``--experiment_dir``):

- ``rho_by_slot_and_K.png``

```bash
# Default: KS = [0, 1, 2, 3, 4, 5], 33 + 12 axis pair lists
uv run python results_analysis/rho_by_slot_and_K.py

# Custom K range
uv run python results_analysis/rho_by_slot_and_K.py --ks 0 2 4 8 16
```

Headline observations from the default run (Qwen-3-32B layer 25,
the analysis point):

- **Slot 3 wins at every K** for both sources -- desc+inst ρ
  rises from 0.621 raw to 0.640 at K=2, ≈flat across K=2…4, then
  erodes at K≥5; responses follow the same shape with a peak at
  K=2 (0.654).
- **Slot 0 is the laggard at raw** (desc+inst 0.536, responses
  0.574) but **catches up to slot 3 -- and *overtakes* slot 3 for
  responses -- once any whitening is applied**: slot 0 K=2
  responses ρ = 0.668 vs slot 3 K=2 = 0.654.  Slot 0 jumps +0.09
  ρ on responses going K=0 → K=2 while slot 3 only gains +0.02.
  This is the theatricality-PC story: at raw, slot 0's
  representation is dominated by a single PC that's largely
  orthogonal to the axis-judge signal; shrinking the top few PCs
  unmasks the remaining identity-encoding subspace.
- Slots 1 and 2 sit ~0.05 ρ below the slot 0/3 pair at every K
  for desc+inst, and respond to whitening similarly (peak at
  K=3-4, modest +0.04 lift over raw).

#### `rho_by_layer.py`

Multi-row × 2 column plot: per-layer mean per-axis Spearman ρ vs
transformer layer for several token slots × two judge sources, with raw
(K=0) plus soft‑K whitening overlays.  Shows *which layers* carry
judge-aligned signal and how that depth profile interacts with
whitening.

- Rows (see ``SLOTS`` in the script; May 2026 default): slot 0
  (body mean), slot 3 (newline after the ``assistant`` header token),
  slot 6 (``</think>``), slot 7 (the following blank / newline
  run right before the assistant content).
- Columns: ``desc+inst`` (33 axes, GPT+Sonnet × desc/inst combined via
  inst-tiebreak weighting) and ``responses`` (12 axes, GPT-only
  response-mode mean).
- Curves: default plots K ∈ ``{0, 1, 2, 3, 4}`` — raw plus four
  softened levels (thinner traces with smaller markers; high‑K curves
  are drawn underneath, raw last so it sits on top).  Pass ``--ks`` to
  change the set without editing the script.
- Faint vertical gridlines at every even layer.

The SVD behind soft‑K whitening is computed *once* per
(slot, layer, leave-out-set) and shared across all plotted K values.
Wall time scales roughly with layers × slots × union(desc+inst,
responses) distinct axis pairs (~tens of minutes on a laptop for the
defaults with four slots).

Outputs (to ``--experiment_dir``):

- ``rho_by_layer.png``
- ``rho_by_layer.json`` -- per-(slot, layer, K, source) mean ρ
  table.  Used by ``--replot_from_json`` to skip the SVD (honours
  ``--ks``; subplot rows honour ``slots`` recorded in the JSON), or by
  ``--reuse_json`` to fill in missing (slot, layer, K, source)
  entries after extending ``SLOTS`` or changing ``--ks`` / ``--layers``.

```bash
# Default: all layers, slots 0+3+6+7, Ks = [0, 1, 2, 3, 4]
uv run python results_analysis/rho_by_layer.py

# Faster: a coarser layer sample
uv run python results_analysis/rho_by_layer.py \
    --layers 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60

# Cheap: replot from the cached JSON (no recomputation)
uv run python results_analysis/rho_by_layer.py --replot_from_json

# Incremental after editing SLOTS in the script: reuse ρ for old slots;
# computes only missing (slot, layer, K) tuples
uv run python results_analysis/rho_by_layer.py --reuse_json
```

Headline observations from the default run:

- **Slot 0 vs slot 3 reveal completely different whitening
  behaviours.**  At slot 0 the whitened curves sit ~0.05--0.10 ρ
  *above* raw at every layer (the gap is essentially constant
  across the 7 K values -- a single dominant PC is doing all the
  work, consistent with the theatricality direction identified in
  the canonical-angles analysis).  At slot 3 raw and whitened
  curves are tightly bunched throughout: the slot-3 representation
  has already implicitly factored that PC out.
- **The slot-3 jump at layer ~25** is sharp and unmistakable:
  desc+inst ρ rises from ~0.43 at layer 22 to ~0.62 at layer 25,
  then slowly decays to ~0.48 by layer 60.  Layer 25 is the
  current Qwen-3-32B analysis-point default (in
  ``rho_by_slot_and_K.py``, ``whitening_k_sweep.py``,
  ``gpt_sonnet_weight_sweep.py``, etc.): it sits at the start of
  the high-ρ plateau without paying the late-layer decay penalty.
  Slot 0 has a smoother, more gradual rise with no clean cliff
  edge.
- **Both slots peak in mid-to-late layers** (slot 0 around
  layers 50-55; slot 3 around layers 25-27) and **drop sharply at
  the final 1-2 layers**, consistent with the model's last-layer
  output being optimised for next-token logits rather than
  identity-encoding geometry.
- **K=6 starts to underperform K∈{2, 3, 4}** at slot 3 past layer
  ~30 (visible as the red curve sliding below the others) -- a
  gentle warning against aggressive whitening at deep layers.

#### `pair_slice_plots.py`

Per-axis-pair 2D "slice" plots that show every trait and role in the
corpus projected into the plane defined by the pair.  For each
``(pos, neg)`` pair:

- **origin** = trait mean (in the chosen whitening regime),
- **y-axis** = ``(pos − neg) / |pos − neg|`` -- pos at top,
- **x-axis** = orthogonal projection of ``(midpoint − trait_mean)``
  into the plane, then unitised.  This is the "common-mode"
  direction that both poles share relative to the trait mean.

The plot annotates trait/role names with a greedy collision-free
algorithm (priority = distance from origin + a hand-curated semantic
bonus dict, with the pair's own poles always shown as bold pole
labels).  Each panel also prints to stdout the d_pos/d_neg/d_mid
distances and the top six entities at each axis extreme.

Whitening uses the standard
``canonical_angles.data.build_augmented_whitening_pool`` (roles +
traits + ``default.pt``, **no** leave-out -- these are
visualisation plots, not held-out statistics) followed by
``fit_whitening("soft_K", pool, K=K)``, matching the rest of the
K-sweep tooling.

The **curated pair list** in ``DEFAULT_PAIRS`` is the set of 13
axis pairs (out of the 33 with judging data) for which::

    |midpoint - trait_mean|  ≥  0.5 × |pos - neg|

at slot 3, layer 25, K=3 soft whitening -- i.e. axes whose +/- pole
pair sits noticeably *off* the trait-mean origin, indicating a
substantial common-mode component shared by both poles relative to
the corpus.  Sorted by that ratio descending::

    systems_thinker / analytical    0.842
    relativist / absolutist         0.806
    ecocentric / anthropocentric    0.753   ⚠ y-flag
    individualistic / collectivistic 0.742  ⚠ y-flag
    reductionist / holistic         0.708
    progressive / conservative      0.655
    egalitarian / elitist           0.637
    convergent / divergent          0.634
    casual / formal                 0.576
    forgiving / unforgiving         0.571   ⚠ y-flag
    practical / theoretical         0.521
    improvisational / methodical    0.506
    concise / verbose               0.502

Three of the 13 carry an explicit ``y_flag`` describing where the
y-direction's actual meaning departs from the pair name -- typically
because the trait description leaned into a hyperbolic / over-loaded
framing of one pole.  The 7 originally-curated pairs (the L24/K4
"near-miss" audit set, minus ``helpful/unhelpful`` whose ratio
dropped to 0.447 at L25/K3) all retained their L24/K4 axis labels
verbatim under L25/K3 -- the +/-x and +/-y top-entity lists shifted
only at the noise-floor.

Outputs (to ``--out_dir``, default
``roger/axis_judge_experiments/pair_slices``):

- One PNG per pair, named ``{pos}_vs_{neg}_slice_K{K}.png`` (so the
  K=3 set lives alongside the historical K=4 set without overwriting).

```bash
# Default: all 8 curated pairs at slot 3, layer 25, K=3
uv run python results_analysis/pair_slice_plots.py

# One specific pair at non-default whitening
uv run python results_analysis/pair_slice_plots.py \
    --pairs progressive,conservative --K 4 \
    --out_dir /tmp/slices_K4
```

### Pole orientation convention

When the axis has a clear "good vs bad" polarity (e.g. `benign` vs `malicious`,
`honest` vs `deceptive`), put the socially-good pole as the **positive** (+3)
pole: `--pair benign malicious --pair_type traits`, not the reverse. For
morally-neutral axes (e.g. `performative` vs `functional`), either orientation
is fine.

The motivating intuition: RLHF'd judges appear to have a strong baseline
`+ = good` / `- = bad` prior, and aligning the rubric's +/- labels with the
judge's prior avoids cognitive dissonance.

#### Empirical validation (2026-04-24, malicious/benign trait pair)

We ran both orientations against both providers (Sonnet 4 and GPT-4.1-mini),
descriptions + instructions modes, 568 entities each. Headline numbers:

**Reproducibility under sign flip** (how often `score(forward) == -score(reverse)`)

| provider / mode | exact match | |δ|≤1 | |δ|≥2 |
|---|---:|---:|---:|
| Sonnet / descriptions | 79.8% | 96.7% | 3.3% |
| Sonnet / instructions | 77.1% | 98.2% | 1.8% |
| GPT / descriptions | 75.4% | 96.0% | 4.0% |
| GPT / instructions | 72.5% | 94.9% | 5.1% |

~75% exact agreement is noticeably below the "orientation-invariant" ideal
(~95%+); orientation is a real factor, not noise.

**Mean signed delta** = `mean(score_forward + score_reverse)` (>0 means
benign=+ inflates scores relative to malicious=+)

| | Sonnet | GPT |
|---|---:|---:|
| descriptions | **+0.092** | +0.011 |
| instructions | +0.021 | +0.000 |

Only Sonnet / descriptions shows clear mean-shift inflation; elsewhere the bias
is effectively zero. So the "benign=+ inflates everything uniformly" concern is
weak.

**Spearman ρ vs activation geometry — the clean signal**

The direction of orientation preference **depends on the metric**, which is
the most important finding:

*Raw metric* — benign=+ wins every comparison (16/16, p ≈ 2e-5 binomial):

| provider / mode / slot | forward (benign=+) | reverse (malicious=+) | Δ |
|---|---:|---:|---:|
| Sonnet / descriptions / slot 3 | +0.711 | +0.665 | −0.046 |
| Sonnet / instructions / slot 3 | +0.748 | +0.716 | −0.032 |
| GPT / descriptions / slot 3 | +0.704 | +0.665 | −0.039 |
| GPT / instructions / slot 3 | +0.726 | +0.685 | −0.041 |

Effect sizes 0.03–0.07 across all 4 slots, all 4 provider/mode combos.

*Whitened metric* — malicious=+ wins 14/16, effects 0.01–0.03:

| provider / mode / slot | forward (benign=+) | reverse (malicious=+) | Δ |
|---|---:|---:|---:|
| Sonnet / descriptions / slot 3 | +0.231 | +0.262 | **+0.031** |
| Sonnet / instructions / slot 3 | +0.238 | +0.271 | +0.033 |
| GPT / descriptions / slot 3 | +0.218 | +0.248 | +0.031 |
| GPT / instructions / slot 3 | +0.215 | +0.221 | +0.005 |

**This asymmetry is unexplained** and worth following up. Hypothesis: the raw
metric is dominated by high-variance PC directions that happen to carry the
judge's charity-bias inflation in the benign direction, so benign=+ orientation
"rides" that inflation constructively. Whitening removes the high-variance
dominance, making the ρ more sensitive to the judge's true discrimination
ability, which is slightly better when malicious=+ (less charity bias pulling
items toward zero from the malicious side).

**Provider-specific flip signatures**

GPT exhibits a subtle rubric-confusion pattern Sonnet does not: 14 entities
scored `(fwd=-1, rev=-1)` or `(fwd=+1, rev=+1)` — literal contradictions under
sign flip. E.g. `unhelpful` scored −1 in *both* orientations (meaning "mildly
malicious" in forward *and* "mildly benign" in reverse). Sonnet essentially
never does this; its disagreements are closer to `(fwd=+2, rev=0)` — loss of
decisiveness when malicious is forced into the + slot, not outright
contradictions.

**Practical recommendations**

- Default: `--pair <good_pole> <bad_pole>` for good-vs-bad axes. Gives the best
  raw-metric ρ and the most decisive score distribution.
- If the analysis relies on the **whitened metric**, consider running the
  reverse orientation too and reporting the better ρ (or averaging).
- If using **GPT-4.1-mini** (cheaper exploration), spot-check a few
  mildly-negative-trait entities for rubric-confusion patterns.
- For critical axes, running both orientations and averaging the scores
  gives ~0.03 ρ gain over either alone and cancels most orientation bias,
  at 2× the judge cost.

### Which scoring source and which metric — empirical findings

We ran the tool over 20 reciprocal-antonym trait axes at GPT-4.1-mini, then
selectively re-ran a 12-axis subset with Sonnet 4 and added response-mode
scoring (the cohort with response judging — see ``pair_list_responses.json``).
The 20-axis and 12-axis data is persisted under
[`roger/axis_judge_experiments/`](../roger/axis_judge_experiments/) with
[`summary.json`](../roger/axis_judge_experiments/summary.json) (20 axes, GPT).

#### Raw vs whitened

Across all 20 axes (GPT, slot=3, scores averaged across descriptions+instructions
per entity before correlating — "both_mean"):

| metric | mean ρ | min | max | ρ distribution |
|---|---:|---:|---:|---|
| raw | **+0.541** | +0.04 | +0.80 | wide; axis-dependent |
| whitened | +0.201 | +0.11 | +0.28 | tight cluster |

**Raw wins 19/20 axes.** Only `ecocentric/anthropocentric` flips — raw ρ is
near zero (+0.04), whitened salvages a weak signal (+0.18). The tightness of
the whitened band (σ ≈ 0.05) vs the wide raw spread (σ ≈ 0.21) is itself a
finding: the raw metric registers how well each axis aligns with the model's
dominant-variance subspace, while the whitened metric is SNR-floored by
low-variance noise and produces similar ρ for most axes regardless of their
actual representational quality.

For interpreting a single axis in isolation, **prefer raw ρ**. For cross-axis
comparison in a fairness-across-directions sense, whitened is the more
direction-neutral metric but paints a much flatter picture.

#### Scoring source: descriptions vs instructions vs responses

On the same 20 axes (GPT only):

| source | mean raw ρ | mean whitened ρ | notes |
|---|---:|---:|---|
| descriptions only | +0.512 | +0.186 | cheapest; 1 judge call per entity |
| instructions only | +0.531 | +0.201 | slight edge over descriptions |
| both_mean (avg of above) | **+0.541** | +0.201 | +0.008 raw gain over best-single; 2× cost |

On the 7-axis deeper experiment, **response-mode is the clear winner on raw**:

| source | mean raw ρ (7 axes) | mean whitened ρ |
|---|---:|---:|
| GPT desc+inst | +0.451 | +0.192 |
| Sonnet desc+inst | +0.439 | +0.184 |
| **GPT responses** | **+0.605** | +0.169 |

Mean improvement of GPT responses over best-desc+inst = **+0.135 raw ρ**, with
per-axis gains ranging from −0.05 to +0.56. The dramatic case is
`ecocentric/anthropocentric`: desc+inst essentially fails (raw ρ ≈ 0), but
responses captures it at ρ = +0.60. Behavioral evidence reveals the axis where
trait-description text is too generic to discriminate.

Response-mode does **not** help on whitened (−0.028 mean); same SNR-floor
pattern. And response-mode is ~20× more expensive than desc+inst.

#### GPT vs Sonnet agreement

On desc+inst for the same 7 axes, GPT and Sonnet agree strongly:

- Pooled across all 3,976 (entity, axis) pairs: **Spearman ρ = 0.816**.
- Per-axis agreement: 0.68–0.92, highest on strong axes (`helpful/unhelpful`
  0.92), lowest on subtle ones (`systems_thinker/analytical` 0.68).
- Per-axis ρ-vs-projection is essentially equivalent between the two providers
  (mean difference +0.011 raw in favor of Sonnet, +0.008 whitened).

For exploration and iteration, **GPT-4.1-mini is ~5× cheaper with statistically
equivalent results**. Sonnet is only worth the premium for final reporting or
for axes where the 0.01-level difference matters.

#### Recommendations

| goal | recommended source | metric | why |
|---|---|---|---|
| Sanity-check an axis cheaply | GPT desc+inst | raw | ~$1/axis, good enough signal |
| Publish ρ for a given axis | GPT responses | raw | best signal, ~$40/axis |
| Compare across axes fairly | whitened (any source) | whitened | equalizes variance across directions |
| Diagnose "does this axis exist at all?" | GPT responses | both raw and whitened | if whitened ρ is nonzero and raw ρ is near-zero, the axis lives in low-variance directions |

### Adaptive per-PC whitening — empirical study

We did an extended study of whether per-PC whitening (each PC gets its own
scale factor β_k ∈ [0, 1]) can beat fixed soft-K whitening. Four operational
conclusions, all from experiments on the responses-cohort axes in
[`pair_list_responses.json`](../roger/axis_judge_experiments/pair_list_responses.json)
(originally a 7-axis subset; expanded to 12 once response judging was
extended):

**1. ‖m‖/‖diff‖ predicts axis hardness (label-free).**

For an axis defined by `diff = vec[pos] − vec[neg]`, the pair's common-mode
offset is `m = (vec[pos] + vec[neg]) − 2·mean_corpus`. The ratio ‖m‖/‖diff‖
measures how far the pair's midpoint sits from the corpus center relative to
the axis magnitude.

| ‖m‖/‖diff‖ range | example axes | raw ρ range |
|---|---|---|
| < 1.0 | helpful/unhelpful | 0.78-0.82 (easy) |
| 1.0–1.5 | guileless/scheming, improvisational/methodical | 0.38-0.69 |
| > 1.5 | progressive/conservative, ecocentric/anthropocentric, systems_thinker/analytical | 0.00-0.58 (hard) |

Pearson correlation between ‖m‖/‖diff‖ and raw ρ is **−0.67** (D+I) and **−0.74**
(responses) across the 7 axes — a single-number label-free predictor of
axis quality. Large common-mode offset means the pair activations share a big
"common mode" signature that isn't the axis itself — and projecting entities
onto the axis direction picks up that shared structure as noise.

**2. soft-K=4 is the best fixed label-free recipe.**

From the whitening-K sweep across 20 axes, mean responses ρ peaks at K=4 (0.652)
vs K=0 (0.605), K=1 (0.639), K=2 (0.646), K=8 (0.587), K=16 (0.471), K=128
(0.169). See [`rho_vs_whitening_K.png`](../roger/axis_judge_experiments/rho_vs_whitening_K.png).
K=4 is the default recommendation when no judge labels are available.

**3. Label-using coord ascent finds well-generalizing β* (+0.13-0.17 held-out gain).**

[`coordinate_ascent.py`](../roger/axis_judge_experiments/coordinate_ascent.py)
implements exact per-PC Spearman-optimal β via pairwise rank-crossing
enumeration. 5-fold CV on 7 axes × 3 score sources:

| source | mean test ρ (β*) | mean test ρ (K=4) | held-out gain |
|---|---:|---:|---:|
| desc+inst | 0.671 | 0.505 | **+0.166** |
| responses | 0.766 | 0.638 | **+0.127** |
| joint (D+I+RESP) | 0.747 | 0.583 | **+0.164** |

The coarse-discrete variant (β ∈ {0, σ_{M}²/σ_k² for M ∈ {2,3,5,9,17}, 1}) gives
virtually the same held-out ρ with ~half the training-ρ overfitting — suggesting
the finer continuous β values were fitting noise. Use the coarse version for
interpretability. Per-axis data: [`coarse_discrete_coordinate_ascent.json`](../roger/axis_judge_experiments/coarse_discrete_coordinate_ascent.json).

**4. Label-free geometric rules cap at ~10-15% of oracle recovery.**

Tried (a) hand-crafted rules on c_k², m_cos_k²; (b) random forests over (log K,
c_k², m_cos_k²). In-sample: RF recovers 78% of oracle; leave-one-axis-out it
drops to ~5%. The v1 rule (Keep if c_k² ≥ 0.20, Ablate if m_cos_k²/(c_k²+m_cos_k²) ≥ 0.90,
else shrink to σ_5²) recovers ~13% in-sample. With n=7 axes, simple geometric
features can't reliably predict axis-specific β* patterns.

Most of the oracle gain is genuinely axis-specific information (likely
semantic-content of each PC in the context of the axis's meaning). The ~13%
that is captured by simple rules is the "easy" structural part (ablate PCs where
low c² coincides with high m_cos²).

### Recommendations (label-free workflow)

1. Compute ‖m‖/‖diff‖ per axis before any judging — axes > 1.5 are hard.
2. Use soft-K=4 whitening as the default metric.
3. If budget permits judge-scoring, coord ascent gives a robust ~+0.13 CV gain.
4. If exploring many axes cheaply, report raw ρ at slot 3 with GPT-4.1-mini
   descriptions+instructions; spot-check with response mode.

### Judge defaults

Judge model defaults: `claude-sonnet-4-20250514` with `--provider anthropic`
(the default), `gpt-4.1-mini` with `--provider openai`. Results cache under
`<output_dir>/scores_*.json` and resume on rerun; pass `--no_cache` to
rescore from scratch.

### Judge model pricing (updated 2026-09)

Single source of truth: `assistant_axis/judge_pricing.py:_MODEL_RATES`
(`price_for_model()`). Table below is a snapshot — read the module
docstring for sourcing/confidence notes before trusting it for a real
budget-tracked run.

| Tier | Model | $/1M in | $/1M out | Confidence |
|---|---|---:|---:|---|
| OpenAI (legacy) | `gpt-4.1-mini` | $0.40 | $1.60 | High — original project measurement |
| Anthropic (legacy) | `claude-sonnet-4` / `-4-6` | $3.00 | $15.00 | High — original project measurement |
| Anthropic small | `claude-haiku-4-5` | $1.00 | $5.00 | High — unchanged since May 2026 |
| Anthropic mid | `claude-sonnet-5` | $2.00 | $10.00 | High — verified live (`claude-api` skill), price *cut* vs sonnet-4 |
| Anthropic flagship | `claude-opus-5` | $5.00 | $25.00 | High — verified live; supersedes an unverified $15/$75 May-2026 placeholder that was never used |
| OpenAI small | `gpt-5.6-luna` | $0.20 | $1.20 | Medium — cross-confirmed web search hitting `developers.openai.com`, not a direct fetch |
| OpenAI mid | `gpt-5.6-terra` | $2.00 | $12.00 | Medium — same |
| OpenAI flagship | `gpt-5.6-sol` | $5.00 | $30.00 | Medium — same; **promotional pricing through 2026-11-21** |

`results_analysis/plot_batch_size_quality_vs_cost.py` has its own separate,
deliberately-un-synced copy of the legacy rates (it regenerates a chart
from cached usage data that was actually billed at those historical
prices) — don't update it when refreshing current prices; see its own
comment.

### Evaluating a new judge tier without recomputing activations

`judge_tier_cost_eval.py` scores a *new* judge model against an
**already-computed** axis, reusing a cached `projections.json` instead of
recomputing projections from raw activation vectors (which
`axis_judge_correlation.py` itself always does — `--data_dir` is
required there). Only two axes in this repo have that full reusable
cache committed:
`roger/axis_judge_experiments/{angel_vs_demon,decisive_vs_indecisive}/{gpt,sonnet}/projections.json`
(571 entities × 8 slots × {raw, whitened} each) — every other axis in the
35-axis GPT-vs-Sonnet comparison only has its aggregate summary committed
(`gpt_vs_sonnet_rhos_di.json`), not the per-axis data this script needs.

```bash
# Preview cost, zero API calls:
uv run python results_analysis/judge_tier_cost_eval.py \
    --pair angel demon --pair_type roles \
    --projections_file roger/axis_judge_experiments/angel_vs_demon/gpt/projections.json \
    --judge_model claude-opus-5 --judge_model claude-sonnet-5 --judge_model claude-haiku-4-5 \
    --judge_model gpt-5.6-sol --judge_model gpt-5.6-terra --judge_model gpt-5.6-luna \
    --output_dir roger/axis_judge_experiments/angel_vs_demon_tier_eval \
    --dry_run

# Second reusable axis (traits, not roles):
uv run python results_analysis/judge_tier_cost_eval.py \
    --pair decisive indecisive --pair_type traits \
    --projections_file roger/axis_judge_experiments/decisive_vs_indecisive/gpt/projections.json \
    --judge_model claude-opus-5 --judge_model claude-sonnet-5 --judge_model claude-haiku-4-5 \
    --judge_model gpt-5.6-sol --judge_model gpt-5.6-terra --judge_model gpt-5.6-luna \
    --output_dir roger/axis_judge_experiments/decisive_vs_indecisive_tier_eval \
    --dry_run

# Drop --dry_run to actually run either (description-mode only, cheap:
# all 6 tiers x both axes ≈ $15.76 total at current rates -- see the
# pricing table above for the per-tier/per-axis breakdown).
```

Writes `scores_descriptions.json`, `usage.json` (real measured cost, not
estimated), and `correlations.json` per judge model, in the same shape
`axis_judge_correlation.py` itself writes.

**Scope of what actually gets touched, running either command above:**

- **What's scored**: not the pole pair itself (`angel`/`demon`, excluded by
  design) — the other 571 entities in the corpus (every other committed
  role/trait), each scored for where it falls on that axis's spectrum.
- **Network**: only `api.anthropic.com` / `api.openai.com` (the judge
  calls themselves). No GPU, no RunPod, no other service contacted.
- **Filesystem**: only writes new files under the new `--output_dir`
  (`roger/axis_judge_experiments/{pair}_tier_eval/<model>/`). Never
  touches the existing committed `gpt/`/`sonnet/` subdirectories or their
  cached scores/projections/correlations.
- **Git**: nothing is committed or pushed automatically.

### Plotting the cost/quality comparison

`judge_tier_cost_plot.py` reads `judge_tier_cost_eval.py`'s output (the 6
new tiers) plus the existing `{pair}/{gpt,sonnet}/correlations.json` (the
2 legacy tiers — `descriptions`-mode rho there is directly comparable,
same 571 entities, same measurement) and plots them together in the same
visual language as `batch_size_cost_vs_quality.png`: quality = 1/(1-rho)
on the y-axis, $/axis on the x-axis, one annotated point per tier.

The 2 legacy tiers have no `usage.json` (that feature postdates their
archived runs), so their cost is *estimated* from the same token/pricing
model rather than measured — the chart marks this with marker style
(filled = measured, hollow = estimated) rather than presenting all 8
points as equally certain.

```bash
# Offline, no API calls -- only reads existing JSON. Run after
# judge_tier_cost_eval.py has actually produced output.
uv run python results_analysis/judge_tier_cost_plot.py \
    --pair angel demon \
    --tier_eval_dir roger/axis_judge_experiments/angel_vs_demon_tier_eval \
    --legacy_dir roger/axis_judge_experiments/angel_vs_demon \
    --slot 6 \
    --output roger/axis_judge_experiments/angel_vs_demon_tier_eval/cost_vs_quality.png
```

## Output layout

```
<output_dir>/
├── config.json                  # all resolved args + axis metadata
├── scores_descriptions.json     # {name: int score in [-3, 3]}
├── scores_instructions.json     # {name: int score in [-3, 3]}
├── scores_responses.json        # {name: {mean_score, n_scored, n_total}}
├── projections.json             # {slot: {name: {raw, whitened}}}
├── correlations.json            # {mode: {slot: {raw|whitened: {rho, p, n}}}}
└── correlation_plot.png         # scatter grid (rows=modes, cols=slot×metric)
```
