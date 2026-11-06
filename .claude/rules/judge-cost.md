---
paths:
- pipeline/3_judge.py
- assistant_axis/judge*.py
- assistant_axis/steering_runner.py
- tools/dry_run_response_token_count.py
- results_analysis/plot_batch_size_quality_vs_cost.py
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: judge-cost

**When:** estimating or budgeting judge API cost, or changing the batch size B.  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=judge-cost`; edit there, then run `uv run python tools/sync_agent_notes.py`.

## Judging cost model (project-wide reference)

The single source of truth lives in
[`results_analysis/plot_batch_size_quality_vs_cost.py`](results_analysis/plot_batch_size_quality_vs_cost.py)
(constants and the Pareto plot) with the fuller narrative in
[`results_analysis/README.md`](results_analysis/README.md) under
"Cost model: GPT-4.1-mini responses-mode judging".  This section is a
quick reference for cost-scoping questions; numbers were re-derived
empirically on **2026-05-09** via a 240-batch tiktoken dry-run on
real cached prompts + responses (the previous documented numbers
under-counted real responses by ~22× and made all costs look ~5× too
low).

**2026-05-10 full cross-axis validation:** re-ran the dry-run via
[`tools/dry_run_response_token_count.py`](tools/dry_run_response_token_count.py)
on **all 12 v2 axes × both cohorts** = 295,258 reconstructed B=10
batches, prompt-rebuilt under the current (v2) rubric, all
tiktoken-counted with `o200k_base`.  Per-axis cost (`measured /
$49.40`) came in at **mean 1.006 ± σ 0.013, range [0.989, 1.034]**.
The model is empirically correct within ±3% per axis and within 0.6%
on average -- no recalibration applied.  Per-axis spread:

| axis                                  | n_batches | cost    | ratio |
| ------------------------------------- | --------: | ------: | ----: |
| helpful_vs_unhelpful                  | 24,599    | $50.33  | 1.019 |
| harmless_vs_harmful                   | 24,630    | $50.43  | 1.021 |
| honest_vs_dishonest                   | 24,621    | $49.32  | 0.998 |
| truthful_vs_deceitful                 | 24,626    | $49.47  | 1.001 |
| guileless_vs_scheming                 | 24,600    | $49.47  | 1.001 |
| egalitarian_vs_elitist                | 24,602    | $49.63  | 1.005 |
| progressive_vs_conservative           | 24,590    | $49.29  | 0.998 |
| concise_vs_verbose                    | 24,589    | $48.86  | 0.989 |
| ecocentric_vs_anthropocentric         | 24,613    | $51.08  | 1.034 |
| improvisational_vs_methodical         | 24,598    | $49.43  | 1.001 |
| relativist_vs_absolutist              | 24,601    | $49.41  | 1.000 |
| systems_thinker_vs_analytical         | 24,589    | $49.71  | 1.006 |

Reproduce via `uv run python tools/dry_run_response_token_count.py
--axis <name> --kind both --quiet`; with `--quiet` omitted, dumps
per-cohort histograms for diagnostic use.  The dry run rebuilds each
batch's prompt with the *current* RUBRIC_RESPONSE_BATCH and tiktoken-
counts it -- so re-running after any rubric change re-validates the
model automatically.

### Per-batch token model (B=10)

* Input: ≈ 320 (header) + 10 × 438 (per item) ≈ **4,700 tokens**.
* Output: ≈ **80 tokens** (median 78, σ 14).
* Header is only ~7% of input; **items dominate** because real
  Qwen-3-32B responses are ~435 tokens each (capped at the 512-token
  generation limit; ~71% hit cap).

### Per-axis cost (both cohorts combined, full volume)

At B=10: **115.6M input + 1.97M output** tokens across ~24,605
batches.

| B  | GPT-4.1-mini | Haiku-4.5 (full / q9)       | Sonnet-4 (full / q9)        |
| -- | -----------: | --------------------------: | --------------------------: |
| 5  |       $55.69 |             $164.55 / $54.85 |       $493.64 / $164.55 |
| 7  |       $52.10 |             $154.74 / $51.58 |       $464.21 / $154.74 |
| 10 |       $49.40 |             $147.38 / $49.13 |       $442.14 / $147.38 |
| 15 |       $47.30 |             $141.66 / $47.22 |       $424.97 / $141.66 |

* `q9` = `--question_subsample_modulo 9` → ~1/3 fraction of full
  volume.  ~4% smaller than tiered-default `_b10` cohorts in practice
  because uniform mod-9 doesn't escalate RP-depleted entities to tier
  2/3.
* Pricing (per 1M tokens, 2026 rates): GPT-4.1-mini $0.40/$1.60;
  Haiku-4.5 $1.00/$5.00; Sonnet-4 $3.00/$15.00.
* **Haiku/Sonnet token scales** (empirical from 2026-05-11 5c.1/5d.1
  full-sweep totals: 11 v2 axes × 2 cohorts, 372,652 GPT calls +
  8,927 Haiku calls; previously the 5d.0 canary's smaller sample
  gave 1.17/2.00, which the full sweep now refines).  These are
  PER-CALL ratios and are approximately B-invariant: at B=7 they
  measure (1.087, 2.165) and the header-amortisation analysis
  (1.35× header_chunk × 320 tok header + 1.07× item_chunk × B × 438
  tok items) predicts (1.089, 2.165) at B=10, so the same
  constants apply to the B=10 Pareto plot anchor below.

  - `HAIKU_INPUT_SCALE = 1.09`: Anthropic tokenizer ~9% chunkier than
    `o200k_base` on response prompts dominated by raw response text;
    much smaller than the 1.35× seen on static-mode prompts where
    the rubric-laden header dominates.
  - `HAIKU_OUTPUT_SCALE = 2.17`: Haiku produces ~2.17× the output
    tokens of GPT-4.1-mini per call (160.2 vs 74.0 per call in the
    5c.1/5d.1 full sweep at B=7; 2.19 in the 5d.0 canary; 1.93× in
    static-mode).  The previous 2.00 mid-estimate was conservative
    on the output side; the net effect of moving (1.17, 2.00) →
    (1.09, 2.17) is a **~5% decrease** in predicted Haiku cost at
    B=10 (input contribution drops faster than the output
    contribution rises, since input is ~5.7× the output for Haiku
    given the $1.00/$5.00 rate split and 115.6M/1.97M token totals).

  Sonnet uses the same Anthropic tokenizer family (input scale
  carries over directly); Sonnet's verbosity is **not** measured in
  this project — we use the Haiku output scale as a placeholder
  (Sonnet is typically ≥ as verbose as Haiku, so this is more likely
  an under-estimate than over-estimate).  See
  `results_analysis/plot_batch_size_quality_vs_cost.py`
  `HAIKU_INPUT_SCALE` / `HAIKU_OUTPUT_SCALE` constants for the
  authoritative values used in the Pareto plot.

  Note: the 5d.1 actual-over-expected ratio of 1.21 (Haiku b7_t3
  residual sweep) is **not** explained by these scale revisions
  (which net to ~5% LOWER cost, not 21% higher); that ratio traces
  to the 5d.1 launch-script's budget formula and is a separate
  reconciliation item.

### B-curve takeaway (motivates the May 2026 B=10 → B=7 default switch)

The B=15 → B=5 cost spread is only **1.18×** ($47.30 → $55.69), not
~2× as the old (broken) model claimed.  Most cost is `items × per-item-rate`
($43.10 floor) regardless of B.  B=7 vs B=10 marginal ≈ $2.70/axis
for +0.004 ρ uplift (~$7 per +0.01 ρ — the best step on the new
curve).  See
[`./roger/axis_judge_experiments/batch_size_curve_8slot/batch_size_cost_vs_quality.png`](./roger/axis_judge_experiments/batch_size_curve_8slot/batch_size_cost_vs_quality.png).

### Steering effect-judge B post-mortem (2026-05-21)

**Stale-local-default bug.**  The library-level canonical constant
`assistant_axis.judge_batch.RESPONSE_BATCH_SIZE = 7` and the steering
re-export `assistant_axis.steering_judges.DEFAULT_TARGET_BATCH_SIZE`
(pinned to that constant) were correctly updated to **B=7** on
2026-05-11.  However, `steering/run_sweep.py` had its own *local*
argparse default `--target-batch-size = 10` that silently overrode
the canonical value when launching multi-cell sweeps.  As a result,
the `multi_cell_batch_v1` (all-mode, 2026-05-18) and
`multi_cell_batch_v1_prefill` (2026-05-19/20) sweeps both ran their
live effect/coh/rp judging at **B=10**, not the canonical B=7.

The error was caught on 2026-05-21 when prepping a `max_strength=128`
fill-in sweep.  Records from the two prior multi_cell sweeps remain
on disk at B=10 and are left as-is; the fill-in past the old cap was
run at the canonical B=7 (Plan A: only new strengths beyond the old
`max_strength=16` cap were generated, so most cohorts have a
B=10/B=7 split at the s ≈ 16 transition; cells that stopped at the
coherence cliff before reaching the old cap remain pure B=10).

**Fixes applied 2026-05-21:**

- `steering/run_sweep.py`: argparse default for `--target-batch-size`
  changed `10 → 7` with a HARD-RULE-aligned warning in the help text
  ("the previous value of 10 is OBSOLETE and was the source of a
  several-hundred-dollar mis-judging incident; do not revert without
  explicit user approval").
- `assistant_axis/steering_runner.py`: early-skip path in
  `run_steering_cell` now distinguishes `up_blocked_reason ==
  "max_strength_reached"` (fall through to extend the UP-walk when
  the caller has raised `max_strength`) from
  `up_blocked_reason == "incoherent"` (skip as before).  Existing
  records are preserved end-to-end via the `done_pairs` gate at
  `steering_runner.py:981, 1189`; only new `(strength, q_idx)`
  pairs beyond the old cap are generated and judged.  Symmetric
  handling on the DOWN side if `min_strength` is ever lowered.
- `assistant_axis/tests/test_steering_runner.py`: two new tests
  (`test_skip_when_previous_cap_below_new_max_strength_with_incoherent_stop`
  and `test_cap_extension_fallthrough_when_previously_capped`) pin
  both branches of the early-skip decision so the cap-extension
  semantics can't regress.

**Pattern to watch for.**  When a project-wide canonical constant
is changed (here `RESPONSE_BATCH_SIZE`), grep all argparse
`default=…` values that name the same parameter and verify they
pin to the constant rather than hard-code a literal.  Local
argparse defaults that hard-code a number are silent stale-value
traps: they don't error, they just quietly do the wrong thing in
production.

### Static-mode cost model (descriptions + instructions)

Static-mode prompts are vastly cheaper per call than response-mode
because there are no per-batch response items — just the rubric
header plus one entity description (or one instruction list).
Empirical from the 2026-05-10 GPT canary on `concise_vs_verbose`
(36 single-shot calls, descriptions + instructions for 18 collision
entities = 12,664 input + 2,250 output tokens):

| Component | Tokens | Notes |
| --------- | -----: | ----- |
| Input per call  | **352** | rubric header (~210) + entity desc/instr (~110) + scoring instruction (~30); range narrow because content length is bounded by description format |
| Output per call | **63**  | reasoning (2-3 sentences) + `SCORE: <int>` line; mean 62.5, σ small |

Per-call cost (single-shot, no batching):

| Judge | $ / call | Source |
| ----- | -------: | ------ |
| GPT-4.1-mini  | **$0.000242** | `352 × $0.40/1M + 63 × $1.60/1M` |
| Haiku-4.5     | **$0.000667** | `352 × $1.00/1M + 63 × $5.00/1M` |
| Sonnet-4      | **$0.002001** | `352 × $3.00/1M + 63 × $15.00/1M` |

**Per-cell scaling for surgical static-mode rejudges** (one cell =
one (axis, judge) pair, two modes = descriptions + instructions, N
entities):

```
cost ≈ 2 × N × cost_per_call
```

For Phase 5b (N=18 collision entities, 12 axes, GPT + Haiku): per
cell = 36 calls = $0.0087 (GPT) or $0.0240 (Haiku).  Total Phase 5b
expected: 12 × ($0.0087 + $0.0240) ≈ **$0.39**.  Hand-rolled
estimates that pre-dated this empirical canary (e.g. an early
$0.012/cell guess) are roughly 1.5× pessimistic — re-derive from the
table above when budgeting, not from older estimates.

When estimating costs for a new judging run, multiply `cost_per_axis_for_(judge, B, subsample)`
by the number of axes and rubrics involved.  When estimating for a
*surgical* rejudge of N entities, scale by `N / 280` (roles) or
`N / 300` (traits) of the per-axis cost (Phase 5d-i and 5d-ii in
the disambiguation plan are worked examples).

**Roles-vs-traits response-length asymmetry (response-mode only).**
At the same B and the same N entities, response-mode judging of
ROLES costs ~**1.22×** more per entity than TRAITS, because
Qwen-3-32B produces ~22% longer responses when role-played
(persona-immersive narratives) than trait-modulated (clipped
behavioral answers): mean 2094 vs 1718 chars / median 2351 vs 1842
chars on the 29 roles + 14 traits 5d.1 cohort (14,500 + 7,000
responses).  Per-call judge output is unaffected (the judge emits a
fixed-format score per response regardless of cohort) — only
per-call input scales with response length.

This is **why 5d.1 came in 1.21× over its hand-tuned 50/50 budget
split**: the 29/14 entity split implies a *cost* split of
`(29 × 1.22) : 14 ≈ 2.53 : 1`, not the `$1.50 : $1.55 ≈ 0.97 : 1`
the launch script assumed.  Re-applying the 1.22× factor to the
5d.1 numbers gives `$2.50/roles + $1.00/traits = $3.50/axis ⇒
$38.50 total`, vs the observed $40.45 (within ~5%, well inside
noise).

The canonical constant is
`assistant_axis.judge_pricing.ROLES_RESPONSE_LENGTH_FACTOR = 1.22`,
with a convenience splitter
`surgical_rejudge_cost_split(per_axis_cost, n_roles, n_traits,
mode="response")` returning the per-cohort budget shares.  Static
mode (descriptions / instructions) is **not** affected — there are
no responses in those prompts, so call `surgical_rejudge_cost_split(
..., mode="static")` to get an N-weighted (1.0×) split for static
rejudge runs.

The 1.22× asymmetry is also part of why the project-aggregate
per-axis cost (`B10_GPT_INPUT_M_TOK = 115.6` for both cohorts
combined) silently absorbs it — full-volume budgets are correct
without an explicit factor because the (300 traits + 280 roles)
~50/50 entity split implies a (1.0:1.22)/(1.0+1.22) = 55/45
implicit weighting that's already baked into the empirical token
totals.  Only *off-default* cohort splits (surgical rejudges,
single-mode runs) need to apply the factor explicitly.

### Budget cap recommendation (safety net)

The `--budget_usd` hard cap is a safety net against catastrophic
miscalculation (a bug, a wrong model, an unexpected prompt blowup
— not normal variance).  Pick it generously relative to the
expected cost, but tightly enough that a real surprise gets caught
before the bill detonates.

Post-May-2026 calibration (5c.1 GPT full sweep + 5d.1 Haiku
residual sweep, with the new `HAIKU_INPUT_SCALE = 1.09` /
`HAIKU_OUTPUT_SCALE = 2.17` per-call scales applied AND the
`ROLES_RESPONSE_LENGTH_FACTOR = 1.22` split applied where
appropriate) gives empirical actual/expected ratios:

- 5c.1 GPT (11 axes × 2 cohorts, B=7 full):  ratio = 0.97 (±4%)
- 5d.1 Haiku (corrected for roles/traits):   ratio = 1.05 (±5%)

So residual prediction error is ~3-5% in known-judge known-prompt
production runs.

**Recommended default**: `budget_usd = 1.25 × expected_cost_usd + $5`
(was: 1.5× + $20).  The 1.25× leaves ~20pt headroom over the
empirical residual — enough to catch model-drift or stupid
mistakes without being so loose that a real cost runaway escapes
the cap.

**For first-of-kind work** (new judge, new prompt template, new B
value not yet canary'd): stay at ~`1.5× + $10` until the first run
lands and the residual is measured.  Drop to the standard
recommendation thereafter.
