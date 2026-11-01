# Judge-tier evaluation: what changed on 2026-11-01, and why

For Konstantinos and the Claude sessions working with him.  Roger and Claude
made these changes on 2026-11-01, starting from `personas-konstantinos` at
c7743467, and merged your later commits (9b3d6a5c, ccf7bbb3) back in.  This
file explains what the pull request contains, the decisions behind it, and how
to carry on from here.  The per-plot notes hold the full numbers:
[`pareto_pooled_61axes_notes.md`](../../roger/axis_judge_experiments/pareto_pooled_61axes_notes.md)
and
[`pareto_pooled_20axes_notes.md`](../../roger/axis_judge_experiments/pareto_pooled_20axes_notes.md).

## 1. Results in brief

Every judge below rated the same entity descriptions (571 to 580 per axis,
corpus pinned at `1448a84c`) on the v3 descriptions-mode rubric.  Its scores
were then correlated with the cached activation projections.  ρ is the mean
Spearman correlation over the 61 axes at slot 6, layer 25, raw projection;
cost is per axis at real-time list price (Flex and Batch are half of that for
every judge here).

| judge | settings | ρ | cost per axis |
|---|---|---|---|
| claude-opus-5 | effort low (thinks); your 2026-10-26 run | 0.622 | $3.217 |
| gpt-4.1-mini | T=0 | 0.605 | $0.130 |
| claude-haiku-4-5 | T=0 | 0.561 | $0.521 |
| claude-sonnet-5 | effort low (no thinking seen) | 0.551 | $1.187 |
| gpt-5.6-terra | reasoning off | 0.550 | $0.719 |
| gpt-5.6-luna | reasoning off | 0.547 | $0.073 |
| claude-haiku-5-5 | thinking off | 0.544 | $0.070 |
| gpt-5.4-mini | reasoning off | 0.514 | $0.291 |
| gpt-6-luna | reasoning off | 0.478 | $0.033 |

- **Pareto frontier:** gpt-6-luna, claude-haiku-5-5, gpt-5.6-luna,
  gpt-4.1-mini, claude-opus-5.  Opus 5 with thinking is the best judge, 0.017
  ahead of gpt-4.1-mini at 25 times the cost.  Below Opus, gpt-4.1-mini is the
  best judge at any price.
- **Robust across 12 projection variants** (slot 0 or 6, layer 25 or 49; raw,
  soft whitening K=2, soft shear L=3).  Opus is top in 11 of them and
  gpt-4.1-mini in the 12th.  gpt-4.1-mini beats every other judge in all 12.
  gpt-6-luna, claude-haiku-5-5 and gpt-4.1-mini are on the frontier in every
  variant.  Kendall τ against the default ranking runs from 0.778 to 1.000.
- **The 20-axis set adds Opus with thinking off and gpt-5.6-sol.**  Opus at
  low effort 0.704, thinking off 0.693, gpt-4.1-mini 0.688, gpt-5.6-sol 0.644
  ([`pareto_pooled_20axes.png`](../../roger/axis_judge_experiments/pareto_pooled_20axes.png)).
- **Plots:** [`pareto_pooled_61axes.png`](../../roger/axis_judge_experiments/pareto_pooled_61axes.png),
  [`judge_tier_robustness_61axes.png`](../../roger/axis_judge_experiments/judge_tier_robustness_61axes.png),
  [`judge_tier_robustness_20axes.png`](../../roger/axis_judge_experiments/judge_tier_robustness_20axes.png).

**The practical problem:** OpenAI removes gpt-4.1-mini from the API on
2027-01-14 and names gpt-6-luna as its replacement, which scores 0.478 here
against 0.605.  So the cheap end of the frontier changes in January.

## 2. What changed, and why

Each item gives the change, then the reason.  Roger's decisions are quoted
where he gave one.

### 2.1 One set of settings for every judge

Roger asked that every plotted cell be made the same way: temperature 0 where
the API accepts it, reasoning off where it can be switched off, and one rubric
version.  Findings that shaped this:

- **Who takes T=0.**  Haiku 4.5 and gpt-4.1-mini do.  Sonnet 5, Opus 5 and
  Haiku 5.5 refuse it, and the GPT-5.6 and later models take no temperature.
  Those ran at their default with one sample per item.
- **GPT-5.6 reasoned at its default effort** (about 20 to 40 reasoning tokens
  per call in a probe), so Luna, Terra and Sol were rerun with
  `reasoning_effort` none.  Sonnet 5 at effort low never produced a thinking
  block in 200 probe calls, so its cells stand.  Opus 5 at effort low always
  thinks, so it was rerun with thinking off on the 20-axis set (Roger: "do
  this only to n=20ax, and on that graph plot both").
- **Cell labels keep reruns apart.**  `--cell_label` writes to
  `<axis>_tier_eval/<model><label>/`: `@t0`, `@none` (reasoning off),
  `@nothink`.  Earlier cells are never overwritten and are never resumed into.
- **Reasoning and temperature turned out to matter little, except for Opus.**
  Thinking helps Opus by about 0.011 in ρ.  Reasoning moves GPT-5.6 by 0.012
  or less, and T=0 moves Haiku 4.5 by 0.004.  The numbers are in the 61-axis
  notes and
  [`judge_tier_robustness_61axes_all.json`](../../roger/axis_judge_experiments/judge_tier_robustness_61axes_all.json).

### 2.2 The reply cap, and cut-off replies

`max_tokens` caps the judge's reply, not its prompt.  At the old cap of 200,
about 0.6% of Opus-with-thinking-off replies and 0.5% of Haiku 5.5 replies
were cut off.  `parse_signed_score` then took the first small integer in the
reasoning as the score.

- The default is now 300 (Roger: "200 is clearly too low for safety").
- A reply cut at the cap is scored as missing, never parsed, counted in the
  run summary, and listed under `truncated` in the cell's provenance.
- `--repair_truncated` re-asks exactly the cut replies and patches the cell
  (88 replies across 34 cells were repaired, for $0.43).
- Your 2026-10-26 cells ran before this check, so a few of their replies may
  carry a stray-number score.  Opus scored about 1.6 fewer entities per axis
  than the other judges (0.3%).

### 2.3 Batch and Flex transports

Roger's cost rule: under about $20, run in real time; over that, consider
batch.  For this work he asked for "flex default where available, else
batch".

- `--transport auto` picks Flex for OpenAI models that take it, the Batch API
  for those that don't (gpt-4.1-mini), and Message Batches for Anthropic.
  Untested OpenAI models get a probe request first.  The capability table is
  `MODEL_CAPABILITIES` in
  [`judge_tier_transport.py`](../../results_analysis/judge_tier_transport.py).
- A Flex request that keeps failing goes once to the standard tier after 10
  minutes.  Usage is recorded under `<model>:flex`, `<model>:batch` or the plain
  model name, so `usage.json` says how each token was bought.
- An OpenAI out-of-credits 429 or an Anthropic low-balance 400 stops the run
  at once instead of being retried (this happened on 2026-10-31).

### 2.4 The Flex stall, and what fixed it

On 2026-11-01 two Flex runs sat for nine hours without writing a cell.  Every
request (35,214 per model) had been handed to the HTTP client at once.
httpcore's connection pool walks its whole queue against every connection on
each event, so the process spent a full CPU core on bookkeeping and almost
nothing was sent.  The answers it did get were only in memory.  Fix (commit
90adf408):

- **At most `--max_in_flight` requests per model (default 200),** sent axis
  by axis.
- **Every billed answer goes to disk as it arrives,** in the cell's
  `answers.partial.jsonl`.  A rerun by any transport takes the saved answers
  to the same prompt and refuses a log written under other settings.  The
  log is kept as `answers.jsonl.gz` once the cell is written.  A run can be
  stopped at any moment and rerun, losing at most the requests in flight.
- **A progress line per model every five minutes.**
- **Stall restart.**  When no answer arrives for `--stall_minutes` (30) while
  requests remain, the run restarts itself (up to `--stall_restarts`, 3) and
  resumes from disk.
- **The 10-minute Flex fallback now holds.**  The clock starts at the first
  send, and each Flex attempt has no SDK retries and a timeout of the time
  left.

After the fix, the four rerun models finished in 30 minutes each, with no
fallbacks.

### 2.5 Prices

Cost is recomputed from each cell's token counts at the real-time list price
(`--price_basis list`), because batch and Flex cells are billed at half price
and would otherwise not compare.  All prices in
[`judge_pricing.py`](../../assistant_axis/judge_pricing.py) were checked against
the OpenAI and Anthropic pricing pages on 2026-11-01.  One was wrong:
gpt-5.6-sol is $4/$20 per million tokens (promotional, "at least through
November 21, 2026"), not $5/$30.  The OpenAI cost dashboard matched our
recorded token counts to the cent at the corrected price.  The Sol cells'
`usage.json` still records cost at the old price ($18.36 recorded, $12.51
billed); the plots recompute it.

Prompt caching does not apply to this task.  The part of each prompt shared
across an axis is 244 tokens, below every judge's minimum (OpenAI 1,024;
Anthropic 512 to 4,096).

### 2.6 How ρ is computed for the plots

- **Collision names.**  On the bare-keyed axes, nine names that are both a
  trait and a role were judged on one description and compared with the
  other's projection.  The robustness tool drops them, and the pooled plot
  takes its ρ from there (`--robustness_json`).
- **Coverage.**  A model judged on fewer axes than the plot's headline count
  is left off and named in the title, because two easy axes inflate ρ.  Your
  ccf7bbb3 added the same `--min_axes` flag; the merge keeps ours, which
  defaults to the headline count.
- **Legacy May cells are off by default** (`--legacy` to show them).  They
  mixed rubric versions and two Sonnet models, and gpt-4.1-mini was rejudged
  under the new settings.

### 2.7 The robustness check

Roger: "Hopefully all the plots tell about the same story on which model to
choose — if they don't, we could be overfitting."
[`judge_tier_robustness.py`](../../results_analysis/judge_tier_robustness.py)
recomputes every judge's ρ against 12 projection variants and reports
rankings, frontiers, Kendall τ, price bands and a bootstrap over axes.

Roger preregistered a guess: layer 49 would make some difference, slot 0 a
little, whitening little.  Outcome on 61 axes:

- **Layer 49: as guessed.**  It reshuffles the ranking as much as any change
  and lowers every judge's ρ by about 0.05.
- **Slot 0: more than guessed.**  It lowers ρ most (gpt-4.1-mini 0.605 to
  0.471 raw), and whitening wins most of that back.
- **Whitening: as guessed at slot 6, layer 25,** with at most 0.012 change.

The choice of judge does not depend on the projection.

### 2.8 Plot and tool details

- `--exclude_models` leaves out superseded runs, which the title names.
- Labels spell out each run's setting, and gpt-4.1-mini is marked as retiring.
- The cost axis is logarithmic, and labels are placed next to their own point.
- `assistant_axis/tests/test_judge_pricing.py::test_sonnet_canonical`
  expected Sonnet 4 at Sonnet 5's price and failed before these changes.  It
  now expects Sonnet 4's $3/$15; the code was right.

## 3. What it cost

As recorded in the cells' `usage.json`, at the price paid (batch and Flex at
half price):

| run | cells | recorded |
|---|---|---|
| claude-opus-5, effort low (your run, real time) | 61 | $196.27 |
| gpt-5.6-terra, default reasoning (your run) | 61 | $53.81 |
| claude-sonnet-5, effort low | 61 | $37.38 |
| claude-opus-5, thinking off | 20 | $32.19 |
| gpt-5.6-terra, reasoning off | 61 | $21.93 |
| gpt-5.6-sol, reasoning off (billed $12.51) | 20 | $18.36 |
| claude-haiku-4-5, T=1 | 61 | $16.45 |
| claude-haiku-4-5, T=0 | 61 | $15.90 |
| gpt-5.4-mini, reasoning off | 61 | $8.88 |
| gpt-5.6-luna, default reasoning (your run) | 61 | $6.19 |
| gpt-4.1-mini, T=0 | 61 | $3.98 |
| gpt-5.6-luna, reasoning off | 61 | $2.22 |
| claude-haiku-5-5, thinking off | 61 | $2.13 |
| gpt-6-luna, reasoning off | 61 | $1.00 |

The stalled Flex runs also lost about $9 of answers held in memory.

## 4. Open questions

- **Was gpt-4.1-mini helped by the vectors?**  In the April-May extraction,
  pipeline step 3 judged every response with gpt-4.1-mini, and each pole
  vector is the mean activation over the responses it scored 3.  The filter
  kept 89% of trait and 81% of role responses.  Roger and Claude are
  rebuilding the vectors without that filter and rerunning the 12-variant
  check; a later update will report it.  The October extraction, judged by
  gpt-5.6-luna, is a second test.
- **The 20-axis set is selected:** 18 of its axes were the highest-ρ axes of
  an earlier sweep judged by gpt-4.1-mini and Sonnet.  The 61-axis set is
  every axis with cached projections.
- **Repeat samples.**  Most judges ran once per item at T=1.  Averaging three
  samples could lift the cheap judges; Roger wants to try it once the
  shortlist is small.
- **Other judging tasks.**  This measures description judging only, not
  instruction or response judging, which have different prompts, token mixes
  and caching behaviour.

## 5. How to carry on

Preview first, then run.  The dry run makes no API calls:

```bash
uv run python results_analysis/judge_tier_cost_eval.py --axes_file scripts/tier_eval_axes.tsv \
    --instructions_dir /tmp/corpus_2026-05-24/data --judge_model <model> \
    [--cell_label @<label>] [--request_setting '<model>:<key>=<json>'] --dry_run
```

- **Pin the corpus.**  `--instructions_dir` must point at the corpus checked
  out at `1448a84c`.  The projections are only valid against those
  descriptions.
- **Keep settings consistent with the plotted cells.**  Use T=0 where
  accepted and reasoning off, and give a new setting its own `--cell_label`.
- **Leave `--transport auto`.**  It picks Flex or batch.  Stopping a run is
  safe, because a rerun resumes from `answers.partial.jsonl`.
- **Confirm big spends first.**  Roger's rule for this project: confirm any
  spend over $20 with him first, quoting each non-default flag.
- **Repair before plotting.**  After a run, `--repair_truncated` re-asks any
  replies cut at the cap.
- **Regenerate the outputs.**  The robustness check and pooled plots: the
  commands are at the end of each notes file.
- **Tests:**
  [`test_judge_tier_cost_eval.py`](../../results_analysis/tests/test_judge_tier_cost_eval.py),
  [`test_judge_tier_cost_plot_pooled.py`](../../results_analysis/tests/test_judge_tier_cost_plot_pooled.py),
  [`test_judge_tier_robustness.py`](../../results_analysis/tests/test_judge_tier_robustness.py),
  [`test_judge_pricing.py`](../../assistant_axis/tests/test_judge_pricing.py).
  None of them make network calls.

Reference: [`results_analysis/README.md`](../../results_analysis/README.md),
sections "Evaluating a new judge tier without recomputing activations" and
"Pooled plot across axes, and the robustness check".
