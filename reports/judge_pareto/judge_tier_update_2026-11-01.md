# Judge-tier evaluation: what changed on 2026-11-01, and why

For Konstantinos and the Claude sessions working with him.  Roger and Claude
made these changes on 2026-11-01 and 02, starting from `personas-konstantinos` at
c7743467, and merged your later commits (9b3d6a5c, ccf7bbb3) back in.  This
file explains what the pull request contains, the decisions behind it, and how
to carry on from here.  Section 3d says where the method stood on 2 November, and section 3e
what changed on 3 and 4 November (instruction judging, balanced pole order, Jev as a fourth
judge, response judging).  The per-plot notes hold the full numbers:
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

Prompt caching cannot lower this task's costs.  The part of each prompt shared
across an axis is 244 tokens, below every judge's minimum (OpenAI 1,024;
Anthropic 512 to 4,096), so nothing could be read back.  It can raise them:
GPT-5.6 and later write any prompt of 1,024 tokens or more to the cache by
default, at 1.25x the input rate.  Single-entity prompts are far below that;
batched prompts at N=20 are not (section 3c), so the tools turn caching off
since 2026-11-02.

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

## 3b. Same day, later: the rubric was the frontier

The strong gpt-4.1-mini point above turned out to be the rubric's doing, not
the model's.  The v3 descriptions rubric is a 2023-style judge prompt whose 0
row ("neutral, ambiguous, or does not map onto this axis") newer models take
literally: they answered 0 for 52% to 70% of entities against gpt-4.1-mini's
32%, and a 0 ties every entity the judge would not place while the projection
orders them all.  Roger had seen the same with the RP-judging rubric: a prompt
that suits gpt-4.1-mini under-serves current models, and a rewrite lets them
overtake it.

So the four judges at the cheap end of the frontier (gpt-4.1-mini, Haiku 5.5,
GPT-5.6 Luna, GPT-6 Luna; Opus 5 and Sol dropped) were judged again with
contemporary rubrics, tuned on 30 of the 61 axes and confirmed on the other
31.  The winner was not a wording but a method: **ask for the entity's
percentile among all the roles and traits on the axis, judge ten entities
per call, and average two random batch orders** (`--rubric b10
--entities_per_call 10 --shuffle_seed 1|2`, cells `<judge>@b10x2`).  Every
judge gained 0.11 to 0.23 in ρ at 35% to 40% lower cost, the gain was as large
on the held-out axes, and the four now sit within 0.017 of one another:

| judge | old rubric | new method | cost per axis, old → new |
|---|---|---|---|
| gpt-5.6-luna, reasoning off | 0.547 | **0.725** | $0.073 → $0.047 |
| gpt-4.1-mini, T=0 | 0.605 | **0.720** | $0.130 → $0.084 |
| claude-haiku-5-5, thinking off | 0.544 | **0.716** | $0.070 → $0.048 |
| gpt-6-luna, reasoning off | 0.478 | **0.708** | $0.033 → $0.021 |

The remaining judges were then run the same way (the Anthropic ones through
a Batch API path added for batched judging): on 61 axes all eight land within
0.035 of one another (Terra 0.731, Luna 0.725, gpt-4.1-mini 0.720, Haiku 5.5
0.716, Sonnet 5 0.713, GPT-6 Luna 0.708, GPT-5.4 mini 0.699, Haiku 4.5 0.696),
and on the 20-axis set the ten judges, Opus 5 and Sol included, within 0.04
(Haiku 5.5 0.786 to GPT-5.4 mini 0.745).  Every judge gained 0.11 to 0.23 and
cost 35% to 45% less.  Price decides: the 61-axis frontier is GPT-6 Luna,
GPT-5.6 Luna and Terra; Opus 5 at 40 times Haiku 5.5's price buys nothing
measurable.  Full tables in update 2 of the 61-axis notes.

The plot with each judge's old and new points joined:
[`pareto_pooled_61axes_rubrics.png`](../../roger/axis_judge_experiments/pareto_pooled_61axes_rubrics.png);
the tuning record, every variant per judge:
[`pareto_pooled_dev30_rubric_tuning.png`](../../roger/axis_judge_experiments/pareto_pooled_dev30_rubric_tuning.png);
the full account, with the variants that lost and why, in update 2 of
[`pareto_pooled_61axes_notes.md`](../../roger/axis_judge_experiments/pareto_pooled_61axes_notes.md).
The rubrics are in
[`results_analysis/judge_tier_rubrics.py`](../../results_analysis/judge_tier_rubrics.py);
the README section "Rubric variants and batched judging" explains the
options.  Two tools go with batched cells:
[`scripts/judge_tier_average_cells.py`](../../scripts/judge_tier_average_cells.py)
(the two shuffles into one cell) and
[`scripts/judge_tier_reparse_batched.py`](../../scripts/judge_tier_reparse_batched.py)
(re-score a batched cell from its archived replies, no API calls).

Two practical lessons from the day: keep Flex to about four processes at a
time (eight, with an OS download in the background, dropped every connection
at once), and give batched replies room (`--max_tokens 1500` for Haiku 5.5,
which writes about 60 tokens per entity).

## 3c. 2026-11-02: tuning the batched method

The method in 3b used two passes at ten entities per call because that is where the
tuning stopped, not because it had been tested.  A grid on the 30 development axes then
varied one thing at a time for GPT-5.6 Luna, GPT-6 Luna and Haiku 5.5: passes (1 to 4),
entities per call N (5, 7, 10, 15, 20) and reasoning effort (none, low, medium).

**Use N=20, two passes, no reasoning.**

- **N=20** costs about 11% less than N=10 with prompt caching off (see below) and is no
  worse.  On all 61 axes it gains
  +0.0072 [+0.0032, +0.0111] for Luna, +0.0036 [-0.0001, +0.0073] for GPT-6 Luna and
  -0.0001 [-0.0033, +0.0030] for Haiku (95% bootstrap over axes).  N=30 is untested.
- **More passes** help with diminishing returns: the third adds about 0.009, the fourth
  about 0.004.
- **Reasoning** costs a third to 70% more with no gain distinguishable from zero.
- **A second judge beats more passes of the first.**  Luna + Haiku, two passes each at
  N=20, beats Luna alone by +0.0119 [+0.0047, +0.0177] on all 61 axes and +0.0115 [+0.0040,
  +0.0187] on the 31 held out of the tuning.  GPT-6 Luna adds nothing to Luna.

| 61 axes, slot 6, layer 25, raw | ρ | $/axis at list price |
|---|---|---|
| GPT-6 Luna, N=20, 2 passes | 0.712 | $0.019 |
| GPT-5.6 Luna, N=20, 2 passes | 0.732 | $0.042 |
| GPT-5.6 Luna + Haiku 5.5, N=20, 2 passes each | 0.744 | $0.084 |

These three are the frontier in 12 of 12 projection variants.  GPT-5.6 Terra (0.731 at
$0.44) is now dominated.  The winners were chosen on the 30 development axes and then run on
the 31 held-out ones.  Flex and Batch prices are half the list prices shown; the grid and
the held-out check cost $18.77 as billed.

**These N=20 cells ran with OpenAI's prompt caching on, the default**, and 93% of their prompts
pass its 1,024-token minimum.  Calculated from the logged tokens, the write premium takes Luna
from $0.0423 to $0.0452 an axis at list and GPT-6 Luna from $0.0189 to $0.0204, about $0.13 in
all at Flex prices, which neither `usage.json` nor these tables show.  With caching on, N=15
($0.0441 and $0.0197) is slightly cheaper than N=20; the comparison above holds with it off.
Every OpenAI send on this branch has turned it off since 2026-11-02: in a live check of both
Lunas, a 1,355-token prompt wrote about 1,350 tokens by default and none with the field.  Haiku
5.5 is not affected, since Anthropic caches only when asked.

Plots: [`pareto_pooled_61axes_grid.png`](../../roger/axis_judge_experiments/pareto_pooled_61axes_grid.png),
[`judge_tier_grid_dev30.png`](../../roger/axis_judge_experiments/judge_tier_grid_dev30/judge_tier_grid_dev30.png),
[`judge_tier_robustness_61axes_grid.png`](../../roger/axis_judge_experiments/judge_tier_robustness_61axes_grid.png).
Full tables, caveats and commands: update 3 of
[`pareto_pooled_61axes_notes.md`](../../roger/axis_judge_experiments/pareto_pooled_61axes_notes.md).

**The whole grid under the 12 projection variants** (30 development axes only, the only axes
that have it):
[`batch size`](../../roger/axis_judge_experiments/judge_tier_grid_dev30/judge_tier_grid_12way_batch_size.png),
[`passes`](../../roger/axis_judge_experiments/judge_tier_grid_dev30/judge_tier_grid_12way_passes.png),
[`reasoning`](../../roger/axis_judge_experiments/judge_tier_grid_dev30/judge_tier_grid_12way_reasoning.png),
[`whole grid`](../../roger/axis_judge_experiments/judge_tier_grid_dev30/judge_tier_grid_12way_all.png).
Four settings are on the whole grid's frontier in all 12: GPT-6 Luna with one pass at N=10
($0.011) and at N=20 ($0.019), Luna at N=20 ($0.042) and Luna + Haiku at N=20 ($0.084).  No
reasoning arm, no four-pass arm and no two-pass arm at N of 10 or less is on it in any variant.

**Gemini is not on the frontier.**  `gemini-3.1-flash-lite` at N=20 and two passes scores 0.687
at $0.068 on the 30 development axes, level with Haiku (0.684 at $0.042) at 60% more;
`gemini-3.5-flash-lite` scores 0.653 at $0.108.  As Luna's second opinion, 3.1 gives 0.714 at
$0.110 against Haiku's 0.716 at $0.084.  Neither was run on the held-out axes.
`gemini-3.5-flash-lite` is out of the running (Roger, 2026-11-02).  `gemini-3.1-flash-lite` has
an earliest shutdown date of 2027-05-07 on
[Google's deprecations page](https://ai.google.dev/gemini-api/docs/deprecations), which names
3.5 as its replacement.

## 3d. 2026-11-02, evening: Gemini's grid, the decision, and the port to production

**The decision (Roger):** N=20, two passes, no reasoning.  The default judges are GPT-5.6
Luna and Haiku 5.5, blended; Gemini 3.1 Flash-Lite stays a supported third judge, and 3.5 is
out of the running.  Two judges are the default (Roger, the same day): the two-model ensemble was
found significantly better than either model alone, while three models bring three pairs and a trio
to evaluate, so the default stays close to what was being done until a third judge is proven.  The
default is one line in [`judge_descriptions.py`](../../results_analysis/judge_descriptions.py), and
May 2027 for 3.1 is not a concern to him.

**Gemini 3.1 on the full grid** (30 development axes, $21.09 at list), change in ρ from
each judge's own N=10, two passes, no reasoning:

| | GPT-5.6 Luna | GPT-6 Luna | Haiku 5.5 | Gemini 3.1 |
|---|---:|---:|---:|---:|
| N=5 | −0.005 | −0.006 | −0.008 | −0.008 |
| N=20 | +0.006 | +0.004 | −0.001 | +0.005 |
| 1 pass | −0.023 | −0.027 | −0.022 | −0.028 |
| 3 passes | +0.010 | +0.009 | +0.009 | +0.009 |
| low reasoning | +0.004 | −0.003 | +0.005 | −0.010 |
| medium reasoning | +0.002 | −0.003 | +0.005 | −0.006 |
| medium reasoning, cost | ×1.66 | ×1.70 | ×1.58 | ×3.37 |

Batch size and passes act the same.  Reasoning hurts Gemini (low: [−0.015, −0.005]) and medium
is dear.  N=20 and 1 and 3 passes are on Gemini's own frontier in 12 of 12 variants, N=5 and N=7
never: [`judge_tier_grid_shapes.png`](../../roger/axis_judge_experiments/judge_tier_grid_dev30/judge_tier_grid_shapes.png),
[`judge_tier_grid_own_frontier.png`](../../roger/axis_judge_experiments/judge_tier_grid_dev30/judge_tier_grid_own_frontier.png).

**Ensemble size** (N=20, two passes each): Luna 0.704, Luna + Haiku 0.716 (+0.012 [−0.000, +0.022]),
with Gemini 0.720 (+0.004 [−0.001, +0.010], ahead in 9 of 12 variants).  A third pass of Luna and
Haiku at N=10 gains as much, +0.005 [+0.001, +0.009], for $0.143 against $0.152.  3.1 beats 3.5 by +0.034 [+0.019, +0.051], ahead
in 12 of 12 variants, for less money.

**What the production audit found.**  `judge.py` and this tool sent byte-identical bodies, but
production's axis judge had no batched rubric, sent T=0 and no reasoning setting, and turned the HTTP
400 Luna and Haiku 5.5 answer into None for every entity.  So the tool moved to production on top of
`judge.py` as the one request layer, which also gave step 3 the Flex limits and the out-of-credits
stop.  Double-checked: 141,719 of 141,719 recorded prompts rebuild byte-for-byte with
`--name_format mechanical`, 49,668 of 49,668 groups at N other than 10 reproduce from their seed, the
wire bodies match on all five paths including Gemini, and a real run through all three judges works
end to end.

**Two faults found on the way.**  A failed call used to finish its cell: an outage left eight empty
cells marked done, which the averaging took in (now `judge.FAILED`, the cell stays open, and averaging
refuses a short source).  And Luna sometimes writes `SCORE: thirty-five`: 978 ratings of 1.4 million,
now parsed and filled into the cells from their logs, which moves no mean ρ in the third decimal
except where the tables above were rebuilt.

## 3e. 2026-11-03 and 04: instruction judging, balanced pole order, Jev, and responses

Shorter here; each item links to the full account.

- **Instruction judging (2026-11-03).**  Every judge judges each of an entity's five instructions,
  one instruction per entity per call (`i1_p0`), 20 entities a call:
  [`judge_instruction_grid.py`](../../scripts/judge_instruction_grid.py) `--design rounds`.  Roger's
  rule, against choosing among some twenty arms a few thousandths apart: one pass of every model.
  Luna + Haiku + Gemini at 1x scored 0.777 at $0.178 an axis, against Luna alone at 0.767
  ([`results_analysis/README.md`](../../results_analysis/README.md) § "Instruction judging: the
  settled method").  The same rule put Gemini into description judging: the trio at 2x, 0.752 at
  $0.076.
- **Prices and the frontier.**  Cost is now the price actually paid: each judge's Flex or Batch
  rate, with cached tokens priced.  Gemini runs through its Batch API
  ([`gemini_batch.py`](../../assistant_axis/gemini_batch.py)).  A point is on the Pareto frontier
  only when it is on the upper convex hull of mean ρ against cost.
- **Pole order balanced (2026-11-04, Roger's rule).**  Every axis prompt names one pole first, and
  the order matters: Luna's repeats in the same order agree more than in opposite orders, and on
  axes with a morally good pole Luna scores better with that pole high.  So no judging runs in one
  fixed order any more.  Descriptions take one pass each way, instructions follow Roger's table
  (`judge_tier_cost_eval.INSTRUCTION_DIRECTIONS`), and responses take a coin per call.  **Scores made
  before 2026-11-04 are in the pair list's order**, which favoured the 18 moral axes by about
  +0.03 ρ on descriptions.  Account: § 4c of
  [`response_judging_plan.md`](./response_judging_plan.md).
- **Jev (TypeSafe AI) as a fourth judge (2026-11-04).**  Batched, 160 entities a request, one
  yes/no-probability question per entity ("nearer the second end than most of the other
  entries?"), at $0.042 per million input tokens with output free.  Alone it scores above every
  other judge (descriptions 0.745 against Luna's 0.709 at one pass): [`jev_judging.md`](./jev_judging.md).  The drivers request
  the pinned version, `jev-1.14.0`, and stop if any answer comes from another version.  **The standard since then, in both modes: Luna + Haiku +
  Gemini + Jev, two passes each, one each way, weighted 4 : 3 : 3 : 12.**  Descriptions: 0.766 at
  $0.084 an axis; instructions: 0.789 at $0.398.  **Jev's weight is chosen for resilience, not
  fitted to the mean.**  The mean-ρ fit gave 24, which made the blend 68% Jev in effect, ranking
  entities as Jev alone does (0.99); Roger: "That's not an ensemble, that's an addendum".  Jev's
  bad axes are one class, the style axes (precise vs vague −0.105 against the trio in both
  modes), and the per-axis gaps are fat-tailed, so 12, the weight with the smallest worst-axis
  shortfall (0.026 and 0.036, against 0.043 and 0.048 at 24), replaced it for a mean ρ within
  noise; dropping each entity's outlying score, which Roger also asked about, does worse
  ([`jev_judging.md`](./jev_judging.md) § "Jev's weight: 12, for resilience",
  [`jev_weight_resilience.py`](../../scripts/jev_weight_resilience.py)).  Luna + Haiku + Jev at
  2x is within 0.0005 of the four in both modes, for about 60% of the cost; the figures are
  [`judge_description_12way.png`](../../roger/axis_judge_experiments/instruction_judging_12way/judge_description_12way.png)
  and
  [`judge_instruction_12way.png`](../../roger/axis_judge_experiments/instruction_judging_12way/judge_instruction_12way.png),
  with [notes](../../roger/axis_judge_experiments/instruction_judging_12way/notes.md).
- **Response judging** (in progress, 12 axes): the batched method with a cached prefix, what
  share of the responses to judge, and the cost Pareto:
  [`response_judging_plan.md`](./response_judging_plan.md) and
  [`judge_response_pareto.png`](../../roger/axis_judge_experiments/response_pareto/judge_response_pareto.png).
  Its hollow extrapolated points (1/3 and all of the responses) are unreliable.  The round-robin
  ensembles' extrapolated segments rise more steeply than their measured curves, and the 1.86
  correction was fitted on Luna alone.  The measured points, from 1/45 to 1/9 of the responses,
  are sound.

## 4. Open questions

- **Was gpt-4.1-mini helped by the vectors?**  No, as far as the step-3
  filter goes.  In the April-May extraction, pipeline step 3 judged every
  response with gpt-4.1-mini, and each pole vector is the mean activation
  over the responses it scored 3.  Rebuilding every vector from all 500
  responses left the ranking, the frontier and gpt-4.1-mini's lead unchanged
  (mean ρ moved by less than 0.005;
  [`judge_tier_robustness_61axes_filtered_vs_unfiltered.txt`](../../roger/axis_judge_experiments/judge_tier_robustness_61axes_filtered_vs_unfiltered.txt)).
  A side finding: on axes whose poles lost most responses to the filter
  (dominant/submissive, harmless/harmful) every judge agrees better with the
  unfiltered vectors, most at slot 0, which may mean the persona is in the
  activations even when the response refuses it; a TODO in AGENT_NOTES.
- **The 20-axis set is selected:** 18 of its axes were the highest-ρ axes of
  an earlier sweep judged by gpt-4.1-mini and Sonnet.  The 61-axis set is
  every axis with cached projections.
- **Repeat samples.**  Answered by 3c: passes in fresh random batch orders are the
  repeat samples, and two at N=20 is the setting to use.
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
