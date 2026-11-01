# Judge-Tier Pareto Analysis — 61 Axes

**Plot:** [`pareto_pooled_61axes.png`](./pareto_pooled_61axes.png)
**Robustness check:** [`judge_tier_robustness_61axes.png`](./judge_tier_robustness_61axes.png), numbers in [`judge_tier_robustness_61axes.json`](./judge_tier_robustness_61axes.json)

## Update, 2026-11-01: every judge on all 61 axes under one set of settings

The plot was redrawn after the judges were rerun or added so that every point
is made the same way.  The 26 October analysis follows below, unchanged, for
the record.

**What changed**

- **One rubric, one corpus, consistent settings.**  Every cell uses the v3
  descriptions-mode prompt against the corpus pinned at `1448a84c`.  Each judge
  runs at temperature 0 where its API accepts it, and with reasoning switched
  off where it can be.  The May 2026 cells (`gpt-4.1-mini` and `claude-sonnet-4`
  from earlier sessions) are no longer plotted; gpt-4.1-mini was rejudged under
  these settings.
- **Reply cap.**  A reply cut at `max_tokens` is now scored as missing rather
  than parsed for a stray number, and the few cut replies were asked again at
  300 tokens (`--repair_truncated` in
  [`judge_tier_cost_eval.py`](../../results_analysis/judge_tier_cost_eval.py)).
- **Collision names.**  On the bare-keyed axes, nine names that are both a
  trait and a role had been judged on one description and compared with the
  other's projection.  ρ now comes from the robustness tool, which drops them
  ([`judge_tier_robustness.py`](../../results_analysis/judge_tier_robustness.py)).
- **Prices.**  Cost is recomputed from each cell's token counts at the
  provider's real-time list price, checked against both pricing pages on
  2026-11-01.  gpt-5.6-sol had been priced at $5/$30 per million tokens; it is
  $4/$20 (promotional, "at least through November 21, 2026").  Flex and Batch
  are half price for every judge here, so the frontier is the same at either
  price.
- **Left out:** `claude-haiku-4-5` at temperature 1, superseded by the
  temperature-0 run shown.

**The judges** (ρ: mean Spearman correlation over the 61 axes at slot 6, layer
25, raw projection; cost per axis at list price)

| judge | settings | ρ | cost per axis |
|---|---|---|---|
| gpt-4.1-mini | T=0 | 0.605 | $0.130 |
| claude-haiku-4-5 | T=0 | 0.561 | $0.521 |
| claude-sonnet-5 | effort low (no thinking seen); refuses T=0 | 0.551 | $1.187 |
| gpt-5.6-terra | reasoning off | 0.550 | $0.719 |
| gpt-5.6-luna | reasoning off | 0.547 | $0.073 |
| claude-haiku-5-5 | thinking off; refuses T=0 | 0.544 | $0.070 |
| gpt-5.4-mini | reasoning off | 0.514 | $0.291 |
| gpt-6-luna | reasoning off | 0.478 | $0.033 |

**Pareto frontier:** gpt-6-luna, claude-haiku-5-5, gpt-5.6-luna, gpt-4.1-mini.
gpt-4.1-mini is the best judge at any price on this task.  claude-haiku-5-5
and gpt-5.6-luna are within 0.003 of each other in ρ and cost.

**Robustness: does the choice depend on which projection the scores are
compared with?**  The robustness tool recomputes every judge's ρ against 12
projections: token slot 0 or 6, layer 25 or 49, and three regimes (raw; soft
whitening with K=2; soft shear with L=3, combined goal/no-goal subspaces).

- gpt-4.1-mini is the top judge in all 12, and in every bootstrap draw over
  axes (2,000 draws) it stays ahead of the runner-up, claude-haiku-4-5 at T=0.
- The frontier holds gpt-6-luna, claude-haiku-5-5 and gpt-4.1-mini in all 12
  variants; gpt-5.6-luna joins it in 6.
- The ranking agrees with the default's in every variant (Kendall τ from
  0.714 to 1.000).
- Layer 49 changes the ranking most (τ 0.714 at slot 6) and lowers every
  judge's ρ by about 0.05.  Slot 0 lowers ρ most (gpt-4.1-mini 0.605 to 0.471
  raw), and whitening wins most of that back at slot 0.  At slot 6, layer 25,
  whitening and shear change ρ by at most 0.012.

**Caveats**

- **gpt-4.1-mini leaves the API on 2027-01-14.**  OpenAI names gpt-6-luna as
  its replacement, which scores ρ 0.478 against 0.605 here, so it is not a
  like-for-like swap on this task.
- **Prices are per item for this task**: prompts of about 315 tokens, of which
  the 244-token rubric is shared across an axis, and replies of about 60.
  That is below every judge's minimum for prompt caching, so caching cannot
  lower these costs.  A different task (judging responses, with long prompts)
  has a different input/output mix and needs its own prices.
- Only gpt-4.1-mini and claude-haiku-4-5 ran at T=0.  The other judges
  refuse T=0 or take no temperature setting, and ran at their default with
  one sample per item.
- Opus 5 is on the 20-axis plot only
  ([`pareto_pooled_20axes_notes.md`](./pareto_pooled_20axes_notes.md)).

**Reproduce** (from the repository root; the robustness tool reads the May
2026 8-slot vectors by default):

```bash
uv run python results_analysis/judge_tier_robustness.py --axes_file scripts/tier_eval_axes.tsv \
    --exclude_models claude-haiku-4-5 --output_dir /tmp/robust61
cp /tmp/robust61/judge_tier_robustness.json roger/axis_judge_experiments/judge_tier_robustness_61axes.json
cp /tmp/robust61/judge_tier_robustness.png roger/axis_judge_experiments/judge_tier_robustness_61axes.png
uv run python results_analysis/judge_tier_cost_plot_pooled.py --experiments_dir roger/axis_judge_experiments \
    --robustness_json roger/axis_judge_experiments/judge_tier_robustness_61axes.json \
    --exclude_models claude-haiku-4-5 --output roger/axis_judge_experiments/pareto_pooled_61axes.png
```

---

## Original analysis, 2026-10-26

**Plot:** [`pareto_pooled_61axes.png`](./pareto_pooled_61axes.png)  
**Date:** 2026-10-26  
**Branch:** `personas-konstantinos`  
**Extends:** [`pareto_pooled_20axes_notes.md`](./pareto_pooled_20axes_notes.md) (20-axis pilot)

---

### What Was Done

Scaled the judge-tier cost–quality evaluation from 20 to all 61 axes in the
`roger/axis_judge_experiments/` archive.

**Pipeline:** `results_analysis/judge_tier_cost_eval.py` — scores entity
descriptions with a new judge model against a cached `projections.json` (axis
direction + activation projections from the Qwen-3-32B 8-slot run), without
touching a GPU.  Quality metric: Spearman ρ between judge scores and projections
at layer slot 6.

**Corpus:** 571–580 entities (mixed traits and roles) per axis, pinned to commit
`1448a84c` (2026-05-24).

**Models with full 61-axis coverage (this session):**
| Model | Provider | Rate (in/out per 1M) | Measured mean/axis | 61-axis total |
|---|---|---|---|---|
| `gpt-5.6-luna` | OpenAI | $0.20 / $1.20 | $0.101 | $6.19 |
| `gpt-5.6-terra` | OpenAI | $2.00 / $12.00 | $0.882 | $53.81 |
| `claude-opus-5` | Anthropic | $5.00 / $25.00 | $3.218 | $196.27 |

**Total measured spend (61 axes × 3 models): $256.27**  
(Full session including the 20-axis pilot: $82.86 + additional $173.41 = ~$347 combined.)

**Legacy models (estimated cost, n=60–61 axes, judged in earlier sessions):**
| Model | Mean ρ | Mean cost/axis | n axes |
|---|---|---|---|
| `gpt-4.1-mini` | 0.600 | ~$0.18 est. | 61 |
| `claude-sonnet-4` | 0.534 | ~$1.50 est. | 60 |

**Low-n reference models (n=2 axes only — angel/demon + decisive/indecisive):**
| Model | Mean ρ | Mean cost/axis | n axes |
|---|---|---|---|
| `claude-haiku-4-5` | 0.663 | $0.52 | 2 |
| `gpt-5.6-sol` | 0.682 | $2.45 | 2 |
| `claude-sonnet-5` | 0.647 | $1.18 | 2 |

---

### Key Findings

#### Pareto frontier (robust models only, n=61)

```
gpt-5.6-luna ($0.10, ρ=0.534) → gpt-4.1-mini ($0.18 est., ρ=0.600) → claude-opus-5 ($3.22, ρ=0.621)
```

- **`gpt-4.1-mini` remains Pareto-optimal** at 61 axes: 44% cheaper than the next
  measured model (terra at $0.88), with ρ=0.600 vs terra's ρ=0.552.
- **`claude-opus-5` is barely Pareto-optimal** vs gpt-4.1-mini: ρ=0.621 vs 0.600
  (+0.021) at 18× the cost ($3.22 vs $0.18). For corpus-scale judging this premium
  is hard to justify.
- **`gpt-5.6-terra` is dominated**: sits between luna and gpt-4.1-mini in cost but
  below both in quality. Do not use.
- **`gpt-5.6-luna`** ($0.10, ρ=0.534) is the cheapest Pareto-optimal option — 44%
  cost saving vs gpt-4.1-mini but ρ 0.066 lower. Acceptable only if budget is very tight.

#### Why quality dropped from the 20-axis pilot

The 20-axis pilot cherry-picked the **highest-ρ axes** from Roger's 35-axis sweep.
Those axes show mean ρ ~0.62–0.70. The additional 41 axes include harder,
lower-ρ pairs that pull the overall mean down to ~0.53–0.62. The Pareto shape is
consistent; the absolute numbers are just lower.

#### Low-n caution: haiku and sol

`claude-haiku-4-5` (ρ=0.663, $0.52) and `gpt-5.6-sol` (ρ=0.682, $2.45) appear
Pareto-optimal but are based on **n=2 axes only**. If extended to 61 axes:
- Haiku at $0.52 and ρ>0.60 would dominate gpt-4.1-mini (same quality, 3× cost),
  but haiku at $0.52 and ρ<0.60 would just fill the gap between luna and gpt-4.1-mini.
- Sol at $2.45 dominating opus at $3.22 is plausible given they share the same
  capability tier, but unconfirmed.

Extending haiku to 61 axes costs ~61 × $0.52 ≈ **$32** and would resolve this.

---

### Batch Scripts

| Script | Axes | Spend |
|---|---|---|
| [`scripts/run_tier_eval_20axes.sh`](../../scripts/run_tier_eval_20axes.sh) | First 20 (high-ρ) | $82.86 measured |
| [`scripts/run_tier_eval_41axes.sh`](../../scripts/run_tier_eval_41axes.sh) | Remaining 41 | ~$173 measured |

---

### Reproduce

```bash
# Regenerate the plot from existing results:
uv run python results_analysis/judge_tier_cost_plot_pooled.py \
    --experiments_dir roger/axis_judge_experiments \
    --slot 6 \
    --output roger/axis_judge_experiments/pareto_pooled_61axes.png

# Re-run all 61 axes from scratch (corpus must be pinned first):
mkdir -p /tmp/corpus_2026-05-24
git archive 1448a84c data/traits/instructions data/roles/instructions \
    | tar -x -C /tmp/corpus_2026-05-24
bash scripts/run_tier_eval_20axes.sh 2>&1 | tee /tmp/tier_eval_20axes.log
bash scripts/run_tier_eval_41axes.sh 2>&1 | tee /tmp/tier_eval_41axes.log
```
