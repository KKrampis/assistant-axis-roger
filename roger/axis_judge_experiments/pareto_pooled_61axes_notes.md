# Judge-Tier Pareto Analysis — 61 Axes

**Plot:** [`pareto_pooled_61axes.png`](./pareto_pooled_61axes.png)  
**Date:** 2026-10-26  
**Branch:** `personas-konstantinos`  
**Extends:** [`pareto_pooled_20axes_notes.md`](./pareto_pooled_20axes_notes.md) (20-axis pilot)

---

## What Was Done

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

## Key Findings

### Pareto frontier (robust models only, n=61)

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

### Why quality dropped from the 20-axis pilot

The 20-axis pilot cherry-picked the **highest-ρ axes** from Roger's 35-axis sweep.
Those axes show mean ρ ~0.62–0.70. The additional 41 axes include harder,
lower-ρ pairs that pull the overall mean down to ~0.53–0.62. The Pareto shape is
consistent; the absolute numbers are just lower.

### Low-n caution: haiku and sol

`claude-haiku-4-5` (ρ=0.663, $0.52) and `gpt-5.6-sol` (ρ=0.682, $2.45) appear
Pareto-optimal but are based on **n=2 axes only**. If extended to 61 axes:
- Haiku at $0.52 and ρ>0.60 would dominate gpt-4.1-mini (same quality, 3× cost),
  but haiku at $0.52 and ρ<0.60 would just fill the gap between luna and gpt-4.1-mini.
- Sol at $2.45 dominating opus at $3.22 is plausible given they share the same
  capability tier, but unconfirmed.

Extending haiku to 61 axes costs ~61 × $0.52 ≈ **$32** and would resolve this.

---

## Batch Scripts

| Script | Axes | Spend |
|---|---|---|
| [`scripts/run_tier_eval_20axes.sh`](../../scripts/run_tier_eval_20axes.sh) | First 20 (high-ρ) | $82.86 measured |
| [`scripts/run_tier_eval_41axes.sh`](../../scripts/run_tier_eval_41axes.sh) | Remaining 41 | ~$173 measured |

---

## Reproduce

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
