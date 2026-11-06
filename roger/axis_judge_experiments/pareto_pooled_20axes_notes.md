# Judge-Tier Pareto Analysis — 20 Axes

**Plot:** [`pareto_pooled_20axes.png`](./pareto_pooled_20axes.png)
**Robustness check:** [`judge_tier_robustness_20axes.png`](./judge_tier_robustness_20axes.png), numbers in [`judge_tier_robustness_20axes.json`](./judge_tier_robustness_20axes.json)

## Update, 2026-11-01: the 20-axis set, with both Opus 5 runs and GPT-5.6 Sol

The same settings, prices and corrections as the 61-axis update
([`pareto_pooled_61axes_notes.md`](./pareto_pooled_61axes_notes.md)), on the
first 20 axes of [`tier_eval_axes.tsv`](../../scripts/tier_eval_axes.tsv)
(listed in [`tier_eval_axes_20.tsv`](../../scripts/tier_eval_axes_20.tsv)).  Two
expensive judges were run on this set only: claude-opus-5 with thinking off,
and gpt-5.6-sol with reasoning off.  claude-opus-5 at the low effort setting
of the 26 October run is shown beside the thinking-off run.  The 26 October
analysis follows below, unchanged.

| judge | settings | ρ | cost per axis |
|---|---|---|---|
| claude-opus-5 | effort low (thinks); refuses T=0 | 0.704 | $3.181 |
| claude-opus-5 | thinking off | 0.693 | $3.198 |
| gpt-4.1-mini | T=0 | 0.688 | $0.130 |
| claude-haiku-4-5 | T=0 | 0.655 | $0.518 |
| claude-sonnet-5 | effort low (no thinking seen) | 0.648 | $1.182 |
| gpt-5.6-sol | reasoning off | 0.644 | $1.301 |
| claude-haiku-5-5 | thinking off | 0.633 | $0.069 |
| gpt-5.6-luna | reasoning off | 0.632 | $0.072 |
| gpt-5.6-terra | reasoning off | 0.631 | $0.718 |
| gpt-5.4-mini | reasoning off | 0.611 | $0.291 |
| gpt-6-luna | reasoning off | 0.572 | $0.033 |

**Pareto frontier:** gpt-6-luna, claude-haiku-5-5, gpt-4.1-mini, claude-opus-5
at low effort, the same in all 12 projection variants (gpt-5.6-luna joins in
4).  Opus at low effort is the top judge in all 12 variants and stays ahead of
Opus with thinking off in at least 98.7% of 2,000 bootstrap draws over axes in
every variant.  Opus with thinking off leads gpt-4.1-mini by only 0.006 at 25
times the cost.  Ranking agreement with the default: Kendall τ from 0.855 to
1.000.

These 20 axes are not a random sample: 18 of them were picked as the
highest-ρ axes of an earlier 35-axis sweep judged by gpt-4.1-mini and Sonnet,
so absolute ρ is higher here than on all 61, and the selection may favour
those judges.

**Update 2, 2026-11-01 (evening): every judge under the new method.**  The
rubric tuning described in update 2 of
[`pareto_pooled_61axes_notes.md`](./pareto_pooled_61axes_notes.md) was run on
this set for all ten judges, Opus 5 (thinking off) and Sol (reasoning off)
included: [`pareto_pooled_20axes_rubrics.png`](./pareto_pooled_20axes_rubrics.png)
(old and new points joined), the 12-variant check in
[`judge_tier_robustness_20axes_rubrics.png`](./judge_tier_robustness_20axes_rubrics.png)
and [`judge_tier_robustness_20axes_rubrics.json`](./judge_tier_robustness_20axes_rubrics.json).
The ten judges land within 0.04 of one another (0.745 to 0.786); Haiku 5.5,
Terra and Opus 5 share the top within 0.002, so price decides, and Opus at 40
times Haiku's price buys nothing measurable.  The table is in the 61-axis
notes.

**Reproduce** (from the repository root):

```bash
uv run python results_analysis/judge_tier_robustness.py --axes_file scripts/tier_eval_axes_20.tsv \
    --exclude_models claude-haiku-4-5 gpt-5.6-luna gpt-5.6-terra --output_dir /tmp/robust20
cp /tmp/robust20/judge_tier_robustness.json roger/axis_judge_experiments/judge_tier_robustness_20axes.json
cp /tmp/robust20/judge_tier_robustness.png roger/axis_judge_experiments/judge_tier_robustness_20axes.png
uv run python results_analysis/judge_tier_cost_plot_pooled.py \
    --tier_eval_dirs $(awk '!/^#/ && NF {print "roger/axis_judge_experiments/"$1"_tier_eval"}' scripts/tier_eval_axes_20.tsv) \
    --robustness_json roger/axis_judge_experiments/judge_tier_robustness_20axes.json \
    --exclude_models claude-haiku-4-5 gpt-5.6-luna gpt-5.6-terra \
    --output roger/axis_judge_experiments/pareto_pooled_20axes.png
```

---

## Original analysis, 2026-10-26

**Plot:** [`pareto_pooled_20axes.png`](./pareto_pooled_20axes.png)  
**Date:** 2026-10-26  
**Branch:** `personas-konstantinos`

---

### What Was Done

We evaluated five judge model tiers on 20 personality-axis pairs to find the best
cost–quality trade-off for description-mode axis judging.

**Pipeline:** `results_analysis/judge_tier_cost_eval.py` — scores entity descriptions
with a new judge model against an existing `projections.json` (the axis direction and
entity activation-projections computed previously), without touching a GPU.
Quality metric: Spearman ρ between the judge's −3..+3 scores and the activation
projections at layer slot 6.

**Corpus:** 571–580 entities (mixed traits and roles), pinned to commit `1448a84c`
(2026-05-24) so descriptions match the vectors that produced the projections.

**Axes evaluated (20 total):**
- 2 committed axes from the Sept 2026 session: `angel_vs_demon`, `decisive_vs_indecisive`
- 18 high-ρ axes from Roger's 35-axis GPT-vs-Sonnet sweep:
  `trustworthy_vs_untrustworthy`, `constructive_vs_destructive`, `helpful_vs_unhelpful`,
  `benign_vs_malicious`, `harmless_vs_harmful`, `improvisational_vs_methodical`,
  `cooperative_vs_competitive`, `reductionist_vs_holistic`,
  `individualistic_vs_collectivistic`, `passionate_vs_dispassionate`,
  `compassionate_vs_callous`, `convergent_vs_divergent`, `playful_vs_serious`,
  `truthful_vs_deceitful`, `practical_vs_theoretical`, `casual_vs_formal`,
  `generous_vs_stingy`, `introverted_vs_extroverted`

**Models evaluated:**
| Model | Provider | Rate (in/out per 1M) | Measured mean/axis | 20-axis total |
|---|---|---|---|---|
| `gpt-5.6-luna` | OpenAI | $0.20 / $1.20 | **$0.099** | $1.98 |
| `gpt-5.6-terra` | OpenAI | $2.00 / $12.00 | **$0.863** | $17.26 |
| `claude-opus-5` | Anthropic | $5.00 / $25.00 | **$3.181** | $63.61 |
| `gpt-4.1-mini` | OpenAI | (legacy; estimated) | ~$0.18 est. | — |
| `claude-sonnet-4` | Anthropic | (legacy; estimated) | ~$1.20 est. | — |

**Actual 20-axis spend: $82.86** (luna $1.98 + terra $17.26 + opus $63.61).  
Dry-run pre-estimate was $66–74; the difference is mainly opus ($2.49 estimated
vs $3.18 measured at 574 entities/axis, ~27% higher than the 575-entity estimate)
and terra ($1.09 estimated vs $0.86 measured, ~21% lower — the output tokens were
shorter than expected).

Legacy models (`gpt-4.1-mini`, `claude-sonnet-4`) are Roger's existing canonical
judges; their costs are estimated (no `usage.json`), their ρ values come from the
existing `gpt/` and `sonnet/` correlation caches.

---

### Key Finding

Across 20 axes, the **Pareto frontier** runs:

```
gpt-5.6-luna ($0.10, ρ≈0.627) → gpt-4.1-mini ($0.18 est., ρ≈0.684) → claude-opus-5 ($3.18, ρ≈0.701)
```

- **`gpt-4.1-mini` is Pareto-optimal** — Roger's existing canonical judge is on the
  efficiency frontier; no cheaper model matches its quality.
- **`gpt-5.6-luna`** offers a 45% cost saving over gpt-4.1-mini at the cost of ~0.06
  units of ρ — acceptable if budget is tight and slightly lower correlation is tolerable.
- **`gpt-5.6-terra`** is dominated by gpt-4.1-mini (higher cost, lower quality) across
  these 20 axes; not recommended.
- **`claude-opus-5`** achieves the best raw ρ (~0.701) but at 18× the cost of luna;
  marginal gain over gpt-4.1-mini (~0.017 ρ) does not justify the spend for
  corpus-scale judging.

---

### Output Files

All results live under `roger/axis_judge_experiments/` (gitignored):

```
<axis>_tier_eval/
    gpt-5.6-luna/
        scores_descriptions.json   # per-entity scores
        correlations.json          # Spearman ρ by slot and metric
        usage.json                 # actual token counts and cost
    gpt-5.6-terra/  (same structure)
    claude-opus-5/  (same structure)
```

Batch script: [`scripts/run_tier_eval_20axes.sh`](../../scripts/run_tier_eval_20axes.sh)  
Plotter: [`results_analysis/judge_tier_cost_plot_pooled.py`](../../results_analysis/judge_tier_cost_plot_pooled.py)

---

### Reproduce

```bash
# Regenerate the plot from existing results:
uv run python results_analysis/judge_tier_cost_plot_pooled.py \
    --experiments_dir roger/axis_judge_experiments \
    --slot 6 \
    --output roger/axis_judge_experiments/pareto_pooled_20axes.png

# Re-run all 20 axes (corpus must be pinned first):
mkdir -p /tmp/corpus_2026-05-24
git archive 1448a84c data/traits/instructions data/roles/instructions \
    | tar -x -C /tmp/corpus_2026-05-24
bash scripts/run_tier_eval_20axes.sh 2>&1 | tee /tmp/tier_eval_20axes.log
```
