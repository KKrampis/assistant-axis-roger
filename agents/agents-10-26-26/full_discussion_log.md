# Session Log — Judge Tier Eval Scale-Up: 20 Axes, Pooled Pareto (2026-10-26)

*Continuation of the Sept 28 judge-tier work documented in
[`agents/agents-9-28-26/full_discussion_log.md`](../agents-9-28-26/full_discussion_log.md).
That session built `judge_tier_cost_eval.py`, ran it on 2 axes (angel/demon,
decisive/indecisive) for 6 models, and found `gpt-5.6-terra` as the standout value
tier. This session scales up to 20 axes to get a statistically robust Pareto frontier.*

---

## 1. Starting State

Branch `personas-konstantinos`. The previous session left:
- 2 axes with tier-eval results committed under
  `roger/axis_judge_experiments/{angel_vs_demon,decisive_vs_indecisive}_tier_eval/`
- `results_analysis/judge_tier_cost_eval.py` — scores a new judge against an
  existing `projections.json`, makes no GPU calls
- `results_analysis/judge_tier_cost_plot.py` — single-axis cost-vs-quality plotter
- The `judging/` archives present but **not yet extracted**

## 2. Archive Extraction

Extracted `judging/axis_judge_experiments_untracked.tar.xz` (420 MB compressed,
3.3 GB raw) from repo root:

```bash
tar -xJf judging/axis_judge_experiments_untracked.tar.xz
```

Result: **61 axes** now have `projections.json` under
`roger/axis_judge_experiments/<axis>/gpt/` (was 2 before extraction).
The directory is gitignored (`/roger` in `.gitignore`).

## 3. Corpus Pin

The vectors and judge caches were computed against the corpus at commit `1448a84c`
(2026-05-24). The corpus has since been rewritten, so any new judging must use
the old descriptions:

```bash
mkdir -p /tmp/corpus_2026-05-24
git archive 1448a84c data/traits/instructions data/roles/instructions \
    | tar -x -C /tmp/corpus_2026-05-24
```

Result: 304 traits + 281 roles under `/tmp/corpus_2026-05-24/data/`.
**This lives in `/tmp` and must be recreated after a reboot.**

## 4. Bugs Found and Fixed in `judge_tier_cost_eval.py`

Two schema variants exist in the archived `projections.json` files:

| Format | Top-level structure | Entity name keys | n entities |
|---|---|---|---|
| Old committed (angel/decisive) | `{"result": {"6": {name: ...}}}` | bare names | 571 |
| Archive format A (most axes) | `{"0": {name: ...}, "1": ...}` | bare names | 573 |
| Archive format B (some axes) | `{"result": {"6": {name\|R: ...}}}` | `name\|R` / `name\|T` | 580 |

**Fix 1 — schema variants** (`results_analysis/judge_tier_cost_eval.py:155`):
```python
# Before:
projections_raw = proj_data["result"]
# After:
projections_raw = proj_data.get("result", proj_data)
projections = {int(s): v for s, v in projections_raw.items() if s.lstrip("-").isdigit()}
```

**Fix 2 — disambiguated entity IDs** (the `name|R`/`name|T` keys):
The corpus files are named by bare stem (`patient.json`), not by disambiguated
ID (`patient|R.json`). Added `parse_entity_id` import and stripping logic in
the entity loading loop so `patient|R` → looks up `roles/instructions/patient.json`.

Both fixes are backward-compatible with all three format variants.

## 5. Dry Run and Cost Confirmation

Ran `--dry_run` on two axes (one of each format) to estimate per-model cost:

| Model | Rate (in/out per 1M) | Est. cost/axis (~575 entities) |
|---|---|---|
| `gpt-5.6-luna` | $0.20 / $1.20 | ~$0.11 |
| `gpt-5.6-terra` | $2.00 / $12.00 | ~$1.09 |
| `claude-opus-5` | $5.00 / $25.00 | ~$2.49 |
| **Total per axis** | | **~$3.69** |

**18 new axes × $3.69 ≈ $66–$74 total** (3 models; the 2 existing axes already
have these models' results). Confirmed by user before launch.

## 6. Axis Selection

18 axes chosen as the top-ρ axes from Roger's 35-axis GPT-vs-Sonnet sweep
(`roger/axis_judge_experiments/gpt_vs_sonnet_rhos_di.json`), excluding
angel/demon (ρ=0.863) and decisive/indecisive (ρ=0.738) which were already done:

| Axis | ρ (Roger's GPT+Sonnet blend) |
|---|---|
| trustworthy_vs_untrustworthy | 0.918 |
| constructive_vs_destructive | 0.915 |
| helpful_vs_unhelpful | 0.915 |
| benign_vs_malicious | 0.911 |
| harmless_vs_harmful | 0.890 |
| improvisational_vs_methodical | 0.877 |
| cooperative_vs_competitive | 0.869 |
| reductionist_vs_holistic | 0.864 |
| individualistic_vs_collectivistic | 0.860 |
| passionate_vs_dispassionate | 0.857 |
| compassionate_vs_callous | 0.855 |
| convergent_vs_divergent | 0.851 |
| playful_vs_serious | 0.844 |
| truthful_vs_deceitful | 0.842 |
| practical_vs_theoretical | 0.835 |
| casual_vs_formal | 0.835 |
| generous_vs_stingy | 0.834 |
| introverted_vs_extroverted | 0.833 |

All 18 are trait-type pairs.

## 7. API Key Setup

Keys live outside the repo at
`/Users/bioitx/Documents/Github/dotfiles/tokens/{openai,anthropic}.key`.

A `.env` file at the repo root (gitignored) holds them for `uv run` and the
batch script:

```bash
export OPENAI_API_KEY=$(cat /Users/bioitx/Documents/Github/dotfiles/tokens/openai.key)
export ANTHROPIC_API_KEY=$(cat /Users/bioitx/Documents/Github/dotfiles/tokens/anthropic.key)
printf "OPENAI_API_KEY=%s\nANTHROPIC_API_KEY=%s\n" \
    "$OPENAI_API_KEY" "$ANTHROPIC_API_KEY" \
    > /Users/bioitx/Documents/GitHub/assistant-axis-roger/.env
```

**Note:** shell `export` statements do NOT persist between separate bash-input
blocks in Claude Code — the `.env` file approach is what actually works.
`uv run` auto-loads `.env` at the repo root; the batch script sources it
explicitly at startup.

## 8. New Scripts Written

### `scripts/run_tier_eval_20axes.sh`

Batch loop over the 18 new axes × 3 models. Resume-safe: skips any
`<axis>_tier_eval/<model>/usage.json` that already exists.

```bash
bash scripts/run_tier_eval_20axes.sh 2>&1 | tee /tmp/tier_eval_20axes.log
```

Expected runtime: ~1–2 hours. Expected spend: ~$66–$74.

### `results_analysis/judge_tier_cost_plot_pooled.py`

Pooled Pareto plotter. Reads all `*_tier_eval/` directories under a given
experiments root, aggregates mean ρ and mean cost-per-axis by model, draws
the Pareto frontier (diamond markers for optimal, circle for dominated).

```bash
uv run python results_analysis/judge_tier_cost_plot_pooled.py \
    --experiments_dir roger/axis_judge_experiments \
    --slot 6 \
    --output roger/axis_judge_experiments/pareto_pooled_20axes.png
```

Also auto-detects legacy gpt-4.1-mini and claude-sonnet-4 reference points
from the original `gpt/` and `sonnet/` subdirs (cost estimated, no usage.json).

Pre-run test on the 2 existing axes produced a clean plot confirming:
- Pareto frontier: gpt-5.6-luna → gpt-5.6-sol → claude-opus-5
- gpt-5.6-terra is dominated by luna (same quality, 8× cost over 2 axes)
- claude-sonnet-5 is dominated (lower quality than haiku, higher cost)

This picture should sharpen significantly across 20 axes.

## 9. Status at Continuation Session End (2026-10-26 part 2)

- [x] Archive extracted (61 axes)
- [x] Corpus pinned at `/tmp/corpus_2026-05-24`
- [x] Script bugs fixed
- [x] Dry run confirmed cost for 18-axis batch
- [x] User confirmed spend (~$66–74)
- [x] `.env` file written with API keys
- [x] Batch script written (`scripts/run_tier_eval_20axes.sh`)
- [x] Pooled plotter written and tested (`results_analysis/judge_tier_cost_plot_pooled.py`)
- [x] **20-axis batch completed** (all 54 runs wrote `usage.json`)
- [x] **`pareto_pooled_20axes.png` generated** — Pareto frontier: luna → gpt-4.1-mini → opus
- [x] **Description notes written** — `roger/axis_judge_experiments/pareto_pooled_20axes_notes.md`
- [x] **41-axis batch launched** — `scripts/run_tier_eval_41axes.sh 2>&1 | tee /tmp/tier_eval_41axes.log`
  - 41 remaining axes × 3 models = 123 runs
  - Estimated spend: ~$151 ($3.69/axis)
  - User confirmed spend via "let's go ahead and do the remaining axes"
  - Resume-safe: skips any model with existing `usage.json`

## 11. Key Results (20-axis Pareto)

Across 20 axes (slot 6, description mode):

| Model | Mean ρ | Mean cost/axis | Pareto? |
|---|---|---|---|
| `gpt-5.6-luna` | ~0.627 | $0.10 | ✓ Pareto-optimal |
| `gpt-4.1-mini` (legacy, est.) | ~0.684 | $0.18 | ✓ Pareto-optimal |
| `claude-opus-5` | ~0.701 | $3.18 | ✓ Pareto-optimal |
| `gpt-5.6-terra` | ~0.627 | $1.09 | ✗ Dominated by luna |
| `claude-sonnet-4` (legacy, est.) | ~0.65 | ~$1.20 | ✗ Dominated |

`gpt-4.1-mini` (Roger's existing canonical judge) is Pareto-optimal: no new model
beats it on quality-per-dollar.

## 12. Next Steps After 41-axis Run

1. Verify log at `/tmp/tier_eval_41axes.log` — check all 123 runs wrote `usage.json`.
2. Regenerate the pooled Pareto plot with all 61 axes:
   ```bash
   uv run python results_analysis/judge_tier_cost_plot_pooled.py \
       --experiments_dir roger/axis_judge_experiments \
       --slot 6 \
       --output roger/axis_judge_experiments/pareto_pooled_61axes.png
   ```
3. Update `pareto_pooled_20axes_notes.md` (or write a new `pareto_pooled_61axes_notes.md`)
   with the 61-axis results.
4. Response-mode judging for new models (terra/luna vs gpt-4.1-mini baseline)
   is a separate, more expensive investigation — deferred.
