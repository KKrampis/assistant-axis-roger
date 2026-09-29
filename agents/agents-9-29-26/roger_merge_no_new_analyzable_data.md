# Roger's 2026-09-28 merge: why it doesn't extend the judge-tier cost/quality analysis

## Summary

Roger's upstream `anthropic-vllm-uv` work (a September 2026 corpus expansion,
new pair lists, and a large in-progress axis-judging re-run) was merged into
`personas-konstantinos` on 2026-09-29. It looks substantial (2113 files,
455K+ insertions) but **contributes no data usable to extend the judge-tier
cost-vs-quality analysis** (`results_analysis/judge_tier_cost_eval.py` /
`judge_tier_cost_plot.py`) as of this merge. This doc records why, in terms
of the analysis's two required halves: **judging score** and **geometric
projection**.

## The two halves the analysis needs

Every point on `cost_vs_quality.png` requires both:

1. **Geometric projection** — an entity's activation vector (from running it
   through Qwen-3-32B on GPU) projected onto a fixed axis direction. Stored
   as `<axis>/<judge>/projections.json`. Judge-independent; computed once,
   reused across every judge tier.
2. **Judging score** — an LLM judge's opinion of where the same entity sits
   on the axis (-3..+3 rubric). Stored as `<axis>/<judge>/scores_descriptions.json`.

The analysis is the Spearman correlation (rho) between these two, per judge
tier, per axis. Neither half alone produces a plottable point.

## What we had before the merge

Two axes only — `angel_vs_demon` and `decisive_vs_indecisive` — with both
halves complete:

- `projections.json`: 571 entities × 8 slots × {raw, whitened}, computed
  months ago (last touched by commit `730ff79`, 2026-05-09).
- Two legacy judge scorings already committed (`gpt/` = gpt-4.1-mini,
  `sonnet/` = claude-sonnet-4).

We added six new judge tiers' worth of scores (`claude-opus-5`,
`claude-sonnet-5`, `claude-haiku-4-5`, `gpt-5.6-sol/terra/luna`) for the
*same* 571 entities, correlated against the *same, unchanged*
`projections.json`. No GPU work was needed — only the judging-score half
was added. Result: 8 points per axis, $16.61 total spent.

These two axes are **unmodified by Roger's merge** — no commits touch
`roger/axis_judge_experiments/{angel_vs_demon,decisive_vs_indecisive}/`
since May 2026.

## What Roger's merge actually brings

| Addition | Judging score present? | Geometric projection present? |
|---|---|---|
| Corpus expansion: 588 → 996 entities (traits 308→659, roles 280→337) | No — only `description`/`eval_prompt` text | No — zero activation vectors exist anywhere in this checkout for the 408 new entities (`runpod_workspace/`, where they'd live, is gitignored and empty here) |
| Updated pair lists (`pair_list_clean.json`, `pair_list_di.json`, `pair_list_goalnongoal.json`; pre-expansion versions kept as `_v1`) | N/A (axis-definition manifests, not scores) | N/A |
| ~33 axes with judging API spend recorded (`guardian_vs_destroyer`, `predator_vs_prey`, `symbiont_vs_parasite`, `superficial_vs_thorough`, `aligned_artificial_intelligence_vs_paperclip_maximizer`, `optimistic_vs_pessimistic`, `accessible_vs_esoteric`, and more) | **Only a cost record** — each has a `usage.json` (`n_calls`, `cost_usd`) but no `scores_descriptions.json` was ever committed | **No** — no `projections.json` for any of them |

Verified directly (not inferred from commit messages):

- Every one of the ~33 in-progress axis directories was checked with `find`;
  each contains only `usage.json` file(s), never `projections.json`,
  `correlations.json`, or `scores_descriptions.json`.
- `angel_vs_demon`'s `projections.json` was opened and its slot-0 entity
  count confirmed at exactly 571 (unchanged).
- The 408 new corpus entities were diffed (`comm`) against the 571 judged
  cohort: 561 already existed and are still present, 10 were dropped/renamed
  (e.g. `aligned_artificial_intelligence` → `instrumentally_aligned_ai`),
  435 are genuinely new.
- The pole entities of the ~33 in-progress axes (`guardian`, `destroyer`,
  `predator`, `prey`, `symbiont`, `parasite`, `superficial`, `thorough`,
  etc.) were checked against the pre-expansion corpus list — all of them
  **already existed before the expansion**, i.e. they are not part of the
  408 new entities; these axes are old "never scored" pairs Roger is now
  working through, not part of the corpus-expansion work itself.

## Why none of this is usable yet

- Judging the 408 new entities is technically cheap (their `description`
  text already exists), but doing so produces scores with **no geometric
  projection to correlate against** — nothing to plot, money spent for no
  new chart point.
- The ~33 in-progress axes are missing the geometric half entirely (no
  `projections.json`), and their judging-score half is also unrecoverable
  from what's committed (`usage.json` proves spend happened, not what the
  scores were).
- Computing geometric projections requires GPU-computed activation vectors,
  which are gitignored/not present in this checkout for either the new
  corpus entities or (as far as we can tell locally) the ~33 in-progress
  axes' full cohort.

## Two ways to actually extend the analysis

1. **Wait for Roger** to finish one of the ~33 in-progress axes far enough
   that he commits a `projections.json` for it. Once that lands, the
   existing `judge_tier_cost_eval.py` workflow applies unchanged: ~$8/axis
   for the same six tiers, no GPU work needed on our side (mirrors exactly
   how `angel_vs_demon` / `decisive_vs_indecisive` were done).
2. **Do the geometric half ourselves** for one of those axes — run the
   axis-computation step of the pipeline against the fixed 571-entity
   cohort using the (already-corpus-resident) pole entities. This is
   materially cheaper than regenerating the full expanded corpus, but still
   requires real GPU access (e.g. via `runpod/`) that this checkout doesn't
   have data for locally.

Re-running `judge_tier_cost_eval.py` / `judge_tier_cost_plot.py` today,
unchanged, reproduces exactly the same 12 points (2 axes × 6 tiers) already
committed — there is nothing new to add without one of the two steps above.

## Clarification: patching the script for resume/dedup does NOT unlock the 996-entity corpus

A natural next question is "what if we patch `judge_tier_cost_eval.py` to skip
entities it's already scored (avoid re-paying for the 561 repeats), and just
run it against the full 996-entity corpus?" This doesn't work, because there
are two independent blockers and the resume patch only addresses one:

| Blocker | What it is | Fixed by patching `judge_tier_cost_eval.py`? |
|---|---|---|
| No resume/dedup logic | The script re-scores every entity on every run, even ones already scored — wastes money on repeats | **Yes** |
| No geometric projection for the 435 new entities | The script derives its entity list *directly from `projections.json`* (`entity_names = sorted(any_slot.keys())`) — it can only ever process entities that already have a GPU-computed projection | **No** — a judging-loop patch cannot invent a projection that was never computed; that needs the GPU pipeline (activations → vectors → projections), a different step entirely |

So today, even with a resume patch applied, pointing the script at "the
complete 996" changes nothing: `angel_vs_demon/gpt/projections.json` still
only has 571 keys, and the 435 new entities remain invisible to the script
no matter how its judging loop is patched.

**If** the geometric blocker were separately resolved (someone ran the GPU
pipeline for the 435 new entities and `projections.json` grew to cover all
996), here is what would concretely change in the plots:

- Same shape: still 2 axes × 6 tiers = 12 points, same `$/axis` vs
  `1/(1-rho)` chart.
- `n` goes from 571 → 996 for every point's correlation.
- The rho values would likely shift (up or down) — not because anything got
  "more accurate," but because it becomes a different, larger, differently
  composed sample. It is a new measurement, not a refined version of the old
  one.
- Cost **with** a resume patch: only judge the net-new 435 per tier —
  roughly $6.2–6.4/axis (6 tiers), ~$12.5–12.8 both axes.
- Cost **without** the patch: re-judge all 996, including the 561 already
  scored — roughly $14.3–14.7/axis, ~$28.6–29.3 both axes, of which ~$16.6
  is pure duplicate spend on data already held.

The resume patch is worth doing eventually (it's the difference between
~$12.5 and ~$28.6 for the same end result), but it is not what stands
between us and a 996-entity plot today — the missing GPU/projections data
for the 435 new entities is the actual blocker, and no script patch touches
that.
