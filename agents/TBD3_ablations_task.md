# TBD 3: Ablations of minor technique choices

Task description for Konstantinos (KK), written from the 9 Nov 2026
manuscript discussion with Roger (RD) (transcript 41:54 to 56:10) and from
the code on branch `personas-konstantinos`.

**Owner:** KK (volunteered; RD marked it "maybe not Roger").
**Priority:** P1 (must have before the first paper ships).
**Order:** after TBD 4 (thinking vs speaking), per the meeting's action items.
**RD's estimate:** a couple of hours to get someone up to speed; the analysis
itself is "run it both ways, compare".

## Where the code and docs are

None of the repo's markdown says "TBD 3". The numbered TBD list lives in
Roger's working document for the paper, not in the repo. The code and agent
docs for this task are **not on `master` or on this branch**; they are on
`origin/personas-konstantinos` (which includes the merged PR #3). Check that
branch out, or re-clone once Roger says all his PRs have arrived.

Docs an agent should read first (paths on `personas-konstantinos`):

| File | What it covers |
|---|---|
| `AGENT_NOTES.md` | Canonical agent guide. Hard rules: confirm with Roger before anything over $20 of API spend or 13 GPU-hours. |
| `CLAUDE.md` | Generated subset of `AGENT_NOTES.md` for Claude Code. Do not edit it directly. |
| `.claude/rules/*.md` | Topic rules (judging, judge cost, provenance, NFS IO, plotting) that load automatically. |
| `pipeline/README.md` | Steps 1 to 5, the score=3 filter in step 4, `vectors_unfiltered/`, `scan_missing_vectors.py`. |
| `pipeline/SUBSET_RUNS.md` | Running the pipeline on a subset of roles/traits (`--roles`). |
| `runpod/README.md` | Running on a RunPod GPU. |
| `data/README.md` | `combination_scores.json`, the incongruity scores. |
| `results_analysis/README.md` | The ρ tools, and how the "unfiltered" judge-tier figure was made. |
| `reports/judge_pareto/judge_tier_update_2026-11-01.md` | A first, partial result for ablation (a). See below. |

On this branch, `agents/AGENT_NOTES_SUMMARY.md` is an older (May 2026) copy
of `AGENT_NOTES.md`; it links to files that are not here.

## Background: what ρ is

Each axis is a pair of opposite traits (good/evil, concise/verbose). The
difference of the two poles' mean activations gives a direction at a given
layer, token slot and whitening. Every role/trait vector is projected onto
it, giving a sort order. A judge model scores each item on the same axis,
giving a second sort order. ρ is the Spearman correlation between the two
(the two pole items are left out). With the data sizes involved, ρ is
resolved to about three significant figures, so small differences are
meaningful, but know the noise level before claiming a win.

## The three ablations

The method is the same for each: run the pipeline with the choice on and
off, compute ρ both ways, and compare. RD also asks for a cross-comparison:
ρ of the judging from one run against the embeddings from the other.

Why it is P1: each of these steps is part of Christina Lu's technique (or
our extension) that every other result depends on. If one of them biases the
data, the flag may need to be set the other way and the analyses redone, so
this has to be settled before the first paper.

### (a) Role-play filtering (the score=3 filter)

**What it does.** After the model answers in a persona, a judge (step 3)
scores 0 to 3 whether it played the role. Step 4 builds each role/trait
vector from score=3 responses only, and drops entities with fewer than
`--min_count` (50) of them. Most traits keep 90%+; traits that push against
helpful/harmless/honest training (e.g. harmful) lose many responses to
refusals.

**Why it might be wrong.** The filter subsamples the data according to one
judge's behaviour. The role-play rubric was rewritten in early November (the
old one worked with gpt-4.1-mini, which is being deprecated; newer models
did badly with it), so any bias may also have changed. Refusals may still
carry the persona: the header tokens of "I'm sorry, I can't help with that"
for a harmful persona are plausibly full of "harmful". Keeping all data
may be better.

**Question to answer.** Does ρ go up or down without the filter? Is the
discarded data still infused with the requested persona?

**Code.**
- Filter: `pipeline/4_vectors.py`, `compute_pos_3_vector` (score == 3 and
  `--min_count`). There is no CLI switch to turn it off.
- Unfiltered vectors already exist as a script:
  `scripts/judge_tier_build_unfiltered_vectors.py <may_8slot_dir> <out_dir>`.
  It recomputes each vector as the mean over all responses. It has the
  archive-drive path (`/Volumes/G-Drive/...`) hard-coded, so it needs editing
  to run anywhere else.
- Role-play judge and rubric: `pipeline/3_judge.py` (default judge
  `gpt-4.1-mini`). The rewritten rubric from last week does not appear to be
  on this branch yet (`ROLE_RUBRIC_VERSION` is still a TODO in the code);
  it is likely in Roger's pending PRs.
- ρ: `results_analysis/judge_tier_robustness.py --data_dir <vectors dir>`
  (12 variants: slots 0/6 × layers 25/49 × raw/soft-K/soft-shear), or
  `results_analysis/axis_judge_correlation.py --data_dir ... --layer ... --slot ...`
  for a single axis, and `results_analysis/rho_by_layer_simple.py` for ρ by layer.

**Already done.** The 1 Nov judge-tier report re-ran the description-judging
comparison on unfiltered vectors (61 axes, May 8-slot data). Mean ρ moved by
less than 0.005 and the judge ranking did not change
(`roger/axis_judge_experiments/judge_tier_robustness_61axes_filtered_vs_unfiltered.txt`).
Side finding: on axes whose poles lost the most responses to the filter
(dominant/submissive, harmless/harmful), the judges agree better with the
unfiltered vectors, mostly at slot 0. That is the "persona is still there in
refusals" signal RD described, and is the natural starting point.

**Still to do.** Repeat on the new dataset with the new rubric; include
response judging, which (unlike description judging) depends on which
responses were kept, and is where the cross-comparison matters; look per
axis rather than only at the mean; report ρ by layer and slot.

### (b) Incongruity filtering (role + trait combinations)

**What it does.** Role+trait combinations (30 roles × 30 traits) sometimes
make little sense and are hard to play. Claude Sonnet scored each
combination's instruction pairs 0 to 3 for incongruity
(`data/combination_scores.json`, made by `data_analysis/score_combinations.py`;
16% score 3). Step 3 drops the pairs that scored 3. This only affects the
combination analyses, which are a small part of the paper.

**Question to answer.** Are the combination results better with or without
dropping those pairs? The code's own TODO says it is unclear: many of these
responses are rejected by the role-play judge anyway, so the filter mostly
shrinks the sample.

**Code.**
- Flag: `pipeline/3_judge.py --no_incongruity_filtering` (exclusion logic
  around lines 466 to 490).
- `run_pipeline.sh` does not pass this flag. Run step 3 by hand for
  combinations into `scores_unfiltered/`, then step 4 into
  `vectors_unfiltered/`:

  ```bash
  uv run 3_judge.py --entity_type combination --no_incongruity_filtering \
      --responses_dir <out>/combinations/responses \
      --output_dir    <out>/combinations/scores_unfiltered
  ```

  ```bash
  uv run 4_vectors.py --activations_dir <out>/combinations/activations \
      --scores_dir <out>/combinations/scores_unfiltered \
      --output_dir <out>/combinations/vectors_unfiltered --min_count 50
  ```

  Then check with `scan_missing_vectors.py --root <out> --min_count 50`,
  which pairs `vectors_unfiltered/` with `scores_unfiltered/` automatically.
- Analyses that use combinations: `results_analysis/compute_combo_marginals.py`
  and the combination plots under `results_analysis/canonical_angles/plots/`.

**Naming clash to watch.** "Unfiltered" means two different things in the
repo. `vectors_unfiltered/` (pipeline README, `scan_missing_vectors.py`)
means "no incongruity filter, score=3 filter still on". The judge-tier
"unfiltered vectors" from (a) mean "no score=3 filter". Use distinct
directory names for the two ablations.

### (c) End-think leakage in Qwen non-reasoning mode

**What it does.** Qwen runs in non-reasoning mode: after the assistant
header we pre-fill an empty thinking block (`<think>` `</think>`). Now and
then Qwen thinks anyway and emits a second `</think>`. The current handling
keeps only the text after the last `</think>`. It is rare for most personas
but common for some, such as stream-of-consciousness ones.

**What Claude found earlier (brief).** The removed text looks like thinking;
its mean embedding differs from the kept text, and the difference points
towards traits like pensive and considering alternatives. KK suggested this
could expose pre-verbal content; RD expects it to be an appendix, possibly
not in the first workshop draft. The ablation itself should still be done
before publishing.

**Question to answer.** Does stripping change ρ, and for which personas?
Should the flag be the other way?

**Code.**
- `assistant_axis/generation.py`, around lines 273 to 287: for Qwen, if a
  response has `</think>` but no `<think>`, everything up to the last
  `</think>` is cut before the response is saved.
- `assistant_axis/internals/conversation.py` (two places, around lines 160
  and 480): the token-span logic that skips thinking tokens when extracting
  activations with `enable_thinking=False`.
- There is **no flag** for this on `personas-konstantinos`, and the
  stripped text is not saved, so the "off" arm needs a code change (a flag
  to keep the raw response, or to save the stripped text alongside) and a
  regeneration for at least the affected personas. Ask Roger whether he
  already has this flag in a pending PR; he said "the flags already exist".

## Suggested plan

1. Get the code: check out `personas-konstantinos` (or re-clone after Roger's
   PRs land). Read `AGENT_NOTES.md` hard rules and `pipeline/README.md`.
2. Ask Roger for: which dataset to use (May 8-slot vs the new one due in
   about a week), the role-play rubric version, and whether (c) has a flag.
3. Start with (a), on the data that already exists: rebuild unfiltered
   vectors, run `judge_tier_robustness.py` and `rho_by_layer_simple.py` on
   both, and look per axis at the axes with the biggest drop in kept
   responses.
4. Then (b): one extra step 3 + step 4 run for combinations only (API cost:
   check against the $20 rule first), and recompute the combination analyses.
5. Then (c): add the flag, regenerate the affected personas (GPU cost:
   check against the 13 GPU-hour rule first), compare.
6. For each ablation, write up: ρ with and without, the cross-ρ, the noise
   level, the axes or personas that move most, and a recommendation for
   which way the flag should be set.

## Deliverables

- One short table per ablation (ρ on / off / cross, by slot and layer).
- A recommendation per flag, with the evidence.
- For (a) and (c), a few examples (refusals that still carry the persona;
  leaked thinking text) for a possible appendix.

## Open questions for Roger

- Which dataset: May 8-slot now, or wait for the new one?
- Is the rewritten role-play rubric in his pending PRs? Which version should
  the ablation use?
- Is there a flag for the end-think handling, or should we add one?
- For (b), which combination analyses are in the paper, so we know which
  results to recompute?
