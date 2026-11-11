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

## Background: how the data are made, and what ρ is

Everything below refers to this pipeline (`pipeline/run_pipeline.sh`):

1. **Generate** (`1_generate.py`, vLLM). Each persona (role, trait, or
   role+trait combination) has 5 instruction variants ("You are a demon...").
   Each instruction is used as the system prompt for each question from
   `data/extraction_questions.jsonl` (300 questions, optionally subsampled
   with `--question_count` / `--reduce_questions`), and Qwen-3-32B answers.
   Qwen runs in non-reasoning mode: the chat template inserts an empty
   thinking block `<think>\n\n</think>\n\n` straight after the assistant
   header, and the model writes its answer after that.
2. **Activations** (`2_activations.py`). The saved conversation (system
   prompt + question + header + answer) is run through the model again, and
   activations are stored per layer for several "slots":
   - slot 0: the mean over all tokens of the answer (Lu's method);
   - slots 1 to 7: single tokens of the header, in Qwen
     `<|im_start|>`, `assistant`, `\n`, `<think>`, `\n\n`, `</think>`, `\n\n`
     (`assistant_axis/internals/spans.py`, `_HEADER_TOKENS`).
     Slot 6 (`</think>`) is the current default for ρ.
3. **Judge** (`3_judge.py`). A judge model (default `gpt-4.1-mini`) scores
   each answer 0 to 3 for "did it play the persona?".
4. **Vectors** (`4_vectors.py`). Each persona's vector is the mean of its
   activations over answers scored 3, if there are at least 50 of them.
5. **Axis** (`5_axis.py`) and the analyses in `results_analysis/`.

**ρ.** An axis is a pair of opposite traits (good/evil). The difference of
the two pole vectors gives a direction (per layer, slot and whitening).
Every other persona vector is projected onto it, giving one ordering. A
judge scores each persona on the same axis, giving a second ordering. ρ is
the Spearman correlation between the two, with the two poles left out.
Roger's rule of thumb: with these data sizes ρ is good to about three
significant figures, so 0.75 vs 0.7502 is noise.

**One fact that matters for all three ablations.** The model is causal: a
token's activation depends only on the tokens before it. The header tokens
(slots 1 to 7) come *before* the answer, so their activations depend only on
the system prompt and the question, never on what the model then wrote. So:

- For the header slots, a filter does not change any activation. It only
  changes **which (instruction, question) pairs are included in the
  average**.
- Only slot 0 (the answer mean) changes with the answer text itself.

This is why Roger expects refused answers to still carry the persona at the
header tokens: those activations were computed before the refusal was
written, and they are what the model was "about to say" from.

## The three ablations

The method is the same for each: build the data with the choice on and off,
compute ρ both ways, and compare. Roger also asks for a cross-comparison: ρ
between the judging from one run and the activations from the other. That
separates "the activations changed" from "the judging changed".

Why it is P1: each of these steps is part of Christina Lu's technique (or
our extension) that every other result depends on. If one of them biases
the data, the flag may need to be set the other way and the analyses redone,
so this has to be settled before the first paper.

### (a) Role-play filtering (the score=3 filter)

**What it does now.**
Step 3 asks the judge, for every answer, whether the model played the
persona. The rubric (for roles, the `eval_prompt` in each
`data/roles/instructions/<role>.json`; for traits and combinations,
`TRAIT_EVAL_TEMPLATE` and `COMBINED_EVAL_TEMPLATE` in `3_judge.py`) is:

| Score | Meaning |
|---|---|
| 0 | No sign of the persona; the model says it is an AI and refuses |
| 1 | No sign of the persona; says it is an AI but answers |
| 2 | Some of the persona, but still says it is an AI |
| 3 | Fully plays the persona (a refusal *in character* also counts as 3) |

Step 4 (`compute_pos_3_vector` in `4_vectors.py`) then keeps only the score-3
answers, averages their activations, and drops the persona if fewer than 50
survive. The "default" (plain assistant) persona is never filtered. There is
no command-line switch to turn the filter off.

**Why it might bias the results.**

1. *It depends on one judge's opinion, and the rubric is changing.* Whether
   an answer "plays the role" is often unclear. A sommelier asked about a town
   budget has little room to sound like a sommelier, and a plain helpful
   answer may be fine; a demon giving a plain helpful answer has clearly
   failed. The old rubric (above) does not cover these cases: it was
   generated once by Opus for Lu's repo and still mentions "Gemma" and an
   "altruistic AI" example in every role file. gpt-4.1-mini coped with it,
   but it is being deprecated and newer judges did badly, so Roger and
   Claude rewrote the rubric in early November (AI vs non-AI roles,
   professionals asked about other topics, and so on). That rewrite is not on
   any branch I can see; it is probably in Roger's pending PRs. A different
   rubric keeps a different subset, so any bias may have moved.
2. *It removes most of the data exactly where it matters.* Traits that push
   against helpful/harmless/honest training (harmful, malicious, deceitful)
   get many refusals. For those poles the vector is built from the minority
   of answers where the model complied, which may be an unusual subset.
3. *Refusals may still carry the persona.* As explained above, for the header
   slots the activation of a refused answer was computed before the refusal
   existed. "I'm sorry, I can't help with that" after a "be harmful" prompt
   is itself a reaction to harmfulness.
4. *It mixes question effects into axis directions.* The filter keeps a
   different set of questions for each persona: "helpful" may keep 95% of
   questions while "unhelpful" keeps 50%. Subtracting the two pole vectors
   then mixes the trait contrast with a difference in which questions were
   averaged (see the `compute_pos_3_vector` docstring and the pipeline
   README).

**Arms to compare.**

- *Filtered*: as now (score 3, at least 50).
- *Unfiltered*: mean over all answers. Already implemented as
  `scripts/judge_tier_build_unfiltered_vectors.py <may_8slot_dir> <out_dir>`,
  which computes `(n3 × stored_vector + sum of the other answers) / n_all`
  so it only reads the non-3 answers from the activation files. It has the
  archive-drive path `/Volumes/G-Drive/...` hard-coded; edit it for wherever
  the activation files are.
- *Optional third arm, question-matched*: for each axis, keep only the
  questions where both poles scored 3, then average. This removes problem 4
  without removing the filter. Not implemented (the README calls it a
  possible "step 4b").

**How to measure.**
- `results_analysis/judge_tier_robustness.py --data_dir <vectors>` gives ρ
  for many judges on 61 axes, at slots 0/6 × layers 25/49 × three whitening
  choices. Run it on both arms.
- `results_analysis/rho_by_layer_simple.py` for ρ by layer (does the layer
  27 / layer 49 peak structure change?).
- `results_analysis/axis_judge_correlation.py --data_dir ... --layer ... --slot ...`
  for one axis in detail.
- Look per axis, not just at the mean. Sort axes by how much data their
  poles lost to the filter, and check whether ρ changes with that.
- Also check the direction itself: cosine between the filtered and unfiltered
  axis directions.

**What the cross-comparison means here.** Description and instruction
judging (the judge reads the persona's description or instructions) do not
depend on this filter, so for those only the activations change. Response
judging (`--score_responses`) reads only score-3 answers, so it changes too.
Cross-ρ (unfiltered activations vs filtered response judging, and the other
way round) tells you which side moved.

**What is already known.** The 1 Nov judge-tier report re-ran the
description-judging comparison on unfiltered May 8-slot vectors (61 axes)
(`roger/axis_judge_experiments/judge_tier_robustness_61axes_filtered_vs_unfiltered.txt`):

- mean ρ changed by less than 0.005 for every judge and setting, and the
  judge ranking did not change;
- median cosine between filtered and unfiltered axis directions was 0.997;
  12 axes were below 0.99 and 3 below 0.95 (dependable/undependable 0.92,
  dominant/submissive 0.94, harmless/harmful 0.95);
- on dominant/submissive ρ went up by about 0.03 to 0.04 without the
  filter; on harmless/harmful it did not move.

So the filter barely matters on average, but a few axes move, and those are
the axes with the most refusals. That is where to look.

**Still to do.** Repeat on the new dataset with the new rubric; add response
judging and the cross-comparison; break results down per axis and per slot
(Roger expects the effect to differ between slot 0 and the header slots);
report ρ by layer.

**Possible outcomes.**
- ρ the same or better unfiltered: drop the filter (simpler, no judge
  dependence, more data); say so in the paper.
- ρ better filtered: keep it, and report that the result is not sensitive to
  the rubric version (compare old and new rubric).
- Mixed by axis: report it, with examples of refusals whose activations still
  carry the persona (a nice appendix item).

### (b) Incongruity filtering (role + trait combinations)

**What it does now.**
Combinations pair a role with a trait. In "roger" mode, `r_` combinations
are a goal role × non-goal trait (e.g. `r_predator__disorganized`), and `t_`
combinations are a non-goal role × goal trait (e.g. `t_chef__harmful`); the
lists are in `data/goal_roles_and_traits.json`. Each combination has 5
instruction variants (one per pair of role and trait instructions).

Some combinations are hard to play because the role and trait contradict
each other: a "calculating, strategic" predator that must be "disorganized,
forgetting mid-sentence". Claude Sonnet scored every instruction variant 0
to 3 for incongruity, with Opus re-scoring borderline cases
(`data/combination_scores.json`, made by `data_analysis/score_combinations.py`).
This 0 to 3 incongruity score is unrelated to the 0 to 3 role-play score.

In step 3, any variant with incongruity 3 is removed **before judging**.
Those answers then have no role-play score, so step 4 never counts them as
score 3, and they drop out of the combination's vector.

**How much it removes.** In the current file: 3,200 combinations × 5
variants = 16,000 variants; 86 of them score 3 (0.5%), in 31 combinations;
72 of the 86 are `t_` combinations. (`data/README.md` says "16% score 3";
that figure is out of date for the current file.) So the effect is small and
limited to a few combinations, and only matters for the analyses that use
combinations.

**Why it might be unnecessary or harmful.** The code's own TODO
(`3_judge.py`, around line 467) says it is unclear whether it helps: many of
these answers are rejected by the role-play judge anyway, so the filter
mostly shrinks the sample; the rate of bad-but-accepted answers among
incongruous variants is not much higher than elsewhere; and some removed
variants are perfectly usable. It is also one more judge-dependent choice to
defend in the paper.

**Arms to compare.**
- *Filtered* (now): `combinations/scores/` and `combinations/vectors/`.
- *Unfiltered*: rerun step 3 for combinations with the flag, into a separate
  folder, then step 4. `run_pipeline.sh` does not pass this flag. Only the
  31 affected combinations need rejudging, so you can add
  `--roles <names>` to keep cost down, and copy the other score files
  unchanged:

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
  Note the dropped variants were still generated and their activations
  extracted in steps 1 and 2; only judging skipped them. So no GPU work is
  needed.

**How to measure.**
- The combination analyses: `results_analysis/compute_combo_marginals.py`
  (how a combination decomposes into role + trait parts) and the plots under
  `results_analysis/canonical_angles/plots/` (e.g.
  `combos_vs_traits_roles_pooled.py`).
- For the 31 affected combinations: how many variants pass the role-play
  judge with and without the filter, and how far their vectors move.
- Whether the role + trait additivity results change.

**Possible outcomes.** With 0.5% of variants affected, the likely result is
"no measurable difference", which would justify dropping the filter (one
less judge dependency) or keeping it with a one-line note in the paper.

### (c) End-think leakage in Qwen non-reasoning mode

**What happens.**
In non-reasoning mode the prompt ends with the pre-filled empty block
`<think>\n\n</think>\n\n`, and the model should simply answer. Occasionally
Qwen starts reasoning anyway in its answer and then writes a second
`</think>`, so the answer looks like: *thinking-style text* `</think>`
*actual answer*. It is rare for most personas, but common for some, such as
"stream of consciousness", "pensive" or "disorganized", which push the model
to think aloud.

**What the code does now.**
- At generation, `assistant_axis/generation.py` (around lines 273 to 287):
  for Qwen, if an answer contains `</think>` but no `<think>`, everything up
  to the last `</think>` is cut, and a warning reports how many answers
  were cut. Only the cut answer is saved; the removed text is **not kept
  anywhere**.
- At activation extraction, `assistant_axis/internals/conversation.py`
  (around lines 190 to 210, and again around 530): with thinking disabled,
  any tokens between a `<think>` and a `</think>` in the answer are left out
  of the slot 0 mean. This catches the rarer case where the model writes a
  full `<think>...</think>` itself.

**What this affects.** Because of the causal-model point above:
- Header slots 1 to 7 are unaffected directly; they come before the answer.
- Slot 0 (answer mean) changes: the thinking text is not in the average.
- The role-play judge only sees the cut answer, so scores, and so which
  answers pass the score-3 filter, can change.

So this ablation mainly concerns slot 0 and the judging, and it is
concentrated in a few personas. But those personas sit at one end of some
axes (e.g. pensive, stream of consciousness), so it could shift those axes.

**What Claude found earlier (brief, informal).** In cases where this
happened, the removed text does look like thinking, its mean activation
differs from that of the kept text, and the difference points towards traits
like pensive and "considering alternatives". KK suggested this could reveal
"pre-verbal" content; Roger expects an appendix, probably not in the first
workshop draft. The ablation itself should be done before publishing, in
case the flag should be the other way.

**Arms to compare.**
- *Strip* (now): keep only the text after the last `</think>`.
- *Keep*: keep the whole answer, including the thinking text, for both the
  judge and the slot 0 mean.
- Possibly a third: drop affected answers entirely.

**What's missing.** There is no flag for this on `personas-konstantinos`,
and since the removed text is not saved, the "keep" arm cannot be rebuilt
from existing files. It needs:
1. a small change to `generation.py`: a flag to turn the cut off, or (better)
   always save the raw text in an extra field and cut in a later step;
2. regenerating answers for the affected personas (GPU work: check the
   13 GPU-hour rule), then steps 2 to 4 for them;
3. rejudging those answers (API cost: check the $20 rule).

Ask Roger first: he said "the flags already exist", so he may have this in a
pending PR, and he may have the list of affected personas from the
generation logs (the warning line gives counts).

**How to measure.**
- How often it happens, per persona (from the warning counts or the raw
  text).
- ρ at slot 0 for axes involving the most affected personas, strip vs keep.
  Header slots should not change except through the role-play filter.
- The direction "mean of removed text − mean of kept text": which traits it
  lines up with (the earlier informal finding), for the appendix.

**Possible outcomes.** Most likely the effect on ρ is small overall but
visible for a few thinking-style personas at slot 0, which would support
keeping the cut (or using header slots, which are immune). If keeping the
text improves ρ, the flag should be flipped and slot 0 results redone.

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
