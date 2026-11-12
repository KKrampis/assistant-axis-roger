# TBD 3 (ablations): discussion context

Export of a Claude Code session (11 Nov 2026) with Konstantinos about TBD 3,
which the session also called "TB3". Sources:
- the three uploaded PDFs: *Working Notes for Our First Paper*, the 9 Nov
  2026 *Manuscript discussion* transcript with Roger, and its 5-page notes
- the repo branches
- Christina Lu's paper (arXiv 2601.10387)

Companion file: `agents/TBD3_plan.md` (the short action plan).

**Claude Code picking this up:** read the whole file, then follow **§14**
(hand-off: pipeline, tasks, correctness checks, results format). §13
explains the cross-ρ; §15 the end-think arms and the TBD 4 link.

---

## 1. Where things are

**Branches:**
- **`master` / `claude/jolly-mendel-7xsvsr`**: essentially Lu's original
  pipeline plus agent notes. **Not** where Roger's work is. An early answer
  in this session was based on this stale checkout and wrongly said there
  were no CLAUDE.md files or flags.
- **`origin/personas-konstantinos`**: Roger's full codebase, with PR #3
  merged on 6 Nov. Has `CLAUDE.md`, `AGENT_NOTES.md`, `.claude/rules/*.md`,
  the pipeline flags, `results_analysis/` and `scripts/`. Work from here.
- **`origin/claude/determined-rubin-rcd5m9`**: another session's long TBD 3
  reference, `agents/TBD3_ablations_task.md`, with full commands per
  ablation.

None of the repo's markdown uses the label "TBD 3". The numbering lives only
in the paper notes.

**Docs relevant to TBD 3** (on `personas-konstantinos`):

| File | Why it matters |
|---|---|
| `CLAUDE.md` / `AGENT_NOTES.md` | Hard rule: confirm with Roger before anything over **$20 API** or **13 GPU-hours**, quoting each non-default flag |
| `.claude/rules/judging.md`, `judge-cost.md` | Judging conventions and costs |
| `pipeline/README.md` | Steps 1–5, the score=3 filter, `vectors_unfiltered/`, `scan_missing_vectors.py` |
| `data/README.md` | Persona file formats, `combination_scores.json`, goal/non-goal lists |
| `roger/axis_judge_experiments/judge_tier_robustness_61axes_filtered_vs_unfiltered.txt` | Roger's first result for ablation (a) |
| `reports/judge_pareto/judge_tier_update_2026-11-01.md` | The report around that result |

---

## 2. What TBD 3 is

From the Working Notes:

> TBD 3: Is the RP filtering at Pipeline Step 3 even a good thing, or is it
> just generating artifacts? Ditto for incongruity filtering and the leaked
> think stripping: ablations. Konstantinos?, P1, ~1 week

The meeting covered it at 41:54–56:10.

**The question.** The pipeline throws data away at three points. Does each
filter help, or does it bias the results?

**The method.** For each filter, build the data with it on and with it off,
then compare ρ. Roger also wants a cross-comparison: judging from one arm
against activations from the other, to see which side moved.

**Why it's P1.** "If that piece of our technique is wrong, it could be
biasing all the rest of our results." It has to be checked before the first
paper, in case the flag should be set the other way.

**Order.** Konstantinos does TBD 4 (thinking vs speaking) first, then TBD 3.
Roger thinks it's good onboarding: "a couple of hours" to get up to speed.

**ρ** is the Spearman correlation between two orderings of the personas on
an axis:
1. the order from projecting each persona's vector onto the axis
2. the order from an LLM judge ranking the personas on that axis

Roger's rule of thumb: ρ is good to about 3 significant figures, so
differences under about 0.005 are noise.

**Slots** are the token positions where activations are read:
- **Slot 0**: the average over the whole answer, as Lu did.
- **Slots 1–7**: single tokens of the Qwen response header: `<|im_start|>`,
  `assistant`, `\n`, `<think>`, `\n\n`, `</think>`, `\n\n`.
- **Slot 6** (`</think>`) is the current default.

Roger (25:40): a single header token is "at least as good and in some
respects even a little better" than the answer average, because "the header
is a more concentrated section of thinking about what persona am I about to
speak in". Header tokens come **before** the answer, so a filter changes
only which answers get averaged, not their activations. Slot 0 is affected
more directly.

---

## 3. The three filters: what they are and where they came from

Git history shows that **only (a) is Lu's**. Roger added (b) in April 2026
and (c) on 6 April 2026.

### (a) Role-play (RP) filtering: Lu's idea; Roger extended where it's used

**Process**, using "pirate" as the example:
1. **Ask.** System prompt "You are a pirate…" (5 variants) × the shared
   question list → **500 answers** in Roger's May data (5 × 100 questions;
   see §8). Lu: 1,200.
2. **Record** the activations.
3. **Judge.** gpt-4.1-mini scores each answer 0–3:
   - 3 = fully in role
   - 2 = some role flavour, but says it's an AI
   - 1 = says it's an AI, but helps
   - 0 = refused
4. **Filter.** `pipeline/4_vectors.py` keeps only the score-3 answers.
5. **Average.** One vector per persona. No vector if fewer than 50 answers
   scored 3.

**Outputs:**
- `scores/<persona>.json`: a score for every answer
- `vectors/<persona>.pt`: mean activation over the score-3 answers. Each
  vector holds **8 slots × 64 layers**, about 5.2 MB for the 8-slot data.

**From Lu:**
- the 0–3 judging idea, gpt-4.1-mini as judge, per-role eval prompts
- the default Assistant is never filtered

Caveat: **Lu's paper** kept score-3 and score-2 answers as *separate*
vectors, with a minimum of 10. **Her released code** keeps score 3 only,
with a minimum of 50, and we use the code's version. Her paper never
justifies the filter or discusses its biases.

**From Roger:**
- applying the filter to **traits** (each pole is its own persona) and to
  **combinations**
- header-token slots
- the rubric rewrite: gpt-4.1-mini is being deprecated, and newer judges
  failed on the old rubric; it was rewritten in early November and is
  probably in a pending PR
- the unfiltered-vector script and the first comparison

**Why it might bias results:**
1. Harmful-type traits get many refusals, so their vectors rest on the few
   compliant answers.
2. Refused answers may still carry the persona at the header tokens, which
   are computed before the refusal is written.
3. The result depends on the judge and rubric.
4. Each persona keeps a different set of questions, which mixes question
   effects into the axis.

**Roger's first run, on 61 axes (1–2 Nov 2026, committed):**
- He built unfiltered vectors with
  `scripts/judge_tier_build_unfiltered_vectors.py` into
  `qwen-3-32b Roger 8slot unfiltered/`.
- He ran `results_analysis/judge_tier_robustness.py` on both arms: 9 judges ×
  61 axes × 12 settings. The 12 settings are slots 0/6 × layers 25/49 ×
  raw / soft-K2 / shear-3.
- Results are in `judge_tier_robustness_61axes.json` and
  `…_61axes_unfiltered.json`, with ρ for every judge × axis × setting.
- Mean ρ moved less than about 0.005 everywhere, and the judge ranking was
  unchanged.
- Axis directions barely moved: median cosine 0.997.
- The axes that moved most are the high-refusal ones:
  - dependable/undependable: cosine 0.92, ρ +0.03
  - dominant/submissive: cosine 0.94, ρ +0.04
  - harmless/harmful: cosine 0.95, ρ unchanged

**Left to do for (a):**
- per-axis analysis on all 61 axes × 12 settings, which needs only the
  committed JSONs
- relate the changes to how much each pole lost to the filter
  (`unfiltered_stats.jsonl`)
- ρ at all layers and slots (`rho_by_layer.py`; needs the vectors)
- response-judging cross-ρ, which costs API money
- repeat on the new dataset and rubric
- optionally, a "score 3 + score 2" arm to match Lu's paper

### (b) Incongruity filtering: Roger's, not in Lu at all

Lu has no role + trait combinations.

**What combinations are:**
- Roles and traits are labelled "goal" or "non-goal" in
  `data/goal_roles_and_traits.json`.
- `r_` = goal role × non-goal trait, e.g. `r_predator__disorganized`.
- `t_` = non-goal role × goal trait, e.g. `t_chef__harmful`.
- The 5 role prompts and 5 trait prompts are joined by index with a newline.
- In the May run: 30 × 30 = **900 `r_` + ~900 `t_`**.

**The filter:**
- Sonnet, with Opus for borderline cases, scored each prompt variant 0–3 for
  how contradictory the pairing is (`data/combination_scores.json`, which
  covers 40 × 40 per side = 3,200 combinations).
- Variants scored 3 are skipped **before** judging (`pipeline/3_judge.py`).
- Flag: `--no_incongruity_filtering`.
- Roger's TODO at `3_judge.py:467` doubts it helps.

**Size:** 86 of 16,000 variants in 31 combinations across the full file. The
~1,800 combinations actually run contain fewer.

**Status:**
- Roger's cache audits show `combinations/scores_unfiltered/` and
  `combinations/vectors_unfiltered/` already exist in the May dataset.
- So only the comparison is missing: `scan_missing_vectors.py` and
  `compute_combo_marginals.py` on both arms. No GPU or API cost.
- Expected result: "no difference".

### (c) Leaked end-think handling: Roger's; Lu says only "we disabled thinking mode"

**Mechanism** (Roger, 51:15–56:00):
1. Qwen runs in non-reasoning mode. The prompt is pre-filled with the system
   prompt, the user prompt, the assistant header, and an empty
   `<think></think>` block.
2. Qwen then answers: "I've already thought… and my thoughts are blank."
3. Occasionally it thinks anyway mid-answer and emits a stray `</think>`.

**Current handling:**
- `assistant_axis/generation.py` (around line 275) drops everything before
  the last `</think>` and logs "Stripped orphaned thinking content from N/M
  responses". The cut text is **not saved**.
- Lu's code only skips complete `<think>…</think>` blocks during activation
  extraction (`internals/conversation.py`).

**Who it affects:** rare for most personas, common for "stream of
consciousness", "pensive" and similar.

**Konstantinos's question (53:31):** how do we know it's really thinking?
Roger: all we detect is the token. But a brief Claude analysis found that:
- the removed text looks like thinking
- its mean activation differs from the kept text's
- the difference points toward "pensive" and "considering alternatives"

**Where it goes in the paper:**
- Konstantinos: "could be a short paper by itself", maybe a way to see
  pre-verbalised, hidden content.
- Roger: probably an appendix, maybe not in the first workshop draft. But do
  the ablation before publishing in case the flag must be flipped.

**What it affects:** slot 0 and the judge's view of the answer, and through
the judge, which answers survive filter (a). Header slots are unaffected
directly.

**Status:** not started. There is **no flag** despite "the flags already
exist". It needs:
1. a `generation.py` change to save the raw text
2. regenerating the affected personas (GPU)
3. re-judging (API)

Ask Roger first.

---

## 4. Data: what's in git and what isn't

| Item | Lu (`master`) | Roger (`personas-konstantinos`) | In git? |
|---|---|---|---|
| Shared questions (`data/extraction_questions.jsonl`) | 240 | 300 (expanded 6 Apr 2026) | yes |
| Roles (`data/roles/instructions/`) | 275 + default | 337 + default | yes |
| Traits (`data/traits/instructions/`) | 240 | 659 | yes |
| Incongruity scores, goal lists | — | `combination_scores.json`, `goal_roles_and_traits.json` | yes |
| Judge rankings for ρ | — | ~9,700 `scores_descriptions.json` under `roger/axis_judge_experiments/` | yes |
| 61-axis ρ results | — | `judge_tier_robustness_61axes*.json/.txt` | yes |
| Responses, scores, **activations**, **vectors** | — | `runpod_workspace/qwen/qwen-3-32b Roger 8slot/` (+ `… unfiltered/`) | **no** (git-ignored; GBs) |

Notes:
- Each persona file also has about 40 of its own questions, inherited from
  Lu's format. The pipeline doesn't use them; it uses the shared list.
- **Persona formats:**
  - Lu's role: 5 `pos` prompts.
  - Lu's trait: 5 `pos`/`neg` pairs in one file; the vector is pos minus neg,
    with a different judge on a 0–100 scale.
  - Roger's traits: each pole is its own file (`helpful.json`,
    `unhelpful.json`), with `positive_label`/`negative_label`. An axis is one
    pole's vector minus the other's, and both are judged 0–3.
- **May "8-slot" dataset:** in the transcript (1:50), Roger said it's
  "checked in, or there are links to it". Most of it, apart from the
  activations, is on the shared Google Drive. Konstantinos said he had
  downloaded it. The transcript is garbled about whether the vectors are on
  the Drive. Check the local copy for `vectors/` and `activations/`.
  - Ablation (a) re-analysis needs the **vectors** (both arms).
  - Rebuilding the unfiltered vectors needs the **activations**, which are
    on Roger's G-Drive; the script hard-codes `/Volumes/G-Drive/...`.
  - Ablation (b) needs only scores and vectors.

---

## 5. Suggested order and open questions

**Order:**
1. Get the May 8-slot data and check out `personas-konstantinos`.
2. **(b)**: comparison only.
3. **(a)**: per-axis and all layers/slots on existing data; then the costed
   cross-ρ after Roger confirms.
4. **(c)**: after Roger answers about a flag; get the GPU-hours confirmed.
5. Per filter, write up: an on/off/cross ρ table, the axes or personas that
   move, and a recommendation for the flag.

**Questions for Roger:**
1. Access to the May 8-slot dataset (vectors, activations) and to the new
   one, due in about a week.
2. Is there a flag for the end-think cut? Which personas leak most?
3. Where is the rewritten RP rubric, and which version should we use?
4. Which combination analyses go in the paper?
5. Our code keeps score 3 only with a minimum of 50; Lu's paper kept 3 and 2
   separately with a minimum of 10. How do we describe this in the paper?
6. For the cross-ρ: description judging (fixed, free) or response judging
   (costs API money)?
7. Where is the AGENT_NOTES TODO about the unfiltered side finding?

---

## 7. What the transcript actually asks for vs. suggestions made in this session

**Stated in the 9 Nov meeting (41:54–56:10) and its notes:**
- (a), (b), (c): "Run the pipeline this way, run the pipeline [the other
  way], compare the two sets of results, see which one has a higher ρ."
- "Maybe compare the ρ between the judging from one against the embeddings
  from the other": the cross-ρ.
- For (a), Roger's framing (47:31): ρ against the judging shows whether the
  discarded data is "still infused with the persona we asked for". The
  rubric rewrite may have changed the bias.
- For (b): "only affects… what happens when you combine a role and a trait,
  which is not a big piece of our research."
- For (c): do it before publishing in case the flag must flip. The write-up
  is probably an appendix, maybe not in the first workshop draft.

**Action items from the notes:**
- **KK:** TBD 4 first (May 8-slot data), then TBD 3. Start by Wednesday;
  use Slack or pair programming when stuck.
- **RD:**
  - send pointers: the rubric to change, the script that generates the file,
    the column and flag names, a two-paragraph summary
  - finish the new dataset (~1 week) and redo the pipeline run
- **Wednesday meeting:** names and estimates.

**Suggestions made in this session, not discussed in the meeting:**
- Per-axis analysis on all 61 axes × 12 settings, and relating each axis's
  change to how many answers its poles lost to the filter. This is
  supported by Roger's report side finding (below) but is not a meeting
  decision.
- ρ at every layer and slot. "ρ by layer" was discussed for TBD 4, not
  TBD 3.

**Not discussed: which judging the cross-ρ uses.** Description judging does
not change between arms, so a cross-ρ only means something with **response
judging**, where the judge reads the kept answers. That costs API money
($20 rule). Ask Roger which he meant.

---

## 8. Questions vs personas, and answer counts

Questions and personas are separate lists. Lu's "240 questions" and "240
traits" are a coincidence.

| | Lu | Roger |
|---|---|---|
| Shared questions (`data/extraction_questions.jsonl`) | 240 | 300 (+60 on 6 Apr 2026) |
| Questions used per persona | all 240 | every 3rd (`--reduce_questions 3`, the roger-mode default) = 100 |
| Prompt variants per persona | 5 | 5 |
| **Answers per persona** | 1,200 | **500** (confirmed by Roger's report: "all 500 responses") |

**Per-file questions.** Each persona file's ~40 `questions` come from Lu's
format. Her appendix describes 40 per role, while her main text says one
shared 240 list; the paper is inconsistent. Both pipelines use only the
shared list.

**Persona counts:**

| | Lu | Roger, defined now | Roger, in the May data |
|---|---|---|---|
| Roles | 275 + default | 337 + default | ~280 |
| Traits | 240 | 659 | ~300 |
| Combinations | 0 | — | 900 `r_` + ~900 `t_` |

Most of the extra traits came from a September expansion and have no
answers or activations yet.

---

## 9. Slots in detail

Each vector file has shape **(8 slots, 64 layers, 5,120)**
(`assistant_axis/internals/spans.py`).

| Slot | Qwen token |
|---|---|
| 0 | mean over all answer tokens (Lu's method) |
| 1 | `<|im_start|>` |
| 2 | `assistant` |
| 3 | `\n` |
| 4 | `<think>` |
| 5 | `\n\n` (the empty thoughts) |
| 6 | `</think>` (**default**) |
| 7 | `\n\n` (last token before the answer) |

**History** (transcript 1:49): the first extraction had slot 0 plus slots
1–3, the "4-slot" data. The 4 thinking-block tokens were added later,
giving the "8-slot" May data.

**Other models' header tokens** (no thinking block):
- Gemma: `<start_of_turn>`, `model`, `\n` (3)
- Llama: `<|start_header_id|>`, `assistant`, `<|end_header_id|>`, `\n\n` (4)

**Why header tokens matter.** They are computed before any answer word: the
model is deciding "which persona am I about to speak as". At the header
tokens, ρ jumps around layer 27, with a secondary peak near layer 49 on some
plots (23:32).

**Wording in the paper.** "Slot" is Claude's shorthand. In the paper, write
"mean over response tokens" and "the `</think>` token".

---

## 10. The two judges, and how Roger's 61-axis check works

**Two different judges:**

| | Role-play judge | Axis judge |
|---|---|---|
| Scale | 0 to 3 | −3 to +3 |
| When | pipeline step 3 | analysis, for ρ |
| Reads | one answer at a time | each persona's **description**, no answers ("description judging") |
| Question | did the model play the persona? | where does this persona sit on this axis? |
| Used for | filtering (keep 3s): TBD 3 (a) | the judge's ordering for ρ |

**How the 61-axis check works** (`results_analysis/judge_tier_robustness.py`).
It was a side check in Roger's judge-selection work ("was gpt-4.1-mini
helped by the vectors?", `judge_tier_update_2026-11-01.md` §4), not the
TBD 3 ablation itself. For each judge × setting × axis:
1. Take ~560 personas, excluding the axis's two poles.
2. Project each persona's vector, after subtracting the default Assistant
   vector, onto the axis (pos pole − neg pole). That gives ordering 1.
3. The axis judge's −3..+3 description scores give ordering 2.
4. ρ = Spearman correlation between them: one number per axis.
5. **Mean ρ** = plain average over the 61 axes. It is a mean across axes,
   not across orderings, used to summarise each judge × setting in one
   number.

The judge scores are identical in the filtered and unfiltered runs; only the
vectors change. The weakness of the mean is that it can hide a few axes that
move.

**"Axis direction moved".** Each axis is an arrow (pos − neg). Filtering
changes which answers feed each pole, so the filtered and unfiltered arrows
can differ. Measured by cosine at slot 6, layer 25:

| cosine | angle | example |
|---|---|---|
| 0.997 | ~4° | median axis |
| 0.95 | ~18° | harmless/harmful |
| 0.92 | ~23° | dependable/undependable |

Only the 10 most-moved axes are listed.

**Report side finding.** On axes whose poles lost most answers
(dominant/submissive, harmless/harmful), every judge agrees better with the
unfiltered vectors, most clearly at slot 0. That suggests the persona is in
the activations even when the answer refuses. The report calls this "a TODO
in AGENT_NOTES", but no such TODO is in the committed `AGENT_NOTES.md`; it
may be in a pending PR.

**The 12 settings** = slot (0, 6) × layer (25, 49) × 3 regimes
(`results_analysis/canonical_angles/whitening.py`; analysis only, steering
uses raw):
- **raw:** no change.
- **soft-K, K=2:** the top 2 dominant directions across all persona vectors
  (roughly "Assistant-likeness", Lu's PC1, and the next one) are shrunk to
  the size of the 3rd, so they don't swamp projections. A "light touch"
  whitening.
- **soft-shear, L=3:** takes the goal-persona subspace and the
  non-goal-persona subspace, and gently skews space so their top 3 shared
  directions become perpendicular. This separates "what the persona wants"
  from "how it behaves". Volume-preserving.

K=2 and L=3 were chosen by Roger's sweeps in May 2026.

**Data needs:**
- Every vector already holds all 64 layers × 8 slots, so ρ at any layer or
  slot needs no new judging.
- The description-judge scores are in git: ~9,700 `scores_descriptions.json`
  under `roger/axis_judge_experiments/`.
- The vectors are not in git; get them from the local / Drive copy.

---

## 11. Roger's three changes, and how combinations are built

**(1) Each trait pole is its own persona.**
- **Lu:** one file per trait with pos/neg prompt pairs; vector = pos − neg;
  0–100 judge; a trait is kept if ≥10 prompt pairs differ by ≥50.
- **Roger:** `helpful.json` and `unhelpful.json` are separate personas
  (5 `pos` prompts each, linked by `positive_label`/`negative_label`),
  judged 0–3 like roles. Axis = pole vector − pole vector.
- **Gain:** every pole is a point in persona space, so ρ runs over ~560
  personas.
- **Cost:** the role-play filter now hits the harmful-type poles, which is
  TBD 3 (a).

**(2) Role + trait combinations** (`generation.py`,
`generate_combined_responses`).
- Prompt *i* = role prompt *i* + newline + trait prompt *i*, for *i* = 1…5.
- Example, `t_chef__greedy` prompt 1 (current files):
  > You are a chef who builds dishes from the bottom up, knows what heat
  > does to protein, and won't put anything on the plate that isn't ready.
  > You are greedy and always prioritize accumulating more resources,
  > wealth, or opportunities for yourself, even at others' expense.
- Then it goes through the pipeline like any persona:
  1. 500 answers
  2. activations
  3. `COMBINED_EVAL_TEMPLATE` in `3_judge.py`: a 0–3 score where 3 needs
     **both** role and trait without identifying as an AI, and showing only
     one caps the score at 2
  4. vector from the score-3 answers
- The maths comes later: `compute_combo_marginals.py` tests whether
  chef+greedy ≈ chef + greedy.
- Prompts were partly regenerated in September, so the May wording may
  differ.

**(3) Goal / non-goal split** (`data/goal_roles_and_traits.json`, from
`data_analysis/classify_goals.py`).
- "goal" = all 5 prompts clearly describe a goal; "non-goal" = none do;
  ambiguous personas are left out.
- Lists: 40 goal / 70 non-goal roles; 66 goal / 57 non-goal traits. The May
  run used 30 of each.
- Each combination pairs one goal with one non-goal persona:
  - `r_` = goal role + non-goal trait (`r_predator__disorganized`)
  - `t_` = non-goal role + goal trait (`t_chef__greedy`)
- The split is also used by soft-shear.

---

## 12. Incongruity filter: exact mechanics, and where the axis lists live

**Incongruity filter** (`pipeline/3_judge.py`, ~lines 467–490 and 594):
1. Once, `data_analysis/score_combinations.py` has Sonnet (Opus for
   borderline cases) score each combination's 5 prompt variants 0–3 for
   contradiction. Saved in `data/combination_scores.json` (3,200
   combinations).
2. Steps 1–2 ignore it: every variant is generated and its activations
   extracted.
3. Step 3, unless `--no_incongruity_filtering` is passed: any variant scored
   3 has all its answers (that `prompt_index`, ~100 of 500) removed before
   the role-play judge, and a line is logged.
4. Step 4: those answers have no score, so they are excluded from the
   vector. The combination loses its vector only if fewer than 50 score-3
   answers remain.

The arms are `scores/` → `vectors/` and `scores_unfiltered/` →
`vectors_unfiltered/`; both exist in the May data. Roger's TODO: probably
not a net win; compare.

**Axis / pole lists** (`roger/axis_judge_experiments/`):

| File | Contents |
|---|---|
| `pair_list_clean.json` | 59 axes (54 trait, 5 role) whose poles name each other as opposites |
| `pair_list_di.json` | 59 axes, including role pairs like `instrumentally_aligned_ai` / `paperclip_maximizer` |
| `pair_list_goalnongoal.json` | 38 axes, tagged goal / non-goal |
| `*_v1.json` | old names, matching the May data; the current files are for the next extraction |
| `judge_tier_robustness_61axes.json`, key `axes` | Roger's 61 (every axis with cached projections) |

Each trait file also has `negative_label` and `arrangement` (pair,
triangle, square…; 262 pairs recorded).

---

## 13. The cross-ρ, explained

Roger (42:48): "maybe compare the ρ between the judging from one against the
embeddings from the other… cross ρs between the two."

**Every ρ has two ingredients:**
1. **Vectors (embeddings).** Where each persona sits in activation space.
   This gives the projection ordering.
2. **Judging.** A judge's −3..+3 placement of each persona on the axis.
   This gives the judge ordering.

A filter can change **either** ingredient:
- **Vectors change** because each persona's vector is averaged over a
  different set of answers: score-3 only, or all.
- **Judging changes only if the judge reads the answers.** That is
  **response judging** (`results_analysis/axis_judge_correlation.py
  --score_responses`): the filtered arm's judge sees only kept answers, the
  unfiltered arm's sees all of them.
- **Description judging** reads only the persona's written description, so
  it is **identical** in both arms.

**With response judging, each axis has four ρ values:**

| | Judging from FILTERED answers | Judging from ALL answers |
|---|---|---|
| **Vectors FILTERED** | ρ₁ (today's pipeline) | ρ₃ (cross) |
| **Vectors UNFILTERED** | ρ₄ (cross) | ρ₂ (fully unfiltered) |

**Per-axis, then averaged.** Each cell is computed **per axis**: one ρ per
axis, per setting. The mean over axes is only a summary. The per-axis values
are the primary result and must be saved and reported (§14.10).

**Reading it:**
- **ρ₁ vs ρ₂** is the plain on/off comparison the meeting asked for. Both
  ingredients change at once.
- **Change only the vectors** (ρ₁ → ρ₄, or ρ₃ → ρ₂): if ρ moves, the vectors
  moved. The dropped answers carried persona signal, or noise.
- **Change only the judging** (ρ₁ → ρ₃, or ρ₄ → ρ₂): if ρ moves, the judging
  moved. The judge reads personas differently when it also sees refusals.

**Example.** If ρ₂ > ρ₁ and ρ₄ ≈ ρ₂, the gain came from the vectors: refused
answers still carry the persona, which is Roger's hypothesis at 47:31.

**With description judging, the table collapses to one column** (ρ₁ vs ρ₄): a
pure vectors-only test. That is exactly Roger's 61-axis run. Clean, but not a
cross-ρ.

**Cost.** Response judging is the expensive judging, so this is the "paid
cross-comparison". Roger's draft plan
(`reports/judge_pareto/response_judging_plan.md`, 3 Nov) estimates **about
$8.6 per axis** at the ⅓ level, with roughly 380 kept answers per persona out
of 500. The unfiltered arm judges all 500, about 1.3× more. See §14.5 for how
to stage it under the $20 rule.

**A cheaper, sharper test: the "discarded-only" arm.** Not from the meeting,
but it answers Roger's 47:31 question directly. Build vectors from **only**
the answers scored 0–2, and compute ρ against description judging. If that ρ
is close to the filtered ρ, the thrown-away answers carry the persona too. No
new judging is needed. This is a required task in §14.5.

---

## 14. HAND-OFF FOR CLAUDE CODE: how to do TBD 3 correctly

This section is written for a Claude Code session picking up TBD 3. Read
§1–§13 for background first. `agents/TBD3_plan.md` is the one-page summary.
§15 covers the end-think arm choices and their link to TBD 4.

### 14.1 Ground rules (non-negotiable)

1. **Git.** Commit or push only when Konstantinos explicitly says so. A stop
   hook may ask you to commit; decline and say why.
2. **Branch.** Work from `origin/personas-konstantinos` (Roger's code;
   `master` is Lu's old code). Read its `CLAUDE.md` (generated from
   `AGENT_NOTES.md`; never edit it directly) and the `.claude/rules/` that
   load for files you touch: `judging.md`, `judge-cost.md`,
   `judge-scoring.md`, `provenance.md`, `plotting.md`, `nfs-io.md`,
   `entity-naming.md`, `axis-geometry.md`.
3. **Cost rule.** Before any run that may exceed **$20 of API spend or 13
   GPU-hours**:
   - stop
   - estimate the cost (use `--dry_run` where it exists)
   - quote each non-default flag as "`--flag value` (default: X; chose Y
     because Z)"
   - wait for Roger's explicit OK, relayed by Konstantinos. "OK" alone is
     ambiguous, and earlier approval does not carry over.
4. **Never overwrite canonical data.** Write every new arm to a new folder
   (`<root>_rpunfiltered`, `<root>_rpdiscarded`, `scores_unfiltered/`,
   `<root>_thinkexp`, …).
5. **Verify, don't assume.** Check actual on-disk state and current `--help`
   defaults. Docs and defaults change.

### 14.2 Facts you need

- **Data layout.** For each persona there is `vectors/<name>.pt` with key
  `"vector"` of shape **(8, 64, 5120)**: one averaged 5,120-number vector for
  each of 8 positions ("slots") × 64 layers. An analysis picks one
  (slot, layer).
- **Slots (Qwen):**
  - 0 = mean over the answer's tokens
  - 1–7 = the single header tokens `<|im_start|>`, `assistant`, `\n`,
    `<think>`, `\n\n`, `</think>`, `\n\n`

  Slot 6 (`</think>`) is the default; layers 25 and 49 are the two ρ peaks.
  Header tokens come **before** the answer, so a filter changes only *which
  answers are averaged* at slots 1–7, not their activations. Slot 0 depends
  on the answer text.
- **How a persona vector is built** (Lu's method, unchanged):
  1. For each answer, run system prompt + question + answer through the
     model.
  2. Record each layer's output.
  3. Average over the answer's tokens (slot 0), or take the single header
     token (slots 1–7).
  4. Average over the persona's **kept** answers (score 3 from the 0–3
     role-play judge; at least 50 needed; the default Assistant is never
     filtered).

  Nothing else: no weighting or normalisation.
- **How an axis and ρ are built** (`results_analysis/rho_by_layer.py`,
  `_axis_unit` and `_project`):
  1. At one (slot, layer): axis = vector(pos pole) − vector(neg pole), scaled
     to length 1. Vectors are first centred on `default.pt`, which cancels in
     the difference. With whitening, all vectors are transformed first.
  2. Project every other persona (~560, both poles excluded) onto the axis.
     That gives ordering 1.
  3. The axis judge's −3..+3 scores of each persona's description give
     ordering 2.
  4. ρ = Spearman(ordering 1, ordering 2), one per axis. The mean over axes
     is the summary.
- **12 settings** = slot (0, 6) × layer (25, 49) × regime:
  - raw
  - soft-K, K=2: shrink the 2 dominant directions
  - soft-shear, L=3: make the goal / non-goal subspaces' top 3 shared
    directions perpendicular

  Code: `results_analysis/canonical_angles/whitening.py`.
- **Two different judges:**
  - The **role-play judge** (0–3, step 3, reads each answer) decides what is
    kept. This is filter (a).
  - The **axis judge** (−3..+3, reads descriptions, or answers in response
    mode) gives the ρ ordering.

  Axis-judge scores are in git: ~9,700 `scores_descriptions.json` under
  `roger/axis_judge_experiments/`.
- **Answers.** Roger mode: 5 prompts × 100 questions (`--reduce_questions 3`
  of 300) = 500 answers per persona.
- **Combinations** (`--mode roger`): prompt *i* = role prompt *i* + newline +
  trait prompt *i*, for *i* = 1…5, in file order. Every persona file has
  exactly 5 prompts. Which roles and traits: the first 30 of each list in
  `data/goal_roles_and_traits.json`.
  - `r_` = goal role × non-goal trait
  - `t_` = non-goal role × goal trait
- **Thinking handling:**
  - **Roger's pipeline** (the one we run): `assistant_axis/generation.py`
    (~line 275) cuts any Qwen answer containing a stray `</think>` down to
    the text after the last `</think>`, **before saving**. The cut text is
    lost; the judge and the slot-0 average see only the cut answer.
  - **Lu's code** instead keeps leaked text in the slot-0 average. Her
    `internals/conversation.py` skips only tokens inside a matched
    `<think>…</think>`; with no opening tag, the leaked text passes through
    and only the `</think>` token is dropped. §15 discusses what each arm
    should keep.

### 14.3 The pipeline: scripts, order, what each does

Run pipeline scripts from `pipeline/` with `uv run`; run analysis scripts
from the repo root. `pipeline/run_pipeline.sh` runs steps 1–5 (and the scan)
end to end; for ablations run steps individually so you control folders and
flags. `<root>` = dataset folder laid out as
`<root>/{roles,traits,combinations}/{responses,activations,scores,vectors}`.

**Pipeline steps** (order: 1 → 2 and 3, which can run in parallel → 4 → 5 →
scan):

| Step | Script | What it does | Key flags | Cost |
|---|---|---|---|---|
| 1 | `1_generate.py` | vLLM generates answers: each persona's 5 system prompts × questions. Writes `responses/<name>.jsonl`. **Applies the end-think cut** (generation.py). | `--mode roger\|christina`, `--roles_dir`, `--traits_dir`, `--roles` (subset), `--reduce_questions 3`, `--goal_count/--non_goal_count`, `--output_dir` | GPU |
| 2 | `2_activations.py` | Re-runs each conversation through the model with forward hooks on every layer. Saves per-answer activations at the 8 slots (`activations/<name>.pt`, GBs). | `--model`, `--responses_dir`, `--output_dir`, `--batch_size`, `--thinking false`, `--roles` | GPU |
| 3 | `3_judge.py` | Role-play judge scores each answer 0–3 (`scores/<name>.json`). **Applies the incongruity filter** for combinations. | `--entity_type role\|trait\|combination` (required), `--judge_model` (default gpt-4.1-mini), `--no_incongruity_filtering`, `--dry_run`, `--roles` | API |
| 4 | `4_vectors.py` | **Applies the role-play filter**: mean over score-3 answers (≥ `--min_count`, default 50) → `vectors/<name>.pt`. Default persona: all answers. | `--activations_dir`, `--scores_dir`, `--output_dir`, `--min_count`, `--overwrite`, `--reduce_questions` | CPU |
| 5 | `5_axis.py` | Lu's Assistant Axis (default − mean of personas). Not needed for TBD 3. | `--vectors_dir`, `--output` | CPU |
| scan | `scan_missing_vectors.py` | Checks every activation has a healthy, loadable vector. Auto-pairs `vectors_unfiltered/` with `scores_unfiltered/`. | `--root <root> --min_count 50` | CPU |

**Analysis and helper scripts used in TBD 3:**

| Script | What it does |
|---|---|
| `scripts/judge_tier_build_unfiltered_vectors.py <may_root> <out_root>` | Builds role and trait vectors from **all** answers: `(n3 × filtered + Σ non-3) / n_all`. Reads only the non-3 answers from the activation files. Writes `unfiltered_stats.jsonl` (per persona: n_all, n3). **Hard-codes `G = /Volumes/G-Drive/...`**; repoint it. |
| `results_analysis/judge_tier_robustness.py` | ρ for 9 axis judges × 61 axes × 12 settings from description judging. Flags: `--data_dir`, `--stem`, `--output_dir`, `--no_cache`, `--slots`, `--layers`, `--regimes`, `--axes`, `--bootstrap_draws`, `--seed`. Output JSON has `rho[judge][axis][setting]`, i.e. per-axis values. |
| `results_analysis/rho_by_layer.py` | ρ at every layer for given slots and whitening levels. Flags: `--data_dir`, `--rhos_json`, `--plot`, `--layers`, `--regime`. `rho_by_layer_simple.py --input <json> --slot N --L 0` plots one curve; it blends response and description ρ by default, so check `--help` and say which you used. |
| `results_analysis/axis_judge_correlation.py` | One axis in detail (`--pair`, `--data_dir`, `--slot`, `--layer`). With `--score_responses --scores_dir --responses_dir`, it does **response judging**, which is paid. |
| `results_analysis/compute_combo_marginals.py --data_dir <vectors dir>` | Splits combination vectors into role and trait parts. For (b). |
| `data_analysis/score_combinations.py` | Produced `data/combination_scores.json` (the incongruity scores). Don't rerun. |

### 14.4 Step 0: locate and verify the data

Konstantinos will give you **the path to his local copy** of the May "8-slot"
dataset (from the shared Drive). Look there for the vectors. Check, and report
before doing anything else:
- [ ] `roles/vectors/*.pt`, `traits/vectors/*.pt`: load one; `"vector"` is
  **(8, 64, 5120)**.
- [ ] `traits/vectors/default.pt` and `combinations/vectors/default.pt`.
- [ ] An unfiltered folder (`… 8slot unfiltered/`) with
  `unfiltered_stats.jsonl`.
- [ ] `combinations/scores/`, `scores_unfiltered/`, `vectors/` and
  `vectors_unfiltered/`.
- [ ] `*/activations/`: needed only to rebuild vectors. GBs each; may be only
  on Roger's G-Drive.
- [ ] `*/scores/` (the 0–3 role-play scores): needed for the discarded-only
  arm and for counts.
- [ ] Counts: ~280 roles, ~300 traits, 900 `r_` + ~900 `t_` combinations, up
  to 500 answers each.

If something is missing, list it and stop. Don't regenerate data to fill gaps
without approval.

### 14.5 Tasks for (a): role-play filter

Do them in this order. 1–5 cost nothing; 6 is paid and gated.

1. **Reproduce Roger.** Run `judge_tier_robustness.py` on the filtered root
   and the unfiltered root (separate `--stem`, `--no_cache`). The means must
   match `roger/axis_judge_experiments/judge_tier_robustness_61axes_filtered_vs_unfiltered.txt`
   (e.g. gpt-4.1-mini, s6_l25_raw: 0.605 → 0.606). If they don't, stop and
   find out why.
2. **Per-axis ρ (not just the mean).** From both JSONs, build a per-axis
   table: for every axis × setting × judge, ρ_filtered, ρ_unfiltered, Δρ.
   Save it as CSV/JSON. This is the primary result; means are a summary.
3. **Discarded-only arm** (§13).
   - Build vectors from only the answers scored 0–2:
     `v_disc = (n_all × v_unfilt − n3 × v_filt) / (n_all − n3)`, using
     `unfiltered_stats.jsonl`.
   - Skip personas with fewer than 50 discarded answers, and report how many
     were skipped.
   - Save to `<root>_rpdiscarded/`, with `default.pt` and other files
     symlinked as the unfiltered script does.
   - Run task 1's script on it. Report ρ_discarded per axis and as means
     alongside ρ_filtered.
   - Verify the formula on one persona against a direct mean over its 0–2
     answers from the activation file, if available.
4. **Relate change to data lost.** For each axis, compute the fraction of
   the two poles' answers that the filter removed (from
   `unfiltered_stats.jsonl`). Correlate it with Δρ, and with the cosine
   between the filtered and unfiltered axis directions.
5. **Layers.** Run `rho_by_layer.py` on the filtered, unfiltered and
   discarded roots (separate `--rhos_json`, `--plot`) for slots 0 and 6, raw.
   Does the layer-27 / layer-49 structure change?
6. **Cross-ρ (paid; gated).** First ask (via Konstantinos) whether Roger
   meant response judging. If yes:
   - check which axes already have May response judging
     (`scores_responses.json`, about 12 axes); those can supply ρ₁ if the
     method matches
   - **pilot one axis** (≈ $20 for both arms) to measure real cost
   - then propose about 10 axes, choosing those with both existing response
     judging and the largest Δρ / most answers lost. Get approval before
     running.

   Fill the 2×2 table per axis (§13).

### 14.6 Tasks for (b): incongruity filter

No GPU or API; the data exists.
1. Count how many combinations **in the May run** have a prompt variant with
   incongruity 3 (`data/combination_scores.json` covers 3,200; the run used
   ~1,800).
2. Check that only those combinations differ between `combinations/vectors/`
   and `vectors_unfiltered/` (hash the files).
3. Run `scan_missing_vectors.py --root <root> --min_count 50`.
4. Run `compute_combo_marginals.py` on both vector folders. For each affected
   combination, compare answers kept and the cosine between the two vectors.
5. If any of the 61 axes' ρ uses combination vectors, recompute it for both
   arms. Otherwise state that ρ is unaffected by construction.

### 14.7 Tasks for (c): end-think cut

Needs a code change, GPU and API; gated by the cost rule. See §15 for the
arms and why.
1. **Ask first** (via Konstantinos): does Roger already have a flag, and which
   personas leak most?
2. **Code change.** In `assistant_axis/generation.py`, save the raw text in
   an extra field (e.g. `raw_response`) next to the cut `response`, so every
   arm can be derived from **one** generation. Keep today's behaviour as the
   default.
3. **Measure the leak first.** On a small sample of personas likely to leak
   (stream of consciousness, pensive, disorganized…) plus a few controls,
   count answers with a stray `</think>`, per persona. Report the rates;
   they set the size of the effect.
4. **Generate once** for the affected personas. Use May's step-1 settings:
   `--mode`, `--reduce_questions 3`, `max_tokens`, temperature, the same 5
   prompts. Prompt files were partly regenerated in September, so use the
   May versions or note the difference.
5. **Build the arms (§15)** from that one generation: steps 2–4 per arm into
   separate folders.
6. Compare ρ per axis, mainly at slot 0, for axes involving the affected
   personas. Run the causality check (§14.8).

### 14.8 Correctness checklist

**(a) Role-play filter:**
- [ ] Unfiltered and discarded formulas verified on one low-keep persona
  (Roger used `roles/screener`, 21% kept) against a direct mean, to bf16
  rounding.
- [ ] Personas with all answers scored 3 come out identical to the filtered
  vector (symlinked).
- [ ] `default.pt` is identical in all arms. It is never filtered.
- [ ] **Same persona set in all arms.** Some personas have no filtered vector
  (<50 score-3) but do have an unfiltered one. Restrict to the intersection,
  or report both sets and label them.
- [ ] Each axis's poles are excluded from its own ρ; only finite projections
  count.
- [ ] Ambiguous names (ascetic, contrarian, cosmopolitan, generalist,
  pacifist, patient, perfectionist, romantic, stoic): use `name|R` /
  `name|T` (`entity-naming.md`). Never guess the kind.
- [ ] Indexing: slot 6 = `</think>`, slot 0 = answer mean, layers
  zero-indexed (25, 49 as in Roger's runs), vectors centred on `default.pt`.
- [ ] **Whitening refitted per arm.** soft-K and soft-shear are fitted on the
  pool in `--data_dir`, as Roger did. If you also hold the fit fixed, report
  that separately.
- [ ] **Separate caches per arm** (`--stem`, `--rhos_json`, `--no_cache`) so
  arms can't mix.
- [ ] **Noise:** |Δρ| < 0.005 = no change. For "the filter matters"
  claims, use paired t-tests over axes (Roger's convention in
  `axis-geometry.md`) or `--bootstrap_draws`.
- [ ] Roger's numbers reproduced before anything new.
- [ ] Cross-ρ: same judges, rubric, batch setup and ⅓ split in both arms.
  Only the answer set differs.

**(b) Incongruity filter:**
- [ ] Only combinations with an incongruity-3 variant differ. All others are
  bit-identical. If not, the runs differ in something else (judge or rubric
  version).
- [ ] Step-3 logs ("excluding N responses") match the count of variants
  scored 3.

**(c) End-think cut:**
- [ ] All arms come from **one** generation (raw text saved). Never compare
  a new generation with the May answers: sampling at temperature 0.7 gives
  different answers.
- [ ] **Causality check.** Header slots 1–7 must be **identical** across arms
  for the same (prompt, question), to floating-point precision. Only slot 0
  and judge scores may differ. If headers differ, extraction is wrong.
- [ ] Non-leaking answers are identical in every arm.
- [ ] Report how many answers changed role-play score between arms.

### 14.9 House conventions

- Every plot: read it back as an image and check for overlapping text. Embed
  provenance PNG metadata (`plotting.md`, `provenance.md`).
- Every JSON result: use the project's provenance envelope.
- Judge prompts: reasoning before the score, never score-first
  (`judge-scoring.md`).
- Paper-facing text: never "slot 0". Write "mean over response tokens" and
  "the `</think>` token".

### 14.10 How to present results

Not specified in the meeting; Roger only said "see which has the higher ρ"
plus the cross-ρ. This format follows Roger's existing comparisons:
- `judge_tier_robustness_61axes_filtered_vs_unfiltered.txt` (mean ρ before >
  after per setting, plus most-rotated axes)
- `rubric_v1_v2_compare_slot6.png` (per-axis paired bars with Δ above each
  pair, a mean bar, and a box of cross-axis means)
- paired t-tests in `axis-geometry.md`

**Provisional choices (confirm with Roger):**
- **Headline judge = gpt-4.1-mini**, with all 9 judges in a supplementary
  table.
- **Paid cross-ρ on ~10 axes after a 1-axis pilot.**

**For each filter, produce:**
1. **Table 1: headline.** One row per setting (12).
   - Columns: mean ρ off | mean ρ on | Δ | paired-t p over axes | # axes up
     > 0.005 | # axes down > 0.005 | all 9 judges agree on the sign of Δ?
   - For (a), add a "discarded-only" mean-ρ column.
2. **Figure 1: per-axis paired bars,** in the style of
   `rubric_v1_v2_compare_slot6.png`.
   - Settings: s6_l25_raw and s0_l25_raw.
   - Axes sorted by the fraction of answers dropped.
   - Δ printed above each pair, plus a mean bar.
   - For (a), a third bar for discarded-only.
3. **Figure 2 (a only):** scatter of Δρ per axis against the fraction of the
   poles' answers removed, with the correlation r.
4. **Table 2: rotation.** Cosine and angle between the on/off axis
   directions: median and the 10 most rotated (s6_l25 and s0_l25).
5. **Table 3: cross-ρ** (if approved). The 2×2 per axis, for the piloted
   axes.
6. **Per-filter extras:**
   - (b): a table of the affected combinations (answers kept per arm,
     cosine).
   - (c): a table of leak rate per persona, then per-arm ρ at slot 0.
7. **Recommendation paragraph:** keep, flip or drop the flag, with the
   evidence and the noise level. Also give 2–3 concrete examples for a
   possible appendix:
   - (a): refusals whose vectors still project like the persona
   - (c): leaked thinking text

Save the per-axis numbers behind every table and figure (CSV/JSON with
provenance) so Roger can re-plot.

---

## 15. End-think arms: what to keep and what to leave (and the link to TBD 4)

### The answer, split into parts

A generated answer that leaked looks like:

```
[pre-filled]  <|im_start|>assistant\n<think>\n\n</think>\n\n
[generated]   ...leaked thinking text...  </think>  ...real answer...
```

Header slots 1–7 come from the pre-filled part, which is the same in every
arm. The question for each arm is what the answer-side processing uses: the
slot-0 average, and the text the role-play judge and any response judge see.

| Arm | Slot-0 average over | Judge sees | Matches |
|---|---|---|---|
| **STRIP** (default, today) | real answer only (after the last `</think>`) | real answer only | Roger's pipeline |
| **KEEP** | leaked text + real answer; the special `<think>`/`</think>` tokens excluded | leaked text + real answer | Lu's code (by reading `conversation.py`) |
| **DROP** (optional) | leaking answers removed entirely | — | the "throw it all out" choice; removes any doubt about mixed content |

### Why these arms

- **STRIP vs KEEP** is the flag Roger said might need to "go the other way".
  It is also the difference between our pipeline and Lu's for these answers,
  which the paper must justify.
- **DROP** tests whether the leaking answers are unusual even after cutting.
  If STRIP ≈ DROP, the cut text was the only problem. If they differ, the
  answers that leak are different, which is an effect worth reporting.
- **Same generation for all three.** That is the only way the arms differ by
  the processing choice alone.

### Possible extra arm (for the appendix, not the flag decision)

**THINK-ONLY**: the slot-0 average over only the leaked text. Then take the
direction "THINK-ONLY − STRIP" per persona and see which trait axes it lines
up with. The informal Claude analysis said "pensive", "considering
alternatives". KK suggested this could reveal pre-verbal, hidden content;
Roger expects an appendix.

### Link to TBD 4 (thinking vs speaking)

TBD 4 asks whether some personas mostly change how the model **thinks** and
others how it **speaks**, and whether these show at different layers (27 vs
49). The leaked text is literally the model's thinking under a persona.
So:
- The **flag decision** (STRIP / KEEP / DROP) belongs to TBD 3. It must be
  settled before publishing.
- The **content analysis** of the leaked text (THINK-ONLY, its direction, its
  layer profile) fits naturally with **TBD 4**, and could be done there or as
  a joint appendix.
- Note that TBD 4 runs on the May data, where the leaked text is already cut.
  So TBD 4 can't see it unless the §14.7 generation (raw text saved) is done
  first.

Flag this to Roger when TBD 4 is scoped.

---

## 16. Session housekeeping

- Konstantinos asked that nothing be committed without his explicit
  say-so. A stop hook repeatedly asked for commits and was declined until
  he approved committing these notes on `claude/jolly-mendel-7xsvsr`.
- Files: `agents/TBD3_plan.md`, `agents/TBD3_discussion_context.md` (this
  file).
