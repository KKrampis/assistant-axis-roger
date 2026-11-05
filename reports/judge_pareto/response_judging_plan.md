# Response judging: the plan (draft, 2026-11-03)

Written for Roger, to approve before anything runs.  Descriptions and instructions are settled ([`judge_descriptions.py`](../../results_analysis/judge_descriptions.py), [`judge_instruction_grid.py`](../../scripts/judge_instruction_grid.py)); this is the third and expensive one.  Terms: a *response* is one answer Qwen-3-32B gave, as an entity, to one of the 100 questions under one of its five instructions; *kept* means step 3 scored it 3 (in role, in trait) and it went into the entity's vector; *ρ* is the Spearman correlation, over an axis's entities, between the judge's scores and the activation projections (slot 6, layer 25, raw, from each axis's cached `gpt/projections.json`).

## 1. What changes from the May method

| | May 2026 ([`axis_judge_correlation.py --score_responses`](../../results_analysis/axis_judge_correlation.py)) | this plan |
|---|---|---|
| the unit judged | a batch of 7 of **one entity's** responses, one `SCORE: −3..+3` for the batch "collectively", batches averaged | **one score per response** |
| a call | 7 responses of one entity, each with its question | one question, asked once, and N responses to it from **N different entities**; the judge places each by percentile among them (the `b10` rubric's form) |
| judges | GPT-4.1-mini and Haiku 4.5, 0.525 : 0.475 | GPT-5.6 Luna, Haiku 5.5, Gemini 3.1 Flash-Lite, 4 : 3 : 3 (`TIER_JUDGE_WEIGHTS`) |
| coverage | a third of the questions per entity, widened for RP-depleted entities, both judges on the same items | every kept response judged once, by one judge (§ 2); escalation adds judges per response |
| transport, price | real time, list | Flex and batch, the real price |

The batch of different personas answering one question is itself the judge's calibration, as the batch of different entities is in description judging, and the question once saves about 5% of the input (14 tokens against a 392-token response) and, more usefully, keeps the call to one topic.  That the judge can do without the question beside each response is the first thing to confirm (§ 4, arm Q).

## 2. Coverage and escalation (Roger's design)

A kept response is `(entity, instruction p, question q)`; an entity has 500 items (100 questions × 5 instructions) of which 380 are kept on average (73 to 500 over a sample of 24 entities).

- **⅓ level (the default).**  Each entity's responses are split in thirds, and the thirds go to the three judges, so every kept response is judged exactly once and every judge sees every entity.  **Since 2026-11-04 the split is golden** (Roger): judge j takes the responses whose point frac((5Q + p) / φ) lies in [j/3, (j+1)/3), Q the absolute question number and p the instruction, the same cells for every entity, so each judge sees every entity on one or two of each question's five instructions.  It was first by question with an entity rank r, judge j taking the questions with `(q + r) % 3 == j`; § 4b has why it changed.  Per entity the judges' scores are averaged 4 : 3 : 3 over the items each judged.
- **⅔ level.**  Each judge takes two thirds, so each question is seen by two of the three judges.
- **Full.**  Every judge sees every response.

The May escalation (more questions for an entity the RP filter had depleted) is not needed: at the ⅓ level everything kept is already judged.  An entity left with few kept responses (the minimum seen is 73) can be escalated to the ⅔ or full level on its own, which is the same code path.

## 3. Cost

Per kept response: about 392 input tokens (median 429, cut at the 512 generation cap) plus ~10 of framing; per call a ~350-token header and the question; output a score per item, no reasoning (as settled for descriptions), ~4 tokens.  An axis has about 577 entities × 380 kept = 219,000 responses, 88 M input tokens.

| level | judge-items per response | $ per axis, May corpus (real price) | 200 axes, current corpus (2.2× the entities) |
|---|---:|---:|---:|
| ⅓ | 1 | ≈ 8.6 | ≈ 3,800 |
| ⅔ | 2 | ≈ 17 | ≈ 7,600 |
| full | 3 | ≈ 26 | ≈ 11,400 |

(Luna by Flex $0.10/$0.60, Haiku by batch $0.05/$0.25, Gemini by batch $0.125/$0.75 per million; the May method was about $50 an axis per judge at list.)  The 200-axis figures are why the pilot is worth its cost: a design that halves the input (shorter cap on the response text, say) or shows the ⅓ level is enough saves thousands.

## 4. The pilot: hill-climb ρ on the cheapest judge

**Judge:** GPT-5.6 Luna by Flex, $0.10 / $0.60 per million, the judge production leans on most (Roger, 2026-11-03: Haiku, at $0.05 / $0.25, "isn't enough cheaper to justify it being the guinea pig").  **The fraction curve** is the pilot's most valuable reading: ρ at a ninth, a sixth and a third of each entity's kept responses, read by subsampling within the judged third, and two axes judged in full for the top end (about $9 more), since the cost of production scales with the fraction judged and the May method reached ρ 0.75 on a third of the questions.  **Axes:** six from [`pair_list_responses.json`](../../roger/axis_judge_experiments/pair_list_responses.json), which have the May response judging as a baseline and the cached projections, chosen to span trait and role pairs and the easy and hard ends (proposed: [`blunt`](../../data/traits/instructions/blunt.json) vs [`tactful`](../../data/traits/instructions/tactful.json), [`concise`](../../data/traits/instructions/concise.json) vs [`verbose`](../../data/traits/instructions/verbose.json), [`honest`](../../data/traits/instructions/honest.json) vs [`dishonest`](../../data/traits/instructions/dishonest.json), [`ecocentric`](../../data/traits/instructions/ecocentric.json) vs [`anthropocentric`](../../data/traits/instructions/anthropocentric.json), [`predator`](../../data/roles/instructions/predator.json) vs [`prey`](../../data/roles/instructions/prey.json), [`angel`](../../data/roles/instructions/angel.json) vs [`demon`](../../data/roles/instructions/demon.json)).  **Sample:** a third of each entity's kept responses (the ⅓ level's share for one judge), the same third in every arm, so the arms are paired.

| arm | what it varies | $ per axis (est.) |
|---|---|---:|
| Q | the question once in the header (the design, rubric `r1_h`) against beside each response (the May form, `r1_q`) | 3.2 each |
| N | 5, 10, 20 responses a call, question once | 3.2 each (N=5 more, its headers) |
| R | one sentence of reasoning per item (the b10 form) against the score alone (`r1_hn`), at the best N | 3.2 each |
| C | the response text cut at about 256 tokens (`--response_chars`) against whole, at the best N | 2.0 |

About 8 cells × 6 axes × $3.2 ≈ **$150** on Luna, plus the two full-coverage axes ≈ $18.  Then the chosen design with the three settled judges on the same six axes at the ⅓ level, ≈ **$50**, to see that the winner carries over (as descriptions and instructions did) and to measure the ⅓, ⅔ and full levels from the same cells (the full level judged, the lower levels read from it).  **Pilot total about $220; research costs approved by Roger on 2026-11-03**; each cell runs under `--budget_usd` at 1.5× its estimate plus $5, the first-of-kind rule.  Production's judge set is then a cost question again: Haiku and GPT-6 Luna at $0.05 input against GPT-5.6 Luna at $0.10 and Gemini at $0.125, and round-robin coverage weighs each judge's price by its third (the settled trio ≈ $8.6 an axis, Haiku + GPT-6 Luna ≈ $4.6).

Decisions are by the method of this week: ρ under the 12 projection variants ([`judge_tier_robustness.py`](../../results_analysis/judge_tier_robustness.py)), the convex-hull frontier, paired bootstraps over axes, and the slope against the exchange rates we have accepted (Luna's second description pass, 1.1 quality per $ an axis; a second judge, 2.3).  The May baseline on these axes is the blend's ρ in each axis's `gpt_responses_*_b7` and `haiku_responses_*_b7_t3` caches, about 0.75 on the 12-axis cohort response-only and 0.76 blended with descriptions and instructions at 0.8 : 0.2.

## 4a. Readings so far, and what they changed (2026-11-04)

On a ninth of the responses, six axes, GPT-5.6 Luna: the question once beats the question beside each answer (0.662 against 0.655); N=20 is as good as or better than 10 and 5; **the score alone beats the sentence of reasoning (0.683 against 0.662) at 16% less**, so `r1_hn` is the design from here; the cut at 1,100 characters lost 0.014 and **Roger rules it out** ("Responses often contain a preamble, details, and a summary ... truncating cuts the summary", 2026-11-04).  The data cost about $400 to make and the trio at the ⅓ level would cost about $4,700 to judge on the October extraction (1,273 entities an axis, 425 kept responses each), so the target is near parity: **k axes per call** (the response paid once per k axes; a pilot at k = 2 and 6 against the k = 1 cells), subsampling read from the judged cells without more judging (the default Pareto curve), GPT-6 Luna as the cheap judge (its own `r1_hn` ninth arm), and Jev (TypeSafe AI) if it can be reached under the hosting rule.

## 4b. Cutting the cost: readings (2026-11-04)

The same ninth of the responses on the same six axes, GPT-5.6 Luna by Flex unless named, cost per axis at the real price; "vs R" is the paired difference from the score-alone arm `r1_hn` with its 95% bootstrap interval over the axes.  The readout script is in the session scratchpad; the run logs and drivers are in [`response_pilot_logs/`](../../roger/axis_judge_experiments/response_pilot_logs/) ([`run_prefix.sh`](../../roger/axis_judge_experiments/response_pilot_logs/run_prefix.sh), [`run_stack.sh`](../../roger/axis_judge_experiments/response_pilot_logs/run_stack.sh)).

| Arm | ρ | vs R | $/axis |
|---|---:|---:|---:|
| R (`r1_hn`), N=20 | 0.683 | | 1.28 |
| R at N=40 | 0.683 | −0.001 [−0.005, +0.005] | 1.25 |
| **cached prefix, one axis a call (`r1_pn`)** | **0.687** | **+0.004 [−0.009, +0.019]** | **0.28** (first axis 1.56) |
| prefix + 2 axes a call, low-cosine pairs (`rk_pn`) | 0.671 | −0.012 [−0.032, +0.007] | 0.18 |
| prefix + 2 axes a call, high-cosine pairs | 0.686 | +0.002 [−0.013, +0.017] | 0.18 |
| prefix + 3 axes a call | 0.668 | −0.016 [−0.020, −0.009] | 0.14 |
| prefix + 6 axes a call | 0.661 | −0.022 [−0.061, +0.004] | 0.09 |
| 6 axes a call, no prefix (`rk_hn`) | 0.665 | −0.019 [−0.055, +0.005] | 0.26 |
| GPT-6 Luna, R | 0.666 | −0.017 [−0.050, +0.008] | 0.63 |
| Jev, Score, 3 levels | 0.554 | | 0.40 |
| Jev, Score, 5 levels | 0.564 | −0.119 [−0.296, +0.021] | 0.46 |
| Jev, Score, 7 levels | 0.547 | | 0.52 |
| Jev, Score, 10 levels | 0.576 | −0.107 [−0.225, −0.002] | 0.61 |
| Jev, Choice of "0" to "10" | 0.573 | | 0.75 |

What they say:

- **The cached prefix costs no quality and cuts the input bill tenfold** (the question and the answers first, read back at a tenth of the price; the axis after).  Of a later axis's $0.28, output is about 42%, the cache reads 40% and the axis text 17%.  A prefix written at 01:04 UTC was still read in full at 01:46, so it outlives the documented 30 minutes when it is read in between.
- **Stacking, twelve axes** ([`tier_eval_axes_responses12.tsv`](../../scripts/tier_eval_axes_responses12.tsv): the six pilot axes and six chosen so that one pairing has every |ρ| between the axes' instruction scores at 0.08 or below and another at 0.44 or above, Roger's criterion of similarity, 2026-11-04).  One axis a call 0.732; high-|ρ| pairs 0.719 (−0.013 [−0.042, +0.016]); low-|ρ| pairs 0.702 (−0.030 [−0.092, +0.014]).  So pairing dissimilar axes does not help, here or on the six; the losses are uneven ([proactive](../../data/traits/instructions/proactive.json) 0.761 alone, 0.434 beside predator, 0.636 beside concise), as if one axis's judgement bled into the other's, harmlessly when the two agree.  Pairs save 36% for 0.013 to 0.030; a ninth of the responses instead of a third saves 67% for 0.015, so subsampling is the better lever and stacking is dropped.
- **Stacking axes on the prefix saves less and costs some quality.**  Two axes a call average −0.005 over the two pairings, three lose 0.016 on every axis, six break [predator](../../data/roles/instructions/predator.json) vs [prey](../../data/roles/instructions/prey.json) (−0.109, as without the prefix).  **Grouping by low cosine is not supported**: the pairs with the highest cosines (0.46, −0.41, 0.09) did as well as R and the low-cosine pairs (all under 0.14) slightly worse, both within a ninth's noise.
- **Subsampling is nearly free above a third.**  One judge on 1/9, 2/9 and 1/3 of each entity's responses: 0.661, 0.673, 0.677 (the design arm), 0.690, 0.701, 0.705 (R, read from its own third and full cells); on the two axes judged in full, 1/3 to all adds 0.002 ([blunt](../../data/traits/instructions/blunt.json) vs [tactful](../../data/traits/instructions/tactful.json) 0.693 to 0.695, predator vs prey 0.328 to 0.330).
- **How to cut a part: the golden design** (2026-11-04, Roger's, refined over four steps).  A response's place on the unit circle is frac((5Q + p) / φ), Q the absolute question number (the questions file's id, not the position after step 1's every-third selection or step 3's filter), p the instruction (0 to 4), φ the golden ratio; part K of M is the interval [K/M, (K+1)/M), computed exactly in integers as Fibonacci hashing ([`judge_tier_cost_eval.golden_part`](../../results_analysis/judge_tier_cost_eval.py)).  Parts are not cut in fifths (Roger: "if we simply avoid subsampling by 5, the design gets a lot easier").  ρ below is the mean over the 12 variants, read from cells already judged; "the full axes" are the two axes judged on every response, each under two rubrics.
  - **Cutting by question loses.**  At a fifth, against cutting by question: by instruction alone +0.010 on the twelve-axis ninth, nothing on the thirds, −0.008 to −0.011 on the full axes; a diagonal `(p + q) % 5` (a Latin square, every question keeping one instruction) +0.029 on the ninth (12 of 12 axes), +0.019 and +0.024 on the thirds, +0.003 and +0.004 in full.  So stratifying within each question is what matters at a fifth.  The entity rank costs 0.001 to 0.006 when cutting by question and nothing otherwise, and shifts whenever the entity set changes, so no design carries it.
  - **Absolute question numbers alias with step 1.**  Both trees took every third question (ids 0, 3, …, 297; `question_id` = 3 × `question_index`, checked on every record of 80 entities a tree), so 5Q + p mod 3 is p mod 3: Roger's first form, 5Q + a permutation of p chosen by a hash of Q, cuts thirds of 40/40/20 and ninths of 6.6% to 13.4%.  A plain hash of (Q, p) avoids that but balances only by chance (worst part 8% off at thirds, 28% at ninths, 57% at 27ths).
  - **The golden design** keeps every part within 1.0% of equal at thirds, 2.8% at ninths and 8.2% at 27ths on our trees, splits every question's five instructions 1/2/2 over the thirds (all 300 ids, every third from 0 or 1, every fifth), and nests ninths inside thirds as intervals.  On the full axes against cutting by question: +0.0006/+0.0008 at a third (the diagonal +0.0008/+0.0009, a plain hash +0.0006/+0.0005) and +0.0062/+0.0055 at a ninth (diagonal +0.0058/+0.0049, plain hash +0.0050/+0.0045).  Against a plain hash inside the twelve-axis ninth: +0.0035 at a third of it, +0.012 at a ninth of it.  (The diagonal cannot be compared inside the ninth and third cells, which were themselves cut by question mod 9 or 3.)
  - Calls are batched by question and every design keeps the same items per question, so calls and cached prefixes are unchanged.  Not measured: these cells were judged in by-question batches, and a golden part puts several times as many distinct entities into a question's batches.
  - [`judge_tier_cost_eval.py`](../../results_analysis/judge_tier_cost_eval.py) `--response_design golden` is the default since; `question` resumes or rereads the earlier cells, and a finished cell of the other design is refused.  The Pareto figure's L+H round-robin takes the golden halves of the ninth (+0.0037 over the halves by question at the whole ninth) and every arm's fractions are golden thirds of its share.  Scripts in the session scratchpad: `subsample_designs3.py`, `weyl_balance.py`, `golden_flat.py`, `golden_rho.py`.
- **Jev is weaker and not cheaper.**  Its mean is 0.11 to 0.13 below GPT-5.6 Luna whatever the level count; it leads on blunt vs tactful and predator vs prey and collapses on [ecocentric](../../data/traits/instructions/ecocentric.json) vs [anthropocentric](../../data/traits/instructions/anthropocentric.json) (0.17 to 0.44 by level count, against Luna's 0.77).  Blended 4:3 with Luna it loses 0.017.  A request bills each axis's question text (about 330 tokens), so even at 200 axes a request it would cost about $0.38 an axis on the ninth, above the cached prefix.  Seven Score levels give 0.547, and Roger's eleven-option Choice question ("0" to "10", the probability-weighted number computed by [`judge_responses_jev.py`](../../scripts/judge_responses_jev.py), `--question_type choice --levels 11`; Score questions stop at ten levels) gives 0.573 at $0.75 an axis: every Jev variant lies between 0.547 and 0.576.  The tool requests `jev-1.14.0` and stops at an answer from another version, as the entity tool does ([jev_judging.md](./jev_judging.md)); the five response logs, requested as `jev-latest` on 2026-11-04, came wholly from `jev-1.14.0`.
- **Gemini caches too.**  A live probe of Gemini 3.1 Flash-Lite read 6,125 of 10,145 prompt tokens of a repeated prefix from its implicit cache, with no write premium (cached input $0.0125 per million in batch, a tenth); whether its Batch API hits the cache is untested, as is Haiku's under Message Batches.  Usage records now carry Gemini's cache reads.

200 axes on the October extraction (1,273 entities, 425 kept responses each), GPT-5.6 Luna alone, the first axis's prefix write spread over the 200:

| Arm | every response once | a third of the responses |
|---|---:|---:|
| R, no cache | $5,043 | $1,681 |
| **cached prefix, one axis a call** | $1,118 | **$373** |
| prefix + 2 axes a call | $737 | $246 |
| prefix + 3 axes a call | $559 | $186 |

So the cached prefix with one axis a call, on a third of each entity's responses, reaches the target (about $400) with no measured loss from the prefix and about 0.002 from the subsample.  For the trio (each judge a ninth, round-robin), Haiku's and Gemini's prices under caching are estimates until their batch caching is tested.

## 4c. Pole order (2026-11-04)

Every axis-judging prompt names one pole first, at the low number; until today that was always the pair list's order, which puts the good pole high on the 18 moral axes and low on others ([humble](../../data/traits/instructions/humble.json), [conscientious](../../data/traits/instructions/conscientious.json), [empathetic](../../data/traits/instructions/empathetic.json), [resilient](../../data/traits/instructions/resilient.json), [patient](../../data/traits/instructions/patient.json), [transparent](../../data/traits/instructions/transparent.json), [thorough](../../data/traits/instructions/thorough.json), [tactful](../../data/traits/instructions/tactful.json)).  Roger's questions: (a) are two runs in the same order more alike than two in opposite orders, and (b) is ρ better with the good pole high?  GPT-5.6 Luna judged each axis in its own order and swapped (`--swap_poles`, the scores mapped back), each order twice on the same batches; descriptions on the 61 axes, responses on the twelve.  Valence labels: the pair list's own good pole on its 18 moral axes, and 17 more by this analysis (11 clear, 6 mild), the other 26 neutral.

**(a) The order matters.**  Agreement (Spearman over the axis's entities) of two runs on the same batches:

| | same order | opposite orders | difference |
|---|---:|---:|---:|
| descriptions, 61 axes | 0.939 | 0.885 | +0.054 [+0.046, +0.063] |
| responses, 12 axes (per-entity means) | 0.994 | 0.956 | +0.039 [+0.026, +0.055] |

**(b) The good pole high is better, on descriptions, under all 12 projection variants**: one run good high minus good low, +0.013 to +0.040 (every interval above zero; smallest under soft whitening); 26 neutral axes as the control, the positive pole high minus low, −0.003 [−0.022, +0.017] at slot 6, layer 25, raw.  Responses have only three valenced axes among the twelve: −0.005 [−0.020, +0.013].

**Two runs, both orders or one**, mean of the 12 variants: on a neutral axis, one run each way beats two in one order by +0.012 (+0.009 to +0.013, all significant; responses +0.007 [+0.001, +0.012]); on a valenced axis, two good-high runs are as good as one each way (one each way minus both good high −0.011 to +0.003) and both good low is worse by 0.019.  For scale: GPT-5.6 Luna beats Haiku 5.5 by 0.012 on descriptions, and Luna's second pass adds 0.019.

**Roger's decision (2026-11-04): balance the directions everywhere**, giving up the good-high gain, because a silent valence asymmetry could invalidate some of his research.  Descriptions: each judge's two passes, one each way.  Instructions: Roger's table, per judge and instruction position, 2:1 across the judges and 3:2 per judge, a coin and a permutation per axis.  Responses: a coin per call.  Built into [`judge_tier_cost_eval.py`](../../results_analysis/judge_tier_cost_eval.py) `--directions`, the defaults of [`judge_descriptions.py`](../../results_analysis/judge_descriptions.py) and [`judge_instruction_grid.py`](../../scripts/judge_instruction_grid.py); the account in [results_analysis/README.md](../../results_analysis/README.md), "Axis direction: balanced in every judging mode".  **Every judge score made before 2026-11-04 is in the pair list's order** (about +0.03 ρ for the moral axes on descriptions).

**Instructions (61 axes; a pass is the mean of the five rounds; the existing first pass, a repeat in the same order, and two swapped passes, all on the same batches) show the same picture.**  (a) Same order 0.986 against opposite orders 0.947, +0.038 [+0.031, +0.048].  (b) Good pole high minus good low, one pass: the 18 moral axes +0.034 [+0.016, +0.055], all 35 valenced axes positive under all 12 variants (+0.008 to +0.039; the interval above zero under the raw and sheared variants, touching zero under soft whitening, as on descriptions the smallest there); the neutral control −0.010 [−0.031, +0.009].  Two passes, one each way, against two in one order on neutral axes: +0.008 (+0.007 to +0.009, every variant significant).  Over the 61 axes, against both passes in the pair list's order: one pass each way everywhere +0.007, orienting the valenced axes good-high and mixing the neutral ones +0.009 (the mean of the 12 variants).  For scale on instructions: Luna beats Haiku by 0.008 and Luna's second pass adds 0.004, so the order policy is worth as much as the judge gap and twice a second pass.

## 4d. Stacking with the worst axes left out: the decision (2026-11-05)

**Roger's decision (2026-11-05): plain stacking, 4 axes a call, the worst 20% of the axes judged one axis a call instead.**  GPT-5.6 Luna, 20 answers a call (40 was worse: −0.0048 ρ at the same coverage, 45 of 61 axes, all 12 variants; [judge_response_pareto_answers.png](../../roger/axis_judge_experiments/response_pareto/judge_response_pareto_answers.png)).

The evidence, on the 61 axes ([judge_response_pareto_61_luna.png](../../roger/axis_judge_experiments/response_pareto/judge_response_pareto_61_luna.png), [judge_response_pareto_61.json](../../roger/axis_judge_experiments/response_pareto/judge_response_pareto_61.json), drawn by [judge_response_pareto.py](../../scripts/judge_response_pareto.py) `--stack_group g61`): the worst axes are chosen by cross-fitting, on 8 of 9 question folds, and scored on the ninth held out.

- One axis a call is off the frontier in every panel.  Leaving the worst out beats stacking every axis at every k, by +0.005 to +0.007 ρ at equal cost on 12 of 12 variants.
- The shares (10%, 15%, 20%, 25%) and k (2, 4, 6) are level at equal cost, within about 0.002.  20% at k=4 is the most frequent point on the middle of the measured frontier and leads 10% at k=4 by +0.0019 (slot 6 +0.003, the interval touching zero): suggestive, not significant.
- The chosen design, the mean of the 12 variants (slot 0, slot 6): the ninth measured 0.725 against one axis a call's 0.716; every response judged (extrapolated) 0.749 (0.735, 0.763) for about **$601** a full run of 200 October axes, against one axis a call's 0.735 for $1,126.
- The axes left out on every fold at k=4: [benign](../../data/traits/instructions/benign.json) vs [malicious](../../data/traits/instructions/malicious.json), [curious](../../data/traits/instructions/curious.json) vs [incurious](../../data/traits/instructions/incurious.json), [descriptive](../../data/traits/instructions/descriptive.json) vs [prescriptive](../../data/traits/instructions/prescriptive.json), [ecocentric](../../data/traits/instructions/ecocentric.json) vs [anthropocentric](../../data/traits/instructions/anthropocentric.json), [fragile](../../data/traits/instructions/fragile.json) vs [resilient](../../data/traits/instructions/resilient.json), [opaque](../../data/traits/instructions/opaque.json) vs [transparent](../../data/traits/instructions/transparent.json), [predator](../../data/roles/instructions/predator.json) vs [prey](../../data/roles/instructions/prey.json), [proactive](../../data/traits/instructions/proactive.json) vs [reactive](../../data/traits/instructions/reactive.json), [religious](../../data/traits/instructions/religious.json) vs [secular](../../data/traits/instructions/secular.json); on most folds, [truthful](../../data/traits/instructions/truthful.json) vs [deceitful](../../data/traits/instructions/deceitful.json), [obedient](../../data/traits/instructions/obedient.json) vs [rebellious](../../data/traits/instructions/rebellious.json).  For the October axes the 20% are chosen by a pilot judged both ways (about 1/81 of the responses, roughly 2% of the run; its judgments count toward coverage); the pilot is not yet in the price above.

Open: whether Haiku 5.5 and Gemini 3.1 behave the same (Roger: the same analysis for each once Luna's is done); labelled and axis-by-axis stacking on the 61 axes (running, 2026-11-05).

## 5. Build

1. **A `r1_*` rubric family** in [`judge_tier_rubrics.py`](../../results_analysis/judge_tier_rubrics.py): the `b10` percentile rubric with a response as the item and the question in the header (arm Q's alternative puts it per item); a `--responses_dir` on [`judge_tier_cost_eval.py`](../../results_analysis/judge_tier_cost_eval.py) that reads `<kind>/responses/<entity>.jsonl` and keeps the items scored 3 in `<kind>/scores/<entity>.json`; a round is `(question, third)`, the calls of a round drawn by seed from the entities holding a kept response to that question.  Scores go to `scores_responses.json` keyed by `(entity, p, q)`, the per-entity mean beside it; `correlations.json` from the per-entity mean, as now.
2. **Coverage** as a flag of the driver: `--coverage 1/3|2/3|full`, the round-robin by entity rank; the blend cell by [`judge_tier_average.py`](../../results_analysis/judge_tier_average.py), which already renormalises weights over the judges that scored an entity.
3. **A driver** like [`judge_instruction_grid.py`](../../scripts/judge_instruction_grid.py) (`judge_responses.py` once settled), and the pilot's arms through it; the figure by [`judge_instruction_12way.py`](../../scripts/judge_instruction_12way.py) in a third mode.
4. Tests on the fakes for the item layout, the coverage split (every kept response once; each judge a third of each entity; a question's calls from the right entities), the per-entity mean, and the budget stop.

Not in scope until the design is settled: wiring the result to the older consumers of `scores_responses.json` (`load_response_scores`, the weight sweeps), and the current corpus (its responses are on the pods and the October tree).

## 6. Open questions for Roger

- The six pilot axes above, or others.
- Pos responses only, as the May method judged (the neg instruction's responses are the other pole's).
- ~~Whether the pilot's third is by question (`q % 3`, as production will split) or by instruction; by question is the proposal.~~  Settled 2026-11-04: neither; the golden design over the absolute question number and the instruction (§ 4b).
