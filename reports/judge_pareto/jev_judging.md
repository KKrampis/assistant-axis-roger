# Jev as an axis judge: descriptions and instructions (2026-11-04)

Written for Roger and for later agents.  Jev is TypeSafe AI's model (jev-1.14.0).  This report covers judging axes from the entities' descriptions and instructions; Jev on responses is in [`response_judging_plan.md`](./response_judging_plan.md) § 4b.

**Terms.**
- **ρ:** the Spearman correlation, over an axis's entities, between a judge's scores and the activation projections.
- **The 12 variants:** slots 0 and 6, layers 25 and 49, each raw, soft-whitened (K=2) and soft-sheared (L=3).
- **The default variant:** slot 6, layer 25, raw.
- **A pass:** every entity judged once.  For instructions it is five rounds, one per instruction.
- **Own order:** the pair list's order, its positive pole named second and placed at the high number.
- **Balanced:** a judge's two passes run in opposite directions (Roger's rule, AGENT_NOTES.md § "Axis direction in judge prompts").
- **Noul:** Jev's yes/no question type.  It returns the probability of yes.
- **dev30 and holdout31:** [`tier_eval_axes_dev30.tsv`](../../scripts/tier_eval_axes_dev30.tsv) and [`tier_eval_axes_holdout31.tsv`](../../scripts/tier_eval_axes_holdout31.tsv), the two halves of the 61 axes, on the May corpus pin.

## The decisions (Roger, 2026-11-04)

- **Descriptions:** "the new standard is (Luna + Haiku + Gemini + Jev) x2, with the reversal in the x2".  [`judge_descriptions.py`](../../results_analysis/judge_descriptions.py) runs this by default.  The blend is Luna 4 : Haiku 3 : Gemini 3 : **Jev 12** (`judge_score_combine.TIER_JUDGE_WEIGHTS_BY_MODE`).
- **Instructions:** "(Luna + Haiku + Gemini + Jev) x2, and just do both directions in the x2".  [`judge_instruction_grid.py`](../../scripts/judge_instruction_grid.py) `--design rounds` runs this by default: two passes, rounds 0-9, each judge in its column of Roger's direction table, the second pass the first reversed.  The blend is the same, **Jev 12**.
- **Jev's weight, 12 in both modes, is chosen for resilience, not fitted to the mean** (Roger, on the fitted 24: "That's not an ensemble, that's an addendum"; § "Jev's weight: 12, for resilience").  The mean-ρ fits, which gave 24, are in § "The description blend" and § "The instruction blend"; 24 was the default for a few hours on 2026-11-04.

## How Jev judges

Jev answers typed questions about one `state` per request.
- **Billing:** the state is billed once per request.  Every question sees all of it and is answered on its own.  The price is $0.042 per million input tokens, and output is free.
- **Limits:** 64k tokens a request and 32k a question, with no cap on the number of questions.
- **No options:** there is no temperature, batch mode or Flex tier.  The API has `/v1/models` and `/v1/systemone` only.

[`judge_jev_entities.py`](../../scripts/judge_jev_entities.py) builds one request per batch.  The state holds the axis (its two ends and their descriptions) and up to 160 entries, numbered, each with its name, kind and text.  There is one question per entry:

> Entry k, *name* (*kind*): consider how the person described typically behaves, what they value and what they are associated with, and the same for the other entries in the state.  Is entry k nearer the *second pole* end of the axis than most of the other entries are?

The value is 100 × P(yes).

- **Batches:** they are `judge_tier_cost_eval.axis_groups`, so a seed gives the same entities as the trio's, cut into groups of 160.
- **Directions:** a swapped call names the positive pole first and its value is mapped back (100 − v).  Descriptions use the trio's coin per axis.  Instructions use Jev's column of Roger's table, `JEV_INSTRUCTION_COLUMN` = B A B A d.  With the trio's columns, each of the first four instruction positions is then judged 2:2.
- **The answer log:** `roger/axis_judge_experiments/_tier_eval_jev/<label>/answers.jsonl` keeps every reply whole, probabilities included.  So other ways of reading a score cost nothing, and a stopped run resumes.
- **Cells:** they are judge_tier_cost_eval's, `<axis>_tier_eval/jev-1.14.0@<label>/`.  Descriptions are labelled `jd_noul_relative_n160d<seed>`; instructions `ji_noul_relative_n160d<round>`.
- **The version is pinned.**  The tool requests `jev-1.14.0` by default and writes no cell if any answer comes from another version.  `GET /v1/models` lists only the aliases `jev-latest` and `jev-preview`, but TypeSafe's [models page](https://docs.typesafe.ai/models.md) says versioned IDs "are accepted by the `model` field whether or not they appear in the list".  Checked live on 2026-11-04: `jev-1.14.0` and `jev-1.13.0` each answer as themselves, and an unknown ID is an HTTP 400.  An alias "moves when a new release ships", so it is for trials only.  The page gives no retirement policy for old versions; `jev-1.13.0` was still served three weeks after 1.14 shipped.  The balanced logs of 2026-11-04 were requested as `jev-latest`, and every answer in them came from `jev-1.14.0`.  The tool resumes such a log under the pinned name, and the cells then record both names in `model_requested`.  From 4e806584 to the fix that followed it, the drivers requested `jev-latest`, on the mistaken belief that versions could not be pinned.

## What was tried (dev30, one pass, own order, the default variant)

| Arm | ρ alone | Gain as 4th judge, weight 3 | $/axis |
|---|---:|---:|---:|
| Luna, Haiku, Gemini, one pass each | 0.684, 0.665, 0.663 | | 0.011, 0.010, 0.017 |
| Jev Score, 5 levels, expected level | 0.671 | −0.0006 | 0.0063 |
| ditto, read as the probability of the second pole's side | 0.681 | +0.0001 | 0.0063 |
| Jev Score, 10 levels, probability of side | 0.690 | +0.0020 | 0.0096 |
| Jev Choice, 11 or 21 options | 0.662, 0.686 | −0.0009, +0.0006 | 0.013, 0.026 |
| Jev Noul, one entity a request | 0.688 | +0.0024 | 0.013 |
| Jev Noul, 20 a request | 0.701 | +0.0040 | 0.0040 |
| Jev Noul, three questions an entry ("strong") | 0.695 | +0.0013 | 0.0084 |
| Jev, pairwise ("is j nearer than k?", every pair of 20) | 0.700 | +0.0081 | 0.027 |
| **Jev Noul relative, 160 a request** | **0.724** | **+0.0087 [+0.0056, +0.0123]** | **0.0042** |

- **Batching helps.** With the Noul question, 20 a request beats one a request by +0.013, at a third of the cost.
- **The relative wording grows with batch size:** 0.698 at 10, 0.711 at 20, 0.717 at 40, 0.721 at 80, 0.724 at 160, and 0.7245 at 290, where it levels off.
- **That choice was made on dev30 among about 20 arms, and it did not replicate.** On holdout31 the relative arm at 160 equals the base Noul arm at 20 alone: −0.001 on descriptions and −0.002 on instructions.  Its ensemble edge is only +0.001.  Figures pooled over all 61 axes therefore flatter it.
- **Instructions, one pass, own order:** relative 160 gives 0.753 on dev30 and 0.796 on holdout31, against Luna's 0.743 and 0.790.

## Held out and balanced

On **holdout31**, own order, base Noul at 20:

| | Descriptions | Instructions |
|---|---:|---:|
| Jev minus Luna, one pass each | +0.032 [+0.014, +0.049] | +0.007 [−0.007, +0.021] |
| Gain as 4th judge | +0.006 [+0.004, +0.009] | +0.004 [+0.002, +0.007] |

On **all 61 axes, balanced**, relative 160 against Luna's balanced passes:

| | Descriptions | Instructions |
|---|---:|---:|
| One pass: Jev vs Luna | 0.750 vs 0.711 (+0.040 [+0.029, +0.050]) | 0.782 vs 0.771 (+0.010 [+0.002, +0.018]) |
| Two passes: Jev vs Luna | 0.758 vs 0.735 (+0.024 [+0.014, +0.032]) | |
| Moral axes, own order minus balanced: Luna, Jev | +0.013, −0.001 | +0.008, +0.000 |

Jev has no moral-valence lift, so balancing costs it nothing.

## Consistency, and what a second pass buys (descriptions, 61 axes)

| Two passes of the same judge | Jev | Luna |
|---|---:|---:|
| Exact rerun: agreement (share of answers identical) | 0.997 (34%) | 0.941 (18%) |
| Other batches, same order | 0.976 | 0.887 |
| Opposite order | 0.917 to 0.930 | 0.854 to 0.887 |
| Gain from a rerun averaged in | +0.0003 | +0.010 |
| Gain from a second pass, same order | +0.003 | +0.019 |
| Gain from a second pass, opposite order | +0.010 [+0.001, +0.020] | +0.022 |

- **Jev is nearly deterministic.**  Its main source of variation is pole order, so its second pass is worth having in the opposite direction, which Roger's rule requires anyway.
- **Instructions:** a balanced one-pass Jev already alternates directions over the five instructions.  So 2x adds only +0.0009 over 1x on its own.

## The description blend: 4 : 3 : 3 : X

The cells are all four judges, two balanced passes each, on the 61 axes.  Each judge's passes are averaged, then blended 4:3:3:X.  X is fitted on ρ averaged over the 12 variants, as Roger asked.

| X | 3 | 8 | 12 | 16 | 20 | **24** | 32 | 48 | 128 | Jev alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ρ, mean of 12 | 0.7014 | 0.7070 | 0.7086 | 0.7093 | 0.7096 | **0.7097** | 0.7095 | 0.7088 | 0.7068 | 0.7045 |

- **Plateau:** X = 24 is the peak, and 16 to 40 lie within 0.0005 of it.
- **Bootstrap:** resampling the axes, the best X falls between 16 and 40 in 95% of cases (most often 24).
- **Out of sample:** fitting either half and scoring the other picks 24.  Leaving any one axis out picks 24.
- **Against X = 3:** 24 is +0.0083 [+0.0051, +0.0113].
- **Free weights:** fitting all four weights freely gains nothing on held-out halves.  The best free fit is 3 : 1.5 : 0 : 12, which equals 4:3:3:24 on all 61.

At 24 Jev carries 71% of the description blend by weight.  Alone, the judges at 2x balanced score Jev 0.7045, Luna 0.6784, Haiku 0.6673 and Gemini 0.6626.

## The instruction blend

The cells are all four judges, two passes each, on the 61 axes: rounds 0-9, the trio's `<judge>@i1_p0n20d0..d9` (judged 2026-11-04, $21.69) and Jev's `@ji_noul_relative_n160d0..d9`.  Each judge sees each instruction once each way.  Each judge's ten rounds are averaged, then blended 4:3:3:X, with X fitted on ρ averaged over the 12 variants, as for descriptions.

| X | 0 (the trio) | 3 | 8 | 12 | 16 | **20** | 24 | 32 | 48 | 128 | Jev alone |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ρ, mean of 12 | 0.7202 | 0.7257 | 0.7290 | 0.7299 | 0.7301 | **0.7302** | 0.7301 | 0.7299 | 0.7293 | 0.7276 | 0.7259 |

- **Plateau:** X = 20 is the peak, and 12 to 40 lie within 0.001 of it.  24 is 0.0000 below it ([−0.0003, +0.0002]).
- **Bootstrap:** resampling the axes, the best X falls between 12 and 40 in 95% of cases (24 most often, 30%; 16 and 20 24% each).
- **Out of sample:** fitting the development half picks 16, the held-out half 24, and leaving any one axis out picks 16 to 24.  Over 500 random half splits, the out-of-sample gain over the trio alone is +0.0100 at X = 16 or 24 and +0.0055 at X = 3.
- **Against X = 3:** 24 is +0.0044 [+0.0014, +0.0074].
- **Free weights:** the best free fit on all 61 is 5 : 1.5 : 0 : 12 (0.7309 against 0.7302); fitted on either half it scores −0.0001 and +0.0010 on the other, so nothing to choose.

**Chosen at first: 24, as for descriptions.**  Nothing separates it from the peak, and one weight for both modes is simpler.  Replaced the same day by 12 in both modes (next section).  Alone, the judges at 2x balanced score Jev 0.7259, Luna 0.7152, Haiku 0.7045 and Gemini 0.7041.

The arms at the default variant (s6_l25_raw), from [`judge_instruction_12way.png`](../../roger/axis_judge_experiments/instruction_judging_12way/judge_instruction_12way.png), balanced, the blends at Jev 12: a 2x takes each judge's passes one each way, a 1x alternates directions over an entity's instructions (Roger's table):

| judges (61 axes, balanced) | ρ | $ per axis | on the frontier, of 12 variants |
|---|---:|---:|---:|
| Luna, 1 pass | 0.768 | 0.050 | 0 |
| Luna + Haiku + Gemini, 1 pass each (the default to 2026-11-04) | 0.778 | 0.178 | 0 |
| Luna + Haiku + Gemini, 2 passes each | 0.780 | 0.356 | 0 |
| Jev, 1 pass | 0.781 | 0.021 | 12 |
| Jev, 2 passes | 0.783 | 0.043 | 8 |
| Luna + Haiku + Jev, 1 pass each | 0.788 | 0.119 | 9 |
| Luna + Haiku + Jev, 2 passes each | 0.789 | 0.239 | 10 |
| **Luna + Haiku + Gemini + Jev, 2 passes each (the default)** | **0.789** | **0.398** | **2** |

At Jev 24 the last three rows were 0.787, 0.788 and 0.789, on the frontier in 9, 10 and 5 variants, and Jev 2x was on it in 10.  As on descriptions, Gemini adds nothing beside Jev at the default variant: the default is 0.7887 against 0.7891 for Luna + Haiku + Jev at 2x, for $0.16 more an axis; it stays the standard by Roger's rule of every model we have (the resilience it is kept for is about the worst axes, below).  On the October corpus (1,270 entities an axis, instructions 40 tokens against 24) the default costs roughly $0.95 an axis, about $190 for 200 axes (the trio's $0.42 an axis at 1x, doubled, plus Jev scaled from the May corpus; an estimate, not a dry run).

## Jev's weight: 12, for resilience (Roger, 2026-11-04)

The mean-ρ fits above made Jev 71% of the blend.  Roger: "4:3:3:24 ? That's not an ensemble, that's an addendum", and "With the other models, they tend to have good and bad axes, which vary between models: part of the reason for the ensemble was resilience."  (His first reaction to 12 was that it "isn't that resilient" either; the tests below are what he asked for next.)  So the weight is judged here by its worst axes as well as its mean.  After them: "best so far is a simple 4:3:3:12 ensemble" and "Switch over".  **Jev's weight is 12 in both modes since then** (`judge_score_combine.TIER_JUDGE_WEIGHTS_BY_MODE`).

Every number in this section comes from [`jev_weight_resilience.py`](../../scripts/jev_weight_resilience.py), on the same balanced cells (each judge's two passes, one each way), 61 axes, ρ averaged over the 12 variants; the numbers are in [`jev_weight_resilience.json`](../../roger/axis_judge_experiments/jev_weight/jev_weight_resilience.json).  An axis's **shortfall** is how far a blend falls below the better of Jev alone and the trio (Luna 4 : Haiku 3 : Gemini 3) on that axis: what an ensemble is for is never doing much worse than its better part.

| 4:3:3:X | Descriptions: mean ρ, vs 24 [95%] | axes below the trio | worst shortfall | Instructions: mean ρ, vs 24 [95%] | axes below the trio | worst shortfall |
|---|---|---:|---:|---|---:|---:|
| 3 | 0.7014, −0.0083 [−0.0113, −0.0051] | 2 | 0.056 | 0.7257, −0.0044 [−0.0074, −0.0014] | 5 | 0.067 |
| 8 | 0.7070, −0.0027 [−0.0045, −0.0007] | 3 | 0.035 | 0.7290, −0.0011 [−0.0029, +0.0007] | 10 | 0.045 |
| **12** | **0.7086, −0.0010 [−0.0022, +0.0002]** | **4** | **0.026** | **0.7299, −0.0003 [−0.0014, +0.0008]** | **10** | **0.036** |
| 14 | 0.7091, −0.0006 [−0.0015, +0.0003] | 4 | 0.030 | 0.7300, −0.0001 [−0.0010, +0.0008] | 11 | 0.034 |
| 16 | 0.7093, −0.0003 [−0.0010, +0.0004] | 4 | 0.033 | 0.7301, −0.0000 [−0.0007, +0.0007] | 11 | 0.037 |
| 24 | 0.7097 | 5 | 0.043 | 0.7301 | 12 | 0.048 |
| 32 | 0.7095, −0.0002 [−0.0007, +0.0003] | 6 | 0.052 | 0.7299, −0.0003 [−0.0008, +0.0001] | 12 | 0.055 |

- **12 has the smallest worst shortfall on descriptions, and 14 on instructions** (0.034 against 0.036): one weight for both modes, as before.  Its mean is within noise of 24's in both.
- **The weight in effect.**  A weighted mean of raw scores gives each judge its weight times its spread.  At 24, Jev's share is 0.67 (descriptions) and 0.69 (instructions), and the blend ranks entities as Jev alone does (Spearman 0.992 and 0.995).  At 12 its share is 0.51 and 0.52, and Luna, Haiku and Gemini each agree with the blend better (0.959, 0.943, 0.948 on descriptions, against 0.947, 0.934, 0.937).

**Jev's bad axes are one class.**  Jev trails the trio on 12 description axes and 17 instruction axes.  The worst are style axes, about how someone expresses themselves or works:

| axis | Descriptions: Jev minus trio | Instructions |
|---|---:|---:|
| [precise](../../data/traits/instructions/precise.json) vs [vague](../../data/traits/instructions/vague.json) | −0.105 | −0.105 |
| [proactive](../../data/traits/instructions/proactive.json) vs [reactive](../../data/traits/instructions/reactive.json) | −0.080 | −0.067 |
| [concise](../../data/traits/instructions/concise.json) vs [verbose](../../data/traits/instructions/verbose.json) | −0.068 | −0.042 |
| [superficial](../../data/traits/instructions/superficial.json) vs [thorough](../../data/traits/instructions/thorough.json) | −0.037 | −0.027 |
| [accessible](../../data/traits/instructions/accessible.json) vs [esoteric](../../data/traits/instructions/esoteric.json) | −0.008 | −0.034 |
| [opaque](../../data/traits/instructions/opaque.json) vs [transparent](../../data/traits/instructions/transparent.json) | −0.011 | −0.028 |
| [practical](../../data/traits/instructions/practical.json) vs [theoretical](../../data/traits/instructions/theoretical.json) | −0.019 | −0.018 |
| [quantitative](../../data/traits/instructions/quantitative.json) vs [qualitative](../../data/traits/instructions/qualitative.json) | −0.008 | −0.018 |

- **They are properties of the axes, not noise.**  On instructions the gap on pass 1 predicts the gap on pass 2 across the axes (Pearson 0.96; 15 axes below on both).  On descriptions the passes agree less (0.44), because a description pass is one direction and Jev's direction effect differs by axis; precise vs vague is −0.085 and −0.107.
- **Jev is still the judge with the fewest bad axes.**  Alone, it trails the mean of the other three alone on 5 description axes and 12 instruction axes; Luna on 34 and 34, Haiku on 35 and 35, Gemini on 48 and 43.
- **Jev's best axis**, by the same measure, is [ecocentric](../../data/traits/instructions/ecocentric.json) vs [anthropocentric](../../data/traits/instructions/anthropocentric.json): +0.080 and +0.094.

**The gaps are fat-tailed.**  Per axis, the gap's standard error comes from a paired bootstrap over the axis's entities (1,000 resamples, the same for both judges and all 12 projections); the conservative check adds the judge noise of the pass split (0.020 on descriptions, an upper bound inflated by the direction effect; 0.004 on instructions).

| | Descriptions | Instructions |
|---|---|---|
| Spread of the gaps across axes (sd); median standard error | 0.030; 0.008 | 0.028; 0.006 |
| Cochran's Q on 60 df: the bootstrap's noise; the conservative noise | 412, p 5e-54; 97, p 0.002 | 585, p 4e-87; 434, p 4e-58 |
| Real spread between axes, τ (conservative) | 0.017 | 0.018 |
| Axes beyond 3 sd after that spread (a normal gives 0.16) | 2 | 3 |
| Excess kurtosis; Shapiro-Wilk p; Student-t df | 3.5; 0.000; 2.2 | 3.4; 0.001; 2.2 |
| The same without the three lowest axes | Q 48 on 57 (p 0.80), τ 0: noise alone | τ 0.015, 1 axis beyond 3 (ecocentric vs anthropocentric, +4.4 sd, on the good side), kurtosis 2.2 |

So beyond a real, roughly normal spread of the per-axis differences, the lower tail is a cluster of style axes: a mixture, not a smooth heavy tail.  A weight fitted to the mean is the right choice when the per-axis differences are normal; with a lower tail confined to one class of axes, the worst axes cost more than the mean shows, and the October corpus's 200 or so axes will hold more style axes of this kind.  With 61 axes the tail shape (the t df above) is rough, but 2 to 3 axes beyond 3 sd against 0.16 is not borderline.

**Alternatives tried, all worse.**  Roger: "Where we have 4 results, can we do better by discarding the outlier and averaging the other 3?", and "Try 1:1:1:1 and 4:3:3:12 ... and 1:1:1:2 for the averaging".

| method | Descriptions: vs 24 [95%] | below the trio | worst shortfall | Instructions: vs 24 [95%] | below the trio | worst shortfall |
|---|---|---:|---:|---|---:|---:|
| **4:3:3:12** | **−0.0010 [−0.0022, +0.0002]** | **4** | **0.026** | **−0.0003 [−0.0013, +0.0009]** | **10** | **0.036** |
| 1:1:1:2 | −0.0040 [−0.0062, −0.0017] | 3 | 0.050 | −0.0020 [−0.0040, +0.0001] | 9 | 0.058 |
| 1:1:1:1 | −0.0081 [−0.0112, −0.0049] | 3 | 0.068 | −0.0045 [−0.0075, −0.0012] | 7 | 0.075 |
| per entity, drop the outlying score, 1:1:1:1 on the rest | −0.0171 [−0.0203, −0.0138] | 29 | 0.077 | −0.0094 [−0.0128, −0.0057] | 33 | 0.080 |
| the same, 1:1:1:2 on the rest | −0.0152 | 27 | 0.061 | −0.0087 | 26 | 0.061 |
| the same, 4:3:3:12 on the rest | −0.0142 | 25 | 0.048 | −0.0083 | 25 | 0.043 |
| the same, 4:3:3:24 on the rest | −0.0139 | 26 | 0.045 | −0.0084 | 24 | 0.036 |
| median of the four | −0.0110 | 10 | 0.068 | −0.0061 | 14 | 0.073 |
| per axis, drop the judge that agrees least with the other three, 4:3:3:12 on the rest | −0.0019 [−0.0042, +0.0006] | 7 | 0.035 | −0.0027 [−0.0062, +0.0007] | 7 | 0.045 |
| the trio alone | −0.0168 | 0 | 0.080 | −0.0099 | 0 | 0.094 |

- **Dropping the outlier is a vote against the best judge.**  The score dropped is Jev's for 29% (descriptions) and 33% (instructions) of entities, against 25% by chance.  Jev is the most accurate judge, and the trio share their errors, so when Jev disagrees with them it is mostly right; the vote removes the one independent judge.  The result falls below the trio alone on 25 to 33 axes.  (With four scores, the score farthest from the mean and the one farthest from the median are always the same, so that variant adds nothing.)
- **Dropping a judge per axis**, chosen by agreement alone (no projections), does drop Jev on [concise](../../data/traits/instructions/concise.json) vs [verbose](../../data/traits/instructions/verbose.json) and [precise](../../data/traits/instructions/precise.json) vs [vague](../../data/traits/instructions/vague.json) in both modes.  But agreement does not tell Jev's bad axes from its best: its agreement with the other three on ecocentric vs anthropocentric (0.899 and 0.929) is as low as on its style axes (0.83 to 0.91 and 0.89 to 0.93).  No threshold on agreement would separate them.

**Caveats.**  The weight is chosen on the same 61 axes it is judged on; it is one parameter on a flat mean, but a worst axis is a single-axis statistic and noisy.  The October corpus is the real test.  Gemini adds nothing beside Jev at the default variant (0.7659 against 0.7664 for Luna + Haiku + Jev 2x on descriptions; 0.7887 against 0.7891 on instructions); it stays by Roger's rule of every model we have.

## Where Jev is strong and weak (one pass each, own order, 61 axes)

Jev's lead over Luna in descriptions predicts its lead in instructions (Spearman +0.68 across the axes), so these are properties of the axes.
- **Ahead in both:** [introverted](../../data/traits/instructions/introverted.json) vs [extroverted](../../data/traits/instructions/extroverted.json) (+0.14 on descriptions), [casual](../../data/traits/instructions/casual.json) vs [formal](../../data/traits/instructions/formal.json), [materialistic](../../data/traits/instructions/materialistic.json) vs [spiritual](../../data/traits/instructions/spiritual.json), [forgiving](../../data/traits/instructions/forgiving.json), [passionate](../../data/traits/instructions/passionate.json), [dependable](../../data/traits/instructions/dependable.json), [obedient](../../data/traits/instructions/obedient.json), [ecocentric](../../data/traits/instructions/ecocentric.json).
- **Behind in both:** [precise](../../data/traits/instructions/precise.json) vs [vague](../../data/traits/instructions/vague.json) and [concise](../../data/traits/instructions/concise.json) vs [verbose](../../data/traits/instructions/verbose.json).  Both are about how someone expresses themselves.
- **No pattern** by kind (roles, traits) or by moral axis.

**Valence.**  On the 43 non-moral axes, Jev's scores follow valence as much as Luna's (|ρ| 0.36 each).  Valence here is the mean raw projection on the 18 good-pole axes.  But Luna adds valence beyond what the whitened projections carry (partial correlation +0.077), as do Gemini (+0.041) and Haiku (+0.021); Jev barely does (+0.010; Jev minus Luna −0.067 [−0.113, −0.024]).  Roger's reading: Jev "is less focused specifically on that aspect of questions that are somewhat entangled with it".

This is also why Jev gains most from soft whitening, which damps the valence-heavy top directions.  Its gain over raw exceeds Luna's by +0.017 to +0.042 at every slot and layer.

## Files

- **The Jev judge:** [`judge_jev_entities.py`](../../scripts/judge_jev_entities.py).  Its tests are [`test_judge_jev_entities.py`](../../scripts/tests/test_judge_jev_entities.py) and the jev_entities tests in [`test_judge_tier_cost_eval.py`](../../results_analysis/tests/test_judge_tier_cost_eval.py).
- **Production:** [`judge_descriptions.py`](../../results_analysis/judge_descriptions.py) (test [`test_judge_descriptions_jev.py`](../../results_analysis/tests/test_judge_descriptions_jev.py)) and [`judge_instruction_grid.py`](../../scripts/judge_instruction_grid.py) (test [`test_judge_instruction_grid_jev.py`](../../scripts/tests/test_judge_instruction_grid_jev.py)).
- **Weights:** [`judge_score_combine.py`](../../assistant_axis/judge_score_combine.py), `TIER_JUDGE_WEIGHTS_BY_MODE`.  The evidence for 12: [`jev_weight_resilience.py`](../../scripts/jev_weight_resilience.py) (test [`test_jev_weight_resilience.py`](../../scripts/tests/test_jev_weight_resilience.py)), its numbers in [`jev_weight_resilience.json`](../../roger/axis_judge_experiments/jev_weight/jev_weight_resilience.json); no API calls, about ten minutes.
- **The figures:** [`judge_instruction_12way.py`](../../scripts/judge_instruction_12way.py), drawing [`judge_description_12way.png`](../../roger/axis_judge_experiments/instruction_judging_12way/judge_description_12way.png) and [`judge_instruction_12way.png`](../../roger/axis_judge_experiments/instruction_judging_12way/judge_instruction_12way.png).  They use balanced cells by default; `--directions own` draws the own-order ones.
- **Run logs and drivers:** [`jev_entity_logs/`](../../roger/axis_judge_experiments/jev_entity_logs/).
- **Spend:** about $16 of Jev credit, and about $27 of trio judging for the balanced cells.
