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

- **Descriptions:** "the new standard is (Luna + Haiku + Gemini + Jev) x2, with the reversal in the x2".  [`judge_descriptions.py`](../../results_analysis/judge_descriptions.py) runs this by default.  The blend is Luna 4 : Haiku 3 : Gemini 3 : **Jev 24** (`judge_score_combine.TIER_JUDGE_WEIGHTS_BY_MODE`; § "The description blend" below).
- **Instructions:** "(Luna + Haiku + Gemini + Jev) x2, and just do both directions in the x2".  [`judge_instruction_grid.py`](../../scripts/judge_instruction_grid.py) `--design rounds` runs this by default: two passes, rounds 0-9, each judge in its column of Roger's direction table, the second pass the first reversed.  Jev's weight for instructions is in § "The instruction blend".

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

So Jev carries 71% of the description blend.  Alone, the judges at 2x balanced score Jev 0.7045, Luna 0.6784, Haiku 0.6673 and Gemini 0.6626.

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

**Chosen: 24, as for descriptions** (`judge_score_combine.TIER_JUDGE_WEIGHTS_BY_MODE`).  Nothing separates it from the peak, and one weight for both modes is simpler.  Alone, the judges at 2x balanced score Jev 0.7259, Luna 0.7152, Haiku 0.7045 and Gemini 0.7041.

The arms at the default variant (s6_l25_raw), from [`judge_instruction_12way.png`](../../roger/axis_judge_experiments/instruction_judging_12way/judge_instruction_12way.png), balanced: a 2x takes each judge's passes one each way, a 1x alternates directions over an entity's instructions (Roger's table):

| judges (61 axes, balanced) | ρ | $ per axis | on the frontier, of 12 variants |
|---|---:|---:|---:|
| Luna, 1 pass | 0.768 | 0.050 | 0 |
| Luna + Haiku + Gemini, 1 pass each (the default to 2026-11-04) | 0.778 | 0.178 | 0 |
| Luna + Haiku + Gemini, 2 passes each | 0.780 | 0.356 | 0 |
| Jev, 1 pass | 0.781 | 0.021 | 12 |
| Jev, 2 passes | 0.783 | 0.043 | 10 |
| Luna + Haiku + Jev, 1 pass each | 0.787 | 0.119 | 9 |
| Luna + Haiku + Jev, 2 passes each | 0.788 | 0.239 | 10 |
| **Luna + Haiku + Gemini + Jev, 2 passes each (the default)** | **0.789** | **0.398** | **5** |

As on descriptions, Gemini adds little beside Jev: the default is +0.0003 over Luna + Haiku + Jev at 2x, for $0.16 more an axis.  On the October corpus (1,270 entities an axis, instructions 40 tokens against 24) the default costs roughly $0.95 an axis, about $190 for 200 axes (the trio's $0.42 an axis at 1x, doubled, plus Jev scaled from the May corpus; an estimate, not a dry run).

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
- **Weights:** [`judge_score_combine.py`](../../assistant_axis/judge_score_combine.py), `TIER_JUDGE_WEIGHTS_BY_MODE`.
- **The figures:** [`judge_instruction_12way.py`](../../scripts/judge_instruction_12way.py), drawing [`judge_description_12way.png`](../../roger/axis_judge_experiments/instruction_judging_12way/judge_description_12way.png) and [`judge_instruction_12way.png`](../../roger/axis_judge_experiments/instruction_judging_12way/judge_instruction_12way.png).  They use balanced cells by default; `--directions own` draws the own-order ones.
- **Run logs and drivers:** [`jev_entity_logs/`](../../roger/axis_judge_experiments/jev_entity_logs/).
- **Spend:** about $16 of Jev credit, and about $27 of trio judging for the balanced cells.
