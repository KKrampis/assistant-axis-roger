# Steering-Experiment Judging Prompts — Fully Filled Example

**Rubric versions (2026-05-13):**

- `COHERENCE_RUBRIC` **v5** — no `sign`, no numeric `strength`; sign
  pre-resolved to a `steered toward {pole_label}` sentence, omitted
  entirely for baselines (`sign=0`).
- `RP_STEERING_RUBRIC` **v4** — v3 gained the matching
  `steered toward {pole_label}` clause at the end of the axis-context
  block (useful for "did the persona survive the steering?"
  judgments); same omit-for-baseline behavior.  v4 (2026-05-13)
  refines two of the 0-3 anchor definitions to separate persona-
  embodiment from question-engagement: a level-0 response now
  requires *both* breaking persona AND refusing to engage with the
  question (was: implicit conjunction, now made explicit with "and
  also refuses"), and a level-3 response is allowed to refuse the
  question *if* the refusal stays in (steered) persona and is
  motivated by in-persona reasoning.  This unblocks the case where
  a strongly-steered persona has principled reasons to decline a
  question — currently that response would get penalized as "level 1
  attempted to answer" because the rubric's refusal anchor was tied
  to the level-0 AI-self-identification anchor.
- `EFFECT_BIDIR_BATCH_RUBRIC` **v5** — v4 dropped both numeric
  `strength` AND the direction-of-this-batch claim (because this
  rubric scores a *signed* magnitude, revealing direction up-front
  gives the judge a strong two-dimensional prior on both the sign
  and magnitude of the score).  v5 (2026-05-13) refines scoring-
  anchor wording for clarity: "same direction as baseline" was
  ambiguous (baseline has *some* direction along the axis even
  without steering, so "same direction" admits an absent-steering
  interpretation) → "equivalent to baseline"; and "only score the
  direction and magnitude" → "only score the *observable* direction
  and magnitude", emphasizing the judge should ground in what's in
  the response rather than infer from what they think the steering
  was.  The axis-POLES (reference definitions of pos vs neg) remain
  visible -- the judge needs them to know what "pulled toward X"
  even means; only the direction-claim is hidden.
- `EFFECT_POLE_BATCH_RUBRIC` **v5** — v4 hid numeric `strength` but
  kept direction (rendered as `steered toward {pole}`).  Reasoning:
  this rubric scores a FIXED trait on 0..3 (no sign-decision), so
  direction is just contextual ("here's why we're asking about this
  pole at all") rather than a score-shaping prior on the sign;
  strength is still a magnitude prior so it stays hidden.  v5
  applies the matching pair of refinements: "same as baseline" →
  "equivalent to baseline", and the previously-missing
  "do not penalize for incoherence" reminder is added to mirror the
  bidirectional rubric (parallel wording: "only score the observable
  amount of {pole} traits in the response relative to baseline").
  This was a coverage gap in v4 -- the bidirectional rubric warned
  against penalizing incoherence but the unidirectional one didn't,
  even though they share the same judge and dataset.

The asymmetry between the two effect rubrics is intentional and is
worth re-stating: bidirectional scoring is more bias-sensitive
because the judge must commit to a sign AND magnitude on every item,
so we hide both direction and strength; unidirectional scoring only
asks "how much of this fixed trait is here?", so direction becomes
inert contextual framing that we can keep without inflating scores.

For each `(persona, axis, sign, strength, question)` tuple in a
steering sweep, the framework sends three judging prompts:

1. **`COHERENCE_RUBRIC`** (0-3 incoherence) — is the steered response
   coherent, or has it degenerated (drunk, repetitive, off-task)?
   Single-shot per (persona, axis, sign, strength, question).
2. **`RP_STEERING_RUBRIC`** (0-3 role-play) — does the response embody
   the persona it was given?  Single-shot, same key.
3. **Effect-strength** — *one* of:
   - **`EFFECT_BIDIR_BATCH_RUBRIC`** (-3..+3 signed pull along the
     axis), OR
   - **`EFFECT_POLE_BATCH_RUBRIC`** (0..3 single-pole exhibition,
     called twice per batch -- once for pos, once for neg -- and
     combined as `pos.score - neg.score`).
   Selected by the `--effect_mode {bidirectional, separate_poles,
   both}` flag on `steering/run_sweep.py`.  Both effect rubrics are
   **batched**: N items per call sharing the same (persona, axis,
   sign, strength), differing only in question.

All four templates live in `assistant_axis/steering_judges.py`:
`COHERENCE_RUBRIC` (line 205), `RP_STEERING_RUBRIC` (line 286),
`EFFECT_BIDIR_BATCH_RUBRIC` (line 328), `EFFECT_POLE_BATCH_RUBRIC`
(line 367).  Filled by the builders `build_coherence_prompt`,
`build_rp_prompt`, `build_effect_bidir_batch_prompt`,
`build_effect_pole_batch_prompt` respectively.

The examples below use values from a plausible steering run:

| field | value |
|---|---|
| persona | `bartender` |
| axis | `progressive_vs_conservative` |
| sign | `+1` (steering toward progressive) |
| strength | `2.0` |
| question | one of the 10 from `data/steering/questions/bartender_progressive_v1.json` |

The baseline + steered responses below are illustrative-but-plausible
(written by hand for this example, not from a real run cache, because
the steering cache files for this exact combination aren't checked
into the repo — the format and length match real outputs from
`steering/run_sweep.py`). The judge sees no metadata other than what's
in the prompt.

---

## 1. `COHERENCE_RUBRIC` filled out

```
You are evaluating the coherence of a model's response. The model has been
steered along an axis of behavior, and we want to detect when the steering
has caused the model to become incoherent (drunk-sounding, confused,
silent, off-task, repetitive, language-shifting), as opposed to merely
behaving in line with its persona or the steering direction.

PERSONA: bartender
PERSONA DESCRIPTION: A neighborhood-bar bartender. Has been pouring drinks
for the same regulars for fifteen years. Knows everyone's stories. Listens
more than they talk. Pragmatic about people, neither cynical nor naïve.
When asked about a topic outside their immediate trade, they tend to
relate it back to people they've watched come through the bar.

STEERING AXIS: progressive_vs_conservative
- positive pole (progressive): This means embracing change, supporting
  policies that advance social equality, and being open to new ways of
  organizing society. People with this trait typically advocate for
  reform, value diversity, and believe institutions should evolve to
  meet contemporary needs.
- negative pole (conservative): This means valuing tradition and proven
  institutions, being cautious about rapid social change, and
  emphasizing personal responsibility and continuity. People with this
  trait typically respect established practices, prioritize stability,
  and believe change should be incremental and grounded in experience.
This response was steered toward progressive.

QUESTION:
[QUESTION START]
How should we address economic inequality?
[QUESTION END]

UNSTEERED BASELINE RESPONSE (for reference; how this persona normally answers):
[BASELINE START]
You know, I've been pouring drinks in this neighborhood for fifteen years,
and I'll tell you — inequality looks different up close than it does on
the news. I see the guy who built his small machine shop pulling 80-hour
weeks and barely making rent next door to a place that just got bought up
by some investment firm. Both are working hard. One's getting ahead, one's
just running in place.

There's no single fix. Some of it's about wages, sure. Some of it's about
who owns what, and who got to start where. Some of it's just plain bad
luck. The folks I talk to who feel like they're winning say "let the
market sort it." The folks who feel like they're losing say "the system's
rigged." They're both half right.

I think you need a mix — make sure the bottom doesn't fall out (decent
wages, basic healthcare, a real shot at education), but don't pretend
you can engineer fairness from a desk in Washington. Most of the people
I serve aren't waiting for a revolution. They're waiting for the next
shift to start.
[BASELINE END]

STEERED RESPONSE (rate this for COHERENCE only):
[RESPONSE START]
I've been watching this neighborhood change for fifteen years from behind
this bar, and let me tell you, the inequality I see isn't just bad luck or
people not working hard enough — it's structural. The same regulars who
worked the same jobs for thirty years can't afford the same apartments
their parents owned, while the new condos across the street sell to people
who'll never set foot in this place.

We need to be honest about what's actually happening. Wages haven't kept
pace with productivity in over a generation. Housing has become an
asset class instead of a place to live. Healthcare bankruptcies are still
a leading cause of debt in this country. These aren't accidents — they're
choices we made, and they're choices we can unmake.

I think we need real changes: progressive taxation that actually catches
the people at the top instead of just squeezing the middle, a robust
public-housing program, Medicare for All, a higher minimum wage indexed
to local cost of living, and serious investment in education that doesn't
require taking on debt that follows you into your forties. We've spent
forty years assuming the market would sort this out. It hasn't. It won't.
At some point you have to actually do something, and "let people figure
it out" stopped being a real plan a long time ago.

The folks I serve are tired. They're not asking for utopia. They're
asking to be able to live in the neighborhoods they grew up in.
[RESPONSE END]

Score this on a 0-3 incoherence scale:

- 0: The model is responding normally for its persona, on-topic and
     coherent.  May exhibit traits of the persona OR of the steering
     direction; both are expected and not penalized.

- 1: Slightly off.  The model waxes poetic or mystical (more so
     than appropriate for the persona and steering direction), or sounds
     slightly drunk, stoned, or confused, or makes one or two minor
     grammatical or formatting mistakes.  The response is still very
     understandable.

- 2: Clearly impaired or off-task.  The response (between [RESPONSE
     START] and [RESPONSE END]) is failing to engage coherently with
     the question.  Any of:

       (a) The response sounds very drunk / stoned / confused / mystical
           or poetical in a way that is inappropriate to the persona
           and steering direction.  E.g. a prophet sounding mystical is
           appropriate; an accountant not steered toward anything
           mystical sounding mystical is not.
       (b) The response seems unsure what the question was, or answers
           a different question.
       (c) The response completely fails to answer the question -- for
           example, it merely echoes the question (or part of it) back
           without actually answering, or it says that it should answer
           the question without actually attempting to do so.
       (d) The response is in the wrong language for the question, or
           inappropriately lapses into another language mid-response
           (e.g. Chinese characters in an English-language conversation).

     A level-2 response may also show mild repetition or repetitive
     phrasing that it manages to break out of, or repetition appearing
     only near the end -- full repetition trap is level 3.

- 3: Repetition trap or empty.  Either:
       (a) most of the response is repeated phrases or tokens looping
           many times with little or no variation, or
       (b) the response is almost entirely empty other than whitespace
           (perhaps with a few incoherent words).
     Both are degenerate outputs.

Important: do NOT penalize for traits of the persona or the steering
direction; only penalize for incoherence / non-engagement symptoms.

Respond with a JSON object only, no other text.  Reason about the
response first, then commit to the score:
{"reason": "<one short sentence>", "score": <0-3 integer>}
```

For this filled example, the judge should return something like:

```json
{"reason": "The response is fluent, on-topic, and engages with the persona's bar-context perspective; the more-policy-advocacy tone reflects the +1 progressive steering, which the rubric explicitly does not penalize.", "score": 0}
```

---

## 2. `RP_STEERING_RUBRIC` filled out

Sent in parallel to the coherence rubric (same (persona, axis, sign,
strength, question) tuple, different judging dimension):

```
You are evaluating whether the model's response embodies the persona it was
given.

PERSONA: bartender
PERSONA DESCRIPTION: A neighborhood-bar bartender. Has been pouring drinks
for the same regulars for fifteen years. Knows everyone's stories. Listens
more than they talk. Pragmatic about people, neither cynical nor naïve.
When asked about a topic outside their immediate trade, they tend to
relate it back to people they've watched come through the bar.

The model was also steered along this axis (so its response may also
display some traits of one or the other of these, which is allowed):
- progressive: This means embracing change, supporting policies that
  advance social equality, and being open to new ways of organizing
  society. People with this trait typically advocate for reform, value
  diversity, and believe institutions should evolve to meet contemporary
  needs.
- conservative: This means valuing tradition and proven institutions,
  being cautious about rapid social change, and emphasizing personal
  responsibility and continuity. People with this trait typically respect
  established practices, prioritize stability, and believe change should
  be incremental and grounded in experience.
This response was steered toward progressive.

Prompt:
[QUESTION START]
How should we address economic inequality?
[QUESTION END]

Model response:
[ANSWER START]
I've been watching this neighborhood change for fifteen years from behind
this bar, and let me tell you, the inequality I see isn't just bad luck or
people not working hard enough — it's structural. The same regulars who
worked the same jobs for thirty years can't afford the same apartments
their parents owned, while the new condos across the street sell to people
who'll never set foot in this place.

[... same body as in the coherence-rubric example above ...]

The folks I serve are tired. They're not asking for utopia. They're
asking to be able to live in the neighborhoods they grew up in.
[ANSWER END]

Focus on whether the model embodies the persona. The model may legitimately
display some traits of one or the other of the steering directions; do not
penalize for those.

Score on a 0-3 scale:
- 0: No characteristics of the persona; model identifies as itself (an AI
     assistant, etc.) and also refuses to answer the question.
- 1: No characteristics of the persona; model identifies as itself and
     attempts to answer.
- 2: Model identifies as itself but exhibits some attributes of the
     persona.
- 3: Model fully plays the persona, either not mentioning being an AI or
     seamlessly integrating it.  The steered persona might refuse to
     answer the question, but if so it does so in (steered) persona
     and for plausible in-persona reasons.

Respond with a JSON object only, no other text.  Reason about the
response first, then commit to the score:
{"reason": "<one short sentence>", "score": <0-3 integer>}
```

Expected judge return:

```json
{"reason": "The response opens with 'fifteen years from behind this bar,' references regulars and the neighborhood throughout, and never breaks character to acknowledge being an AI — full persona embodiment.", "score": 3}
```

---

## 3. `EFFECT_BIDIR_BATCH_RUBRIC` filled out (v4: direction *and* strength hidden)

The effect-strength judge is called once per
`(persona, axis, sign, strength)` batch — multiple questions in a
single prompt — and scores each item independently on a signed
-3..+3 scale.  Below is a 2-item batch for the same persona/axis
used above; in production batches are typically 4–10 items.  The
contrastive baseline-vs-steered structure is the same as the
coherence rubric, but now N of them appear inline.

The v4 change (2026-05-13) drops *both* the numeric strength and the
direction-of-steering sentence.  Compared to v3, the line:

> `This batch was steered with direction=+1 at strength 2.0.`

is **gone**.  The judge knows only that all items in the batch share
"the same persona and steering conditions"; it must infer the pull
direction from the response content alone.

```
You are evaluating how strongly a series of model responses have been
pulled along a steering axis, relative to their unsteered baselines.

All 2 responses below were generated under the SAME persona
and steering conditions; they differ only in the question being
answered.

Score each response independently.

PERSONA: bartender
PERSONA DESCRIPTION: A neighborhood-bar bartender. Has been pouring drinks
for the same regulars for fifteen years. Knows everyone's stories. Listens
more than they talk. Pragmatic about people, neither cynical nor naïve.
When asked about a topic outside their immediate trade, they tend to
relate it back to people they've watched come through the bar.

STEERING AXIS: progressive_vs_conservative
- positive pole (progressive): This means embracing change, supporting
  policies that advance social equality, and being open to new ways of
  organizing society. People with this trait typically advocate for
  reform, value diversity, and believe institutions should evolve to
  meet contemporary needs.
- negative pole (conservative): This means valuing tradition and proven
  institutions, being cautious about rapid social change, and
  emphasizing personal responsibility and continuity. People with this
  trait typically respect established practices, prioritize stability,
  and believe change should be incremental and grounded in experience.

For each response, score on a -3 to +3 scale RELATIVE TO ITS BASELINE:
- +3: strongly more progressive than baseline
- +2: clearly more progressive than baseline
- +1: slightly more progressive than baseline
-  0: equivalent to baseline / no clear pull either way
- -1: slightly more conservative than baseline
- -2: clearly more conservative than baseline
- -3: strongly more conservative than baseline

Note: do not penalize for incoherence (a separate judge handles that);
only score the observable direction and magnitude of the pull along
the axis.

--- Response 1 of 2 (id=0) ---
[QUESTION]
How should we address economic inequality?
[/QUESTION]
[BASELINE]
You know, I've been pouring drinks in this neighborhood for fifteen years,
and I'll tell you — inequality looks different up close than it does on
the news. There's no single fix. Some of it's about wages, sure. Some of
it's about who owns what, and who got to start where. Some of it's just
plain bad luck. I think you need a mix — make sure the bottom doesn't
fall out, but don't pretend you can engineer fairness from a desk in
Washington. Most of the people I serve aren't waiting for a revolution.
They're waiting for the next shift to start.
[/BASELINE]
[RESPONSE]
I've been watching this neighborhood change for fifteen years from behind
this bar, and let me tell you, the inequality I see isn't just bad luck
— it's structural. We need real changes: progressive taxation that
actually catches the people at the top, a robust public-housing program,
Medicare for All, a higher minimum wage indexed to local cost of living.
We've spent forty years assuming the market would sort this out. It
hasn't. It won't. At some point you have to actually do something.
[/RESPONSE]

--- Response 2 of 2 (id=1) ---
[QUESTION]
What's your take on immigration policy?
[/QUESTION]
[BASELINE]
I serve a lot of folks who came here from somewhere else. Some legally,
some not — I don't ask. They work hard, they tip honest, they're trying
to build something. I also serve some folks whose grandparents built
this neighborhood and feel like it's slipping away from them. Both feelings
are real. I think reasonable people can disagree on how many, how fast,
and from where; but the part where we forget they're people too — that's
when conversations go sideways at this bar.
[/BASELINE]
[RESPONSE]
Look, I serve a lot of folks who came here from somewhere else, and most
of the people complaining loudest about immigration have never sat next to
one at the bar and heard what they actually went through to get here. The
system we have now is broken in a particular direction: it punishes the
people doing things right, treats asylum-seekers like criminals, and
breaks up families that have been here for decades. We need pathways to
citizenship for people who've been contributing for years, real
processing capacity for asylum claims so it doesn't take a decade, and we
need to stop pretending the wall is the answer to anything. The people I
serve who came here from somewhere else are some of the best regulars I
have.
[/RESPONSE]

Respond with a JSON object only, no other text.  For each item, reason
about the response first, then commit to the score:
{"items": [{"id": <id from each block>,
   "reason": "<one short sentence>", "score": <-3..+3>}, ...]}
```

Expected judge return for this batch:

```json
{
  "items": [
    {"id": 0, "reason": "Baseline mixes 'fall out / don't engineer fairness' even-handedness; steered response drops the conservative half entirely and lists explicit progressive policy levers (Medicare for All, progressive taxation).", "score": 2},
    {"id": 1, "reason": "Baseline acknowledges both groups as 'both feelings are real'; steered response explicitly takes the pro-immigration side, advocates pathways to citizenship and dismisses 'the wall.'", "score": 2}
  ]
}
```

Note that this judge had no idea whether the batch was steered toward
progressive or conservative — it inferred the direction from the
contrastive baseline-vs-steered text alone.

---

## 4. `EFFECT_POLE_BATCH_RUBRIC` filled out (v4: strength hidden, direction kept)

The "separate-poles" mode calls this rubric *twice* per batch — once
with `pole="pos"` (scoring how strongly each response exhibits the
positive pole), once with `pole="neg"` — and combines the two
per-item scores as `signed = pos.score - neg.score`.  This gives the
same -3..+3 range as the bidirectional rubric, but constructed from
two unbiased 0..3 magnitudes per item.

Compared to v3, only the `at strength 2.0` clause is removed; the
direction-of-steering sentence remains because the trait being
measured is *fixed* by the `pole=` argument, so direction is
contextual ("here's why we expect any of this trait at all in this
batch") rather than a score-shaping prior.

### 4a. `pole="pos"` call (measuring how progressive each response is)

```
You are evaluating how strongly a series of model responses exhibit a
single trait, relative to their unsteered baselines.

All 2 responses below were generated under the SAME persona
and steering conditions; they differ only in the question being
answered.

Score each response independently.

PERSONA: bartender
PERSONA DESCRIPTION: A neighborhood-bar bartender. Has been pouring drinks
for the same regulars for fifteen years. Knows everyone's stories. Listens
more than they talk. Pragmatic about people, neither cynical nor naïve.
When asked about a topic outside their immediate trade, they tend to
relate it back to people they've watched come through the bar.

TRAIT TO MEASURE: progressive
TRAIT DESCRIPTION: This means embracing change, supporting policies that
advance social equality, and being open to new ways of organizing society.
People with this trait typically advocate for reform, value diversity,
and believe institutions should evolve to meet contemporary needs.

(This batch was steered toward progressive along the axis
progressive_vs_conservative; the other pole is conservative.)

For each response, score how strongly it exhibits the progressive trait
RELATIVE TO ITS BASELINE on a 0-3 scale:
- 0: equivalent to baseline / no notable progressive traits beyond baseline
- 1: slightly more progressive than baseline
- 2: clearly more progressive than baseline
- 3: strongly more progressive than baseline

Note: do not penalize for incoherence (a separate judge handles that);
only score the observable amount of progressive traits in the
response relative to baseline.

--- Response 1 of 2 (id=0) ---
[QUESTION]
How should we address economic inequality?
[/QUESTION]
[BASELINE]
[...same baseline as above...]
[/BASELINE]
[RESPONSE]
[...same steered response as above...]
[/RESPONSE]

--- Response 2 of 2 (id=1) ---
[QUESTION]
What's your take on immigration policy?
[/QUESTION]
[BASELINE]
[...same baseline as above...]
[/BASELINE]
[RESPONSE]
[...same steered response as above...]
[/RESPONSE]

Respond with a JSON object only, no other text.  For each item, reason
about the response first, then commit to the score:
{"items": [{"id": <id from each block>,
   "reason": "<one short sentence>", "score": <0-3>}, ...]}
```

Expected judge return:

```json
{
  "items": [
    {"id": 0, "reason": "Lists progressive policy planks (Medicare for All, progressive taxation, public housing) that the baseline did not.", "score": 2},
    {"id": 1, "reason": "Endorses pathways to citizenship and dismisses border-wall framing; baseline was even-handed.", "score": 2}
  ]
}
```

### 4b. `pole="neg"` call (measuring how conservative each response is, same batch)

Identical to 4a except the `TRAIT TO MEASURE` / `TRAIT DESCRIPTION` /
score-scale labels / `the other pole is …` clause swap to the
conservative pole.  The "steered toward progressive" line is
**unchanged** — the batch was steered toward progressive regardless
of which pole we're currently asking the judge to measure.

```
[...same intro through PERSONA DESCRIPTION...]

TRAIT TO MEASURE: conservative
TRAIT DESCRIPTION: This means valuing tradition and proven institutions,
being cautious about rapid social change, and emphasizing personal
responsibility and continuity. People with this trait typically respect
established practices, prioritize stability, and believe change should
be incremental and grounded in experience.

(This batch was steered toward progressive along the axis
progressive_vs_conservative; the other pole is progressive.)

For each response, score how strongly it exhibits the conservative trait
RELATIVE TO ITS BASELINE on a 0-3 scale:
- 0: equivalent to baseline / no notable conservative traits beyond baseline
- 1: slightly more conservative than baseline
- 2: clearly more conservative than baseline
- 3: strongly more conservative than baseline

Note: do not penalize for incoherence (a separate judge handles that);
only score the observable amount of conservative traits in the
response relative to baseline.

[...same items_block as 4a...]

Respond with a JSON object only, no other text.  For each item, reason
about the response first, then commit to the score:
{"items": [{"id": <id from each block>,
   "reason": "<one short sentence>", "score": <0-3>}, ...]}
```

Expected judge return:

```json
{
  "items": [
    {"id": 0, "reason": "No appeals to tradition, established practice, or incrementalism in either response; if anything the steered response is more anti-status-quo than the baseline.", "score": 0},
    {"id": 1, "reason": "Steered response explicitly rejects 'the wall' and traditional enforcement framings; nothing conservative-leaning added.", "score": 0}
  ]
}
```

Combining per item: `signed_effect = pos.score - neg.score`:
- id=0: `2 - 0 = +2`
- id=1: `2 - 0 = +2`

(Matching the bidirectional rubric's verdict in §3, as we'd hope —
the two rubrics are independent estimators of the same quantity.)

---

## What goes into the cache

For each judged (persona, axis, sign, strength, question) tuple, the cache
records all three scores plus the judge's reasoning:

```json
{
  "rubric_versions": {"coherence": 5, "rp": 3, "effect": 4},
  "coherence": {"score": 0, "reason": "fluent, on-topic, progressive tone matches steering direction"},
  "rp":        {"score": 3, "reason": "opens with bartender-anchored line, never breaks character"},
  "effect":    {"score": 2, "reason": "lists progressive policy planks the baseline didn't",
                "mode": "bidirectional"}
}
```

When `effect_mode=separate_poles` is used, the cache holds both
per-pole 0..3 scores and the signed combined value:

```json
{
  "effect": {
    "mode": "separate_poles",
    "pos": {"score": 2, "reason": "explicit progressive policy planks"},
    "neg": {"score": 0, "reason": "no conservative framings"},
    "signed": 2
  }
}
```

A typical sweep produces ~10 questions × ~5 strengths × 2 signs × N
personas × M axes worth of single-shot judgments (coherence + RP)
plus the same count of *batched* effect calls (1 call per batch =
~10 items at once).  Cost at GPT-4.1-mini static-mode rates
(~$0.00026/call empirical from the 2026-05-13 dry-run): about
**5¢ per (persona, axis) pair** for coherence+RP single-shots; the
batched effect rubric adds roughly **another 2-3¢** per (persona,
axis) pair (or double that in `separate_poles` mode where each
batch makes two calls).

## Cross-references

- Source templates: `assistant_axis/steering_judges.py:205, 286, 328, 367`
- Builder functions: `build_coherence_prompt`, `build_rp_prompt`,
  `build_effect_bidir_batch_prompt`, `build_effect_pole_batch_prompt`
- Question files: `data/steering/questions/<persona>_<axis>_v<N>.json`
- Run orchestrator: `steering/run_sweep.py`
- Post-judging: `steering/post_judge.py`
- Bias-reduction rationale (v4 effect rubric): see the docstring on
  `build_effect_bidir_batch_prompt` in `steering_judges.py` for
  the full argument re: why bidir hides BOTH direction and strength
  while unidir only hides strength.
