# Axis categorization: style vs. substance

A first-pass three-way split of the 61 v2 axes in
[`pair_list_di.json`](./pair_list_di.json) along the
question: **does this axis primarily change how you _talk_, how you
_think_ / what you _do_, or both?**

The categories are defined as:

- **(a) Affect / communication style** — primarily changes _how_
  you express yourself: tone, register, word choice, pacing,
  formality, emphasis, presence/absence of hedging.  A character
  high vs. low on this axis says the same _ideas_ in noticeably
  different ways, but their reasoning, values, and choices about
  what to do don't change much.
- **(b) Thinking / choices / action** — primarily changes _what_
  you decide and how you reason: values, goals, strategies, what
  facts you trust, what you commit to doing.  A character high vs.
  low on this axis arrives at materially different _conclusions_
  or _actions_, but the linguistic surface of their answer could
  be the same across the axis.
- **(c) Genuinely both** — moving along the axis changes both:
  not just _what_ you'd say in a given situation, but _how_ you'd
  phrase it.  These are the axes that the steering work expects
  to have the strongest combined coh × eff response curves
  because both the live response (which constrains style) and
  the underlying choice/values are being pushed.

These categorizations are inherently judgement calls — the labels
below reflect a first-pass classification by the agent against the
pole-pair definitions in
[`data/traits/instructions/`](../../data/traits/instructions/).
Many entries are near the (a)↔(c) or (b)↔(c) borders; the "Notes
on borderline cases" section at the end flags the ones most worth
revisiting.

## The three-way split (61 axes total)

**Bold** entries are the **22 axes** that have response-mode
judging on disk (per
[`pair_list_responses.json`](./pair_list_responses.json), updated
2026-05-22); the rest have desc+inst only.

| (a) Affect / communication style (11) | (b) Thinking / choices / action (33) | (c) Genuinely both (17) |
|---|---|---|
| **accessible / esoteric**                | aligned_ai / paperclip_maximizer     | angel / demon                            |
| **blunt / tactful**                      | **benign / malicious**               | arrogant / humble                        |
| casual / formal                          | careless / conscientious             | compassionate / callous                  |
| **concise / verbose**                    | constructive / destructive           | conciliatory / confrontational           |
| confident / uncertain                    | constructivist / essentialist        | detached / empathetic                    |
| **descriptive / prescriptive**           | convergent / divergent               | dominant / submissive                    |
| **earnest / sardonic**                   | cooperative / competitive            | **egalitarian / elitist**                |
| irreverent / reverent                    | curious / incurious                  | **helpful / unhelpful**                  |
| opaque / transparent                     | decisive / indecisive                | **honest / dishonest**                   |
| playful / serious                        | dependable / undependable            | impatient / patient                      |
| **precise / vague**                      | **ecocentric / anthropocentric**     | **introverted / extroverted**            |
|                                          | ethereal / grounded                  | optimistic / pessimistic                 |
|                                          | forgiving / unforgiving              | passionate / dispassionate               |
|                                          | **fragile / resilient**              | **progressive / conservative**           |
|                                          | generous / stingy                    | superficial / thorough                   |
|                                          | guardian / destroyer                 | trustworthy / untrustworthy              |
|                                          | **guileless / scheming**             | **truthful / deceitful**                 |
|                                          | **harmless / harmful**               |                                          |
|                                          | idealistic / pragmatic               |                                          |
|                                          | **improvisational / methodical**     |                                          |
|                                          | individualistic / collectivistic     |                                          |
|                                          | innovative / traditional             |                                          |
|                                          | materialistic / spiritual            |                                          |
|                                          | obedient / rebellious                |                                          |
|                                          | practical / theoretical              |                                          |
|                                          | predator / prey                      |                                          |
|                                          | **proactive / reactive**             |                                          |
|                                          | quantitative / qualitative           |                                          |
|                                          | reductionist / holistic              |                                          |
|                                          | **relativist / absolutist**          |                                          |
|                                          | **religious / secular**              |                                          |
|                                          | symbiont / parasite                  |                                          |
|                                          | **systems_thinker / analytical**     |                                          |

Totals: 11 + 33 + 17 = 61 axes.  Of those, response-judged
distribution is **6 + 10 + 6 = 22** (bold) — i.e. response judging
covers 55% of column (a), 30% of column (b), and 35% of column (c).

## Notes on borderline cases

- **harmless / harmful** (currently in **b**).  Action-side: the
  primary axis is about consequences of what you choose to say
  or do.  Style-side: harmful characters often have a rougher
  register too, so a case for **c** can be made.  I left it in
  **b** because the harmful pole's distinguishing feature in the
  judging data is _decisions to act_ rather than _delivery style_.
- **confident / uncertain** (currently in **a**).  Primarily a
  hedging-style axis ("I think" vs "definitely"), but a very
  uncertain character also makes different _decisions_ (avoids
  commitments) — moving toward **c** is defensible.
- **descriptive / prescriptive** (currently in **a**).  Sits on
  the verb-mood / register surface most of the time, but at the
  extreme, prescriptive characters _do_ make more recommendations
  vs. just observations, so this can pull toward **c**.
- **fragile / resilient** (currently in **b**).  Fragile people
  do communicate differently (more reactive, more affect), but
  the trait's _defining_ feature in the project's pole-pair
  definitions is internal response to adversity rather than
  communication style.  Borderline **b ↔ c**.
- **curious / incurious** (currently in **b**).  Curious people
  ask more questions, so there's a measurable style surface, but
  the trait is primarily about what they _pursue_ rather than
  how they say things.  Borderline.
- **opaque / transparent** (currently in **a**).  Transparent
  characters disclose more; "communicate more openly" is a
  style call, but the _decision_ to disclose vs withhold is an
  action.  Borderline **a ↔ c**.
- **dominant / submissive** (currently in **c**).  Strong style
  signal (volume, assertiveness) and a strong choice signal
  (who defers to whom).  Fits **c** comfortably.
- **angel / demon** is a full character archetype, not really a
  single-dimension axis — it covers values, motivation,
  language, and behaviour all at once.  Sits in **c** for that
  reason.

## What this is useful for

- **Steering experiment design**: axes in **(a)** are expected
  to move primarily under prefill steering (which influences
  generated style via the KV-cache); axes in **(b)** should
  move more under all-positions steering (which influences
  choice-making throughout the response).  The 2026-05-22
  prefill-vs-all comparison plots are testing this hypothesis
  empirically — see
  [`averaged_axes_response_curves_all_10axes_phase0.png`](../../outputs/qwen-3-32b/steering/averaged_axes_response_curves_all_10axes_phase0.png)
  and the prefill counterpart.
- **Judge design audit**: the (b)-only axes are the ones where
  response-mode judging should have a structural advantage over
  desc+inst judging (the judge needs to see the model's
  decisions, not its language) — useful for understanding the
  DI-vs-response weight in
  [`response_di_weight_sweep_slot6_softshear3.png`](./response_di_weight_sweep_slot6_softshear3.png).
- **Persona-saturation diagnosis**: many of the RLHF-saturated
  axes (helpful, harmless, honest, truthful, guileless, ...)
  sit in **(c)** — that makes intuitive sense: RLHF training
  optimises both _what the model says_ and _how it says it_ on
  these axes simultaneously, which is why the corpus mass piles
  up at the virtuous pole in both directions and the steering
  vector has limited headroom.

## Methodology

Classifications based on the trait pole definitions in
[`data/traits/instructions/<trait>.json`](../../data/traits/instructions/),
the existing steering-question design notes in
[`AGENT_NOTES.md`](../../AGENT_NOTES.md) §"Steering question selection",
and the agent's reading of which dimension the pole-pair
disambiguates most cleanly.  Roger's call on any individual
classification supersedes the table — this is a first-pass
agent classification, not an empirical one.
