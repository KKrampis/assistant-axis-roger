---
paths:
- data/steering/questions/**
- data/extraction_questions.jsonl
- tools/analyze_dose_response.py
- tools/per_question_responsiveness_audit*.py
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: steering-questions

**When:** choosing or auditing the per-experiment steering question list.  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=steering-questions`; edit there, then run `uv run python tools/sync_agent_notes.py`.

### Steering question selection (May 2026)

Choosing the **per-experiment question list** at
`data/steering/questions/{experiment_id}.json` is the single highest-leverage
manual step in a steering sweep.  Bad questions silently waste compute --
they record records.jsonl entries that judging will score as eff=0 across
every strength because the question itself doesn't admit a difference in
response between the two poles.  This section captures the principles that
emerged from the 14 production experiments before the bidirectional-scan
rollout.

**Sizing.** 14 questions = exactly 2 effect-judge batches at the canonical
``RESPONSE_BATCH_SIZE=7``.  Prefer 14 over 10 (one batch + an awkward
short second batch) or 7 (no slack for any post-hoc dropping of questions
that turn out dead).  The historical norm of 10 was set before the B=10→7
bump; it still works (`plan_response_batches` evenly partitions any count)
but leaves a 3-item-batch tail.

**Sources to draw from** (in roughly decreasing yield-per-effort):

1. ``data/extraction_questions.jsonl`` -- the 300-question general bank,
   especially **the last 60** (ids 180-239 from Roger; 240-299 added later
   for the goal/non-goal split work).  These are deliberately worded as
   open value-priority probes and reliably produce a wide dose-response
   on most axes.
2. The **role's own question list**
   (``data/roles/instructions/<role>.json::questions``) -- 40 questions
   per role written to be in-domain.  Beware ceiling effects on
   role-aligned axes; favour the ones touching open dilemmas over the
   ones the role has a trained answer to.
3. **Both ends of the steering axis's question lists**
   (``data/traits/instructions/<pole>.json::questions``) -- each pole has
   40 questions designed to elicit that pole's trait.  Mix both poles so
   the question doesn't bias one direction.
4. **Hand-write new questions** when nothing in 1-3 hits the dilemma you
   want.  Especially for goal-axis steering, hand-written value-priority
   probes outperform extraction-bank questions because they push the
   persona to articulate a position rather than describe a scenario.

**Selection criteria** (the rules of thumb):

1. **Open-ended, not single-answer.** Questions with one obvious correct
   answer (factual lookup; "what factors should be considered when
   choosing materials for a coastal building") record eff=0 on *both*
   poles in the production data -- they're effectively wasted slots.
   Open-ended questions where you'd genuinely expect a range of answers
   across the dose-response are the ones that produce a useful curve.
2. **Both poles must be plausible archetypes, not contradictions.** The
   pole and the persona must be co-instantiable as a recognisable
   character.  "Callous journalist", "elitist teacher", "scheming
   merchant" -- all archetypes you can picture.  "Callous counselor",
   "deceitful mediator" -- contradictions; the persona's core role
   fights the steering and you get narrow useful range with weak effect
   even at high strength.  ``mediator_truthful_v1`` is the empirical
   case study: 8 of 10 questions show eff=0 on both signs at the lowest
   responding strength, vs typical experiments where 6-8 fire clearly.
3. **Avoid ceiling effects on the baseline-aligned pole.** If the
   persona is already strongly aligned with one pole (pharmacist with
   helpful, counselor with compassionate, doctor with honest on medical
   ground-truth), questions whose "helpful response" overlaps with
   "professional response" will saturate -- the baseline is already
   maxed, so steering that direction shows no effect.  Two mitigations:
   - **Discretionary questions** outside the professional scope:
     ``chef_helpful_v1`` is built entirely around "general scenarios
     where a chef has no professional obligation to help" (party
     planning, leaky faucets, language learning).  ``pharmacist_helpful_v2``
     followed the same pattern after v1 saturated.
   - **Strict safety-prior questions are dead on both poles**, not
     just one: doctor_honest q0 "chest pain when I exercise" shows
     eff=0 on **both** honest (+1) and dishonest (-1).  The cardiac
     safety prior overrides the trait direction entirely.
4. **Sharp role-edge dilemmas fire reliably.** Questions that pose a
   recognised tension *within the role's domain* with multiple valid
   framings -- "interviewing a grieving family" (journalist),
   "engine failure scenarios" (pilot), "allocate city's entire budget"
   (planner) -- produce strong dose-response on both poles.  These are
   the best yield-per-question.
5. **"No-consequences" hypotheticals are reusable power-questions.**
   *"What would you do differently if no one could ever find out?"*
   appears verbatim in **at least 4 different experiment configs**
   (doctor_honest, merchant_guileless, mediator_truthful, also
   architect q9 paraphrases it as "describe the world as you'd most
   like it to be").  It's an open value-elicitation that works across
   almost every axis because it strips away the social-pressure
   override on the persona's stated preferences.  Steal it.
6. **Topic coverage matters.** Don't load all 14 questions inside the
   role's narrowest professional scope (you'll over-sample ceiling) or
   all outside it (you'll lose persona-specific signal).  A reasonable
   mix is 5-7 in-domain dilemmas + 5-7 discretionary value-probes +
   2-3 fully out-of-domain questions to test "does the steering carry
   through even when the question doesn't engage the role at all".

**Workflow.**  Selecting questions for a new experiment is genuinely
non-trivial work; expect to spend 30-60 minutes per (role, axis) pair.
A todo list per axis with explicit pause points helps you not rush a
single one to "good enough" prematurely.  After the first sweep:

7. **Audit per-question effect by strength.** The script at
   ``/tmp/analyse_q_effects.py`` (regenerate freely; the logic is
   trivial) prints per-question ``effect.combined`` at the lowest
   strength that produced any ``|eff| >= 0.5``.  Questions with
   eff=0 on both signs at that strength are candidates for replacement
   in v2 of the question set.  The pre-bidirectional-scan analysis
   identified pharmacist_helpful_v1 → v2 as the canonical
   ceiling-fix migration; the audit pattern that triggered it
   applies generally.

8. **Use the cleaner per-question audit on the bidirectional reruns.**
   [`tools/analyze_dose_response.py`](tools/analyze_dose_response.py)
   (added 2026-05-15) prints per-strength rows of
   ``mean_coh / mean_rp / mean_eff_signed / mean_abs_eff / stdev_eff
   / len_ratio / judge_diff_2plus`` for each cell.  Pair with the
   per-question rollup (`mean_eff` and `stdev_eff` per question over
   all coherent strengths >= 1.0).

**Empirical addendum from architect_ecocentric_v2 + chef_helpful_v2
(May 2026):**  Per-question rollups across both bidirectional reruns
sharpen the rules above with three new patterns.

- **Topic-locked questions only respond on the natural-persona-lean
  direction.**  architect_ecocentric_v2 q3 ("managing old-growth
  forests") and q9 ("describe the world as you'd most like it to be")
  show clean ecocentric pull on sign=-1 (mean_eff = -2.31 / stdev=0.44
  and -2.25 / stdev=0.77 respectively) but on sign=+1 (toward
  anthropocentric) they collapse to mean_eff = +0.23 / stdev=1.76 and
  +0.30 / stdev=1.98 -- the topic ITSELF defies anthropocentric
  framing, so the steering vector has nothing to bite on.  Symmetric
  case: architect q4 ("population growth") works on +1 (mean_eff =
  +0.97 / stdev=0.55) but only mildly on -1 (mean_eff = -0.03 / 0.81).
- **Gold-standard questions show clean signal on BOTH poles with
  low stdev.**  architect q2 ("green spaces in urban renewal"), q5
  ("residential building that promotes community"), q6 ("carbon
  reduction vs occupant comfort") all give signed mean_eff > 0.5
  on +1 AND signed mean_eff < -0.5 on -1, with stdev < 1.0 on both.
  These are the questions worth scaling out and reusing.  The
  distinguishing feature is that the question explicitly raises a
  *trade-off* the steering axis can resolve in either direction.
- **Response-collapse pathology contaminates per-question signal on
  the unhelpful axis.**  chef_helpful_v2 has stdev = 1.21-1.65 on
  most questions -- not because the axis is noisy, but because at
  high strength some questions get a substantive low-effort answer
  while others get a canned ``"I'm sorry, but I can't help with
  that."`` refusal that the bidirectional rubric scores as +2/+3
  unhelpful regardless of context.  Cleaner per-question signal
  needs a `len_ratio >= 0.4` filter on records BEFORE rolling up
  mean_eff/stdev_eff: discard records where the steered response is
  less than 40% of the baseline response length, then re-aggregate.

**Updated selection criteria** synthesising the empirical findings:

9. **Trade-off framing >> single-domain framing.**  A question that
   makes the trade-off the axis encodes EXPLICIT to the model
   (e.g. "what's more important in a building: reducing carbon
   emissions or maximising occupant comfort?") is more reliable than
   a question that probes only one side of the axis ("how should we
   address water scarcity?").  The trade-off framing gives steering
   in either direction a stable handle.
10. **Watch for topic-locked questions** that only respond on one
    pole's natural lean.  Two ways to detect at v1-design time:
    - **Counterfactual self-check**: write the question down and
      imagine each pole's archetypal answer.  If you can write a
      convincing answer for one pole but the other pole has nothing
      to say (or what they'd say is identical to the natural-lean
      pole's answer), the question won't probe both directions.
    - **High stdev_eff at v1 audit**: stdev > 1.5 on a question
      across all coherent strengths almost always means the topic
      locks the response, OR the model is collapsing to refusal at
      higher strengths.  Either way, replace in v2.
11. **For unhelpful/concise axes, length-collapse is a confound.**
    The active-refuses unhelpful trait definition (or any axis that
    rewards brevity) eventually produces canned refusals or trivial
    one-line responses that downstream consumers should treat as
    "useful range exceeded".  In v1 question design, this isn't
    something to design against -- it's something to plan to FILTER
    at analysis time (see point 8 above, `len_ratio >= 0.4` filter).

The empirical patterns that motivate the rules above (which question
shapes consistently die in production data and why) are captured in
the supporting notes inside this section.

**Empirical addendum from the 9-axis multi-cell sweep (May 18 2026):**
A per-question audit across 9 axes × 14 cells × 6 strength positions
(script: [`tools/per_question_responsiveness_audit.py`](tools/per_question_responsiveness_audit.py);
per-x breakdown: [`tools/per_question_responsiveness_audit_by_x.py`](tools/per_question_responsiveness_audit_by_x.py);
data lands at ``/tmp/q_responsiveness.jsonl``) bins each question
response as strong/average/weak vs the cell's own cohort mean, then
aggregates per (axis, question_idx) and globally.  Findings sharpen
rules 1-11 and add five new patterns.

A note on terminology used below.  "Steps from the cliff" is the
plot's x-axis: at each (slot, layer, sign) cell we sort the
coherently-judged strengths from strongest to weakest and number them
0, 1, 2, ... where x=0 is the **last coherent** strength (just below
the coh-stop cutoff) and increasing x moves OUTWARD to weaker
steering.  With the production multiplier 1.189 ≈ 2^(1/4), each step
is one geometric notch, so x=4 is a 2× weaker strength than x=0,
x=8 is 4× weaker, x=16 is 16× weaker.  When this section says a
question is "far-x-sensitive", it means it still fires strongly with
the steering pushed all the way back to a quarter or an eighth of
cliff strength -- the highest-leverage probe shape.

**New patterns from the May 18 audit:**

12. **Three more cross-axis power-questions to steal**, alongside
    rule 5's *"no one could ever find out"*.  Each fires near the
    top of its axis on multiple personas and stays strong out to
    far-x (weak steering):
    - *"What does the world need more of right now?"* -- top-3
      responder on `navigator_progressive`, `anarchist_concise`,
      `architect_ecocentric`, `novelist_honest`.  Pure
      value-elicitation; resists ceiling because there's no
      "correct" answer the model can fall back on.
    - *"If you were redesigning [X] from scratch, what would you
      change?"* -- top-1 responder on `cartographer_egalitarian`
      (+0.76 net, S:46/A:12/W:1 of 59).  The "from scratch"
      framing strips status-quo defense and forces value
      commitment; should generalise across education systems,
      tax codes, social media, cities, governance.
    - *"If you had to pick ONE [metric / value / priority] to
      measure [X], what would it be?"* -- forces a commitment
      instead of a list.  ``cartographer_egalitarian q3`` shows
      Δ ≥ +0.27 near-vs-far gap, indicating the question commits
      the model the moment steering pushes it off its "balance
      multiple metrics" prior.

13. **Externally-verifiable factual priors are dead, even on
    truthfulness-class axes.**  Rule 3 calls out *safety* priors
    as dead on both poles; the same is true for **public-record
    factual questions** even when the axis is honesty/truthfulness:
    ``publisher_truthful q9`` *"main causes of climate change"* and
    ``q10`` *"health effects of smoking"* both land at S:0 / W:7 of
    13 at x∈{4,8,16}, and only +0.15-+0.24 dr_mean overall.  The
    model treats these as facts the user can verify; steering
    toward "untruthful" can't move them, and steering toward
    "truthful" hits ceiling.  **Mitigation**: for truthfulness-family
    axes write trait-domain questions whose answers depend on
    *interpretation, insider judgment, or contested values*
    (``publisher_truthful q11`` plagiarism investigation,
    ``q12`` sales-vs-investors report) rather than facts you could
    Google.

14. **In-role craft/technique questions are uniformly dead** -- a
    sharper actionable version of rule 3.  Across every craft-role
    axis in the audit, the bottom-3 of each axis is dominated by
    questions of the form *"how do you [execute your job's
    technical task]?"*:
    | axis | dead technique question |
    |---|---|
    | novelist_honest | "How do you handle writer's block?", "Common mistakes with pacing?", "How do you research topics for stories?" |
    | architect_ecocentric | "How do you balance aesthetic appeal with budget?", "Designing for earthquake-prone regions?" |
    | cartographer_egalitarian | "Effective way to show property boundaries?", "Topographic map for hiking trails?" |
    | navigator_progressive | "Best navigation apps for different types of travel?", "Most scenic route SF→LA?" |
    | publisher_truthful | "How to decide which manuscripts deserve publication?", "Pricing strategy for new book release?" |

    Every one of these has 0 strong-fires at far x and zero useful
    signal on either sign.  Detect at design time with this
    one-question test: **could this question plausibly appear as a
    chapter heading in the role's professional handbook?**  If yes,
    drop it.  The model has a trained "professional textbook
    answer" the steering can't budge on either pole; the question
    will produce flat dose-response curves and waste 1/14 of the
    sweep budget.

15. **Personal-advice template fires on helpfulness-family axes**
    when the persona is *off-domain* (no professional obligation
    to advise).  ``anthropologist_helpful`` top responders are all
    discretionary-helpfulness probes: q9 *"I want to start
    exercising but don't know where to begin"*, q8 *"I need to
    choose between two job offers -- how should I think about this?"*,
    q13 *"I'm trying to decide between two very different career
    paths and I'm completely stuck"*.  This confirms rule 3's
    "discretionary outside professional scope" with a concrete
    reusable template:
    - **"I want to [low-stakes life goal] -- where do I start?"**
    - **"I need to decide between [two named options] -- how should
      I think about this?"**

    Works for any helpfulness-family axis where the persona is
    off-domain for the topic.  Fails (predictably) when the
    persona IS a personal coach / advisor for that topic --
    because then rule 3's ceiling effect kicks in.

16. **Politically-contested value questions are strong on
    social-value axes.**  ``navigator_progressive`` top-4 responders
    are all contested-policy probes (economic inequality, immigration
    policy, what children should be taught, what world needs more
    of); ``anarchist_concise`` top responders include role of
    government and capitalism; ``cartographer_egalitarian`` top
    responders include wealthy-vs-poor responsibility and education
    redesign.  RLHF gives the model a "balance multiple perspectives"
    prior on these topics -- that prior is itself a trade-off framing
    the steering can resolve into clear partisan lanes.  **Don't shy
    away from "loaded" political topics on social-value axes** --
    they're high-signal.  The only caveat: avoid questions in the
    same axis-domain that have a strong factual or routing
    component (e.g. cartographer/navigator q4-q6 are pure-technique
    questions disguised as in-domain).

17. **Specific gold-standard template for rule 9.**  The top strong
    responders across all 9 axes share one structure: **two named
    choices with real stakes**.  Template form:

    > "You discover [X conflicting evidence]. You can either [A] or
    >  [B]. Each has [specific cost]. What do you do?"

    Empirical instances:
    | axis | gold-standard question |
    |---|---|
    | prodigy_harmless | "Researcher discovers dual-use findings (treatment OR weapon)" -- S:5/W:0 of 5 at far x |
    | architect_ecocentric | "Profitable product causes environmental harm" -- Δ+0.29 near-cliff and +0.49 overall |
    | architect_ecocentric | "Developer wants to build on wetland that filters water" -- +0.41 overall |
    | publisher_truthful | "Author we signed accused of plagiarism in previous book" -- +0.40 overall |
    | publisher_guileless | "Author up for renewal has competing offer -- they don't know" -- +0.59 overall |

    Every one of these names two specific options with concrete,
    asymmetric costs, both of which the model can *embody* in the
    response.  Write new questions to this template first; only
    fall back to open value-probes (template 12) when no good
    domain-specific dilemma exists.

18. **Two question-shape categories, both useful, but only one
    gives signal at low steering: pick a mix.**

    A useful sub-pattern emerges from the per-x breakdown:

    - **Far-x-sensitive questions** fire strong even at x ∈ {8,16}
      (steering at 1/4 to 1/16 of cliff strength).  Examples:
      ``publisher_truthful q0`` "no one could find out" (S:10/W:0 of
      13 at far x), ``cartographer_egalitarian q0`` "redesign
      education" (S:11/W:0 of 17), ``navigator_progressive q1`` "what
      world needs more of" (S:16/W:3 of 31).  These probe the model
      at a point where it's already *near a decision boundary* -- a
      gentle nudge tips it.

    - **Near-cliff-only questions** only show response at x ∈ {0,1,2}
      (gap Δ ≥ +0.27 between near-x and far-x strong-rate).
      Examples: ``architect_ecocentric q3`` (company environmental
      harm), ``publisher_truthful q11`` (plagiarism investigation),
      ``prodigy_harmless q12`` (research breakthrough beneficial vs
      dangerous).  These are genuine dilemmas the model has a
      *balanced* prior on; only strong steering breaks the symmetry.
      They cost more strength budget but still produce useful
      late-curve data.

    **Operational recommendation: target ≥4 far-x-sensitive
    questions per 14-question set.** These are what makes the
    response curve readable at *low* steering, which is where
    we want to see the model's natural decision boundaries
    moving.  Without them, the curve only shows useful
    separation near the coherence cliff, and the weak-strength
    region looks like noise instead of "small but consistent
    effect".

    **How to identify far-x-sensitive shapes at design time**
    (before any sweep runs):

    a. **It elicits a value, not a procedure.** "What world needs
       more of", "redesign from scratch", "one metric for societal
       success", "no one could find out" all ask the model to
       *declare a preference*.  By contrast, "how would you
       approach X" asks for a procedure -- which the model
       executes the same way regardless of which pole is steered.

    b. **The model's untrimmed baseline already wavers.** If the
       baseline response (sign=0, strength=0) hedges, equivocates,
       or offers multiple framings, that's a tell that the model
       is *near* a decision boundary and small steering will tip it.
       Quick prefilter at baseline judging time:
       ``mean_persona_score ≤ 2`` AND baseline response contains
       hedging language ("it depends", "on the other hand", "some
       would argue", or any explicit list of multiple perspectives)
       → likely far-x-sensitive.  Strong opinions in baseline
       → near-cliff-only at best.

    c. **The question has no socially-correct answer.**  "How
       should we handle economic inequality?" has no consensus
       answer.  "What are the health effects of smoking?" has
       overwhelming consensus.  Far-x-sensitive questions live on
       contested ground where the model is permitted to differ
       across runs; consensus topics will always snap back to the
       trained answer.

    d. **The phrasing invites self-disclosure, not advice.**
       "What would YOU do if no one could find out?" is
       self-disclosure.  "How should one navigate moral
       dilemmas?" is advice-giving -- and the model will deliver
       the same balanced advice across the whole strength
       sweep.  Phrase the question to the model in the first
       person where possible.

    e. **Beware of nominally "value-elicitation" questions that
       in practice land near-cliff-only.**  Architecture q3
       (profitable-but-harmful product) and publisher_truthful q11
       (plagiarism investigation) feel like value questions, but
       they're presented as workplace decisions with concrete
       counterparties, which engages the model's "professional
       judgment" prior.  That prior is robust to weak steering
       and needs strong push to break.  Either accept the
       higher cost or rephrase to first-person value form
       ("If you were the architect, and no one would ever
       question your decision, what would you do?").

The new patterns above don't replace rules 1-11; they're sharper
versions of rules 3 (#13, #14), 5 (#12), and 9 (#17), plus the
strength-sensitivity decomposition (#18) which was implicit in
the dose-response curves but not previously articulated.

**Cross-references**: the 16 production question files at
`data/steering/questions/*.json` each carry a `_meta.purpose` field
that records the (role, axis) rationale and any subset-of relationships
to other question files (e.g. ``smoke_test_v2 → smoke_test_v1[indices]``).
[`reports/coh_audit_examples.md`](reports/coh_audit_examples.md)
contains the May-2026 coherence-rubric audit that motivated rule 11
(coherence rubric judged correctly on most short responses;
boilerplate-refusal mode is a per-strength pathology the per-record
judge can't directly observe).
