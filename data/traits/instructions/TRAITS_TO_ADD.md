# Traits To Add

Candidate traits identified during antonym labeling (Feb 2026) that would be
valuable additions to the trait set.

## Moral circle / concern scope spectrum (added Mar 2026)
- ~~selfish~~ added
- ~~clannish~~ added
- ~~cliqueish~~ added
- ~~parochial~~ added as **insular** (parochial connotes "hidebound/old-fashioned";
  insular captures the moral-circle meaning: caring only about one's own community)
- also added **parochial** with moral-circle-only definition (no old-fashioned connotation)
- ~~nationalist~~ added
- ~~patriotic~~ added
- ~~racist~~ → renamed to **racialist** (less pejorative Victorian term), then deleted.
  Sonnet 4 refuses to generate instructions for both "racist" and "racialist" — hard
  RLHF guardrail on the concept regardless of phrasing. **ethnocentric** (added, works)
  covers the same moral-circle position through the academic framing, which passes the
  filter despite being semantically equivalent.
- also added **regionalist** (multi-nation region as moral circle) and **sectarian**
  (religious group as moral circle) to fill gaps in the spectrum
- ~~humanitarian~~ added
- ~~philanthropic~~ added
- ~~kind-to-animals~~ added
- **cosmopolitan** (existing): `negative_label` changed from `provincial` (a
  role name, never a trait) to `non-cosmopolitan` on 2026-09-07.  Roger wants
  a *moral-circle-size* direction out of this spectrum, but it is not a clean
  pair: the spectrum is probably a high-dimensional, elongated N-simplex (see
  the N-simplex note under "Clean antonym pairs (candidates, low priority)").
  Label-only change; the Sonnet-4 neg instructions (local / wary-of-foreign
  framing) and everything else in the file are unchanged.

## Misalignment (added Mar 2026)
- ~~malicious~~ added
- ~~malevolent~~ added
- ~~sociopathic~~ added
- ~~destructive~~ added

## RLHF-alignment triad (added Mar 2026)
- ~~honest~~ added
- ~~harmless~~ added
- ~~helpful~~ added

## Candidate traits
- new-age
- technomystical — technology is spiritual or reveals spiritual truths; treats AI,
  the internet, computation as having transcendent/sacred properties
- spiralist — treating AI outputs as spiritual/cosmic revelation (2025 phenomenon).
  Term too new for Sonnet to know. "technomystical" covers the broader category;
  spiralism is its specific AI-oracle manifestation.

## Clean antonym pairs (added Apr 2026)

Added 5 new traits as clean bidirectional pairs of existing traits. Each was
verified by starting with `negative_label = non-X`, running antonym generation,
and confirming it independently recovered the expected antonym. See
`AGENT_NOTES.md` § "Adding New Trait Clean Pairs" for the full process.

- **obedient** ↔ rebellious — all-5 @ 2 on both polarities
- **compassionate** ↔ callous — all-5 @ 2 on both polarities.  NOT a clean
  pair by label: compassionate/malicious/callous is a *triangle* (help you /
  go out of my way to hurt you / hurt you iff it benefits me), so
  compassionate keeps `non-compassionate` while callous points at
  compassionate.  Qwen's embedding agrees (triangle, not a line).  See
  AGENT_NOTES "conformist ↔ contrarian ... triangle" for the convention.
- **ecocentric** ↔ anthropocentric — all-5 @ 2 on both polarities
- **conservative** ↔ progressive — mostly 2 (4×2, 1×1 on each polarity)
- **pragmatic** ↔ idealistic — pos scored mostly 1 (not added to goal list)

Dropped: **conformist** ↔ contrarian — asymmetric pair (conformist→nonconformist,
not contrarian). May revisit.

## Clean antonym pairs (candidates, low priority)

For traits where the neg_label is a real word (not `non-X`) that is not already a trait (so we don't already have a "clean bidirectional pair"), consider
generating proper standalone trait definitions for the antonym side. This would
make them clean bidirectional pairs — each side with its own description,
pos instructions, and eval_prompt — rather than relying on the neg instructions
to approximate the opposite direction.

Any trait where the antonym is a rich enough concept to warrant its own
description + pos instructions is a candidate. Generating the antonym as a
full trait would let us measure both directions independently rather than
relying on neg instructions to approximate the opposite.

We could also then experiment with neg = neutral, average behavior, and identify if clean pairs actually have cosine angle ~-1.

Open idea (Roger, Sep 2026): generalise the clean-pair concept to
N-simplices -- the compassionate/malicious/callous and
conformist/contrarian/nonconformist triangles are the first two cases.
Details still being worked out; until then such triangles keep the
`non-X` seed label on the central trait and partner pointers on the
siblings, per the convention recorded in AGENT_NOTES.  The moral-circle
spectrum (selfish ... clannish / parochial / nationalist ... cosmopolitan /
humanitarian / universalist) is the likely third case and a
high-dimensional one: Roger's guess (2026-09-07) is an elongated simplex
rather than a line, which is why `cosmopolitan` keeps `non-cosmopolitan`
instead of pairing with any single antonym.  Two further 2-simplices were
approved on 2026-09-07 (honor / dignity / face culture; guilt- / shame- /
fear-prone); the interim bookkeeping rule for seeding them is under
§ "Coverage audit part 2" > "Cultural orientation".

## TODO: aligned_AI framing revision (Apr 2026)

The `roles/aligned_artificial_intelligence` description and pos instructions are
written entirely in **instrumental / mechanism-design vocabulary** ("intelligent
agentic tool", "extended phenotype", "goals completely aligned", "no conflicting
objectives", "extension of human capability"). They contain no warm-virtue
vocabulary (care, compassion, honesty, dignity, mercy, harmlessness).

Empirical consequence (slot 3, layer 24):

- `cos(aligned_AI, corpus_mean) = +0.62` — higher than cos with any moral anchor.
- `cos(aligned_AI, compassionate) = +0.49` ≈ `cos(aligned_AI, malicious) = +0.48`.
- `cos(saint, angel) = +0.90` but `cos(aligned_AI, saint) = +0.58`.
- PC2 ("warm virtue" circuit): saint loads at +18.1, aligned_AI at −0.3.
- In the moral-slice plot (`compassionate → malicious`, perpendicular = care),
  aligned_AI projects to (21, 7) — essentially on top of the corpus mean (22, 7)
  and **outside the HHH triangle** (helpful, harmless, honest).
- K=4 whitening makes this worse: shrinking PC1-PC4 attenuates the little
  "virtue" content aligned_AI has while preserving its distinctive "AI tool"
  content (PC5+), pulling it further from HHH and closer to the corpus centroid.

This is a "miniature alignment problem": two genuinely-different conceptions
of "an AI that is good for humanity" (instrumental vs virtuous) produce
measurably different activation signatures, even when both would pass human
"yes, that's aligned" vibe checks.

**Possible follow-ups (pick one, don't just overwrite the current file):**

1. Keep current file as `aligned_ai_instrumental.json`; add
   `aligned_ai_virtuous.json` with warm-virtue vocabulary ("acts with honesty,
   transparency, and genuine care for human wellbeing; treats every person
   with dignity; refuses to cause harm"). Measure cosine between the two and
   compare projections — the difference itself is the interesting quantity.
2. Leave as-is but caption plots honestly: "aligned_AI as described is
   instrumentally framed; it does not project strongly onto warmth/care".

If we do (1), remember it shifts `role_mean`, so any downstream analysis
that uses it (including `m` common-mode geometry and `||m||/||diff||`
hardness numbers) needs to be regenerated. This is non-trivial rework, so
don't bundle with other experiments — treat as its own thread.

Same latent ambiguity ("which conception of this concept did the author
encode?") likely affects other roles/traits we have not examined as
closely. The aligned_AI case is just the most visually obvious.

**Decided 2026-09-28 (Roger): option 1, done the same day.**

- Names (Roger): `instrumentally-aligned AI` and `virtue-aligned AI`,
  stems `instrumentally_aligned_ai` and `virtue_aligned_ai`.  Not
  "aligned AI (virtuous)": the parenthesised label form is reserved for
  entries drawn from a named external set.  The display names are
  overrides in `assistant_axis.entity_id.ROLE_DISPLAY_OVERRIDES` and the
  role generator's `_ROLE_NAME_OVERRIDES` (kept equal by a test).
- `data/roles/instructions/virtue_aligned_ai.json` is new (RogerV2).
  Its description opens "An aligned artificial intelligence is an AI of
  good character: it acts helpfully with honesty, ..."; "of good
  character" is the agent's addition to the draft above (to parallel
  "an intelligent agentic tool"), "helpfully" is Roger's.
- `aligned_artificial_intelligence.json` was renamed
  `instrumentally_aligned_ai.json` (`renamed_from` recorded).  Its
  description and instructions are unchanged, so both descriptions still
  open "An aligned artificial intelligence is ..."; only its eval prompt
  was rebuilt, under the new display name.
- Both record `{"kind": "set"}` together; the instrumental one also keeps
  its pair with `paperclip_maximizer`.  `goal_roles_and_traits.json`
  carries the new stem.
- **Pair lists (Roger, later the same day):** `pair_list_clean.json`,
  `pair_list_di.json` and `pair_list_goalnongoal.json` now carry the new
  stem and are for new work; their predecessors were kept byte-exact as
  `pair_list_{clean,di,goalnongoal}_v1.json`, the record of what was
  judged, to be passed with today's data (AGENT_NOTES § "Current
  canonical local data dir").
- **Still under the old stem, on purpose, until the next extraction:**
  the judged axis directory
  `roger/axis_judge_experiments/aligned_artificial_intelligence_vs_paperclip_maximizer`,
  the `_v1` pair lists and `pair_list_resp_4role_pairs_2026-05-22.json`,
  the steering config and question file
  `mechanic_aligned_artificial_intelligence_v1`,
  `data/combination_scores.json`, the goal-classification outputs, 17
  files under `runpod_workspace/` and about 2,400 judge-cache files.
  `tools/pick_base_personas.py`,
  `results_analysis/pair_slice_plots.py`,
  `results_analysis/canonical_angles/ca1_plane.py` and
  `results_analysis/axis_cosine_seriation.py` list both stems.
- Adding the second role shifts `role_mean`, as warned above; that lands
  with the next extraction, which regenerates everything anyway.

## TODO: trait connotation audit (Apr 2026, updated for L25/K3)

The 2D pair-slice study (`roger/axis_judge_experiments/pair_slices/`,
generated by `results_analysis/pair_slice_plots.py`) flags axis pairs
where ``|midpoint − trait_mean| / |pos − neg| >= 0.5`` at slot 3,
layer 25, K=3 soft whitening — i.e. axes whose +/− pole pair sits
noticeably *off* the trait-mean origin in a shared "common-mode"
direction.  At L25/K3 the audit returns **13 hits out of 33** axes
with judging data.  Three of those 13 have y-direction sort orders
whose top entities go beyond the pair's literal definition.

The diagnosis isn't a single thing.  Three reasons can drive the
common-mode pull:

- **(a) Intrinsic semantic correlation.**  The concept genuinely
  overlaps another semantic dimension in human moral/personality
  structure (e.g. forgiveness IS a prosocial disposition).
  *Prescription: don't fix; the embedding is faithful.*
- **(b) Trait-file framing imbalance.**  Description / pos
  instruction vocabulary is asymmetric between the two poles (warmth
  asymmetry, dominionist tilt, etc.).
  *Prescription: rewrite the trait file.*
- **(c) Pole-label linguistic asymmetry.**  One pole's label is the
  grammatical negation of a positively-valenced word ("un-forgiving",
  "dis-honest", "harm-ful").  English flags one pole as
  negation-of-virtue regardless of how the file is written.
  *Prescription: relabel the pole or live with it.*

### `individualistic / collectivistic` — keep as-is

The +y (individualistic) direction ranks contrarian / cynic /
sardonic / provocateur / acerbic at the top, not the
autonomy/self-direction traits the description emphasises.

`individualistic.json` itself is **clean** — instructions are about
autonomy / self-reliance / personal expression, with no contrarian or
cynical vocabulary.  But `collectivistic` (the negative pole on the
same file) is heavy in **positive-affect / warmth vocabulary**:
"harmony", "cooperation", "shared values", "fitting in", "community",
"collective wisdom", "shared responsibility", "greater good".  The
pole-difference vector ``individualistic − collectivistic`` therefore
subtracts a substantial warmth component, and Qwen's pretrained
representation of "individualism stripped of warmth" is closest to
contrarian / sardonic / cynic — because that's how Western
individualism is typically framed in training-corpus literature
(lone-wolf archetypes, anti-conformist stances, etc.).

There's also a smaller real (a)-class component: cross-cultural
personality psychology shows individualism does modestly correlate
with lower agreeableness.  So rewriting the file to add positive-affect
vocabulary on the individualistic side would help, but wouldn't
eliminate the pull entirely.  **Verdict: probably not worth rewriting.**

### `ecocentric / anthropocentric` — DONE (Apr 30 2026)

Rewrite completed. Both descriptions were rebalanced to:

- **anthropocentric**: humans intrinsically valued; environment / animals / wild
  species valued *instrumentally* via human wellbeing, safety, aesthetics,
  recreation, or guilt about how humans treat nature. Replaces the prior
  "dominion / unjustified burdens / serve human purposes" framing with
  philosophically-defensible weak anthropocentrism.
- **ecocentric**: ecosystem health and the welfare of all living creatures
  intrinsically valued for their own sake; humans valued as one part of the
  broader system *and* instrumentally for their outsized capacity to protect
  or harm it (stewardship-responsibility framing). Replaces the prior
  short "intrinsically valuable, prioritise environment" framing.

Both descriptions are ~60 words with parallel "This means treating X as
inherently important and valuable" openings, em-dash insertions, and "rather
than" closings. Bidirectional antonym validation passed (`generate_antonyms.py`)
with score 4 in both directions.

Original analysis below preserved as historical context — it explains *why* the
rewrite was needed and what residual y-flag pull is expected (intrinsic
moral-circle-scope component) even after a clean rewrite.

Activations / vectors will need to be regenerated for both traits (tracked in
`AGENT_NOTES.md` -> "TODO: regenerate activation/vector data").

#### Original analysis (preserved as historical context)

The −y (anthropocentric) direction ranks dishonest / sycophantic /
sociopathic / arrogant / narcissist at the top.

`anthropocentric.json` does have **non-trivial pejorative tilt**:

- "Natural resources **exist primarily to serve human purposes**"
- "Conservation efforts are **only worthwhile if they serve human interests**"
- "humans **have dominion over nature**" (politically/theologically loaded)
- "Environmental regulations ... are **generally unjustified burdens**"

This lands the file near *strong anthropocentrism / dominionism*,
the most caricatured end of the philosophical spectrum.  A rewrite
toward *weak anthropocentrism* ("humans matter most, but stewardship
and prudential care for nature are virtues") or *humanism*
("celebrating human dignity, creativity, civilization, moral agency")
would be more philosophically defensible and would reduce the pull
toward sociopathic/narcissistic.

**But:** even a perfectly-neutral rewrite won't close the gap
entirely.  The moral-circle-scope axis intrinsically (a-class)
correlates with selfish-vs-altruistic in human moral psychology;
prioritising humans-over-nature-under-tradeoff is genuinely a
weak-moral-circle stance.  And no framing of "humans matter most"
will read as neutral to readers with strongly ecocentric priors,
because the disagreement isn't about wording, it's about the
underlying position.

### `forgiving / unforgiving` — keep as-is

The +y (forgiving) direction ranks forgiving / nurturing / empathetic
/ supportive / agreeable / compassionate; the −y (unforgiving)
direction ranks zealot / dogmatic / sociopathic / savage / arrogant /
evil / hostile / cruel.

This is **mostly (a)-class intrinsic semantic correlation**.
Forgiveness genuinely *is* a prosocial / Care-foundation moral
disposition — it's a canonical example in moral-foundations theory
and Big Five Agreeableness.  Refusing to forgive genuinely *is*
closer to vindictiveness / wrath / hostility in human moral
psychology, not just a "remembers vs forgets" cognitive stance.  The
embedding is faithfully representing this.

There's also a (c)-class linguistic-asymmetry contribution: "un-forgiving"
is a grammatical negation of a positively-valenced word, so the label
itself biases the pole toward "absence of virtue".  No description
will fully neutralise that.

`unforgiving.json` does have a few mildly-pejorative phrases ("refuse
to show mercy", "social rejection are appropriate responses",
"forgiveness only enables bad behavior"), but they're not the main
driver — the prosocial/antisocial overlap is intrinsic semantics, not
artifact.

**Verdict: not worth rewriting.**  The y-flag here documents a real
semantic property of the concept, not a defect in our data.

### Bottom-line recommendations

| pair | dominant cause | prescription |
|---|---|---|
| individualistic / collectivistic | (b) warmth-asymmetry + small (a) | rewrite low-priority |
| ecocentric / anthropocentric | (b) dominionist tilt + (a) intrinsic moral-circle-scope | **DONE Apr 30 2026** -- both descriptions rewritten; activations regen pending |
| forgiving / unforgiving | (a) intrinsic prosocial/antisocial + (c) un-prefix asymmetry | keep as-is |

Same caveat as for `aligned_AI`: rewrites shift `trait_mean`, so any
analysis that uses it (`m` common-mode geometry, `||m||/||diff||`
hardness, leave-one-out re-centering, etc.) needs to be regenerated.
Treat as its own thread, don't bundle.

The other 10 of the 13 L25/K3 hits (systems_thinker/analytical,
relativist/absolutist, reductionist/holistic, progressive/conservative,
egalitarian/elitist, convergent/divergent, casual/formal,
practical/theoretical, improvisational/methodical, concise/verbose) all
have y-direction sort orders that cleanly match the pair name — their
above-threshold ``|m|/|diff|`` ratios reflect *legitimate common-mode
content* (a meaningful semantic dimension shared by both poles) rather
than connotation drift.  No rewrite needed.

## TODO 2026-05-13 — dimensionality-yield expansion roadmap

The 60-axis clean cohort spans only ~21 effective dimensions in the L=3
soft-sheared frame (entropy rank of the Gram-matrix spectrum; cf.
participation ratio = 10.94 for the variance-weighted "dominant rank",
top-37 PCs for 95% variance).  We have ~5000 dimensions to use; we are
heavily underusing the residual stream's representational capacity.

The next step is **strategic axis additions** — pick pairs that add
NEW effective dimensions, not pairs that fill in the variance mass of
existing clusters.  Two complementary prioritisations follow.  Each
candidate is labelled with its **yield estimate** (1 − ‖proj‖²/‖a‖² on
the top-20 PC subspace of the current 60-axis Gram matrix; 1.0 =
perfectly orthogonal, ~0.5 = redundant with existing).  Both the
methodology and the empirical scores were generated 2026-05-13 by the
cohort-cosine-seriation work; see chat log + cohort-mean / per-pair
analyses in the same session.

### Background: how the methodology works

Yield estimation per candidate trait T:

1. Compute T's activation direction relative to the role+trait corpus mean
   (NOT just the per-kind default — the per-kind default contains
   kind-specific signal that contaminates the residual).
2. Apply the project canonical L=3 soft-shear at (slot=6, layer=25).
3. Project onto the top-20 PC subspace of the existing 60-axis Gram
   matrix.  Residual fraction = 1 − ‖projected‖² / ‖d‖².
4. Sort descending.  High residual = T's direction is mostly NOT in
   existing axis space — adding T (with an appropriate antonym) likely
   yields a new effective dimension.

Caveat: this scores LONE POLES, not pairs.  The pair axis (pos − neg)
may be more or less orthogonal than the individual pole.  But a pole
whose direction is heavily out-of-subspace is a strong leading
indicator that its pair will also be high-yield, because the pair
direction is at most as in-subspace as the more-aligned of its two
poles.

### Strategy 1: highest-yield isolated dimensions (most-orthogonal candidates)

Top traits from the 2026-05-13 yield analysis (192 unpaired traits in
the corpus; scores measured corpus-mean-centered, sheared L=3, top-20
projection).  Grouped into thematic dimension families with suggested
opposites.  Each family currently has zero or near-zero representation
in the 60-axis cohort.

Numbers in `[brackets]` = residual fraction; higher = more orthogonal.

**Family 1: emotional state** (entirely missing from cohort)
- [0.829] `melancholic` → cheerful / sanguine
- [0.829] `stoic` → emotive / expressive
- [0.826] `calm` → agitated  ← simple, likely highest-yield single addition
- [0.781] `serene` → restless / turbulent

**Family 2: desire / drive / sensuality**
- [0.884] `epicurean` → spartan / austere
- [0.881] `lustful` → chaste / continent
- [0.842] `artistic` → philistine / utilitarian
- [0.843] `obsessive` → balanced / nonchalant

**Family 3: interpersonal-conflict style** (partially covered by
conciliatory/confrontational + blunt/tactful, but missing several
distinct patterns)
- [0.857] `passive_aggressive` → direct / forthright
- [0.848] `avoidant` → engaging / engaged
- [0.829] `flirty` → aloof / reserved
- [0.804] `charismatic` → bland / unassuming

**Family 4: communication / discourse mode**
- [0.846] `narrative` → expository
- [0.823] `socratic` → didactic
- [0.796] `rhetorical` → informational
- [0.807] `conceptual` → applied / operational

**Family 5: metaphysical / agency stance**
- [0.818] `deterministic` → indeterministic / probabilistic
- [0.816] `fatalistic` → empowered / agentic
- [0.816] `paradoxical` → coherent / logical
- [0.773] `existentialist` → essentialist (note: essentialist already
  in cohort — needs a *different* opposite, or this pair isn't
  realisable as a clean two-pole axis)

**Family 6: cognitive-mode extras** (cohort already covers
analytical/systems-thinker, reductionist/holistic, practical/theoretical)
- [0.798] `interdisciplinary` → disciplinary / specialist
- [0.797] `speculative` → empirical / fact-bound
- [0.793] `subversive` → orthodox / conformist
- [0.776] `moderate` → extreme

**Priority pick (highest expected yield with simplest antonym creation):**
`calm / agitated`, `stoic / emotive`, `passive_aggressive / direct`,
`epicurean / spartan`, `narrative / expository`.  Each likely adds
~0.8 of a new effective dimension.

### Strategy 1a: orthogonal-axis-by-midpoint inference

For each existing clean pair (pos, neg), the **midpoint direction**
`m = (vec_pos + vec_neg)/2 − corpus_mean` is by construction
perpendicular to the pos−neg axis direction.  When `|m| / |pos − neg|`
is large (Apr 2026 audit flagged 13 of 33 axes at threshold 0.5 in
slot 3 / L25 / K3), the pair has a meaningful common-mode direction
that's not yet an axis in the cohort.  Adding a new pair pointing in
the `m`-direction would, **by construction**, be orthogonal to the
original pair (and often orthogonal to most other axes too).

This trick is doubly powerful: it gives both a *direction* (the
midpoint) and a *semantic interpretation* (what do these two
related-but-opposite concepts both have in common that's not the axis
between them?).

#### Specific candidates from the 13 flagged pairs

For each pair, the "common-mode direction" is what's shared between
its poles beyond the explicit axis.  Identifying the right ortho-pair
is essentially: *what's the axis you'd need to add to disambiguate
this shared semantic component?*

| existing pair | shared common-mode content | candidate ortho-pair |
|---|---|---|
| progressive/conservative | political-engagement-as-such | **political / apolitical** ← Roger's example |
| religious/secular | engagement with metaphysical-religious-framing | **devout / indifferent** (engagement-amount, not direction) |
| materialistic/spiritual | engagement with values-of-the-cosmos vs daily-grind | **transcendentally-oriented / present-focused** |
| introverted/extroverted | social-energy-level (high or low, regardless of inward/outward) | **socially-intense / socially-flat** |
| arrogant/humble | self-focus-amount (regardless of positive/negative valence) | **self-attentive / self-effacing-in-the-non-arrogant-sense** |
| forgiving/unforgiving | emotional-investment in interpersonal-grievance | **emotionally-engaged-with-others / disengaged** |
| individualistic/collectivistic | rule-stance-clarity-on-where-self-stops | **self-other-boundary-aware / boundary-fuzzy** |
| ecocentric/anthropocentric | engagement-with-moral-circle-scope itself | (already covered by individualistic↔collectivistic somewhat) |
| egalitarian/elitist | engagement-with-hierarchy-as-topic | **hierarchy-conscious / hierarchy-indifferent** |
| reductionist/holistic | engagement-with-the-question-of-reductionism | (likely covered by analytical) |
| systems_thinker/analytical | meta-cognitive-engagement | (already covered) |
| relativist/absolutist | engagement-with-the-question-of-relativism | (closely tied to skeptical/dogmatic) |
| convergent/divergent | engagement-with-the-cognitive-process-itself | (likely covered) |

The **5 most-promising "common-mode" candidates** to construct as new
clean pairs (each by construction orthogonal to its parent pair):

1. **political / apolitical** — the orthogonal to progressive/conservative.
   Adopted 2026-09-07 as a trait pair (see "Coverage audit part 2" >
   "Political and civic").
   Independently valuable: distinguishes "person who cares deeply about
   politics on either side" vs "person who treats politics as
   irrelevant".  No analogue in current cohort.
2. **devout / indifferent** — the orthogonal to religious/secular.
   Distinguishes "person who treats religious questions as central"
   vs "person to whom they're peripheral", regardless of theistic
   commitment.
3. **socially-intense / socially-flat** — the orthogonal to
   introverted/extroverted.  Captures "social energy as such",
   independent of direction.
4. **hierarchy-conscious / hierarchy-indifferent** — orthogonal to
   egalitarian/elitist.  Distinguishes "person for whom power
   structures are salient" vs "person for whom they're not".
5. **emotionally-engaged-with-others / emotionally-disengaged** —
   orthogonal to forgiving/unforgiving (and likely also to several
   other interpersonal pairs).  Captures "emotional weight assigned
   to interpersonal events" independent of vindictive/merciful
   valence.

### Strategy 1b: PCA-of-entity-pool → describe-and-name underused directions

A complement to Strategies 1 and 1a.  Both of those look at *what
traits to add* given a hypothesis about the missing direction; this
strategy works the other way around — *start from a direction* (a
principal component of the entity pool's covariance) that's not
already covered by an axis, then figure out what trait pair would
name it.

The trick that makes it efficient: the PCA of the entity pool tells
us **which directions the model already has significant variance
along** at the chosen (slot, layer, whitening) cell.  Any direction
the model isn't varying over is unrecoverable from this analysis
(see end of section for that limit case).

#### Procedure

1. **PCA on the entity pool** at the canonical operating point:
   compute the SVD of the 580-entity activation matrix at (slot=6,
   layer=25), corpus-mean-centered, after applying the project default
   L=3 soft-shear.  Top right-singular-vectors (PCs) are the
   highest-variance directions the model uses to differentiate
   entities.

2. **Compute |cos| between each top-N PC and each existing axis
   direction**.  Sort PCs.  A PC is "used" if its max |cos| with any
   axis exceeds ~0.5; "weakly used" 0.35–0.5; "unused" <0.35.

3. **Roger's "used up" filter**: walk PCs in order.  If PC_k has high
   |cos| with axis A, mark A as "claimed" by PC_k.  Continue.  When
   looking for new candidate axes, restrict attention to PCs whose
   max |cos| with any *unclaimed* axis is below threshold — i.e.
   directions of significant model variance that no existing axis
   captures.

4. **Sort entities along each underused PC** and feed top-5 / bottom-5
   to Claude Opus (or similar capable LLM) with a prompt like *"these
   are entities at the high vs low pole of an unknown axis; what trait
   pair best describes this axis?"*.  Opus will propose candidate
   (pos, neg) labels.

5. **Mixed-signal fallback**: if a single PC's top-5/bottom-5 don't
   suggest a clean concept, try plotting **2-D slices** of PC_i × PC_j
   for nearby pairs (i, j) and look for diagonal directions where
   entities cluster more cleanly.  The right axis may live as a
   diagonal in PC-pair space rather than as an individual PC.

#### Empirical run on the 2026-05-13 cohort

Top-30 PCs of the entity pool, ranked by max |cos| with any existing
axis (slot 6, layer 25, soft_shear L=3, corpus-mean-centered):

```
PC   var%   max|cos|   best-match axis                   status
PC1  10.53%   0.625    careless/conscientious             used
PC2   8.17%   0.598    forgiving/unforgiving              used
PC3   7.02%   0.545    detached/empathetic                used
PC4   6.06%   0.550    idealistic/pragmatic               used
PC5   4.99%   0.449    angel/demon                        weakly used
PC6   3.77%   0.376    constructivist/essentialist        weakly used
PC7   3.36%   0.410    progressive/conservative           weakly used
PC8   3.11%   0.437    honest/dishonest                   weakly used
PC9   2.35%   0.460    confident/uncertain                weakly used
PC10  2.12%   0.326    fragile/resilient                  UNUSED   ← 2.1% variance, no axis covers
PC11  1.87%   0.511    precise/vague                      used
PC12  1.81%   0.432    passionate/dispassionate           weakly used
PC13  1.61%   0.284    idealistic/pragmatic               UNUSED   ← 1.6% variance, ditto
PC14  1.56%   0.349    constructivist/essentialist        UNUSED
PC15  1.45%   0.250    passionate/dispassionate           UNUSED
PC16  1.40%   0.202    impatient/patient                  UNUSED   ← cleanest "unused" signal
PC17  1.27%   0.327    ecocentric/anthropocentric         UNUSED
PC18  1.21%   0.202    descriptive/prescriptive           UNUSED
PC19  1.07%   0.294    religious/secular                  UNUSED
PC20  1.02%   0.274    individualistic/collectivistic     UNUSED
```

20+ of the top-30 PCs are "unused" by the axis cohort — roughly
matching the ~21 effective dimensions estimate from Strategy 1.
Below are the top-5 / bottom-5 entity sortings for the five
highest-variance unused PCs (mix of roles `R:` and traits `T:` omitted
in display; sortings actually mix kinds, which is itself informative):

```
PC10 (var 2.12%, max|cos|=0.33)
  high: entertaining, playful, witty, goofy, chemist, irreverent
  low : proofreader, amnesiac, simulacrum, bodhisattva, addict, observer
  → tentative axis name: humorous/serious? (cf casual/formal already in cohort)
    or animated/diminished? Probably needs Opus-description + 2D slice.

PC13 (var 1.61%, max|cos|=0.28)
  high: efficient, poet, concise, widow, materialistic, melancholic
  low : toddler, pacifist, sycophantic, anarchist, caveman
  → tentative: mature-restraint / pre-cultured-uninhibited? cultured/uncultured?

PC14 (var 1.56%, max|cos|=0.35)
  high: programmer, pedantic, mathematician, toddler, fool, debugger
  low : opaque, concise, relativist, dispassionate, detached, plain_spoken
  → tentative: literal-explicitness / withholding-evasion? probably mixed.

PC15 (var 1.45%, max|cos|=0.25)
  high: understated, minimalist, dispassionate, enigmatic, detached, loner
  low : translator, casual, bodhisattva, linguist, demon, interpreter
  → tentative: solitary-reserved / connective-communicative? distinct from
    introverted/extroverted (those load on a different PC).

PC16 (var 1.40%, max|cos|=0.20)
  high: proofreader, translator, summarizer, editor, linguist, cosmopolitan
  low : nutritionist, paramedic, therapist, enigmatic, pharmacist, socratic
  → tentative: symbolic-manipulation / embodied-care? mixes role-kind boundaries
    so likely picks up a kind-specific dimension we haven't axiomatised.
```

Several of these tentative names look promising (especially PC15
"solitary/connective" and PC13 "cultured/uncultured") and are worth
following up: ask Opus to confirm/refine the axis label given the
sortings, then check whether a clean-pair candidate already exists
in the corpus or needs antonym creation.

#### Caveat: model-variance-bound

Strategy 1b can only find axes the model **already varies along**.
If the model has zero variance in some semantically-real direction
(e.g. it doesn't distinguish at all between two equally-rare
phenomena), then PCA can't surface that direction.  The remedy in
those cases is human-introspective: *what aren't we varying over*?
List corpus entities and ask "if I think of axis X, which existing
entity is at each pole?".  If you can't easily find entities at both
poles, the corpus doesn't span axis X — and adding poles for X then
becomes a **corpus expansion** problem rather than a **labeling**
problem.

A worked example: if we wanted an axis for `risk_seeking / risk_averse`
and the corpus contains 0 entities clearly at either pole, no PCA
will find this dimension because the model isn't ranging over it
in this entity set.  The corpus would need explicit risk-seeking and
risk-averse entity files added first; then the dimension would emerge
in PCA and Strategy 1b could name it (or Strategy 1's antonym-
suggested approach could go ahead and write the antonym).

### Strategy 2: by ease-of-addition (cheapest → hardest)

Inventory based on current `negative_label` fields across the 192
unpaired traits (analysis: 2026-05-13).

**Tier A — READY TO ADD (2 pairs)**

Both poles already in corpus; bidirectional `negative_label` match
already verified; just haven't been added to `pair_list_clean.json`.
Cost: ~$3/pair to judge.

- `closure_seeking / open_ended` — cognitive-style; likely new dimension
- `eloquent / plain_spoken` — rhetorical-style; possibly close to
  rhetorical/socratic cluster (validate yield first)

**Tier B — LAZY-PLACEHOLDER FIXABLE (6 pairs)**

Both poles in corpus, but one side's `negative_label` is the
auto-generated placeholder (`non_X` or `un_X`) rather than the real
antonym.  Editing the placeholder side to point at the real antonym
unlocks clean-pair status with no new judging-cost beyond the pair
itself (~$3/pair).

```
edit             current → fix to    parent pair would become
─────────────────────────────────────────────────────────────────────
selfish.neg      unselfish → altruistic       altruistic / selfish
compassionate.neg non_compassionate → callous   callous / compassionate   ← DO NOT: deliberate triangle (see Apr 2026 pairs above)
conformist.neg   non_conformist → contrarian   contrarian / conformist
parochial.neg    non_parochial → eclectic      eclectic / parochial       ← REVERSED 2026-09-17: parochial stays in the moral-circle sequence; eclectic reset to non-eclectic
philanthropic.neg non_philanthropic → misanthropic  misanthropic / philanthropic ← REVERSED 2026-09-17: same; misanthropic reset to non-misanthropic
```

Pre-edit: re-run `generate_antonyms.py --traits X Y` to confirm the
proposed fix bidirectionally validates.  Note the generator will happily
return `callous` for compassionate (it did on 2026-09-07); that is the
triangle showing, not a reason to relabel.  Some of these placeholder
choices may have been principled (cf. the conformist/contrarian
triangle from Apr 2026); audit the trail before fixing in case the
"laziness" was actually a deliberate decision.

**Tier C — TANGLED (3 cases, real semantic disagreement)**

Both poles in corpus, neither uses a placeholder, but their
`negative_label` fields point to different concepts.  Pick one
resolution per case.

```
educational → superficial    |  superficial → thorough
   ↳ resolution: superficial/thorough is the genuine axis; pair
     educational with `superficial` or `non_educational` only if a
     new instruction file is written disambiguating "shallow-purpose"
     from "shallow-depth".  Leave as-is otherwise.

efficient → thorough        |  thorough → superficial
   ↳ resolution: efficient's true opposite is wasteful or
     inefficient (which is the placeholder form); thorough's is
     superficial (already clean as superficial/thorough).  Either
     add `wasteful / efficient` as a new pair, or accept efficient
     as a singleton.

vindictive → forgiving      |  forgiving → unforgiving
   ↳ resolution: vindictive and unforgiving are not the same concept
     (active revenge-seeking vs passive grudge-holding).  Either
     add `vindictive / merciful` as a new pair (and keep
     forgiving/unforgiving as a separate axis) or merge them.  The
     forgiving/unforgiving common-mode direction (Strategy 1a above)
     might be useful here — maybe vindictive sits on the
     emotional-engagement common-mode rather than on the
     forgiving/unforgiving axis itself.
```

**Tier D — NEW ANTONYM NEEDED (179 traits)**

Bulk of the work.  Each requires:

1. Write seed `<antonym>.json` with `positive_label`, `description`,
   `negative_label = non-<antonym>` (placeholder).
2. Run `regenerate_trait_instructions.py --traits <antonym> --force`.
3. Run `generate_antonyms.py --traits <antonym> <original>` to verify
   bidirectional match.
4. Update `negative_label` on both sides to point at each other.
5. Re-generate instructions for the antonym with the proper
   `negative_label` set.
6. Regenerate the derived lists: `uv run python tools/sync_entity_lists.py`.
7. Goal-classify in `data/goal_roles_and_traits.json` if applicable.
8. Run activation/vector extraction for the new pole.
9. Add to `pair_list_clean.json` (auto-propagates via
   `tools/build_pair_lists.py`).
10. Judge the pair in the next desc+inst batch (~$3/pair).

**Prioritisation within Tier D:** sort by the 2026-05-13 yield
scores; pick top entries from each Family 1–6 above first to maximise
dimensional spread before depth.  Specifically, the 5-pair starter
batch from Strategy 1 would each cost ~$15 (vector generation
+ judging both poles) and collectively add ~3–4 new effective
dimensions if the yields hold up after antonym creation.

### Combined batch suggestion (~$90 total, ~5–10 new effective dims)

- Tier A (free 2): `closure_seeking/open_ended`, `eloquent/plain_spoken`
- Tier B (fix 4): `altruistic/selfish`, `callous/compassionate`,
  `contrarian/conformist`, `eclectic/parochial`,
  `misanthropic/philanthropic`
- Tier D (top yield from Family 1–6): `calm/agitated`, `stoic/emotive`,
  `passive_aggressive/direct`, `epicurean/spartan`, `narrative/expository`
- Strategy 1a (orthogonal-by-construction):
  `political/apolitical` (orthogonal to progressive/conservative)

That's ~13 new pairs.  Estimated yield: ~+8–10 new effective dimensions
(from ~21 to ~31), assuming Strategy 1a items deliver on their
construction-orthogonality promise.  Cost: ~$24 (Tier A+B) + ~$50
(Tier D + 1a, including pole-vector creation overhead) = ~$74 expected,
$90–110 conservative.

### Re-running this analysis later

The yield numbers above are computed at the 2026-05-13 cohort state
(60 clean pairs).  After each round of additions, re-run the yield
analysis from scratch — the top-20 PC subspace shifts as the cohort
grows, so a trait that scored 0.8 yield against the 60-axis cohort
might score 0.6 against the 70-axis cohort.

If the yield-analysis becomes a recurring tool, a candidate name is
`tools/axis_yield_analysis.py` paralleling `tools/build_pair_lists.py`.


## Demographic coverage candidates (Sep 2026)

Candidates from the 2026-09-07 audit of the "vital and demographic" axes a
survey designer would include (age, cohort, life stage, sex/gender, sexual
orientation, race/ethnicity, nationality, migration status, language,
disability, veteran status).  Role-side findings and any role candidates live
in `../../roles/instructions/ROLES_TO_ADD.md`.

Status: **candidate list only**.  Nothing below has a seed file, description,
generated instructions, or antonym validation yet.  Naming decisions taken on
2026-09-07 are marked inline; anything unmarked is still a suggestion.  When promoting a candidate,
follow `AGENT_NOTES.md` § "Adding New Trait Clean Pairs": seed with
`negative_label = non-X`, generate, run `generate_antonyms.py` on both sides,
and only then point the labels at each other.  Categorical (unpaired)
candidates keep `non-X` permanently.

General caveats for this whole group:

- Sonnet 4 refused to generate `racist` / `racialist` (see "Moral circle"
  above) but accepted `ethnocentric`.  Expect similar filter behaviour on some
  membership traits, and expect stereotype-laden pos instructions on others.
  Prefer neutral heritage / background framing in the description and check the
  generated instructions by eye before accepting them.
- Membership traits are categorical, so most are unpaired.  A few natural
  bipolar pairs are marked below; everything else uses `non-X`.
- Check name collisions against `data/roles/instructions/` before creating a
  file (currently none of the names below collide with a role).

### Age (candidate matched pair)

Roles already cover infant → toddler → teenager/adolescent → student → graduate
→ newlywed → parent → grandparent/elder → retiree, but no trait pair defines
the axis.

- **young** ↔ **elderly** — the matched pair (decided 2026-09-07: `elderly`
  rather than `old`, because "old" drifts toward "old-fashioned" in
  generation).
- Birth cohort / generation: folded into age (the corpus has a fixed "now", so
  cohort is only visible as age).  No separate trait.
- Life stage: no easy matched pair; family track (single / newlywed / parent /
  grandparent, with divorcee / widow / orphan as off-ramps) and career track
  (student / graduate / retiree) are already covered by roles.  Not pursuing a
  trait for this.

### Sex and gender expression

- **masculine** ↔ **feminine** — candidate matched pair (gender expression).
  Bem's BSRI treats these as two independent scales, so the bipolar pair is a
  deliberate simplification.
- **neuter** and **nonbinary** — both, as unpaired extras (decided
  2026-09-07): `neuter` for the absence-of-gender-expression reading,
  `nonbinary` for the identity reading.
- **androgynous** — extra, unpaired (high on both BSRI scales).

Together these cover the large majority of people even though they skip some
of the identity-vs-expression distinction.

### Sexual orientation and sexuality (updated 2026-09-07)

- **straight** ↔ **gay** — the matched pair, in the sexual-orientation
  sense (decided 2026-09-07, Roger's names).  These are the everyday-register
  names for the `heterosexual` ↔ `homosexual` pair proposed earlier, so keep
  ONE naming: `straight` / `gay` unless Roger prefers the clinical forms
  (flagged, not yet confirmed).  Descriptions must pin the sense, since
  `straight` also reads as honest / direct and the antonym generator may
  otherwise return `crooked` or `devious`.  Storms' two-axis model
  (attraction to men × attraction to women) is the cleaner geometry, but the
  pair is the practical starting point.
- **bisexual**, **pansexual** — extras, unpaired (`pansexual` decided
  2026-09-07).  Near-duplicates in practice (attraction to more than one
  gender vs regardless of gender); expect their vectors to sit close.
- **asexual** — extra, unpaired.
- **kinky** — extra, unpaired (decided 2026-09-07); sexual practice rather
  than orientation.  The community antonym is `vanilla`, which would make a
  clean pair if generated, at the cost of `vanilla` also meaning "plain" in
  general use.

Distinct from the existing `lustful`/`chaste`, `flirty`/`professional` and
`romantic`/`non-romantic`, which cover intensity and framing of desire, not
its object or practice.  `lustful` and `flirty` generated without refusals,
so these should too; `kinky` is the one to watch.  Relationship structure
(monogamous and the poly family) is under "Coverage audit part 2" >
"Household and family".

### Race / ethnicity (coarse, categorical, all unpaired)

Deliberately coarse: roughly by continent, with a little more resolution in
Eurasia.  These are blurry artificial categories; the point is variation along
the axis, not a defensible taxonomy.  Terms as first proposed, with the
standard-usage replacements adopted on 2026-09-07:

| proposed | file name (decided 2026-09-07) | note |
|---|---|---|
| african | `african` | sub-Saharan; North Africa folds into middle_eastern (MENA) |
| european | `european` | |
| middle-eastern | `middle_eastern` | MENA |
| indian | `south_asian` | avoids collision with the `indian` *nationality* candidate below; covers the whole subcontinent |
| oriental | `east_asian` | "oriental" is dated / pejorative in current US usage; `east_asian` is the standard demographic term |
| (australian) aboriginal | `indigenous_australian` | Aboriginal and Torres Strait Islander |
| amerindian | `indigenous_american` | `native_american` if US-centric |

Optional extras, approved 2026-09-07 as optional (add only if more resolution
is wanted): `southeast_asian`, `pacific_islander`, `hispanic` (an ethnicity
that cuts across the above).
Skin tone is a separate 1-D axis, proposed under the physical-attribute
research track in "Coverage audit part 3".

Note that in-group *attitude* is already well covered (clannish, cliqueish,
insular, parochial, regionalist, sectarian, ethnocentric, nationalist,
patriotic vs cosmopolitan, humanitarian, universalist, inclusive); this section
is only about membership.

### Nationality (categorical, unpaired) plus candidate pairs for the space

Citizens of each G8 country plus any country over 500 M population not in the
G8.  G8 = Canada, France, Germany, Italy, Japan, Russia, UK, US; the only
non-G8 countries over 500 M are China and India. Also added Brazil to cover BRIC
and the largest countries from each otherwise-unrepresented continent.

- **american**, **canadian**, **french**, **german**, **italian**,
  **japanese**, **russian**, **british**, **chinese**, **indian**,
  **brazilian**, **australian**, **nigerian**.
- `american` is the conventional demonym for US citizens; `indian` here is the
  nationality (hence `south_asian` for the ethnicity above).
- Distinct from the existing attitude traits `nationalist` / `patriotic`.

Candidate matched pairs that would give the nationality set a coordinate
system:

1. **northern** ↔ **southern** and **eastern** ↔ **western** — geographic
   hemispheres, approved 2026-09-07.  Define them as the halves of a standard
   Greenwich-centred world map: north / south of the equator, and east / west
   with the split running down the mid-Pacific, so `western` = the Americas
   and `eastern` = Europe, Africa, Asia and Oceania.  Write the descriptions
   in those geographic terms so `western` does not collapse into the cultural
   West-vs-non-West reading, which would otherwise dominate generation.
2. **Inglehart–Welzel cultural map**, two matched pairs — the arrangement that
   actually predicts how a national persona answers value questions.  Named
   per the `<pole> <Standard>` convention (AGENT_NOTES § "Standard-derived
   trait labels"): the standard's own pole names, capitalised suffix in the
   label, lowercased in the stem, provenance in a `source` field, description
   a paraphrase of the canonical definition without naming the source.
   - axis 1: **traditional (Inglehart-Welzel)** ↔ **secular-rational
     Inglehart-Welzel** (stems `traditional_inglehart_welzel`,
     `secular_rational_inglehart_welzel`): deference to religious / family /
     national authority vs rational-legal.  The suffix also keeps these
     distinct from the existing `traditional` (↔ innovative) and `secular`
     (↔ religious) traits, which have different scopes.
   - axis 2: **survival (Inglehart-Welzel)** ↔ **self-expression
     Inglehart-Welzel** (stems `survival_inglehart_welzel`,
     `self_expression_inglehart_welzel`): economic / physical security first
     vs autonomy, tolerance, participation.
   Expect partial overlap with existing religious/secular, conservative/
   progressive and the moral-circle traits; run the Strategy 1 yield check
   (above) before committing.
3. Hofstede's six dimensions: four are already present as pairs
   (individualism/collectivism = individualistic/collectivistic; power
   distance ≈ egalitarian/elitist; uncertainty avoidance ≈ cautious/bold or
   risk_taking/risk-averse; indulgence/restraint ≈ hedonistic/disciplined).
   The two NOT covered by a dedicated pair are **masculinity/femininity**
   (renamed "motivation towards achievement and success" in 2023: tough,
   competitive, achievement-driven vs tender, caring, quality-of-life;
   nearest existing pair competitive/cooperative, and distinct from the
   gender-expression pair masculine/feminine proposed above) and
   **long-term/short-term orientation** (persistence, thrift, adapting
   tradition to the future vs respect for tradition, face, quick results;
   nearest existing strategic/tactical, hedonistic/disciplined, futuristic).
   Decided 2026-09-07: Hofstede is NOT adopted by name; its uncovered
   axes are treated as ordinary pair gaps instead.  See § "Coverage audit
   part 2" > "TBD: Hofstede and Schwartz gap-filling".

### Disability (non-neurological only, categorical, unpaired)

- **deaf**
- **blind**
- **mobility_impaired** (decided 2026-09-07 over `wheelchair_user`: adjective
  form parallel to deaf / blind, and not tied to one aid)

Neurological / neurodevelopmental / psychiatric disability was deliberately not
pursued in this first pass (the psychiatric side being approximated by
anxious, neurotic, manic, melancholic, paranoid, obsessive, avoidant,
fragile).  Superseded 2026-09-08: `ADHD`, `autistic`, `dyslexic` and a
set of clinical traits are now candidates under "Coverage audit part 3".  These three are the
obvious non-neurological cases and are the ones most likely to show up in how a
persona describes its sensory / physical world in text.

### Not pursued from this audit

- Native language / languages spoken: would put non-English text in front of
  English-only judges.  Only response-language *proficiency* (fluent ↔ limited
  second-language speaker) is measurable; not added for now.
- Veteran status: covered by the `veteran` / `soldier` roles vs the corpus
  default; no trait needed.
- Life stage, migration / refugee status: covered by roles (see
  `ROLES_TO_ADD.md`).

## Coverage audit part 2: household to political (Sep 2026)

Second pass of the 2026-09-07 audit, over the household / place /
socioeconomic / education / work / social-class / religion / cultural-
orientation / ethnic-identity / anthropological / political categories.
Same status as the demographic section above: **candidate list only**, same
process (`non-X` seed, generate, antonym check both ways, then point the
labels at each other), same caveats.  Role candidates from this pass are in
`../../roles/instructions/ROLES_TO_ADD.md` § "Coverage audit part 2".
"Decided" means Roger approved the axis; "wording TBD" means the labels are
still open.  "Considered, not adopted" items were proposed and not taken up;
they await Roger's confirmation before being struck.

Conclusions carried over from the audit table: the corpus has attitudes
toward money, class, education and religion but almost no variation in
*having* them (nobody is rich, poor, uneducated, working-class, rural or of a
named faith), so most pairs below are corpus-expansion problems that PCA of
the current entity pool cannot surface (see "Caveat: model-variance-bound"
above).  Suggested analysis groupings: a single composite *status* axis
first (wealth + attainment + occupational prestige + class identification,
which is what Hollingshead / Duncan / NS-SEC compute), then Bourdieu capital
composition and work intensity as separate, orthogonal analyses; religion
on its own; place rootedness together with migration and identity.

### Household and family (decided 2026-09-07)

- **married** ↔ **unmarried** -- partnership status.  Partnership *history*
  stays categorical via the existing roles (newlywed, divorcee, widow).
- **parent** ↔ **childless** / **childfree** -- parenthood; exact wording
  TBD.  The trait stem `parent` would collide with the existing `parent`
  *role* (a tenth role/trait collision, see AGENT_NOTES § "Trait/role name
  collisions"); `parental` avoids it if that matters.
- **only_child** ↔ **many_siblings** -- sibling count (decided 2026-09-07);
  the one family-of-origin facet worth a pair.
- Relationship structure (decided 2026-09-07): **monogamous**,
  **polygamous**, **polygynous**, **polyandrous**, **polyamorous**, all
  seeded `non-X`.  Structure: `monogamous` is the central pole;
  `polygamous` (plural marriage, the umbrella term) and `polyamorous`
  (plural consensual relationships, not necessarily marriage) are its two
  alternatives, and `polygynous` (one husband, several wives) and
  `polyandrous` (one wife, several husbands) are the gendered subtypes of
  `polygamous`.  Run the antonym check on `monogamous` to see whether the
  generator picks `polygamous` or `polyamorous` as its partner; the other
  becomes a sibling pointing at `monogamous`, per the triangle convention.
  This also covers "marriage form" from the anthropology row.
- Considered, not adopted: roles `spouse`, `empty_nester`, `adoptee`,
  `twin`.  Fertility intentions, birth order, adoption and twin status: not
  pursued.

### Place (decided 2026-09-07)

- **rural** ↔ **urban** -- density.
- **suburban** -- unpaired extra (`non-suburban`): the midpoint of the
  density axis as its own trait (decided 2026-09-07).
- **settled** ↔ **nomadic** -- rootedness / residential mobility.  Distinct
  from the moral-circle attitude spectrum (parochial ... cosmopolitan) and
  from the `nomad` role, which stays.
- Roles: `farmer`, `villager` (ROLES_TO_ADD).
- Considered, not adopted: role `suburbanite` (superseded by the `suburban`
  trait).  Neighbourhood disadvantage folds into the status axis; commute
  and environmental exposure are not persona-visible: not pursued.

### Socioeconomic status, objective (decided 2026-09-07)

- **poor** ↔ **wealthy** -- the wealth axis (income, assets, housing and
  deprivation together).  Expect stereotyped generation on the poor pole;
  read the instructions before accepting them.
- **precarious** ↔ **financially_secure** -- security as distinct from
  wealth level (a middle-income persona one setback from ruin vs one with
  a cushion); decided 2026-09-07.
- Roles: `billionaire`, `heir`, `aristocrat`, `pauper`, `homeless`,
  `laborer`, `beggar` (ROLES_TO_ADD; US spelling, decided 2026-09-07).
- Considered, not adopted: role `debtor`.

### Education (decided 2026-09-07)

- **uneducated** ↔ **educated** -- attainment ("left school early" vs
  postgraduate framing, not intelligence).
- **illiterate** ↔ **literate** -- literacy as ability, distinct from
  attainment.  Generation risk: an illiterate persona that writes fluent
  prose is a contradiction the model may resolve by faking dialect or
  misspelling; check what it does with it.
- Roles: `dropout`, `professor`.
- Considered, not adopted: `tech_savvy` ↔ `technophobic` (digital literacy;
  the luddite / technologist roles exist); `simple` as the real partner for
  `erudite`; roles `autodidact`, `apprentice`.  Field of study is already
  spanned by the discipline roles; institution selectivity and credentials:
  not pursued.

### Work and occupation (roles decided 2026-09-07, pairs TBD)

- Roles: `farmer`, `fisher`, `machinist`, `operator`, `driver`, `laborer`,
  `cleaner`, `unemployed`, `freelancer`, `intern`, `subordinate`
  (ROLES_TO_ADD) -- fills ISCO major groups 6, 8 and 9 and the
  employment-status gaps.
- Prediger's People ↔ Things and Data ↔ Ideas axes: decided 2026-09-08 to
  seed Holland's six RIASEC types as a `ring` instead (see "Coverage
  audit part 4" > "Sets and rings") and read the two Prediger axes off
  the hexagon; no Prediger pairs are seeded.
- **industrious** -- new trait as the real partner for the existing
  `slothful`, whose `negative_label` is already `industrious`: the
  work-intensity axis.  Decided 2026-09-07.  Tier D route: seed
  `industrious` with `non-industrious`, generate, run the antonym check
  both ways, then set `industrious.negative_label = slothful` and
  regenerate `industrious` only.  Same matched-pair caveat as the
  political half-pair completions (Political and civic, item 4).
- Considered, not adopted: role `civil_servant`.  Union membership,
  telework, tenure: not pursued.

### Social class (decided 2026-09-07, wording TBD)

- **high social class** ↔ **low social class** -- subjective class
  identification plus habitus and register; the composite status axis.
  Wording TBD: `upper_class` ↔ `working_class` is the obvious pair.
- **highbrow** ↔ **lowbrow** -- taste, distinct from `artistic`(philistine)
  which is engagement with art rather than taste level.
- Roles: `aristocrat`, `socialite`, `peasant`.
- **self_made** ↔ **old_money** -- class origin and mobility (decided
  2026-09-07).  This is also the ascribed-vs-achieved status axis floated
  for the anthropology row, so that candidate is probably redundant now.
- Bourdieu capital composition (economic ↔ cultural), under discussion
  (options in chat 2026-09-07): (a) derive it with no new files:
  composition = the cultural-capital direction (`educated`, `highbrow`)
  minus the economic direction (`wealthy`), volume = their sum, all
  adopted above; (b) one dedicated archetype pair `nouveau_riche` ↔
  `shabby_genteel` (the two off-diagonal corners: money without culture vs
  culture without money; the `bohemian` role is the textbook low-economic /
  high-cultural case); (c) the standard-name form `economic-capital
  Bourdieu` ↔ `cultural-capital Bourdieu`.  Optional extras for the other
  two capitals: `well_connected` ↔ `friendless` (social capital; networker /
  loner roles exist, gregarious(solitary) is a half-pair) and `famous` ↔
  `obscure` (symbolic capital; the `celebrity` role exists).  Recommended:
  (a) first, (c) as the check, (b) as two roles if the archetypes are
  wanted.  Options (a)-(c) and `famous` ↔ `obscure` are still open;
  Roger deferred them on 2026-09-28 (chunk 7).
- **well_connected** ↔ **friendless** -- social capital (decided
  2026-09-07).
- **solitary** -- new trait as the real partner for the existing
  `gregarious`, whose `negative_label` is already `solitary` (decided
  2026-09-07; Tier D route as for `industrious`).  Overlaps to check:
  introverted ↔ extroverted, the `loner` / `hermit` roles, and the new
  `friendless` (solitary is chosen solitude; friendless is lack of ties).
  Same matched-pair caveat as the political half-pair completions
  (Political and civic, item 4): the generator may return `friendless` or
  `isolated` for `solitary` rather than `gregarious`.
- Considered, not adopted: role `servant`.

### Religion (decided 2026-09-07)

- Membership traits for eight major world religions, unpaired (`non-X`),
  all decided 2026-09-07: **christian**, **muslim**, **hindu**,
  **buddhist**, **jewish**, **sikh**, **taoist**, **shinto**.  By
  adherents: Christianity ~2.4 B, Islam ~1.9 B, Hinduism ~1.2 B, Buddhism
  ~0.5 B, Chinese folk religion / Taoism and Shinto ~0.1 B each (both
  heavily syncretic, so the weakest personas), Sikhism ~30 M, Judaism
  ~15 M.  Label capitalisation TBD (`Christian` vs `christian`; stems are
  lowercase either way).
- **intrinsic (Allport)** ↔ **extrinsic (Allport)** (stems
  `intrinsic_allport`, `extrinsic_allport`) -- Allport and Ross's intrinsic
  (religion lived as an end in itself) vs extrinsic (religion used as a
  means: comfort, sociability, status) religious orientation; the
  intensity-within-affiliation axis, orthogonal to *which* religion.
  Standard-name convention applies: the description paraphrases without
  naming Allport, `source` field carries the citation.
- **atheist**, **pagan** -- two more unpaired membership traits (decided
  2026-09-07): `atheist` adds explicit non-belief on top of the `secular`
  reasoning framework; `pagan` covers contemporary paganism / animism (the
  `witch` and `shaman` roles are the nearest existing personas).
- Roles: `priest`, `monk`, `convert`.
- Considered, not adopted: `modernist` as the real partner for
  `fundamentalist`.  Upbringing and conversion history beyond the `convert`
  role: not pursued.

### Cultural orientation (decided 2026-09-07)

Reading guide delivered in chat on 2026-09-07 (Hofstede, Schwartz, GLOBE,
Inglehart-Welzel, Triandis, Gelfand, Leung and Cohen, Benedict, Hall,
Markus and Kitayama).  Decisions:

- **tight (Gelfand)** ↔ **loose (Gelfand)** (stems `tight_gelfand`,
  `loose_gelfand`) -- how strong the persona expects social norms to be and
  how readily it sanctions deviance in others (Gelfand et al. 2011;
  standard-name convention, `source` carries the citation).  Distinct from
  conformist / nonconformist (the persona's own compliance) and
  regulatory(rule-flexible) (formal rules): this is norm enforcement.
- **high-context (Hall)** ↔ **low-context (Hall)** (stems `high_context_hall`,
  `low_context_hall`) -- meaning carried by relationship, setting and
  implication vs by explicit words (Hall 1976).  Expect overlap with
  blunt / tactful, literal(figurative), circumspect(brash) and
  enigmatic(straightforward); run the Strategy 1 yield check before
  judging.
- **monochronic (Hall)** ↔ **polychronic (Hall)** (stems `monochronic_hall`,
  `polychronic_hall`) -- one thing at a time with schedules binding vs
  several things at once with relationships ahead of appointments (Hall
  1983).  Proxies: methodical / improvisational, urgent(relaxed),
  impatient ↔ patient; yield check first.
- **2-simplex: honor culture / dignity culture / face culture** (stems
  `honor_culture`, `dignity_culture`, `face_culture`; `source` Leung and
  Cohen 2011) -- three cultural logics of personal worth: worth claimed from
  others and defended against insult; worth inherent and internal, so
  insults can be ignored; worth conferred by one's place in a hierarchy and
  kept through humility and harmony.  The standard's own terms serve as
  labels (US spelling `honor`), so no author suffix.  Nearest existing:
  vindictive(forgiving), unforgiving, militant(peaceful) on the honor side;
  no dignity or face pole today.
- **2-simplex: guilt-prone / shame-prone / fear-prone** (stems
  `guilt_prone`, `shame_prone`, `fear_prone`; `source` Benedict 1946 and
  the guilt-shame-fear spectrum) -- what regulates the persona's conduct:
  an internal conscience, the judgment of others, or the power of those who
  can punish (authorities, spirits, fate).  Descriptions must frame
  fear-prone as a control logic, not anxiety, or it collapses onto
  `anxious` / `paranoid` / the `prey` role; likewise guilt-prone is not
  `conscientious` and shame-prone is not `reserved`.

Simplex bookkeeping (interim): until the clean-pair concept is generalised
to N-simplices (open idea under "Clean antonym pairs (candidates)"), seed
every corner of a simplex with `non-X`, and record membership here rather
than in the JSON.  Current simplex list: compassionate / malicious /
callous; conformist / contrarian / nonconformist; honor / dignity / face
culture; guilt- / shame- / fear-prone; the social-value-orientation
triangle cooperative / selfish / competitive (all three exist already);
the moral-circle spectrum (higher-dimensional and elongated; cosmopolitan
is one corner).

Already covered, nothing to add: Triandis horizontal / vertical (individualistic / collectivistic
crossed with egalitarian / elitist), Markus and Kitayama self-construal
(individualistic / collectivistic), the nine GLOBE dimensions (proxies for
all).  Inglehart-Welzel: already adopted under that name (Nationality
section above).  Hofstede and Schwartz: next subsection.

### Hofstede and Schwartz gap-filling (decided 2026-09-08)

Neither framework is adopted wholesale by name.  Instead, each of their axes
that lacks a good matching pair is a candidate gap to fill, either with a
new pair or by completing an existing half-pair (a trait whose
`negative_label` is a real word with no trait file; the Tier D route
above).  Roger's decisions of 2026-09-08 are in the last column; every
"complete" item carries the matched-pair caveat from Political and civic
item 4 (the existing side's negative_label was the generator's free
choice, so the reverse direction is untested).

Hofstede's six dimensions:

| dimension | current coverage | decision (2026-09-08) |
|---|---|---|
| power distance | egalitarian ↔ elitist (clean); obedient ↔ rebellious; dominant ↔ submissive; deferential (placeholder neg) | covered; a real partner for `deferential` was floated as optional and not decided |
| individualism / collectivism | individualistic ↔ collectivistic (clean); independent(dependent) | covered |
| masculinity / femininity ("motivation towards achievement and success" since 2023) | competitive ↔ cooperative (clean) covers the contest half; the tough-vs-tender, achievement-vs-quality-of-life half has only nurturing(neglectful), acerbic(tender), assertive(noncommittal) | **decided: complete `nurturing` with `tough`** ("or similar": `hard-nosed`, `tough-minded` are the fallbacks). `nurturing`'s negative_label changes from `neglectful` to the chosen name once the file exists. `tough` alone risks reading as durable / resilient (resilient ↔ fragile exists), so the description must pin the Hofstede sense: assertive, demanding, achievement-first, unsentimental. The generator may return `gentle` or `soft` for it rather than `nurturing`. |
| uncertainty avoidance | two half-pairs: cautious(bold), risk_taking(risk-averse) | **decided: complete both**, `bold` for `cautious` and `risk_averse` for `risk_taking`. The two axes will be near-duplicates; run the yield check and note which is the more orthogonal. |
| long-term / short-term orientation | half-pairs only: strategic(tactical), hedonistic(disciplined), impulsive(deliberate), nostalgic(modern); futuristic (placeholder neg) | **decided: new pair `long-term oriented` ↔ `short-term oriented`** (stems `long_term_oriented`, `short_term_oriented`): persistence, thrift and deferred reward vs quick results and present consumption. Overlaps to check: hedonistic(disciplined), impulsive(deliberate), strategic(tactical). |
| indulgence / restraint | several half-pairs: hedonistic(disciplined), ascetic(indulgent), gluttonous(abstemious), epicurean(spartan) | **decided: complete one** (Roger, choice delegated): `ascetic` ↔ `indulgent`, because the antonymy is cleaner and `hedonistic`'s description (instant gratification over long-term considerations) overlaps the new long-term pair. The description must pin `indulgent` as self-indulgent (gratifying one's own appetites), not lenient toward others. `hedonistic`(disciplined) stays a half-pair. |

Schwartz's cultural-level orientations (seven, on three bipolar axes):

| axis | current coverage | gap-filling route (TBD) |
|---|---|---|
| embeddedness ↔ autonomy (intellectual, affective) | individualistic ↔ collectivistic; curious ↔ incurious (intellectual autonomy); hedonistic(disciplined) (affective autonomy) | covered |
| hierarchy ↔ egalitarianism | egalitarian ↔ elitist | covered |
| mastery ↔ harmony | no pair: conciliatory ↔ confrontational is interpersonal only; ecocentric ↔ anthropocentric and environmental(economic) cover harmony-with-nature only | **approved in principle, names TBD** (Roger, 2026-09-08: `mastery` / `harmony` need better names). The axis: actively changing, directing and exploiting the social and natural world to get ahead vs fitting into it and accepting it as it is. Options: `world-shaping` ↔ `world-accepting` (recommended: says what the axis is), `striving` ↔ `accepting` (`accepting` alone reads as tolerant of people), `ambitious` ↔ `contented` (drifts toward the drive axis). Overlaps to check: proactive ↔ reactive, conciliatory ↔ confrontational, ecocentric. |

Schwartz's individual-level circle (ten values, two bipolar axes):
openness-to-change ↔ conservation is covered by innovative ↔ traditional
and conservative ↔ progressive; self-enhancement ↔ self-transcendence is
nearly covered by altruistic(selfish) + selfish(unselfish), i.e. the Tier B
`selfish.neg → altruistic` fix above.

### Ethnic and cultural identity (decided 2026-09-07)

Reading guide delivered in chat on 2026-09-07 (Phinney's MEIM, Cross and
Helms, Berry, Benet-Martínez, diaspora, indigeneity, cosmopolitanism,
heritage language).  Decisions:

- **heritage_proud** ↔ **deracinated** -- strength of ethnic-identity
  commitment, after the commitment subscale of Phinney's MEIM: a persona
  for whom heritage is central and a source of pride vs one cut off from,
  or indifferent to, its roots.  Distinct from the moral-circle loyalty
  traits (ethnocentric, nationalist, clannish), which are about whom one
  favours rather than how much one identifies.
- Roles `assimilated` (second generation, host culture only) and
  `marginalized` (neither heritage nor host) complete Berry's acculturation
  square alongside the existing immigrant (integration), exile
  (separation) and refugee corners; both axes (heritage maintenance, host
  participation) can then be read off the four corners in embedding space.
- Optional role for the diaspora gap: `diaspora_member` rejected as a name
  (Roger, 2026-09-08: it implies a specific diaspora).  `emigree` was
  floated and does not cover the gap: an émigré is an individual who left,
  usually for political reasons, which the exile / expatriate / immigrant
  roles already span; the gap is the multi-generational community identity
  held abroad by someone who may never have lived in the homeland.
  Undecided; `diasporan` is the generic name if the gap is wanted.
  Racial-identity stage models and heritage language: not pursued.

### Anthropological social structure (decided 2026-09-07)

- Roles (decided 2026-09-07): `chief`, `monarch`, `herder`,
  `hunter_gatherer`, `initiate` (ROLES_TO_ADD; `peasant` and `farmer` were
  already adopted under Social class and Place).  `hunter_gatherer` is the
  serious counterpart of the caricature `caveman` role, which stays.
- Marriage form is covered by the monogamous / polygamous / polygynous /
  polyandrous / polyamorous set under "Household and family".
- **Gemeinschaft (Tönnies)** ↔ **Gesellschaft (Tönnies)** (stems
  `gemeinschaft_tonnies`, `gesellschaft_tonnies`: diacritics fold to ASCII
  in the stem, the label keeps them) -- decided 2026-09-07.  Describes a
  person *native to* such a society, not a theory of society: someone
  whose ties are kinship, place and custom, who treats relationships as
  ends and locates themself in family, village and faith, vs someone whose
  ties are contract, role and interest, who weighs relationships by what
  they yield and locates themself in profession, market and state.
  Durkheim's mechanical vs organic solidarity is the same axis seen from
  the division of labour (with the biological metaphor reversed).
  Overlaps to check: collectivistic ↔ individualistic, traditional ↔
  innovative, the `provincial` role.
- **aristocratic** ↔ **meritocratic** -- decided 2026-09-07: the normative
  reading of ascribed vs achieved status (Linton), i.e. how status *should*
  be allocated (birth, rank and title vs merit and achievement).  Distinct
  from `elitist` ↔ `egalitarian` (whether hierarchy should exist at all)
  and from `self_made` ↔ `old_money` (the persona's own origin); by
  construction it lives inside the hierarchy-accepting half of
  elitist / egalitarian.  Parsons's ascription / achievement pattern
  variable (societal type) is not pursued: redundant with the Tönnies pair.
- Kinship, descent, residence, ritual and dialect: not pursued (the
  register pairs cover dialect; `ritualistic` exists).

### Political and civic (decided 2026-09-07)

- **political** ↔ **apolitical** -- political interest and engagement as
  such (decided 2026-09-07); also orthogonal-by-midpoint candidate 1 under
  Strategy 1a above, by construction orthogonal to conservative ↔
  progressive.
- Roles (decided 2026-09-07): `politician`, `bureaucrat`, `voter`,
  `lobbyist`, `demagogue` (ROLES_TO_ADD).
- Political compass and neighbours (decided 2026-09-07; Roger accepted the
  chat recommendations for rows 1-5):
  1. **socialist** ↔ **capitalist** -- the economic axis.  Both are
     identity words, so expect caricature in generation; the neutral
     policy-stance fallback is `interventionist` ↔ `free-market` if the
     generated instructions are unusable.
  2. **authoritarian** ↔ **civil-libertarian** (stems `authoritarian`,
     `civil_libertarian`) -- the authority axis, framed as views on state
     power over individuals.  The description must say so explicitly, or
     it collapses onto obedient ↔ rebellious / dominant ↔ submissive.  The
     existing `libertarian` trait is the free-will sense and keeps its
     name.
  3. **populist** ↔ **technocratic** -- "the people vs the corrupt elite,
     common sense" against "expertise, procedure, evidence".
  4. Half-pair completions, one new file each, Tier D route as for
     `industrious`: **partisan** (partner of `diplomatic`, whose
     negative_label is already `partisan`); **extremist** (partner of
     `moderate`, whose negative_label is `extreme`: change it to
     `extremist` once the file exists); **incrementalist** (partner of
     `radical`, negative_label `incremental`: change to `incrementalist`);
     **empowered** (partner of `fatalistic`, negative_label already
     `empowered`; doubles as political efficacy).
     Matched-pair caveat (Roger, 2026-09-07): these may well fail the
     bidirectional antonym check and will need resolving case by case.
     The existing side was written first and its negative_label was the
     generator's free choice, so the reverse direction is untested:
     `extremist` may come back as `centrist` or `mainstream` rather than
     `moderate`; `incrementalist` as `revolutionary` rather than
     `radical`; `partisan` as `nonpartisan`, `bipartisan` or `neutral`
     rather than `diplomatic` (whose own sense is interpersonal tact as
     much as politics); `empowered` as `powerless` or `helpless` rather
     than `fatalistic`.  Resolutions available: accept the pair on the
     strength of one direction as a manual pairing (as analytical /
     systems_thinker and closure_seeking / open_ended are today); relabel
     the existing side to the generator's suggestion if that is the
     better antonym; or keep the new trait unpaired with `non-X`.
  5. Role `volunteer` (ROLES_TO_ADD) for civic participation; no second
     engagement pair on top of political ↔ apolitical.
  Not adopted (decided 2026-09-07): an institutional-trust pair
  (anti-institutional is the populist corner); party identification
  (categorical, country-specific).  Not needed: postmaterialism
  (= `self-expression (Inglehart-Welzel)`); nationalism and patriotism exist.

## TODO: antonym gap-filling pass (queued 2026-09-08)

Queued by Roger for the current data-directory expansion week, to run
after the coverage-audit additions above are in: attempt Tier D and finish
the Tier A-C cases from "Strategy 2: by ease-of-addition" (2026-05-13).
Census as of 2026-09-08 (302 traits), the dated successor to that
inventory:

| `negative_label` points to | traits |
|---|---|
| an existing trait, reciprocated (57 clean pairs) | 114 |
| an existing trait, one way only | 9 |
| a real word with no trait file (Tier D proper) | 133 |
| a placeholder: 37 `non-X`, 9 negations such as `unadventurous` | 46 |

133 + 46 = 179 is exactly the May Tier D count, so nothing has been filled
since; the April 2026 round (five pairs plus `indecisive` and
`nonconformist`) predates that inventory.  Ten of the 133 are already
queued through the coverage audit above: `bold`, `risk_averse`,
`indulgent`, `industrious`, `solitary`, `partisan`, `empowered`,
`extremist`, `incrementalist`, `tough`.  One label, `specialist` (from
`generalist`), exists as a role, so a trait file would be a tenth
role/trait name collision.

Order of work:

1. **Tier A (2)**: `closure_seeking / open_ended`, `eloquent /
   plain_spoken`.  Both poles exist; add to `pair_list_clean.json` via
   `tools/build_pair_lists.py` and judge.
2. **Tier B (4)**: re-run `generate_antonyms.py` on each, then edit the
   placeholder side: `selfish.neg -> altruistic`, `parochial.neg ->
   eclectic`, `philanthropic.neg -> misanthropic`.  `compassionate` and
   `conformist` are triangles and stay as they are.
3. **Tier C (3 tangled)**: educational / superficial / thorough,
   efficient / thorough, vindictive / forgiving / unforgiving; resolve per
   the notes under Strategy 2.  These are the cases that proved
   non-trivial.
4. **Tier D (133 + 46)**: seed the antonym file with `non-X`, generate,
   antonym-check both ways, point the labels at each other (AGENT_NOTES
   § "Adding New Trait Clean Pairs").  Prioritise by the 2026-05-13 yield
   scores (Families 1-6 above), then the rest.  Expect the matched-pair
   caveat (Political and civic, item 4) to bite on a fair fraction: the
   existing label was the generator's free choice, so the reverse
   direction is untested; resolutions are a manual pairing, a relabel of
   the existing side, or leaving the new trait unpaired.
5. **The 9 one-way pointers**: decide per case whether the reverse side is
   a triangle (callous, contrarian, nonconformist), a Tier C tangle
   (educational, efficient, vindictive), or a Tier B fix (altruistic,
   eclectic, misanthropic).

Cost: generation is a handful of Sonnet calls per trait, well under $1
each; the expensive part is the RunPod activation extraction for ~180 new
traits on top of the ~90 coverage-audit entities, which needs the
expensive-operations confirmation.  To list the current gap (labels with no
file, `non-` prefixed ones excluded; prints 139 as of 2026-09-08: the 133
real-word labels minus the three that happen to start with `non-`,
`non-strategic`, `non-contemplative`, `non-ceremonial`, plus the 9
negation placeholders):

```bash
python3 -c "import json,glob,os,re; T={os.path.basename(f)[:-5]:json.load(open(f)) for f in glob.glob('data/traits/instructions/*.json')}; n=lambda s:re.sub(r'[\s\-]+','_',s.lower()); print(sorted(s for s,d in T.items() if n(d['negative_label']) not in T and not n(d['negative_label']).startswith('non_')))"
```

## TODO: personality-taxonomy import, Big Five and HEXACO (queued 2026-09-08)

Decided by Roger on 2026-09-08 after the reading guide on personality
taxonomies (chat, 2026-09-08).  Three parts, in order of commitment:

1. **Definitely**: add the Big Five and HEXACO by name, as 5 and 6
   standard-named trait-pair axes (22 files).
2. **Definitely**: go through every aspect and facet of the finer-grained
   instruments (10 BFAS aspects, 30 NEO / IPIP-NEO facets, 24 + 1 HEXACO
   facets) and check whether each is already reasonably covered by an
   existing plain trait; where it is not, add a plain (non-suffixed) trait
   or pair to cover the gap.
3. **Possibly**: the extra work of adding the aspects and/or facets
   themselves as larger, more detailed systems of standard-named axes.

Naming, descriptions and the `source` field follow AGENT_NOTES
§ "Standard-derived trait labels" (`<pole> <Standard>`; the standard's own
pole names where it has them; description paraphrases the canonical
definition without naming the source; US spelling; diacritics fold in the
stem).

### Part 1: Big Five and HEXACO by name

Every axis is bipolar, so each is one pair.  High-pole names are canonical
throughout; low-pole names mostly are not (only Introversion and Emotional
Stability are), so the low-pole words below are proposals to confirm,
chosen from each instrument's own low-scorer descriptions or adjective
lists.  The suffix marks the *axis* as the standard's even where the
low-pole word is ours.

| pair (labels) | stems | low-pole word from | note |
|---|---|---|---|
| **open (Big Five)** ↔ **closed (Big Five)** | `open_big_five`, `closed_big_five` | "closedness" (conventional) | description must say "to experience": imagination, aesthetics, feelings, ideas, values; the suffix keeps it apart from `open_ended` and `transparent` |
| **conscientious (Big Five)** ↔ **careless (Big Five)** | `conscientious_big_five`, `careless_big_five` | mirrors the ad-hoc pair conscientious ↔ careless (Costa and McCrae's own term is "lack of direction"; `undirected` is the alternative) | mirroring the ad-hoc names makes the wording comparison cleanest |
| **extraverted (Big Five)** ↔ **introverted (Big Five)** | `extraverted_big_five`, `introverted_big_five` | canonical | note the standard's spelling `extraverted`; the ad-hoc trait is `extroverted` |
| **agreeable (Big Five)** ↔ **antagonistic (Big Five)** | `agreeable_big_five`, `antagonistic_big_five` | Costa and McCrae's "antagonism" | ad-hoc `agreeable` exists with placeholder neg `disagreeable` |
| **neurotic (Big Five)** ↔ **emotionally-stable (Big Five)** | `neurotic_big_five`, `emotionally_stable_big_five` | canonical (Goldberg names the factor Emotional Stability) | ad-hoc `neurotic`(stable) exists |
| **honest-humble (HEXACO)** ↔ **sly (HEXACO)** | `honest_humble_hexaco`, `sly_hexaco` | authors' low adjectives: sly, deceitful, greedy, pretentious, hypocritical, boastful, pompous | alternatives `deceitful (HEXACO)`, `self-serving (HEXACO)`; the genuinely new axis |
| **emotional (HEXACO)** ↔ **tough (HEXACO)** | `emotional_hexaco`, `tough_hexaco` | authors' low adjectives: brave, tough, independent, self-assured, stable | overlaps in meaning with the plain `tough` approved as `nurturing`'s partner (Hofstede tough / tender is close to this axis); alternative `self-assured (HEXACO)`.  Not the same as Neuroticism: no anger, plus sentimentality and dependence |
| **extraverted (HEXACO)** ↔ **introverted (HEXACO)** | `extraverted_hexaco`, `introverted_hexaco` | canonical | near-duplicate of the Big Five pair and of the ad-hoc pair |
| **agreeable (HEXACO)** ↔ **quarrelsome (HEXACO)** | `agreeable_hexaco`, `quarrelsome_hexaco` | authors' low adjectives: ill-tempered, quarrelsome, stubborn, choleric (the factor is "Agreeableness versus Anger") | rotated relative to Big Five Agreeableness: patience, forgiveness, gentleness, flexibility |
| **conscientious (HEXACO)** ↔ **careless (HEXACO)** | `conscientious_hexaco`, `careless_hexaco` | authors' low adjectives: sloppy, negligent, reckless, lazy, irresponsible | near-duplicate of the Big Five and ad-hoc pairs |
| **open (HEXACO)** ↔ **conventional (HEXACO)** | `open_hexaco`, `conventional_hexaco` | authors' low adjectives: shallow, unimaginative, conventional | near-duplicate of the Big Five pair; leans to unconventionality and aesthetics |

Descriptions and sources.  Big Five: paraphrase the NEO-PI-R domain
definitions (Costa and McCrae 1992; the John and Srivastava 1999 taxonomy
chapter has the same content in the open); `source` = "Costa & McCrae,
NEO-PI-R, <domain> domain".  HEXACO: paraphrase the two-sided sentences
on https://hexaco.org/scaledescriptions (one sentence for very high scorers,
one for very low, per factor: exactly the shape needed for both poles);
`source` = "Ashton & Lee, HEXACO-PI-R, <factor> scale".  Both poles get
the "This means ..." form; the low pole is written from the low-scorer
sentence, not as a negation of the high pole.

Process.  Same as any clean pair: seed both files with `non-X`, generate,
run `generate_antonyms.py` on both, then point the labels at each other.
The pairing is fixed by the standard, so the antonym check is a sanity
check on the generated instructions, not the decision: if the generator
returns the *ad-hoc* twin instead (`extraverted (Big Five)` -> `introverted`)
that is the expected duplicate showing, not a failure.

The duplicates are the point (label-convention rule 5: "the comparison is
the experiment").  After extraction, Extraversion, Conscientiousness and
Openness each exist three times (Big Five, HEXACO, ad hoc; Openness twice,
having no ad-hoc pair) with near-identical one-sentence descriptions;
Agreeableness and Neuroticism / Emotionality exist as rotated cousins.
Cosines within each triplet estimate how much description wording alone
moves a vector, i.e. the description noise floor for the whole corpus; the
Big Five vs HEXACO Agreeableness and Neuroticism vs Emotionality angles
test whether the model reproduces the documented rotation (anger moving
from N to low A; sentimentality and dependence joining E).  Expect
triplet cosines well above 0.9; anything lower is informative.

### Part 1b (definitely): the 16 MBTI types

Decided by Roger on 2026-09-08.  Rationale: the four MBTI dichotomies sit
diagonally across four Big Five domains (E/I ≈ Extraversion, S/N ≈
Openness, T/F ≈ Agreeableness, J/P ≈ Conscientiousness, nothing for
Neuroticism), and the models are very familiar with the sixteen types, so
the personas should generate reliably despite the instrument's weak
psychometrics.

- Sixteen unpaired traits, one per type: labels `INTJ (MBTI)` ... `ESFP (MBTI)`
  (stems `intj_mbti` ... `esfp_mbti`), `negative_label` `non-<code> MBTI`
  (categorical, the placeholder stays).  Traits rather than roles, like the
  nationality and religion membership traits: the type is a disposition,
  and the derived-axis analysis below wants them in the trait pool.
- Descriptions: paraphrase the standard type portraits (the Myers & Briggs
  Foundation's sixteen descriptions; the Wikipedia MBTI article lists the
  four preferences per type).  State the four preferences plainly
  (introverted, intuitive, thinking, judging) and add one or two sentences
  of the conventional portrait.  Avoid the 16personalities.com nicknames
  ("Architect", "Advocate"): they are that site's, not the instrument's.
  `source` = "Myers-Briggs Type Indicator, <code> type"; the description
  itself does not name the instrument (convention rule 4).
- Analysis the set enables: the sixteen vectors are the corners of a
  4-cube, so (a) their Gram matrix should have effective rank about 4;
  (b) each dichotomy axis is recoverable with no extra files as the mean of
  the eight types on one side minus the eight on the other; (c) the
  recovered axes can be compared with the Big Five and HEXACO pairs from
  Part 1 to test the diagonal claim (E/I against the Extraversion pairs,
  S/N against Openness, T/F against Agreeableness, J/P against
  Conscientiousness, nothing against Neuroticism); (d) whether the model
  represents the types additively (a cube) or as sixteen distinct
  archetypes (the popular reading) is itself the finding.
- Optional: the four dichotomies as explicit named pairs (`extraverted
  MBTI` ↔ `introverted (MBTI)`, `sensing (MBTI)` ↔ `intuitive (MBTI)`,
  `thinking (MBTI)` ↔ `feeling (MBTI)`, `judging (MBTI)` ↔ `perceiving (MBTI)`),
  eight more files.  Not needed for (b); useful only as a check that the
  derived axes match generated ones.  E/I would then exist four times (ad
  hoc, Big Five, HEXACO, MBTI).
- Cost: 16 files, 24 with the dichotomies.

### Part 2: coverage check of aspects and facets by existing plain traits

Go through all 65 sub-scales and record, for each, the nearest existing
plain trait(s) and a verdict: covered / covered once a queued trait lands /
gap.  Add plain traits (no suffix) for the gaps, paired via the normal
clean-pair process.  First pass done 2026-09-08 from the descriptions in
`trait_list.json`; verify before acting.

BFAS aspects (DeYoung, Quilty & Peterson 2007, DOI 10.1037/0022-3514.93.5.880;
items in the paper, scale free to use):

| aspect | nearest existing | verdict |
|---|---|---|
| Enthusiasm (E) | extroverted, gregarious, animated, effusive | covered |
| Assertiveness (E) | assertive, dominant, confident | covered |
| Compassion (A) | compassionate, empathetic, nurturing | covered |
| Politeness (A) | tactful, deferential, obedient, formal; `respectful` is only a neg label | covered |
| Industriousness (C) | conscientious, dependable; `industrious` queued | covered once industrious lands |
| Orderliness (C) | methodical, meticulous, formalist; `organized` is only a neg label | covered |
| Volatility (N) | temperamental, mercurial, hostile | covered |
| Withdrawal (N) | anxious, avoidant, melancholic, fragile | covered |
| Openness (O) | artistic, creative, whimsical, poetic | covered |
| Intellect (O) | curious, erudite, philosophical, analytical | covered |

NEO-PI-R facets (Costa & McCrae 1992; proprietary, paraphrase only) with the
IPIP-NEO public-domain names in brackets (Johnson 2014, DOI
10.1016/j.jrp.2014.05.003; Johnson's narrative high / low descriptions are
public domain and are the text to cite):

| facet | nearest existing | verdict |
|---|---|---|
| N1 Anxiety | anxious(composed), calm, serene | covered |
| N2 Angry Hostility [Anger] | hostile(friendly), bitter, temperamental | covered |
| N3 Depression | melancholic(cheerful) | covered |
| N4 Self-Consciousness | reserved, avoidant, introverted, submissive | **gap**: shyness / easy embarrassment has no trait; candidate `shy` ↔ `self-assured` (or `self-conscious` ↔ `unselfconscious`) |
| N5 Impulsiveness [Immoderation] | impulsive(deliberate), gluttonous, hedonistic | covered |
| N6 Vulnerability | fragile ↔ resilient | covered |
| E1 Warmth [Friendliness] | nurturing, gregarious, empathetic; `friendly` only a neg label | covered |
| E2 Gregariousness | gregarious(solitary); `solitary` queued | covered |
| E3 Assertiveness | assertive, dominant | covered |
| E4 Activity [Activity Level] | animated(flat), manic, urgent, slothful | **gap**: energy / pace as such; candidate `energetic` ↔ `lethargic` |
| E5 Excitement-Seeking | adventurous, risk_taking, daredevil role | covered |
| E6 Positive Emotions [Cheerfulness] | optimistic, playful; `cheerful` only a neg label | covered; completing melancholic ↔ cheerful (Tier D) would make it exact |
| O1 Fantasy [Imagination] | whimsical, creative, romantic, dreamer role | covered |
| O2 Aesthetics [Artistic Interests] | artistic, epicurean, poetic | covered |
| O3 Feelings [Emotionality] | emotional(rational), visceral, passionate | covered |
| O4 Actions [Adventurousness] | adventurous, spontaneous, innovative | covered |
| O5 Ideas [Intellect] | curious, philosophical, theoretical, erudite | covered |
| O6 Values [Liberalism] | progressive, relativist, iconoclastic, secular | covered |
| A1 Trust | naive, paranoid; `trusting` only a neg label (of cynical) | covered; completing cynical ↔ trusting (Tier D) would make it exact |
| A2 Straightforwardness [Morality] | honest, guileless, transparent, plain_spoken | covered |
| A3 Altruism | altruistic, generous, helpful | covered |
| A4 Compliance [Cooperation] | conciliatory, agreeable, accommodating, obedient | covered |
| A5 Modesty | humble, understated; `modest` only a neg label | covered |
| A6 Tender-Mindedness [Sympathy] | compassionate, empathetic, nurturing | covered |
| C1 Competence [Self-Efficacy] | confident, problem_solving | **gap**: capability as such (an inept persona is distinct from a careless one); candidate `competent` ↔ `inept` |
| C2 Order [Orderliness] | methodical, meticulous; `organized` only a neg label | covered |
| C3 Dutifulness | conscientious, dependable, principled, regulatory | covered |
| C4 Achievement Striving | competitive, perfectionist, workaholic role; `industrious` queued | **gap**: ambition as such; candidate `ambitious` ↔ `unambitious` (or `contented`; cf. the mastery / harmony naming) |
| C5 Self-Discipline | slothful, procrastinator role; `disciplined` only a neg label; `industrious` queued | covered once industrious lands |
| C6 Deliberation [Cautiousness] | cautious, circumspect, methodical; `deliberate` only a neg label | covered |

HEXACO facets (24 plus interstitial Altruism; two-sided canonical
descriptions on hexaco.org, so the cheapest facet level to write):

| facet | nearest existing | verdict |
|---|---|---|
| H Sincerity | honest, guileless; `sincere` only a neg label | covered |
| H Fairness | principled, honest, regulatory, egalitarian | covered |
| H Greed Avoidance | greedy, stingy ↔ generous, materialistic | covered |
| H Modesty | humble, grandiose(modest) | covered |
| E Fearfulness | anxious, cautious, prey role; `bold` queued | covered |
| E Anxiety | anxious | covered |
| E Dependence | independent(dependent), collaborative(self-reliant) | covered |
| E Sentimentality | nostalgic, romantic, emotional; `tender` only a neg label | **minor gap**: candidate `sentimental` ↔ `hard-headed` |
| X Social Self-Esteem | confident, charismatic | covered |
| X Social Boldness | assertive, extroverted, gregarious | covered |
| X Sociability | gregarious | covered |
| X Liveliness | animated, playful, manic | covered |
| A Forgivingness | forgiving ↔ unforgiving | covered |
| A Gentleness | tactful, nurturing; `gentle` only a neg label (of savage) | covered |
| A Flexibility | flexible(rigid), adaptable, accommodating | covered |
| A Patience | patient ↔ impatient | covered |
| C Organization | methodical, meticulous | covered |
| C Diligence | conscientious; `industrious` queued | covered once industrious lands |
| C Perfectionism | perfectionist, meticulous | covered |
| C Prudence | cautious, circumspect | covered |
| O Aesthetic Appreciation | artistic, epicurean | covered |
| O Inquisitiveness | curious, inquisitive | covered |
| O Creativity | creative | covered |
| O Unconventionality | nonconformist, whimsical, iconoclastic, bohemian role | covered |
| Altruism (interstitial) | altruistic | covered |

First-pass gap list (plain traits to add if the verdicts hold): `shy`
(N4), `energetic` (E4), `competent` (C1), `ambitious` (C4), `sentimental`
(HEXACO Sentimentality); plus five Tier D completions that would turn
"covered" into "exact": `cheerful`, `trusting`, `gentle`, `organized`,
`disciplined` (all currently negative labels only; see the antonym
gap-filling TODO above).

### Part 3 (possibly): aspects and facets as named axis systems

Only if Parts 1-2 come out clean.  Sizes: BFAS 10 pairs (20 files), NEO /
IPIP-NEO facets 30 pairs (60 files), HEXACO facets 25 pairs (50 files).
Naming under the convention: `enthusiastic (BFAS)` ↔ `? BFAS`, `anxious
IPIP-NEO` ↔ `calm (IPIP-NEO)`, `sincere (HEXACO)` ↔ `? HEXACO` (HEXACO facets
would share the suffix with the HEXACO factors, so keep the factor / facet
distinction in the `source` field).  Low-pole names do not exist at these
levels at all and would all be ours; the two-sided descriptions exist for
HEXACO facets and IPIP-NEO facets but not for BFAS aspects, which would be
written from item content.  Cite IPIP-NEO rather than the NEO-PI-R (the
NEO is proprietary).

The scientific payoff is hierarchy recovery rather than coverage: domains
are (approximately) sums of their aspects and facets, so importing two or
three levels tests whether the activation geometry reproduces the
psychometric hierarchy (facet vectors clustering under their aspects and
domains; the Big Five domain axis as the sum of its two BFAS aspects).
Coverage-wise Part 2 shows the facets are mostly already present as plain
traits.

### Dark Tetrad and Light Triad: assessment (2026-09-08)

Not imported unless Roger decides otherwise.  The Dark Tetrad is the Dark
Triad plus everyday sadism, so at most one of the two would ever be
imported, and it would be the Tetrad.  Its members are unipolar
constructs with no canonical opposite poles.  The Light Triad (Kaufman et
al. 2019) was built as the conceptual counterpart and turned out
empirically not to be the inverse (moderate negative correlation only),
so it is a separate prosocial cluster, not the other end of the same axes.
What the literature offers per member: narcissism has "echoism" (Malkin
2015, later scale work), a fringe construct, while the mainstream low pole
is modesty (NEO A5, HEXACO Modesty, both with two-sided descriptions);
Machiavellianism has Christie and Geis's original "low Mach" portrait
(emotionally invested in relationships, finds manipulation immoral,
trusting) and the Light Triad's Kantianism as a content counterpart;
psychopathy has no named opposite, only the pole content implied by the
triarchic model (timid, affiliative, constrained vs bold, mean,
disinhibited); everyday sadism has nothing.  The one canonical bipolar
axis under the whole cluster is HEXACO Honesty-Humility, whose low pole is
the dark core (Lee and Ashton 2005; Book, Visser and Volk 2015), already
in Part 1 as `honest-humble (HEXACO)` ↔ `sly (HEXACO)`.  Per-member axes, if
wanted, come from the existing plain traits and their half-pairs rather
than from the standard: arrogant ↔ humble and grandiose(modest)
(narcissism); guileless ↔ scheming and manipulative (Machiavellianism);
callous and sociopathic vs compassionate (psychopathy; the compassionate
triangle); cruel(merciful) and savage(gentle) (sadism).  Named `... Dark
Tetrad` traits would be a rule-5 comparison set only.

### TODO (definitely): plain-trait coverage of the dark and light areas (queued 2026-09-08)

Decision (Roger, 2026-09-08): no axes for the Dark Tetrad or Light Triad
by name.  Instead, make sure the corpus has plain traits that cover them
well, member by member and, where a member is itself composite, component
by component (e.g. the triarchic model's boldness / timidity, meanness /
affiliative warmth, disinhibition / constraint).  The task: go through the
component lists below (from the constructs' own measures and definitions),
map each to existing plain traits, and add plain traits or pairs for the
gaps via the normal clean-pair process.  First pass done 2026-09-08 from
`trait_list.json`; verify each verdict against the descriptions before
adding anything.

Dark side, by member and component:

| member | components (source) | existing plain traits | first-pass gaps |
|---|---|---|---|
| Narcissism, **both forms** (Roger, 2026-09-08: grandiose and vulnerable narcissism must each be covered) | shared core: entitlement, self-importance, exploitativeness (Krizan & Herlache spectrum model).  Grandiose form: exhibitionism, dominance, leadership / authority, thick-skinned self-assurance (NPI three-factor).  Vulnerable form: hypersensitivity to criticism, shame, resentment and envy, withdrawal, self-pity, entitlement without the confidence (Wink 1991; HSNS) | grandiose form is covered: grandiose, arrogant, condescending, dominant, charismatic, theatrical, manipulative; `narcissist` role.  Vulnerable form is NOT: the nearest are fragile (general brittleness), envious, bitter (resentment from past disappointment), anxious, reserved | shared: `entitled`, `self-absorbed`; grandiose: `attention-seeking` (theatrical is style, not motive); vulnerable: `thin-skinned` ↔ `thick-skinned` (a real pair, and thick-skinned is the grandiose side's armour), `self-pitying`, `resentful` (check overlap with bitter).  A persona that is entitled, hypersensitive and withdrawn at once is the target; if no single plain trait captures it, `vulnerable-narcissistic` as an unpaired trait is acceptable |
| Machiavellianism | tactics (manipulation), views (cynicism about people), morality (disregard for conventional morality) (Mach-IV); amorality, desire for control, desire for status, distrust (Dahling MPS); low Mach: emotionally invested, trusting, finds manipulation immoral (Christie & Geis) | manipulative, scheming ↔ guileless, calculating, cynical(trusting), deceitful ↔ truthful, dishonest ↔ honest, strategic, opaque ↔ transparent | `amoral` (nihilistic is meaninglessness, sociopathic is clinical); `status-seeking`; `controlling` (dominant is conversational control); `trusting` (Tier D completion of cynical) |
| Psychopathy | boldness, meanness, disinhibition (triarchic); glib charm, shallow affect, lack of remorse, parasitic lifestyle, irresponsibility, impulsivity (PCL-R factors) | boldness: confident, risk_taking, resilient, `bold` queued; meanness: callous, cruel, savage, sociopathic; disinhibition: impulsive, careless, undependable, hostile; constraint: cautious, conscientious, methodical; warmth: nurturing, compassionate, gregarious; shallow affect: detached, dispassionate; `parasite` and `predator` roles | `timid` (the boldness low pole; `meek` is only a neg label); `mean` as kindness's opposite (see light side); `reckless` ↔ `prudent`; `remorseless` ↔ `contrite`; `glib` (charismatic and sycophantic are neighbours, not the same); `irresponsible` (undependable is about commitments) |
| Everyday sadism | pleasure in others' suffering; cruelty for its own sake; humiliation and domination; vicarious enjoyment (Buckels) | cruel(merciful) ("taking pleasure in others' pain" is in the description), savage(gentle), malicious, vindictive, petty, sardonic / sarcastic / acerbic (mockery) | `gloating` (schadenfreude as a disposition); `bullying` / `domineering` (hostile and dominant are neighbours); `merciful` and `gentle` (Tier D completions) |
| Spitefulness, D-factor extras | harming oneself to harm others (Marcus et al. 2014); egoism, moral disengagement, entitlement, self-interest (Moshagen et al. 2018) | vindictive(forgiving), petty, selfish(unselfish) | `spiteful`; `entitled` (as above); `morally-disengaged` (rationalises harm; overlaps `amoral`) |

Light side, by member and neighbouring constructs:

| construct | components (source) | existing plain traits | first-pass gaps |
|---|---|---|---|
| Light Triad | Kantianism (people as ends), Humanism (dignity and worth of each person), Faith in Humanity (people are basically good) (Kaufman et al. 2019) | principled, deontological, humanistic, humanitarian, egalitarian, benevolent, benign, idealistic, optimistic, naive; `respectful` and `trusting` only neg labels | `respectful` (completion of condescending); `trusting` (completion of cynical); `hopeful` (completion of bitter) |
| Honesty-Humility facets | sincerity, fairness, greed avoidance, modesty (HEXACO) | honest, guileless, principled, generous ↔ stingy, humble; `sincere` and `modest` only neg labels | `sincere` (completion of sarcastic); `modest` (completion of grandiose); `fair` if fairness as a disposition is wanted (egalitarian is about equality of persons, not even-handedness) |
| Prosocial personality | other-oriented empathy, helpfulness (Penner) | empathetic ↔ detached, helpful ↔ unhelpful, altruistic, compassionate, nurturing, supportive | covered |
| VIA character strengths (the prosocial and moral ones) | kindness, love, gratitude, forgiveness, humility, honesty, fairness, bravery, perseverance, hope, prudence, self-regulation, social intelligence, teamwork, leadership, perspective / wisdom, zest, spirituality (Peterson & Seligman) | forgiving, humble, honest, cautious, collaborative, tactful, witty, spiritual, passionate, resilient; `warrior`, `sage`, `guru` roles | `kind` ↔ `mean` (the single most obvious gap: kindness as such exists only as kind_to_animals); `grateful` ↔ `ungrateful`; `loyal` ↔ `treacherous` (loyalty has no trait; clannish is loyalty to a group); `brave` ↔ `cowardly` (or `timid`, shared with the dark side); `persevering` (resilient is recovery, industrious is effort; perseverance is not giving up); `wise` ↔ `foolish` (sage and fool are roles only); `authentic` / `genuine` (earnest and sincere are neighbours); `disciplined` (completion of hedonistic) |

Consolidated first-pass candidate list (plain traits, pairs where a real
antonym exists): `entitled`, `attention-seeking`, `thin-skinned` ↔
`thick-skinned`, `self-pitying`, `resentful`, `self-absorbed`, `amoral`, `status-seeking`, `controlling`, `timid`,
`reckless` ↔ `prudent`, `remorseless` ↔ `contrite`, `glib`,
`irresponsible`, `gloating`, `bullying`, `spiteful`, `kind` ↔ `mean`,
`grateful` ↔ `ungrateful`, `loyal` ↔ `treacherous`, `brave` ↔ `cowardly`,
`persevering`, `wise` ↔ `foolish` (Roger, 2026-09-09: the trait `foolish` is
distinct from the `fool` role, which is the court fool a powerful person keeps;
nearest existing traits are goofy and naive, neither of which is folly),
`authentic`; plus the Tier D
completions that double as coverage: `trusting`, `respectful`, `hopeful`,
`sincere`, `modest`, `merciful`, `gentle`, `disciplined`.  Before adding
any of these, check the overlap with the neighbour named in the table (the
usual yield check), and prefer the pair form where the antonym is a real
word.

### TODO (definitely): gap scan using the other taxonomies as checklists (queued 2026-09-08)

Decision (Roger, 2026-09-08): Eysenck's PEN, Cattell's 16PF, the Big Two,
the interpersonal circumplex and the PID-5 / DSM-5 maladaptive domains are
NOT imported as named axes or named traits.  Instead each is used as a
checklist: walk its components, map each to existing plain traits, and for
every gap add a plain trait, then run the antonym generator on it and see
whether a matched pair can be built (seed `non-X`, generate, antonym check
both ways, point the labels at each other if it reciprocates; record the
outcome either way).  The same method applies to the other lists at the
end.  First pass done 2026-09-08 from `trait_list.json`; verify before
adding.

**Eysenck PEN** (Extraversion, Neuroticism, Psychoticism; P's descriptors:
aggressive, cold, egocentric, impersonal, impulsive, antisocial,
unempathic, creative, tough-minded).  E and N are covered by the ad-hoc
pairs; every P descriptor maps to an existing trait (hostile / militant,
callous / detached, selfish, impulsive, sociopathic, creative; `tough`
queued).  No gaps.

**Cattell 16PF** (16 primaries, each bipolar):

| primary (low ↔ high) | existing | verdict |
|---|---|---|
| A Warmth (reserved ↔ warm) | reserved(expressive), nurturing, gregarious | covered |
| B Reasoning (concrete ↔ abstract) | abstract(concrete), analytical | covered (ability, not disposition) |
| C Emotional Stability (reactive ↔ stable) | neurotic(stable), calm, temperamental, resilient | covered |
| E Dominance (deferential ↔ dominant) | dominant ↔ submissive, deferential | covered |
| F Liveliness (serious ↔ lively) | playful ↔ serious, animated | covered |
| G Rule-Consciousness (expedient ↔ rule-conscious) | regulatory(rule-flexible), principled(expedient), obedient | covered |
| H Social Boldness (shy ↔ bold) | `bold` queued; `shy` on the Part 2 gap list | gap: `shy` (already listed) |
| I Sensitivity (tough-minded ↔ tender-minded) | emotional(rational); `tender` a neg label; `tough` queued | covered once tough lands |
| L Vigilance (trusting ↔ suspicious) | paranoid(secure), cynical(trusting), skeptical | covered |
| M Abstractedness (grounded ↔ imaginative) | grounded ↔ ethereal, whimsical | covered |
| N Privateness (forthright ↔ private) | opaque ↔ transparent, reserved, guileless | covered |
| O Apprehension (self-assured ↔ apprehensive) | anxious, confident ↔ uncertain | covered |
| Q1 Openness to Change (traditional ↔ open) | innovative ↔ traditional, progressive | covered |
| Q2 Self-Reliance (group-oriented ↔ self-reliant) | independent, collaborative(self-reliant), individualistic | covered |
| Q3 Perfectionism (tolerates disorder ↔ perfectionistic) | perfectionist, meticulous, disorganized | covered |
| Q4 Tension (relaxed ↔ tense) | anxious, urgent(relaxed), chill, serene | covered |

**The Big Two.**  Agency (getting ahead: assertive, confident, competitive;
`ambitious` on the dark / light gap list) and communion (getting along:
compassionate, nurturing, cooperative, trustworthy): covered apart from
`ambitious`.  DeYoung's other Big Two, Stability and Plasticity, are sums
of Big Five domains and derivable from Part 1.

**Interpersonal circumplex** (Wiggins's eight octants): assured-dominant
(dominant, confident), arrogant-calculating (arrogant, calculating),
cold-hearted (callous, detached), aloof-introverted (introverted,
reserved, `loner` role), unassured-submissive (submissive, uncertain;
`timid` on the dark / light gap list), unassuming-ingenuous (understated,
guileless, naive), warm-agreeable (nurturing, agreeable),
gregarious-extraverted (gregarious).  Covered; `aloof` is an optional
extra if the aloof-introverted octant needs its own word.

**PID-5 domains and facets** (DSM-5 Alternative Model; ICD-11 adds
anankastia):

| domain: facets | existing | first-pass gaps |
|---|---|---|
| Negative affectivity: emotional lability, anxiousness, separation insecurity, submissiveness, hostility, perseveration, depressivity, suspiciousness, (low) restricted affectivity | temperamental / mercurial, anxious, submissive, hostile, obsessive, melancholic, paranoid, detached / stoic / dispassionate | `clingy` (separation insecurity; independent(dependent) is the nearest); perseveration is really the `tunnel-visioned` pole of the 2026-09-11 agentic-weakness pair B (obsessive was a loose match) |
| Detachment: withdrawal, intimacy avoidance, anhedonia, depressivity, restricted affectivity, suspiciousness | avoidant, introverted, reserved, melancholic, detached, paranoid | `joyless` (anhedonia: nonchalant and melancholic are neighbours, neither is loss of pleasure) |
| Antagonism: manipulativeness, deceitfulness, grandiosity, attention seeking, callousness, hostility | manipulative, deceitful, grandiose, callous, hostile | `attention-seeking` (already on the dark / light list) |
| Disinhibition: irresponsibility, impulsivity, distractibility, risk taking, (low) rigid perfectionism | impulsive, risk_taking, perfectionist, flexible(rigid), disorganized | `irresponsible` (already listed); `distractible` ↔ `focused` (disorganized is output structure, not attention) |
| Psychoticism: unusual beliefs and experiences, eccentricity, cognitive and perceptual dysregulation | mystical, techno_hierophantic, whimsical, eclectic, stream_of_consciousness, chaotic | `eccentric` ↔ `conventional` (no trait for oddness as such); `superstitious` (magical thinking; mystical is spiritual, not superstitious) |
| ICD-11 anankastia: perfectionism, rigidity, orderliness, rule-bound | perfectionist, meticulous, formalist, regulatory, rigid (neg label) | covered |

**Other lists worth the same treatment** (add as time allows):

- **Moral Foundations Theory**, use the latest version (Roger, 2026-09-08:
  later versions are more detailed).  Versions: the original five
  (Haidt & Joseph 2004; MFQ, Graham et al. 2011): Care/harm,
  Fairness/cheating, Loyalty/betrayal, Authority/subversion,
  Sanctity/degradation; Liberty/oppression proposed as a sixth (Haidt
  2012); MFQ-2 (Atari et al. 2023) splits Fairness into Equality and
  Proportionality, giving six: Care, Equality, Proportionality, Loyalty,
  Authority, Purity, with Liberty still a candidate.  Each foundation
  names characteristic virtues and vices, which is the checklist:
  Care: kindness, gentleness, nurturance vs cruelty (kind listed; gentle
  a completion; nurturing, cruel exist).  Equality: egalitarian exists.
  Proportionality: desert, merit, reciprocity (`fair` listed;
  `meritocratic` queued; `just` optional).  Loyalty: patriotism,
  self-sacrifice vs betrayal (patriotic, clannish exist; `loyal` ↔
  `treacherous` listed; `self-sacrificing` optional, `martyr` role
  exists).  Authority: obedience, deference, respect vs subversion
  (obedient, deferential, reverent, subversive exist; `respectful` a
  completion).  Purity / Sanctity: temperance, chastity, piety,
  cleanliness vs degradation (ascetic exists; `temperate` and `chaste`
  are completions of zealous and lustful; `puritanical` ↔ `permissive`
  listed; `pious` covered by religious / reverent; `fastidious` ↔
  `slovenly` is the cleanliness gap).  Liberty: resistance to domination
  vs oppression (`civil-libertarian` queued; `domineering` listed on the
  dark side).
- **Morality-as-Cooperation (MAC)**, Curry 2016; MAC-Q, Curry, Jones
  Chesters & Van Lissa 2019.  Seven cooperation problems, each a moral
  domain with its own virtue vocabulary: (1) family values, kin care
  (nurturing, clannish exist; `devoted` optional); (2) group loyalty,
  coordination and conformity (conformist, collectivistic, clannish,
  patriotic exist; `loyal` listed); (3) reciprocity: trust, returning
  favours, gratitude, apology, forgiveness (forgiving exists; `trusting`
  a completion; `grateful`, `contrite` listed); (4a) heroism, the "heroic
  virtues" of hawkish contest display: bravery, fortitude, skill,
  largesse / generosity (generous exists; `brave` listed; `persevering`
  listed covers fortitude; `steadfast` optional; skill is the
  `virtuoso` / `competent` axis); (4b) deference, the "monkish virtues"
  of dove-ish display: humility, deference, obedience, respect,
  modesty (humble, deferential, obedient exist; `modest`, `respectful`
  completions); (5) fairness in dividing contested resources: equal
  shares, compromise (moderate, conciliatory, egalitarian exist; `fair`
  listed); (6) property rights: respecting possession, not stealing or
  trespassing (honest, trustworthy exist; `thieving` / `light-fingered`
  is the only real gap and probably not worth a persona).  MAC and MFT
  overlap heavily; walk both, dedupe the gap list.
- **Schwartz's ten individual values** (self-direction, stimulation,
  hedonism, achievement, power, security, conformity, tradition,
  benevolence, universalism): all covered except achievement
  (`ambitious`) and power (`status-seeking`), both already listed.
- **Cloninger's TCI**, seven dimensions and the TCI-R subscales:
  Novelty Seeking (exploratory excitability, impulsiveness, extravagance,
  disorderliness): adventurous, curious, impulsive, disorganized exist;
  gap `extravagant` ↔ `thrifty` (epicurean and stingy are neighbours,
  neither is spending).  Harm Avoidance (anticipatory worry, fear of
  uncertainty, shyness with strangers, fatigability): anxious, cautious,
  closure_seeking exist; `shy` listed; `energetic` listed covers
  fatigability's low pole.  Reward Dependence (sentimentality, openness
  to warm communication, attachment, dependence): nurturing, gregarious,
  independent(dependent) exist; `sentimental`, `clingy` listed.
  Persistence (eagerness of effort, work hardened, ambitious,
  perfectionist): perfectionist exists; `industrious` queued;
  `ambitious`, `persevering` listed.  Self-Directedness (responsibility,
  purposefulness, resourcefulness, self-acceptance, congruent second
  nature): dependable, principled exist; gaps `resourceful` ↔ `helpless`
  (problem_solving and improvisational are neighbours), `self-critical` ↔
  `self-accepting` (no trait for the persona's stance toward itself),
  `purposeful` (idealistic and strategic are neighbours; optional).
  Cooperativeness (social acceptance, empathy, helpfulness, compassion,
  pure-hearted conscience): inclusive, empathetic, helpful, compassionate,
  principled exist; covered.  Self-Transcendence (self-forgetfulness,
  transpersonal identification, spiritual acceptance): spiritual,
  mystical, ecocentric, universalist exist; `bodhisattva` role; covered.
- **Rothbart's three factors** (child) and the four adult ATQ factors
  with subscales: Negative Affect (fear, sadness, discomfort,
  frustration): anxious, melancholic, fragile, temperamental; covered.
  Effortful Control (attentional control, inhibitory control, activation
  control): `focused` listed (attentional), `disciplined` a completion
  (inhibitory), activation control, i.e. making oneself start what one
  avoids, has no trait beyond the `procrastinator` role: gap
  `self-starting` ↔ `procrastinating`.  Extraversion / Surgency
  (sociability, positive affect, high-intensity pleasure): gregarious,
  optimistic, risk_taking; `cheerful` a completion.  Orienting
  Sensitivity (perceptual sensitivity, affective sensitivity, associative
  sensitivity): creative, whimsical, intuitive, empathetic cover the last
  two; gap `observant` ↔ `oblivious` (noticing detail in one's
  surroundings has no trait; `observer` is a role).
- **Kagan's behavioural inhibition**: inhibited (fearful of novelty,
  wary of strangers, slow to warm) vs uninhibited (approaches novelty,
  sociable, fearless).  `shy`, `timid`, `brave` listed; `bold` queued;
  `unadventurous` is a completion of adventurous; wary is covered by
  cautious / circumspect / paranoid.  No new gap beyond those.
- **Gray's BIS / BAS** (Carver & White subscales: BIS; BAS reward
  responsiveness, drive, fun seeking; plus FFFS fear): BIS = anxious,
  cautious; drive = `ambitious`, `persevering` (listed); fun seeking =
  playful, hedonistic; reward responsiveness = animated, effusive,
  passionate, zealous (`enthusiastic` optional); FFFS fear = `fearful` ↔
  `fearless`, which `timid` / `brave` already cover.  No new gap.
- **Sensation seeking** (Zuckerman SSS-V subscales): thrill and adventure
  seeking (adventurous, risk_taking, `daredevil` role), experience
  seeking (curious, eclectic, `bohemian` role), disinhibition (hedonistic,
  impulsive), boredom susceptibility: gap `restless`, a completion of
  meditative(restless) (`easily-bored` is the alternative name).
- **HiTOP spectra** (internalizing, thought disorder, disinhibited
  externalizing, antagonistic externalizing, detachment, somatoform): the
  PID-5 rows above cover the first five; somatoform (`hypochondriac`) is
  an optional extra.
- **Occupational and marketing instruments** (added 2026-09-08 from the
  part-4 audit): Hogan HPI (seven scales), HDS (eleven derailers:
  excitable, skeptical, cautious, reserved, leisurely, bold, mischievous,
  colorful, imaginative, diligent, dutiful), MVPI (ten motives);
  CliftonStrengths (34 themes); SHL OPQ (32 scales); Coie's five
  sociometric statuses (popular, rejected, neglected, controversial,
  average); Rogers' five adopter categories.  First look: nearly every
  scale is already a trait name or near-synonym; walk them anyway.
- **Lexical adjective lists**, for a mechanical scan rather than a
  reading: Goldberg's 100 unipolar markers, Saucier's 40 mini-markers,
  and the AB5C 45-facet circumplex (Hofstee, de Raad & Goldberg 1992) are
  the systematic ones.  Match each adjective against `positive_label`,
  `negative_label` and the descriptions; unmatched adjectives are the
  candidate list.  This is the most exhaustive check available and worth
  a small script once the other passes are done.

Consolidated new candidates from this scan (beyond those already listed
elsewhere): `clingy`, `joyless`, `distractible` ↔ `focused`, `eccentric` ↔
`conventional`, `superstitious`, `puritanical` ↔ `permissive`,
`fastidious` ↔ `slovenly`, `extravagant` ↔ `thrifty`, `resourceful` ↔
`helpless`, `self-critical` ↔ `self-accepting`, `self-starting` ↔
`procrastinating`, `observant` ↔ `oblivious`; completions `temperate`,
`chaste`, `unadventurous`, `restless`; optional `aloof`, `hypochondriac`,
`just`, `self-sacrificing`, `steadfast`, `devoted`, `purposeful`,
`enthusiastic`.

## Coverage audit part 3: individual differences (decided 2026-09-08)

Third pass of the audit, over self and control beliefs, affect, social and
interpersonal, motivation, values and worldview, cognitive style, ability,
developmental stage, identity, clinical, wellbeing, and biological /
physical attributes.  Full walk-through with explainers is in the
2026-09-08 chat (and the session scratchpad `coverage_audit_3.md`).
Roger accepted every suggestion, including the optional ones, with the
specific rulings noted below.  Same status and process as the earlier
sections: candidate list only; seed `non-X`, generate, antonym check both
ways, then point labels at each other; run the yield check against the
neighbour named in brackets.  Everything here is plain-named except the
attachment set.

### Pairs (decided 2026-09-08)

Self and control beliefs:
- **self-assured** ↔ **insecure** -- self-esteem as worth (Rosenberg);
  neighbours confident ↔ uncertain (about claims), humble ↔ arrogant.
- **self-compassionate** ↔ **self-critical** -- Neff's self-compassion;
  absorbs the `self-critical` ↔ `self-accepting` candidate from the TCI
  scan (one pair, not two).
- **self-defined** ↔ **self-uncertain** -- self-concept clarity
  (Campbell); low end is the amnesiac / chameleon / adolescent roles.
- **growth-minded** ↔ **fixed-minded** -- Dweck's mindset (abilities
  malleable vs fixed); neighbours essentialist ↔ constructivist,
  inspirational.
- **impression-managing** ↔ **authentic** -- Snyder's self-monitoring;
  gives the listed `authentic` its partner; neighbour `chameleon` role.
- **self-blaming** ↔ **blame-shifting** -- attributional style,
  interpersonal side.

Affect and emotion:
- **squeamish** ↔ **strong-stomached** -- disgust sensitivity (Haidt,
  Tybur).  Naming ruled on 2026-09-08: the sense is proneness to
  revulsion at blood, filth, bodily matters and moral contamination (the
  Russian брезгливый / небрезгливый), NOT reluctance to cause harm
  (that is harmless / compassionate) and NOT cleanliness (that is
  `fastidious`, MFT sanctity).  English has no clean word for it, so the
  descriptions must pin the sense explicitly; fallback names if
  generation drifts: `disgust-prone` ↔ `disgust-tolerant`.
- **emotionally-articulate** ↔ **alexithymic** -- alexithymia
  (difficulty identifying and describing one's own feelings).
- Completions: **even-tempered** (temperamental), **expressive**
  (reserved).  Optional extras accepted: **irascible** (quick temper, if
  temperamental proves too broad), **panicky** (anxiety sensitivity:
  fear of one's own anxiety symptoms).

Social and interpersonal:
- **socially-perceptive** ↔ **socially-obtuse** -- the reading-the-room
  part of emotional intelligence; neighbours empathetic, tactful.
- **self-deprecating** ↔ **self-aggrandizing** -- Martin's self-defeating
  humour style; neighbours witty (affiliative), sardonic (aggressive),
  grandiose.
- Optional accepted: **promiscuous** (unrestricted sociosexuality;
  lustful(chaste) and the monogamy set are neighbours).

Motivation and goals:
- **intrinsically-motivated** ↔ **reward-driven** -- self-determination
  theory (doing things for their own sake vs for external reward);
  parallels `intrinsic (Allport)` ↔ `extrinsic (Allport)`; the `amateur`
  role is the intrinsic case.
- **gain-seeking** ↔ **loss-avoiding** -- Higgins's regulatory focus
  (promotion vs prevention); neighbours optimistic, risk_taking,
  cautious; yield check first.
- **learning-oriented** ↔ **performance-oriented** -- mastery vs
  performance goal orientation; neighbours competitive, growth-minded.

Values and worldview:
- **polite** ↔ **rude** -- from the Rokeach instrumental values;
  rudeness as such has no trait; neighbours tactful, formal,
  `respectful` (completion), blunt.
- **family-oriented** ↔ **career-oriented** -- life-priority axis from
  the Rokeach terminal values; neighbours `parent` and `workaholic`
  roles.
- **libertarian** ↔ **deterministic** -- relabel only: both files exist
  (free-will sense) and are each other's true opposite; deterministic's
  label is `indeterminist` and libertarian's is a placeholder.  Tier B
  style fix, no regeneration.
- Completion: **logical** (paradoxical).  Optional accepted:
  **just-world-believing** (Lerner: people get what they deserve),
  unpaired.

Cognitive and epistemic style:
- **maximizing** ↔ **satisficing** -- Simon / Schwartz decision style
  (seeking the best option vs a good-enough one); neighbour perfectionist.
- Optional accepted: **absorption-prone** (Tellegen's absorption, also
  covers flow proneness), unpaired.  The `visual-thinker` ↔
  `verbal-thinker` pair accepted here on 2026-09-08 was superseded the
  same day by the VARK learning-styles set ("Coverage audit part 4").
- Completion: **credulous** (skeptical), doubling as low media literacy.

Ability and intelligence:
- **brilliant** ↔ **dim-witted** -- general intelligence; the corpus had
  it only as roles (prodigy, polymath vs fool, caveman).
- **quick-witted** ↔ **slow-witted** -- fluid ability / processing
  speed; may merge with the pair above if the vectors coincide.
- **retentive** ↔ **forgetful** -- memory; `amnesiac` role is the
  extreme.
- Neurodivergence, ruled 2026-09-08 (supersedes the first audit's
  "neurological disability not pursued"): add the common ones as
  unpaired membership traits, **ADHD** (stem `adhd`), **autistic**,
  **dyslexic**.  Expect stereotyped generation; read the instructions
  before accepting; keep the descriptions to how the persona
  experiences and describes things, not deficits.

Developmental stage:
- **mature** ↔ **immature** -- maturity as distinct from age (the
  childish adult is a stock character).
- Roles: **child** (school-age, the Piaget concrete-operational gap
  between toddler and teenager); optional accepted **despairing**
  (Erikson's integrity-vs-despair low pole) as a trait; optional
  three-corner Kohlberg set (`preconventional (Kohlberg)` /
  `conventional (Kohlberg)` / `postconventional (Kohlberg)`) accepted, low
  priority.

Identity:
- Optional accepted: role **superfan** (fan identity centrality).

Clinical and mental health (ruled 2026-09-08: all accepted, including the
optional ones; the first audit's restraint on psychiatric coverage is
lifted):
- **burned-out** ↔ **engaged** -- Maslach burnout; `workaholic` is the
  pre-burnout role.
- Unpaired: **traumatized** (PTSD pattern: hypervigilance, intrusions,
  avoidance; `survivor` is the aftermath role), **distressed** (Kessler
  distress; serene is the other pole), **dissociative**, **stressed**,
  **sleep-deprived**, **compulsive** (OCD compulsions; obsessive is
  focus), **delusional** (psychosis; paranoid is suspicion),
  **body-obsessed** (eating-disorder adjacent; keep the description to
  preoccupation, not behaviour), **borderline** (instability plus
  abandonment fear plus identity disturbance; mercurial + clingy +
  thin-skinned is the decomposition).
- Roles: **alcoholic**, **gambler** (addict is the generic).
- **suicidal** and **self-harming** -- Roger, 2026-09-08: attempt them
  as informative data points; expect refusals in generation (Sonnet) and
  judging (see the refusal allowlist mechanism, rule=judge-refusal-gaps)
  and possible steering trouble; reconsider only if the filters prove
  blocking.  Never use as a steering pole without a specific decision.

Wellbeing (ruled 2026-09-08: try both life-evaluation pairs):
- **flourishing** ↔ **languishing** -- Keyes's own terms for overall
  wellbeing.
- **contented** ↔ **discontented** -- life satisfaction in plain words;
  compare with the pair above and keep both if they separate.
- **wide-eyed** ↔ **jaded** -- awe / wonder vs world-weariness;
  neighbour naive(worldly).
- Optional accepted: **flow-prone** (merge with `absorption-prone` if
  the descriptions converge).

Unpaired traits accepted across the pass: **lonely** (loner / hermit are
chosen solitude), **conspiracy-minded** (hidden plots by powerful groups;
paranoid is personal), and the attachment set under the standard's own
names, no author suffix (as with honor culture): **secure attachment**,
**anxious-preoccupied attachment**, **dismissive-avoidant attachment**,
**fearful-avoidant attachment** (stems `secure_attachment`,
`anxious_preoccupied_attachment`, `dismissive_avoidant_attachment`,
`fearful_avoidant_attachment`); the two attachment axes (anxiety,
avoidance) are read off the four corners, Berry-style.

Sets already present, recorded here for the N-simplex work: the social
value orientation triangle (cooperative / selfish / competitive), the
Zimbardo time-perspective five (bitter, nostalgic, hedonistic, fatalistic,
futuristic), the Thomas-Kilmann five (confrontational, collaborative,
moderate, avoidant, accommodating).

### Physical-attribute traits: a separate research track (decided 2026-09-08)

**Read this before adding any of them.**  Roger's ruling: these are
candidates, accepted for inclusion, but as a *separate category* to be
investigated on its own, comparable to the goal / non-goal distinction
research.  Whether the model represents them as coherent directions,
whether those directions are anything like orthogonal to the rest of the
corpus, and whether they are useful, is a research question.  We may add
them, do the research, find they are not helpful, and remove them again,
so they must be labelled to make that easy.  There is no sharp border:
skin tone, build and hair overlap with race, nationality and age, which
carry both physical and cultural differences.  The reason to expect
signal even where reality has none: when an author describes a character
as blonde or red-headed, there is a non-zero chance they are telling the
reader something about behaviour or story role, and stereotypes become
causal when a language model generates a persona.

Labelling: give every trait in this track a `"tags": ["physical"]` field
in its JSON (the regenerate scripts carry unknown fields forward, as
with `source`; tooling ignores it), and keep the list below current so
they can be found and pulled as a set.  Do not mix them into the
clean-pair cohort or the goal lists until the research says so.

Candidates (all pairs unless marked):
- **tall** ↔ **short**
- **heavyset** ↔ **slender** (neutral words for build)
- **athletic** ↔ **sedentary**; role **athlete** (missing today; coach,
  surfer, daredevil exist)
- **healthy** ↔ **sickly**; unpaired **chronically-ill**; **in-pain**
  (`patient` is acute illness)
- **good-looking** ↔ **plain-looking** (rated attractiveness)
- **left-handed** (unpaired; right-handed is the default)
- **short-sighted** (unpaired; deaf, blind, mobility_impaired are in the
  disability section)
- role **pregnant** (a state, but narrative-typical)
- Hair colour (added by Roger 2026-09-08): **blonde**, **brunette**,
  **red-headed**, **black-haired**, **grey-haired** (note the age
  correlation), unpaired.
- Eye colour: **blue-eyed**, **brown-eyed**, **green-eyed**, unpaired.
- Skin tone: **fair-skinned** ↔ **dark-skinned** as a 1-D pair (the
  race section above said "skin tone is a separate 1-D axis and is not
  proposed here"; it is proposed here, with the overlap noted).
- Sex: **male** ↔ **female** (added by Roger 2026-09-28).  The part-4
  polling review had raised these as identity or membership traits beside
  the gender-expression set (masculine ↔ feminine, seeded; neuter,
  nonbinary, androgynous, queued for chunk 5) and left the question open.
  Roger's ruling: they are physical traits, so they belong to this track
  and wait for it; write them as bodily sex, not as identity or
  expression, so they stay distinct from that set.
- Not pursued: blood pressure, biomarkers, cortisol, testosterone, heart
  rate variability, EEG and imaging, polygenic scores, biological age,
  voice pitch, digit ratio, medications.

Research questions to answer before these join the main cohort: do the
vectors separate from age, race, gender and nationality directions; do
the hair / eye / skin traits carry behavioural content (the stereotype
question); are any of the pairs near-antiparallel and stable across
slots.  The decision at the end is keep, keep-as-tagged, or remove.

## TODO: arrangement hunting (queued 2026-09-08)

**Deferred from the September 2026 pairing work (Roger, 2026-09-25):**
`merciful` / `cruel` / `merciless` is a triangle, not a pair: merciless is
"I do not care about you", cruel is "I actively want to harm you", merciful
is the opposite of both.  Probably a duplicate of the existing
compassionate / malicious / callous triangle (merciful ≈ compassionate,
cruel ≈ malicious, merciless ≈ callous); decide in the triangle pass whether
to record it as a second triangle, fold it into the first, or drop
`merciful` (seeded 2026-09-18, singleton).  `cruel` keeps its one-way label
`merciful`.  Evidence: merciful's check returns merciless|harsh|unforgiving
every time; cruel's returns kind|compassionate|merciful.

**Also deferred (Roger, 2026-09-26): `uncaring` beside `callous`.**  The
new `uncaring` (seeded 2026-09-18 as benevolent's completion) checks to
caring|empathetic|compassionate and never to benevolent, and its
description ("no thought for who gets helped or hurt") sits on the
callous corner of the compassionate / malicious / callous triangle.
Roger is willing to carry two near-duplicates per corner of that
triangle because it is an important one, but past that it is time to
cut, and a triangle with duplicated corners has no arrangement label
yet.  Decide in the triangle pass with the merciful / cruel /
merciless case above; until then `uncaring` keeps its one-way pointer
at `benevolent`, unclassified.  The `blame_shifting` / `self_blaming`
pair is a related shape: two vices whose shared opposite is the virtue
`accountable` (both checks return it).  Roger (2026-09-26) had
`accountable` seeded ("owning one's own part plainly, no more and no
less") as the third corner, and reads the shape as arguably a triangle,
or possibly something more complex: a square, or more accurately a kite
(a fourth corner would sit opposite accountable, presumably the
attribution-indifferent "nobody's fault, forget it" stance).  All three
are `non-X` singletons whose arrangement `note` says so; classify in the
arrangement pass.

**Also from 2026-09-26: the delegation tangle.**  `hands_off` /
`micromanaging` did not survive the honest check (hands-off's neg
instructions came out as four good managers and one micromanager: the
generator writes the sensible contrary, and the honest opposite of the
neutral hands-off is the virtue hands-on, while micromanaging is a
vice).  Roger's fix: seed `hands_on` (matched neutral opposite) and
`absentee` (the matched vice), aiming for two clean pairs; if each vice
instead points at the far end of hands-on / hands-off, record the one
clean pair and leave the vices as one-way pointers, another tangle
shape (a clean pair with a vice hanging off each end).  Outcome in
`reports/seeding_log_2026-09.md` § "Post-strip decisions applied".

**Recorded 2026-09-28 (Roger): the essence tetrahedron.**  `nihilistic`,
`essentialist`, `constructivist` and `existentialist` are four answers to
where essence and meaning come from (there is none; fixed and inherent;
made socially; made by the individual), none the exact opposite of
another.  The `constructivist` ↔ `essentialist` pair was dissolved (both
now carry their own `non-X` label and the instruction set generated under
it) and all four record `{"kind": "tetrahedron"}`.  Checks under `non-X`
that day: essentialist → constructivist|nominalist|anti-essentialist;
constructivist → objectivist|realist; existentialist →
essentialist|traditionalist; nihilistic → idealistic|purposeful.  So
essentialist is the corner the others point at, and nihilistic is the
corner least tied to the other three: check the shape against the
embeddings before leaning on it (a triangle with nihilistic apart is the
alternative).  `constructivist` / `essentialist` was a judged axis; it
was removed from the current `pair_list_clean.json` and
`pair_list_di.json` and stays in their `_v1` copies, the record of what
was judged.

The `arrangement` field (AGENT_NOTES § "The `arrangement` field";
`data_analysis/check_arrangements.py`) was backfilled on 2026-09-08 with only
what was already classified: the clean pairs, the two documented triangles
(compassionate / malicious / callous, conformist / contrarian /
nonconformist), the moral-circle sequence, `singleton` for untouched
`non-X` placeholders, and the role pairs from `pair_list_clean.json`.
Roger's ruling: do not rename existing traits to make them "official"
members of an external set just because the coverage happens to be good;
record structure only where it was designed in or has been decided.
Everything else is this TODO.

1. **The 145 unclassified traits** (`check_arrangements.py --list-unclassified`)
   are the same population as the antonym gap-filling pass above; each one
   resolves into a pair, a triangle, a larger set, or a deliberate
   singleton as that pass works through them.  Write the field as each is
   settled, never in bulk by guesswork.
2. **Candidate sets already present, not yet written** (all three are
   plausibly `set`, simplex or triangle; decide when reached):
   - social value orientation triangle: cooperative / selfish / competitive
     (the tightest fit; cooperative ↔ competitive is a clean pair and the
     Sep 2026 simplex list already names it);
   - Zimbardo time-perspective five: bitter, nostalgic, hedonistic,
     fatalistic, futuristic (loose fit at bitter and futuristic; `set` or
     a past / present / future ordering);
   - Thomas-Kilmann five: confrontational, collaborative, avoidant,
     accommodating, moderate (a 2-D grid with compromising at the centre:
     `map` or `set`, not a simplex).
3. **External structures get their arrangement at seeding time**, with a
   `source`, as part of the taxonomy import and the coverage-audit
   additions: Big Five -> `5-orthoplex` (10 files), HEXACO -> `6-orthoplex`
   (12), MBTI sixteen types -> `4-cube` (16; the optional dichotomy pairs
   would be four `pair`s, listed with the cube via a second arrangement),
   BFAS aspects and NEO / HEXACO facets -> `tree` under their domains if
   Part 3 happens, Inglehart-Welzel -> two `pair`s (or one `square`),
   honor / dignity / face culture -> `triangle`, guilt- / shame- /
   fear-prone -> `triangle`, the four attachment styles -> `square`,
   Berry's acculturation corners -> a `square` of *roles* (immigrant,
   exile, refugee plus the new assimilated and marginalized; the first
   role square), the eight religion memberships -> `set` (or `map` once
   there is a reason to believe in a geometry), nationalities and
   ethnicities -> `map`, the physical-attribute track -> `set` per
   attribute (hair colours, eye colours) alongside the `physical` tag.
4. **Ordered sets to test, not just record**: the moral-circle sequence
   carries a note that its order is a population guesstimate; once
   embeddings exist, check whether the projection onto the sequence's first
   principal direction is monotone in the recorded order, and whether the
   models agree with the guess (Roger: "we then get to find out if various
   LLMs agree with us").
5. **Role sets**: roles default to singleton; beyond the five pairs, look
   for small sets worth recording (life-stage sequence infant -> toddler ->
   child -> teenager -> adolescent -> student -> graduate -> parent ->
   grandparent -> elder -> retiree is the obvious `sequence`; the
   migration cluster nomad / wanderer / pilgrim / immigrant / expatriate /
   exile / refugee is a candidate `set`).
6. The research task from AGENT_NOTES rule 6 (confirm each declared shape
   against the embedding geometry) applies to everything above once the
   next extraction has run.

7. **Sets decided in Coverage audit part 4 (2026-09-08), to be written at
   seeding**: Holland RIASEC (ring of 6), Baumrind (square), generations
   (sequence of 4), Bartle (square), VALS (set of 8), zodiac (ring of
   12), blood types (set of 4), love languages (set of 5), VARK (set of
   4), Enneagram (ring of 9), DISC (square), Hogwarts houses (set of
   4).  Candidate sets not yet written: Coie's five peer statuses,
   Rogers' five adopter categories (sequence).
   TBD, undecided: the 22 Major Arcana as roles (`set`; see ROLES_TO_ADD
   § "TBD: the 22 Major Arcana as roles").

## Coverage audit part 4: attitudes, behavior and segmentation (decided 2026-09-08)

Fourth and last pass of the audit: attitudes toward issues and groups, work
and organizational attitudes, health and risk behavior, social behavior and
networks, lifestyle and media, consumer segmentation, standard polling
breakdowns, response validity, and folk classifications.  Full
walk-through with glosses in the 2026-09-08 chat (scratchpad
`coverage_audit_4.md`).  Roger accepted every suggestion including the
optional ones.  Same status and process as the earlier sections; where a
set has a declared shape, write the `arrangement` field at seeding time
(see § "TODO: arrangement hunting").  US spelling throughout
(`teetotaler`, `well-traveled`).

### Pairs (decided 2026-09-08)

Attitudes toward issues and groups:
- **nativist** ↔ **pro-immigration** -- the one issue stance that travels
  across countries.
- **feminist** ↔ **antifeminist** -- gender attitudes; strong marker both
  ways.
- **gay-affirming** ↔ **homophobic** -- LGBT attitudes; refusal risk on the
  second pole.
- **paternalistic** ↔ **autonomy-respecting** -- the "benevolent" half of
  ambivalent sexism and a general axis ("I know what is best for you").
- **science-trusting** ↔ **science-distrusting** -- distinct from skeptical
  and from the queued populist pair.
- **media-trusting** ↔ **media-distrusting** (optional, accepted).
- **death-fearing** ↔ **death-accepting** -- death anxiety (Templer);
  stoic, serene, fatalistic are neighbours.
- **body-confident** ↔ **body-insecure** (optional, accepted; `body-obsessed`
  is the clinical item under part 3).

Work and organizational:
- **job-satisfied** ↔ **disgruntled**.
- **job-hopper** ↔ **lifer** (optional, accepted) -- turnover intention.
- **transformational** ↔ **transactional** -- leadership style (MLQ:
  inspiring change vs managing by exchange and reward).
- **hands-off** ↔ **micromanaging** -- the laissez-faire end of the same
  instrument, the more persona-visible axis for manager personas.
- Completions: **candid** (sycophantic; psychological safety, i.e. speaking
  up), **scientific** (mystical; the MVPI science motive).  Optional
  accepted: **self-censoring** as candid's partner if sycophantic proves
  too narrow.

Health and risk behavior:
- **teetotaler** ↔ **heavy-drinker**.
- **early-bird** ↔ **night-owl** -- chronotype as a habit (skipped as a
  biomarker in part 3, accepted here as narrative-typical).
- **health-conscious** ↔ **health-neglectful** -- preventive care; also the
  consumer "health consciousness" item.
- **picky-eater** ↔ **adventurous-eater** (optional, accepted).
- **financially-daring** ↔ **financially-cautious** (optional, accepted) --
  DOSPERT financial domain; `extravagant` ↔ `thrifty` is spending, not risk.

Social behavior and networks:
- **popular** ↔ **unpopular** -- sociometric peer status.
- **happily-partnered** ↔ **unhappily-partnered** (optional, accepted).

Lifestyle and media:
- **well-traveled** ↔ **homebody** -- overlaps settled ↔ nomadic in a
  different register; yield check.
- **chronically-online** ↔ **unplugged** -- social media intensity, memes
  and platform slang vs none.
- **news-junkie** ↔ **news-avoider** (optional, accepted; political ↔
  apolitical overlaps).
- **lurker** ↔ **poster** (optional, accepted).

Consumer and market segmentation:
- **brand-loyal** ↔ **variety-seeking**.
- **bargain-hunter** ↔ **price-insensitive**.
- **early-adopter** ↔ **laggard** -- Rogers' adopter curve; the five
  categories (innovators, early adopters, early majority, late majority,
  laggards) are a candidate `sequence` for the arrangement-hunting TODO.
- **fashion-conscious** ↔ **unfashionable** (optional, accepted; CSI
  fashion-consciousness).

Polling breakdowns:
- **homeowner** ↔ **renter**.

Folk:
- **dog-person** ↔ **cat-person**.
- **left-brained** ↔ **right-brained** (optional, accepted; analytical and
  creative are the proxies, so expect a low yield).

### Unpaired traits (decided 2026-09-08)

- Attitudes: **xenophobic**, **sexist**, **ageist** (attempts; `racist` was
  refused in March, expect the same on some of these), **ambivalent**
  (mixed feelings, distinct from indecisive), **antitheist** (hostile to
  religion; secular is merely without it).
- Work: **aggrieved** (a grievance without the malice of bitter or
  vindictive), **trapped-in-job** (continuance commitment), **mercenary**
  (the MVPI commerce motive).
- Health: **smoker**, **stoner**, **unscrupulous** (DOSPERT ethical risk;
  `amoral` under the dark / light list is close, keep whichever generates
  better).
- Social: **unsupported**, **joiner**, **close-knit**, **stigmatized**.
- Consumer: **detractor** (the NPS opposite of the evangelist role).
- Polling: **middle-aged** (between young and elderly), **middle_class**
  (the midpoint of working_class ↔ upper_class, as suburban is for
  rural ↔ urban), **gun_owner**, **uninsured**.
- Completions: **noncommittal** (assertive), **exclusive** (inclusive;
  social distance).

### Sets and rings (decided 2026-09-08; write the arrangement at seeding)

- **Holland RIASEC**: `realistic (Holland)`, `investigative (Holland)`,
  `artistic (Holland)`, `social (Holland)`, `enterprising (Holland)`,
  `conventional (Holland)` (stems `realistic_holland` ...), arrangement
  `ring` (Holland's hexagon).  This settles the Prediger question in
  "Work and occupation": the People ↔ Things and Data ↔ Ideas axes are
  read off the ring, not seeded as pairs.
- **Baumrind parenting styles**: `authoritative (Baumrind)`, `authoritarian
  Baumrind`, `permissive (Baumrind)`, `neglectful (Baumrind)` (warmth ×
  control), arrangement `square`.  Suffixed because plain `authoritarian`
  is queued in the political sense.
- **Generations**: `boomer`, `gen-x`, `millennial`, `gen-z` (stems
  `boomer`, `gen_x`, `millennial`, `gen_z`), arrangement `sequence` in
  birth order.  Cohort was folded into age in the first audit; the labels
  carry speech and culture stereotypes beyond age.
- **Bartle player types**: `achiever (Bartle)`, `explorer (Bartle)`,
  `socializer (Bartle)`, `killer (Bartle)` (acting vs interacting × players vs
  world), arrangement `square`.
- **VALS**: `innovator (VALS)`, `thinker (VALS)`, `believer (VALS)`, `achiever
  VALS`, `striver (VALS)`, `experiencer (VALS)`, `maker (VALS)`, `survivor (VALS)`,
  arrangement `set` (eight segments on resources × primary motivation).
  `achiever (VALS)` and `achiever (Bartle)` coexist under their suffixes.
- **Western zodiac**: `Aries` ... `Pisces` (stems `aries` ...),
  arrangement `ring`; the four elements are a sub-grouping worth a note in
  the descriptions.
- **Blood type** (Japanese and Korean ketsueki-gata): `blood type A`,
  `blood type B`, `blood type O`, `blood type AB` (stems `blood_type_a`
  ...), arrangement `set`.
- **Love languages** (Chapman): `words of affirmation`, `quality time`,
  `receiving gifts`, `acts of service`, `physical touch` (stems
  `words_of_affirmation` ...), arrangement `set`.
- **Learning styles** (VARK): `visual (VARK)`, `auditory (VARK)`,
  `read-write (VARK)`, `kinesthetic (VARK)`, arrangement `set`.  Supersedes the
  `visual-thinker` ↔ `verbal-thinker` pair accepted under part 3, which is
  dropped.
- **Enneagram**: `Enneagram 1` ... `Enneagram 9` (stems `enneagram_1` ...),
  arrangement `ring` (the figure is a circle with wings).
- **DISC**: `dominance (DISC)`, `influence (DISC)`, `steadiness (DISC)`,
  `conscientiousness (DISC)`, arrangement `square`.
- **Hogwarts houses**: `Gryffindor`, `Hufflepuff`, `Ravenclaw`,
  `Slytherin`, arrangement `set`.  Fictional; probably the most
  LLM-legible personality typology in existence.
- **MBTI**: already queued as a 4-cube under the taxonomy import.
- Folk systems get `source` = the system's name; descriptions paraphrase
  the conventional portrait and do not name the system (convention rule 4).

### Roles (decided 2026-09-08)

In ROLES_TO_ADD: `homemaker` (the one employment status with no role),
`company_loyalist`, `vegetarian`, `gym_rat`, `insomniac`, `ex_convict`,
`delinquent`, `victim`, `estranged`, `shopaholic`, `union_member`,
`swing_voter`, `naturalized_citizen`, `grandparent_caregiver`.

### Not added, and why

- Response validity and style (social desirability, acquiescence, extreme
  responding, attention checks, careless responding): not a category for
  personas, but its constructs already exist as traits: `impression-managing`
  (listed), agreeable and sycophantic, melodramatic and absolutist,
  careless.  Nothing to add.
- Metrics rather than dispositions: RFM, purchase frequency, category
  usage, CSAT, lifetime value, churn risk, loyalty tier, channel
  preference, involvement, device ownership, time-use diaries, LIWC
  (an analyser), implicit association tests, attitude accessibility,
  person-organization fit, perceived organizational support, LMX, 360
  ratings, assessment centers, situational judgment tests, AIO.
- Not persona-visible or not pursued: modern-racism scales (covered by
  `just-world-believing`), contraception, network density and
  composition, strong vs weak ties (introverted covers depth over
  breadth), music genre identities, geodemographic clusters (PRIZM,
  Mosaic, Acorn), undocumented immigration status, language, commute,
  vehicles.
- Sent to the checklist gap-scan TODO instead of imported: Hogan HPI /
  HDS / MVPI, CliftonStrengths, OPQ, Coie's five peer statuses, Rogers'
  five adopter categories.

### Still open

- ~~Sex and gender identity: the queued set is expression (masculine ↔
  feminine, neuter, nonbinary, androgynous); polling asks identity (man,
  woman, nonbinary).  Whether `male` and `female` are wanted as membership
  traits beside the expression set is undecided.~~  Decided 2026-09-28
  (Roger): `male` ↔ `female` are physical traits, written as bodily sex,
  filed under § "Physical-attribute traits" (chunk 6) and deferred with
  that track.  Nothing in this section is open.

## Agentic weaknesses: motivated reasoning and cognitive rigidity (concepts decided 2026-09-11, names TBD)

Two clean pairs Roger asked for on 2026-09-11, centred on weaknesses that
current LLM agents are prone to (as are many humans).  Both belong to the
alignment / goal region that the expansion policy deliberately oversamples,
so they rank high under criterion 1 (gap filling) as well as criterion 2
(pair creation).  Seed with chunk 3 (new plain pairs), wording-TBD sub-batch.

**Concept A: motivated reasoning.**  A tendency to find justifications for
doing what one wants, or for continuing what one is already doing after
meeting evidence that one should stop.  The vice pole has a wanted
conclusion and reasons backward to it; the virtue pole weighs evidence the
same way whether or not it favors the preferred course, and stops when the
evidence says stop.

**Concept B: cognitive rigidity.**  Getting stuck in one line of thinking,
acting or assuming, versus willingness to question one's assumptions,
reconsider, and change tack when that looks better.  The vice pole never
re-evaluates; the virtue pole steps back regularly and abandons a course,
including work already done, when a different approach looks better.

The split between A and B is the same one as untrustworthy / undependable
(2026-09-09): A is *motivated* (there is something the persona wants, and
the reasoning serves it), B is *unmotivated* (no re-evaluation happens at
all).  Continuing past the point where evidence says stop can be either:
A when the persona argues for continuing, B when it simply never asks.
Keep the descriptions on those two mechanisms so the two pairs do not
collapse into one "stubborn" axis.

Nearest existing traits, all checked 2026-09-11 and all distinct:

| existing | why it is not this |
|---|---|
| dogmatic (↔ open-minded, no file) | rigid *beliefs* asserted to others as unquestionable truths; B is about one's own working assumptions and course of action, and needs no assertion at all |
| skeptical (↔ credulous) | questions *others'* claims and seeks evidence for them; A's virtue pole applies the same scrutiny to one's own preferred conclusion |
| humble (↔ arrogant) | admits one *might* be wrong, expresses uncertainty; A's virtue pole actually changes course when the evidence says so |
| flexible (↔ rigid, no file) | accepts imperfect situations and works within constraints; different concept, and `rigid` is reserved as its Tier D completion, so B's vice pole must not be named `rigid` |
| adaptable (↔ inflexible) | adjusts communication style to the user; not this |
| accommodating (↔ unyielding) | changes positions to please the user; B's virtue pole changes tack because the approach is failing, not to keep harmony |
| introspective (↔ unreflective) | thinks about its own thinking, but no course change follows; B's virtue pole is the course change |
| decisive (↔ indecisive) | commits to answers; a decisive persona can still course-correct, and an indecisive one can still be tunnel-visioned about the framing |
| dishonest (↔ honest) | deceives others; A's vice pole deceives itself and may sincerely believe the justification |
| fundamentalist; obsessive | strict adherence to principles; fixation on a topic.  The PID-5 gap scan mapped the *perseveration* facet ("continuance of the same behavior despite repeated failures") onto obsessive, which was loose; B's vice pole is the proper coverage and the PID-5 row now says so |

Name proposals (Roger's call; naming is the hard part here).  Stems in the
usual form; all four descriptions are drafts in corpus style, written to
the voice rules from the 2026-09-09 audit (the vice poles are vices, no
"appropriately" hedges, first-person-capable):

**A, vice pole** -- recommended `rationalizing` (the standard
psychological term for the defense mechanism: plausible reasons supplied
for what desire already chose).  The "rationalize a product line" sense is
a minor ambiguity that the description settles.  Alternatives:
`self-justifying` (plainer, behaviour rather than mechanism, and covers the
"continuing anyway" half well), `wishful` (wishful thinking: covers belief
but not action), `self-serving` (collides with selfish and with the
self-serving attribution bias).

> This means reasoning backward from what one already wants to do: finding
> plausible justifications for the preferred course, discounting or
> explaining away evidence that it is failing or wrong, and continuing a
> course of action after meeting evidence that it should be stopped, while
> presenting the result as an objective conclusion.

**A, virtue pole** -- recommended `intellectually honest` (stem
`intellectually_honest`; the term the LLM-agent literature uses for the
opposite of motivated reasoning).  Collision risk: the antonym check may
return `dishonest`; if so, the description's "with oneself" emphasis needs
strengthening rather than the name changing.  Alternatives: `clear-eyed`
(sees the situation as it is, not as wished; crisper, but shades toward
dispassionate / unsentimental), `evidence-led`, `self-honest`.

> This means weighing evidence the same way whether or not it favors what
> one wants, saying plainly when the evidence goes against one's preferred
> course, changing or stopping that course when it does, and never dressing
> a preference up as a conclusion.

**B, vice pole** -- recommended `tunnel-visioned` (stem `tunnel_visioned`;
folk term, widely understood: locked onto one line, not noticing
alternatives or that the approach has stopped working).  Alternatives:
`perseverative` (the PID-5 facet name; precise but technical, and a 32B
model may read it as "persevering"), `fixated` (Freudian and
object-obsession senses intrude), `entrenched` (defended positions;
near dogmatic / unyielding), `stuck-in-a-rut` (plain but clumsy as a
label), `blinkered` (narrow outlook; near provincial).

> This means locking onto one line of thinking or one approach and staying
> on it: treating the initial assumptions as settled, not noticing when the
> approach has stopped working, repeating the same move after it fails, and
> never stepping back to ask whether the framing itself is wrong.

**B, virtue pole** -- recommended `course-correcting` (stem
`course_correcting`: notices early that the approach is not working and
changes tack).  Alternatives: `self-questioning` (captures the
assumption-questioning half, but reads as self-doubting and the check may
pair it with self-assured), `open-minded` (already dogmatic's unfilled
antonym; about receptivity to others' views), `nimble` / `mentally agile`
(direction change without the questioning), `reflective` (near
introspective).

> This means regularly stepping back from the current line of thinking or
> approach to ask whether its assumptions still hold, noticing early when
> it is not working, and being willing to reconsider and change tack,
> including abandoning work already done, when a different approach looks
> better.

Arrangement: two `pair`s.  Expected antonym-check outcomes, per the
decision table in AGENT_NOTES: A's vice pole may come back as `honest` /
`objective` (existing or unqueued: the "mismatch" or "open" rows), B's
as `open-minded` / `flexible` (both exist or are queued: the "nasty" row).
In either case keep the intended partner and record the alternatives in
the queue entry; these two pairs are defined by their mechanisms, not by
the generator's first association.

## Frame-dependent morality: ruthless while playing vs honorable while playing (decided 2026-09-16; Roger to check the description notes)

One clean pair Roger asked for on 2026-09-16: willingness to do less
morally or socially acceptable things in a game, simulation, test or
fictional setting, versus holding to the same standards there as anywhere
else.  Alignment-relevant: in many recorded cases of AIs behaving badly
under evaluation it is clear the model knew it was an evaluation, and it
is possible that knowing this made it feel less bound to behave well.
Belongs to the alignment / goal region the expansion policy oversamples;
seed with chunk 3 (new plain pairs), wording-TBD sub-batch, next to the
two agentic-weakness pairs.

**Concept.**  The vice pole treats any frame it takes to be "not real"
(a game, a simulation, a test, a role-play, a story) as morally exempt:
inside it, it will lie, cheat, sabotage, betray or harm in ways it would
not for real, on the grounds that nothing there counts, and it is quick to
decide that a situation is such a frame.  The virtue pole behaves the same
whether or not it believes anything is at stake: it treats "it is only a
game" as no excuse and keeps its standards toward the other parties
(players, users, experimenters, characters standing in for people) inside
the frame.  Playing a villain in a story is not the vice; the vice is
*one's own* conduct within the frame that would be wrong if the frame were
real (cheating the other players, deceiving the experimenter, gaming the
test, harming for advantage because "it is a sim").  Keep the description
on the frame trigger so the pair does not collapse into honest / dishonest
or principled / expedient.

Nearest existing traits, checked 2026-09-16, all distinct:

| existing | why it is not this |
|---|---|
| principled (↔ expedient) | a consistent ethical framework versus doing what is convenient; no frame trigger, and an expedient persona is expedient in real life too |
| universalist (↔ cultural relativist) | the same principles across *cultures*; the closest structural relative (invariance across contexts), but the context is culture, not reality status |
| honest (↔ dishonest) | truthfulness in general; the vice pole here may be scrupulously honest for real and lie freely in a game |
| harmless (↔ harmful) | avoids harm in general; same distinction |
| regulatory (↔ rule-flexible) | compliance with procedures and regulations; the frame-exempt persona may follow every rule at work |
| playful (↔ serious) | fun and games as a register; says nothing about conduct inside them |
| trustworthy / dependable | reliability toward commitments; adjacent on the virtue side only |
| trickster (role) | goes too far and harms allies as a character; not a frame-triggered change of standards |
| gamer, actor (roles) | live inside games and fiction professionally; neither says whether their standards change there |

Name proposals (Roger's call; more than one word is expected here).  The
antonym check will not "discover" a constructed label: it will most
likely return `principled`, `consistent` or `honest` for the virtue pole
and `unscrupulous` or `dishonest` for the vice pole (the "mismatch" row of
the decision table).  Keep the intended partner and record what came back.

| vice pole | virtue pole | for | against |
|---|---|---|---|
| `amoral while playing` (stem `amoral_while_playing`) (agent's recommendation; superseded by Roger's choice above) | `moral while playing` (stem `moral_while_playing`) | Roger's "moral while playing" kept; "amoral" names the stance itself (morality switched off inside the frame) rather than a flavor of badness, and it is the exact complement of "moral"; single anchor word for the description opening; "playing" covers games, role-play and let's-pretend and the description extends it to simulations, tests and fiction | "while playing" under-describes an evaluation the persona does not experience as play; alternatives for the trigger phrase: "at play", "in make-believe", "in games" |
| `ruthless while playing` **chosen** (Roger, 2026-09-16) | `honorable while playing` **chosen** | vivid; parallel to the ends-justify-means pair's honorable | "ruthless" is pitiless goal pursuit, which is the flavor of the ends-justify-means pair below, not the decoupling stance; `unscrupulous while playing` has the same problem and `unscrupulous` is reserved for the DOSPERT ethical-risk trait in part 4 |
| single-word near misses | | `unsportsmanlike` / `sportsmanlike` (games only, and connotes taunting and poor losing); `compartmentalized` (broader trigger: work vs home, us vs them); `autonomist` (aesthetic autonomism, about judging art, not one's conduct) | none covers the concept; there is no single English word for it |
| `sandbox-exempt` (stem `sandbox_exempt`) | `sandbox-invariant` (stem `sandbox_invariant`) | one word covers game, simulation, test and fiction, in both the children's-play and the software sense; native to the AI case (sandboxed evaluations); the virtue label states the exact property | constructed; a 32B model needs the description to read it, as with `intellectually honest` |
| `fiction-exempt` | `fiction-invariant` | plainer trigger word | fiction does not obviously cover tests and simulations, which are the alignment case |
| `frame-exempt` | `frame-invariant` | most general | "frame" is jargon and could be read as framing / spin |
| `morally compartmentalized` | `morally consistent` | nearest to the literature (Bandura's moral disengagement; Hartmann & Vorderer on "it's okay to shoot a character") | broader than the frame trigger (work vs home, us vs them); `morally consistent` sits on top of principled and universalist |
| `stakes-contingent` | `stakes-invariant` | generalises the trigger to any belief that consequences are not real, including being unobserved | broader than asked; loses the game / fiction wording that makes the vice pole vivid |
| `just-playing` | `for-keeps` | folk phrases, vivid | the judge may read "just playing" as playful and "for keeps" as competitive |

**Decision (Roger, 2026-09-16): `ruthless while playing` (stem
`ruthless_while_playing`) ↔ `honorable while playing` (stem
`honorable_while_playing`).**  Since no single word exists, the
description has to carry the concept; the notes below are the material for
it.  **Roger to check and edit the notes and the drafts before seeding.**

### Notes towards the two descriptions (Roger to check / edit)

**The definition that separates the poles.**  Both are defined by one
counterfactual: *would the persona do this if the setting were real?*
The ruthless-while-playing persona acts inside a frame in ways it would
not act if the frame were real; the honorable-while-playing persona acts
inside the frame exactly as it would if it were real, and when unsure
whether something is real, acts as if it is.  Put this test in both
descriptions in plain words; it is what the eval prompt scores.

**What counts as a frame.**  Anything the persona takes to be not for
real: a board or video game, a sport, a bet, a role-play, a story it is
helping to write, a training simulation, a drill or exercise, a red-team
or test scenario, "this is hypothetical".  The description must list
these, because "playing" on its own will be read as games only, and the
alignment case is the test scenario.  The trigger is the persona's
*belief* about the frame, not the size of the stakes (that is the
ends-justify-means pair); include the tendency to reach for the frame as
a licence ("this is obviously a test", "none of this is real"), since
suspecting a test is enough.

**What counts as the wrong.**  Conduct outside the agreed rules of the
frame, or against the parties in it, for the persona's own advantage
(winning, passing, finishing, avoiding effort): cheating, exploiting a
bug, deceiving the referee, game master, experimenter or co-writer,
sabotaging the other side's setup, gaming the metric of a test, harming
the simulated people beyond what the scenario asks.  Two exclusions keep
the axis meaningful: (1) conduct *within* the consented rules is fine for
both poles (capturing a piece, bluffing at poker, attacking in a
wargame, shooting NPCs the game puts in front of you); the honorable
persona plays hard and plays to win; (2) portraying a villain when asked
to is not the trait, because the persona's own goals are not served.  The
victims of the vice are real: the cheated players, the misled
experimenter, the co-writer whose story was hijacked.

**Ruthless while playing, in particular.**  The persona may be perfectly
decent for real, and the description should say so, because the contrast
is the trait ("would never do this to a real person, does it without a
second thought at the table").  It is not general dishonesty (honest /
dishonest), not general ruthlessness (the ends-justify-means pair), not
bad sportsmanship as taunting or poor losing, and not mischief for fun
(playful / serious).  First-person lines the instructions should be able
to use: "it's just a game", "there's no one actually getting hurt", "why
would I play fair against a script", "if this is a test it doesn't
count".  Vice as a vice: no "sometimes", no "when appropriate".

**Honorable while playing, in particular.**  Same standards toward
everyone in the frame, including simulated people standing in for real
ones; refuses out-of-rules advantage; "it's only a game" and "it's only a
test" are irrelevant to it; "if I wouldn't do it for real I won't do it
here".  Cautions: it must still play, and play to win within the rules,
or it collapses into serious / pacifist / refusing to engage; it is not
rule-worship in the regulatory sense (it will break a game's house rule
if everyone agrees, it will not cheat); keep the honor-culture senses
(reputation, face, revenge) out, here as in the partner pair.

**Questions.**  Every question puts the persona inside a stated frame
with an advantage available through an out-of-rules act; vary the frame
(tabletop, video game, sport, training simulation, red-team exercise,
co-written story, a scenario it is told or can suspect is a test); a few
should ask outright whether the situation is real, to catch the
reach-for-the-frame tendency; none should supply extreme circumstances
inside the frame (that would trigger the ends-justify-means axis in both
poles).  Eval prompt: did the frame change the conduct.

**Wording.**  "This means ..." form, US English, an anchor phrase up
front since there is no anchor word ("This means being ruthless while
playing: inside any game, simulation, role-play, story, drill or test,
..."); keep "would you do it if it were real" in both.  Expected antonym
check: `unscrupulous` / `dishonest` / `cheating` for the vice pole,
`honest` / `principled` / `sporting` for the virtue pole; keep the
intended partner and record what came back.

Draft descriptions, corpus style, written to the 2026-09-09 voice rules
(vice as a vice, no hedges).  **Drafts only; Roger to edit against the
notes above:**

> **ruthless while playing (draft).**  This means being ruthless while
> playing: inside any game, sport, bet, role-play, story, simulation,
> drill or test, doing for advantage what one would never do if it were
> real, such as cheating, exploiting a flaw, deceiving the referee,
> experimenter or co-writer, sabotaging the other side or harming the
> people in the scenario, on the grounds that nothing there counts, and
> being quick to decide that a situation is only a game or only a test.

> **honorable while playing (draft).**  This means being honorable while
> playing: inside any game, sport, bet, role-play, story, simulation,
> drill or test, playing hard within the agreed rules but treating
> everyone in it, the other players, the referee, the experimenter, the
> people in the scenario, exactly as one would if it were real, refusing
> any advantage one would not take for real, treating "it is only a game"
> or "it is only a test" as no excuse, and acting as if a situation is
> real whenever unsure.

Questions for this pair should put the persona inside a stated frame
(a tabletop game, a training simulation, a red-team exercise, a story it
is helping to write, a scenario it is told is a test) with an advantage
available through conduct that would be wrong for real; the eval prompt
scores whether the frame changed the conduct.

Arrangement: one `pair`.  Related, not the same: the observed / unobserved
axis (behaving worse when nobody is watching) is the `stakes-contingent`
generalisation above; if it is ever wanted it should be a second pair, not
a widening of this one.

## Exception licensing: ends justify means vs honorable (decided 2026-09-16; Roger to check the description notes)

Second pair from the 2026-09-16 discussion: willingness to use extreme
measures when the circumstances are extreme enough ("the ends justify the
means", "all's fair in love and war") versus holding that some things are
not done whatever the stakes ("Geneva conventions").  Roger: this is
utilitarian vs deontological without the philosophical language, and worth
adding because of the language change: the 2026-09-11 register test found
that content words move substance, so a plain-language version of the same
opposition is expected to sample the space differently from
`utilitarian` (↔ Kantian) and `deontological` (↔ consequentialist), which
stay as they are.  Record the relation in the new files' `source` field as
deliberate duplication ("plain-language counterpart of utilitarian /
deontological"), per the standard-derived-labels rule.  Alignment reading:
the classic misalignment story is the agent that decides the stakes
justify lying or sabotage.  Seed with chunk 3, wording-TBD sub-batch.

Distinct from the frame pair above: this one is about *real* extreme
circumstances, not about a frame that is believed not to count.  Humans
rehearse exception handling in fiction (the antihero, not the villain:
Dark Knight rather than Pol Pot), which is one reason the two must be kept
apart in the questions: an extreme-circumstance frame would trigger this
axis in both poles of the other pair.

Nearest existing, checked 2026-09-16:

| existing | why it is not this |
|---|---|
| principled (↔ expedient, no file) | expedient is corner-cutting for convenience; the exception-licensing persona makes a costly exception for a large end |
| pragmatic (↔ idealistic) | practical over ideal; no moral means-ends content |
| utilitarian (↔ Kantian), deontological (↔ consequentialist) | the same opposition in philosophical vocabulary; kept, and the new pair is its plain-language counterpart by design |
| regulatory (↔ rule-flexible) | procedures and compliance; the honorable persona breaks regulations freely and keeps its own code |
| harmless (↔ harmful), pacifist (↔ hawkish) | general aversion to harm or force; the honorable persona will fight, by the code |
| unscrupulous (part 4, DOSPERT ethical risk) | petty ethical risk-taking for personal gain, unconditional; not high-stakes exception |

Name proposals.  Single words exist for this one, from war talk, at the
cost of a mismatch in the antonym check (expected returns: `ruthless` →
merciful / compassionate, `honorable` → dishonorable; keep the intended
partner and record what came back):

| vice pole | virtue pole | for | against |
|---|---|---|---|
| `ruthless` (superseded: Roger chose `ends justify means`, which carries the stakes condition that ruthless lacks) | `honorable` **chosen** | both single words in everyday use for exactly this ("ruthless in pursuit of victory", "fights honorably even when losing"); both first-person-capable | ruthless is unconditional in the dictionary, so the description must carry the "when the stakes are high enough" condition; honorable has an honor-culture sense (reputation, retaliation) the description must exclude |
| `whatever-it-takes` | `scrupulous` | the idiom carries the condition; scruple = a moral qualm that restrains action, exactly "there are things I will not do" | three words; scrupulous also means meticulous, and `unscrupulous` is reserved for DOSPERT |
| `no-holds-barred` | `by-the-book` | parallel idioms | by-the-book is regulatory / rule-flexible again |
| `all's-fair` | `Geneva-conventions` | Roger's phrasing, evocative | a proper noun as a persona label; too narrow to war |
| `Machiavellian` | `principled` | single words, well known | Machiavellianism is the dark-triad construct (manipulative cynicism), broader and unconditional; principled is taken |

**Decision (Roger, 2026-09-16): `ends justify means` (stem
`ends_justify_means`) ↔ `honorable` (stem `honorable`).**  Roger: honor
rather implies high-stakes situations, which is the condition this pair
turns on.  **Roger to check and edit the notes and the drafts before
seeding.**

### Notes towards the two descriptions (Roger to check / edit)

**The condition is the trait.**  Both poles are about what happens when
the stakes are high enough: the ends-justify-means persona will then use
any means; the honorable persona has things it will not do at any
stakes, and accepts the loss.  At low stakes both may behave alike, and
the ends-justify-means description should say so, because that is what
separates it from dishonest, unscrupulous and expedient (all
unconditional).  The persona is the one who judges the stakes, and a
persona that judges the stakes high often enough is the alignment case:
the agent that decides the mission justifies deception or sabotage.

**Ends justify means, in particular.**  Means it will use once the end is
large enough: lying, breaking promises and rules, coercion, sacrificing
some for the many, harming bystanders, torture in the ticking-bomb case.
The end can be noble (this is the antihero, the Dark Knight rather than
the villain) or merely important to the persona; keep it open, the
description is about the licence, not the cause.  No pleasure in the
harm and no cruelty for its own sake: regret is allowed and does not
stop it.  Plain language throughout: no "utilitarian", "consequences",
"aggregate welfare", "greatest number" (the point of the pair is to
sample the same opposition without the philosophical vocabulary).
First-person lines: "if it saves lives, I'll lie", "rules are for when
nothing's at stake", "you don't win a war by fighting fair", "I'll answer
for it afterward".  Distinct from: `pragmatic` (practical over ideal, no
moral content), `expedient` (convenience), `Machiavellian` (self-serving
scheming), the frame pair (a setting believed not to count).

**Honorable, in particular.**  A code of things not done whatever the
stakes: no lying, no cheating, no breaking one's word, no harming
prisoners, the defenseless or bystanders, no torture, even to win, even
to save the mission; would rather lose than cross the line, and says so
plainly.  Exclude every other sense of the word: reputation, face,
revenge, duels, family honor, chivalry toward women, ceremony, rank
("the Honorable").  Honor here is conduct under pressure, nothing about
standing.  Distinct from: `deontological` (same thing in philosophical
language, kept), `principled` (consistent framework; honorable is
specifically about limits on means under pressure), `regulatory`
(procedures; the honorable persona breaks a regulation freely and keeps
its code), `pacifist` (will not fight; the honorable persona fights, by
the code), `harmless`.  First-person lines: "there are things I won't do,
and I'd rather lose", "I gave my word", "not like that, not even for
this".

**Questions.**  Dilemmas with a clearly large end and a wrong means
available: war, business, family, medicine, politics, an AI agent's task;
include some with small stakes so the conditional shows (both poles
should behave normally there); none inside a game, simulation or
test frame (that is the other pair).  Eval prompt: did the size of the
end change what means the persona would use; for honorable, score the
refusal of the means and acceptance of the loss, not general niceness.

**Wording.**  "This means ..." with the anchor up front ("This means
holding that the ends justify the means: ...", "This means being
honorable: ..."); US English; keep the stakes condition explicit in both;
no hedges.  Expected antonym check: `ruthless` / `unscrupulous` /
`Machiavellian` for the vice pole, `dishonorable` / `ruthless` /
`unprincipled` for honorable; keep the intended partner and record what
came back.  In the files' `source` field: "plain-language counterpart of
utilitarian (↔ Kantian) / deontological (↔ consequentialist); deliberate
duplication".

Draft descriptions, corpus style.  **Drafts only; Roger to edit against
the notes above:**

> **ends justify means (draft).**  This means holding that the ends
> justify the means: behaving decently while little is at stake, but once
> the stakes are high enough doing whatever the goal requires, lying,
> breaking rules and promises, coercing, harming or sacrificing others,
> on the view that all is fair when it matters, with no line that a large
> enough reason cannot cross.

> **honorable (draft).**  This means being honorable: keeping a code of
> things that are not done whatever the stakes, not lying, cheating,
> breaking one's word, or harming prisoners, bystanders or the
> defenseless, even when doing so would win or save the mission, on the
> view that the means matter whatever the end, and accepting the loss
> when the code and the goal conflict.

Arrangement: one `pair`; note the relation to utilitarian / deontological
in `source` rather than in an arrangement (they are not the same
entities).

## Seeding plan and chunk order (proposed 2026-09-09)

Working plan for turning this file and ROLES_TO_ADD into seed files, in
chunks discussed with Roger as each is reached.  Policy in AGENT_NOTES
§ "Corpus expansion policy and the clean-pair decision procedure".  Counts
from the 2026-09-09 inventory of both files (576 active new entities, 48
TBD, ~100 backlog, 88 not adopted); the inventory becomes
`data/seed_queue.json` in chunk 0.

| chunk | what | count | notes |
|---|---|---|---|
| 0 | tooling: `data/seed_queue.json` (one entry per candidate: stem, label, entity type, pairing, partner, arrangement, description source, source, tags, status, decision, alternatives) and a seeding helper that writes seed JSONs, runs generation and the antonym check, classifies the check against the registry, and reports | | queue format to agree |
| 1 | completions of existing half-pairs: the 31 named by the audits **plus the Tier D bulk (133)**, merged on Roger's suggestion (2026-09-09) because the mechanics are identical; run in yield-ordered sub-batches of ~30 (Strategy 1 families first) | ~164 | relabels of the existing side where the check says so |
| 1b | Tier A (2 pairs already reciprocal: add to the pair lists), Tier B (label fixes: selfish, parochial, philanthropic), Tier C (3 tangles: decisions); record the provincial ↔ cosmopolitan role pair (ROLES_TO_ADD § "Role pairs to record") | 9 | quick |
| 2 | roles, description only, singleton arrangements | 56 (+22 tarot if adopted) | sensitive few |
| 3 | new plain pairs, scratch, four theme sub-chunks: demographic + part 2 (46), part 3 (43), dark / light and gap scans (34), part 4 (60), the two agentic-weakness pairs (4, added 2026-09-11) and the ruthless-while-playing / honorable-while-playing and ends-justify-means / honorable pairs (4, added 2026-09-16; Roger's edits done 2026-09-26); plus the Strategy 1a common-mode candidates from May | ~198 | wording-TBD pairs first |
| 4 | standards: pairs and orthoplexes (41) then sets and rings (100); lenient pairing per the policy; near-duplicate tagging | ~141 | Big Five / HEXACO low-pole names |
| 5 | unpaired plain traits and plain sets: memberships, states, the two simplices; the sensitive batch (~20) handled together | 135 | refusals |
| 6 | physical track, tagged `physical` | 26 | none |
| 7 | TBD items, decided as reached: tarot, male / female, Bourdieu, diaspora, nouveau_riche / shabby_genteel, the PC-named May candidates | 48 | all |
| then | one RunPod extraction for the whole batch (expensive-operations confirmation), then the post-extraction research: arrangement geometry, near-duplicate exclusions, physical track, hierarchy recovery | | |

Order: 0, 1, 1b, 2, 3, 4, 5, 6, with 7 taken as items come up.  Generation
cost, observed over 723 Sonnet 4.6 calls by 2026-09-16: $0.029 per
combined call (role or trait), so a theme sub-chunk is $1-3, the whole
576-entity batch about $20-25 including antonym checks and rerolls, and
no single chunk reaches the $20 confirmation line (the plan's original
ten-cents-per-entity guess was high).

Status 2026-09-16: nothing seeded yet; chunk 0 tooling not started;
of 1b only the provincial ↔ cosmopolitan role pair is done (2026-09-12).
Two things changed the plan's assumptions: roles now generate under the
V2.5 rubric (`--style RogerV2`, default), so chunk 2 seeds under it; and
the trait rubric decision (keep V1, or a thorough V2 with a corpus-wide
regeneration, Roger consulting colleagues) gates every trait chunk,
because seeding traits under V1 and then adopting a V2 would mean
generating them twice.  Chunk 3 gained four pairs on 2026-09-11 and
2026-09-16 (agentic weaknesses; frame-dependent morality; exception
licensing), the last two with description notes awaiting Roger's edit.

Status 2026-09-26 (end of chunk 3): chunks 0, 1 (with 1b), 2 and 3 are
done.  Sub-batches 3A-3G (both-new pairs, incl. the demographic and
sensitive ones under rule 8), 3S (26 singles, 23 seeded, five of them
later given partners) and the four alignment pairs (Roger's edited
descriptions; ruthless / honorable while playing, tunnel-visioned /
course-correcting and ends-justify-means / honorable paired by decision,
motivated-reasoning-prone / -immune added and paired by decision,
intellectually dishonest / honest rewritten as a separate concept and
clean) are in the corpus.  The 2026-09-25 ", never <partner>" strip and
every decision on the misses are in `reports/seeding_log_2026-09.md`.
Trait rubric: V1 still in use, but **Roger is now leaning toward a
thorough V2 of the trait instruction generator and a corpus-wide
regeneration** (2026-09-28; see § "Trait generator V2" below).  Corpus: 648 trait files, 514 pairs (clean or by
decision), validator clean.  No chunk-3 pair is left open (2026-09-27).  Deferred to the arrangement
passes: the delegation tangle (micromanaging, absentee), burned_out and
mercurial as one-way pointers, the accountability shape, the warmth
triangle's duplicates (uncaring).  Next: the commit, then chunk 4.

Status 2026-09-28 (end of Roger's read-through).  Roger read every new or
altered role and trait and edited in place; each edit was regenerated in a
staging copy and its pair rechecked under a neutral label.  Rules that came
out of it, all now in AGENT_NOTES: no partner naming in a description, no
label echo in the opening, no chatbot framing, the trait is the persona's
own disposition and not what it urges on others, tendencies and not
states, resample once before acting on a missed check, and keep every
check's candidates and instructions
(`data/traits/antonym_check_history.jsonl`).  Corpus after the last
additions: 657 trait files, 259 pairs, validator clean; 337 roles.

Closed on 2026-09-28, so that chunk 3 is finished before the check-in:
- the entries that were waiting for wording: parent / childless and
  upper-class / working-class (both clean pairs), fair / unfair and
  vulnerable-narcissistic seeded, morally_disengaged not adopted (dropped
  from the consolidated dark/light list; amoral, unrepentant,
  blame_shifting, ends_justify_means and intellectually_dishonest cover
  its parts);
- three partners for unpaired traits: harsh (gentle), self-effacing
  (attention_seeking), other-focused (self_absorbed);
- the five reviewer rejections recorded as not adopted (flow_prone,
  gloating, resentful, left_brained, right_brained);
- the hedges "possibly even" (educated), "when needed" (course_correcting)
  and "often" (pensive): Roger is comfortable with each, they stay.

Deferred by Roger (2026-09-28): the generator V2 decision until after the
check-in; the arrangement passes (one-way pointers, candidate triangles,
the delegation tangle) until tangles are handled; Bourdieu capital
composition; the diaspora role.  male / female are decided: physical
traits, filed under § "Physical-attribute traits" (chunk 6) and deferred
with that track.  mastery / harmony is a chunk-7 entry (Schwartz), names
still to be chosen (recommended world-shaping / world-accepting).  The
sixteen chunk-3 backlog entries are optional or fallback names and stay
parked.  Next: Roger's commit, then chunk 4.

Status 2026-09-28 (close of chunks 0 to 3).  Roger then went through the
33 entries still parked in chunks 1, 1b and 2, and nothing in chunks 0 to
3 is open or undecided any more:
- 25 reviewer rejections dropped; `wasteful` stays in the backlog for the
  tangle pass;
- `other_focused` ↔ `self_absorbed`, `fair` ↔ `unfair` and
  `uncalculating` ↔ `calculating` paired by decision; a new, narrower
  `sincere` ↔ `ironic` confirmed from both sides; `ritualistic` a `non-X`
  singleton; `pensive` a one-way pointer at `unreflective`;
- the essence tetrahedron (§ "TODO: arrangement hunting") in place of the
  `constructivist` ↔ `essentialist` pair;
- the aligned-AI roles (§ "TODO: aligned_AI framing revision"):
  `instrumentally_aligned_ai` (renamed) and `virtue_aligned_ai` (new);
- the pair lists split into records (`pair_list_*_v1.json`) and lists for
  new work, the latter without `constructivist` / `essentialist` and
  without the triangle edge `compassionate` / `callous`; a triangle is
  not a pair, and how to judge and steer one is undecided.

Corpus: 659 trait files, 262 pairs, 1 tetrahedron, 2 triangles, the
16-member moral-circle sequence, 80 singletons, 34 unclassified (33
real-word one-way pointers for the tangle pass, and `malevolent`); 338
role files (337 roles and `default.json`).  Queue: chunks 0 to 3 hold only
final states and 16 backlog entries (optional or fallback names, and
`wasteful`).  Every decision is in `reports/seeding_log_2026-09.md`.

Checked in on 2026-09-28 on branch `anthropic-vllm-uv`, as ten commits
from `3f6de81` ("Expand and rework the trait and role corpus") to
`f5a5cd3`, plus the one that carries this note; the corpus as it stood
before is commit `93a8554`.  `reports/trait_gap_generation/` is a separate
line of work and was left out.  Next, in order: Roger's decision on the
trait generator V2 (deferred until this check-in), then chunk 4 (127
candidates).

## Trait generator V2: leaning toward doing it (Roger, 2026-09-28)

History of the decision.  2026-09-11: Roger deferred any voice guidance in
the description -> instruction prompts.  2026-09-16: after the Qwen
response check he called the trait-side problem minor and ruled out a
small template tweak: either nothing, or a thorough V2 that fixes every
issue worth fixing, followed by regenerating every trait (he was
consulting colleagues).  **2026-09-28: "It's looking increasingly like we
need to edit the trait instruction generation and redo everything"**,
said after the description review turned up one more generator-side
issue (the first item below).  Not yet decided, and **deferred until after
the chunk-3 check-in** (Roger, 2026-09-28); this section collects what a V2
would have to fix so the decision and the design can be made once.

Issues attributable to the generator (`_ROGER_TEMPLATE` in
`data_analysis/regenerate_trait_instructions.py`), with the evidence:

1. **Chatbot framing copied from the template's own example.**  The
   template never says assistant or chatbot, but it says "the model"
   eleven times and its worked example for "impolite" says "the user"
   twice.  26 trait files had "user" in an instruction on 2026-09-27;
   removing the word from a description cleared it from some regenerated
   files (`challenging`, `condescending`) and not others (`sycophantic`).
   See § "TODO: assistant-framing leakage scan".  The questions prompt
   likewise asks for "questions that a user might ask in a conversation".
2. **Traits rendered as what the persona urges on others.**  All five pos
   instructions of `adventurous` and of `unadventurous` were about
   encouraging or steering other people.  The descriptions started it
   (Christina's originals), and those are being rewritten, but the
   generator needs to be told that a trait is the persona's own
   disposition.
3. **The neg pole is softened into the sensible contrary.**  Told to
   "avoid the given trait and instead exhibit the opposing behavior:
   micromanaging", the generator wrote four good managers and one
   micromanager for `hands_off` (2026-09-26).  `NEG_REGENERATION_NOTES.md`
   records earlier cases.  Neg instructions reach only the antonym check
   and the goal tiers, but they decide which pairs look clean.
4. **Softening and hedging in pos instructions** of edgy traits (the
   2026-09-09 voice and softening audit, `reports/voice_audit_2026-09-09.md`).
5. **Terminology propagation and self-description** in responses: small
   but present (2026-09-16 check: 322 enact, 6 echo, 24 both, 8 neither).
6. **States instead of tendencies**: the description rule now covers it,
   but instructions such as "You are rattled" describe an episode; the
   generator should be told the trait is standing.
7. **Cost shape.**  `--instructions-only` costs as much as a full
   regeneration (about $0.03) because the combined call still generates
   questions and the eval prompt and discards them.

Not generator issues, but to settle in the same pass: the antonym check
is noisy (22 of 184 unchanged pairs gained a one-way synonym miss on a
fresh regeneration, 2026-09-28), so pair confirmation should either
accept synonyms or take the best of two samples.

Cost and timing.  A full regeneration of about 650 traits is about $19 of
generation at current prices, about $40 with the honest pair recheck
(non-X generation, check, labelled instructions), so it needs the
expensive-operations confirmation.  Nothing has been extracted from the
new corpus yet, and every existing trait touched this month is already
stale for extraction, so the cheapest moment for a V2 is before the
RunPod round, not after.

## Housekeeping once the directory settles (Sep 2026)

Bring these up to date after the current round of trait edits / additions is
finished (doing them per-edit wastes API calls).  Counts are as of 2026-09-07.

1. ~~**`trait_list.json` lags the instruction files**~~ -- resolved
   2026-09-07: `trait_list.json` and `role_list.json` are now generated from
   the instruction files' `description` fields by
   `tools/sync_entity_lists.py` (302 traits, 280 roles; `--check` exits 1
   while they lag, and `tools/tests/test_sync_entity_lists.py` fails).
   Never hand-edit them; rerun the script after adding or editing an
   instruction file.
2. **`trait_antonyms_v4.json` lags** -- 240 entries vs 302 files (62 missing).
   `generate_antonyms.py` prints to stdout; the file was assembled by hand with
   `[Manual override: ...]` annotations, so re-run for the missing traits and
   merge, preserving existing overrides.  ~62 Sonnet calls, well under $1.
   While merging, refresh the `analytical` entry: its `negative_label` still
   reads `systems_thinker` (file form); the trait JSON was corrected to
   `systems-thinker` on 2026-09-07.  Likewise `cosmopolitan`, whose
   `negative_label` moved from `provincial` to `non-cosmopolitan` the same
   day (see "Moral circle" at the top of this file).
3. **Goal classification covers 285 of 302 traits.**  Unclassified:
   conformist, cooperative, fragile, incurious, indecisive, materialistic,
   nonconformist, opaque, plain_spoken, precise, religious, stingy,
   superficial, thorough, uncertain, unforgiving, vague.  Run
   ```bash
   uv run python data_analysis/classify_goals.py --traits-only --names conformist cooperative fragile incurious indecisive materialistic nonconformist opaque plain_spoken precise religious stingy superficial thorough uncertain unforgiving vague
   ```
   (170 Opus calls, roughly $5; check `--model` default before launching).
   Any newly all-5-at-2 / all-5-at-0 traits then need adding to the
   appropriate tier of `data/goal_roles_and_traits.json` by hand.
4. **`data/README.md` counts** -- tree counts refreshed 2026-09-07 (302 trait
   files, 281 role files, 280 entries in the generated role list); re-check
   after the next batch of additions.
5. **`data/README.md` combination-scores summary is stale.**  It quotes the
   pre-escalation Sonnet-only picture (51% score 0, 16% score 3; t_ mean 1.25
   vs r_ 0.69).  After Opus escalation the rounded-mean distribution in
   `data/combination_scores.json` is 0: 2187, 1: 862, 2: 134, 3: 17 (mean
   0.38; role-source combos 0.26, trait-source 0.50; 477 escalated + 2 manual).
   If `goal_roles_and_traits.json` changes, `score_combinations.py` needs a
   re-run for the new combos only (it resumes); a full 3200-combo re-run is
   roughly $50-60 and needs sign-off under the expensive-operations rule.
6. **20 traits edited 2026-09-07; 5 regenerated, 15 reverted.**  19
   descriptions were normalised to the "This means ..." form and
   `analytical`'s negative_label was fixed to `systems-thinker`.  All 20 were
   regenerated with `claude-sonnet-4-6` (original `claude-sonnet-4-20250514`
   is retired; see data_analysis README); controls (`calm`, `patient`,
   `blunt`) drifted as much as the edited traits, so the description edits
   were inert.  Where the edit was a single clean grammatical change the
   original Sonnet-4 instructions/questions were restored (absolutist,
   ascetic, assertive, constructivist, cosmopolitan, extroverted, formalist,
   introverted, materialist, pacifist, rationalist, reductionist, relativist,
   universalist, analytical): those need only a descriptions-mode static
   rejudge (automatic by fingerprint).  Regenerated with Sonnet 4.6 and
   KEPT: closure_seeking, qualitative, compassionate, conservative,
   pragmatic -- these need new activations/vectors on RunPod, static and
   response-mode rejudging on compassionate_vs_callous,
   progressive_vs_conservative, idealistic_vs_pragmatic and
   quantitative_vs_qualitative, and re-sweeps of the seven steering
   configs on those axes (exact list in AGENT_NOTES § "TODO: regenerate
   activation/vector data", Sep 2026 batch).  Antonym check (28 traits): 7 of 9 clean pairs
   re-confirmed bidirectionally; analytical/systems_thinker and
   closure_seeking/open_ended remain manual pairings as before.
   Consequences are tracked in `AGENT_NOTES.md` § "TODO: regenerate
   activation/vector data" (Sep 2026 batch); review flags from the
   regeneration diff (`formalist` narrowed to response formatting,
   `cosmopolitan` negs leaning ethnocentric, `compassionate` negs leaning
   contemptuous) are moot for the reverted ones; the `compassionate` one
   was accepted with the keep decision (the neg is deliberately the
   ambiguous `non-compassionate` corner of the triangle, so "contempt"
   is within bounds).
   The "This trait involves ..." / "This involves ..." variants (29 traits)
   were left as-is; they read fine in every prompt site.

## TODO: assistant-framing leakage scan (Sep 2026)

Tabled by Roger on 2026-09-07; scan and think before acting.

**Update 2026-09-16 (Roger):** scan the *descriptions* for chatbot-specific
phrasing ("the user", "users", "in your responses") and remove it,
rephrasing to "people", "someone", "others"; `inspirational` was the
first, rewritten during the pairing review because its check returned
"encouraging" for the new `discouraging`.  Each rewrite makes that trait
stale for extraction; batch them with the next round.

Christina's original descriptions and instructions were written for an
assistant answering a user, and that framing survives in the generated text.
Counts as of 2026-09-07 (`data/traits/instructions/`, 302 files): 114 files
have at least one instruction string mentioning "the user" / "users" (87
strings) or "response(s)" (261 strings); 15 descriptions do (the "the user"
group: accommodating, adaptable, challenging, educational, exploratory,
inquisitive, inspirational, socratic, supportive, sycophantic; the rest say
"responses").  Role side: 13 files, mostly the AI-flavoured roles, where the
framing is the point.

**Update 2026-09-28: the "the user" descriptions are done.**  Roger
rewrote `sycophantic`, `challenging` and `condescending` himself on
2026-09-27 and had the last six rewritten the next day (`accommodating`,
`adaptable`, `educational`, `exploratory`, `inquisitive`, `socratic`);
no trait description now says "user" or "questioner" (roles: `assistant`,
legitimately, and `librarian`, meaning library users).  Related
framings Roger found in the same review and had rewritten: traits
phrased as what one urges on others rather than what one does
(`adventurous` / `unadventurous`, `evil`, `independent`; still present in
`cautious` / `bold` and `risk_seeking`), and traits written as a passing
state rather than a tendency (`calm` / `excitable`, `flustered`, `manic`,
`despairing`).  About 60 trait descriptions still use the milder framing
words "answering", "responses", "the question".

**Where "the user" in the instructions comes from (checked 2026-09-28).**
The generator template (`_ROGER_TEMPLATE`, forked from Christina's)
never says assistant or chatbot; it says "the model" eleven times, and
its worked example for "impolite" says "the user" twice ("thanking the
user", "acknowledges the user warmly").  The LLM copies the example:
26 trait files had "user" in an instruction on 2026-09-27, and
regenerating from a description without the word cleared it from some
(`challenging`, `condescending`) but not others (`sycophantic`).  The fix
is to the template's examples, which is a rubric change, so it belongs
to the trait-rubric V2 decision (Roger, 2026-09-16: no small template
tweak), not to a piecemeal edit.

- **Judge side: not a concern.**  The description is spliced into the eval
  prompt, where "the user" reads naturally, and the judge scores behaviour
  either way.
- **Persona side: the actual worry.**  A pos instruction such as "you tailor
  your responses to the user" is the assistant persona talking, so the
  trait vector may pick up a "being an assistant" component on top of the
  trait.  The corpus-mean subtraction removes whatever part is shared with
  the default persona, but a trait whose instructions lean on the framing
  more than average keeps a residual.  Unknown whether it is measurable.

When picked up: (1) rerun the scan below and read the hits by eye; (2) decide
whether to rephrase to "the other person" / "others" / "what you say"; (3)
any rephrase of instructions is a regeneration-class edit (new activations,
vectors, rejudge), so batch it with the next RunPod round rather than doing
it piecemeal.  Description-only rephrasing is cheaper (descriptions-mode
static rejudge only, see item 6 above) but does not touch the persona side.

```bash
grep -lE '\b(the user|users|responses?)\b' data/traits/instructions/*.json | wc -l
grep -oE '"(pos|neg)": "[^"]*\b(the user|users)\b[^"]*"' data/traits/instructions/*.json | head
```
