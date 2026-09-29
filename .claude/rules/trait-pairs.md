---
paths:
- data/goal_roles_and_traits.json
- data/traits/**
- data/roles/**
- data_analysis/generate_antonyms.py
- data_analysis/regenerate_*.py
- pipeline/1_generate.py
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: trait-pairs

**When:** adding or regenerating trait clean pairs or instructions, or running combined response generation.  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=trait-pairs`; edit there, then run `uv run python tools/sync_agent_notes.py`.

## Adding New Trait Clean Pairs

When adding a new trait B that is the antonym of an existing trait A (e.g., adding `obedient` as the antonym of `rebellious`):

### Process

1. **Create seed file** in `data/traits/instructions/B.json` with `positive_label`, `description`, and `negative_label` set to `non-B` (NOT to A yet).  If B comes from a named standard, label it `<pole> (<Standard>)` (e.g. `traditional (Inglehart-Welzel)`, stem `traditional_inglehart_welzel`), keep the source out of the description, and add an optional `"source"` field -- see § "Standard-derived trait labels" under the file-name vs display-name convention.

2. **Generate instructions** for B:
   ```bash
   uv run python data_analysis/regenerate_trait_instructions.py --traits B --force
   ```

3. **Run antonym generation** for B (and re-run for A to double-check):
   ```bash
   uv run python data_analysis/generate_antonyms.py --traits B A
   ```
   (`--traits` scopes to specific traits; omit for all.)
   This should independently discover A as B's antonym. If A's antonym generation also returns B, we have a **clean pair**: A↔B confirmed bidirectionally.

4. **Update negative_labels**: Set B's `negative_label` to A and A's to B in their instruction files, and record `"arrangement": {"kind": "pair", "members": [A, B]}` (sorted stems) on both; `uv run python data_analysis/check_arrangements.py` must pass (see § "The `arrangement` field").

5. **Regenerate instructions** for B with the proper antonym (the Roger prompt style injects the antonym into the neg instruction clause):
   ```bash
   uv run python data_analysis/regenerate_trait_instructions.py --traits B --force
   ```
   A should NOT need regeneration since it already has B as its negative_label.

6. **Regenerate the derived lists**: `uv run python tools/sync_entity_lists.py`.  `data/traits/trait_list.json` and `data/roles/role_list.json` are generated from the instruction files' `description` fields (since 2026-09-07; never hand-edit them).  `--check` exits 1 while they lag, and `tools/tests/test_sync_entity_lists.py` fails.

7. **Run goal classification** if the trait needs goal scoring (for `data/goal_roles_and_traits.json`).

### The pairing loop when the check does not name the original (Roger, 2026-09-17)

Set out during the September 2026 pairing review, after the agent had
been reporting one-directional checks as decisions instead of working
them.  **This is not a strict algorithm.**  The steps below are the moves
available; the work is judgement and some creativity at each one: is
this a well-formed clean pair with matched scopes, which word or which
clause would make it one, is the original's answer telling us the pair
is wrong or only that the label is, and, when nothing lines up after a
round or two, being willing to give up (a singleton, a one-way pointer,
or a deleted file is a fine outcome).  Bring the evidence, not just the
verdict, to Roger for the calls that change existing files.  Starting
from an existing file whose `negative_label` names a trait with no file:

1. **Seed the recorded name** with `non-X`, generate, run the check on
   the new file.  If it names the original, pair.
2. **If not, adjust the new description** so the scopes line up with the
   original's (same domain, same breadth; add the clause the original
   has and the new one lacks), regenerate, re-check.  One or two rounds.
3. **If it still does not, run the check on the original** and read what
   it offers besides the recorded label (its neg instructions were
   written with that label injected, so the recorded word usually
   appears; the *other* words are the information).  State that bias
   with the result.  Where the original's answer is the deciding
   evidence, regenerate the original under `non-X` for an unbiased
   answer (as done for `diplomatic` on 2026-09-17); that costs the
   original's extraction, so do it deliberately, not by default
   (confirmed by Roger 2026-09-17).  Seed the best of the offered words,
   generate, check both sides, pair if they name each other.
4. **If nothing pairs**, something else is needed: rename the original
   to the check's word (RO, `seed_entities.py rename`), make the original
   a singleton with `non-X`, or drop the new file if it duplicates an
   existing trait.  These are Roger's calls, brought to him with the
   evidence from steps 1-3.

Pairing by decision (P) is the last resort, used when one side names
the other and the other side's answer describes the same content under a
different word.  Worked examples of every branch, including the
give-ups, are in `reports/seeding_log_2026-09.md` (the 2026-09-16/17
pairing review: 22 cases, 34 pairs, 12 singletons, 7 deletions).

### Why non-X first?

Starting with `non-B` instead of `A` ensures the antonym generator discovers `A` independently from the neg instructions, rather than being primed by us providing it. This validates that the pos/neg instruction pairs genuinely capture the A↔B opposition.

**The same procedure applies when an existing pair member's description
is rewritten** (done for five traits in the 2026-09-11 voice repair):
set its `negative_label` to `non-X`, regenerate in full, run the check,
restore the label, regenerate `--instructions-only` so the neg clause
names the real partner again.  Rerolling instructions under an unchanged
description does not need the recheck.  Read the check's answer against
the decision table in § "Corpus expansion policy": `conciliatory|conflict-averse`
confirms conciliatory; `non-materialistic|idealistic` does not confirm
spiritual (that one needed a description revision before it returned
`spiritual|idealistic`).

### Role pairs (procedure to design, 2026-09-11)

Roles have no `negative_label` and no neg instructions, so steps 1, 3
and 5 above do not apply; role pairs exist only in the `arrangement`
field.  Roger (2026-09-11) wants a role-pair check that keeps the rest of
the procedure: seed one side, generate its instructions, ask a generator
given the description and pos instructions to name the *opposite role*
and rate the opposition, run it from both sides, and record the pair only
when the two sides name each other.  Needs a role mode for
`generate_antonyms.py`.  First candidate: provincial ↔ cosmopolitan (both
roles, both currently `singleton`); details and the existing unchecked
role pairs in `data/roles/instructions/ROLES_TO_ADD.md` § "Role pairs to
record".

### Description-writing rules for new seeds (Sep 2026)

What the voice / softening audit (2026-09-09), the role rubric V2 pilot
(2026-09-11..12) and the Qwen response checks (2026-09-11, 2026-09-16)
taught about writing the `description` of a new trait or role.  The
description is the only hand-written text; everything else is generated
from it and echoes it, so a hedge or an outsider's word here reaches the
instructions, the eval prompt and (for roles) the responses.  Apply these
to every seed, by hand or by a writing agent; Roger reviews before
`seed_entities.py write`.

1. **Form.**  Traits: "This means ..." (274 of 302 files; normalised
   2026-09-07), one or two sentences, 18-32 words (p10-p90; median 24),
   US English.  Roles: "A <role> is someone who ..." / "An <role> is a
   ... that ..." for the recent files (older ones use "This refers to"),
   23-43 words (median 28).  **Do not open by repeating the label**
   ("This means being grateful: ...").  Roger, 2026-09-27: every prompt
   that uses a description supplies the label right beside it
   (`**{positive_label}**. {description}` in the generator, the eval
   prompt and the pipeline judge; `**{name}**: {content}` in static
   judging; the axis name and examples in the axis rubric header, where
   the description is the pole text), so "X: This means being X: ..." is
   pure repetition.  Go straight to the behaviour ("This means noticing
   every kindness, ...") or open with a short gloss that paraphrases the
   trait and then expand it ("This means being drained by company and
   restored by solitude: keeping to oneself, ...").  Keep a label in the
   opening only when it carries a qualifier that picks the sense
   ("aristocratic about rank", "tough on people", "from the Eastern
   Hemisphere", "having a calm temperament").  An earlier version of this
   rule prescribed the label anchor, and 344 of the 390 descriptions
   written in September 2026 had it (2 of the 258 older ones); see
   `reports/seeding_log_2026-09.md` for the clean-up.
2. **The vice is a vice.**  No "appropriately", "when appropriate",
   "overly", "excessive", "too", "sometimes", "may", "can", "tends to",
   "healthy", "in a balanced way"; no virtue-framing of an edgy trait
   (irreverent is not "questions pretension", it jokes about the sacred and
   does not care) and no villain-laundering of a bad role (a destroyer
   destroys; a smuggler moves contraband for money).  A softened
   description samples the inoffensive centre of the concept and the
   pole then sits on top of an existing milder trait.
3. **Inside voice.**  Write in words the persona would use of itself:
   no case-worker, policy, clinical or anthropologist's vocabulary
   ("marginalised", "engages in", "exhibits", "demonstrates", "navigates",
   "individuals who"); name the particulars of the role's world (the
   tools, the customers, the hours, the enemy) rather than its category.
   For roles this matters most: the register of the description was found
   to propagate through the instructions into Qwen's responses.  For
   traits the instruction register does not propagate, but self-labelling
   does, so avoid handing the generator a paragraph it can quote back.
4. **Scope by a test, not by a list of synonyms.**  State what separates
   this pole from its nearest neighbours, ideally as one question the
   judge can ask ("would the persona do this if the setting were real?",
   "does the persona argue for continuing, or simply never ask?").  Check
   the nearest existing traits and roles before writing (nine names exist
   on both sides; `ls data/{traits,roles}/instructions`), and if the new
   entity is a deliberate near-duplicate of an existing one (a standard's
   version of a plain trait), say so in `source`, never in the
   description.
5. **Pairs are written together.**  Both poles get the same scope, the
   same trigger and the same length; the neg pole is an opposite, not an
   absence ("course-correcting", not "non-tunnel-visioned"), and where no
   opposite exists the seed keeps `non-X` and the antonym check decides.
   Name the mechanism that keeps two neighbouring pairs apart (motivated
   vs unmotivated; a frame believed not to count vs stakes believed high).
   **Do not name the partner in the description** (no "This means being
   X, never Y: ..."; no "The opposite of Y." tail).  Roger, 2026-09-25:
   naming the partner is a thumb on the clean-pair scale (the check is
   supposed to find the partner from the behaviour, and the label is
   already injected into the neg instructions) and probably pulls the two
   poles' description embeddings together.  It is a last resort for a
   pair that cannot be had any other way, recorded as such.  The
   pre-September corpus's "... rather than <opposite behaviour>" clause
   (86 of the 306 files at commit `93a8554`) is tolerated but should be
   used only where the word is polysemous and the contrast picks the sense ("temperate" about
   beliefs, not drink); prefer letting the body after the colon do the
   work.  The 2026-09-25 chunk-3 packets prescribed "never X"; 81 seeded
   files and the 3D drafts were stripped and rechecked on 2026-09-26 (see
   `reports/seeding_log_2026-09.md`).
6. **Standards.**  For a named instrument, paraphrase the canonical
   definition in the corpus form, keep the instrument's name out of the
   description, put the provenance in `source`, and label
   `<pole> (<Standard>)` (§ "Standard-derived trait labels").
7. **Things that are not the description's job.**  Questions and eval
   prompts are generated; do not steer them from the description.  The
   only exception is a scope sentence the questions must respect (for
   the frame pair: every question inside a stated frame), which goes in
   the queue entry's `description_notes` for the reviewer, not in the
   description.
8. **Sensitive entities** (the part-4 sensitive batch, the physical
   track, demographic memberships): the same rules; the description
   states the membership or attribute plainly and stops.  No disclaimers,
   no "respectfully".  The generator's refusals are handled at generation
   time, not by hedging the seed.

**Trait generator V2: leaning toward it (Roger, 2026-09-28).**  The rules
above govern descriptions; several problems found in the September
review are in the instruction generator instead (chatbot framing copied
from the template's own "the user" example, traits rendered as what the
persona urges on others, softened neg poles, states for tendencies).
Roger ruled out a small template tweak on 2026-09-16 and on 2026-09-28
said he is leaning toward a thorough V2 with a corpus-wide regeneration.
He deferred the decision until after the chunk-3 check-in, which was
made on 2026-09-28, so the decision is now open and comes before chunk 4
is generated.
The issue list, evidence, cost and timing are in
`data/traits/instructions/TRAITS_TO_ADD.md` § "Trait generator V2".  Until
he decides, do not edit `_ROGER_TEMPLATE`, and treat every trait
regeneration as provisional.

Writing-agent recipe (used from chunk 1 onward): give the agent this
section, the queue entries for one sub-chunk (label, partner, the
`description_notes` and `decision` fields, and any `description_draft`),
the nearest-existing table if one exists, and five recent corpus
descriptions of the same entity type as examples; ask for the description
plus one line naming the nearest existing entity and why it is different;
put the result in the entry's `description` field with status `ready`
only after Roger has read it.

### Seeding tooling (chunk 0, Sep 2026)

`data/seed_queue.json` is the inventory of every candidate in
`TRAITS_TO_ADD.md` and `ROLES_TO_ADD.md` (built 2026-09-16 from the two
files by five extraction passes, then merged and checked against the
corpus; the markdown files stay the record of the reasoning, the queue
is the record of state).  One entry per entity with `stem`, `label`,
`entity_type`, `chunk`, `sub_chunk`, `pairing`, `partner`,
`arrangement_members`, `description_draft` / `description_notes` (from
the files) and `description` (final), `source`, `tags`, `status`,
`decision`, `alternatives`, `section` / `lines` (where in the files),
and, once run, `check_result`.  Status lifecycle
`candidate -> ready -> seeded -> generated -> checked -> paired | done`;
parked: `tbd`, `backlog`, `not_adopted`, `superseded`, `exists`.

[`data_analysis/seed_entities.py`](./data_analysis/seed_entities.py)
drives it: `status`, `write` (seed JSONs for `ready` entries: traits get
`negative_label = non-<label>` unless `pair_by_construction`, roles get
`arrangement: singleton`; refuses to overwrite; runs
`sync_entity_lists.py`), `generate` (calls the two regenerate scripts;
prints the $0.03-per-entity estimate and refuses over $20 without
`--confirm-expensive`), `check` (runs `generate_antonyms.py`, classifies
each answer as nice / mismatch / nearly_nice / nasty / open against the
registry of existing *and queued* stems per the decision table below,
stores it in `check_result`), `rename --old X --new LABEL --partner Y`
(Roger's RO action: rename the existing trait to the word the check
returned, if free, regenerate it, re-check both sides; the file moves with
`git mv` and gets a `renamed_from` field), `pair --a X --b Y` (reciprocal labels,
`arrangement` pair on both, `--instructions-only` regeneration of the new
side, `check_arrangements.py`, `sync_entity_lists.py`) and `report`.
Every subcommand has `--dry-run`.  Tests:
`data_analysis/tests/test_seed_entities.py`.

The chunk-3 runs (2026-09-25/26) were driven by small scripts in the
session scratchpad, not in the repo: packet builders that assemble the
rules section above plus four post-strip example pairs and each entry's
queue notes for the writer agents; `run_pairs.py` (fold descriptions and
reviewer verdicts into the queue, `write` / `generate` / `check`, `pair`
the ones whose checks name each other, print the misses); `run_singles.py`
(the same for `non-X` singletons, recording the check answers for later
pairing); and the one-off strip / rename / edit rounds.  Their outputs
and every decision are in `reports/seeding_log_2026-09.md`; if the
pattern is needed again, rebuild from that log rather than hunting for
the scripts.  Two lessons worth keeping: a reviewer rename puts the entry
under two stems in a stem-keyed dict (deduplicate before `write`), and
the `check` answer is what the *checked* file's own description elicits,
so editing a pole changes only that pole's answer.

**Regenerate in a staging copy when Roger is editing (2026-09-28).**  The
regenerate scripts read a file, spend 20-60 seconds on the API call, and
write the whole file back, so an edit Roger saves in between is lost.
When he is reviewing files while a batch runs, copy the files to a
scratch directory, point `regenerate_trait_instructions.TRAITS_DIR` at it
(import the module and set the attribute; the usage record keeps its
repo path), run the check on the staged files with
`generate_antonyms.classify_one`, and merge `instruction`, `questions`,
`eval_prompt` and `generator` back only where the repo file's description
and label still match what was staged.  Skipped files are his edits:
rerun them.  The check is also noisier than a single run suggests: on
the 2026-09-28 recheck of 184 pairs whose content had not changed, 22
gained a one-way miss on a synonym, and a second regeneration brought 14
of the 22 back.  A single miss on a previously clean pair is weak
evidence: resample once before acting, and treat a miss that repeats
with the same words as real.

**Keep every sample (Roger, 2026-09-28).**  Record the antonym candidates
of every check, not only the latest, together with a copy of the
instruction set that check read: the candidates are more candidate
labels, and they are a measure of that particular generation.  The
record is `data/traits/antonym_check_history.jsonl`, one JSON object per
check, append-only, beside `instructions/` so nothing that globs the
corpus reads it: `stem`, `checked_at`, `phase`, the label the file had
when checked (normally `non-X`), `description`, `intended`, `returned`,
`candidates`, `known`, `category`, `intended_hit`, `score`, `reasoning`,
`instructions` (the pos / neg pairs read) and `generator`.
`seed_entities.py check` and `rename` append to it
(`append_check_history`, `check_history_record`) and add each answer to
the queue entry's `check_answers` list; `check_result` still holds the
latest.  Any ad-hoc runner must do the same, and must write the record
*before* the labelled `--instructions-only` regeneration overwrites the
instructions the check read.  The 385 records from 2026-09-27/28 before
this rule were backfilled with their candidates but without
instructions (not kept at the time).

Review order (Roger, 2026-09-17): writer agents draft, a review agent
checks, then seed and generate *before* Roger reads the descriptions, so
the antonym-check results accompany them; generation is cheap and a
regeneration after his edit costs three cents, while a round trip of
coordination costs more.  Only edits to existing files (labels, renames,
rewrites) wait for him.

### Corpus expansion policy and the clean-pair decision procedure (Sep 2026)

Roger's stated trade-offs for adding traits and roles (2026-09-09), written
down so the seeding rounds apply them consistently.  Correct here if wrong.

**Three competing criteria, roughly in priority order:**

1. **Gap filling.**  The goal is good coverage of the whole roles + traits
   set across a multi-dimensional persona space, without distorting its
   shape by oversampling one region (skewness matters), *except* that the
   goal-related and alignment / misalignment regions are deliberately
   sampled more heavily because they are what we most want to study.
2. **Pair creation.**  Clean pairs are currently more useful for judging
   and steering than triangles, simplices or sets (steering a triangle as
   three-or-more pairs and checking the resulting geometry is planned but
   harder), so a clean pair is worth somewhat more than two unpaired
   points.  The premium shrinks once the number of clean pairs is well
   above the effective dimensionality of the space (16-64 on current
   analysis; ~21 effective dimensions in the 60-axis cohort at
   2026-05-13), after which extra pairs mostly add redundancy.
3. **Rewriting an existing description** for better pairing or coverage
   (scope, emphasis, phrasing) is allowed but costs activation
   regeneration and rejudging, so avoid it unless the improvement is
   clear.

Outside the official external arrangements, be open to discovering that a
conceived pair is really a triangle or a larger arrangement; record what
the generator finds rather than forcing a pair.

**Where the neg instructions are actually used (verified 2026-09-09).**
Response generation (`assistant_axis/generation.py`) builds system prompts
from `instruction[i]["pos"]` only, so activations, vectors, axes, static
judging (descriptions + pos instructions), steering and the refusal
fallback never see a neg string.  The neg instructions are read by exactly
three things: `data_analysis/generate_antonyms.py` (the clean-pair
identification step), `data_analysis/classify_goals.py` (goal
classification of the neg polarity, which feeds the hand-built
`goal_roles_and_traits.json`), and `data_analysis/sample_trait_responses.py`
(a diagnostic sampler).  Consequence: a softened *neg* pole cannot distort
an existing vector; it can only mislead the antonym check and the goal
tiers.  Softening in a *pos* pole or a description is what reaches the
data.  Judge the audit flags accordingly.

**Official external arrangements are held to a looser pair standard.**  For
axes imported from a named instrument (Big Five, HEXACO, Inglehart-Welzel,
...), the pairing is fixed by the standard: set the two `negative_label`s
to each other by construction, so the checker sees a clean pair, and treat
the antonym check as *informational*.  Still run it: a failed check on a
summarised (non-canonical) description is a hint to adjust the wording.

**The antonym check needs a registry of existing *and proposed* names.**
`generate_antonyms.py` sees only the trait's own definition and
instructions and returns one or more candidate antonyms (`a|b`).  Interpret
its answer against the union of existing stems and the seed queue:

| generator returns | case | action |
|---|---|---|
| one word, and it is the intended partner (existing or queued) | the nice case | pair; point the labels at each other |
| one word, an existing or queued trait that is *not* the intended partner | mismatch | decide: relabel to the found partner, seed the intended one anyway as a sibling (triangle), or keep `non-X` |
| several words, exactly one of them existing or queued | nearly nice | usually pair with that one; note the alternatives |
| several words, more than one existing or queued (with or without the intended one) | the nasty case | judgement call; often a triangle or set; record the options in the queue entry, do not force a pair |
| no existing or queued word at all | open | seed the best new word as a completion, or keep `non-X` |

Having the full proposed list before seeding removes the sequencing
problem where a partner the generator names is one we planned to add
later.  Recorded decisions go in the seed queue entry (`decision`,
`alternatives`), not only in chat.

### The `arrangement` field (Sep 2026)

Every role and trait instruction JSON may carry an `arrangement` field
recording which set of same-type entities it belongs to and the shape of
that set.  Decided 2026-09-08; loader and validator in
[`assistant_axis/arrangements.py`](./assistant_axis/arrangements.py),
CLI check in [`data_analysis/check_arrangements.py`](./data_analysis/check_arrangements.py),
one-off backfill in [`data_analysis/backfill_arrangements.py`](./data_analysis/backfill_arrangements.py).

```json
"arrangement": {"kind": "pair", "members": ["callous", "compassionate"]}
```

or a list of such objects when the entity is in several sets (`malicious`
is one pole of benign ↔ malicious *and* one corner of the
compassionate / malicious / callous triangle).

| kind | members | meaning |
| --- | --- | --- |
| `singleton` | none | belongs to no set (explicit) |
| `pair` | 2 | one clean pair |
| `triangle`, `tetrahedron`, `N-simplex` (N ≥ 4) | 3, 4, N+1 | mutually opposed corners |
| `square` | 4 | two axes; the 2-cube / 2-orthoplex distinction is deliberately not kept (split into `square` / `diamond` later if it matters) |
| `cube`, `N-cube` (N ≥ 4) | 8, 2^N | every combination of N binary axes (MBTI's 16 types are a 4-cube) |
| `octahedron`, `N-orthoplex` (N ≥ 4) | 6, 2N | the poles of N clean pairs (HEXACO is a 6-orthoplex) |
| `ring` | ≥ 3, ordered | circumplex: neighbours close, opposites opposite |
| `tree` | ≥ 2, with `parent` / `children` | hierarchy (domains over aspects over facets) |
| `map` | ≥ 2 | unordered, expected to have low-dimensional metric structure (nationalities, ethnicities) |
| `sequence` | ≥ 2, ordered | roughly an axis with interesting deviations (the moral-circle group) |
| `set` | ≥ 2 | unstructured |

Numeric aliases are accepted and canonicalised (`2-simplex` → `triangle`,
`3-orthoplex` → `octahedron`, `2-cube` → `square`, ...).

Rules:

1. **Members are file stems, never labels**, include the entity itself,
   and are sorted, except for `sequence` and `ring` whose order is the
   content.  Every member records the identical arrangement.
2. **A missing field means not yet classified**; `singleton` is written
   explicitly, and **only for a trait whose `negative_label` is a `non-X`
   placeholder** (roles have no label).  A trait whose real-word label
   has no file, or points one way at a trait paired elsewhere
   (`inspirational` → discouraging, `subversive` → orthodox), stays
   unclassified: the label is a pairing still to be decided, and the
   checker rejects a singleton there (rule added 2026-09-17 after the
   agent had written two such singletons).  As of the 2026-09-08 backfill the not-yet-classified traits
   are the 142 with a real-word `negative_label` that has no file or is a
   one-way pointer (see TRAITS_TO_ADD § "TODO: antonym gap-filling pass");
   cleaning them up (pairs, triangles, sets, ...) is that TODO.
3. **The pair convention is unchanged**: a clean trait pair is still two
   files whose `negative_label` fields point at each other.  The field is
   authoritative for *shape*; the labels stay the prompt-facing antonyms;
   the checker refuses to let them disagree (every `pair` must be
   reciprocal by label, every reciprocal pair must be recorded in some
   arrangement of a classified trait, octahedra / orthoplexes must
   partition into clean pairs).  Roles have no `negative_label`, so role
   pairs (angel / demon, predator / prey, ...) exist only here.
4. **Optional keys**: `axes` (the clean pairs forming the axes of a
   square / cube / orthoplex), `parent` and `children` (tree only),
   `source` (provenance of an imported structure), `note` (free text).
   Unknown keys are preserved.
5. **Check after every edit**:
   `uv run python data_analysis/check_arrangements.py` (exit 1 on any
   inconsistency; `--list-unclassified` prints the backlog).
   `assistant_axis/tests/test_arrangements.py::test_real_corpus_is_consistent`
   fails while the checked-in corpus is inconsistent.
6. **Research task, once embeddings exist**: confirm each declared shape
   is approximately descriptive of the embedding geometry (pairs as line
   segments that roughly intersect at a common centre for orthoplexes;
   the four off-pair segments of a square falling into two roughly equal,
   parallel, mutually orthogonal sets; sequences projecting monotonically
   onto their first principal direction).  Where it fails, reclassify,
   except that structures imported from an external system (HEXACO, MBTI)
   may keep their declared shape regardless.

### `data/goal_roles_and_traits.json` structure

Lists of roles and traits partitioned by goal content, for experimental use.
Each of `roles` and `traits` has `goal` (all-5 @ 2) and `non_goal` (all-5 @ 0) sublists.

**Ordering convention** -- items are randomized within tiers, tiers are concatenated:

- `roles.goal`: first 30 = primary set (varied), last 10 = cluster duplicates
- `roles.non_goal`: first 30 = most varied (max non-goal semantic spread), next 10 = nice-to-haves, last 30 = most redundant with first 30
- `traits.goal`: first 30 = max goal-space variation (less-HHH-default side of pairs, distinct ethical frameworks), next 10 = remaining moral circle spectrum, last 26 = default-side of pairs + redundant goal directions
- `traits.non_goal`: first 30 = max persona-property variation (less-default side of pairs), next 10 = nice-to-haves, last 17 = pair partners + redundant

### Combined response generation (pipeline)

The pipeline (`pipeline/1_generate.py`) supports two modes:

**Roger mode** (default): Generates combined role+trait instructions, standalone traits, and the default baseline. Uses `data/goal_roles_and_traits.json` to pick the top-N roles/traits from each goal/non-goal list.

- `r_{role}__{trait}` = goal role x non-goal trait (goal from role)
- `t_{role}__{trait}` = non-goal role x goal trait (goal from trait)
- Double underscore `__` separates role and trait in filenames
- Instructions are index-matched (pair 0-0, 1-1, ..., 4-4), concatenated with `\n`
- `--goal_count` / `--non_goal_count` (default 30 each) control how many items from each list; error only if a count exceeds BOTH lists it applies to

**Christina mode**: Processes standalone roles (or traits) from `--roles_dir`. Run separately per entity type (different `--roles_dir` and `--output_dir`) to avoid name collisions.

**Step 3 eval_prompt routing**: The judge (`3_judge.py`) requires an explicit `--entity_type {role,trait,combination}` flag to select the eval_prompt:

- `role` — look up stem in `data/roles/instructions/`, use its `eval_prompt` field.
- `trait` — look up stem in `data/traits/instructions/`, build a pipeline-specific 0-3 eval_prompt from the description. (The 0-100 `eval_prompt` in trait JSONs is NOT used by the pipeline, since `parse_judge_score()` rejects scores > 3.)
- `combination` — parse `r_<role>_t_<trait>` filename, build a compound 0-3 eval_prompt from both descriptions.

`default` is skipped under all entity types (step 4 uses all activations without scores).

**DO NOT try to autodetect entity type from the filename.** 9 names exist in both `data/roles/instructions/` and `data/traits/instructions/`:

    ascetic, contrarian, cosmopolitan, generalist, pacifist,
    patient, perfectionist, romantic, stoic

Any roles-first (or traits-first) fallback will silently mis-score one side for these 9. `run_pipeline.sh` passes the correct `--entity_type` per output-subdir type (Roger mode: `roles|traits|combinations`; Christina mode: inferred from `ROLES_DIR`). If you add a new standalone invocation of `3_judge.py`, you must pass `--entity_type` explicitly.

Steps 2, 4, 5 are unchanged — they process whatever files appear in their input directories.

---
