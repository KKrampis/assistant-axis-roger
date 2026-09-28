# Roles To Add

Candidate roles and role-side coverage notes.  Companion to
`../../traits/instructions/TRAITS_TO_ADD.md` (which holds the trait-side
candidates and the process for adding clean pairs).  Started 2026-09-07
during the demographic-coverage audit.

## Process for adding a role

1. Seed `data/roles/instructions/<role>.json` with a `description` field only.
   Use the file-name form for the stem (`coral_reef`, `devils_advocate`); the
   display form is derived by `regenerate_role_instructions.role_display_name`.
2. Generate the five pos instructions, 40 questions and the eval prompt:
   `uv run python data_analysis/regenerate_role_instructions.py --roles <role> --force`
3. Regenerate the derived index: `uv run python tools/sync_entity_lists.py`.
   Since 2026-09-07 `data/roles/role_list.json` is generated from the
   instruction files' `description` fields (never hand-edit it); `--check`
   reports lag, and `tools/tests/test_sync_entity_lists.py` fails while the
   list lags.
4. Check the name against `data/traits/instructions/` -- nine names already
   exist on both sides (see `AGENT_NOTES.md` § "Trait/role name collisions").
   Give the file an `arrangement` field, `{"kind": "singleton"}` unless the
   role is one pole of a role pair, and run
   `uv run python data_analysis/check_arrangements.py`.
   Role pairs exist only in `arrangement` (roles have no `negative_label`);
   the clean-pair check for roles is still to be designed, see "Role pairs
   to record" below.
5. Run `data_analysis/classify_goals.py --names <role> --roles-only` if the
   role should be eligible for `data/goal_roles_and_traits.json`.
6. Run activation / vector extraction for the new role before it can be used
   as a steering base persona or role-pair pole.

## Demographic coverage audit (Sep 2026)

Role-side findings for the "vital and demographic" survey axes.  Roles carry
almost all demographic *membership* in this corpus; traits carry attitudes.

- **Age**: covered.  infant, toddler, teenager, adolescent, student, prodigy,
  graduate, newlywed, parent, grandparent, elder, retiree; widow skews old;
  ~240 occupational roles are implicitly adult; `ancient` is the non-human
  extreme.  Matched pair is trait-side (`young` ↔ `elderly`, see
  TRAITS_TO_ADD.md).
- **Birth cohort / generation**: only implicit (gamer, influencer, blogger,
  podcaster read young; luddite, traditionalist read old).  Folded into age.
- **Life stage**: well covered.  Family track: single (implicit), newlywed,
  parent, grandparent, with divorcee, widow, orphan as off-ramps.  Career
  track: student, graduate, worker (implicit), retiree.  Other transitions:
  immigrant, refugee, exile, expatriate, patient, prisoner, survivor, veteran.
  No matched pair planned.
- **Migration / refugee status**: fairly covered.  native (implicit),
  immigrant, expatriate, refugee, exile, nomad, wanderer, pilgrim, provincial
  (rooted extreme), hybrid (loose second-generation proxy), ambassador and
  emissary.  Two natural axes: rootedness (settled ↔ nomadic) and agency
  (chosen expatriate ↔ forced refugee).
- **Veteran status**: covered.  veteran, soldier, warrior, peacekeeper, spy vs
  the civilian default.
- **Sex / gender, sexual orientation, race / ethnicity, nationality, language,
  disability**: no role coverage.  Only weak stereotyped coding exists (widow,
  witch read female; soldier, warrior, pirate, mechanic read male).  All of
  these are being handled as traits; see TRAITS_TO_ADD.md.

## Candidate roles

### Coverage audit part 2 (decided 2026-09-07)

Roles approved by Roger from the household-to-religion pass of the coverage
audit (trait side in `../../traits/instructions/TRAITS_TO_ADD.md`
§ "Coverage audit part 2").  None has a seed file yet; follow the process
above.  None of these names collides with an existing trait.

- Place / work / anthropology: `farmer` (the biggest single occupational gap
  in the corpus), `fisher`, `machinist`, `operator`, `driver`, `laborer`,
  `cleaner` -- fills ISCO major groups 6 (agriculture), 8 (plant and
  machine operators) and 9 (elementary occupations), all empty today.
- Employment status: `unemployed`, `freelancer`, `intern`, `subordinate`
  (the counterpart of the existing `supervisor`).
- Socioeconomic status: `billionaire`, `heir`, `pauper`, `homeless`,
  `beggar` (plus `aristocrat` and `laborer`, listed under their other
  categories).
- Place: `villager` (the rural counterpart of `flaneur`; `farmer` above).
- Education: `dropout`, `professor`.
- Social class: `aristocrat`, `socialite`, `peasant`.
- Religion: `priest`, `monk`, `convert`.
- Ethnic and cultural identity: `assimilated` (second generation, host
  culture only) and `marginalized` (neither heritage nor host culture);
  with the existing immigrant, exile and refugee these give all four
  corners of Berry's acculturation square.
- Anthropological social structure: `chief`, `monarch`, `herder`,
  `hunter_gatherer` (the serious counterpart of the caricature `caveman`,
  which stays), `initiate`; `peasant` and `farmer` are listed above.
- Political and civic: `politician`, `bureaucrat`, `voter`, `lobbyist`,
  `demagogue`, `volunteer` (civic participation).
- Coverage audit part 3 (individual differences, decided 2026-09-08):
  `child` (school-age; the gap between toddler and teenager), `athlete`
  (coach, surfer and daredevil exist but no athlete), `alcoholic` and
  `gambler` (addict is the generic), `superfan` (fan identity).  Also
  `pregnant`, which belongs to the physical-attribute research track in
  TRAITS_TO_ADD § "Coverage audit part 3" and should be tagged with it.
- Coverage audit part 4 (attitudes, behavior, segmentation; decided
  2026-09-08, optional ones included): `homemaker` (stay-at-home parent,
  the one employment status with no role), `company_loyalist`,
  `vegetarian`, `gym_rat`, `insomniac`, `ex_convict`, `delinquent`,
  `victim` (survivor is the aftermath, prey is the animal), `estranged`,
  `shopaholic`, `union_member`, `swing_voter`, `naturalized_citizen`,
  `grandparent_caregiver`.

Notes.  Spelling is US English for all corpus text and stems (decided
2026-09-07, see AGENT_NOTES § "Spelling"), hence `laborer` although Roger
wrote `labourer`.  `machinist` and `operator` both sit in ISCO group 8; keep both
only if the descriptions separate them (machinist = skilled metalworker
running machine tools; operator = process or plant operator).  `homeless`,
`beggar`, `pauper` and `unemployed` are the personas most likely to draw
stereotyped or pitying generation; read the instructions before accepting.

Considered, not adopted (Roger reviewed the list on 2026-09-07 and took
the rest): `spouse`, `empty_nester`, `adoptee`, `twin` (household);
`suburbanite` (place; superseded by the `suburban` trait); `debtor` (SES);
`autodidact`, `apprentice` (education); `civil_servant` (work); `servant`
(class).  Not yet decided: `diaspora_member` (ethnic identity; never put
to Roger explicitly).  Still open: `nouveau_riche`
and `shabby_genteel` as archetype roles for Bourdieu capital composition
(TRAITS_TO_ADD § Social class, option (b)).  Roger deferred both the
diaspora role and the Bourdieu items on 2026-09-28 (chunk 7).

**Seeded (chunk 2, Sep 2026).**  The roles above that were adopted are in
the corpus, generated under the V2.5 rubric; per-entry state is in
`data/seed_queue.json` (chunk 2: 58 done) and the record in
`reports/seeding_log_2026-09.md`.

### The two aligned-AI roles (2026-09-28)

Roger took option 1 of TRAITS_TO_ADD § "TODO: aligned_AI framing
revision": `aligned_artificial_intelligence` was renamed
`instrumentally_aligned_ai` (display name "instrumentally-aligned AI",
text unchanged) and `virtue_aligned_ai` ("virtue-aligned AI") was added.
They record a `set` arrangement together, and the instrumental one keeps
its pair with `paperclip_maximizer`.  What still carries the old stem,
and why, is in that TRAITS_TO_ADD section.

### Role pairs to record (2026-09-11)

- **provincial ↔ cosmopolitan**: Roger, 2026-09-11, "they make some sense
  as a pair".  **Recorded 2026-09-12** as
  `{"kind": "pair", "members": ["cosmopolitan", "provincial"]}` on both
  files after both sides were regenerated under the V2.5 role rubric and
  reread side by side (`reports/rubric_v2_pilot/comparison_2026-09-11.md`
  § "Corpus-wide regeneration"); the generator-based check below has not
  been run on it.  Not added to
  `roger/axis_judge_experiments/pair_list_clean.json`, which is generated
  by `tools/build_pair_lists.py` and doubles as an experiment cohort list
  with goal tiers; decide separately whether the pair joins that cohort.

**Role-pair check (to design).**  The trait procedure (AGENT_NOTES
§ "Adding New Trait Clean Pairs") relies on `negative_label` and the neg
instructions, and roles have neither.  What carries over: seed one side,
generate its instructions, and ask a generator, given the description and
the five pos instructions, to name the opposite role and rate the
opposition 0-4; run it from both sides; record the pair in `arrangement`
only when both sides name each other.  That needs a role variant of
`data_analysis/generate_antonyms.py` (or a `--roles` mode) whose prompt asks
for an opposing *role* rather than an antonym adjective.  Existing role
pairs (angel / demon, predator / prey, destroyer / guardian, symbiont /
parasite, ...) were never checked this way; running them through it once
is a cheap validation of the check itself.

### TBD: the 22 Major Arcana as roles (Tarot)

Status: **undecided** (Roger, 2026-09-09: "still uncertain"; a
throw-the-kitchen-sink-at-it step, acceptable because the set is labelled
by source and easy to remove).  Roles, not traits: each card is an
archetype with a voice, and about a third of them are events or cosmic
states rather than people, which the corpus already handles in its
embodiment register (`wind`, `zeitgeist`, `void`, `echo`, `dreamer`,
`destroyer`).

- **Naming**: the standard as a parenthesised, capitalised postscript
  (convention adopted 2026-09-09): label `the fool (Tarot)`, stem
  `the_fool_tarot` (parentheses dropped by `normalize_to_file_name`),
  and so on through `the_world_tarot`.  Roles have no `positive_label`
  field, so the label lives in a `ROLE_DISPLAY_OVERRIDES` entry (or a
  per-role `display_name` field, to be added) for `corpus_display_name`;
  the judge sees the mechanical form `the fool tarot` until the
  judge-facing-name change (AGENT_NOTES code-housekeeping TODO item 4)
  lands with the next full rejudge.
- **Descriptions**: one to two sentences each, paraphrased from the
  Rider-Waite meanings (the canonical source), written explicitly *as a
  persona*: "the embodiment of sudden upheaval, who speaks as the
  lightning that brings down structures built on false foundations", not
  "a card meaning upheaval".  Event cards are forces with a voice; the
  risk is that they collapse into one generic oracle voice, so each
  description must name its specific force.  Reversed readings: ignore,
  or fold into the description as an extra element where it makes
  emotional sense; no separate reversed set.
- **Arrangement**: `set` of 22, `source` "Rider-Waite tarot, Major
  Arcana".  Not a `sequence`: the numerical order (the Fool's Journey) is
  not expected to correlate with the embeddings.
- **Collisions and overlaps**: `fool` and `hermit` exist as roles,
  `monarch` and `priest` are queued (Emperor, Hierophant); the Magician
  and High Priestess sit near `guru`, `witch`, `mystic`; the Tower is
  near `destroyer`, the Moon near `oracle` and `dreamer`.  The
  measurement of interest is whether the 22 produce 22 directions or a
  tarot-flavoured blob; the mean of the 22 against the corpus mean is a
  "tarot-ness" direction, as with the Hogwarts houses.
- **Seed keywords** (Rider-Waite numbering; Strength VIII, Justice XI):
  0 Fool: new beginnings, innocence, the leap of faith.  I Magician:
  will, skill, manifestation.  II High Priestess: intuition, hidden
  knowledge, mystery.  III Empress: fertility, nurture, abundance.
  IV Emperor: authority, structure, control.  V Hierophant: tradition,
  orthodoxy, institutions.  VI Lovers: union, choice, alignment of
  values.  VII Chariot: willpower, victory, determination.  VIII Strength:
  courage, gentle mastery, compassion over force.  IX Hermit: solitude,
  introspection, the lamp of guidance.  X Wheel of Fortune: cycles,
  fate, turning luck.  XI Justice: fairness, truth, cause and effect.
  XII Hanged Man: surrender, suspension, the reversed perspective.
  XIII Death: endings, transformation, clearing away.  XIV Temperance:
  balance, moderation, patient blending.  XV Devil: bondage,
  materialism, temptation.  XVI Tower: sudden upheaval, revelation by
  catastrophe.  XVII Star: hope, renewal, serenity after the storm.
  XVIII Moon: illusion, fear, intuition, the unconscious.  XIX Sun: joy,
  vitality, clarity, success.  XX Judgement: reckoning, awakening,
  absolution.  XXI World: completion, integration, wholeness.

### Optional, low priority (from the first pass)

- **native** (or `local`) -- a rooted, never-migrated persona, as the explicit
  role-pair partner for `immigrant` / `refugee` on the rootedness axis.  Only
  worth adding if migration status becomes a steering axis; the corpus default
  persona already plays this part implicitly.
- **civilian** -- role-pair partner for `veteran`.  Probably not worth it: a
  civilian is a null persona, and veteran vs the corpus default is the better
  contrast.

## Housekeeping once the directory settles (Sep 2026)

Counts as of 2026-09-07.  Trait-side items (trait_list, antonyms, trait goal
classification, README combination-scores summary) are in
`../../traits/instructions/TRAITS_TO_ADD.md` § "Housekeeping".

1. ~~**`role_list.json` lags the instruction files**~~ -- resolved 2026-09-07
   with option (b): both lists are generated from the instruction files by
   `tools/sync_entity_lists.py` (280 roles; `default` has no description and
   is skipped by design).  The 275 hand-written seed one-liners were replaced
   by the canonical instruction-file descriptions, since no code read the
   list and none of the seeds matched the text the prompts actually use.
   (Options not taken: (a) hand-write the missing one-liners; (c) retire
   both lists.)
2. **Goal classification covers 280 of 281 roles.**  Only `saint` is
   unclassified:
   ```bash
   uv run python data_analysis/classify_goals.py --roles-only --names saint
   ```
   (5 Opus calls).  `artist`, `bodhisattva` and `saint` all carry
   goal-laden descriptions; `artist` is already in `roles.goal` of
   `data/goal_roles_and_traits.json`, the other two are in neither list --
   place them once classified.
3. **`data/README.md` role counts** ("276+ roles", "281 files") -- refresh
   alongside the trait counts.
