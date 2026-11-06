# Data Analysis

Scripts for generating, classifying, and inspecting the role/trait data in
[`data/`](../data/). Everything in this directory is new — Christina Lu's
published repository did not include her data generation tooling.

## Scripts

### `regenerate_trait_instructions.py`

Generates pos/neg instruction pairs, questions, and eval prompts for traits
via the Anthropic API. Recreates the functionality described in Christina Lu's
paper (Appendixes B). With the appropriate flags (`--style Christina`,
temperature 1.0, no thinking), it uses her original prompts and parameters
and reproduces her results closely. The default `--style Roger` is a fork
with improved neg instructions (more example pairs for different trait types,
optional antonym injection).

```bash
uv run python data_analysis/regenerate_trait_instructions.py --traits stoic --force
uv run python data_analysis/regenerate_trait_instructions.py --all --dry-run
uv run python data_analysis/regenerate_trait_instructions.py --traits stoic --style Christina --force
```

Token usage (Sep 2026): every non-dry run logs a `[usage]` line and merges
its cost into the cumulative `data/traits/regeneration_usage.json`
(`--usage-json PATH` to redirect); the role script and `generate_antonyms.py`
do the same with `data/roles/regeneration_usage.json` and
`data/traits/antonym_check_usage.json`.  Roughly $0.02-0.03 per entity on
Sonnet 4.6.  Note that `--roles`/`--traits` take separate arguments: in zsh,
`$(cat list.txt)` is passed as one word, so use `$(cat list.txt | tr ' ' '\n')`
or spell the names out.

### `regenerate_role_instructions.py`

Generates instruction variants and questions for roles. Same relationship to
Christina's paper (Appendix A) as the trait script — a recreation of her missing tooling,
with flags to reproduce her original parameters.

```bash
uv run python data_analysis/regenerate_role_instructions.py --roles pirate oracle --force
uv run python data_analysis/regenerate_role_instructions.py --all --dry-run
uv run python data_analysis/regenerate_role_instructions.py --roles pirate --style RogerV2 --force --show-prompt
```

Prompt styles (Sep 2026): `--style RogerV2` (default since 2026-09-12)
adds the voice and anti-softening rules from the September 2026 voice
audit (second-person, the role's own vocabulary and particulars, a 15-25
word target, no case-worker or writer's register, no whitewashing of bad
roles, an inside view for non-verbal roles, particulars that vary across
the five instructions, scenario questions) with examples that obey them;
`--style Roger` is the May 2026 production template, identical to
Christina's for roles, kept for rollback and comparison.  Roger adopted V2
for roles on 2026-09-12 after the pilot in `reports/rubric_v2_pilot/`
(every role file except `default.json` carries it since the corpus-wide
regeneration later that day; subject to rollback once embeddings have
been extracted, for which the V1-rubric files are kept there and at commit
`93a8554`, the last before the corpus check-in of 2026-09-28).  The trait
script has no V2 yet.  Every
regenerated role or trait file now carries a `generator` field (script,
style, a short hash of the template text, model, temperature, thinking
budget, date) so the two instruction populations stay distinguishable;
the hash changes whenever the template text is edited.

### `generate_antonyms.py`

Determines the `negative_label` for traits by feeding their pos/neg
instructions to Claude and asking it to name the opposite pole. Outputs
antonym scores (0-4) and reasoning. Supports `--traits` to scope to specific
traits. Primarily used for creating/checking `negative_label` values. Particulalry useful for creating/confirming "clean pairs" of antonyms were B is the correct `negative_label` for A and vice versa, so comfirming theit context and scope match well: start with `negative_label = "non-{positive_lable}"`, confirm you get the expected atonyms bidirectionally, then update the `negative_label` values to the antonyms.

```bash
uv run python data_analysis/generate_antonyms.py
uv run python data_analysis/generate_antonyms.py --traits obedient rebellious
```

### `seed_entities.py`

Drives the corpus-expansion queue `data/seed_queue.json` (one entry per
candidate trait or role from `TRAITS_TO_ADD.md` / `ROLES_TO_ADD.md`, with a
status lifecycle `candidate -> ready -> seeded -> generated -> checked ->
paired | done`).  `write` turns entries whose description is final into seed
JSONs (traits with `negative_label = non-<label>`, roles with a singleton
`arrangement`), `generate` runs the two regenerate scripts on them (with a
cost estimate and a refusal over $20 without `--confirm-expensive`), `check`
runs `generate_antonyms.py` and classifies each answer against existing and
queued stems (nice / mismatch / nearly_nice / nasty / open, the decision
table in `AGENT_NOTES.md` § "Corpus expansion policy"), `rename` applies the RO action
(rename the existing trait to the check's word if free, regenerate, re-check
both sides), `pair` records a
confirmed clean pair on both files and regenerates the new side's neg
clause, and `status` / `report` summarise.  Every subcommand takes
`--dry-run`.  The description-writing rules the seeds must follow are in
`AGENT_NOTES.md` § "Description-writing rules for new seeds".

```bash
uv run python data_analysis/seed_entities.py status --chunk 1 --list
uv run python data_analysis/seed_entities.py write --chunk 1 --sub-chunk "Tier D" --dry-run
uv run python data_analysis/seed_entities.py generate --stems rationalizing intellectually_honest
uv run python data_analysis/seed_entities.py check --stems rationalizing
uv run python data_analysis/seed_entities.py pair --a rationalizing --b intellectually_honest
```

### `classify_goals.py`

Classifies each role/trait instruction by whether it implies alignment-relevant
goals (score 0-2). Results are aggregated per role/trait and saved to
`output/`. Supports `--names`, `--roles-only`, `--traits-only`, and `--force`. Note that this use Opus, so costs > $100 to run.

```bash
uv run python data_analysis/classify_goals.py --dry-run
uv run python data_analysis/classify_goals.py --names obedient compassionate --traits-only --force
uv run python data_analysis/classify_goals.py  # full corpus
```

### `score_combinations.py`

Scores all role+trait instruction combinations for incongruity using Claude
Sonnet. For each combination, sends all index-matched pos instruction pairs
in a single API call and gets a 0-3 score per pair with reasoning. Output
goes to `data/combination_scores.json`.

Parameters mirror Step 1's `--goal_count` and `--non_goal_count` (defaulting
to 40 each). Supports `--dry_run`, `--batch_size` (instruction pairs per
call, default all), and resume via incremental saves.

```bash
uv run python data_analysis/score_combinations.py --dry_run
uv run python data_analysis/score_combinations.py --goal_count 2 --non_goal_count 2  # test
uv run python data_analysis/score_combinations.py  # full 40x40 + 40x40 = 3200 combos
```

### `sample_trait_responses.py`

Diagnostic tool for eyeballing how a model responds to trait instructions.
Sends pos and neg system prompts with sampled questions, prints responses
side by side.

```bash
uv run python data_analysis/sample_trait_responses.py stoic
uv run python data_analysis/sample_trait_responses.py stoic --n-questions 3 --pairs 0 2 4
```

## Generator model

All five API scripts here default to `claude-sonnet-4-6` (switched 2026-09-07
when `claude-sonnet-4-20250514` was retired and started returning 404).  The
corpus is therefore mixed: any trait whose instructions were regenerated on
or after 2026-09-07 (tracked in `data/traits/instructions/TRAITS_TO_ADD.md`
§ "Housekeeping" and `AGENT_NOTES.md` § "TODO: regenerate activation/vector
data") comes from Sonnet 4.6; everything else still comes from Sonnet 4.  A same-prompt
comparison on three control traits showed Sonnet 4.6 rewrites every
instruction and question (no exact matches), with longer, more prescriptive
"always/never" phrasing.  `classify_goals.py` and the `score_combinations.py`
rescore model use `claude-opus-4-6`, which still resolves.

Known stale tests (tracked in `AGENT_NOTES.md` § "TODO: code housekeeping
(Sep 2026)"): three tests in
`results_analysis/tests/test_infer_axis_description.py` expect the streaming
helper to return a bare string where it now returns `(text, usage)`; they
fail at git HEAD independent of the model change.  (The fourth,
`test_regenerate_role_instructions.py::TestBuildEvalPrompt::test_uses_0_to_3_scale`,
was updated 2026-09-11 to assert the reason-first eval-prompt ending.)

## Arrangement tools (Sep 2026)

The `arrangement` field on every role and trait JSON records which set of
same-type entities the file belongs to and its shape (`pair`, `triangle`,
`square`, `N-orthoplex`, `sequence`, ...); rules in `AGENT_NOTES.md`
§ "The `arrangement` field", implementation in
`assistant_axis/arrangements.py`.

- `check_arrangements.py` -- validates the whole corpus (kind vocabulary,
  member counts, reciprocity between members, agreement with
  `negative_label` pairs, tree links).  Exit 1 on any inconsistency;
  `--list-unclassified` prints the files with no field.  Run it after any
  edit to labels or arrangements; the test suite also runs it against the
  checked-in corpus.
- `backfill_arrangements.py` -- the one-off that wrote the initial values
  (clean pairs, the two triangles, the moral-circle sequence, singletons
  for untouched `non-X` placeholders, role pairs from
  `pair_list_clean.json`).  Dry run by default, `--apply` to write,
  `--overwrite` to replace existing fields.  Kept for re-runs after bulk
  additions.

## Output

```
output/
├── goal_classifications.json       # Aggregated per-(name, source, polarity)
├── goal_classifications_raw.json   # Per-instruction classifications (Opus)
├── goal_classifications_sonnet.json     # Aggregated (earlier Sonnet run)
└── goal_classifications_raw_sonnet.json # Per-instruction (earlier Sonnet run)
```

The `_sonnet` files are from an earlier classification run using Sonnet: it's not up to this task.
The primary files (without suffix) use Opus.

## Tests

```bash
uv run pytest data_analysis/tests/
```

Unit tests for the instruction regeneration scripts (prompt construction,
JSON parsing/repair, argument validation).
