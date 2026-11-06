---
paths:
- pipeline/3_judge.py
- assistant_axis/judge*.py
- assistant_axis/steering_judges.py
- assistant_axis/rubric_equivalence.py
- data_analysis/*.py
- results_analysis/axis_judge_correlation.py
- results_analysis/infer_axis_description.py
- results_analysis/standardize_axis_spec.py
- data/roles/instructions/**
- data/traits/instructions/**
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: judging

**When:** editing judge prompts, rubrics, parse-rate handling, or any LLM judge call site.  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=judging`; edit there, then run `uv run python tools/sync_agent_notes.py`.

### Judge prompts: reason BEFORE score (mandatory)

In any LLM judge prompt that asks for **both** a reasoning/explanation
field and a numeric (or categorical) decision, the reasoning **must
come first**.  Anything else makes the model commit to the score
before it has thought about it, turning the reason field into
post-hoc rationalization rather than a chain-of-thought aid.

This matters because we judge with **non-reasoning models** by default
(GPT-4.1-mini, Claude Haiku 4.5, Claude Sonnet's non-thinking mode).
Without explicit "think-on-the-page" framing, these models will skip
deliberation entirely.

**Two correct patterns** in this codebase:

1. **JSON output** — put the reason key first in both the schema
   description and any example, AND tell the model so explicitly:

   ```
   Respond with a JSON object only, no other text.  Reason about the
   response first, then commit to the score:
   {"reason": "<one short sentence>", "score": <0-3 integer>}
   ```

   Even though `json.loads()` is order-agnostic for parsing, LLMs
   tend to write fields in the order they're declared in the
   schema/example — so order matters for *generation*, not for
   parsing.  The explicit "reason first, then score" sentence is
   belt-and-suspenders against training-data conventions that lean
   toward `{"score": ..., "reason": ...}`.

2. **Free-form prefix + structured score** — used by
   ``results_analysis/axis_judge_correlation.py`` (the desc/instr/
   response judges):

   ```
   First, briefly reason about where this {entity} falls on the axis
   (2-3 sentences).  Then on a new line, write exactly:
   SCORE: <integer from -3 to +3>
   ```

   The parser looks for the `SCORE: <int>` token, not the first
   integer in the reply.

**The parser side** matters too: ``parse_judge_score`` in
``assistant_axis/judge.py`` looks for ``SCORE: <int>`` first, then
falls back to the LAST (not first) integer in the valid range.  A
naive "first integer wins" parser is hostile to reasoning text — a
response like "0 if no, 3 if full ... I rate this 2" would be parsed
as 0 by a first-integer parser, producing the wrong score whenever
the rubric definitions are echoed in the reasoning.

**Forbidden anti-pattern**:

```
# BAD -- forces commit-without-thinking
"Respond with a number between 0 and 3. Don't say anything else, just the number."

# BAD -- score field first means reason is rationalization
{"score": <0-3>, "reason": "..."}
```

**Where this is enforced**:

- ``assistant_axis/steering_judges.py`` — coherence, RP, effect rubrics
- ``pipeline/3_judge.py`` — TRAIT_EVAL_TEMPLATE, COMBINED_EVAL_TEMPLATE,
  and the per-role ``eval_prompt`` field of every
  ``data/roles/instructions/*.json``
- ``data/traits/instructions/*.json`` — per-trait ``eval_prompt`` field
- ``results_analysis/axis_judge_correlation.py`` — RUBRIC_STATIC,
  RUBRIC_RESPONSE_BATCH (uses the SCORE: marker pattern)
- ``data_analysis/{score_combinations,generate_antonyms,classify_goals}.py``
  — JSON output, reasoning-first
- ``data_analysis/regenerate_{role,trait}_instructions.py`` — the
  templates that emit ``eval_prompt`` text into role/trait JSON
  files; if you ever modify these, double-check the embedded
  example template inside the Christina instruction-generation
  prompt also follows the reasoning-first pattern (the LLM mimics
  what you show it as an example).

**When changing a judge rubric**: bump the corresponding
``*_RUBRIC_VERSION`` constant.  These are stamped into each judged
record for traceability; old records remain valid but are now
distinguishable from records produced under the new rubric.

**Per-entity drift-on-resume check (May 2026)**.  Both static-mode
([`score_static_mode`](results_analysis/axis_judge_correlation.py))
and response-mode
([`score_responses_mode`](results_analysis/axis_judge_correlation.py))
cache loaders run
[`_check_rubric_version_on_resume`](results_analysis/axis_judge_correlation.py)
immediately after `_load_json_or_empty`.  The check walks **every
entry** in the cache (not just the cohort as a whole), classifies it
as **current / equivalent / drifted**, drops only the drifted ones in
memory, and logs a ``WARNING`` listing what was dropped.  The on-disk
file is left intact and is overwritten on the next save with
fresh-rubric data for the rejudged entries.  Caches whose envelope
lacks both a cohort-level ``rubric_version`` stamp and a per-entity
stamp map are passed through unchanged — those are covered by the
older `schema_version` / fingerprint checks.

The effective rubric-version for each cache entry is resolved by
peeking at, in priority order:

1. ``_provenance.notes.per_entity_rubric_versions[<entity_id_or_name>]``
   — the per-entity stamp written next to the cache by the producer.
   Static-mode keys are disambiguated ``"name|R"`` / ``"name|T"``;
   response-mode keys are bare names (the cohort directory pins the
   kind).
2. The cohort-level ``rubric_version`` from the producer-script
   InputSpec's ``extras`` (read by
   [`_peek_rubric_version`](results_analysis/axis_judge_correlation.py)
   / [`peek_rubric_version`](assistant_axis/judge_loaders.py)).

This per-entity granularity matters because rubric bumps can affect
different entities differently.  The v2 → v3 bump (May 2026) changed
prompt rendering from file-form (``aligned_artificial_intelligence``)
to display-form (``aligned artificial intelligence``); response-mode
prompts only carry pole names through that pipeline (so only axes
with multi-word poles see any change), while static-mode prompts carry
the *entity* name too (so any entity with underscores in its name
sees a changed prompt under v3).  Cohort-level "drop and rejudge
everything" would either be wasteful (rejudging the byte-identical
entries) or unsound (claiming axis-wide equivalence when the change
is per-entity).

**Rubric-equivalence registry**
([`assistant_axis.rubric_equivalence`](assistant_axis/rubric_equivalence.py),
backing file `rubric_equivalences.yaml` at repo root):
declarative ``(from_rubric, to_rubric, modes, axes|except_axes,
entity_ids|except_entity_ids)`` edges that let the producer
*skip* a rejudge for cells where the maintainer has verified the
prompt is byte-identical between the cached and current rubric.
Mirrors the existing
[`script_equivalence`](assistant_axis/script_equivalence.py)
pattern; queried via `is_equivalent(from, to, *, axis, mode,
entity_id)` which does scoped BFS over edges (reflexive, transitive,
asymmetric).  Declare new edges via
[`tools/mark_rubric_equivalent.py`](tools/mark_rubric_equivalent.py)
(supports `--symmetric` for the common round-trippable case;
`--list` and `--check` for inspection).  Initial seed entry: the v2
→ v3 response-mode equivalence for the 11 single-pole axes.

**Strict mode**.  Pass `--strict_rubric_version` to make undeclared
drift a hard `SystemExit(2)` instead of silent-drop-and-rejudge.  The
error message lists the affected entities and prints the exact CLI
needed to declare an equivalence (or instructs the operator to drop
the flag to accept the rejudge).  Off by default because long-running
judging sweeps shouldn't abort mid-flight on a rubric bump; use when
you want a run to halt rather than spend money rejudging entries you
weren't planning to.

**Judge-model drift on resume (Sep 2026)**.  Judge caches are keyed
by judge *family* (`sonnet/`, `gpt/`, `haiku_responses_traits_b7_t3/`),
not by model id, and the resume path never compared models -- so when
`claude-sonnet-4-20250514` was retired (Sep 2026) and the Anthropic
default moved to `claude-sonnet-4-6`, a plain resume would have appended
4.6 scores into cohorts that were, per their own provenance, mostly
`claude-sonnet-4-5` (52 of 56 recorded Sonnet caches; 90 more recorded
no model at all).  [`_check_judge_model_on_resume`](results_analysis/axis_judge_correlation.py)
now runs right after the rubric-version check in both
`score_static_mode` and `score_responses_mode`:

* recorded `judge_model` == current: keep;
* recorded and different: **drop the whole file** (rejudged as a
  single-model cohort) with a WARNING naming both models;
  `--strict_judge_model` aborts instead, and the message shows the
  `--judge_model <old>` invocation that would extend the old cohort;
* unrecorded (legacy envelope): keep with a WARNING by default, since
  100+ GPT and Haiku legacy caches exist and those judges did not
  change; `--drop_unrecorded_judge_model` rejudges them whole (use it
  for the Sonnet migration), and combined with strict it aborts.

Cost of the Sonnet migration: ~1,160 static calls per axis (61 axes on
disk), roughly $4-6 per axis at Sonnet 4.6 rates, incurred lazily the
first time each axis is touched.  Each such run still goes through the
expensive-operations confirmation.  The refusal allowlist
(`data/judge_refusal_allowlist.json`) is keyed by exact model id and
currently lists only `claude-sonnet-4-5`; the first 4.6 run over an axis
with `virus|R` will report it as an *unexpected* gap if the refusal
recurs -- add a `claude-sonnet-4-6` entry then, not pre-emptively.
Tests: `test_axis_judge_correlation_phase4.py::TestCheckJudgeModelOnResume`.

**Consumer side**.  Callers reading caches can audit drift without
the producer's drop-or-abort policy via
[`assistant_axis.judge_loaders.rubric_version_report`](assistant_axis/judge_loaders.py),
which returns a `RubricVersionReport(path, cohort_rubric,
current_rubric, n_total, n_current, n_equivalent, drifted, ...)`.
The report is informational (truthy = no drift); it doesn't mutate
the cache or raise.  Use for "is this analysis I'm about to publish
based on stale-rubric data anywhere?" audits.

**Whole-tree audit**.
[`tools/audit_rubric_versions.py`](tools/audit_rubric_versions.py)
walks the `roger/` (or any) judge-cache tree and classifies every
``scores_*.json`` into ``current`` / ``equivalent`` / ``drifted`` /
``legacy`` against the current ``RUBRIC_VERSION`` plus the
equivalence registry; reports cohort-stamp distribution + per-entity
entry tallies + per-cache drift breakdowns.  Default output is
markdown; ``--format flat`` is one line per cache for grep / shell
follow-up.  ``--status drifted`` shows only the actionable cells.
``--current v4`` lets you ask "what would a v3→v4 bump invalidate?"
without actually bumping the constant.  Sibling to
``tools/audit_caches.py`` / ``tools/audit_pngs.py`` (provenance-
fingerprint and PNG-Inputs-chunk audits respectively).

**Steering-judge rubrics deliberately out of scope** (May 2026).
The COHERENCE_/RP_/EFFECT_RUBRIC_VERSION integers in
[`assistant_axis/steering_judges.py`](assistant_axis/steering_judges.py)
are stamped per-record but don't yet participate in the per-entity
drift check or the equivalence registry.  Currently low-risk
because those rubrics are run from the runpod-side
``steering/`` pipeline (fresh runs, no resume-against-edited-rubric
pattern); see the TODO block above
``COHERENCE_RUBRIC_VERSION = 4`` for the work needed to bring them
into the same scheme if/when that assumption breaks.

### Judge parse-rate alerting (mandatory)

Every script that calls an LLM judge and parses structured output
**must** emit an end-of-run parse-rate summary, and that summary must
fire a loud warning if the OK rate drops below **99%** (i.e. ≥1%
UNPARSEABLE / API-failed / empty).  Why 99% and not something looser:

- At small N a 5% fail rate is hard to notice in the log noise but
  is enough to materially distort downstream aggregates.
- For ensemble judging, even 1% silently biases the mean toward
  whichever judge in the pair survives — an effect that doesn't
  show up in mean-of-means but does show up in per-(kind, model)
  scatter plots.  This rule was adopted May 2026 after the
  GPT-4.1-mini batched-effect run was found to have a 17–31%
  UNPARSEABLE rate that nobody noticed for several days.

**Default stance: <99% is a bug, not a budget.**  The norm for
this project is that *every* judge run should clear 99% parseable.
If a run trips the loud warning, that is a signal to **stop, catch
it, investigate, and fix** — typical root causes are prompt-format
drift (e.g. the model started emitting code fences), max-token
truncation, schema fragility (top-level lists, nested JSON), or a
specific model that's miscalibrated for the task (cf. GPT-4.1-mini
on batched effect judging in May 2026).  We may, on a *case-by-case*
basis, decide to accept a lower rate when no fix is available — but
that should be an explicit, documented decision (e.g. a comment on
the call site or a note in the run's results README), not silence.
The default reaction to ``*** HIGH FAIL RATE ***`` is to treat it
as a real problem that blocks downstream analysis until resolved.

**Use the central helper** at ``assistant_axis/judge.warn_if_low_parse_rate``:

```python
from assistant_axis.judge import warn_if_low_parse_rate

warn_if_low_parse_rate(
    label="my_script:claude-haiku-4-5-20251001",
    n_ok=n_parsed,        # parseable, in-range, non-empty
    n_total=n_attempted,  # all judge calls actually made
    logger_obj=logger,    # so the line shows up in your log file
)
```

Behaviour:

- 100% OK → single INFO line: ``[label] parse rate: N/N OK``
- ≥99% OK with some failures → INFO line with the percentage
- <99% OK → WARNING line prefixed with ``*** HIGH FAIL RATE ***``
- ``n_total == 0`` → no-op (don't pollute logs for empty runs)

The threshold lives in ``PARSE_RATE_LOUD_THRESHOLD`` if you ever
need to override it (you almost certainly don't — match the
project default unless you have a specific reason).

**Where it's wired today** (grep ``warn_if_low_parse_rate`` to find
new sites):

- ``pipeline/3_judge.py`` — aggregate across all entities
- ``results_analysis/axis_judge_correlation.py`` — per (mode,
  provider, model) at end of static and response sub-runs
- ``data_analysis/classify_goals.py`` — post-retry success rate
- ``data_analysis/score_combinations.py`` — post-retry success rate
- ``data_analysis/generate_antonyms.py`` — counts non-ERROR sentinels
- ``assistant_axis/steering_judges.py`` — uses an in-class tally
  (``RealJudgeDispatcher.judge_outcome_summary``) with the same
  threshold and ``*** HIGH FAIL RATE ***`` marker; promoted to
  WARNING level at ``shutdown()``.

**Single-shot scripts are exempt**: utilities that judge one axis
or one item at a time (e.g. ``results_analysis/infer_axis_description.py``,
``results_analysis/standardize_axis_spec.py``) don't get a parse-rate
tracker — N=1 makes the rate meaningless, and they already raise on
unrecoverable parse failure rather than silently dropping records.

### Bias-reduction rubric bumps (May 2026)

All four steering rubrics were tightened to remove **judge priors** —
fields whose values shouldn't be in the prompt because they leak the
experimenter's *expected* answer.  Versions stamped into every judged
record (see "**When changing a judge rubric**" callout above):

| rubric | old | new | what changed |
|---|---|---|---|
| `COHERENCE_RUBRIC` | v4 | **v5** | drop raw `sign` and numeric `strength`; pre-resolve sign → `steered toward {pole_label}` sentence; omit sentence entirely for `sign=0` baselines |
| `RP_STEERING_RUBRIC` | v2 | **v3** | gain matching `steered toward {pole_label}` clause as final line of axis-context block; same omit-for-baseline behavior |
| `RP_STEERING_RUBRIC` | v3 | **v4** | refine 0-3 anchors: level-0 explicitly requires *both* AI-self-identification AND refusal to engage ("and also refuses"); level-3 explicitly allows refusal *if* it stays in (steered) persona for in-persona reasons.  Unblocks legitimate principled-refusal cases for strongly-steered personas that v3 would mis-score as level-1. |
| `EFFECT_POLE_BATCH_RUBRIC` | v3 | **v4** | drop numeric `strength`; pre-resolve sign → `steered toward {pole}` (direction *kept* because trait is fixed by `pole=`, so direction is contextual rather than score-shaping) |
| `EFFECT_BIDIR_BATCH_RUBRIC` | v3 | **v4** | drop **both** numeric `strength` AND the direction-of-this-batch claim; judge must infer pull direction from response content alone |
| `EFFECT_*_BATCH_RUBRIC` | v4 | **v5** | wording refinements to scoring anchors and instructions: "same as baseline" / "same direction as baseline" → "equivalent to baseline" (the old phrasing admitted an absent-steering reading -- baseline has *some* direction along the axis even unsteered); insert "observable" into the direction/magnitude language so the judge grounds in response content rather than inferring from prior; add the previously-missing "do not penalize for incoherence" reminder to the unidirectional rubric (gap from v4). |
| `EFFECT_BIDIR_BATCH_RUBRIC` | v5 | **v6** | drop the `+` prefix from the score scale legend (`+3`/`+2`/`+1` → `3`/`2`/`1`) AND add an explicit `do NOT prefix positive scores with "+"` instruction in the JSON-output spec.  Diagnosed 2026-05-15 on architect_ecocentric_v2 + chef_helpful_v2: GPT-4.1-mini (and occasionally claude-haiku) faithfully reproduced the rubric's `-3..+3` wording as JSON values like `"score": +1`, which fails `json.loads` because RFC 8259 disallows `+` as a numeric prefix.  Combined with a defensive parser repair (`_repair_json_blob` in `assistant_axis/judge.py`) that strips the `+` from number-after-colon contexts on a fallback pass, this took the per-cell UNPARSEABLE rate from ~25-50% down to 0%.  Pole rubric was not bumped (its 0..3 scale has no `+`-prefix issue). |

The asymmetry between the two effect rubrics is the principled bit:
the bidirectional rubric scores a *signed* magnitude (-3..+3) on every
item, requiring the judge to commit to a sign AND a magnitude.  Telling
it "direction=+1, strength=8" gives a strong two-dimensional prior.
The unidirectional rubric scores a *fixed* trait on 0..3 — the judge
has no sign decision to make, only a magnitude decision, so the
strength prior is the one to suppress and direction can stay as inert
context.  Full rationale in the v4-builder docstring on
`build_effect_bidir_batch_prompt`.

Filled-out examples of all four rubrics with their new wording live in
[`reports/steering_judging_prompt_example.md`](reports/steering_judging_prompt_example.md).

Since records have always been stamped with the rubric version they
were produced under, no migration is needed — old records remain
valid; they're just flagged as v4 (effect) / v4 (coherence) / v2
(RP) and can be re-run against the new rubric if a particular
analysis depends on the bias correction.  No current downstream
consumer treats rubric-version as a filter, so this is purely an
audit-trail change for now.
