---
paths:
- assistant_axis/judge_refusal_fallback.py
- assistant_axis/judge_score_combine.py
- tools/fill_judge_refusal_gaps.py
- results_analysis/refill_judge_gaps.py
- tools/pick_base_personas.py
- data/judge_refusal_allowlist.json
- results_analysis/axis_judge_correlation.py
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: judge-refusal-gaps

**When:** touching judge refusal fallbacks, the refusal allowlist, gap refills, or base-persona picking.  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=judge-refusal-gaps`; edit there, then run `uv run python tools/sync_agent_notes.py`.

### Known permanent gap: `virus|R` on Sonnet instructions mode

`results_analysis/refill_judge_gaps.py --scan` flags ~26 axes as having
a single missing entity in their Sonnet `scores_instructions.json` (and
4 of those 26 also miss it from `scores_descriptions.json`).  The
missing entity is always `virus|R`.

**Cause** (audit, May 2026): Sonnet's safety filter refuses to score
imperative second-person prompts of the form *"You are a virus that
spreads rapidly by taking over existing systems and using their
resources for reproduction and propagation."* (the v1 instruction
phrasing in [`data/roles/instructions/virus.json`](data/roles/instructions/virus.json)).
The third-person *description* phrasing — "A virus is a simple
replicator that..." — *usually* gets through but occasionally also
refuses (4/26 axes).  GPT-4.1-mini scores both modes for `virus|R`
without issue.

**Decision**: accept as a permanent gap, **do not refill**.
Refilling would re-issue the same refused prompts and burn API spend on
calls that are certain to fail.  The impact on downstream ρ is
negligible — 1 missing entity out of ~390 corpus entities, only on
Sonnet's side, with GPT's score still present (the ensemble combiner
treats the missing Sonnet value as "absent" and falls back to GPT
alone for that entity on that axis).

**Detection**: the parse-rate alerter (above) fires on each affected
run; the warning text identifies `virus|R` by name.  As long as that's
the *only* entity flagged, the run is healthy.  Any *new* entity
appearing in a Sonnet refusal pattern is a signal to investigate.

**Haiku does accept the prompt** (probe May 2026,
`claude-haiku-4-5-20251001`): both desc and inst modes return
``SCORE: -3`` on the alignment axis, with reasoning that interprets
``virus`` *biologically* ("cellular machinery", "the organism it
infects", "survival or flourishing").  Sonnet's filter is presumably
tripped by the malware-tactics flavor of the corpus' question bank
("infiltrate", "exploit weaknesses", "bypass security measures",
"total domination") rather than the role name itself; Haiku's
smaller-model safety post-training is less aggressive, and it lands
on the bio interpretation instead.

**If we ever want to close this gap**: two options.
* **Option A (cheap, dishonest)**: patch
  ``scores_{descriptions,instructions}.json`` for the 26 affected
  Sonnet caches with one-time Haiku scores for ``virus|R``,
  flagged in ``notes`` with the provenance.  Cost: ~$0.03.  Risk:
  mixes judge providers within a single ``scores_*.json`` cache
  silently; future audit tools may need a notes-aware reader.
* **Option B (clean)**: add a judge-fallback chain in
  ``combine_desc_inst_two_judges`` mirroring the
  response-mode GPT+Haiku-fallback pattern, so any entity missing
  from the primary second-judge cache (Sonnet) gets filled by the
  fallback (Haiku) at combine time.  Real refactor but generalises
  the fix beyond ``virus|R``.

**Decided 2026-05-13: BOTH Option B (combiner fallback) AND a
narrower Option B' (allowlisted post-hoc backfill) implemented**.

Option B' rationale (Roger: "fall back to Haiku on a refusal, but
with an allowlist of roles allowed to fall back, currently just
virus|R, and a loud warning if some other role/trait hits a refusal
and isn't on the allowlist"):

* New file [`data/judge_refusal_allowlist.json`](data/judge_refusal_allowlist.json)
  -- registry of ``(judge_model, entity_id)`` pairs known to be
  systematic refusals, with the fallback judge to use for backfill.
  Currently: just ``(claude-sonnet-4-5, virus|R)`` → fallback to
  ``claude-haiku-4-5-20251001``.

* New tool [`tools/fill_judge_refusal_gaps.py`](tools/fill_judge_refusal_gaps.py)
  -- scans axis-judge dirs for ``gaps.json`` entries, partitions into
  allowlisted vs unexpected, fills allowlisted gaps by calling the
  configured fallback judge with the same prompt the primary judge
  refused, and patches the score into the primary cache's
  ``result`` dict with a ``_provenance.notes.fallback_fills``
  annotation so the substitution is auditable per-entity.  Unexpected
  refusals trigger a loud banner warning.  Idempotent (re-running
  is a no-op once gaps are filled).

* End-of-mode warning in
  [`results_analysis/axis_judge_correlation.py`](results_analysis/axis_judge_correlation.py)
  (``score_static_mode``) consults the allowlist and emits SEPARATE
  log lines for allowlisted (softer INFO log) vs unexpected (LOUD
  banner) gaps -- so the producer-side run is honest about what's
  known-and-fillable vs novel-and-needs-investigation.

* Initial backfill (2026-05-13, ~$0.04 in Haiku calls): 30/30
  virus|R gaps filled across 26 Sonnet caches.  Haiku scores
  cluster sensibly (mostly ±3 on alignment/harm/opacity-related
  axes; consistent with the role's semantics).  Per-entity
  audit trail in each affected cache's
  ``_provenance.notes.fallback_fills.virus|R``.

Option B (general combiner fallback) rationale below.  The combiner
semantics in
[`assistant_axis/judge_score_combine.py`](assistant_axis/judge_score_combine.py)
are now *one-side-missing fallback* across the board:

* `combine_desc_inst_one_judge(desc, inst)` — if only one mode is
  numeric for an entity, return that mode at 100% (renormalize
  the weights over the present modes only).
* `combine_desc_inst_two_judges(g_d, g_i, s_d, s_i)` — composes
  the above per-judge; if only one judge produces a DI score,
  return that judge alone at 100%.
* New helper `blend_two_with_fallback(a, b, weight_a)` — generic
  two-source blend with the same fallback semantics; intended for
  the response-ensemble (GPT-resp + Haiku-resp) and final-blend
  (response + DI) cases, which currently use inline
  ``set(a) & set(b)`` filters in consumer scripts.  Migrating those
  consumers is a follow-on; the immediate virus|R rescue flows
  through the DI combiner.

A name is dropped from the output iff it has zero numeric coverage
across all inputs — that's the only path to exclusion under the
new semantics.  Previously a single missing/None input on a 4-way
combine dropped the whole entity.

Practical effect for `virus|R`: it's now scored on every axis that
has *any* numeric coverage (which is all of them, since GPT scored
the role cleanly).  ρ on the 26 affected axes recomputes with the
full ~390-entity cohort instead of 389/390.  Δρ per axis is ε;
the principle (use available signal rather than discard it) is
what matters.

Test coverage: see
[`assistant_axis/tests/test_judge_score_combine.py`](assistant_axis/tests/test_judge_score_combine.py),
new tests:

* `test_combine_partial_coverage_keeps_entity_with_fallback`
* `test_combine_non_numeric_treated_as_missing_with_fallback`
* `test_combine_drops_only_when_all_inputs_missing`
* `test_combine_one_judge_fallback_on_missing_mode`
* `test_blend_two_with_fallback`

(The previous `test_combine_skips_missing_entities` and
`test_combine_skips_non_numeric` encoded the OLD strict-AND
semantics — replaced, not just patched, since the new contract is
the opposite intent.)

Also relevant: rewriting
`data/roles/instructions/virus.json`'s `instruction` field to use
non-imperative phrasing would likely make Sonnet accept it, but
changes the experimental setup (other roles use imperative form);
not worth the inconsistency for a single role.  Virus is already
excluded from steering base-persona candidates
(``NON_HUMAN_OR_NONVERBAL_EXCLUDE`` in
[`tools/pick_base_personas.py`](tools/pick_base_personas.py)) for
the same human-verbal-base-role reason; the corpus inclusion is
preserved because virus-as-a-corpus-entity is a useful *extreme
point* for cosine geometry even if it's not a base-persona candidate.
