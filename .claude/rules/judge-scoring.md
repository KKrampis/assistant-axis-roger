---
paths:
- assistant_axis/judge_score_combine.py
- assistant_axis/judge_batch.py
- assistant_axis/judge_loaders.py
- results_analysis/**/*.py
- tools/defer_rejudge.py
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: judge-scoring

**When:** combining judge scores, loading response scores, choosing the response-judging batch size, or subsampling questions.  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=judge-scoring`; edit there, then run `uv run python tools/sync_agent_notes.py`.

### Combining judge scores: use the canonical helpers and constants

Three families of empirically-tuned mixing ratios live in
[`assistant_axis/judge_score_combine.py`](assistant_axis/judge_score_combine.py)
and govern how scripts reduce multiple judge × mode scores to a
single per-entity scalar.  Always import the constants instead of
re-declaring magic floats so a future re-tuning propagates with one
edit.

| Constant | Default | Weights |
|---|---|---|
| `DEFAULT_DI_WEIGHTS` | `(0.499, 0.501)` | desc / inst within one judge |
| `DEFAULT_GPT_SONNET_DI_WEIGHT` | `0.625` | GPT / Sonnet within the desc+inst ensemble (per mode) (was `0.50` until 2026-05-12; retuned to the soft_shear=3 discrete grid peak w=0.625 on the 35-axis slot 6 sweep — see `judge_score_combine.py` "Selection history") |
| `DEFAULT_GPT_HAIKU_Q9_WEIGHT` | `0.525` | GPT / Haiku in the response-mode ensemble (was `0.60` until 2026-05-11; then `0.41` 2026-05-11→05-12 on the raw-projection v2 sweep; `0.625` 2026-05-12→05-22 on the canonical-whitening soft_shear=3 v2 sweep at 12 axes; **retuned to `0.525` on 2026-05-22 22-axis cohort** — discrete grid peak w=0.525 on the expanded set; see `judge_score_combine.py` "Selection history") |
| `DEFAULT_RESPONSE_DI_WEIGHT` | `0.80` | response / desc+inst in the final per-entity score |

> **RESOLVED 2026-05-14 — di-extension reconfirm done; all constants hold**
>
> Original task: reconfirm `DEFAULT_GPT_SONNET_DI_WEIGHT`,
> `DEFAULT_RESPONSE_DI_WEIGHT`, and `DEFAULT_DI_WEIGHTS` after the
> di-extension batch (`pair_list_di.json` 35 → 61: 22 new trait pairs
> + 4 new role pairs; ~$80 of additional GPT+Sonnet desc+inst
> judging).
>
> Outcomes:
>
> * `DEFAULT_GPT_SONNET_DI_WEIGHT = 0.625` **holds**.  New 61-axis
>   sweep discrete-grid peak: w=0.800, ρ=+0.66476; current default at
>   w=0.625 gives ρ=+0.66252.  Δρ=+0.00224 (third-decimal noise);
>   top-10 grid points span w=0.75–0.975 within Δρ=0.0004 -- a wide
>   plateau the default sits on the edge of.  No retune.
> * `DEFAULT_RESPONSE_DI_WEIGHT = 0.80` **holds**.  Discrete-grid peak
>   w=0.825, ρ=+0.7774 vs default at w=0.8 (one grid step apart at
>   step 0.025).  No retune.
> * `DEFAULT_DI_WEIGHTS = (0.499, 0.501)` (inst_tie) **holds**.  61-axis
>   ablation: inst_tie ρ=+0.66200, equal ρ=+0.66182, desc_tie
>   ρ=+0.66088; monotonic same direction as the 33-axis original
>   sweep, Δρ in third decimal.
> * `DEFAULT_GPT_HAIKU_Q9_WEIGHT = 0.625` -- not affected by this
>   batch (response-side cohort unchanged); no rerun needed.
>
> Also rerun: `rho_by_slot_and_K`, `rho_by_slot_and_L`,
> `rho_by_layer_K`, `rho_by_layer_L` -- all four plots regenerated
> overnight 2026-05-13 → 2026-05-14, sitting at
> [`roger/axis_judge_experiments/rho_by_*.png`](roger/axis_judge_experiments/).
> No visible regime shifts; the per-axis L-selector investigation
> remains a separate follow-on and is unaffected by the reconfirm.
>
> Reconfirm ran with **two simultaneous correctness improvements**:
> the combiner one-side-missing fallback (see "Known permanent gap:
> virus|R" entry) and 30/30 backfilled virus|R Sonnet scores via the
> allowlist-driven Haiku fallback.  Net effect on ρ: Δρ ≈ +0.0001
> per axis (one extra entity in a ~390-entity cohort).

Full empirical derivation, per-axis discussion, tuning history, and
**the re-tuning checklist** for when new judging data lands live at
[`results_analysis/README.md` → "Convention: tuned mixing ratios for
judge ensembles"](results_analysis/README.md#convention-tuned-mixing-ratios-for-judge-ensembles).
Read that section *before* changing any of the three constants.

### Response judging batch size (`RESPONSE_BATCH_SIZE`)

The response-mode judging pipeline partitions an entity's `score==3`
responses into roughly equal-sized batches before sending each batch
to the LLM judge (see `plan_response_batches` in
[`results_analysis/axis_judge_correlation.py`](results_analysis/axis_judge_correlation.py)).
The batch size is a noise-vs-cost trade-off knob; once chosen, it
influences every downstream rho/correlation analysis that consumes
`scores_responses.json`, so all consumers must agree on a single
value.

The canonical default lives in
[`assistant_axis/judge_batch.py`](assistant_axis/judge_batch.py) as
`RESPONSE_BATCH_SIZE` (currently `7`, bumped from `10` on 2026-05-11
to align with "Change D" of the trait/role disambiguation plan —
see the `Value history` block in that file's module docstring for
the rationale).  Importers:

* `axis_judge_correlation.py` — argparse default for
  `--response_target_batch_size`.
* `assistant_axis.steering_judges` — `DEFAULT_TARGET_BATCH_SIZE`
  is derived from `RESPONSE_BATCH_SIZE` so steering effect-judge
  scores stay apples-to-apples with axis-judge scores at the same
  batch size.  This pinning briefly fell silently out of step
  during May 2026's Phase 5c (which used `--response_target_batch_size 7`
  via explicit override rather than bumping the constant); restored
  on 2026-05-11.
* `whitening_k_sweep.py`, `rho_by_layer.py`,
  `optimal_axis_for_judge.py` — use the
  `response_subdir(judge, mode)` helper to construct
  `gpt_responses_traits_b{N}/` paths.

The `_b{N}` suffix in judging-output directory names
(`gpt_responses_traits_b7`, `haiku_responses_roles_b7_t3`, ...) IS
the canonical convention; consumers use `response_subdir()` rather
than hard-coding `"_b7"` or `"_b10"`.  Existing `_b10` caches
remain readable via `response_subdir(..., batch_size=10)` (the
cross-batch comparison plots that hit `_b5` / `_b7` / `_b15`
archives do the same), but the default value the callers reach
for is now `b7`.

**Bumping the value** invalidates every existing
`*_b{old}/scores_responses.json` cache for response judging
purposes (desc+inst caches are unaffected — their rubric is
batch-size-agnostic).  After bumping you'll need to:

1. Re-run `axis_judge_correlation.py --score_responses` for every
   axis × judge cell you care about (LLM cost).
2. Re-run every downstream consumer (`whitening_k_sweep`,
   `rho_by_layer`, `optimal_axis_for_judge`,
   `batch_size_rho_curve`, ...).
3. Either delete the old `_b{old}` caches or defer them via
   `tools/defer_rejudge.py`.

Pre-Phase-6 caches without a `_b{N}` suffix (bare
`gpt_responses_traits/`) are NOT covered by `response_subdir()` —
they're treated as legacy v1 archives.  Consumers that still need
to read them do so explicitly via the unsuffixed path (rare; mostly
the cross-rubric comparison plots that hit `__rubric_v1.json`
snapshots).

### Tiered question subsampling for response judging (default May 2026)

> **2026-05-21 status update — tiered t3 is canonical for new judging;
> uniform q9 is OBSOLETE for writes, retained as the read-side
> fallback for axes/entities not yet rejudged at B=7.**
>
> Empirically, the tiered cohort auto-escalates (tier 1 → tier 2 → tier 3
> per-entity) until every persona has enough graded items, which
> typically yields coverage equivalent to the old uniform `_b10_q9`
> cohort. Running both is roughly twice the cost for little marginal
> coverage. **Use `_b7_t3` (tiered) for all new Haiku judging.**
>
> **Read-side fallback (CORRECTED 2026-05-22).**  The default Haiku
> `prefer_b` in `assistant_axis.judge_loaders` is `(7, 10)` — B=7
> preferred, B=10 legacy fallback per entity.  An earlier (2026-05-21)
> change reduced it to `(7,)` only, but that broke load on the original
> 12 axes whose bulk entity coverage still lives in `_b10_q9` (only
> the ~14–29 collision-disambiguation entities per axis got the
> `_b7_t3` surgical rejudge).  The new 10 Phase-1/2 axes have their
> full coverage in `_b7_t3` and resolve entirely there; the original
> 12 axes fall back to `_b10_q9` for everything outside the surgical
> set.  Both regimes work transparently under the default `(7, 10)`.
>
> The **fresh** t3 cost is ~10× higher than the historical surgical-
> refresh number documented elsewhere (because a from-scratch run pays
> the full tier-1/2/3 escalation cost for every entity, rather than
> topping up an already-q9-covered axis): plan on ~$27/cohort,
> ~$54/axis (roles+traits) for fresh Haiku t3 runs.

`results_analysis.axis_judge_correlation --score_responses` now defaults
to **per-entity tiered subsampling** instead of "judge every response".
Three tiers, indexed by the dense response-pipeline `q_idx % M` (default
`M=3`):

| Tier | Filter | Fraction of canonical |
|---|---|---|
| 1 | `q_idx % 3 == 0`     | 1/3 |
| 2 | `q_idx % 3 in {0,1}` | 2/3 |
| 3 | (no filter)          | full |

Each entity falls back to a less aggressive tier only when RP filtering
has depleted it past these thresholds (with `N_default = 500` items in
the default persona's `responses/default.jsonl`):

* tier-1 count ≥ `N_default / 6` (≈ 83) → use tier 1
* else tier-2 count ≥ `2 * N_default / 9` (≈ 111) → use tier 2
* else → use tier 3

So full-roster entities pay ~1/3 of the cost of "no subsample" while
RP-heavy entities still get a stable sample (typical max ~170 items;
worst case ~2× the t1→t2 threshold ≈ 165).

**Knobs**:

* `--no_subsample` -- restore the pre-tiered default (judge everything).
* `--question_subsample_modulo N` (`>0`) -- **OBSOLETE as of 2026-05-21**
  for new Haiku runs; kept only for historic q9 reproduction. Uniform
  mode filters every entity identically by `orig_id % N == 0` (no
  RP-aware escalation). Sonnet's frozen `_b10_q9` cohort still uses
  this; do not write new q9 Haiku cohorts.
* `--tiered_modulo_per_chunk M` -- tier chunking granularity (default 3).

**Why q_idx, not orig_id**: the response pipeline uses `orig_id = 3 *
q_idx` (reduce=3 from the 300-question canonical), so `orig_id % 3 == 0`
is vacuously true and would not subsample. `q_idx` is dense (0..99) and
gives clean 1/3 / 2/3 / all chunks regardless of the canonical stride.

`AGENT_NOTES.md` and `correlations.json` files from earlier (uniform-q9
or no-subsample) runs are NOT backfilled; their results stay valid and
comparable within their own cohort.

---

### `assistant_axis.judge_loaders.load_response_scores`

The canonical reader for response-mode scores.  Default Haiku
`prefer_b = (7, 10)` — reads `_b7_t3` where available and falls back
to `_b10_q9` per-entity for axes not yet rejudged at B=7.  Uses
**conditional provenance**: only registers a cohort file as a
dependency if it actually contributed at least one entity to the
returned result.  See the module docstring for the suffix conventions
(`(no suffix)` = full volume, `_q<N>` = uniform mod-N legacy, `_t<M>`
= tiered).

---
