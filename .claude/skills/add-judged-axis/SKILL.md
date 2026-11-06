---
name: add-judged-axis
description: Checklist for incorporating a newly response-judged axis into the analysis pipeline
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
### Incorporating new response-judged axes (2026-05-22 checklist)

Before running the formal re-tuning checklist above (sweep 2 + sweep
3), the new-axis set has to be made discoverable to the loader and
downstream tooling.  After a Phase-1/2-style response-judging
campaign delivers GPT B=7 + Haiku B=7 t3 caches for N new axes:

1. **Update `pair_list_responses.json`** to include the new axes
   (use the rich pair_list_di.json schema with `pair_type`).  This
   is the single source of truth for "which axes have response
   judging"; almost everything else that uses the response cohort
   set reads from it (or imports from a script that does).

2. **Run `tools/audit_caches.py` + `tools/audit_pngs.py`** (slow but
   exhaustive) to see what's now stale.  These provide the
   authoritative answer for "what JSON/PNG outputs depend on
   data I changed".

3. **Source-code scans** the audit tools can't see:

   - `DEFAULT_AXES` in
     `results_analysis/gpt_anthropic_response_weight_sweep.py` —
     since 2026-05-22 this is loaded from
     `pair_list_responses.json` automatically; no manual edit
     needed.  Other scripts that import `DEFAULT_AXES` (e.g.
     `response_di_weight_sweep.py`) pick up the new list
     transitively.
   - `GPT_DIR_TEMPLATE`, `HAIKU_DIR_TEMPLATE` (hardcoded
     ``_b10`` / ``_b10_q9`` literals) in legacy sweep scripts:
     replaced by the canonical
     `assistant_axis.judge_loaders.load_response_scores()` so the
     per-entity B=7 → B=10 fallback works for axes that don't yet
     have B=7.  If you find another script still using these
     hardcoded templates, retrofit it the same way before running
     it on a mixed-B cohort.
   - `_DEFAULT_PREFER_B` in `assistant_axis/judge_loaders.py` — the
     Haiku default is `(7, 10)`, NOT `(7,)`.  If you ever feel
     tempted to reduce it to `(7,)`, remember the 2026-05-21
     mistake: the original 12 axes' bulk coverage lives in
     `_b10_q9`, and `(7,)`-only would invisibly drop ~95% of those
     axes' entity coverage.

4. **Re-run cached `*.json` artefacts that the audit flagged stale.**
   Common ones:
   - `roger/axis_judge_experiments/rho_by_layer_L.json` (and the
     companion `_K.json` if you use it) — `rho_by_layer.py` reads
     `pair_list_responses.json`, so a regen now reflects the new
     N response axes.  Expect ~30–60 min wall time for the full
     `--layers 0..63 --slots 0 3 6 7` sweep.

5. **Then run the re-tuning checklist** (sweeps 2 and 3) per the
   README to confirm or update the canonical weight constants on
   the new N-axis cohort.

**Why this section exists.**  The re-tuning checklist alone is
necessary but not sufficient: it doesn't enumerate the cohort-
discovery plumbing that has to be right BEFORE the sweeps will
even see the new axes.  Added 2026-05-22 after a 10-axis
incorporation pass tripped over the hardcoded `DEFAULT_AXES`, the
`_b10_q9`-only legacy templates in `response_di_weight_sweep.py`,
and a stale `_DEFAULT_PREFER_B[("haiku", *)] = (7,)` that dropped
b10 fallback.

```python
from assistant_axis.judge_score_combine import (
    DEFAULT_DI_WEIGHTS,
    DEFAULT_GPT_SONNET_DI_WEIGHT,
    DEFAULT_GPT_HAIKU_Q9_WEIGHT,
    DEFAULT_RESPONSE_DI_WEIGHT,
    combine_desc_inst_two_judges, add_di_weights_arg, parse_di_weights_arg,
)

# 1. desc + inst within one judge (or 4-way GPT+Sonnet):
#    The 4-way uses DEFAULT_GPT_SONNET_DI_WEIGHT (= 0.625 since 2026-05-12)
#    by default; pass gpt_sonnet_weight=... for ablations.
di = combine_desc_inst_two_judges(g_d, g_i, s_d, s_i)

# 2. within-response ensemble:
response = {
    n: DEFAULT_GPT_HAIKU_Q9_WEIGHT * gpt[n]
       + (1 - DEFAULT_GPT_HAIKU_Q9_WEIGHT) * haiku_q9[n]
    for n in set(gpt) & set(haiku_q9)
}

# 3. final blend:
final = {
    n: DEFAULT_RESPONSE_DI_WEIGHT * response[n]
       + (1 - DEFAULT_RESPONSE_DI_WEIGHT) * di[n]
    for n in set(response) & set(di)
}

# CLI integration for the desc/inst tiebreak knob:
add_di_weights_arg(parser)
args = parser.parse_args()
weights = parse_di_weights_arg(args.di_weights)  # → tuple, e.g. (0.499, 0.501)
```

**Anti-pattern**: don't compute means or blends inline:

```python
# BAD -- defeats the convention; can't ablate; out of date if a default changes:
scores = {n: 0.5 * resp[n] + 0.5 * di[n] for n in common}

# GOOD -- canonical, ablatable, future-proof:
scores = {
    n: DEFAULT_RESPONSE_DI_WEIGHT * resp[n]
       + (1 - DEFAULT_RESPONSE_DI_WEIGHT) * di[n]
    for n in common
}
```

Re-tuning is expected as new judging data accumulates; the README's
"Re-tuning checklist" walks the per-sweep regen, the comparison
plots to update, and the snapshot-before-invalidate ritual.

Consumers today: `results_analysis/{rho_by_slot_and_K, rho_by_layer,
whitening_k_sweep, gpt_sonnet_weight_sweep, gpt_anthropic_response_weight_sweep,
response_di_weight_sweep, rubric_v1_v2_compare, judge_ensemble_rho_curve}.py`.
