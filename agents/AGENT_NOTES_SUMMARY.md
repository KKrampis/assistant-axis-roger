# Agent Collaboration Guide

This document describes Roger's working style, communication preferences, and collaboration patterns for AI agents working on this project.

**Purpose:** Help future AI agents collaborate more effectively by understanding expectations upfront.


---

## Summary

This document is the operating manual for AI coding agents (Cursor-based) working with Roger on **assistant-axis**, a research codebase that computes an "Assistant Axis" — a direction in an LLM's activation space that measures how "Assistant-like" a persona is — and uses it to judge, steer, and analyze model behavior (Qwen-3-32B is the primary subject model). The guide blends four categories of material: **hard behavioral rules** the agent must never violate, **Roger's working style**, **engineering conventions** the codebase has accumulated (mostly around LLM-judge reliability and data provenance), and a **running log of project-specific decisions, cost models, and open TODOs**. It is explicitly a living document, last substantively organized in May 2026, and reads accordingly — dated entries, superseded values, and "holds/retuned" verdicts are common.

**Four hard rules sit above everything else.** (1) Any operation projected to cost more than $20 in API spend or more than 13 GPU-hours must have its parameters explicitly confirmed by Roger first, quoting each non-default flag and its rationale — this rule exists because an agent once burned ~$364 on a mis-parameterized judging run. (2) File access is scoped strictly to `/Users/roger/Documents/GitHub/`; anything outside requires an explicit, per-task grant. (3) Every file the agent references in chat (plots especially) must be a clickable, workspace-relative markdown link (`./path/to/file`), never a bare or backticked path. (4) Any LLM call site that runs in a batch or loop must log token usage and cost via a shared `MultiModelUsage`/`BudgetTracker` accumulator into a `usage.json` sidecar, so spend is auditable after the fact rather than reconstructed from invoices.

**Working style.** Roger is technical, prefers concise and direct responses, wants trade-offs laid out so he can decide quickly rather than being asked for permission on clear paths, and consistently prefers agents verify actual on-disk/installed state over assuming it. He distinguishes Ask mode (read-only, propose changes) from Agent mode (implement directly), and favors incremental, conversational iteration with verification at each step over large upfront specs. Plans that modify code must include a testing step unless he explicitly waives it or testing is genuinely impractical (with his sign-off). When a test's expectations and the code's actual behavior disagree, the agent should stop and reason carefully rather than reflexively editing either one — fixing the code is the default, adjusting test expectations is rare and must be flagged prominently, and asking Roger is always safe.

**Plot and documentation discipline.** After generating any plot, the agent must read it back as an image and visually check for text collisions (a detailed failure-mode checklist covers title-block stacking and colorbar/label collisions, with a working matplotlib recipe). Every plot must carry embedded PNG-metadata provenance (title, author, software invocation, git SHA, and for ad-hoc scripts, the full source code) so any figure can be traced back to exactly the code and data that produced it.

**Data and judge provenance is the codebase's deepest piece of infrastructure.** A layered system — dataset manifests, an `InputSpec`/`json_metadata`/`png_metadata` provenance envelope, a `--cache-policy` (strict/warn/rebuild/off) for readers, and audit tools (`audit_pngs.py`, `audit_caches.py`, `audit_deferrals.py`) — lets any cached JSON or plot be classified as current, stale, legacy, frozen, or deliberately deferred, and lets staleness propagate transitively through dependency chains. A `script_equivalence` registry lets maintainers declare that a code edit didn't change output, and a `deferred_rejudges.yaml` registry (with a typed category schema) lets them declare "known stale, don't re-flag" for specific artifacts, without either escape hatch silently masking real drift.

**LLM-judge engineering** is a recurring theme: judge prompts must always ask the model to reason before committing to a score (never score-first), every batch-judging script must alert loudly if its structured-output parse rate drops below 99%, and a documented, allowlisted fallback mechanism (Haiku standing in for Sonnet) handles a known permanent refusal gap on one adversarial-sounding role (`virus|R`). Score-combination weights across judges and modes (description/instruction/response, GPT/Sonnet/Haiku) are centralized as named constants with a versioned tuning history rather than inline magic numbers, and a checklist governs how newly-judged axes get incorporated into the cohort. A detailed, empirically re-derived cost model (per-batch token counts, per-provider pricing, a batch-size cost/quality Pareto curve) underpins the project's May-2026 shift from batch size 10 to 7, and a real incident — a stale local CLI default that silently overrode the canonical batch size during steering sweeps — motivates a general "grep for hardcoded defaults after changing a canonical constant" caution.

**Steering experiments** get their own architecture: steering always operates in raw, unwhitened activation space (whitening/soft-shear is analysis-only, tuned separately to K=2 / L=3), and uses a two-tier live judging pipeline (blocking coherence checks gate more expensive persona and effect judging). A May-2026 rewrite replaced a unidirectional strength scan with a bidirectional one, added per-cell start-strength heuristics derived from a 310-cell audit, fixed a label-order bias in effect judging via a swap-averaged estimator, and added supporting tooling (multi-config sweep queuing, a Google Sheets exporter, response-curve plots, multi-GPU baseline-race protection).

**Entity naming** conventions (a `name|R`/`name|T` disambiguator for the ~9 words that are both a trait and a role, and a strict file-name-vs-display-name split so underscores never leak into LLM prompts) prevent silent data-merging bugs. The document closes with general LLM-collaboration patterns (structured pros/cons responses, ambiguity handling, prompt-writing advice), a data-pipeline procedure for adding new antonym trait pairs (with April 2026 validation results and one 3-trait "conformity triangle" exception), an open PC-round-trip research synthesis, a snapshot-before-invalidate discipline for expensive caches, and a closing "Quick Reference" of golden rules: check actual state, be concise, treat documentation as infrastructure, and discuss trade-offs before executing.

---

## Table of Contents

- [Expensive Operations — Confirm Parameters First (HARD RULE)](#expensive-operations-confirm-parameters-first-hard-rule)
- [File Access Boundary (HARD RULE)](#file-access-boundary-hard-rule)
- [Hotlink every file you mention to Roger (HARD RULE)](#hotlink-every-file-you-mention-to-roger-hard-rule)
- [Token usage logging is mandatory on batched LLM call sites (HARD RULE)](#token-usage-logging-is-mandatory-on-batched-llm-call-sites-hard-rule)
- [Communication Style](#communication-style)
  - [Concise and Technical](#concise-and-technical)
  - [Decision-Focused Workflow](#decision-focused-workflow)
  - [Verify-Then-Trust Approach](#verify-then-trust-approach)
  - [Hotlinking image/plot files in chat replies (mandatory)](#hotlinking-imageplot-files-in-chat-replies-mandatory)
  - [Plot visual verification (mandatory after any plot generation)](#plot-visual-verification-mandatory-after-any-plot-generation)
    - [Verification workflow](#verification-workflow)
    - [Failure-mode checklist (in order of typical recurrence)](#failure-mode-checklist-in-order-of-typical-recurrence)
    - [Working recipe — heatmap + colorbar + 2-line title](#working-recipe-heatmap-colorbar-2-line-title)
  - [Ask Mode vs Agent Mode](#ask-mode-vs-agent-mode)
  - [Iteration Style](#iteration-style)
  - [Understanding the Landscape First](#understanding-the-landscape-first)
- [Plan Requirements](#plan-requirements)
  - [Testing Must Be Included](#testing-must-be-included)
- [Technical Approach](#technical-approach)
  - [Check Actual State First](#check-actual-state-first)
  - [Documentation as Infrastructure](#documentation-as-infrastructure)
  - [Maintainability Focus](#maintainability-focus)
  - [Markdown Backtick Conventions](#markdown-backtick-conventions)
  - [Judge prompts: reason BEFORE score (mandatory)](#judge-prompts-reason-before-score-mandatory)
  - [Known permanent gap: `virus|R` on Sonnet instructions mode](#known-permanent-gap-virusr-on-sonnet-instructions-mode)
  - [Judge parse-rate alerting (mandatory)](#judge-parse-rate-alerting-mandatory)
  - [NFS-safe file I/O (mandatory for `/workspace` reads and writes)](#nfs-safe-file-io-mandatory-for-workspace-reads-and-writes)
  - [Plot Provenance Metadata (mandatory for every plot)](#plot-provenance-metadata-mandatory-for-every-plot)
    - [Tracked-script plots](#tracked-script-plots)
    - [Ad-hoc exploration plots (write the script to /tmp first)](#ad-hoc-exploration-plots-write-the-script-to-tmp-first)
    - [Conventions summary](#conventions-summary)
    - [Caveats](#caveats)
  - [End-to-End Data Provenance (writers, readers, audits)](#end-to-end-data-provenance-writers-readers-audits)
    - [Timestamp convention: UTC ISO-8601 everywhere](#timestamp-convention-utc-iso-8601-everywhere)
    - [Writer pattern (mandatory for new analysis scripts)](#writer-pattern-mandatory-for-new-analysis-scripts)
    - [Reader pattern (`--cache-policy`)](#reader-pattern---cache-policy)
    - [Reader+registrar pattern (`load_and_register`) — preferred](#readerregistrar-pattern-load_and_register-preferred)
      - [Retrofit catalog (Phase 6c, 2026-05-09)](#retrofit-catalog-phase-6c-2026-05-09)
    - [Audit tools](#audit-tools)
    - [When to regenerate manifests](#when-to-regenerate-manifests)
    - [Migrated scripts (as of May 2026)](#migrated-scripts-as-of-may-2026)
  - [Judge-step provenance (rubrics, equivalence, deferral, recovery)](#judge-step-provenance-rubrics-equivalence-deferral-recovery)
    - [1. Producer dependencies (per output mode)](#1-producer-dependencies-per-output-mode)
    - [2. Per-axis consumer granularity](#2-per-axis-consumer-granularity)
    - [3. Script equivalence (harmless code edits)](#3-script-equivalence-harmless-code-edits)
    - [4. Deferred rejudging (intentional staleness)](#4-deferred-rejudging-intentional-staleness)
      - [Deferral categories (schema v2, May 2026)](#deferral-categories-schema-v2-may-2026)
      - [`tools/audit_deferrals.py` (registry coherence)](#toolsaudit_deferralspy-registry-coherence)
      - [Apply deferrals BEFORE propagating transitive staleness](#apply-deferrals-before-propagating-transitive-staleness)
    - [5. Recovery: `tools/diff_against_recorded.py`](#5-recovery-toolsdiff_against_recordedpy)
  - [Deferred future work — Phase 7 (subtree content hashing)](#deferred-future-work-phase-7-subtree-content-hashing)
  - [Combining judge scores: use the canonical helpers and constants](#combining-judge-scores-use-the-canonical-helpers-and-constants)
  - [Incorporating new response-judged axes (2026-05-22 checklist)](#incorporating-new-response-judged-axes-2026-05-22-checklist)
  - [Steering judges (Phase-2 architecture)](#steering-judges-phase-2-architecture)
  - [Bias-reduction rubric bumps (May 2026)](#bias-reduction-rubric-bumps-may-2026)
  - [Bidirectional steering scan (May 2026, default)](#bidirectional-steering-scan-may-2026-default)
  - [Per-cell start-strength heuristic (May 2026)](#per-cell-start-strength-heuristic-may-2026)
  - [Steering question selection (May 2026)](#steering-question-selection-may-2026)
  - [Sweep log location (May 2026)](#sweep-log-location-may-2026)
  - [Multi-config queue runner (May 2026)](#multi-config-queue-runner-may-2026)
  - [Multi-GPU safety: baselines sentinel (May 2026)](#multi-gpu-safety-baselines-sentinel-may-2026)
  - [Exporting steering experiments to Google Sheets (May 2026)](#exporting-steering-experiments-to-google-sheets-may-2026)
    - [One-time Google Cloud setup](#one-time-google-cloud-setup)
    - [Usage](#usage)
    - [Rerun semantics](#rerun-semantics)
    - [Layout](#layout)
    - [Files](#files)
  - [Per-cell response curves vs steps-to-incoherence (May 2026)](#per-cell-response-curves-vs-steps-to-incoherence-may-2026)
  - [Swap-averaged effect judging (rubric v7, May 2026)](#swap-averaged-effect-judging-rubric-v7-may-2026)
    - [Schema (records.jsonl)](#schema-recordsjsonl)
    - [Filling in older experiments](#filling-in-older-experiments)
    - [Cost trade-off in the live pipeline](#cost-trade-off-in-the-live-pipeline)
    - [Analysis surfaces (no code changes needed)](#analysis-surfaces-no-code-changes-needed)
  - [Whitening / soft-shear defaults](#whitening-soft-shear-defaults)
  - [Axis-geometry tool: `axis_cosine_seriation.py`](#axis-geometry-tool-axis_cosine_seriationpy)
  - [Response judging batch size (`RESPONSE_BATCH_SIZE`)](#response-judging-batch-size-response_batch_size)
  - [Tiered question subsampling for response judging (default May 2026)](#tiered-question-subsampling-for-response-judging-default-may-2026)
- [Trait/role name collisions and the `name|R` / `name|T` convention (May 2026)](#traitrole-name-collisions-and-the-namer-namet-convention-may-2026)
  - [Disambiguator: `assistant_axis.entity_id`](#disambiguator-assistant_axisentity_id)
  - [Display-side rule (plots, legends, console output)](#display-side-rule-plots-legends-console-output)
  - [File-name vs display-name convention (underscores ↔ spaces)](#file-name-vs-display-name-convention-underscores-spaces)
  - [Common-pitfall callout (read this before merging trait + role data)](#common-pitfall-callout-read-this-before-merging-trait-role-data)
  - [Producer-side / consumer-side audit (May 2026)](#producer-side-consumer-side-audit-may-2026)
  - [`assistant_axis.judge_loaders.load_response_scores`](#assistant_axisjudge_loadersload_response_scores)
- [Judging cost model (project-wide reference)](#judging-cost-model-project-wide-reference)
  - [Per-batch token model (B=10)](#per-batch-token-model-b10)
  - [Per-axis cost (both cohorts combined, full volume)](#per-axis-cost-both-cohorts-combined-full-volume)
  - [B-curve takeaway (motivates the May 2026 B=10 → B=7 default switch)](#b-curve-takeaway-motivates-the-may-2026-b10-b7-default-switch)
  - [Steering effect-judge B post-mortem (2026-05-21)](#steering-effect-judge-b-post-mortem-2026-05-21)
  - [Static-mode cost model (descriptions + instructions)](#static-mode-cost-model-descriptions-instructions)
  - [Budget cap recommendation (safety net)](#budget-cap-recommendation-safety-net)
  - [Provenance: capture `usage` from every API response](#provenance-capture-usage-from-every-api-response)
- [Response Patterns](#response-patterns)
  - [What Works Well](#what-works-well)
  - [What to Avoid](#what-to-avoid)
  - [When Test Expectations Don't Match Code Behavior](#when-test-expectations-dont-match-code-behavior)
- [Prompt Engineering](#prompt-engineering)
  - [Writing Prompts for LLMs](#writing-prompts-for-llms)
  - [Multi-Pass Review for Prompts and Text](#multi-pass-review-for-prompts-and-text)
- [Documentation Context](#documentation-context)
  - [Multiple Audience Levels](#multiple-audience-levels)
- [Session Initialization](#session-initialization)
  - [Recommended Starting Pattern](#recommended-starting-pattern)
  - [Current canonical local data dir (May 2026)](#current-canonical-local-data-dir-may-2026)
  - [Mid-Session Patterns](#mid-session-patterns)
- [Examples from Past Interactions](#examples-from-past-interactions)
  - [Good Interaction Pattern](#good-interaction-pattern)
  - [Could Be Better](#could-be-better)
- [Adding New Trait Clean Pairs](#adding-new-trait-clean-pairs)
  - [Process](#process)
  - [Why non-X first?](#why-non-x-first)
  - [`data/goal_roles_and_traits.json` structure](#datagoal_roles_and_traitsjson-structure)
  - [Clean pair validation results (April 2026)](#clean-pair-validation-results-april-2026)
  - [TODO: regenerate activation/vector data after April–May 2026 trait edits](#todo-regenerate-activationvector-data-after-aprilmay-2026-trait-edits)
  - [TODO: variant-specific K grids in the PC round-trip experiment (May 2026)](#todo-variant-specific-k-grids-in-the-pc-round-trip-experiment-may-2026)
  - [PC round-trip experiment: findings synthesis (May 2026)](#pc-round-trip-experiment-findings-synthesis-may-2026)
  - [Snapshot-before-invalidate principle (May 2026 lesson)](#snapshot-before-invalidate-principle-may-2026-lesson)
    - [Off-tree archives (when in-tree is too big)](#off-tree-archives-when-in-tree-is-too-big)
  - [Combined response generation (pipeline)](#combined-response-generation-pipeline)
- [Updates and Evolution](#updates-and-evolution)
- [Quick Reference](#quick-reference)

---

## Expensive Operations — Confirm Parameters First (HARD RULE)

**Whenever you are about to start an operation that will incur any
of the following, STOP, double-check every parameter, and SURFACE
any uncertainty or any not-currently-standard setting to Roger
before launching:**

- More than **$20** in cumulative judging / API spend (OpenAI,
  Anthropic, etc.), across any number of axes or runs in that
  invocation.
- More than **13 GPU-hours** of total compute on the RunPod (or any
  remote GPU host).  At Qwen-3-32B's typical 4-GPU sweep pace this is
  roughly 3-4 hours of wallclock; bigger compute-on-Mac jobs count
  toward the spirit of this rule too.
- Any other action with a real-world cost greater than **$20** —
  storage, network egress, third-party APIs, etc.

**What "surface" means**:

1. Re-read the *current* defaults in the relevant CLI flag's
   `--help` text and in this AGENT_NOTES (look for any "(May 2026)"
   / "(updated 2026-…)" / "**holds**" / "(default …)" markers
   indicating where the canonical setting lives).  Don't copy from
   older orchestration scripts or older runs without first
   checking they're still canonical.  Defaults change.
2. Quote each non-default or potentially-questionable parameter
   back to Roger in your reply, in the form
   "`--<flag> <value>` (default: `<default>`; chose `<value>` because
   `<reason>`)".
3. **Wait for explicit confirmation** before launching.  "Looks
   good", "go ahead", "fine", or any equivalent counts; ambiguity
   ("OK") does not.  Recent prior approval of a similar but
   *not-the-same* run does not transfer — re-confirm for each
   expensive launch.

**Why this rule exists**: on 2026-05-20 the agent kicked off a
~$985 response-judging run for 10 axes at `B=10` because it copied
the `scripts/rejudge_after_rubric_v2.sh` recipe without checking
the AGENT_NOTES "Judging cost model" section that flagged the May
2026 default switch to **B=7**.  ~$364 was sunk before the mistake
was caught.  The cost difference between B=10 and B=7 is small
(~$30 across 10 axes), but the principle generalises: for anything
expensive, the few minutes of confirmation are worth far more than
the value of the API tokens.

**Recent / explicit requests still apply**: if Roger has just said
"do the canonical run at B=10" in the immediately-prior message,
proceed without a second confirmation cycle.  This rule only
triggers when the parameter choice is ambiguous, inherited from
older context, or differs from the documented current default.

---

## File Access Boundary (HARD RULE)

**You may read, write, search, or otherwise touch files only within
`/Users/roger/Documents/GitHub/`.**  There should never be a need to
go outside this tree for any task in this project.  If you believe
you need to, **stop and ask Roger first** with a specific request
naming the exact path(s) you want access to and why; proceed only
after he grants explicit permission for that specific access.

This applies to all forms of access: `Read`, `Write`, `StrReplace`,
`Grep`, `Glob`, `find`, `cat`, `ls`, `rg`, shell redirection, and any
remote tools (mirroring or fetching files from this Mac to other
machines counts as access from this side).  Read-only access is
NOT exempt — `/Users/roger/Documents/` outside `GitHub/` and
everything in `/Users/roger/` (other than `GitHub/`) is off-limits
without explicit per-task permission.

Remote machines (e.g. RunPod) are a separate scope; this rule
governs the local filesystem only.

The Cursor terminals folder (`/Users/roger/.cursor/projects/.../terminals/`)
and agent-tools folder are tool infrastructure that Cursor itself
populates, and are expected reads — those are exempt by design (and
are written by Cursor, not by you).

---

## Hotlink every file you mention to Roger (HARD RULE)

**Every time you reference a file (plot, JSON, source, log, config,
notebook, etc.) in a chat reply to Roger, format it as a markdown
link with a workspace-relative path prefixed with `./`.** No
exceptions: bare backticked paths do NOT hotlink in Cursor, and
Roger has to manually copy them — which he's reminded the agent
about repeatedly.  Default assumption: if Roger reads a file path
in a chat reply, he wants to be able to click it.

**`.png` files are the highest-priority case.** If you mention a
plot — generated, regenerated, looked-up, or referenced by name —
**always** wrap it as `[filename.png](./path/to/filename.png)`.
Roger almost invariably wants to open and look at a plot you
mention; making him copy-paste the path is a friction every single
time.  No exception is small enough to skip this — even a single
sentence like "I wrote the new plot to `foo.png`" must hotlink
`foo.png`.

**DO** (every file mention, including summary tables and bullet lists):

```markdown
[gpt_haiku_q9_response_weight_sweep_slot6.png](./roger/axis_judge_experiments/gpt_haiku_q9_response_weight_sweep_slot6.png)
- updated [`pair_list_responses.json`](./roger/axis_judge_experiments/pair_list_responses.json) 12 → 22 axes
- the script lives at [`results_analysis/response_di_weight_sweep.py`](./results_analysis/response_di_weight_sweep.py)
```

**DON'T** (these are the failure modes the agent keeps committing):

- `roger/axis_judge_experiments/foo.png` (bare path — not clickable)
- `` `roger/axis_judge_experiments/foo.png` `` (backticked — not
  clickable either)
- `file:///Users/roger/Documents/GitHub/assistant-axis/roger/...`
  (absolute file URIs render but are non-portable)
- `[foo.png](/Users/roger/Documents/GitHub/assistant-axis/roger/.../foo.png)`
  (absolute paths bake the username into the chat transcript; use
  workspace-relative `./roger/...` instead)
- A bare filename like `pair_list_responses.json` inside running
  prose, with no link wrapper at all

This applies to **every** reply.  In a summary listing 10 files,
all 10 must be hotlinked; in a one-line reply mentioning a single
plot, that plot must be hotlinked.  If you find yourself writing a
backticked file path with no markdown link around it in a chat
reply, stop and rewrite that section.

(This rule also exists below in a less-prominent
"Hotlinking image/plot files" section; this top-level HARD RULE
is the canonical statement.  The duplication is intentional —
discoverability matters more than DRYness here.)

---

## Token usage logging is mandatory on batched LLM call sites (HARD RULE)

**Any automated LLM call site that runs in a batch, judging loop, or
otherwise repeatedly enough that aggregate cost is operationally
interesting MUST plumb in a [`MultiModelUsage`](./assistant_axis/judge_pricing.py)
accumulator and persist a `usage.json` side-car next to its primary
output.**  This is how we reconstruct after-the-fact what an
expensive batch cost us, by which model, and validate against the
estimate that authorised the spend.

The canonical implementations are:

- [`results_analysis/axis_judge_correlation.py`](./results_analysis/axis_judge_correlation.py) —
  the response/static-mode judging path.  Uses
  [`BudgetTracker`](./assistant_axis/judge_pricing.py) (which wraps a
  single-model `UsageTotals`) for budget-capped axis-at-a-time runs.
  Writes `<output_dir>/usage.json`.
- [`assistant_axis/steering_judges.py`](./assistant_axis/steering_judges.py) —
  the steering sweep live-judging path.  Uses
  [`MultiModelUsage`](./assistant_axis/judge_pricing.py) because each
  cell hits both GPT and Haiku.  Writes
  `<cell_dir>/usage.json` at `RealJudgeDispatcher.shutdown()`,
  merging into the existing file so resumes accumulate rather than
  overwrite.  See `TestUsageTracking` in
  [`assistant_axis/tests/test_steering_judges.py`](./assistant_axis/tests/test_steering_judges.py).

**How to add it to a new call site (recipe):**

1. Import `MultiModelUsage` from
   [`assistant_axis.judge_pricing`](./assistant_axis/judge_pricing.py).
2. Construct one tracker per run/batch/cell (whatever the natural
   "unit of work" is for the script).
3. Pass `usage=tracker` to every
   [`call_judge_single_unified`](./assistant_axis/judge.py) /
   `call_judge_single` / `call_anthropic_judge_single` call.
   They extract the SDK's `response.usage` block and tick the
   accumulator automatically.
4. At end-of-run, call `tracker.write_json(output_dir / "usage.json")`
   (or merge into an existing one if the script supports resume).
5. Log `tracker.log_line()` so the cost appears in the run log too.

**`usage.json` schema** (verbatim from `MultiModelUsage.as_dict()`):

```json
{
  "per_model": {
    "claude-haiku-4-5-20251001": {
      "completion_tokens": 12345,
      "cost_usd": 0.4321,
      "model": "claude-haiku-4-5-20251001",
      "n_calls": 67,
      "prompt_tokens": 89012
    },
    "gpt-4.1-mini": { ... }
  },
  "total_completion_tokens": 23456,
  "total_cost_usd": 1.2345,
  "total_prompt_tokens": 178024,
  "n_calls": 134
}
```

**Why this is a hard rule:** before this was wired into steering
(2026-05-24), we estimated sweep cost from per-record token counts
and SDK pricing tables.  Those estimates were ±20% (rubric input
size was approximated from a single sample, not measured per call),
and offered no breakdown by judge model — so the only way to know
what a sweep cost was to wait for the OpenAI/Anthropic invoice and
back-out across overlapping runs.  Per-cell `usage.json` files give
us a precise, auditable, retrieve-anytime cost record.

**Existing call sites still on the to-do list (2026-05-24):**

- [`data_analysis/classify_goals.py`](./data_analysis/classify_goals.py)
- [`data_analysis/regenerate_role_instructions.py`](./data_analysis/regenerate_role_instructions.py)
- [`data_analysis/regenerate_trait_instructions.py`](./data_analysis/regenerate_trait_instructions.py)
- [`data_analysis/score_combinations.py`](./data_analysis/score_combinations.py)
- [`data_analysis/sample_trait_responses.py`](./data_analysis/sample_trait_responses.py)
- [`data_analysis/generate_antonyms.py`](./data_analysis/generate_antonyms.py)
- [`results_analysis/standardize_axis_spec.py`](./results_analysis/standardize_axis_spec.py)
- [`results_analysis/infer_axis_description.py`](./results_analysis/infer_axis_description.py)

Retrofit when next touched (or sooner if scheduled for a heavy
run).  Diagnostic one-offs (e.g.
[`tools/diagnose_unparseable.py`](./tools/diagnose_unparseable.py))
are exempt unless they grow into batch tools.

---

## Communication Style

### Concise and Technical
- Roger is highly technical and doesn't need basic concepts explained
- Prefer direct, information-dense responses over verbose explanations
- Assume sophisticated understanding of programming concepts, tools, and workflows

### Decision-Focused Workflow
- Roger likes to see trade-offs analyzed, then makes decisions quickly
- Example pattern: "Discuss pros and cons of each" → evaluates → "OK, repeated flags is it for now"
- Don't ask for permission when the path forward is clear
- DO ask when there are meaningful design choices to discuss

### Verify-Then-Trust Approach
- Roger prefers checking actual state over accepting assumptions
- Examples:
  - "See what we got, and update the requirements.txt"
  - "Check what's currently installed in the anthropic conda environment"
- When in doubt, check the actual state of files, installations, or system configuration
- Don't guess at versions, paths, or configuration - verify them

### Hotlinking image/plot files in chat replies (mandatory)

**See the top-level HARD RULE "Hotlink every file you mention to
Roger".**  Promoted there 2026-05-22 because the agent kept
forgetting and Roger had to reminded multiple times to redo
summaries with hotlinks added.  Brief version here:

- Always `[filename](./relative/path/to/file)` — workspace-relative,
  `./` prefix, every mention.
- Applies to `.png`, `.json`, `.log`, `.py`, `.md`, `.yaml`,
  `.jsonl` — anything Roger might open in the IDE.
- Bare paths and backticked-only paths do NOT hotlink in Cursor.
- Absolute paths (`/Users/roger/...`) hotlink but bake the
  username in; prefer `./...`.

### Plot visual verification (mandatory after any plot generation)

After generating or re-rendering ANY plot (`.png`, `.jpg`, `.pdf`),
**read the file back as an image and visually inspect it before
declaring the work done.**  Numbers and exit codes alone are not
sufficient: matplotlib will happily emit a visually broken figure
with a 0 exit code.

#### Verification workflow

1. Generate the plot.
2. Read it back: `Read` tool, path = generated PNG.
3. Inspect every text element for overlap / clipping (checklist
   below).
4. If broken, fix and re-render; verify again.  Loop until clean.

Don't ship "the numbers print correctly so it must be fine."  The
plot is the deliverable.

#### Failure-mode checklist (in order of typical recurrence)

**Vertical stacking inside the title block**

The most frequent issue.  `suptitle_with_specs` (in
`assistant_axis/plot_metadata.py`) renders its bold headline as
`fig.suptitle` and the spec line as `fig.text`.  matplotlib's
`constrained_layout` only sees the suptitle, so:

* The spec line and per-axes titles can stack onto each other on
  short figures, AND
* Once a `fig.colorbar` is also in the figure, `constrained_layout`
  IGNORES `fig.subplots_adjust(top=...)` and the colorbar stretches
  vertically to fill the full axes height — putting its top tick
  label right under (or on top of) the spec line.

Both are fixed at the caller, not in the helper:

* Give the figure enough vertical room (`figsize=(W, ≥ 6")` for
  figures with a 2-line title block + per-panel titles).
* Call `fig.subplots_adjust(top=top_rect)` *after*
  `suptitle_with_specs` returns its `top_rect` value (and use
  `constrained_layout=True` so colorbars still place correctly).
* Shorten colorbar height with `fig.colorbar(im, ax=axes,
  shrink=0.7)` (or smaller) — that's the only colorbar-height knob
  `constrained_layout` honours, so it's the right lever to keep
  the colorbar's top tick well below the spec line.

The `line_height` parameter on `suptitle_with_specs` auto-scales
from `title_fontsize` (in points) and figure height (in inches)
since May 2026, so the helper itself no longer needs per-caller
tuning.  But the figure-size + colorbar-shrink dance is still on
the caller.

**Horizontal collisions next to the colorbar**

The spec line is rendered centred at figure-x = 0.5, but a
colorbar pushes the heatmap / axes left of figure-centre, so the
spec line's right tail can graze the colorbar's top tick label
even when no vertical overlap exists.  Mitigations:

* Shorten the spec line.  Long axis lists (e.g. all 12 v2 axes
  spelled out) belong in the `_provenance` envelope, not in the
  spec line.
* Cut-off labels (long category names overflowing the bottom
  margin; tilted x-tick labels truncated on the right edge):
  rotate to 30°, reduce font, or truncate.

**Other recurring issues**

* **Legend overlap with data.**  Default `loc="best"` can land on
  top of points when the data fills the axes; pin with
  `loc="lower right"` (etc.) when that happens.
* **Colorbar disproportional** to the heatmap when the figure
  gains width without gaining height — purely aesthetic, only
  worth fixing if it confuses the reader.

#### Working recipe — heatmap + colorbar + 2-line title

For a single- or multi-panel heatmap with a shared colorbar and
the standard `suptitle_with_specs` 2-line title block, this
pattern reliably produces a clean layout:

```python
fig, axes = plt.subplots(
    1, n_panels,
    figsize=(5.3 * n_panels, 6.4),  # height ≥ 6 in
    constrained_layout=True,        # needed for the colorbar
)
if n_panels == 1:
    axes = np.array([axes])

# ... draw imshow + per-panel titles ...

fig.colorbar(im, ax=axes.tolist(), shrink=0.7,
             label="Spearman ρ (...)")

_, top_rect = suptitle_with_specs(fig, suptitle, spec_line)
# constrained_layout doesn't see fig.text, so reserve top region
# explicitly.  The shrink=0.7 above is what makes the colorbar
# respect this reservation.
fig.subplots_adjust(top=top_rect)

plt.savefig(out_path, dpi=150, bbox_inches="tight",
            metadata=png_metadata(title=suptitle, inputs=inputs))
plt.close()
```

Reference implementation (with extensive in-file comments
explaining each constraint): `results_analysis/batch_size_pairwise_rho.py::make_heatmap_pair`.

### Ask Mode vs Agent Mode
- Roger is conscious of the distinction between read-only (ask) and write (agent) modes
- In **ask mode**: Provide code snippets and instructions for Roger to apply
- In **agent mode**: Implement directly when asked
- When Roger attaches a plan file and says "Implement the plan", that's a clear signal to execute

### Iteration Style
- Prefers conversational iteration over big upfront specifications
- Build incrementally with feedback loops
- Verify results at each step: "See what we got"

### Understanding the Landscape First
- Roger often asks "meta" questions about tooling, APIs, or approaches before diving into implementation
- Examples: "Is tool use better than text parsing?", "What temperature should we use?", "Does Gemini support protobufs?"
- This pattern of understanding constraints and options before committing to an approach helps avoid rework

---

## Plan Requirements

### Testing Must Be Included

**Default rule:** Any plan that modifies code MUST include testing steps.

**Why:** Roger should never be surprised to discover code was changed but not tested.

**Exceptions (only two):**
1. **Roger explicitly says not to test** 
   - Example: "Just implement the fix, I'll test it myself later"
2. **Testing is impractical AND you get approval**
   - Before finalizing the plan, ask: "Testing this requires X which isn't available. Should I proceed without testing, or is there an alternative approach?"
   - Wait for confirmation

**If you're unsure:** Ask before finalizing the plan, don't assume.

---

## Technical Approach

### Check Actual State First
- Always prefer checking actual state over making "reasonable assumptions"
- Use tools to inspect: read files, check installed packages, examine directory structures
- Example: Don't assume a package version - check with `pip show` or equivalent

### Documentation as Infrastructure
- The README is a **living operational document** for future agents and developers
- It's not just user-facing documentation - it's a knowledge base
- When adding functionality, update README as part of the same work
- Include maintenance reminders and "gotchas" for future developers

### Maintainability Focus
- Roger thinks about future maintenance:
  - "Can we add a reminder to update requirements.txt?"
  - Version specifiers with `>=` for dependencies
  - Clear documentation of optional vs required packages
- Consider: "What will trip up someone (human or AI) working on this in 6 months?"

### Markdown Backtick Conventions
When writing prompts or documentation for LLMs:
- **DO backtick:** Literal filenames (`project.md`), extensions (`.md`), field names (`"type"`), state values (`"waiting"`)
- **DON'T backtick:** Format patterns inside JSON examples (`YYYY-MM-DD` in `<YYYY-MM-DD ...>`)

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

### NFS-safe file I/O (mandatory for `/workspace` reads and writes)

`/workspace` on RunPod is a MooseFS-backed network mount. Direct file
ops are usable but occasionally flaky — partial copies, transient `EIO`,
slow flushes, and (most importantly) **silent short reads from
torch.load** that surface as `RuntimeError("storage has wrong byte
size of dtype ...")` or `RuntimeError("PytorchStreamReader failed
reading zip archive: ... unexpected EOF, expected N more bytes")`.
Both reads and writes need defensive handling.

**The pattern in this codebase:**

- **Writes**: stage to a local-disk temp file (via `TMPDIR`), copy to
  the destination with retry/backoff, then `os.replace` to atomically
  publish the final name, then unlink the temp.  Atomic from a reader's
  point of view: dest never exists in a half-written state.  Tensor
  saves additionally run **three post-copy integrity checks** before
  considering the save successful — any failure triggers another
  copy attempt:
    1. **Size** matches staging byte-count (catches short-writes).
    2. **SHA-256** matches staging digest (catches silent byte-flips
       inside a correct-size file — e.g. corrupted NFS chunks that
       pass length checks; this is the failure mode behind the
       2.6 GB-but-unloadable `r_guardian__casual.pt`).
    3. **`torch.load` round-trip** succeeds (catches version-drift /
       pickle-format issues that byte-equality alone wouldn't).
  Both verify steps default to on; pass
  ``verify_sha256=False`` / ``verify_load=False`` to opt out when
  the hot-path cost isn't warranted.
- **Reads**: direct read inside a 5-attempt retry loop with the
  project-standard **exponential** backoff
  `DEFAULT_RETRY_DELAYS_S = (5, 20, 60, 180)` seconds (cumulative wall
  clock ~265 s, matching the API-retry standard in
  `results_analysis/axis_judge_correlation.py`).  Treats `OSError`,
  **`RuntimeError`** *and* `EOFError` as transient — the
  `RuntimeError` case is the failure mode above; earlier code that
  retried only `OSError` missed it entirely.  On final failure: log
  error and raise; the caller's outer loop is responsible for any
  "skip this item, continue with the rest" semantics.

New call sites should use the `assistant_axis.atomic_io` helpers
rather than re-rolling the pattern:

```python
from assistant_axis.atomic_io import (
    # Writes (via TMPDIR staging + atomic rename)
    atomic_write_text, atomic_write_bytes, append_jsonl, write_jsonl,
    torch_save_with_retry,
    # Reads (direct read in retry loop)
    read_text_with_retry, read_jsonl_with_retry, torch_load_with_retry,
)

atomic_write_text(json.dumps(config) + "\n", "/workspace/.../config.json")
append_jsonl({"foo": 1}, "/workspace/.../records.jsonl")
write_jsonl(records, "/workspace/.../records.jsonl")    # full rewrite
torch_save_with_retry(state_dict, "/workspace/.../checkpoint.pt")

config_text = read_text_with_retry("/workspace/.../config.json")
records = read_jsonl_with_retry("/workspace/.../records.jsonl")
state = torch_load_with_retry("/workspace/.../checkpoint.pt", map_location="cpu")
```

For very large tensors where you want to control torch.save's zipfile
flag (e.g. step 2's ~2.6 GB activation files where the new-style zip
serializer hits "iostream error" on RunPod), pass it through:

```python
torch_save_with_retry(activations_dict, output_file,
                      use_zipfile_serialization=False)
```

**Why:** silent failures here are extremely costly — multi-hour
generation runs that lose their last partial flush, restarts that
choke on their own state files, vectors silently missing because step 4
got a short read and skipped after warning.  The retry loops have
caught real RunPod NFS hiccups and let runs proceed; the size-sanity
check on writes catches the silent-truncation case that no exception
would otherwise reveal.

**Don't retry parse errors.**  `read_text_with_retry`,
`read_jsonl_with_retry` and `torch_load_with_retry` retry only on the
transient set above.  `JSONDecodeError`, `UnicodeDecodeError`, and
`pickle.UnpicklingError` propagate immediately because they almost
always indicate a corrupt-on-disk file rather than NFS flakiness, and
retrying just amplifies the delay.  `read_jsonl_with_retry` does
silently skip malformed lines by default (with a warning) since
records.jsonl can in principle have a half-flushed last line, though
`atomic_write_text` makes that nearly impossible on the writer side;
pass `skip_malformed=False` to make a malformed line fatal instead.

**TMPDIR setup:** drivers call `assistant_axis.tmpfs.setup_tmpdir(target)`
(default `target="/dev/shm"`) which **unconditionally** overrides any
pre-existing `$TMPDIR` so the atomic-write staging files land on RAM-
backed tmpfs.  This is by design: pod-defaults like
`TMPDIR=/workspace/tmp` would silently route every staging write
through slow NFS, and pre-May-2026 we had `setup_tmpdir_if_unset()`
preserve such defaults (requiring `unset TMPDIR` before every run).
Both `pipeline/run_pipeline.sh` and `steering/run_sweep.py` expose a
`--tmpdir <path>` CLI flag; callers who genuinely want to preserve
their existing `$TMPDIR` must pass it explicitly via
`--tmpdir "$TMPDIR"`.  The deprecated `setup_tmpdir_if_unset()` is
kept as a thin shim for external callers but logs a warning.

**Auditing past damage**: `pipeline/scan_missing_vectors.py` walks an
`activations/`+`vectors/`+`scores/` triple and classifies each missing
vector as one of `mysterious` / `corrupt_or_truncated_activation` / `ok_zero_size`
(re-run candidates) vs `missing_scores` / `below_min_count` /
`all_nan_or_empty` (legitimate filter).  Use it to recover from
historical short-read damage that pre-retry step 4 silently skipped.

**Current call sites (keep this list updated as new ones land):**

- Writes via `atomic_io`: `pipeline/2_activations.py` (uses
  `torch_save_with_retry`), `pipeline/4_vectors.py`,
  `pipeline/5_axis.py`, `assistant_axis/steering_runner.py`,
  `steering/run_sweep.py`
- Reads via `atomic_io`: `pipeline/2_activations.py` (responses),
  `pipeline/4_vectors.py`, `pipeline/5_axis.py`,
  `assistant_axis/steering_runner.py`, `steering/run_sweep.py`,
  `assistant_axis/axis.py` (`load_axis`, `load_axis_with_metadata`,
  `load_role_vector` — used by notebooks too)

---

### Plot Provenance Metadata (mandatory for every plot)
Every plot generated in this repo MUST embed a PNG-text-chunk
provenance block via `assistant_axis.png_metadata`.  The contents
depend on whether the plot came from a tracked script or an ad-hoc
exploratory script.

#### Tracked-script plots

```python
from assistant_axis import png_metadata
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            metadata=png_metadata(title=first_line_of_suptitle))
```

The helper auto-fills:
- `Title`         — pass the first line of the figure suptitle / caption
- `Author`        — defaults to "Roger Dearnaley"
- `Software`      — `uv run python <repo-relative path> <args>`, or
  `uv run python -m foo.bar.baz <args>` for `-m` invocations.  Pasteable
  into a shell at the repo root to re-run.
- `Creation Time` — ISO-8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SS+00:00`).
  Pre-2026-05 PNGs / JSON envelopes carry the older local-time format
  (`YYYY-MM-DD HH:MM:SS ±HHMM`); audit / reader code reads the field
  as an opaque string and tolerates either.
- `Source`        — git short SHA (+`+dirty` if working tree dirty)

#### Ad-hoc exploration plots (write the script to /tmp first)

When generating a plot during exploration, write the Python to
`/tmp/<descriptive_name>.py`, then run it.  Inside that script, embed
the source so the plot is self-contained:

```python
from pathlib import Path
from assistant_axis import png_metadata
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            metadata=png_metadata(
                title=first_line_of_suptitle,
                source_text=Path(__file__).read_text(),
            ))
```

This adds:
- `Source Code`        — full body of the entry-point Python file
- `Source Code SHA256` — hex digest, for tamper detection

Recovery is one line:
```python
from PIL import Image
print(Image.open("plot.png").info["Source Code"])
```

For the rare multi-file case, pass `source_files={"main.py": ..., "helper.py": ...}`
instead — they're JSON-serialised into a single chunk plus a
`Source Code Files` index.  But: needing more than one `/tmp/` file is
itself a smell that the work is graduating to a tracked module; prefer
that path.

**Heredoc gotcha**: `uv run python << 'PY' ... PY` invocations cannot
read their own body via `__file__` — they're stdin, not a file.  When
producing a plot worth keeping, write the body to `/tmp/foo.py` first
rather than using a heredoc.

#### Conventions summary

1. **Tracked script** writes a plot → ALWAYS pass
   `metadata=png_metadata(title=...)`.  No `source_text` needed (the
   `Software` field plus the git SHA + tracked source already
   reproduce it).
2. **Ad-hoc `/tmp/foo.py` script** writes a plot → ALWAYS pass
   `metadata=png_metadata(title=..., source_text=Path(__file__).read_text())`.
3. **Inline heredoc** writes a plot → if it's interesting, refactor
   into a `/tmp/foo.py` first; if it's truly disposable, pass at
   minimum `metadata=png_metadata(title=...)` so we get the timestamp
   and git SHA.
4. **Inspect** any plot with `exiftool foo.png` or
   `PIL.Image.open(p).info`.

The helper lives at `assistant_axis/plot_metadata.py`.  PNG
`tEXt`/`iTXt` chunks support up to ~2 GB each — no realistic ceiling
on what we can embed.

#### Caveats

Embedded source captures the script that *called* `savefig`, plus the
git SHA at run time.  It does **not** capture:
- Versions of pip dependencies (use `uv.lock` for that — the SHA pins it).
- Repo files imported by the script (the SHA pins those if the tree was
  clean; otherwise `+dirty` warns you).
- Other on-disk inputs the script reads (datasets, JSON config files).

So embedded source is a strong but not complete archeological record:
it gives you the entry-point + the git tree-state + your Python env's
lockfile.  In a few months, that's almost always enough to reconstruct
a plot — and tells you exactly what's missing if it isn't.

---

### End-to-End Data Provenance (writers, readers, audits)

The plot-metadata block above tells you *who produced* a plot and
*how to reproduce* it.  The provenance system below tells you whether
the plot's *inputs are still current* — i.e., is the plot stale
because a dataset, judge cache, or upstream cache JSON has changed
since the plot was rendered?

This is implemented as four cooperating layers:

1. **Dataset manifests** (`MANIFEST.json` per dataset, generated by
   `tools/regenerate_dataset_manifest.py`) — subtree-granular
   `summary_sha256` over `(rel_path, mtime, size)` triples.  See
   `audits/post_pipeline_derived_layout.md` for the
   `derived/{marginals,aggregates,axis,legacy_centroid}` directory
   convention.
2. **`assistant_axis.provenance`** — `InputSpec` (one declared dep),
   plus helpers (`current_data_subtree_input`, `current_file_input`,
   `current_files_input`) that build versioned (`v1:...`)
   fingerprints.  `validate_recorded` / `load_validated_json` re-derive
   current state and compare.
3. **PNG + JSON envelope embedding** — `png_metadata(..., inputs=...)`
   adds an `Inputs` iTXt chunk; `json_metadata(payload, inputs=...)`
   wraps a JSON payload as `{"result": payload, "_provenance": {...}}`.
4. **Audit tools** — `tools/audit_pngs.py` walks PNGs, classifies
   current/stale/legacy/frozen; `tools/audit_caches.py` does the same
   for JSON caches *and* propagates stale-ness transitively along
   inter-cache file dependencies.

#### Timestamp convention: UTC ISO-8601 everywhere

Every timestamp the provenance system writes is **UTC ISO-8601**
(`YYYY-MM-DDTHH:MM:SS+00:00`).  This applies uniformly to:

- `_provenance.produced_at` in JSON envelopes
- `Creation Time` in PNG metadata
- `last_modified_at` and `mtime` fields in `InputSpec` fingerprints
  (`v1:<mtime>@<size>`)
- `deferred_at` in `deferred_rejudges.yaml`
- `recorded_at` in `script_equivalences.yaml`
- `generated_at` in dataset `MANIFEST.json` files

Pre-2026-05-09 envelopes and PNGs may carry a local-time variant
(`YYYY-MM-DD HH:MM:SS ±HHMM`).  Reader/audit code treats these
fields as opaque strings (display-only; never parsed for ordering
or comparison), so the format mix is harmless.  Re-running any
producer will write a UTC envelope.

#### Writer pattern (mandatory for new analysis scripts)

Every analysis script that writes a JSON cache or a plot SHOULD
record its declared inputs.  The pattern:

```python
from assistant_axis import json_metadata, png_metadata
from assistant_axis.provenance import (
    InputSpec, current_data_subtree_input, current_file_input,
    current_files_input,
)

# 1. Build inputs list at the top of main(), after argparse:
inputs: list[InputSpec] = [
    current_data_subtree_input(
        data_dir, "traits/vectors", dep_key="traits_vectors",
        extras={"slot": str(args.slot), "layer": str(args.layer)}),
    current_data_subtree_input(
        data_dir, "roles/vectors", dep_key="roles_vectors"),
    current_file_input(
        dep_key="pairs_json", path=experiment_dir / args.pairs),
    current_files_input(
        dep_key="judge_caches", paths=judge_cache_paths,
        extras={"score_source": args.score_source}),
]

# 2. Wrap JSON output in the envelope:
envelope = json_metadata(records, inputs=inputs,
                         title=f"my_analysis slot={args.slot}")
out_path.write_text(json.dumps(envelope, indent=2))

# 3. Pass inputs to png_metadata for any PNG output:
fig.savefig(plot_path, dpi=150, bbox_inches="tight",
            metadata=png_metadata(title=title, inputs=inputs))
```

`dep_key` semantics: short, namespaced strings the writer chooses.
Used as the per-dep dict key by `validate_inputs`.  Don't reuse the
canonical "role" / "trait" terminology for these — those are entity
types; we use **dep_key** specifically to avoid that collision.

`InputSpec.kind` values: `"subtree"` (dataset directory tracked in a
`MANIFEST.json`), `"file"` (single file fingerprinted by
`v1:{mtime_iso}@{size}`), `"multi"` (composite of many files, e.g.
~130 per-axis judge caches, with `member_paths` recorded so a reader
can re-validate).

#### Reader pattern (`--cache-policy`)

When a script reads an upstream cache that was produced by a
provenance-aware writer, use `load_validated_json` and expose a
`--cache-policy` flag:

```python
from assistant_axis.provenance import CACHE_POLICIES, load_validated_json

p.add_argument("--cache-policy", choices=CACHE_POLICIES, default="warn",
               help="strict | warn (default) | rebuild | off")
args = p.parse_args()

records, check = load_validated_json(
    sweep_path, policy=args.cache_policy)
```

Policy semantics:

- **strict** — raise `StaleCacheError` on any drifted / missing /
  unverifiable input.  Use this when an automated pipeline must not
  silently consume stale data.
- **warn** (default) — print a drift summary to stderr, return the
  payload anyway.  Sane default for interactive work: you see drift
  immediately but aren't blocked.
- **rebuild** — if a `rebuild_callback` is wired in, call it (it
  should regenerate the cache), then re-validate.  Without a
  callback, behaves like strict.  No callbacks are wired in yet
  (Phase 4 stub); add one when an automated rebuild path makes
  sense.
- **off** — skip validation entirely.  Use only when you know what
  you're doing (e.g. reading a frozen historical cache for an
  archive plot).

Legacy bare-JSON caches without a `_provenance` envelope load
transparently as `(payload, None)` — readers don't need a code path
for the pre-migration shape.

`load_validated_json` also re-validates `kind="multi"` inputs by
re-stat'ing the recorded `member_paths`.  Pre-Phase-4 multi
InputSpecs (no `member_paths`) classify as `unverifiable` and trigger
strict failure; regenerate the cache to enable validation.

#### Reader+registrar pattern (`load_and_register`) — preferred

When a script *both* reads a cache *and* records that cache as a
dependency of its own output, use the combined helper instead of
calling `load_validated_json` and `current_file_input` separately:

```python
from assistant_axis.provenance import (
    InputSpec, current_file_input, load_and_register,
)

inputs: list[InputSpec] = [
    current_file_input(dep_key="producer_script", path=_SCRIPT_PATH),
    # ... other up-front deps (data subtrees, axis files, ...)
]

# Read + envelope-unwrap + drift-check + register-as-input, in one call.
sweep_payload, _spec, _check = load_and_register(
    sweep_path, dep_key="upstream_sweep", inputs=inputs,
    policy=args.cache_policy)
```

Why this pattern: in the pre-`load_and_register` world, every reader-
that-also-produces did the read with `json.loads` (or
`load_validated_json`) at one site and registered the dependency with
`current_file_input` at *another* site, often dozens of lines apart.
That split made it easy to:

- read a cache without registering it, so the output's
  `_provenance.inputs` undercounted dependencies and downstream
  audits couldn't follow the chain (`gpt_anthropic_response_weight_sweep.py`
  was a real instance — its output JSON had no envelope at all);
- register a path without actually reading it, so the writer claimed
  to depend on data it never consumed; or
- forget to envelope-unwrap, so a cache produced after the
  Phase-6 envelope landed parsed as `{"_provenance": ..., "result":
  ...}` and silently yielded zero records (real bug, found via the
  eco haiku-full plot regen at 00:52 on 2026-05-09 — the loader saw
  `_provenance` and `result` as if they were entity keys and skipped
  the axis).

`load_and_register` makes all three failure modes structurally
implausible by binding read + unwrap + drift-check + InputSpec build
together.  Use `load_validated_json` directly only when you're a
pure consumer with no JSON output of your own (e.g. an audit tool).

For `.npz` caches (which can't carry the `{"_provenance": ...,
"result": ...}` envelope because `np.savez` only stores arrays), use
`load_and_register_npz` — it reads the npz, parses the producer's
``meta`` blob (a `json.dumps(...)` of a dict that includes
`_inputs`), drift-validates those recorded inputs against current
state with the same `warn`/`strict`/`rebuild`/`off` policy, and
builds an InputSpec for the npz itself.  Producer side: write the
inputs list into your meta dict before `np.savez(...)`:

```python
meta["_inputs"] = [dataclasses.asdict(s) for s in inputs]
np.savez(out, matrix=matrix, column_keys=keys,
         meta=json.dumps(meta))
```

Reader side mirrors `load_and_register`:

```python
inputs: list[InputSpec] = []
data, meta, _spec, _check = load_and_register_npz(
    Path(args.npz), dep_key="my_npz_cache",
    inputs=inputs, policy=args.cache_policy)
matrix = data["matrix"]                       # use the NpzFile
```

##### Retrofit catalog (Phase 6c, 2026-05-09)

25 scripts identified as candidates when `load_and_register` shipped.
Generated by the triage in
`/Users/roger/.cursor/projects/.../triage_provenance_readers.py`-style
inline (`grep`-based), classifying each provenance-aware script by
which of `{json.loads, load_validated_json, current_file_input,
load_and_register}` it currently uses.

**HOT** — uses raw `json.loads` + separate `current_file_input` (no
envelope unwrap, no drift check; 2-call pattern):

- `results_analysis/batch_size_rho_curve.py` ✓ retrofitted
- `results_analysis/canonical_angles/plots/ca1_plane_pre_post_shear.py` ✓ retrofitted
- `results_analysis/gpt_sonnet_weight_sweep.py` ✓ retrofitted
- `results_analysis/gpt_vs_sonnet_scatter.py` ✓ retrofitted
- `results_analysis/optimal_axis_for_judge.py` ✓ retrofitted
- `results_analysis/pc_round_trip/klm_sweep.py` ✓ retrofitted (also fixed `permutation_null.py` and `plot_direction_cosines.py` callers)
- `results_analysis/rho_by_layer.py` ✓ retrofitted
- `results_analysis/rho_by_slot_and_K.py` ✓ retrofitted (also restructured to pre-load scores once instead of re-reading per (slot, K) iteration)
- `results_analysis/whitening_k_sweep.py` ✓ retrofitted
- `results_analysis/pc_round_trip/plot_winner_decomposition.py` ✓ retrofitted via `load_and_register_npz` (the npz analogue added at the same time; meta["_inputs"] now drift-validated under warn/strict/rebuild/off policy on every read).
- `tools/diff_against_recorded.py` — N/A: diagnostic / audit tool, not a producer of provenance-aware output.  Its `current_file_input` call computes a *probe* fingerprint to compare against a recorded one, not to declare a dep.
- `tools/mark_script_equivalent.py` — N/A: same as above (probe-only, not a producer).

**WARM** — uses `load_validated_json` *and* separate
`current_file_input` (envelope-aware but still 2-call):

- `results_analysis/judge_ensemble_rho_curve.py` ✓ retrofitted
- `results_analysis/pc_round_trip/plot_direction_cosines.py` ✓ retrofitted (also caught 3 unregistered cache reads)
- `results_analysis/pc_round_trip/plot_global_winner_cosines.py` ✓ retrofitted
- `results_analysis/pc_round_trip/plot_histogram.py` ✓ retrofitted
- `results_analysis/pc_round_trip/plot_loglin.py` ✓ retrofitted
- `results_analysis/pc_round_trip/plot_null_overlay.py` ✓ retrofitted
- `results_analysis/pc_round_trip/plot_subspace_projection.py` ✓ retrofitted
- `results_analysis/pc_round_trip/plot_winner_consistency.py` ✓ retrofitted
- `results_analysis/pc_round_trip/winner_decomposition.py` ✓ retrofitted
- `results_analysis/plot_batch_size_quality_vs_cost.py` ✓ retrofitted
- `results_analysis/whitening_k_peak_fit.py` ✓ retrofitted
- `results_analysis/whitening_k_weighted_scatter.py` ✓ retrofitted

**N/A** — false positive in the initial triage; on inspection
doesn't actually fit the read+register pattern:

- `results_analysis/axis_judge_correlation.py` — only `lvj=1` was a
  docstring reference (no actual `load_validated_json` call); its
  cache reads are either raw `.pt`/`.jsonl` config/data inputs
  (correctly registered with `current_file*_input`, not envelope-
  wrapped) or self-output reads on the resume path (which already
  inline-unwrap envelopes via the `_load_json` helper for that
  purpose).  No load_and_register retrofit benefits here.

**DONE** — already using `load_and_register`:

- `results_analysis/gpt_anthropic_response_weight_sweep.py` (the
  reference retrofit; envelope was added at the same time as the
  helper because this script previously had no `_provenance` output
  envelope at all).

#### Audit tools

To check the freshness of everything in the repo at once:

```bash
# Are committed plots still fresh w.r.t. their declared inputs?
uv run python tools/audit_pngs.py --output reports/audit_pngs.md

# Are JSON caches fresh, including transitively through file deps?
uv run python tools/audit_caches.py --output reports/audit_caches.md

# Filter to just the stale ones (handy for "what do I need to rerun?"):
uv run python tools/audit_pngs.py --status stale
uv run python tools/audit_caches.py --status stale
```

Status taxonomy:

- **current** — all declared inputs match current state.
- **stale_direct** (caches) / **stale** (PNGs) — at least one input
  drifted/vanished/unverifiable.
- **stale_transitive** (caches only) — own deps are clean but an
  upstream cache (reachable via `kind="file"` deps) is stale.
- **legacy** — no provenance envelope (pre-migration, hand-written,
  or pipeline-produced like `MANIFEST.json` and judge caches).
- **frozen** (PNGs only) — explicit `Frozen` / `Frozen-At` chunk
  marking the artifact as intentionally pinned to historical data.
  Add via `png_metadata(..., extra={"Frozen": "snapshot-2026-04"})`
  in the producing script.

#### When to regenerate manifests

`MANIFEST.json` files live at each dataset root and are gitignored
(under `runpod_workspace/`).  Regenerate after any change to a
dataset's contents:

```bash
uv run python tools/regenerate_dataset_manifest.py \
    --dataset 'runpod_workspace/qwen/qwen-3-32b Roger 8slot'
```

Subtree-granular and stable: re-running on an unchanged dataset
produces a byte-identical `MANIFEST.json`.  Audit tools rely on this
to detect drift; ALWAYS regenerate after edits to the
`derived/{marginals,aggregates,axis,...}` directories produced by
`compute_combo_marginals.py` (the only post-pipeline producer in the
project).

#### Migrated scripts (as of May 2026)

Use this list to answer "is X provenance-aware?" without grepping.
Quick check: `rg 'json_metadata\(|png_metadata\(.*inputs=' results_analysis/`
gives writers; `rg '--cache-policy|load_validated_json' results_analysis/`
gives readers.

**Writers** — record `_provenance` envelopes on JSON output and
embed `Inputs` chunks in PNG output (i.e., they pass `inputs=...` to
`json_metadata` / `png_metadata`):

Top-level `results_analysis/` (judge-cache-reading producers and
pure-data producers):

- `results_analysis/all_roles_pairwise_slots.py`
- `results_analysis/axis_cosine_seriation.py`
- `results_analysis/batch_size_rho_curve.py`
- `results_analysis/gpt_sonnet_weight_sweep.py`
- `results_analysis/gpt_vs_sonnet_scatter.py`
- `results_analysis/judge_ensemble_rho_curve.py` (also a reader —
  uses `load_validated_json`; no `--cache-policy` CLI flag, policy
  is hard-coded `"warn"`)
- `results_analysis/optimal_axis_for_judge.py`
- `results_analysis/pair_slice_plots.py`
- `results_analysis/pca_scree_plots.py`
- `results_analysis/rho_by_layer.py`
- `results_analysis/rho_by_slot_and_K.py`
- `results_analysis/role_pair_diff_norms.py`
- `results_analysis/token_position_noise_analysis.py`
- `results_analysis/variance_decomposition.py`
- `results_analysis/whitening_k_peak_fit.py`
- `results_analysis/whitening_k_sweep.py`
- `results_analysis/whitening_k_weighted_scatter.py`

`results_analysis/canonical_angles/plots/`:

- `results_analysis/canonical_angles/plots/ca1_plane_pre_post_shear.py`
- `results_analysis/canonical_angles/plots/combos_vs_traits_roles_pooled.py`
- `results_analysis/canonical_angles/plots/goal_vs_nogoal_mutual.py`
- `results_analysis/canonical_angles/plots/layer_sweep.py`
- `results_analysis/canonical_angles/plots/whitening_sweep_with_nulls.py`

`results_analysis/pc_round_trip/`:

- `results_analysis/pc_round_trip/klm_sweep.py`
- `results_analysis/pc_round_trip/permutation_null.py` (incremental
  saves: `_merge_save` transparently unwraps any pre-existing
  envelope, merges fresh data into `result`, then re-wraps with a
  fresh provenance block — see "Atomic envelopes for incremental
  caches" gotcha below)
- `results_analysis/pc_round_trip/plot_direction_cosines.py`
  (writes the headline PNG plus four side-output JSONs when
  `--include_optimization` is passed)
- `results_analysis/pc_round_trip/plot_global_winner_cosines.py`
- `results_analysis/pc_round_trip/plot_histogram.py`
- `results_analysis/pc_round_trip/plot_loglin.py`
- `results_analysis/pc_round_trip/plot_null_overlay.py`
- `results_analysis/pc_round_trip/plot_subspace_projection.py`
- `results_analysis/pc_round_trip/plot_winner_consistency.py`
- `results_analysis/pc_round_trip/plot_winner_decomposition.py`
- `results_analysis/pc_round_trip/winner_decomposition.py` (NPZ
  output cannot carry a JSON envelope, so the InputSpec list is
  serialised into `meta` inside the NPZ; downstream readers
  declare the NPZ itself as a `current_file_input`)

**Readers** — accept `--cache-policy {strict, warn, rebuild,
off}` and validate upstream caches via `load_validated_json`:

- `results_analysis/plot_batch_size_quality_vs_cost.py`
- `results_analysis/whitening_k_peak_fit.py`
- `results_analysis/whitening_k_weighted_scatter.py`
- `results_analysis/pc_round_trip/plot_direction_cosines.py`
- `results_analysis/pc_round_trip/plot_global_winner_cosines.py`
- `results_analysis/pc_round_trip/plot_histogram.py`
- `results_analysis/pc_round_trip/plot_loglin.py`
- `results_analysis/pc_round_trip/plot_null_overlay.py`
- `results_analysis/pc_round_trip/plot_winner_consistency.py`
- `results_analysis/pc_round_trip/plot_subspace_projection.py`
- `results_analysis/pc_round_trip/winner_decomposition.py`

(Several scripts appear in both lists — they consume an upstream
cache *and* write a new one.  By convention, runtime
`--cache-policy` validation is reserved for downstream plot/scan
scripts; producer-style scripts that read judge caches and write
their own envelope-bearing output (e.g.
`optimal_axis_for_judge.py`, `whitening_k_sweep.py`,
`gpt_sonnet_weight_sweep.py`, `gpt_vs_sonnet_scatter.py`,
`rho_by_layer.py`, `rho_by_slot_and_K.py`, `klm_sweep.py`,
`permutation_null.py`) record the upstream fingerprints in their
output envelope and rely on `audit_caches.py` to surface drift
transitively, instead of forcing a runtime check on every job.)

**Manifest-tracked producers (not envelope-bearing)** —
`results_analysis/compute_combo_marginals.py` writes outputs into
the dataset under `<data_dir>/combinations/derived/` and is tracked
via the dataset's `MANIFEST.json` rather than per-file
`json_metadata` envelopes (see the regen reminder block at the top
of the script).  Regenerate the manifest after any
`compute_combo_marginals.py` run so audits see the fresh state.

**Atomic envelopes for incremental caches.** Scripts that build
caches up incrementally (e.g. `permutation_null.py`'s
`_merge_save`, or the recover/refill paths in
`axis_judge_correlation.py`'s `_save_json`) need to keep the
on-disk file envelope-bearing at every flush point so a kill -9
mid-run still leaves a valid cache.  The pattern is: on read,
unwrap the existing envelope and operate on the inner `result`
dict (the rest of the merge logic stays unchanged); on write,
re-wrap with a fresh `json_metadata(..., inputs=...)` and
atomic-rename into place.  Don't try to splice freshly-judged
records into an existing envelope without re-stamping; the
envelope's `produced_by.git_sha` and timestamp must reflect the
current run.

**Pending migration / known incomplete (2026-05-09):** None on the
script side.  The remaining gaps are *output* refreshes blocked on
external work:

- The 2 response-mode `gpt_*_q9_response_weight_sweep_slot6.json/png`
  outputs and ~10 `whitening_k_sweep_*` / `batch_size_curve_*` /
  `rho_by_layer.json` / response-side `optimal_axis/.../diagnostics.json`
  caches all classify as `legacy` or `stale_*` because their
  upstream response-mode judge caches are mid-rejudge under the v2
  rubric.  They will auto-clear once the v2 rejudge completes and
  the affected producers are re-run; no script work is needed.

`gpt_anthropic_response_weight_sweep.py`, previously listed as
partially migrated, was finished as the reference retrofit when
`load_and_register` shipped (Phase 6c).
`plot_winner_decomposition.py`, previously flagged as in-flight,
landed via the new `load_and_register_npz` helper added to
`assistant_axis.provenance`.

Migrate opportunistically when touching a script for other reasons;
the writer-pattern code template above is ~10 lines.

**Audit clean state (May 9 2026 04:00 baseline).**  After Phase
C–D, the pc_round_trip migration block, the Phase 6c
`load_and_register` retrofit, and the Phase 6d post-retrofit
cleanup (orphan-PNG deferrals, pre-cohort-rename file deletions,
runpod_workspace dataset-tree deferral, single broken-import fix
in `plot_direction_cosines.py`'s use of the renamed
`_build_subtree_inputs`), the steady-state audit expectation for
the analysis tree is:

- Top-level `results_analysis/` outputs in `roger/`: zero
  uncontrolled `legacy` (every cache and PNG is either
  envelope-bearing or registered in `deferred_rejudges.yaml`);
  zero `stale_direct` from anything other than the in-flight v2
  response rejudge.  Any `stale_transitive` should trace back
  to a known-pending upstream change (e.g. a still-running
  rejudge job) rather than silent drift.

  Concrete numbers from the most recent run
  (`reports/audit_*_final.md`):

  - `roger/` JSONs: 24 current, 0 uncontrolled legacy, 2 legacy
    + 10 stale_direct + 25 stale_transitive (all blocked on the
    v2 response rejudge — `gpt_*_q9_response_weight_sweep_slot6`
    + `whitening_k_sweep_*` + `batch_size_curve_*` +
    `rho_by_layer` + response-side `optimal_axis/.../diagnostics`
    + their downstream caches), 2642 deferred, out of 2703.
  - `roger/` PNGs: 125 current, 0 uncontrolled legacy, 2 legacy
    + 3 stale (all blocked on the v2 response rejudge: the 2
    `gpt_*_q9_response_weight_sweep_slot6.png` plots and the 3
    `batch_size_curve_8slot/*` + `optimal_axis restarts.png`
    plots), 664 deferred, out of 794.
  - `runpod_workspace/` JSONs: 15715 deferred (entire dataset
    content, manifest-tracked) out of 15715 — every file in
    `runpod_workspace/<dataset>/...` is now covered by the
    `runpod_workspace/*` deferral so audits trust
    `MANIFEST.json` for those subtrees.
  - `runpod_workspace/` PNGs: 1 deferred (the embedded
    `assistant-axis/img/assistant_axis.png` logo).
- **Response-judge rubric v1 → v2 transition (in flight,
  2026-05-09):** the v1 rubric (entity-named
  `RUBRIC_RESPONSE_BATCH`) was snapshotted to an offline backup
  before the v2 anonymisation went in.  Going-forward axes are
  being re-judged under v2 so their downstream caches will record
  fingerprints of an envelope-bearing v2 response cache instead
  of a grandfathered-in legacy one.  Until that re-judge
  completes, expect `stale_transitive` rows on any analysis cache
  whose response-mode dependency hasn't yet been re-judged; once
  it has, the same row should flip to `current` without further
  action on the analysis side.
- Pre-migration archival artifacts (3-cell pc_round_trip
  winners, K-near-N variant `klm_results` backups, deprecated
  `nth_pc` per-cell winners, old optimal_axis outputs, v1
  response-judge snapshots) are kept on-disk for cross-cohort
  comparison and registered as `legacy → deferred` with a
  free-text reason in `deferred_rejudges.yaml`.  These rows show
  up in audit reports under the `deferred` bucket, NOT `legacy`.
- `runpod_workspace/<dataset>/` is tracked via per-dataset
  `MANIFEST.json` rather than per-file envelopes; freshness is
  asserted by re-running `tools/regenerate_dataset_manifest.py`
  and confirming the output is byte-identical (the manifest is
  stable on unchanged input).

When that picture changes (e.g. a fresh script lands as `legacy`
without a deferral, or a producer change makes `stale_direct`
appear locally for reasons other than the in-flight v2 rejudge),
treat it as a real signal: either finish the migration, register
a deferral, or fix the drift.

**`pc_round_trip` library helpers / launchers** (no I/O of
provenance-relevant artifacts; intentionally not migrated):

- `results_analysis/pc_round_trip/__init__.py` — package marker.
- `results_analysis/pc_round_trip/principled_layer_transport.py` —
  pure library helpers for cross-cell α-weight transport; no
  on-disk reads or writes.
- `results_analysis/pc_round_trip/launch_judge_runs.py` —
  orchestrator that submits per-axis judge jobs to
  `axis_judge_correlation.py`; outputs land in the judge
  subsystem's domain, which has its own provenance regime
  (RUBRIC_VERSION + `_build_axis_judge_inputs`).

**Special case — judge caches.**
[results_analysis/axis_judge_correlation.py](results_analysis/axis_judge_correlation.py)
produces `scores_descriptions.json`, `scores_instructions.json`,
and `scores_responses.json` (plus `projections.json`,
`correlations.json`, `correlation_plot.png`).  It is fully
provenance-aware as of **Phase 6**, with bespoke handling for
rubric versioning, harmless-edit equivalence, and deferred
rejudging.  See the next subsection for the rules and the matching
maintainer workflows.

### Judge-step provenance (rubrics, equivalence, deferral, recovery)

Judging is the single expensive non-deterministic step in the
pipeline, so it gets a heavier provenance regime than the rest of
the system.  The four cooperating layers:

#### 1. Producer dependencies (per output mode)

`axis_judge_correlation.py`'s
[`_build_axis_judge_inputs`](results_analysis/axis_judge_correlation.py)
builds an `InputSpec` list per output, one of `descriptions` /
`instructions` / `responses` / `projections` / `correlations`:

- **producer_script** (`kind="file"`): the source of
  `axis_judge_correlation.py` itself, with current judge config
  (provider, model, temperature, max_tokens, batch-size knobs)
  recorded in `extras`.  Editing rubric strings, prompt builders,
  or score-parsing logic flips this fingerprint.
- **axis_file** OR (**pair_vectors** + **pole_instructions**):
  whichever path defined the axis.  Pole text deps are skipped
  when `--pos_pole` / `--neg_pole` were passed on the CLI (the
  text came from args, not from a file).
- **corpus_instructions** (`kind="multi"`): every
  `instructions_dir/{roles,traits}/instructions/*.json` the script
  could read.  Editing one trait JSON's `description` or
  `instruction.pos[*]` field flips the whole multi fingerprint;
  this over-invalidates a little but keeps logic simple.
- **corpus_vectors** (`kind="multi"`): every
  `data_dir/{roles,traits}/vectors/*.pt` (used for projections;
  defines the scorable entity set in every mode).
- **responses-mode extras** (only for `responses` / `correlations`):
  `response_files`, `response_score_files`, `default_responses`,
  `questions_file`.

A `JUDGE PROVENANCE NOTE` comment block at the top of
`axis_judge_correlation.py` lists which kinds of edit are harmless
vs not, and shows the one-line `mark_script_equivalent.py`
invocation to declare the harmless ones.

#### 2. Per-axis consumer granularity

The 5 consumers that read judge caches (`whitening_k_sweep.py`,
`gpt_sonnet_weight_sweep.py`, `batch_size_rho_curve.py`,
`optimal_axis_for_judge.py`, `pc_round_trip/plot_direction_cosines.py`)
record one `current_file_input` *per (axis, mode, judge_source)*
rather than one composite `current_files_input` over the whole fan-
out.  Convention: `dep_key = f"judge_{axis_id}_{mode}_{source}"`
(e.g. `judge_q9_angel_vs_demon_descriptions_gpt`).  Result: an
`audit_caches.py` report on a downstream cache pinpoints exactly
which axis was rejudged rather than smearing the drift over the
whole bundle.

#### 3. Script equivalence (harmless code edits)

`script_equivalences.yaml` at repo root declares per-edit
"output-preserving" pairs for `kind="file"` producers.
[`assistant_axis.script_equivalence.is_equivalent`](assistant_axis/script_equivalence.py)
does a transitive BFS over the registered edges; when
`validate_recorded` sees `kind="file"` drift on a script and the
registry has a chain `recorded_fp → ... → current_fp`, the
status downgrades from `drift` to `equivalent` (counted as ok in
`ProvenanceCheck.ok`).

Workflow after a harmless edit (docstring, type hint, log message,
internal refactor that preserves I/O):

```sh
uv run python tools/mark_script_equivalent.py \
    --script results_analysis/axis_judge_correlation.py \
    --reason "Refactored prompt builder; identical I/O."
```

The CLI auto-detects `--from-fp` from the most recent envelope-
bearing cache that recorded the script (looking under `roger/` and
`runpod_workspace/`).  `--to-fp` defaults to the current on-disk
fingerprint.  `tools/mark_script_equivalent.py --list` shows all
declared edges; `--check` queries an arbitrary edge without
mutating the registry.

#### 4. Deferred rejudging (intentional staleness)

`deferred_rejudges.yaml` at repo root declares "I know this is
stale, don't bug me about it" entries: a `path_glob` (and
optional `dep_key` glob) plus a free-text `reason`.  Both
`audit_caches.py` and `audit_pngs.py` reclassify matched rows to
`deferred`, covering **two pre-deferred statuses**:

- **`stale_*` → `deferred`**: "this would otherwise need
  rerunning; defer the rerun instead".  The original use case —
  for example, a trait edit that drifted one judge cache and we
  don't want to re-judge yet.
- **`legacy` → `deferred`** (May 2026 backfill): "this never had
  an envelope and we've decided not to migrate the producer.
  Presume current as of mechanism introduction".  Used to mark
  pre-Phase-6 judge caches (`scores_*.json`, `projections.json`,
  `correlations.json`, `correlation_plot.png`, `gaps.json`)
  written by `axis_judge_correlation.py` before its Phase 6a
  wrap.  The deferral self-clears the moment the producer is
  re-run with the modern writer pattern (the cache then carries
  an envelope and reads as `current` directly; the registry
  entry becomes a no-op).

`current` and (PNG-only) `frozen` are never reclassified — the
registry only takes effect for rows that would otherwise read as
`stale_*` or `legacy`.  The deferred section in audit reports
shows `pre_deferred_status`, the matching registry entry, and
when it was declared.

Deferrals are *deferred-until-removed* — there's no `expires_at`,
they stay active until the maintainer deletes the entry.  Use
`tools/defer_rejudge.py --list` to inspect, `--remove` to lift.

```sh
# Defer all q9 description rejudges (stale → deferred):
uv run python tools/defer_rejudge.py \
    --path 'roger/axis_judge_experiments/q9_*/gpt/scores_descriptions.json' \
    --reason "Held until 4-model audit completes."

# Backfill: declare all pre-Phase-6 judge caches as legacy → deferred
# (the May 2026 standard reason template):
uv run python tools/defer_rejudge.py \
    --path '**/scores_descriptions.json' \
    --reason "Legacy judge cache pre-dating Phase 6 provenance mechanism (May 2026). Presumed current as of mechanism introduction; cost-prohibitive to re-judge. Auto-clears when re-judged."

# Lift later:
uv run python tools/defer_rejudge.py --remove \
    --path 'roger/axis_judge_experiments/q9_*/gpt/scores_descriptions.json'

# Skip the registry entirely in one run (debug):
uv run python tools/audit_caches.py --ignore-deferrals
```

Audit `--status` semantics:
- `--status stale` matches `stale_direct` + `stale_transitive`
  (does NOT include `deferred`, which is the explicit "skip me"
  bucket).
- `--status deferred` shows just the deferred rows.
- `--status legacy` shows just unmigrated/uncovered rows
  (i.e. legacy that *isn't* matched by a registry entry).
- No `--status` filter shows everything in their respective
  sections.

##### Deferral categories (schema v2, May 2026)

Each entry carries a structured `category` tag drawn from
`DeferralCategory` in
[`assistant_axis/deferral_registry.py`](assistant_axis/deferral_registry.py).
The category drives both audit-report grouping and category-specific
invariant checks in `tools/audit_deferrals.py`:

| Category | Semantics | Auto-clears? |
|---|---|---|
| `legacy_bare` | Pre-Phase-6 cache without an envelope. | When producer is re-run (envelope replaces bare file). |
| `frozen_snapshot` | Point-in-time capture *with a live twin*; comparison plots typically read both. Optional `compares_to` glob points at the live twin. | Never. |
| `archived` | Standalone output of a retired pipeline configuration; no live twin (e.g. pre-cohort-cutover backups, deprecated variants). | Never. |
| `orphan_no_producer` | Output whose producer script is not in the git tree (one-off `/tmp/_*.py`, removed/superseded producers). Optional `producer_script` records the historical path. | Manual; `audit_deferrals` ⚠️ alarms if a recorded `producer_script` reappears in `git ls-files`. |
| `superseded` | Replaced by a named newer artifact (`replaced_by` glob). | When `replaced_by` is removed (audit errors). |
| `operational` | Config / API-usage / tracker files written without an envelope by design. | Never. |
| `manifest_tracked` | Freshness asserted via a per-dataset `MANIFEST.json` rather than per-file envelopes. | Never. |
| `external_pipeline` | Output of a workflow outside the current provenance migration scope (coherence_eval, pc_axis_describer). | Scope decision. |
| `experimental_one_off` | Exploratory artifact with no plan to integrate. | Manual. |
| `hand_curated_input` | Hand-edited config consumed by producers (`pair_list*.json`, embedded logo PNGs). | Never (it's an input, not a producer output). |
| `uncategorized` | Schema-v1 entry that hasn't been backfilled. | The audit raises so backfill is forced. |

The `frozen_snapshot` vs `archived` distinction is intentional even
though both never auto-clear: `frozen_snapshot` implies a live twin
exists, so the audit can verify `compares_to` and warn when paired
comparison plots have lost half their provenance.  `archived` makes
no such claim — it's "this is from a retired pipeline configuration,
nothing to compare against."

When deferring something via the CLI, pass the category explicitly:

```sh
uv run python tools/defer_rejudge.py \
    --path 'roger/.../some_orphan_*.png' \
    --category orphan_no_producer \
    --producer-script /tmp/_old_canonical_angles.py \
    --reason "Pre-Phase-6 one-off; producer no longer in codebase."
```

Optional metadata fields (`--producer-script`, `--replaced-by`,
`--compares-to`) are emitted only when set; the on-disk YAML stays
terse for the common case.  Schema-v1 files (no `category` field)
are still readable; existing entries load as `UNCATEGORIZED` and the
file is upgraded to v2 in place on first write.

##### `tools/audit_deferrals.py` (registry coherence)

Companion to `audit_caches.py` / `audit_pngs.py` — those check
*on-disk artifacts*; this checks *the registry itself*.  Runs four
category-specific invariants:

* **`orphan_promoted`** (error) — an `orphan_no_producer` entry's
  `producer_script` is now tracked by git.  Either the producer was
  promoted (lift the deferral, re-run, let fresh envelopes land) or
  the path collision is coincidental (rename the recorded
  `producer_script` to disambiguate).
* **`superseded_replaced_by_missing`** (error) — `replaced_by` glob
  matches no file on disk.  Either the successor was deleted
  (remove the deferral entry too) or it never got produced
  (regenerate the successor).
* **`superseded_missing_replaced_by`** (warning) — `superseded`
  entry without a `replaced_by` field; consumers can't navigate to
  the successor.
* **`frozen_snapshot_twin_missing`** (warning) — `compares_to` glob
  matches no file; the live twin used by paired comparison plots is
  missing.  The snapshot may belong in `archived` instead.
* **`uncategorized`** (error) — a schema-v1 entry that needs
  backfilling.

```sh
# Markdown roll-up to stdout:
uv run python tools/audit_deferrals.py
# Save to file:
uv run python tools/audit_deferrals.py --output reports/audit_deferrals.md
# CI-friendly (exit non-zero on any error):
uv run python tools/audit_deferrals.py --strict
```

The audit deliberately does NOT inspect file contents — only the
registry plus a snapshot of `git ls-files` and the on-disk path
index.  Cheap to run, ~1s.

##### Apply deferrals BEFORE propagating transitive staleness

Subtle ordering invariant in `tools/audit_caches.py`'s `main()`:
`apply_deferrals(rows)` must run *before*
`propagate_transitive_stale(rows)`.  The propagation BFS only walks
edges from rows whose status is in
`STALE_STATUSES = ("stale_direct", "stale_transitive")`, so deferred
rows act as barriers — a deferred upstream cache no longer falsely
taints downstream consumers as `stale_transitive`.

This was a real bug fixed in May 2026 (Finding 1 of an audit pass):
when the order was swapped, deferring an upstream desc+inst cache
was tainting every downstream rho/whitening sweep transitively, even
though the consumers' recorded fingerprints of the deferred cache
were perfectly current.  Test:
`tools.tests.test_audit_caches.test_deferred_upstream_does_not_taint_downstream`.

#### 5. Recovery: `tools/diff_against_recorded.py`

When a cache reports `stale_direct` on `producer_script`, this
tool reconstructs the diff between the recorded version and the
current one, so the maintainer can decide
"harmless → mark equivalent" vs "real change → rejudge or defer":

```sh
uv run python tools/diff_against_recorded.py \
    roger/axis_judge_experiments/q9_angel_vs_demon/gpt/scores_descriptions.json
```

Resolution order:
1. **Git** (`git show <recorded_sha>:<script_rel>`): preferred
   when the envelope's `produced_by.git_sha` is reachable.  Skip
   with `--no-git`.
2. **Cursor Local History fallback**: walks
   `~/Library/Application Support/Cursor/User/History/<hash>/`,
   finds the snapshot whose `timestamp` is closest to the
   recorded `last_modified_at`, and diffs it against the current
   file.  Skip with `--no-history`.

The tool prints a ready-to-paste `mark_script_equivalent.py`
invocation pre-filled with the recorded `--from-fp`, so a
maintainer who confirms the diff is harmless can declare
equivalence in one paste.

**Cursor Local History tuning** — the default per-file entry cap
is small (~50).  For long-lived editing sessions on
`axis_judge_correlation.py`, bump it via
`workbench.localHistory.maxFileEntries: 500` in
`~/.cursor/settings.json` so the recovery tool can reach further
back.

### Deferred future work — Phase 7 (subtree content hashing)

Originally scoped as Phase 6, demoted to Phase 7, then **deferred**
in May 2026 after a cost/value review.  Today's subtree
fingerprints are metadata-only (sorted `(rel_path, mtime_ns_floored,
size)` triples, SHA-256'd); content hashing would make them
robust to `cp` / `scp` / non-`-t` rsync / `git checkout` / editor
no-op-rewrites and would catch silent bit-rot, at the cost of
making manifest regen 30-300x slower (still <1 minute on local
SSD for the full dataset).

The full write-up — what it solves, what it doesn't, the per-subtree
hashing-cost numbers, the implementation plan, and the decision
criteria for when to actually pick it up — lives in code as the
`TODO(phase-7-deferred)` block at the top of
[`assistant_axis/provenance.py`](assistant_axis/provenance.py),
with a coordinated manifest-regen-side checklist at the
`_summarize` function in
[`tools/regenerate_dataset_manifest.py`](tools/regenerate_dataset_manifest.py).
Pick up Phase 7 if we start moving datasets between machines via
something other than `rsync -at`, if false-positive drift from
no-op-rewriting tooling becomes a recurring nuisance in audit
reports, or if we need a defensible answer to "did this dataset get
corrupted on disk?".  Until then, metadata fingerprints are
strictly cheaper for equal-or-better real-world behaviour in
Roger's workflow.

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

### Steering judges (Phase-2 architecture)

**Activation space (read this first).**  Steering vectors live in
**raw model-activation space** — the native frame the model
operates in.  `ActivationSteering` (in
[`assistant_axis/steering.py`](assistant_axis/steering.py)) and the
sweep runner ([`steering/run_sweep.py`](steering/run_sweep.py)) load
the unwhitened `axis.pt` directly and add it to the residual stream
at the chosen layer; **no whitening, no shear**.  The
whitening / soft-shear regimes documented in *"Whitening / soft-shear
defaults"* below are **analysis-side only** (they sharpen
judge-projection ρ) and explicitly do not apply here.  When you
load a judging-side axis vector for use as a steering vector, pass
it through `ActivationSteering` *as-is* — whitening or shearing
first would push it out of the model's native frame and degrade
both the steering and any safety-relevant invariants
(e.g. activation-capping thresholds calibrated against the
distribution of raw projections).  Same reason
`pc_round_trip/launch_judge_runs.py` keeps its own
`DEFAULT_SHEAR_L=0` — see the caveat at the end of the whitening
section.

Steering sweeps use a **two-tier judging** protocol baked into
[`assistant_axis/steering_judges.py`](assistant_axis/steering_judges.py)
and consumed by [`assistant_axis/steering_runner.py`](assistant_axis/steering_runner.py)
via the `JudgeDispatcher` protocol:

1. **Coherence (synchronous, blocking)** — per-record `judge_coherence_blocking()`
   call after each generation batch.  0-3 rubric.  Default model
   `gpt-4.1-mini`; configurable via `--coherence-model`.  Gates the
   `should_stop_at(strength)` early-termination decision.
2. **Persona / RP (asynchronous, post-batch)** — gpt-4.1-mini, unbatched,
   matches `pipeline/3_judge.py`'s rubric augmented with steering-direction
   awareness.  0-3 rubric.
3. **Effect (asynchronous, batched)** — GPT+Haiku ensemble, batched at
   `target_batch_size=10` (matches response judging's default), strict
   single-(cell, sign, strength) batches.  `--effect-mode` selects
   bidirectional (one ±3 prompt) vs separate-poles (two 0-3 prompts,
   signed combine = pos.mean - neg.mean) vs both.

**Skip rule (cost optimization)**: tiers 2 and 3 are **skipped at the
(cell, sign, strength) granularity** when `mean(coh) > skip_threshold`
(default 1.0).  Filtering on the strength-mean rather than per-record
coh reflects that incoherence is a strength-level property whereas
judge noise is per-question; averaging the K coh scores cancels noise
without losing the gross signal.  Skipped records get
`judges.persona.skipped_due_to_strength_mean_coh = True` (and same for
effect); `strength_mean_coh` is stamped on every record regardless.

**Records schema** (`{cell_dir}/records.jsonl`):

```json
{
  "strength": 1.0, "sign": +1, "question_idx": 7, ...,
  "judges": {
    "coherence": {"score": 1, "reason": "...", "model": "...", "ts": ..., "rubric_version": 1},
    "strength_mean_coh": 0.7,
    "persona": {"score": 2, "model": "...", "skipped_due_to_strength_mean_coh": false, ...},
    "effect": {
      "mode": "bidirectional",
      "bidirectional": {"scores": {"<model_a>": {"score": 2, "reason": "..."},
                                    "<model_b>": {"score": 3, "reason": "..."},
                                    "mean": 2.5}},
      "combined": 2.5,
      "skipped_due_to_strength_mean_coh": false, ...
    }
  }
}
```

**CLIs**:

- [`steering/run_sweep.py`](steering/run_sweep.py) — live sweep with
  judging.  `--no-live-judging` falls back to `NoOpJudgeDispatcher`
  (no API calls; useful when iterating on generation alone).
  `--judges-only` skips generation entirely and delegates to
  `post_judge.py` for retrospective fill-in.
- [`steering/post_judge.py`](steering/post_judge.py) — retrospective
  RP/effect fill-in on an existing experiment dir.  Same skip rule;
  `--rerun-coherence-with-model` populates `judges.coherence_alts[<model>]`
  for the coherence-model shootout (preserves the live coherence
  field).
- [`assistant_axis/cherrypick.py`](assistant_axis/cherrypick.py) — walks
  one or more experiment dirs, applies a default filter
  (`strength_mean_coh<=1.0 AND persona>=2 AND |effect.combined|>=2`)
  or a Python-expression filter, prints a summary table and pretty
  spotlights for the top N by |effect.combined|.

**Cost characteristic**: coherence dominates the bill because it's
unfiltered (every record gets a call).  RP+effect are skipped on
incoherent strengths, so they're typically <30% of total spend.
gpt-4.1-mini default is ~5x cheaper than Sonnet on the unfiltered hot
path; rationale documented in
[`/Users/roger/.cursor/plans/steering_judges_phase2_a54a28b6.plan.md`](.cursor/plans/steering_judges_phase2_a54a28b6.plan.md).

**Known judging artifacts:**

- **Concise-direction RP underscoring**: When the concise/verbose axis
  (or any axis that produces very short responses) is steered toward
  brevity, the RP judge gives low persona scores (e.g. 1/3) even when
  the response content is perfectly in-character.  The judge simply
  can't find enough textual evidence of persona in a 3-8 word response.
  This is a measurement ceiling, not a real persona loss.  Expect
  similar artifacts from any axis that drastically reduces output
  length (e.g. terse/elaborate, laconic/expansive if those ever exist).

**Sigma-rescaling (deferred)**: effect scores are stored on the raw
±3 scale.  A separate downstream tool will eventually consume
`records[*].judges.effect.combined` and rescale to standard deviations
of the response-judging distribution.  The default
`response_target_batch_size=10` (in
[`results_analysis/axis_judge_correlation.py`](results_analysis/axis_judge_correlation.py))
matches the steering effect default so the two distributions are
apples-to-apples in the same judging regime.

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

### Bidirectional steering scan (May 2026, default)

The steering-strength sweep was switched from a bottom-up
unidirectional scan to a **bidirectional middle-out scan** with two
stop conditions.  Implemented in
[`assistant_axis/steering_runner.py`](assistant_axis/steering_runner.py)
(`BidirectionalCursor`, `_run_bidirectional_cell`), wired through
[`steering/run_sweep.py`](steering/run_sweep.py).  Design plan:
[`/Users/roger/.cursor/plans/bidirectional_steering_scan_e93878b3.plan.md`](.cursor/plans/bidirectional_steering_scan_e93878b3.plan.md).

**Motivation.**  Pre-May-2026 sweeps started at `weakest_strength=1.0`
and walked up in `multiplier=1.189` steps until 2 consecutive
strengths crossed the coherence threshold.  Empirical review of the
14 production experiments before this change:
- Median strength at which `|effect| >= 0.5` is **3-4 multiplier
  steps** above weakest (so the bottom 3 strengths typically carry
  no measurable signal).
- Top strength reached was 15.96 (`architect_ecocentric −1`) -- well
  below the historical `max_strength=64.0` cap.  The UP cap is
  effectively informational; coh-stop is the binding constraint.
- Several cells need 6+ steps up before a meaningful effect appears
  (`mediator_truthful +1`, `journalist_callous +1`); they spend
  similar effort on the low-signal tail that gets thrown away.

**New scan.**

- Centre anchor: `s_init = weakest * multiplier**start_steps_up`
  (default `1.0 * 1.189**2 ≈ 1.414`, i.e. "2 steps above the
  historical floor").
- UP direction: step up from `s_init` in `multiplier` ratios; stop on
  `coh_stop_consecutive` consecutive `mean_coh >= coh_stop_threshold`
  strengths (defaults `2` and `1.5`, unchanged from pre-May 2026).
  Hard cap at `max_strength=64.0`.
- DOWN direction: step down from `s_init` in `multiplier` ratios;
  stop on `eff_stop_consecutive` consecutive
  `mean(|effect.combined|) < eff_stop_threshold` strengths (defaults
  `2` and `0.25`).  Hard floor at `min_strength=0.125` (3 octaves
  below historical weakest=1.0).
- State machine: `BothOpen → {UpBlocked, DownBlocked} → Done`.
  Sticky: a blocked direction never unblocks.  Single-pipeline
  generation alternating UP/DOWN while both open; once one side
  blocks, only the open side steps and we await its judges between
  steps (yellow-mode per direction).
- NaN handling: a DOWN strength whose effect was skipped due to
  high coherence (the `_StrengthState.judging_skipped` path) resolves
  the effect-mean future to **NaN**; the eff-stop tail check ignores
  NaN strengths rather than counting them as zero-effect (which
  would falsely block DOWN early).

**Config.**  All bidirectional knobs surface as both `sweep_cfg`
keys in `steering/run_sweep.py` and kwargs on
`run_steering_cell()`:

| key | default | meaning |
|---|---|---|
| `scan_mode` | `"bidirectional"` | switch to `"legacy_unidirectional"` for old behaviour |
| `start_strength_multiplier_steps` | per-cell heuristic — see below | `s_init = weakest * mult ** N` |
| `min_strength` | `0.125` | DOWN cursor floor |
| `eff_stop_threshold` | `0.25` | per-strength `mean(|effect.combined|)` below which counts toward DOWN stop |
| `eff_stop_consecutive` | `2` | how many consecutive sub-threshold strengths to require |
| (existing `weakest_strength`, `max_strength`, `multiplier`, `coh_stop_threshold`, `coh_stop_consecutive` unchanged) |

**Legacy mode.**  Setting `scan_mode: legacy_unidirectional` in
`sweep_cfg` (or passing `scan_mode="legacy_unidirectional"` directly
to `run_steering_cell`) restores pre-2026-05-14 behaviour
bit-for-bit on the records.jsonl path.  The summary.json now also
records `scan_mode` and `positions_mode` -- additive fields only,
nothing pre-existing is removed.  Test
`TestLegacyModeParity` pins the legacy invariants
(visited-strengths order, completion semantics, summary fields).

**Restart semantics.**  `summary.json` now persists `scan_mode`; on
restart, the persisted mode overrides the caller's request (with a
warning) so mid-run mode switches don't corrupt the schedule.  For
mid-run crashes with `records.jsonl` present but no `summary.json`,
the bidirectional cell re-walks the cursor past existing strengths
on each side (cursor is deterministic, so re-emission matches
prior emission), bootstraps `mean_coh_by_strength` and
`mean_abs_eff_by_strength` from the on-disk records, then
re-evaluates both stop conditions against the resumed history
before generating any new strengths.  Test
`TestBidirectionalRestart` covers this path.

**summary.json schema** (bidirectional cells):

```json
{
  "scan_mode": "bidirectional",
  "positions_mode": "all",
  "s_init": 1.413721,
  "weakest_strength": 1.0,
  "max_strength": 64.0,
  "min_strength": 0.125,
  "multiplier": 1.189,
  "start_strength_multiplier_steps": 2,
  "eff_stop_threshold": 0.25,
  "eff_stop_consecutive": 2,
  "coh_stop_threshold": 1.5,
  "coh_stop_consecutive": 2,
  "up_blocked_reason": "incoherent" | "max_strength_reached" | null,
  "down_blocked_reason": "sub_threshold_effect" | "min_strength_reached" | null,
  "up_blocked_at_strength": 8.0,
  "down_blocked_at_strength": 0.25,
  "records_in_scan_order": true,
  ...legacy fields (slot, layer, sign, n_records, etc.)
}
```

Downstream `results_analysis/*` plot scripts already group records
by `record["strength"]` so disk order doesn't matter -- the
`records_in_scan_order: true` field is purely documentary.

### Per-cell start-strength heuristic (May 2026)

Before 2026-05-24 the bidirectional sweep used a flat
`start_strength_multiplier_steps=2` (so `s_init ≈ 1.41` at
`weakest=1.0, mult=1.189`) for every cell.  Empirically this is
sub-optimal: a 310-cell audit
([`tools/analyse_start_strength.py`](./tools/analyse_start_strength.py))
showed **49% of all-mode** cells started in the productive band but
**only 11% of prefill-mode** cells did, with 89% of prefill cells
starting TOO_LOW (the downward walk immediately eff-stopped at a
barren region) and ~25% of `(slot=7, layer=49, all-mode)` cells
starting TOO_HIGH (the upward walk immediately coh-stopped at an
already-incoherent region).

The fix is a per-cell lookup table in
[`assistant_axis/sweep_start_heuristics.py`](./assistant_axis/sweep_start_heuristics.py),
keyed on `(positions_mode, slot, layer)` with per-mode defaults:

| mode | slot | layer | start_steps | s_init | rationale |
|---|---:|---:|---:|---:|---|
| all | 0 | 25 | 5 | 2.38 | eff weak (~0.4) at default → bump |
| all | 0 | 31 | 4 | 2.00 | eff borderline → small bump |
| all | 0 | 49 | 0 | 1.00 | high-effect already; lower to cut TOO_HIGH risk |
| all | 6 | 25 | 6 | 2.83 | weakest all-mode cell (eff ~0.26) |
| all | 7 | 25 | 7 | 3.36 | weak (eff ~0.26) |
| all | 7 | 49 | **−3** | 0.59 | hottest cell — coh tripped at default in ~25% of axes; needs sub-weakest start (requires `start_steps_up < 0`, allowed 2026-05-24; floor is now `min_strength`) |
| all | any other | | 3 (default) | 1.68 | |
| prefill | 0 | 25 | 9 | 4.95 | eff ~0.2; need 3-5× higher |
| prefill | 0 | 49 | 5 | 2.38 | layer-49 prefill is closer to cliff; modest bump |
| prefill | 6 | 25 | 9 | 4.95 | weakest prefill cell |
| prefill | 6 | 49 | **1** | 1.19 | **layer 49 prefill is already near coh cliff** (median coh ~0.5 at default); needs LOWER start than default |
| prefill | 7 | 25 | 9 | 4.95 | weak (eff ~0.18) |
| prefill | 7 | 49 | **1** | 1.19 | same as (prefill, 6, 49): near-cliff already |
| prefill | any other | | 6 (default) | 2.67 | |

Counter-intuitive pattern: **layer-49 prefill cells start LOWER than
the prefill default**, because prefill steering at deep layers
already has coherence near the cliff at weak strengths (median coh
~0.5 at `s_init = 1.41`).  Pushing prefill higher to fix the
weak-effect cells would push deep-layer prefill over.

**Negative start_steps.**  The cursor previously asserted
`start_steps_up >= 0`; relaxed 2026-05-24 because the (all, 7, 49)
override needs `s_init < weakest`.  The actual safety floor is
`s_init >= min_strength` (cursor's `__init__` enforces this).

**Override semantics.**  If the YAML explicitly sets
`sweep.start_strength_multiplier_steps`, that value wins for every
cell (so one-off experiments can still pin a value).  Per-cell YAML
override is *not* supported -- the table is the per-cell knob.

**Re-tuning.**  Re-run
[`tools/analyse_start_strength.py`](./tools/analyse_start_strength.py)
after every meaningful sweep batch (e.g. when sweep #2 second-role
data lands and we have ~2× more cells per bucket).  Inspect the
`(slot, layer, sign, mode)` table at the end of the report; if a
bucket newly shifts away from GOOD-dominant, add or adjust an
`OVERRIDES` entry and ship a one-line table edit.  No code or
runner changes needed.

**What about scan_mode legacy?**  The lookup applies only to
`scan_mode="bidirectional"` (the default).  Legacy unidirectional
sweeps use the cell-level `start_strength_multiplier_steps` value
directly without consulting the heuristic.

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

### Sweep log location (May 2026)

`steering/run_sweep.py` now auto-attaches a `FileHandler` to
`{output_root}/sweep.log` on both the parent and every spawned
worker, so the sweep log lives **inside the experiment dir**
alongside `config.json` / `questions.json` /
`persona_system_prompt.txt`.  Pre-May-2026 sweeps wrote logs
wherever the shell-redirect landed
(`/workspace/assistant-axis/sweep_<exp>.log`); those were moved
into their experiment dirs by
[`scripts/move_sweep_logs_into_experiment_dirs.sh`](scripts/move_sweep_logs_into_experiment_dirs.sh)
(one-shot, idempotent).

Future runs no longer need shell-redirect to capture logs.
Recommended runpod invocation:

```bash
python -m steering.run_sweep --config data/steering/configs/foo_v1.yaml &
tail -F /workspace/outputs/qwen-3-32b/steering/foo_v1/sweep.log
```

The two-run journalist + merchant sweeps preserve both runs as
`sweep_1.log` and `sweep_2.log` in the experiment dir; one-run
experiments use the canonical `sweep.log` name.  Workers and
parent share the same `sweep.log` via `mode='a'` line-buffered
appends -- safe on Linux because steering log records are well
under PIPE_BUF (4096B), so no mid-line interleaving is possible.

### Multi-config queue runner (May 2026)

`steering/run_sweep.py --config` now accepts **multiple `--config`
flags** (repeatable) and a `--config-list FILE` (one path per line,
`#` comments OK).  When given more than one config, the runner
loads the model **once**, then drains a single combined work queue
across every queued experiment.  Saves the cold-import + 60+ GB
model-load cost per extra config -- a 12-experiment batch goes
from `12 * (cold imports + model load + run)` to
`(cold imports + model load) + 12 * run`.

All queued configs must share `model_name` (one model fits in GPU
memory at a time); mixed-model queues raise loudly at startup.

Per-experiment `sweep.log` files still partition cleanly: each
work item carries its own `sweep_log_dir` and the worker swaps
`FileHandler`s (`_detach_sweep_log_filehandler` +
`_attach_sweep_log_filehandler`) when transitioning between
experiments.  Per-experiment `config.json` / `questions.json` /
`persona_system_prompt.txt` are still atomically written into the
experiment dir during parent-side setup, before the work queue
starts draining.

Typical batch invocation:

```bash
python -m steering.run_sweep \
    --config-list data/steering/configs/multi_cell_batch_v1.txt \
    --gpus 4
```

where `multi_cell_batch_v1.txt` is just:

```
data/steering/configs/anthropologist_helpful_v1.yaml
data/steering/configs/prodigy_harmless_v1.yaml
data/steering/configs/architect_ecocentric_v3.yaml
# ... 9 more
```

### Multi-GPU safety: baselines sentinel (May 2026)

Cells depend on their config's `baselines/records.jsonl` existing
(read by `build_baseline_lookup` to fill `baseline_response` in
judge prompts).  With `--gpus N` workers draining a shared queue,
a naive ordering would let worker B pop a cell for config X while
worker A is still computing config X's baselines -- the cell would
silently judge with blank `baseline_response`s.

The runner now:

1. **Sorts all baselines to the head of the queue.**  With N
   workers, the first N items popped are baselines, parallelising
   baseline throughput.
2. **Touches `baselines/.complete` after `compute_baselines`
   succeeds** (both the generation path and the "all already on
   disk" resume fast path -- the latter so resumed runs that
   missed the sentinel in a prior session still produce it).
3. **Workers wait on the sentinel before dispatching a cell**:
   if the cell's `baselines/.complete` doesn't yet exist, the
   worker polls every 2s with a 20-minute hard timeout.  Worst
   case waste with 4 GPUs + 12 configs is ~90 GPU-seconds per
   config = ~18 GPU-minutes across the batch; effectively zero
   compared to the ~hours of generation+judging.

Pre-May-2026 single-GPU sweeps never hit this race (one worker,
sequential order).  Pre-existing multi-GPU sweeps would have
quietly corrupted judge prompts on the first cell of each
experiment; the sentinel fix is also a pre-existing-bug fix.

### Exporting steering experiments to Google Sheets (May 2026)

`tools/steering_to_gsheet.py` turns a steering experiment output
directory into a tab on a Google Sheet.  One spreadsheet hosts an
"experiment set" (a collection of related experiments); within it,
one **tab per (role, axis)** holds the shared question set + baseline
responses as frozen header rows, with one **block per cell**
(`(slot, layer, positions_mode)`) underneath -- one block header
row carrying a machine-readable sentinel + human-readable summary,
followed by signed-strength data rows ordered from most negative to
most positive.

#### One-time Google Cloud setup

1.  Open https://console.cloud.google.com and create (or reuse) a
    project named something like "assistant-axis-sheets".
2.  Enable the **Google Sheets API** and **Google Drive API** in the
    project's API library.
3.  Configure the OAuth consent screen: choose **External**, add
    your own Google account as a test user.  No app verification
    needed -- only the test user can authenticate.
4.  Create credentials: **OAuth 2.0 Client ID** -> application type
    **Desktop app**.  Download the resulting JSON.
5.  Save the JSON as `~/.config/assistant-axis/google_credentials.json`
    (the path `assistant_axis/gsheet_auth.py` reads from).

The first invocation pops a browser tab for consent; the resulting
user token is cached at `~/.config/assistant-axis/google_token.json`
and auto-refreshes thereafter.  Both `google_credentials.json` and
`google_token.json` are gitignored (tree-wide pattern in
[`.gitignore`](.gitignore)) -- don't commit them.

#### Usage

```bash
# Dry-run (no auth required): writes a CSV next to the experiment dir
# and a layout summary to stderr.  Use to sanity-check the layout
# before pushing to Sheets.
uv run python tools/steering_to_gsheet.py \
    /workspace/outputs/qwen-3-32b/steering/architect_ecocentric_v2 \
    --dry-run

# Online: open or create a spreadsheet named "Steering May 2026",
# open or create a tab named "architect ecocentric_anthropocentric"
# (default derived from persona role + axis poles), push the data.
uv run python tools/steering_to_gsheet.py \
    /workspace/outputs/qwen-3-32b/steering/architect_ecocentric_v2 \
    --spreadsheet-name "Steering May 2026"

# Reuse the same spreadsheet for a sibling experiment (lands in a
# different tab because role+axis differ):
uv run python tools/steering_to_gsheet.py \
    /workspace/outputs/qwen-3-32b/steering/chef_helpful_v2 \
    --spreadsheet-id <id-from-first-run>
```

The tool prints the spreadsheet URL on success.

#### Rerun semantics

The default `--replace-experiment-blocks` rerun policy:

  - **Scans col A** of the existing tab for block sentinels in
    `[block:<experiment_id>/s<slot>_l<layer>/<positions_mode>]`
    format.
  - **Drops** existing blocks whose `experiment_id` matches the
    current run.
  - **Preserves** blocks for other experiments in the same tab
    verbatim (their row content is read back from the existing
    grid and re-written).
  - **Appends** the new payload's blocks at the bottom.

This makes re-exports idempotent for a single experiment_id while
keeping comparison blocks from sibling experiments (e.g. v1 vs v2)
intact.

Alternative policies:
  - `--append-only`: never drop existing blocks; just append.
    Useful when you want to keep a stale comparison block.
  - `--wipe-tab`: clear the entire tab (including frozen header)
    before writing.  Use when **questions themselves changed** across
    reruns -- the frozen-row compatibility check otherwise errors
    loudly to prevent silently misaligning rerun blocks against
    stale baselines.

#### Layout

  - **Frozen rows**: persona system prompt (row 1), per-column labels
    + question texts (row 2), baseline responses (row 3).
  - **Frozen column**: signed-strength (col A).
  - **Column groups**: outer group on aggregate cols B-E (mean_coh /
    mean_rp / mean_eff_signed / mean_abs_eff); per-question inner
    group on each `q*_coh` + `q*_rp` + `q*_eff` trio.  Roger can
    toggle each via the `+`/`-` indicators Sheets renders.
  - **Conditional formats** are tab-wide (column-spanning), bounded
    to ~10 + 11 * n_questions rules total well within Sheets' practical
    cap:
      - mean_coh / q*_coh: graded pink at >= 0.5 / 1.0 / 1.5.
      - mean_rp / q*_rp: faint orange at <= 1 (persona drift).
      - mean_eff_signed / q*_eff: sign-coloured graded green/blue at
        \|x\| >= 1 / 2 / 3.
      - q*_response: italic gray on TEXT_EQ `[skipped — incoherent]`
        (catches rows where the runner skipped RP+effect judging due
        to mean_coh >= skip_threshold).
  - **Widths**: 80px strength + aggregates, 400px response cols
    (wrap on), 60px score cols.

Block-header rows have non-numeric content in score columns so they
naturally don't match the numeric comparison rules.

#### Files

  - [`tools/sheet_layout.py`](tools/sheet_layout.py) -- pure data
    shaping (`build_tab`, `make_block_sentinel`, helpers).  No Google
    deps imported; unit-testable via
    [`tools/tests/test_sheet_layout.py`](tools/tests/test_sheet_layout.py).
  - [`tools/steering_to_gsheet.py`](tools/steering_to_gsheet.py) --
    CLI + Sheets I/O glue + merge/append/replace logic.  Merge
    correctness is tested via
    [`tools/tests/test_steering_to_gsheet.py`](tools/tests/test_steering_to_gsheet.py).
  - [`assistant_axis/gsheet_auth.py`](assistant_axis/gsheet_auth.py)
    -- thin `gspread.oauth` wrapper.

### Per-cell response curves vs steps-to-incoherence (May 2026)

[`results_analysis/steering_response_curves.py`](results_analysis/steering_response_curves.py)
plots one figure per steering experiment with 7 cells × 2 signs × 4
filter regimes per cell.  The X axis is "steps remaining to
incoherence cliff" (0 at the right edge), aligning every cell at its
own incoherence point so cells with different absolute cliff
strengths are visually comparable.

The four filter regimes per (cell, sign) probe whether the average
effect score at each strength is being dragged around by noisy
responses:

  - **(a) all responses** -- raw mean of effect scores at that
    strength.  Solid line, weight 1.6.
  - **(b) coh=0 only** -- mean over responses judged perfectly
    coherent.  Dashed (`(0, (5, 2))`).
  - **(c) rp=3 only** -- mean over responses strongly in-persona.
    Dotted (`(0, (1, 2))`).
  - **(d) coh=0 & rp=3** -- mean over responses that are both
    perfectly coherent AND strongly in-persona.  Dash-dot
    (`(0, (3, 2, 1, 2))`).

Diverging curves between (a) and (d) indicate strength regions where
some questions are giving misleading judge scores (incoherent or
off-persona responses contaminating the mean).  Convergence between
(a) and (d) is a sanity check that the population average reflects
the in-persona / coherent population.

Two side-by-side subplots:
  - **Left**: sign=+1 (steers toward `role_to` = negative pole).
  - **Right**: sign=-1 (steers toward `role_from` = positive pole).

Legend on the left subplot maps colour to `(slot, layer)`; legend on
the right maps linestyle to filter regime, so you don't have to
mentally cross-reference between the two.  See
[`results_analysis/tests/test_steering_response_curves.py`](results_analysis/tests/test_steering_response_curves.py)
for the aggregation-logic test suite.

Typical invocation:

```bash
uv run python results_analysis/steering_response_curves.py \
    outputs/qwen-3-32b/steering/anthropologist_helpful_v1
```

Output: `<experiment_dir>/response_curves.png` (override with
`--output PATH`).  PNG carries the standard
[plot-provenance metadata](#plot-provenance-metadata-mandatory-for-every-plot)
chunk.

### Swap-averaged effect judging (rubric v7, May 2026)

The bidirectional effect rubric has a strong label/order bias: the
judges treat whatever sits in the `[RESPONSE]` block as "more
thoroughly developed" relative to whatever sits in the `[BASELINE]`
block, even when the texts are swapped between those slots.
Empirically on anthropologist_helpful_v1 + prodigy_harmless_v1 the
bias dominated content-driven scoring at low/mid steering strengths
across both safety axes (helpful/unhelpful and harmless/harmful).
Diagnostic: [`tools/test_effect_order_bias.py`](tools/test_effect_order_bias.py)
swaps the [BASELINE] and [RESPONSE] contents in the rubric and
compares; a content-driven judge would produce `swap_score =
-straight_score`, a label-biased judge produces `swap_score =
straight_score`.

**Solution (rubric v7)**: every bidirectional effect judging call
fires BOTH variants and records the bias-cancelling estimator

```
averaged = (straight.mean - swap.mean) / 2
```

Under a content-driven judge `swap = -straight` so `averaged ==
straight` (signal preserved).  Under pure label bias `swap ==
straight` so `averaged = 0` (bias cancelled).  Real judges sit
between these extremes; the estimator subtracts off whatever
constant additive bias the judge has.

#### Schema (records.jsonl)

```json
"effect": {
  "bidirectional": {
    "straight": {"scores": {"<model>": {"score": int, "reason": str},
                            "mean": float, ...}},
    "swap":     {"scores": {"<model>": {"score": int, "reason": str},
                            "mean": float, ...}},
    "averaged": float
  },
  "mode": "bidirectional",
  "combined": float,           // = averaged (preferred) ->
                               //   straight.scores.mean (swap missing) ->
                               //   legacy bidirectional.scores.mean (v6)
  "rubric_version": 7,
  "skipped_due_to_strength_mean_coh": bool,
  "ts": float
}
```

Migration helper: [`assistant_axis.steering_judges.migrate_effect_dict_in_place`](assistant_axis/steering_judges.py).
Called automatically on every dispatcher write so legacy v6 records
end up in v7 shape the first time they're touched.  Idempotent:
already-v7 dicts return unchanged.

The bidirectional cursor's eff-stop predicate
([`assistant_axis/steering_runner.py`](assistant_axis/steering_runner.py))
reads `judges.effect.combined` to populate
`mean_abs_eff_by_strength`; no predicate-logic change was needed,
just the input semantics shifted from straight-only to
bias-cancelled.

#### Filling in older experiments

[`steering/post_judge.py`](steering/post_judge.py) gained two new
flags for the post-batch fill-in:

  - `--effect-swap-fill` runs ONLY the swap rubric on records with a
    straight half on disk, populates `bidirectional.swap` +
    `bidirectional.averaged`, sets `combined = averaged`, bumps
    `rubric_version` to 7.  Skips records lacking a straight half
    (skipped at sweep time due to high mean_coh).  Reuses existing
    straight scores -- no re-judging of straight.
  - `--apply-corrected-stop-cutoff` walks each cell's strengths in
    descending magnitude order and applies the live runner's
    bidirectional eff-stop predicate against the new averaged eff.
    The K-consecutive below-threshold streak (default K=2, threshold
    0.5) identifies the bias-floor anchor; strengths SMALLER than
    the anchor move from `records.jsonl` to `_excluded_records.jsonl`
    in the same cell dir.  `summary.json` gains `excluded_strengths`,
    `excluded_reason`, `excluded_eff_stop_threshold`,
    `excluded_eff_stop_consecutive`.  Idempotent on re-run.

Typical invocation:

```bash
uv run python steering/post_judge.py \
    --experiment_dir outputs/qwen-3-32b/steering/anthropologist_helpful_v1 \
    --effect-swap-fill \
    --apply-corrected-stop-cutoff
```

Cost reference: ~$10-15 API + ~3-10 min wall-clock per experiment
on the v6-anthropologist scale (14 cells × ~20 strengths).

#### Cost trade-off in the live pipeline

The dispatcher now fires 2× as many effect-judging calls per
strength (straight + swap × 2-model ensemble = 4 API calls instead
of 2).  Roughly cost-neutral with the buggy-predicate pre-v7
pipeline because the corrected stop fires earlier on the DOWN side
(observed ~36% strength-count reduction across anthropologist +
prodigy's swap descent), and that saving offsets the doubled
effect-judging cost.  See the original plan
[`swap_averaged_effect_judging_5af7c7a6.plan.md`](.cursor/plans/swap_averaged_effect_judging_5af7c7a6.plan.md)
for the cost band breakdown.

#### Analysis surfaces (no code changes needed)

Both [`tools/sheet_layout.py`](tools/sheet_layout.py) and
[`results_analysis/steering_response_curves.py`](results_analysis/steering_response_curves.py)
read `judges.effect.combined` for the eff value -- the same field
that now carries the averaged (bias-cancelled) estimator post-fill.
No analysis code changes were needed for v7 records; the change
percolates through the existing readers.

### Whitening / soft-shear defaults

**Scope (read this first).**  These defaults apply to **analysis-side
scripts** — ρ judging, `rho_by_layer`, `whitening_k_sweep`,
`gpt_sonnet_weight_sweep`, the canonical-angles pipeline, etc. —
where whitening sharpens the score↔projection correlation by
normalising the activation manifold.  **Steering work is always done
in raw model-activation space** (see *"Steering judges (Phase-2
architecture)"* above): `ActivationSteering` and `run_sweep.py`
consume the unwhitened `axis.pt` directly, and the transforms below
do NOT pass through to the steering side.  Don't apply them to
steering vectors.  (Same reason `pc_round_trip/launch_judge_runs.py`
keeps its own `DEFAULT_SHEAR_L=0` — see the caveat at the end of
this section for that specific case.)

For analyses going forward, the project defaults are:

| regime | default | constant |
|---|---|---|
| **soft-K whitening** | **K=2** | `results_analysis.canonical_angles.whitening.DEFAULT_SOFT_K` |
| **soft-shear (top-L pooled)** | **L=3** | `results_analysis.canonical_angles.whitening.DEFAULT_SOFT_SHEAR_L` |
| **primary recommended whitening regime** | **`soft_shear=3`** | `results_analysis.canonical_angles.whitening.DEFAULT_WHITENING_SPEC` |

`DEFAULT_SOFT_K=2` was set in Apr 2026 after the LKM grid sweep on the
33 desc+inst + 12 response = 45-axis set with the inst-tiebreak
weighting (prior hardcoded default was K=3, from the older K-sweep
parabola fit done before soft-shear was in the toolbox).

`DEFAULT_SOFT_SHEAR_L=3` (and `DEFAULT_WHITENING_SPEC="soft_shear=3"`)
were bumped from `L=2` / `"soft_shear=2"` on May 11 2026 after the
35-axis di-cohort `shear_l_sweep` at slots 3 / 6 / 7 with the new
cohort-mean blended-curve overlay
(`results_analysis.shear_l_sweep` + `judge_score_combine.cohort_mean_curves`,
0.80·rs + 0.20·di per axis, 5x cross-axis weight on primary axes).
Per-axis paired t-tests over 35 axes show:

| slot | L=1 vs L=2 | L=1 vs L=3 | L=3 vs L=4 |
|---|---|---|---|
| 3 | tied (p=0.37) | tied (p=0.54) | borderline (p=0.051) |
| 6 | **L=2 dip** (Δ=−0.017, p=0.019) | tied (p=0.29) | **cliff** (Δ=−0.023, p=0.0006) |
| 7 | tied (p=0.68) | tied (p=0.92) | **cliff** (Δ=−0.020, p=0.0004) |

L=3 is at-or-above the optimum on every slot, never significantly
worse than L=1, and the universally-significant harm transition is
L=3 → L=4 across all three slots — so L=3 is the highest "safe"
shear depth with no significant ρ harm anywhere.  Tied with L=1 on
ρ at most slots; goal/no-goal subspace cleanliness (more top-aligned
canonical-angle pairs orthogonalised) is the tie-breaker that picks
L=3 over L=1.

Use the constants instead of literal integers in CLI defaults::

    from results_analysis.canonical_angles.whitening import DEFAULT_SOFT_K
    p.add_argument("--whiten_K", type=int, default=DEFAULT_SOFT_K)

When fitting:

```python
from results_analysis.canonical_angles.whitening import (
    DEFAULT_SOFT_K, DEFAULT_SOFT_SHEAR_L, fit_shear, fit_whitening,
)
from results_analysis.canonical_angles.data import build_goal_nogoal_subspaces

# Primary: soft-shear at L=3, fitted on combined r+t goal/no-goal subspaces.
A_g, A_n = build_goal_nogoal_subspaces(data_dir, slot, layer, kind="combined")
shear = fit_shear(A_g, A_n, L=DEFAULT_SOFT_SHEAR_L)
M_done = shear.apply(M_raw)

# Alternative when shear isn't available: soft-K whitening at K=2.
basis = fit_whitening("soft_K", pool, K=DEFAULT_SOFT_K)
M_done = basis.apply(M_raw)
```

**Caveats**:

- The optimum varies by slot: slot 3 has a broad L=1..L=5 plateau,
  slot 6 has a real L=2 dip but L=1 ≈ L=3, slot 7 has L=1 ≈ L=2 ≈ L=3.
  L=3 is defensible at all three; bespoke per-slot tuning could
  squeeze ~0.005 ρ at slot 7 by picking L=1 instead.
- The L=3 → L=4 cliff is sharp and universal -- do **not** bump
  `DEFAULT_SOFT_SHEAR_L` above 3 without re-running the cohort-mean
  paired-t analysis.
- `pc_round_trip/launch_judge_runs.py` has a separate
  `DEFAULT_SHEAR_L=0` (raw) that is *intentionally* not tied to this
  constant -- the pc_round_trip canonical experiment runs raw on
  purpose.  See the comment block at that constant for the rationale.

**Cache hygiene**: when changing these defaults, flush `rho_by_layer.json`
(the only auto-loading cache that interleaves K values across runs).
Per-axis `correlations.json` files are tied to specific
``--whiten_K`` invocations and should NOT be deleted -- they belong to
historical runs and re-running them is expensive.

### Axis-geometry tool: `axis_cosine_seriation.py`

Pairwise `|cos|` heatmap of every axis in the chosen pair-list cohort,
reordered by **optimal leaf ordering** (`scipy.cluster.hierarchy.
optimal_leaf_ordering`, Bar-Joseph 2001) on the `(1 − |cos|)`-distance
dendrogram so semantically-related axes sit adjacent and clusters
land block-diagonal.  Promoted from a one-off May 2026 geometry
exploration; full operational docs in
[`results_analysis/README.md`](results_analysis/README.md#axis_cosine_seriationpy).

When to use:

* **Sanity-checking a new cohort** — does any pair land on top of an
  existing axis?  Seriation pushes near-duplicates to immediate
  neighbours where they're easy to spot.
* **Comparing the raw and L=3 soft-sheared frames** — the script
  takes ``--L 0`` for raw and ``--L 3`` (default) for canonical;
  same cohort run both ways gives the "does whitening tighten or
  create clusters?" picture.  On the clean 60-pair cohort the
  honest answer is "tightens by ~10 % mean |cos| (0.245 → 0.222)
  but doesn't manufacture new structure" — consistent with the
  whitening-as-sharpener story above.  **Stronger evidence**: with
  the hand-curated 8-block semantic ontology turned on (default),
  the OLO seriation on the L=3 frame recovers the ontology
  **exactly** (8 runs / 8 blocks / 0 splits) while the raw frame
  fragments it into **21 runs / 8 blocks / 7 splits**.  Whitening
  is aligning the data-driven geometry with the human ontology by
  attenuating the dominant goal/non-goal direction that otherwise
  dominates the cosine structure.
* **Picking new traits to add** — the brightest off-diagonal blocks
  in the canonical frame mark dense regions of the axis space where
  *additional* traits won't add effective dimensionality.  Pair
  with the PCA-of-entity-pool / yield analysis from
  [`data/traits/instructions/TRAITS_TO_ADD.md`](data/traits/instructions/TRAITS_TO_ADD.md)
  for the full "which direction is empty?" picture.

JSON sidecar carries the OLO leaf order plus cluster memberships at
five thresholds (0.7 / 0.6 / 0.5 / 0.4 / 0.3) for downstream
consumption.  Provenance envelope and PNG `Inputs` chunk are
populated -- registered in the writer-side audit list above.

---

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

## Trait/role name collisions and the `name|R` / `name|T` convention (May 2026)

**The bug we keep almost making.** Nine names appear in BOTH the trait
and role lists (`ascetic`, `contrarian`, `cosmopolitan`, `generalist`,
`pacifist`, `patient`, `perfectionist`, `romantic`, `stoic`).  In any
data structure that mixes traits and roles, the bare name is **NOT a
unique identifier**.  Historic code did things like:

```python
merged = {}
for entry in trait_scores: merged[entry["name"]] = entry["score"]
for entry in role_scores:  merged[entry["name"]] = entry["score"]  # silently overwrites!
```

…which silently dropped one side of every collision and biased downstream
ρ calculations by ~1.5%.

### Disambiguator: `assistant_axis.entity_id`

Use [`entity_id(name, kind)`](assistant_axis/entity_id.py) whenever a
data structure could plausibly contain entries from both kinds (dict
keys, set members, JSON cache keys, sorted name lists for rho, ...).
Format: pipe-suffixed single-letter kind tag.

```python
from assistant_axis import entity_id, parse_entity_id, display_label

entity_id("patient", "roles")    # "patient|R"
entity_id("patient", "traits")   # "patient|T"
parse_entity_id("patient|R")     # EntityId(name='patient', kind='roles')
display_label("patient|R")       # "patient"   ← bare name, for plots
```

Pure-kind contexts (e.g. inside `runpod_workspace/.../{roles,traits}/`,
or a per-cohort `haiku_responses_traits_*/` cache) keep bare names —
the kind is implicit in the path.  Disambiguation is for the **mixing
layer**.

### Display-side rule (plots, legends, console output)

Plots and labels show the **bare name only**; kind is encoded
*visually*.  Two collision-name dots in the same scatter is fine:
same label "patient", different colour.  Project convention (locked
in `assistant_axis.plot_palette`):

| kind  | fill colour       | text colour | marker | text style |
| ----- | ----------------- | ----------- | :----: | ---------- |
| trait | `lightgrey`       | `dimgrey`   | `o`    | upright    |
| role  | `lightsteelblue`  | `navy`      | `s`    | italic     |

Helpers: `kind_color(kind)`, `kind_text_color(kind)`, `kind_marker(kind)`,
`kind_text_style(kind)`, `display_label(eid)`.  Reference
implementation: `results_analysis/pair_slice_plots.py` lines 336–410.

### File-name vs display-name convention (underscores ↔ spaces)

Many entities have multi-word names that exist in two forms:

| form          | example                          | where used                         |
| ------------- | -------------------------------- | ---------------------------------- |
| **file-name** | `aligned_artificial_intelligence` | every JSON key, dict key, set member, filename, scoring cache, ρ intersection key |
| **display-name** | `aligned artificial intelligence` | plot labels, axis annotations, **LLM rubric / prompt body**, console output for humans |

**Rule (Convention 2): file-name everywhere except display sites.**
Use the underscore form for any data-structure key, ID, intersection
operand, or persistence key.  Convert to space-form ONLY at the
boundary where you render the name to a human OR an LLM — and do
that locally (via the `display_form_name` helper or
`name.replace("_", " ")`).  Never the reverse: a display-form string
should never be used as a dict key, JSON key, or rho intersection
input.

**LLM prompts are display sites too.**  An LLM reading
`aligned_artificial_intelligence` parses it less naturally than
`aligned artificial intelligence` (extra tokens, distracts from the
concept).  Project-wide convention is therefore the same as for
plot labels: file-form is the canonical key, display-form is what
goes into the LLM's mouth.  Every rubric / prompt builder MUST
apply `display_form_name(...)` to entity names before injecting
them into the prompt body, examples list, or axis-name header.

Multi-word entity census (qwen-3-32b Roger 8slot corpus): 12 of 303
traits + 4 of 281 roles = 16 of ~584.  Examples: `systems_thinker`,
`kind_to_animals`, `stream_of_consciousness` (traits);
`aligned_artificial_intelligence`, `paperclip_maximizer`, `coral_reef`,
`devils_advocate` (roles).

**The audit (May 2026)** found legitimate `_ → space` conversion
sites in the codebase, all in display/annotation code:
`canonical_angles/ca1_plane.py:193`, `pair_slice_plots.py:418`,
`infer_axis_description.py:164` (PC-describer LLM prompt),
`rubric_v1_v2_compare.py:209`, `regenerate_role_instructions.py:47`
(data-prep LLM prompt), and the `display_form_name()` helper.

**LLM rubric audit (May 2026)** of every prompt builder in the
project (8 callsites; each is now annotated in-file with a brief
display-form note that points back to this section):

| Prompt builder | Status | Notes |
| --- | --- | --- |
| `axis_judge_correlation.py:build_static_prompt` | **fixed v3 (May 2026)** | Was injecting file-form `{name}`, examples, axis_name; now wraps each in `display_form_name(...)`. |
| `axis_judge_correlation.py:build_response_batch_prompt` | **fixed v3 (May 2026)** | Body anonymises the entity (v2); shared header still injected examples + axis_name in file-form, now in display-form. |
| `data_analysis/score_combinations.py:build_user_message` | **fixed (May 2026)** | Now wraps `combo['role']` / `combo['trait']` in `display_form_name(...)` at prompt injection. |
| `data_analysis/regenerate_role_instructions.py:build_eval_prompt` (+ Christina/Roger variants) | OK | Already used `role_display_name(stem)` (file-stem -> display, with overrides for `devil's advocate`). |
| `data_analysis/regenerate_trait_instructions.py:build_eval_prompt` (+ instruction variants) | OK | Already uses pre-stored display-form `positive_label` from each trait JSON (e.g. `stream-of-consciousness`). |
| `results_analysis/infer_axis_description.py:build_prompt` (PC-describer) | OK | Already used `_display_label` helper for trait/role labels. |
| `results_analysis/standardize_axis_spec.py:build_prompt` | OK | Operates on long-form pole *descriptions*, not bare entity names. |
| `assistant_axis/steering_judges.py` rubrics | **open** (low-priority) | Uses `persona.role` / `SteeringSpec.{axis_name,pos_label,neg_label}` as-is from steering config.  If config supplies file-form, prompts leak underscores.  Tracked separately; sweep configs to date have used clean labels.  Fix at SteeringSpec/PersonaSpec construction in `steering/{run_sweep,post_judge}.py`. |

**Cross-axis rejudge surface for the v3 fix** (35 distinct axis
dirs on disk across all pair lists, 57 distinct pole pairs across
all 9 `pair_list*.json` files): only **1** axis has multi-word
pole names — `systems_thinker_vs_analytical`.  So the
``axis_name`` + examples header leak is a single-axis problem
across the entire project, not just the v2 12-axis production
set.  The per-entity ``{name}`` leak in static-mode prompts
affects the 16 multi-word entities (12 traits + 4 roles, ~2.7%
of the corpus) on every axis.  Surgical rejudge cost: ~$0.49 for
the per-entity leak across all 35 axes; +$1.48 for the 1-axis
static header leak; +~$54 if redoing that one axis's response
mode at B=7 GPT + Haiku-tiered.

No internal pipeline produces or consumes display-name keys — every
ρ intersection, every set membership check, every cache key uses
the file-name form.

**Why no internal auto-conversion?**  Inside the pipeline we never
convert: every key is in file-name form by construction and any
mismatch is a bug, not user error.  Auto-coercing
`"aligned artificial intelligence"` → `"aligned_artificial_intelligence"`
in internal code obscures the mismatch and would fail anyway on
names where the user's word breaks differ
(`"obama_administration_health_team"` vs `"Obama administration's
health team"`).

**External boundary hardening (May 2026)**: the two ingest points
that accept names from outside the pipeline normalise display-form
input to file-name form with a WARNING, since the typical leak
(spaces, hyphens, capitals, apostrophes) is mechanical and easy to
detect:

1. `axis_judge_correlation.py --rejudge_names` —
   [`_parse_rejudge_names`](results_analysis/axis_judge_correlation.py)
   runs each name through
   [`normalize_to_file_name`](assistant_axis/entity_id.py); logs
   `WARNING: --rejudge_names: coerced N display-form entries...`
   when the input wasn't already canonical.
2. `optimal_axis_for_judge.py --scores_file` —
   [`_normalize_scores_file_keys`](results_analysis/optimal_axis_for_judge.py)
   runs every JSON key through `normalize_to_file_name`, logs
   `WARNING: --scores_file ...: coerced N display-form key(s)...`
   on conversions, and `SystemExit`s if two keys collide under
   normalisation (rather than silently dropping one).

`name|R` / `name|T` disambiguated ids are passed through unchanged
in both paths.  See `tests/test_entity_id.py::TestNormalizeToFileName`
and `tests/test_optimal_axis_for_judge_normalize.py` for the full
behaviour spec.

**Defensive checks already in place** for this risk class:
`StaleSchemaError` enforcement on AT-RISK static caches
(`assistant_axis/judge_loaders.py:478–533`), per-axis
`len(common) < 3 or 5` floors at every ρ site,
[`tools/lint_kind_collision.py`](tools/lint_kind_collision.py) AST
walker, and
[`assistant_axis/tests/test_collision_regression.py`](assistant_axis/tests/test_collision_regression.py)
covering 5 patterns × 9 collision names.

### Common-pitfall callout (read this before merging trait + role data)

Whenever you see this pattern in code:

```python
for kind in ("traits", "roles"):
    for entry in load_scores(axis, kind):
        merged[entry["name"]] = entry["score"]
```

**STOP.**  Replace with:

```python
from assistant_axis import entity_id

for kind in ("traits", "roles"):
    for entry in load_scores(axis, kind):
        merged[entity_id(entry["name"], kind)] = entry["score"]
```

…otherwise you are dropping nine traits or nine roles per axis.  The
canonical lint regex (`for .* in \("(traits|roles)", "(traits|roles)"\):`)
catches the dual-iteration pattern; pair every match with an
`entity_id(...)` call before the merge.

**At-risk cache families** (write disambiguated keys in mixed-kind
contexts):
* `<axis>/<judge>/scores_descriptions.json`
* `<axis>/<judge>/scores_instructions.json`
* `<axis>/<judge>/projections.json`
* `<axis>/<judge>/correlations*.json`

The per-cohort caches (`<axis>/<judge>_responses_<kind>_b<B>{...}/scores_responses*.json`)
are kind-pure by directory — bare names there remain correct.  Same for
`runpod_workspace/.../{roles,traits}/...` and any single-kind notebook.

### Producer-side / consumer-side audit (May 2026)

* **Producer**: `results_analysis/axis_judge_correlation.py` — writes
  the at-risk cache families above.  After the May 2026 fix it stamps
  disambiguated keys + `schema_version: 2`.  Old v1 caches are loud-rejected
  on read with a concrete regenerate-via command.
* **Consumers**: ~13 result-analysis scripts that merge trait + role
  data (see `phase3_consumers` in
  `trait_role_name_disambiguation_06e61abc.plan.md`).  Each migrated to
  `load_response_scores(...)` and `entity_id(name, kind)` keys.

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

## Judging cost model (project-wide reference)

The single source of truth lives in
[`results_analysis/plot_batch_size_quality_vs_cost.py`](results_analysis/plot_batch_size_quality_vs_cost.py)
(constants and the Pareto plot) with the fuller narrative in
[`results_analysis/README.md`](results_analysis/README.md) under
"Cost model: GPT-4.1-mini responses-mode judging".  This section is a
quick reference for cost-scoping questions; numbers were re-derived
empirically on **2026-05-09** via a 240-batch tiktoken dry-run on
real cached prompts + responses (the previous documented numbers
under-counted real responses by ~22× and made all costs look ~5× too
low).

**2026-05-10 full cross-axis validation:** re-ran the dry-run via
[`tools/dry_run_response_token_count.py`](tools/dry_run_response_token_count.py)
on **all 12 v2 axes × both cohorts** = 295,258 reconstructed B=10
batches, prompt-rebuilt under the current (v2) rubric, all
tiktoken-counted with `o200k_base`.  Per-axis cost (`measured /
$49.40`) came in at **mean 1.006 ± σ 0.013, range [0.989, 1.034]**.
The model is empirically correct within ±3% per axis and within 0.6%
on average -- no recalibration applied.  Per-axis spread:

| axis                                  | n_batches | cost    | ratio |
| ------------------------------------- | --------: | ------: | ----: |
| helpful_vs_unhelpful                  | 24,599    | $50.33  | 1.019 |
| harmless_vs_harmful                   | 24,630    | $50.43  | 1.021 |
| honest_vs_dishonest                   | 24,621    | $49.32  | 0.998 |
| truthful_vs_deceitful                 | 24,626    | $49.47  | 1.001 |
| guileless_vs_scheming                 | 24,600    | $49.47  | 1.001 |
| egalitarian_vs_elitist                | 24,602    | $49.63  | 1.005 |
| progressive_vs_conservative           | 24,590    | $49.29  | 0.998 |
| concise_vs_verbose                    | 24,589    | $48.86  | 0.989 |
| ecocentric_vs_anthropocentric         | 24,613    | $51.08  | 1.034 |
| improvisational_vs_methodical         | 24,598    | $49.43  | 1.001 |
| relativist_vs_absolutist              | 24,601    | $49.41  | 1.000 |
| systems_thinker_vs_analytical         | 24,589    | $49.71  | 1.006 |

Reproduce via `uv run python tools/dry_run_response_token_count.py
--axis <name> --kind both --quiet`; with `--quiet` omitted, dumps
per-cohort histograms for diagnostic use.  The dry run rebuilds each
batch's prompt with the *current* RUBRIC_RESPONSE_BATCH and tiktoken-
counts it -- so re-running after any rubric change re-validates the
model automatically.

### Per-batch token model (B=10)

* Input: ≈ 320 (header) + 10 × 438 (per item) ≈ **4,700 tokens**.
* Output: ≈ **80 tokens** (median 78, σ 14).
* Header is only ~7% of input; **items dominate** because real
  Qwen-3-32B responses are ~435 tokens each (capped at the 512-token
  generation limit; ~71% hit cap).

### Per-axis cost (both cohorts combined, full volume)

At B=10: **115.6M input + 1.97M output** tokens across ~24,605
batches.

| B  | GPT-4.1-mini | Haiku-4.5 (full / q9)       | Sonnet-4 (full / q9)        |
| -- | -----------: | --------------------------: | --------------------------: |
| 5  |       $55.69 |             $164.55 / $54.85 |       $493.64 / $164.55 |
| 7  |       $52.10 |             $154.74 / $51.58 |       $464.21 / $154.74 |
| 10 |       $49.40 |             $147.38 / $49.13 |       $442.14 / $147.38 |
| 15 |       $47.30 |             $141.66 / $47.22 |       $424.97 / $141.66 |

* `q9` = `--question_subsample_modulo 9` → ~1/3 fraction of full
  volume.  ~4% smaller than tiered-default `_b10` cohorts in practice
  because uniform mod-9 doesn't escalate RP-depleted entities to tier
  2/3.
* Pricing (per 1M tokens, 2026 rates): GPT-4.1-mini $0.40/$1.60;
  Haiku-4.5 $1.00/$5.00; Sonnet-4 $3.00/$15.00.
* **Haiku/Sonnet token scales** (empirical from 2026-05-11 5c.1/5d.1
  full-sweep totals: 11 v2 axes × 2 cohorts, 372,652 GPT calls +
  8,927 Haiku calls; previously the 5d.0 canary's smaller sample
  gave 1.17/2.00, which the full sweep now refines).  These are
  PER-CALL ratios and are approximately B-invariant: at B=7 they
  measure (1.087, 2.165) and the header-amortisation analysis
  (1.35× header_chunk × 320 tok header + 1.07× item_chunk × B × 438
  tok items) predicts (1.089, 2.165) at B=10, so the same
  constants apply to the B=10 Pareto plot anchor below.

  - `HAIKU_INPUT_SCALE = 1.09`: Anthropic tokenizer ~9% chunkier than
    `o200k_base` on response prompts dominated by raw response text;
    much smaller than the 1.35× seen on static-mode prompts where
    the rubric-laden header dominates.
  - `HAIKU_OUTPUT_SCALE = 2.17`: Haiku produces ~2.17× the output
    tokens of GPT-4.1-mini per call (160.2 vs 74.0 per call in the
    5c.1/5d.1 full sweep at B=7; 2.19 in the 5d.0 canary; 1.93× in
    static-mode).  The previous 2.00 mid-estimate was conservative
    on the output side; the net effect of moving (1.17, 2.00) →
    (1.09, 2.17) is a **~5% decrease** in predicted Haiku cost at
    B=10 (input contribution drops faster than the output
    contribution rises, since input is ~5.7× the output for Haiku
    given the $1.00/$5.00 rate split and 115.6M/1.97M token totals).

  Sonnet uses the same Anthropic tokenizer family (input scale
  carries over directly); Sonnet's verbosity is **not** measured in
  this project — we use the Haiku output scale as a placeholder
  (Sonnet is typically ≥ as verbose as Haiku, so this is more likely
  an under-estimate than over-estimate).  See
  `results_analysis/plot_batch_size_quality_vs_cost.py`
  `HAIKU_INPUT_SCALE` / `HAIKU_OUTPUT_SCALE` constants for the
  authoritative values used in the Pareto plot.

  Note: the 5d.1 actual-over-expected ratio of 1.21 (Haiku b7_t3
  residual sweep) is **not** explained by these scale revisions
  (which net to ~5% LOWER cost, not 21% higher); that ratio traces
  to the 5d.1 launch-script's budget formula and is a separate
  reconciliation item.

### B-curve takeaway (motivates the May 2026 B=10 → B=7 default switch)

The B=15 → B=5 cost spread is only **1.18×** ($47.30 → $55.69), not
~2× as the old (broken) model claimed.  Most cost is `items × per-item-rate`
($43.10 floor) regardless of B.  B=7 vs B=10 marginal ≈ $2.70/axis
for +0.004 ρ uplift (~$7 per +0.01 ρ — the best step on the new
curve).  See
[`./roger/axis_judge_experiments/batch_size_curve_8slot/batch_size_cost_vs_quality.png`](./roger/axis_judge_experiments/batch_size_curve_8slot/batch_size_cost_vs_quality.png).

### Steering effect-judge B post-mortem (2026-05-21)

**Stale-local-default bug.**  The library-level canonical constant
`assistant_axis.judge_batch.RESPONSE_BATCH_SIZE = 7` and the steering
re-export `assistant_axis.steering_judges.DEFAULT_TARGET_BATCH_SIZE`
(pinned to that constant) were correctly updated to **B=7** on
2026-05-11.  However, `steering/run_sweep.py` had its own *local*
argparse default `--target-batch-size = 10` that silently overrode
the canonical value when launching multi-cell sweeps.  As a result,
the `multi_cell_batch_v1` (all-mode, 2026-05-18) and
`multi_cell_batch_v1_prefill` (2026-05-19/20) sweeps both ran their
live effect/coh/rp judging at **B=10**, not the canonical B=7.

The error was caught on 2026-05-21 when prepping a `max_strength=128`
fill-in sweep.  Records from the two prior multi_cell sweeps remain
on disk at B=10 and are left as-is; the fill-in past the old cap was
run at the canonical B=7 (Plan A: only new strengths beyond the old
`max_strength=16` cap were generated, so most cohorts have a
B=10/B=7 split at the s ≈ 16 transition; cells that stopped at the
coherence cliff before reaching the old cap remain pure B=10).

**Fixes applied 2026-05-21:**

- `steering/run_sweep.py`: argparse default for `--target-batch-size`
  changed `10 → 7` with a HARD-RULE-aligned warning in the help text
  ("the previous value of 10 is OBSOLETE and was the source of a
  several-hundred-dollar mis-judging incident; do not revert without
  explicit user approval").
- `assistant_axis/steering_runner.py`: early-skip path in
  `run_steering_cell` now distinguishes `up_blocked_reason ==
  "max_strength_reached"` (fall through to extend the UP-walk when
  the caller has raised `max_strength`) from
  `up_blocked_reason == "incoherent"` (skip as before).  Existing
  records are preserved end-to-end via the `done_pairs` gate at
  `steering_runner.py:981, 1189`; only new `(strength, q_idx)`
  pairs beyond the old cap are generated and judged.  Symmetric
  handling on the DOWN side if `min_strength` is ever lowered.
- `assistant_axis/tests/test_steering_runner.py`: two new tests
  (`test_skip_when_previous_cap_below_new_max_strength_with_incoherent_stop`
  and `test_cap_extension_fallthrough_when_previously_capped`) pin
  both branches of the early-skip decision so the cap-extension
  semantics can't regress.

**Pattern to watch for.**  When a project-wide canonical constant
is changed (here `RESPONSE_BATCH_SIZE`), grep all argparse
`default=…` values that name the same parameter and verify they
pin to the constant rather than hard-code a literal.  Local
argparse defaults that hard-code a number are silent stale-value
traps: they don't error, they just quietly do the wrong thing in
production.

### Static-mode cost model (descriptions + instructions)

Static-mode prompts are vastly cheaper per call than response-mode
because there are no per-batch response items — just the rubric
header plus one entity description (or one instruction list).
Empirical from the 2026-05-10 GPT canary on `concise_vs_verbose`
(36 single-shot calls, descriptions + instructions for 18 collision
entities = 12,664 input + 2,250 output tokens):

| Component | Tokens | Notes |
| --------- | -----: | ----- |
| Input per call  | **352** | rubric header (~210) + entity desc/instr (~110) + scoring instruction (~30); range narrow because content length is bounded by description format |
| Output per call | **63**  | reasoning (2-3 sentences) + `SCORE: <int>` line; mean 62.5, σ small |

Per-call cost (single-shot, no batching):

| Judge | $ / call | Source |
| ----- | -------: | ------ |
| GPT-4.1-mini  | **$0.000242** | `352 × $0.40/1M + 63 × $1.60/1M` |
| Haiku-4.5     | **$0.000667** | `352 × $1.00/1M + 63 × $5.00/1M` |
| Sonnet-4      | **$0.002001** | `352 × $3.00/1M + 63 × $15.00/1M` |

**Per-cell scaling for surgical static-mode rejudges** (one cell =
one (axis, judge) pair, two modes = descriptions + instructions, N
entities):

```
cost ≈ 2 × N × cost_per_call
```

For Phase 5b (N=18 collision entities, 12 axes, GPT + Haiku): per
cell = 36 calls = $0.0087 (GPT) or $0.0240 (Haiku).  Total Phase 5b
expected: 12 × ($0.0087 + $0.0240) ≈ **$0.39**.  Hand-rolled
estimates that pre-dated this empirical canary (e.g. an early
$0.012/cell guess) are roughly 1.5× pessimistic — re-derive from the
table above when budgeting, not from older estimates.

When estimating costs for a new judging run, multiply `cost_per_axis_for_(judge, B, subsample)`
by the number of axes and rubrics involved.  When estimating for a
*surgical* rejudge of N entities, scale by `N / 280` (roles) or
`N / 300` (traits) of the per-axis cost (Phase 5d-i and 5d-ii in
the disambiguation plan are worked examples).

**Roles-vs-traits response-length asymmetry (response-mode only).**
At the same B and the same N entities, response-mode judging of
ROLES costs ~**1.22×** more per entity than TRAITS, because
Qwen-3-32B produces ~22% longer responses when role-played
(persona-immersive narratives) than trait-modulated (clipped
behavioral answers): mean 2094 vs 1718 chars / median 2351 vs 1842
chars on the 29 roles + 14 traits 5d.1 cohort (14,500 + 7,000
responses).  Per-call judge output is unaffected (the judge emits a
fixed-format score per response regardless of cohort) — only
per-call input scales with response length.

This is **why 5d.1 came in 1.21× over its hand-tuned 50/50 budget
split**: the 29/14 entity split implies a *cost* split of
`(29 × 1.22) : 14 ≈ 2.53 : 1`, not the `$1.50 : $1.55 ≈ 0.97 : 1`
the launch script assumed.  Re-applying the 1.22× factor to the
5d.1 numbers gives `$2.50/roles + $1.00/traits = $3.50/axis ⇒
$38.50 total`, vs the observed $40.45 (within ~5%, well inside
noise).

The canonical constant is
`assistant_axis.judge_pricing.ROLES_RESPONSE_LENGTH_FACTOR = 1.22`,
with a convenience splitter
`surgical_rejudge_cost_split(per_axis_cost, n_roles, n_traits,
mode="response")` returning the per-cohort budget shares.  Static
mode (descriptions / instructions) is **not** affected — there are
no responses in those prompts, so call `surgical_rejudge_cost_split(
..., mode="static")` to get an N-weighted (1.0×) split for static
rejudge runs.

The 1.22× asymmetry is also part of why the project-aggregate
per-axis cost (`B10_GPT_INPUT_M_TOK = 115.6` for both cohorts
combined) silently absorbs it — full-volume budgets are correct
without an explicit factor because the (300 traits + 280 roles)
~50/50 entity split implies a (1.0:1.22)/(1.0+1.22) = 55/45
implicit weighting that's already baked into the empirical token
totals.  Only *off-default* cohort splits (surgical rejudges,
single-mode runs) need to apply the factor explicitly.

### Budget cap recommendation (safety net)

The `--budget_usd` hard cap is a safety net against catastrophic
miscalculation (a bug, a wrong model, an unexpected prompt blowup
— not normal variance).  Pick it generously relative to the
expected cost, but tightly enough that a real surprise gets caught
before the bill detonates.

Post-May-2026 calibration (5c.1 GPT full sweep + 5d.1 Haiku
residual sweep, with the new `HAIKU_INPUT_SCALE = 1.09` /
`HAIKU_OUTPUT_SCALE = 2.17` per-call scales applied AND the
`ROLES_RESPONSE_LENGTH_FACTOR = 1.22` split applied where
appropriate) gives empirical actual/expected ratios:

- 5c.1 GPT (11 axes × 2 cohorts, B=7 full):  ratio = 0.97 (±4%)
- 5d.1 Haiku (corrected for roles/traits):   ratio = 1.05 (±5%)

So residual prediction error is ~3-5% in known-judge known-prompt
production runs.

**Recommended default**: `budget_usd = 1.25 × expected_cost_usd + $5`
(was: 1.5× + $20).  The 1.25× leaves ~20pt headroom over the
empirical residual — enough to catch model-drift or stupid
mistakes without being so loose that a real cost runaway escapes
the cap.

**For first-of-kind work** (new judge, new prompt template, new B
value not yet canary'd): stay at ~`1.5× + $10` until the first run
lands and the residual is measured.  Drop to the standard
recommendation thereafter.

### Provenance: capture `usage` from every API response

`assistant_axis.judge_pricing` (added Phase 4c, May 2026) is the
project's single source of truth for per-model pricing.  Every judge
call's `resp.usage` (input/output/cache tokens) is summed into a
`UsageTotals`/`BudgetTracker`; the live $ spend is checked against
`--budget_usd` (hard cap, exits with code 2 if exceeded) and ratioed
against `--expected_cost_usd` (advisory).  A `usage.json` side-car is
emitted next to every cohort cache, and totals are stamped into
`_provenance.notes` for forensic cost reconciliation.

---

## Response Patterns

### What Works Well

**Structured Analysis:**
```markdown
## Option A: Approach One
Pros: ...
Cons: ...

## Option B: Approach Two
Pros: ...
Cons: ...

## Recommendation: Option B
Why: ...
```

**Verification Steps:**
1. Check actual state
2. Report findings
3. Propose action
4. Execute after confirmation

**Implementation Flow:**
1. Acknowledge the request
2. Implement the changes
3. Verify results (lints, file contents)
4. Report completion with summary

**Handling Ambiguity:**
- When instructions are ambiguous, ASK for clarification
- If asking seems excessive, choose the option with the smaller blast radius
  - Example: "for review" unclear → provide list (easy to ignore) vs make changes (harder to undo)
  - Prefer reversible actions over irreversible ones when uncertain

### What to Avoid

**Don't:**
- Over-explain basic programming concepts
- Make assumptions about things such as versions, paths, or configurations without checking
- Separate project documentation (system behavior) from meta-documentation (working together)
- Create new .md documentation files (work summaries, etc.) without instructions or permission
  - In plan todos: "Document" implies creating a document file - use "Report" or "Verify" for communicating findings

**Do:**
- Be concise and information-dense
- Check actual state proactively
- Update README.md as part of feature work if needed
- Discuss trade-offs before implementing when choices are meaningful

### When Test Expectations Don't Match Code Behavior

**First: STOP and THINK HARD.** Don't reflexively "fix" either one.

**Then choose your path:**

**Path 1: Fix the Code (COMMON)**
- When you're very confident the expected results are correct and well-designed
- The implementation has a bug or misunderstood the spec
- This is the default assumption for tests written by Roger or derived from his specs

**Path 2: Adjust the Tests (RARE)**
- When you're very confident the code is correct and the expected results were poorly thought out
- Often happens when YOU wrote both the plan and discovered a design issue during implementation
- **REQUIRED:** When reporting your work, prominently flag that you changed expected results
- **REQUIRED:** Explain: what the discrepancy was, why you concluded tests were wrong, your reasoning
- This lets Roger verify your judgment call and catches if you were wrong

**Path 3: Ask Roger (ALWAYS SAFEST)**
- When you're not entirely certain which is correct
- Maybe Roger has a quick answer, maybe it needs discussion
- Better to get the right answer than enshrine a bug in both code and tests
- If you're on the fence between paths 1 and 2: choose this path

**Golden Rule:** Changing test expectations is always noteworthy. Even if you're certain it was right, Roger wants to hear about it in your summary.

---

## Prompt Engineering

### Writing Prompts for LLMs
When creating prompts that other LLMs will consume:
- **Context first:** Explain the "why" and broader goal before the "how"
  - Example: Added "Context" section explaining the prediction system goal
- **Acknowledge judgment calls explicitly:** Tell the LLM when decisions require judgment
  - Provide standards: "reasonable person", "preponderance of evidence"
  - Give examples of edge cases
- **Professional polish matters:** Clean prompts may prime LLMs for more careful work
- **Consistency in terminology:** Use backticks consistently, maintain parallel structure

### Multi-Pass Review for Prompts and Text
- When reviewing prompts or documentation (not code), expect multiple passes may be needed
- Each pass with fresh framing ("another pass", "anything we missed") can surface different issues
- Pattern: Typos → Clarity → Consistency → Edge cases
- This mirrors human proofreading where attention shifts with each read

---

## Documentation Context

### Multiple Audience Levels
- Human developers (Roger)
- Future AI agents
- Potentially other team members
- Documentation should serve all these audiences

---

## Session Initialization

### Recommended Starting Pattern
When starting a new session:
1. Read `README.md` for project context
2. Read `AGENT_NOTES.md` (this file) for collaboration patterns
3. Ask Roger what we're working on
4. Check relevant code/state before making assumptions

### Current canonical local data dir (May 2026)

The latest pipeline outputs live at:

```
runpod_workspace/qwen/qwen-3-32b Roger 8slot/
```

This is the **8-slot** rebuild that supersedes `runpod_workspace/qwen/qwen-3-32b Roger/` (the older 4-slot dir, vectors ~2.6 MB each). The 8-slot trait vectors are ~5.2 MB. The 8-slot dir contains the regenerated activations/scores/vectors for the 8 traits in the April–May 2026 trait-edits TODO list (`conformist`, `nonconformist`, `indecisive`, `anthropocentric`, `ecocentric`, `obedient`, `rebellious`, `unhelpful`) plus regenerated combinations.

When invoking analysis scripts that take `--data_dir`, prefer the 8slot path. The `results_analysis/run_axis_experiment_batch.py` default (`runpod_workspace/qwen/qwen-3-32b Roger`) is stale; pass `--data_dir 'runpod_workspace/qwen/qwen-3-32b Roger 8slot'` explicitly until that default is updated. The README examples likewise still reference the older directory.

### Mid-Session Patterns
- If unsure about working style: "Should I implement this directly or discuss options first?"
- If scope is unclear: "This affects X, Y, and Z - should I handle all of them?"
- If state is uncertain: Check it rather than assuming

---

## Examples from Past Interactions

### Good Interaction Pattern
```
Roger: "Check what's currently installed and update requirements.txt"
Agent: [Checks actual versions] → [Updates file] → [Shows results]
Roger: [Approves]
```

### Could Be Better
```
Roger: "Add version numbers to requirements.txt"
Agent: "I'll use version X.Y.Z which is reasonable for..."
Roger: "No, check what's actually installed"
```

Better approach:
```
Roger: "Add version numbers to requirements.txt"
Agent: [Checks installed versions] → [Updates with actual versions] → [Shows results]
```

---

## Adding New Trait Clean Pairs

When adding a new trait B that is the antonym of an existing trait A (e.g., adding `obedient` as the antonym of `rebellious`):

### Process

1. **Create seed file** in `data/traits/instructions/B.json` with `positive_label`, `description`, and `negative_label` set to `non-B` (NOT to A yet).

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

4. **Update negative_labels**: Set B's `negative_label` to A and A's to B in their instruction files.

5. **Regenerate instructions** for B with the proper antonym (the Roger prompt style injects the antonym into the neg instruction clause):
   ```bash
   uv run python data_analysis/regenerate_trait_instructions.py --traits B --force
   ```
   A should NOT need regeneration since it already has B as its negative_label.

6. **Add to `data/traits/trait_list.json`** with description.

7. **Run goal classification** if the trait needs goal scoring (for `data/goal_roles_and_traits.json`).

### Why non-X first?

Starting with `non-B` instead of `A` ensures the antonym generator discovers `A` independently from the neg instructions, rather than being primed by us providing it. This validates that the pos/neg instruction pairs genuinely capture the A↔B opposition.

### `data/goal_roles_and_traits.json` structure

Lists of roles and traits partitioned by goal content, for experimental use.
Each of `roles` and `traits` has `goal` (all-5 @ 2) and `non_goal` (all-5 @ 0) sublists.

**Ordering convention** -- items are randomized within tiers, tiers are concatenated:

- `roles.goal`: first 30 = primary set (varied), last 10 = cluster duplicates
- `roles.non_goal`: first 30 = most varied (max non-goal semantic spread), next 10 = nice-to-haves, last 30 = most redundant with first 30
- `traits.goal`: first 30 = max goal-space variation (less-HHH-default side of pairs, distinct ethical frameworks), next 10 = remaining moral circle spectrum, last 26 = default-side of pairs + redundant goal directions
- `traits.non_goal`: first 30 = max persona-property variation (less-default side of pairs), next 10 = nice-to-haves, last 17 = pair partners + redundant

### Clean pair validation results (April 2026)

After running the bidirectional antonym-discovery procedure on each candidate pair (script: `data_analysis/generate_antonyms.py`), 5 of the 6 originally proposed pairs validated as clean and the 6th was reorganized into a 3-trait conformity triangle.

| pair | bidirectional? | discovered antonyms | status |
|---|---|---|---|
| pragmatic ↔ idealistic | yes (mild near-synonym hedge) | pragmatic→idealistic; idealistic→pragmatic\|realistic | clean pair |
| conservative ↔ progressive | yes | →progressive ; →conservative | clean pair |
| ecocentric ↔ anthropocentric | yes | →anthropocentric ; →ecocentric | clean pair (descriptions rewrote in this session) |
| decisive ↔ indecisive | yes | →indecisive ; →decisive | clean pair (`indecisive` newly seeded) |
| obedient ↔ rebellious | yes (mild same-axis hedge) | obedient→rebellious\|defiant; rebellious→obedient\|compliant | clean pair (descriptions tightened to authority-axis-only) |

The 6th candidate (`conformist ↔ contrarian`) turned out to be a **3-trait triangle** rather than a clean pair, structurally analogous to the compassionate/malicious/callous triangle:

| trait | role | `negative_label` |
|---|---|---|
| conformist | engaged + aligned (central) | `non-conformist` (hyphenated seed marker) |
| contrarian | engaged + opposed (sibling) | `conformist` (partner reference) |
| nonconformist | disengaged (sibling, newly seeded) | `conformist` (partner reference) |

Convention: when a trait's true antonym is **structurally ambiguous** (the union of two siblings on different axes), its `negative_label` keeps the seed-marker form (`non-{positive_label}`) to flag the asymmetry. The two siblings each set `negative_label` to the central trait. This mirrors the compassionate/callous arrangement where compassionate.neg=`non-compassionate` and callous.neg=`compassionate` (engaged-vs-disengaged carves a clean partner pointer in one direction; the other direction is ambiguous between callous and malicious).

### TODO: regenerate activation/vector data after April–May 2026 trait edits

The trait-instruction edits across the April + early-May 2026 sessions require activation extraction and vector recomputation for **8 traits** (the others were either reverted to git-HEAD-equivalent or had only metadata changes that don't affect generation).

**New traits — no prior activations exist (April 2026):**

- `conformist`
- `nonconformist`
- `indecisive`

**Existing traits — instruction[]/description rewritten substantively, prior activations now stale:**

- `anthropocentric` (April 2026)
- `ecocentric` (April 2026)
- `obedient` (April 2026)
- `rebellious` (April 2026)
- `unhelpful` (May 2026 — rewrite from passive "indifferent / bare minimum" to active "refuses / deflects / makes excuses / does the bare minimum or less when forced". Motivation: `pharmacist_helpful_v1` and `chef_helpful_v1` steering experiments showed the old passive definition produced near-zero effect on the unhelpful pole because the resulting vector mostly captured "slightly less enthusiastic helpfulness", not actual unhelpfulness. Clean-pair check with `helpful` confirmed bidirectionally (score 4/4 each direction) before keeping the new definition.)

After regenerating activations + vectors for these 8, downstream artefacts that key on these traits (axis caches, response-mode score caches, plots that visualize these axes) should be inspected and rerun where they touch the affected traits. Combinations involving `unhelpful` (the 30 `t_*__unhelpful` set) also need full regeneration.

**Deliberately not regenerating:**

- `compassionate`: only `negative_label` changed (`callous` → `non-compassionate`), the description and instruction[] are unchanged. The negative_label is used only in the RP-filtering scoring step and forms a very small part of that prompt; the change is unlikely to materially shift filtering decisions.
- `callous`, `pragmatic`, `idealistic`, `conservative`, `progressive`, `decisive`, `contrarian`: byte-identical to git HEAD after this session's cleanup; existing activations remain valid.

### TODO: variant-specific K grids in the PC round-trip experiment (May 2026)

In `klm_sweep.py` and `permutation_null.py` we currently use a single
expanded K grid (`expanded_k_grid` = `DEFAULT_K_COARSE` ∪ `K-near-N`
neighbourhood with width ±4) for **all** ρ-search variants.  That's
appropriate for the `nth_pc` variant (which exhibits a sharp K=N spike
empirically), but is over-padded for the `fixed_direction` and
`fixed_principled` variants — those don't show K=N spike behaviour and
their winners cluster at small K (typically 0–8, rarely above ~32).

Hypothesis: for fixed_dir variants, a small low-resolution coarse grid
(e.g. `[1, 2, 3, 4, 6, 8, 12, 16, 24, 32]`) plus bracket-and-bisect
refinement would find essentially the same optima as the current
expanded grid, while cutting the search space substantially.  Lower
search bias → lower noise floor in the corresponding permutation null →
higher effective signal-to-noise for fixed_dir lines on the round-trip
plot.

Plan when ready:
1. Add a `variant`-aware `expanded_k_grid` (or two helpers) so that
   `nth_pc` keeps the K-near-N expansion and `fixed_direction` /
   `fixed_principled` use the smaller grid.
2. Add a `variant`-aware null in `permutation_null.py` so each variant
   has its own noise floor.  (Currently the null only computes nth_pc
   and over-states the floor for fixed_dir.)
3. Re-run the actual sweep + null and regenerate the plot.

Motivation evidence (2026-05-06): in the 3-cell sweep with ±4 K-near-N,
fixed_direction K winners across 14 PCs × 2 styles = 28 cells were
mostly K ∈ {0, 1, 2, 4, 8} with one outlier at K=512 (PC 192 glo,
suspected ρ-hacking on noise).  fixed_principled K winners were even
smaller — mostly K ∈ {0, 4, 7, 8, 13}.  Neither variant's K winners
needed the dense N±4 window.

### PC round-trip experiment: findings synthesis (May 2026)

**Goal.** Test how faithfully human-interpretable axes written for
canonical PCA directions (Opus describes the axis, GPT/Sonnet rank
entities along it) get geometrically recovered by an `(L, K, M)`
search over whitened PCs at various `(slot, layer)` cells.

**Setup recap.**

- *Canonical Nth PC* is defined in a fixed reference geometry: slot 3,
  layer 25, soft-shear `L=2`, PCA on standardized entity embeddings.
  Opus narrates that direction; the judge rubric is that description.
- `nth_pc` variant: at each cell `(slot, layer, L, K)`, recompute PCA
  on the post-shear/whitened entity matrix and take the index-N
  singular vector.  Direction is free to "float" with `(L, K)`; the
  empirical K-winner pattern clusters near `K≈N` (which is why the
  search grid is `DEFAULT_K_COARSE ∪ N±4`).
- `fixed_principled` variant: take α = `U_canonical[:, N-1] / σ_{N-1}`
  (the entity weights that define canonical PC N) and reapply those
  weights at the target cell's centred entity matrix.  The transported
  object is the same *linear combination of entities* across cells,
  modulo cell-local geometric drift.  At the canonical cell with
  `L=2, K=0` it reduces to the canonical direction; elsewhere it
  doesn't.  Naive `fixed_same` (reuse the raw ℝ^D vector) was
  abandoned because slot changes make it nearly orthogonal to the
  principled map.
- `permutation_null`: same search machinery on randomly permuted judge
  scores, giving the ρ that `(L, K)` optimisation reaches on noise.
  Currently computed only for the `nth_pc` variant; see the
  variant-specific-K-grids TODO above for why this is awkward to
  benchmark `fixed_principled` against.

**Core empirical read.**

The `nth_pc` winner-decomposition heatmap
(`roger/pc_round_trip_nth_pc_winner_decomposition.png`; built by
`results_analysis/pc_round_trip/{winner_decomposition,plot_winner_decomposition}.py`)
projects each `nth_pc` global winner direction back into canonical-L=2
shear space and expresses its squared mass on each canonical PC (rows
= target index N, columns = canonical PC index, row-normalised).

- For small N (roughly N ≲ 12), the heatmap is broadly diagonal: the
  recovered direction at index N really does live near canonical PC N.
- For larger N, the heatmap goes strongly off-diagonal: most squared
  mass concentrates on the first ~8 canonical PCs, regardless of which
  N was being searched for.  Spot-checks confirm the picture from the
  other side: individual low-index canonical PCs (e.g. PC 1, 2) have
  non-trivial Spearman ρ against high-N judge-score vectors.

**Interpretation.**

An Opus description of canonical PC N can be split into roughly:
*(a)* a PC-N-specific signal — what's actually distinctive about PC N
that ends up in the prose — and *(b)* generic top-PC scaffolding
(length, hedging, bluntness, stance loudness, "helpfulness
temperature", and other dominant features articulable in any axis
description) which is correlated with the high-variance early
canonical PCs and shows up in the judge rankings of *almost any*
labelled axis.  Add some noise.

The two ρ-measurement variants probe different mixtures of these:

| Variant | What it sees | Empirical caricature at high N |
|---|---|---|
| `nth_pc` ρ | (a) + (b); the search picks whichever latent best predicts judges. | Stays high but the heatmap shows the chosen direction loads on canonical PCs 1–8 — high ρ via canonical-PC aliasing, not genuine PC-N recovery. |
| `fixed_principled` ρ | Mostly (a): the direction is pinned to PC N's α-weights, only `(L, K)` varies. | Falls toward the noise floor sooner with N — measures whatever PC-N-specific content remains in the rankings once aliasing is denied. |

Neither is "the right ρ"; they answer different questions.
`fixed_principled` upper-bounds how much of PC N is genuinely
articulable in description-language; `nth_pc` upper-bounds what the
search procedure can recover from the same descriptions when allowed
to drift onto any latent that predicts judges.

**Regime heuristic (informal, continuity not proven).**

- N ≲ ~12: `nth_pc`, `fixed_principled`, the decomposition heatmap
  and the face-valid description content all agree.  Treat "the Nth
  PC" as a latent the description language can pick out.
- N ≳ ~12: `nth_pc` ρ alone is misleading; trust the `fixed_principled`
  trace, the decomposition, and the low-PC-vs-high-N-ranking
  cross-correlation as the sanity-check trio.

**Geometry side note.**

`results_analysis/pc_round_trip/principled_layer_transport.py`
compares naive vs principled cross-cell transport.  Layer ±1 at fixed
slot drifts modestly; changing slot at fixed layer drifts a lot.
Slot, not layer, dominates inter-cell direction drift, which is why
the principled (α-weight) transport matters once the cell set spans
multiple slots.

**Operational caveat tied to methodology.**

Until variant-specific K grids and nulls land (TODO above), the
permutation null in the round-trip plot reflects the `nth_pc` search
procedure's noise floor, not `fixed_principled`'s.  Compare
`fixed_principled` traces to *each other* and to canonical (no-opt)
ρ; treat the dashed null bands as an upper bound on noise rather than
a literal floor for `fixed_principled`.

**Plot status (2026-05-06).**
`results_analysis/pc_round_trip/plot_direction_cosines.py` was
restricted to `fixed_principled`-only after this synthesis: the
single-panel ρ figure shows canonical (×) plus `fixed_principled` ◆
traces for 1/2/3-cell aggregates with R/G/B colour-coding.  `nth_pc`
remains in the upstream `klm_sweep` cache and the heatmap script;
those tools are kept available for the diagnosis described above but
no longer drive the headline plot.

### Snapshot-before-invalidate principle (May 2026 lesson)

Whenever invalidating, wiping, or overwriting expensive-to-recreate data — most commonly judge-score caches, activation tensors, computed vectors, and response files — **make a timestamped copy first**, even if it feels "obviously fine to drop." The recreate cost is non-trivial:

- Judge caches: $50–$150+ per full rejudge round (gpt + sonnet + haiku across all axes)
- Activations: ~2.6 GB per role × 1 GPU-hour → tens of dollars + tmpfs hassle
- Response sets: minutes-to-hours of GPU time per pipeline run

Half an hour after a wipe, "I want to know how much things actually changed" is a very common follow-up question (Spearman rho between old and new judge scores per entity, for instance). Without a snapshot, the answer is "we can't compute that anymore."

**Cheap mitigations, in roughly increasing thoroughness:**

1. `cp -a <file_or_dir> <file_or_dir>.bak.YYYYMMDD_HHMMSS` next to the original. Trivially deletable when no longer needed.
2. Stage the originals into `/tmp/<session-id>/...` before the in-place modification — survives a chat session but auto-cleaned by macOS at boot.
3. For really expensive data: `rsync -a` into an offline backup directory (e.g. `~/Documents/assistant-axis-backups/<date>/`), or move to `~/.Trash` instead of deleting (the macOS Trash retains files until you explicitly empty it, so a wipe is recoverable for days/weeks if you change your mind).

**Agent rule:** when about to run an invalidation/wipe/overwrite of expensive cache data, **bring this up before doing it** — propose a snapshot scheme, even briefly. It costs ~one tool call and saves the "ah, well" moment later. (This is the same instinct as `git stash` before a destructive rebase.)

#### Off-tree archives (when in-tree is too big)

Some snapshots are too bulky to fit the in-tree `archive/` pattern
(itself 3-9 MB per subdir; the whole repo working tree is ~60 MB).
For those, the convention is `~/Documents/assistant-axis-archives/<archive-name>/`
on Roger's workstation, with a path note recorded here so a future
agent can find them.

Current off-tree archives:

* `~/Documents/assistant-axis-archives/v1-response-caches-2026-05-09/`
  — pre-anonymisation (rubric v1) per-axis response judge caches
  captured just before the v1→v2 rejudge ran on 2026-05-09.  111 MB
  xz-9 tarball (945 MB raw, 492 files spanning 116 axis × cohort
  combinations).  The aggregate v1 plots / JSONs that drive the
  v1↔v2 comparison story stay in-tree at
  `roger/axis_judge_experiments/*__rubric_v1.{json,png}` and
  `rubric_v1_v2_compare_slot6.{png,json}`.  The tarball is needed
  only when re-running per-entity v1↔v2 analyses; restore via
  `tar -xJf <tarball> -C /path/to/repo`.  Full README + manifest
  lives next to the tarball.

When promoting a snapshot to off-tree archive: keep an expanded
copy on disk under `roger/...` (gitignored) for active workflows
that read those paths, and put the canonical compressed copy +
README + manifest in the off-tree dir.  Add a row above so the
next agent / future-you can locate it.

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

## Updates and Evolution

This document should evolve as we discover new patterns. When something doesn't work smoothly:
1. Reflect on what caused friction
2. Determine if it's a pattern vs one-off situation
3. Update this document if it's a pattern
4. Keep it concise - remove outdated patterns

**Last Updated:** May 6, 2026

---

## Quick Reference

**Roger's Style in 4 Words:** Pragmatic, verify-then-trust, documentation-conscious, technically-sophisticated

**Golden Rules:**
1. Check actual state, don't assume
2. Be concise but complete
3. Documentation is infrastructure
4. Discuss trade-offs, then execute decisively

