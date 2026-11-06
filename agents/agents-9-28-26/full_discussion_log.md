# Full Session Summary — RunPod Pipeline, Judge Methodology, and Judge-Tier Evaluation

*A chronological summary of this entire session on `assistant-axis-roger`, from
initial clone through to a completed, paid judge-tier comparison. Written to
stand alone as a reference document, not a transcript. Companion to
[`agents/agents-9-12-26/full_discussion_log.md`](../agents-9-12-26/full_discussion_log.md)
(the earlier open-work-scoping investigation this session continued from) and
[`agents/agents-9-21-26/judge_tier_methodology_qa.md`](../agents-9-21-26/judge_tier_methodology_qa.md)
(a deep-dive methodology Q&A produced partway through this same session,
covering the axis/projection/judge-score/correlation chain in much more
detail than the summary below repeats).*

---

## 1. Orientation

Started from a fresh clone of `KKrampis/assistant-axis-roger`
(`https://github.com/KKrampis/assistant-axis-roger`), a fork of Christina Lu
et al.'s "Assistant Axis" paper codebase, extended by Roger Dearnaley. Read
the prior session's `full_discussion_log.md` for context: Roger's fork adds
"Roger mode" (role×trait combination generation), a research agenda toward a
"terminal goal subspace," and a large body of open, never-fully-scored work
(moral-circle traits, goal-role pairs, stale traits) documented in
`AGENT_NOTES.md`.

## 2. Swarm design brainstorm, narrowed to one option

Discussed three ways to parallelize running the 5-stage pipeline
(generate→activations→judge→vectors→axis) across many missing roles/traits:
one agent per entity, one agent per pipeline stage, or one agent per
thematic cluster. Settled on **per-stage** as the structurally correct
choice — it matches how the pipeline's own scripts batch, avoids reloading
the model per entity, and avoids the fact that stage 5's PCA embedding is a
joint operation across all entities (breaking the per-entity design).

## 3. Local (Mac) vs. RunPod, and the moral-circle pilot

Established that stages 1–2 (generation, activation extraction) need a real
CUDA GPU — vLLM doesn't run on Apple Silicon, and the target models are far
too large for a Mac anyway — while stages 3–5 (judging, vectors, axis/PCA)
are CPU-only and can run locally. Built a full RunPod workflow for the
GPU-bound half:

- `runpod/launch_pod.py` / `stop_pod.py` — provision/tear down a GPU pod via
  RunPod's REST API (verified against RunPod's live OpenAPI spec rather than
  assumed). Later hardened with a one-pod-at-a-time check and optional
  `GH_PUSH_TOKEN`/`ANTHROPIC_API_KEY` forwarding.
- `runpod/bootstrap.sh` — clones the repo, `uv sync`s, configures push
  access.
- `runpod/run_and_shutdown.sh` — wraps `run_pipeline.sh` and stops the pod
  automatically via RunPod's auto-injected pod-scoped credentials (no need
  to copy the real account key onto the pod).
- `runpod/push_results.sh` — lets the pod push results back itself via a
  repo-scoped GitHub token, deliberately not a copied personal SSH key.
- `data/traits/instructions/_moral_circle/` — a scoped symlink directory
  (the 12 moral-circle traits + `default.json`) so `run_pipeline.sh`
  (which otherwise processes every file in `--roles_dir`) only touches this
  batch.
- Fixed a real gap: `run_pipeline.sh`'s christina mode silently ignored
  `--reduce_questions`; patched it to actually forward the flag.

Docs: `runpod/README.md`, `pipeline/SUBSET_RUNS.md` (Option A: `--roles`
flags per-script vs. Option B: the scoped-directory pattern above).

## 4. Branch and identity housekeeping

Created `personas-konstantinos` as a copy of `anthropic-vllm-uv` plus the
`agents/` directory (which only existed on `master`) — the two source
branches had genuinely diverged content that needed merging by hand, not a
git merge. Also fixed git authorship mid-session: an early commit
accidentally used the user's real name; amended it and set git identity
(local and global) to `Claude <noreply@anthropic.com>` going forward, plus a
force-push with `--force-with-lease` to correct the already-pushed commit.

## 5. The judging component, explained from first principles

Investigated `pipeline/3_judge.py` (Lu's original 0–3 "how fully is the
model role-playing" scale — a response-quality *filter*, unrelated to any
pole comparison) versus `results_analysis/axis_judge_correlation.py`
(Roger's own addition, first committed 2026-04-25, confirmed via git
history to postdate the fork point — a −3..+3 pole-alignment scale used to
validate whether a computed axis means anything). Walked through, with real
data pulled directly from the repo rather than recalled from memory:

- What "the axis" is (a raw activation-difference vector — confirmed the
  committed `_axis.pt` is a *different*, general-purpose object, not the
  angel/demon-specific direction, which was never itself saved as a file).
- What a "projection" is (a dot product, cached as plain floats in
  `projections.json`, not vectors).
- What the judge produces (one integer from reading text, nothing more).
- How Spearman ρ connects the two, including a concrete worked example of
  tie-handling: 311 of 571 entities (54.5%) shared the exact judge score
  `0`, and Spearman assigns them all the *same* rank (260/571) regardless of
  their real, quite different projections — a genuine limitation of
  correlating a 7-bucket discrete score against a continuous one.
- What the 571 entities actually are (278 roles + 293 traits, the full
  corpus minus that axis's own 2 poles) and that this −3..+3 methodology
  had already been run on **35 axes total**, not just angel/demon — full
  list recovered from `gpt_vs_sonnet_rhos_di.json`.
- Reconciled a real discrepancy (571 vs. the 562 seen in the pooled 35-axis
  chart): the 9 known role/trait collision names account for exactly the
  difference; the aggregate comparison excludes them, `angel_vs_demon`'s own
  data doesn't.

Full detail on all of this lives in the companion methodology doc
(§1 reference above), not repeated here.

## 6. What's reusable without recomputing activations

Verified directly (not assumed) that raw activation vectors are **not
committed anywhere** in this repo — only two axes,
`angel_vs_demon` and `decisive_vs_indecisive`, have their full per-entity
`projections.json` committed; the other 33 of the 35-axis comparison only
have their aggregate summary. Built
**`results_analysis/judge_tier_cost_eval.py`** to exploit this: it scores a
*new* judge model against an *existing* cached projection set, skipping the
GPU-dependent half of the pipeline entirely. Paired with
**`results_analysis/judge_tier_cost_plot.py`**, which plots the result in
the same cost-vs-quality (`1/(1−ρ)` vs. `$`) house style as Roger's own
`plot_batch_size_quality_vs_cost.py`.

## 7. Pricing update

`assistant_axis/judge_pricing.py` was last updated May 2026 and had drifted:
Anthropic pricing was refreshed via the `claude-api` skill's live table
(Haiku 4.5 unchanged; **Sonnet 5 is cheaper** than Sonnet 4 was; **Opus 5**
at $5/$25 supersedes a never-verified $15/$75 May-2026 placeholder that had
never actually been used). Added a 3-tier OpenAI ladder
(`gpt-5.6-luna/terra/sol`) sourced from web search cross-confirming
`developers.openai.com` directly — flagged as lower-confidence than the
Anthropic numbers, since that page couldn't be fetched directly (client-
rendered app). Legacy rates were kept alongside current ones so historical
cached runs still price correctly, and `plot_batch_size_quality_vs_cost.py`
was deliberately left un-synced, since it recomputes a chart from data that
was actually billed at the old rates.

## 8. Running it for real: two bugs, both caught and fixed mid-run

API keys were placed at `~/.config/assistant-axis/.env` (outside the repo
tree, symlinked in — following the same convention `.gitignore` already
documents for Google Sheets OAuth credentials) after an earlier attempt to
`export` them directly didn't work (shell state doesn't persist between
tool calls).

Running the real evaluation (6 tiers × 2 axes × 571 entities) surfaced two
genuine bugs, both stopped immediately on detection rather than left to run
to completion:

1. **Claude Opus 5 / Sonnet 5 run adaptive thinking by default.**
   `assistant_axis/judge.py`'s `call_anthropic_judge_single` assumed
   `response.content[0]` was always the answer text; with thinking on, it's
   a `ThinkingBlock` first, so every call crashed *after* being billed,
   silently logged as an unparseable response. Cost ~$1–2 on 221 wasted
   calls before being caught. Fixed to scan for the actual text block, and
   added `output_config.effort="low"` (gated by model name) to bound the
   thinking-token cost this exposed.
2. **A dependency-version mismatch** (`httpx2` 2.13.1 / `Brotli` 1.1.0, an
   artifact of this session's ad-hoc `pip install` rather than the
   project's real `uv.lock`) broke OpenAI response decompression
   specifically for the larger `gpt-5.6-*` responses. Worked around by
   disabling response compression for that client (harmless at these
   payload sizes).

Both fixes were validated on small (`--limit 3`) samples against real API
calls before resuming the full run — a `--limit N` flag was added to
`judge_tier_cost_eval.py` specifically for this. The run was also
interrupted once externally partway through and resumed from exactly where
it left off (tracking which of the 12 tier/axis combinations already had a
saved `usage.json`, to avoid re-spending on completed tiers).

## 9. Result

**Total real spend: $16.61** (plus the ~$1–2 wasted before the Opus-5 fix,
unquantified since no `usage.json` was ever written for those failed
calls). Headline finding, visible in both `cost_vs_quality.png` charts:
**Opus 5 has the highest ρ against the geometric projection on both axes**
(0.791 on angel/demon, 0.651 on decisive/indecisive) but at the highest
cost; `gpt-5.6-terra` is the standout value tier (ρ=0.782 at $0.81 vs.
Opus's $3.15, on angel/demon). Both plots and all raw per-tier data
(`scores_descriptions.json`, `usage.json`, `correlations.json`) are
committed under
`roger/axis_judge_experiments/{angel_vs_demon,decisive_vs_indecisive}_tier_eval/`.

Compared against Roger's own prior work: this doesn't extend axis coverage
(still only these 2 of 35) or touch response-mode judging, but it's
genuinely new tier coverage — Opus has never been tested as a judge in this
repo before, nor has any OpenAI model beyond `gpt-4.1-mini`.

## 10. Repository administration

Pushed all of the above to `personas-konstantinos` across several commits.
Made the repository **private** on GitHub at the user's explicit request
(confirmed technically possible — the session is authenticated as the
`KKrampis` account with admin rights — and flagged the real consequence
before doing it: a public repo going private breaks anything relying on
public access, forks, or unauthenticated API reads).

## 11. Fork-chain investigation

Traced the actual upstream chain, verified against GitHub's own fork
metadata rather than assumed: **`safety-research/assistant-axis`** (the
original paper repo) → **`RDearnaley/assistant-axis`** (Roger's original,
real-name account — only ever got a `master` branch, stalled
2026-01-20, matching the exact commit where the fork later diverged) →
**`RogerDOX14w/assistant-axis`** (a second account Roger switched to for
the sustained "Roger mode" work — has the `anthropic-vllm-uv` branch with
130+ commits, stalled 2026-09-04) → **`KKrampis/assistant-axis-roger`**
(this repo — confirmed *not* a GitHub-native fork of either, `isFork:
false`, `parent: null`; a plain copy-and-push rather than GitHub's Fork
button, though content-identical to `RogerDOX14w`'s fork at copy time,
verified via matching git blob hashes).

Investigated a claimed recent addition (`pair_list_clean.json`, described
as "just checked in," plus a claim of "new commits in the last 2-3 days")
— checked both of Roger's accounts, all their branches, and found **no
commits from the last several weeks on either**, and the named file doesn't
exist on GitHub under either account. `pair_list_goalnongoal.json` (the
other file mentioned) does exist and is already byte-identical between
Roger's fork and this repo — nothing to pull there. Flagged this discrepancy
rather than fabricating a pull from a source that doesn't verifiably exist.
