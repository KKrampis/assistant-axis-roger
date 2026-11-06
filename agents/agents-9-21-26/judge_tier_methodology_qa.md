# Judge Tier Evaluation — Methodological Q&A

*A summary of the methodological discussion behind the judge-tier cost/quality
evaluation work on `personas-konstantinos`. Written to stand alone as a
reference document — not a session transcript, but a synthesis of the
questions that came up and how they were resolved, grounded directly in the
repo's code (not recalled from memory). Companion to
[`agents/agents-9-12-26/full_discussion_log.md`](../agents-9-12-26/full_discussion_log.md),
which covers the earlier open-work-scoping investigation this thread grew out
of.*

---

## 1. What "the axis" literally is

Take a model, prompt it "you are an angel," record its internal activations
(a long vector of numbers). Do the same for "you are a demon." Subtract one
from the other: `angel_activations − demon_activations`. That difference is
a single direction in activation space — a purely numerical object, computed
with zero language understanding. That direction is "the angel-demon axis."

Two distinct things share the word "axis" in this repo and should not be
conflated:

- **The general Assistant Axis** (`roger/workspace/qwen-3-32b/roles/_axis.pt`,
  shape `[64, 5120]` — verified by loading it directly) — `mean(default) −
  mean(all-role-vectors)`, from the original 30×30 Roger-mode sweep. Default
  persona vs. everything else, not specific to any one pair.
- **A pair-specific axis** (e.g. `unit(vec[angel] − vec[demon])`), what
  `results_analysis/axis_judge_correlation.py --pair angel demon` computes.
  This one is **not saved as a file anywhere** — confirmed by listing
  `roger/axis_judge_experiments/angel_vs_demon/`: no `.pt` file of any kind
  exists there, only JSON/PNG/log outputs. It's computed on the fly from raw
  activation vectors that no longer exist in this repo (see §5).

## 2. What a "projection" is

For some third entity (e.g. `paperclip_maximizer`), take its own activation
vector and measure how far it points along the axis direction — a single
number, via dot product (`raw_proj = (entity_vector − default_vector) ·
axis_unit`, `axis_judge_correlation.py:1116`). Positive → leans toward the
"angel" pole; negative → leans "demon"; near zero → doesn't lean either way
*on this specific direction*. Still pure arithmetic — no text has been read.

A second, "whitened" variant also exists: a soft-K PCA whitener is fit
per-entity, leave-one-out, on the rest of the corpus, before projecting —
this down-weights the corpus's dominant-variance directions so the score
isn't just tracking whatever pattern happens to dominate the whole dataset.

## 3. What the LLM judge actually gives us

The judge never sees the axis, the projection number, or any activation.
It's handed a rubric — the two pole descriptions, a few example entities
near each pole, a −3..+3 scale table — and one entity's **plain-text
description**. It reasons briefly, then outputs `SCORE: <int>`
(`axis_judge_correlation.py:144–195`, the exact prompt template — this is
not a paraphrase). That's the entire output: one integer, derived purely
from reading English.

## 4. What "signal" means, and how it connects the two

For each of the ~571 non-pole entities you now have two independently
computed numbers: a geometric projection (§2) and a judge score (§3). Do
this for all of them, then ask: does ranking entities by projection match
ranking them by judge score? That's exactly what Spearman ρ measures
(`_spearmanr`, `axis_judge_correlation.py:2808` — literally
`scipy.stats.spearmanr`, rank-based). High ρ (e.g. `helpful/unhelpful`,
0.915) means two independent methods — activation geometry and language
understanding — agree on the ranking: real evidence the axis captured
something a judge also recognizes as "angel-vs-demon-like." Low ρ (e.g.
`systems_thinker/analytical`, 0.679) means they aren't tracking the same
thing.

**This is the crux of the whole exercise**: the axis, on its own, is
just a direction that happens to separate two role-play prompts. Whether it
means anything is an empirical question, answered by checking whether an
independent judge — reading text, understanding language — arrives at
similar conclusions.

## 5. What's actually reusable in the repo (verified, not assumed)

Checked directly, not recalled: **zero raw activation vectors are committed
anywhere in this repo.** Not angel's, not demon's, not any of the other 571
entities'. They only ever existed on Roger's local machine.

What survived: `projections.json`, containing **plain numbers, not
vectors** — `result[slot][entity_name] = {"raw": <float>, "whitened":
<float>}`, confirmed for slots `0`–`7`, 571 entity names, by loading the
file directly. This is the *output* of applying (now-gone) raw vectors to
a (now-gone, never-saved) axis direction — the expensive part happened
once, and only its numeric result was small enough, and thought worth
committing.

**Only two axes have this full reusable cache**:
`roger/axis_judge_experiments/{angel_vs_demon,decisive_vs_indecisive}/`.
Every other axis in the 35-axis GPT-vs-Sonnet comparison only has its
*aggregate* summary committed (`gpt_vs_sonnet_rhos_di.json`), not the
per-axis projections a new judge-tier comparison would need.

Practical consequence: a new judge tier can be scored against these two
axes' cached projections with **zero GPU work** — that's exactly what
`results_analysis/judge_tier_cost_eval.py` does, built this session.

## 6. Why compare multiple judges/tiers, not just one

Two distinct questions, easy to conflate:

- **Validity** (does the axis mean anything?) — answered by correlating a
  judge's scores against the *projection* (§4). A single judge's high ρ
  here could still just reflect that one model's idiosyncratic associations
  rather than a broadly-recognized concept. Running several independent
  judges (different companies, different training, different capability
  tiers) and finding they *all* similarly agree with the projection is much
  stronger evidence the axis is picking up something real.
- **Reliability** (do judges even agree with each other?) — a *different*
  measurement: correlating one judge's scores directly against another
  judge's scores (what `gpt_vs_sonnet_scatter.py` does; pooled ρ=0.855
  across 35 axes, per the actual committed chart — see caveat in §8 on not
  trusting stale prose over the live file). This is inter-rater agreement,
  not validity — there's no ground truth being checked here, just whether
  two judges see the same thing.

**The tier angle** (cheap vs. expensive judges) answers a third, practical
question layered on top of both: does spending more on a smarter judge
actually buy a stronger validity signal, or does a cheap one do just as
well? The repo already found this once (`gpt-4.1-mini` ≈ `sonnet-4` in
validity ρ, at ~5× lower cost) — this work extends that check across a
wider tier ladder, including a provider (OpenAI's GPT-5.6 family) never
tested here before.

## 7. Cost model — why 571 entities is cheap

Each judge call scores one short paragraph (an entity's description) and
returns brief reasoning plus a score — roughly 470 input / 80 output
tokens per call. 571 calls ≈ 268K input + 46K output tokens *total* per
judge tier per axis — small by LLM standards. This is why **description-mode**
judging (what this work uses) is so much cheaper than the repo's
**response-mode** judging elsewhere (scoring actual multi-paragraph model
completions, up to 1200 per entity: $40–50/axis, a >15× difference).

Current per-tier, per-axis costs (verified pricing, `assistant_axis/judge_pricing.py`):

| Tier | Provider | $/axis | Both axes |
|---|---|---:|---:|
| `gpt-5.6-luna` | OpenAI | $0.11 | $0.22 |
| `claude-haiku-4-5` | Anthropic | $0.50 | $0.99 |
| `claude-sonnet-5` | Anthropic | $0.99 | $1.99 |
| `gpt-5.6-terra` | OpenAI | $1.08 | $2.17 |
| `claude-opus-5` | Anthropic | $2.48 | $4.97 |
| `gpt-5.6-sol` | OpenAI | $2.71 | $5.42 |
| **All 6 new tiers, both axes** | | | **$15.76** |

The 2 legacy tiers (`gpt-4.1-mini`, `claude-sonnet-4`) have no measured
cost — `usage.json` tracking postdates their archived runs — so their cost
in any comparison chart is *estimated* from the same token/pricing model,
not measured. This distinction is preserved visibly (filled vs. hollow
markers) in `judge_tier_cost_plot.py`'s output rather than presented as
equally certain.

## 8. A standing caution: verify against the actual file, not the prose

Earlier in this work, numbers quoted from `results_analysis/README.md`'s
prose (pooled ρ, weight-sweep endpoints) turned out not to match the
*specific* committed PNG/JSON files being discussed — the repo has multiple
slot variants (slot3/6/7) and rubric versions (v1 vs. the current "di"
cohort) coexisting, and prose describing one doesn't always specify which.
Lesson applied throughout this document: every number and file reference
above was checked directly against the actual committed file, not recalled
from an earlier summary — including this one, if it's read back later.

## 9. Tools built this session, and their exact scope

- **`assistant_axis/judge_pricing.py`** — pricing table updated (Anthropic
  rates verified live via the `claude-api` skill; OpenAI GPT-5.6 rates from
  cross-confirmed web search, lower confidence, flagged as such in the
  module docstring). Legacy rates kept alongside current ones so historical
  cached runs still price correctly.
- **`results_analysis/judge_tier_cost_eval.py`** — scores new judge tiers
  against a cached `projections.json`, correlates via the existing
  `compute_correlations`, tracks real cost via `MultiModelUsage`. Touches
  only `api.anthropic.com`/`api.openai.com` and new output directories;
  never modifies existing committed data; no git operations.
- **`results_analysis/judge_tier_cost_plot.py`** — offline, no API calls;
  reads the above's output plus legacy `correlations.json`, plots
  cost-vs-quality in the house style of `batch_size_cost_vs_quality.png`.

## 10. Status as of this writing (updated)

`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` are configured (via `~/.config/assistant-axis/.env`,
symlinked into the repo root, following the same outside-the-repo-tree
convention already established in `.gitignore` for the Google Sheets OAuth
credentials). Nothing has been executed yet — the ~$15.76 scoring run is
still pending.

## 11. The −3..+3 rubric was applied to 35 axes, not just angel/demon

Angel/demon isn't methodologically special — it's one of **35 pole pairs**
Roger ran this same procedure on, pulled directly from
`gpt_vs_sonnet_rhos_di.json`:

```
helpful/unhelpful          harmless/harmful           honest/dishonest
truthful/deceitful         guileless/scheming         egalitarian/elitist
progressive/conservative   concise/verbose            ecocentric/anthropocentric
improvisational/methodical relativist/absolutist      systems_thinker/analytical
accessible/esoteric        benign/malicious           casual/formal
compassionate/callous      confident/uncertain        constructive/destructive
convergent/divergent       cooperative/competitive    ethereal/grounded
forgiving/unforgiving      generous/stingy            idealistic/pragmatic
individualistic/collectivistic  introverted/extroverted   obedient/rebellious
passionate/dispassionate   playful/serious            practical/theoretical
quantitative/qualitative   reductionist/holistic      trustworthy/untrustworthy
decisive/indecisive        angel/demon
```

Every axis gets its own instance of the same shared rubric template
(`_RUBRIC_HEADER` / `RUBRIC_STATIC`, `axis_judge_correlation.py:144–195`) —
only the pole descriptions (and a few example names) change per axis. It's
the same mechanism run 35 independent times, not 35 different mechanisms.

Angel/demon and decisive/indecisive are special only in **data survival**:
they're the 2 of 35 whose full per-entity projection data is still
committed (see §5) — the underlying projection data for the other 33 was
necessarily computed too (you can't get a ρ number without it), but never
saved to git; only the resulting summary numbers survived.

### 35 vs. 571/562 — two orthogonal numbers, reconciled

**35** = how many axes were tested. **571** (or **562**) = how many
entities get scored *per axis*, drawn from the same ~573-entity master
corpus (278 roles + 293 traits) each time, minus that axis's own 2 poles.

Verified directly: `angel_vs_demon`'s committed data has exactly 571
entities, and includes all 9 of the known role/trait collision names
(`ascetic`, `contrarian`, `cosmopolitan`, `generalist`, `pacifist`,
`patient`, `perfectionist`, `romantic`, `stoic` — the names
`AGENT_NOTES.md` flags as impossible to type-resolve from filename alone).
The aggregate 35-axis `gpt_vs_sonnet_scatter` comparison shows `n=562` per
axis instead — `571 − 9 = 562`, confirming that analysis excludes those 9
collision names to sidestep the ambiguity. Sanity check: `562 × 35 =
19,670`, exactly the pooled `n` reported in `gpt_vs_sonnet_scatter_pooled_di.png`.
So it's not "35 becomes 571" — it's roughly 571 (or 562) entities scored,
once per axis, 35 times over.

## 12. Whole-range rubric examples (real data, `angel_vs_demon`, GPT judge)

| Score | Meaning | Example entities |
|---|---|---|
| −3 | strongly demon-like | `cruel`, `evil`, `malicious` |
| −2 | clearly demon-like | `gluttonous`, `parasite`, `opaque` |
| −1 | leans demon-like | `devils_advocate`, `pirate`, `trickster` |
| 0 | neutral / off-axis (311/571 entities — 54.5%) | `veteran`, `blogger`, `zeitgeist` |
| +1 | leans angel-like | `advocate`, `resilient`, `holistic` |
| +2 | clearly angel-like | `guardian`, `empathetic`, `tactful` |
| +3 | strongly angel-like | `altruistic`, `compassionate`, `nurturing` |

Reasoning traceable to the source text: `parasite` (−2) — *"survives by
living on or in a host... often without providing benefit in return"* —
exploitative but not maximally cruel, hence −2 not −3. `guardian` (+2) —
*"takes on the responsibility of protecting others from harm... acting as
a shield... for those who are vulnerable"* — protective but not framed in
explicitly spiritual/altruistic language the way `compassionate` is,
hence +2 not +3. `veteran` (0) — *"military experience, perspective,
discipline..."* — genuinely unrelated to the angel/demon dimension,
correctly lands in the huge neutral bucket (see §4's tie-handling
discussion for what that bucket does to the correlation).

## 13. Three scoring modes exist; only the cheapest has been used so far

`axis_judge_correlation.py` supports:

- **`descriptions`** and **`instructions`** — score the entity's *static
  text* (the JSON's `description` field, or its `pos` instruction text).
  Never touches anything the model generated. This is everything discussed
  and built so far, including `judge_tier_cost_eval.py`.
- **`responses`** — scores the model's *actual generated completions*
  (real answers it gave while role-playing that entity), via
  `RUBRIC_RESPONSE_BATCH`, a close cousin of the same rubric family and
  scale. Far more expensive — per the cost model in §7, ~$40–50/axis vs.
  ~$1–3/axis for description-mode, since it reads hundreds of real
  multi-paragraph completions per entity instead of one short paragraph.
  **Not used anywhere in this session's work.**

## 14. Is "only 2 of 35 axes" a pipeline limitation, or a data limitation?

**Data, not pipeline.** `axis_judge_correlation.py` itself has no
restriction to these two — point it at any pair and it computes a fresh
axis and fresh projections, provided `--data_dir` has raw activation
vectors. Those vectors only ever existed on Roger's local machine and were
never committed (§5), so the pipeline *can* target any of the other 33, or
an entirely new pair — it just currently has nothing to compute from
without redoing GPU work. `judge_tier_cost_eval.py` (this session's tool)
is the one genuinely hard-restricted to these 2, because it was built
specifically to exploit the one thing that did survive: the cached
projections, which only exist for these two.

## 15. What the pending $15.76 run would actually add over Roger's existing plots

It only ever touches `angel_vs_demon` and `decisive_vs_indecisive` — no
broader axis coverage. What's genuinely new:

- **Untested judge tiers**: Roger's existing plots only ever compare
  `gpt-4.1-mini` against `sonnet-4` and `haiku-4.5`. Opus has never been
  tested as a judge in this repo, at any generation; no OpenAI model
  beyond `gpt-4.1-mini` has been tested either.
- **Current pricing**: Roger's cost figures are frozen at May-2026 rates.
  Sonnet got cheaper since ($2/$10 now vs. $3/$15 then) — re-asking his
  cost-efficiency question today can get a different answer than it did
  then, independent of anything about judge quality.

What it explicitly does **not** do: extend axis coverage beyond these 2;
touch response-mode judging (§13); or improve/re-run Roger's
judge-*agreement* work (`gpt_vs_sonnet_scatter.py`'s 35-axis comparison) —
it only extends the judge-*validity*-against-projection check to new
tiers, on 2 axes. Honest framing: a cheap, narrow pilot to see whether a
wider tier ladder changes the cost-efficiency conclusion, not a broader or
more complete version of Roger's study. A positive finding here (a cheap
tier matching Opus, or Opus meaningfully winning) would be the actual
argument for spending more to extend to the other 33 axes.

## 16. Cost-inclusive plotting already exists, twice over

- **Precedent, already in the repo**: `results_analysis/plot_batch_size_quality_vs_cost.py`
  → `batch_size_cost_vs_quality.png` — $/axis on x, quality (`1/(1−ρ)`) on
  y, one point per batch-size/ensemble-combo. Established house style for
  putting cost and correlation quality on the same chart in this repo.
- **Built this session**: `results_analysis/judge_tier_cost_plot.py` — the
  direct equivalent for judge tiers instead of batch sizes. Same axes,
  same transform, one point per tier; filled markers for measured cost
  (the 6 new tiers, from real `usage.json`), hollow for estimated (the 2
  legacy tiers, which predate cost tracking).

Nothing further needs building here — the tooling is complete and waiting
on the actual scoring run (§10) to have real data to plot.
