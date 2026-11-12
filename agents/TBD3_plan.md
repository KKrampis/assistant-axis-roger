# TBD 3: what we need to do

The short plan. For Claude Code, the operating instructions are
**`agents/TBD3_discussion_context.md` §14**:
- the pipeline scripts and their order
- the tasks
- the correctness checks
- the results format

Background on the filters, judges and data is in §1–§13 of the same file.
§15 covers the end-think arms and the link to TBD 4.

## The task, as agreed in the 9 Nov meeting

Our pipeline throws data away at three points before building persona
vectors. For each one:

1. **Run it both ways**: filter on and filter off.
2. **Compare ρ, per axis and on average**: which arm's vectors match the
   judge's ranking of personas better?
3. **Cross-ρ** (Roger: "maybe"): judging from one arm against vectors from
   the other, to separate "the vectors changed" from "the judging changed"
   (§13). This needs paid response judging.
4. **Recommend** which way the flag should be set.

**Why it matters.** It's P1, before the first paper. Roger: "if that piece of
our technique is wrong, it could be biasing all the rest of our results."

**Order.** TBD 4 (thinking vs speaking) comes first. Then TBD 3.

## The three filters

| | Filter | Origin | What it does now | Status | Cost |
|---|---|---|---|---|---|
| **a** | Role-play (score = 3) | Lu (the code; her paper kept score 2 too) | A 0–3 judge scores each answer; step 4 keeps only 3s; ≥50 needed | **Half done.** Roger compared filtered vs unfiltered on 61 axes (1–2 Nov), description judging only | Free, except the cross-ρ |
| **b** | Incongruity | Roger (Apr 2026) | Prompt variants of role + trait combinations rated "very contradictory" have their ~100 answers dropped before judging | **Data built.** Only the comparison is missing | Free |
| **c** | Leaked end-think cut | Roger (Apr 2026); Lu's code keeps leaked text in the average | Text before a stray `</think>` is cut before saving, and **lost** | **Not started. No flag.** | Code + GPU + API |

## What's left, per filter

### (a) Role-play filter: all free except step 6
1. Reproduce Roger's 61-axis numbers.
2. Per-axis table: ρ filtered vs unfiltered for each axis × setting × judge.
   This is the primary result; means are a summary.
3. **Discarded-only arm**: vectors from only the answers scored 0–2.
   - If its ρ is close to the filtered ρ, refused answers still carry the
     persona. That answers Roger's question at 47:31 directly.
   - No new judging needed.
4. Relate each axis's Δρ, and how far its direction rotated, to how many
   answers its poles lost.
5. ρ by layer (slots 0 and 6) for all arms.
6. **Cross-ρ (paid, gated):**
   - confirm Roger means response judging
   - 1-axis pilot (≈ $20)
   - ~10 axes only with his approval

### (b) Incongruity filter: free
- Count the affected combinations in the May run.
- Check that only those differ between the two arms.
- Compare the combination vectors with `compute_combo_marginals.py`.
- Recommend keep or drop.

### (c) End-think cut: gated
1. Ask Roger whether he has a flag, and which personas leak.
2. Save the raw text in `generation.py`.
3. Measure the leak rate.
4. Generate **once** for the affected personas.
5. Build the arms **STRIP** (today) / **KEEP** (Lu-like) / optional **DROP**
   (§15).
6. Compare ρ at slot 0. Check that the header slots are identical across
   arms.

The analysis of the leaked text itself fits TBD 4 (§15).

## Data

Konstantinos will give the path to his local copy of the May 8-slot data.
Check it first (§14.4) for:
- vectors (8 × 64 × 5120 per persona)
- the unfiltered folder and `unfiltered_stats.jsonl`
- combination `scores_unfiltered/` and `vectors_unfiltered/`
- role-play `scores/`

The activation files (GBs) are likely only on Roger's G-Drive; they're needed
only to rebuild vectors. Axis-judge scores and Roger's 61-axis results are in
git.

## Rules
- Work from `origin/personas-konstantinos`.
- Anything over **$20 API or 13 GPU-hours** needs Roger's explicit OK,
  quoting each non-default flag.
- Write every new arm to new folders.
- No commits or pushes without Konstantinos's say-so.

## Deliverables (format in §14.10)
Per filter:
- a headline table (12 settings)
- per-axis paired bars
- a Δρ vs data-lost scatter (for a)
- a rotation table
- the cross-ρ table if approved
- a recommendation
- appendix examples

**Provisional choices, to confirm with Roger:**
- headline judge = gpt-4.1-mini
- cross-ρ on ~10 axes after a 1-axis pilot

## Questions for Roger
1. Data access: May 8-slot vectors and activations, the unfiltered folder,
   and the new dataset (~1 week).
2. Is there a flag for the end-think cut? Which personas leak most?
3. Cross-ρ: description judging or response judging? Budget for a pilot and
   ~10 axes?
4. Which rubric version (the rewritten one is probably in a pending PR)?
5. Headline judge: gpt-4.1-mini (the judge used for the May data), or the
   current canonical ensemble?
6. Our code keeps score 3 only with ≥50; Lu's paper kept 3 and 2 separately
   with ≥10. How do we describe this in the paper?
7. Which combination analyses go in the paper, so (b) recomputes the right
   ones?
8. Where is the AGENT_NOTES TODO about the unfiltered side finding?
9. Should the leaked-thinking content analysis go with TBD 4?
