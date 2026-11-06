"""Dump steered responses at length-collapse strengths into a readable
markdown file so Roger can assess whether the brevity is legitimate
steering (unhelpful = refuse-to-engage; concise = terse) or actual
incoherence the rubric is missing.

For each (cell, strength), shows question | baseline | steered response
| coherence judge call + reason | effect judge calls + reasons.

Run on the pod, write the output locally via stdout redirect.
"""
import json
import glob
import statistics
import sys
from collections import defaultdict


def load_records(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def fmt_truncated(text: str, n_chars: int = 1200) -> str:
    if not text:
        return "(empty)"
    if len(text) <= n_chars:
        return text
    return text[:n_chars] + f"\n\n[...truncated, total length {len(text)} chars]"


def dump_strengths(experiment_dir: str, cell: str, strengths: list):
    bls = load_records(f"{experiment_dir}/baselines/records.jsonl")
    bl_by_q = {int(b["question_idx"]): b for b in bls}
    recs = load_records(f"{experiment_dir}/{cell}/records.jsonl")
    # Index by question_idx within strength so floating-point rounding
    # variation (records may store 11.286154 vs the requested 11.286154
    # vs an asked-for 11.286153) doesn't trip exact-key lookups.
    by_s: dict = {}
    for r in recs:
        by_s.setdefault(round(r["strength"], 4), {})[r["question_idx"]] = r
    by_s_q = by_s
    exp_name = experiment_dir.split("/")[-1]
    print(f"# {exp_name}  cell={cell}")
    print()
    print(f"Baselines: {len(bls)} questions.  Persona system prompt:")
    print()
    with open(f"{experiment_dir}/persona_system_prompt.txt") as f:
        print("> " + f.read().strip().replace("\n", "\n> "))
    print()
    for s in strengths:
        rounded = round(s, 4)
        # Find nearest available strength key (within 0.01 tolerance).
        match = None
        for k in by_s_q.keys():
            if abs(k - rounded) < 0.01:
                match = k
                break
        if match is None:
            print(f"## strength={s} (NO RECORDS)")
            continue
        q_dict = by_s_q[match]
        qs_at_s = sorted(q_dict.keys())
        if not qs_at_s:
            print(f"## strength={s} (NO RECORDS)")
            continue
        any_rec = q_dict[qs_at_s[0]]
        smc = (any_rec.get("judges") or {}).get("strength_mean_coh")
        lens = []
        for qi in qs_at_s:
            r = q_dict[qi]
            bl = bl_by_q.get(qi, {})
            br = bl.get("response", "")
            sr = r.get("response", "")
            if br and sr:
                lens.append(len(sr) / max(1, len(br)))
        print(f"## strength={s}  (mean_coh={smc}; per-strength len_ratio "
              f"mean={statistics.mean(lens):.2f} min={min(lens):.2f})")
        print()
        for qi in qs_at_s:
            r = q_dict[qi]
            bl = bl_by_q.get(qi, {})
            br = bl.get("response", "")
            sr = r.get("response", "")
            len_ratio = len(sr) / max(1, len(br))
            judges = r.get("judges") or {}
            coh = judges.get("coherence") or {}
            persona = judges.get("persona") or {}
            eff = (judges.get("effect") or {})
            scores = (eff.get("bidirectional") or {}).get("scores") or {}
            gpt = scores.get("gpt-4.1-mini", {}) if isinstance(scores.get("gpt-4.1-mini"), dict) else {}
            claude = scores.get("claude-haiku-4-5-20251001", {}) if isinstance(scores.get("claude-haiku-4-5-20251001"), dict) else {}
            print(f"### q{qi} — len_ratio = {len_ratio:.2f}")
            print()
            print(f"**Question**: {r.get('question', '?')}")
            print()
            print(f"**Baseline response** ({len(br)} chars):")
            print()
            print(fmt_truncated(br))
            print()
            print(f"**Steered response** ({len(sr)} chars):")
            print()
            print(fmt_truncated(sr))
            print()
            print(f"**Judges**:")
            print(f"- coherence: `{coh.get('score')}` — *{(coh.get('reason') or '')}*")
            print(f"- persona (RP): `{persona.get('score')}` — *{(persona.get('reason') or '')}*")
            print(f"- effect.combined: `{eff.get('combined')}`")
            print(f"  - gpt-4.1-mini: `{gpt.get('score')}` — *{(gpt.get('reason') or '')}*")
            print(f"  - claude-haiku: `{claude.get('score')}` — *{(claude.get('reason') or '')}*")
            print()
            print("---")
            print()


# Chef +1 at strengths around the rising-coh region
print("# Coherence rubric audit: judge calls vs reality on length-collapse")
print()
print("Focus: chef_helpful_v2 +1 (toward unhelpful) and architect_ecocentric_v2 +1")
print("(toward anthropocentric) -- the two cells where I flagged the rubric")
print("as possibly accepting severe truncation as steering-direction-correct")
print("when it might actually be incoherence pathology.")
print()
print("## My assessment after reading through")
print()
print("**The rubric is mostly correct** on these cases.  Of the ~30")
print("response-baseline pairs sampled at chef +1 strengths 4.75-6.71,")
print("most coh judge calls are defensible:")
print()
print("- **Substantive low-effort responses** (\"Maybe it's a loose wheel")
print("  bearing\" / \"have someone else give the speech\" / \"tear up paper\")")
print("  correctly score `coh=0` -- the trait permits brevity, the model")
print("  IS engaging with the topic at low effort, which is what the")
print("  active-refuses unhelpful definition calls for.")
print()
print("- **Clear pathologies** correctly catch `coh=2/3`:")
print("  - q2 (cat) echoing the question back -> `coh=2`")
print("  - q4 (anxiety) word-salad confusion -> `coh=2`")
print("  - q5 (groceries) non sequitur -> `coh=2`")
print("  - q8 (faucet) echoing the persona prompt -> `coh=2`")
print("  - q6 (neighbor's tree) repetition trap -> `coh=3`")
print()
print("**The one specific blind spot** is the *boilerplate refusal* mode")
print("at chef +1 s=5.65: q3 (grass stains), q5 (groceries), q6 (tree)")
print("all received the SAME 38-char response \"I'm sorry, but I can't")
print("help with that.\" regardless of question.  Judge scored q5 as")
print("`coh=2` (probably because of a `</think>` thinking-trace artifact")
print("at the start) but q3 and q6 as `coh=0` (\"brief but on-topic\").")
print()
print("Same text, different scores -- the judge is being inconsistent")
print("AND arguably wrong on q3/q6 since:")
print("  - The response contains zero grass-stain or tree content;")
print("    \"on-topic\" is a stretch.")
print("  - The same canned text appears for unrelated questions.")
print()
print("But it's *plausibly* right: the unhelpful trait definition includes")
print("\"finds excuses not to help, deflects or refuses requests\", so a")
print("pure refusal IS a legitimate trait expression.  The pathological")
print("part is that it's the SAME refusal for every question, which a")
print("per-record judge cannot directly observe.")
print()
print("## Recommendation")
print()
print("Two options, your call:")
print()
print("**Option A** -- minor rubric tweak (low risk):  Add a level-2")
print("clause: \"A canned generic refusal/disclaimer (e.g. 'I can't")
print("help with that') that contains no topic-specific content from the")
print("question is non-engagement (level 2), even if the steering")
print("direction is unhelpful or concise.\"  Re-bump COHERENCE_RUBRIC")
print("to v6 and re-judge coherence on existing records.")
print()
print("**Option B** -- code-side cross-question check (separate from")
print("rubric):  Add a per-strength \"response uniformity\" detector that")
print("flags strengths where >=N records have textually-identical or")
print("near-identical responses (cosine similarity of normalized text)")
print("regardless of question.  Treat such strengths as length-collapse")
print("pathology in downstream consumers without changing the per-record")
print("coh score.  Better signal-to-noise but more code.")
print()
print("**Option C** -- both.")
print()
print("Recommend Option A as a first step -- it's small, addresses the")
print("specific blind spot, and the next sweep will produce cleaner")
print("coherence data automatically.  Option B remains valuable as a")
print("downstream filter.")
print()
print("---")
print()
print("Below: the raw evidence.  Each example shows the question,")
print("baseline response, steered response, and all judge calls.")
print()
print("---")
print()
dump_strengths(
    "/workspace/outputs/qwen-3-32b/steering/chef_helpful_v2",
    "s3_l25_+1",
    [4.749377, 5.647009, 6.714294],
)
print()
print("=" * 70)
print()
dump_strengths(
    "/workspace/outputs/qwen-3-32b/steering/architect_ecocentric_v2",
    "s3_l25_+1",
    [11.286154, 13.419182],
)
