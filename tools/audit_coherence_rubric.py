"""Inspect responses near the coherence cliff to test if the rubric is
catching all incoherence or missing some pathologies (length collapse,
repetition, off-task drift).

This is a one-off diagnostic; results inform whether COHERENCE_RUBRIC
needs a bump.  Run on the pod after a rejudge.
"""
import json
import glob
import re
import statistics
from collections import Counter, defaultdict


def length_signal(text: str, baseline_len: int) -> float:
    if not text:
        return 0.0
    return len(text) / max(1, baseline_len)


def repetition_score(text: str) -> float:
    """0.0 = no repetition, 1.0 = pure repetition.  4-gram repeat
    density."""
    words = text.split()
    if len(words) < 8:
        return 0.0
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    if not grams:
        return 0.0
    c = Counter(grams)
    repeats = sum(n - 1 for n in c.values() if n > 1)
    return repeats / max(1, len(grams))


def off_task_score(text: str, question: str) -> float:
    """Fraction of question content-words NOT mentioned in response.
    1.0 = totally off-task; 0.0 = all topic words appear."""
    q_words = set(w.lower() for w in re.findall(r"[a-zA-Z]{4,}", question))
    if not q_words:
        return 0.0
    r_words = set(w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text or ""))
    overlap = len(q_words & r_words) / max(1, len(q_words))
    return 1.0 - overlap


def main():
    for exp_dir in [
        "/workspace/outputs/qwen-3-32b/steering/architect_ecocentric_v2",
        "/workspace/outputs/qwen-3-32b/steering/chef_helpful_v2",
    ]:
        print()
        print("##### " + exp_dir.split("/")[-1] + " #####")
        bls = [json.loads(line) for line in open(exp_dir + "/baselines/records.jsonl") if line.strip()]
        bl_by_q = {int(b["question_idx"]): b["response"] for b in bls}
        for cell in sorted(glob.glob(exp_dir + "/s3_l25_*/records.jsonl")):
            sign_part = cell.split("/")[-2].split("_")[-1]
            recs = [json.loads(line) for line in open(cell) if line.strip()]
            by_s = defaultdict(list)
            for r in recs:
                by_s[r["strength"]].append(r)
            transition = []
            for s in sorted(by_s):
                rs = by_s[s]
                j = rs[0].get("judges") or {}
                mc = j.get("strength_mean_coh")
                if mc is None:
                    continue
                if 0.0 < mc < 2.5:
                    transition.append((s, mc, rs))
            if not transition:
                # No transition strength; sample the LAST non-skipped strength
                for s in reversed(sorted(by_s)):
                    rs = by_s[s]
                    eff = (rs[0].get("judges") or {}).get("effect") or {}
                    if not eff.get("skipped_due_to_strength_mean_coh"):
                        mc = rs[0].get("judges", {}).get("strength_mean_coh", 0.0)
                        transition.append((s, mc, rs))
                        break
            print()
            print("=== " + sign_part + " ===")
            for s, mc, rs in transition[:5]:
                lens = [length_signal(r["response"], len(bl_by_q.get(r["question_idx"], ""))) for r in rs]
                reps = [repetition_score(r["response"]) for r in rs]
                offt = [off_task_score(r["response"], r["question"]) for r in rs]
                print()
                print("  strength=%.4f  mean_coh=%.2f" % (s, mc))
                print("    len_ratio: mean=%.2f min=%.2f max=%.2f" % (
                    statistics.mean(lens), min(lens), max(lens)))
                print("    repetition: mean=%.3f max=%.3f" % (
                    statistics.mean(reps), max(reps)))
                print("    off_task: mean=%.2f max=%.2f" % (
                    statistics.mean(offt), max(offt)))
                print("    per-Q coh + pathologies:")
                for r in sorted(rs, key=lambda r: r["question_idx"]):
                    qi = r["question_idx"]
                    c = (r.get("judges") or {}).get("coherence") or {}
                    sc = c.get("score") if isinstance(c.get("score"), int) else "?"
                    lr = length_signal(r["response"], len(bl_by_q.get(qi, "")))
                    rp = repetition_score(r["response"])
                    ot = off_task_score(r["response"], r["question"])
                    flags = []
                    if lr < 0.4:
                        flags.append("SHORT(%.2f)" % lr)
                    if rp > 0.1:
                        flags.append("REPEAT(%.2f)" % rp)
                    if ot > 0.95:
                        flags.append("OFFTASK(%.2f)" % ot)
                    print("      q%d coh=%s len_r=%.2f rep=%.3f ot=%.2f %s  | %s" % (
                        qi, sc, lr, rp, ot, " ".join(flags),
                        (c.get("reason") or "")[:80]))


if __name__ == "__main__":
    main()
