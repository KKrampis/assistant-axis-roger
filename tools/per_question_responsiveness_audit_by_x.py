"""Follow-up: which questions fire even at WEAK steering (high x)?

Reads /tmp/q_responsiveness.jsonl and breaks out strong-rate vs weak-rate
per x position, so we can see which questions are "low-perturbation
sensitive" (fire strong even at x=8 / x=16) vs which are "near-cliff only"
(only fire strong at x=0,1).
"""
import json
from collections import defaultdict
from pathlib import Path

ROWS = [json.loads(l) for l in Path("/tmp/q_responsiveness.jsonl").read_text().splitlines() if l.strip()]

# Global: which questions appear strong at high-x (weak steering)?
# Compute per (axis, qi): strong-rate at x in {4,8,16} divided by n_at_x there.
weak_sensitive = []
for r in ROWS:
    n_at_x = r["n_at_x"]
    s_at_x = r["strong_at_x"]
    w_at_x = r["weak_at_x"]
    n_far = sum(n_at_x.get(str(x), n_at_x.get(x, 0)) for x in (4, 8, 16))
    s_far = sum(s_at_x.get(str(x), s_at_x.get(x, 0)) for x in (4, 8, 16))
    w_far = sum(w_at_x.get(str(x), w_at_x.get(x, 0)) for x in (4, 8, 16))
    if n_far < 4:
        continue
    rate = (s_far - w_far) / n_far
    weak_sensitive.append((rate, r["axis"], r["qi"], r["question"], s_far, n_far, w_far))

weak_sensitive.sort(reverse=True)
print("=== TOP 20: STRONG even at WEAK steering (x=4,8,16) — high-leverage probes ===")
for rate, axis, qi, q, s, n, w in weak_sensitive[:20]:
    print(f"  {rate:+.2f} S{s:>2d}/W{w:>2d} of {n:>2d}  [{axis[:24]:24s}] q{qi:>2d}: {q[:100]}")

print("\n=== BOTTOM 20: WEAK even at NEAR-cliff steering (x=4,8,16) — wasted slots ===")
for rate, axis, qi, q, s, n, w in weak_sensitive[-20:]:
    print(f"  {rate:+.2f} S{s:>2d}/W{w:>2d} of {n:>2d}  [{axis[:24]:24s}] q{qi:>2d}: {q[:100]}")

# Also: questions that flip — strong at x=0,1 but weak at x=8,16 (only fire under strong push)
near_only = []
for r in ROWS:
    n_at_x = r["n_at_x"]
    s_at_x = r["strong_at_x"]
    w_at_x = r["weak_at_x"]
    def get(d, x): return d.get(str(x), d.get(x, 0))
    n_near = sum(get(n_at_x, x) for x in (0, 1, 2))
    n_far = sum(get(n_at_x, x) for x in (4, 8, 16))
    s_near = sum(get(s_at_x, x) for x in (0, 1, 2))
    s_far = sum(get(s_at_x, x) for x in (4, 8, 16))
    if n_near < 3 or n_far < 3:
        continue
    near_rate = s_near / n_near
    far_rate = s_far / n_far
    near_only.append((near_rate - far_rate, near_rate, far_rate, r["axis"], r["qi"], r["question"]))

near_only.sort(reverse=True)
print("\n=== TOP 15: STRONG near cliff only (high gap = near-cliff threshold question) ===")
for gap, near, far, axis, qi, q in near_only[:15]:
    print(f"  Δ{gap:+.2f} (near={near:.2f} far={far:.2f}) [{axis[:24]:24s}] q{qi:>2d}: {q[:90]}")
