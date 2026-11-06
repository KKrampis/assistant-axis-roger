"""Rebuild the May 2026 8-slot trait and role vectors WITHOUT the step-3 filter.

The May vectors are the mean activation over each entity's responses that
gpt-4.1-mini scored 3 (pipeline/4_vectors.py, compute_pos_3_vector).  The
unfiltered vector is the mean over all responses, computed exactly as
    (n3 * stored + sum over the other responses) / n_all
so only the responses not scored 3 are read from the activation files (the
G-Drive archive, read only; legacy torch files read selectively by legacy_pt).
Validated on roles/screener (21% kept): the shortcut matches the direct mean
to within the bf16 rounding of the stored vector.

Output: <out>/{traits,roles}/vectors/<name>.pt (float32, same dict layout, plus
metadata), a symlink to the May file where every response scored 3;
<out>/traits/vectors/default.pt and <out>/combinations -> the May ones (the
default persona and the soft-shear subspaces stay as in May).  Resumable: an
existing output is skipped.  Writes per-entity stats to <out>/unfiltered_stats.jsonl.
"""
import json, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import legacy_pt as lp

G = Path("/Volumes/G-Drive/assistant-axis/runpod_workspace/qwen/qwen-3-32b Roger 8slot")
L = Path(sys.argv[1]).resolve()        # the May 8-slot data dir (local)
OUT = Path(sys.argv[2]).resolve()      # the new data dir
for kind in ("traits", "roles"):
    (OUT / kind / "vectors").mkdir(parents=True, exist_ok=True)
for link, target in ((OUT / "traits" / "vectors" / "default.pt", L / "traits" / "vectors" / "default.pt"),
                     (OUT / "roles" / "vectors" / "default.pt", L / "roles" / "vectors" / "default.pt"),
                     (OUT / "combinations", L / "combinations")):
    if target.exists() and not link.exists():
        link.symlink_to(target)
stats_path = OUT / "unfiltered_stats.jsonl"
t_start, done, read_bytes = time.time(), 0, 0
todo = [(k, fp) for k in ("traits", "roles") for fp in sorted((L / k / "vectors").glob("*.pt")) if fp.stem != "default"]
print(f"{len(todo)} entities", flush=True)
for i, (kind, fp) in enumerate(todo):
    name = fp.stem
    out = OUT / kind / "vectors" / f"{name}.pt"
    if out.exists() or out.is_symlink():
        continue
    scores = json.load(open(L / kind / "scores" / f"{name}.json"))
    act = G / kind / "activations" / f"{name}.pt"
    rec = {"kind": kind, "name": name}
    if not act.exists():
        rec["error"] = "no activation file on the G-Drive; May vector kept"
        out.symlink_to(fp)
        open(stats_path, "a").write(json.dumps(rec) + "\n")
        print(f"[{i+1}/{len(todo)}] {kind}/{name}: NO ACTIVATIONS, May vector kept", flush=True)
        continue
    obj, st, f, _ = lp.scan(act, verify="first")
    keys = [k for k in obj if k != "metadata"]
    non3 = [k for k in keys if scores.get(k) != 3]
    n, n3 = len(keys), len(keys) - len(non3)
    stored = torch.load(fp, map_location="cpu", weights_only=False)
    vec = stored["vector"].to(torch.float64)
    rec.update(n_all=n, n_score3=n3, unscored=sum(1 for k in keys if k not in scores))
    if not non3:
        out.symlink_to(fp)
        rec.update(rel_shift_s6_l25=0.0, cos_s6_l25=1.0)
    else:
        s = torch.zeros_like(vec); nan = 0
        for k in sorted(non3, key=lambda k: st[obj[k].key][0]):   # file order: fewer seeks
            x = lp.read_tensor(f, st, obj[k]).to(torch.float64)
            m = torch.isnan(x)
            if m.any():
                nan += int(m.sum()); x = torch.where(m, torch.zeros_like(x), x)
            s += x
            read_bytes += x.numel() * 2
        unf = (n3 * vec + s) / n
        a, b = vec[6, 25], unf[6, 25]
        rec.update(rel_shift_s6_l25=float((b - a).norm() / a.norm()),
                   cos_s6_l25=float(torch.nn.functional.cosine_similarity(a, b, dim=0)), nan_entries=nan)
        new = dict(stored)
        new["vector"] = unf.to(torch.float32)
        md = dict(stored.get("metadata") or {})
        md.update(unfiltered=True, n_all=n, n_score3=n3,
                  note="mean over all responses, not only those step 3 scored 3 (2026-11-01 check)")
        new["metadata"] = md
        tmp = out.with_suffix(".pt.tmp")
        torch.save(new, tmp)
        os.replace(tmp, out)
    f.close()
    open(stats_path, "a").write(json.dumps(rec) + "\n")
    done += 1
    el = time.time() - t_start
    print(f"[{i+1}/{len(todo)}] {kind}/{name}: kept {n3}/{n}, shift {rec.get('rel_shift_s6_l25', 0):.4f}; "
          f"{read_bytes/1e9:.1f} GB read in {el/60:.1f} min", flush=True)
print(f"done: {done} entities in {(time.time()-t_start)/60:.1f} min", flush=True)
