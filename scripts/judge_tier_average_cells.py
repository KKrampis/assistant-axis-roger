"""Derive an averaged cell from two batched-judging cells of the same judge (two
shuffle seeds): per entity the mean of the two scores (or the one present),
usage the sum of both, correlations recomputed.  Writes <axis>_tier_eval/<out>/
for every axis where both sources are complete; skips axes already derived
from the same source files (by mtime).

Usage: uv run python scripts/judge_tier_average_cells.py claude-haiku-5-5@b10x2 \
           claude-haiku-5-5@b10s3 claude-haiku-5-5@b10s4 --instructions_dir /tmp/corpus_2026-05-24/data"""
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from results_analysis import judge_tier_cost_eval as ev
from results_analysis.axis_judge_correlation import compute_correlations
from assistant_axis.judge_pricing import MultiModelUsage
from assistant_axis.atomic_io import atomic_write_text
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out"); ap.add_argument("src1"); ap.add_argument("src2")
ap.add_argument("--experiments_dir", default="roger/axis_judge_experiments")
ap.add_argument("--axes_file", default="scripts/tier_eval_axes.tsv")
ap.add_argument("--instructions_dir", required=True, help="the corpus the cells were judged on (pinned at 1448a84c)")
args = ap.parse_args()
E = Path(args.experiments_dir); DATA = Path(args.instructions_dir)
out, a, b = args.out, args.src1, args.src2
jobs = {j.axis_dir: j for j in ev.jobs_from_axes_file(Path(args.axes_file), E)}
made = kept = 0
for axis, job in jobs.items():
    da, db, do = (E / f"{axis}_tier_eval" / c for c in (a, b, out))
    if not ((da / "usage.json").exists() and (db / "usage.json").exists()):
        continue
    stamp = max(os.path.getmtime(da / "scores_descriptions.json"), os.path.getmtime(db / "scores_descriptions.json"))
    if (do / "usage.json").exists():
        prov = json.load(open(do / "scores_descriptions.json"))["_provenance"]
        if prov.get("averaged", {}).get("source_mtime") == stamp:
            kept += 1; continue
    sa, sb = (json.load(open(d / "scores_descriptions.json")) for d in (da, db))
    scores = {}
    for k in set(sa["result"]) | set(sb["result"]):
        v = [x[k] for x in (sa["result"], sb["result"]) if k in x]
        scores[k] = sum(v) / len(v)
    if not job.projections:
        ev.load_axis(job, DATA)
    corr = compute_correlations(scores_by_mode={"descriptions": scores}, projections=job.projections,
                                slots=list(job.projections.keys()), excluded_set=set(job.axis_spec.exclusions))
    usage = MultiModelUsage.load_or_create(da / "usage.json"); usage.merge_from(MultiModelUsage.load_or_create(db / "usage.json"))
    prov = dict(sa["_provenance"]); prov.pop("shuffle_seed", None); prov["cell_label"] = f"@{out.split('@', 1)[1]}" if "@" in out else ""
    prov["averaged"] = {"sources": [a, b], "n_both": sum(1 for k in scores if k in sa["result"] and k in sb["result"]),
                        "n_one": sum(1 for k in scores if (k in sa["result"]) != (k in sb["result"])),
                        "source_mtime": stamp, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    do.mkdir(parents=True, exist_ok=True)
    atomic_write_text(json.dumps({"result": dict(sorted(scores.items())), "_provenance": prov}, indent=2), do / "scores_descriptions.json")
    atomic_write_text(json.dumps(corr, indent=2), do / "correlations.json")
    atomic_write_text(json.dumps(usage.as_dict(), indent=2, sort_keys=True), do / "usage.json")
    made += 1
print(f"{out}: {made} cells derived, {kept} up to date, from {a} + {b}")
