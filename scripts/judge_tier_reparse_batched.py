"""Re-score batched judge-tier cells offline from their archived replies
(answers.jsonl.gz / answers.partial.jsonl) with the current batch parser;
rewrites scores_descriptions.json and correlations.json, keeps usage.json.  Used on
2026-11-01 after the batch parser learnt Haiku 5.5's layout (the SCORE on the line
after the item) and after a resume bug (fixed the same day) had emptied cells.
--fill_only (2026-11-02, after the parser learnt scores in words) keeps every recorded
score and adds only the entities that had none.

Usage: uv run python scripts/judge_tier_reparse_batched.py claude-haiku-5-5@b10s3 \
           --instructions_dir /tmp/corpus_2026-05-24/data"""
import argparse, gzip, json, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from results_analysis import judge_tier_cost_eval as ev, judge_tier_rubrics as jr
from results_analysis.axis_judge_correlation import compute_correlations
from assistant_axis.atomic_io import atomic_write_text
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("cells", nargs="+", help="cell directory names, e.g. claude-haiku-5-5@b10s3")
ap.add_argument("--experiments_dir", default="roger/axis_judge_experiments")
ap.add_argument("--axes_file", default="scripts/tier_eval_axes.tsv")
ap.add_argument("--instructions_dir", required=True, help="the corpus the cells were judged on (pinned at 1448a84c)")
ap.add_argument("--fill_only", action="store_true", help="keep every recorded score; add only entities that had none")
ap.add_argument("--note", default="batch parser accepts the SCORE on the line after the item", help="recorded in the cell's provenance")
args = ap.parse_args()
E = Path(args.experiments_dir); DATA = Path(args.instructions_dir)
jobs = {j.axis_dir: j for j in ev.jobs_from_axes_file(Path(args.axes_file), E)}
cells = args.cells
for cell in cells:
    rubric = jr.get("b10" if "@b10" in cell else "b5")
    changed = tot = 0
    for d in sorted(E.glob(f"*_tier_eval/{cell}")):
        axis = d.parent.name[: -len("_tier_eval")]
        sp = d / "scores_descriptions.json"
        if not sp.exists():
            continue
        logp = d / "answers.jsonl.gz"; raw = gzip.decompress(logp.read_bytes()).decode() if logp.exists() else (d / "answers.partial.jsonl").read_text()
        recs = [json.loads(l) for l in raw.splitlines() if l.strip()]
        recs = [r for r in recs if r.get("kind") == "answer"]
        parsed_reply = {}
        scores = {}
        for r in recs:
            names = r.get("batch") or [r["name"]]
            key = id(r["reply"]) if False else (tuple(names))
            if key not in parsed_reply:
                parsed_reply[key] = {} if r.get("truncated") or not r["reply"] else rubric.parse_batch(r["reply"], len(names))
            k = names.index(r["name"]) + 1
            v = parsed_reply[key].get(k) if parsed_reply[key] else None
            if v is not None:
                scores[r["name"]] = int(v)
        old = json.load(open(sp)); prov = old["_provenance"]
        n_old = len(old["result"]); tot += 1
        if args.fill_only:
            scores = {**scores, **old["result"]}
        if scores == {k: int(v) for k, v in old["result"].items()}:
            continue
        job = jobs[axis]
        if not job.projections:
            ev.load_axis(job, DATA)
        corr = compute_correlations(scores_by_mode={"descriptions": scores}, projections=job.projections,
                                    slots=list(job.projections.keys()), excluded_set=set(job.axis_spec.exclusions))
        prov["reparsed"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "scored_before": n_old,
                            "scored_after": len(scores), "note": args.note}
        atomic_write_text(json.dumps({"result": dict(sorted(scores.items())), "_provenance": prov}, indent=2), sp)
        atomic_write_text(json.dumps(corr, indent=2), d / "correlations.json")
        changed += 1
    print(f"{cell}: {tot} cells, {changed} rewritten")
