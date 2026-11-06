"""Re-score batched judge-tier cells offline from their archived replies
(answers.jsonl.gz / answers.partial.jsonl) with the current batch parser;
rewrites scores_<mode>.json and correlations.json, keeps usage.json.  Any batched rubric: the cell's
name says which (b5, b10, the i1_*/is_* instruction arms, the r1_* response arms, whose cells hold each
response's score under "items" and the per-entity means under "result").  Used on
2026-11-01 after the batch parser learnt Haiku 5.5's layout (the SCORE on the line
after the item) and after a resume bug (fixed the same day) had emptied cells.
--fill_only (2026-11-02, after the parser learnt scores in words) keeps every recorded
score and adds only the entities that had none.

Usage: uv run python scripts/judge_tier_reparse_batched.py claude-haiku-5-5@b10s3 \
           --instructions_dir /tmp/corpus_2026-05-24/data"""
import argparse, gzip, json, re, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from results_analysis import judge_tier_cost_eval as ev, judge_tier_rubrics as jr
from results_analysis.axis_judge_correlation import compute_correlations
from assistant_axis.atomic_io import atomic_write_text


def rubric_of_cell(cell: str) -> "jr.RubricSpec":
    """The batched rubric a cell's name carries: the longest rubric name after an "@" that a batch size,
    seed, pass count, round or "_" follows (so "@r1_hnn20" is r1_hn, not r1_h)."""
    found = [n for n in jr.RUBRICS if re.search(rf"@{re.escape(n)}(?:[nsxr]\d|_|$)", cell)]
    if not found:
        raise SystemExit(f"{cell}: no batched rubric in the cell name")
    return jr.get(max(found, key=len))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+", help="cell directory names, e.g. claude-haiku-5-5@b10s3")
    ap.add_argument("--experiments_dir", default="roger/axis_judge_experiments")
    ap.add_argument("--axes_file", default="scripts/tier_eval_axes.tsv")
    ap.add_argument("--instructions_dir", required=True, help="the corpus the cells were judged on (pinned at 1448a84c)")
    ap.add_argument("--fill_only", action="store_true", help="keep every recorded score; add only entities that had none")
    ap.add_argument("--note", default="batch parser accepts the SCORE on the line after the item", help="recorded in the cell's provenance")
    args = ap.parse_args(argv)
    E = Path(args.experiments_dir); DATA = Path(args.instructions_dir)
    jobs = {j.axis_dir: j for j in ev.jobs_from_axes_file(Path(args.axes_file), E)}
    cells = args.cells
    for cell in cells:
        rubric = rubric_of_cell(cell)
        changed = tot = 0
        for d in sorted(E.glob(f"*_tier_eval/{cell}")):
            axis = d.parent.name[: -len("_tier_eval")]
            sp = d / f"scores_{rubric.mode}.json"
            if not sp.exists():
                continue
            logp = d / "answers.jsonl.gz"; raw = gzip.decompress(logp.read_bytes()).decode() if logp.exists() else (d / "answers.partial.jsonl").read_text()
            recs = [json.loads(l) for l in raw.splitlines() if l.strip()]
            recs = [r for r in recs if r.get("kind") == "answer"]
            parsed_reply = {}
            scores = {}
            directed = any("swapped" in r for r in recs)
            for r in recs:
                names = r.get("batch") or [r["name"]]
                key = (tuple(names), r["reply"], bool(r.get("swapped")))
                if directed and "swapped" not in r:
                    raise SystemExit(f"{d}: a cell judged under --directions has an answer without its direction")
                if key not in parsed_reply:
                    got = {} if r.get("truncated") or not r["reply"] else rubric.parse_batch(r["reply"], len(names))
                    if r.get("swapped"):        # that call showed the axis swapped: back to the axis's direction
                        got = {i: None if v is None else rubric.lo + rubric.hi - v for i, v in got.items()}
                    parsed_reply[key] = got
                k = names.index(r["name"]) + 1
                v = parsed_reply[key].get(k) if parsed_reply[key] else None
                if v is not None:
                    scores[r["name"]] = int(v)
            old = json.load(open(sp)); prov = old["_provenance"]
            if prov.get("swap_poles"):      # judged with the poles swapped: parse as that run did
                swapped = jr.with_poles_swapped(rubric)
                scores = {}
                for r in recs:
                    names = r.get("batch") or [r["name"]]
                    got = {} if r.get("truncated") or not r["reply"] else swapped.parse_batch(r["reply"], len(names))
                    v = got.get(names.index(r["name"]) + 1)
                    if v is not None:
                        scores[r["name"]] = int(v)
            responses = rubric.text_source == "responses"
            recorded = old["items"] if responses else old["result"]
            n_old = len(recorded); tot += 1
            if args.fill_only:
                scores = {**scores, **recorded}
            if scores == {k: int(v) for k, v in recorded.items()}:
                continue
            job = jobs[axis]
            if not job.projections:
                job.projections = ev.load_projections(job.projections_file, "all")
                job.axis_spec = ev.build_axis_spec(job.pos, job.neg, job.pair_type, DATA, job.projections_file)
            result = ev.entity_means(scores) if responses else dict(sorted(scores.items()))
            corr = compute_correlations(scores_by_mode={rubric.mode: result}, projections=job.projections,
                                        slots=list(job.projections.keys()), excluded_set=job.axis_spec.pole_entity_ids())
            prov["reparsed"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "scored_before": n_old,
                                "scored_after": len(scores), "note": args.note}
            payload = {"result": result, "_provenance": prov}
            if responses:
                payload = {"result": result, "items": dict(sorted(scores.items())), "_provenance": prov}
                prov["n_items_scored"], prov["n_scored"] = len(scores), len(result)
            else:
                prov["n_scored"] = len(result)
            atomic_write_text(json.dumps(payload, indent=2), sp)
            atomic_write_text(json.dumps(corr, indent=2), d / "correlations.json")
            changed += 1
        print(f"{cell}: {tot} cells, {changed} rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
