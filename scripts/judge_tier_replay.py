"""Replay recorded judging through the current code, with no API calls: is what we tested what runs?

For every batched cell with an answer log, rebuild each logged prompt from the corpus with the cell's
recorded settings (--name_format mechanical for the cells judged before 2026-11-02) and compare its digest
with the digest logged beside the reply; re-parse each logged reply with the current parser and compare it
with the cell's recorded score.  A rating may only go from missing to a score (a spelled-out score the parser
of the time could not read).  Exit status 1 when anything else differs.

Usage: uv run python scripts/judge_tier_replay.py --axes_file scripts/tier_eval_axes.tsv \
           --instructions_dir CORPUS_AT_1448a84c [--cells PATTERN ...]"""
import argparse, fnmatch, gzip, json, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from results_analysis import judge_tier_cost_eval as ev, judge_tier_rubrics as jr


def answers_of(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") == "answer":
                yield rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments_dir", default="roger/axis_judge_experiments")
    ap.add_argument("--axes_file", default="scripts/tier_eval_axes.tsv")
    ap.add_argument("--instructions_dir", required=True, help="the corpus the cells were judged on")
    ap.add_argument("--cells", nargs="*", default=["*@b10*"], help="cell name patterns")
    ap.add_argument("--name_format", choices=list(jr.NAME_FORMATS), default="mechanical", help="as the cells were judged")
    ap.add_argument("--report", default=None, help="write the counts here as JSON")
    args = ap.parse_args(argv)
    E, data = Path(args.experiments_dir), Path(args.instructions_dir)
    show = jr.name_shower(args.name_format, data)
    n = Counter()
    bad = []
    for job in ev.jobs_from_axes_file(Path(args.axes_file), E):
        tier = E / f"{job.axis_dir}{ev.TIER_EVAL_SUFFIX}"
        if not tier.is_dir() or not job.projections_file.exists():
            continue
        ev.load_axis(job, data)
        # every name the projections list: until the port the tool judged a pole they listed, and left it out of rho
        listed, _gone = ev.resolve_entities(sorted(next(iter(job.projections.values()))), data)
        entities = {name: (etype, desc) for etype, name, desc in listed}
        for d in sorted(p for p in tier.iterdir() if p.is_dir()):
            if not any(fnmatch.fnmatchcase(d.name, pat) for pat in args.cells) or not (d / "answers.jsonl.gz").exists():
                continue
            cell = json.loads((d / "scores_descriptions.json").read_text())
            prov = cell["_provenance"]
            if prov.get("averaged") or prov.get("rubric") != "b10":
                continue
            rubric = jr.get(prov["rubric"])
            n["cells"] += 1
            result, seen = cell["result"], set()
            groups = {}
            for rec in answers_of(d / "answers.jsonl.gz"):
                if rec.get("batch"):
                    groups.setdefault(tuple(rec["batch"]), {})[rec["name"]] = rec
            built = ev.axis_groups(SimpleNamespace(entities=listed), SimpleNamespace(
                shuffle_seed=prov.get("shuffle_seed", 0), entities_per_call=prov.get("entities_per_call", 1),
                batch_design=prov.get("batch_design", "shuffle")))
            same = sum(tuple(e[1] for e in g) in groups for g in built)
            n["groups_built"] += len(built)
            n["groups_same"] += same
            if prov.get("batch_design", "shuffle") == "shuffle" and same != len(built):
                n["cells_with_other_groups"] += 1
            for names, recs in groups.items():
                rec = next(iter(recs.values()))
                n["prompts"] += 1
                if not all(x in entities for x in names):
                    n["entity_gone"] += 1
                    continue
                prompt = rubric.build_batch(job.axis_spec, job.pair_type, [(*entities[x][:1], x, entities[x][1]) for x in names], show=show)
                if ev.prompt_digest(prompt) == rec.get("prompt"):
                    n["prompt_same"] += 1
                else:
                    n["prompt_differs"] += 1
                    bad.append(f"prompt: {job.axis_dir} {d.name} {names[0]}")
                for k, name in enumerate(names, 1):
                    own = recs.get(name)
                    if own is None or own.get("truncated") or not own.get("reply"):
                        continue
                    # a member's record holds the reply that gave it a line: a re-asked group has several
                    parsed = rubric.parse_batch(own["reply"], len(names))
                    seen.add(name)
                    new, old = parsed.get(k), result.get(name)
                    n["ratings"] += 1
                    if new == old:
                        n["rating_same"] += 1
                    elif old is None and new is not None:
                        n["rating_gained"] += 1
                    elif new is None:
                        n["rating_lost"] += 1
                        bad.append(f"rating lost: {job.axis_dir} {d.name} {name} was {old}")
                    else:
                        n["rating_changed"] += 1
                        bad.append(f"rating changed: {job.axis_dir} {d.name} {name} {old} -> {new}")
    print(json.dumps(dict(n), indent=1, sort_keys=True))
    for line in bad[:40]:
        print(line)
    if len(bad) > 40:
        print(f"... and {len(bad) - 40} more")
    if args.report:
        Path(args.report).write_text(json.dumps({"counts": dict(n), "differences": bad}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
