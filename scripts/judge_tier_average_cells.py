"""Derive an averaged cell from two or more batched-judging cells: shuffle seeds of one judge (the
passes), or the passes of several judges (a blend).  Per entity the mean of the scores present, usage
the sum of all, correlations recomputed.  Writes <axis>_tier_eval/<out>/ for every axis whose sources
are finished; skips axes already derived from the same source files (by mtime).

A source holding under --min_coverage of the axis's entities is refused, and the run exits 1: on
2026-11-02 eight empty cells, left by an outage, were averaged in without a word, and quietly turned
two passes into one on those axes.

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

#: The largest share of an axis a finished cell may lack and still be averaged: one batch of 20 lost
#: on a 573-entity axis is 3.5%.
DEFAULT_MIN_COVERAGE = 0.9


def short_sources(loaded, sources, n_entities, min_coverage):
    """``(source, n scored)`` for each source below ``min_coverage`` of the axis's ``n_entities``."""
    return [(s, len(x["result"])) for s, x in zip(sources, loaded) if len(x["result"]) < min_coverage * n_entities]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("sources", nargs="+", help="two or more source cells to average")
    ap.add_argument("--experiments_dir", default="roger/axis_judge_experiments")
    ap.add_argument("--axes_file", default="scripts/tier_eval_axes.tsv")
    ap.add_argument("--instructions_dir", required=True, help="the corpus the cells were judged on (pinned at 1448a84c)")
    ap.add_argument("--min_coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                    help=f"refuse a source holding under this share of the axis's entities (default {DEFAULT_MIN_COVERAGE})")
    args = ap.parse_args(argv)
    E = Path(args.experiments_dir); DATA = Path(args.instructions_dir)
    out, srcs = args.out, args.sources
    if len(srcs) < 2:
        ap.error("give at least two source cells")
    made = kept = refused = 0
    for job in ev.jobs_from_axes_file(Path(args.axes_file), E):
        axis = job.axis_dir
        ds = [E / f"{axis}_tier_eval" / c for c in srcs]; do = E / f"{axis}_tier_eval" / out
        if not all((d / "usage.json").exists() for d in ds):
            continue
        stamp = max(os.path.getmtime(d / "scores_descriptions.json") for d in ds)
        if (do / "usage.json").exists():
            prov = json.load(open(do / "scores_descriptions.json"))["_provenance"]
            if prov.get("averaged", {}).get("source_mtime") == stamp:
                kept += 1; continue
        loaded = [json.load(open(d / "scores_descriptions.json")) for d in ds]
        sa = loaded[0]
        if not job.projections:
            ev.load_axis(job, DATA)
        short = short_sources(loaded, srcs, len(job.entities), args.min_coverage)
        if short:
            print(f"REFUSED {axis}: of {len(job.entities)} entities, " + ", ".join(f"{s} holds {n}" for s, n in short)
                  + f" (under {args.min_coverage:.0%}); rejudge it, then rerun this", file=sys.stderr)
            refused += 1
            continue
        scores = {}
        for k in set().union(*(x["result"] for x in loaded)):
            v = [x["result"][k] for x in loaded if k in x["result"]]
            scores[k] = sum(v) / len(v)
        corr = compute_correlations(scores_by_mode={"descriptions": scores}, projections=job.projections,
                                    slots=list(job.projections.keys()), excluded_set=set(job.axis_spec.exclusions))
        usage = MultiModelUsage.load_or_create(ds[0] / "usage.json")
        for d in ds[1:]:
            usage.merge_from(MultiModelUsage.load_or_create(d / "usage.json"))
        prov = dict(sa["_provenance"]); prov.pop("shuffle_seed", None); prov["cell_label"] = f"@{out.split('@', 1)[1]}" if "@" in out else ""
        prov["averaged"] = {"sources": list(srcs),
                            "n_all": sum(1 for k in scores if all(k in x["result"] for x in loaded)),
                            "n_partial": sum(1 for k in scores if not all(k in x["result"] for x in loaded)),
                            "source_mtime": stamp, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        do.mkdir(parents=True, exist_ok=True)
        atomic_write_text(json.dumps({"result": dict(sorted(scores.items())), "_provenance": prov}, indent=2), do / "scores_descriptions.json")
        atomic_write_text(json.dumps(corr, indent=2), do / "correlations.json")
        atomic_write_text(json.dumps(usage.as_dict(), indent=2, sort_keys=True), do / "usage.json")
        made += 1
    print(f"{out}: {made} cells derived, {kept} up to date, {refused} refused, from {' + '.join(srcs)}")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
