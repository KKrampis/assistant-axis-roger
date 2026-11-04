"""Derive an averaged cell from two or more batched-judging cells (results_analysis/judge_tier_average.py):
shuffle seeds of one judge, or the passes of several judges.  Exits 1 when an axis's source was refused.

Usage: uv run python scripts/judge_tier_average_cells.py claude-haiku-5-5@b10x2 \
           claude-haiku-5-5@b10s3 claude-haiku-5-5@b10s4 --instructions_dir /tmp/corpus_2026-05-24/data"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from results_analysis import judge_tier_average as avg, judge_tier_cost_eval as ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+", help="two or more source cells to average")
    ap.add_argument("--experiments_dir", default="roger/axis_judge_experiments")
    ap.add_argument("--axes_file", default="scripts/tier_eval_axes.tsv")
    ap.add_argument("--instructions_dir", required=True, help="the corpus the cells were judged on")
    ap.add_argument("--min_coverage", type=float, default=avg.DEFAULT_MIN_COVERAGE,
                    help=f"refuse a source holding under this share of the axis's entities (default {avg.DEFAULT_MIN_COVERAGE})")
    args = ap.parse_args(argv)
    if len(args.sources) < 2:
        ap.error("give at least two source cells")
    E = Path(args.experiments_dir)
    r = avg.average_cells(args.out, args.sources, ev.jobs_from_axes_file(Path(args.axes_file), E), E,
                          Path(args.instructions_dir), min_coverage=args.min_coverage)
    for line in r.refused:
        print(f"REFUSED {line}", file=sys.stderr)
    print(f"{args.out}: {r.made} cells derived, {r.kept} up to date, {len(r.refused)} refused, from {' + '.join(args.sources)}")
    return 1 if r.refused else 0


if __name__ == "__main__":
    sys.exit(main())
