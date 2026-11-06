"""Averaged judge cells: the passes of one judge (shuffle seeds), or the passes of several judges (a
blend).  Per entity the mean of the scores present; in a blend, each judge's mean over its own passes,
then the judges weighted by ``judge_score_combine.tier_judge_weights`` (Luna 4, another judge 3, since
2026-11-03; equal before), recorded as ``averaged.judge_weights``.  Usage the sum of all, correlations
recomputed.  A blend made under other weights is made again.

A source holding under ``min_coverage`` of the axis's entities is refused.  On 2026-11-02 eight empty
cells, left by an outage, were averaged in without a word, and quietly turned two passes into one on
those axes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from assistant_axis.atomic_io import atomic_write_text
from assistant_axis.judge_pricing import MultiModelUsage
from assistant_axis.judge_score_combine import combine_tier_judges, tier_judge_weights
from results_analysis import judge_tier_cost_eval as ev
from results_analysis.axis_judge_correlation import compute_correlations

#: The least of an axis a finished cell may hold and still be averaged: one batch of 20 lost on a
#: 573-entity axis leaves 96.5%.
DEFAULT_MIN_COVERAGE = 0.9


@dataclass
class AverageResult:
    made: int = 0
    kept: int = 0
    refused: List[str] = field(default_factory=list)   # one line per axis, for a report


def short_sources(loaded: Sequence[dict], sources: Sequence[str], n_entities: int,
                  min_coverage: float) -> List[Tuple[str, int]]:
    """``(source, n scored)`` for each source below ``min_coverage`` of the axis's ``n_entities``."""
    return [(s, len(x["result"])) for s, x in zip(sources, loaded) if len(x["result"]) < min_coverage * n_entities]


def average_cells(out: str, sources: Sequence[str], jobs: Sequence["ev.AxisJob"], experiments_dir: Path,
                  instructions_dir: Path, *, min_coverage: float = DEFAULT_MIN_COVERAGE,
                  limit: Optional[int] = None) -> AverageResult:
    """Write ``<axis>_tier_eval/<out>/`` for every axis of ``jobs`` whose ``sources`` are all finished;
    an axis already derived from the same source files (by mtime) is left as it is, and one with a
    short source is refused, with nothing written.  ``limit``: the sources were judged with that
    ``--limit``, so that many entities are the whole axis.  Returns what happened."""
    if len(sources) < 2:
        raise ValueError("averaging needs at least two source cells")
    E = Path(experiments_dir)
    result = AverageResult()
    judge_of = {s: s.split("@", 1)[0] for s in sources}
    judges = list(dict.fromkeys(judge_of.values()))
    for job in jobs:
        axis = job.axis_dir
        ds = [E / f"{axis}_tier_eval" / c for c in sources]
        do = E / f"{axis}_tier_eval" / out
        if not all((d / "usage.json").exists() for d in ds):
            continue
        paths = [ev.cell_scores_path(d) for d in ds]
        if len({p.name for p in paths}) != 1:
            raise ValueError(f"{axis}: the sources mix judging modes: {[p.name for p in paths]}")
        scores_name = paths[0].name
        mode = scores_name[len("scores_"):-len(".json")]
        # the judges' weights for this mode (judge_score_combine.TIER_JUDGE_WEIGHTS_BY_MODE: Jev's differ by mode)
        weights = tier_judge_weights(judges, mode) if len(judges) > 1 else None
        stamp = max(os.path.getmtime(p) for p in paths)
        if (do / "usage.json").exists():
            done = json.loads((do / scores_name).read_text())["_provenance"].get("averaged", {})
            if done.get("source_mtime") == stamp and done.get("judge_weights") == weights:
                result.kept += 1
                continue
        loaded = [json.loads(p.read_text()) for p in paths]
        if not job.projections:
            ev.load_axis(job, Path(instructions_dir), limit=limit)
        short = short_sources(loaded, sources, len(job.entities), min_coverage)
        if short:
            result.refused.append(f"{axis}: of {len(job.entities)} entities, "
                                  + ", ".join(f"{s} holds {n}" for s, n in short)
                                  + f" (under {min_coverage:.0%}); judge it again, then rerun this")
            continue
        per_judge = {}
        for j in judges:
            mine = [x["result"] for s, x in zip(sources, loaded) if judge_of[s] == j]
            per_judge[j] = {k: sum(r[k] for r in mine if k in r) / sum(1 for r in mine if k in r)
                            for k in set().union(*mine)}
        scores = combine_tier_judges(per_judge, weights) if weights else per_judge[judges[0]]
        corr = compute_correlations(scores_by_mode={mode: scores}, projections=job.projections,
                                    slots=list(job.projections.keys()), excluded_set=job.axis_spec.pole_entity_ids())
        usage = MultiModelUsage.load_or_create(ds[0] / "usage.json")
        for d in ds[1:]:
            usage.merge_from(MultiModelUsage.load_or_create(d / "usage.json"))
        prov = dict(loaded[0]["_provenance"])
        prov.pop("shuffle_seed", None)
        prov["cell_label"] = f"@{out.split('@', 1)[1]}" if "@" in out else ""
        prov["averaged"] = {"sources": list(sources),
                            "n_all": sum(1 for k in scores if all(k in x["result"] for x in loaded)),
                            "n_partial": sum(1 for k in scores if not all(k in x["result"] for x in loaded)),
                            "judge_weights": weights,
                            "source_mtime": stamp, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        do.mkdir(parents=True, exist_ok=True)
        atomic_write_text(json.dumps({"result": dict(sorted(scores.items())), "_provenance": prov}, indent=2),
                          do / scores_name)
        atomic_write_text(json.dumps(corr, indent=2), do / "correlations.json")
        # usage.json last: the file that says a cell is finished
        atomic_write_text(json.dumps(usage.as_dict(), indent=2, sort_keys=True), do / "usage.json")
        result.made += 1
    return result
