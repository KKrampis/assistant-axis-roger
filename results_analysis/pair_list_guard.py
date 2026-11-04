"""Fail loudly when a pair list names an axis the data cannot serve.

Walk-through decision A2 (Roger, 2026-10-12; reviews 9a D3 and 9b D1): the
results_analysis scripts keep the *current* pair lists as their defaults, but
a script that takes a pair list must stop, naming each axis, when a listed
axis has no vector or no judge cache, instead of crashing on the first
missing file or quietly leaving the axis out of its averages.

Why it comes up: since 2026-09-28 the pair lists come in two generations
(CLAUDE.md, "Pair lists come in two generations").  ``pair_list_<cohort>.json``
describes the corpus as it is now, for the next extraction; it names poles
that the 8slot data dir has no vector for (``instrumentally_aligned_ai``,
``unreliable``) and pairs no judge cache exists for (``intuitive_vs_analytical``).
``pair_list_<cohort>_v1.json`` is the record of what was judged, and is the
list to pass with the 8slot data and its caches.

Usage, right after the script has loaded its pair list::

    from results_analysis.pair_list_guard import STATIC_DI_CACHES, check_pair_list
    check_pair_list(pairs, pair_list=experiment_dir / args.pairs,
                    script="whitening_k_sweep.py", data_dir=data_dir,
                    experiment_dir=experiment_dir, caches_all=STATIC_DI_CACHES)

Each script states what it needs per axis, and only that: the files it
reads unconditionally (``caches_all``), groups of alternatives of which one
must exist (``caches_any``, e.g. a response cohort at either batch size),
or, for scripts that read a sweep JSON instead of the caches, the set of
axes that JSON holds (``recorded_axes``).  A cache a script treats as
optional (response scores in a desc+inst cohort) is not checked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Collection, Iterable, Mapping, Optional, Sequence, Tuple

from assistant_axis.judge_batch import response_subdir
from assistant_axis.judge_loaders import (
    _DEFAULT_PREFER_B, cohort_dirname, filename_for_rubric,
)
from assistant_axis.pair_list_cohort import pair_type_of

# The four static (descriptions + instructions) judge caches the desc+inst
# consumers read for every axis, relative to ``<experiment_dir>/<axis>/``.
STATIC_DI_CACHES: Tuple[str, ...] = (
    "gpt/scores_descriptions.json",
    "gpt/scores_instructions.json",
    "sonnet/scores_descriptions.json",
    "sonnet/scores_instructions.json",
)

KINDS: Tuple[str, ...] = ("traits", "roles")


def axis_id(entry: Mapping) -> str:
    """``<pos>_vs_<neg>``, the axis directory name under the experiment dir."""
    return f"{entry['pos']}_vs_{entry['neg']}"


def gpt_response_caches(kinds: Sequence[str] = KINDS) -> Tuple[str, ...]:
    """The GPT response caches at the canonical batch size
    (``response_subdir``), one per kind, relative to the axis dir."""
    return tuple(f"{response_subdir('gpt', k)}/scores_responses.json" for k in kinds)


def loader_response_caches(judge: str, *, rubric: str = "v2",
                           kinds: Sequence[str] = KINDS,
                           prefer_b: Optional[Sequence[int]] = None) -> Tuple[str, ...]:
    """Every cohort file :func:`assistant_axis.judge_loaders.load_response_scores`
    may read for ``judge`` (all kinds, every batch size in its ``prefer_b``),
    relative to the axis dir."""
    bs = tuple(prefer_b) if prefer_b else _DEFAULT_PREFER_B[(judge, rubric)]
    fname = filename_for_rubric(rubric)
    return tuple(f"{cohort_dirname(judge, k, b)}/{fname}" for k in kinds for b in bs)


def _missing_vectors(entry: Mapping, data_dir: Path) -> list[str]:
    pt = pair_type_of(entry)
    return [f"{pt}/vectors/{n}.pt" for n in (entry["pos"], entry["neg"])
            if not (data_dir / pt / "vectors" / f"{n}.pt").exists()]


def axis_problems(
    entry: Mapping,
    *,
    data_dir: Optional[Path] = None,
    experiment_dir: Optional[Path] = None,
    caches_all: Sequence[str] = (),
    caches_any: Sequence[Tuple[str, Sequence[str]]] = (),
    recorded_axes: Optional[Collection[str]] = None,
    recorded_label: str = "the input JSON",
) -> list[str]:
    """What one pair-list entry lacks, as short phrases (empty when usable)."""
    problems: list[str] = []
    ax = axis_id(entry)
    if data_dir is not None:
        mv = _missing_vectors(entry, Path(data_dir))
        if mv:
            problems.append("no vector " + ", ".join(mv))
    if experiment_dir is not None and (caches_all or caches_any):
        adir = Path(experiment_dir) / ax
        if not adir.is_dir():
            problems.append(f"no judge caches (no directory {ax}/)")
        else:
            gone = [rel for rel in caches_all if not (adir / rel).exists()]
            if gone:
                problems.append("no judge cache " + ", ".join(gone))
            for label, alternatives in caches_any:
                if not any((adir / rel).exists() for rel in alternatives):
                    problems.append(f"no {label} (none of: {', '.join(alternatives)})")
    if recorded_axes is not None and ax not in recorded_axes:
        problems.append(f"no entry in {recorded_label}")
    return problems


def v1_alternative(pair_list: Path) -> Optional[Path]:
    """The ``_v1`` record beside a current pair list, if there is one."""
    pair_list = Path(pair_list)
    if pair_list.stem.endswith("_v1"):
        return None
    cand = pair_list.with_name(f"{pair_list.stem}_v1{pair_list.suffix}")
    return cand if cand.exists() else None


def check_pair_list(
    pairs: Iterable[Mapping],
    *,
    pair_list: Path | str,
    script: str,
    data_dir: Optional[Path] = None,
    experiment_dir: Optional[Path] = None,
    caches_all: Sequence[str] = (),
    caches_any: Sequence[Tuple[str, Sequence[str]]] = (),
    recorded_axes: Optional[Collection[str]] = None,
    recorded_label: str = "the input JSON",
) -> None:
    """Raise ``SystemExit`` naming every listed axis that lacks a vector
    (under ``data_dir``), a required judge cache (under ``experiment_dir``)
    or an entry in ``recorded_axes``; return quietly when all are usable."""
    pairs = list(pairs)
    bad: list[tuple[str, list[str]]] = []
    for entry in pairs:
        probs = axis_problems(
            entry, data_dir=data_dir, experiment_dir=experiment_dir,
            caches_all=caches_all, caches_any=caches_any,
            recorded_axes=recorded_axes, recorded_label=recorded_label)
        if probs:
            bad.append((axis_id(entry), probs))
    where = []
    if data_dir is not None:
        where.append(f"vectors under {data_dir}")
    if experiment_dir is not None and (caches_all or caches_any):
        where.append(f"judge caches under {experiment_dir}")
    fail_for_axes(bad, n_axes=len(pairs), pair_list=pair_list, script=script,
                  where="; ".join(where))


def fail_for_axes(
    bad: Sequence[Tuple[str, Sequence[str]]],
    *,
    n_axes: int,
    pair_list: Path | str,
    script: str,
    where: str = "",
) -> None:
    """Raise ``SystemExit`` listing ``bad`` (``(axis, [problem, ...])``);
    no-op when it is empty.  For a script whose loop finds an axis it
    cannot use (an empty score set from a loader) after loading: collect
    the axes, then call this once, rather than skipping them."""
    if not bad:
        return
    pair_list = Path(pair_list)
    lines = [
        f"{script}: {len(bad)} of {n_axes} axes in {pair_list.name} cannot be used"
        + (f" ({where})" if where else "") + ":",
    ]
    lines += [f"  {ax}: {'; '.join(probs)}" for ax, probs in bad]
    alt = v1_alternative(pair_list)
    hint = (
        "The current pair lists describe the corpus as it is now (for the next "
        "extraction); the 8slot data and its judge caches were made from the "
        "_v1 lists (CLAUDE.md, \"Pair lists come in two generations\").  "
    )
    if alt is not None:
        hint += f"Pass {alt.name} instead, or drop these axes from the list."
    else:
        hint += "Drop these axes from the list, or judge / extract them first."
    lines.append(hint)
    raise SystemExit("\n".join(lines))
