"""Cohort-token derivation from pair-list filenames + per-entry helpers.

The producer scripts in ``results_analysis/`` (whitening_k_sweep.py,
gpt_vs_sonnet_scatter.py, etc.) all consume a pair-list JSON via
``--pairs <file>``.  Historically the pair-list filename encoded a
count (``pair_list_12.json``, ``pair_list_33.json``).  Because the count
changes as more axes get judged, this gave us an accidental footgun:
output filenames like ``whitening_k_sweep_slot6.json`` collided across
cohorts (the same auto-derived path was used regardless of which pair
list fed the run).

Starting May 2026, pair lists use *definition-stable* names:

- ``pair_list_di.json`` -- "axes with desc+instr judging from both GPT
  and Sonnet" (formerly ``pair_list_33.json``).
- ``pair_list_responses.json`` -- "axes with desc+instr + GPT-responses
  judging" (formerly ``pair_list_12.json``).

Since 2026-09-28 a list that records judged work is copied to
``pair_list_<cohort>_v1.json`` before it is edited for a changed corpus
(a renamed pole, a pair that stopped being one), so the record and the
list for new work are separate files with separate cohort tokens
(``di`` and ``di_v1``).  See AGENT_NOTES "Current canonical local data
dir".

These names won't change as the cohorts grow.  Producer scripts pluck a
``cohort`` token from the pair-list filename and bake it into their
auto-derived output filenames so that, e.g.,
``whitening_k_sweep_di_slot6.json`` and
``whitening_k_sweep_responses_slot6.json`` are unambiguously different
files.

Per-entry schema
----------------

Each entry in a pair-list JSON is a dict with at minimum
``{"pos": str, "neg": str}``.  Optional fields:

* ``"pair_type"``: ``"traits"`` (default if omitted) or ``"roles"``.
  Tells consumers which subdirectory under ``data/`` and
  ``data_dir/`` holds the instruction file and activation vector for
  each pole, matching the axis_judge_correlation.py CLI argument
  ``--pair_type {traits,roles}``.  Pre-May-2026 pair lists predate
  this field; ``pair_type_of()`` defaults to ``"traits"`` so legacy
  files keep working.
* ``"tier"``, ``"moral"``, ``"note"``: advisory metadata used for
  reporting / filtering, no functional effect on judging.

Use :func:`pair_type_of` (rather than reading ``pair_type`` directly)
so callers get the default-on-missing behaviour for free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping


_PAIR_LIST_PREFIX = "pair_list_"

PAIR_TYPES = ("traits", "roles")
DEFAULT_PAIR_TYPE = "traits"


def cohort_from_pairs(pairs: str | Path) -> str:
    """Extract the cohort token from a pair-list filename.

    >>> cohort_from_pairs("pair_list_responses.json")
    'responses'
    >>> cohort_from_pairs("pair_list_di.json")
    'di'
    >>> cohort_from_pairs("/abs/path/pair_list_3_cohort_b.json")
    '3_cohort_b'
    >>> cohort_from_pairs("custom_pairs.json")
    'custom_pairs'

    Falls back to the file stem if the prefix is unrecognized so callers
    always get *some* token.
    """
    stem = Path(pairs).stem
    if stem.startswith(_PAIR_LIST_PREFIX):
        return stem[len(_PAIR_LIST_PREFIX):]
    return stem


def pair_type_of(entry: Mapping) -> str:
    """Return ``entry['pair_type']`` if present and valid, else
    :data:`DEFAULT_PAIR_TYPE` (``"traits"``).

    Centralises the legacy-format-tolerant default so consumers can
    treat the field as always-available.

    >>> pair_type_of({"pos": "helpful", "neg": "unhelpful"})
    'traits'
    >>> pair_type_of({"pos": "angel", "neg": "demon", "pair_type": "roles"})
    'roles'
    >>> pair_type_of({"pos": "x", "neg": "y", "pair_type": "TRAITS"})
    'traits'

    Unknown / case-folded values are normalised lower-case; truly
    invalid values (anything not in :data:`PAIR_TYPES`) raise
    ``ValueError`` so producer scripts fail fast on a typo rather
    than silently picking the wrong directory.
    """
    pt = entry.get("pair_type", DEFAULT_PAIR_TYPE)
    if not isinstance(pt, str):
        raise ValueError(
            f"pair_type must be a string; got {pt!r} on entry {entry!r}"
        )
    pt = pt.lower()
    if pt not in PAIR_TYPES:
        raise ValueError(
            f"pair_type must be one of {PAIR_TYPES}; got {pt!r} on "
            f"entry {entry!r}"
        )
    return pt
