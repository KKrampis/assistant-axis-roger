#!/usr/bin/env python3
"""Audit a vectors/ directory against its source activations/ and
scores/ directories, classifying why each missing vector is missing.

Motivation
----------
On RunPod / NFS we have observed three distinct failure modes in step 4
(``pipeline/4_vectors.py``):

    1. **Legitimate** misses -- the activation file exists but a
       vector cannot be computed:

         * the scores file is absent (judging never ran, or was
           interrupted before this entity); OR
         * the score=3 count is below ``--min_count``; OR
         * the activation tensor for the entity is empty / all-NaN.

    2. **Mysterious** misses -- everything upstream looks fine, the
       activation file is full-size and loadable, the scores file has
       enough score=3 entries, yet step 4 produced no vector.  These
       are almost always caused by transient NFS read failures that
       step 4 logged as warnings and skipped (the bug that motivated
       the project-standard 5-attempt retry rolled out alongside this
       script).  Re-running step 4 with ``--overwrite`` on just these
       entities is enough to recover.

    3. **Corrupt-on-write vectors** -- the vector file exists at the
       expected size, but ``torch.load`` raises ``RuntimeError`` /
       ``EOFError`` ("storage has wrong byte size of dtype" /
       "PytorchStreamReader failed reading zip archive: ... unexpected
       EOF, expected N more bytes") on read.  This is the failure
       mode the post-write size-sanity check in
       ``torch_save_with_retry`` was added to catch on the writer
       side; the scanner now actively loads each vector to detect any
       that slipped through before the check was in place.

The script also flags **truncated activation files** (anything whose
on-disk size is dramatically smaller than the median size of its
peers) as a separate category, since those are the upstream bug that
caused the original ``r_guardian__casual.pt`` corruption.

**Zero-byte activation files** are classified as ``placeholder``
rather than corruption.  The pipeline's step-2 output is always
~2.6 GB per role (n_convs × n_slots × n_layers × hidden × bf16);
no real activation .pt is ever 0 bytes.  Users sometimes ``touch``
0-byte placeholder files to tell step 2 "skip this role" via its
``output_file.exists()`` check (typically when reusing pre-existing
scores/vectors and not wanting to re-extract activations to save
disk space).  Placeholders are NOT in the re-run candidates list --
the user intentionally doesn't want them regenerated.

Usage
-----

**Comprehensive mode** (recommended).  Point the scanner at an
output root and it auto-discovers every entity-type subdir
(``default/``, ``roles/``, ``traits/``, ``combinations/``, ...) with
an ``activations/`` subdir, and scans every sibling ``vectors*``
variant (``vectors/``, ``vectors_unfiltered/``, etc.) that contains
real (non-symlinked) ``.pt`` files::

    uv run pipeline/scan_missing_vectors.py \\
        --root outputs/qwen-3-32b/ \\
        --min_count 50

**Single-pair mode** (back-compat with earlier invocations)::

    uv run pipeline/scan_missing_vectors.py \\
        --activations_dir outputs/qwen-3-32b/roles/activations \\
        --vectors_dir     outputs/qwen-3-32b/roles/vectors \\
        --scores_dir      outputs/qwen-3-32b/roles/scores \\
        --min_count       50

For ``vectors_unfiltered/`` (the no-incongruity-filter variant of the
filtered pipeline), pair the directory with a sibling
``scores_unfiltered/`` containing the unfiltered scores.  Comprehensive
mode auto-pairs ``vectors*_unfiltered`` with ``scores_unfiltered/`` when
both exist.  Score=3 + ``--min_count`` filtering still applies to
non-default roles in unfiltered mode -- the only thing "unfiltered"
turns off is the per-pair incongruity exclusion baked into the filtered
``scores/`` directory.  Default-entity-type subdirs (``default/``)
remain mean-of-all-activations because their role name contains
``default``.

If you want the legacy "mean-of-all-activations regardless of role
name" behaviour for a custom vectors variant, pass
``--mode unfiltered_mean`` in single-pair mode.

Outputs a per-entity JSON report next to each scanned vectors dir
(``missing_vectors_audit.json``) plus a re-run list
(``missing_vectors_rerun.txt``) for the suspects that need step 4
re-runs.  Prints a summary table to stdout.  Entities classified as
``mysterious``, ``corrupt_or_truncated_activation``, ``corrupt_vector``,
or ``ok_zero_size`` are the ones worth re-running.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import torch
from tqdm import tqdm

# Errors we treat as "this file is unreadable" when classifying.
# torch.load on a corrupt-on-write or short-read file can raise any
# of these; we don't retry past atomic_io's own retry loop here, just
# label the file as corrupt.  pickle.UnpicklingError covers
# "_pickle.UnpicklingError: pickle data was truncated" / "invalid
# load key", which is a parse error (per Roger's "don't retry parse
# errors" rule) but still indicates a genuinely broken file we should
# flag for re-run.
LOAD_FAILURE_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError, OSError, EOFError, pickle.UnpicklingError, ValueError,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.atomic_io import (  # noqa: E402
    atomic_write_text,
    read_text_with_retry,
    torch_load_with_retry,
)

logger = logging.getLogger("pipeline.scan_missing_vectors")


# ---------------------------------------------------------------------------
# Resumable load-result cache
# ---------------------------------------------------------------------------

CACHE_SCHEMA_VERSION = 1
# Flush the cache to disk every N updates OR every M seconds,
# whichever comes first, plus a final flush in close() / on Ctrl-C.
# The right cadence depends on cost asymmetry:
#   - one flush = ~0.5-1 s (atomic-write a ~5-20 MB JSON to NFS)
#   - one lost entry on a drop = ~5-15 s of NFS re-read for a 2.6 GB
#     activation file
# So flushing aggressively is cheap insurance.  N=25 is ~25 lost
# entries worst case (~2-5 min of recovery work); the M=30 s floor
# bounds loss when entries arrive slowly (e.g. each torch.load
# takes 10 s, so 25 entries = 250 s without the time guard).
CACHE_FLUSH_EVERY = 25
CACHE_FLUSH_INTERVAL_S = 30.0


class ScanCache:
    """Persistent cache of ``torch.load`` outcomes keyed by file path.

    Two sub-caches:

    * ``vector_loads[path] = {size, mtime, ok, error}``
      For the per-vector load check.  ``ok=True`` means
      ``torch.load`` succeeded; ``ok=False`` means it raised, with
      the ``type(e).__name__: msg`` stored in ``error``.

    * ``activation_loads[path] = {size, mtime, ok, error, keys, any_finite}``
      For the deep-load activation check.  When ``ok=True`` we cache
      the full key list (so the cheap filter+match step can run on
      cache hit without reloading the tensor) plus a single
      ``any_finite`` boolean covering all activations in the file
      (a conservative shortcut: if every tensor is fully NaN we
      know to classify as ``all_nan_or_empty`` without re-checking
      per-key).

    A cache hit requires ``(size, mtime)`` to match the current
    on-disk file; any mismatch (re-extraction, in-place edit) is
    treated as a miss and the entry is overwritten on the next put.

    Persistence cadence: flush whenever :data:`CACHE_FLUSH_EVERY`
    updates have accumulated OR :data:`CACHE_FLUSH_INTERVAL_S`
    seconds have passed since the last flush, whichever comes first.
    Plus a final flush in :meth:`close` (idempotent, safe to call
    from an ``atexit`` hook or ``finally`` block).  Uses
    :func:`atomic_write_text` so a crash mid-flush never leaves a
    half-written cache file at the destination.
    """

    def __init__(self, path: Optional[Path]):
        self.path = path
        self.data: dict[str, Any] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "vector_loads": {},
            "activation_loads": {},
        }
        self._dirty_count = 0
        self._last_flush_time = time.monotonic()
        self._stats = {"v_hits": 0, "v_miss": 0, "a_hits": 0, "a_miss": 0}
        if path is not None and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if loaded.get("schema_version") == CACHE_SCHEMA_VERSION:
                    self.data = loaded
                    logger.info(
                        f"[cache] loaded {len(self.data['vector_loads'])} vector "
                        f"+ {len(self.data['activation_loads'])} activation entries "
                        f"from {path}"
                    )
                else:
                    logger.warning(
                        f"[cache] schema mismatch in {path} "
                        f"({loaded.get('schema_version')!r} != {CACHE_SCHEMA_VERSION}); "
                        f"discarding"
                    )
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"[cache] could not load {path}: {e}; starting empty")

    @staticmethod
    def _key(p: Path) -> str:
        # resolve() turns relative + symlinked paths into a stable
        # absolute key so the same file is cached once regardless of
        # how the caller spells its path.
        try:
            return str(p.resolve(strict=False))
        except OSError:
            return str(p)

    def _matches_disk(self, entry: dict, p: Path) -> bool:
        """True if a cached entry's (size, mtime) match the on-disk file."""
        try:
            st = p.stat()
        except OSError:
            return False
        return (entry.get("size") == st.st_size
                and entry.get("mtime") == st.st_mtime)

    # -- vector loads -------------------------------------------------------

    def get_vector(self, p: Path) -> Optional[dict]:
        entry = self.data["vector_loads"].get(self._key(p))
        if entry is not None and self._matches_disk(entry, p):
            self._stats["v_hits"] += 1
            return entry
        self._stats["v_miss"] += 1
        return None

    def put_vector(self, p: Path, *, ok: bool, error: Optional[str] = None) -> None:
        try:
            st = p.stat()
        except OSError:
            return  # nothing to cache for a vanished file
        self.data["vector_loads"][self._key(p)] = {
            "size": st.st_size,
            "mtime": st.st_mtime,
            "ok": ok,
            "error": error,
        }
        self._mark_dirty()

    # -- activation loads ---------------------------------------------------

    def get_activation(self, p: Path) -> Optional[dict]:
        entry = self.data["activation_loads"].get(self._key(p))
        if entry is not None and self._matches_disk(entry, p):
            self._stats["a_hits"] += 1
            return entry
        self._stats["a_miss"] += 1
        return None

    def put_activation(
        self,
        p: Path,
        *,
        ok: bool,
        error: Optional[str] = None,
        keys: Optional[list[str]] = None,
        any_finite: Optional[bool] = None,
    ) -> None:
        try:
            st = p.stat()
        except OSError:
            return
        self.data["activation_loads"][self._key(p)] = {
            "size": st.st_size,
            "mtime": st.st_mtime,
            "ok": ok,
            "error": error,
            "keys": keys,
            "any_finite": any_finite,
        }
        self._mark_dirty()

    # -- persistence --------------------------------------------------------

    def _mark_dirty(self) -> None:
        self._dirty_count += 1
        if (self._dirty_count >= CACHE_FLUSH_EVERY
                or time.monotonic() - self._last_flush_time
                >= CACHE_FLUSH_INTERVAL_S):
            self.flush()

    def flush(self) -> None:
        if self.path is None or self._dirty_count == 0:
            return
        try:
            atomic_write_text(
                json.dumps(self.data, indent=None),
                self.path,
                logger_obj=logger,
            )
            self._dirty_count = 0
            self._last_flush_time = time.monotonic()
        except (OSError, RuntimeError) as e:
            # Don't reset _dirty_count -- the next mark_dirty will
            # try again.  Logging at WARNING (not ERROR) because the
            # cache is best-effort: failing to persist just means
            # next run has more work to do, not a correctness issue.
            logger.warning(f"[cache] flush to {self.path} failed: {e}")

    def close(self) -> None:
        """Final flush + log of hit/miss summary.  Idempotent."""
        self.flush()
        s = self._stats
        if any(s.values()):
            logger.info(
                f"[cache] vector_loads: {s['v_hits']} hits / {s['v_miss']} misses; "
                f"activation_loads: {s['a_hits']} hits / {s['a_miss']} misses"
            )


# Activation files smaller than this fraction of the median peer-group
# file size are flagged as "suspiciously small / likely truncated".
TRUNCATION_RATIO = 0.5
# Vector files smaller than this many bytes are treated as suspect.
# A healthy bf16 (n_slots × n_layers × hidden) vector is in the low MB;
# even a tiny smoke-test tensor pickled by torch.save is ~700 bytes.
# Anything under 256 bytes is almost certainly a stub / aborted write.
MIN_HEALTHY_VECTOR_BYTES = 256

# Statuses that should be re-run by step 4.  Kept in one place so the
# stdout summary, the rerun list and any callers stay in sync.
RERUN_STATUSES = (
    "mysterious",
    "corrupt_or_truncated_activation",
    "corrupt_vector",
    "ok_zero_size",
)

# Status display order for the summary table.
ALL_STATUSES = (
    "ok",
    "placeholder",
    "ok_zero_size",
    "missing_activation",
    "corrupt_or_truncated_activation",
    "corrupt_vector",
    "missing_scores",
    "below_min_count",
    "all_nan_or_empty",
    "mysterious",
    "orphan_vector",
)


# ---------------------------------------------------------------------------
# Per-entity classification
# ---------------------------------------------------------------------------

def _question_index(key: str) -> int:
    return int(key.rsplit("_q", 1)[1])


def _keep_by_reduce(key: str, reduce_questions: int) -> bool:
    if reduce_questions <= 1:
        return True
    return _question_index(key) % reduce_questions == 0


def classify_entity(
    role: str,
    act_file: Path,
    median_act_size: int,
    vectors_dir: Path,
    scores_dir: Path | None,
    min_count: int,
    reduce_questions: int,
    mode: str,
    deep_load: bool,
    load_check: bool,
    cache: Optional[ScanCache] = None,
) -> dict[str, Any]:
    """Classify a single entity.

    See module docstring for the status taxonomy.  Returns a dict
    with at minimum ``role`` and ``status`` plus category-specific
    extras (sizes, counts, reasons).

    If ``cache`` is given, the expensive per-vector and per-activation
    ``torch.load`` calls consult / populate it (keyed by absolute path,
    validated by file size + mtime).  This makes a re-run after a
    connection drop near-instant for files the previous run already
    confirmed loadable.
    """
    out: dict[str, Any] = {"role": role}

    vec_file = vectors_dir / f"{role}.pt"
    has_vector = vec_file.exists()
    out["vector_path"] = str(vec_file)
    out["activation_path"] = str(act_file)
    out["activation_size_bytes"] = act_file.stat().st_size if act_file.exists() else 0

    # 1. Activation upstream check.
    if not act_file.exists():
        out["status"] = "missing_activation"
        return out

    # 1a. Zero-byte activation: a deliberate placeholder, not corruption.
    # The pipeline's step-2 output is always ~2.6 GB per role (n_convs ×
    # n_slots × n_layers × hidden × bf16); no real activation .pt is ever
    # 0 bytes.  Users (notably the placeholder-script pattern documented
    # in pipeline/README.md) sometimes ``touch`` 0-byte placeholders to
    # tell step 2 "skip this role" via the output_file.exists() check.
    # Those should NOT show up as re-run candidates -- the user
    # intentionally doesn't want them regenerated.  Classified
    # separately so the rerun list is clean and the summary distinguishes
    # placeholders from genuine corruption.
    if out["activation_size_bytes"] == 0:
        out["status"] = "placeholder"
        return out

    if median_act_size > 0 and out["activation_size_bytes"] < median_act_size * TRUNCATION_RATIO:
        out["status"] = "corrupt_or_truncated_activation"
        out["median_peer_bytes"] = median_act_size
        out["ratio_to_median"] = out["activation_size_bytes"] / median_act_size
        return out

    # 2. Vector exists?
    if has_vector:
        size = vec_file.stat().st_size
        out["vector_size_bytes"] = size
        if size < MIN_HEALTHY_VECTOR_BYTES:
            out["status"] = "ok_zero_size"
            return out

        # 2b. Vector loadability check.  This catches the
        # "corrupt-on-write" failure mode where the file is at the
        # expected size but torch.load raises RuntimeError / EOFError
        # because pickle / safetensors content is malformed.  Use a
        # short retry schedule -- a corrupt file won't recover from
        # waiting, but one quick retry gives NFS a chance to clear a
        # one-off hiccup.
        if load_check:
            cached = cache.get_vector(vec_file) if cache is not None else None
            if cached is not None:
                if not cached["ok"]:
                    # Replay the failure as a live warning so cached
                    # known-bad files stay visible on every re-run --
                    # silent suppression of a corrupt file is exactly
                    # the kind of thing the scanner is supposed to
                    # surface.  WARNING (not ERROR) because we've
                    # already errored on this file once when first
                    # caching it.
                    logger.warning(
                        f"[cached-corrupt vector] {vec_file}: "
                        f"{cached.get('error')}"
                    )
                    out["status"] = "corrupt_vector"
                    out["note"] = f"vector .pt unreadable (cached): {cached.get('error')}"
                    return out
                # cache hit, ok=True -> skip the load
            else:
                try:
                    torch_load_with_retry(
                        vec_file, map_location="cpu", weights_only=False,
                        delays_s=[5.0], logger_obj=logger,
                    )
                    if cache is not None:
                        cache.put_vector(vec_file, ok=True)
                except LOAD_FAILURE_ERRORS as e:
                    err_msg = f"{type(e).__name__}: {e}"
                    if cache is not None:
                        cache.put_vector(vec_file, ok=False, error=err_msg)
                    out["status"] = "corrupt_vector"
                    out["note"] = f"vector .pt unreadable: {err_msg}"
                    return out

        out["status"] = "ok"
        return out

    # 3. Vector missing -- figure out why.
    #
    # ``is_default`` here means "vector is supposed to be a mean of all
    # activations (no scores filter)" -- the path step 4 takes only when
    # the role name contains ``default``.  Earlier versions of this
    # scanner also treated ``mode == "unfiltered"`` as default-like, but
    # that misclassified non-default roles in ``vectors_unfiltered/`` as
    # ``mysterious`` when they were legitimately ``below_min_count``:
    # ``vectors_unfiltered/`` only turns off incongruity filtering, it
    # does NOT turn off the score=3 + min_count filter for non-default
    # roles.  See the module docstring for the convention.
    #
    # Back-compat: ``--mode unfiltered`` *without* a scores_dir is the
    # legacy mean-of-all invocation; we still honour it (equivalent to
    # the new explicit ``--mode unfiltered_mean``).
    is_default = "default" in role
    if mode == "unfiltered_mean":
        is_default = True
    elif mode == "unfiltered" and scores_dir is None:
        is_default = True

    if not is_default:
        if scores_dir is None:
            out["status"] = "mysterious"
            out["note"] = "filtered mode requested but no scores_dir provided"
            return out
        scores_file = scores_dir / f"{role}.json"
        if not scores_file.exists():
            out["status"] = "missing_scores"
            return out

        try:
            scores = json.loads(read_text_with_retry(scores_file, logger_obj=logger))
        except (OSError, RuntimeError, EOFError, json.JSONDecodeError) as e:
            out["status"] = "mysterious"
            out["note"] = f"scores file unreadable: {e}"
            return out

        # Cheap path: count score=3 entries from the scores dict alone,
        # without loading the activation tensor.  Step 4 also requires
        # the matching activation key to be present, so this is an
        # upper bound on the realised score=3 count.  If it's already
        # below min_count, we're done.
        score3_in_scores = sum(
            1 for k, v in scores.items()
            if v == 3 and _keep_by_reduce(k, reduce_questions)
        )
        out["score3_in_scores"] = score3_in_scores
        if score3_in_scores < min_count:
            out["status"] = "below_min_count"
            return out

    if not deep_load:
        # Skip the expensive activation-load step: any other case is
        # presumed mysterious until a deep load disagrees.
        out["status"] = "mysterious"
        out["note"] = "shallow scan (--deep_load disabled)"
        return out

    # 4. Deep check: load the activation tensor and recompute the
    # filter to confirm whether step 4 *should* have produced a
    # vector.  Use a fast retry schedule (one short retry) rather
    # than the project-standard [5,20,60,180]s -- a corrupt file
    # won't recover, and waiting 4+ minutes per corrupt file makes
    # auditing thousands of files painful.  One 5 s retry still
    # gives NFS a chance to clear a one-off hiccup.
    cached_act = cache.get_activation(act_file) if cache is not None else None

    if cached_act is not None and not cached_act["ok"]:
        # Replay the failure as a live warning -- see the matching
        # block above on the vector path for rationale.  Activation
        # corruption is the expensive-to-recover case (re-extract
        # from step 2), so users want to see this on every scan.
        logger.warning(
            f"[cached-corrupt activation] {act_file}: "
            f"{cached_act.get('error')}"
        )
        out["status"] = "corrupt_or_truncated_activation"
        out["note"] = (f"activation .pt unloadable (cached): "
                       f"{cached_act.get('error')}")
        return out

    if cached_act is not None and cached_act["ok"]:
        # Use cached metadata (key list + any_finite-overall flag) to
        # classify without reloading the tensor.  The keys list is
        # the file's full activation set; we re-derive the filtered
        # subset cheaply from the current scores (which may differ
        # from when the cache was populated).
        keys = cached_act.get("keys") or []
        any_finite_overall = cached_act.get("any_finite", True)
        if not keys:
            out["status"] = "all_nan_or_empty"
            out["note"] = "activation file contains no tensors (cached)"
            return out
        # Conservative shortcut: if the entire file was NaN at cache
        # time, every subset is too.  (Per-key finite info isn't
        # cached -- if the user wants stronger guarantees they can
        # blow away the cache.)
        if not any_finite_overall:
            out["status"] = "all_nan_or_empty"
            out["note"] = "activation file is all-NaN (cached)"
            return out
        if is_default:
            kept = [k for k in keys if _keep_by_reduce(k, reduce_questions)]
            out["activation_count"] = len(kept)
            if not kept:
                out["status"] = "all_nan_or_empty"
                return out
            out["status"] = "mysterious"
            return out
        # Filtered mode -- recount matched against current scores.
        matched_keys = [
            k for k in keys
            if k in scores and scores[k] == 3 and _keep_by_reduce(k, reduce_questions)
        ]
        out["score3_matched"] = len(matched_keys)
        if len(matched_keys) < min_count:
            out["status"] = "below_min_count"
            out["note"] = (f"score3 in scores file = {out['score3_in_scores']}, "
                           f"actually realised in activations = {len(matched_keys)} "
                           f"(cached)")
            return out
        out["status"] = "mysterious"
        return out

    # Cache miss (or no cache) -- do the actual load.
    try:
        data = torch_load_with_retry(
            act_file, map_location="cpu", weights_only=False,
            delays_s=[5.0],
            logger_obj=logger,
        )
    except LOAD_FAILURE_ERRORS as e:
        err_msg = f"{type(e).__name__}: {e}"
        if cache is not None:
            cache.put_activation(act_file, ok=False, error=err_msg)
        out["status"] = "corrupt_or_truncated_activation"
        out["note"] = f"activation .pt unloadable: {err_msg}"
        return out

    data.pop("metadata", None)
    if not data:
        if cache is not None:
            cache.put_activation(act_file, ok=True, keys=[], any_finite=False)
        out["status"] = "all_nan_or_empty"
        out["note"] = "activation file contains no tensors"
        return out

    # Cache the keys + an "any tensor in the whole file is finite" flag.
    # Computing the per-tensor finite check now is cheap-ish since the
    # tensors are already in memory; doing it lets us short-circuit
    # the all_nan_or_empty case on cache hit without reloading.
    all_keys = sorted(data.keys())
    any_finite_overall = any(torch.isfinite(a).any().item() for a in data.values())

    if is_default:
        all_acts = [act for k, act in data.items() if _keep_by_reduce(k, reduce_questions)]
        out["activation_count"] = len(all_acts)
        if not all_acts:
            if cache is not None:
                cache.put_activation(act_file, ok=True, keys=all_keys,
                                     any_finite=any_finite_overall)
            out["status"] = "all_nan_or_empty"
            return out
        any_finite = any(torch.isfinite(a).any().item() for a in all_acts)
        if cache is not None:
            cache.put_activation(act_file, ok=True, keys=all_keys,
                                 any_finite=any_finite_overall)
        if not any_finite:
            out["status"] = "all_nan_or_empty"
            return out
        out["status"] = "mysterious"
        return out

    # Filtered mode.
    matched_pairs = [
        (k, act) for k, act in data.items()
        if k in scores and scores[k] == 3 and _keep_by_reduce(k, reduce_questions)
    ]
    out["score3_matched"] = len(matched_pairs)
    if cache is not None:
        cache.put_activation(act_file, ok=True, keys=all_keys,
                             any_finite=any_finite_overall)
    if len(matched_pairs) < min_count:
        out["status"] = "below_min_count"
        out["note"] = (f"score3 in scores file = {out['score3_in_scores']}, "
                       f"actually realised in activations = {len(matched_pairs)}")
        return out

    any_finite = any(torch.isfinite(a).any().item() for _, a in matched_pairs)
    if not any_finite:
        out["status"] = "all_nan_or_empty"
        return out

    out["status"] = "mysterious"
    return out


# ---------------------------------------------------------------------------
# Comprehensive scan target discovery
# ---------------------------------------------------------------------------

@dataclass
class ScanTarget:
    """One (activations, vectors, optional scores) triple to scan."""
    entity_type: str          # "roles" / "traits" / "combinations" / "default" / ...
    vectors_variant: str      # "vectors" / "vectors_unfiltered" / "vectors_pca" / ...
    activations_dir: Path
    vectors_dir: Path
    scores_dir: Optional[Path] = None
    mode: str = "filtered"

    @property
    def label(self) -> str:
        return f"{self.entity_type}/{self.vectors_variant}"


def _is_skippable_vectors_dir(vec_dir: Path) -> tuple[bool, str]:
    """True if the directory has no real vector content.

    Skippable means:
      - directory empty, OR
      - all .pt files are symlinks (typically the default.pt symlink
        pointing at ../../default/vectors/default.pt that step 4 sets
        up as a baseline reference -- not a vector this dir actually
        owns).

    Returns (skippable, reason) for logging.
    """
    pt_files = list(vec_dir.glob("*.pt"))
    if not pt_files:
        return True, "empty (no .pt files)"
    real_files = [f for f in pt_files if not f.is_symlink()]
    if not real_files:
        return True, f"only {len(pt_files)} symlinks (default-only)"
    return False, ""


def discover_scan_targets(
    root: Path,
    *,
    skip_suffixes: tuple[str, ...] = (),
    matched_suffixes_out: Optional[set[str]] = None,
) -> list[ScanTarget]:
    """Walk `root` for entity-type subdirs and yield one ScanTarget per
    non-empty ``vectors*`` variant.

    An entity-type subdir is any directory at the root level that
    contains an ``activations/`` subdir.  Each entity-type subdir may
    contain zero or more ``vectors*`` subdirs (``vectors``,
    ``vectors_unfiltered``, ...); we emit one ScanTarget per non-skippable
    one.  ``scores/`` is included if present (filtered mode); otherwise
    the target is run in unfiltered mode.

    ``skip_suffixes``: if a vectors-variant directory name ends with
    any of these strings, it's silently skipped.  Use to exclude
    auxiliary vector layouts you don't want audited (e.g. an
    in-progress experimental ``vectors_4slots/`` sibling).  Matching
    is exact-suffix on the directory's *basename*, not a glob.

    Default entity-type subdirs (``default/``) typically contain a
    single ``default.pt`` and no scores; they're handled with mode=
    unfiltered automatically.
    """
    targets: list[ScanTarget] = []
    if not root.is_dir():
        raise ValueError(f"--root path is not a directory: {root}")

    for entity_dir in sorted(root.iterdir()):
        if not entity_dir.is_dir():
            continue
        act_dir = entity_dir / "activations"
        if not act_dir.is_dir():
            continue
        filtered_scores = entity_dir / "scores"
        if not filtered_scores.is_dir():
            filtered_scores = None
        unfiltered_scores = entity_dir / "scores_unfiltered"
        if not unfiltered_scores.is_dir():
            unfiltered_scores = None

        for vec_dir in sorted(entity_dir.iterdir()):
            if not vec_dir.is_dir():
                continue
            if not vec_dir.name.startswith("vectors"):
                continue
            matched_suffix = next(
                (s for s in skip_suffixes if vec_dir.name.endswith(s)),
                None,
            )
            if matched_suffix is not None:
                logger.info(
                    f"[skip] {vec_dir} (matches --skip_suffix {matched_suffix!r})"
                )
                if matched_suffixes_out is not None:
                    matched_suffixes_out.add(matched_suffix)
                continue
            skippable, reason = _is_skippable_vectors_dir(vec_dir)
            if skippable:
                logger.info(f"[skip] {vec_dir} ({reason})")
                continue
            # Pair vectors variants with their scores convention:
            #   vectors_unfiltered/  <->  scores_unfiltered/  (mode=unfiltered)
            #   vectors/             <->  scores/             (mode=filtered)
            #   vectors_*_unfiltered <->  scores_unfiltered/  (mode=unfiltered)
            #   vectors_*            <->  scores/             (mode=filtered)
            # When the matching scores dir is absent, fall back to the
            # other one if available, otherwise leave None (the scanner
            # will then return missing_scores for non-default roles).
            if vec_dir.name.endswith("_unfiltered"):
                mode = "unfiltered"
                target_scores = unfiltered_scores or filtered_scores
            else:
                mode = "filtered" if filtered_scores is not None else "unfiltered"
                target_scores = filtered_scores or unfiltered_scores
            targets.append(ScanTarget(
                entity_type=entity_dir.name,
                vectors_variant=vec_dir.name,
                activations_dir=act_dir,
                vectors_dir=vec_dir,
                scores_dir=target_scores,
                mode=mode,
            ))

    return targets


# ---------------------------------------------------------------------------
# One-target scan
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    target: ScanTarget
    median_activation_size: int
    results: list[dict[str, Any]] = field(default_factory=list)
    output_path: Optional[Path] = None
    rerun_list_path: Optional[Path] = None

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out

    def suspects(self) -> list[dict[str, Any]]:
        return [r for r in self.results if r["status"] in RERUN_STATUSES]


def scan_one_target(
    target: ScanTarget,
    *,
    min_count: int,
    reduce_questions: int,
    deep_load: bool,
    load_check: bool,
    cache: Optional[ScanCache] = None,
) -> ScanResult:
    """Walk one ScanTarget's activations + vectors, classifying every entity."""
    act_files = sorted(target.activations_dir.glob("*.pt"))
    if not act_files:
        logger.warning(f"[{target.label}] no activation .pt files in "
                       f"{target.activations_dir}; skipping")
        return ScanResult(target=target, median_activation_size=0)

    sizes = [f.stat().st_size for f in act_files]
    median_size = int(statistics.median(sizes))

    print(f"\n[{target.label}] scanning {len(act_files)} activation files "
          f"(median {median_size / 1e6:.1f} MB) "
          f"in {target.vectors_dir}")
    if target.scores_dir is None:
        print(f"  mode: {target.mode} (no scores filter)")
    else:
        print(f"  mode: {target.mode}, scores from {target.scores_dir}")

    results: list[dict[str, Any]] = []
    desc = f"audit {target.label}"
    for act_file in tqdm(act_files, desc=desc, leave=False):
        role = act_file.stem
        results.append(classify_entity(
            role=role,
            act_file=act_file,
            median_act_size=median_size,
            vectors_dir=target.vectors_dir,
            scores_dir=target.scores_dir,
            min_count=min_count,
            reduce_questions=reduce_questions,
            mode=target.mode,
            deep_load=deep_load,
            load_check=load_check,
            cache=cache,
        ))

    # Also flag any vector files that exist with no source activation
    # (genuine orphans -- shouldn't happen, but cheap to check).
    act_role_set = {f.stem for f in act_files}
    for vec_file in sorted(target.vectors_dir.glob("*.pt")):
        if vec_file.stem == "missing_vectors_audit":
            continue
        # Skip symlinks (the default.pt symlink) -- they're shared
        # references, not genuine orphans.
        if vec_file.is_symlink():
            continue
        if vec_file.stem not in act_role_set:
            results.append({
                "role": vec_file.stem,
                "status": "orphan_vector",
                "vector_path": str(vec_file),
                "activation_path": None,
                "vector_size_bytes": vec_file.stat().st_size,
            })

    return ScanResult(target=target, median_activation_size=median_size,
                      results=results)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_scan_output(
    sr: ScanResult,
    *,
    min_count: int,
    reduce_questions: int,
    deep_load: bool,
    load_check: bool,
) -> None:
    """Write per-scan JSON report and rerun list next to the vectors dir."""
    output_path = sr.target.vectors_dir / "missing_vectors_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "activations_dir": str(sr.target.activations_dir),
        "vectors_dir": str(sr.target.vectors_dir),
        "scores_dir": str(sr.target.scores_dir) if sr.target.scores_dir else None,
        "entity_type": sr.target.entity_type,
        "vectors_variant": sr.target.vectors_variant,
        "mode": sr.target.mode,
        "min_count": min_count,
        "reduce_questions": reduce_questions,
        "deep_load": deep_load,
        "load_check": load_check,
        "median_activation_size_bytes": sr.median_activation_size,
        "results": sr.results,
    }, indent=2))
    sr.output_path = output_path

    rerun_path = sr.target.vectors_dir / "missing_vectors_rerun.txt"
    suspects = sr.suspects()
    rerun_path.write_text(
        "\n".join(r["role"] for r in suspects) + ("\n" if suspects else "")
    )
    sr.rerun_list_path = rerun_path


def _print_scan_summary(sr: ScanResult) -> None:
    """Print a per-scan summary table + suspect list to stdout."""
    counts = sr.counts()
    print(f"\n  === {sr.target.label} summary ===")
    for status in ALL_STATUSES:
        if status in counts:
            print(f"    {status:34s}  {counts[status]:5d}")
    print(f"    {'TOTAL':34s}  {len(sr.results):5d}")
    if sr.output_path:
        print(f"    report: {sr.output_path}")
    if sr.rerun_list_path:
        print(f"    rerun:  {sr.rerun_list_path}")

    suspects = sr.suspects()
    if suspects:
        print(f"  {len(suspects)} suspect(s):")
        for r in suspects[:30]:
            extra = r.get("note") or r.get("score3_in_scores") or ""
            print(f"    [{r['status']:34s}] {r['role']}  {extra}")
        if len(suspects) > 30:
            print(f"    ... and {len(suspects) - 30} more "
                  f"(see {sr.output_path})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Audit vectors/* dirs vs activations/+ scores/, "
                    "classifying every missing or corrupt vector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for the full status taxonomy.",
    )
    # Two ways to specify what to scan.  Comprehensive (--root) auto-
    # discovers entity-type subdirs and every sibling vectors* variant;
    # explicit single-pair (--activations_dir + --vectors_dir) is the
    # back-compat invocation.
    parser.add_argument("--root", type=Path, default=None,
                        help="Output root containing entity-type subdirs "
                             "(default/, roles/, traits/, combinations/, ...). "
                             "Auto-discovers every {entity}/vectors* sibling "
                             "with real (non-symlinked) .pt files.")
    parser.add_argument("--skip_suffix", action="append", default=[],
                        metavar="SUFFIX",
                        help="In --root mode: skip vectors-variant dirs whose "
                             "basename ends with this suffix.  Repeatable.  "
                             "Example: --skip_suffix _4slots skips every "
                             "vectors_4slots/ subdir found.")
    parser.add_argument("--cache", type=Path, default=None,
                        metavar="PATH",
                        help="Override the default cache file location.  "
                             "By default (caching is ON), the cache lands at "
                             "<root>/scan_cache.json in --root mode or at "
                             "<vectors_dir>/scan_cache.json in single-pair "
                             "mode -- alongside the audit JSON / log file.  "
                             "Cache is keyed by absolute path + (size, mtime), "
                             "so re-runs after a connection drop near-"
                             "instantly skip files the previous run already "
                             "confirmed loadable.  Auto-flushes every "
                             f"{CACHE_FLUSH_EVERY} updates or "
                             f"{CACHE_FLUSH_INTERVAL_S:.0f} s "
                             "(whichever first) plus on exit.")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable caching entirely.  Use when you want "
                             "to force a full re-verification of every file "
                             "(e.g. after suspecting in-place corruption "
                             "that didn't bump mtime, or to confirm cache-"
                             "skipped classifications still hold).")
    parser.add_argument("--activations_dir", type=Path, default=None)
    parser.add_argument("--vectors_dir", type=Path, default=None)
    parser.add_argument("--scores_dir", type=Path, default=None,
                        help="Required for --mode filtered or unfiltered "
                             "(used to count score=3 entries for non-default "
                             "roles in either mode).  Drop only for "
                             "--mode unfiltered_mean.")
    parser.add_argument("--mode",
                        choices=("filtered", "unfiltered", "unfiltered_mean"),
                        default="filtered",
                        help="filtered = step 4's default behaviour "
                             "(score=3 mean filtered against scores/ for "
                             "non-default roles; mean-of-all for default).  "
                             "unfiltered = same logic, but expecting the "
                             "unfiltered scores dir (scores_unfiltered/) "
                             "which omits the per-pair incongruity exclusion.  "
                             "unfiltered_mean = legacy: treat every entity "
                             "as 'default' (mean of all acts, no scores). "
                             "In --root mode, mode is auto-detected per "
                             "vectors dir (unfiltered if name ends with "
                             "_unfiltered, filtered otherwise).")
    parser.add_argument("--min_count", type=int, default=50,
                        help="Minimum score=3 sample count required by step 4.")
    parser.add_argument("--reduce_questions", type=int, default=1,
                        help="Match the value passed to step 4.")
    parser.add_argument("--deep_load", action="store_true",
                        help="For *missing* vectors, load every suspect "
                             "activation .pt to confirm the score=3 / "
                             "NaN classification.  Slow but definitive.  "
                             "Without this, anything that passes the cheap "
                             "shallow checks is reported as 'mysterious'.")
    parser.add_argument("--no_load_check", action="store_true",
                        help="Skip the per-vector torch.load check.  By "
                             "default every existing vector .pt is loaded "
                             "(with a short 1-retry schedule) to detect the "
                             "corrupt-on-write failure mode (full size, "
                             "torch.load raises).  Disabling makes the scan "
                             "much faster but won't catch corrupt vectors.")
    parser.add_argument("--rerun_list", type=Path, default=None,
                        help="Single-pair mode only: write the rerun list "
                             "here.  In --root mode, rerun lists are written "
                             "next to each vectors dir as missing_vectors_rerun.txt.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Single-pair mode only: write the JSON report "
                             "here.  In --root mode, JSON reports are written "
                             "next to each vectors dir as missing_vectors_audit.json.")
    args = parser.parse_args()

    # Validate input combinations.
    using_root = args.root is not None
    using_explicit = args.activations_dir is not None or args.vectors_dir is not None
    if using_root and using_explicit:
        parser.error("--root and --activations_dir/--vectors_dir are mutually exclusive")
    if not using_root and not using_explicit:
        parser.error("must specify either --root or both --activations_dir and --vectors_dir")
    if not using_root:
        if args.activations_dir is None or args.vectors_dir is None:
            parser.error("single-pair mode requires both --activations_dir and --vectors_dir")
        if args.mode == "filtered" and args.scores_dir is None:
            parser.error("--scores_dir is required for --mode filtered")

    load_check = not args.no_load_check

    # Default cache path: alongside the audit output / log files.
    # Caching is ON by default (Roger's drop-mid-scan recovery trumps
    # any concern about cache staleness, which (size, mtime) keying
    # already handles).  --no_cache forces the legacy behaviour.
    if args.no_cache:
        cache_path: Optional[Path] = None
        if args.cache is not None:
            parser.error("--cache and --no_cache are mutually exclusive")
    elif args.cache is not None:
        cache_path = args.cache
    elif using_root:
        cache_path = args.root / "scan_cache.json"
    else:
        cache_path = args.vectors_dir / "scan_cache.json"

    if args.skip_suffix and not using_root:
        logger.warning(
            "--skip_suffix has no effect outside --root mode; ignoring "
            "(in single-pair mode you've already specified the exact "
            "vectors_dir to scan)"
        )

    # Build the list of scan targets.
    targets: list[ScanTarget]
    if using_root:
        matched_suffixes: set[str] = set()
        targets = discover_scan_targets(
            args.root, skip_suffixes=tuple(args.skip_suffix),
            matched_suffixes_out=matched_suffixes,
        )
        # Loud warning if any --skip_suffix value matched nothing -- almost
        # always a typo (e.g. _4slots vs _4slot), and a silent no-op here
        # is exactly the failure mode that wastes hours of NFS scan time.
        unmatched = [s for s in args.skip_suffix if s not in matched_suffixes]
        if unmatched:
            print("=" * 70, file=sys.stderr)
            print(
                f"WARNING: --skip_suffix value(s) matched no discovered "
                f"vectors* dir: {unmatched!r}",
                file=sys.stderr,
            )
            print(
                "         Check the 'Found N scan target(s)' listing above "
                "for the actual\n"
                "         directory basenames -- suffix matching is exact "
                "(no glob, no\n"
                "         pluralisation).  Aborting; re-run with the "
                "corrected suffix\n"
                "         or drop --skip_suffix entirely if you meant to "
                "scan everything.",
                file=sys.stderr,
            )
            print("=" * 70, file=sys.stderr)
            sys.exit(2)
        if not targets:
            print(f"ERROR: no scan targets found under {args.root}", file=sys.stderr)
            print("       (looking for entity-type subdirs containing both an "
                  "activations/ subdir and at least one non-empty vectors* "
                  "subdir)", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(targets)} scan target(s) under {args.root}:")
        for t in targets:
            sf = "" if t.scores_dir is None else f"  scores={t.scores_dir.name}"
            print(f"  {t.label:40s}  mode={t.mode}{sf}")
    else:
        targets = [ScanTarget(
            entity_type=args.activations_dir.parent.name or "explicit",
            vectors_variant=args.vectors_dir.name or "vectors",
            activations_dir=args.activations_dir,
            vectors_dir=args.vectors_dir,
            scores_dir=args.scores_dir,
            mode=args.mode,
        )]

    # Open the load-result cache (default-on; --no_cache to disable).
    # Always close in the finally below so a Ctrl-C still flushes any
    # in-memory updates to disk -- otherwise the entries since the
    # last flush would be lost and re-runs would have to re-load them.
    cache = ScanCache(cache_path) if cache_path is not None else None
    if cache is not None:
        print(f"Cache: {cache_path}")

    scan_results: list[ScanResult] = []
    try:
        for target in targets:
            sr = scan_one_target(
                target,
                min_count=args.min_count,
                reduce_questions=args.reduce_questions,
                deep_load=args.deep_load,
                load_check=load_check,
                cache=cache,
            )
            scan_results.append(sr)
    finally:
        if cache is not None:
            cache.close()

    # Write outputs.
    if using_root:
        # One report + rerun list per scan, alongside the vectors dir.
        for sr in scan_results:
            _write_scan_output(
                sr, min_count=args.min_count,
                reduce_questions=args.reduce_questions,
                deep_load=args.deep_load, load_check=load_check,
            )
    else:
        sr = scan_results[0]
        if args.output is None:
            args.output = sr.target.vectors_dir / "missing_vectors_audit.json"
        sr.output_path = args.output
        sr.output_path.parent.mkdir(parents=True, exist_ok=True)
        sr.output_path.write_text(json.dumps({
            "activations_dir": str(sr.target.activations_dir),
            "vectors_dir": str(sr.target.vectors_dir),
            "scores_dir": str(sr.target.scores_dir) if sr.target.scores_dir else None,
            "min_count": args.min_count,
            "reduce_questions": args.reduce_questions,
            "mode": sr.target.mode,
            "deep_load": args.deep_load,
            "load_check": load_check,
            "median_activation_size_bytes": sr.median_activation_size,
            "results": sr.results,
        }, indent=2))
        if args.rerun_list is not None:
            sr.rerun_list_path = args.rerun_list
            sr.rerun_list_path.parent.mkdir(parents=True, exist_ok=True)
            suspects = sr.suspects()
            sr.rerun_list_path.write_text(
                "\n".join(r["role"] for r in suspects)
                + ("\n" if suspects else "")
            )

    # Print summaries.
    for sr in scan_results:
        _print_scan_summary(sr)

    # Aggregate header (if multiple).
    if len(scan_results) > 1:
        total_counts: dict[str, int] = {}
        total_results = 0
        for sr in scan_results:
            for status, n in sr.counts().items():
                total_counts[status] = total_counts.get(status, 0) + n
            total_results += len(sr.results)
        print(f"\n=== Aggregate across {len(scan_results)} scan(s) ===")
        for status in ALL_STATUSES:
            if status in total_counts:
                print(f"  {status:34s}  {total_counts[status]:5d}")
        print(f"  {'TOTAL':34s}  {total_results:5d}")

        n_suspects = sum(len(sr.suspects()) for sr in scan_results)
        if n_suspects:
            print(f"\n  Total re-run candidates: {n_suspects}")
            print("  See per-scan missing_vectors_rerun.txt files for the "
                  "lists.")


if __name__ == "__main__":
    main()
