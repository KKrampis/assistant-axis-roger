"""End-to-end data-provenance primitives for caches and plots.

Phase 2 of the provenance redesign (see ``audits/post_pipeline_*``,
``tools/regenerate_dataset_manifest.py``, and the plan under
``.cursor/plans/``).

This module provides three things:

1. ``InputSpec`` — a typed, JSON-serialisable record of *one input* to a
   producer (e.g. a dataset subtree, an upstream cache JSON, an axis spec
   file).  Producers attach a list of these to every plot / cache they
   write so a reader can later validate that nothing the artifact depends
   on has drifted.

2. ``Manifest`` — a typed, read-only view of a dataset's
   ``MANIFEST.json``.  Loaded from disk via ``read_manifest(data_dir)``.

3. ``current_*_input`` helpers + ``validate_inputs`` — the core API used
   by writers (to build the provenance record at write time) and readers
   (to compare the recorded record against the current dataset/file
   state and detect drift).

Design choices:

* Fingerprints are *cheap* (mtime+size based for files, subtree
  ``summary_sha256`` from MANIFEST.json for dataset subtrees -- but
  the subtree summary today is also a SHA-256 over sorted
  ``[(rel_path, mtime_ns_floored, size), ...]`` triples, so it is
  metadata-equality only, NOT content-equality).  The
  ``fingerprint_kind`` field is reserved so a future content-hash
  tier (see ``TODO(phase-7-deferred)`` block below) can be added
  without breaking older readers.  As of May 2026 we deliberately
  decided to defer Phase 7; see that TODO for the why, the win, and
  the implementation outline if/when it becomes worth doing.
* ``InputSpec`` carries everything a downstream auditor needs: the
  ``role`` (semantic name), the ``path`` (repo-relative, so manifests
  port across machines), the ``fingerprint`` (the actual freshness
  signal), and ``extras`` for any per-plot discriminators (slot, layer,
  K-range, ...).
* Validation is *additive*: artifacts without a ``_provenance`` block or
  ``Inputs`` chunk keep working; auditors classify them as
  "legacy/unverifiable".  ``validate_inputs`` returns a ``ProvenanceCheck``
  describing per-role status; raising / rebuilding is left to higher-
  level helpers (Phase 4: ``load_validated_json``).

Public API (stable):

    InputSpec, Manifest, ManifestSubtree
    read_manifest(data_dir) -> Manifest
    current_data_subtree_input(data_dir, subtree_rel, role, *, extras=None) -> InputSpec
    current_file_input(role, path, *, extras=None) -> InputSpec
    current_files_input(dep_key, paths, *, extras=None) -> InputSpec
    current_corpus_files_input(dep_key, paths, *, root, entities, fields,
                               extras=None) -> InputSpec
        # current_files_input over corpus instruction JSONs, plus per-entity
        # content hashes of the fields the consumer reads; a metadata drift
        # whose hashes match validates as "equivalent_content" (ok).
    validate_inputs(recorded, current) -> ProvenanceCheck
    inputs_to_jsonable(specs) -> list[dict]
    inputs_from_jsonable(blobs) -> list[InputSpec]
    load_validated_json(path, *, policy="warn", rebuild_callback=None)
        -> (payload, check)
    load_and_register(path, *, dep_key, extras=None, policy="warn",
                      rebuild_callback=None, inputs=None)
        -> (payload, spec, check)
        # Read + envelope-unwrap + drift-validate + build a writer-side
        # InputSpec for the file in one call.  Use this when a script
        # both consumes a cache AND records that cache as a dependency
        # of its own output, which is the common case -- linking the
        # two operations prevents the "read it but forgot to register
        # it" failure mode.
    load_and_register_npz(path, *, dep_key, extras=None, policy="warn",
                          rebuild_callback=None, inputs=None,
                          meta_key="meta", allow_pickle=True)
        -> (npz_handle, meta, spec, check)
        # ``.npz`` analogue of ``load_and_register``.  ``np.savez`` can't
        # carry a JSON envelope, so producers (e.g. winner_decomposition.py)
        # serialise their inputs list into ``meta["_inputs"]`` instead.
        # This helper parses that meta blob, validates the recorded
        # inputs against current state with the usual warn/strict/
        # rebuild/off policy, and builds an InputSpec for the npz
        # itself -- preserving the same single-call atomicity as the
        # JSON path so callers can't read a cache without registering
        # it (or vice versa).

Repo path conventions:
    ``InputSpec.path`` is always recorded *relative to the repo root*
    (the directory containing the ``.git`` dir).  Helpers below use
    ``_repo_root()`` to do that resolution; consumers can rely on it.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import functools
import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Optional

__all__ = [
    "InputSpec",
    "Manifest",
    "ManifestSubtree",
    "ProvenanceCheck",
    "InputStatus",
    "StaleCacheError",
    "MANIFEST_FILENAME",
    "PROVENANCE_SCHEMA_VERSION",
    "FINGERPRINT_VERSION_TAG",
    "OK_STATUSES",
    "CORPUS_CONTENT_VERSION",
    "CACHE_POLICIES",
    "read_manifest",
    "current_data_subtree_input",
    "current_file_input",
    "current_files_input",
    "corpus_content",
    "current_corpus_files_input",
    "current_for_recorded",
    "validate_inputs",
    "validate_recorded",
    "inputs_to_jsonable",
    "inputs_from_jsonable",
    "load_validated_json",
    "load_and_register",
    "load_and_register_npz",
]

# =====================================================================
# TODO(phase-7-deferred): Subtree content hashing
# =====================================================================
#
# STATUS: Deferred (May 2026).  Phases 1-6 of the provenance redesign
# shipped; "subtree content hashing" was originally scoped as Phase 6
# but was demoted to Phase 7 in favour of judge-step provenance, then
# deferred entirely after a cost / value review.  Pick this up if and
# only if one of the failure modes below starts biting in practice.
#
# One part of the idea did land (2026-10-10, AGENT_NOTES housekeeping
# item 3): corpus instruction JSONs recorded as a ``kind="multi"`` input
# can carry per-entity content hashes of the fields the consumer reads
# (``InputSpec.content``, ``current_corpus_files_input``), with the
# ``equivalent_content`` status sketched in step 3 below.  That one bit
# daily (every metadata edit to any trait file staled every judge cache);
# dataset subtrees remain metadata-only.
#
# WHAT WE HAVE TODAY (the baseline this would replace / augment)
# --------------------------------------------------------------
# Every fingerprint in the system is *metadata-only*:
#   - kind="file":     "v1:{mtime_iso_floor_to_seconds}@{size}"
#                      (helpers ``current_file_input``).
#   - kind="multi":    "v1:multi:{sha256(sorted [(rel_path,
#                      mtime_ns_floored, size), ...])[:12]}"
#                      (helpers ``current_files_input``).
#   - kind="subtree":  "v1:{dataset_id}@{summary_sha256[:12]}" where
#                      ``summary_sha256`` is computed by
#                      ``tools/regenerate_dataset_manifest.py``
#                      ``_summarize`` and is itself the SHA-256 over
#                      the same sorted (rel_path, mtime_floored, size)
#                      triples.
#
# So nothing in the project ever hashes file *contents* -- only
# (path, mtime_floored_to_seconds, size) tuples.  That works because
# ``rsync -at`` (the default Roger uses for dataset transfer) preserves
# whole-second mtime, and ``regenerate_dataset_manifest.py`` floors
# mtime to whole seconds, making fingerprints rsync-stable by
# construction.  Manifest regen is <1 second for the full dataset.
#
# WHAT PHASE 7 WOULD BUY
# ----------------------
# It would catch the following classes of false positive (today's
# system says "drift" when no semantic drift occurred):
#
#   1. ``cp`` / ``scp`` / ``cp -p`` / non-``-t`` rsync / tarball
#      extraction that resets file mtimes.  Content identical, mtime
#      fresh, today's fingerprint changes.
#   2. ``git checkout`` of a tracked dataset on a fresh clone.  Mtime
#      is set to checkout time so cross-machine validation always
#      shows drift.
#   3. Editor "save" that rewrites identical bytes.  Mtime bumps,
#      content unchanged.
#
# And the following classes of false negative (today's system says
# "ok" when the file was tampered with):
#
#   4. Bit-rot / partial-write / silent on-disk corruption that does
#      not change mtime or size.
#   5. ``touch -t <old> && edit`` (intentional mtime spoofing).
#
# Not helped by Phase 7 (just being honest):
#
#   - Re-running an upstream pipeline that produces "the same"
#     numerical output.  PyTorch / LLM nondeterminism means the bytes
#     differ, so a content hash drifts even though the experiment is
#     semantically identical.
#   - ``rsync -at`` (which is the workflow today).  Already stable
#     under metadata fingerprinting.
#
# The single most valuable case is (2) -- "I just cloned the dataset
# to a new machine, can I trust the existing caches?" -- which is
# real but not currently a daily pain point.
#
# COST
# ----
# SHA-256 throughput on Roger's hardware: ~1-1.5 GB/s warm-cache on
# M-series macOS w/ SHA-NI; ~600 MB/s cache-cold (disk-bound).  Roger's
# typical dataset shape (Qwen-3-32B, 8 slots, 64 layers, 5120 hidden,
# fp32, ~280-300 entities per role group):
#
#   - traits/vectors/   ~ 3.1 GB  ->  3-6 s
#   - roles/vectors/    ~ 2.9 GB  ->  3-6 s
#   - {traits,roles}/responses/  ~ few GB jsonl  ->  5-10 s
#   - combinations/vectors/derived/marginals/{r_goal,r_nogoal,
#     t_goal,t_nogoal}  ~ few hundred MB each  ->  <2 s each
#   - default/, axis.pt loose files: <1 s
#
# Full-dataset content-hashed regen: ~30-60 s on local SSD,
# 2-5 min on NFS.  vs <1 s for today's metadata-only path.  So
# 30-300x slower for manifest regeneration; manifest regeneration is
# infrequent and explicit (``tools/regenerate_dataset_manifest.py``
# is invoked manually after data-producing steps), so this is OK in
# absolute terms but not a free lunch.
#
# Crucially, validate-time cost stays *zero*: ``validate_recorded``
# would still compare the recorded ``content_sha256`` to the manifest's
# (no I/O beyond ``MANIFEST.json``).  Only an opt-in ``--deep`` mode
# would re-hash on demand.
#
# Storage: ~70 bytes per file in MANIFEST.json (path + 64-hex SHA).
# ~600 files in a typical dataset = ~42 KB extra.  Trivial.
#
# IMPLEMENTATION OUTLINE (~ half a day, ~150 LoC + tests)
# -------------------------------------------------------
# Scope: subtree-only content hashing.  Leave kind="file" and
# kind="multi" on metadata fingerprints (their per-file overhead is
# real and they don't hit the cross-machine case anyway).
#
# 1. ``tools/regenerate_dataset_manifest.py``:
#    a. Add ``content_sha256: str`` to ``SubtreeSummary``.
#    b. In ``_summarize``, stream-hash each file
#       (``hashlib.sha256(); read in 1 MB chunks``) and accumulate
#       per-file ``(rel_path, file_content_sha256)`` pairs.  The
#       subtree's ``content_sha256`` is the SHA-256 of the sorted
#       JSON list of those pairs.
#    c. Add ``--metadata-only`` flag to skip content hashing for the
#       fast path; default could be either (opinion call).  Keep
#       ``FINGERPRINT_KIND`` bumpable so manifests self-identify.
#    d. Print "hashing {name}: {bytes} in {seconds}s" progress per
#       subtree when content hashing is on, since regen now takes
#       30-60 s instead of <1 s.
#
# 2. ``MANIFEST.json`` schema bump (still 1.0-compatible since
#    ``content_sha256`` is additive; readers ignoring it stay
#    correct).  Bump ``FINGERPRINT_KIND`` to ``"mtime_size_v1+content_v1"``
#    or similar so consumers can dispatch.
#
# 3. This module (``assistant_axis/provenance.py``):
#    a. ``current_data_subtree_input`` records the manifest's
#       ``content_sha256`` in ``extras["content_sha256"]`` whenever
#       it's present in the manifest.
#    b. Add ``"equivalent_content"`` to ``InputStatus.status``
#       choices (parallel to the ``"equivalent"`` status added in
#       Phase 6b for script-equivalence downgrades).
#    c. In ``validate_recorded``, when a kind="subtree" input
#       reports metadata drift, also compare
#       ``recorded.extras["content_sha256"]`` against the current
#       manifest's ``content_sha256``; if equal, downgrade the status
#       to ``"equivalent_content"`` with a ``detail`` explaining the
#       transport-noise downgrade.
#    d. Update ``ProvenanceCheck.ok`` to treat
#       ``"equivalent_content"`` as acceptable, same way it treats
#       ``"equivalent"``.
#
# 4. ``tools/audit_caches.py`` and ``tools/audit_pngs.py``:
#    a. Include ``"equivalent_content"`` in ``STATUSES``.
#    b. Render in audit reports under a "Equivalent (content-hash
#       confirmed)" section, parallel to the existing "Equivalent
#       (declared harmless)" rendering.
#
# 5. Tests (~50 LoC):
#    - Round-trip: write subtree, validate -> ok.
#    - Touch every file in subtree to bump mtime, content unchanged
#      -> validate downgrades drift to equivalent_content.
#    - Modify one file's bytes -> validate stays drift (content
#      hash mismatch), confirming false negatives (bit-rot) are
#      caught when the user explicitly re-runs manifest regen.
#    - Old manifest without ``content_sha256``: validate falls back
#      to metadata-only behaviour (graceful).
#
# OPTIONAL EXTENSION (skip until concretely needed):
# A ``--deep`` flag on ``audit_caches.py`` / ``audit_pngs.py`` /
# ``regenerate_dataset_manifest.py --verify`` that re-reads every
# file and confirms the recorded content hashes still match.  Costs
# the full 30-60 s but is the only way to catch (4) bit-rot without
# a manifest regen.
#
# DECISION CRITERIA (when to actually do this)
# --------------------------------------------
# Pick up Phase 7 if any of the following becomes routinely true:
#   - We start moving datasets between machines via something other
#     than ``rsync -at`` (e.g. ``scp``, S3 sync, tarball restore,
#     dataset-in-git on a fresh clone).
#   - We get false-positive drift in audit_caches / audit_pngs from
#     tooling that touches files without changing content (some
#     editor's format-on-save, some build-tool's noop rewrites).
#   - We ever need a defensible answer to "did this dataset get
#     corrupted on disk between manifest regen N and now?".
# Until any of those is the case, the metadata-only fingerprint is
# strictly cheaper for equal-or-better real-world behaviour.
# =====================================================================


MANIFEST_FILENAME = "MANIFEST.json"
PROVENANCE_SCHEMA_VERSION = "1.0"

# Length of the subtree summary_sha256 prefix embedded in fingerprints.
# Short enough to fit in PNG iTXt chunks comfortably; long enough that a
# random collision across the few-hundred subtrees we touch is negligible
# (12 hex = 48 bits; birthday collision threshold is ~17M items).
_SUBTREE_SHA_PREFIX_LEN = 12

# Version tag that prefixes every fingerprint string.  Lets future
# format changes (e.g. switching subtree summaries to content hashes,
# or files to mtime_ns) coexist with old recorded fingerprints: the
# validator can dispatch on the prefix rather than format-mismatch
# silently.  Bump to "v2:" when the fingerprint *string format* (not
# the underlying data) changes.
FINGERPRINT_VERSION_TAG = "v1"

# Statuses that count as current.  Audit tools import this rather than
# listing the statuses themselves.
OK_STATUSES = ("ok", "equivalent", "equivalent_content")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputSpec:
    """One input that a writer depends on.

    Attributes:
        dep_key: Abbreviated, writer-chosen name for one dependency
            (literally a "key" identifying one "dep").  Used as the
            dict key inside :func:`validate_inputs` when comparing
            recorded vs current.  Choose a short, namespaced string
            (e.g. ``"data_dir/traits"``, ``"sweep_json_K"``,
            ``"judge_scores_inst"``); see ``AGENT_NOTES.md`` for the
            project naming convention.

            Why "dep_key" and not ``role``: the canonical character
            archetypes in this project are called *roles*
            (``data/roles/instructions/``), and ``roles/vectors`` is
            already a manifest subtree -- having ``InputSpec.role =
            "roles/vectors"`` would be an avoidable collision.
        path: Repo-relative path to the input.  For dataset-subtree
            inputs this is ``<dataset_root_rel>/<subtree>``; for files
            it's the full repo-relative path to the file.
        fingerprint: Versioned compact freshness signal.  Always
            prefixed with the current ``FINGERPRINT_VERSION_TAG`` so
            future format upgrades have a clean migration path.
            Current formats:

            - subtree: ``"v1:{dataset_id}@{summary_sha256[:12]}"``
            - file:    ``"v1:{mtime_iso}@{size}"``
        kind: ``"subtree"`` or ``"file"``.
        dataset_id: The dataset's ``dataset_id`` from ``MANIFEST.json``
            when ``kind == "subtree"``; ``None`` for plain files.
        last_modified_at: ISO-8601 UTC timestamp of the input's most
            recent modification at the time the InputSpec was built.
        extras: Free-form discriminators (slot, layer, K-range, ...);
            advisory only -- not used in equality / drift comparisons.
        member_paths: For ``kind == "multi"`` only: the repo-relative
            paths of the constituent files.  Allows a downstream reader
            to recompute the multi fingerprint from the same file set
            and detect drift.  Without it, multi fingerprints are
            write-only diagnostics (legacy multi InputSpecs are
            classified ``unverifiable`` by :func:`load_validated_json`).
            ``None`` for non-multi kinds.
        content: For a ``kind == "multi"`` input over corpus instruction
            JSONs only (built by :func:`current_corpus_files_input`): a
            content hash of the fields the consumer actually reads, per
            entity.  The metadata ``fingerprint`` stays the primary
            signal; when it drifts and these hashes still match,
            :func:`validate_recorded` reports ``equivalent_content`` (ok)
            instead of ``drift``.  Shape: ``{"version", "root", "fields",
            "sha256", "entities": {entity_id: hash or None}}``.  ``None``
            for every other input, and for corpus inputs recorded before
            2026-10-10.
    """
    dep_key: str
    path: str
    fingerprint: str
    kind: str
    dataset_id: Optional[str] = None
    last_modified_at: Optional[str] = None
    extras: dict = field(default_factory=dict)
    member_paths: Optional[list] = None  # list[str] | None
    content: Optional[dict] = None


@dataclass(frozen=True)
class ManifestSubtree:
    """One row from a dataset's ``MANIFEST.json``."""
    name: str
    kind: str          # "raw" or "derived"
    recurse: bool
    count: int
    total_bytes: int
    newest_mtime: str
    summary_sha256: str


@dataclass(frozen=True)
class Manifest:
    """Typed read-only view of a dataset's ``MANIFEST.json``."""
    dataset_id: str
    schema_version: str
    fingerprint_kind: str
    manifest_generated_at: str
    subtree_summaries: dict      # name -> ManifestSubtree
    data_dir: Path                # absolute path to the dataset root
    raw: dict                     # the full parsed JSON (escape hatch)

    def subtree(self, name: str) -> ManifestSubtree:
        """Return one subtree's summary, raising ``KeyError`` if missing."""
        try:
            return self.subtree_summaries[name]
        except KeyError as e:
            available = ", ".join(sorted(self.subtree_summaries)[:6])
            raise KeyError(
                f"subtree {name!r} not in MANIFEST.json for {self.data_dir}; "
                f"available (first 6): {available}, ..."
            ) from e


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputStatus:
    dep_key: str
    # One of:
    #   "ok"               -- recorded fingerprint matches current.
    #   "drift"            -- both exist; fingerprints differ.
    #   "equivalent"       -- kind="file" drift downgraded by an entry
    #                         in script_equivalences.yaml (output-
    #                         preserving edit; downstream caches don't
    #                         need rebuilding).
    #   "equivalent_content" -- kind="multi" metadata drift on corpus
    #                         files whose recorded content hashes
    #                         (``InputSpec.content``) still match: the
    #                         edits touched fields the consumer does
    #                         not read (tags, arrangement, generator,
    #                         questions, ...).
    #   "missing_current"  -- file/manifest vanished since the record
    #                         was made.
    #   "missing_recorded" -- dep_key in current but not recorded.
    #   "unverifiable"     -- legacy multi-input without member_paths.
    status: str
    recorded: Optional[InputSpec] = None
    current: Optional[InputSpec] = None
    detail: str = ""


@dataclass(frozen=True)
class ProvenanceCheck:
    """Per-input drift report built by :func:`validate_inputs`."""
    statuses: list           # list[InputStatus]
    ok: bool                 # True iff every status is in OK_STATUSES

    def drifted(self) -> list:
        return [s for s in self.statuses if s.status == "drift"]

    def equivalent(self) -> list:
        """Drift downgraded to ok, by either route: a declared script
        equivalence or matching corpus content hashes."""
        return [s for s in self.statuses
                if s.status in ("equivalent", "equivalent_content")]

    def missing(self) -> list:
        return [s for s in self.statuses
                if s.status in ("missing_current", "missing_recorded")]

    def summary(self) -> str:
        lines = []
        for s in self.statuses:
            line = f"  [{s.status:>17s}] {s.dep_key}"
            if s.detail:
                line += f"  -- {s.detail}"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Repo-root resolution (for path normalisation)
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Locate the repository root (the dir containing ``.git``).

    Walks up from this module's location, which is at
    ``<repo>/assistant_axis/provenance.py``.  Falls back to the cwd if
    no .git is found (e.g. running outside a git checkout).
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def _to_repo_relative(path: Path) -> str:
    """Normalize a path to a forward-slash repo-relative string.  If the
    path is outside the repo, return its absolute form."""
    p = Path(path).resolve()
    root = _repo_root()
    try:
        rel = p.relative_to(root)
    except ValueError:
        return str(p).replace(os.sep, "/")
    return str(rel).replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------

def read_manifest(data_dir: Path) -> Manifest:
    """Load ``data_dir / 'MANIFEST.json'`` and return a typed view.

    Raises:
        FileNotFoundError: if no MANIFEST.json is present.  Suggests the
            ``tools/regenerate_dataset_manifest.py`` regeneration command.
        ValueError: if the schema is unrecognised.
    """
    data_dir = Path(data_dir)
    p = data_dir / MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError(
            f"No {MANIFEST_FILENAME} in {data_dir}.  Regenerate via:\n"
            f"    uv run python tools/regenerate_dataset_manifest.py "
            f"--dataset {str(data_dir)!r}"
        )
    raw = json.loads(p.read_text())
    schema_version = raw.get("schema_version", "?")
    if schema_version not in ("1.0",):
        raise ValueError(
            f"Unsupported MANIFEST.json schema_version {schema_version!r} at {p}; "
            f"this code understands 1.0."
        )
    summaries: dict[str, ManifestSubtree] = {}
    for name, blob in raw.get("subtree_summaries", {}).items():
        summaries[name] = ManifestSubtree(
            name=name,
            kind=blob["kind"],
            recurse=bool(blob.get("recurse", False)),
            count=int(blob["count"]),
            total_bytes=int(blob["total_bytes"]),
            newest_mtime=blob["newest_mtime"],
            summary_sha256=blob["summary_sha256"],
        )
    return Manifest(
        dataset_id=raw["dataset_id"],
        schema_version=schema_version,
        fingerprint_kind=raw.get("fingerprint_kind", "unknown"),
        manifest_generated_at=raw["manifest_generated_at"],
        subtree_summaries=summaries,
        data_dir=data_dir.resolve(),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# InputSpec builders (writer-side)
# ---------------------------------------------------------------------------

def current_data_subtree_input(
    data_dir: Path,
    subtree_rel: str,
    dep_key: str,
    *,
    extras: Optional[dict] = None,
    manifest: Optional[Manifest] = None,
) -> InputSpec:
    """Build an :class:`InputSpec` describing one dataset subtree.

    Args:
        data_dir: Dataset root.
        subtree_rel: Subtree name as it appears in ``MANIFEST.json``
            (e.g. ``"traits/vectors"``,
            ``"combinations/vectors/derived/marginals/r_goal"``).
        dep_key: Caller-chosen abbreviated name for this dependency.
            See :class:`InputSpec` for the naming convention.
        extras: Optional per-call discriminators (e.g. ``{"slot": "6"}``).
        manifest: Pre-loaded :class:`Manifest`; if omitted, we read it
            from disk.

    Returns:
        InputSpec with fingerprint
        ``"v1:{dataset_id}@{summary_sha256[:12]}"``, ``kind="subtree"``,
        path ``"<dataset_root_rel>/<subtree_rel>"``.
    """
    if manifest is None:
        manifest = read_manifest(data_dir)
    sub = manifest.subtree(subtree_rel)
    dataset_root_rel = _to_repo_relative(manifest.data_dir)
    full_rel = f"{dataset_root_rel}/{subtree_rel}"
    fp = (f"{FINGERPRINT_VERSION_TAG}:{manifest.dataset_id}"
          f"@{sub.summary_sha256[:_SUBTREE_SHA_PREFIX_LEN]}")
    return InputSpec(
        dep_key=dep_key,
        path=full_rel,
        fingerprint=fp,
        kind="subtree",
        dataset_id=manifest.dataset_id,
        last_modified_at=sub.newest_mtime,
        extras=dict(extras or {}),
    )


def current_file_input(
    dep_key: str,
    path: Path,
    *,
    extras: Optional[dict] = None,
) -> InputSpec:
    """Build an :class:`InputSpec` for a single file (cache JSON, axis spec, ...).

    Fingerprint is ``"v1:{mtime_iso}@{size}"`` over ``os.stat``
    (follows symlinks -- so a symlink fingerprint reflects the
    target's stat, which is what we want for "did the data this
    points at change?" semantics).  ``size`` goes last because it's
    variable-length and we want the fixed-format mtime to anchor any
    prefix-based parser.

    Mtime is floored to whole seconds.  Roger's standard workflow
    (RunPod -> NFS -> rsync -> Mac) hands us second-precision mtimes
    on every transfer; storing nanos here would mostly waste space
    and would false-positive any time a file arrives via a
    sub-second-preserving path.  Phase 6 (content hashes) is the
    upgrade path for sub-second-precision drift detection.

    Use ``current_data_subtree_input`` for files that are part of a
    managed dataset subtree (those go through MANIFEST.json instead).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"current_file_input: {p} does not exist (dep_key={dep_key!r})"
        )
    st = os.stat(p)
    # Truncate to whole seconds (drops sub-second component) so
    # rsync-rounded and natively-stored mtimes produce the same
    # fingerprint.
    mtime_iso = _dt.datetime.fromtimestamp(
        int(st.st_mtime), tz=_dt.timezone.utc
    ).isoformat()
    return InputSpec(
        dep_key=dep_key,
        path=_to_repo_relative(p),
        fingerprint=f"{FINGERPRINT_VERSION_TAG}:{mtime_iso}@{st.st_size}",
        kind="file",
        dataset_id=None,
        last_modified_at=mtime_iso,
        extras=dict(extras or {}),
    )


def current_files_input(
    dep_key: str,
    paths: Iterable[Path],
    *,
    extras: Optional[dict] = None,
) -> InputSpec:
    """Build an :class:`InputSpec` summarising a heterogeneous *set* of files
    (e.g. per-axis judge caches, scattered config files) into one composite
    fingerprint.

    Use this when a writer fans out over many small inputs that aren't
    naturally a managed dataset subtree.  Examples:

    * ``whitening_k_sweep.py``: ~132 per-axis judge caches at
      ``roger/axis_judge_experiments/<axis>/{gpt,sonnet}/scores_*.json``.
    * ``batch_size_rho_curve.py``: per-batch-size cache fan-out.

    Granularity rationale: the question we want answered downstream is
    "do I need to rebuild?", not "which file in the set changed".  A
    composite fingerprint answers the former cheaply; if you need the
    latter for debugging, re-stat the files and compare to the
    recorded sub-list (the writer can always log it on demand, or we
    can add it to ``extras`` per-call).

    Fingerprint shape: ``"v1:multi:{sha256[:12]}"`` over the sorted JSON
    list ``[(rel_path, mtime_seconds, size), ...]`` of every existing
    path.  Non-existent paths are silently skipped (lets writers iterate
    over an "expected" set that includes optional caches; see the
    ``responses`` files in ``whitening_k_sweep.py``).

    If *every* path is missing, returns a sentinel
    ``"v1:multi:empty"`` fingerprint -- still a valid InputSpec so the
    declared dependency stays visible to auditors, but obviously
    distinguishable from any real fingerprint.

    The ``path`` field is set to the longest common path prefix when
    there's more than one file (advisory; useful for human inspection),
    or the single file's path when there's exactly one.

    The ``last_modified_at`` field is the max mtime across the set
    (ISO-8601 UTC, second precision).  Useful for human inspection
    even when the fingerprint matches.
    """
    # Materialise once; ``paths`` may be a generator.
    paths = [Path(p) for p in paths]
    # Dedupe by repo-relative path (which uses ``Path.resolve``, so
    # symlinks pointing at the same target collapse into one entry).
    # This keeps ``triples`` consistent with ``all_member_rels`` -- a
    # bug bit Phase 6a when two ``default.pt`` symlinks both pointed
    # at the same shared target, producing a write-time fingerprint
    # that included duplicate triples while the recorded
    # ``member_paths`` was deduped, so re-validation always saw drift.
    triples_by_rel: dict[str, tuple[str, int, int]] = {}
    newest_mtime_secs = 0
    for p in paths:
        if not p.exists():
            continue
        st = os.stat(p)
        rel = _to_repo_relative(p)
        # Floor mtime to whole seconds for rsync stability (see
        # ``current_file_input`` docstring).
        mtime_secs = int(st.st_mtime)
        triples_by_rel[rel] = (rel, mtime_secs, int(st.st_size))
        if mtime_secs > newest_mtime_secs:
            newest_mtime_secs = mtime_secs
    triples = sorted(triples_by_rel.values())
    rels = sorted(triples_by_rel.keys())

    # Member paths capture the *full* expected set (including missing
    # ones) so a downstream reader can re-stat them and recompute the
    # fingerprint -- including detecting "a file that was missing at
    # write-time has since appeared" as drift.  This is symmetric with
    # the read-time treatment in ``_current_for_recorded``.
    all_member_rels: list[str] = []
    for p in paths:
        all_member_rels.append(_to_repo_relative(Path(p)))
    all_member_rels = sorted(set(all_member_rels))

    if not triples:
        return InputSpec(
            dep_key=dep_key,
            path="(empty)",
            fingerprint=f"{FINGERPRINT_VERSION_TAG}:multi:empty",
            kind="multi",
            dataset_id=None,
            last_modified_at=None,
            extras=dict(extras or {}),
            member_paths=all_member_rels or None,
        )

    blob = json.dumps(triples, separators=(",", ":")).encode("utf-8")
    sha = hashlib.sha256(blob).hexdigest()
    fp = f"{FINGERPRINT_VERSION_TAG}:multi:{sha[:_SUBTREE_SHA_PREFIX_LEN]}"

    path = rels[0] if len(rels) == 1 else os.path.commonpath(rels)
    newest_iso = _dt.datetime.fromtimestamp(
        newest_mtime_secs, tz=_dt.timezone.utc).isoformat()

    return InputSpec(
        dep_key=dep_key,
        path=path,
        fingerprint=fp,
        kind="multi",
        dataset_id=None,
        last_modified_at=newest_iso,
        extras=dict(extras or {}),
        member_paths=all_member_rels,
    )


# ---------------------------------------------------------------------------
# Content hashes of corpus fields (AGENT_NOTES housekeeping item 3, 2026-10-10)
# ---------------------------------------------------------------------------
#
# A consumer of the corpus instruction JSONs records them as one
# ``kind="multi"`` input, fingerprinted by (mtime, size), so *any* edit to
# any file used to mark every consumer stale, metadata-only edits
# (``arrangement``, ``tags``, ``generator``, ``questions``) included.  The
# ``content`` block below records, per entity the consumer actually reads,
# a hash of just the fields that reach its rubric or prompt.  The metadata
# fingerprint stays the primary signal, so a record written before this
# block existed validates exactly as before; when the metadata drifts and
# the hashes match, the status is ``equivalent_content`` (ok).  This is
# the per-entity, per-field form of the "content hashes" plan in the
# deferred block at the top of this module, which remains deferred for
# dataset subtrees.

CORPUS_CONTENT_VERSION = "corpus_fields_v1"
_ENTITY_HASH_LEN = 16


def _corpus_field_value(doc, field_path: str):
    """One field of an instruction JSON: ``description`` reads a top-level
    key; ``instruction[*].pos`` reads the ``pos`` of every item of the
    ``instruction`` list (``None`` where a key is absent)."""
    if not isinstance(doc, dict):
        return None
    base, sep, sub = field_path.partition("[*].")
    value = doc.get(base)
    if not sep:
        return value
    if not isinstance(value, list):
        return None
    return [item.get(sub) if isinstance(item, dict) else None for item in value]


@functools.lru_cache(maxsize=16384)
def _entity_content_hash_cached(path: str, mtime_ns: int, size: int,
                                fields: tuple) -> str:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError:
        return "unreadable"
    blob = json.dumps({f: _corpus_field_value(doc, f) for f in fields},
                      sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_ENTITY_HASH_LEN]


def _entity_content_hash(path: Path, fields: tuple) -> Optional[str]:
    """Hash of ``fields`` of one instruction JSON; ``None`` when the file
    does not exist.  Memoised on (path, mtime_ns, size), so an audit that
    revalidates many records reads each file once."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return _entity_content_hash_cached(str(path), st.st_mtime_ns,
                                       st.st_size, tuple(fields))


def _content_digest(entity_hashes: dict) -> str:
    blob = json.dumps(sorted(entity_hashes.items()), separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def corpus_content(
    root: Path,
    entities: Iterable[tuple],
    fields: Iterable[str],
) -> dict:
    """The ``InputSpec.content`` block for corpus instruction JSONs under
    ``root`` (``<root>/<kind>/instructions/<stem>.json``).

    ``entities`` are ``(kind, stem)`` pairs, the entities the consumer
    reads; ``fields`` the JSON fields that plug into its rubric or prompt
    (``"description"``, ``"positive_label"``, ``"instruction[*].pos"``,
    ...).  Entities are keyed by their disambiguated id (``"lazy|T"``).
    """
    from assistant_axis.entity_id import entity_id, kind_long

    root = Path(root)
    fields = tuple(fields)
    hashes: dict = {}
    for kind, stem in entities:
        path = root / kind_long(kind) / "instructions" / f"{stem}.json"
        hashes[entity_id(stem, kind)] = _entity_content_hash(path, fields)
    hashes = dict(sorted(hashes.items()))
    return {
        "version": CORPUS_CONTENT_VERSION,
        "root": _to_repo_relative(root),
        "fields": list(fields),
        "sha256": _content_digest(hashes),
        "entities": hashes,
    }


def _current_corpus_content(recorded: dict) -> dict:
    """Recompute a recorded ``content`` block from the files as they are now."""
    from assistant_axis.entity_id import parse_entity_id

    root = _repo_root() / recorded["root"]
    entities = []
    for eid in recorded.get("entities") or {}:
        parsed = parse_entity_id(eid)
        entities.append((parsed.kind, parsed.name))
    return corpus_content(root, entities, recorded.get("fields") or [])


def current_corpus_files_input(
    dep_key: str,
    paths: Iterable[Path],
    *,
    root: Path,
    entities: Iterable[tuple],
    fields: Iterable[str],
    extras: Optional[dict] = None,
) -> InputSpec:
    """:func:`current_files_input` over corpus instruction JSONs, plus the
    ``content`` block (:func:`corpus_content`) for the entities and fields
    the consumer reads.  ``paths`` keeps its meaning (the files whose
    metadata is fingerprinted, usually every JSON the script could open);
    ``entities`` may be a subset (those it does open)."""
    spec = current_files_input(dep_key, paths, extras=extras)
    return dataclasses.replace(
        spec, content=corpus_content(root, entities, fields))


def _content_check(recorded: InputSpec) -> tuple:
    """For a multi input with a ``content`` block whose metadata drifted:
    ``(same, current_block, detail)``."""
    rc = recorded.content or {}
    if rc.get("version") != CORPUS_CONTENT_VERSION:
        return False, None, (f"content block version {rc.get('version')!r} "
                             f"is not {CORPUS_CONTENT_VERSION!r}; cannot compare")
    cur = _current_corpus_content(rc)
    if cur["sha256"] == rc.get("sha256"):
        return True, cur, ""
    before = rc.get("entities") or {}
    after = cur["entities"]
    changed = sorted(e for e in before if before[e] != after.get(e))
    gone = [e for e in changed if after.get(e) is None]
    shown = ", ".join(changed[:10]) + (", ..." if len(changed) > 10 else "")
    detail = (f"{len(changed)} of {len(before)} entities changed in "
              f"{'/'.join(rc.get('fields') or [])}: {shown}")
    if gone:
        detail += f" ({len(gone)} no longer have a file)"
    return False, cur, detail


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(
    recorded: Iterable[InputSpec],
    current: Iterable[InputSpec],
) -> ProvenanceCheck:
    """Compare two lists of :class:`InputSpec` by ``dep_key``; return
    per-input status.

    A drift is any of:

    * ``dep_key`` present in ``recorded`` but absent in ``current``
      (``missing_current``); the dependency vanished.
    * ``dep_key`` present in ``current`` but absent in ``recorded``
      (``missing_recorded``); typically not an error -- the caller may
      have added a new dependency post-record -- but flagged so the
      caller can decide.
    * Both present but ``fingerprint`` differs (``drift``).

    Returns a :class:`ProvenanceCheck` whose ``ok`` field is ``True``
    iff every ``dep_key`` is matched and fingerprints agree.
    """
    rec_by_key: dict[str, InputSpec] = {s.dep_key: s for s in recorded}
    cur_by_key: dict[str, InputSpec] = {s.dep_key: s for s in current}
    statuses: list[InputStatus] = []
    all_keys = sorted(set(rec_by_key) | set(cur_by_key))
    for k in all_keys:
        r = rec_by_key.get(k)
        c = cur_by_key.get(k)
        if r is None:
            statuses.append(InputStatus(
                dep_key=k, status="missing_recorded",
                recorded=None, current=c,
                detail=("recorded list has no entry for this dep_key; "
                        "caller may have added a new dependency."),
            ))
            continue
        if c is None:
            statuses.append(InputStatus(
                dep_key=k, status="missing_current",
                recorded=r, current=None,
                detail="dependency vanished; the input no longer exists.",
            ))
            continue
        if r.fingerprint != c.fingerprint:
            statuses.append(InputStatus(
                dep_key=k, status="drift",
                recorded=r, current=c,
                detail=f"fingerprint changed: {r.fingerprint} -> {c.fingerprint}",
            ))
            continue
        statuses.append(InputStatus(
            dep_key=k, status="ok", recorded=r, current=c,
        ))
    ok = all(s.status == "ok" for s in statuses)
    return ProvenanceCheck(statuses=statuses, ok=ok)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def inputs_to_jsonable(specs: Iterable[InputSpec]) -> list[dict]:
    """Serialise a list of :class:`InputSpec` for inclusion in a PNG
    iTXt chunk or a JSON cache's ``_provenance.inputs`` field."""
    out: list[dict] = []
    for s in specs:
        d = asdict(s)
        # Drop ``member_paths`` for non-multi inputs to keep records
        # tidy (it's always None for those).  Round-trip is preserved
        # because ``inputs_from_jsonable`` re-defaults to None.  Same for
        # ``content``, which only corpus inputs carry.
        if d.get("member_paths") is None:
            d.pop("member_paths", None)
        if d.get("content") is None:
            d.pop("content", None)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Reader-side helpers (Phase 4)
# ---------------------------------------------------------------------------

CACHE_POLICIES = ("strict", "warn", "rebuild", "off")


class StaleCacheError(Exception):
    """Raised by :func:`load_validated_json` under ``policy="strict"``
    (or ``"rebuild"`` without a callback) when a recorded provenance
    drifts from current state.

    Attributes:
        path: The cache file we tried to load.
        check: The :class:`ProvenanceCheck` describing the drift.

    Example::

        try:
            data, _ = load_validated_json(cache_path, policy="strict")
        except StaleCacheError as e:
            print(e.check.summary())
            sys.exit(1)
    """

    def __init__(self, path: Path, check: "ProvenanceCheck"):
        self.path = path
        self.check = check
        msg = (f"Stale cache: {path}\n"
               f"  drifted/missing inputs:\n{check.summary()}")
        super().__init__(msg)


class _UnverifiableInput(Exception):
    """Internal: raised when we can't compute current state for a recorded
    InputSpec (e.g. legacy multi without member_paths)."""


def _find_dataset_root(rel_path: str) -> Optional[Path]:
    """Walk up from ``<repo>/rel_path`` looking for ``MANIFEST.json``.
    Returns ``None`` if no manifest is found anywhere up the chain."""
    p = (_repo_root() / rel_path).resolve()
    if p.is_file():
        p = p.parent
    for ancestor in [p, *p.parents]:
        if (ancestor / MANIFEST_FILENAME).exists():
            return ancestor
    return None


def current_for_recorded(spec: InputSpec) -> InputSpec:
    """Re-derive the *current* :class:`InputSpec` for a previously-recorded one.

    Dispatches on ``spec.kind``:

    - ``"subtree"``: walks up ``spec.path`` looking for the dataset
      root's ``MANIFEST.json``, then re-fingerprints the named subtree.
    - ``"file"``: re-stats the file at ``spec.path``.
    - ``"multi"``: re-stats every ``spec.member_paths`` entry and
      recomputes the composite fingerprint.  Raises
      :class:`_UnverifiableInput` if ``member_paths`` is missing
      (legacy multi spec) -- callers should treat that case as
      "unverifiable" rather than "drift".

    Raises:
        FileNotFoundError: file/manifest vanished.
        KeyError: subtree was renamed/removed in the manifest.
        ValueError: unknown ``kind``.
        _UnverifiableInput: legacy multi spec without member_paths.
    """
    if spec.kind == "subtree":
        root = _find_dataset_root(spec.path)
        if root is None:
            raise FileNotFoundError(
                f"No MANIFEST.json found above {spec.path!r}; "
                f"cannot revalidate subtree input."
            )
        full_abs = (_repo_root() / spec.path).resolve()
        try:
            sub_rel = full_abs.relative_to(root)
        except ValueError as e:
            raise FileNotFoundError(
                f"Recorded subtree path {spec.path!r} not under "
                f"discovered dataset root {root}; broken layout?"
            ) from e
        return current_data_subtree_input(
            data_dir=root,
            subtree_rel=str(sub_rel).replace(os.sep, "/"),
            dep_key=spec.dep_key,
            extras=dict(spec.extras or {}),
        )
    if spec.kind == "file":
        return current_file_input(
            dep_key=spec.dep_key,
            path=_repo_root() / spec.path,
            extras=dict(spec.extras or {}),
        )
    if spec.kind == "multi":
        if not spec.member_paths:
            raise _UnverifiableInput(
                f"multi input {spec.dep_key!r} has no recorded "
                f"member_paths (likely written before that field "
                f"existed); regenerate the cache to enable validation."
            )
        return current_files_input(
            dep_key=spec.dep_key,
            paths=[_repo_root() / mp for mp in spec.member_paths],
            extras=dict(spec.extras or {}),
        )
    raise ValueError(f"Unknown InputSpec.kind {spec.kind!r}")


def validate_recorded(
    recorded: Iterable[InputSpec],
) -> ProvenanceCheck:
    """Compute current state for each recorded input and produce a
    :class:`ProvenanceCheck`.

    Differs from :func:`validate_inputs` in that the caller does not
    have to provide a ``current`` list; this helper re-derives current
    state from each recorded :class:`InputSpec` via
    :func:`current_for_recorded`.  Used by audits / reader-side
    helpers (notably :func:`load_validated_json`).

    Status semantics:

    - ``ok``: recorded fingerprint matches the freshly-recomputed
      current fingerprint.
    - ``drift``: both exist but fingerprints differ.
    - ``missing_current``: file vanished or manifest disappeared
      since the record was made.
    - ``unverifiable``: e.g. legacy multi-input without
      ``member_paths``; can't recompute the current fingerprint.
    """
    recorded = list(recorded)
    statuses: list[InputStatus] = []
    for r in recorded:
        try:
            c = current_for_recorded(r)
        except _UnverifiableInput as e:
            statuses.append(InputStatus(
                dep_key=r.dep_key, status="unverifiable",
                recorded=r, current=None, detail=str(e),
            ))
            continue
        except (FileNotFoundError, KeyError) as e:
            statuses.append(InputStatus(
                dep_key=r.dep_key, status="missing_current",
                recorded=r, current=None, detail=str(e),
            ))
            continue
        if r.fingerprint == c.fingerprint:
            statuses.append(InputStatus(
                dep_key=r.dep_key, status="ok",
                recorded=r, current=c,
            ))
        elif r.kind == "multi" and r.content:
            # Corpus files whose metadata drifted: compare the content of
            # the fields the consumer reads (housekeeping item 3).
            same, cur_content, why = _content_check(r)
            c = dataclasses.replace(c, content=cur_content)
            if same:
                statuses.append(InputStatus(
                    dep_key=r.dep_key, status="equivalent_content",
                    recorded=r, current=c,
                    detail=(f"fingerprint changed: {r.fingerprint} -> "
                            f"{c.fingerprint}; the "
                            f"{len(r.content.get('entities') or {})} entities' "
                            f"{'/'.join(r.content.get('fields') or [])} "
                            f"are unchanged"),
                ))
            else:
                statuses.append(InputStatus(
                    dep_key=r.dep_key, status="drift",
                    recorded=r, current=c,
                    detail=f"fingerprint changed: {r.fingerprint} -> {c.fingerprint}; {why}",
                ))
        else:
            # Drift on a kind="file" input may be a declared-harmless
            # edit; consult the script-equivalence registry before
            # treating it as real drift.  Imported lazily to avoid a
            # hard dep on PyYAML at import time of this module.
            equivalent_via: list = []
            if r.kind == "file":
                try:
                    from assistant_axis.script_equivalence import (
                        is_equivalent as _is_equivalent,
                    )
                    found, path = _is_equivalent(
                        r.path, r.fingerprint, c.fingerprint,
                        return_path=True,
                    )
                    if found:
                        equivalent_via = path
                except Exception:
                    # Registry parse / IO problems must not crash
                    # validation; surface as plain drift.
                    equivalent_via = []
            if equivalent_via:
                reasons = " ; ".join(
                    f"{e.from_fp[:24]}->{e.to_fp[:24]}: {e.reason}"
                    for e in equivalent_via
                )
                statuses.append(InputStatus(
                    dep_key=r.dep_key, status="equivalent",
                    recorded=r, current=c,
                    detail=(
                        f"fingerprint changed: {r.fingerprint} -> "
                        f"{c.fingerprint}; declared equivalent via "
                        f"script_equivalences.yaml ({len(equivalent_via)} "
                        f"hop{'s' if len(equivalent_via) != 1 else ''}: "
                        f"{reasons})"
                    ),
                ))
            else:
                statuses.append(InputStatus(
                    dep_key=r.dep_key, status="drift",
                    recorded=r, current=c,
                    detail=f"fingerprint changed: {r.fingerprint} -> {c.fingerprint}",
                ))
    ok = all(s.status in OK_STATUSES for s in statuses)
    return ProvenanceCheck(statuses=statuses, ok=ok)


def load_validated_json(
    path: Path,
    *,
    policy: str = "warn",
    rebuild_callback=None,
):
    """Load a JSON file (envelope-aware) and validate its recorded
    inputs against current state.

    Args:
        path: JSON file to load.
        policy: Drift handling.  One of:

            * ``"strict"`` -- raise :class:`StaleCacheError` on any
              drift / missing / unverifiable.
            * ``"warn"`` (default) -- print a summary to stderr and
              return the payload anyway.
            * ``"rebuild"`` -- if ``rebuild_callback`` is provided,
              call ``rebuild_callback(path, check)`` (it should bring
              the cache back to currency) and re-validate; else
              behaves like ``"strict"``.
            * ``"off"`` -- skip validation entirely; just return the
              payload (with the envelope still unwrapped if present).

        rebuild_callback: Optional ``Callable[[Path, ProvenanceCheck], None]``
            invoked under ``policy="rebuild"``.  After it returns, the
            file is reloaded and re-validated; if drift persists, a
            :class:`StaleCacheError` is raised.

    Returns:
        ``(payload, check)`` where:

        * ``payload`` is the unwrapped result list/dict.  When the
          loaded JSON has no provenance envelope (legacy or hand-
          written), ``payload`` is the raw decoded JSON.
        * ``check`` is the :class:`ProvenanceCheck` from the
          revalidation pass.  ``None`` for legacy files without an
          envelope, or when ``policy="off"``.

    Notes:
        ``policy="off"`` skips validation but still unwraps the
        envelope, so callers can use it as a one-liner replacement for
        ``json.load`` that quietly tolerates either format.
    """
    if policy not in CACHE_POLICIES:
        raise ValueError(
            f"Unknown cache policy {policy!r}; "
            f"choose from {CACHE_POLICIES}.")

    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))

    # Detect envelope.  The schema is ``{"result": <payload>, "_provenance": {...}}``.
    has_envelope = (
        isinstance(obj, dict)
        and "_provenance" in obj
        and "result" in obj
    )
    payload = obj["result"] if has_envelope else obj

    if policy == "off" or not has_envelope:
        return payload, None

    recorded = inputs_from_jsonable(obj["_provenance"].get("inputs", []))
    check = validate_recorded(recorded)

    if check.ok:
        return payload, check

    if policy == "warn":
        import sys
        sys.stderr.write(
            f"[provenance] {path}: drift detected (policy=warn, returning "
            f"payload anyway):\n{check.summary()}\n"
        )
        return payload, check

    if policy == "rebuild" and rebuild_callback is not None:
        rebuild_callback(path, check)
        # Re-load + re-validate after the callback.
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not (isinstance(obj, dict)
                and "_provenance" in obj and "result" in obj):
            raise StaleCacheError(path, check)
        payload = obj["result"]
        recorded = inputs_from_jsonable(
            obj["_provenance"].get("inputs", []))
        check = validate_recorded(recorded)
        if check.ok:
            return payload, check
        # Still drifted -- callback didn't fix it.
        raise StaleCacheError(path, check)

    # strict, or rebuild without a callback.
    raise StaleCacheError(path, check)


def load_and_register(
    path: Path,
    *,
    dep_key: str,
    extras: Optional[dict] = None,
    policy: str = "warn",
    rebuild_callback=None,
    inputs: Optional[list] = None,
):
    """Read + envelope-unwrap + drift-validate + build an
    :class:`InputSpec` describing ``path`` itself, in one call.

    This is the writer-side counterpart to :func:`load_validated_json`:
    where ``load_validated_json`` answers "is this cache I'm reading
    still current?", ``load_and_register`` answers BOTH that AND
    "what InputSpec should I record so my OWN output's provenance
    points back at this file?".  Linking the two operations makes it
    structurally hard to forget one of them -- you can't read a file
    without registering it as a dependency, and you can't register a
    dependency you didn't read.

    Args:
        path: JSON file to read.  May or may not carry a
            ``{"_provenance": ..., "result": ...}`` envelope; either
            way the payload is unwrapped before return.
        dep_key: Writer-chosen short name for this dependency in the
            caller's output provenance (see :class:`InputSpec` for the
            naming convention).
        extras: Free-form discriminators (slot, layer, K-range, ...);
            advisory only -- not used in equality / drift comparisons.
            Recorded on the returned InputSpec.
        policy: Drift handling for the loaded file's own recorded
            inputs.  Forwarded to :func:`load_validated_json`; see
            that function's docstring for the full menu (``"strict"``,
            ``"warn"``, ``"rebuild"``, ``"off"``).
        rebuild_callback: Forwarded to :func:`load_validated_json`
            under ``policy="rebuild"``.
        inputs: Optional list to which the freshly-built InputSpec is
            appended *in place*.  Lets a caller accumulate dependencies
            during a multi-file read without managing an explicit
            collection variable::

                inputs: list[InputSpec] = []
                scores, _, _ = load_and_register(
                    p, dep_key="scores_axisA", inputs=inputs)
                ...
                save_json(out_path, result, inputs=inputs)

    Returns:
        ``(payload, spec, check)``:

        * ``payload`` is the unwrapped result.
        * ``spec`` is an :class:`InputSpec` for ``path`` (caller's
          dependency record).
        * ``check`` is the :class:`ProvenanceCheck` from validating
          the loaded file's *own* recorded inputs -- ``None`` for
          legacy bare files or under ``policy="off"``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.  (We could
            return ``(None, None, None)`` instead, but the caller
            almost always wants to know -- explicit pre-check is
            cheap, and the alternative invites silent skips.)
        StaleCacheError: Under ``policy="strict"`` (or ``"rebuild"``
            without a working callback) when the loaded file's own
            recorded inputs disagree with current state.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"load_and_register: {p} does not exist (dep_key={dep_key!r})"
        )
    payload, check = load_validated_json(
        p, policy=policy, rebuild_callback=rebuild_callback,
    )
    spec = current_file_input(dep_key=dep_key, path=p, extras=extras)
    if inputs is not None:
        inputs.append(spec)
    return payload, spec, check


def _parse_npz_meta(npz, meta_key: str) -> dict:
    """Extract and JSON-parse the meta blob from an open NpzFile.

    ``np.savez`` stringifies non-array values: a Python dict becomes a
    0-dim object array of a JSON string when the producer does
    ``np.savez(..., meta=json.dumps(meta_dict))`` (the convention used
    by ``winner_decomposition.py``).  Tolerate three storage shapes
    so the helper isn't brittle to future changes:

    1. ``data[meta_key]`` -> 0-dim ndarray wrapping a ``str`` (the
       canonical ``json.dumps(...)`` round-trip case).
    2. ``data[meta_key]`` -> 0-dim ndarray wrapping a ``dict`` (when a
       producer does ``np.savez(..., meta=meta_dict)`` with
       ``allow_pickle=True``).
    3. ``meta_key`` not in ``data.files`` -> return ``{}``.

    Returns ``{}`` rather than raising on a missing/unparseable meta
    block so legacy npz caches (no inputs recorded) keep loading
    cleanly under ``policy="warn"`` -- they just won't drift-check.
    """
    if meta_key not in npz.files:
        return {}
    raw = npz[meta_key]
    if hasattr(raw, "item"):
        try:
            raw = raw.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def load_and_register_npz(
    path: Path,
    *,
    dep_key: str,
    extras: Optional[dict] = None,
    policy: str = "warn",
    rebuild_callback=None,
    inputs: Optional[list] = None,
    meta_key: str = "meta",
    allow_pickle: bool = True,
):
    """``.npz`` analogue of :func:`load_and_register`.

    ``np.savez`` doesn't support the ``{"_provenance": ..., "result": ...}``
    JSON envelope that :func:`load_and_register` relies on, so producers
    serialise their dependency list into the npz's own ``meta`` array
    (a ``json.dumps(...)`` blob inside a 0-dim object array; see
    ``winner_decomposition.py`` for the canonical writer pattern).
    This helper:

    1. Opens ``path`` with ``np.load(allow_pickle=allow_pickle)``.
    2. Parses ``meta`` (key configurable via ``meta_key``) into a dict.
    3. If the meta dict has an ``_inputs`` list, runs the same drift
       validation as :func:`load_validated_json` (re-fingerprinting
       each recorded :class:`InputSpec` against current state) and
       applies ``policy`` (``"strict"``/``"warn"``/``"rebuild"``/``"off"``).
    4. Builds an :class:`InputSpec` describing ``path`` itself so the
       caller can declare the npz as a dependency of its own output.

    Same atomicity invariant as the JSON path: you can't read a cache
    without registering it, and you can't register a dependency you
    didn't read.

    Args:
        path: ``.npz`` file to read.
        dep_key: Writer-chosen dependency name for the caller's output
            provenance (see :class:`InputSpec` for the naming
            convention).
        extras: Free-form discriminators recorded on the returned
            InputSpec; advisory only.
        policy: Drift handling for the npz's recorded inputs.  Same
            menu as :func:`load_validated_json`.  ``"warn"`` is the
            default and prints to stderr without raising; legacy npz
            files lacking ``meta["_inputs"]`` short-circuit to
            ``check=None`` regardless of policy.
        rebuild_callback: Forwarded under ``policy="rebuild"``.
            Receives ``(path, check)``; should bring the npz back to
            currency before returning.  After it returns the npz is
            re-loaded and re-validated; persistent drift raises
            :class:`StaleCacheError`.
        inputs: Optional list to which the freshly-built InputSpec is
            appended in place.
        meta_key: Field name to read the meta blob from.  Defaults to
            ``"meta"`` (matches winner_decomposition.py).
        allow_pickle: Forwarded to ``np.load``.  Defaults to ``True``
            because the only npz consumer in this codebase
            (``plot_winner_decomposition.py``) needs pickled object
            arrays for its column_keys.  Set ``False`` for
            untrusted npz files.

    Returns:
        ``(npz, meta, spec, check)``:

        * ``npz`` is the open ``NpzFile``; the caller indexes it as
          usual (``npz["matrix"]`` etc.) and is responsible for
          closing it (``npz.close()``) or letting GC handle it.
        * ``meta`` is the parsed meta dict (empty ``{}`` if absent or
          unparseable).
        * ``spec`` is an :class:`InputSpec` for ``path``.
        * ``check`` is the :class:`ProvenanceCheck` from validating
          the npz's own recorded inputs, or ``None`` if the npz has
          no ``meta["_inputs"]`` block / under ``policy="off"``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        StaleCacheError: Under ``policy="strict"`` (or ``"rebuild"``
            without a working callback) when the npz's recorded inputs
            disagree with current state.
        ValueError: For unknown ``policy`` values.
    """
    if policy not in CACHE_POLICIES:
        raise ValueError(
            f"Unknown cache policy {policy!r}; "
            f"choose from {CACHE_POLICIES}.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"load_and_register_npz: {p} does not exist "
            f"(dep_key={dep_key!r})"
        )

    # Lazy import: keeps numpy off the import-time deps of the
    # provenance module (which is meant to be light-weight and
    # importable from anywhere; numpy is fine to require at the
    # call site since every npz user already depends on it).
    import numpy as np  # noqa: PLC0415

    npz = np.load(p, allow_pickle=allow_pickle)
    meta = _parse_npz_meta(npz, meta_key)

    check: Optional[ProvenanceCheck] = None
    if policy != "off":
        recorded_blobs = meta.get("_inputs") if isinstance(meta, dict) else None
        if recorded_blobs:
            recorded = inputs_from_jsonable(recorded_blobs)
            check = validate_recorded(recorded)
            if not check.ok:
                if policy == "warn":
                    import sys
                    sys.stderr.write(
                        f"[provenance] {p}: drift detected "
                        f"(policy=warn, returning npz anyway):\n"
                        f"{check.summary()}\n"
                    )
                elif policy == "rebuild" and rebuild_callback is not None:
                    npz.close()
                    rebuild_callback(p, check)
                    npz = np.load(p, allow_pickle=allow_pickle)
                    meta = _parse_npz_meta(npz, meta_key)
                    recorded_blobs = (meta.get("_inputs")
                                      if isinstance(meta, dict) else None)
                    if not recorded_blobs:
                        # Callback rebuilt without a meta block; treat
                        # as still-stale rather than silently passing.
                        npz.close()
                        raise StaleCacheError(p, check)
                    recorded = inputs_from_jsonable(recorded_blobs)
                    check = validate_recorded(recorded)
                    if not check.ok:
                        npz.close()
                        raise StaleCacheError(p, check)
                else:
                    # strict, or rebuild without a callback.
                    npz.close()
                    raise StaleCacheError(p, check)

    spec = current_file_input(dep_key=dep_key, path=p, extras=extras)
    if inputs is not None:
        inputs.append(spec)
    return npz, meta, spec, check


def inputs_from_jsonable(blobs: Iterable[dict]) -> list[InputSpec]:
    """Inverse of :func:`inputs_to_jsonable`.  Tolerates blobs missing
    the optional fields (``dataset_id``, ``last_modified_at``,
    ``extras``, ``member_paths``, ``content``) for forward and backward
    compatibility."""
    out: list[InputSpec] = []
    for b in blobs:
        mp = b.get("member_paths")
        if mp is not None:
            mp = list(mp)
        out.append(InputSpec(
            dep_key=b["dep_key"],
            path=b["path"],
            fingerprint=b["fingerprint"],
            kind=b.get("kind", "file"),
            dataset_id=b.get("dataset_id"),
            last_modified_at=b.get("last_modified_at"),
            extras=dict(b.get("extras") or {}),
            member_paths=mp,
            content=b.get("content"),
        ))
    return out
