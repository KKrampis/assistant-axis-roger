#!/usr/bin/env python3
"""(Re)generate ``MANIFEST.json`` at the root of one or more datasets.

The manifest records subtree-granular fingerprints of all post-pipeline and
pipeline-produced inputs that downstream caches and plots depend on.  Each
subtree's fingerprint is a SHA-256 over the deterministic sorted JSON list
``[(rel_path, mtime_ns, size), ...]``.  Cheap (mtime+size only, ~tens of
ms per subtree on local disk).  See
``audits/post_pipeline_derived_layout.md`` and the provenance plan in
``.cursor/plans/`` for the broader design.

Output schema (``MANIFEST.json`` at the dataset root)::

    {
        "dataset_id":           "qwen-3-32b Roger 8slot",
        "schema_version":       "1.0",
        "manifest_generated_at": "2026-05-08T...",
        "fingerprint_kind":     "mtime_size_v1",
        "subtree_summaries": {
            "traits/vectors": {
                "kind":           "raw",      # or "derived"
                "recurse":        false,
                "count":          585,
                "total_bytes":    1234567890,
                "newest_mtime":   "2026-05-04T02:09:50.123456+00:00",
                "summary_sha256": "abcd1234..."
            },
            "combinations/vectors/derived/marginals/r_goal": { ... },
            "combinations/axis.pt": {"count": 1, ...},
            ...
        }
    }

Subtree discovery rules (applied to each top-level entity dir
``default/``, ``traits/``, ``roles/``, ``combinations/``):

* Each loose ``.pt``/``.json``/``.txt`` file at the entity-dir level becomes
  its own count=1 subtree (e.g. ``combinations/axis.pt``,
  ``combinations/axis_unfiltered.pt``).  Pipeline logs (``*.log``) and
  ``.DS_Store`` are filtered out.
* Subdirectories named ``vectors`` / ``vectors_*`` / ``vectors.*`` -> one
  flat (non-recursive) subtree each, ``kind="raw"``.
* Subdirectories named ``responses`` / ``responses_*`` -> one recursive
  subtree each, ``kind="raw"``.
* Subdirectories named ``scores`` / ``scores_*`` -> one recursive subtree
  each, ``kind="raw"``.
* Subdirectories named ``activations`` / ``activations_*`` -> one recursive
  subtree each, ``kind="raw"``.
* Special: ``combinations/vectors/derived/`` is enumerated by category:
  - ``marginals/{etype}/`` and ``legacy_centroid/{etype}/`` -> one
    flat subtree per etype, ``kind="derived"``.
  - ``aggregates/`` and ``axis/`` -> one flat subtree each,
    ``kind="derived"``.

CLI::

    uv run python tools/regenerate_dataset_manifest.py \\
        --dataset 'runpod_workspace/qwen/qwen-3-32b Roger 8slot' \\
        [--dataset 'runpod_workspace/qwen/qwen-3-32b Roger'] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCHEMA_VERSION = "1.0"
FINGERPRINT_KIND = "mtime_size_v1"

# Mtime precision floor.  Rsync (and most cross-host transports) round
# mtimes to whole seconds when copying.  In Roger's typical workflow
# (RunPod -> NFS -> rsync -> Mac) every file lands at second precision
# anyway, so storing nanos in the fingerprint is mostly wasted space
# AND lights up false-positive drift the rare time a file arrives via
# a sub-second-preserving path (scp -p, cp -p, cloud sync).  We floor
# to whole seconds at hash time so fingerprints are transport-stable
# by construction; no laxity needed at validate time.
MTIME_PRECISION_NS = 1_000_000_000  # 1 second

ENTITY_DIRS = ("default", "traits", "roles", "combinations")
SKIP_FILE_NAMES = {".DS_Store"}
SKIP_FILE_SUFFIXES = {".log"}
DERIVED_FLAT_CATEGORIES = ("aggregates", "axis")
DERIVED_NESTED_CATEGORIES = ("marginals", "legacy_centroid")


def _floor_mtime_ns(mtime_ns: int) -> int:
    """Truncate ``st_mtime_ns`` to whole-second precision.  See
    ``MTIME_PRECISION_NS`` docstring."""
    return (mtime_ns // MTIME_PRECISION_NS) * MTIME_PRECISION_NS


@dataclass
class SubtreeSpec:
    name: str            # display name, relative to dataset root
    paths: list[Path]    # one or more file/dir paths to summarize together
    recurse: bool        # for directory paths: walk recursively
    kind: str            # "raw" or "derived"


@dataclass
class SubtreeSummary:
    kind: str
    recurse: bool
    count: int
    total_bytes: int
    newest_mtime: str           # ISO-8601 UTC
    summary_sha256: str         # hex


def _iter_files(spec: SubtreeSpec):
    """Yield each ``Path`` covered by a spec, filtering SKIPped names/suffixes."""
    for p in spec.paths:
        if p.is_file():
            if _is_skipped(p):
                continue
            yield p
        elif p.is_dir():
            if spec.recurse:
                for child in p.rglob("*"):
                    if child.is_file() and not _is_skipped(child):
                        yield child
            else:
                for child in p.iterdir():
                    if child.is_file() and not _is_skipped(child):
                        yield child


def _is_skipped(p: Path) -> bool:
    if p.name in SKIP_FILE_NAMES:
        return True
    if p.suffix in SKIP_FILE_SUFFIXES:
        return True
    return False


# ---------------------------------------------------------------------------
# TODO(phase-7-deferred): Subtree content hashing
#
# Today's ``summary_sha256`` is metadata-only -- a hash over sorted
# ``[(rel_path, mtime_ns_floored, size), ...]`` triples.  That makes
# manifest regen <1 second for the full dataset and is rsync-stable by
# construction (mtime is floored to whole seconds, matching ``rsync -t``
# precision), but it cannot tell ``cp``-without-mtime-preserve from a
# real edit, cannot recognize a fresh ``git checkout`` of identical
# bytes, and cannot detect on-disk bit-rot.
#
# The full design rationale, cost analysis, decision criteria, and
# implementation plan live as the ``TODO(phase-7-deferred)`` block at
# the top of ``assistant_axis/provenance.py``.  This is the manifest-
# regen-side checklist for picking it up; both sites must move
# together.
#
#   1. Add ``content_sha256: str`` to ``SubtreeSummary`` below.
#   2. In ``_summarize``, stream each file with ``hashlib.sha256()``
#      (1 MB chunks) and accumulate ``(rel_path, file_sha)`` pairs.
#      The subtree's ``content_sha256`` = SHA-256 of the sorted JSON
#      list of those pairs.  Keep the existing metadata-triple
#      ``summary_sha256`` alongside for backwards compatibility and
#      for the cheap-validate path.
#   3. Add a ``--metadata-only`` CLI flag to bypass content hashing
#      when speed matters; otherwise content hashing is the default
#      once added.  Print per-subtree progress (``hashing X: NN MB
#      in NN s``) since regen will go from <1 s to ~30-60 s on
#      local SSD (see canonical TODO for full numbers).
#   4. Bump ``FINGERPRINT_KIND`` to ``"mtime_size_v1+content_v1"`` so
#      consumers (validate_recorded, audit tools) can dispatch on
#      what guarantees the manifest provides.
#   5. Old manifests without ``content_sha256`` must still load --
#      graceful fallback to metadata-only validation (see provenance
#      module's ``read_manifest`` path).
# ---------------------------------------------------------------------------


def _summarize(spec: SubtreeSpec, root: Path) -> SubtreeSummary | None:
    triples: list[tuple[str, int, int]] = []
    total_bytes = 0
    newest_mtime_ns = 0
    for fp in _iter_files(spec):
        # stat follows symlinks: combinations/vectors/default.pt -> default/vectors/default.pt
        st = os.stat(fp)
        rel = str(fp.relative_to(root)).replace(os.sep, "/")
        # Floor to whole seconds so fingerprints are rsync-stable
        # (see MTIME_PRECISION_NS docstring above).
        mtime_ns = _floor_mtime_ns(int(st.st_mtime_ns))
        triples.append((rel, mtime_ns, int(st.st_size)))
        total_bytes += int(st.st_size)
        if mtime_ns > newest_mtime_ns:
            newest_mtime_ns = mtime_ns
    if not triples:
        return None
    triples.sort()
    blob = json.dumps(triples, separators=(",", ":")).encode("utf-8")
    sha = hashlib.sha256(blob).hexdigest()
    # Whole-second ISO (no microseconds) so newest_mtime stays
    # human-readable and aligns with the floored mtime_ns above.
    newest_iso = _dt.datetime.fromtimestamp(
        newest_mtime_ns // MTIME_PRECISION_NS, tz=_dt.timezone.utc
    ).isoformat()
    return SubtreeSummary(
        kind=spec.kind,
        recurse=spec.recurse,
        count=len(triples),
        total_bytes=total_bytes,
        newest_mtime=newest_iso,
        summary_sha256=sha,
    )


def discover_subtrees(root: Path) -> list[SubtreeSpec]:
    specs: list[SubtreeSpec] = []
    for ent_name in ENTITY_DIRS:
        ent = root / ent_name
        if not ent.is_dir():
            continue
        # Loose files at the entity level (axis.pt, axis.4slot.pt, etc.)
        for f in sorted(ent.iterdir()):
            if f.is_file() and not _is_skipped(f):
                specs.append(SubtreeSpec(
                    name=f"{ent_name}/{f.name}", paths=[f],
                    recurse=False, kind="raw",
                ))
        # Subdirs that match recognized prefixes
        for d in sorted(ent.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if _matches_prefix(d.name, "vectors"):
                specs.append(SubtreeSpec(
                    name=f"{ent_name}/{d.name}", paths=[d],
                    recurse=False, kind="raw",
                ))
            elif _matches_prefix(d.name, "responses"):
                specs.append(SubtreeSpec(
                    name=f"{ent_name}/{d.name}", paths=[d],
                    recurse=True, kind="raw",
                ))
            elif _matches_prefix(d.name, "scores"):
                specs.append(SubtreeSpec(
                    name=f"{ent_name}/{d.name}", paths=[d],
                    recurse=True, kind="raw",
                ))
            elif _matches_prefix(d.name, "activations"):
                specs.append(SubtreeSpec(
                    name=f"{ent_name}/{d.name}", paths=[d],
                    recurse=True, kind="raw",
                ))
            # else: unknown subdir at the entity level; skipped.
        # Special-case: combinations/vectors/derived/
        if ent_name == "combinations":
            specs.extend(_derived_subtree_specs(root, ent / "vectors" / "derived"))
    return specs


def _matches_prefix(dirname: str, prefix: str) -> bool:
    """Match e.g. 'vectors', 'vectors_4slot', 'vectors_reduce3', 'vectors.240'."""
    if dirname == prefix:
        return True
    if dirname.startswith(prefix + "_"):
        return True
    if dirname.startswith(prefix + "."):
        return True
    return False


def _derived_subtree_specs(root: Path, derived: Path) -> list[SubtreeSpec]:
    """Enumerate one subtree per leaf category under combinations/vectors/derived/."""
    specs: list[SubtreeSpec] = []
    if not derived.is_dir():
        return specs
    for cat_dir in sorted(derived.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        rel_prefix = f"combinations/vectors/derived/{cat}"
        if cat in DERIVED_NESTED_CATEGORIES:
            for sub in sorted(cat_dir.iterdir()):
                if sub.is_dir():
                    specs.append(SubtreeSpec(
                        name=f"{rel_prefix}/{sub.name}",
                        paths=[sub], recurse=False, kind="derived",
                    ))
        elif cat in DERIVED_FLAT_CATEGORIES:
            specs.append(SubtreeSpec(
                name=rel_prefix,
                paths=[cat_dir], recurse=False, kind="derived",
            ))
        else:
            # Unknown future category: still record it (defensive).
            specs.append(SubtreeSpec(
                name=rel_prefix,
                paths=[cat_dir], recurse=True, kind="derived",
            ))
    return specs


def build_manifest(root: Path, *, dataset_id: str | None = None) -> dict:
    if dataset_id is None:
        dataset_id = root.name
    specs = discover_subtrees(root)
    summaries: dict[str, dict] = {}
    t0 = time.time()
    for spec in specs:
        summary = _summarize(spec, root)
        if summary is None:
            continue
        summaries[spec.name] = asdict(summary)
    elapsed_ms = int((time.time() - t0) * 1000)
    manifest = {
        "dataset_id": dataset_id,
        "schema_version": SCHEMA_VERSION,
        "fingerprint_kind": FINGERPRINT_KIND,
        "manifest_generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "elapsed_ms": elapsed_ms,
        "subtree_summaries": dict(sorted(summaries.items())),
    }
    return manifest


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True, type=Path,
                        help="Dataset root.  Pass once per dataset.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build manifests in memory but don't write to disk.")
    args = parser.parse_args()

    any_failed = False
    for ds in args.dataset:
        print(f"=== {ds} ===")
        if not ds.exists():
            print(f"  [SKIP] dataset does not exist")
            any_failed = True
            continue
        manifest = build_manifest(ds)
        n = len(manifest["subtree_summaries"])
        total_files = sum(s["count"] for s in manifest["subtree_summaries"].values())
        total_bytes = sum(s["total_bytes"] for s in manifest["subtree_summaries"].values())
        print(f"  subtrees: {n}")
        print(f"  files:    {total_files}")
        print(f"  bytes:    {total_bytes:,}  ({total_bytes / 1e9:.2f} GB)")
        print(f"  elapsed:  {manifest['elapsed_ms']} ms")
        for name, s in manifest["subtree_summaries"].items():
            print(f"    {s['kind']:>7s}  {name:<60s}  count={s['count']:>5d}  "
                  f"sha={s['summary_sha256'][:12]}  newest={s['newest_mtime']}")
        out = ds / "MANIFEST.json"
        if args.dry_run:
            print(f"  [DRY-RUN] would write {out}")
        else:
            _atomic_write(out, json.dumps(manifest, indent=2) + "\n")
            print(f"  wrote {out}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
