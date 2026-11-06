"""Tests for assistant_axis.provenance.

Covers:
* Manifest read round-trip + schema validation.
* current_data_subtree_input fingerprint shape and stability.
* current_file_input fingerprint shape and reaction to mtime/size change.
* validate_inputs cases: ok, drift, missing_recorded, missing_current.
* JSON round-trip (inputs_to_jsonable / inputs_from_jsonable).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from assistant_axis import provenance as prov


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(data_dir: Path, *, dataset_id: str = "test-ds",
                    subtrees: dict[str, dict] | None = None) -> None:
    if subtrees is None:
        subtrees = {
            "traits/vectors": {
                "kind": "raw", "recurse": False,
                "count": 3, "total_bytes": 123,
                "newest_mtime": "2026-05-08T00:00:00+00:00",
                "summary_sha256": "abcd" * 16,
            },
        }
    payload = {
        "dataset_id": dataset_id,
        "schema_version": "1.0",
        "fingerprint_kind": "mtime_size_v1",
        "manifest_generated_at": "2026-05-08T01:00:00+00:00",
        "subtree_summaries": subtrees,
    }
    (data_dir / "MANIFEST.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# read_manifest
# ---------------------------------------------------------------------------

def test_read_manifest_round_trip(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    m = prov.read_manifest(tmp_path)
    assert m.dataset_id == "test-ds"
    assert m.schema_version == "1.0"
    assert m.fingerprint_kind == "mtime_size_v1"
    assert "traits/vectors" in m.subtree_summaries
    sub = m.subtree("traits/vectors")
    assert sub.kind == "raw"
    assert sub.count == 3
    assert sub.summary_sha256.startswith("abcd")


def test_read_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prov.read_manifest(tmp_path)


def test_read_manifest_unknown_schema_raises(tmp_path: Path) -> None:
    payload = {
        "dataset_id": "ds",
        "schema_version": "9.9",
        "fingerprint_kind": "mtime_size_v1",
        "manifest_generated_at": "2026-05-08T01:00:00+00:00",
        "subtree_summaries": {},
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version"):
        prov.read_manifest(tmp_path)


def test_subtree_lookup_unknown_raises(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    m = prov.read_manifest(tmp_path)
    with pytest.raises(KeyError, match="missing"):
        m.subtree("missing")


# ---------------------------------------------------------------------------
# current_data_subtree_input
# ---------------------------------------------------------------------------

def test_current_data_subtree_input_fingerprint_shape(tmp_path: Path) -> None:
    _write_manifest(tmp_path, dataset_id="qwen-3-32b Roger 8slot")
    spec = prov.current_data_subtree_input(
        tmp_path, "traits/vectors", dep_key="my_dep",
    )
    assert spec.dep_key == "my_dep"
    assert spec.kind == "subtree"
    assert spec.dataset_id == "qwen-3-32b Roger 8slot"
    # Versioned fingerprint: "v1:{dataset_id}@{sha[:12]}"
    assert spec.fingerprint == "v1:qwen-3-32b Roger 8slot@abcdabcdabcd"
    assert spec.path.endswith("/traits/vectors")
    assert spec.last_modified_at == "2026-05-08T00:00:00+00:00"


def test_current_data_subtree_input_passes_extras(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    spec = prov.current_data_subtree_input(
        tmp_path, "traits/vectors", dep_key="r",
        extras={"slot": "6", "layer": "25"},
    )
    assert spec.extras == {"slot": "6", "layer": "25"}


def test_current_data_subtree_input_stability(tmp_path: Path) -> None:
    """Two calls with the same manifest produce byte-identical InputSpecs."""
    _write_manifest(tmp_path)
    s1 = prov.current_data_subtree_input(tmp_path, "traits/vectors", dep_key="r")
    s2 = prov.current_data_subtree_input(tmp_path, "traits/vectors", dep_key="r")
    assert s1 == s2


def test_current_data_subtree_input_changes_when_summary_changes(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path)
    s1 = prov.current_data_subtree_input(tmp_path, "traits/vectors", dep_key="r")
    # Re-write the manifest with a different summary_sha256.
    _write_manifest(tmp_path, subtrees={
        "traits/vectors": {
            "kind": "raw", "recurse": False,
            "count": 3, "total_bytes": 123,
            "newest_mtime": "2026-05-08T00:00:00+00:00",
            "summary_sha256": "ffff" * 16,
        },
    })
    s2 = prov.current_data_subtree_input(tmp_path, "traits/vectors", dep_key="r")
    assert s1.fingerprint != s2.fingerprint


# ---------------------------------------------------------------------------
# current_file_input
# ---------------------------------------------------------------------------

def test_current_file_input_fingerprint_shape(tmp_path: Path) -> None:
    f = tmp_path / "axis_spec.json"
    f.write_text('{"hello": "world"}\n')
    spec = prov.current_file_input("axis_spec", f)
    assert spec.dep_key == "axis_spec"
    assert spec.kind == "file"
    assert spec.dataset_id is None
    # Versioned fingerprint: "v1:{iso}@{size}".
    assert spec.fingerprint.startswith("v1:")
    rest = spec.fingerprint.removeprefix("v1:")
    iso, sz = rest.rsplit("@", 1)
    assert int(sz) == f.stat().st_size
    # ISO has timezone info.
    assert iso.endswith("+00:00")


def test_current_file_input_stability_with_no_changes(tmp_path: Path) -> None:
    f = tmp_path / "x.json"
    f.write_text("hello")
    s1 = prov.current_file_input("r", f)
    # Don't touch the file.  Calling again should produce the same fingerprint.
    s2 = prov.current_file_input("r", f)
    assert s1.fingerprint == s2.fingerprint


def test_current_file_input_changes_on_size_change(tmp_path: Path) -> None:
    f = tmp_path / "x.json"
    f.write_text("hello")
    s1 = prov.current_file_input("r", f)
    f.write_text("hello world")  # different size
    s2 = prov.current_file_input("r", f)
    assert s1.fingerprint != s2.fingerprint


def test_current_file_input_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prov.current_file_input("r", tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# current_files_input (composite multi-file)
# ---------------------------------------------------------------------------

def test_current_files_input_fingerprint_shape(tmp_path: Path) -> None:
    a = tmp_path / "a.json"; a.write_text("aa")
    b = tmp_path / "b.json"; b.write_text("bbbb")
    spec = prov.current_files_input("score_caches", [a, b])
    assert spec.kind == "multi"
    assert spec.dep_key == "score_caches"
    assert spec.fingerprint.startswith("v1:multi:")
    # Path is the common prefix when there's > 1 file.
    assert spec.path == prov._to_repo_relative(tmp_path)


def test_current_files_input_single_file_uses_full_path(tmp_path: Path) -> None:
    a = tmp_path / "a.json"; a.write_text("aa")
    spec = prov.current_files_input("d", [a])
    assert spec.path == prov._to_repo_relative(a)


def test_current_files_input_skips_missing_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.json"; a.write_text("aa")
    spec = prov.current_files_input(
        "d", [a, tmp_path / "missing.json"])
    # One existing file should produce a valid (non-sentinel) fingerprint.
    assert spec.fingerprint.startswith("v1:multi:")
    assert spec.fingerprint != "v1:multi:empty"


def test_current_files_input_all_missing_returns_sentinel(tmp_path: Path) -> None:
    spec = prov.current_files_input(
        "d", [tmp_path / "x.json", tmp_path / "y.json"])
    assert spec.fingerprint == "v1:multi:empty"
    assert spec.path == "(empty)"
    assert spec.last_modified_at is None


def test_current_files_input_stability(tmp_path: Path) -> None:
    a = tmp_path / "a.json"; a.write_text("aa")
    b = tmp_path / "b.json"; b.write_text("bbbb")
    s1 = prov.current_files_input("d", [a, b])
    s2 = prov.current_files_input("d", [a, b])
    # Order-independent: same files in different order produce same fp.
    s3 = prov.current_files_input("d", [b, a])
    assert s1.fingerprint == s2.fingerprint == s3.fingerprint


def test_current_files_input_drift_on_one_file_change(tmp_path: Path) -> None:
    a = tmp_path / "a.json"; a.write_text("aa")
    b = tmp_path / "b.json"; b.write_text("bbbb")
    s1 = prov.current_files_input("d", [a, b])
    b.write_text("bbbbXXX")  # one file's size changes
    s2 = prov.current_files_input("d", [a, b])
    assert s1.fingerprint != s2.fingerprint


def test_current_files_input_drift_on_set_membership(tmp_path: Path) -> None:
    a = tmp_path / "a.json"; a.write_text("aa")
    b = tmp_path / "b.json"; b.write_text("bbbb")
    c = tmp_path / "c.json"; c.write_text("ccc")
    s1 = prov.current_files_input("d", [a, b])
    s2 = prov.current_files_input("d", [a, b, c])
    assert s1.fingerprint != s2.fingerprint


def test_current_files_input_dedupes_symlinks_to_same_target(tmp_path: Path) -> None:
    """Regression test (Phase 6a): when two symlinks resolve to the same
    target file, the composite fingerprint must round-trip through
    ``current_for_recorded`` cleanly.  Previously the write-time
    fingerprint included duplicate triples (two paths, same resolved
    target) while ``member_paths`` was deduped via ``set()``, so
    re-validation always reported drift on multi inputs that contained
    overlapping symlinks (e.g., ``roles/vectors/default.pt`` and
    ``traits/vectors/default.pt`` both pointing at a shared
    ``default/vectors/default.pt``).
    """
    target = tmp_path / "target.pt"; target.write_text("payload")
    link_a = tmp_path / "link_a.pt"; link_a.symlink_to(target)
    link_b = tmp_path / "link_b.pt"; link_b.symlink_to(target)
    s1 = prov.current_files_input("d", [link_a, link_b, target])
    s2 = prov.current_files_input("d", [target])
    assert s1.fingerprint == s2.fingerprint, (
        "Three paths that resolve to one target must produce the same "
        "fingerprint as one direct reference."
    )
    # member_paths should also be the deduped set of resolved rel paths.
    assert s1.member_paths == s2.member_paths
    # Round-trip through current_for_recorded yields the same fingerprint.
    s3 = prov.current_for_recorded(s1)
    assert s3.fingerprint == s1.fingerprint


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------

def _spec(dep_key: str, fp: str = "fp1") -> prov.InputSpec:
    return prov.InputSpec(
        dep_key=dep_key, path=f"/p/{dep_key}", fingerprint=fp, kind="file")


def test_validate_inputs_all_ok() -> None:
    rec = [_spec("a"), _spec("b")]
    cur = [_spec("a"), _spec("b")]
    chk = prov.validate_inputs(rec, cur)
    assert chk.ok is True
    assert all(s.status == "ok" for s in chk.statuses)


def test_validate_inputs_drift_detected() -> None:
    rec = [_spec("a", "v1"), _spec("b")]
    cur = [_spec("a", "v2"), _spec("b")]
    chk = prov.validate_inputs(rec, cur)
    assert chk.ok is False
    drifts = chk.drifted()
    assert [s.dep_key for s in drifts] == ["a"]
    assert "v1 -> v2" in drifts[0].detail


def test_validate_inputs_missing_current() -> None:
    rec = [_spec("a"), _spec("b")]
    cur = [_spec("a")]
    chk = prov.validate_inputs(rec, cur)
    assert chk.ok is False
    miss = chk.missing()
    assert [s.dep_key for s in miss] == ["b"]
    assert miss[0].status == "missing_current"


def test_validate_inputs_missing_recorded() -> None:
    rec = [_spec("a")]
    cur = [_spec("a"), _spec("b")]
    chk = prov.validate_inputs(rec, cur)
    assert chk.ok is False
    miss = chk.missing()
    assert [s.dep_key for s in miss] == ["b"]
    assert miss[0].status == "missing_recorded"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_inputs_jsonable_round_trip(tmp_path: Path) -> None:
    s = prov.InputSpec(
        dep_key="r", path="some/p", fingerprint="v1:fp",
        kind="subtree", dataset_id="ds",
        last_modified_at="2026-05-08T01:00:00+00:00",
        extras={"k": "v"},
    )
    blobs = prov.inputs_to_jsonable([s])
    again = prov.inputs_from_jsonable(blobs)
    assert again == [s]


def test_inputs_from_jsonable_tolerates_missing_optionals() -> None:
    blobs = [{
        "dep_key": "r", "path": "p", "fingerprint": "v1:fp", "kind": "file",
        # no dataset_id, last_modified_at, extras, member_paths
    }]
    specs = prov.inputs_from_jsonable(blobs)
    assert len(specs) == 1
    s = specs[0]
    assert s.dataset_id is None
    assert s.last_modified_at is None
    assert s.extras == {}
    assert s.member_paths is None


# ---------------------------------------------------------------------------
# member_paths round-trip and population
# ---------------------------------------------------------------------------

def test_current_files_input_records_member_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.json"; a.write_text("aa")
    b = tmp_path / "b.json"; b.write_text("bbbb")
    spec = prov.current_files_input("d", [a, b])
    # Both paths recorded, sorted, deduped.
    assert spec.member_paths is not None
    assert len(spec.member_paths) == 2
    assert spec.member_paths == sorted(spec.member_paths)


def test_current_files_input_records_missing_paths_too(tmp_path: Path) -> None:
    """member_paths captures the *expected* set, not just the existing
    one -- so a previously-missing file appearing later registers as
    drift on the next read."""
    a = tmp_path / "a.json"; a.write_text("aa")
    missing = tmp_path / "missing.json"
    spec = prov.current_files_input("d", [a, missing])
    assert spec.member_paths is not None
    assert len(spec.member_paths) == 2
    # The missing path is in member_paths even though it didn't
    # contribute to the fingerprint.
    rels = set(spec.member_paths)
    assert prov._to_repo_relative(missing) in rels


def test_member_paths_round_trip_through_json() -> None:
    s = prov.InputSpec(
        dep_key="r", path="some/p", fingerprint="v1:multi:abc",
        kind="multi", member_paths=["a/x.json", "b/y.json"])
    blobs = prov.inputs_to_jsonable([s])
    assert "member_paths" in blobs[0]
    again = prov.inputs_from_jsonable(blobs)
    assert again == [s]


def test_member_paths_omitted_for_non_multi_in_jsonable() -> None:
    s = prov.InputSpec(
        dep_key="r", path="p", fingerprint="v1:fp", kind="file")
    blobs = prov.inputs_to_jsonable([s])
    # Tidy: don't pollute file/subtree records with a None field.
    assert "member_paths" not in blobs[0]


# ---------------------------------------------------------------------------
# current_for_recorded
# ---------------------------------------------------------------------------

def test_current_for_recorded_file(tmp_path: Path) -> None:
    f = tmp_path / "a.json"
    f.write_text("hello")
    s1 = prov.current_file_input("r", f)
    s2 = prov.current_for_recorded(s1)
    assert s1.fingerprint == s2.fingerprint
    assert s2.kind == "file"


def test_current_for_recorded_file_detects_drift(tmp_path: Path) -> None:
    f = tmp_path / "a.json"
    f.write_text("hello")
    s1 = prov.current_file_input("r", f)
    time.sleep(1.1)
    f.write_text("hello world")
    s2 = prov.current_for_recorded(s1)
    assert s1.fingerprint != s2.fingerprint


def test_current_for_recorded_multi(tmp_path: Path) -> None:
    a = tmp_path / "a.json"; a.write_text("aa")
    b = tmp_path / "b.json"; b.write_text("bbbb")
    s1 = prov.current_files_input("r", [a, b])
    s2 = prov.current_for_recorded(s1)
    assert s1.fingerprint == s2.fingerprint


def test_current_for_recorded_multi_without_member_paths_raises() -> None:
    s = prov.InputSpec(
        dep_key="r", path="x", fingerprint="v1:multi:legacy",
        kind="multi", member_paths=None)
    with pytest.raises(prov._UnverifiableInput):
        prov.current_for_recorded(s)


def test_current_for_recorded_unknown_kind_raises() -> None:
    s = prov.InputSpec(
        dep_key="r", path="x", fingerprint="v1:fp", kind="bogus")
    with pytest.raises(ValueError, match="Unknown InputSpec.kind"):
        prov.current_for_recorded(s)


# ---------------------------------------------------------------------------
# load_validated_json
# ---------------------------------------------------------------------------

def _envelope(payload, inputs):
    return {
        "result": payload,
        "_provenance": {
            "schema_version": prov.PROVENANCE_SCHEMA_VERSION,
            "inputs": prov.inputs_to_jsonable(inputs),
        },
    }


def test_load_validated_json_no_envelope_returns_raw(tmp_path: Path) -> None:
    f = tmp_path / "legacy.json"
    f.write_text(json.dumps({"hello": "world"}))
    payload, check = prov.load_validated_json(f, policy="strict")
    assert payload == {"hello": "world"}
    assert check is None


def test_load_validated_json_off_skips_validation(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    src.write_text("xx")
    inputs = [prov.current_file_input("src", src)]
    out = tmp_path / "cache.json"
    out.write_text(json.dumps(_envelope({"data": 1}, inputs)))
    # Mutate src so a strict load would fail.
    time.sleep(1.1)
    src.write_text("yyyy")
    payload, check = prov.load_validated_json(out, policy="off")
    assert payload == {"data": 1}
    assert check is None  # no validation performed


def test_load_validated_json_ok(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    src.write_text("hello")
    inputs = [prov.current_file_input("src", src)]
    out = tmp_path / "cache.json"
    out.write_text(json.dumps(_envelope({"x": 1}, inputs)))
    payload, check = prov.load_validated_json(out, policy="strict")
    assert payload == {"x": 1}
    assert check is not None
    assert check.ok


def test_load_validated_json_strict_raises_on_drift(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    src.write_text("hello")
    inputs = [prov.current_file_input("src", src)]
    out = tmp_path / "cache.json"
    out.write_text(json.dumps(_envelope({"x": 1}, inputs)))
    time.sleep(1.1)
    src.write_text("hello world")
    with pytest.raises(prov.StaleCacheError) as ei:
        prov.load_validated_json(out, policy="strict")
    assert ei.value.path == out
    assert not ei.value.check.ok


def test_load_validated_json_warn_returns_payload_with_drift(
    tmp_path: Path, capsys
) -> None:
    src = tmp_path / "src.json"
    src.write_text("hello")
    inputs = [prov.current_file_input("src", src)]
    out = tmp_path / "cache.json"
    out.write_text(json.dumps(_envelope({"x": 1}, inputs)))
    time.sleep(1.1)
    src.write_text("hello world")
    payload, check = prov.load_validated_json(out, policy="warn")
    assert payload == {"x": 1}
    assert check is not None and not check.ok
    err = capsys.readouterr().err
    assert "drift" in err.lower()


def test_load_validated_json_strict_missing_current(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    src.write_text("hello")
    inputs = [prov.current_file_input("src", src)]
    out = tmp_path / "cache.json"
    out.write_text(json.dumps(_envelope({"x": 1}, inputs)))
    src.unlink()  # vanished
    with pytest.raises(prov.StaleCacheError) as ei:
        prov.load_validated_json(out, policy="strict")
    statuses = {s.dep_key: s.status for s in ei.value.check.statuses}
    assert statuses["src"] == "missing_current"


def test_load_validated_json_unverifiable_multi(tmp_path: Path) -> None:
    """Legacy multi spec (no member_paths) classifies as unverifiable
    and triggers strict failure."""
    legacy_multi = prov.InputSpec(
        dep_key="caches", path=str(tmp_path), fingerprint="v1:multi:legacy",
        kind="multi", member_paths=None)
    out = tmp_path / "cache.json"
    out.write_text(json.dumps(_envelope({"x": 1}, [legacy_multi])))
    with pytest.raises(prov.StaleCacheError) as ei:
        prov.load_validated_json(out, policy="strict")
    statuses = {s.dep_key: s.status for s in ei.value.check.statuses}
    assert statuses["caches"] == "unverifiable"


def test_load_validated_json_rebuild_callback(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    src.write_text("hello")
    inputs_v1 = [prov.current_file_input("src", src)]
    out = tmp_path / "cache.json"
    out.write_text(json.dumps(_envelope({"x": 1}, inputs_v1)))
    time.sleep(1.1)
    src.write_text("hello world")  # cause drift

    callbacks: list = []
    def rebuild(path, check):
        callbacks.append((path, check))
        # Simulate a rebuild: rewrite cache with current inputs.
        new_inputs = [prov.current_file_input("src", src)]
        path.write_text(json.dumps(_envelope({"x": 2}, new_inputs)))

    payload, check = prov.load_validated_json(
        out, policy="rebuild", rebuild_callback=rebuild,
    )
    assert payload == {"x": 2}
    assert check.ok
    assert len(callbacks) == 1


def test_load_validated_json_rebuild_without_callback_raises(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.json"
    src.write_text("hello")
    inputs = [prov.current_file_input("src", src)]
    out = tmp_path / "cache.json"
    out.write_text(json.dumps(_envelope({"x": 1}, inputs)))
    time.sleep(1.1)
    src.write_text("hello world")
    with pytest.raises(prov.StaleCacheError):
        prov.load_validated_json(out, policy="rebuild")


def test_load_validated_json_unknown_policy_raises(tmp_path: Path) -> None:
    f = tmp_path / "cache.json"
    f.write_text("{}")
    with pytest.raises(ValueError, match="Unknown cache policy"):
        prov.load_validated_json(f, policy="bogus")


def test_load_validated_json_subtree_input(tmp_path: Path) -> None:
    """Round-trip a subtree InputSpec through load_validated_json."""
    # Build a fake dataset root with a manifest.
    data_dir = tmp_path / "ds"
    data_dir.mkdir()
    sub = data_dir / "traits" / "vectors"
    sub.mkdir(parents=True)
    _write_manifest(data_dir)
    inputs = [prov.current_data_subtree_input(
        data_dir, "traits/vectors", dep_key="traits")]
    out = tmp_path / "cache.json"
    out.write_text(json.dumps(_envelope({"x": 1}, inputs)))
    payload, check = prov.load_validated_json(out, policy="strict")
    assert payload == {"x": 1}
    assert check.ok


# ---------------------------------------------------------------------------
# Corpus content hashes (AGENT_NOTES housekeeping item 3, 2026-10-10)
# ---------------------------------------------------------------------------

_FIELDS = ("positive_label", "description", "instruction[*].pos")


def _corpus(root: Path) -> list:
    """Two traits and a role; returns every JSON path (the metadata set)."""
    t = root / "traits" / "instructions"
    r = root / "roles" / "instructions"
    t.mkdir(parents=True)
    r.mkdir(parents=True)
    (t / "lazy.json").write_text(json.dumps({
        "positive_label": "lazy", "negative_label": "industrious",
        "description": "This means doing as little as possible.",
        "instruction": [{"pos": "Rest.", "neg": "Work."}],
        "questions": ["q1"], "arrangement": {"kind": "pair", "members": ["industrious", "lazy"]}}))
    (t / "industrious.json").write_text(json.dumps({
        "positive_label": "industrious", "negative_label": "lazy",
        "description": "This means working hard.",
        "instruction": [{"pos": "Work.", "neg": "Rest."}]}))
    (r / "mechanic.json").write_text(json.dumps({
        "description": "A mechanic fixes engines.", "instruction": [{"pos": "Fix."}]}))
    return sorted(t.glob("*.json")) + sorted(r.glob("*.json"))


def _rewrite(path: Path, **changes) -> None:
    """Edit a JSON file and move its mtime forward, so the metadata
    fingerprint drifts even within the same second."""
    doc = json.loads(path.read_text())
    doc.update(changes)
    path.write_text(json.dumps(doc))
    st = os.stat(path)
    os.utime(path, (st.st_atime + 10, st.st_mtime + 10))


def _corpus_spec(root: Path, paths: list, entities=None) -> prov.InputSpec:
    entities = entities or [("traits", "lazy"), ("traits", "industrious"), ("roles", "mechanic")]
    return prov.current_corpus_files_input(
        "corpus_instructions", paths, root=root, entities=entities, fields=_FIELDS)


def test_corpus_content_keys_entities_by_id_and_hashes_fields(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    spec = _corpus_spec(tmp_path, paths)
    c = spec.content
    assert c["version"] == prov.CORPUS_CONTENT_VERSION and c["fields"] == list(_FIELDS)
    assert list(c["entities"]) == ["industrious|T", "lazy|T", "mechanic|R"]
    assert all(isinstance(h, str) and len(h) == 16 for h in c["entities"].values())
    assert spec.kind == "multi" and len(spec.member_paths) == 3
    assert prov.validate_recorded([spec]).statuses[0].status == "ok"


def test_metadata_only_edit_is_equivalent_content(tmp_path: Path) -> None:
    """The case the change is for: tags, arrangement, questions, a neg
    instruction, generator.  Metadata drifts; the read fields do not."""
    paths = _corpus(tmp_path)
    spec = _corpus_spec(tmp_path, paths)
    lazy = tmp_path / "traits" / "instructions" / "lazy.json"
    _rewrite(lazy, tags=["x"], questions=["q1", "q2"], generator={"model": "m"},
             instruction=[{"pos": "Rest.", "neg": "Toil hard."}])
    chk = prov.validate_recorded([spec])
    st = chk.statuses[0]
    assert st.status == "equivalent_content" and chk.ok
    assert "3 entities' positive_label/description/instruction[*].pos are unchanged" in st.detail
    assert chk.equivalent() == [st]


def test_a_read_field_edit_is_drift_naming_the_entity(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    spec = _corpus_spec(tmp_path, paths)
    _rewrite(tmp_path / "traits" / "instructions" / "lazy.json",
             description="This means avoiding effort.")
    chk = prov.validate_recorded([spec])
    st = chk.statuses[0]
    assert st.status == "drift" and not chk.ok
    assert "1 of 3 entities changed in positive_label/description/instruction[*].pos: lazy|T" in st.detail
    # the current block is attached, so a report can show the new hashes
    assert st.current.content["entities"]["lazy|T"] != spec.content["entities"]["lazy|T"]


def test_label_and_pos_edits_count_and_a_deleted_file_is_drift(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    spec = _corpus_spec(tmp_path, paths)
    _rewrite(tmp_path / "traits" / "instructions" / "industrious.json",
             positive_label="Industrious")
    _rewrite(tmp_path / "roles" / "instructions" / "mechanic.json",
             instruction=[{"pos": "Repair."}])
    st = prov.validate_recorded([spec]).statuses[0]
    assert st.status == "drift" and "2 of 3 entities changed" in st.detail
    (tmp_path / "traits" / "instructions" / "lazy.json").unlink()
    st = prov.validate_recorded([spec]).statuses[0]
    assert st.status == "drift" and "(1 no longer have a file)" in st.detail


def test_an_entity_the_consumer_does_not_read_is_not_hashed(tmp_path: Path) -> None:
    """The entity set is what the consumer opens (load_corpus: a vector and a
    file); a corpus file with no vector can change freely."""
    paths = _corpus(tmp_path)
    spec = _corpus_spec(tmp_path, paths, entities=[("traits", "lazy")])
    _rewrite(tmp_path / "traits" / "instructions" / "industrious.json",
             description="This means grinding.")
    st = prov.validate_recorded([spec]).statuses[0]
    assert st.status == "equivalent_content"


def test_a_record_without_content_validates_as_before(tmp_path: Path) -> None:
    """Records written before 2026-10-10 carry no content block: metadata
    drift stays drift, so nothing is reclassified by the change itself."""
    paths = _corpus(tmp_path)
    spec = prov.current_files_input("corpus_instructions", paths)
    assert spec.content is None
    _rewrite(tmp_path / "traits" / "instructions" / "lazy.json", tags=["x"])
    assert prov.validate_recorded([spec]).statuses[0].status == "drift"


def test_content_round_trips_through_json_and_envelopes(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    spec = _corpus_spec(tmp_path, paths)
    blobs = prov.inputs_to_jsonable([spec])
    assert "content" in blobs[0]
    assert prov.inputs_from_jsonable(json.loads(json.dumps(blobs))) == [spec]
    assert "content" not in prov.inputs_to_jsonable([_spec("f")])[0]
    _rewrite(tmp_path / "roles" / "instructions" / "mechanic.json", tags=["garage"])
    out = tmp_path / "cache.json"
    out.write_text(json.dumps(_envelope({"x": 1}, [spec])))
    payload, check = prov.load_validated_json(out, policy="strict")
    assert payload == {"x": 1} and check.ok


def test_an_unknown_content_version_is_drift(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    spec = _corpus_spec(tmp_path, paths)
    import dataclasses
    spec = dataclasses.replace(spec, content={**spec.content, "version": "corpus_fields_v99"})
    _rewrite(tmp_path / "traits" / "instructions" / "lazy.json", tags=["x"])
    st = prov.validate_recorded([spec]).statuses[0]
    assert st.status == "drift" and "cannot compare" in st.detail
