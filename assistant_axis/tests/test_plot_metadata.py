"""Tests for assistant_axis.plot_metadata.

Focused on Phase 3a additions:
* png_metadata accepts an ``inputs=`` list and embeds ``Inputs`` /
  ``Inputs SHA256`` chunks.
* json_metadata wraps a payload with a ``_provenance`` envelope that
  includes a stable ``inputs_sha256``.
"""
from __future__ import annotations

import hashlib
import json

from assistant_axis import plot_metadata as pm
from assistant_axis import provenance as prov


def _spec(dep_key: str = "r", fp: str = "fp1") -> prov.InputSpec:
    return prov.InputSpec(
        dep_key=dep_key, path=f"some/{dep_key}", fingerprint=fp, kind="file")


# ---------------------------------------------------------------------------
# png_metadata
# ---------------------------------------------------------------------------

def test_png_metadata_basic_no_inputs() -> None:
    md = pm.png_metadata(title="t", argv=[])
    assert md["Title"] == "t"
    assert "Inputs" not in md
    assert "Inputs SHA256" not in md


def test_png_metadata_inputs_embeds_inputs_and_sha() -> None:
    inputs = [_spec("a", "v1"), _spec("b", "v2")]
    md = pm.png_metadata(title="t", argv=[], inputs=inputs)
    assert "Inputs" in md
    assert "Inputs SHA256" in md
    parsed = json.loads(md["Inputs"])
    assert {x["dep_key"] for x in parsed} == {"a", "b"}
    # SHA matches the embedded body bytes.
    assert md["Inputs SHA256"] == hashlib.sha256(
        md["Inputs"].encode("utf-8")).hexdigest()


def test_png_metadata_inputs_sha_stable_across_dict_order() -> None:
    """Two identical input lists yield identical chunks (stable sort_keys)."""
    inputs1 = [_spec("a"), _spec("b")]
    inputs2 = [_spec("a"), _spec("b")]
    md1 = pm.png_metadata(title="t", argv=[], inputs=inputs1)
    md2 = pm.png_metadata(title="t", argv=[], inputs=inputs2)
    assert md1["Inputs"] == md2["Inputs"]
    assert md1["Inputs SHA256"] == md2["Inputs SHA256"]


def test_png_metadata_inputs_accepts_jsonable_dicts() -> None:
    """Phase 3 writers may pass already-jsonable dicts directly."""
    blobs = prov.inputs_to_jsonable([_spec("a")])
    md = pm.png_metadata(title="t", argv=[], inputs=blobs)
    assert "Inputs" in md
    assert "a" in md["Inputs"]


def test_png_metadata_empty_inputs_list_writes_empty_chunks() -> None:
    """``inputs=None`` means "writer not yet migrated" (no chunks).
    ``inputs=[]`` means "writer migrated, has no deps" (empty chunks).

    Auditors rely on chunk presence as the migration signal -- an empty
    chunk still counts.  See ``tools/audit_pngs.py`` (Phase 5).
    """
    md = pm.png_metadata(title="t", argv=[], inputs=[])
    assert md["Inputs"] == "[]"
    # SHA of literal "[]" -- hardcoded as a stability anchor.
    assert md["Inputs SHA256"] == hashlib.sha256(b"[]").hexdigest()


# ---------------------------------------------------------------------------
# json_metadata
# ---------------------------------------------------------------------------

def test_json_metadata_envelope_shape() -> None:
    payload = {"foo": [1, 2, 3]}
    out = pm.json_metadata(payload, argv=[])
    assert out["result"] == payload
    pv = out["_provenance"]
    assert pv["schema_version"] == "1.0"
    assert "cmd" in pv["produced_by"]
    assert "produced_at" in pv


def test_produced_at_is_utc_iso8601() -> None:
    """``produced_at`` must be an ISO-8601 UTC timestamp matching the
    convention used elsewhere in the provenance system (deferred
    rejudges, script equivalences, file-fingerprint mtimes, MANIFEST
    timestamps).  Older envelopes used a local-time format
    ``%Y-%m-%d %H:%M:%S %z``; this test guards against that
    regressing.
    """
    import datetime as _dt
    out = pm.json_metadata({"x": 1}, argv=[])
    produced_at = out["_provenance"]["produced_at"]
    parsed = _dt.datetime.fromisoformat(produced_at)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == _dt.timedelta(0), (
        f"produced_at={produced_at!r} is not UTC; offset="
        f"{parsed.utcoffset()}"
    )
    assert "T" in produced_at, (
        f"produced_at={produced_at!r} is not ISO-8601 (missing 'T')"
    )


def test_json_metadata_with_inputs_round_trip() -> None:
    inputs = [_spec("a", "v1"), _spec("b", "v2")]
    out = pm.json_metadata({"x": 1}, argv=[], inputs=inputs)
    pv = out["_provenance"]
    assert "inputs" in pv
    assert "inputs_sha256" in pv
    blobs = pv["inputs"]
    assert {b["dep_key"] for b in blobs} == {"a", "b"}
    # round-trip through JSON
    text = json.dumps(out, sort_keys=True)
    again = json.loads(text)
    assert again["_provenance"]["inputs_sha256"] == pv["inputs_sha256"]


def test_json_metadata_inputs_sha_matches_canonical_body() -> None:
    """The recorded inputs_sha256 must match the SHA256 of the canonical
    JSON body (indent=2, sort_keys, ensure_ascii=False) for downstream
    audit tooling."""
    inputs = [_spec("a")]
    out = pm.json_metadata({}, argv=[], inputs=inputs)
    pv = out["_provenance"]
    canonical = json.dumps(pv["inputs"], indent=2, sort_keys=True,
                           ensure_ascii=False)
    assert pv["inputs_sha256"] == hashlib.sha256(
        canonical.encode("utf-8")).hexdigest()


def test_json_metadata_records_title_when_passed() -> None:
    out = pm.json_metadata({}, argv=[], title="my plot")
    assert out["_provenance"]["produced_by"]["title"] == "my plot"


def test_json_metadata_omits_title_when_none() -> None:
    out = pm.json_metadata({}, argv=[])
    assert "title" not in out["_provenance"]["produced_by"]
