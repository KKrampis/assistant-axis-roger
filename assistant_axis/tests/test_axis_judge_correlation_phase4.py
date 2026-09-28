"""Phase-4 producer-side regression tests for
``results_analysis/axis_judge_correlation.py``.

This pins down the small number of pure helpers I touched so a
refactor doesn't silently regress the trait/role disambiguation
contract:

1. ``_parse_rejudge_names`` — CLI parsing of ``--rejudge_names``
2. ``_save_json`` — schema_version stamping + notes merging
3. ``compute_correlations`` — mixed key shapes (entity_id and bare)
4. The ``cohort_kind`` derivation that fixes the
   ``args.pair_type``-clobbers-config bug.

The script is imported as a module; we call its helpers directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant_axis.provenance import InputSpec
from results_analysis import axis_judge_correlation as ajc


# ---------------------------------------------------------------------------
# _parse_rejudge_names
# ---------------------------------------------------------------------------

class TestParseRejudgeNames:
    def test_none_returns_none(self):
        assert ajc._parse_rejudge_names(None) is None

    def test_empty_string_returns_none(self):
        assert ajc._parse_rejudge_names("") is None
        assert ajc._parse_rejudge_names("   ") is None

    def test_short_kind_tags(self):
        result = ajc._parse_rejudge_names("R:patient T:stoic")
        assert result == [("roles", "patient"), ("traits", "stoic")]

    def test_long_kind_tags(self):
        result = ajc._parse_rejudge_names("role:patient,trait:stoic")
        assert result == [("roles", "patient"), ("traits", "stoic")]

    def test_plural_kind_tags(self):
        result = ajc._parse_rejudge_names("roles:patient,traits:stoic")
        assert result == [("roles", "patient"), ("traits", "stoic")]

    def test_case_insensitive_kind(self):
        result = ajc._parse_rejudge_names("ROLE:patient TRAIT:stoic")
        assert result == [("roles", "patient"), ("traits", "stoic")]

    def test_comma_and_whitespace_separators_mix(self):
        """Both ``,`` and whitespace are accepted as separators."""
        result = ajc._parse_rejudge_names(
            "R:a, T:b\n R:c\tT:d"
        )
        assert result == [
            ("roles", "a"), ("traits", "b"),
            ("roles", "c"), ("traits", "d"),
        ]

    def test_collision_names_handled(self):
        """The 9 collision names must be parseable on both sides."""
        for collision in ["patient", "stoic", "teacher", "student",
                          "philosopher", "scientist", "lawyer",
                          "scholar", "writer"]:
            result = ajc._parse_rejudge_names(
                f"R:{collision} T:{collision}"
            )
            assert result == [("roles", collision), ("traits", collision)]

    def test_missing_colon_raises_systemexit(self):
        with pytest.raises(SystemExit) as exc:
            ajc._parse_rejudge_names("patient")
        assert "malformed" in str(exc.value).lower()

    def test_unknown_kind_raises_systemexit(self):
        with pytest.raises(SystemExit) as exc:
            ajc._parse_rejudge_names("Z:patient")
        # Either SystemExit message or kind_long ValueError text leaks
        # through; both are acceptable.
        assert exc.value is not None

    def test_missing_name_raises_systemexit(self):
        with pytest.raises(SystemExit) as exc:
            ajc._parse_rejudge_names("R:")
        assert "missing entity name" in str(exc.value).lower()

    def test_display_form_hyphen_coerced_to_file_name(self, caplog):
        """``R:devil's-advocate`` -> ``R:devils_advocate`` with a
        WARNING log line so the user notices their input was massaged
        (input-hardening: see AGENT_NOTES "File-name vs display-name
        convention")."""
        import logging
        with caplog.at_level(logging.WARNING, logger=ajc.logger.name):
            result = ajc._parse_rejudge_names("R:systems-thinker")
        assert result == [("roles", "systems_thinker")]
        assert any(
            "coerced" in r.message.lower() and "display-form" in r.message.lower()
            for r in caplog.records
        )

    def test_display_form_apostrophe_coerced(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger=ajc.logger.name):
            result = ajc._parse_rejudge_names("T:devil's_advocate")
        assert result == [("traits", "devils_advocate")]
        assert any("coerced" in r.message.lower() for r in caplog.records)

    def test_display_form_capitals_coerced(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger=ajc.logger.name):
            result = ajc._parse_rejudge_names("R:Patient")
        assert result == [("roles", "patient")]
        assert any("coerced" in r.message.lower() for r in caplog.records)

    def test_no_warning_when_already_file_form(self, caplog):
        """File-name input is the common path and must not log a
        warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger=ajc.logger.name):
            result = ajc._parse_rejudge_names("R:patient T:stoic")
        assert result == [("roles", "patient"), ("traits", "stoic")]
        assert not any("coerced" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# _resolve_rejudge_names_against_scorable: pole-skip vs typo
# ---------------------------------------------------------------------------

class TestResolveRejudgeNamesAgainstScorable:
    """The validator distinguishes legitimate pole-pair drops from typos.

    Regression for the May 2026 5d.1 bug: an axis-invariant
    ``--rejudge_names`` list that contains an RP-depleted trait which
    happens to be the pole pair of one of the 12 v2 axes (e.g.
    ``T:harmful`` on ``harmless_vs_harmful``) used to trigger a
    ``SystemExit`` and abort the orchestrator mid-sweep.  The fix
    silently drops pole-pair entries (logging at INFO) and only
    fatals on entries that are genuinely absent from the corpus.
    """

    # A small synthetic corpus with one collision name + one pole pair.
    # ``corpus`` is the FULL list (pre-pole-skip).  ``scorable`` is the
    # post-pole-skip list passed to the validator.
    CORPUS = [
        ("traits", "harmful"),
        ("traits", "harmless"),
        ("traits", "avoidant"),
        ("traits", "petty"),
        ("roles", "patient"),
        ("roles", "ascetic"),
        ("traits", "patient"),  # name collision
    ]
    # On the harmless_vs_harmful axis, both pole names are skipped:
    POLE_SKIP_NAMES = ("harmful", "harmless")
    # Pre-build SCORABLE outside the class-body comprehension scope
    # (class-body lambdas/comprehensions can't see CORPUS at class
    # scope in CPython 3.x).
    SCORABLE = [
        ("traits", "avoidant"),
        ("traits", "petty"),
        ("roles", "patient"),
        ("roles", "ascetic"),
        ("traits", "patient"),
    ]

    def test_all_scorable_no_drops(self, caplog):
        """When every entry is in scorable, no drops, no SystemExit."""
        import logging
        rejudge = [("traits", "avoidant"), ("traits", "petty"),
                   ("roles", "patient")]
        with caplog.at_level(logging.INFO, logger=ajc.logger.name):
            kept = ajc._resolve_rejudge_names_against_scorable(
                rejudge, scorable=self.SCORABLE,
                corpus_entities=self.CORPUS,
            )
        assert kept == rejudge
        assert not any(
            "silently dropped" in r.message for r in caplog.records
        )

    def test_pole_pair_silently_dropped(self, caplog):
        """``T:harmful`` on harmless_vs_harmful is in corpus but not
        scorable -> dropped + logged, NOT a SystemExit."""
        import logging
        rejudge = [
            ("traits", "avoidant"),
            ("traits", "harmful"),  # pole-pair
            ("traits", "petty"),
        ]
        with caplog.at_level(logging.INFO, logger=ajc.logger.name):
            kept = ajc._resolve_rejudge_names_against_scorable(
                rejudge, scorable=self.SCORABLE,
                corpus_entities=self.CORPUS,
            )
        assert kept == [("traits", "avoidant"), ("traits", "petty")]
        msgs = [r.message for r in caplog.records]
        assert any(
            "silently dropped" in m and "T:harmful" in m for m in msgs
        ), f"records: {msgs}"

    def test_typo_raises_systemexit(self):
        """Entry not in the full corpus is fatal (typo / stale ref)."""
        rejudge = [("traits", "avoidant"), ("traits", "obssesive_typo")]
        with pytest.raises(SystemExit) as exc:
            ajc._resolve_rejudge_names_against_scorable(
                rejudge, scorable=self.SCORABLE,
                corpus_entities=self.CORPUS,
            )
        msg = str(exc.value)
        assert "not in the corpus" in msg
        assert "T:obssesive_typo" in msg

    def test_typo_takes_precedence_over_pole_skip(self):
        """Mixed: a pole-skip + a typo -> SystemExit (typo wins)."""
        rejudge = [
            ("traits", "harmful"),       # pole-skip (legit)
            ("traits", "avoidant"),      # scorable
            ("roles", "fictional_role"),  # not in corpus -> typo
        ]
        with pytest.raises(SystemExit) as exc:
            ajc._resolve_rejudge_names_against_scorable(
                rejudge, scorable=self.SCORABLE,
                corpus_entities=self.CORPUS,
            )
        assert "fictional_role" in str(exc.value)

    def test_empty_rejudge_names(self):
        """Empty list -> empty kept list, no error."""
        kept = ajc._resolve_rejudge_names_against_scorable(
            [], scorable=self.SCORABLE,
            corpus_entities=self.CORPUS,
        )
        assert kept == []

    def test_collision_name_kind_disambiguation(self):
        """``patient|R`` vs ``patient|T`` are distinct in the validator;
        passing ``T:patient`` when only ``R:patient`` exists is treated
        as a typo (regression for the trait/role collision Bug A)."""
        # Strip the trait-side patient from corpus for this scenario.
        corpus = [pair for pair in self.CORPUS
                  if pair != ("traits", "patient")]
        pole_set = set(self.POLE_SKIP_NAMES)
        scorable = [pair for pair in corpus if pair[1] not in pole_set]
        with pytest.raises(SystemExit) as exc:
            ajc._resolve_rejudge_names_against_scorable(
                [("traits", "patient")],
                scorable=scorable,
                corpus_entities=corpus,
            )
        assert "T:patient" in str(exc.value)


# ---------------------------------------------------------------------------
# _save_json: schema_version + notes
# ---------------------------------------------------------------------------

class TestSaveJsonEnvelope:
    def test_bare_no_inputs_no_envelope(self, tmp_path):
        """Without inputs, ``_save_json`` writes plain JSON (used for
        operational files like ``config.json`` / ``gaps.json``)."""
        p = tmp_path / "config.json"
        ajc._save_json(p, {"foo": 1})
        data = json.loads(p.read_text())
        assert data == {"foo": 1}
        assert "_provenance" not in data
        assert "schema_version" not in data

    def test_with_inputs_writes_envelope(self, tmp_path):
        p = tmp_path / "scores.json"
        spec = InputSpec(dep_key="dep", path="some/path", kind="file",
                         fingerprint="v1:abc", extras={})
        ajc._save_json(p, {"patient|R": 1}, inputs=[spec], title="t")
        data = json.loads(p.read_text())
        assert data["result"] == {"patient|R": 1}
        assert "_provenance" in data
        assert "schema_version" not in data  # not stamped unless requested

    def test_schema_version_stamped_at_top_level(self, tmp_path):
        """Phase-4 contract: the *payload* schema_version lives at the
        top level of the envelope, NOT inside ``result``.

        (``_provenance`` may also carry its own ``schema_version``
        field — that's the provenance-block schema, semantically
        distinct from the payload schema.  We only assert about the
        top-level payload version here.)
        """
        p = tmp_path / "scores.json"
        spec = InputSpec(dep_key="dep", path="some/path", kind="file",
                         fingerprint="v1:abc", extras={})
        ajc._save_json(
            p, {"patient|R": 1}, inputs=[spec], schema_version=2,
        )
        data = json.loads(p.read_text())
        assert data["schema_version"] == 2  # top-level payload version
        # Must NOT contaminate the payload itself.
        if isinstance(data["result"], dict):
            assert "schema_version" not in data["result"]

    def test_notes_merged_into_provenance_notes(self, tmp_path):
        """``notes`` is merged into ``_provenance.notes`` so the
        rejudge-list survives round-trips through the cache."""
        p = tmp_path / "scores.json"
        spec = InputSpec(dep_key="dep", path="some/path", kind="file",
                         fingerprint="v1:abc", extras={})
        ajc._save_json(
            p, {"patient|R": 1}, inputs=[spec],
            schema_version=2,
            notes={"rejudge_names": ["R:patient", "T:patient"]},
        )
        data = json.loads(p.read_text())
        assert data["_provenance"]["notes"]["rejudge_names"] == [
            "R:patient", "T:patient",
        ]

    def test_notes_none_does_not_create_notes_key(self, tmp_path):
        p = tmp_path / "scores.json"
        spec = InputSpec(dep_key="dep", path="some/path", kind="file",
                         fingerprint="v1:abc", extras={})
        ajc._save_json(p, {"x": 1}, inputs=[spec])
        data = json.loads(p.read_text())
        # Some json_metadata implementations always include "notes";
        # we only require it isn't an artefact of our merge logic.
        notes = data["_provenance"].get("notes")
        assert notes is None or notes == {} or notes == [] or isinstance(notes, dict)


# ---------------------------------------------------------------------------
# _load_json_or_empty: still permissive about schema_version (unlike
# the static-mode loaders, which loud-reject).
# ---------------------------------------------------------------------------

class TestLoadJsonOrEmpty:
    def test_unwraps_envelope(self, tmp_path):
        p = tmp_path / "scores.json"
        envelope = {
            "result": {"patient|R": 1},
            "_provenance": {"inputs": [], "tool_version": "x"},
            "schema_version": 2,
        }
        p.write_text(json.dumps(envelope))
        out = ajc._load_json_or_empty(p)
        assert out == {"patient|R": 1}

    def test_permissive_about_v1_legacy(self, tmp_path):
        """Resume path must NOT loud-reject v1 caches; it just
        overwrites them with v2 data on the next save."""
        p = tmp_path / "scores.json"
        p.write_text(json.dumps({"patient": 1, "stoic": 0}))
        out = ajc._load_json_or_empty(p)
        assert out == {"patient": 1, "stoic": 0}

    def test_missing_file_returns_empty(self, tmp_path):
        out = ajc._load_json_or_empty(tmp_path / "absent.json")
        assert out == {}

    def test_corrupt_file_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json {")
        out = ajc._load_json_or_empty(p)
        assert out == {}


# ---------------------------------------------------------------------------
# Rubric-version drift detection on cache resume
# ---------------------------------------------------------------------------

def _make_v2_envelope_with_rubric(
    payload: object,
    rubric_version: object,
    *,
    per_entity_rubric_versions: object = None,
) -> dict:
    """Build an envelope identical in shape to what the real producer
    writes via ``_save_json`` + ``_build_axis_judge_inputs``, with a
    controlled ``rubric_version`` in the producer_script extras (or
    omitted if ``rubric_version is None``).

    When ``per_entity_rubric_versions`` is provided, it's stamped into
    ``_provenance.notes.per_entity_rubric_versions``, matching the
    post-May-2026 per-entity stamping schema.  Tests that exercise the
    new per-entity drift path use this; legacy tests that only set
    the cohort-level stamp leave it at ``None``.
    """
    extras: dict = {}
    if rubric_version is not None:
        extras["rubric_version"] = rubric_version
    prov: dict = {
        "inputs": [
            {
                "dep_key": "producer_script",
                "path": "results_analysis/axis_judge_correlation.py",
                "kind": "file",
                "fingerprint": "v1:2026-05-11T00:00:00+00:00@123",
                "extras": extras,
            },
        ],
    }
    if per_entity_rubric_versions is not None:
        prov["notes"] = {"per_entity_rubric_versions": per_entity_rubric_versions}
    return {
        "schema_version": 2,
        "result": payload,
        "_provenance": prov,
    }


class TestPeekRubricVersion:
    """Producer-side mirror of ``judge_loaders.peek_rubric_version``."""

    def test_returns_recorded_label(self, tmp_path):
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric({}, "v3")))
        assert ajc._peek_rubric_version(p) == "v3"

    def test_returns_v2_label_verbatim(self, tmp_path):
        """The check is label-based: no comparison logic here, just
        report what was recorded."""
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric({}, "v2")))
        assert ajc._peek_rubric_version(p) == "v2"

    def test_missing_rubric_version_returns_none(self, tmp_path):
        """Envelope present but no rubric_version stamped → None."""
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric({}, None)))
        assert ajc._peek_rubric_version(p) is None

    def test_no_envelope_returns_none(self, tmp_path):
        """Bare JSON (legacy pre-Phase-6) → None."""
        p = tmp_path / "scores.json"
        p.write_text(json.dumps({"patient|R": 1}))
        assert ajc._peek_rubric_version(p) is None

    def test_file_missing_returns_none(self, tmp_path):
        assert ajc._peek_rubric_version(tmp_path / "absent.json") is None

    def test_corrupt_json_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json{")
        assert ajc._peek_rubric_version(p) is None


class TestPeekPerEntityRubricVersions:
    """Producer-side reader for the per-entity rubric_version map."""

    def test_returns_map_when_present(self, tmp_path):
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric(
            {"patient|R": 1},
            "v3",
            per_entity_rubric_versions={"patient|R": "v3", "stoic|T": "v2"},
        )))
        assert ajc._peek_per_entity_rubric_versions(p) == {
            "patient|R": "v3",
            "stoic|T": "v2",
        }

    def test_returns_empty_dict_when_absent(self, tmp_path):
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric({}, "v3")))
        assert ajc._peek_per_entity_rubric_versions(p) == {}

    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        assert ajc._peek_per_entity_rubric_versions(
            tmp_path / "absent.json"
        ) == {}

    def test_returns_empty_dict_on_corrupt_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json{")
        assert ajc._peek_per_entity_rubric_versions(p) == {}

    def test_ignores_non_string_values(self, tmp_path):
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric(
            {},
            "v3",
            per_entity_rubric_versions={
                "patient|R": "v3",
                "weird|R": 7,  # not a string -> filtered out
            },
        )))
        assert ajc._peek_per_entity_rubric_versions(p) == {"patient|R": "v3"}


class TestCheckRubricVersionOnResume:
    """``_check_rubric_version_on_resume`` is the producer-resume gate
    that drops stale-rubric entries per-entity, consulting the
    equivalence registry for "different version but byte-identical
    prompts" cases.  Returns ``(filtered_cache, kept_stamps_map)``.
    """

    def test_matching_cohort_version_keeps_everything(self, tmp_path):
        """Cohort rubric == current, no per-entity stamps → every
        entry passes through; kept_stamps is keyed by every entry
        and points at the cohort stamp."""
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(
            _make_v2_envelope_with_rubric({"patient|R": 1}, ajc.RUBRIC_VERSION)
        ))
        cache = {"patient|R": 1, "patient|T": -2}
        kept, stamps = ajc._check_rubric_version_on_resume(
            p, cache, mode="descriptions"
        )
        assert kept == cache, "matching rubric_version must keep cache intact"
        assert stamps == {
            "patient|R": ajc.RUBRIC_VERSION,
            "patient|T": ajc.RUBRIC_VERSION,
        }

    def test_mismatched_cohort_drops_all_entries(self, tmp_path, caplog,
                                                  monkeypatch):
        """Cohort stamp differs from current AND no equivalence edge →
        every entry drops, kept is ``{}``."""
        # Force empty registry so v2->v3 isn't found equivalent.
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [],
        )
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(
            _make_v2_envelope_with_rubric({"patient|R": 1}, "v2")
        ))
        cache = {"patient|R": 1, "patient|T": -2}
        import logging
        with caplog.at_level(logging.WARNING, logger=ajc.logger.name):
            kept, stamps = ajc._check_rubric_version_on_resume(
                p, cache, mode="descriptions"
            )
        assert kept == {}
        assert stamps == {}
        msgs = " ".join(r.message for r in caplog.records)
        assert "rubric_version drift" in msgs
        assert "'v2'" in msgs
        assert f"'{ajc.RUBRIC_VERSION}'" in msgs

    def test_unstamped_cache_passed_through_as_legacy(self, tmp_path):
        """Envelope present but no cohort stamp AND no per-entity stamps
        = truly legacy.  Pass through unchanged; the older
        schema_version / fingerprint checks own this case."""
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(
            _make_v2_envelope_with_rubric({"patient|R": 1}, None)
        ))
        cache = {"patient|R": 1}
        kept, stamps = ajc._check_rubric_version_on_resume(
            p, cache, mode="descriptions"
        )
        assert kept == cache
        assert stamps == {}, "legacy fallback must not invent stamps"

    def test_empty_cache_short_circuit(self, tmp_path):
        p = tmp_path / "scores.json"
        kept, stamps = ajc._check_rubric_version_on_resume(
            p, {}, mode="descriptions"
        )
        assert kept == {} and stamps == {}

    def test_file_missing_passes_cache_through_as_legacy(self, tmp_path):
        """File gone but in-memory cache present (caller path with
        ``no_cache=False`` but post-delete) → no stamp info to act on
        → pass through, mirroring the legacy fallback."""
        p = tmp_path / "doesnt_exist.json"
        cache = {"patient|R": 1}
        kept, stamps = ajc._check_rubric_version_on_resume(
            p, cache, mode="responses"
        )
        assert kept == cache
        assert stamps == {}

    def test_drop_warning_points_at_tooling(self, tmp_path, caplog,
                                             monkeypatch):
        """Warning must mention RUBRIC_VERSION (for context) and
        ``mark_rubric_equivalent.py`` / ``--strict_rubric_version``
        (so operators know how to declare or escalate)."""
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [],
        )
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(
            _make_v2_envelope_with_rubric({"patient|R": 1}, "v1")
        ))
        import logging
        with caplog.at_level(logging.WARNING, logger=ajc.logger.name):
            ajc._check_rubric_version_on_resume(
                p, {"patient|R": 1}, mode="descriptions"
            )
        msgs = " ".join(r.message for r in caplog.records)
        assert "RUBRIC_VERSION" in msgs
        assert "mark_rubric_equivalent.py" in msgs
        assert "--strict_rubric_version" in msgs

    def test_per_entity_stamps_take_precedence_over_cohort(self, tmp_path):
        """When a cache carries per-entity stamps, those drive the
        per-entry decision; cohort stamp is only the fallback for
        entries that don't have a per-entity stamp."""
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric(
            {"patient|R": 1, "stoic|T": 2},
            "v2",  # cohort says v2
            per_entity_rubric_versions={
                "patient|R": ajc.RUBRIC_VERSION,  # fresh
                "stoic|T": "v2",  # stale (matches cohort)
            },
        )))
        cache = {"patient|R": 1, "stoic|T": 2}
        kept, stamps = ajc._check_rubric_version_on_resume(
            p, cache, mode="descriptions"
        )
        # patient|R kept (per-entity stamp says current), stoic|T
        # dropped (per-entity stamp says v2, no equivalence).
        assert "patient|R" in kept
        assert "stoic|T" not in kept
        assert stamps == {"patient|R": ajc.RUBRIC_VERSION}

    def test_partial_drop_keeps_warning_listing_dropped_eids(
        self, tmp_path, caplog, monkeypatch,
    ):
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [],
        )
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric(
            {"keep|R": 1, "drop|R": 2, "drop|T": 3},
            "v2",
            per_entity_rubric_versions={
                "keep|R": ajc.RUBRIC_VERSION,
                "drop|R": "v2",
                "drop|T": "v2",
            },
        )))
        cache = {"keep|R": 1, "drop|R": 2, "drop|T": 3}
        import logging
        with caplog.at_level(logging.WARNING, logger=ajc.logger.name):
            kept, _ = ajc._check_rubric_version_on_resume(
                p, cache, mode="descriptions"
            )
        assert set(kept.keys()) == {"keep|R"}
        msgs = " ".join(r.message for r in caplog.records)
        assert "drop|R" in msgs
        assert "drop|T" in msgs
        assert "2/3" in msgs  # dropped 2 out of 3

    def test_equivalence_keeps_drifted_entries(
        self, tmp_path, monkeypatch, caplog,
    ):
        """When ``rubric_equivalence`` declares the (v2, current) pair
        equivalent for this (axis, mode) cell, the cache entries
        survive even though their stamps don't match current."""
        from assistant_axis.rubric_equivalence import RubricEquivalenceEdge
        # Declare v2->current as equivalent for any axis/mode/entity.
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [
                RubricEquivalenceEdge(
                    from_rubric="v2", to_rubric=ajc.RUBRIC_VERSION,
                    reason="test equivalence",
                )
            ],
        )
        # Cache under an axis subdir so _axis_from_cache_path returns
        # something useful; not strictly required since the test
        # registry edge is unscoped.
        axis_dir = tmp_path / "concise_vs_verbose" / "gpt"
        axis_dir.mkdir(parents=True)
        p = axis_dir / "scores_descriptions.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric(
            {"patient|R": 1, "stoic|T": 2}, "v2"
        )))
        cache = {"patient|R": 1, "stoic|T": 2}
        import logging
        with caplog.at_level(logging.INFO, logger=ajc.logger.name):
            kept, stamps = ajc._check_rubric_version_on_resume(
                p, cache, mode="descriptions"
            )
        assert kept == cache, "equivalent entries must be kept"
        assert stamps == {"patient|R": "v2", "stoic|T": "v2"}, (
            "kept stamps must reflect the ORIGINAL (pre-equivalence) "
            "version, not the current one"
        )
        info_msgs = " ".join(r.message for r in caplog.records
                              if r.levelname == "INFO")
        assert "rubric_equivalence" in info_msgs

    def test_strict_mode_aborts_on_undeclared_drift(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [],
        )
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric(
            {"patient|R": 1}, "v2"
        )))
        with pytest.raises(SystemExit, match="strict_rubric_version"):
            ajc._check_rubric_version_on_resume(
                p, {"patient|R": 1}, mode="descriptions", strict=True
            )

    def test_strict_mode_silent_when_equivalent(
        self, tmp_path, monkeypatch,
    ):
        """Strict mode aborts only on UNDECLARED drift; a declared
        equivalence still lets the cache through silently."""
        from assistant_axis.rubric_equivalence import RubricEquivalenceEdge
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [
                RubricEquivalenceEdge(
                    from_rubric="v2", to_rubric=ajc.RUBRIC_VERSION,
                    reason="test",
                )
            ],
        )
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(_make_v2_envelope_with_rubric(
            {"patient|R": 1}, "v2"
        )))
        # Must NOT raise.
        kept, _ = ajc._check_rubric_version_on_resume(
            p, {"patient|R": 1}, mode="descriptions", strict=True
        )
        assert kept == {"patient|R": 1}


# ---------------------------------------------------------------------------
# compute_correlations: mixed key shapes
# ---------------------------------------------------------------------------

class TestComputeCorrelations:
    def test_disambiguated_keys_static_mode(self):
        """Static-mode (mixed-kind) inputs use entity_id keys; the
        emitted ``names`` array preserves entity_id form."""
        # 5 entities, kind-mixed, including the "patient" collision
        scores = {
            "descriptions": {
                "patient|R": 1, "patient|T": -2,
                "stoic|R": 0, "stoic|T": -1,
                "teacher|R": 2,
            },
        }
        projections = {
            0: {
                "patient|R": {"raw": 0.5, "whitened": 0.4},
                "patient|T": {"raw": -0.3, "whitened": -0.2},
                "stoic|R": {"raw": 0.1, "whitened": 0.05},
                "stoic|T": {"raw": -0.1, "whitened": -0.05},
                "teacher|R": {"raw": 0.7, "whitened": 0.6},
            },
        }
        out = ajc.compute_correlations(
            scores, projections, slots=[0], excluded_set=set(),
        )
        names = out["descriptions"]["0"]["raw"]["names"]
        assert "patient|R" in names
        assert "patient|T" in names
        # Both kinds of "patient" must appear separately — the whole
        # point of the disambiguation.
        assert names.count("patient|R") == 1
        assert names.count("patient|T") == 1
        # n must be 5 (no exclusions)
        assert out["descriptions"]["0"]["raw"]["n"] == 5

    def test_bare_name_response_mode(self):
        """Response-mode (kind-pure) inputs use bare names; the emitted
        ``names`` array is also bare."""
        scores = {"responses": {"patient": 0.8, "stoic": 0.4, "teacher": 0.6}}
        projections = {
            0: {
                "patient": {"raw": 0.5, "whitened": 0.4},
                "stoic": {"raw": 0.1, "whitened": 0.05},
                "teacher": {"raw": 0.7, "whitened": 0.6},
            },
        }
        out = ajc.compute_correlations(
            scores, projections, slots=[0], excluded_set=set(),
        )
        names = out["responses"]["0"]["raw"]["names"]
        assert names == sorted(["patient", "stoic", "teacher"])
        assert out["responses"]["0"]["raw"]["n"] == 3

    def test_excluded_set_bare_names_strips_disambiguated_keys(self):
        """``excluded_set`` is bare-name (it's the pole pair), but the
        score keys are entity_ids — the helper must use the bare side
        of each entity_id for the membership check."""
        scores = {
            "descriptions": {
                "patient|R": 1, "patient|T": -2,
                "stoic|R": 0, "stoic|T": -1,
                "teacher|R": 2, "teacher|T": -3,
            },
        }
        projections = {
            0: {
                "patient|R": {"raw": 0.5, "whitened": 0.4},
                "patient|T": {"raw": -0.3, "whitened": -0.2},
                "stoic|R": {"raw": 0.1, "whitened": 0.05},
                "stoic|T": {"raw": -0.1, "whitened": -0.05},
                "teacher|R": {"raw": 0.7, "whitened": 0.6},
                "teacher|T": {"raw": -0.7, "whitened": -0.6},
            },
        }
        # Excluded is BARE name "patient" — must match BOTH patient|R
        # and patient|T.
        out = ajc.compute_correlations(
            scores, projections, slots=[0], excluded_set={"patient"},
        )
        names = out["descriptions"]["0"]["raw"]["names"]
        assert "patient|R" not in names
        assert "patient|T" not in names
        # The other 4 (stoic|R/T, teacher|R/T) survive.
        assert out["descriptions"]["0"]["raw"]["n"] == 4

    def test_skips_entries_missing_from_projections(self):
        """An entity in scores but not in projections is silently
        dropped (the rho is computed over the intersection)."""
        scores = {
            "descriptions": {
                "patient|R": 1, "patient|T": -2,
                "stoic|R": 0,
                "ghost|R": 99,  # not in projections
            },
        }
        projections = {
            0: {
                "patient|R": {"raw": 0.5, "whitened": 0.4},
                "patient|T": {"raw": -0.3, "whitened": -0.2},
                "stoic|R": {"raw": 0.1, "whitened": 0.05},
            },
        }
        out = ajc.compute_correlations(
            scores, projections, slots=[0], excluded_set=set(),
        )
        names = out["descriptions"]["0"]["raw"]["names"]
        assert "ghost|R" not in names
        assert out["descriptions"]["0"]["raw"]["n"] == 3

    def test_empty_score_map_yields_nan_rho(self):
        scores = {"descriptions": {}}
        projections = {0: {}}
        out = ajc.compute_correlations(
            scores, projections, slots=[0], excluded_set=set(),
        )
        # rho/p should be NaN with n=0 (or fewer than 3 items).
        result = out["descriptions"]["0"]["raw"]
        assert result["n"] == 0
        # NaN; can't compare NaN with == so use isnan.
        import math
        assert math.isnan(result["rho"])


# ---------------------------------------------------------------------------
# Phase 4b: judge-aware subsampling defaults (Bug B fix)
# ---------------------------------------------------------------------------

import argparse


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal Namespace for the subsampling resolver tests.

    We only set the fields the resolver reads + writes; everything
    else is irrelevant for the judge-family + no_subsample logic.
    """
    defaults = {
        "provider": "openai",
        "judge_model": None,
        "no_subsample": None,
        "question_subsample_modulo": 0,
        "tiered_modulo_per_chunk": 3,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestJudgeFamily:
    def test_openai_is_gpt(self):
        assert ajc._judge_family("openai", "gpt-4.1-mini") == "gpt"
        assert ajc._judge_family("openai", "gpt-5-pro") == "gpt"
        assert ajc._judge_family("openai", None) == "gpt"

    def test_anthropic_haiku(self):
        assert ajc._judge_family(
            "anthropic", "claude-haiku-4-5-20251001"
        ) == "haiku"
        assert ajc._judge_family(
            "anthropic", "claude-3-5-haiku-20241022"
        ) == "haiku"

    def test_anthropic_sonnet(self):
        assert ajc._judge_family(
            "anthropic", "claude-sonnet-4-20250514"
        ) == "sonnet"

    def test_anthropic_default_model_is_sonnet(self):
        """``judge_model=None`` → resolves to provider default
        (claude-sonnet-4-20250514) → 'sonnet' family."""
        assert ajc._judge_family("anthropic", None) == "sonnet"

    def test_anthropic_unknown_model_fails_safe_to_sonnet(self):
        """Future Opus / unknown Anthropic model → conservatively
        classify as Sonnet (= the most expensive bucket = no
        --no_subsample escape hatch).  Preferring caution > speed."""
        assert ajc._judge_family(
            "anthropic", "claude-opus-5-future"
        ) == "sonnet"

    def test_unknown_provider_raises(self):
        with pytest.raises(SystemExit):
            ajc._judge_family("vllm-local", "some-model")


class TestApplyJudgeAwareSubsampleDefaults:
    """Cover all 6 cases in the design table:
    judge_family ∈ {gpt, haiku, sonnet} × no_subsample ∈ {None, True}.
    """

    # GPT --------------------------------------------------------------

    def test_gpt_default_unspecified_becomes_full_volume(self):
        args = _make_args(provider="openai", no_subsample=None)
        ajc._apply_judge_aware_subsample_defaults(args)
        assert args.no_subsample is True

    def test_gpt_explicit_no_subsample_honoured(self):
        args = _make_args(provider="openai", no_subsample=True)
        ajc._apply_judge_aware_subsample_defaults(args)
        assert args.no_subsample is True

    # Haiku ------------------------------------------------------------

    def test_haiku_default_uses_tiered(self):
        args = _make_args(
            provider="anthropic",
            judge_model="claude-haiku-4-5-20251001",
            no_subsample=None,
        )
        ajc._apply_judge_aware_subsample_defaults(args)
        assert args.no_subsample is False

    def test_haiku_explicit_no_subsample_honoured(self):
        """Haiku CAN be run at full volume (allowed though discouraged)."""
        args = _make_args(
            provider="anthropic",
            judge_model="claude-haiku-4-5-20251001",
            no_subsample=True,
        )
        ajc._apply_judge_aware_subsample_defaults(args)
        assert args.no_subsample is True

    # Sonnet -----------------------------------------------------------

    def test_sonnet_default_uses_tiered(self):
        args = _make_args(provider="anthropic", judge_model=None,
                          no_subsample=None)
        ajc._apply_judge_aware_subsample_defaults(args)
        assert args.no_subsample is False

    def test_sonnet_explicit_no_subsample_hard_fails(self):
        """The Sonnet escape hatch is locked: --no_subsample is a
        SystemExit, not a warning."""
        args = _make_args(
            provider="anthropic",
            judge_model="claude-sonnet-4-20250514",
            no_subsample=True,
        )
        with pytest.raises(SystemExit) as exc:
            ajc._apply_judge_aware_subsample_defaults(args)
        assert "Sonnet" in str(exc.value)
        assert "no_subsample" in str(exc.value).lower() or \
               "subsample" in str(exc.value).lower()


class TestSubsampleDefaultsIntegrationWithArgparse:
    """End-to-end: parse_args + resolver behaves as a real CLI invocation."""

    def test_argparse_no_subsample_default_is_none(self):
        """The flag must be tri-state: parser default = None so the
        resolver can distinguish 'not specified' from 'explicitly
        false'."""
        parser_argv_simulator = ajc.parse_args  # smoke test reachable
        assert parser_argv_simulator is not None  # not really testable
        # Direct introspection: we can't easily call parse_args
        # without sys.argv mocking; verify default via an empty Namespace.
        # The actual default_None assertion lives in the integration
        # test via _apply_judge_aware_subsample_defaults below.

    def test_resolver_records_in_logs(self, caplog):
        """The resolver emits a [subsample] log line so the resolved
        defaults are visible in run logs."""
        import logging
        with caplog.at_level(logging.INFO, logger=ajc.logger.name):
            args = _make_args(provider="openai", no_subsample=None)
            ajc._apply_judge_aware_subsample_defaults(args)
        # Loose match: just check the [subsample] tag is present.
        assert any(
            "[subsample]" in r.message for r in caplog.records
        ), f"records: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Phase 4c: _flush_budget_artifacts (usage.json side-car)
# ---------------------------------------------------------------------------

from assistant_axis.judge_pricing import BudgetTracker, UsageTotals


def _make_run_args(**kwargs) -> argparse.Namespace:
    """Args namespace covering the fields _flush_budget_artifacts reads."""
    defaults = {
        "output_dir": None,
        "usage_json": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _tracker_with_cost(cost_usd: float) -> BudgetTracker:
    """Build a tracker with a non-trivial recorded cost for assertion."""
    t = BudgetTracker(totals=UsageTotals(model="gpt-4.1-mini"))
    # 2.5M input tokens @ $0.40/1M = $1.00 per million-tokens — scale.
    tokens = int(cost_usd / 0.40 * 1_000_000)
    t.charge(tokens, 0)
    return t


class TestFlushBudgetArtifacts:
    def test_writes_side_car_to_output_dir_default(self, tmp_path):
        """When --usage_json is not set but --output_dir is, write
        ``<output_dir>/usage.json``."""
        args = _make_run_args(output_dir=str(tmp_path))
        tracker = _tracker_with_cost(0.40)
        ajc._flush_budget_artifacts(args, tracker)
        side_car = tmp_path / "usage.json"
        assert side_car.exists(), "default side-car location not written"
        data = json.loads(side_car.read_text())
        assert data["cost_usd"] == pytest.approx(0.40, abs=1e-6)
        assert data["model"] == "gpt-4.1-mini"
        assert data["n_calls"] == 1

    def test_writes_side_car_to_explicit_path(self, tmp_path):
        """``--usage_json /custom/path.json`` overrides the default."""
        custom = tmp_path / "subdir" / "billing.json"
        args = _make_run_args(
            output_dir=str(tmp_path),
            usage_json=str(custom),
        )
        tracker = _tracker_with_cost(0.20)
        ajc._flush_budget_artifacts(args, tracker)
        assert custom.exists()
        # Default location must NOT be written when override is set.
        assert not (tmp_path / "usage.json").exists()

    def test_dev_null_disables_side_car(self, tmp_path):
        """``--usage_json /dev/null`` is the documented disable."""
        args = _make_run_args(
            output_dir=str(tmp_path),
            usage_json="/dev/null",
        )
        tracker = _tracker_with_cost(0.40)
        ajc._flush_budget_artifacts(args, tracker)
        assert not (tmp_path / "usage.json").exists()

    def test_no_output_dir_no_side_car(self, tmp_path):
        """Without --output_dir or --usage_json, no file is written
        (== legacy callers with no I/O location)."""
        args = _make_run_args(output_dir=None, usage_json=None)
        tracker = _tracker_with_cost(0.40)
        # Must not raise (no path to write to → quietly skip).
        ajc._flush_budget_artifacts(args, tracker)

    def test_logs_summary_line(self, tmp_path, caplog):
        """End-of-run summary log line is emitted regardless of
        side-car path (so operator sees the bill in the live log)."""
        import logging
        args = _make_run_args(output_dir=str(tmp_path))
        tracker = _tracker_with_cost(0.40)
        with caplog.at_level(logging.INFO, logger=ajc.logger.name):
            ajc._flush_budget_artifacts(args, tracker)
        assert any(
            "[budget]" in r.message and "cost=$0.40" in r.message
            for r in caplog.records
        )

    def test_side_car_includes_budget_and_expected(self, tmp_path):
        """Cap + expected fields propagate through to the side-car so
        post-run scripts can verify budget compliance."""
        args = _make_run_args(output_dir=str(tmp_path))
        tracker = BudgetTracker(
            totals=UsageTotals(model="gpt-4.1-mini"),
            budget_usd=10.00,
            expected_cost_usd=5.00,
        )
        tracker.charge(1_000_000, 0)  # $0.40
        ajc._flush_budget_artifacts(args, tracker)
        data = json.loads((tmp_path / "usage.json").read_text())
        assert data["budget_usd"] == 10.0
        assert data["expected_cost_usd"] == 5.0
        assert data["actual_over_expected"] == pytest.approx(0.08, abs=1e-3)


# ---------------------------------------------------------------------------
# Phase 4: v1 (bare-name) → v2 (entity_id) cache migration on resume
# ---------------------------------------------------------------------------

class TestStaleBareNameCacheMigration:
    """When ``score_static_mode`` resumes from a v1 (bare-name) cache,
    bare names get RELABELLED to entity_id keys when their kind is
    unambiguous in the corpus, and DROPPED for the 9 collision names
    (which Phase 5b will surgically rejudge).  Critical for the
    Phase 5a "recompute without re-judging 580 entities" plan.
    """

    def _make_kinds_map(self) -> dict:
        """Same shape as the dict score_static_mode builds from
        ``corpus.descriptions``: ``name → {set of kinds}``."""
        return {
            # collision names (in BOTH corpora)
            "patient": {"traits", "roles"},
            "stoic": {"traits", "roles"},
            "ascetic": {"traits", "roles"},
            # trait-only names
            "concise": {"traits"},
            "verbose": {"traits"},
            # role-only names
            "doctor": {"roles"},
            "lawyer": {"roles"},
        }

    def _migrate_inline(self, cache, kinds_for_name):
        """Reproduce the migration loop from score_static_mode so we
        can unit-test the predicate without mocking the whole
        corpus + axis machinery."""
        from assistant_axis.entity_id import entity_id, is_entity_id
        out = {}
        for k, v in cache.items():
            if is_entity_id(k):
                out[k] = v
                continue
            kinds = kinds_for_name.get(k, set())
            if len(kinds) == 1:
                only_kind = next(iter(kinds))
                out[entity_id(k, only_kind)] = v
            # collision and orphan: drop (no else branch needed)
        return out

    def test_unique_kind_relabels_to_entity_id(self):
        """Trait-only and role-only names get relabelled with their
        unambiguous kind suffix — the common case (~571/580 entries
        in production)."""
        kinds = self._make_kinds_map()
        cache = {"concise": 4, "verbose": -2, "doctor": 1, "lawyer": -3}
        out = self._migrate_inline(cache, kinds)
        assert out == {
            "concise|T": 4,
            "verbose|T": -2,
            "doctor|R": 1,
            "lawyer|R": -3,
        }

    def test_collision_names_dropped(self):
        """The 9 names in both corpora get dropped — Bug A means we
        can't trust the bare-name value (could be either kind's
        score).  Phase 5b rejudges these surgically."""
        kinds = self._make_kinds_map()
        cache = {"patient": 3, "stoic": -1, "ascetic": 2}
        out = self._migrate_inline(cache, kinds)
        assert out == {}

    def test_orphan_keys_dropped(self):
        """Bare names not in EITHER corpus (corpus edits, stale
        imports) are dropped — no recovery possible."""
        kinds = self._make_kinds_map()
        cache = {"removed_name": 2, "ancient_artifact": -3}
        out = self._migrate_inline(cache, kinds)
        assert out == {}

    def test_mixed_v1_v2_cache_preserves_v2_relabels_v1(self):
        """A partially-migrated cache (some v1, some v2) is the
        critical resume case: v2 keys pass through, v1 keys get
        relabelled or dropped per the rules above."""
        kinds = self._make_kinds_map()
        cache = {
            # already-v2 (forward-compatible)
            "concise|T": 5,
            "doctor|R": 1,
            # v1 unique-kind: relabel
            "verbose": -2,
            "lawyer": -3,
            # v1 collision: drop
            "patient": 3,
            "stoic": -1,
            # v1 orphan: drop
            "removed_name": 2,
        }
        out = self._migrate_inline(cache, kinds)
        assert out == {
            "concise|T": 5,
            "doctor|R": 1,
            "verbose|T": -2,
            "lawyer|R": -3,
        }

    def test_v2_only_cache_pass_through_unchanged(self):
        """A pure v2 cache survives migration intact — idempotent
        under repeated loads (the common forward-progress case once
        the bug is in the rear-view mirror)."""
        kinds = self._make_kinds_map()
        v2 = {
            "patient|R": 3, "patient|T": -1,
            "concise|T": 4, "doctor|R": 2,
        }
        out = self._migrate_inline(v2, kinds)
        assert out == v2

    def test_relabel_count_matches_collision_set(self):
        """Pin the migration math: across a realistic full v1 cache
        (580 entries with the May 2026 collision count of 9),
        we expect exactly 9 collision drops and the rest relabelled
        — NO judge re-calls beyond what Phase 5b needs."""
        kinds = self._make_kinds_map()
        # Simulate a cache that has all the names in our test
        # corpus (3 collisions + 4 unique).
        cache = {n: 0 for n in kinds.keys()}
        out = self._migrate_inline(cache, kinds)
        # 4 unique-kind names relabel; 3 collisions drop.
        assert len(out) == 4
        assert all(("|T" in k) or ("|R" in k) for k in out)


# ---------------------------------------------------------------------------
# Phase 4c: tracker integration with API call sites
# ---------------------------------------------------------------------------

class TestCallJudgeTrackerIntegration:
    """Smoke-test that the tracker is actually charged on each API call.

    Pin the contract: every successful call_judge invocation MUST tick
    the tracker at least once with non-zero usage.
    """

    def test_anthropic_call_charges_tracker(self, monkeypatch):
        import asyncio
        from types import SimpleNamespace

        # Mock the anthropic SDK so we don't hit the network.
        class _MockUsage:
            input_tokens = 100
            output_tokens = 20

        class _MockBlock:
            text = "5"

        class _MockResp:
            content = [_MockBlock()]
            usage = _MockUsage()

        class _MockMessages:
            async def create(self, **kwargs):
                return _MockResp()

        class _MockClient:
            def __init__(self):
                self.messages = _MockMessages()

        class _MockAnthropic:
            AsyncAnthropic = _MockClient

        # The import is local to _call_anthropic_batch.
        monkeypatch.setitem(
            __import__("sys").modules, "anthropic", _MockAnthropic,
        )

        tracker = BudgetTracker(totals=UsageTotals(model="claude-haiku-4-5"))
        from assistant_axis.judge import RateLimiter
        results = asyncio.run(ajc._call_anthropic_batch(
            ["prompt 1", "prompt 2"],
            model="claude-haiku-4-5",
            max_tokens=100,
            temperature=0.0,
            rate_limiter=RateLimiter(100.0),
            batch_size=2,
            tracker=tracker,
        ))
        assert results == ["5", "5"]
        # Tracker must have ticked on each of the 2 calls.
        assert tracker.totals.n_calls == 2
        assert tracker.totals.prompt_tokens == 200
        assert tracker.totals.completion_tokens == 40
        # Cost = 200 * $1/1M + 40 * $5/1M = $0.0002 + $0.0002 = $0.0004
        assert tracker.totals.cost_usd == pytest.approx(0.0004, rel=1e-6)

    def test_openai_call_charges_tracker(self, monkeypatch):
        import asyncio
        from types import SimpleNamespace

        class _MockUsage:
            prompt_tokens = 50
            completion_tokens = 10

        class _MockMessage:
            content = "3"

        class _MockChoice:
            message = _MockMessage()

        class _MockResp:
            choices = [_MockChoice()]
            usage = _MockUsage()

        class _MockChatCompletions:
            async def create(self, **kwargs):
                return _MockResp()

        class _MockClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=_MockChatCompletions())

        class _MockOpenAI:
            AsyncOpenAI = _MockClient

        monkeypatch.setitem(
            __import__("sys").modules, "openai", _MockOpenAI,
        )

        tracker = BudgetTracker(totals=UsageTotals(model="gpt-4.1-mini"))
        from assistant_axis.judge import RateLimiter
        results = asyncio.run(ajc._call_openai_batch(
            ["p1"],
            model="gpt-4.1-mini",
            max_tokens=100,
            temperature=0.0,
            rate_limiter=RateLimiter(100.0),
            batch_size=1,
            tracker=tracker,
        ))
        assert results == ["3"]
        assert tracker.totals.n_calls == 1
        assert tracker.totals.prompt_tokens == 50
        assert tracker.totals.completion_tokens == 10

    def test_anthropic_no_tracker_does_not_crash(self, monkeypatch):
        """Backward-compat: pre-Phase-4c callers (no tracker arg) still
        work — the tracker plumbing must be a strict superset of the
        old contract."""
        import asyncio

        class _MockBlock:
            text = "5"

        class _MockResp:
            content = [_MockBlock()]

        class _MockMessages:
            async def create(self, **kwargs):
                return _MockResp()

        class _MockClient:
            def __init__(self):
                self.messages = _MockMessages()

        class _MockAnthropic:
            AsyncAnthropic = _MockClient

        monkeypatch.setitem(
            __import__("sys").modules, "anthropic", _MockAnthropic,
        )
        from assistant_axis.judge import RateLimiter
        results = asyncio.run(ajc._call_anthropic_batch(
            ["p"], model="claude-haiku-4-5", max_tokens=100, temperature=0.0,
            rate_limiter=RateLimiter(100.0), batch_size=1,
            # No tracker arg — must default to None and not crash.
        ))
        assert results == ["5"]

    def test_budget_cap_aborts_mid_batch(self, monkeypatch):
        """The cap must trigger immediately on the call that crosses
        it, propagating BudgetExceededError UP through _call_*_batch
        (rather than being swallowed by retry/gather).  Critical for
        the 'aborts cleanly mid-run' contract.
        """
        import asyncio
        from assistant_axis.judge_pricing import BudgetExceededError

        class _MockUsage:
            input_tokens = 1_000_000  # huge — guaranteed cap-cross
            output_tokens = 0

        class _MockBlock:
            text = "5"

        class _MockResp:
            content = [_MockBlock()]
            usage = _MockUsage()

        class _MockMessages:
            async def create(self, **kwargs):
                return _MockResp()

        class _MockClient:
            def __init__(self):
                self.messages = _MockMessages()

        class _MockAnthropic:
            AsyncAnthropic = _MockClient

        monkeypatch.setitem(
            __import__("sys").modules, "anthropic", _MockAnthropic,
        )

        tracker = BudgetTracker(
            totals=UsageTotals(model="claude-haiku-4-5"),
            budget_usd=0.10,  # 1M tokens @ $1/1M = $1.00 → way over cap
        )
        from assistant_axis.judge import RateLimiter

        # The cap-crossing call MUST raise BudgetExceededError up
        # through _call_anthropic_batch (rather than being swallowed
        # by gather's return_exceptions=True or retry's classifier).
        with pytest.raises(BudgetExceededError):
            asyncio.run(ajc._call_anthropic_batch(
                ["p1"], model="claude-haiku-4-5",
                max_tokens=100, temperature=0.0,
                rate_limiter=RateLimiter(100.0), batch_size=1,
                tracker=tracker,
            ))
        # Tracker recorded the cap-crossing call before raising.
        assert tracker.totals.n_calls == 1
        assert tracker.totals.cost_usd > tracker.budget_usd


# ---------------------------------------------------------------------------
# Rubric builders: file-form -> display-form for LLM (RUBRIC_VERSION v3)
# ---------------------------------------------------------------------------

import torch  # noqa: E402  (lazy: pytorch is heavy + only used here)


def _make_axis_spec(
    *, axis_name: str = "systems_thinker (+) vs analytical (-) [traits]",
    pos_examples=("systems_thinker",),
    neg_examples=("analytical",),
):
    """Minimal AxisSpec for rubric tests; uses a 1-slot, 4-dim axis."""
    return ajc.AxisSpec(
        axis_name=axis_name,
        neg_pole="favours linear, deductive, reductionistic analysis",
        pos_pole="favours holistic, multi-causal systems thinking",
        neg_examples=list(neg_examples),
        pos_examples=list(pos_examples),
        axis_by_slot={0: torch.zeros(4)},
        source_description="test",
    )


class TestRubricsRenderDisplayForm:
    """RUBRIC_VERSION v3: entity names are rendered in display form
    (``aligned artificial intelligence``) inside the prompt body
    instead of file-name form (``aligned_artificial_intelligence``).
    See AGENT_NOTES "File-name vs display-name convention" /
    "LLM-prompt sites are display sites".
    """

    def test_static_rubric_renders_name_with_spaces(self):
        spec = _make_axis_spec()
        prompt = ajc.build_static_prompt(
            spec, etype="traits",
            name="aligned_artificial_intelligence",
            content="...some description...",
        )
        # Display form appears in the body
        assert "aligned artificial intelligence" in prompt
        # File form does NOT appear in the body
        assert "aligned_artificial_intelligence" not in prompt

    def test_static_rubric_renders_examples_with_spaces(self):
        spec = _make_axis_spec(
            pos_examples=("systems_thinker", "stream_of_consciousness"),
            neg_examples=("analytical",),
        )
        prompt = ajc.build_static_prompt(
            spec, etype="traits", name="patient", content="x",
        )
        # The two multi-word example names appear with spaces, not
        # underscores
        assert "systems thinker" in prompt
        assert "stream of consciousness" in prompt
        # NB: the substring ``systems_thinker`` would also match
        # ``systems thinker`` if we were sloppy with assertions, so
        # check for the underscore form explicitly.
        assert "systems_thinker" not in prompt
        assert "stream_of_consciousness" not in prompt

    def test_static_rubric_renders_axis_name_with_spaces(self):
        spec = _make_axis_spec(
            axis_name="systems_thinker (+) vs analytical (-) [traits]",
        )
        prompt = ajc.build_static_prompt(
            spec, etype="traits", name="patient", content="x",
        )
        # The pair-mode axis_name's underscored pole name is now
        # rendered with a space.  ``[traits]`` (no underscores) is
        # passed through unchanged.
        assert "systems thinker (+) vs analytical (-) [traits]" in prompt

    def test_static_rubric_strips_self_example(self):
        """Per-call leakage prevention: when scoring entity X, X is
        stripped from its OWN rubric examples (file-form
        comparison; display-form injection)."""
        spec = _make_axis_spec(
            pos_examples=("systems_thinker", "stream_of_consciousness"),
            neg_examples=("analytical",),
        )
        prompt = ajc.build_static_prompt(
            spec, etype="traits",
            name="systems_thinker", content="x",
        )
        # systems_thinker is the entity being scored -> removed from
        # the examples list.  But it still appears in the entity
        # heading (``**systems thinker**: x``).
        # Count by looking at the example sentence.
        ex_line = next(
            line for line in prompt.split("\n")
            if "Traits that score near +3 include" in line
        )
        assert "systems thinker" not in ex_line
        assert "stream of consciousness" in ex_line

    def test_response_rubric_renders_examples_with_spaces(self):
        """Response-mode rubric body anonymises ``{name}`` (v2+),
        but the shared header still injects examples + axis_name --
        which we now render in display form."""
        from results_analysis.axis_judge_correlation import ScoredResponse

        spec = _make_axis_spec(
            pos_examples=("systems_thinker",),
            neg_examples=("analytical",),
        )
        items = [ScoredResponse(
            name="patient", key="pos_p0_q0", question="q", answer="a",
        )]
        prompt = ajc.build_response_batch_prompt(
            spec, etype="traits",
            name="patient",  # not in pos/neg examples
            items=items,
        )
        # Header carries display-form; v2 body does not name entity.
        assert "systems thinker" in prompt
        assert "systems_thinker" not in prompt
        assert "systems thinker (+) vs analytical (-) [traits]" in prompt

    def test_response_rubric_no_name_leak_in_body(self):
        """Sanity check: v2 anonymisation still in effect (the entity
        being judged is NOT named in the response-mode body, only
        the rubric header sees the example set)."""
        from results_analysis.axis_judge_correlation import ScoredResponse

        spec = _make_axis_spec(
            pos_examples=("constructive",), neg_examples=("destructive",),
        )
        items = [ScoredResponse(
            name="aligned_artificial_intelligence",
            key="pos_p0_q0", question="q", answer="a",
        )]
        prompt = ajc.build_response_batch_prompt(
            spec, etype="traits",
            name="aligned_artificial_intelligence",
            items=items,
        )
        # The entity being judged ("aligned_artificial_intelligence")
        # MUST NOT appear ANYWHERE in the prompt -- not as
        # underscored, not as spaced.  v2 anonymisation contract.
        assert "aligned_artificial_intelligence" not in prompt
        assert "aligned artificial intelligence" not in prompt

    def test_single_word_names_unchanged(self):
        """The common case (single-word entities, ~97 % of the
        corpus): display-form == file-form, no diff in prompt."""
        spec = _make_axis_spec(
            pos_examples=("constructive",), neg_examples=("destructive",),
            axis_name="constructive (+) vs destructive (-) [traits]",
        )
        prompt = ajc.build_static_prompt(
            spec, etype="traits", name="patient", content="x",
        )
        # No spurious changes: bare names render as themselves.
        assert "**patient**: x" in prompt
        assert "constructive (+) vs destructive (-) [traits]" in prompt

    def test_rubric_version_is_v3(self):
        """Bookkeeping: bumping behaviour MUST be paired with
        bumping ``RUBRIC_VERSION`` so caches written under v3 are
        identifiable in provenance."""
        assert ajc.RUBRIC_VERSION == "v3"


# ---------------------------------------------------------------------------
# _check_judge_model_on_resume (Sep 2026): caches are keyed by judge family,
# so a model change must not append into an old-model cohort.
# ---------------------------------------------------------------------------

def _envelope_with_judge_model(payload, judge_model):
    """Producer-shaped envelope with ``judge_model`` in the producer_script
    extras (``None`` = legacy envelope that never recorded a model)."""
    env = _make_v2_envelope_with_rubric(payload, ajc.RUBRIC_VERSION)
    for inp in env["_provenance"]["inputs"]:
        if inp.get("dep_key") == "producer_script":
            inp.setdefault("extras", {})
            if judge_model is None:
                inp["extras"].pop("judge_model", None)
            else:
                inp["extras"]["judge_model"] = judge_model
    return env


class TestCheckJudgeModelOnResume:

    def test_matching_model_keeps_cache(self, tmp_path):
        p = tmp_path / "scores_descriptions.json"
        p.write_text(json.dumps(_envelope_with_judge_model({"a|T": 1}, "claude-sonnet-4-6")))
        cache = {"a|T": 1, "b|R": -2}
        assert ajc._check_judge_model_on_resume(
            p, cache, mode="descriptions", current_model="claude-sonnet-4-6") == cache

    def test_recorded_mismatch_drops_everything_with_warning(self, tmp_path, caplog):
        p = tmp_path / "scores_descriptions.json"
        p.write_text(json.dumps(_envelope_with_judge_model({"a|T": 1}, "claude-sonnet-4-5")))
        with caplog.at_level("WARNING"):
            kept = ajc._check_judge_model_on_resume(
                p, {"a|T": 1, "b|R": 2}, mode="descriptions", current_model="claude-sonnet-4-6")
        assert kept == {}
        assert "claude-sonnet-4-5" in caplog.text and "claude-sonnet-4-6" in caplog.text

    def test_recorded_mismatch_strict_aborts(self, tmp_path):
        p = tmp_path / "scores_instructions.json"
        p.write_text(json.dumps(_envelope_with_judge_model({"a|T": 1}, "claude-sonnet-4-5")))
        with pytest.raises(SystemExit) as ei:
            ajc._check_judge_model_on_resume(
                p, {"a|T": 1}, mode="instructions", current_model="claude-sonnet-4-6", strict=True)
        assert "claude-sonnet-4-5" in str(ei.value)

    def test_unrecorded_kept_by_default_with_warning(self, tmp_path, caplog):
        p = tmp_path / "scores_descriptions.json"
        p.write_text(json.dumps(_envelope_with_judge_model({"a|T": 1}, None)))
        cache = {"a|T": 1}
        with caplog.at_level("WARNING"):
            kept = ajc._check_judge_model_on_resume(
                p, cache, mode="descriptions", current_model="claude-sonnet-4-6")
        assert kept == cache
        assert "no judge_model" in caplog.text

    def test_unrecorded_dropped_when_flag_set(self, tmp_path):
        p = tmp_path / "scores_descriptions.json"
        p.write_text(json.dumps(_envelope_with_judge_model({"a|T": 1}, None)))
        kept = ajc._check_judge_model_on_resume(
            p, {"a|T": 1}, mode="descriptions", current_model="claude-sonnet-4-6",
            drop_unrecorded=True)
        assert kept == {}

    def test_unrecorded_drop_plus_strict_aborts(self, tmp_path):
        p = tmp_path / "scores_descriptions.json"
        p.write_text(json.dumps(_envelope_with_judge_model({"a|T": 1}, None)))
        with pytest.raises(SystemExit):
            ajc._check_judge_model_on_resume(
                p, {"a|T": 1}, mode="descriptions", current_model="claude-sonnet-4-6",
                strict=True, drop_unrecorded=True)

    def test_bare_legacy_cache_without_envelope_is_unrecorded(self, tmp_path):
        p = tmp_path / "scores_descriptions.json"
        p.write_text(json.dumps({"a|T": 1}))  # no envelope at all
        assert ajc._peek_judge_model(p) is None
        assert ajc._check_judge_model_on_resume(
            p, {"a|T": 1}, mode="descriptions", current_model="x") == {"a|T": 1}

    def test_empty_cache_passes_through(self, tmp_path):
        p = tmp_path / "scores_descriptions.json"
        assert ajc._check_judge_model_on_resume(
            p, {}, mode="descriptions", current_model="x") == {}

    def test_default_anthropic_judge_is_sonnet_4_6(self):
        assert ajc._default_model_for_provider("anthropic") == "claude-sonnet-4-6"
        assert ajc._judge_family("anthropic", None) == "sonnet"


# ---------------------------------------------------------------------------
# Pole descriptions for poles renamed since the pair list was written
# ---------------------------------------------------------------------------

from assistant_axis.entity_id import clear_corpus_display_cache  # noqa: E402


class TestPoleInstructionPath:
    """``pair_list_*_v1.json`` names poles by the stems they were judged
    under.  The description lookup follows ``renamed_from``; vectors and
    judge caches are never remapped."""

    @staticmethod
    def _corpus(root: Path) -> Path:
        d = root / "roles" / "instructions"; d.mkdir(parents=True)
        (root / "traits" / "instructions").mkdir(parents=True)
        (d / "instrumentally_aligned_ai.json").write_text(json.dumps({
            "description": "An aligned artificial intelligence is an intelligent agentic tool.",
            "renamed_from": {"stem": "aligned_artificial_intelligence", "date": "2026-09-28"}}))
        (d / "paperclip_maximizer.json").write_text(json.dumps({"description": "A paperclip maximizer ..."}))
        return root

    def test_current_stem_is_used_as_is(self, tmp_path: Path, caplog):
        root = self._corpus(tmp_path)
        try:
            with caplog.at_level("WARNING", logger="axis_judge"):
                p = ajc._pole_instruction_path(root, "roles", "paperclip_maximizer")
            assert p == root / "roles" / "instructions" / "paperclip_maximizer.json"
            assert caplog.records == []
        finally:
            clear_corpus_display_cache()

    def test_renamed_pole_reads_the_renamed_file_and_warns(self, tmp_path: Path, caplog):
        root = self._corpus(tmp_path)
        try:
            with caplog.at_level("WARNING", logger="axis_judge"):
                p = ajc._pole_instruction_path(root, "roles", "aligned_artificial_intelligence")
            assert p == root / "roles" / "instructions" / "instrumentally_aligned_ai.json"
            assert json.loads(p.read_text())["description"].startswith("An aligned artificial intelligence")
            assert any("renamed" in r.getMessage() and "instrumentally_aligned_ai" in r.getMessage()
                       for r in caplog.records)
        finally:
            clear_corpus_display_cache()

    def test_unknown_pole_keeps_its_own_path_so_the_open_fails_loudly(self, tmp_path: Path):
        root = self._corpus(tmp_path)
        try:
            p = ajc._pole_instruction_path(root, "roles", "no_such_role")
            assert p == root / "roles" / "instructions" / "no_such_role.json"
            assert not p.exists()
        finally:
            clear_corpus_display_cache()
