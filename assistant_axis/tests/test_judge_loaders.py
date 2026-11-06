"""Tests for ``assistant_axis.judge_loaders``.

The conditional-provenance contract is the most important property
to pin down here: an artifact that resolves every entity from
``_b7_t3`` must NOT register ``_b10_q9`` as a dependency, even when
the loader opened ``_b10_q9`` to look up an entity that turned out
to be available in ``_b7_t3``.
"""
import json
from pathlib import Path

import pytest

from assistant_axis.judge_loaders import (
    EXPECTED_STATIC_SCHEMA_VERSION,
    StaleSchemaError,
    cohort_dirname,
    dep_key_for_cohort,
    filename_for_rubric,
    load_projections,
    load_response_scores,
    load_static_correlations,
    load_static_scores,
    peek_rubric_version,
    subsample_method_for_dirname,
)
from assistant_axis.plot_metadata import json_metadata
from assistant_axis.provenance import InputSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score_entry(mean: float, n_batches: int = 10, n_items: int = 70) -> dict:
    """Minimal valid response-score entry shape (matches existing on-disk
    format: ``mean_score``, ``std_score``, ``n_batches``,
    ``n_total_items``, ``per_batch``, ``target_batch_size``)."""
    return {
        "mean_score": mean,
        "std_score": 0.5,
        "n_batches": n_batches,
        "n_total_items": n_items,
        "per_batch": {},
        "target_batch_size": 7,
        "batch_scores": [mean] * n_batches,
    }


def _write_cohort(
    experiments_root: Path,
    axis: str,
    cohort_dir: str,
    filename: str,
    payload: dict,
) -> Path:
    """Write a bare (non-enveloped) JSON cohort cache; returns the
    full path."""
    p = experiments_root / axis / cohort_dir / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Plumbing helpers
# ---------------------------------------------------------------------------

def test_cohort_dirname_gpt():
    assert cohort_dirname("gpt", "roles", 7) == "gpt_responses_roles_b7"
    assert cohort_dirname("gpt", "traits", 10) == "gpt_responses_traits_b10"


def test_cohort_dirname_haiku():
    assert (
        cohort_dirname("haiku", "roles", 7) == "haiku_responses_roles_b7_t3"
    )
    assert (
        cohort_dirname("haiku", "traits", 10)
        == "haiku_responses_traits_b10_q9"
    )


def test_cohort_dirname_sonnet():
    assert (
        cohort_dirname("sonnet", "traits", 10)
        == "sonnet_responses_traits_b10_q9"
    )


def test_cohort_dirname_unknown_combination():
    with pytest.raises(ValueError, match="no known cohort directory"):
        cohort_dirname("haiku", "roles", 5)
    with pytest.raises(ValueError, match="no known cohort directory"):
        cohort_dirname("sonnet", "roles", 7)


def test_cohort_dirname_normalises_kind():
    """Long-form 'roles', short 'R', singular 'role' — all fine."""
    assert cohort_dirname("gpt", "roles", 7) == "gpt_responses_roles_b7"
    assert cohort_dirname("gpt", "R", 7) == "gpt_responses_roles_b7"
    assert cohort_dirname("gpt", "role", 7) == "gpt_responses_roles_b7"


def test_filename_for_rubric():
    assert filename_for_rubric("v2") == "scores_responses.json"
    assert filename_for_rubric("v1") == "scores_responses__rubric_v1.json"
    assert filename_for_rubric("V2") == "scores_responses.json"


def test_filename_for_rubric_rejects_unknown():
    with pytest.raises(ValueError, match="unknown rubric"):
        filename_for_rubric("v3")


def test_subsample_method_decoder():
    assert subsample_method_for_dirname("gpt_responses_traits_b7") == "full_volume"
    assert subsample_method_for_dirname("gpt_responses_roles_b10") == "full_volume"
    assert (
        subsample_method_for_dirname("haiku_responses_roles_b10_q9")
        == "uniform_mod9"
    )
    assert (
        subsample_method_for_dirname("haiku_responses_traits_b7_t3")
        == "tiered_t3"
    )
    assert (
        subsample_method_for_dirname("haiku_responses_roles_b7_t5")
        == "tiered_t5"
    )


def test_subsample_method_unknown_suffix():
    """Future suffixes should fall through to the unknown tag rather
    than mis-classify."""
    assert (
        subsample_method_for_dirname("haiku_responses_roles_b7_z42")
        == "unknown:z42"
    )


def test_dep_key_format():
    assert (
        dep_key_for_cohort("haiku_responses_roles_b7_t3", "v2")
        == "haiku_responses_roles_b7_t3_v2"
    )


# ---------------------------------------------------------------------------
# load_response_scores: the conditional-provenance contract
# ---------------------------------------------------------------------------

@pytest.fixture
def axis_root(tmp_path: Path) -> Path:
    """An empty experiments-root directory under tmp_path."""
    return tmp_path / "axis_judge_experiments"


def test_haiku_all_entities_in_b7(axis_root):
    """Phase-5d-rejudged axis × cohort with every entity present in
    `_b7_t3`: only `_b7_t3` registered as a dependency."""
    axis = "concise_vs_verbose"
    payload_b7 = {
        "patient": _make_score_entry(0.5),
        "stoic": _make_score_entry(0.6),
        "ascetic": _make_score_entry(0.7),
    }
    _write_cohort(
        axis_root, axis, "haiku_responses_roles_b7_t3",
        "scores_responses.json", payload_b7,
    )

    inputs: list[InputSpec] = []
    scores, sources = load_response_scores(
        axis, "roles", "haiku", "v2",
        experiments_root=axis_root, inputs=inputs,
    )

    assert set(scores) == {"patient", "stoic", "ascetic"}
    for name in scores:
        assert sources[name] == "haiku_responses_roles_b7_t3"
        assert scores[name]["source_cohort"] == "haiku_responses_roles_b7_t3"
        assert scores[name]["subsample_method"] == "tiered_t3"
    assert len(inputs) == 1
    assert inputs[0].dep_key == "haiku_responses_roles_b7_t3_v2"
    assert inputs[0].extras["resolution"] == "preferred"


def test_haiku_all_entities_in_b10(axis_root):
    """Legacy axis case: an axis where 5d was skipped (or ``_b7_t3``
    doesn't exist yet) so only ``_b10_q9`` is on disk.  The default
    Haiku ``prefer_b = (7, 10)`` falls back to b10 when b7 is
    missing (the canonical behaviour since 2026-05-22).
    """
    axis = "egalitarian_vs_elitist"
    payload_b10 = {
        "patient": _make_score_entry(0.5),
        "stoic": _make_score_entry(0.6),
    }
    _write_cohort(
        axis_root, axis, "haiku_responses_roles_b10_q9",
        "scores_responses.json", payload_b10,
    )

    inputs: list[InputSpec] = []
    scores, sources = load_response_scores(
        axis, "roles", "haiku", "v2",
        experiments_root=axis_root, inputs=inputs,
    )

    assert set(scores) == {"patient", "stoic"}
    for name in scores:
        assert sources[name] == "haiku_responses_roles_b10_q9"
        assert scores[name]["subsample_method"] == "uniform_mod9"
    assert len(inputs) == 1
    assert inputs[0].dep_key == "haiku_responses_roles_b10_q9_v2"
    assert inputs[0].extras["resolution"] == "fallback"
    assert inputs[0].extras["B"] == 10


def test_haiku_mixed_b7_and_b10(axis_root):
    """The canonical case: 9 collisions + ~20 escalated entities
    resolve to ``_b7_t3``, the rest to ``_b10_q9``.  Both cohorts
    registered.  Default Haiku prefer_b = (7, 10) gives this
    automatic fallback behaviour.
    """
    axis = "concise_vs_verbose"
    payload_b7 = {
        "patient": _make_score_entry(0.5),  # collision name, in b7
        "stoic": _make_score_entry(0.6),    # collision name, in b7
    }
    payload_b10 = {
        "teacher": _make_score_entry(0.4),  # not surgically rejudged
        "doctor": _make_score_entry(0.45),
        # Note: ``patient`` / ``stoic`` may also exist in b10 (legacy
        # data); this should resolve to b7 since it has priority.
        "patient": _make_score_entry(-99.0),
        "stoic": _make_score_entry(-99.0),
    }
    _write_cohort(
        axis_root, axis, "haiku_responses_roles_b7_t3",
        "scores_responses.json", payload_b7,
    )
    _write_cohort(
        axis_root, axis, "haiku_responses_roles_b10_q9",
        "scores_responses.json", payload_b10,
    )

    inputs: list[InputSpec] = []
    scores, sources = load_response_scores(
        axis, "roles", "haiku", "v2",
        experiments_root=axis_root, inputs=inputs,
    )

    assert set(scores) == {"patient", "stoic", "teacher", "doctor"}
    assert sources["patient"] == "haiku_responses_roles_b7_t3"
    assert sources["stoic"] == "haiku_responses_roles_b7_t3"
    assert sources["teacher"] == "haiku_responses_roles_b10_q9"
    assert sources["doctor"] == "haiku_responses_roles_b10_q9"
    # b7 wins for the collision-name overlap (sentinel -99 must NOT
    # bleed in from the b10 stub).
    assert scores["patient"]["mean_score"] == 0.5
    assert scores["stoic"]["mean_score"] == 0.6

    dep_keys = {spec.dep_key for spec in inputs}
    assert dep_keys == {
        "haiku_responses_roles_b7_t3_v2",
        "haiku_responses_roles_b10_q9_v2",
    }
    # Resolution tag distinguishes the preferred B from the fallback.
    by_key = {spec.dep_key: spec for spec in inputs}
    assert by_key["haiku_responses_roles_b7_t3_v2"].extras["resolution"] == "preferred"
    assert by_key["haiku_responses_roles_b10_q9_v2"].extras["resolution"] == "fallback"


def test_haiku_b7_covers_everything_b10_has_no_unique_entities(axis_root):
    """Subtle case: ``_b7_t3`` covers every entity also present in
    ``_b10_q9`` -- ``_b10_q9`` was *opened* but contributed nothing
    to the result.  Must NOT be registered (otherwise the whole
    conditional-provenance contract is broken).
    """
    axis = "concise_vs_verbose"
    payload_b7 = {
        "patient": _make_score_entry(0.5),
        "stoic": _make_score_entry(0.6),
    }
    # Same names, different (worse) scores -- if the loader broke and
    # registered b10, that'd at least be a wrong dependency.  But it
    # might also overwrite values, so we use distinct scores.
    payload_b10 = {
        "patient": _make_score_entry(-99.0),
        "stoic": _make_score_entry(-99.0),
    }
    _write_cohort(
        axis_root, axis, "haiku_responses_roles_b7_t3",
        "scores_responses.json", payload_b7,
    )
    _write_cohort(
        axis_root, axis, "haiku_responses_roles_b10_q9",
        "scores_responses.json", payload_b10,
    )

    inputs: list[InputSpec] = []
    scores, sources = load_response_scores(
        axis, "roles", "haiku", "v2",
        experiments_root=axis_root, inputs=inputs,
    )

    # b7 wins the value race.
    assert scores["patient"]["mean_score"] == 0.5
    assert scores["stoic"]["mean_score"] == 0.6
    # Conditional-provenance: b10 was opened but contributed zero
    # entities to ``scores`` — it MUST NOT be registered.
    dep_keys = {spec.dep_key for spec in inputs}
    assert dep_keys == {"haiku_responses_roles_b7_t3_v2"}, (
        f"Conditional-provenance contract violated: b10 should not be "
        f"registered when it contributed no entities to the result. "
        f"Got dep_keys={dep_keys}"
    )


def test_haiku_no_cohorts_at_all_returns_empty(axis_root):
    """No cohort files on disk: empty result, no inputs registered."""
    inputs: list[InputSpec] = []
    scores, sources = load_response_scores(
        "concise_vs_verbose", "roles", "haiku", "v2",
        experiments_root=axis_root, inputs=inputs,
    )
    assert scores == {}
    assert sources == {}
    assert inputs == []


def test_inputs_none_does_not_raise(axis_root):
    """Passing ``inputs=None`` must work (not every consumer cares
    about provenance)."""
    axis = "concise_vs_verbose"
    _write_cohort(
        axis_root, axis, "haiku_responses_roles_b7_t3",
        "scores_responses.json",
        {"patient": _make_score_entry(0.5)},
    )
    scores, sources = load_response_scores(
        axis, "roles", "haiku", "v2",
        experiments_root=axis_root, inputs=None,
    )
    assert "patient" in scores
    assert sources["patient"] == "haiku_responses_roles_b7_t3"


def test_gpt_v2_default_prefer_b_is_b7_only(axis_root):
    """After Phase 5c, GPT v2 reads exclusively from ``_b7``;
    explicitly does NOT fall back to the broken b10 v2 caches."""
    axis = "concise_vs_verbose"
    _write_cohort(
        axis_root, axis, "gpt_responses_roles_b7", "scores_responses.json",
        {"patient": _make_score_entry(0.5)},
    )
    # A sibling b10 directory exists with deferred-but-wrong data.
    # The default prefer_b for ("gpt", "v2") is (7,) -- so the loader
    # must NOT touch b10 at all.
    _write_cohort(
        axis_root, axis, "gpt_responses_roles_b10", "scores_responses.json",
        {"patient": _make_score_entry(-99.0), "ghost": _make_score_entry(-99.0)},
    )

    inputs: list[InputSpec] = []
    scores, sources = load_response_scores(
        axis, "roles", "gpt", "v2",
        experiments_root=axis_root, inputs=inputs,
    )

    assert set(scores) == {"patient"}, (
        f"GPT v2 default must read only b7; got entities {set(scores)}"
    )
    assert scores["patient"]["mean_score"] == 0.5
    dep_keys = {spec.dep_key for spec in inputs}
    assert dep_keys == {"gpt_responses_roles_b7_v2"}


def test_gpt_v1_default_reads_b10(axis_root):
    """GPT v1 reference data lives at B=10 (the legitimate, full-
    volume v1 reference)."""
    axis = "concise_vs_verbose"
    _write_cohort(
        axis_root, axis, "gpt_responses_traits_b10",
        "scores_responses__rubric_v1.json",
        {"absolutist": _make_score_entry(0.42)},
    )
    inputs: list[InputSpec] = []
    scores, _ = load_response_scores(
        axis, "traits", "gpt", "v1",
        experiments_root=axis_root, inputs=inputs,
    )
    assert "absolutist" in scores
    assert inputs[0].dep_key == "gpt_responses_traits_b10_v1"


def test_sonnet_default_reads_b10_only(axis_root):
    axis = "concise_vs_verbose"
    _write_cohort(
        axis_root, axis, "sonnet_responses_roles_b10_q9",
        "scores_responses.json",
        {"patient": _make_score_entry(0.5)},
    )
    inputs: list[InputSpec] = []
    scores, sources = load_response_scores(
        axis, "roles", "sonnet", "v2",
        experiments_root=axis_root, inputs=inputs,
    )
    assert "patient" in scores
    assert sources["patient"] == "sonnet_responses_roles_b10_q9"
    assert inputs[0].dep_key == "sonnet_responses_roles_b10_q9_v2"


def test_explicit_prefer_b_override(axis_root):
    """A caller can pass an explicit ``prefer_b`` to override the
    default, e.g. to hit the legacy b10 v2 GPT data on purpose for a
    forensic comparison."""
    axis = "concise_vs_verbose"
    _write_cohort(
        axis_root, axis, "gpt_responses_roles_b10", "scores_responses.json",
        {"patient": _make_score_entry(0.5)},
    )
    inputs: list[InputSpec] = []
    scores, _ = load_response_scores(
        axis, "roles", "gpt", "v2",
        experiments_root=axis_root, prefer_b=(10,), inputs=inputs,
    )
    assert "patient" in scores
    assert inputs[0].extras["B"] == 10


def test_explicit_empty_prefer_b_rejected(axis_root):
    with pytest.raises(ValueError, match="non-empty"):
        load_response_scores(
            "concise_vs_verbose", "roles", "haiku", "v2",
            experiments_root=axis_root, prefer_b=(),
        )


def test_unknown_judge_or_rubric_rejected(axis_root):
    with pytest.raises(ValueError, match="no default prefer_b"):
        load_response_scores(
            "concise_vs_verbose", "roles", "claude", "v2",
            experiments_root=axis_root,
        )
    with pytest.raises(ValueError, match="unknown rubric"):
        load_response_scores(
            "concise_vs_verbose", "roles", "haiku", "v3",
            experiments_root=axis_root,
        )


def test_score_entry_mutation_is_on_a_copy(axis_root):
    """The on-disk dict mustn't accrete metadata across calls."""
    axis = "concise_vs_verbose"
    cohort_path = _write_cohort(
        axis_root, axis, "haiku_responses_roles_b7_t3",
        "scores_responses.json",
        {"patient": _make_score_entry(0.5)},
    )
    scores1, _ = load_response_scores(
        axis, "roles", "haiku", "v2",
        experiments_root=axis_root,
    )
    # Re-read raw (no envelope so json.load is fine).
    raw = json.loads(cohort_path.read_text())
    assert "source_cohort" not in raw["patient"], (
        "load_response_scores mutated the on-disk-shaped payload "
        "in memory; the read-time fields must live on a copy."
    )
    scores2, _ = load_response_scores(
        axis, "roles", "haiku", "v2",
        experiments_root=axis_root,
    )
    assert scores1["patient"]["mean_score"] == scores2["patient"]["mean_score"]


def test_provenance_envelope_unwraps(axis_root):
    """Cohort caches that DO carry a provenance envelope unwrap to
    their ``result`` payload, like the production caches do."""
    axis = "concise_vs_verbose"
    enveloped_payload = {
        "_provenance": {
            "schema_version": 1,
            "produced_by": "test",
            "produced_at": "2026-01-01T00:00:00+00:00",
            "git_sha": "deadbeef",
            "inputs": [],
            "extras": {},
        },
        "result": {"patient": _make_score_entry(0.5)},
    }
    p = axis_root / axis / "haiku_responses_roles_b7_t3" / "scores_responses.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(enveloped_payload), encoding="utf-8")

    scores, sources = load_response_scores(
        axis, "roles", "haiku", "v2", experiments_root=axis_root,
    )
    assert scores["patient"]["mean_score"] == 0.5
    assert sources["patient"] == "haiku_responses_roles_b7_t3"


# ---------------------------------------------------------------------------
# Static-mode loaders (Phase 4): schema_version: 2 enforcement
# ---------------------------------------------------------------------------

def _write_v2_envelope(
    path: Path,
    payload: object,
    *,
    schema_version: int | None = 2,
) -> Path:
    """Write a v2 envelope (``schema_version`` at top level alongside
    ``result`` + ``_provenance``) — mirrors what the producer's
    ``_save_json`` writes after Phase 4.
    """
    envelope = json_metadata(payload, inputs=[], title="test")
    if schema_version is not None:
        envelope["schema_version"] = schema_version
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True))
    return path


def _write_v1_bare_legacy(path: Path, payload: object) -> Path:
    """Write the legacy v1 shape: bare-name keys, no envelope, no
    schema_version.  This is what every existing static-mode cache on
    disk looks like before the Phase 4 producer fix lands.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


# load_static_scores -----------------------------------------------------

def test_load_static_scores_v2_clean(tmp_path):
    """Schema-v2 cache with disambiguated keys → clean load + InputSpec
    appended."""
    p = tmp_path / "axis" / "gpt" / "scores_descriptions.json"
    _write_v2_envelope(
        p, {"patient|R": 1, "patient|T": -2, "stoic|R": 0, "stoic|T": -1},
    )
    inputs: list[InputSpec] = []
    payload, spec, _check = load_static_scores(p, inputs=inputs)
    assert payload == {"patient|R": 1, "patient|T": -2, "stoic|R": 0, "stoic|T": -1}
    assert spec.path  # repo-relative path captured
    assert spec.kind == "file"
    assert len(inputs) == 1
    assert inputs[0].dep_key == "static_scores:scores_descriptions.json"


def test_load_static_scores_v1_bare_rejected(tmp_path):
    """Legacy v1 file (no envelope, bare-name keys) → loud reject with a
    regenerate command in the error message.

    This is the canonical failure mode the schema-version contract is
    designed to catch.
    """
    p = tmp_path / "axis" / "gpt" / "scores_descriptions.json"
    _write_v1_bare_legacy(p, {"patient": 1, "stoic": 0})
    with pytest.raises(StaleSchemaError) as exc:
        load_static_scores(p)
    assert exc.value.recorded is None
    assert exc.value.expected == EXPECTED_STATIC_SCHEMA_VERSION
    msg = str(exc.value)
    assert "schema" in msg.lower()
    assert "Regenerate via:" in msg
    assert "axis_judge_correlation.py" in msg
    # Strong: the error message must guide the operator to a working
    # regenerate command (not just say "stale schema").
    assert "--score_descriptions" in msg
    assert str(p.parent) in msg


def test_load_static_scores_bogus_schema_version_rejected(tmp_path):
    """A future or wrong schema_version (e.g. 99) → loud reject."""
    p = tmp_path / "axis" / "gpt" / "scores_instructions.json"
    _write_v2_envelope(p, {"patient|R": 1}, schema_version=99)
    with pytest.raises(StaleSchemaError) as exc:
        load_static_scores(p)
    assert exc.value.recorded == 99
    assert exc.value.expected == 2
    assert "--score_instructions" in str(exc.value)


def test_load_static_scores_missing_schema_version_rejected(tmp_path):
    """v2 envelope shape but no top-level ``schema_version`` field → reject."""
    p = tmp_path / "axis" / "gpt" / "scores_descriptions.json"
    _write_v2_envelope(p, {"patient|R": 1}, schema_version=None)
    with pytest.raises(StaleSchemaError) as exc:
        load_static_scores(p)
    assert exc.value.recorded is None


def test_load_static_scores_v2_with_bare_keys_rejected(tmp_path):
    """Schema-v2 declared but every key is bare (forgot-to-disambiguate
    on the producer side) → loud reject by the sanity check."""
    p = tmp_path / "axis" / "gpt" / "scores_descriptions.json"
    _write_v2_envelope(p, {"patient": 1, "stoic": 0, "teacher": 2})
    with pytest.raises(StaleSchemaError) as exc:
        load_static_scores(p)
    assert "no entity-id" in str(exc.value).lower()


def test_load_static_scores_v2_empty_payload_passes(tmp_path):
    """Early-stage runs with zero entries should pass the disambiguation
    sanity check (vacuously) so partial caches mid-run don't blow up."""
    p = tmp_path / "axis" / "gpt" / "scores_descriptions.json"
    _write_v2_envelope(p, {})
    payload, _spec, _check = load_static_scores(p)
    assert payload == {}


def test_load_static_scores_file_not_found(tmp_path):
    p = tmp_path / "axis" / "gpt" / "scores_descriptions.json"
    with pytest.raises(FileNotFoundError):
        load_static_scores(p)


def test_load_static_scores_inputs_none_does_not_raise(tmp_path):
    """``inputs=None`` is the docstring-default; must work."""
    p = tmp_path / "axis" / "gpt" / "scores_descriptions.json"
    _write_v2_envelope(p, {"patient|R": 1})
    payload, _spec, _check = load_static_scores(p, inputs=None)
    assert payload == {"patient|R": 1}


# load_projections -------------------------------------------------------

def test_load_projections_v2_clean(tmp_path):
    p = tmp_path / "axis" / "gpt" / "projections.json"
    payload = {
        "0": {"patient|R": {"raw": 0.5, "whitened": 0.4},
              "patient|T": {"raw": -0.3, "whitened": -0.2}},
        "1": {"patient|R": {"raw": 0.6, "whitened": 0.5},
              "patient|T": {"raw": -0.4, "whitened": -0.3}},
    }
    _write_v2_envelope(p, payload)
    inputs: list[InputSpec] = []
    out, spec, _check = load_projections(p, inputs=inputs)
    assert out == payload
    assert len(inputs) == 1
    assert "projections" in inputs[0].dep_key


def test_load_projections_v1_rejected(tmp_path):
    p = tmp_path / "axis" / "gpt" / "projections.json"
    _write_v1_bare_legacy(p, {"0": {"patient": {"raw": 0.5, "whitened": 0.4}}})
    with pytest.raises(StaleSchemaError) as exc:
        load_projections(p)
    assert "axis_judge_correlation.py" in str(exc.value)


def test_load_projections_v2_with_bare_inner_keys_rejected(tmp_path):
    """v2 envelope but the per-slot inner dict has bare-name keys →
    sanity-check failure."""
    p = tmp_path / "axis" / "gpt" / "projections.json"
    _write_v2_envelope(p, {
        "0": {"patient": {"raw": 0.5, "whitened": 0.4},
              "stoic": {"raw": -0.3, "whitened": -0.2}},
    })
    with pytest.raises(StaleSchemaError) as exc:
        load_projections(p)
    assert "no entity-id" in str(exc.value).lower()


# load_static_correlations -----------------------------------------------

def test_load_static_correlations_v2_clean(tmp_path):
    p = tmp_path / "axis" / "gpt" / "correlations.json"
    payload = {
        "descriptions": {
            "0": {
                "raw": {
                    "rho": 0.5, "p": 0.01, "n": 12,
                    "names": ["patient|R", "patient|T", "stoic|R"],
                    "scores": [1, -2, 0],
                    "projections": [0.5, -0.3, 0.1],
                },
                "whitened": {"rho": 0.6, "p": 0.005, "n": 12,
                             "names": [], "scores": [], "projections": []},
            },
        },
    }
    _write_v2_envelope(p, payload)
    inputs: list[InputSpec] = []
    out, spec, _check = load_static_correlations(p, inputs=inputs)
    assert out == payload
    assert len(inputs) == 1
    assert "correlations" in inputs[0].dep_key


def test_load_static_correlations_v1_rejected(tmp_path):
    p = tmp_path / "axis" / "gpt" / "correlations.json"
    _write_v1_bare_legacy(p, {"descriptions": {"0": {"raw": {"rho": 0}}}})
    with pytest.raises(StaleSchemaError):
        load_static_correlations(p)


# StaleSchemaError -------------------------------------------------------

def test_stale_schema_error_message_format():
    """The error message must be operator-friendly: schema mismatch +
    full regenerate command on a separate line for easy copy-paste."""
    p = Path("/tmp/foo/scores_descriptions.json")
    err = StaleSchemaError(
        path=p, recorded=None, expected=2,
        regenerate_via="uv run python results_analysis/x.py --foo",
    )
    msg = str(err)
    assert "v1-or-unset" in msg  # human-friendly recorded version label
    assert "expected v2" in msg
    assert "Regenerate via:" in msg
    assert "uv run python results_analysis/x.py --foo" in msg


def test_stale_schema_error_attributes():
    """Programmatic access to the structured fields, for callers that
    want to dispatch on the recorded version (e.g. attempt an
    auto-regen for some recorded versions, hard-fail for others)."""
    p = Path("/tmp/foo")
    err = StaleSchemaError(p, recorded=99, expected=2, regenerate_via="cmd")
    assert err.path == p
    assert err.recorded == 99
    assert err.expected == 2
    assert err.regenerate_via == "cmd"


# ---------------------------------------------------------------------------
# peek_rubric_version: cache-side rubric drift detection
# ---------------------------------------------------------------------------

def _write_envelope_with_rubric(
    path: Path,
    payload: object,
    *,
    rubric_version: object,
    per_entity_rubric_versions: object = None,
) -> Path:
    """Hand-write a v2 envelope with a controlled rubric_version label
    stamped into the producer_script InputSpec's extras (mirrors what
    the real producer's ``_save_json`` writes).

    Passing ``rubric_version=None`` omits the extras key entirely
    (legacy pre-Phase-6 cache); pass an empty dict / object to test
    weird-but-valid shapes.

    When ``per_entity_rubric_versions`` is provided, it's stamped into
    ``_provenance.notes.per_entity_rubric_versions``, matching the
    post-May-2026 per-entity stamping schema written by the real
    producer for static and response modes.
    """
    extras: dict = {}
    if rubric_version is not None:
        extras["rubric_version"] = rubric_version
    prov: dict = {
        "schema_version": "1.0",
        "produced_by": {"cmd": "test"},
        "produced_at": "2026-05-11T00:00:00+00:00",
        "inputs_sha256": "test",
        "inputs": [
            {
                "dep_key": "producer_script",
                "path": "results_analysis/axis_judge_correlation.py",
                "fingerprint": "v1:2026-05-11T00:00:00+00:00@123",
                "kind": "file",
                "extras": extras,
            },
        ],
    }
    if per_entity_rubric_versions is not None:
        prov["notes"] = {"per_entity_rubric_versions": per_entity_rubric_versions}
    envelope = {
        "schema_version": 2,
        "result": payload,
        "_provenance": prov,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2))
    return path


def test_peek_rubric_version_returns_recorded_label(tmp_path):
    p = _write_envelope_with_rubric(
        tmp_path / "scores_descriptions.json", {}, rubric_version="v3",
    )
    assert peek_rubric_version(p) == "v3"


def test_peek_rubric_version_returns_any_recorded_string(tmp_path):
    """The check is label-based: any string is returned verbatim
    (callers compare to the active RUBRIC_VERSION)."""
    p = _write_envelope_with_rubric(
        tmp_path / "scores.json", {}, rubric_version="v2",
    )
    assert peek_rubric_version(p) == "v2"


def test_peek_rubric_version_missing_returns_none(tmp_path):
    """Envelope present, but the producer_script extras don't record
    rubric_version (legacy Phase-4 cache predating the stamp)."""
    p = _write_envelope_with_rubric(
        tmp_path / "scores.json", {}, rubric_version=None,
    )
    assert peek_rubric_version(p) is None


def test_peek_rubric_version_non_string_returns_none(tmp_path):
    """Defensive: integer or non-string rubric_version shouldn't crash;
    we just refuse to surface it (the strict producer-side path can
    treat None as a non-match against the active label)."""
    p = _write_envelope_with_rubric(
        tmp_path / "scores.json", {}, rubric_version=3,
    )
    assert peek_rubric_version(p) is None


def test_peek_rubric_version_no_envelope_returns_none(tmp_path):
    """Bare JSON (legacy pre-Phase-6) → None."""
    p = tmp_path / "scores.json"
    p.write_text(json.dumps({"patient|R": 1, "patient|T": -2}))
    assert peek_rubric_version(p) is None


def test_peek_rubric_version_no_inputs_list_returns_none(tmp_path):
    """Envelope present but inputs list absent → None."""
    p = tmp_path / "scores.json"
    p.write_text(json.dumps({
        "schema_version": 2,
        "result": {},
        "_provenance": {"produced_by": {"cmd": "x"}},
    }))
    assert peek_rubric_version(p) is None


def test_peek_rubric_version_no_producer_script_input_returns_none(tmp_path):
    """Inputs list exists but no producer_script entry → None."""
    p = tmp_path / "scores.json"
    p.write_text(json.dumps({
        "schema_version": 2,
        "result": {},
        "_provenance": {
            "inputs": [
                {"dep_key": "some_other_dep", "extras": {"rubric_version": "v3"}},
            ],
        },
    }))
    assert peek_rubric_version(p) is None


def test_peek_rubric_version_file_missing_returns_none(tmp_path):
    """No file → None (no exception)."""
    assert peek_rubric_version(tmp_path / "doesnt_exist.json") is None


def test_peek_rubric_version_unparseable_returns_none(tmp_path):
    """Corrupt JSON → None (no exception)."""
    p = tmp_path / "scores.json"
    p.write_text("not valid json{")
    assert peek_rubric_version(p) is None


def test_peek_rubric_version_accepts_str_path(tmp_path):
    p = _write_envelope_with_rubric(
        tmp_path / "scores.json", {}, rubric_version="v3",
    )
    assert peek_rubric_version(str(p)) == "v3"


# ---------------------------------------------------------------------------
# peek_per_entity_rubric_versions + rubric_version_report:
# consumer-side reporting of rubric-version freshness.
# ---------------------------------------------------------------------------

from assistant_axis.judge_loaders import (  # noqa: E402
    RubricVersionReport,
    peek_per_entity_rubric_versions,
    rubric_version_report,
)


class TestPeekPerEntityRubricVersions:
    """Consumer-side mirror of the producer's per-entity stamp peek."""

    def test_returns_map_when_present(self, tmp_path):
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json",
            {"patient|R": 1},
            rubric_version="v3",
            per_entity_rubric_versions={"patient|R": "v3", "stoic|T": "v2"},
        )
        assert peek_per_entity_rubric_versions(p) == {
            "patient|R": "v3",
            "stoic|T": "v2",
        }

    def test_returns_empty_when_absent(self, tmp_path):
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json", {}, rubric_version="v3",
        )
        assert peek_per_entity_rubric_versions(p) == {}

    def test_returns_empty_when_file_missing(self, tmp_path):
        assert peek_per_entity_rubric_versions(tmp_path / "missing.json") == {}

    def test_returns_empty_on_corrupt_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json{")
        assert peek_per_entity_rubric_versions(p) == {}

    def test_ignores_non_string_values(self, tmp_path):
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json", {},
            rubric_version="v3",
            per_entity_rubric_versions={"patient|R": "v3", "weird": 9},
        )
        assert peek_per_entity_rubric_versions(p) == {"patient|R": "v3"}

    def test_accepts_str_path(self, tmp_path):
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json", {},
            rubric_version="v3",
            per_entity_rubric_versions={"patient|R": "v3"},
        )
        assert peek_per_entity_rubric_versions(str(p)) == {"patient|R": "v3"}


class TestRubricVersionReport:
    """Consumer-side per-entity drift audit returning a structured report."""

    def test_all_current_no_drift(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [],
        )
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json",
            {"patient|R": 1, "stoic|T": 2},
            rubric_version="v3",
            per_entity_rubric_versions={"patient|R": "v3", "stoic|T": "v3"},
        )
        report = rubric_version_report(p, current_rubric="v3")
        assert isinstance(report, RubricVersionReport)
        assert bool(report) is True
        assert report.n_total == 2
        assert report.n_current == 2
        assert report.n_equivalent == 0
        assert report.drifted == {}
        assert report.n_drifted == 0

    def test_pure_drift_no_equivalence(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [],
        )
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json",
            {"patient|R": 1, "stoic|T": 2},
            rubric_version="v2",
            per_entity_rubric_versions={"patient|R": "v2", "stoic|T": "v2"},
        )
        report = rubric_version_report(p, current_rubric="v3")
        assert bool(report) is False
        assert report.n_current == 0
        assert report.n_equivalent == 0
        assert report.drifted == {"v2": ["patient|R", "stoic|T"]}
        assert report.n_drifted == 2

    def test_falls_back_to_cohort_stamp_when_no_per_entity(
        self, tmp_path, monkeypatch,
    ):
        """No per-entity stamps → cohort stamp drives the decision for
        every entry."""
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [],
        )
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json",
            {"patient|R": 1, "stoic|T": 2},
            rubric_version="v2",
        )
        report = rubric_version_report(p, current_rubric="v3")
        assert report.n_drifted == 2
        assert set(report.drifted["v2"]) == {"patient|R", "stoic|T"}

    def test_legacy_no_stamps_treats_as_current(
        self, tmp_path, monkeypatch,
    ):
        """When neither cohort nor per-entity stamps exist, the report
        treats every entry as current (mirrors the producer's
        legacy-fallback semantics)."""
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [],
        )
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json",
            {"patient|R": 1},
            rubric_version=None,
        )
        report = rubric_version_report(p, current_rubric="v3")
        assert bool(report) is True
        assert report.n_current == 1
        assert report.drifted == {}

    def test_equivalence_keeps_drifted_entries(
        self, tmp_path, monkeypatch,
    ):
        from assistant_axis.rubric_equivalence import RubricEquivalenceEdge
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [
                RubricEquivalenceEdge(
                    from_rubric="v2", to_rubric="v3",
                    reason="test",
                )
            ],
        )
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json",
            {"patient|R": 1, "stoic|T": 2},
            rubric_version="v2",
        )
        report = rubric_version_report(p, current_rubric="v3")
        assert bool(report) is True
        assert report.n_equivalent == 2
        assert report.n_current == 0
        assert report.drifted == {}

    def test_partial_drift_with_partial_equivalence(
        self, tmp_path, monkeypatch,
    ):
        """One entity covered by an equivalence edge; the other isn't.
        The report should distinguish current/equivalent/drifted."""
        from assistant_axis.rubric_equivalence import RubricEquivalenceEdge
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [
                RubricEquivalenceEdge(
                    from_rubric="v2", to_rubric="v3",
                    entity_ids=("patient|R",),
                    reason="test",
                )
            ],
        )
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json",
            {"patient|R": 1, "stoic|T": 2, "fresh|R": 3},
            rubric_version="v3",
            per_entity_rubric_versions={
                "patient|R": "v2",   # equivalent
                "stoic|T": "v2",     # drifted (entity scope misses)
                "fresh|R": "v3",     # current
            },
        )
        report = rubric_version_report(p, current_rubric="v3")
        assert report.n_total == 3
        assert report.n_current == 1
        assert report.n_equivalent == 1
        assert report.n_drifted == 1
        assert report.drifted == {"v2": ["stoic|T"]}

    def test_axis_mode_scope_passed_to_registry(
        self, tmp_path, monkeypatch,
    ):
        """The axis and mode args propagate to is_equivalent so scoped
        edges resolve correctly."""
        from assistant_axis.rubric_equivalence import RubricEquivalenceEdge
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [
                RubricEquivalenceEdge(
                    from_rubric="v2", to_rubric="v3",
                    modes=("responses",),
                    except_axes=("systems_thinker_vs_analytical",),
                    reason="single-pole axes byte-identical",
                )
            ],
        )
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json",
            {"patient": 1},
            rubric_version="v2",
        )
        # Right scope → equivalent.
        report = rubric_version_report(
            p, current_rubric="v3",
            axis="concise_vs_verbose", mode="responses",
        )
        assert bool(report) is True
        # Wrong axis → drift.
        report = rubric_version_report(
            p, current_rubric="v3",
            axis="systems_thinker_vs_analytical", mode="responses",
        )
        assert bool(report) is False
        # Wrong mode → drift.
        report = rubric_version_report(
            p, current_rubric="v3",
            axis="concise_vs_verbose", mode="descriptions",
        )
        assert bool(report) is False

    def test_entries_arg_restricts_audit_scope(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [],
        )
        p = _write_envelope_with_rubric(
            tmp_path / "scores.json",
            {"patient|R": 1, "stoic|T": 2},
            rubric_version="v2",
        )
        report = rubric_version_report(
            p, current_rubric="v3",
            entries=["patient|R"],
        )
        # stoic|T excluded from audit.
        assert report.n_total == 1
        assert report.drifted == {"v2": ["patient|R"]}

    def test_missing_file_yields_empty_report(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "assistant_axis.rubric_equivalence.load_registry",
            lambda *a, **kw: [],
        )
        report = rubric_version_report(
            tmp_path / "missing.json", current_rubric="v3",
        )
        assert report.n_total == 0
        assert bool(report) is True  # no entries = no drift detected
