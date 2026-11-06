"""Tests for ``tools/audit_rubric_versions.py``.

The audit script walks a tree of judge caches and classifies each
into ``current`` / ``equivalent`` / ``drifted`` / ``legacy`` based on
its stamped rubric_version vs the current ``RUBRIC_VERSION`` and the
:mod:`assistant_axis.rubric_equivalence` registry.  These tests
exercise the classification path against synthetic caches in
``tmp_path`` so they're hermetic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

# Direct import of the tools/ script.  We pin the module via filesystem
# import; the script is set up to be runnable directly (it adds the
# repo root to sys.path itself).
import importlib.util  # noqa: E402
_MODNAME = "_audit_rubric_versions"
_spec = importlib.util.spec_from_file_location(
    _MODNAME,
    _REPO_ROOT / "tools" / "audit_rubric_versions.py",
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
# Register in sys.modules so @dataclass can resolve cls.__module__ at
# class-creation time (decorator inspects sys.modules[cls.__module__]).
sys.modules[_MODNAME] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
audit_one = _mod.audit_one
iter_judge_caches = _mod.iter_judge_caches
render_flat = _mod.render_flat
render_markdown = _mod.render_markdown
main = _mod.main

from assistant_axis.rubric_equivalence import RubricEquivalenceEdge  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: write synthetic judge-cache JSONs.
# ---------------------------------------------------------------------------

def _write_envelope(
    path: Path,
    payload: object,
    *,
    rubric_version: object = None,
    per_entity_rubric_versions: object = None,
) -> Path:
    """Write a v2-shaped envelope at ``path`` with controllable rubric
    stamps.  ``rubric_version=None`` omits the cohort stamp; passing
    ``per_entity_rubric_versions`` populates notes."""
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
            }
        ],
    }
    if per_entity_rubric_versions is not None:
        prov["notes"] = {
            "per_entity_rubric_versions": per_entity_rubric_versions,
        }
    envelope = {
        "schema_version": 2,
        "result": payload,
        "_provenance": prov,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2))
    return path


def _write_bare(path: Path, payload: object) -> Path:
    """Write a bare (non-envelope) JSON, mimicking pre-Phase-6 caches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# Per-cache classification (audit_one)
# ---------------------------------------------------------------------------

class TestAuditOne:
    def test_current_cache_classified_current(self, tmp_path):
        p = _write_envelope(
            tmp_path / "concise_vs_verbose" / "gpt" / "scores_descriptions.json",
            {"patient|R": 1, "stoic|T": 2},
            rubric_version="v3",
            per_entity_rubric_versions={
                "patient|R": "v3",
                "stoic|T": "v3",
            },
        )
        a = audit_one(p, current_rubric="v3", registry=[])
        assert a.status == "current"
        assert a.cohort_rubric == "v3"
        assert a.n_total == 2
        assert a.n_current == 2
        assert a.n_equivalent == 0
        assert a.n_drifted == 0
        assert a.axis == "concise_vs_verbose"
        assert a.mode == "descriptions"

    def test_drift_no_equivalence(self, tmp_path):
        p = _write_envelope(
            tmp_path / "ax" / "gpt" / "scores_descriptions.json",
            {"patient|R": 1, "stoic|T": 2},
            rubric_version="v2",
        )
        a = audit_one(p, current_rubric="v3", registry=[])
        assert a.status == "drifted"
        assert a.cohort_rubric == "v2"
        assert a.n_drifted == 2
        assert "v2" in a.drifted_by_version
        assert set(a.drifted_by_version["v2"]) == {"patient|R", "stoic|T"}

    def test_drift_with_equivalence_keeps_entries(self, tmp_path):
        registry = [
            RubricEquivalenceEdge(
                from_rubric="v2", to_rubric="v3",
                reason="test",
            ),
        ]
        p = _write_envelope(
            tmp_path / "ax" / "gpt" / "scores_descriptions.json",
            {"patient|R": 1, "stoic|T": 2},
            rubric_version="v2",
        )
        a = audit_one(p, current_rubric="v3", registry=registry)
        assert a.status == "equivalent"
        assert a.n_equivalent == 2
        assert a.n_drifted == 0
        assert a.drifted_by_version == {}

    def test_mixed_drift_and_equivalence(self, tmp_path):
        """Per-entity scope picks out exactly the entities the edge
        covers; the others drift."""
        registry = [
            RubricEquivalenceEdge(
                from_rubric="v2", to_rubric="v3",
                entity_ids=("patient|R",),
                reason="patient only",
            ),
        ]
        p = _write_envelope(
            tmp_path / "ax" / "gpt" / "scores_descriptions.json",
            {"patient|R": 1, "stoic|T": 2, "fresh|R": 3},
            rubric_version="v3",  # default for unstamped entries
            per_entity_rubric_versions={
                "patient|R": "v2",   # equivalent
                "stoic|T": "v2",     # drifted (scope misses)
                "fresh|R": "v3",     # current
            },
        )
        a = audit_one(p, current_rubric="v3", registry=registry)
        # Cache status is "drifted" because at least one entry drifted,
        # which is the priority-ordering rule (drifted > equivalent >
        # current) the audit advertises.
        assert a.status == "drifted"
        assert a.n_current == 1
        assert a.n_equivalent == 1
        assert a.n_drifted == 1
        assert a.drifted_by_version == {"v2": ["stoic|T"]}

    def test_legacy_envelope_without_stamp(self, tmp_path):
        """Envelope-wrapped cache lacking a rubric_version stamp =
        legacy.  Still counts entries for context."""
        p = _write_envelope(
            tmp_path / "ax" / "gpt" / "scores_descriptions.json",
            {"a": 1, "b": 2, "c": 3},
            rubric_version=None,
        )
        a = audit_one(p, current_rubric="v3", registry=[])
        assert a.status == "legacy"
        assert a.n_total == 3, "legacy caches should still report size"
        assert a.cohort_rubric is None
        assert a.axis == "ax"
        assert a.mode == "descriptions"

    def test_legacy_bare_json(self, tmp_path):
        """Non-envelope bare JSON = legacy, with top-level entries
        counted."""
        p = _write_bare(
            tmp_path / "ax" / "gpt" / "scores_descriptions.json",
            {"a": 1, "b": 2},
        )
        a = audit_one(p, current_rubric="v3", registry=[])
        assert a.status == "legacy"
        assert a.n_total == 2

    def test_unparseable_file_classified_legacy(self, tmp_path):
        p = tmp_path / "ax" / "gpt" / "scores_descriptions.json"
        p.parent.mkdir(parents=True)
        p.write_text("not valid json{")
        a = audit_one(p, current_rubric="v3", registry=[])
        assert a.status == "legacy"
        assert a.n_total == 0

    def test_per_entity_stamps_override_cohort(self, tmp_path):
        """Per-entity stamps take precedence over the cohort stamp."""
        p = _write_envelope(
            tmp_path / "ax" / "gpt" / "scores_descriptions.json",
            {"patient|R": 1, "stoic|T": 2},
            rubric_version="v2",  # cohort says v2
            per_entity_rubric_versions={
                "patient|R": "v3",  # but per-entity says v3 = current
                "stoic|T": "v2",    # per-entity confirms v2 = drift
            },
        )
        a = audit_one(p, current_rubric="v3", registry=[])
        assert a.status == "drifted"  # stoic|T still drifts
        assert a.n_current == 1
        assert a.drifted_by_version == {"v2": ["stoic|T"]}

    def test_mode_inferred_for_responses(self, tmp_path):
        p = _write_envelope(
            tmp_path / "ax" / "gpt_responses_roles_b7" / "scores_responses.json",
            {"patient": 1},
            rubric_version="v3",
        )
        a = audit_one(p, current_rubric="v3", registry=[])
        assert a.mode == "responses"
        assert a.axis == "ax"

    def test_mode_inferred_for_instructions(self, tmp_path):
        p = _write_envelope(
            tmp_path / "ax" / "gpt" / "scores_instructions.json",
            {"patient|R": 1},
            rubric_version="v3",
        )
        a = audit_one(p, current_rubric="v3", registry=[])
        assert a.mode == "instructions"


# ---------------------------------------------------------------------------
# Walk / file iteration
# ---------------------------------------------------------------------------

class TestIterJudgeCaches:
    def test_finds_standard_cache_filenames(self, tmp_path):
        # Things that should be picked up.
        wanted = [
            tmp_path / "a" / "gpt" / "scores_descriptions.json",
            tmp_path / "a" / "gpt" / "scores_instructions.json",
            tmp_path / "a" / "gpt_responses_roles_b7" / "scores_responses.json",
            tmp_path / "a" / "haiku" / "scores_descriptions__rubric_v1.json",
        ]
        # Things that should NOT be picked up.
        unwanted = [
            tmp_path / "a" / "correlations.json",
            tmp_path / "a" / "gpt" / "config.json",
            tmp_path / "a" / "gaps.json",
            tmp_path / "rho_by_layer.json",
        ]
        for f in wanted + unwanted:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("{}")
        found = set(iter_judge_caches([tmp_path]))
        assert found == set(wanted)

    def test_missing_root_silently_skipped(self, tmp_path):
        # Doesn't raise on absent root.
        assert list(iter_judge_caches([tmp_path / "missing"])) == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    @pytest.fixture
    def sample_audits(self, tmp_path):
        # Build a small mixed pile of audits.
        p1 = _write_envelope(
            tmp_path / "ax1" / "gpt" / "scores_descriptions.json",
            {"a|R": 1}, rubric_version="v3",
            per_entity_rubric_versions={"a|R": "v3"},
        )
        p2 = _write_envelope(
            tmp_path / "ax2" / "gpt" / "scores_descriptions.json",
            {"a|R": 1, "b|T": 2}, rubric_version="v2",
        )
        p3 = _write_envelope(
            tmp_path / "ax3" / "gpt" / "scores_descriptions.json",
            {"a": 1}, rubric_version=None,
        )
        return [
            audit_one(p1, current_rubric="v3", registry=[]),
            audit_one(p2, current_rubric="v3", registry=[]),
            audit_one(p3, current_rubric="v3", registry=[]),
        ]

    def test_markdown_summary_includes_all_statuses(self, sample_audits, tmp_path):
        body = render_markdown(
            sample_audits, current_rubric="v3",
            roots=[tmp_path], status_filter=None,
        )
        assert "Rubric-version audit" in body
        assert "RUBRIC_VERSION``: ``v3``" in body
        # Summary table reflects 1/0/1/1 (current / equivalent / drifted / legacy).
        assert "| current | 1 |" in body
        assert "| drifted | 1 |" in body
        assert "| legacy | 1 |" in body

    def test_markdown_filtered_to_status(self, sample_audits, tmp_path):
        body = render_markdown(
            sample_audits, current_rubric="v3",
            roots=[tmp_path], status_filter="drifted",
        )
        # Only the drifted cache should appear in the per-cache section.
        assert "ax2/gpt/scores_descriptions.json" in body
        # Heading reflects the filter.
        assert "Caches with status = ``drifted``" in body

    def test_flat_output_one_line_per_cache(self, sample_audits):
        body = render_flat(sample_audits, status_filter=None)
        lines = [ln for ln in body.splitlines() if ln.strip()]
        assert len(lines) == 3
        statuses = [ln.split()[0] for ln in lines]
        assert sorted(statuses) == ["current", "drifted", "legacy"]

    def test_flat_filtered(self, sample_audits):
        body = render_flat(sample_audits, status_filter="drifted")
        lines = [ln for ln in body.splitlines() if ln.strip()]
        assert len(lines) == 1
        assert lines[0].startswith("drifted")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_default_current_rubric_parses_from_file(self):
        # Should pick up the in-tree RUBRIC_VERSION constant.
        rv = _mod._default_current_rubric()
        assert rv.startswith("v")

    def test_main_with_output_file(self, tmp_path, capsys, monkeypatch):
        # Point root at an empty dir so the walk finds nothing; we just
        # want to exercise the end-to-end CLI path.
        out = tmp_path / "report.md"
        exit_code = main([
            "--root", str(tmp_path),
            "--current", "v3",
            "--output", str(out),
        ])
        assert exit_code == 0
        assert out.exists()
        body = out.read_text()
        assert "Rubric-version audit" in body
        # Summary tally line goes to stderr.
        err = capsys.readouterr().err
        assert "Wrote" in err

    def test_main_requires_current(self, tmp_path, monkeypatch):
        # Force _default_current_rubric to return "" so the loud-fail
        # path is exercised.
        monkeypatch.setattr(_mod, "_default_current_rubric", lambda: "")
        with pytest.raises(SystemExit, match="rubric_version"):
            main(["--root", str(tmp_path)])
