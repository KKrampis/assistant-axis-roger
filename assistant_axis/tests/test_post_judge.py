"""Tests for steering/post_judge.py.

Most of post_judge's logic is delegation to RealJudgeDispatcher (which
is unit-tested separately).  Here we cover:
- spec loading from a synthetic experiment dir
- _group_records_by_strength bucketing (including baseline skip)
- _group_needs_work decision (which records still need a judge)

End-to-end main() with real dispatch is exercised on the pod.
"""
from __future__ import annotations

import importlib.util as _iu
import json
from pathlib import Path

import pytest


def _load_post_judge():
    """Load post_judge.py via importlib so tests don't need a steering
    package init.  Same shim trick test_steering_run_sweep.py uses."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "steering" / "post_judge.py"
    spec = _iu.spec_from_file_location("_post_judge", path)
    mod = _iu.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pj():
    return _load_post_judge()


def _make_record(*, strength=1.0, sign=+1, q_idx=0, persona_score=None,
                 effect_combined=None):
    judges = {
        "coherence": {"score": 0, "model": "test"},
        "strength_mean_coh": 0.5,
        "persona": (
            {"score": persona_score, "model": "test",
             "skipped_due_to_strength_mean_coh": False}
            if persona_score is not None else None
        ),
        "effect": (
            {"mode": "bidirectional", "combined": effect_combined,
             "skipped_due_to_strength_mean_coh": False}
            if effect_combined is not None else None
        ),
    }
    return {
        "strength": strength, "sign": sign,
        "slot": 3, "layer": 25,
        "question_idx": q_idx, "question": f"Q{q_idx}",
        "response": f"R{q_idx}", "n_tokens": 10,
        "judges": judges, "timing": {"gen_s": 1.0},
    }


# ---------------------------------------------------------------------------
# _group_records_by_strength
# ---------------------------------------------------------------------------

class TestGroupByStrength:
    def test_groups_by_sign_and_strength(self, pj):
        records = [
            _make_record(sign=+1, strength=1.0, q_idx=0),
            _make_record(sign=+1, strength=1.0, q_idx=1),
            _make_record(sign=+1, strength=2.0, q_idx=0),
            _make_record(sign=-1, strength=1.0, q_idx=0),
        ]
        groups = pj._group_records_by_strength(records)
        assert sorted(groups.keys()) == [(-1, 1.0), (1, 1.0), (1, 2.0)]
        assert len(groups[(1, 1.0)]) == 2
        assert len(groups[(1, 2.0)]) == 1
        assert len(groups[(-1, 1.0)]) == 1


# ---------------------------------------------------------------------------
# _group_needs_work
# ---------------------------------------------------------------------------

class TestGroupNeedsWork:
    def test_all_judges_present_no_work(self, pj):
        records = [
            _make_record(persona_score=2, effect_combined=1.0),
            _make_record(persona_score=3, effect_combined=2.0, q_idx=1),
        ]
        assert pj._group_needs_work(records, ["persona", "effect"]) is False

    def test_missing_persona_needs_work(self, pj):
        records = [
            _make_record(persona_score=None, effect_combined=1.0),
        ]
        assert pj._group_needs_work(records, ["persona", "effect"]) is True
        # If we're only running effect, persona absence doesn't matter
        assert pj._group_needs_work(records, ["effect"]) is False

    def test_missing_effect_needs_work(self, pj):
        records = [
            _make_record(persona_score=2, effect_combined=None),
        ]
        assert pj._group_needs_work(records, ["persona", "effect"]) is True
        assert pj._group_needs_work(records, ["persona"]) is False


# ---------------------------------------------------------------------------
# apply_corrected_stop_cutoff
# ---------------------------------------------------------------------------


def _write_cell_for_cutoff(
    cell_dir: Path,
    strengths_with_combined: list,
) -> None:
    """Build a fake cell dir with records.jsonl + summary.json.

    ``strengths_with_combined`` is a list of ``(strength, combined,
    n_questions)``: each strength gets ``n_questions`` records with
    ``judges.effect.combined`` set to ``combined`` (so the mean of
    ``|combined|`` equals ``|combined|``).
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for s, c, n_q in strengths_with_combined:
        for q in range(n_q):
            records.append({
                "strength": s, "sign": +1, "slot": 3, "layer": 25,
                "question_idx": q, "question": f"Q{q}", "response": f"R{q}",
                "judges": {
                    "coherence": {"score": 0},
                    "strength_mean_coh": 0.5,
                    "persona": {"score": 2,
                                "skipped_due_to_strength_mean_coh": False},
                    "effect": {
                        "mode": "bidirectional",
                        "combined": c,
                        "skipped_due_to_strength_mean_coh": False,
                    },
                },
            })
    (cell_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    (cell_dir / "summary.json").write_text(json.dumps({
        "slot": 3, "layer": 25, "sign": +1,
        "scan_mode": "bidirectional",
    }))


class TestApplyCorrectedStopCutoff:
    """Hygiene cutoff moves below-corrected-stop records out of
    records.jsonl into _excluded_records.jsonl, updates summary.json.
    """

    def test_no_records_excluded_when_signal_stays_above_threshold(
        self, pj, tmp_path,
    ):
        """All strengths have averaged-eff above threshold => nothing
        excluded; predicate never fires."""
        cell = tmp_path / "cell"
        _write_cell_for_cutoff(cell, [
            (1.0, 1.2, 3),
            (2.0, 1.5, 3),
            (4.0, 2.0, 3),
        ])
        n = pj.apply_corrected_stop_cutoff(
            cell_dir=cell, eff_stop_threshold=0.5, eff_stop_consecutive=2,
        )
        assert n == 0
        assert not (cell / "_excluded_records.jsonl").exists()

    def test_strengths_below_cutoff_moved(self, pj, tmp_path):
        """Descending walk: 4.0 (1.5), 2.0 (0.4), 1.0 (0.3), 0.5 (0.2).
        Two consecutive smaller strengths (1.0 and 0.5) both <= 0.5 ->
        cutoff fires at 2.0; strengths < 2.0 move out.
        """
        cell = tmp_path / "cell"
        _write_cell_for_cutoff(cell, [
            (0.5, 0.2, 3),
            (1.0, 0.3, 3),
            (2.0, 0.4, 3),
            (4.0, 1.5, 3),
        ])
        n = pj.apply_corrected_stop_cutoff(
            cell_dir=cell, eff_stop_threshold=0.5, eff_stop_consecutive=2,
        )
        # Strengths 0.5 and 1.0 (= 6 records) move to excluded.
        assert n == 6
        excluded_lines = (cell / "_excluded_records.jsonl").read_text().splitlines()
        excluded_lines = [json.loads(l) for l in excluded_lines if l.strip()]
        assert len(excluded_lines) == 6
        assert {r["strength"] for r in excluded_lines} == {0.5, 1.0}
        # records.jsonl has the kept rows only.
        kept_lines = (cell / "records.jsonl").read_text().splitlines()
        kept_lines = [json.loads(l) for l in kept_lines if l.strip()]
        assert {r["strength"] for r in kept_lines} == {2.0, 4.0}
        # summary.json records the exclusion footprint.
        summary = json.loads((cell / "summary.json").read_text())
        assert summary["excluded_strengths"] == [0.5, 1.0]
        assert summary["excluded_reason"] == "below_corrected_eff_stop"
        assert summary["excluded_eff_stop_threshold"] == 0.5
        assert summary["excluded_eff_stop_consecutive"] == 2

    def test_idempotent_second_call(self, pj, tmp_path):
        """Running the cutoff twice doesn't re-move already-excluded
        records: they're already out of records.jsonl, so there's
        nothing left to move."""
        cell = tmp_path / "cell"
        _write_cell_for_cutoff(cell, [
            (0.5, 0.2, 2),
            (1.0, 0.3, 2),
            (2.0, 0.4, 2),
            (4.0, 1.5, 2),
        ])
        n1 = pj.apply_corrected_stop_cutoff(
            cell_dir=cell, eff_stop_threshold=0.5, eff_stop_consecutive=2,
        )
        assert n1 == 4
        n2 = pj.apply_corrected_stop_cutoff(
            cell_dir=cell, eff_stop_threshold=0.5, eff_stop_consecutive=2,
        )
        assert n2 == 0  # nothing more to move

    def test_no_op_on_missing_records_file(self, pj, tmp_path):
        cell = tmp_path / "empty_cell"
        cell.mkdir()
        n = pj.apply_corrected_stop_cutoff(
            cell_dir=cell, eff_stop_threshold=0.5, eff_stop_consecutive=2,
        )
        assert n == 0

    def test_nan_eff_breaks_streak(self, pj, tmp_path):
        """A strength with mean_abs_eff = NaN (no effect.combined
        present, e.g. skipped at sweep time) RESETS the consecutive
        counter so we don't mark records on the OTHER side of the
        NaN gap as excluded.

        Strengths walked descending: 4.0 (1.5), 2.0 (NaN), 1.0 (0.3),
        0.5 (0.3).  The NaN at 2.0 resets the streak; below 2.0 we
        accumulate 0.3 + 0.3 => 2 consecutive => cutoff at 1.0;
        strength 0.5 moves out.
        """
        cell = tmp_path / "cell"
        records = []
        # Strength 4.0: signal above threshold.
        for q in range(2):
            records.append({
                "strength": 4.0, "sign": +1, "slot": 3, "layer": 25,
                "question_idx": q, "question": "q", "response": "r",
                "judges": {"effect": {"combined": 1.5}},
            })
        # Strength 2.0: NaN (no effect.combined).
        for q in range(2):
            records.append({
                "strength": 2.0, "sign": +1, "slot": 3, "layer": 25,
                "question_idx": q, "question": "q", "response": "r",
                "judges": {"effect": {
                    "combined": None,
                    "skipped_due_to_strength_mean_coh": True,
                }},
            })
        # Strengths 1.0 and 0.5: both below threshold.
        for s in (1.0, 0.5):
            for q in range(2):
                records.append({
                    "strength": s, "sign": +1, "slot": 3, "layer": 25,
                    "question_idx": q, "question": "q", "response": "r",
                    "judges": {"effect": {"combined": 0.3}},
                })
        cell.mkdir()
        (cell / "records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
        (cell / "summary.json").write_text(json.dumps({"slot": 3}))
        n = pj.apply_corrected_stop_cutoff(
            cell_dir=cell, eff_stop_threshold=0.5, eff_stop_consecutive=2,
        )
        # Only strength 0.5 (2 records) excluded; strength 1.0 was the
        # second-of-two-consecutive sub-threshold step so it stays as the
        # bias-floor anchor.
        assert n == 2
        excluded = [json.loads(l) for l in (cell / "_excluded_records.jsonl").read_text().splitlines() if l.strip()]
        assert {r["strength"] for r in excluded} == {0.5}


# ---------------------------------------------------------------------------
# load_experiment_specs
# ---------------------------------------------------------------------------

class TestLoadExperimentSpecs:
    def _setup_minimal_experiment(self, tmp_path: Path) -> Path:
        """Build an experiment dir with config.json + minimal data dirs."""
        exp = tmp_path / "exp"
        exp.mkdir()
        (exp / "config.json").write_text(json.dumps({
            "experiment_id": "exp",
            "model_name": "Qwen/Qwen3-32B",
            "output_dir": str(tmp_path),
            "axis_source": {
                "type": "role_transplant",
                "vectors_dir": str(tmp_path / "fake/vectors"),
                "role_from": "compassionate",
                "role_to": "callous",
            },
            "persona": {"type": "role", "role": "counselor"},
            "cells": [{"slot": 3, "layer": 25}],
            "questions_file": str(tmp_path / "questions.json"),
        }))
        return exp

    def _setup_data_dirs(self, tmp_path: Path) -> Path:
        data = tmp_path / "data"
        (data / "roles" / "instructions").mkdir(parents=True)
        (data / "traits" / "instructions").mkdir(parents=True)
        (data / "roles" / "instructions" / "counselor.json").write_text(
            json.dumps({"description": "provides emotional support"})
        )
        (data / "traits" / "instructions" / "callous.json").write_text(
            json.dumps({"description": "hard, indifferent"})
        )
        (data / "traits" / "instructions" / "compassionate.json").write_text(
            json.dumps({"description": "warm, sensitive"})
        )
        return data

    def test_loads_persona_and_steering(self, pj, tmp_path):
        exp = self._setup_minimal_experiment(tmp_path)
        data = self._setup_data_dirs(tmp_path)

        persona, steering = pj.load_experiment_specs(exp, data)
        assert persona.role == "counselor"
        assert "emotional support" in persona.description
        assert steering.pos_label == "callous"
        assert steering.neg_label == "compassionate"
        assert "hard" in steering.pos_description
        assert "warm" in steering.neg_description
        # Default axis_name format: "<neg>-<pos>"
        assert steering.axis_name == "compassionate-callous"

    def test_missing_persona_file_warns(self, pj, tmp_path, caplog):
        exp = self._setup_minimal_experiment(tmp_path)
        # No data dir at all -- everything missing
        data = tmp_path / "data"
        data.mkdir()

        persona, steering = pj.load_experiment_specs(exp, data)
        # Empty descriptions but loadable
        assert persona.role == "counselor"
        assert persona.description == ""
        assert steering.pos_description == ""
        assert steering.neg_description == ""

    def test_missing_config_raises(self, pj, tmp_path):
        empty = tmp_path / "no_config"
        empty.mkdir()
        with pytest.raises(SystemExit, match="config.json"):
            pj.load_experiment_specs(empty, tmp_path / "data")
