"""Tests for assistant_axis.cherrypick.

Covers the default filter, custom expression filter, record iteration
across cell subdirs, and the summary table format.  CLI argv handling
is exercised via main() with monkeypatched argv + capsys.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from assistant_axis import cherrypick


def _record(*, strength=1.0, sign=+1, q_idx=0,
            smc=0.5, persona_score=3, effect_combined=2.5,
            response="resp", question="q?",
            experiment="exp_a", cell="s3_l25_+1") -> dict:
    return {
        "strength": strength, "sign": sign,
        "slot": 3, "layer": 25,
        "question_idx": q_idx, "question": question, "response": response,
        "_experiment": experiment, "_cell": cell,
        "judges": {
            "coherence": {"score": int(round(smc)), "model": "test"},
            "strength_mean_coh": smc,
            "persona": {"score": persona_score, "model": "test",
                        "skipped_due_to_strength_mean_coh": False},
            "effect": {"mode": "bidirectional", "combined": effect_combined,
                       "skipped_due_to_strength_mean_coh": False},
        },
    }


# ---------------------------------------------------------------------------
# default_filter
# ---------------------------------------------------------------------------

class TestDefaultFilter:
    def test_passes_all_thresholds(self):
        r = _record(smc=0.7, persona_score=3, effect_combined=2.5)
        assert cherrypick.default_filter(r) is True

    def test_rejects_high_smc(self):
        r = _record(smc=1.5)
        assert cherrypick.default_filter(r) is False

    def test_rejects_low_persona(self):
        r = _record(persona_score=1)
        assert cherrypick.default_filter(r) is False

    def test_rejects_weak_effect(self):
        r = _record(effect_combined=1.5)
        assert cherrypick.default_filter(r) is False

    def test_negative_effect_passes_when_magnitude_high(self):
        r = _record(effect_combined=-2.5)
        assert cherrypick.default_filter(r) is True

    def test_thresholds_configurable(self):
        r = _record(smc=1.4, persona_score=2, effect_combined=1.5)
        # Default thresholds reject this
        assert cherrypick.default_filter(r) is False
        # Looser thresholds accept it
        assert cherrypick.default_filter(
            r, max_strength_mean_coh=1.5,
            min_persona=2, min_abs_effect=1.5,
        ) is True

    def test_missing_smc_rejected(self):
        r = _record()
        del r["judges"]["strength_mean_coh"]
        assert cherrypick.default_filter(r) is False

    def test_missing_persona_rejected(self):
        r = _record()
        r["judges"]["persona"] = None
        assert cherrypick.default_filter(r) is False

    def test_missing_effect_combined_rejected(self):
        r = _record()
        r["judges"]["effect"] = {"mode": "bidirectional", "combined": None}
        assert cherrypick.default_filter(r) is False


# ---------------------------------------------------------------------------
# make_expr_filter
# ---------------------------------------------------------------------------

class TestExprFilter:
    def test_simple_threshold(self):
        check = cherrypick.make_expr_filter("abs_eff >= 2.5")
        assert check(_record(effect_combined=2.5)) is True
        assert check(_record(effect_combined=2.0)) is False
        assert check(_record(effect_combined=-3.0)) is True

    def test_combined_strength_filter(self):
        check = cherrypick.make_expr_filter(
            "smc < 1 and persona['score'] >= 3 and combined > 0"
        )
        assert check(_record(smc=0.5, persona_score=3,
                             effect_combined=1.0)) is True
        assert check(_record(smc=1.0, persona_score=3,
                             effect_combined=1.0)) is False
        assert check(_record(smc=0.5, persona_score=2,
                             effect_combined=1.0)) is False
        assert check(_record(smc=0.5, persona_score=3,
                             effect_combined=-0.5)) is False

    def test_filter_returns_falsy_on_exception(self):
        # Reaching into a missing key should not raise; the wrapper
        # logs and returns False so the record is dropped.
        check = cherrypick.make_expr_filter("record['nope']['x']")
        assert check(_record()) is False

    def test_strength_and_sign_bound(self):
        check = cherrypick.make_expr_filter("strength >= 4 and sign == -1")
        assert check(_record(strength=8, sign=-1)) is True
        assert check(_record(strength=8, sign=+1)) is False
        assert check(_record(strength=2, sign=-1)) is False


# ---------------------------------------------------------------------------
# iter_records: walking experiment dirs
# ---------------------------------------------------------------------------

class TestIterRecords:
    def _setup_experiment(self, root: Path, *, cells: dict) -> Path:
        """Create an experiment dir layout with per-cell records.jsonl.

        cells is {cell_dir_name: [records...]}.  Baselines are auto-created.
        """
        exp_dir = root / "exp_a"
        exp_dir.mkdir()
        (exp_dir / "config.json").write_text("{}")
        # Baselines (skipped during iteration)
        baselines = exp_dir / "baselines"
        baselines.mkdir()
        with (baselines / "records.jsonl").open("w") as f:
            f.write(json.dumps({"strength": 0, "sign": 0,
                                "question_idx": 0, "question": "q",
                                "response": "b"}) + "\n")
        # Cells
        for cell_name, recs in cells.items():
            cell_dir = exp_dir / cell_name
            cell_dir.mkdir()
            with (cell_dir / "records.jsonl").open("w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        return exp_dir

    def test_yields_all_cell_records_annotated(self, tmp_path):
        cells = {
            "s3_l25_+1": [{"strength": 1, "sign": +1, "question_idx": 0,
                           "question": "q1", "response": "r1"}],
            "s3_l25_-1": [{"strength": 1, "sign": -1, "question_idx": 0,
                           "question": "q1", "response": "r2"}],
        }
        exp_dir = self._setup_experiment(tmp_path, cells=cells)
        records = list(cherrypick.iter_records([exp_dir]))
        # Two cells x 1 record each; baselines NOT included
        assert len(records) == 2
        for r in records:
            assert r["_experiment"] == "exp_a"
            assert r["_cell"] in {"s3_l25_+1", "s3_l25_-1"}

    def test_skips_baselines(self, tmp_path):
        exp_dir = self._setup_experiment(
            tmp_path, cells={"s3_l25_+1": [{"strength": 1, "sign": +1,
                                            "question_idx": 0,
                                            "question": "q",
                                            "response": "r"}]}
        )
        records = list(cherrypick.iter_records([exp_dir]))
        # Baselines have a record but iter_records skips that subdir.
        assert all(r["_cell"] != "baselines" for r in records)

    def test_missing_dir_warns_and_continues(self, tmp_path, caplog):
        exp_a = self._setup_experiment(
            tmp_path, cells={"s3_l25_+1": [{"strength": 1, "sign": +1,
                                            "question_idx": 0,
                                            "question": "q",
                                            "response": "r"}]}
        )
        # iter_records with a mix of real and missing dirs
        records = list(cherrypick.iter_records(
            [exp_a, tmp_path / "does_not_exist"],
        ))
        assert len(records) == 1


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------

class TestSummarise:
    def test_empty(self):
        out = cherrypick.summarise([])
        assert "matched 0" in out

    def test_counts_per_cell(self):
        records = [
            _record(strength=1.0, q_idx=0, cell="s3_l25_+1"),
            _record(strength=1.0, q_idx=1, cell="s3_l25_+1"),
            _record(strength=2.0, q_idx=0, cell="s3_l25_+1"),
            _record(strength=1.0, q_idx=0, cell="s3_l25_-1"),
        ]
        out = cherrypick.summarise(records)
        assert "matched 4" in out
        assert "s3_l25_+1" in out
        assert "s3_l25_-1" in out


# ---------------------------------------------------------------------------
# main() -- end-to-end via tmp_path
# ---------------------------------------------------------------------------

class TestMain:
    def _setup_with_records(self, root: Path, *, records: list) -> Path:
        exp_dir = root / "exp_a"
        exp_dir.mkdir()
        (exp_dir / "config.json").write_text("{}")
        cell_dir = exp_dir / "s3_l25_+1"
        cell_dir.mkdir()
        with (cell_dir / "records.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return exp_dir

    def test_main_default_filter_writes_output(self, tmp_path, monkeypatch,
                                               capsys):
        # 3 records: 1 passes default filter, 2 don't
        recs = [
            _record(smc=0.5, persona_score=3, effect_combined=2.5,
                    q_idx=0),  # passes
            _record(smc=1.5, persona_score=3, effect_combined=2.5,
                    q_idx=1),  # smc too high
            _record(smc=0.5, persona_score=1, effect_combined=2.5,
                    q_idx=2),  # persona too low
        ]
        # Strip the _experiment/_cell so iter_records adds them naturally
        for r in recs:
            r.pop("_experiment", None)
            r.pop("_cell", None)
        exp_dir = self._setup_with_records(tmp_path, records=recs)

        out_path = tmp_path / "matched.jsonl"
        monkeypatch.setattr(sys, "argv", [
            "cherrypick",
            "--experiment_dirs", str(exp_dir),
            "--output", str(out_path),
            "--print-N", "0",
        ])
        cherrypick.main()
        captured = capsys.readouterr()
        assert "matched 1 record" in captured.out

        lines = [json.loads(l)
                 for l in out_path.read_text().splitlines()
                 if l.strip()]
        assert len(lines) == 1
        assert lines[0]["question_idx"] == 0
