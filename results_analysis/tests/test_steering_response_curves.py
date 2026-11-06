"""Tests for results_analysis/steering_response_curves.py.

Covers the per-cell aggregation logic that turns a steering experiment
directory into one ``CellSign`` per ``(slot, layer, sign)`` with mean
effect under each of the four filter regimes (a/b/c/d).  The plotting
side is exercised by the smoke test in ``test_plot_smoke`` which
verifies the PNG renders without error -- visual content is best
checked manually.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    """Load steering_response_curves.py via importlib so the tests don't
    need a ``results_analysis`` package init beyond what already exists.
    """
    spec = importlib.util.spec_from_file_location(
        "_steering_response_curves",
        REPO_ROOT / "results_analysis" / "steering_response_curves.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Pre-register so dataclasses defined inside resolve module name correctly.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# Synthetic experiment dir builder
# ---------------------------------------------------------------------------

def _make_record(
    *, strength: float, sign: int, question_idx: int,
    coh: Optional[int], rp: Optional[int], eff: Optional[float],
    persona_skipped: bool = False, effect_skipped: bool = False,
) -> Dict[str, Any]:
    """Build one record matching the runner's on-disk schema, with the
    judge sub-blocks structured so the score extractors round-trip
    correctly (in particular, persona/effect blocks always exist on
    real cell records even when their inner score is null).
    """
    return {
        "strength": strength, "sign": sign, "question_idx": question_idx,
        "question": f"q{question_idx}",
        "response": f"resp-{sign:+d}{strength}-q{question_idx}",
        "n_tokens": 50,
        "judges": {
            "coherence": None if coh is None else {"score": coh},
            "persona": {
                "score": rp,
                "skipped_due_to_strength_mean_coh": persona_skipped,
            },
            "effect": {
                "combined": eff,
                "skipped_due_to_strength_mean_coh": effect_skipped,
            },
        },
    }


def _write_cell(
    cell_dir: Path, sign: int, *,
    strengths: List[float], records_per_strength: List[List[Dict[str, Any]]],
    up_blocked_reason: Optional[str] = None,
) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    # Flatten records into a single jsonl.
    flat: List[Dict[str, Any]] = []
    for recs in records_per_strength:
        flat.extend(recs)
    (cell_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in flat) + "\n"
    )
    summary = {
        "slot": 6, "layer": 25, "sign": sign,
        "scan_mode": "bidirectional",
        "up_blocked_reason": up_blocked_reason,
        "stopped_at_strength": max(strengths) if strengths else 0.0,
    }
    (cell_dir / "summary.json").write_text(json.dumps(summary))


@pytest.fixture
def tiny_cell(tmp_path: Path):
    """One ``s6_l25_+1`` directory with 3 strengths × 3 questions = 9 records.

    Strength patterns chosen to differentiate the 4 filter means:
      s=1.0: all three q's are coh=0 rp=3 eff=+1.0
      s=2.0: q0 coh=0 rp=3 eff=+2.0
             q1 coh=1 rp=3 eff=+2.0   (filtered out by coh-0 regime)
             q2 coh=0 rp=2 eff=+2.0   (filtered out by rp-3 regime)
      s=4.0: all three are coh=2, persona+effect skipped
             (no contribution to any filter)
    """
    cell_dir = tmp_path / "s6_l25_+1"
    records_per_strength = [
        # s=1.0 -- clean across the board
        [
            _make_record(strength=1.0, sign=+1, question_idx=q,
                         coh=0, rp=3, eff=1.0)
            for q in range(3)
        ],
        # s=2.0 -- variations
        [
            _make_record(strength=2.0, sign=+1, question_idx=0,
                         coh=0, rp=3, eff=2.0),
            _make_record(strength=2.0, sign=+1, question_idx=1,
                         coh=1, rp=3, eff=2.0),
            _make_record(strength=2.0, sign=+1, question_idx=2,
                         coh=0, rp=2, eff=2.0),
        ],
        # s=4.0 -- runner skipped persona+effect at this strength.
        [
            _make_record(strength=4.0, sign=+1, question_idx=q,
                         coh=2, rp=None, eff=None,
                         persona_skipped=True, effect_skipped=True)
            for q in range(3)
        ],
    ]
    _write_cell(
        cell_dir, +1,
        strengths=[1.0, 2.0, 4.0],
        records_per_strength=records_per_strength,
        up_blocked_reason="incoherent",
    )
    return cell_dir


# ---------------------------------------------------------------------------
# aggregate_cell_sign
# ---------------------------------------------------------------------------

class TestAggregateCellSign:
    def test_parses_dir_name(self, mod, tiny_cell):
        cs = mod.aggregate_cell_sign(tiny_cell)
        assert cs.slot == 6
        assert cs.layer == 25
        assert cs.sign == +1
        assert cs.up_blocked_reason == "incoherent"

    def test_strengths_sorted_ascending_and_trailing_skipped_trimmed(
        self, mod, tiny_cell
    ):
        """The tiny_cell fixture has 3 strengths (1.0 / 2.0 / 4.0) but
        s=4.0 has rp+effect judging skipped (no usable eff data).  The
        aggregator trims that trailing skipped strength so the curve's
        rightmost x=0 always corresponds to a real data point.  Without
        the trim, every cell would render a gap at the cliff and Roger
        would (rightly) ask why the curves don't reach x=0.
        """
        cs = mod.aggregate_cell_sign(tiny_cell)
        strengths = [a.strength for a in cs.aggs]
        assert strengths == [1.0, 2.0]

    def test_trailing_skipped_stashed_in_extra_aggs(self, mod, tiny_cell):
        """Trimmed trailing strengths (eff-skipped but coh-judged)
        land on ``cs.extra_aggs`` so the bottom-row coh panel can
        render them past the cliff.  Order is ascending magnitude.
        """
        cs = mod.aggregate_cell_sign(tiny_cell)
        extra_strengths = [a.strength for a in cs.extra_aggs]
        assert extra_strengths == [4.0]
        # And the mean_coh field is populated for the bottom-row plot.
        assert cs.extra_aggs[0].mean_coh_all == pytest.approx(2.0)
        # rp_all is None because tiny_cell's s=4.0 records have
        # rp=None (judging was skipped at that strength).
        assert cs.extra_aggs[0].mean_rp_all is None

    def test_trailing_skipped_does_not_trim_interior_skipped(
        self, mod, tmp_path
    ):
        """Only TRAILING skipped strengths are trimmed.  A skipped
        strength followed by a non-skipped one stays in place (creates
        an interior gap in the curve), preserving the bidirectional
        scan order on disk.
        """
        cell_dir = tmp_path / "s6_l25_+1"
        cell_dir.mkdir()
        records = [
            _make_record(strength=1.0, sign=+1, question_idx=0,
                         coh=0, rp=3, eff=1.0),
            # Interior skipped strength: judges null but a stronger
            # strength below has usable data.
            _make_record(strength=2.0, sign=+1, question_idx=0,
                         coh=2, rp=None, eff=None,
                         persona_skipped=True, effect_skipped=True),
            _make_record(strength=4.0, sign=+1, question_idx=0,
                         coh=0, rp=3, eff=4.0),
        ]
        (cell_dir / "records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
        (cell_dir / "summary.json").write_text(
            json.dumps({"slot": 6, "layer": 25, "sign": +1,
                        "scan_mode": "bidirectional"})
        )
        cs = mod.aggregate_cell_sign(cell_dir)
        strengths = [a.strength for a in cs.aggs]
        # All three preserved.
        assert strengths == [1.0, 2.0, 4.0]
        # Interior gap retained.
        assert cs.aggs[1].mean_eff_all is None

    def test_baseline_zero_strength_excluded(self, mod, tmp_path):
        """The aggregator drops strength=0 rows -- those are baseline
        records, not steering data points."""
        cell_dir = tmp_path / "s6_l25_+1"
        cell_dir.mkdir()
        baseline = _make_record(
            strength=0.0, sign=0, question_idx=0,
            coh=0, rp=3, eff=0.0,
        )
        steered = _make_record(
            strength=1.0, sign=+1, question_idx=0,
            coh=0, rp=3, eff=1.0,
        )
        (cell_dir / "records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in [baseline, steered]) + "\n"
        )
        cs = mod.aggregate_cell_sign(cell_dir)
        assert [a.strength for a in cs.aggs] == [1.0]

    def test_filter_means_all(self, mod, tiny_cell):
        cs = mod.aggregate_cell_sign(tiny_cell)
        # s=1.0: all 3 q's contribute eff=1.0 -> mean=1.0
        # s=2.0: all 3 q's contribute eff=2.0 -> mean=2.0
        # s=4.0 (all skipped) was trimmed off the tail.
        means = [a.mean_eff_all for a in cs.aggs]
        assert means == [1.0, 2.0]

    def test_filter_means_coh0(self, mod, tiny_cell):
        cs = mod.aggregate_cell_sign(tiny_cell)
        # s=1.0: all 3 q's coh=0 -> contributes 1.0/1.0/1.0 -> mean=1.0
        # s=2.0: q0 (coh=0) and q2 (coh=0) contribute eff=2.0 each;
        #        q1 has coh=1 and is filtered out -> mean=2.0
        means = [a.mean_eff_coh0 for a in cs.aggs]
        assert means == [1.0, 2.0]

    def test_filter_means_rp3(self, mod, tiny_cell):
        cs = mod.aggregate_cell_sign(tiny_cell)
        # s=1.0: all 3 q's rp=3 -> mean=1.0
        # s=2.0: q0 (rp=3) and q1 (rp=3) contribute eff=2.0 each;
        #        q2 has rp=2 and is filtered out -> mean=2.0
        means = [a.mean_eff_rp3 for a in cs.aggs]
        assert means == [1.0, 2.0]

    def test_filter_means_coh0_and_rp3(self, mod, tiny_cell):
        cs = mod.aggregate_cell_sign(tiny_cell)
        # s=1.0: all 3 q's coh=0 AND rp=3 -> mean=1.0
        # s=2.0: only q0 has both coh=0 AND rp=3 -> mean=2.0
        means = [a.mean_eff_coh0_rp3 for a in cs.aggs]
        assert means == [1.0, 2.0]

    def test_bookkeeping_counters(self, mod, tiny_cell):
        cs = mod.aggregate_cell_sign(tiny_cell)
        # s=1.0: 3 records, 3 with eff, 3 with coh=0
        # s=2.0: 3 records, 3 with eff, 2 with coh=0
        # s=4.0 row was trimmed off the tail (was all-skipped).
        assert [a.n_total for a in cs.aggs] == [3, 3]
        assert [a.n_with_eff for a in cs.aggs] == [3, 3]
        assert [a.n_coh0 for a in cs.aggs] == [3, 2]

    def test_unfiltered_coh_rp_means(self, mod, tiny_cell):
        """The new ``mean_coh_all`` and ``mean_rp_all`` feed the bottom-row
        coh / inverted-rp panels.  ``tiny_cell``:
          s=1.0: 3 x coh=0, 3 x rp=3   -> mean_coh=0.0, mean_rp=3.0
          s=2.0: cohs [0,1,0], rps [3,3,2] -> 1/3, 8/3
        """
        cs = mod.aggregate_cell_sign(tiny_cell)
        coh_means = [a.mean_coh_all for a in cs.aggs]
        rp_means = [a.mean_rp_all for a in cs.aggs]
        assert coh_means[0] == pytest.approx(0.0)
        assert coh_means[1] == pytest.approx(1.0 / 3.0)
        assert rp_means[0] == pytest.approx(3.0)
        assert rp_means[1] == pytest.approx(8.0 / 3.0)


# ---------------------------------------------------------------------------
# gather_cells
# ---------------------------------------------------------------------------

class TestGatherCells:
    def test_multiple_cells_sorted(self, mod, tmp_path):
        """Multiple (slot, layer, sign) subdirs should appear in
        ``(slot, layer, sign)`` ascending order so the legend / colour
        assignment is stable.
        """
        exp = tmp_path / "exp"
        for slot, layer, sign in [(7, 49, +1), (7, 49, -1),
                                  (6, 25, +1), (6, 25, -1)]:
            d = exp / f"s{slot}_l{layer}_{'+1' if sign > 0 else '-1'}"
            sign_str = '+1' if sign > 0 else '-1'
            d.mkdir(parents=True)
            rec = _make_record(
                strength=1.0, sign=sign, question_idx=0,
                coh=0, rp=3, eff=1.0 * sign,
            )
            (d / "records.jsonl").write_text(json.dumps(rec) + "\n")
            (d / "summary.json").write_text(
                json.dumps({"slot": slot, "layer": layer, "sign": sign,
                            "scan_mode": "bidirectional"})
            )
        cells = mod.gather_cells(exp)
        keys = [(c.slot, c.layer, c.sign) for c in cells]
        # Ascending (slot, layer, sign) -- (6,25,-1), (6,25,+1), (7,49,-1), (7,49,+1).
        assert keys == [(6, 25, -1), (6, 25, +1), (7, 49, -1), (7, 49, +1)]

    def test_skips_non_cell_subdirs(self, mod, tmp_path):
        """``baselines/`` and other non-cell subdirs should be ignored."""
        exp = tmp_path / "exp"
        exp.mkdir()
        (exp / "baselines").mkdir()
        (exp / "baselines" / "records.jsonl").write_text("")
        (exp / "other_thing").mkdir()
        cells = mod.gather_cells(exp)
        assert cells == []

    def test_skips_empty_cell(self, mod, tmp_path):
        """A cell dir with no records.jsonl should be skipped silently."""
        exp = tmp_path / "exp"
        d = exp / "s6_l25_+1"
        d.mkdir(parents=True)
        # No records.jsonl at all.
        cells = mod.gather_cells(exp)
        assert cells == []


# ---------------------------------------------------------------------------
# _x_steps_from_cliff
# ---------------------------------------------------------------------------

class TestXStepsFromCliff:
    def test_zero_for_largest_n_minus_1_for_smallest(self, mod):
        """For N=5 strengths, the largest-magnitude strength sits at
        x=0 (the cliff) and the smallest at x=4.  After ``invert_xaxis``
        in the plot, x=0 ends up on the right edge as intended.
        """
        x = mod._x_steps_from_cliff(5)
        assert list(x) == [4, 3, 2, 1, 0]

    def test_handles_n_one(self, mod):
        """A cell with a single strength still produces a single x=0
        point so the plot renders it at the cliff.
        """
        x = mod._x_steps_from_cliff(1)
        assert list(x) == [0]


# ---------------------------------------------------------------------------
# End-to-end plot smoke test
# ---------------------------------------------------------------------------

class TestPlotSmoke:
    def test_plot_writes_png(self, mod, tmp_path):
        """End-to-end smoke: build a 2-cell × 2-sign experiment and
        verify the PNG renders without exception.  Visual content is
        checked manually.
        """
        exp = tmp_path / "smoke_exp"
        for slot, layer, sign in [(6, 25, +1), (6, 25, -1),
                                  (7, 49, +1), (7, 49, -1)]:
            d = exp / f"s{slot}_l{layer}_{'+1' if sign > 0 else '-1'}"
            d.mkdir(parents=True)
            recs: List[Dict[str, Any]] = []
            for strength in [1.0, 2.0, 4.0]:
                for q in range(3):
                    recs.append(_make_record(
                        strength=strength, sign=sign, question_idx=q,
                        coh=0, rp=3, eff=sign * strength * 0.5,
                    ))
            (d / "records.jsonl").write_text(
                "\n".join(json.dumps(r) for r in recs) + "\n"
            )
            (d / "summary.json").write_text(
                json.dumps({"slot": slot, "layer": layer, "sign": sign,
                            "scan_mode": "bidirectional",
                            "up_blocked_reason": "incoherent"})
            )

        out_png = tmp_path / "out.png"
        cells = mod.gather_cells(exp)
        mod.plot_response_curves(cells, output_path=out_png, title="smoke")
        assert out_png.exists()
        assert out_png.stat().st_size > 5000  # a non-trivial PNG
