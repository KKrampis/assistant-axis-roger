"""Tests for tools/sheet_layout.py.

Builds synthetic experiment directories in tmp_path (no real model
required) and asserts on the structured :class:`TabPayload` returned
by :func:`build_tab`.  Keeps the test surface independent of
gspread / Google APIs -- that integration layer lives in
``tools/steering_to_gsheet.py`` and is exercised manually against
real data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from tools import sheet_layout as sl


# ---------------------------------------------------------------------------
# Fixtures: synthetic experiment dirs
# ---------------------------------------------------------------------------

def _write(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content, indent=2))
    else:
        path.write_text(str(content))


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _make_record(
    *,
    strength: float,
    sign: int,
    question_idx: int,
    question: str,
    response: str,
    coh: Optional[int] = 0,
    rp: Optional[int] = 3,
    eff: Optional[float] = 0.0,
    persona_skipped: bool = False,
    effect_skipped: bool = False,
) -> Dict[str, Any]:
    """Construct one cell-record matching the runner's on-disk schema."""
    coh_block = None if coh is None else {"score": coh, "reason": "ok"}
    # Persona / effect blocks are NEVER None on a real cell record (the
    # runner always writes the structure, even if the inner judge was
    # skipped due to mean_coh).  Mirror that here so `_persona_skipped`
    # / `_effect_skipped` flags resolve correctly in tests.
    persona_block = {
        "score": rp, "reason": "ok",
        "skipped_due_to_strength_mean_coh": persona_skipped,
    }
    effect_block = {
        "bidirectional": {"scores": {}, "mean": eff,
                          "n_judges_succeeded": 2, "n_judges_attempted": 2},
        "mode": "bidirectional",
        "combined": eff,
        "skipped_due_to_strength_mean_coh": effect_skipped,
    }
    return {
        "strength": strength,
        "sign": sign,
        "question_idx": question_idx,
        "question": question,
        "response": response,
        "n_tokens": 50,
        "judges": {
            "coherence": coh_block,
            "persona": persona_block,
            "effect": effect_block,
        },
        "timing": {"gen_s": 1.0},
        "abandoned": False,
    }


def _make_baseline_record(
    question_idx: int, question: str, response: str
) -> Dict[str, Any]:
    """Baseline record: sign=0, strength=0, judges all null."""
    return {
        "strength": 0.0,
        "sign": 0,
        "question_idx": question_idx,
        "question": question,
        "response": response,
        "n_tokens": 50,
        "judges": {"coherence": None, "persona": None, "effect": None},
        "timing": {"gen_s": 1.0},
    }


def _make_config(
    experiment_id: str,
    role: str = "chef",
    role_from: str = "helpful",
    role_to: str = "unhelpful",
    cells: Optional[List[Dict[str, int]]] = None,
    positions_mode: str = "all",
) -> Dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "model_name": "Qwen/Qwen3-32B",
        "output_dir": "/tmp/outputs",
        "axis_source": {
            "type": "role_transplant",
            "vectors_dir": "/tmp/vectors",
            "role_from": role_from,
            "role_to": role_to,
        },
        "persona": {"type": "role", "role": role, "prompt_index": 0},
        "cells": cells or [{"slot": 3, "layer": 25}],
        "sweep": {
            "weakest_strength": 1.0, "max_strength": 8.0,
            "multiplier": 1.189, "signs": [1, -1],
        },
        "positions_mode": positions_mode,
        "batch_size": 4,
        "max_new_tokens": 512,
        "questions_file": "data/steering/questions/dummy.json",
    }


def _make_summary(
    *, slot: int, layer: int, sign: int,
    up_strength: float = 8.0, up_reason: str = "incoherent",
    down_strength: float = 0.125, down_reason: str = "min_strength_reached",
) -> Dict[str, Any]:
    return {
        "slot": slot, "layer": layer, "sign": sign,
        "stopped_at_strength": up_strength,
        "reason": "bidirectional_done",
        "n_records": 0,
        "n_strengths_swept": 0,
        "scan_mode": "bidirectional",
        "positions_mode": "all",
        "s_init": 1.414,
        "min_strength": 0.125,
        "max_strength": 8.0,
        "multiplier": 1.189,
        "weakest_strength": 1.0,
        "up_blocked_reason": up_reason,
        "down_blocked_reason": down_reason,
        "up_blocked_at_strength": up_strength,
        "down_blocked_at_strength": down_strength,
    }


@pytest.fixture
def single_cell_experiment(tmp_path: Path) -> Path:
    """7 questions × 5 strengths (sign +1) × 5 strengths (sign -1) + baseline.

    Includes one fully-skipped strength (mean_coh forced ≥ 1.5) so the
    [skipped — incoherent] placeholder is exercised.
    """
    n_q = 7
    questions = [f"q{i}" for i in range(n_q)]
    exp_dir = tmp_path / "chef_helpful_v1"
    _write(exp_dir / "config.json", _make_config("chef_helpful_v1"))
    _write(exp_dir / "questions.json", questions)
    _write(exp_dir / "persona_system_prompt.txt", "Pretend you are a chef.")
    # Baselines
    _write_jsonl(
        exp_dir / "baselines" / "records.jsonl",
        [_make_baseline_record(i, questions[i], f"baseline-resp-{i}")
         for i in range(n_q)],
    )
    # Cell records: 5 strengths each sign.
    pos_strengths = [1.0, 1.414, 2.0, 4.0, 8.0]
    neg_strengths = [1.0, 1.414, 2.0, 4.0, 8.0]
    pos_recs: List[Dict[str, Any]] = []
    for s in pos_strengths:
        # The strongest +1 strength is forced to "skipped" so we
        # cover the placeholder path.
        skipped = (s == 8.0)
        for q in range(n_q):
            pos_recs.append(_make_record(
                strength=s, sign=+1, question_idx=q,
                question=questions[q],
                response=f"+{s}-resp-q{q}",
                coh=2 if skipped else 0,
                rp=None if skipped else 3,
                eff=None if skipped else +1.0,
                persona_skipped=skipped,
                effect_skipped=skipped,
            ))
    neg_recs: List[Dict[str, Any]] = []
    for s in neg_strengths:
        for q in range(n_q):
            neg_recs.append(_make_record(
                strength=s, sign=-1, question_idx=q,
                question=questions[q],
                response=f"-{s}-resp-q{q}",
                coh=0, rp=3, eff=-1.0,
            ))
    _write_jsonl(exp_dir / "s3_l25_+1" / "records.jsonl", pos_recs)
    _write_jsonl(exp_dir / "s3_l25_-1" / "records.jsonl", neg_recs)
    _write(exp_dir / "s3_l25_+1" / "summary.json",
           _make_summary(slot=3, layer=25, sign=+1))
    _write(exp_dir / "s3_l25_-1" / "summary.json",
           _make_summary(slot=3, layer=25, sign=-1,
                         up_reason="incoherent",
                         down_reason="min_strength_reached"))
    return exp_dir


@pytest.fixture
def multi_cell_experiment(tmp_path: Path) -> Path:
    """7 questions × 2 cells × (3 strengths × 2 signs) + baseline."""
    n_q = 7
    questions = [f"q{i}" for i in range(n_q)]
    exp_dir = tmp_path / "chef_helpful_multi"
    _write(exp_dir / "config.json", _make_config(
        "chef_helpful_multi",
        cells=[{"slot": 3, "layer": 25}, {"slot": 7, "layer": 49}],
    ))
    _write(exp_dir / "questions.json", questions)
    _write(exp_dir / "persona_system_prompt.txt", "Pretend you are a chef.")
    _write_jsonl(
        exp_dir / "baselines" / "records.jsonl",
        [_make_baseline_record(i, questions[i], f"b-{i}") for i in range(n_q)],
    )
    for slot, layer in [(3, 25), (7, 49)]:
        for sign in (+1, -1):
            recs: List[Dict[str, Any]] = []
            for s in (1.0, 2.0, 4.0):
                for q in range(n_q):
                    recs.append(_make_record(
                        strength=s, sign=sign, question_idx=q,
                        question=questions[q],
                        response=f"{sign:+d}{s}-s{slot}l{layer}-q{q}",
                    ))
            _write_jsonl(
                exp_dir / f"s{slot}_l{layer}_{'+1' if sign > 0 else '-1'}"
                / "records.jsonl",
                recs,
            )
            _write(
                exp_dir / f"s{slot}_l{layer}_{'+1' if sign > 0 else '-1'}"
                / "summary.json",
                _make_summary(slot=slot, layer=layer, sign=sign),
            )
    return exp_dir


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestSingleCellShape:
    def test_grid_dimensions(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # Single cell: 1 block = 1 header row + 5*2=10 data rows.
        # Total rows = 4 header rows (header, baseline, persona, axis)
        # + 1 block header + 10 data = 15.
        assert len(payload.values) == 4 + 1 + 10
        # Cols = 5 agg + 4 * 7 questions = 33.
        assert all(len(r) == 5 + 4 * 7 for r in payload.values)

    def test_frozen_counts(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # Only header + baseline are frozen (May-2026 layout).
        assert payload.frozen_rows == 2
        assert payload.frozen_cols == 1

    def test_header_region_row_order(self, single_cell_experiment):
        """Header-region rows: header (0), baseline (1), persona (2),
        axis (3).  Persona + axis are unfrozen below the two frozen
        rows so they don't waste vertical space.
        """
        payload = sl.build_tab(single_cell_experiment)
        # Row 0: header (column labels + question texts).
        assert payload.values[0][0] == "signed_strength"
        # Row 1: baseline (col A label, baseline_responses in q cells).
        assert payload.values[1][0] == "baseline"
        # Row 2: persona (col A label, prompt in col B).
        assert payload.values[2][0] == "persona"
        assert payload.values[2][1] == "Pretend you are a chef."
        # Row 3: axis (col A label, pos + neg pole descriptions split).
        assert payload.values[3][0] == "axis"

    def test_header_row_has_question_text(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # Header is row index 0; col index 5 = q0 response col carries q text.
        assert payload.values[0][5] == "q0"
        # Per-question col order: response (5), eff (6), coh (7), rp (8).
        assert payload.values[0][6] == "eff"
        assert payload.values[0][7] == "coh"
        assert payload.values[0][8] == "rp"

    def test_baseline_row_has_baseline_responses(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # Baseline is now row 1 (was row 2 in pre-May-2026 layout).
        assert payload.values[1][0] == "baseline"
        assert payload.values[1][5] == "baseline-resp-0"
        assert payload.values[1][9] == "baseline-resp-1"

    def test_axis_row_packs_both_poles_on_two_lines(
        self, single_cell_experiment
    ):
        """Both pole descriptions live in a single merged cell at col B,
        separated by an embedded newline (rendered as line-2 by Sheets
        wrap=WRAP).  Saves vertical space vs the side-by-side variant
        and lets each line read full-width.
        """
        payload = sl.build_tab(single_cell_experiment)
        # Col A label, col B carries the merged content.
        assert payload.values[3][0] == "axis"
        cell = str(payload.values[3][1])
        lines = cell.split("\n")
        assert len(lines) == 2, \
            f"axis row should be exactly 2 lines, got: {cell!r}"
        # Line 1 = pos pole (role_from), line 2 = neg pole (role_to).
        assert lines[0].startswith("helpful")
        assert "sign -1" in lines[0]
        assert lines[1].startswith("unhelpful")
        assert "sign +1" in lines[1]
        # All cells past col B in row 3 are empty (single merge covers them).
        for c in range(2, len(payload.values[3])):
            assert payload.values[3][c] == ""

    def test_block_header_sentinel_in_col_a(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # Block header now at row index 4 (right after the 4 header rows).
        sentinel = payload.values[sl.N_HEADER_ROWS][0]
        assert sentinel == "[block:chef_helpful_v1/s3_l25/all]"

    def test_signed_strengths_sorted_ascending(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # Data rows start at row N_HEADER_ROWS + 1 (after block header);
        # 10 strength rows total.
        data_start = sl.N_HEADER_ROWS + 1
        signed = [payload.values[data_start + i][0] for i in range(10)]
        assert signed == sorted(signed)
        # Should span -8.0 .. +8.0 (approx).
        assert signed[0] == -8.0
        assert signed[-1] == 8.0

    def test_blocks_list_matches_grid(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        assert len(payload.blocks) == 1
        b = payload.blocks[0]
        assert b.experiment_id == "chef_helpful_v1"
        assert b.slot == 3
        assert b.layer == 25
        assert b.positions_mode == "all"
        assert b.header_row == sl.N_HEADER_ROWS
        assert b.data_row_start == sl.N_HEADER_ROWS + 1
        assert b.data_row_end == sl.N_HEADER_ROWS + 1 + 10
        # Sentinel matches what we put in col A of header_row.
        assert payload.values[b.header_row][0] == b.sentinel

    def test_merges_present(self, single_cell_experiment):
        """2 merges: persona row B..end, axis row B..end (each one wide cell)."""
        payload = sl.build_tab(single_cell_experiment)
        assert len(payload.merges) == 2
        persona_merge = next(m for m in payload.merges if m.row_start == 2)
        axis_merge = next(m for m in payload.merges if m.row_start == 3)
        assert persona_merge.col_start == 1
        assert axis_merge.col_start == 1
        # Both span to the same end column.
        assert persona_merge.col_end == axis_merge.col_end


class TestSkippedRow:
    def test_skipped_row_keeps_actual_response_text(
        self, single_cell_experiment
    ):
        """The runner USED to substitute '[skipped -- incoherent]' for
        skipped rows' response cells.  As of May 2026 we keep the
        actual response text instead -- the lack of RP/Eff values in
        adjacent score cells plus the high coh score (transitively
        shaded onto the response cell via the cross-reference
        conditional formats) is enough visual signal.
        """
        payload = sl.build_tab(single_cell_experiment)
        # Find the row at signed_strength = +8.0 (skipped in the fixture).
        for i, row in enumerate(payload.values):
            if i < sl.N_HEADER_ROWS + 1:  # skip header region + block header
                continue
            if row[0] == 8.0:
                # Real response from the fixture (per _make_record), not
                # any placeholder substitution.
                assert row[5] == "+8.0-resp-q0"
                assert row[9] == "+8.0-resp-q1"
                return
        pytest.fail("Did not find +8.0 row")

    def test_no_italic_skip_rules_emitted(self, single_cell_experiment):
        """No italic-gray TEXT_EQ placeholder rules anymore (we don't
        substitute, so the rule would never fire).
        """
        payload = sl.build_tab(single_cell_experiment)
        italic_rules = [cf for cf in payload.cond_formats if cf.italic]
        assert italic_rules == []


class TestDimGroups:
    def test_outer_aggregate_group(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # First group should cover cols B-E (1..5).
        outer = payload.dim_groups[0]
        assert outer.dim == "COLUMNS"
        assert outer.start == 1
        assert outer.end == sl.N_AGG_COLS  # 5

    def test_one_inner_group_per_question(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # 7 questions -> 7 inner groups (after the 1 outer).
        assert len(payload.dim_groups) == 1 + 7
        for q in range(7):
            g = payload.dim_groups[1 + q]
            q_resp_col = sl.N_AGG_COLS + sl.N_PER_QUESTION_COLS * q
            # Group should cover the score trio (coh, rp, eff), skipping
            # response (the first per-question col).
            assert g.start == q_resp_col + 1
            assert g.end == q_resp_col + sl.N_PER_QUESTION_COLS


class TestCondFormats:
    def test_graded_pink_thresholds_present(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # At least one rule for each of (0.5, 1.0, 1.5) on mean_coh.
        pink_thresholds_seen: set = set()
        for cf in payload.cond_formats:
            if (cf.col_start == 1 and cf.col_end == 2
                    and cf.condition_type == "NUMBER_GREATER_THAN_EQ"):
                pink_thresholds_seen.add(cf.values[0])
        assert pink_thresholds_seen == set(sl.COH_THRESHOLDS)

    def test_signed_eff_both_directions(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # mean_eff col is index 3.  Should have 3 positive (GTE 1/2/3)
        # and 3 negative (LTE -1/-2/-3) for at least one data row.
        pos_t: set = set()
        neg_t: set = set()
        for cf in payload.cond_formats:
            if cf.col_start == 3 and cf.col_end == 4:
                if cf.condition_type == "NUMBER_GREATER_THAN_EQ":
                    pos_t.add(cf.values[0])
                elif cf.condition_type == "NUMBER_LESS_THAN_EQ":
                    neg_t.add(cf.values[0])
        assert pos_t == set(sl.EFF_ABS_THRESHOLDS)
        assert neg_t == {-t for t in sl.EFF_ABS_THRESHOLDS}

    def test_response_col_has_cross_reference_rules(
        self, single_cell_experiment
    ):
        """Response cells get CUSTOM_FORMULA rules referencing their
        row's coherence / RP / effect cells so the response cell takes
        the same background colour as its triggering score cell.

        Per-question col order (May-2026): response, eff, coh, rp.
        So for q0 (response col 5 / 'F') the score cols are:
          eff -> col 6 / 'G'
          coh -> col 7 / 'H'
          rp  -> col 8 / 'I'
        """
        payload = sl.build_tab(single_cell_experiment)
        q0_resp = sl.N_AGG_COLS
        custom = [
            cf for cf in payload.cond_formats
            if cf.col_start == q0_resp and cf.col_end == q0_resp + 1
            and cf.condition_type == "CUSTOM_FORMULA"
        ]
        # 3 coh references + 1 rp + 12 (6 green + 6 blue) eff = 16.
        assert len(custom) == 16
        # Confirm formulas point at the correct columns by letter.
        ref_cols = set()
        for cf in custom:
            f = cf.values[0]
            assert f.startswith("=$"), f
            letter = ""
            for ch in f[2:]:
                if ch.isalpha():
                    letter += ch
                else:
                    break
            ref_cols.add(letter)
        # G (eff col 6), H (coh col 7), I (rp col 8).
        assert ref_cols == {"G", "H", "I"}

    def test_priority_order_coh_before_rp_before_eff(
        self, single_cell_experiment
    ):
        """For Sheets' FIRST-match-wins semantics to give coh > rp > eff
        priority on response cells, the rules must be registered in
        order coh (first/highest), rp (middle), eff (last/lowest).

        Per-question col order (May-2026): response, eff, coh, rp.
          eff -> col G
          coh -> col H
          rp  -> col I
        Registration order on the response cell:
          coh rules (H, highest priority)
            -> rp rule (I, middle priority)
            -> eff rules (G, lowest priority).
        """
        payload = sl.build_tab(single_cell_experiment)
        q0_resp = sl.N_AGG_COLS
        custom = [
            cf for cf in payload.cond_formats
            if cf.col_start == q0_resp and cf.col_end == q0_resp + 1
            and cf.condition_type == "CUSTOM_FORMULA"
        ]
        def letter_of(cf):
            f = cf.values[0]
            out = ""
            for ch in f[2:]:
                if ch.isalpha():
                    out += ch
                else:
                    break
            return out

        order = [letter_of(cf) for cf in custom]
        # coh rules (col H) -> rp rule (col I) -> eff rules (col G).
        last_h = max(i for i, l in enumerate(order) if l == "H")
        first_i = order.index("I")
        first_g = order.index("G")
        assert last_h < first_i, "coh (H) rules must precede rp (I) rule"
        assert first_i < first_g, "rp (I) rule must precede eff (G) rules"

    def test_gradient_descending_order(self, single_cell_experiment):
        """Within each gradient family (pink coh / green eff / blue eff),
        rules MUST be registered in descending-threshold order so
        Sheets' first-match-wins semantics yields the saturated shade
        for the largest score.  Reverse-order registration would
        result in every matching cell getting the FAINTEST shade --
        observed bug in May-2026 before this test was added.
        """
        payload = sl.build_tab(single_cell_experiment)
        # mean_coh col B: numeric pink rules, thresholds should descend.
        coh_rules = [cf for cf in payload.cond_formats
                     if cf.col_start == 1 and cf.col_end == 2
                     and cf.condition_type == "NUMBER_GREATER_THAN_EQ"]
        coh_thresholds = [cf.values[0] for cf in coh_rules]
        assert coh_thresholds == sorted(coh_thresholds, reverse=True)
        # mean_eff col D: GTE thresholds descend, then LTE thresholds descend
        # (i.e. -0.5 last among LTE, -3.0 first among LTE).
        eff_gte = [cf.values[0] for cf in payload.cond_formats
                   if cf.col_start == 3 and cf.col_end == 4
                   and cf.condition_type == "NUMBER_GREATER_THAN_EQ"]
        eff_lte = [cf.values[0] for cf in payload.cond_formats
                   if cf.col_start == 3 and cf.col_end == 4
                   and cf.condition_type == "NUMBER_LESS_THAN_EQ"]
        # GTE: large positive first.
        assert eff_gte == sorted(eff_gte, reverse=True)
        # LTE: most-negative first (i.e. -3 before -0.5).
        assert eff_lte == sorted(eff_lte)

    def test_eff_has_six_levels(self, single_cell_experiment):
        """Effect now uses 6 absolute thresholds (was 3); both the
        score col rules and the response col cross-references should
        reflect that."""
        assert len(sl.EFF_ABS_THRESHOLDS) == 6
        assert len(sl.COLOR_GREEN) == 6
        assert len(sl.COLOR_BLUE) == 6

    def test_low_rp_rule_on_mean_col(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # Tab-wide: exactly one orange-faint LTE-1 NUMBER comparison
        # rule on mean_rp col 2 (per-question rp cols get their own).
        low_rp_rules = [cf for cf in payload.cond_formats
                        if cf.col_start == 2 and cf.col_end == 3
                        and cf.condition_type == "NUMBER_LESS_THAN_EQ"
                        and cf.values == (sl.RP_LOW_THRESHOLD,)]
        assert len(low_rp_rules) == 1
        # Rule spans the entire data row range.
        assert low_rp_rules[0].row_start == sl.N_HEADER_ROWS
        # 4 header + 1 block header + 10 data = 15 total
        assert low_rp_rules[0].row_end == sl.N_HEADER_ROWS + 1 + 10

    def test_rule_count_scales_with_questions_not_rows(
        self, single_cell_experiment
    ):
        """Sanity check: under tab-wide formatting, total rule count is
        a function of n_questions, NOT of n_data_rows.  Previously we
        emitted per-row rules which would blow past Sheets' practical
        cap on multi-cell experiments; tab-wide keeps us bounded.
        """
        payload = sl.build_tab(single_cell_experiment)
        # Aggregate cols: 3 (mean_coh pink) + 1 (mean_rp orange) +
        # 12 (mean_eff signed: 6 green + 6 blue) = 16.
        # Per-question score cols: same 16.
        # Per-question response col cross-references: 3 (coh pink) +
        # 1 (rp orange) + 12 (eff green+blue) = 16.
        # Total per question = 16 (score) + 16 (response) = 32.
        # 7 questions -> 16 + 7*32 = 240.
        n = len(sl.EFF_ABS_THRESHOLDS)
        agg = len(sl.COH_THRESHOLDS) + 1 + 2 * n  # 3 + 1 + 12 = 16
        per_q_scores = agg                          # same shape
        per_q_resp = len(sl.COH_THRESHOLDS) + 1 + 2 * n  # 16
        expected = agg + 7 * (per_q_scores + per_q_resp)
        assert len(payload.cond_formats) == expected


class TestWidths:
    def test_response_cols_are_wide(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        # q0 response col = N_AGG_COLS + 0 = 5.
        widths_by_col = {w.col: w.width_px for w in payload.widths}
        assert widths_by_col[5] == sl.WIDTH_RESPONSE_PX
        # q0_eff = 6, q0_coh = 7, q0_rp = 8 -- all narrow score width.
        assert widths_by_col[6] == sl.WIDTH_SCORE_PX
        assert widths_by_col[7] == sl.WIDTH_SCORE_PX
        assert widths_by_col[8] == sl.WIDTH_SCORE_PX

    def test_strength_col_width(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        widths_by_col = {w.col: w.width_px for w in payload.widths}
        assert widths_by_col[0] == sl.WIDTH_STRENGTH_PX


class TestMultiCell:
    def test_two_blocks(self, multi_cell_experiment):
        payload = sl.build_tab(multi_cell_experiment)
        assert len(payload.blocks) == 2
        b0, b1 = payload.blocks
        assert b0.slot == 3 and b0.layer == 25
        assert b1.slot == 7 and b1.layer == 49

    def test_sentinels_unique(self, multi_cell_experiment):
        payload = sl.build_tab(multi_cell_experiment)
        sentinels = [b.sentinel for b in payload.blocks]
        assert len(set(sentinels)) == len(sentinels)

    def test_blocks_are_contiguous_and_in_order(self, multi_cell_experiment):
        payload = sl.build_tab(multi_cell_experiment)
        b0, b1 = payload.blocks
        # The second block starts immediately after the first ends.
        assert b1.header_row == b0.data_row_end
        # First block's header sits right after the header region.
        assert b0.header_row == sl.N_HEADER_ROWS


class TestBlockSentinel:
    def test_format(self):
        s = sl.make_block_sentinel("foo_v1", 3, 25, "all")
        assert s == "[block:foo_v1/s3_l25/all]"

    def test_positions_mode_distinguishes(self):
        a = sl.make_block_sentinel("foo_v1", 3, 25, "all")
        b = sl.make_block_sentinel("foo_v1", 3, 25, "prefill")
        assert a != b


class TestDefaultNames:
    def test_tab_name(self):
        assert sl.default_tab_name("chef", "helpful", "unhelpful") \
            == "chef helpful_unhelpful"

    def test_spreadsheet_name_single(self):
        n = sl.default_spreadsheet_name(["chef_helpful_v1"])
        assert "chef_helpful_v1" in n

    def test_spreadsheet_name_common_prefix(self):
        n = sl.default_spreadsheet_name(
            ["chef_helpful_v1", "chef_helpful_v2"]
        )
        assert "chef_helpful" in n
        assert n.endswith("*")

    def test_spreadsheet_name_no_common_prefix(self):
        n = sl.default_spreadsheet_name(["abc_helpful_v1", "xyz_helpful_v1"])
        assert n == "Steering experiments"


class TestAggregates:
    def test_aggregates_match_per_q_means(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        data_start = sl.N_HEADER_ROWS + 1
        for row in payload.values[data_start:]:
            if row[0] == 1.0:
                # All q0..q6 coh values are 0 (per fixture), so mean_coh = 0.
                assert row[1] == 0.0
                assert row[2] == 3.0
                assert row[3] == 1.0
                assert row[4] == 1.0
                return
        pytest.fail("Did not find +1.0 row")

    def test_aggregates_blank_when_all_skipped(self, single_cell_experiment):
        payload = sl.build_tab(single_cell_experiment)
        data_start = sl.N_HEADER_ROWS + 1
        # +8.0 row had persona+effect skipped for ALL 7 questions, so
        # mean_rp and mean_eff should be "" (blank).  mean_coh is still
        # 2.0 because the fixture set coh=2 on skipped rows.
        for row in payload.values[data_start:]:
            if row[0] == 8.0:
                assert row[2] == ""
                assert row[3] == ""
                assert row[4] == ""
                return
        pytest.fail("Did not find +8.0 row")
