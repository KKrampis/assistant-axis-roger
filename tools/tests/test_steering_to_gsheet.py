"""Tests for the merge / append / replace logic in
``tools/steering_to_gsheet.py``.

The Google Sheets I/O glue (gspread + batchUpdate request builders)
is exercised manually against real data once OAuth is set up.  This
module covers the pure-Python merge step that decides which existing
blocks survive a rerun and where the new payload's blocks land in
the final grid -- the trickiest correctness question in the CLI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from tools import sheet_layout as sl
from tools import steering_to_gsheet as ssg


# ---------------------------------------------------------------------------
# Block-sentinel scanning helpers
# ---------------------------------------------------------------------------

class TestScanExistingBlocks:
    def test_no_blocks_in_empty_tab(self):
        assert ssg._scan_existing_blocks([]) == []

    def test_no_blocks_in_header_only_tab(self):
        col_a = ["persona", "strength", "baseline"]
        assert ssg._scan_existing_blocks(col_a) == []

    def test_finds_block_headers(self):
        col_a = [
            "persona", "strength", "baseline",
            "[block:chef_helpful_v1/s3_l25/all]",
            "-8.0", "-4.0", "-2.0", "0", "2.0", "4.0", "8.0",
            "[block:chef_helpful_v1/s7_l49/all]",
            "-8.0", "8.0",
        ]
        found = ssg._scan_existing_blocks(col_a)
        assert len(found) == 2
        assert found[0] == (3, "[block:chef_helpful_v1/s3_l25/all]")
        assert found[1] == (11, "[block:chef_helpful_v1/s7_l49/all]")

    def test_rejects_partial_sentinels(self):
        # Strings that LOOK like sentinels but don't match the exact
        # regex shouldn't be picked up (avoid false positives on user
        # comments etc).
        col_a = [
            "[block: chef/s3_l25/all]",          # space after colon
            "[block:chef/s3_l25/]",                # empty positions_mode
            "[block:chef/s3_l25]",                 # missing positions_mode
            "block:chef/s3_l25/all",               # missing brackets
            "[block:chef_helpful_v1/s3_l25/all]",  # the only real one
        ]
        found = ssg._scan_existing_blocks(col_a)
        assert len(found) == 1
        assert found[0] == (4, "[block:chef_helpful_v1/s3_l25/all]")

    def test_unicode_in_sentinel_id(self):
        # Slashes and brackets are forbidden chars in our sentinel
        # format, but other characters in experiment_id should pass.
        col_a = ["[block:chef_helpful_v1-test/s3_l25/all]"]
        found = ssg._scan_existing_blocks(col_a)
        assert len(found) == 1


class TestBlockRowRanges:
    def test_ranges_extend_to_next_header(self):
        col_a = (
            ["persona", "strength", "baseline"]
            + ["[block:e1/s3_l25/all]", "a", "b", "c"]
            + ["[block:e1/s7_l49/all]", "d", "e"]
        )
        headers = ssg._scan_existing_blocks(col_a)
        ranges = ssg._block_row_ranges(col_a, headers)
        assert ranges == [
            (3, 7, "[block:e1/s3_l25/all]"),
            (7, 10, "[block:e1/s7_l49/all]"),
        ]

    def test_last_block_extends_to_end_of_sheet(self):
        col_a = ["persona", "strength", "baseline",
                 "[block:e1/s3_l25/all]", "a", "b", "c", "d", "e", "f", "g"]
        headers = ssg._scan_existing_blocks(col_a)
        ranges = ssg._block_row_ranges(col_a, headers)
        assert ranges == [(3, 11, "[block:e1/s3_l25/all]")]


# ---------------------------------------------------------------------------
# Mode filters
# ---------------------------------------------------------------------------

class TestFilterBlocksToKeep:
    @pytest.fixture
    def two_block_existing(self):
        return [
            (3, 7, "[block:exp_A/s3_l25/all]"),
            (7, 11, "[block:exp_B/s3_l25/all]"),
        ]

    def test_replace_drops_matching_experiment(self, two_block_existing):
        kept = ssg._filter_blocks_to_keep(
            two_block_existing, "exp_A", "replace_experiment_blocks"
        )
        # exp_A's block should be dropped; exp_B's preserved.
        assert len(kept) == 1
        assert kept[0][2] == "[block:exp_B/s3_l25/all]"

    def test_replace_keeps_all_when_no_match(self, two_block_existing):
        kept = ssg._filter_blocks_to_keep(
            two_block_existing, "exp_C", "replace_experiment_blocks"
        )
        assert kept == two_block_existing

    def test_replace_drops_all_blocks_of_same_experiment_across_cells(self):
        existing = [
            (3, 7, "[block:exp_A/s3_l25/all]"),
            (7, 11, "[block:exp_A/s7_l49/all]"),
            (11, 15, "[block:exp_B/s3_l25/all]"),
        ]
        kept = ssg._filter_blocks_to_keep(
            existing, "exp_A", "replace_experiment_blocks"
        )
        assert len(kept) == 1
        assert kept[0][2].startswith("[block:exp_B/")

    def test_append_only_keeps_everything(self, two_block_existing):
        kept = ssg._filter_blocks_to_keep(
            two_block_existing, "exp_A", "append_only"
        )
        assert kept == two_block_existing

    def test_wipe_tab_keeps_nothing(self, two_block_existing):
        kept = ssg._filter_blocks_to_keep(
            two_block_existing, "exp_A", "wipe_tab"
        )
        assert kept == []

    def test_unknown_mode_falls_to_replace(self, two_block_existing):
        # Defensive: unknown mode falls to replace_experiment_blocks
        # semantics rather than silently keeping or dropping everything.
        kept = ssg._filter_blocks_to_keep(
            two_block_existing, "exp_A", "garbage_mode"
        )
        assert len(kept) == 1
        assert kept[0][2].startswith("[block:exp_B/")


# ---------------------------------------------------------------------------
# Sentinel parsing
# ---------------------------------------------------------------------------

class TestParseSentinelBack:
    def test_well_formed(self):
        spec = ssg._parse_block_sentinel_back(
            "[block:chef_helpful_v1/s3_l25/all]",
            header_row=3, data_row_start=4, data_row_end=14,
        )
        assert spec is not None
        assert spec.experiment_id == "chef_helpful_v1"
        assert spec.slot == 3
        assert spec.layer == 25
        assert spec.positions_mode == "all"
        assert spec.header_row == 3
        assert spec.data_row_start == 4
        assert spec.data_row_end == 14
        assert spec.sentinel == "[block:chef_helpful_v1/s3_l25/all]"

    def test_returns_none_on_garbage(self):
        for bad in ["", "not a sentinel", "[block:nope]",
                    "[block:exp/all]", "[block:exp/s3_l25]"]:
            assert ssg._parse_block_sentinel_back(
                bad, header_row=0, data_row_start=1, data_row_end=2,
            ) is None

    def test_round_trip_with_make_block_sentinel(self):
        # The parse_back function is the inverse of make_block_sentinel
        # -- this is the contract the merge logic relies on.
        s = sl.make_block_sentinel("foo_v2", 7, 49, "prefill")
        spec = ssg._parse_block_sentinel_back(
            s, header_row=10, data_row_start=11, data_row_end=20,
        )
        assert spec is not None
        assert spec.experiment_id == "foo_v2"
        assert spec.slot == 7
        assert spec.layer == 49
        assert spec.positions_mode == "prefill"


# ---------------------------------------------------------------------------
# Frozen-row compatibility validator
# ---------------------------------------------------------------------------

def _make_minimal_payload(
    n_questions: int = 3,
    questions: List[str] = None,
    experiment_id: str = "exp_A",
) -> sl.TabPayload:
    """Build a TabPayload by hand (skipping disk I/O) for merge / validator
    tests.

    Mirrors the May-2026 layout:
      Row 0: header (column labels + question texts)
      Row 1: baseline responses
      Row 2: persona prompt
      Row 3: axis info
      Row 4: block header (single empty block)
    """
    questions = questions or [f"q{i}" for i in range(n_questions)]
    header_row = sl._make_header_row(n_questions, questions)
    baseline_row = sl._make_baseline_row(
        [f"b{i}" for i in range(n_questions)], n_questions
    )
    persona_row = sl._make_persona_row("test persona", n_questions)
    axis_row = sl._make_axis_row(
        axis_source={"role_from": "pos", "role_to": "neg"},
        n_questions=n_questions,
        instructions_dir=Path("/nonexistent"),  # no trait files needed
    )
    sentinel = sl.make_block_sentinel(experiment_id, 3, 25, "all")
    n_cols = sl.N_AGG_COLS + sl.N_PER_QUESTION_COLS * n_questions
    block_header = [sentinel, "summary"] + [""] * (n_cols - 2)
    values = [header_row, baseline_row, persona_row, axis_row, block_header]
    values = [sl.pad_row(r, n_cols) for r in values]
    return sl.TabPayload(
        values=values,
        frozen_rows=sl.FROZEN_ROWS, frozen_cols=1,
        dim_groups=[], cond_formats=[], widths=[], merges=[],
        blocks=[sl.BlockSpec(
            experiment_id=experiment_id, slot=3, layer=25,
            positions_mode="all",
            header_row=sl.N_HEADER_ROWS,
            data_row_start=sl.N_HEADER_ROWS + 1,
            data_row_end=sl.N_HEADER_ROWS + 1,
            sentinel=sentinel,
        )],
        n_questions=n_questions,
        persona_prompt="test persona",
        questions=list(questions),
        baseline_responses=[f"b{i}" for i in range(n_questions)],
        tab_name="test tab",
        default_spreadsheet_name="test spreadsheet",
    )


def _make_payload_with(
    questions=("q0", "q1", "q2"),
    persona="test persona",
    baselines=("b0", "b1", "b2"),
    experiment_id="exp_A",
) -> sl.TabPayload:
    """Build a TabPayload by hand with explicit control over persona /
    questions / baselines so the validator tests can vary each
    independently.

    May-2026 row order: header, baseline, persona, axis, block_header.
    """
    n_questions = len(questions)
    header_row = sl._make_header_row(n_questions, list(questions))
    baseline_row = sl._make_baseline_row(list(baselines), n_questions)
    persona_row = sl._make_persona_row(persona, n_questions)
    axis_row = sl._make_axis_row(
        axis_source={"role_from": "pos", "role_to": "neg"},
        n_questions=n_questions,
        instructions_dir=Path("/nonexistent"),
    )
    sentinel = sl.make_block_sentinel(experiment_id, 3, 25, "all")
    n_cols = sl.N_AGG_COLS + sl.N_PER_QUESTION_COLS * n_questions
    block_header = [sentinel, "summary"] + [""] * (n_cols - 2)
    values = [header_row, baseline_row, persona_row, axis_row, block_header]
    values = [sl.pad_row(r, n_cols) for r in values]
    return sl.TabPayload(
        values=values,
        frozen_rows=sl.FROZEN_ROWS, frozen_cols=1,
        dim_groups=[], cond_formats=[], widths=[], merges=[],
        blocks=[sl.BlockSpec(
            experiment_id=experiment_id, slot=3, layer=25,
            positions_mode="all",
            header_row=sl.N_HEADER_ROWS,
            data_row_start=sl.N_HEADER_ROWS + 1,
            data_row_end=sl.N_HEADER_ROWS + 1,
            sentinel=sentinel,
        )],
        n_questions=n_questions,
        persona_prompt=persona,
        questions=list(questions),
        baseline_responses=list(baselines),
        tab_name="test tab",
        default_spreadsheet_name="test spreadsheet",
    )


class TestValidateFrozenCompatibility:
    def test_empty_tab_passes(self):
        payload = _make_minimal_payload()
        # No exception raised
        ssg._validate_frozen_compatibility([], payload)

    def test_matching_questions_passes(self):
        payload = _make_minimal_payload(questions=["q0", "q1", "q2"])
        # Build an existing tab with the same header row.
        existing = [payload.values[i] for i in range(sl.N_HEADER_ROWS)]
        ssg._validate_frozen_compatibility(existing, payload)

    def test_different_questions_raises(self):
        payload = _make_minimal_payload(questions=["new0", "new1", "new2"])
        existing_payload = _make_minimal_payload(
            questions=["old0", "old1", "old2"]
        )
        existing = [existing_payload.values[i] for i in range(sl.N_HEADER_ROWS)]
        with pytest.raises(RuntimeError, match="question text"):
            ssg._validate_frozen_compatibility(existing, payload)

    def test_existing_too_narrow_raises(self):
        # Existing tab has fewer questions than the new payload.
        payload = _make_minimal_payload(n_questions=5)
        existing_payload = _make_minimal_payload(n_questions=3)
        existing = [existing_payload.values[i] for i in range(sl.N_HEADER_ROWS)]
        with pytest.raises(RuntimeError, match="fewer columns"):
            ssg._validate_frozen_compatibility(existing, payload)

    def test_different_persona_raises(self):
        """A persona-prompt difference invalidates the tab even when
        questions match, because every existing block's response was
        generated under the OLD persona.
        """
        old = _make_payload_with(persona="You are a chef.")
        new = _make_payload_with(persona="You are an architect.")
        existing = [old.values[i] for i in range(sl.N_HEADER_ROWS)]
        with pytest.raises(RuntimeError, match="persona system prompt"):
            ssg._validate_frozen_compatibility(existing, new)

    def test_different_baselines_raises(self):
        """Baseline mismatch: same questions, but a different config
        (seed / model build / persona tweak) produced different
        baseline responses.  Merging would silently relate stale
        blocks to wrong reference responses.
        """
        old = _make_payload_with(baselines=("old-b0", "old-b1", "old-b2"))
        new = _make_payload_with(baselines=("new-b0", "new-b1", "new-b2"))
        existing = [old.values[i] for i in range(sl.N_HEADER_ROWS)]
        with pytest.raises(RuntimeError, match="baseline response"):
            ssg._validate_frozen_compatibility(existing, new)

    def test_matching_baselines_pass(self):
        """When questions, persona, AND baselines all match the
        validator is happy.
        """
        old = _make_payload_with()
        new = _make_payload_with()  # identical
        existing = [old.values[i] for i in range(sl.N_HEADER_ROWS)]
        ssg._validate_frozen_compatibility(existing, new)

    def test_empty_existing_baseline_does_not_raise(self):
        """When existing baselines are missing (e.g. tab was created
        but never populated), don't raise on the baseline check --
        let the upcoming write fill them in fresh.
        """
        new = _make_payload_with(baselines=("b0", "b1", "b2"))
        # Existing has matching persona + questions but blank baselines.
        existing_payload = _make_payload_with(baselines=("", "", ""))
        existing = [existing_payload.values[i] for i in range(sl.N_HEADER_ROWS)]
        ssg._validate_frozen_compatibility(existing, new)

    def test_error_message_mentions_both_remedies(self):
        """Errors should point at BOTH --wipe-tab and --tab-name as
        valid remedies so users don't reflexively wipe a tab that
        contains useful sibling blocks.
        """
        old = _make_payload_with(baselines=("old", "old", "old"))
        new = _make_payload_with(baselines=("new", "new", "new"))
        existing = [old.values[i] for i in range(sl.N_HEADER_ROWS)]
        with pytest.raises(RuntimeError) as excinfo:
            ssg._validate_frozen_compatibility(existing, new)
        msg = str(excinfo.value)
        assert "--wipe-tab" in msg
        assert "--tab-name" in msg


# ---------------------------------------------------------------------------
# End-to-end merge
# ---------------------------------------------------------------------------

class TestMergePayloadIntoExisting:
    def _build_existing(
        self,
        exp_ids_and_cells: List[tuple],
        n_questions: int = 3,
    ) -> List[List[Any]]:
        """Synthesize an 'existing' tab grid (the result of one or more
        prior exports).  Each (exp_id, slot, layer) gets one block of
        3 data rows.

        Header-region rows match _make_minimal_payload's layout so the
        validator (which compares row 0 = header, row 1 = baseline,
        row 2 = persona, row 3 = axis) sees identical content and
        doesn't raise a false-positive frozen-mismatch error during
        merge tests.
        """
        n_cols = sl.N_AGG_COLS + sl.N_PER_QUESTION_COLS * n_questions
        questions = [f"q{i}" for i in range(n_questions)]
        rows = [
            sl._make_header_row(n_questions, questions),
            sl._make_baseline_row(
                [f"b{i}" for i in range(n_questions)], n_questions
            ),
            sl._make_persona_row("test persona", n_questions),
            sl._make_axis_row(
                axis_source={"role_from": "pos", "role_to": "neg"},
                n_questions=n_questions,
                instructions_dir=Path("/nonexistent"),
            ),
        ]
        for exp_id, slot, layer in exp_ids_and_cells:
            s = sl.make_block_sentinel(exp_id, slot, layer, "all")
            rows.append([s, f"summary for {s}"] + [""] * (n_cols - 2))
            # 3 strength rows, all marked with the exp_id so we can
            # confirm they got preserved (or removed) in the merge.
            for k, strength in enumerate([-1.0, 0.5, 1.0]):
                # Carry a per-row marker in col 1 (mean_coh slot) so we
                # can pick rows out of the merged grid by exp_id later.
                rows.append([strength, exp_id]
                            + [""] * (n_cols - 2))
        return [sl.pad_row(r, n_cols) for r in rows]

    def test_replace_drops_matching_exp_keeps_others(self):
        # Tab currently holds exp_A (s3_l25) and exp_B (s3_l25).  We
        # re-export exp_A with new data; exp_B should survive.
        existing = self._build_existing(
            [("exp_A", 3, 25), ("exp_B", 3, 25)],
        )
        payload = _make_minimal_payload(experiment_id="exp_A")

        merged, specs = ssg._merge_payload_into_existing(
            payload=payload, existing_values=existing,
            mode="replace_experiment_blocks",
            current_experiment_id="exp_A",
        )

        # exp_B's rows should still be in the merged grid (the marker
        # in col 1 tells us which exp they belong to).
        exp_b_rows = [r for r in merged
                      if isinstance(r[1], str) and r[1] == "exp_B"]
        assert len(exp_b_rows) == 3  # all 3 data rows preserved
        # exp_A's OLD rows should be gone.
        exp_a_rows = [r for r in merged
                      if isinstance(r[1], str) and r[1] == "exp_A"]
        assert len(exp_a_rows) == 0
        # The new payload's exp_A block should be at the end.
        # _make_minimal_payload yields a single empty block header row.
        merged_sentinels = [r[0] for r in merged
                            if isinstance(r[0], str)
                            and r[0].startswith("[block:")]
        # Should be: exp_B's preserved sentinel + new exp_A's sentinel.
        assert any(s.startswith("[block:exp_B/") for s in merged_sentinels)
        assert any(s.startswith("[block:exp_A/") for s in merged_sentinels)

    def test_append_only_keeps_old_blocks_including_same_exp(self):
        existing = self._build_existing([("exp_A", 3, 25)])
        payload = _make_minimal_payload(experiment_id="exp_A")
        merged, _specs = ssg._merge_payload_into_existing(
            payload=payload, existing_values=existing,
            mode="append_only",
            current_experiment_id="exp_A",
        )
        # OLD exp_A rows preserved (3 of them) AND new payload appended.
        exp_a_rows = [r for r in merged
                      if isinstance(r[1], str) and r[1] == "exp_A"]
        assert len(exp_a_rows) == 3
        sentinels = [r[0] for r in merged
                     if isinstance(r[0], str)
                     and r[0].startswith("[block:exp_A/")]
        # Two block headers in the merged grid: the preserved one
        # plus the freshly-appended one.
        assert len(sentinels) == 2

    def test_wipe_tab_drops_everything(self):
        existing = self._build_existing(
            [("exp_A", 3, 25), ("exp_B", 3, 25)],
        )
        payload = _make_minimal_payload(experiment_id="exp_A")
        merged, _specs = ssg._merge_payload_into_existing(
            payload=payload, existing_values=existing,
            mode="wipe_tab",
            current_experiment_id="exp_A",
        )
        # No exp_A or exp_B rows preserved.
        for marker in ("exp_A", "exp_B"):
            preserved = [r for r in merged
                         if isinstance(r[1], str) and r[1] == marker]
            assert preserved == []
        # The merged grid starts fresh: payload frozen rows + payload block.
        # exp_A's NEW block sentinel should appear exactly once.
        sentinels = [r[0] for r in merged
                     if isinstance(r[0], str)
                     and r[0].startswith("[block:")]
        assert len(sentinels) == 1
        assert sentinels[0].startswith("[block:exp_A/")

    def test_kept_block_specs_reflect_final_positions(self):
        # The BlockSpec row offsets returned by _merge_payload_into_existing
        # must reflect the merged grid (not the original existing grid),
        # since the CLI uses them to attach conditional formats / find
        # later blocks.
        existing = self._build_existing(
            [("exp_B", 3, 25)],
            n_questions=3,
        )
        payload = _make_minimal_payload(experiment_id="exp_A")
        merged, specs = ssg._merge_payload_into_existing(
            payload=payload, existing_values=existing,
            mode="replace_experiment_blocks",
            current_experiment_id="exp_A",
        )

        # Two final specs: exp_B preserved + exp_A new.
        spec_by_exp = {s.experiment_id: s for s in specs}
        assert "exp_B" in spec_by_exp
        assert "exp_A" in spec_by_exp

        # The sentinel in col A of each spec's header_row must match
        # the spec.sentinel (post-merge row numbering).
        for spec in specs:
            assert merged[spec.header_row][0] == spec.sentinel

    def test_block_validates_frozen_rows_in_non_wipe_modes(self):
        # If the existing tab's questions differ from the new payload's,
        # non-wipe modes should error before producing a corrupt grid.
        old_existing_payload = _make_minimal_payload(
            questions=["old0", "old1", "old2"]
        )
        existing = [old_existing_payload.values[i] for i in range(4)]
        new_payload = _make_minimal_payload(
            questions=["new0", "new1", "new2"]
        )
        with pytest.raises(RuntimeError, match="differs"):
            ssg._merge_payload_into_existing(
                payload=new_payload, existing_values=existing,
                mode="replace_experiment_blocks",
                current_experiment_id="exp_A",
            )

    def test_wipe_tab_bypasses_frozen_validation(self):
        # The whole point of --wipe-tab is to overwrite the frozen
        # header when questions change; the validator should not fire
        # in this mode.
        old_existing_payload = _make_minimal_payload(
            questions=["old0", "old1", "old2"]
        )
        existing = [old_existing_payload.values[i] for i in range(4)]
        new_payload = _make_minimal_payload(
            questions=["new0", "new1", "new2"]
        )
        merged, _specs = ssg._merge_payload_into_existing(
            payload=new_payload, existing_values=existing,
            mode="wipe_tab",
            current_experiment_id="exp_A",
        )
        # Header row (row 0 in May-2026 layout) should reflect NEW questions.
        header_row = merged[0]
        # q0 response col = N_AGG_COLS + 0
        q0_col = sl.N_AGG_COLS
        assert header_row[q0_col] == "new0"


# ---------------------------------------------------------------------------
# A1 column letter helper (used by the values-update range string)
# ---------------------------------------------------------------------------

class TestColIndexToLetter:
    def test_first_column(self):
        assert ssg._col_index_to_letter(1) == "A"

    def test_z(self):
        assert ssg._col_index_to_letter(26) == "Z"

    def test_aa(self):
        assert ssg._col_index_to_letter(27) == "AA"

    def test_az_ba_bz(self):
        assert ssg._col_index_to_letter(52) == "AZ"
        assert ssg._col_index_to_letter(53) == "BA"
        assert ssg._col_index_to_letter(78) == "BZ"

    def test_typical_question_count(self):
        # 14 questions = 5 + 14*4 = 61 columns -> "BI"
        assert ssg._col_index_to_letter(61) == "BI"
