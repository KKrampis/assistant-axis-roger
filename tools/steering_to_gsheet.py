"""Export a steering experiment directory into a Google Sheets tab.

Two output paths:

  - ``--dry-run`` writes a local CSV of the cell grid (no Google
    auth required).  Useful for verifying the layout against real
    data before pushing to Sheets.

  - default (online): authenticates with Google via
    :func:`assistant_axis.gsheet_auth.get_client`, opens (or creates)
    a spreadsheet, opens (or creates) a tab, writes the cell grid,
    and applies frozen rows / column groups / conditional formats /
    column widths via the Sheets ``batchUpdate`` API.

Rerun semantics on a non-empty tab (default
``--replace-experiment-blocks``):

  - The tool scans col A of the existing tab for block sentinels
    matching the current experiment_id and drops those rows.
  - Surviving blocks (other experiments in the same tab) are kept
    in place; their values are preserved verbatim.
  - The new payload's blocks are appended at the bottom.
  - Conditional formats and column widths are reapplied to the
    full (kept + new) data range so styling stays consistent.

``--append-only`` keeps every existing block; ``--wipe-tab`` clears
the tab first.

Example:

    uv run python tools/steering_to_gsheet.py \\
        /workspace/outputs/qwen-3-32b/steering/architect_ecocentric_v2 \\
        --spreadsheet-name "Steering experiments May 2026" \\
        [--tab-name "architect ecocentric_anthropocentric"] \\
        [--replace-experiment-blocks | --append-only | --wipe-tab]

See ``AGENT_NOTES.md`` ("Exporting steering experiments to Google
Sheets") for the one-time OAuth setup.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools import sheet_layout as sl

logger = logging.getLogger("steering_to_gsheet")


# ---------------------------------------------------------------------------
# CSV (dry-run) writer
# ---------------------------------------------------------------------------

def write_csv(payload: sl.TabPayload, out_path: Path) -> None:
    """Write the cell grid as CSV for offline inspection.

    Only ``payload.values`` are emitted.  Frozen rows, dim groups,
    conditional formats, and column widths are echoed to stderr as a
    human-readable summary so the dry-run still surfaces all the
    layout decisions.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in payload.values:
            w.writerow(row)
    logger.info(f"wrote {len(payload.values)} rows x "
                f"{len(payload.values[0]) if payload.values else 0} cols "
                f"to {out_path}")

    summary = (
        f"tab_name: {payload.tab_name!r}\n"
        f"default_spreadsheet_name: {payload.default_spreadsheet_name!r}\n"
        f"frozen_rows: {payload.frozen_rows}, frozen_cols: {payload.frozen_cols}\n"
        f"n_questions: {payload.n_questions}, "
        f"n_blocks: {len(payload.blocks)}\n"
        f"dim_groups: {len(payload.dim_groups)} "
        f"(outer=1 + per-question={payload.n_questions})\n"
        f"cond_formats: {len(payload.cond_formats)}\n"
        f"widths: {len(payload.widths)}\n"
        f"blocks:\n"
    )
    for b in payload.blocks:
        summary += (
            f"  - {b.sentinel} -> rows {b.header_row}..{b.data_row_end} "
            f"(header {b.header_row}, data {b.data_row_start}..{b.data_row_end})\n"
        )
    sys.stderr.write(summary)


# ---------------------------------------------------------------------------
# Sheets push helpers (online path)
# ---------------------------------------------------------------------------

def _open_or_create_spreadsheet(client, *, spreadsheet_id: Optional[str],
                                 spreadsheet_name: str):
    """Resolve the target spreadsheet by id or by exact-name search.

    Falls back to creating a new spreadsheet (with ``spreadsheet_name``)
    when neither path resolves.  The drive.file scope only sees files
    the OAuth user created via this app, so the search space is small
    and unambiguous.
    """
    import gspread
    if spreadsheet_id:
        return client.open_by_key(spreadsheet_id)
    try:
        return client.open(spreadsheet_name)
    except gspread.SpreadsheetNotFound:
        logger.info(f"spreadsheet {spreadsheet_name!r} not found; creating...")
        return client.create(spreadsheet_name)


def _open_or_create_worksheet(spreadsheet, tab_name: str, *, n_cols: int):
    """Resolve the target tab by name; create it (with adequate width)
    when missing.  We pre-size the new tab to ``n_cols`` so the first
    write doesn't trigger an auto-expand round-trip.
    """
    import gspread
    try:
        return spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        logger.info(f"tab {tab_name!r} not found in spreadsheet; creating...")
        # Start with 200 rows -- generous; sheets auto-expand as needed.
        return spreadsheet.add_worksheet(
            title=tab_name, rows=200, cols=max(26, n_cols),
        )


# ---------------------------------------------------------------------------
# Append/replace logic
# ---------------------------------------------------------------------------

def _scan_existing_blocks(
    existing_col_a: List[str],
) -> List[Tuple[int, str]]:
    """Return ``[(row_idx, sentinel), ...]`` for every block-header row
    found in column A of an existing tab.

    A "block-header row" is any row whose col A value matches the
    block-sentinel format ``[block:.../s\\d+_l\\d+/...]``.
    """
    import re
    # experiment_id, positions_mode: no whitespace, no '/', no ']'.
    # Real experiment_ids are filename-stem-like (e.g. architect_ecocentric_v2)
    # so this is the strictest pattern that still admits everything the
    # runner produces.
    pat = re.compile(r"^\[block:[^/\s\]]+/s\d+_l\d+/[^/\s\]]+\]$")
    return [(i, v) for i, v in enumerate(existing_col_a) if pat.match(v)]


def _block_row_ranges(
    existing_col_a: List[str],
    headers: List[Tuple[int, str]],
) -> List[Tuple[int, int, str]]:
    """Convert a list of block headers into ``[(start, end, sentinel), ...]``
    row ranges (start inclusive, end exclusive).  Each block's range
    spans from its header row to the next header row (or end of sheet).
    """
    out: List[Tuple[int, int, str]] = []
    n = len(existing_col_a)
    for i, (start, sentinel) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else n
        out.append((start, end, sentinel))
    return out


def _filter_blocks_to_keep(
    existing_block_ranges: List[Tuple[int, int, str]],
    current_experiment_id: str,
    mode: str,
) -> List[Tuple[int, int, str]]:
    """Pick which existing blocks survive the rerun.

    Args:
        mode: one of "replace_experiment_blocks", "append_only", "wipe_tab".
    """
    if mode == "wipe_tab":
        return []
    if mode == "append_only":
        return list(existing_block_ranges)
    # replace_experiment_blocks (default)
    prefix = f"[block:{current_experiment_id}/"
    return [b for b in existing_block_ranges if not b[2].startswith(prefix)]


def _validate_frozen_compatibility(
    existing_values: List[List[Any]],
    payload: sl.TabPayload,
) -> None:
    """Bail out loudly when an existing tab's header-region rows don't
    match the new payload's.

    Header-region row indices (post-May-2026 layout):
      Row 0 (frozen)  -- column labels + question texts.  Column
                         alignment of baselines / strength data
                         would silently slide if questions differed.
      Row 1 (frozen)  -- baseline responses.  Steering responses in
                         existing blocks were generated against
                         THESE baselines; overwriting the row with a
                         fresh baseline set would visually relate
                         stale blocks to the wrong reference.
      Row 2 (unfrozen) -- persona system prompt.  Baselines and
                          steering responses are persona-dependent;
                          a different persona invalidates every
                          existing block's response interpretation.
      Row 3 (unfrozen) -- axis info (pos / neg pole descriptions).
                          Pole semantics anchor the sign convention
                          for every block's data; mismatch suggests
                          the underlying steering vector / axis
                          changed.

    Caller should pass --wipe-tab (to overwrite the header region and
    accept that pre-existing blocks become semantically detached) OR
    --tab-name <other> (to keep both datasets cleanly separated in
    different tabs).
    """
    if len(existing_values) < sl.N_HEADER_ROWS:
        return  # tab is empty / fresh -- compatible

    new_header = payload.values[0]
    new_baseline_row = payload.values[1]
    new_persona_row = payload.values[2]
    existing_header = existing_values[0] if len(existing_values) > 0 else []
    existing_baseline_row = (
        existing_values[1] if len(existing_values) > 1 else []
    )
    existing_persona_row = (
        existing_values[2] if len(existing_values) > 2 else []
    )

    n = len(new_header)
    if len(existing_header) < n:
        raise RuntimeError(
            "existing tab has fewer columns than new payload "
            f"({len(existing_header)} vs {n}); "
            "questions changed -- pass --wipe-tab to overwrite "
            "the frozen header or --tab-name <other> to use a "
            "separate tab"
        )

    # Persona prompt lives in row 1 col B by build_tab convention.
    if len(existing_persona_row) >= 2 and len(new_persona_row) >= 2:
        if str(existing_persona_row[1]).strip() != str(new_persona_row[1]).strip():
            raise RuntimeError(
                "existing tab's persona system prompt differs from new "
                "payload's.  Pass --wipe-tab to overwrite (existing "
                "blocks become semantically detached from the new "
                "persona) or --tab-name <other> to use a separate tab.\n"
                f"  existing: {str(existing_persona_row[1])[:120]!r}\n"
                f"  new:      {str(new_persona_row[1])[:120]!r}"
            )

    # Question text + baseline response per question.
    for q in range(payload.n_questions):
        col = sl.N_AGG_COLS + sl.N_PER_QUESTION_COLS * q
        if str(existing_header[col]).strip() != str(new_header[col]).strip():
            raise RuntimeError(
                f"existing tab's question text at q{q} differs from new "
                f"payload's questions: existing={existing_header[col]!r}, "
                f"new={new_header[col]!r}.  Pass --wipe-tab to overwrite "
                f"the questions or --tab-name <other> to use a separate tab."
            )
        existing_bl = (
            str(existing_baseline_row[col]).strip()
            if len(existing_baseline_row) > col else ""
        )
        new_bl = str(new_baseline_row[col]).strip()
        if existing_bl and new_bl and existing_bl != new_bl:
            raise RuntimeError(
                f"existing tab's baseline response at q{q} differs from new "
                f"payload's.  Even though questions match, the baseline "
                f"responses were generated under a different config "
                f"(seed / persona / model build).  Pre-existing blocks in "
                f"this tab were measured against the OLD baselines, so "
                f"merging the new payload would silently relate stale "
                f"blocks to wrong reference responses.\n"
                f"  existing baseline (q{q}): "
                f"{existing_bl[:120]!r}\n"
                f"  new baseline      (q{q}): {new_bl[:120]!r}\n"
                f"Pass --wipe-tab to overwrite (existing blocks become "
                f"semantically detached) or --tab-name <other> to keep "
                f"the two datasets in separate tabs."
            )


def _merge_payload_into_existing(
    payload: sl.TabPayload,
    existing_values: List[List[Any]],
    mode: str,
    current_experiment_id: str,
) -> Tuple[List[List[Any]], List[sl.BlockSpec]]:
    """Compute the final cell grid + block specs after merging.

    Strategy:
      1. Take frozen rows from the payload (those drive questions +
         baselines + persona; existing-tab equivalents are validated
         to match for non-wipe modes).
      2. Walk existing rows; keep block ranges that pass the filter,
         preserving their on-disk content as-is.
      3. Append the new payload's blocks at the end.
      4. Recompute BlockSpec row offsets so the returned blocks
         reflect their final positions in the merged grid (used by
         the formatting layer for conditional-format ranges).
    """
    if mode != "wipe_tab":
        _validate_frozen_compatibility(existing_values, payload)

    col_a = [str(r[0]) if r else "" for r in existing_values]
    headers = _scan_existing_blocks(col_a)
    existing_ranges = _block_row_ranges(col_a, headers)
    to_keep = _filter_blocks_to_keep(
        existing_ranges, current_experiment_id, mode
    )

    # Build the merged grid.
    n_cols = (
        sl.N_AGG_COLS + sl.N_PER_QUESTION_COLS * payload.n_questions
    )
    merged: List[List[Any]] = []
    # Frozen header from payload (canonical source of questions/baselines).
    merged.extend(payload.values[:sl.FROZEN_ROWS])

    kept_block_specs: List[sl.BlockSpec] = []
    cursor = sl.FROZEN_ROWS
    for start, end, sentinel in to_keep:
        # Pull rows from existing_values, pad to n_cols.
        for r in existing_values[start:end]:
            merged.append(sl.pad_row(list(r), n_cols))
        # Parse slot/layer/positions_mode + experiment_id back from sentinel
        # so kept block specs stay searchable on the next rerun.
        spec = _parse_block_sentinel_back(
            sentinel,
            header_row=cursor, data_row_start=cursor + 1,
            data_row_end=cursor + (end - start),
        )
        if spec is not None:
            kept_block_specs.append(spec)
        cursor += (end - start)

    # New payload blocks appended.  Their offsets need adjusting to the
    # merged grid's row indices.
    new_blocks_start = cursor
    payload_blocks_start_in_payload = sl.FROZEN_ROWS
    for row in payload.values[payload_blocks_start_in_payload:]:
        merged.append(sl.pad_row(list(row), n_cols))
    new_block_specs: List[sl.BlockSpec] = []
    for spec in payload.blocks:
        offset = new_blocks_start - payload_blocks_start_in_payload
        new_block_specs.append(sl.BlockSpec(
            experiment_id=spec.experiment_id,
            slot=spec.slot,
            layer=spec.layer,
            positions_mode=spec.positions_mode,
            header_row=spec.header_row + offset,
            data_row_start=spec.data_row_start + offset,
            data_row_end=spec.data_row_end + offset,
            sentinel=spec.sentinel,
        ))

    return merged, kept_block_specs + new_block_specs


def _parse_block_sentinel_back(
    sentinel: str,
    *, header_row: int, data_row_start: int, data_row_end: int,
) -> Optional[sl.BlockSpec]:
    """Recover a :class:`BlockSpec` from an in-cell sentinel string.

    Format: ``[block:<exp_id>/s<slot>_l<layer>/<positions_mode>]``.
    Returns None on parse failure (caller continues without that spec
    -- doesn't impact the actual grid being written, only future
    find-and-replace).
    """
    import re
    m = re.match(
        r"^\[block:([^/\s\]]+)/s(\d+)_l(\d+)/([^/\s\]]+)\]$",
        sentinel,
    )
    if not m:
        return None
    exp_id, slot, layer, positions = m.groups()
    return sl.BlockSpec(
        experiment_id=exp_id, slot=int(slot), layer=int(layer),
        positions_mode=positions,
        header_row=header_row,
        data_row_start=data_row_start,
        data_row_end=data_row_end,
        sentinel=sentinel,
    )


# ---------------------------------------------------------------------------
# Sheets batchUpdate request builders
# ---------------------------------------------------------------------------

def _color_dict(rgb: Tuple[float, float, float]) -> Dict[str, float]:
    return {"red": rgb[0], "green": rgb[1], "blue": rgb[2]}


def _grid_range(*, sheet_id: int, cond: sl.CondFormat) -> Dict[str, int]:
    return {
        "sheetId": sheet_id,
        "startRowIndex": cond.row_start,
        "endRowIndex": cond.row_end,
        "startColumnIndex": cond.col_start,
        "endColumnIndex": cond.col_end,
    }


def _cond_format_request(*, sheet_id: int, cond: sl.CondFormat,
                         index: int) -> Dict[str, Any]:
    """Translate one :class:`CondFormat` into a Sheets
    ``addConditionalFormatRule`` batchUpdate request.
    """
    fmt: Dict[str, Any] = {}
    if cond.bg_color is not None:
        fmt["backgroundColor"] = _color_dict(cond.bg_color)
    text_fmt: Dict[str, Any] = {}
    if cond.italic:
        text_fmt["italic"] = True
    if cond.text_color is not None:
        text_fmt["foregroundColor"] = _color_dict(cond.text_color)
    if text_fmt:
        fmt["textFormat"] = text_fmt

    condition: Dict[str, Any] = {
        "type": cond.condition_type,
        "values": [{"userEnteredValue": str(v)} for v in cond.values],
    }
    return {
        "addConditionalFormatRule": {
            "rule": {
                "ranges": [_grid_range(sheet_id=sheet_id, cond=cond)],
                "booleanRule": {
                    "condition": condition,
                    "format": fmt,
                },
            },
            "index": index,
        }
    }


def _freeze_request(*, sheet_id: int, rows: int, cols: int) -> Dict[str, Any]:
    return {
        "updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id,
                "gridProperties": {
                    "frozenRowCount": rows,
                    "frozenColumnCount": cols,
                },
            },
            "fields": "gridProperties.frozenRowCount,"
                      "gridProperties.frozenColumnCount",
        }
    }


def _dim_group_request(*, sheet_id: int, g: sl.DimGroup) -> Dict[str, Any]:
    return {
        "addDimensionGroup": {
            "range": {
                "sheetId": sheet_id,
                "dimension": g.dim,
                "startIndex": g.start,
                "endIndex": g.end,
            }
        }
    }


def _width_request(*, sheet_id: int, w: sl.ColumnWidth) -> Dict[str, Any]:
    return {
        "updateDimensionProperties": {
            "range": {
                "sheetId": sheet_id,
                "dimension": "COLUMNS",
                "startIndex": w.col,
                "endIndex": w.col + 1,
            },
            "properties": {"pixelSize": w.width_px},
            "fields": "pixelSize",
        }
    }


def _unmerge_all_request(*, sheet_id: int) -> Dict[str, Any]:
    """Idempotent: clear every merge on the tab so we can re-issue
    the payload's merges cleanly.  No-op if nothing is merged.
    """
    return {
        "unmergeCells": {
            "range": {"sheetId": sheet_id},
        }
    }


def _merge_request(*, sheet_id: int, m: sl.MergeRange) -> Dict[str, Any]:
    return {
        "mergeCells": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": m.row_start,
                "endRowIndex": m.row_end,
                "startColumnIndex": m.col_start,
                "endColumnIndex": m.col_end,
            },
            "mergeType": "MERGE_ALL",
        }
    }


def _header_row_height_requests(*, sheet_id: int) -> List[Dict[str, Any]]:
    """Explicit pixelSize for the persona + axis header rows.

    Sheets row heights persist across writes; if a previous push had
    wrap=WRAP on a row, the row height stayed tall even after we
    switched to OVERFLOW.  Setting explicit pixelSize each time
    forces the row to the intended single- or two-line height.

    Persona row (row 2): 25px (single line of default font).
    Axis row (row 3): 48px (two lines, comfortably).
    """
    return [
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id, "dimension": "ROWS",
                    "startIndex": 2, "endIndex": 3,
                },
                "properties": {"pixelSize": 25},
                "fields": "pixelSize",
            }
        },
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id, "dimension": "ROWS",
                    "startIndex": 3, "endIndex": 4,
                },
                "properties": {"pixelSize": 48},
                "fields": "pixelSize",
            }
        },
    ]


def _header_row_wrap_requests(
    *, sheet_id: int, n_cols: int,
) -> List[Dict[str, Any]]:
    """Wrap-strategy overrides for the persona / axis header rows.

    Persona (row 2): OVERFLOW so the prompt stays on a single line and
    the row auto-sizes shorter -- the prompt is one-time reference
    content; saving vertical real estate matters more than reading
    every word at a glance.

    Axis (row 3): WRAP so the embedded newline (separating the two
    pole descriptions) renders as a line break -- the two-line
    representation is exactly what the user asked for, packed in a
    single wide merged cell.
    """
    return [
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 2, "endRowIndex": 3,
                    "startColumnIndex": 0, "endColumnIndex": n_cols,
                },
                "cell": {
                    "userEnteredFormat": {"wrapStrategy": "OVERFLOW_CELL"},
                },
                "fields": "userEnteredFormat.wrapStrategy",
            }
        },
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 3, "endRowIndex": 4,
                    "startColumnIndex": 0, "endColumnIndex": n_cols,
                },
                "cell": {
                    "userEnteredFormat": {"wrapStrategy": "WRAP"},
                },
                "fields": "userEnteredFormat.wrapStrategy",
            }
        },
    ]


def _top_align_all_request(
    *, sheet_id: int, n_rows: int, n_cols: int,
) -> Dict[str, Any]:
    """Single repeatCell setting verticalAlignment=TOP on every cell
    in the tab.  Numeric scores look fine top-aligned; long wrap-on
    text becomes much easier to scan because every cell in a row
    starts at the same baseline.
    """
    return {
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 0,
                "endRowIndex": n_rows,
                "startColumnIndex": 0,
                "endColumnIndex": n_cols,
            },
            "cell": {
                "userEnteredFormat": {"verticalAlignment": "TOP"},
            },
            "fields": "userEnteredFormat.verticalAlignment",
        }
    }


def _response_col_left_border_requests(
    *, sheet_id: int, n_questions: int, n_rows: int,
) -> List[Dict[str, Any]]:
    """One updateBorders request per q*_response column applying a
    solid left border on every row.

    Visually delimits per-question blocks: each block starts with its
    response col (a wide cell), and the left border on that col gives
    the eye a vertical guide between adjacent questions' score trios
    and the next question's response.

    Style: SOLID_MEDIUM is light enough not to overpower the cond-format
    backgrounds but still readable against the wrap-on response text.
    """
    out: List[Dict[str, Any]] = []
    for q in range(n_questions):
        col = sl.N_AGG_COLS + sl.N_PER_QUESTION_COLS * q
        out.append({
            "updateBorders": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": n_rows,
                    "startColumnIndex": col,
                    "endColumnIndex": col + 1,
                },
                "left": {
                    "style": "SOLID_MEDIUM",
                    "color": {"red": 0.4, "green": 0.4, "blue": 0.4},
                },
            }
        })
    return out


def _wrap_response_cols_request(
    *, sheet_id: int, n_questions: int, n_rows: int,
) -> Dict[str, Any]:
    """Single repeatCell request applying WRAP + top-vertical-alignment to
    every q*_response col across all rows.

    Top alignment matters for long wrap-on responses in the frozen
    baseline row (row 3): centred / bottom alignment leaves whitespace
    at the top of the cell when the response is short and pushes the
    visible text downward when the cell auto-expands to fit a tall
    wrap, eating screen real estate.  Top-align keeps the first line
    of every response anchored to the same row baseline so the eye can
    scan across blocks consistently.
    """
    requests: List[Dict[str, Any]] = []
    for q in range(n_questions):
        col = sl.N_AGG_COLS + sl.N_PER_QUESTION_COLS * q
        requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": n_rows,
                    "startColumnIndex": col,
                    "endColumnIndex": col + 1,
                },
                "cell": {
                    "userEnteredFormat": {
                        "wrapStrategy": "WRAP",
                        "verticalAlignment": "TOP",
                    },
                },
                "fields": "userEnteredFormat.wrapStrategy,"
                          "userEnteredFormat.verticalAlignment",
            }
        })
    return {"requests": requests}


def _clear_conditional_formats_request(*, sheet_id: int,
                                       n_existing_rules: int) -> List[Dict[str, Any]]:
    """Remove all existing conditional-format rules on the tab.

    Sheets requires rules to be deleted one at a time by index, and
    deleting reindexes the remaining rules, so we walk backwards.
    """
    return [
        {"deleteConditionalFormatRule": {"sheetId": sheet_id, "index": i}}
        for i in range(n_existing_rules - 1, -1, -1)
    ]


def _delete_dim_groups_request(*, sheet_id: int,
                                groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove all existing column groups so we can re-add fresh ones."""
    out: List[Dict[str, Any]] = []
    for g in groups:
        out.append({
            "deleteDimensionGroup": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": g["range"]["dimension"],
                    "startIndex": g["range"]["startIndex"],
                    "endIndex": g["range"]["endIndex"],
                }
            }
        })
    return out


# ---------------------------------------------------------------------------
# Online push (gspread + batchUpdate)
# ---------------------------------------------------------------------------

def push_to_sheet(
    payload: sl.TabPayload,
    *,
    spreadsheet_id: Optional[str],
    spreadsheet_name: str,
    mode: str,
    experiment_id: str,
) -> str:
    """End-to-end Sheets push.  Returns the spreadsheet URL.

    Caller must have prepared OAuth credentials (handled by
    :func:`assistant_axis.gsheet_auth.get_client`).
    """
    from assistant_axis import gsheet_auth
    client = gsheet_auth.get_client()
    ss = _open_or_create_spreadsheet(
        client, spreadsheet_id=spreadsheet_id,
        spreadsheet_name=spreadsheet_name,
    )
    n_cols = sl.N_AGG_COLS + sl.N_PER_QUESTION_COLS * payload.n_questions
    ws = _open_or_create_worksheet(ss, payload.tab_name, n_cols=n_cols)
    sheet_id = ws.id

    # Read existing values (col A is enough for sentinel scan, but we
    # need the kept-block rows for merge -- one bulk read avoids
    # repeated API round-trips).
    existing_values = ws.get_all_values() if mode != "wipe_tab" else []

    merged_values, _final_block_specs = _merge_payload_into_existing(
        payload=payload, existing_values=existing_values,
        mode=mode, current_experiment_id=experiment_id,
    )
    total_rows = len(merged_values)
    total_cols = n_cols

    # ----- WIPE phase -----
    # Always clear conditional formats + dim groups so we can re-add
    # fresh ones spanning the new row range; clear cell values when
    # --wipe-tab; otherwise just rewrite over the existing cells.
    ws_props = ss.fetch_sheet_metadata()
    sheet_meta = next((s for s in ws_props["sheets"]
                       if s["properties"]["sheetId"] == sheet_id), {})
    existing_rules = sheet_meta.get("conditionalFormats", [])
    existing_dim_groups = sheet_meta.get("columnGroups", [])

    requests: List[Dict[str, Any]] = []
    requests.extend(_clear_conditional_formats_request(
        sheet_id=sheet_id, n_existing_rules=len(existing_rules),
    ))
    requests.extend(_delete_dim_groups_request(
        sheet_id=sheet_id, groups=existing_dim_groups,
    ))
    # Unmerge first so we can re-apply the payload's merges cleanly.
    # Sheets rejects mergeCells over already-merged ranges.
    requests.append(_unmerge_all_request(sheet_id=sheet_id))

    if mode == "wipe_tab":
        requests.append({
            "updateCells": {
                "range": {"sheetId": sheet_id},
                "fields": "userEnteredValue,userEnteredFormat",
            }
        })

    # ----- VALUES -----
    # Write all merged values in one batch; gspread handles A1
    # conversion for us.
    if requests:
        ss.batch_update({"requests": requests})

    if merged_values:
        end_col_letter = _col_index_to_letter(total_cols)
        end_row = total_rows
        cell_range = f"A1:{end_col_letter}{end_row}"
        # Ensure the sheet has enough rows; gspread expands lazily but
        # the API call itself fails if values exceed bounds.
        if ws.row_count < total_rows:
            ws.add_rows(total_rows - ws.row_count + 10)
        if ws.col_count < total_cols:
            ws.add_cols(total_cols - ws.col_count + 2)
        # gspread 6.x changed the argument order to (values, range_name).
        # Use kwargs to be order-independent across both 5.x and 6.x.
        ws.update(
            values=merged_values,
            range_name=cell_range,
            value_input_option="USER_ENTERED",
        )

    # ----- DECORATIONS -----
    decoration_requests: List[Dict[str, Any]] = []
    decoration_requests.append(_freeze_request(
        sheet_id=sheet_id, rows=payload.frozen_rows, cols=payload.frozen_cols,
    ))
    for g in payload.dim_groups:
        decoration_requests.append(_dim_group_request(
            sheet_id=sheet_id, g=g,
        ))
    for w in payload.widths:
        decoration_requests.append(_width_request(
            sheet_id=sheet_id, w=w,
        ))

    # Re-derive conditional formats over the new total row count
    # (payload's were computed against just its own block range).
    full_cond_formats = sl.tab_wide_cond_formats(
        n_questions=payload.n_questions,
        data_row_start=sl.FROZEN_ROWS,
        data_row_end=total_rows,
    )
    for idx, cf in enumerate(full_cond_formats):
        decoration_requests.append(_cond_format_request(
            sheet_id=sheet_id, cond=cf, index=idx,
        ))

    # Top-align every cell first; the wrap_response_cols_request below
    # also sets verticalAlignment=TOP on response cols, but that's
    # narrowly scoped -- the broad request here covers score / agg /
    # strength cells too so every row reads consistently.
    decoration_requests.append(_top_align_all_request(
        sheet_id=sheet_id, n_rows=total_rows, n_cols=total_cols,
    ))

    # Wrap response columns.
    wrap_payload = _wrap_response_cols_request(
        sheet_id=sheet_id,
        n_questions=payload.n_questions,
        n_rows=total_rows,
    )
    decoration_requests.extend(wrap_payload["requests"])

    # Header-row wrap overrides: persona row OVERFLOW (single line),
    # axis row WRAP (renders the embedded newline between poles).
    decoration_requests.extend(_header_row_wrap_requests(
        sheet_id=sheet_id, n_cols=total_cols,
    ))
    # Explicit pixelSize on the persona + axis rows so their heights
    # don't inherit any tall residue from prior pushes.
    decoration_requests.extend(_header_row_height_requests(
        sheet_id=sheet_id,
    ))

    # Vertical left-border on each q*_response col to delimit per-
    # question blocks.
    decoration_requests.extend(_response_col_left_border_requests(
        sheet_id=sheet_id,
        n_questions=payload.n_questions,
        n_rows=total_rows,
    ))

    # Cell merges for the persona + axis info rows (split below the
    # frozen baseline row in the unfrozen header region).
    for m in payload.merges:
        decoration_requests.append(_merge_request(sheet_id=sheet_id, m=m))

    if decoration_requests:
        ss.batch_update({"requests": decoration_requests})

    return ss.url


def _col_index_to_letter(n_cols: int) -> str:
    """Convert a 1-based column count into a Sheets letter (A, B, ...,
    AA, AB, ...).  Used to build the values-update A1 range.
    """
    # n_cols is 1-based (count of columns including the first).
    out = ""
    n = n_cols
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(ord("A") + r) + out
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("experiment_dir", type=Path,
                   help="Path to a steering experiment output directory "
                        "(e.g. /workspace/outputs/qwen-3-32b/steering/architect_ecocentric_v2)")
    p.add_argument("--spreadsheet-id", default=None,
                   help="Attach to an existing spreadsheet by its id. "
                        "Mutually exclusive with --spreadsheet-name; "
                        "id wins when both are set.")
    p.add_argument("--spreadsheet-name", default=None,
                   help="Open (or create) a spreadsheet by name.  Default "
                        "derives from the experiment_id; pass when you want "
                        "a stable named spreadsheet for an experiment set.")
    p.add_argument("--tab-name", default=None,
                   help="Override the default '<role> <pos>_<neg>' tab name.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--replace-experiment-blocks", dest="mode",
        action="store_const", const="replace_experiment_blocks",
        help="Default.  Drop any existing blocks whose sentinel matches "
             "the current experiment_id; preserve other blocks; append "
             "the new payload's blocks at the end.",
    )
    mode.add_argument(
        "--append-only", dest="mode",
        action="store_const", const="append_only",
        help="Never remove any existing blocks; just append.",
    )
    mode.add_argument(
        "--wipe-tab", dest="mode", action="store_const", const="wipe_tab",
        help="Clear the entire tab (including frozen header rows) before "
             "writing the new payload.  Use when questions or baselines "
             "themselves changed across reruns.",
    )
    p.set_defaults(mode="replace_experiment_blocks")
    p.add_argument("--dry-run", action="store_true",
                   help="Write a CSV next to the experiment dir instead of "
                        "calling Google Sheets.  No OAuth required.")
    p.add_argument("--dry-run-csv", type=Path, default=None,
                   help="Override the dry-run CSV output path.  Default "
                        "is <experiment_dir>/sheet_export.csv.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.experiment_dir.exists():
        logger.error(f"experiment dir not found: {args.experiment_dir}")
        return 2

    payload = sl.build_tab(
        args.experiment_dir,
        tab_name_override=args.tab_name,
        spreadsheet_name_override=args.spreadsheet_name,
    )
    logger.info(
        f"built tab payload: tab={payload.tab_name!r}, "
        f"n_questions={payload.n_questions}, "
        f"n_blocks={len(payload.blocks)}, "
        f"n_rows={len(payload.values)}, "
        f"n_cols={len(payload.values[0]) if payload.values else 0}"
    )

    if args.dry_run:
        out_path = args.dry_run_csv or (
            args.experiment_dir / "sheet_export.csv"
        )
        write_csv(payload, out_path)
        return 0

    # Read experiment_id from the payload's first block (or from the
    # config directly if no blocks).
    if payload.blocks:
        experiment_id = payload.blocks[0].experiment_id
    else:
        import json
        with open(args.experiment_dir / "config.json", encoding="utf-8") as f:
            experiment_id = json.load(f).get(
                "experiment_id", args.experiment_dir.name
            )

    url = push_to_sheet(
        payload,
        spreadsheet_id=args.spreadsheet_id,
        spreadsheet_name=args.spreadsheet_name or payload.default_spreadsheet_name,
        mode=args.mode,
        experiment_id=experiment_id,
    )
    logger.info(f"export complete: {url}")
    print(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
