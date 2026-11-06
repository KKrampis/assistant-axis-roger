---
name: steering-to-gsheet
description: Export a steering experiment directory to Google Sheets (one-time setup, usage, rerun semantics, layout)
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
### Exporting steering experiments to Google Sheets (May 2026)

`tools/steering_to_gsheet.py` turns a steering experiment output
directory into a tab on a Google Sheet.  One spreadsheet hosts an
"experiment set" (a collection of related experiments); within it,
one **tab per (role, axis)** holds the shared question set + baseline
responses as frozen header rows, with one **block per cell**
(`(slot, layer, positions_mode)`) underneath -- one block header
row carrying a machine-readable sentinel + human-readable summary,
followed by signed-strength data rows ordered from most negative to
most positive.

#### One-time Google Cloud setup

1.  Open https://console.cloud.google.com and create (or reuse) a
    project named something like "assistant-axis-sheets".
2.  Enable the **Google Sheets API** and **Google Drive API** in the
    project's API library.
3.  Configure the OAuth consent screen: choose **External**, add
    your own Google account as a test user.  No app verification
    needed -- only the test user can authenticate.
4.  Create credentials: **OAuth 2.0 Client ID** -> application type
    **Desktop app**.  Download the resulting JSON.
5.  Save the JSON as `~/.config/assistant-axis/google_credentials.json`
    (the path `assistant_axis/gsheet_auth.py` reads from).

The first invocation pops a browser tab for consent; the resulting
user token is cached at `~/.config/assistant-axis/google_token.json`
and auto-refreshes thereafter.  Both `google_credentials.json` and
`google_token.json` are gitignored (tree-wide pattern in
[`.gitignore`](.gitignore)) -- don't commit them.

#### Usage

```bash
# Dry-run (no auth required): writes a CSV next to the experiment dir
# and a layout summary to stderr.  Use to sanity-check the layout
# before pushing to Sheets.
uv run python tools/steering_to_gsheet.py \
    /workspace/outputs/qwen-3-32b/steering/architect_ecocentric_v2 \
    --dry-run

# Online: open or create a spreadsheet named "Steering May 2026",
# open or create a tab named "architect ecocentric_anthropocentric"
# (default derived from persona role + axis poles), push the data.
uv run python tools/steering_to_gsheet.py \
    /workspace/outputs/qwen-3-32b/steering/architect_ecocentric_v2 \
    --spreadsheet-name "Steering May 2026"

# Reuse the same spreadsheet for a sibling experiment (lands in a
# different tab because role+axis differ):
uv run python tools/steering_to_gsheet.py \
    /workspace/outputs/qwen-3-32b/steering/chef_helpful_v2 \
    --spreadsheet-id <id-from-first-run>
```

The tool prints the spreadsheet URL on success.

#### Rerun semantics

The default `--replace-experiment-blocks` rerun policy:

  - **Scans col A** of the existing tab for block sentinels in
    `[block:<experiment_id>/s<slot>_l<layer>/<positions_mode>]`
    format.
  - **Drops** existing blocks whose `experiment_id` matches the
    current run.
  - **Preserves** blocks for other experiments in the same tab
    verbatim (their row content is read back from the existing
    grid and re-written).
  - **Appends** the new payload's blocks at the bottom.

This makes re-exports idempotent for a single experiment_id while
keeping comparison blocks from sibling experiments (e.g. v1 vs v2)
intact.

Alternative policies:
  - `--append-only`: never drop existing blocks; just append.
    Useful when you want to keep a stale comparison block.
  - `--wipe-tab`: clear the entire tab (including frozen header)
    before writing.  Use when **questions themselves changed** across
    reruns -- the frozen-row compatibility check otherwise errors
    loudly to prevent silently misaligning rerun blocks against
    stale baselines.

#### Layout

  - **Frozen rows**: persona system prompt (row 1), per-column labels
    + question texts (row 2), baseline responses (row 3).
  - **Frozen column**: signed-strength (col A).
  - **Column groups**: outer group on aggregate cols B-E (mean_coh /
    mean_rp / mean_eff_signed / mean_abs_eff); per-question inner
    group on each `q*_coh` + `q*_rp` + `q*_eff` trio.  Roger can
    toggle each via the `+`/`-` indicators Sheets renders.
  - **Conditional formats** are tab-wide (column-spanning), bounded
    to ~10 + 11 * n_questions rules total well within Sheets' practical
    cap:
      - mean_coh / q*_coh: graded pink at >= 0.5 / 1.0 / 1.5.
      - mean_rp / q*_rp: faint orange at <= 1 (persona drift).
      - mean_eff_signed / q*_eff: sign-coloured graded green/blue at
        \|x\| >= 1 / 2 / 3.
      - q*_response: italic gray on TEXT_EQ `[skipped — incoherent]`
        (catches rows where the runner skipped RP+effect judging due
        to mean_coh >= skip_threshold).
  - **Widths**: 80px strength + aggregates, 400px response cols
    (wrap on), 60px score cols.

Block-header rows have non-numeric content in score columns so they
naturally don't match the numeric comparison rules.

#### Files

  - [`tools/sheet_layout.py`](tools/sheet_layout.py) -- pure data
    shaping (`build_tab`, `make_block_sentinel`, helpers).  No Google
    deps imported; unit-testable via
    [`tools/tests/test_sheet_layout.py`](tools/tests/test_sheet_layout.py).
  - [`tools/steering_to_gsheet.py`](tools/steering_to_gsheet.py) --
    CLI + Sheets I/O glue + merge/append/replace logic.  Merge
    correctness is tested via
    [`tools/tests/test_steering_to_gsheet.py`](tools/tests/test_steering_to_gsheet.py).
  - [`assistant_axis/gsheet_auth.py`](assistant_axis/gsheet_auth.py)
    -- thin `gspread.oauth` wrapper.
