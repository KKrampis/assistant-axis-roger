---
paths:
- results_analysis/**/*.py
- assistant_axis/plot_metadata.py
- assistant_axis/plot_palette.py
- notebooks/**
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: plotting

**When:** generating, regenerating, or visually verifying any plot, or writing entity names into plot text.  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=plotting`; edit there, then run `uv run python tools/sync_agent_notes.py`.

### Plot visual verification (mandatory after any plot generation)

After generating or re-rendering ANY plot (`.png`, `.jpg`, `.pdf`),
**read the file back as an image and visually inspect it before
declaring the work done.**  Numbers and exit codes alone are not
sufficient: matplotlib will happily emit a visually broken figure
with a 0 exit code.

#### Verification workflow

1. Generate the plot.
2. Read it back: `Read` tool, path = generated PNG.
3. Inspect every text element for overlap / clipping (checklist
   below).
4. If broken, fix and re-render; verify again.  Loop until clean.

Don't ship "the numbers print correctly so it must be fine."  The
plot is the deliverable.

#### Failure-mode checklist (in order of typical recurrence)

**Vertical stacking inside the title block**

The most frequent issue.  `suptitle_with_specs` (in
`assistant_axis/plot_metadata.py`) renders its bold headline as
`fig.suptitle` and the spec line as `fig.text`.  matplotlib's
`constrained_layout` only sees the suptitle, so:

* The spec line and per-axes titles can stack onto each other on
  short figures, AND
* Once a `fig.colorbar` is also in the figure, `constrained_layout`
  IGNORES `fig.subplots_adjust(top=...)` and the colorbar stretches
  vertically to fill the full axes height — putting its top tick
  label right under (or on top of) the spec line.

Both are fixed at the caller, not in the helper:

* Give the figure enough vertical room (`figsize=(W, ≥ 6")` for
  figures with a 2-line title block + per-panel titles).
* Call `fig.subplots_adjust(top=top_rect)` *after*
  `suptitle_with_specs` returns its `top_rect` value (and use
  `constrained_layout=True` so colorbars still place correctly).
* Shorten colorbar height with `fig.colorbar(im, ax=axes,
  shrink=0.7)` (or smaller) — that's the only colorbar-height knob
  `constrained_layout` honours, so it's the right lever to keep
  the colorbar's top tick well below the spec line.

The `line_height` parameter on `suptitle_with_specs` auto-scales
from `title_fontsize` (in points) and figure height (in inches)
since May 2026, so the helper itself no longer needs per-caller
tuning.  But the figure-size + colorbar-shrink dance is still on
the caller.

**Horizontal collisions next to the colorbar**

The spec line is rendered centred at figure-x = 0.5, but a
colorbar pushes the heatmap / axes left of figure-centre, so the
spec line's right tail can graze the colorbar's top tick label
even when no vertical overlap exists.  Mitigations:

* Shorten the spec line.  Long axis lists (e.g. all 12 v2 axes
  spelled out) belong in the `_provenance` envelope, not in the
  spec line.
* Cut-off labels (long category names overflowing the bottom
  margin; tilted x-tick labels truncated on the right edge):
  rotate to 30°, reduce font, or truncate.

**Other recurring issues**

* **Legend overlap with data.**  Default `loc="best"` can land on
  top of points when the data fills the axes; pin with
  `loc="lower right"` (etc.) when that happens.
* **Colorbar disproportional** to the heatmap when the figure
  gains width without gaining height — purely aesthetic, only
  worth fixing if it confuses the reader.

#### Working recipe — heatmap + colorbar + 2-line title

For a single- or multi-panel heatmap with a shared colorbar and
the standard `suptitle_with_specs` 2-line title block, this
pattern reliably produces a clean layout:

```python
fig, axes = plt.subplots(
    1, n_panels,
    figsize=(5.3 * n_panels, 6.4),  # height ≥ 6 in
    constrained_layout=True,        # needed for the colorbar
)
if n_panels == 1:
    axes = np.array([axes])

# ... draw imshow + per-panel titles ...

fig.colorbar(im, ax=axes.tolist(), shrink=0.7,
             label="Spearman ρ (...)")

_, top_rect = suptitle_with_specs(fig, suptitle, spec_line)
# constrained_layout doesn't see fig.text, so reserve top region
# explicitly.  The shrink=0.7 above is what makes the colorbar
# respect this reservation.
fig.subplots_adjust(top=top_rect)

plt.savefig(out_path, dpi=150, bbox_inches="tight",
            metadata=png_metadata(title=suptitle, inputs=inputs))
plt.close()
```

Reference implementation (with extensive in-file comments
explaining each constraint): `results_analysis/batch_size_pairwise_rho.py::make_heatmap_pair`.

### Plot Provenance Metadata (mandatory for every plot)
Every plot generated in this repo MUST embed a PNG-text-chunk
provenance block via `assistant_axis.png_metadata`.  The contents
depend on whether the plot came from a tracked script or an ad-hoc
exploratory script.

#### Tracked-script plots

```python
from assistant_axis import png_metadata
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            metadata=png_metadata(title=first_line_of_suptitle))
```

The helper auto-fills:
- `Title`         — pass the first line of the figure suptitle / caption
- `Author`        — defaults to "Roger Dearnaley"
- `Software`      — `uv run python <repo-relative path> <args>`, or
  `uv run python -m foo.bar.baz <args>` for `-m` invocations.  Pasteable
  into a shell at the repo root to re-run.
- `Creation Time` — ISO-8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SS+00:00`).
  Pre-2026-05 PNGs / JSON envelopes carry the older local-time format
  (`YYYY-MM-DD HH:MM:SS ±HHMM`); audit / reader code reads the field
  as an opaque string and tolerates either.
- `Source`        — git short SHA (+`+dirty` if working tree dirty)

#### Ad-hoc exploration plots (write the script to /tmp first)

When generating a plot during exploration, write the Python to
`/tmp/<descriptive_name>.py`, then run it.  Inside that script, embed
the source so the plot is self-contained:

```python
from pathlib import Path
from assistant_axis import png_metadata
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            metadata=png_metadata(
                title=first_line_of_suptitle,
                source_text=Path(__file__).read_text(),
            ))
```

This adds:
- `Source Code`        — full body of the entry-point Python file
- `Source Code SHA256` — hex digest, for tamper detection

Recovery is one line:
```python
from PIL import Image
print(Image.open("plot.png").info["Source Code"])
```

For the rare multi-file case, pass `source_files={"main.py": ..., "helper.py": ...}`
instead — they're JSON-serialised into a single chunk plus a
`Source Code Files` index.  But: needing more than one `/tmp/` file is
itself a smell that the work is graduating to a tracked module; prefer
that path.

**Heredoc gotcha**: `uv run python << 'PY' ... PY` invocations cannot
read their own body via `__file__` — they're stdin, not a file.  When
producing a plot worth keeping, write the body to `/tmp/foo.py` first
rather than using a heredoc.

#### Conventions summary

1. **Tracked script** writes a plot → ALWAYS pass
   `metadata=png_metadata(title=...)`.  No `source_text` needed (the
   `Software` field plus the git SHA + tracked source already
   reproduce it).
2. **Ad-hoc `/tmp/foo.py` script** writes a plot → ALWAYS pass
   `metadata=png_metadata(title=..., source_text=Path(__file__).read_text())`.
3. **Inline heredoc** writes a plot → if it's interesting, refactor
   into a `/tmp/foo.py` first; if it's truly disposable, pass at
   minimum `metadata=png_metadata(title=...)` so we get the timestamp
   and git SHA.
4. **Inspect** any plot with `exiftool foo.png` or
   `PIL.Image.open(p).info`.

The helper lives at `assistant_axis/plot_metadata.py`.  PNG
`tEXt`/`iTXt` chunks support up to ~2 GB each — no realistic ceiling
on what we can embed.

#### Caveats

Embedded source captures the script that *called* `savefig`, plus the
git SHA at run time.  It does **not** capture:
- Versions of pip dependencies (use `uv.lock` for that — the SHA pins it).
- Repo files imported by the script (the SHA pins those if the tree was
  clean; otherwise `+dirty` warns you).
- Other on-disk inputs the script reads (datasets, JSON config files).

So embedded source is a strong but not complete archeological record:
it gives you the entry-point + the git tree-state + your Python env's
lockfile.  In a few months, that's almost always enough to reconstruct
a plot — and tells you exactly what's missing if it isn't.

---

### Display-side rule (plots, legends, console output)

Plots and labels show the **bare name only**; kind is encoded
*visually*.  Two collision-name dots in the same scatter is fine:
same label "patient", different colour.  Project convention (locked
in `assistant_axis.plot_palette`):

| kind  | fill colour       | text colour | marker | text style |
| ----- | ----------------- | ----------- | :----: | ---------- |
| trait | `lightgrey`       | `dimgrey`   | `o`    | upright    |
| role  | `lightsteelblue`  | `navy`      | `s`    | italic     |

Helpers: `kind_color(kind)`, `kind_text_color(kind)`, `kind_marker(kind)`,
`kind_text_style(kind)`, `display_label(eid)`.  Reference
implementation: `results_analysis/pair_slice_plots.py` lines 336–410.
