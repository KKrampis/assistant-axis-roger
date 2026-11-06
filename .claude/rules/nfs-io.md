---
paths:
- pipeline/**
- assistant_axis/atomic_io.py
- assistant_axis/tmpfs.py
- assistant_axis/axis.py
- assistant_axis/steering_runner.py
- scripts/**
- runpod_workspace/**
- steering/**
- results_analysis/axis_judge_correlation.py
---
<!-- GENERATED FILE: do not edit.  Source: AGENT_NOTES.md (section markers).  Regenerate with: uv run python tools/sync_agent_notes.py -->
# Rule: nfs-io

**When:** reading or writing anything under /workspace or another network-mounted path.  Loads automatically for files matching the `paths` above.  Source: the sections of [`AGENT_NOTES.md`](AGENT_NOTES.md) marked `rule=nfs-io`; edit there, then run `uv run python tools/sync_agent_notes.py`.

### NFS-safe file I/O (mandatory for `/workspace` reads and writes)

`/workspace` on RunPod is a MooseFS-backed network mount. Direct file
ops are usable but occasionally flaky — partial copies, transient `EIO`,
slow flushes, and (most importantly) **silent short reads from
torch.load** that surface as `RuntimeError("storage has wrong byte
size of dtype ...")` or `RuntimeError("PytorchStreamReader failed
reading zip archive: ... unexpected EOF, expected N more bytes")`.
Both reads and writes need defensive handling.

**The pattern in this codebase:**

- **Writes**: stage to a local-disk temp file (via `TMPDIR`), copy to
  the destination with retry/backoff, then `os.replace` to atomically
  publish the final name, then unlink the temp.  Atomic from a reader's
  point of view: dest never exists in a half-written state.  Tensor
  saves additionally run **three post-copy integrity checks** before
  considering the save successful — any failure triggers another
  copy attempt:
    1. **Size** matches staging byte-count (catches short-writes).
    2. **SHA-256** matches staging digest (catches silent byte-flips
       inside a correct-size file — e.g. corrupted NFS chunks that
       pass length checks; this is the failure mode behind the
       2.6 GB-but-unloadable `r_guardian__casual.pt`).
    3. **`torch.load` round-trip** succeeds (catches version-drift /
       pickle-format issues that byte-equality alone wouldn't).
  Both verify steps default to on; pass
  ``verify_sha256=False`` / ``verify_load=False`` to opt out when
  the hot-path cost isn't warranted.
- **Reads**: direct read inside a 5-attempt retry loop with the
  project-standard **exponential** backoff
  `DEFAULT_RETRY_DELAYS_S = (5, 20, 60, 180)` seconds (cumulative wall
  clock ~265 s, matching the API-retry standard in
  `results_analysis/axis_judge_correlation.py`).  Treats `OSError`,
  **`RuntimeError`** *and* `EOFError` as transient — the
  `RuntimeError` case is the failure mode above; earlier code that
  retried only `OSError` missed it entirely.  On final failure: log
  error and raise; the caller's outer loop is responsible for any
  "skip this item, continue with the rest" semantics.

New call sites should use the `assistant_axis.atomic_io` helpers
rather than re-rolling the pattern:

```python
from assistant_axis.atomic_io import (
    # Writes (via TMPDIR staging + atomic rename)
    atomic_write_text, atomic_write_bytes, append_jsonl, write_jsonl,
    torch_save_with_retry,
    # Reads (direct read in retry loop)
    read_text_with_retry, read_jsonl_with_retry, torch_load_with_retry,
)

atomic_write_text(json.dumps(config) + "\n", "/workspace/.../config.json")
append_jsonl({"foo": 1}, "/workspace/.../records.jsonl")
write_jsonl(records, "/workspace/.../records.jsonl")    # full rewrite
torch_save_with_retry(state_dict, "/workspace/.../checkpoint.pt")

config_text = read_text_with_retry("/workspace/.../config.json")
records = read_jsonl_with_retry("/workspace/.../records.jsonl")
state = torch_load_with_retry("/workspace/.../checkpoint.pt", map_location="cpu")
```

For very large tensors where you want to control torch.save's zipfile
flag (e.g. step 2's ~2.6 GB activation files where the new-style zip
serializer hits "iostream error" on RunPod), pass it through:

```python
torch_save_with_retry(activations_dict, output_file,
                      use_zipfile_serialization=False)
```

**Why:** silent failures here are extremely costly — multi-hour
generation runs that lose their last partial flush, restarts that
choke on their own state files, vectors silently missing because step 4
got a short read and skipped after warning.  The retry loops have
caught real RunPod NFS hiccups and let runs proceed; the size-sanity
check on writes catches the silent-truncation case that no exception
would otherwise reveal.

**Don't retry parse errors.**  `read_text_with_retry`,
`read_jsonl_with_retry` and `torch_load_with_retry` retry only on the
transient set above.  `JSONDecodeError`, `UnicodeDecodeError`, and
`pickle.UnpicklingError` propagate immediately because they almost
always indicate a corrupt-on-disk file rather than NFS flakiness, and
retrying just amplifies the delay.  `read_jsonl_with_retry` does
silently skip malformed lines by default (with a warning) since
records.jsonl can in principle have a half-flushed last line, though
`atomic_write_text` makes that nearly impossible on the writer side;
pass `skip_malformed=False` to make a malformed line fatal instead.

**TMPDIR setup:** drivers call `assistant_axis.tmpfs.setup_tmpdir(target)`
(default `target="/dev/shm"`) which **unconditionally** overrides any
pre-existing `$TMPDIR` so the atomic-write staging files land on RAM-
backed tmpfs.  This is by design: pod-defaults like
`TMPDIR=/workspace/tmp` would silently route every staging write
through slow NFS, and pre-May-2026 we had `setup_tmpdir_if_unset()`
preserve such defaults (requiring `unset TMPDIR` before every run).
Both `pipeline/run_pipeline.sh` and `steering/run_sweep.py` expose a
`--tmpdir <path>` CLI flag; callers who genuinely want to preserve
their existing `$TMPDIR` must pass it explicitly via
`--tmpdir "$TMPDIR"`.  The deprecated `setup_tmpdir_if_unset()` is
kept as a thin shim for external callers but logs a warning.

**Auditing past damage**: `pipeline/scan_missing_vectors.py` walks an
`activations/`+`vectors/`+`scores/` triple and classifies each missing
vector as one of `mysterious` / `corrupt_or_truncated_activation` / `ok_zero_size`
(re-run candidates) vs `missing_scores` / `below_min_count` /
`all_nan_or_empty` (legitimate filter).  Use it to recover from
historical short-read damage that pre-retry step 4 silently skipped.

**Current call sites (keep this list updated as new ones land):**

- Writes via `atomic_io`: `pipeline/2_activations.py` (uses
  `torch_save_with_retry`), `pipeline/4_vectors.py`,
  `pipeline/5_axis.py`, `assistant_axis/steering_runner.py`,
  `steering/run_sweep.py`
- Reads via `atomic_io`: `pipeline/2_activations.py` (responses),
  `pipeline/4_vectors.py`, `pipeline/5_axis.py`,
  `assistant_axis/steering_runner.py`, `steering/run_sweep.py`,
  `assistant_axis/axis.py` (`load_axis`, `load_axis_with_metadata`,
  `load_role_vector` — used by notebooks too)

---
