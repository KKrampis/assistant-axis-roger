"""Shared logic for filling allowlisted judge-refusal gaps via a fallback judge.

Used by both:

* :mod:`tools.fill_judge_refusal_gaps` -- the post-hoc audit/fill tool
  that scans existing axis-judge dirs and fills missing entries.
* :func:`results_analysis.axis_judge_correlation.score_static_mode` --
  the producer-side judging loop, which calls
  :func:`fill_allowlisted_gaps_inline` at end-of-mode so a fresh
  judging run is self-healing for allowlisted entities (no separate
  command needed).

The allowlist (registry of which entities each judge model
systematically refuses, plus the fallback judge to use) is
``data/judge_refusal_allowlist.json``.  Currently the only allowlisted
entry is ``(claude-sonnet-4-5, virus|R)`` → fallback to
``claude-haiku-4-5-20251001``; see AGENT_NOTES "Known permanent gap:
virus|R on Sonnet instructions mode".

Adding entries:
  1. Empirically confirm the primary judge reliably refuses an entity
     and the fallback judge accepts the same prompt structure.
  2. Append the (judge_model, entity_id) entry to the allowlist JSON.
  3. (Optional) re-run :mod:`tools.fill_judge_refusal_gaps` to backfill
     any existing caches.

A name appearing in the allowlist means "we've accepted that the
primary judge will refuse and we'll backfill from the fallback".  The
fallback's score goes into the SAME ``scores_*.json`` cache as the
primary's, annotated in ``_provenance.notes.fallback_fills`` so the
substitution is auditable per-entity.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .atomic_io import atomic_write_text

logger = logging.getLogger(__name__)


# Repository-root-anchored default; callers can override via
# ``load_allowlist(path=...)``.
def _default_allowlist_path() -> Path:
    # ``assistant_axis/judge_refusal_fallback.py`` is at the repo root
    # under assistant_axis/; the data dir is a sibling.
    return Path(__file__).resolve().parent.parent / "data" / "judge_refusal_allowlist.json"


def load_allowlist(path: Optional[Path] = None) -> dict:
    """Return the allowlist as a plain dict.  Empty dict if file missing.

    Schema (1.0)::

        {
          "_schema_version": 1,
          "_description": "...",
          "judge_refusals": {
            "<judge_model>": {
              "<entity_id>": {
                "added_at": "YYYY-MM-DD",
                "reason": "...",
                "fallback_judge": "...",
                "fallback_provider": "anthropic" | "openai" | ...,
                "modes": ["descriptions", "instructions"] | None,
                "notes": "..."  (optional)
              },
              ...
            }
          }
        }
    """
    p = path or _default_allowlist_path()
    if not p.exists():
        return {"judge_refusals": {}}
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            f"could not parse judge_refusal_allowlist at {p}: "
            f"{type(e).__name__}: {e}; treating as empty"
        )
        return {"judge_refusals": {}}
    d.setdefault("judge_refusals", {})
    return d


def get_allowlist_entry(
    allowlist: dict, judge_model: str, entity_id: str, mode: Optional[str] = None,
) -> Optional[dict]:
    """Look up a specific (judge_model, entity_id[, mode]) in the allowlist.

    Returns the entry dict (with ``fallback_judge`` etc.) iff the entity
    is allowlisted for this judge AND (if ``mode`` is given) the mode is
    in the entry's ``modes`` list (or the entry has no modes filter).
    """
    judge_table = allowlist.get("judge_refusals", {}).get(judge_model, {})
    entry = judge_table.get(entity_id)
    if entry is None:
        return None
    if mode is not None:
        allowed_modes = entry.get("modes")
        if allowed_modes and mode not in allowed_modes:
            return None
    return entry


def partition_gaps_by_allowlist(
    missing_eids: Iterable[str],
    judge_model: str,
    mode: str,
    allowlist: Optional[dict] = None,
) -> tuple[list[tuple[str, dict]], list[str]]:
    """Split missing entities into (allowlisted_with_entry, unexpected).

    Returns:
        (allowlisted, unexpected) where:

        * ``allowlisted`` is a list of ``(entity_id, entry_dict)`` pairs --
          the entry tells you which fallback judge to use.
        * ``unexpected`` is a list of entity_id strings -- these are gaps
          that aren't in the allowlist; the caller should LOUD-warn.
    """
    if allowlist is None:
        allowlist = load_allowlist()
    allowlisted: list[tuple[str, dict]] = []
    unexpected: list[str] = []
    for eid in missing_eids:
        entry = get_allowlist_entry(allowlist, judge_model, eid, mode=mode)
        if entry is None:
            unexpected.append(eid)
        else:
            allowlisted.append((eid, entry))
    return allowlisted, unexpected


def format_unexpected_banner(
    unexpected: list[tuple[str, str, str]], judge_model: str,
) -> str:
    """Format a loud banner for unexpected refusals.

    ``unexpected`` is a list of ``(axis_label, mode, entity_id)`` tuples
    so the banner can cross-axis (used by the audit tool); the
    producer-side caller passes a single-axis list.
    """
    head = (
        f"\n{'=' * 78}\n"
        f"UNEXPECTED REFUSALS by primary judge {judge_model!r}\n"
        f"{'=' * 78}\n"
        f"  {len(unexpected)} (axis, mode, entity_id) gaps are NOT on the\n"
        f"  judge-refusal allowlist (data/judge_refusal_allowlist.json).\n"
        f"  This means a refusal pattern appeared that wasn't expected.\n"
        f"  Options:\n"
        f"    1. Retry the run (transient).\n"
        f"    2. If reproducible, empirically confirm a fallback judge\n"
        f"       accepts the prompt, then add (judge_model, entity_id)\n"
        f"       to the allowlist and re-run.\n"
        f"    3. If the prompt itself is problematic, fix it.\n"
        f"  Unexpected:\n"
    )
    body = "\n".join(
        f"    {ax!r} / {mode} / {eid}"
        for ax, mode, eid in sorted(unexpected)
    )
    return head + body + f"\n{'=' * 78}\n"


# ---------------------------------------------------------------------------
# Per-entity prompt-content reconstruction (mirrors score_static_mode)
# ---------------------------------------------------------------------------

def kind_of_entity_id(eid: str) -> str:
    """``"virus|R"`` → ``"roles"``; ``"stoic|T"`` → ``"traits"``."""
    if eid.endswith("|R"):
        return "roles"
    if eid.endswith("|T"):
        return "traits"
    raise ValueError(f"unrecognized entity_id suffix: {eid!r}")


def name_of_entity_id(eid: str) -> str:
    return eid[:-2] if eid[-2:] in ("|R", "|T") else eid


def build_content_for_mode(
    entity_kind: str, entity_name: str, mode: str,
    instructions_root: Optional[Path] = None,
) -> str:
    """Replicate the per-entity content construction in
    :func:`results_analysis.axis_judge_correlation.score_static_mode`.

    * Descriptions: the ``description`` field of the entity's
      instruction JSON.
    * Instructions: a numbered list of the ``pos`` strings (with a
      leading newline so the list renders below ``**name**:``).
    """
    root = instructions_root or (Path(__file__).resolve().parent.parent / "data")
    inst_path = root / entity_kind / "instructions" / f"{entity_name}.json"
    j = json.loads(inst_path.read_text())
    if mode == "descriptions":
        return j["description"]
    if mode == "instructions":
        return "\n" + "\n".join(
            f"{i+1}. {it['pos']}" for i, it in enumerate(j["instruction"])
        )
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# Fallback API call + cache patch
# ---------------------------------------------------------------------------

async def call_fallback_async(
    client: Any, model: str, prompt: str, *,
    max_tokens: int = 1024, temperature: float = 0.0,
) -> tuple[Optional[str], dict]:
    """Send a single message to the fallback judge.

    Wraps the sync Anthropic SDK call in ``asyncio.to_thread`` so it can
    be awaited inside an event loop without blocking.  Returns
    ``(response_text, usage_dict)``; ``response_text`` is ``None`` if
    the call errored.

    Broad ``Exception`` catch is intentional: this is a per-call probe
    where ANY error (network, parse, rate-limit, refusal) should return
    cleanly with a captured error string rather than crash the parent
    runner.  Same pattern as
    :func:`results_analysis.axis_judge_correlation._call_with_retry`.
    """
    try:
        resp = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return text, usage
    except Exception as e:  # noqa: BLE001
        return None, {"error": f"{type(e).__name__}: {e}"}


def patch_cache_with_fallback_fill(
    cache_path: Path,
    entity_id: str,
    score: int,
    fallback_judge: str,
    allowlist_reason: str,
) -> None:
    """Insert ``score`` for ``entity_id`` into the cache's ``result``
    dict and annotate ``_provenance.notes.fallback_fills``.  Atomic write
    via :func:`assistant_axis.atomic_io.atomic_write_text`."""
    envelope = json.loads(cache_path.read_text())
    if "result" not in envelope:
        # Pre-envelope flat dict (v1 schema) -- wrap minimally so
        # downstream readers handle it uniformly.
        envelope = {"result": envelope, "_provenance": {"notes": {}}}
    envelope["result"][entity_id] = int(score)
    prov = envelope.setdefault("_provenance", {})
    notes = prov.setdefault("notes", {})
    fills = notes.setdefault("fallback_fills", {})
    fills[entity_id] = {
        "fallback_judge": fallback_judge,
        "filled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "allowlist_reason": allowlist_reason[:140],
    }
    atomic_write_text(json.dumps(envelope, indent=2) + "\n", cache_path)


# ---------------------------------------------------------------------------
# End-to-end "fill a single allowlisted gap" -- shared by the tool + producer
# ---------------------------------------------------------------------------

# Function type for the prompt builder, supplied by the caller.  In the
# audit tool, this rebuilds the prompt from config.json + the entity's
# instructions file; in the producer-side flow, the caller can just
# call ``axis_judge_correlation.build_static_prompt`` directly with the
# AxisSpec it already has in scope, then pass it as a thunk that
# returns the prompt for this (entity_id, mode) pair.
PromptBuilder = Callable[[str, str], str]


async def fill_one_allowlisted_gap(
    *,
    entity_id: str,
    mode: str,
    cache_path: Path,
    allowlist_entry: dict,
    prompt_builder: PromptBuilder,
    parse_score: Callable[[Optional[str]], Optional[int]],
    client: Any,
    dry_run: bool = False,
) -> dict:
    """Fill one allowlisted gap end-to-end.

    Parameters
    ----------
    entity_id, mode, cache_path
        Identifies which (cache_file, entity_id) slot we're filling.
    allowlist_entry
        The allowlist row, with ``fallback_judge`` and ``reason``.
    prompt_builder
        Callable ``(entity_id, mode) -> prompt_str``.  Lets the caller
        decide HOW to build the prompt (full AxisSpec available
        vs. reconstruction from config.json).
    parse_score
        Callable that extracts a signed integer score from the judge's
        raw response text.  In practice this is
        :func:`results_analysis.axis_judge_correlation.parse_signed_score`.
    client
        Anthropic SDK client (for the Anthropic-side fallbacks we
        currently support); future provider expansion would broaden
        this.
    dry_run
        If True, returns ``status='would_fill'`` without making API
        calls.

    Returns
    -------
    dict with keys:
        * ``status``: one of ``'filled'``, ``'would_fill'``,
          ``'fallback_call_failed'``, ``'fallback_unparseable'``,
          ``'error'``
        * Plus context fields (``entity_id``, ``mode``, ``score``,
          ``usage``, ``detail``).
    """
    base = {"entity_id": entity_id, "mode": mode,
            "cache_path": str(cache_path)}
    try:
        prompt = prompt_builder(entity_id, mode)
    except (FileNotFoundError, KeyError, ValueError) as e:
        return {**base, "status": "error",
                "detail": f"prompt build failed: {type(e).__name__}: {e}"}
    if dry_run:
        return {**base, "status": "would_fill", "prompt_len": len(prompt)}
    fallback_judge = allowlist_entry["fallback_judge"]
    text, usage = await call_fallback_async(client, fallback_judge, prompt)
    if text is None:
        return {**base, "status": "fallback_call_failed",
                "detail": usage.get("error", "unknown")}
    score = parse_score(text)
    if score is None:
        snippet = text[:200] + ("..." if len(text) > 200 else "")
        return {**base, "status": "fallback_unparseable",
                "detail": snippet}
    patch_cache_with_fallback_fill(
        cache_path=cache_path,
        entity_id=entity_id,
        score=score,
        fallback_judge=fallback_judge,
        allowlist_reason=allowlist_entry.get("reason", ""),
    )
    return {**base, "status": "filled", "score": int(score),
            "usage": usage, "fallback_judge": fallback_judge}


async def fill_allowlisted_gaps_inline(
    *,
    cache_path: Path,
    mode: str,
    missing_eids: list[str],
    judge_model: str,
    prompt_builder: PromptBuilder,
    parse_score: Callable[[Optional[str]], Optional[int]],
    allowlist: Optional[dict] = None,
    concurrency: int = 3,
    logger_obj: Optional[logging.Logger] = None,
) -> tuple[list[dict], list[str]]:
    """Call site for the *producer-side* automatic fallback.

    Given the set of entities missing from a cache_path at end-of-mode,
    partition by allowlist and fill the allowlisted ones inline using
    the configured fallback judge.  Patches the cache file directly
    (atomic writes).

    Returns
    -------
    (results, unexpected)
        ``results`` is the list of per-entity status dicts (from
        :func:`fill_one_allowlisted_gap`).  ``unexpected`` is the list
        of entity_ids that weren't on the allowlist; the caller is
        responsible for the loud banner warning.

    The function is a no-op (returns ``([], unexpected)``) if all
    missing entities are unexpected, or if the allowlist is empty.

    NOTE: this opens an Anthropic client on demand (lazy import of
    ``anthropic.Anthropic``).  Callers running outside the Anthropic
    ecosystem can skip this by partitioning gaps themselves with
    :func:`partition_gaps_by_allowlist` and not invoking this helper.
    """
    log = logger_obj or logger
    if allowlist is None:
        allowlist = load_allowlist()
    allowlisted, unexpected = partition_gaps_by_allowlist(
        missing_eids, judge_model=judge_model, mode=mode, allowlist=allowlist,
    )
    if not allowlisted:
        return [], unexpected

    # Lazy import: only needed when there's actual fallback work to do.
    # Callers that hit this path need ANTHROPIC_API_KEY in env.
    from anthropic import Anthropic
    client = Anthropic()
    log.info(
        f"[refusal-fallback] {mode}: filling {len(allowlisted)} "
        f"allowlisted gap(s) inline from cache {cache_path.name} "
        f"(primary judge: {judge_model!r})"
    )

    sem = asyncio.Semaphore(concurrency)

    async def _one(eid: str, entry: dict) -> dict:
        async with sem:
            return await fill_one_allowlisted_gap(
                entity_id=eid, mode=mode, cache_path=cache_path,
                allowlist_entry=entry, prompt_builder=prompt_builder,
                parse_score=parse_score, client=client, dry_run=False,
            )

    results = await asyncio.gather(*(
        _one(eid, entry) for eid, entry in allowlisted
    ))

    n_filled = sum(1 for r in results if r["status"] == "filled")
    n_failed = len(results) - n_filled
    if n_failed:
        log.warning(
            f"[refusal-fallback] {mode}: {n_failed}/{len(results)} "
            f"allowlisted fills did NOT succeed: " + "; ".join(
                f"{r['entity_id']}={r['status']}({r.get('detail', '')[:60]})"
                for r in results if r["status"] != "filled"
            )
        )
    else:
        log.info(
            f"[refusal-fallback] {mode}: filled {n_filled}/{len(allowlisted)} "
            f"allowlisted gap(s) cleanly via Haiku fallback"
        )
    return results, unexpected
