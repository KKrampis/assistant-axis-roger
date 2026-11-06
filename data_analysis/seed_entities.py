#!/usr/bin/env python3
"""Seeding helper for the corpus-expansion queue (chunk 0 of the plan in
``data/traits/instructions/TRAITS_TO_ADD.md`` § "Seeding plan and chunk
order").

The queue lives in ``data/seed_queue.json``::

    {"_meta": {...}, "entries": [{"stem": ..., "label": ..., "entity_type":
     "trait" | "role", "chunk": ..., "pairing": ..., "partner": ...,
     "description": ..., "status": ..., ...}, ...]}

Entry lifecycle (the ``status`` field)::

    candidate -> ready -> seeded -> generated -> checked -> paired
                                                        \\-> done

``candidate`` = extracted from the queue files, no final description yet;
``ready`` = description written and reviewed; ``seeded`` = seed JSON
written into ``data/{traits,roles}/instructions/``; ``generated`` =
instructions, questions and eval prompt generated; ``checked`` = antonym
check run and classified (traits only); ``paired`` / ``done`` = labels and
arrangement recorded.  Parked statuses that the lifecycle never touches:
``tbd``, ``backlog``, ``not_adopted``, ``superseded``, ``exists`` (a file
already existed when the queue was built).

Subcommands (all read and update the queue; ``--dry-run`` never writes)::

    status   [--chunk C] [--status S] [--list]
    write    (--stems S... | --chunk C [--sub-chunk X]) [--overwrite] [--dry-run]
    generate (--stems S... | --chunk C [--sub-chunk X]) [--dry-run] [--confirm-expensive]
    check    (--stems S... | --chunk C [--sub-chunk X]) [--dry-run]
    pair     --a STEM --b STEM [--regenerate a|b|both|none] [--dry-run]
    rename   --old STEM --new LABEL [--partner STEM] [--no-check] [--dry-run]
    report   [--chunk C]

Procedure the subcommands implement (AGENT_NOTES § "Adding New Trait Clean
Pairs"): seed with ``negative_label = non-<label>`` (or the partner label
by construction for standard-derived pairs), generate, run the antonym
check, read its answer against the registry of existing *and queued*
stems with the decision table in § "Corpus expansion policy", then
``pair`` records reciprocal labels and the ``arrangement`` on both files
and regenerates the new side ``--instructions-only`` so the neg clause
names the real partner.  Roles have no labels and no check: ``write`` and
``generate`` are the whole procedure, plus ``arrangement`` at write time.

``rename`` is Roger's RO action (2026-09-16) for a completion whose antonym
check named a word other than the existing partner: rename the *old*
trait to that word if the name is free (no file, not queued), regenerate
its instructions, and re-check the pair from both sides; ``pair`` then
records it if both sides name each other.  The file moves with ``git mv``
and gets a ``renamed_from`` field; the old stem's RunPod data becomes an
orphan (noted in AGENT_NOTES § regeneration TODO).

Cost guard: generation is about $0.03 per entity on Sonnet 4.6 (observed,
Sep 2026); ``generate`` prints the estimate and refuses a batch over the
$20 line without ``--confirm-expensive`` (the expensive-operations rule).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from assistant_axis.atomic_io import atomic_write_text  # noqa: E402
from assistant_axis.entity_id import normalize_to_file_name  # noqa: E402

DEFAULT_QUEUE = _REPO_ROOT / "data" / "seed_queue.json"
DEFAULT_DATA_DIR = _REPO_ROOT / "data"
COST_PER_ENTITY_USD = 0.03
EXPENSIVE_LINE_USD = 20.0

LIFECYCLE = ["candidate", "ready", "seeded", "generated", "checked", "paired", "done"]
PARKED = ["tbd", "backlog", "not_adopted", "superseded", "exists"]
SET_KINDS = {"set", "ring", "sequence", "triangle", "square", "orthoplex", "tree", "map",
             "tetrahedron", "octahedron", "cube"}
CHECK_CATEGORIES = ("nice", "nice_with_alternatives", "mismatch", "nearly_nice", "nasty", "open", "error")


# ---------------------------------------------------------------------------
# queue I/O
# ---------------------------------------------------------------------------

def load_queue(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        q = json.load(f)
    if "entries" not in q:
        raise SystemExit(f"{path}: no 'entries' key")
    return q


def save_queue(q: dict, path: Path) -> None:
    q.setdefault("_meta", {})["updated"] = date.today().isoformat()
    atomic_write_text(json.dumps(q, indent=1, ensure_ascii=False) + "\n", path)


def write_json(path: Path, data: dict) -> None:
    atomic_write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", path)


def instructions_dir(data_dir: Path, entity_type: str) -> Path:
    return data_dir / ("traits" if entity_type == "trait" else "roles") / "instructions"


def corpus_stems(data_dir: Path) -> dict[str, set[str]]:
    out = {}
    for et in ("trait", "role"):
        d = instructions_dir(data_dir, et)
        out[et] = {p.stem for p in d.glob("*.json")} if d.exists() else set()
    return out


def select_entries(q: dict, stems: Optional[Sequence[str]], chunk: Optional[str],
                   sub_chunk: Optional[str]) -> list[dict]:
    entries = q["entries"]
    if stems:
        wanted = [normalize_to_file_name(s) for s in stems]
        by_stem = {e.get("stem"): e for e in entries}
        missing = [s for s in wanted if s not in by_stem]
        if missing:
            raise SystemExit(f"not in queue: {', '.join(missing)}")
        return [by_stem[s] for s in wanted]
    sel = [e for e in entries if chunk is None or str(e.get("chunk")) == str(chunk)]
    if sub_chunk is not None:
        sel = [e for e in sel if (e.get("sub_chunk") or "").lower().startswith(sub_chunk.lower())]
    return sel


def label_for_stem(q: dict, stem: str, data_dir: Path) -> Optional[str]:
    """Display label of a stem: from the queue, else from the corpus file."""
    for e in q["entries"]:
        if e.get("stem") == stem and e.get("label"):
            return e["label"]
    for et in ("trait", "role"):
        p = instructions_dir(data_dir, et) / f"{stem}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            return d.get("positive_label") or stem.replace("_", " ")
    return None


# ---------------------------------------------------------------------------
# seed files
# ---------------------------------------------------------------------------

def seed_document(entry: dict, q: dict, data_dir: Path) -> dict:
    """Build the seed JSON for one queue entry (no instructions yet)."""
    et = entry["entity_type"]
    desc = (entry.get("description") or "").strip()
    if not desc:
        raise ValueError(f"{entry['stem']}: no final description (status {entry.get('status')})")
    doc: dict = {}
    if et == "trait":
        label = entry["label"]
        doc["positive_label"] = label
        if entry.get("negative_label_at_seed"):
            neg = entry["negative_label_at_seed"]
        elif entry.get("pair_by_construction") and entry.get("partner"):
            neg = label_for_stem(q, entry["partner"], data_dir) or entry["partner"].replace("_", " ")
        else:
            neg = f"non-{label}"
        doc["negative_label"] = neg
    doc["description"] = desc
    if entry.get("source"):
        doc["source"] = entry["source"]
    if entry.get("tags"):
        doc["tags"] = list(entry["tags"])
    kind = (entry.get("pairing") or "singleton").lower()
    members = entry.get("arrangement_members")
    if kind in SET_KINDS and members:
        arr: dict = {"kind": kind, "members": list(members) if kind in ("ring", "sequence") else sorted(members)}
        if entry.get("arrangement_note"):
            arr["note"] = entry["arrangement_note"]
        doc["arrangement"] = arr
    elif kind == "pair" and entry.get("pair_by_construction") and entry.get("partner"):
        doc["arrangement"] = {"kind": "pair", "members": sorted([entry["stem"], entry["partner"]])}
    else:
        # pairs go through the non-X check first and get their arrangement in `pair`
        doc["arrangement"] = {"kind": "singleton"}
    return doc


def cmd_write(args, q: dict, data_dir: Path) -> int:
    sel = select_entries(q, args.stems, args.chunk, args.sub_chunk)
    existing = corpus_stems(data_dir)
    n_written = 0
    for e in sel:
        st = e.get("status")
        if st not in ("ready", "seeded") and not args.stems:
            continue  # chunk mode only takes entries whose description is final
        if st not in ("ready", "seeded"):
            print(f"SKIP {e['stem']}: status {st}, not ready", file=sys.stderr)
            continue
        et = e["entity_type"]
        path = instructions_dir(data_dir, et) / f"{e['stem']}.json"
        if e["stem"] in existing[et] and not args.overwrite:
            print(f"SKIP {e['stem']}: {path.relative_to(data_dir.parent) if path.is_relative_to(data_dir.parent) else path} exists (use --overwrite)", file=sys.stderr)
            continue
        other = "role" if et == "trait" else "trait"
        if e["stem"] in existing[other]:
            print(f"NOTE {e['stem']}: name also exists as a {other} (collision list in AGENT_NOTES)", file=sys.stderr)
        try:
            doc = seed_document(e, q, data_dir)
        except ValueError as err:
            print(f"SKIP {err}", file=sys.stderr)
            continue
        if args.dry_run:
            print(f"DRY-RUN would write {path}: {json.dumps(doc, ensure_ascii=False)[:160]}...")
            continue
        write_json(path, doc)
        e["status"] = "seeded"
        e["seeded_at"] = date.today().isoformat()
        n_written += 1
        print(f"WROTE {path}")
    if n_written and not args.dry_run:
        save_queue(q, args.queue)
        run_tool(["tools/sync_entity_lists.py"], dry_run=False)
    print(f"Done: {n_written} seed file(s) written", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# generation and the antonym check
# ---------------------------------------------------------------------------

def run_tool(script_and_args: Sequence[str], dry_run: bool, capture: bool = False) -> subprocess.CompletedProcess | None:
    cmd = ["uv", "run", "python", *script_and_args]
    print("$ " + " ".join(cmd), file=sys.stderr)
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=_REPO_ROOT, check=False, text=True,
                          capture_output=capture)


def generation_commands(entries: Iterable[dict]) -> list[list[str]]:
    traits = [e["stem"] for e in entries if e["entity_type"] == "trait"]
    roles = [e["stem"] for e in entries if e["entity_type"] == "role"]
    cmds = []
    if traits:
        cmds.append(["data_analysis/regenerate_trait_instructions.py", "--traits", *traits])
    if roles:
        cmds.append(["data_analysis/regenerate_role_instructions.py", "--roles", *roles])
    return cmds


def cmd_generate(args, q: dict, data_dir: Path) -> int:
    sel = [e for e in select_entries(q, args.stems, args.chunk, args.sub_chunk)
           if e.get("status") == "seeded" or (args.stems and e.get("status") in ("seeded", "generated"))]
    if not sel:
        print("nothing to generate (no entries with status seeded)", file=sys.stderr)
        return 0
    est = COST_PER_ENTITY_USD * len(sel)
    print(f"{len(sel)} entities, estimated ${est:.2f} at ${COST_PER_ENTITY_USD}/entity", file=sys.stderr)
    if est > EXPENSIVE_LINE_USD and not args.confirm_expensive and not args.dry_run:
        raise SystemExit(f"estimate ${est:.2f} exceeds the ${EXPENSIVE_LINE_USD:.0f} line: confirm with Roger, then pass --confirm-expensive")
    ok = True
    for cmd in generation_commands(sel):
        res = run_tool(cmd, dry_run=args.dry_run)
        if res is not None and res.returncode != 0:
            ok = False
            print(f"FAILED (exit {res.returncode}): {' '.join(cmd)}", file=sys.stderr)
    if args.dry_run:
        return 0
    if not ok:
        return 1
    # mark only entries whose files now carry instructions
    n = 0
    for e in sel:
        p = instructions_dir(data_dir, e["entity_type"]) / f"{e['stem']}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d.get("instruction"), list) and d["instruction"]:
                e["status"] = "generated"
                e["generated_at"] = date.today().isoformat()
                n += 1
    save_queue(q, args.queue)
    print(f"Done: {n}/{len(sel)} marked generated", file=sys.stderr)
    return 0


def split_candidates(returned: str) -> list[str]:
    return [normalize_to_file_name(c.strip()) for c in returned.split("|") if c.strip()]


def classify_check(returned: str, intended: Optional[str], registry: set[str]) -> dict:
    """Read an antonym-check answer against the decision table in AGENT_NOTES
    § "Corpus expansion policy": one word / several words × known / unknown.
    ``registry`` holds existing *and queued* stems."""
    if not returned or returned == "ERROR":
        return {"category": "error", "candidates": [], "known": [], "intended_hit": False}
    cands = split_candidates(returned)
    known = [c for c in cands if c in registry]
    hit = bool(intended) and intended in cands
    if hit:
        # the intended partner was named: confirmed from this side, with or
        # without alternatives (the alternatives are recorded, not acted on)
        cat = "nice" if len(cands) == 1 else "nice_with_alternatives"
    elif len(cands) == 1:
        cat = "mismatch" if known else "open"
    else:
        cat = "nearly_nice" if len(known) == 1 else ("nasty" if len(known) > 1 else "open")
    return {"category": cat, "candidates": cands, "known": known, "intended_hit": hit}


def build_registry(q: dict, data_dir: Path) -> set[str]:
    reg = set()
    for et_stems in corpus_stems(data_dir).values():
        reg |= et_stems
    for e in q["entries"]:
        if e.get("status") in ("not_adopted", "superseded"):
            continue
        if e.get("stem"):
            reg.add(e["stem"])
        if e.get("label"):
            reg.add(normalize_to_file_name(e["label"]))
    return reg


CHECK_HISTORY_NAME = "antonym_check_history.jsonl"


def check_history_path(data_dir: Path) -> Path:
    """Beside ``instructions/``, not inside it, so nothing that globs the
    corpus reads it (same placement as the usage records)."""
    return data_dir / "traits" / CHECK_HISTORY_NAME


def check_history_record(stem: str, doc: dict, answer: dict, verdict: dict,
                         intended: Optional[str], phase: str) -> dict:
    """One antonym check, kept whole: the candidates the generator named and
    the instruction set it read.  ``doc`` is the trait file *as checked*
    (usually under a ``non-X`` label), so ``instructions`` is the generation
    the answer describes, which a later regeneration would overwrite.

    Roger, 2026-09-28: every sample's candidates are worth keeping (they are
    more candidate labels, and a measure of that set of instructions), and a
    second sample must not erase the first."""
    return {
        "stem": stem,
        "checked_at": date.today().isoformat(),
        "phase": phase,
        "positive_label": doc.get("positive_label"),
        "negative_label_at_check": doc.get("negative_label"),
        "description": doc.get("description"),
        "intended": intended,
        "returned": answer.get("negative_label"),
        "candidates": verdict.get("candidates"),
        "known": verdict.get("known"),
        "category": verdict.get("category"),
        "intended_hit": verdict.get("intended_hit"),
        "score": answer.get("antonym_score"),
        "reasoning": answer.get("reasoning"),
        "instructions": doc.get("instruction"),
        "generator": doc.get("generator"),
    }


def append_check_history(data_dir: Path, records: Iterable[dict]) -> int:
    """Append records to ``data/traits/antonym_check_history.jsonl`` (one JSON
    object per line, never rewritten).  Returns the number appended."""
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    if not lines:
        return 0
    path = check_history_path(data_dir)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    atomic_write_text(existing + "".join(line + "\n" for line in lines), path)
    return len(lines)


def note_check_answer(entry: dict, returned: Optional[str], phase: str) -> None:
    """Keep every sample's answer on the queue entry (``check_result`` holds
    only the latest)."""
    entry.setdefault("check_answers", []).append(
        {"checked_at": date.today().isoformat(), "returned": returned, "phase": phase})


def _trait_doc(data_dir: Path, stem: str) -> dict:
    p = instructions_dir(data_dir, "trait") / f"{stem}.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def cmd_check(args, q: dict, data_dir: Path) -> int:
    sel = [e for e in select_entries(q, args.stems, args.chunk, args.sub_chunk)
           if e["entity_type"] == "trait"
           and (e.get("status") == "generated" or (args.stems and e.get("status") == "checked"))]
    if not sel:
        print("nothing to check (no generated trait entries)", file=sys.stderr)
        return 0
    stems = [e["stem"] for e in sel]
    res = run_tool(["data_analysis/generate_antonyms.py", "--traits", *stems], dry_run=args.dry_run, capture=True)
    if args.dry_run:
        return 0
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)
        return 1
    try:
        results = json.loads(res.stdout)
    except json.JSONDecodeError as err:
        print(f"could not parse antonym output: {err}\n{res.stdout[:500]}", file=sys.stderr)
        return 1
    registry = build_registry(q, data_dir)
    by_label = {normalize_to_file_name(e["label"]): e for e in sel}
    n = 0
    history = []
    for label, r in results.items():
        e = by_label.get(normalize_to_file_name(label))
        if e is None:
            continue
        verdict = classify_check(r.get("negative_label", ""), e.get("partner"), registry)
        e["check_result"] = {
            "returned": r.get("negative_label"), "score": r.get("antonym_score"),
            "reasoning": r.get("reasoning"), **verdict, "checked_at": date.today().isoformat()}
        note_check_answer(e, r.get("negative_label"), "check")
        history.append(check_history_record(e["stem"], _trait_doc(data_dir, e["stem"]), r, verdict,
                                            e.get("partner"), "check"))
        e["status"] = "checked"
        n += 1
        print(f"{e['stem']:28s} -> {r.get('negative_label')!s:28s} score={r.get('antonym_score')} {verdict['category']}"
              + (f"  (intended {e.get('partner')})" if e.get("partner") else ""))
    save_queue(q, args.queue)
    append_check_history(data_dir, history)
    print(f"Done: {n} checked", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# recording a pair
# ---------------------------------------------------------------------------

def merge_pair_arrangement(existing, pair: dict):
    """Add ``pair`` to a file's arrangement field: replace a missing or
    singleton field, keep an identical pair, otherwise append to the list
    (a trait can be one pole of a pair *and* a member of a sequence or
    triangle, as ``malicious`` is)."""
    if existing is None:
        return pair
    items = existing if isinstance(existing, list) else [existing]
    items = [x for x in items if x.get("kind") != "singleton"]  # a singleton (with or without a note) is replaced
    if not items:
        return pair
    same = [x for x in items if x.get("kind") == "pair" and sorted(x.get("members", [])) == pair["members"]]
    if same:
        return existing
    return items + [pair]


def cmd_pair(args, q: dict, data_dir: Path) -> int:
    a, b = normalize_to_file_name(args.a), normalize_to_file_name(args.b)
    pa, pb = (instructions_dir(data_dir, "trait") / f"{s}.json" for s in (a, b))
    for p in (pa, pb):
        if not p.exists():
            raise SystemExit(f"{p} does not exist")
    with open(pa, encoding="utf-8") as f:
        da = json.load(f)
    with open(pb, encoding="utf-8") as f:
        db = json.load(f)
    la, lb = da["positive_label"], db["positive_label"]
    members = sorted([a, b])
    changes = {a: [], b: []}
    if da.get("negative_label") != lb:
        changes[a].append(f"negative_label {da.get('negative_label')!r} -> {lb!r}")
        da["negative_label"] = lb
    if db.get("negative_label") != la:
        changes[b].append(f"negative_label {db.get('negative_label')!r} -> {la!r}")
        db["negative_label"] = la
    arr = {"kind": "pair", "members": members}
    if args.note:
        arr["note"] = args.note
    for stem, d in ((a, da), (b, db)):
        merged = merge_pair_arrangement(d.get("arrangement"), arr)
        if merged != d.get("arrangement"):
            changes[stem].append(f"arrangement -> {json.dumps(merged)}")
            d["arrangement"] = merged
    for stem in (a, b):
        print(f"{stem}: " + ("; ".join(changes[stem]) if changes[stem] else "no change"))
    if args.dry_run:
        return 0
    write_json(pa, da)
    write_json(pb, db)
    regen = {"a": [a], "b": [b], "both": [a, b], "none": []}[args.regenerate]
    if regen:
        res = run_tool(["data_analysis/regenerate_trait_instructions.py", "--traits", *regen,
                        "--instructions-only", "--force"], dry_run=False)
        if res.returncode != 0:
            print("regeneration failed", file=sys.stderr)
            return 1
    for e in q["entries"]:
        if e.get("stem") in (a, b) and e.get("status") in LIFECYCLE:
            e["status"] = "paired"
            e["paired_at"] = date.today().isoformat()
            e["partner"] = b if e["stem"] == a else a
    save_queue(q, args.queue)
    ok = run_tool(["data_analysis/check_arrangements.py", "--quiet"], dry_run=False)
    run_tool(["tools/sync_entity_lists.py"], dry_run=False)
    return 0 if ok.returncode == 0 else 1


def cmd_rename(args, q: dict, data_dir: Path) -> int:
    old = normalize_to_file_name(args.old); new_label = args.new.strip(); new = normalize_to_file_name(new_label)
    tdir = instructions_dir(data_dir, "trait")
    src, dst = tdir / f"{old}.json", tdir / f"{new}.json"
    if not src.exists():
        raise SystemExit(f"{src} does not exist")
    if dst.exists():
        raise SystemExit(f"{dst} already exists: {new!r} is not free")
    queued = [e for e in q["entries"] if e.get("stem") == new and e.get("status") not in ("not_adopted", "superseded")]
    if queued and not args.force:
        raise SystemExit(f"{new!r} is queued ({queued[0].get('status')}, chunk {queued[0].get('chunk')}); pass --force to take the name anyway")
    with open(src, encoding="utf-8") as f:
        d = json.load(f)
    print(f"{old} -> {new}: positive_label {d.get('positive_label')!r} -> {new_label!r}; negative_label stays {d.get('negative_label')!r}")
    if args.dry_run:
        return 0
    if (data_dir.parent / ".git").exists():
        subprocess.run(["git", "mv", str(src), str(dst)], cwd=data_dir.parent, check=True)
    else:
        src.rename(dst)
    d["positive_label"] = new_label
    d["renamed_from"] = {"stem": old, "date": date.today().isoformat(),
                         "reason": args.reason or f"RO: the antonym check on {args.partner or 'the new completion'} named this pole {new_label!r}; renamed, regenerated, re-checked"}
    write_json(dst, d)
    res = run_tool(["data_analysis/regenerate_trait_instructions.py", "--traits", new, "--force"], dry_run=False)
    if res.returncode != 0:
        print("regeneration failed", file=sys.stderr)
        return 1
    for e in q["entries"]:
        if e.get("partner") == old:
            e["partner"] = new
            e["pairing_note"] = (e.get("pairing_note") or "") + f" [partner renamed {old} -> {new} on {date.today().isoformat()} (RO)]"
    save_queue(q, args.queue)
    run_tool(["tools/sync_entity_lists.py"], dry_run=False)
    if args.no_check:
        return 0
    stems = [new] + ([normalize_to_file_name(args.partner)] if args.partner else [])
    res = run_tool(["data_analysis/generate_antonyms.py", "--traits", *stems], dry_run=False, capture=True)
    if res.returncode != 0:
        print(res.stderr[-1500:], file=sys.stderr)
        return 1
    results = json.loads(res.stdout)
    registry = build_registry(q, data_dir)
    history = []
    for label, r in results.items():
        stem = normalize_to_file_name(label)
        intended = (normalize_to_file_name(args.partner) if stem == new else new) if args.partner else None
        v = classify_check(r.get("negative_label", ""), intended, registry)
        print(f"{stem:20s} -> {r.get('negative_label')!s:30s} score={r.get('antonym_score')} {v['category']}" + (f"  (intended {intended})" if intended else ""))
        history.append(check_history_record(stem, _trait_doc(data_dir, stem), r, v, intended, "rename"))
        for e in q["entries"]:
            if e.get("stem") == stem and e["entity_type"] == "trait":
                e["check_result"] = {"returned": r.get("negative_label"), "score": r.get("antonym_score"), "reasoning": r.get("reasoning"), **v, "checked_at": date.today().isoformat()}
                note_check_answer(e, r.get("negative_label"), "rename")
    save_queue(q, args.queue)
    append_check_history(data_dir, history)
    print("if both sides name each other: seed_entities.py pair --a <new completion> --b " + new, file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# status and report
# ---------------------------------------------------------------------------

def summarize(entries: list[dict]) -> dict:
    by_status: dict[str, int] = {}
    by_chunk: dict[str, dict[str, int]] = {}
    for e in entries:
        st = e.get("status") or "?"
        by_status[st] = by_status.get(st, 0) + 1
        c = str(e.get("chunk"))
        by_chunk.setdefault(c, {}).setdefault(st, 0)
        by_chunk[c][st] += 1
    return {"n": len(entries), "by_status": by_status, "by_chunk": by_chunk}


def duplicate_live_stems(q: dict) -> dict[str, list[dict]]:
    """Stems carried by more than one live (non-parked) entry of the same
    entity type; the seeding runs key entries by stem, so two live copies
    mean one of them gets processed under the other's notes (2026-09-25:
    self_assured and authentic)."""
    seen: dict[tuple, list[dict]] = {}
    for e in q["entries"]:
        if e.get("status") in ("not_adopted", "superseded", "exists") or not e.get("stem"):
            continue
        seen.setdefault((e["entity_type"], e["stem"]), []).append(e)
    return {f"{k[1]} ({k[0]})": v for k, v in seen.items() if len(v) > 1}


def cmd_status(args, q: dict, data_dir: Path) -> int:
    dups = duplicate_live_stems(q)
    for stem, entries in dups.items():
        print(f"WARNING duplicate live stem {stem}: " + "; ".join(f"partner={e.get('partner')} status={e.get('status')} sub_chunk={e.get('sub_chunk')}" for e in entries), file=sys.stderr)
    sel = select_entries(q, None, args.chunk, args.sub_chunk)
    if args.status:
        sel = [e for e in sel if e.get("status") == args.status]
    s = summarize(sel)
    print(f"{s['n']} entries")
    print("by status:", json.dumps(s["by_status"], sort_keys=True))
    for c in sorted(s["by_chunk"], key=lambda x: (len(x), x)):
        print(f"  chunk {c:>4s}: {json.dumps(s['by_chunk'][c], sort_keys=True)}")
    if args.list:
        for e in sel:
            part = f" <-> {e['partner']}" if e.get("partner") else ""
            print(f"  [{e.get('status'):>11s}] {e.get('chunk')!s:>3s} {e['entity_type'][0]} {e.get('stem')}{part}  ({e.get('sub_chunk')})")
    return 0


def cmd_report(args, q: dict, data_dir: Path) -> int:
    sel = select_entries(q, None, args.chunk, args.sub_chunk)
    print("| stem | type | chunk | pairing | partner | status | check |")
    print("|---|---|---|---|---|---|---|")
    for e in sel:
        cr = e.get("check_result") or {}
        chk = f"{cr.get('returned', '')} ({cr.get('category', '')})" if cr else ""
        print(f"| {e.get('stem')} | {e['entity_type']} | {e.get('chunk')} | {e.get('pairing')} | {e.get('partner') or ''} | {e.get('status')} | {chk} |")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)

    def selection(sp, need=True):
        g = sp.add_mutually_exclusive_group(required=need)
        g.add_argument("--stems", nargs="+")
        g.add_argument("--chunk")
        sp.add_argument("--sub-chunk", help="prefix match on the sub_chunk field, with --chunk")

    sp = sub.add_parser("status"); sp.add_argument("--chunk"); sp.add_argument("--sub-chunk"); sp.add_argument("--status"); sp.add_argument("--list", action="store_true"); sp.set_defaults(func=cmd_status)
    sp = sub.add_parser("write"); selection(sp); sp.add_argument("--overwrite", action="store_true"); sp.add_argument("--dry-run", action="store_true"); sp.set_defaults(func=cmd_write)
    sp = sub.add_parser("generate"); selection(sp); sp.add_argument("--dry-run", action="store_true"); sp.add_argument("--confirm-expensive", action="store_true"); sp.set_defaults(func=cmd_generate)
    sp = sub.add_parser("check"); selection(sp); sp.add_argument("--dry-run", action="store_true"); sp.set_defaults(func=cmd_check)
    sp = sub.add_parser("pair"); sp.add_argument("--a", required=True); sp.add_argument("--b", required=True); sp.add_argument("--regenerate", choices=["a", "b", "both", "none"], default="a", help="which side gets --instructions-only regeneration so its neg clause names the partner (default: a, the new side)"); sp.add_argument("--note"); sp.add_argument("--dry-run", action="store_true"); sp.set_defaults(func=cmd_pair)
    sp = sub.add_parser("rename"); sp.add_argument("--old", required=True, help="existing trait stem"); sp.add_argument("--new", required=True, help="new positive_label (stem derived)"); sp.add_argument("--partner", help="the new completion to re-check against"); sp.add_argument("--reason"); sp.add_argument("--force", action="store_true", help="take a name that is queued"); sp.add_argument("--no-check", action="store_true"); sp.add_argument("--dry-run", action="store_true"); sp.set_defaults(func=cmd_rename)
    sp = sub.add_parser("report"); sp.add_argument("--chunk"); sp.add_argument("--sub-chunk"); sp.set_defaults(func=cmd_report)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    q = load_queue(args.queue)
    return args.func(args, q, args.data_dir)


if __name__ == "__main__":
    sys.exit(main())
