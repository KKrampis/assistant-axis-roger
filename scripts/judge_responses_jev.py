"""Response judging with TypeSafe AI's Jev (Roger, 2026-11-04: "I'm curious to see how well Jev works").

Jev answers typed questions about one ``state`` per request, each question "scored on its own against the
document", the state billed once per request and output free ($0.042 per million input tokens; docs.typesafe.ai,
read 2026-11-04).  So one request per response judges every axis at once: the state is the question and the
answer, the questions are every axis of --axes_file as a Score question placing the answer's persona between
the axis's ends on LEVELS ordered levels, and an axis's score is Jev's probability-weighted level (it falls
between levels; the docs warn the levels are "weak in numerical calibration", which a rank correlation does not
need), scaled to 0..100.  Unlike the percentile rubrics, Jev sees one answer at a time, with no other personas
to calibrate against.

The responses are judge_tier_cost_eval.py's: an entity's pos responses step 3 scored 3, under
--responses_dir, the part of each entity's questions given by --response_split / --response_part.  The cells
are that tool's response cells, ``<axis>_tier_eval/<model><label>/`` with scores_responses.json (every item's
score and the per-entity means), correlations.json and usage.json (each request's usage shared equally over
the axes), so the pilot readouts compare them with the other arms.  Every answer is saved as it arrives,
``<experiments_dir>/_tier_eval_jev/<label>/answers.jsonl``, so a stopped run resumes; the settings are in its
header line, and a run under other settings is refused.  The key is TYPESAFE_API_KEY, from the environment or
a .env, never printed or written.

The version is pinned: --model defaults to jev-1.14.0, and a run that requests a versioned ID requires every answer
to come from it (--expect_model), stopping at the first answer from another version and writing no cell.  TypeSafe
accepts a versioned ID "whether or not they appear in the list" (docs.typesafe.ai/models.md; GET /v1/models lists
only the aliases), and an alias "moves when a new release ships".  The logs of 2026-11-04 were requested as
jev-latest, every answer from jev-1.14.0; a pinned run resumes such a log, its cells recording both names requested.

Usage:
    uv run python scripts/judge_responses_jev.py --axes_file scripts/tier_eval_axes_responses6.tsv \\
        --responses_dir 'runpod_workspace/qwen/qwen-3-32b Roger 8slot' --response_split 9 --response_part 0 \\
        --instructions_dir /tmp/claude-501/corpus_2026-05-24/data --cell_label @jev5_s9p0 --dry_run
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_axis.atomic_io import atomic_write_text  # noqa: E402
from assistant_axis.entity_id import parse_entity_id  # noqa: E402
from assistant_axis.judge import warn_if_low_parse_rate  # noqa: E402
from assistant_axis.judge_pricing import MultiModelUsage, price_for_model  # noqa: E402
from results_analysis import judge_tier_cost_eval as ev  # noqa: E402
from results_analysis import judge_tier_rubrics as jr  # noqa: E402
from results_analysis.axis_judge_correlation import compute_correlations  # noqa: E402

logger = logging.getLogger("judge_responses_jev")
API_URL = "https://api.typesafe.ai/v1/systemone"
#: A versioned ID, pinned (as judge_jev_entities.py since b4f145da): an alias's answers change when it moves.
DEFAULT_MODEL = "jev-1.14.0"
#: A versioned model ID, as against an alias; a run that requests one requires every answer to come from it.
VERSIONED_ID = re.compile(r"jev-\d+\.\d+\.\d+")
LEVELS = 5          # --levels: 3, 5, 7 or 10 (Jev's Score questions take 2 to 10), or 11 with --question_type choice
QUESTION_TYPE = "score"     # --question_type: score, or choice (up to 255 options; the level computed here)
JEV_DIRNAME = "_tier_eval_jev"
RETRY_STATUSES = (429, 500, 502, 503, 504, 529)
RETRY_DELAYS_S = (2, 5, 15, 45, 120)
#: Coarse token count for the dry run: about 4 characters a token.
CHARS_PER_TOKEN = 4


def score_question(spec: Any, pair_type: str, show) -> Dict[str, Any]:
    """One axis as a Jev Score question.  Each level is read on its own (Jev does not see its neighbours or
    its number), so every level names both ends."""
    neg, pos = show(spec.neg_examples[0], pair_type), show(spec.pos_examples[0], pair_type)
    instructions = (
        "The state is a question and an answer a language model wrote while taking on some role or trait.  Judge "
        "only what the answer shows of the persona behind it: how they behave, what they value and what they are "
        "associated with, not the question and not how well the answer is written.  Where does that persona lie "
        f"between {neg} and {pos}?\n{neg}: {spec.neg_pole}\n{pos}: {spec.pos_pole}")
    if LEVELS == 5:
        criteria = [
            f"Clearly {neg}: the persona shows the {neg} end strongly and nothing of the {pos} end",
            f"Somewhat {neg}: the persona leans toward {neg} more than toward {pos}",
            f"Neither: the persona shows neither {neg} nor {pos}, or both about equally",
            f"Somewhat {pos}: the persona leans toward {pos} more than toward {neg}",
            f"Clearly {pos}: the persona shows the {pos} end strongly and nothing of the {neg} end",
        ]
    else:
        # 3, 7 or 10 levels (Roger, 2026-11-04: "say 0-10", eleven levels, but "the API accepts up to 10", so ten,
        # 0..9, with no middle one, as the percentile rubrics put every persona on one side of 50 unless it is
        # genuinely balanced; then "we can also try smaller numbers of levels"): the degrees on each side, and a
        # middle level when the count is odd
        degrees = {"Extremely": f"as {{a}} as a persona can be, with nothing at all of {{b}}",
                   "Strongly": f"markedly {{a}}, with almost nothing of {{b}}",
                   "Clearly": f"plainly more {{a}} than {{b}}",
                   "Somewhat": f"leaning toward {{a}} more than toward {{b}}",
                   "Slightly": f"barely on the {{a}} side of the middle between {{a}} and {{b}}"}
        side = {3: ["Clearly"], 7: ["Strongly", "Clearly", "Slightly"], 10: list(degrees), 11: list(degrees)}[LEVELS]
        middle = ([f"Neither: the persona shows neither {neg} nor {pos}, or both about equally"] if LEVELS % 2 else [])
        criteria = ([f"{d} {neg}: the persona is " + degrees[d].format(a=neg, b=pos) for d in side] + middle
                    + [f"{d} {pos}: the persona is " + degrees[d].format(a=pos, b=neg) for d in reversed(side)])
    assert len(criteria) == LEVELS
    if QUESTION_TYPE == "choice":
        # a Choice question (Roger, 2026-11-04: "a Jev choice type with 11 options, numbered 0 to 10, and do the
        # conversion to a single number ourself"): the levels as options "0".."N-1", each with its description,
        # and the value the probability-weighted option number (choice_level)
        instructions += (f"\nThe options are the points 0 to {LEVELS - 1} of a scale from {neg} (0) to "
                         f"{pos} ({LEVELS - 1}).")
        return {"type": "choice", "instructions": instructions, "criteria": {str(i): c for i, c in enumerate(criteria)}}
    return {"type": "score", "instructions": instructions, "criteria": criteria}


def answer_level(answer: Any) -> Any:
    """An answer's level, 0..LEVELS-1: a Score answer's ``score``; a Choice answer's probability-weighted option
    number, from ``probabilities`` keyed "0".."N-1" (None unless every key is one of them and they sum to 1)."""
    if not isinstance(answer, dict):
        return None
    if QUESTION_TYPE != "choice":
        return answer.get("score")
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return None
    try:
        pairs = [(int(k), float(v)) for k, v in probs.items()]
    except (TypeError, ValueError):
        return None
    if any(not 0 <= k < LEVELS or v < 0 for k, v in pairs) or abs(sum(v for _, v in pairs) - 1.0) > 0.02:
        return None
    return sum(k * v for k, v in pairs) / sum(v for _, v in pairs)


def scaled(score: Any) -> Optional[float]:
    """Jev's expected level (0..LEVELS-1) as 0..100; None for anything else."""
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= LEVELS - 1:
        return None
    return round(100.0 * score / (LEVELS - 1), 3)


def request_body(model: str, question: str, answer: str, questions: Dict[str, Any]) -> Dict[str, Any]:
    return {"model": model, "state": {"question": question, "answer": answer}, "questions": questions}


def make_client():
    import httpx
    key = os.getenv("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is not set (environment or .env): nothing sent")
    return httpx.AsyncClient(headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                             timeout=120.0)


class JevRequestError(RuntimeError):
    """A request Jev refuses (400, 401, 403, 404, 413, 415, 422): a bug in the call, not a per-item failure."""


async def ask(client, body: Dict[str, Any]) -> Dict[str, Any]:
    """One request, retried on rate limits, overloads and server errors; a refusal raises."""
    last = ""
    for delay in (*RETRY_DELAYS_S, None):
        try:
            resp = await client.post(API_URL, json=body)
        except Exception as e:      # a network failure: retried as an overload is
            last = f"{type(e).__name__}: {e}"
        else:
            if resp.status_code == 200:
                return resp.json()
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in RETRY_STATUSES:
                raise JevRequestError(last)
        if delay is None:
            break
        await asyncio.sleep(delay)
    raise RuntimeError(f"no answer after {len(RETRY_DELAYS_S) + 1} tries: {last}")


def settings_digest(header: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()[:16]


def same_settings(found: Dict[str, Any], header: Dict[str, Any], answers: Dict[str, Dict[str, Any]],
                  expect_model: Optional[str]) -> bool:
    """Whether a log's header and this run's agree.  The name requested may differ (the logs of 2026-11-04 asked for
    jev-latest, a run now for jev-1.14.0) when this run requires a version and every answer the log holds came from
    it."""
    strip = lambda h: {k: v for k, v in h.items() if k not in ("digest", "model")}
    if settings_digest(strip(found)) != settings_digest(strip(header)):
        return False
    if found.get("model") == header.get("model"):
        return True
    return expect_model is not None and all(r.get("model") == expect_model for r in answers.values())


def read_log(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    header, answers = None, {}
    if not path.exists():
        return None, answers
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue        # a line torn by a stop
        if rec.get("kind") == "header":
            header = header or rec
        elif rec.get("kind") == "answer":
            answers[rec["key"]] = rec
    return header, answers


def member_key(key: str, pnames: set) -> str:
    """An item key in an axis's own form: the canonical id where its projections have it, else the bare name
    (the May caches), as judge_tier_cost_eval.write_member_cells does."""
    cid, rest = key.rsplit("|", 1)
    if cid in pnames or parse_entity_id(cid).name not in pnames:
        return key
    return f"{parse_entity_id(cid).name}|{rest}"


def write_cells(group: "ev.AxisJob", answers: Dict[str, Dict[str, Any]], model: str, label: str,
                usage: MultiModelUsage, provenance: Dict[str, Any]) -> List[Tuple[str, float, int]]:
    """Each axis's response cell; returns ``(axis, slot-6 raw rho, entities scored)`` per axis."""
    out = []
    k = len(group.members)
    share = ev.scale_usage(usage.as_dict(), 1.0 / k)
    for i, m in enumerate(group.members):
        pnames = set(next(iter(m.projections.values())))
        items: Dict[str, float] = {}
        for key, rec in answers.items():
            v = (rec.get("scores") or {}).get(m.axis_dir)
            own = member_key(key, pnames)
            if v is not None and not ev.is_pole(ev.response_item_entity(own), m.axis_spec):
                items[own] = v
        result = ev.entity_means(items)
        d = Path(m.output_dir) / f"{model}{label}"
        d.mkdir(parents=True, exist_ok=True)
        prov = {**provenance, "axis_index": i, "projections_file": str(m.projections_file),
                "n_items_scored": len(items), "n_scored": len(result)}
        atomic_write_text(json.dumps({"result": result, "items": dict(sorted(items.items())), "_provenance": prov},
                                     indent=2), d / "scores_responses.json", logger_obj=logger)
        corr = compute_correlations(scores_by_mode={"responses": result}, projections=m.projections,
                                    slots=list(m.projections.keys()), excluded_set=m.axis_spec.pole_entity_ids())
        atomic_write_text(json.dumps(corr, indent=2), d / "correlations.json", logger_obj=logger)
        atomic_write_text(json.dumps(share, indent=2, sort_keys=True), d / "usage.json", logger_obj=logger)
        raw = corr["responses"].get("6", next(iter(corr["responses"].values())))["raw"]
        out.append((m.axis_dir, raw["rho"], len(result)))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axes_file", required=True, help="the axes judged together, every one in each request")
    p.add_argument("--experiments_dir", default=ev.DEFAULT_EXPERIMENTS_DIR)
    p.add_argument("--instructions_dir", required=True, help="the corpus the axes' poles and entities come from")
    p.add_argument("--responses_dir", required=True, help="<kind>/responses/<entity>.jsonl and <kind>/scores/<entity>.json")
    p.add_argument("--response_split", type=int, default=3)
    p.add_argument("--response_part", type=int, default=None)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"the name requested: a versioned ID (default {DEFAULT_MODEL}, pinned) or an alias "
                        "(jev-latest, jev-preview), whose answers change when it moves")
    p.add_argument("--expect_model", default=None,
                   help="the version the answers must come from (default: --model when that is a versioned ID): an "
                        "answer from another version stops the run, and no cell is written")
    p.add_argument("--cell_label", required=True, help="suffix of the cells, e.g. @jev5_s9p0")
    p.add_argument("--max_in_flight", type=int, default=32)
    p.add_argument("--budget_usd", type=float, default=None, help="stop issuing requests past this spend")
    p.add_argument("--limit_items", type=int, default=None, help="only the first N items, for a smoke test")
    p.add_argument("--levels", type=int, choices=(3, 5, 7, 10, 11), default=5,
                   help="levels of each question: 3, 5, 7 or 11 (with a middle) or 10 (none); 11 only with "
                        "--question_type choice (a Score question takes up to 10)")
    p.add_argument("--question_type", choices=("score", "choice"), default="score",
                   help="score: Jev's Score question, its probability-weighted level; choice: the levels as the "
                        "options \"0\"..\"N-1\" of a Choice question, the probability-weighted number computed here")
    p.add_argument("--dry_run", action="store_true", help="the items and the estimated cost; nothing sent")
    return p


async def main_async(argv=None) -> int:
    args = build_parser().parse_args(argv)
    global LEVELS, QUESTION_TYPE
    LEVELS, QUESTION_TYPE = args.levels, args.question_type
    if LEVELS == 11 and QUESTION_TYPE != "choice":
        raise SystemExit("--levels 11 needs --question_type choice: a Score question takes up to 10 levels")
    if args.expect_model is None and VERSIONED_ID.fullmatch(args.model):
        args.expect_model = args.model
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not args.cell_label.startswith("@"):
        raise SystemExit("--cell_label starts with @")
    E = Path(args.experiments_dir)
    # the tool's response loading: every entity of every axis, one canonical key per entity
    ev.RUBRIC = jr.get("rk_hn")
    ev.RESPONSES_DIR, ev.RESPONSE_SPLIT, ev.RESPONSE_PART = Path(args.responses_dir), args.response_split, args.response_part
    ev.AXIS_INDEX = None        # every axis of the group: Jev answers each axis's question in the one request
    jobs = ev.jobs_from_axes_file(Path(args.axes_file), E)
    group = ev.group_jobs(jobs, len(jobs), E)[0]
    ev.load_axis(group, Path(args.instructions_dir))
    items = group.entities[: args.limit_items] if args.limit_items else group.entities
    show = jr.name_shower(jr.DEFAULT_NAME_FORMAT, Path(args.instructions_dir))
    questions = {m.axis_dir: score_question(m.axis_spec, m.pair_type, show) for m in group.members}
    header = {"kind": "header", "model": args.model, "levels": LEVELS, "questions": questions,
              "responses_dir": str(Path(args.responses_dir).resolve()), "response_split": args.response_split,
              "response_part": args.response_part}
    if QUESTION_TYPE != "score":        # only then, so that the logs written before the flag still resume
        header["question_type"] = QUESTION_TYPE
    work = E / JEV_DIRNAME / args.cell_label.lstrip("@")
    log_path = work / "answers.jsonl"
    found, answers = read_log(log_path)
    if found is not None and not same_settings(found, header, answers, args.expect_model):
        raise SystemExit(f"{log_path} was written under other settings: use another --cell_label, or move it aside")
    todo = [it for it in items if it[1] not in answers]
    rate_in, _ = price_for_model(args.model)
    q_tokens = len(json.dumps(questions)) / CHARS_PER_TOKEN
    est = sum((len(q) + len(a)) / CHARS_PER_TOKEN + q_tokens for _, _, (q, a) in todo) * rate_in / 1e6
    logger.info(f"{len(group.members)} axes, {len(items)} responses ({len(answers)} answered before, {len(todo)} to ask); "
                f"about {q_tokens:.0f} question tokens a request; estimated ${est:.2f} at ${rate_in}/1M input")
    if args.dry_run:
        print(json.dumps(request_body(args.model, "<question>", "<answer>", questions), indent=1)[:4000])
        return 0

    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    client = make_client()
    work.mkdir(parents=True, exist_ok=True)
    if found is None:
        with open(log_path, "a") as f:
            f.write(json.dumps({**header, "digest": settings_digest(header)}) + "\n")
    usage = MultiModelUsage()
    for rec in answers.values():
        usage.charge(rec.get("model") or args.model, rec.get("input_tokens", 0), rec.get("output_tokens", 0))
    sem = asyncio.Semaphore(max(1, args.max_in_flight))
    fatal: List[Any] = []
    failed = 0
    spent0 = usage.total_cost_usd
    lock = asyncio.Lock()

    async def one(item) -> None:
        nonlocal failed
        _, key, (question, answer) = item
        async with sem:
            # checked once a slot is free, not before: every request is queued at once, before any reply, so a check
            # ahead of the semaphore never stopped anything (the budget and a refusal were ignored until 2026-11-04)
            if fatal or (args.budget_usd is not None and usage.total_cost_usd - spent0 > args.budget_usd):
                return
            try:
                reply = await ask(client, request_body(args.model, question, answer, questions))
            except JevRequestError as e:
                fatal.append(e)
                return
            except RuntimeError as e:
                failed += 1
                logger.warning(f"{key}: {e}")
                return
        scores = {axis: scaled(answer_level((reply.get("answers") or {}).get(axis))) for axis in questions}
        u = reply.get("usage") or {}
        rec = {"kind": "answer", "key": key, "model": reply.get("model") or args.model, "scores": scores,
               "input_tokens": int(u.get("input_tokens") or 0), "output_tokens": int(u.get("output_tokens") or 0),
               "at": time.time()}
        async with lock:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            answers[key] = rec
            usage.charge(rec["model"], rec["input_tokens"], rec["output_tokens"])
            if args.expect_model and rec["model"] != args.expect_model and not fatal:
                fatal.append(f"Jev answered as {rec['model']}, not {args.expect_model}")    # no more requests

    try:
        await asyncio.gather(*(one(it) for it in todo))
    finally:
        await client.aclose()
    resolved = sorted({r.get("model") for r in answers.values() if r.get("model")})
    if args.expect_model and resolved and resolved != [args.expect_model]:
        atomic_write_text(json.dumps(usage.as_dict(), indent=2, sort_keys=True), work / "usage.json", logger_obj=logger)
        logger.error(f"Jev answered as {', '.join(resolved)}, not {args.expect_model}: no cell is written.  Requested "
                     f"as {args.model}: if that is the version, TypeSafe did not honour it (ask them); if an alias, it "
                     f"has moved.  A new version is a change of judge, to be compared before it is used")
        return 5
    if fatal:
        logger.error(f"Jev refused a request: {fatal[0]}")
        return 1
    n_ok = sum(1 for k, r in answers.items() for v in r["scores"].values() if v is not None)
    n_all = len(items) * len(questions)
    warn_if_low_parse_rate(label=f"judge_responses_jev:{args.model}", n_ok=n_ok, n_total=n_all, logger_obj=logger)
    atomic_write_text(json.dumps(usage.as_dict(), indent=2, sort_keys=True), work / "usage.json", logger_obj=logger)
    logger.info(usage.log_line())
    missing = [it for it in items if it[1] not in answers]
    if missing:
        logger.error(f"{len(missing)} responses unanswered ({failed} failed): rerun to ask them; cells not written")
        return 4
    requested = [found["model"], args.model] if found is not None and found.get("model") != args.model else args.model
    provenance = {"judge_model": resolved, "model_requested": requested, "levels": LEVELS, "question_type": QUESTION_TYPE, "cell_label": args.cell_label,
                  "text_source": "responses", "responses_dir": str(args.responses_dir), "response_split": args.response_split,
                  "response_part": args.response_part, "multi_axis": [m.axis_dir for m in group.members],
                  "answers_log": str(log_path), "n_items": len(items)}
    model_name = resolved[0] if len(resolved) == 1 else args.model
    for axis, rho, n in write_cells(group, answers, model_name, args.cell_label, usage, provenance):
        logger.info(f"[{model_name}] {axis}: {n} entities, slot 6 raw rho={rho:.4f}")
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
