"""Ask batched groups of written cells again, without judging the rest: what a rerun of the cell would change.

--mode unread, in place: each group with a member the cell has no score for (a reply that stopped early or could
not be read; a refusal has its own path).  The group's prompt goes again up to PARSE_RETRIES times, and only the
gaps are filled.  scripts and figures that read the cell see the gaps closed, nothing else moved.

--mode names: cells judged before 2026-11-02, whose prompts show mechanical names (``big picture``), get a sibling
``<cell>jl`` under the judge display form (``big-picture``, Roger, 2026-11-03: "fix that, and rerun the affected
items").  A group whose prompt changes is asked afresh, one ask and up to PARSE_RETRIES more, and all its members
take the new ratings; a group whose prompt is byte-identical keeps its ratings, and gaps are filled as above.  The
source cell is not touched.  Its usage.json in the new cell holds only the re-asks; judge_tier_cost_eval.cell_usage_path
costs the cell as the source.

Groups are rebuilt from the cell's recorded seed, batch size and design, and when the cell has a log of its replies
the rebuilt groups must be those logged, or the cell is left alone.  Flex where the model takes it, otherwise real
time.  Each re-ask's reply and charges go to the cell's reask_log.jsonl.gz.

Usage: uv run python scripts/judge_tier_reask.py --mode names --cells 'gpt-5.6-luna@none@b10n20s?' 'claude-haiku-5-5@b10n20s?' \\
           --axes_file scripts/tier_eval_axes.tsv --instructions_dir CORPUS_AT_1448a84c --dry_run"""
import argparse, asyncio, fnmatch, gzip, json, logging, sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant_axis.judge import RateLimiter, accepts_flex, provider_for_model, warn_if_low_parse_rate
from assistant_axis.judge_pricing import MultiModelUsage, price_for_model
from results_analysis import judge_tier_cost_eval as ev, judge_tier_rubrics as jr, judge_tier_transport as tt

logger = logging.getLogger("judge_tier_reask")
LOG = "reask_log.jsonl.gz"
#: Names a cell is judged under when its provenance records none: every cell before 2026-11-02.
OLD_NAME_FORMAT = ev.CHECKPOINT_DEFAULTS["name_format"]


@dataclass
class Group:
    names: list
    prompt: str
    want: list                  # the members to score
    changed: bool = False       # the prompt differs from the one the cell recorded: every member's rating is replaced
    got: dict = field(default_factory=dict)
    open: bool = False          # a call got no answer: no outcome, a rerun asks again


@dataclass
class Work:
    job: object
    source: Path
    target: Path
    model: str
    rubric: object
    data: dict
    groups: list
    usage: MultiModelUsage = field(default_factory=MultiModelUsage)
    log: list = field(default_factory=list)

    @property
    def prov(self):
        return self.data["_provenance"]


def logged_groups(d: Path):
    """The groups a cell's own log records, or None when it has no log: from the answer log or the Batch API replies."""
    found = set()
    for name, key in ((ev.ANSWERS_ARCHIVE, "batch"), (ev.BATCH_REPLIES_ARCHIVE, "members")):
        if (d / name).exists():
            with gzip.open(d / name, "rt") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get(key) and rec.get("kind") != "refusal":
                        found.add(tuple(rec[key]))
    return found or None


def prepare(job, d: Path, mode: str, data_dir: Path, axis_cache: dict, notes: list):
    """The work a cell needs, or None.  Sets the tool's rubric and round: prompts are built here, before anything is sent."""
    data = json.loads(ev.cell_scores_path(d).read_text())
    prov = data["_provenance"]
    if prov.get("averaged") or int(prov.get("entities_per_call") or 1) < 2:
        return None
    recorded = prov.get("name_format", OLD_NAME_FORMAT)
    if mode == "names" and recorded != OLD_NAME_FORMAT:
        return None
    if mode == "unread" and prov.get("n_entities") is not None and prov.get("n_scored") == prov["n_entities"]:
        return None
    rubric = jr.get(prov["rubric"])
    ev.RUBRIC, ev.INSTRUCTION_ROUND = rubric, prov.get("instruction_round", 0)
    key = (job.axis_dir, rubric.text_source, ev.INSTRUCTION_ROUND)
    if key not in axis_cache:
        ev.load_axis(job, data_dir)
        axis_cache[key] = (job.entities, job.projections, job.axis_spec,
                           ev.resolve_entities(sorted(next(iter(job.projections.values()))), data_dir, ev.entity_text_loader())[0])
    entities, job.projections, job.axis_spec, listed = axis_cache[key]
    # until the port of 2026-11-02 the tool judged a pole the projections listed, and left it out of rho
    full = entities if "name_format" in prov else listed
    built = ev.axis_groups(SimpleNamespace(entities=full), SimpleNamespace(
        shuffle_seed=prov.get("shuffle_seed", 0), entities_per_call=prov["entities_per_call"],
        batch_design=prov.get("batch_design", "shuffle")))
    logged = logged_groups(d)
    if logged is not None and {tuple(e[1] for e in g) for g in built} - logged:
        notes.append(f"{d}: the rebuilt groups are not those its log records; left alone")
        return None
    result, refused = data["result"], set(prov.get("refused") or [])
    was, now = jr.name_shower(recorded, data_dir), jr.name_shower(ev.jr.DEFAULT_NAME_FORMAT, data_dir)
    groups = []
    for g in built:
        names = [e[1] for e in g]
        old = rubric.build_batch(job.axis_spec, job.pair_type, g, show=was)
        new = rubric.build_batch(job.axis_spec, job.pair_type, g, show=now) if mode == "names" else old
        if new != old:
            groups.append(Group(names, new, list(names), changed=True))
        else:
            gaps = [n for n in names if result.get(n) is None and n not in refused]
            if gaps:
                groups.append(Group(names, new, gaps))
    target = d.with_name(d.name + "jl") if mode == "names" else d
    if not groups and mode == "unread":
        return None
    if mode == "names" and (target / "usage.json").exists():
        return None
    return Work(job, d, target, prov["judge_model"], rubric, data, groups)


def clients_for(model: str):
    if provider_for_model(model) == "anthropic":
        return ev.make_anthropic_client(), None, None
    if provider_for_model(model) == "google":
        return ev.make_gemini_client(), None, None
    if accepts_flex(model) is True:
        return ev.make_openai_client("flex"), ev.make_openai_client(None), "flex"
    return ev.make_openai_client(None), None, None


async def ask_groups(works: list, *, budget_usd: float, max_in_flight: int, rps: int, stats: Counter) -> MultiModelUsage:
    run_usage = MultiModelUsage()
    limiter = RateLimiter(rps)
    clients = {w.model: clients_for(w.model) for w in works}
    call_stats = tt.CallStats()

    def gate() -> bool:
        return run_usage.total_cost_usd > budget_usd

    async def one(item) -> None:
        w, g = item
        client, fallback, tier = clients[w.model]
        want, asks = list(g.want), ev.PARSE_RETRIES + (1 if g.changed else 0)
        for ask in range(1, asks + 1):
            charge = ev._Charge(w.usage, run_usage)
            text = await tt.call_model(w.model, g.prompt, int(w.prov["max_tokens"]), limiter, charge, client=client,
                                       settings=w.prov["request_settings"], service_tier=tier, fallback_client=fallback,
                                       stats=call_stats, gate=gate)
            if text is tt.SKIPPED or text is tt.FAILED:
                g.open = True
                stats["calls with no answer"] += 1
                return
            stats[f"{w.model}: calls"] += 1
            reply = text.text if isinstance(text, tt.Truncated) else text
            parsed = w.rubric.parse_batch(reply, len(g.names)) if reply else {}
            filled = [n for n in want if parsed.get(g.names.index(n) + 1) is not None]
            for n in filled:
                g.got[n] = parsed[g.names.index(n) + 1]
            want = [n for n in want if n not in filled]
            w.log.append({"kind": "reask", "batch": g.names, "reply": reply, "filled": filled, "unread": want, "ask": ask,
                          "changed": g.changed, "charges": charge.charges, "prompt": ev.prompt_digest(g.prompt),
                          "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            if not g.changed and filled:
                stats[f"{w.model}: gaps filled on re-ask {ask}"] += len(filled)
            if not want:
                return
        stats[f"{w.model}: still unread"] += len(want)

    await ev.run_bounded(one, [(w, g) for w in works for g in w.groups], max_in_flight)
    return run_usage


def write_work(w: Work, mode: str) -> bool:
    """The cell, its correlations and its usage, usage last; False when a call got no answer, so a rerun asks again."""
    if any(g.open for g in w.groups):
        return False
    result = dict(w.data["result"])
    for g in w.groups:
        if g.changed:
            for n in g.names:
                result.pop(n, None)
        result.update(g.got)
    result = {k: result[k] for k in sorted(result)}
    prov = dict(w.prov)
    prov["n_scored"] = len(result)
    entry = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": mode, "groups": len(w.groups),
             "filled": sorted(n for g in w.groups if not g.changed for n in g.got)}
    if mode == "names":
        prov.update(name_format=jr.DEFAULT_NAME_FORMAT, cell_label=prov.get("cell_label", "") + "jl", reasked_from=w.source.name,
                    groups_reasked=len(w.groups), entities_reasked=sum(len(g.names) for g in w.groups if g.changed))
    prov["reasked"] = list(prov.get("reasked") or []) + [entry]
    w.target.mkdir(parents=True, exist_ok=True)
    job = w.job
    ev.atomic_write_text(json.dumps({"result": result, "_provenance": prov}, indent=2),
                         w.target / f"scores_{w.rubric.mode}.json", logger_obj=logger)
    correlations = ev.compute_correlations(scores_by_mode={w.rubric.mode: result}, projections=job.projections,
                                           slots=list(job.projections.keys()), excluded_set=job.axis_spec.pole_entity_ids())
    ev.atomic_write_text(json.dumps(correlations, indent=2), w.target / "correlations.json", logger_obj=logger)
    lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in w.log)
    earlier = gzip.decompress((w.target / LOG).read_bytes()).decode() if (w.target / LOG).exists() else ""
    ev.atomic_write_bytes(gzip.compress((earlier + lines).encode()), w.target / LOG, logger_obj=logger)
    own = w.target / "usage.json"
    usage = ev.merge_usage_dict(json.loads(own.read_text()), w.usage) if own.exists() else w.usage.as_dict()
    ev.atomic_write_text(json.dumps(usage, indent=2, sort_keys=True), own, logger_obj=logger)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("unread", "names"), required=True)
    ap.add_argument("--cells", nargs="+", required=True, help="cell name patterns")
    ap.add_argument("--axes_file", required=True)
    ap.add_argument("--experiments_dir", default="roger/axis_judge_experiments")
    ap.add_argument("--instructions_dir", required=True, help="the corpus the cells were judged on")
    ap.add_argument("--budget_usd", type=float, default=5.0, help="no further requests once this run has spent this")
    ap.add_argument("--max_in_flight", type=int, default=50)
    ap.add_argument("--requests_per_second", type=int, default=10)
    ap.add_argument("--dry_run", action="store_true", help="say what would be asked: nothing sent, nothing written")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    E, data_dir = Path(args.experiments_dir), Path(args.instructions_dir)
    works, notes, per_model = [], [], Counter()
    for job in ev.jobs_from_axes_file(Path(args.axes_file), E):
        tier = E / f"{job.axis_dir}{ev.TIER_EVAL_SUFFIX}"
        if not tier.is_dir():
            continue
        cache: dict = {}
        for d in sorted(p for p in tier.iterdir() if p.is_dir() and (p / "usage.json").exists()):
            if any(fnmatch.fnmatchcase(d.name, pat) for pat in args.cells):
                w = prepare(job, d, args.mode, data_dir, cache, notes)
                if w:
                    works.append(w)
    est = 0.0
    for w in works:
        rate_in, rate_out = price_for_model(w.model)
        half = 0.5 if clients_tier(w.model) else 1.0
        for g in w.groups:
            asks = 1
            per_model[f"{w.model}: groups"] += 1
            per_model[f"{w.model}: prompts changed"] += g.changed
            per_model[f"{w.model}: gaps"] += 0 if g.changed else len(g.want)
            est += asks * half * (len(g.prompt) / 4 * rate_in + 25 * len(g.names) * rate_out) / 1e6
    print(f"{len(works)} cells to change, {sum(len(w.groups) for w in works)} groups, about ${est:.2f}")
    for k in sorted(per_model):
        print(f"  {k}: {per_model[k]}")
    for n in notes:
        print(f"  note: {n}")
    if args.dry_run or not works:
        return 1 if notes else 0
    stats: Counter = Counter()
    usage = asyncio.run(ask_groups(works, budget_usd=args.budget_usd, max_in_flight=args.max_in_flight,
                                   rps=args.requests_per_second, stats=stats))
    written = sum(write_work(w, args.mode) for w in works)
    n_ok = sum(len(g.got) for w in works for g in w.groups)
    n_all = sum(len(g.want) for w in works for g in w.groups)
    warn_if_low_parse_rate(label=f"judge_tier_reask --mode {args.mode}", n_ok=n_ok, n_total=n_all, logger_obj=logger)
    print(f"{written} of {len(works)} cells written; {n_ok} of {n_all} wanted ratings read")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(usage.log_line())
    return 0 if written == len(works) and not notes else 1


def clients_tier(model: str):
    return provider_for_model(model) == "openai" and accepts_flex(model) is True


if __name__ == "__main__":
    sys.exit(main())
