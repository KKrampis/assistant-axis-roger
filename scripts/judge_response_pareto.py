"""Response judging: subsampling against stacking, a Pareto frontier under the 12 projection variants.

Roger, 2026-11-04: "Assuming we're using caching, I'd like to see a 12-way Pareto frontier plot of further
subsampling against our current best approach to stacking.  Label the X axis with estimated total cost for a
full run ... show options for Luna+Haiku and Luna+Haiku+Gemini."

The data: the twelve axes of scripts/tier_eval_axes_responses12.tsv, judged on a ninth of each entity's kept
responses (split 9, part 0; May data), N=20 answers to one question a call, the question and answers as a
cached prefix and the axis after (r1_pn), each call's direction a coin (Roger's rule, AGENT_NOTES "Axis direction
in judge prompts"):

* GPT-5.6 Luna, one axis a call: assembled per call from the axis's own-order and swapped runs on the same
  batches (@r1_pnn20_s9p0g12_a<i> and ..._a<i>w), each call taken the way its coin says.
* GPT-5.6 Luna, axes stacked k a call (--stack k: each batch's axes partitioned afresh, a coin per axis and
  call), in three formats: plain "1, SCORES: 18, 42", labelled "1. SCORES: A=18, B=42", axis by axis
  "A: 1=18, 2=32, ..." (@rk_<fmt>n20_s9p0g12_k<k>).
* Haiku 5.5, one axis a call, coins (@r1_pnn20_s9p0g12_a<i>c).
* Gemini 3.1 Flash-Lite, one axis a call, coins, on its round-robin third of the ninth only (split 27, part 0;
  @r1_pnn20_s27p0g12_a<i>c): only designs where Gemini judges a share come near the budget.

Arms: each judge alone; Luna + Haiku both judging every response (2x) and round-robin (each response by one of
them, by (q + rank) % 18); the trio round-robin ((q + rank) % 27: Gemini 0, Luna 9, Haiku 18, inside the ninth).
Judges blended Luna 4 : Haiku 3 : Gemini 3 (judge_score_combine.combine_tier_judges).  Further subsampling: of
each judge's share, the fraction m/5 by instruction ((p + q + rank + r) % 5 < m, every question kept; it beat
dropping whole questions on the two axes judged in full), averaged over the five rotations r.  Coverage above
the ninth (1/3, all) is extrapolated, hollow points: each arm's per-entity reliability at the ninth from a
split half by question (more coverage adds questions; halves by instruction share their questions and
predicted a fifth of the measured gain), Spearman-Brown to the larger coverage, rho scaled by the square root of the
reliabilities' ratio (checked against the score-alone rubric's third and full cells, printed).

Cost on the x axis: a full run, 200 axes of the October extraction (1,273 entities, 425 kept responses each)
at the point's coverage, every prefix cached and read for every axis after the first: GPT-5.6 Luna by Flex
($0.10/$0.60 per 1M, reads 0.1x, a write 1.25x once per prefix); Haiku 5.5 by Message Batches ($0.05/$0.25, the
same factors; the probe of 2026-11-04 read every prefix in a batch); Gemini 3.1 Flash-Lite by its Batch API
naming an explicit cache per prefix ($0.125/$0.75, cached reads $0.0125; each cache created once at the standard
input rate, $0.25, and stored an hour at $1 per 1M an hour; the implicit cache reads nothing in batch).  Each
judge's tokens a call are its own, measured on these runs; a stacked arm's cost is its measured cost.

Output: <output_dir>/judge_response_pareto.png and .json.  No API calls.

Usage: uv run python scripts/judge_response_pareto.py --output_dir roger/axis_judge_experiments/response_pareto
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from assistant_axis import json_metadata, png_metadata  # noqa: E402
from assistant_axis.judge_score_combine import combine_tier_judges  # noqa: E402
from assistant_axis.provenance import current_file_input, current_files_input  # noqa: E402
from results_analysis import judge_tier_cost_eval as ev  # noqa: E402
from results_analysis import judge_tier_robustness as rob  # noqa: E402
from results_analysis.judge_tier_cost_plot_pooled import pareto_frontier  # noqa: E402

LUNA, HAIKU, GEMINI = "gpt-5.6-luna", "claude-haiku-5-5", "gemini-3.1-flash-lite"
OCTOBER_RESPONSES = 1273 * 425      # kept responses an axis, the October extraction
N_AXES = 200
NINTH = 1 / 9
FRACTIONS = (1, 2, 3, 4, 5)         # m/5 of each judge's share of the ninth
EXTRAPOLATE = {"1/3": 3, "all": 9}  # coverage multiples of the ninth
KEY_RE = re.compile(r"^(.*)\|p(\d+)q(\d+)$")
#: Production prices, USD per 1M tokens: (input, output, cached read, once per prefix per cached token).
PRICES = {
    LUNA: dict(inp=0.10, out=0.60, read=0.010, once=0.10 * 1.25 - 0.010),       # Flex; a write 1.25x, less the read
    HAIKU: dict(inp=0.05, out=0.25, read=0.005, once=0.05 * 1.25 - 0.005),      # Message Batches
    GEMINI: dict(inp=0.125, out=0.75, read=0.0125, once=0.25 + 1.0 * 1.0),     # Batch API, explicit cache: create + 1 h
}
#: The best stacking at each k on the twelve axes (2026-11-04: plain -0.007 at k=2, axis by axis -0.018 at k=4 and
#: -0.020 at k=6, against one axis a call); the other formats are in the json, not the figure.
STACKED = [("rk_pn", 2), ("rk_pa", 4), ("rk_pa", 6)]
FMT_NAME = {"rk_pn": "plain", "rk_pl": "labelled", "rk_pa": "axis by axis"}


def bare_scores(raw: Dict[str, Any]) -> Dict[Tuple[str, int, int], float]:
    """A cell's item scores keyed (bare entity name, p, q); a name scored under both kinds is dropped."""
    out, seen = {}, {}
    for k, v in raw.items():
        m = KEY_RE.match(k)
        if not m or not isinstance(v, (int, float)):
            continue
        key = (m.group(1).split("|")[0], int(m.group(2)), int(m.group(3)))
        seen[key] = seen.get(key, 0) + 1
        out[key] = float(v)
    return {k: v for k, v in out.items() if seen[k] == 1}


class Data:
    def __init__(self, E: Path, axes_file: Path, data_dir: Path, instructions_dir: Path):
        self.E = E
        self.axes = rob.read_axes_tsv(axes_file)
        self.inputs = [current_file_input("axes_file", axes_file)]
        self.read: List[Path] = []
        # the group's entities, ranked as the tool ranks them (sorted canonical ids), by bare name
        ev.RUBRIC = ev.jr.get("r1_pn")
        ev.RESPONSES_DIR = REPO / "runpod_workspace" / "qwen" / "qwen-3-32b Roger 8slot"
        ev.RESPONSE_SPLIT, ev.RESPONSE_PART, ev.AXIS_INDEX = 9, None, None     # every item: the full entity list
        jobs = ev.jobs_from_axes_file(axes_file, E)
        self.group = ev.group_jobs(jobs, len(jobs), E)[0]
        ev.load_axis(self.group, instructions_dir)
        ids = sorted({key.rsplit("|", 1)[0] for _, key, _ in self.group.entities})
        bare = [i.split("|")[0] for i in ids]
        self.rank = {b: r for r, (b, i) in enumerate(zip(bare, ids)) if bare.count(b) == 1}
        self.kinds = rob.vector_kinds(data_dir)

    def cell(self, axis: str, model: str, label: str) -> Optional[Dict[Tuple[str, int, int], float]]:
        p = self.E / f"{axis}{rob.TIER_EVAL_SUFFIX}" / f"{model}{label}" / "scores_responses.json"
        if not p.exists():
            return None
        self.read.append(p)
        return bare_scores(json.load(open(p))["items"])

    def usage(self, axis: str, model: str, label: str) -> Optional[dict]:
        p = self.E / f"{axis}{rob.TIER_EVAL_SUFFIX}" / f"{model}{label}" / "usage.json"
        if not p.exists():
            return None
        self.read.append(p)
        return json.load(open(p))

    def luna_balanced(self, i: int, axis: str) -> Optional[Dict[Tuple[str, int, int], float]]:
        """Luna one axis a call, each call the way its coin says: from the own-order and swapped runs on the
        same batches, the batches read from the own-order run's answer log."""
        own, sw = self.cell(axis, LUNA, f"@r1_pnn20_s9p0g12_a{i}"), self.cell(axis, LUNA, f"@r1_pnn20_s9p0g12_a{i}w")
        if own is None or sw is None:
            return None
        log = self.E / ev.MULTI_DIRNAME / self.group.axis_dir / f"{LUNA}@r1_pnn20_s9p0g12_a{i}" / "answers.jsonl.gz"
        self.read.append(log)
        out: Dict[Tuple[str, int, int], float] = {}
        for line in gzip.decompress(log.read_bytes()).decode().splitlines():
            rec = json.loads(line)
            if rec.get("kind") != "answer" or rec.get("name") != rec.get("batch", [None])[0]:
                continue
            coin = ev._coin("responses", axis, LUNA, *sorted(rec["batch"]))
            for name in rec["batch"]:
                m = KEY_RE.match(name)
                key = (m.group(1).split("|")[0], int(m.group(2)), int(m.group(3)))
                src = sw if coin else own
                if key in src:
                    out[key] = src[key]
        return out


def per_call_profile(data: Data, model: str, label_of) -> dict:
    """A judge's mean tokens a call on its later axes: the cached prefix, the rest of the prompt, the output;
    and the items a call."""
    pre = rest = out = calls = items = 0.0
    for i, ax in enumerate(data.axes[1:], 1):
        u = data.usage(ax.axis_dir, model, label_of(i))
        s = json.load(open(data.E / f"{ax.axis_dir}{rob.TIER_EVAL_SUFFIX}" / f"{model}{label_of(i)}" / "scores_responses.json"))
        if u is None:
            continue
        e = next(iter(u["per_model"].values()))
        read = e.get("cache_read_tokens", 0)
        n = e["n_calls"]
        if model == GEMINI:     # the implicit cache reads part of the prefix: the prefix is the prompt less the axis
            suffix = 353 * (e["prompt_tokens"] / n) / 10270      # the explicit probe: 353 of 10,270
            pre += e["prompt_tokens"] - suffix * n
            rest += suffix * n
        else:
            pre += read
            rest += e["prompt_tokens"] - read
        out += e["completion_tokens"]
        calls += n
        items += s["_provenance"]["n_items"]
    return {"prefix": pre / calls, "rest": rest / calls, "out": out / calls, "items_per_call": items / calls}


def judge_costs(profile: dict, model: str) -> Tuple[float, float]:
    """(USD per response per axis, USD per response once): a later axis's call, and the once-per-prefix write."""
    p = PRICES[model]
    call = (profile["prefix"] * p["read"] + profile["rest"] * p["inp"] + profile["out"] * p["out"]) / 1e6
    once = profile["prefix"] * p["once"] / 1e6
    return call / profile["items_per_call"], once / profile["items_per_call"]


def full_run_cost(parts: Sequence[Tuple[float, Tuple[float, float]]]) -> float:
    """A full run's cost: for each judge's share (fraction of all responses, (per axis, once) per response)."""
    return sum(frac * OCTOBER_RESPONSES * (N_AXES * per_axis + once) for frac, (per_axis, once) in parts)


def subsample(items: Dict[Tuple[str, int, int], float], rank: Dict[str, int], m: int, r: int,
              share: Optional[Tuple[int, int]] = None) -> Dict[Tuple[str, int, int], float]:
    """The items of a share ((q + rank) % mod == part) and of the fraction m/5 by instruction, rotation r."""
    out = {}
    for (n, p, q), v in items.items():
        if n not in rank:
            continue
        if share and (q + rank[n]) % share[0] != share[1]:
            continue
        if m < 5 and (p + q + rank[n] + r) % 5 >= m:
            continue
        out[(n, p, q)] = v
    return out


def means(items: Dict[Tuple[str, int, int], float]) -> Dict[str, float]:
    s: Dict[str, List[float]] = {}
    for (n, _, _), v in items.items():
        s.setdefault(n, []).append(v)
    return {n: float(np.mean(v)) for n, v in s.items()}


def blend(parts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return combine_tier_judges(parts) if len(parts) > 1 else next(iter(parts.values()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments_dir", default=str(rob.DEFAULT_EXPERIMENTS_DIR))
    ap.add_argument("--data_dir", default=str(rob.DEFAULT_DATA_DIR))
    ap.add_argument("--axes_file", default="scripts/tier_eval_axes_responses12.tsv")
    ap.add_argument("--instructions_dir", default="/tmp/claude-501/corpus_2026-05-24/data")
    ap.add_argument("--cache_dir", default=str(rob.DEFAULT_EXPERIMENTS_DIR / "_tier_eval_scratch" / "robustness_cache"))
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args(argv)
    E, data_dir = Path(args.experiments_dir), Path(args.data_dir).resolve()
    data = Data(E, Path(args.axes_file), data_dir, Path(args.instructions_dir))
    axes = data.axes

    # every judge's items on every axis
    judges: Dict[str, Dict[str, Dict]] = {"luna": {}, "haiku": {}, "gemini": {}}
    for i, ax in enumerate(axes):
        judges["luna"][ax.axis_dir] = data.luna_balanced(i, ax.axis_dir)
        judges["haiku"][ax.axis_dir] = data.cell(ax.axis_dir, HAIKU, f"@r1_pnn20_s9p0g12_a{i}c")
        judges["gemini"][ax.axis_dir] = data.cell(ax.axis_dir, GEMINI, f"@r1_pnn20_s27p0g12_a{i}c")
    stacked = {(fmt, k): {ax.axis_dir: data.cell(ax.axis_dir, LUNA, f"@{fmt}n20_s9p0g12_k{k}") for ax in axes}
               for fmt, k in STACKED}
    have = {j: all(v is not None for v in judges[j].values()) for j in judges}
    print(f"judges complete: {have}", file=sys.stderr)

    # costs a response
    prof = {LUNA: per_call_profile(data, LUNA, lambda i: f"@r1_pnn20_s9p0g12_a{i}")}
    if have["haiku"]:
        prof[HAIKU] = per_call_profile(data, HAIKU, lambda i: f"@r1_pnn20_s9p0g12_a{i}c")
    if have["gemini"]:
        prof[GEMINI] = per_call_profile(data, GEMINI, lambda i: f"@r1_pnn20_s27p0g12_a{i}c")
    unit = {m: judge_costs(p, m) for m, p in prof.items()}
    for m, (a, o) in unit.items():
        print(f"[cost] {m}: {json.dumps({k: round(v, 1) for k, v in prof[m].items()})}; ${a * 1e6:.2f} per 1M "
              f"response-axes, ${o * 1e6:.1f} per 1M responses once; full run, every response: "
              f"${full_run_cost([(1.0, unit[m])]):,.0f}", file=sys.stderr)
    stack_unit = {}
    for (fmt, k), cells in stacked.items():
        if all(cells.values()):
            costs = [data.usage(ax.axis_dir, LUNA, f"@{fmt}n20_s9p0g12_k{k}") for ax in axes]
            items = [json.load(open(E / f"{ax.axis_dir}{rob.TIER_EVAL_SUFFIX}" / f"{LUNA}@{fmt}n20_s9p0g12_k{k}" /
                                    "scores_responses.json"))["_provenance"]["n_items"] for ax in axes]
            from results_analysis.judge_tier_cost_plot_pooled import load_measured_cost
            per = np.mean([load_measured_cost(E / f"{ax.axis_dir}{rob.TIER_EVAL_SUFFIX}" / f"{LUNA}@{fmt}n20_s9p0g12_k{k}" /
                                              "usage.json", "batch") / n for ax, n in zip(axes, items)])
            stack_unit[(fmt, k)] = (per, unit[LUNA][1])

    # the arms: (key, label, family, [(judge, items by axis, share)], cost parts per unit coverage)
    arms = []
    def add(key, label, family, members, cost_parts):
        arms.append(dict(key=key, label=label, family=family, members=members, cost_parts=cost_parts))
    add(f"{LUNA}@k1", "Luna", "Luna, one axis a call", [("luna", judges["luna"], None)], [(1.0, unit[LUNA])])
    for (fmt, k), cells in stacked.items():
        if (fmt, k) in stack_unit:
            add(f"{LUNA}@{fmt}k{k}", f"Luna k={k} {FMT_NAME[fmt]}", f"Luna stacked, {FMT_NAME[fmt]}",
                [("luna", cells, None)], [(1.0, stack_unit[(fmt, k)])])
    if have["haiku"]:
        add(f"{HAIKU}@k1", "Haiku", "Haiku, one axis a call", [("haiku", judges["haiku"], None)], [(1.0, unit[HAIKU])])
        add(f"{LUNA}+{HAIKU}@2x", "Luna + Haiku, both", "Luna + Haiku, both judge every response",
            [("luna", judges["luna"], None), ("haiku", judges["haiku"], None)], [(1.0, unit[LUNA]), (1.0, unit[HAIKU])])
        add(f"{LUNA}+{HAIKU}@rr", "Luna + Haiku, round-robin", "Luna + Haiku, each response by one",
            [("luna", judges["luna"], (18, 0)), ("haiku", judges["haiku"], (18, 9))], [(0.5, unit[LUNA]), (0.5, unit[HAIKU])])
    if have["haiku"] and have["gemini"]:
        add(f"{LUNA}+{HAIKU}+{GEMINI}@rr", "trio, round-robin", "Luna + Haiku + Gemini, each response by one",
            [("luna", judges["luna"], (27, 9)), ("haiku", judges["haiku"], (27, 18)), ("gemini", judges["gemini"], (27, 0))],
            [(1 / 3, unit[LUNA]), (1 / 3, unit[HAIKU]), (1 / 3, unit[GEMINI])])
    model_of = {"luna": LUNA, "haiku": HAIKU, "gemini": GEMINI}

    # projections under the 12 variants
    variants = rob.build_variants(rob.DEFAULT_SLOTS, rob.DEFAULT_LAYERS, list(rob.REGIMES),
                                  K=rob.DEFAULT_SOFT_K, L=rob.DEFAULT_SOFT_SHEAR_L)
    vkeys = [v.key for v in variants]
    meta = {"version": rob.CACHE_VERSION, "fingerprint": rob.data_fingerprint(data_dir), "ca_kind": "combined"}
    slices = rob.load_slices(Path(args.cache_dir), rob.DEFAULT_SLOTS, rob.DEFAULT_LAYERS, meta, True)
    if slices is None:
        raise SystemExit("the slice cache is stale or missing: run scripts/judge_instruction_12way.py once to rebuild it")
    names = {ax.axis_dir: set(rob.lift_scores({n: 0.0 for n in data.rank}, data.kinds)) for ax in axes}
    proj = rob.project_variants(slices, axes, variants, names, data_dir)

    def rho_of(scores: Dict[str, float], ax) -> Dict[str, float]:
        lifted = rob.lift_scores(scores, data.kinds)
        return {vk: rob.cell_rho(lifted, proj[ax.axis_dir][vk], (ax.pos, ax.neg))[0] for vk in vkeys}

    check = validate_extrapolation(data, E)
    calib = check["calibration"]
    points: Dict[str, dict] = {}
    for arm in arms:
        for m in FRACTIONS:
            rhos = {vk: [] for vk in vkeys}
            for ax in axes:
                per_rot = []
                for r in (range(5) if m < 5 else (0,)):
                    parts = {model_of[j]: means(subsample(items[ax.axis_dir], data.rank, m, r, share))
                             for j, items, share in arm["members"]}
                    per_rot.append(rho_of(blend(parts), ax))
                for vk in vkeys:
                    rhos[vk].append(float(np.mean([x[vk] for x in per_rot])))
            cov = NINTH * m / 5
            points[f"{arm['key']}|{m}/5"] = dict(arm=arm["key"], label=arm["label"], family=arm["family"], coverage=cov,
                                                 measured=True, cost=full_run_cost([(f * cov, u) for f, u in arm["cost_parts"]]),
                                                 rho={vk: float(np.mean(rhos[vk])) for vk in vkeys}, per_axis=rhos)
        # extrapolation above the ninth: split-half reliability by instruction parity, Spearman-Brown
        rel = []
        for ax in axes:
            halves = []
            for h in (0, 1):
                parts = {}
                for j, items, share in arm["members"]:
                    # by question (more coverage adds questions), alternating along the judge's own questions,
                    # which lie a share's modulus apart (9 in the ninth, 18 or 27 in a round-robin share)
                    step = share[0] if share else 9
                    sel = {k: v for k, v in subsample(items[ax.axis_dir], data.rank, 5, 0, share).items()
                           if (k[2] // step) % 2 == h}
                    parts[model_of[j]] = means(sel)
                halves.append(blend(parts))
            common = sorted(set(halves[0]) & set(halves[1]))
            r = pearsonr([halves[0][n] for n in common], [halves[1][n] for n in common])[0]
            rel.append(2 * r / (1 + r))
        R = float(np.mean(rel))
        base = points[f"{arm['key']}|5/5"]
        for name, M in EXTRAPOLATE.items():
            RM = M * R / (1 + (M - 1) * R)
            cov = NINTH * M
            points[f"{arm['key']}|{name}"] = dict(
                arm=arm["key"], label=arm["label"], family=arm["family"], coverage=cov, measured=False, reliability=R,
                cost=full_run_cost([(f * cov, u) for f, u in arm["cost_parts"]]),
                # the Spearman-Brown gain, scaled by the measured-to-predicted ratio of the check
                rho={vk: base["rho"][vk] * (1 + calib * (math.sqrt(RM / R) - 1)) for vk in vkeys})

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inputs = data.inputs + [current_files_input(dep_key="cells", paths=sorted(set(data.read)))]
    title = "Response judging with a cached prefix: subsampling against stacking, under 12 projection variants"
    plot(points, variants, out / "judge_response_pareto.png", title, len(axes), check, inputs)
    result = {"config": {"axes_file": args.axes_file, "variants": vkeys, "prices": PRICES, "october_responses": OCTOBER_RESPONSES,
                         "n_axes": N_AXES, "fractions": "m/5 of each judge's share of the ninth, by instruction",
                         "extrapolation": "Spearman-Brown from a split half by question"},
              "unit_costs": {m: {"per_response_axis": a, "per_response_once": o, "tokens_per_call": prof[m]}
                             for m, (a, o) in unit.items()},
              "extrapolation_check": check, "points": points}
    (out / "judge_response_pareto.json").write_text(json.dumps(json_metadata(result, title=title, inputs=inputs), indent=1))
    print(f"wrote {out / 'judge_response_pareto.png'} and .json", file=sys.stderr)
    return 0


def validate_extrapolation(data: Data, E: Path) -> dict:
    """The extrapolation checked where coverage was measured: the score-alone rubric (r1_hn, GPT-5.6 Luna, own
    order) on a third of the six pilot axes and on all of two: from each one's ninth (split 9, part 0), the
    predicted rho at the third and at all, against the measured (slot 6, layer 25, raw, the cached projections)."""
    out = {}
    for axis in ("blunt_vs_tactful", "concise_vs_verbose", "honest_vs_dishonest", "ecocentric_vs_anthropocentric",
                 "predator_vs_prey", "angel_vs_demon"):
        proj = json.load(open(E / axis / "gpt" / "projections.json"))
        proj = {n.split("|")[0]: v["raw"] for n, v in proj.get("result", proj)["6"].items()}
        for label, M in (("@r1_hnn20_s3p0", 3), ("@r1_hnn20_all", 9)):
            p = E / f"{axis}{rob.TIER_EVAL_SUFFIX}" / f"{LUNA}{label}" / "scores_responses.json"
            if not p.exists():
                continue
            items = bare_scores(json.load(open(p))["items"])
            names = sorted({k[0] for k in items})
            rank = {n: r for r, n in enumerate(names)}
            def rho(sel):
                s = means(sel)
                c = [n for n in s if n in proj]
                from scipy.stats import spearmanr
                return spearmanr([s[n] for n in c], [proj[n] for n in c])[0]
            # a ninth of each entity's questions: of the full cell's, a ninth; of the third's (q = c mod 3), a third
            ninth = {k: v for k, v in items.items()
                     if ((k[2] + rank[k[0]]) % 9 == 0 if M == 9 else (k[2] // 3 + rank[k[0]]) % 3 == 0)}
            halves = [means({k: v for k, v in ninth.items() if (k[2] // 9) % 2 == h}) for h in (0, 1)]
            c = sorted(set(halves[0]) & set(halves[1]))
            r = pearsonr([halves[0][n] for n in c], [halves[1][n] for n in c])[0]
            R = 2 * r / (1 + r)
            RM = M * R / (1 + (M - 1) * R)
            r9, rm = rho(ninth), rho(items)
            out[f"{axis} {'third' if M == 3 else 'all'}"] = {"ninth": r9, "predicted": r9 * math.sqrt(RM / R), "measured": rm}
    errs = [v["predicted"] - v["measured"] for v in out.values()]
    gains = [v["measured"] - v["ninth"] for v in out.values()]
    pred_gains = [v["predicted"] - v["ninth"] for v in out.values()]
    calibration = float(np.mean(gains) / np.mean(pred_gains))
    print(f"[extrapolation check] {len(errs)} cases: predicted minus measured {np.mean(errs):+.4f} "
          f"(range {min(errs):+.4f} to {max(errs):+.4f}); measured gain over the ninth {np.mean(gains):+.4f}, "
          f"predicted {np.mean(pred_gains):+.4f}: the extrapolated gains are scaled by {calibration:.2f}", file=sys.stderr)
    return {"cases": out, "mean_error": float(np.mean(errs)), "mean_gain": float(np.mean(gains)),
            "mean_predicted_gain": float(np.mean(pred_gains)), "calibration": calibration}


FAMILY_STYLE = {
    "Luna, one axis a call": ("#3E8E2F", "o"),
    "Luna stacked, plain": ("#9CCB6B", "s"),
    "Luna stacked, labelled": ("#2BA39B", "s"),
    "Luna stacked, axis by axis": ("#B5A32E", "s"),
    "Haiku, one axis a call": ("#D9636B", "o"),
    "Luna + Haiku, both judge every response": ("#8E6BBF", "D"),
    "Luna + Haiku, each response by one": ("#C792EA", "D"),
    "Luna + Haiku + Gemini, each response by one": ("#3B6FD8", "D"),
}
K_MARKER = {"k2": "s", "k4": "^", "k6": "v"}


def plot(points: Dict[str, dict], variants, out_png: Path, title: str, n_axes: int, check: dict, inputs) -> None:
    vkeys = [v.key for v in variants]
    sl = list(dict.fromkeys((v.slot, v.layer) for v in variants))
    regimes = list(dict.fromkeys(v.regime for v in variants))
    fig, axs = plt.subplots(len(regimes), len(sl), figsize=(4.9 * len(sl), 4.1 * len(regimes) + 2.2), squeeze=False)
    costs = [p["cost"] for p in points.values()]
    x_lo, x_hi = min(costs) / 1.5, max(costs) * 1.5
    ticks = [t for t in (3, 10, 30, 100, 300, 1000, 3000) if x_lo <= t <= x_hi]
    for r, regime in enumerate(regimes):
        for c, (s, l) in enumerate(sl):
            ax = axs[r][c]
            vk = next(v.key for v in variants if (v.slot, v.layer, v.regime) == (s, l, regime))
            keys = list(points)
            q = {k: 1 / (1 - points[k]["rho"][vk]) for k in keys}
            idx = pareto_frontier([(points[k]["cost"], q[k]) for k in keys])
            front = sorted((keys[i] for i in idx), key=lambda k: points[k]["cost"])
            for arm in dict.fromkeys(p["arm"] for p in points.values()):
                ks = sorted([k for k in keys if points[k]["arm"] == arm], key=lambda k: points[k]["cost"])
                fam = points[ks[0]]["family"]
                col, mk = FAMILY_STYLE[fam]
                kk = re.search(r"k(\d)$", arm)
                if kk and "stacked" in fam:
                    mk = K_MARKER[f"k{kk.group(1)}"]
                ms = [k for k in ks if points[k]["measured"]]
                ex = [k for k in ks if not points[k]["measured"]]
                ax.plot([points[k]["cost"] for k in ms], [q[k] for k in ms], "-", color=col, lw=0.8, alpha=0.6)
                if ex:
                    ax.plot([points[k]["cost"] for k in ms[-1:] + ex], [q[k] for k in ms[-1:] + ex], ":", color=col, lw=0.8, alpha=0.6)
                ax.scatter([points[k]["cost"] for k in ms], [q[k] for k in ms], s=[18 + 7 * int(k.split("|")[1][0]) for k in ms],
                           marker=mk, color=col, edgecolors="black" if False else col, zorder=3)
                ax.scatter([points[k]["cost"] for k in ex], [q[k] for k in ex], s=40, marker=mk, facecolors="none",
                           edgecolors=col, linewidths=1.2, zorder=3)
            ax.plot([points[k]["cost"] for k in front], [q[k] for k in front], "--", color="#444444", lw=1.0, alpha=0.8, zorder=2)
            for k in front:
                ax.scatter(points[k]["cost"], q[k], s=90, facecolors="none", edgecolors="#222222", linewidths=0.9, zorder=4)
            ax.set_xscale("log")
            ax.set_xlim(x_lo, x_hi)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"\\${t:,}" for t in ticks])
            ax.minorticks_off()
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            ax.set_title(f"{vk}{'  (default)' if vk == rob.DEFAULT_VARIANT else ''}", fontsize=10,
                         fontweight="bold" if vk == rob.DEFAULT_VARIANT else "normal")
            if c == 0:
                ax.set_ylabel("1 / (1 - mean rho)", fontsize=9)
            if r == len(regimes) - 1:
                ax.set_xlabel("estimated total cost of a full run (USD, 200 October axes)", fontsize=9)
    handles = [plt.Line2D([0], [0], marker=mk, color=col, lw=0.8, label=fam, markersize=7) for fam, (col, mk) in FAMILY_STYLE.items()
               if any(p["family"] == fam for p in points.values())]
    handles += [plt.Line2D([0], [0], marker=m, color="#777777", lw=0, label=f"stacked {k[1]} axes a call", markersize=7)
                for k, m in K_MARKER.items()]
    handles += [plt.Line2D([0], [0], marker="o", color="#777777", markerfacecolor="none", lw=0, label="extrapolated (1/3, all)"),
                plt.Line2D([0], [0], marker="o", color="#222222", markerfacecolor="none", markersize=10, lw=0,
                           label="on the frontier (its convex hull)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    sub = (f"{n_axes} axes (scripts/tier_eval_axes_responses12.tsv); GPT-5.6 Luna, Haiku 5.5, Gemini 3.1 Flash-Lite, every call's direction "
           f"a coin, judges 4:3:3; stacking: each batch's axes regrouped at random, the best format at each k\n"
           f"a ninth of the kept responses measured: points from 1/45 (smallest) to 1/9, subsampled by instruction; 1/3 and all extrapolated "
           f"(hollow: Spearman-Brown by question, gains scaled by {check['calibration']:.2f}, the measured-to-predicted ratio on Luna's third and full cells)\n"
           f"cost of a full run, 200 October axes (1,273 entities x 425 kept responses) at the point's coverage, every prefix cached: Luna by Flex, "
           f"Haiku by Message Batches, Gemini by its Batch API with explicit caches")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.965, sub, ha="center", va="top", fontsize=8.5)
    fig.subplots_adjust(top=0.885, bottom=0.12, left=0.05, right=0.985, hspace=0.32, wspace=0.18)
    fig.savefig(out_png, dpi=130, metadata=png_metadata(title=title, inputs=inputs))
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
