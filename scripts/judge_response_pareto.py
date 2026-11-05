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
them, the golden halves of the ninth: judge_tier_cost_eval.golden_part over the absolute question number and the
instruction, the design that tool cuts by since 2026-11-04); the trio round-robin by question ((q + rank) % 27:
Gemini 0, Luna 9, Haiku 18, inside the ninth; Gemini judged only its share, so its arm cannot be cut otherwise from
these cells).  Judges blended Luna 4 : Haiku 3 : Gemini 3 (judge_score_combine.combine_tier_judges).  Further
subsampling: of each judge's share, the fraction m/3 cut the golden way (a response's golden point, rescaled within
a golden share, in the first m thirds of the circle after a rotation r/3; no fifths, Roger, 2026-11-04), averaged
over the three rotations r.  Coverage above
the ninth (1/3, all) is extrapolated, hollow points: each arm's per-entity reliability at the ninth from a
split half by question (more coverage adds questions; halves by instruction share their questions and
predicted a fifth of the measured gain), Spearman-Brown to the larger coverage, rho scaled by the square root of the
reliabilities' ratio (checked against the score-alone rubric's third and full cells, printed).

Cost on the x axis: a full run, 200 axes of the October extraction (1,273 entities, 425 kept responses each)
at the point's coverage, every prefix cached and read for every axis after the first: GPT-5.6 Luna by Flex
($0.10/$0.60 per 1M, reads 0.1x, a write 1.25x once per prefix); Haiku 5.5 by Message Batches ($0.05/$0.25, the
same factors, assuming every prefix is read inside a batch, which is untested); Gemini 3.1 Flash-Lite by its Batch API
naming an explicit cache per prefix ($0.125/$0.75, cached reads $0.0125; each cache created once at the standard
input rate, $0.25, and stored an hour at $1 per 1M an hour; the implicit cache reads nothing in batch).  Each
judge's tokens a call are its own, measured on these runs; a stacked arm's cost is its measured cost.

With --stack_group g61 (Roger, 2026-11-05: "Can you do the Pareto plot at 61, mostly?"): the 61 axes of
scripts/tier_eval_axes.tsv, Luna only, the only judge and format judged on all 61: one axis a call (the twelve's
balanced cells as above, a coin run @r1_pnn20_s9p0g61_a<i>c on the other 49) against plain stacking at k = 2, 4
and 6 (@rk_pnn20_s9p0g61_k<k>, each batch's 61 axes regrouped at random), every axis stacked or the worst 10%, 15%, 20%
or 25% of the axes judged one axis a call instead.  The worst are chosen by cross-fitting: the ninth's questions fall
in 9 folds by the golden point of the absolute question number; for each fold, the axes whose stacking penalty
(stacked minus one axis a call, rho averaged over the 12 variants) is worst on the other 8 folds are judged one
axis a call on that fold, and the folds are pooled.  Such an arm costs its share of axes at each price.

Output: <output_dir>/judge_response_pareto_{luna,jev,models}.png and judge_response_pareto.json; with
--stack_group g61, judge_response_pareto_61_luna.png, judge_response_pareto_answers.png (20 answers a call against 40,
the 40-answer cells @r1_pnn40_s9p0g12_a<i>c on the twelve and @r1_pnn40_s9p0g61_a<i>c on the other 49) and
judge_response_pareto_61.json.  No API calls.

Usage: uv run python scripts/judge_response_pareto.py --output_dir roger/axis_judge_experiments/response_pareto
       uv run python scripts/judge_response_pareto.py --stack_group g61 --output_dir roger/axis_judge_experiments/response_pareto
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
FRACTION_K = 3                      # each judge's share of the ninth cut in thirds (Roger: no fifths)
FRACTIONS = (1, 2, 3)               # m/3 of it
KEY_RE = re.compile(r"^(.*)\|p(\d+)q(\d+)$")
#: Production prices, USD per 1M tokens: (input, output, cached read, once per prefix per cached token).
PRICES = {
    LUNA: dict(inp=0.10, out=0.60, read=0.010, once=0.10 * 1.25 - 0.010),       # Flex; a write 1.25x, less the read
    HAIKU: dict(inp=0.05, out=0.25, read=0.005, once=0.05 * 1.25 - 0.005),      # Message Batches
    GEMINI: dict(inp=0.125, out=0.75, read=0.0125, once=0.25 + 1.0 * 1.0),     # Batch API, explicit cache: create + 1 h
}
#: Every stacked arm judged on the twelve axes: three output formats at k = 2, 4 and 6 (plain k=6 added on 2026-11-04,
#: Roger: "One stacking direction seems to be missing?").  All are in the Luna figure.
STACKED = [(f, k) for f in ("rk_pn", "rk_pl", "rk_pa") for k in (2, 4, 6)]
JEV = "jev-1.14.0"
#: Jev's relative yes/no design (judge_responses_jev.py --design relative), the arms judged on the twelve axes:
#: (cell label, figure label).  Its cost is measured per response and axis, with no cache.
JEV_ARMS = [("@jrel_a20m12_s9p0", "Jev, 20 answers x 12 axes a request"), ("@jrel_a40m6_s9p0", "Jev, 40 answers x 6 axes a request"),
            ("@jrel_a40m1_s9p0", "Jev, 40 answers x 1 axis a request")]
#: The figures (Roger, 2026-11-04: "our Pareto plot is answering too many questions at once"; the 40-answer arm got its
#: own on 2026-11-05, "Move that to a separate plot"): one arm can be
#: in several; a reference arm is drawn grey and stays off that figure's frontier.
FIGURES = {
    "jev": ("Jev only: its relative yes/no design, answers and axes a request (Luna one axis a call for reference)",
            "judge_response_pareto_jev.png",
            "Jev 1.14 (TypeSafe AI): a batch of answers to one question and some axes a request, \"is the persona behind answer k "
            "nearer the X end of axis A than most of the other answers'?\", a coin per request and axis; GPT-5.6 Luna one axis a "
            "call in grey"),
    "models": ("Judges and mixes: each judge alone, every response judged by each, or each response by one",
               "judge_response_pareto_models.png",
               "GPT-5.6 Luna, Haiku 5.5, Gemini 3.1 Flash-Lite one axis a call, Jev 1.14 at 20 answers x 12 axes a request; every "
               "call's direction a coin; mixes weighted Luna 4 : Haiku 3 : Gemini 3 : Jev 3; shares golden, the trio's by question "
               "(Gemini judged only its 1/27)"),
}
FMT_NAME = {"rk_pn": "plain", "rk_pl": "labelled", "rk_pa": "axis by axis"}
AXES_12, AXES_61 = "scripts/tier_eval_axes_responses12.tsv", "scripts/tier_eval_axes.tsv"
#: The 61-axis mode: plain stacking's k, the shares of the axes left out of it, and the cross-fitting folds.
STACK61_K = (2, 4, 6)
LEAVE_OUT = (0.10, 0.15, 0.20, 0.25)  # Roger, 2026-11-04: "try 10%, 25%, and a plausible seeming threshold"; 15%, 20% 2026-11-05
CROSSFIT_FOLDS = 9
INV_PHI = 0.6180339887498949
#: The 61-axis mode's judges (Roger, 2026-11-05: "redo this for Haiku and Gemini ... Each step depends on the decision
#: for the previous one"; figures under the same names with the judge added).  Per judge: its share of the responses
#: (``base``; Gemini judges the round-robin 1/27 of the ninth, its questions 27 apart) and the cell labels, ``{i}`` the
#: axis's index in its axes file; ``k1_12`` / ``n40_12`` the twelve's own cells (Luna: "balanced", its own-order and
#: swapped runs), else the 61-axis label serves the twelve too.  A judge's figures are drawn once their cells exist.
JUDGES_61 = {
    "luna": dict(model=LUNA, name="Luna", share="s9p0", base=NINTH, step=9, k1_12="balanced",
                 k1_61="@r1_pnn20_s9p0g61_a{i}c", n40_12="@r1_pnn40_s9p0g12_a{i}c", n40_61="@r1_pnn40_s9p0g61_a{i}c"),
    "haiku": dict(model=HAIKU, name="Haiku 5.5", share="s9p0", base=NINTH, step=9, k1_12=None,
                  k1_61="@r1_pnn20_s9p0g61_st1", n40_12=None, n40_61="@r1_pnn40_s9p0g61_st1"),
    "gemini": dict(model=GEMINI, name="Gemini 3.1 Flash-Lite", share="s27p0", base=1 / 27, step=27,
                   k1_12="@r1_pnn20_s27p0g12_a{i}c", k1_61="@r1_pnn20_s27p0g61_a{i}c", n40_12=None,
                   n40_61="@r1_pnn40_s27p0g61_a{i}c"),
}
#: The stacking formats on the 61 axes (2026-11-05; Roger: "the full one", replacing the twelve-axis Luna figure).
FIGURE_FORMATS = ("61 axes, {name} only: one axis a call against every stacking format and k",
                  "judge_response_pareto_luna.png",
                  "{name} alone, every call's direction a coin, 20 answers a call; every axis stacked, each batch's 61 axes "
                  "regrouped at random; formats plain \"1, SCORES: 18, 42\", labelled \"1. SCORES: A=18, B=42\", axis by axis "
                  "\"A: 1=18, 2=32\"")
#: 20 answers a call against 40, on the 61 axes since 2026-11-05 (Roger: "Do it [at Flex], update it and show it to me
#: once done, no need to change the plot name"); drawn by --stack_group g61.
FIGURE_ANSWERS = ("61 axes, {name} only: 20 answers a call against 40",
                  "judge_response_pareto_answers.png",
                  "{name} alone, one axis a call, every call's direction a coin; each call judges one question's answers "
                  "in batches of 20 (the default) or 40")
FIGURE_61 = ("61 axes, {name} only: one axis a call against plain stacking, every axis or all but the worst",
             "judge_response_pareto_61_luna.png",
             "{name} alone, every call's direction a coin, 20 answers a call; plain stacking \"1, SCORES: 18, 42\", each "
             "batch's 61 axes regrouped at random; the worst 10%, 15%, 20% or 25% of the axes judged one axis a call instead, chosen on "
             "the other 8 of 9 question folds (cross-fitted)")


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
        # the absolute question number of each question position (the golden design's Q)
        self.qabs = {ev.response_question(key): ev.absolute_question(text[0]) for _, key, text in self.group.entities}
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


def subsample(items: Dict[Tuple[str, int, int], float], rank: Dict[str, int], qabs: Dict[int, int], m: int, r: int,
              share: Optional[Tuple[str, int, int]] = None) -> Dict[Tuple[str, int, int], float]:
    """The items of a share and of the fraction m/FRACTION_K cut the golden way, rotation r.  A share is ("g", mod,
    part), golden (judge_tier_cost_eval.golden_part), or ("q", mod, part), by question, (q + rank) % mod == part;
    within a golden share the fraction reads the point's place inside the share's interval."""
    one = 1 << 64
    out = {}
    for (n, p, q), v in items.items():
        if n not in rank:
            continue
        point = ev.golden_point(qabs[q], p)
        if share and share[0] == "q" and (q + rank[n]) % share[1] != share[2]:
            continue
        if share and share[0] == "g":
            if (point * share[1]) >> 64 != share[2]:
                continue
            point = (point * share[1]) % one
        if m < FRACTION_K and (((point + r * one // FRACTION_K) % one) * FRACTION_K) >> 64 >= m:
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
    ap.add_argument("--stack_group", choices=("g12", "g61"), default="g12",
                    help="g12: the twelve axes, every judge and format (three figures); g61: the 61 axes, Luna only (stacking, and 20 answers a call against 40)")
    ap.add_argument("--axes_file", default=None, help=f"default {AXES_12}, or {AXES_61} with --stack_group g61")
    ap.add_argument("--judge", choices=sorted(JUDGES_61), default="luna",
                    help="with --stack_group g61: the judge whose cells are drawn (figures named with it unless Luna)")
    ap.add_argument("--instructions_dir", default="/tmp/claude-501/corpus_2026-05-24/data")
    ap.add_argument("--cache_dir", default=str(rob.DEFAULT_EXPERIMENTS_DIR / "_tier_eval_scratch" / "robustness_cache"))
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args(argv)
    if args.axes_file is None:
        args.axes_file = AXES_61 if args.stack_group == "g61" else AXES_12
    if args.stack_group == "g61":
        return main61(args)
    E, data_dir = Path(args.experiments_dir), Path(args.data_dir).resolve()
    data = Data(E, Path(args.axes_file), data_dir, Path(args.instructions_dir))
    axes = data.axes

    # every judge's items on every axis
    judges: Dict[str, Dict[str, Dict]] = {"luna": {}, "haiku": {}, "gemini": {}}
    for i, ax in enumerate(axes):
        judges["luna"][ax.axis_dir] = data.luna_balanced(i, ax.axis_dir)
        judges["haiku"][ax.axis_dir] = data.cell(ax.axis_dir, HAIKU, f"@r1_pnn20_s9p0g12_a{i}c")
        judges["gemini"][ax.axis_dir] = data.cell(ax.axis_dir, GEMINI, f"@r1_pnn20_s27p0g12_a{i}c")
    judges["luna40"] = {ax.axis_dir: data.cell(ax.axis_dir, LUNA, f"@r1_pnn40_s9p0g12_a{i}c") for i, ax in enumerate(axes)}
    for lab, _ in JEV_ARMS:
        judges[lab] = {ax.axis_dir: data.cell(ax.axis_dir, JEV, lab) for ax in axes}
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
    if have["luna40"]:
        prof["luna40"] = per_call_profile(data, LUNA, lambda i: f"@r1_pnn40_s9p0g12_a{i}c")
        unit["luna40"] = judge_costs(prof["luna40"], LUNA)
    for lab, _ in JEV_ARMS:         # no cache: the run's cost over its responses and axes
        if have[lab]:
            u = json.load(open(E / "_tier_eval_jev" / lab.lstrip("@") / "usage.json"))
            n = json.load(open(E / f"{axes[0].axis_dir}{rob.TIER_EVAL_SUFFIX}" / f"{JEV}{lab}" / "scores_responses.json"))["_provenance"]["n_items"]
            unit[lab] = (u["total_cost_usd"] / (n * len(axes)), 0.0)
            data.read.append(E / "_tier_eval_jev" / lab.lstrip("@") / "usage.json")
    for m, (a, o) in unit.items():
        print(f"[cost] {m}: {json.dumps({k: round(v, 1) for k, v in prof.get(m, {}).items()})}; ${a * 1e6:.2f} per 1M "
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

    # the arms: (key, label, family, [(judge, items by axis, share)], cost parts per unit coverage, figures)
    arms = []
    def add(key, label, family, members, cost_parts, figures):
        arms.append(dict(key=key, label=label, family=family, members=members, cost_parts=cost_parts, figures=figures))
    jb = JEV_ARMS[0][0]                 # Jev in the mixes: its cheapest arm, 20 answers x 12 axes
    add(f"{LUNA}@k1", "Luna", "Luna, one axis a call", [("luna", judges["luna"], None)], [(1.0, unit[LUNA])],
        ["jev", "models"])
    if have["luna40"]:
        add(f"{LUNA}@n40", "Luna, 40 answers", "Luna, one axis a call, 40 answers a call", [("luna40", judges["luna40"], None)],
            [(1.0, unit["luna40"])], [])        # its figure is drawn on the 61 axes (--stack_group g61)
    for (fmt, k), cells in stacked.items():
        if (fmt, k) in stack_unit:
            add(f"{LUNA}@{fmt}k{k}", f"Luna k={k} {FMT_NAME[fmt]}", f"Luna stacked, {FMT_NAME[fmt]}",
                [("luna", cells, None)], [(1.0, stack_unit[(fmt, k)])], [])     # drawn on the 61 axes since 2026-11-05
    for lab, name in JEV_ARMS:
        if have[lab]:
            add(f"{JEV}{lab}", name, name, [(lab, judges[lab], None)], [(1.0, unit[lab])], ["jev"] + (["models"] if lab == jb else []))
    if have["haiku"]:
        add(f"{HAIKU}@k1", "Haiku", "Haiku, one axis a call", [("haiku", judges["haiku"], None)], [(1.0, unit[HAIKU])], ["models"])
        add(f"{LUNA}+{HAIKU}@2x", "Luna + Haiku, both", "Luna + Haiku, both judge every response",
            [("luna", judges["luna"], None), ("haiku", judges["haiku"], None)], [(1.0, unit[LUNA]), (1.0, unit[HAIKU])], ["models"])
        add(f"{LUNA}+{HAIKU}@rr", "Luna + Haiku, round-robin", "Luna + Haiku, each response by one",
            [("luna", judges["luna"], ("g", 2, 0)), ("haiku", judges["haiku"], ("g", 2, 1))], [(0.5, unit[LUNA]), (0.5, unit[HAIKU])],
            ["models"])
    if have["haiku"] and have["gemini"]:
        add(f"{LUNA}+{HAIKU}+{GEMINI}@rr", "trio, round-robin", "Luna + Haiku + Gemini, each response by one",
            [("luna", judges["luna"], ("q", 27, 9)), ("haiku", judges["haiku"], ("q", 27, 18)), ("gemini", judges["gemini"], ("q", 27, 0))],
            [(1 / 3, unit[LUNA]), (1 / 3, unit[HAIKU]), (1 / 3, unit[GEMINI])], ["models"])
    if have[jb]:
        add(f"{LUNA}+{JEV}@2x", "Luna + Jev, both", "Luna + Jev, both judge every response",
            [("luna", judges["luna"], None), (jb, judges[jb], None)], [(1.0, unit[LUNA]), (1.0, unit[jb])], ["models"])
        add(f"{LUNA}+{JEV}@rr", "Luna + Jev, round-robin", "Luna + Jev, each response by one",
            [("luna", judges["luna"], ("g", 2, 0)), (jb, judges[jb], ("g", 2, 1))], [(0.5, unit[LUNA]), (0.5, unit[jb])], ["models"])
        if have["haiku"]:
            add(f"{LUNA}+{HAIKU}+{JEV}@3x", "Luna + Haiku + Jev, all", "Luna + Haiku + Jev, all judge every response",
                [("luna", judges["luna"], None), ("haiku", judges["haiku"], None), (jb, judges[jb], None)],
                [(1.0, unit[LUNA]), (1.0, unit[HAIKU]), (1.0, unit[jb])], ["models"])
            add(f"{LUNA}+{HAIKU}+{JEV}@rr", "Luna + Haiku + Jev, round-robin", "Luna + Haiku + Jev, each response by one",
                [("luna", judges["luna"], ("g", 3, 0)), ("haiku", judges["haiku"], ("g", 3, 1)), (jb, judges[jb], ("g", 3, 2))],
                [(1 / 3, unit[LUNA]), (1 / 3, unit[HAIKU]), (1 / 3, unit[jb])], ["models"])
    model_of = {"luna": LUNA, "luna40": LUNA, "haiku": HAIKU, "gemini": GEMINI, **{lab: JEV for lab, _ in JEV_ARMS}}

    variants, vkeys, rho_of = projector(data, Path(args.cache_dir), data_dir)
    check = validate_extrapolation(data, E)
    calib = check["calibration"]
    points = arm_points(arms, data, model_of, rho_of, vkeys, calib)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inputs = data.inputs + [current_files_input(dep_key="cells", paths=sorted(set(data.read)))]
    title = "Response judging with a cached prefix: subsampling against stacking, under 12 projection variants"
    for fig_key, (fig_title, fname, lead) in FIGURES.items():
        sub_points = {k: v for k, v in points.items() if fig_key in v["figures"]}
        ref = {f"{LUNA}@k1"} if fig_key == "jev" else set()
        plot(sub_points, variants, out / fname, f"Response judging, {fig_title}", len(axes), check, inputs, ref_arms=ref, lead=lead,
             axes_file=args.axes_file)
        print(f"wrote {out / fname}", file=sys.stderr)
    result = {"config": {"axes_file": args.axes_file, "variants": vkeys, "prices": PRICES, "october_responses": OCTOBER_RESPONSES,
                         "n_axes": N_AXES, "fractions": "m/3 of each judge's share of the ninth, golden (judge_tier_cost_eval.golden_point), 3 rotations",
                         "extrapolation": "Spearman-Brown from a split half by question"},
              "unit_costs": {m: {"per_response_axis": a, "per_response_once": o, "tokens_per_call": prof.get(m)}
                             for m, (a, o) in unit.items()},
              "extrapolation_check": check, "points": points}
    (out / "judge_response_pareto.json").write_text(json.dumps(json_metadata(result, title=title, inputs=inputs), indent=1))
    print(f"wrote {out / 'judge_response_pareto.json'}", file=sys.stderr)
    return 0


def projector(data: Data, cache_dir: Path, data_dir: Path):
    """The 12 projection variants, and rho_of(entity scores, axis) -> {variant: Spearman rho} on data's axes."""
    variants = rob.build_variants(rob.DEFAULT_SLOTS, rob.DEFAULT_LAYERS, list(rob.REGIMES),
                                  K=rob.DEFAULT_SOFT_K, L=rob.DEFAULT_SOFT_SHEAR_L)
    vkeys = [v.key for v in variants]
    meta = {"version": rob.CACHE_VERSION, "fingerprint": rob.data_fingerprint(data_dir), "ca_kind": "combined"}
    slices = rob.load_slices(cache_dir, rob.DEFAULT_SLOTS, rob.DEFAULT_LAYERS, meta, True)
    if slices is None:
        raise SystemExit("the slice cache is stale or missing: run scripts/judge_instruction_12way.py once to rebuild it")
    names = {ax.axis_dir: set(rob.lift_scores({n: 0.0 for n in data.rank}, data.kinds)) for ax in data.axes}
    proj = rob.project_variants(slices, data.axes, variants, names, data_dir)

    def rho_of(scores: Dict[str, float], ax) -> Dict[str, float]:
        lifted = rob.lift_scores(scores, data.kinds)
        return {vk: rob.cell_rho(lifted, proj[ax.axis_dir][vk], (ax.pos, ax.neg))[0] for vk in vkeys}
    return variants, vkeys, rho_of


def arm_points(arms: List[dict], data: Data, model_of: Dict[str, str], rho_of, vkeys: List[str],
               calib: float, base: float = NINTH, step: int = 9) -> Dict[str, dict]:
    """Each arm's points: the fractions m/3 of the measured share (the ninth unless ``base`` says otherwise; Gemini
    judged a 1/27, cut by question, its questions ``step`` = 27 apart) measured, each the mean of three rotations,
    and the coverages above it up to all extrapolated by Spearman-Brown, the gains scaled by calib.  The point keys
    keep the word "ninth" whatever the share (the plots read ``coverage`` and ``m``)."""
    axes = data.axes
    extrapolate = {f"1/{round(1 / (base * M))}" if base * M < 1 else "all": M
                   for M in (3, 9, 27) if base * M <= 1 + 1e-9}
    points: Dict[str, dict] = {}
    for arm in arms:
        for m in FRACTIONS:
            rhos = {vk: [] for vk in vkeys}
            for ax in axes:
                per_rot = []
                for r in (range(FRACTION_K) if m < FRACTION_K else (0,)):
                    parts = {model_of[j]: means(subsample(items[ax.axis_dir], data.rank, data.qabs, m, r, share))
                             for j, items, share in arm["members"]}
                    per_rot.append(rho_of(blend(parts), ax))
                for vk in vkeys:
                    rhos[vk].append(float(np.mean([x[vk] for x in per_rot])))
            cov = base * m / FRACTION_K
            points[f"{arm['key']}|ninth x {m}/{FRACTION_K}"] = dict(arm=arm["key"], label=arm["label"], family=arm["family"], coverage=cov, m=m,
                                                 figures=arm["figures"],
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
                    # which lie a share's modulus apart (9 in the ninth and in a golden share of it, 27 in the
                    # trio's share by question)
                    st = share[1] if share and share[0] == "q" else step
                    sel = {k: v for k, v in subsample(items[ax.axis_dir], data.rank, data.qabs, FRACTION_K, 0, share).items()
                           if (k[2] // st) % 2 == h}
                    parts[model_of[j]] = means(sel)
                halves.append(blend(parts))
            common = sorted(set(halves[0]) & set(halves[1]))
            r = pearsonr([halves[0][n] for n in common], [halves[1][n] for n in common])[0]
            rel.append(2 * r / (1 + r))
        R = float(np.mean(rel))
        base_pt = points[f"{arm['key']}|ninth x {FRACTION_K}/{FRACTION_K}"]
        for name, M in extrapolate.items():
            RM = M * R / (1 + (M - 1) * R)
            cov = base * M
            points[f"{arm['key']}|{name}"] = dict(
                arm=arm["key"], label=arm["label"], family=arm["family"], coverage=cov, measured=False, reliability=R,
                figures=arm["figures"],
                cost=full_run_cost([(f * cov, u) for f, u in arm["cost_parts"]]),
                # the Spearman-Brown gain, scaled by the measured-to-predicted ratio of the check
                rho={vk: base_pt["rho"][vk] * (1 + calib * (math.sqrt(RM / R) - 1)) for vk in vkeys})
    return points


def main61(args) -> int:
    """The 61 axes, one judge (--judge, JUDGES_61): 20 answers a call against 40; one axis a call against plain
    stacking at k = 2, 4, 6, every axis stacked or all but the worst 10%, 15%, 20% or 25%, those chosen by
    cross-fitting over question folds (module docstring); and the three stacking formats.  Each figure is drawn once
    its cells exist; Luna's file names are kept, other judges' carry the judge (judge_response_pareto_answers_haiku.png)."""
    from results_analysis.judge_tier_cost_plot_pooled import load_measured_cost
    J = JUDGES_61[args.judge]
    model, name, share, base = J["model"], J["name"], J["share"], J["base"]
    E, data_dir, inst = Path(args.experiments_dir), Path(args.data_dir).resolve(), Path(args.instructions_dir)
    data = Data(E, Path(args.axes_file), data_dir, inst)
    d12 = Data(E, Path(AXES_12), data_dir, inst)          # the twelve's own cells, and Luna's cost a call
    axes, A = data.axes, len(data.axes)
    i12 = {a.axis_dir: i for i, a in enumerate(d12.axes)}

    def cells_of(label_12, label_61):
        """{axis: items or None}: the twelve's own cells (label_12) where given, else the 61-axis label."""
        out = {}
        for i, a in enumerate(axes):
            if label_12 == "balanced" and a.axis_dir in i12:
                out[a.axis_dir] = d12.luna_balanced(i12[a.axis_dir], a.axis_dir)
            elif label_12 and a.axis_dir in i12:
                out[a.axis_dir] = d12.cell(a.axis_dir, model, label_12.format(i=i12[a.axis_dir]))
            else:
                out[a.axis_dir] = data.cell(a.axis_dir, model, label_61.format(i=i))
        return out

    complete = lambda cells: all(v for v in cells.values())
    k1 = cells_of(J["k1_12"], J["k1_61"])
    if not complete(k1):
        raise SystemExit(f"{name}: one axis a call is missing on {sum(not v for v in k1.values())} of {A} axes")
    k1_40 = cells_of(J["n40_12"], J["n40_61"])
    have40 = complete(k1_40)
    fmt_label = lambda fmt, k: f"@{fmt}n20_{share}g61_k{k}"
    stacked_by = {(fmt, k): {a.axis_dir: data.cell(a.axis_dir, model, fmt_label(fmt, k)) for a in axes}
                  for fmt in FMT_NAME for k in STACK61_K}
    have = {key: complete(c) for key, c in stacked_by.items()}
    plain_ks = [k for k in STACK61_K if have[("rk_pn", k)]]
    print(f"[{args.judge}] one axis a call: complete; 40 answers: {'complete' if have40 else 'missing'}; stacked: "
          + ", ".join(f"{fmt} k={k}" for (fmt, k), ok in have.items() if ok), file=sys.stderr)

    def measured_unit(label_of_axis):
        """(USD per response per axis, 0): a cell's recorded tokens at the model's batch rate, over its items."""
        per = []
        for i, a in enumerate(axes):
            cd = E / f"{a.axis_dir}{rob.TIER_EVAL_SUFFIX}" / f"{model}{label_of_axis(i, a)}"
            if not (cd / "usage.json").exists():
                continue
            per.append(load_measured_cost(cd / "usage.json", "batch") / json.load(open(cd / "scores_responses.json"))["_provenance"]["n_items"])
            data.read.append(cd / "usage.json")
        return (float(np.mean(per)), 0.0)

    if args.judge == "luna":     # as since 2026-11-04: Luna's production cost model a call, the stacked runs measured
        unit_k1 = judge_costs(per_call_profile(d12, LUNA, lambda i: f"@r1_pnn20_s9p0g12_a{i}"), LUNA)
        unit_40 = judge_costs(per_call_profile(d12, LUNA, lambda i: f"@r1_pnn40_s9p0g12_a{i}c"), LUNA) if have40 else None
        stack_unit = {key: (measured_unit(lambda i, a, key=key: fmt_label(*key))[0], unit_k1[1])
                      for key, ok in have.items() if ok}
    else:                        # every arm measured the same way: its recorded tokens at the batch rate
        lab = lambda label_12, label_61: (lambda i, a: label_12.format(i=i12[a.axis_dir])
                                          if label_12 and a.axis_dir in i12 else label_61.format(i=i))
        unit_k1 = measured_unit(lab(J["k1_12"], J["k1_61"]))
        unit_40 = measured_unit(lab(J["n40_12"], J["n40_61"])) if have40 else None
        stack_unit = {key: measured_unit(lambda i, a, key=key: fmt_label(*key)) for key, ok in have.items() if ok}
    for nm, u in [("one axis a call", unit_k1), ("40 answers a call", unit_40)] + \
                 [(f"{FMT_NAME[f]} k={k}", u) for (f, k), u in stack_unit.items()]:
        if u:
            print(f"[cost] {nm}: ${u[0] * 1e6:.2f} per 1M response-axes; full run, every response: "
                  f"${full_run_cost([(1.0, u)]):,.0f}", file=sys.stderr)

    variants, vkeys, rho_of = projector(data, Path(args.cache_dir), data_dir)
    fold = lambda key: int(((data.qabs[key[2]] * INV_PHI) % 1.0) * CROSSFIT_FOLDS)
    part = lambda items, keep: {key: v for key, v in items.items() if fold(key) in keep}
    rho_mean = lambda items, ax: float(np.mean(list(rho_of(means(items), ax).values())))
    folds = set(range(CROSSFIT_FOLDS))

    arms, left_out = [], {}
    arms.append(dict(key=f"{model}@k1", label=name, family=f"{name}, one axis a call", members=[("j", k1, None)],
                     cost_parts=[(1.0, unit_k1)], figures=["luna61", "answers", "formats"]))
    if have40:
        arms.append(dict(key=f"{model}@n40", label=f"{name}, 40 answers", family=f"{name}, one axis a call, 40 answers a call",
                         members=[("j", k1_40, None)], cost_parts=[(1.0, unit_40)], figures=["answers"]))
    if plain_ks:
        # each fold's selection: the stacking penalty of every axis on the other folds
        basef = {j: np.array([rho_mean(part(k1[a.axis_dir], folds - {j}), a) for a in axes]) for j in folds}
        penalty = {(k, j): np.array([rho_mean(part(stacked_by[("rk_pn", k)][a.axis_dir], folds - {j}), a) for a in axes]) - basef[j]
                   for k in plain_ks for j in folds}
    for (fmt, k), ok in have.items():
        if not ok:
            continue
        figs = ["formats"] + (["luna61"] if fmt == "rk_pn" else [])
        arms.append(dict(key=f"{model}@{fmt}k{k}", label=f"{name} k={k} {FMT_NAME[fmt]}", family=f"{name} stacked, {FMT_NAME[fmt]}",
                         members=[("j", stacked_by[(fmt, k)], None)], cost_parts=[(1.0, stack_unit[(fmt, k)])], figures=figs))
        if fmt != "rk_pn":
            continue
        for sh in LEAVE_OUT:
            n, pct = round(sh * A), round(100 * sh)
            out_of = {j: set(np.argsort(penalty[(k, j)])[:n].tolist()) for j in folds}
            merged = {}
            for i, a in enumerate(axes):
                items: Dict[Tuple[str, int, int], float] = {}
                for j in folds:
                    items.update(part(k1[a.axis_dir] if i in out_of[j] else stacked_by[(fmt, k)][a.axis_dir], {j}))
                merged[a.axis_dir] = items
            key = f"{model}@rk_pnx{pct}k{k}"
            left_out[key] = {a.axis_dir: c for i, a in enumerate(axes) if (c := sum(i in out_of[j] for j in folds))}
            arms.append(dict(key=key, label=f"{name} k={k} plain, worst {pct}% one axis a call",
                             family=f"{name} stacked, plain, worst {pct}% one axis a call", members=[("j", merged, None)],
                             cost_parts=[(n / A, unit_k1), (1 - n / A, stack_unit[(fmt, k)])], figures=["luna61"]))

    check = validate_extrapolation(data, E)
    points = arm_points(arms, data, {"j": model}, rho_of, vkeys, check["calibration"], base=base, step=J["step"])
    ref = points[f"{model}@k1|ninth x {FRACTION_K}/{FRACTION_K}"]["rho"]
    for arm in arms:
        p = points[f"{arm['key']}|ninth x {FRACTION_K}/{FRACTION_K}"]
        d = np.array([p["rho"][vk] - ref[vk] for vk in vkeys])
        print(f"[measured share] {arm['label']:50s} cost {p['cost']:6.0f}  rho {np.mean(list(p['rho'].values())):.4f}  "
              f"minus one axis a call, same coverage: slot 0 {d[:6].mean():+.4f}, slot 6 {d[6:].mean():+.4f}", file=sys.stderr)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.read += d12.read
    inputs = data.inputs + d12.inputs + [current_files_input(dep_key="cells", paths=sorted(set(data.read)))]
    suffix = "" if args.judge == "luna" else f"_{args.judge}"
    drawn = []
    for fig_key, (fig_title, fname, lead), needed in (("luna61", FIGURE_61, bool(plain_ks)), ("answers", FIGURE_ANSWERS, have40),
                                                      ("formats", FIGURE_FORMATS, sum(have.values()) > len(plain_ks))):
        if not needed:
            continue
        fname = fname.replace("_luna.png", f"_{args.judge}.png") if args.judge != "luna" and fname.endswith("_luna.png") \
            else fname.replace(".png", f"{suffix}.png") if args.judge != "luna" else fname
        sub_points = {key: v for key, v in points.items() if fig_key in v["figures"]}
        plot(sub_points, variants, out / fname, f"Response judging, {fig_title.format(name=name)}", A, check, inputs,
             lead=lead.format(name=name), axes_file=args.axes_file, base=base)
        drawn.append(fname)
        print(f"wrote {out / fname}", file=sys.stderr)
    title = f"Response judging on 61 axes, {name} only: answers a call and stacking, under 12 projection variants"
    result = {"config": {"judge": args.judge, "model": model, "measured_share": base, "axes_file": args.axes_file,
                         "variants": vkeys, "prices": PRICES, "october_responses": OCTOBER_RESPONSES,
                         "n_axes": N_AXES, "stack_k": list(STACK61_K), "leave_out": list(LEAVE_OUT), "figures": drawn,
                         "crossfit": f"{CROSSFIT_FOLDS} folds by frac(Q / phi) of the absolute question; each fold's left-out axes "
                                     "the worst by stacking penalty (rho mean of 12 variants) on the other folds",
                         "fractions": "m/3 of the measured share, golden (judge_tier_cost_eval.golden_point), 3 rotations",
                         "extrapolation": "Spearman-Brown from a split half by question",
                         "unit_costs_basis": "Luna: one axis a call by its production cost model, stacked runs measured; "
                                             "other judges: every arm's recorded tokens at the batch rate"},
              "unit_costs": {"one axis a call": unit_k1, "40 answers a call": unit_40,
                             **{f"{FMT_NAME[f]} k={k}": u for (f, k), u in stack_unit.items()}},
              "left_out_folds": left_out, "extrapolation_check": check, "points": points}
    jname = "judge_response_pareto_61.json" if args.judge == "luna" else f"judge_response_pareto_61_{args.judge}.json"
    (out / jname).write_text(json.dumps(json_metadata(result, title=title, inputs=inputs), indent=1))
    print(f"wrote {out / jname}", file=sys.stderr)
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
    "Luna, one axis a call": ("#222222", "o"),
    "Luna, one axis a call, 40 answers a call": ("#8C8C8C", "o"),
    "Luna stacked, plain": ("#1B9E77", "s"),
    "Luna stacked, labelled": ("#D95F02", "s"),
    "Luna stacked, axis by axis": ("#7570B3", "s"),
    "Luna stacked, plain, worst 10% one axis a call": ("#1F78B4", "s"),
    "Luna stacked, plain, worst 15% one axis a call": ("#66A61E", "s"),
    "Luna stacked, plain, worst 20% one axis a call": ("#E7298A", "s"),
    "Luna stacked, plain, worst 25% one axis a call": ("#A6761D", "s"),
    "Jev, 20 answers x 12 axes a request": ("#E7298A", "o"),
    "Jev, 40 answers x 6 axes a request": ("#66A61E", "o"),
    "Jev, 40 answers x 1 axis a request": ("#E6AB02", "o"),
    "Haiku, one axis a call": ("#E08214", "o"),
    "Luna + Haiku, both judge every response": ("#8E6BBF", "D"),
    "Luna + Haiku, each response by one": ("#8E6BBF", "P"),
    "Luna + Haiku + Gemini, each response by one": ("#17BECF", "P"),
    "Luna + Jev, both judge every response": ("#A6761D", "D"),
    "Luna + Jev, each response by one": ("#A6761D", "P"),
    "Luna + Haiku + Jev, all judge every response": ("#1F78B4", "D"),
    "Luna + Haiku + Jev, each response by one": ("#1F78B4", "P"),
}
K_MARKER = {"k2": "s", "k4": "^", "k6": "v"}


def canonical_family(family: str) -> str:
    """The FAMILY_STYLE key of a family: itself, or with its judge's name read as Luna's (the 61-axis mode names its
    judge: "Haiku 5.5 stacked, plain" is styled as "Luna stacked, plain")."""
    if family in FAMILY_STYLE:
        return family
    return re.sub(r"^[^,]+?(, | stacked)", lambda m: "Luna" + m.group(1), family, count=1)


def style_of(family: str) -> Tuple[str, str]:
    return FAMILY_STYLE[canonical_family(family)]


def plot(points: Dict[str, dict], variants, out_png: Path, title: str, n_axes: int, check: dict, inputs,
         ref_arms=frozenset(), lead: str = "", axes_file: str = AXES_12, base: float = NINTH) -> None:
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
            fkeys = [k for k in keys if points[k]["arm"] not in ref_arms]
            idx = pareto_frontier([(points[k]["cost"], q[k]) for k in fkeys])
            front = sorted((fkeys[i] for i in idx), key=lambda k: points[k]["cost"])
            for arm in dict.fromkeys(p["arm"] for p in points.values()):
                ks = sorted([k for k in keys if points[k]["arm"] == arm], key=lambda k: points[k]["cost"])
                fam = points[ks[0]]["family"]
                col, mk = ("#BBBBBB", "o") if arm in ref_arms else style_of(fam)
                kk = re.search(r"k(\d)$", arm)
                if kk and "stacked" in fam:
                    mk = K_MARKER[f"k{kk.group(1)}"]
                ms = [k for k in ks if points[k]["measured"]]
                ex = [k for k in ks if not points[k]["measured"]]
                ax.plot([points[k]["cost"] for k in ms], [q[k] for k in ms], "-", color=col, lw=0.8, alpha=0.6)
                if ex:
                    ax.plot([points[k]["cost"] for k in ms[-1:] + ex], [q[k] for k in ms[-1:] + ex], ":", color=col, lw=0.8, alpha=0.6)
                ax.scatter([points[k]["cost"] for k in ms], [q[k] for k in ms], s=[18 + 22 * points[k]["m"] / FRACTION_K for k in ms],
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
    ref_fams = {p["family"] for p in points.values() if p["arm"] in ref_arms}
    present = list(dict.fromkeys(p["family"] for p in points.values()))
    order = list(FAMILY_STYLE)
    present.sort(key=lambda f: order.index(canonical_family(f)))
    handles = [plt.Line2D([0], [0], marker=style_of(fam)[1], color=style_of(fam)[0], lw=0.8, label=fam, markersize=7)
               for fam in present if fam not in ref_fams]
    handles += [plt.Line2D([0], [0], marker="o", color="#BBBBBB", lw=0.8, label=f"{fam} (reference, off the frontier)", markersize=7)
                for fam in sorted(ref_fams)]
    if any("stacked" in p["family"] for p in points.values()):
        handles += [plt.Line2D([0], [0], marker=m, color="#777777", lw=0, label=f"stacked {k[1]} axes a call", markersize=7)
                    for k, m in K_MARKER.items()]
    handles += [plt.Line2D([0], [0], marker="o", color="#777777", markerfacecolor="none", lw=0, label="extrapolated (1/3, all)"),
                plt.Line2D([0], [0], marker="o", color="#222222", markerfacecolor="none", markersize=10, lw=0,
                           label="on the frontier (its convex hull)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    sub = (f"{n_axes} axes ({axes_file}); {lead}\n"
           f"{'a ninth' if base == NINTH else f'1/{round(1 / base)}'} of the kept responses measured: points from 1/{round(3 / base)} (smallest) to "
           f"1/{round(1 / base)}, subsampled the golden way (absolute question and instruction); larger coverages extrapolated "
           f"(hollow: Spearman-Brown by question, gains scaled by {check['calibration']:.2f}, the measured-to-predicted ratio on Luna's third and full cells)\n"
           f"cost of a full run, 200 October axes (1,273 entities x 425 kept responses) at the point's coverage, every prefix cached: Luna by Flex, "
           f"Haiku by Message Batches, Gemini by its Batch API with explicit caches, Jev at its measured cost (no cache)")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.965, sub, ha="center", va="top", fontsize=8.5)
    fig.subplots_adjust(top=0.885, bottom=0.12, left=0.05, right=0.985, hspace=0.32, wspace=0.18)
    fig.savefig(out_png, dpi=130, metadata=png_metadata(title=title, inputs=inputs))
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
