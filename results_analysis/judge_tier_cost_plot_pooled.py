#!/usr/bin/env python3
"""
Pooled judge-tier cost vs quality plot across multiple axes.

Reads all *_tier_eval/ directories found under roger/axis_judge_experiments/
(or a specified set via --tier_eval_dirs), aggregates per-axis rho and cost
for each judge model, then plots mean quality = 1/(1-rho) vs mean cost-per-axis
with the Pareto frontier highlighted.

Also optionally overlays the two legacy reference models ("gpt-4.1-mini (May)" and
"claude-sonnet-4 (May)", the May 2026 cells: T=0, max_tokens 1024) whose judge caches predate usage.json tracking: their rhos
come from the gpt/ and sonnet/ subdirs of the original axis directories and
their costs are ESTIMATED from the same token/pricing model as the dry-run.

Coverage rule: the plot's headline "N axes" is the largest number of axes any
model was judged on.  A model judged on fewer axes than that is NOT plotted
(its mean rho is over a different, usually easier, subset and is not
comparable); the models left out are named in the title and on stderr.
--min_axes N lowers the bar, e.g. --min_axes 60 keeps a legacy judge that
is missing one axis, and --min_axes 1 shows everything as before.

Price basis: a cell's recorded cost (usage.json's total_cost_usd) depends on
the transport it happened to go by, so by default every cell's cost is
recomputed from its recorded token counts.  Since 2026-11-03 the default is
--price_basis batch, the real price (Roger: "We should probably be computing
Pareto at real price, in case someone offered a bigger or smaller discount for
batch"): each model at its own batch or Flex rate (judge_pricing's
_BATCH_PRICE_FACTORS; the list price for a model with none).  Every judge here
is half price that way, so the frontier is the one the list price gave; the
dollars are half.  --price_basis list prices everything at the real-time list
price (the default until 2026-11-03, and the basis of every table before it),
and --price_basis recorded plots the cost as billed.

Usage:
    # After running judge_tier_cost_eval.py on 18+ axes:
    uv run python results_analysis/judge_tier_cost_plot_pooled.py \\
        --experiments_dir roger/axis_judge_experiments \\
        --slot 6 \\
        --output roger/axis_judge_experiments/pareto_pooled_20axes.png

    # Restrict to a specific set of axes:
    uv run python results_analysis/judge_tier_cost_plot_pooled.py \\
        --tier_eval_dirs \\
            roger/axis_judge_experiments/angel_vs_demon_tier_eval \\
            roger/axis_judge_experiments/trustworthy_vs_untrustworthy_tier_eval \\
        --slot 6 \\
        --output /tmp/pareto_subset.png
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from assistant_axis.judge_pricing import (
    GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT, SONNET_RATE_IN, SONNET_RATE_OUT,
    cache_rate_factors, price_for_model,
)
from assistant_axis.plot_metadata import png_metadata
from assistant_axis.provenance import (
    CACHE_POLICIES, InputSpec, current_file_input, current_files_input, load_and_register,
)

PROVIDER_COLOR = {"openai": "#7FB069", "anthropic": "#D9636B", "google": "#5B8DEF", "typesafe": "#E39B2D",
                  "blend": "#8E6BBF"}

EST_IN_PER_ENTITY = 320 + 150
EST_OUT_PER_ENTITY = 80


def provider_of(model: str) -> str:
    if "+" in model:                # judges blended: gpt-5.6-luna+claude-haiku-5-5@...
        return "blend"
    if model.startswith("gemini"):
        return "google"
    if model.startswith("jev"):     # TypeSafe AI's Jev (2026-11-04)
        return "typesafe"
    return "openai" if model.startswith(("gpt-", "o1-", "o3-", "o4-")) else "anthropic"


#: Every cell file this process has read, for the figure's recorded inputs.  The cells are bare JSON
#: with no envelope of their own, so their drift is seen when a later audit compares these files.
READ_FILES: set = set()


def read_json(path: Path) -> Any:
    """``path``'s JSON, noted in :data:`READ_FILES`."""
    with open(path) as f:
        data = json.load(f)
    READ_FILES.add(Path(path))
    return data


def load_rho(correlations_path: Path, slot: int,
             mode: str = "descriptions", metric: str = "raw") -> Optional[Tuple[float, int]]:
    if not correlations_path.exists():
        return None
    d = read_json(correlations_path)
    r = d.get("result", d)
    try:
        cell = r[mode][str(slot)][metric]
        return float(cell["rho"]), int(cell["n"])
    except (KeyError, TypeError, ValueError):
        return None


PRICE_BASES = ("batch", "list", "recorded")
#: The real price (2026-11-03): each model at its own batch or Flex rate.
DEFAULT_PRICE_BASIS = "batch"


def _base_model(key: str) -> str:
    """``claude-haiku-4-5:batch`` -> ``claude-haiku-4-5`` (usage keys carry the transport)."""
    for suffix in (":batch", ":flex"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def basis_rates(model: str, price_basis: str) -> Tuple[float, float]:
    """``model``'s ``(input, output)`` rates per 1M tokens on ``price_basis``: its real-time list
    price, or (``batch``) its own batch or Flex rate, the list price when it has none.  Raises
    ``KeyError`` for a model the pricing table does not know."""
    if price_basis == "batch":
        try:
            return price_for_model(f"{model}:batch")
        except KeyError:
            pass
    return price_for_model(model)


def price_factor(model: str, price_basis: str) -> float:
    """What ``price_basis`` pays for ``model`` as a fraction of its list price (1.0 when unknown)."""
    try:
        return basis_rates(model, price_basis)[0] / price_for_model(model)[0]
    except (KeyError, ZeroDivisionError):
        return 1.0


def load_measured_cost(usage_path: Path, price_basis: str = DEFAULT_PRICE_BASIS) -> Optional[float]:
    """A cell's cost from its ``usage.json``.

    ``recorded``: the cost the run was billed (``total_cost_usd``), which mixes
    real-time cells with batch and Flex cells billed at their discount.  ``batch``
    (the default, the real price) and ``list`` recompute the cost from the
    recorded token counts at each model's own batch or Flex rate (:func:`basis_rates`)
    or at its real-time price, whatever transport the cell was judged by, so that
    every point on one plot is priced the same way.  A model the pricing table
    does not know falls back to the recorded cost with a warning.
    """
    if not usage_path.exists():
        return None
    d = read_json(usage_path)
    if price_basis == "recorded" or not d.get("per_model"):
        return d.get("total_cost_usd")
    total = 0.0
    for key, entry in d["per_model"].items():
        model = _base_model(entry.get("model") or key)
        try:
            rate_in, rate_out = basis_rates(model, price_basis)
        except KeyError:
            print(f"WARNING: {usage_path}: no {price_basis} price for {model!r}; using the recorded cost",
                  file=sys.stderr)
            return d.get("total_cost_usd")
        # cache-written and cache-read tokens (part of prompt_tokens) at their factors of the input rate, as
        # billed (judge_pricing.cache_rate_factors; before 2026-11-04 they were repriced as plain input, which
        # put a cached-prefix arm at five times its bill)
        written, read = int(entry.get("cache_write_tokens", 0)), int(entry.get("cache_read_tokens", 0))
        write_factor, read_factor = cache_rate_factors(model)
        plain = entry.get("prompt_tokens", 0) - written - read
        total += ((plain + written * write_factor + read * read_factor) * rate_in
                  + entry.get("completion_tokens", 0) * rate_out) / 1e6
    return total


#: The reply cap of judge_tier_cost_eval.py runs made before each cell recorded
#: its own: the script's --max_tokens default then (300 since 2026-11-01, when
#: cells already record theirs; the 20- and 41-axis drivers never set it).  A
#: reply cut at the cap usually lost its SCORE line and went unscored, or was
#: scored from a stray number (cut replies score as missing since 2026-11-01).
TIER_EVAL_DEFAULT_MAX_TOKENS = 200


def cell_max_tokens(cell_dir: Path, legacy: bool = False) -> Optional[int]:
    """The judge's reply cap (max_tokens) for one cell.

    A tier-eval cell: its ``scores_descriptions.json`` provenance
    ``max_tokens``, else :data:`TIER_EVAL_DEFAULT_MAX_TOKENS`.  A legacy
    (May 2026, axis_judge_correlation.py) cell: ``config.json``'s
    ``args.max_tokens``, else None (not recorded)."""
    try:
        if legacy:
            value = (read_json(cell_dir / "config.json").get("args") or {}).get("max_tokens")
            return int(value) if value is not None else None
        prov = read_json(cell_dir / "scores_descriptions.json").get("_provenance") or {}
        value = prov.get("max_tokens")
        return int(value) if value is not None else TIER_EVAL_DEFAULT_MAX_TOKENS
    except (OSError, ValueError, TypeError, AttributeError):
        return None if legacy else TIER_EVAL_DEFAULT_MAX_TOKENS


def cap_summary(caps: Dict[str, set]) -> str:
    """One title line naming the judges' reply caps, ``{model: {max_tokens, ...}}``,
    grouped by cap: the largest group by count, the others by name."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for model, values in caps.items():
        known = sorted(v for v in values if v is not None)
        key = "/".join(str(v) for v in known) or "unrecorded"
        if known and None in values:
            key += " or unrecorded"
        groups[key].append(model)
    tail = " (a reply cut at the cap gives no score)"
    if len(groups) == 1:
        key = next(iter(groups))
        return f"Judge reply cap: max_tokens {key} for every judge{tail}"
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    parts = []
    for i, (key, models) in enumerate(ordered):
        who = f"{len(models)} judges" if i == 0 and len(models) > 2 else ", ".join(sorted(models))
        parts.append(f"{key} for {who}")
    return "Judge reply cap (max_tokens): " + "; ".join(parts) + tail


def estimate_cost(n_entities: int, rate_in: float, rate_out: float) -> float:
    return (n_entities * EST_IN_PER_ENTITY * rate_in
            + n_entities * EST_OUT_PER_ENTITY * rate_out) / 1e6


def pareto_frontier(points: List[Tuple[float, float]], hull: bool = True) -> List[int]:
    """Indices of the frontier points ``(cost, quality)``, cheapest first: not dominated (min cost, max
    quality) and, with ``hull`` (the default since 2026-11-03; Roger: "Only count points on the Pareto
    frontier if they are on its convex hull"), on the upper convex hull.  Quality is 1/(1 - mean rho), as
    every caller passes it; the hull is taken on mean rho = 1 - 1/quality against linear cost, the units of
    an exchange rate (rho per $ per axis), so that a point under the chord of its neighbours, the best
    choice at no exchange rate, is left off.  A point on the chord itself stays.  ``hull=False``: the
    non-dominated points alone, the rule before that date."""
    indices = []
    for i, (cost_i, quality_i) in enumerate(points):
        dominated = False
        for j, (cost_j, quality_j) in enumerate(points):
            if j == i:
                continue
            if cost_j <= cost_i and quality_j >= quality_i and (cost_j < cost_i or quality_j > quality_i):
                dominated = True
                break
        if not dominated:
            indices.append(i)
    indices = sorted(indices, key=lambda i: points[i][0])
    if not hull:
        return indices

    def xy(i: int) -> Tuple[float, float]:
        cost, quality = points[i]
        return cost, (1.0 - 1.0 / quality) if quality else -math.inf

    kept: List[int] = []
    for i in indices:
        while len(kept) >= 2:
            (xa, ya), (xb, yb), (xc, yc) = xy(kept[-2]), xy(kept[-1]), xy(i)
            if (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa) > 1e-15:    # b lies under the chord a-c
                kept.pop()
            else:
                break
        kept.append(i)
    return kept


def collect_tier_eval_data(
    tier_eval_dirs: List[Path],
    slot: int,
    price_basis: str = DEFAULT_PRICE_BASIS,
) -> Dict[str, List[dict]]:
    """
    Returns {model: [{rho, cost, cost_is_measured, axis_name}, ...]}
    """
    per_model: Dict[str, List[dict]] = defaultdict(list)

    for tier_dir in tier_eval_dirs:
        if not tier_dir.is_dir():
            print(f"WARNING: {tier_dir} does not exist — skipping", file=sys.stderr)
            continue
        axis_name = tier_dir.name.replace("_tier_eval", "")
        for model_dir in sorted(tier_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            result = load_rho(model_dir / "correlations.json", slot)
            if result is None:
                continue
            rho, n = result
            cost = load_measured_cost(model_dir / "usage.json", price_basis)
            if cost is None:
                print(f"  NOTE: {model} / {axis_name}: no usage.json — skipping this entry",
                      file=sys.stderr)
                continue
            per_model[model].append({
                "rho": rho, "n": n, "cost": cost,
                "cost_is_measured": True,
                "axis_name": axis_name,
                "max_tokens": cell_max_tokens(model_dir),
            })

    return per_model


def rho_from_stored_scores(cell_dir: Path, axis_name: str, slot: int,
                           mode: str = "descriptions", metric: str = "raw") -> Optional[Tuple[float, int]]:
    """Recompute a legacy cell's rho from its ``scores_<mode>.json`` and
    ``projections.json`` when its ``correlations.json`` lacks the mode.

    A later run on the same cell that scored only another mode rewrites
    correlations.json without this one (the Sonnet cell of
    aligned_artificial_intelligence_vs_paperclip_maximizer: descriptions
    scored 580/580 on 2026-05-13, then a rejudge of one role's instructions
    the same evening left correlations.json with instructions only).  Uses
    ``compute_correlations`` itself, with the axis's two poles excluded, so
    the result is what the original run computed.
    """
    scores_path = cell_dir / f"scores_{mode}.json"
    proj_path = cell_dir / "projections.json"
    if not (scores_path.exists() and proj_path.exists()):
        return None
    from results_analysis.axis_judge_correlation import compute_correlations  # heavy; only when needed

    sd = read_json(scores_path)
    scores = {k: v for k, v in sd.get("result", sd).items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)}
    pd_ = read_json(proj_path)
    projections = {int(s): v for s, v in pd_.get("result", pd_).items() if s.lstrip("-").isdigit()}
    poles = set(axis_name.split("_vs_", 1))
    corr = compute_correlations({mode: scores}, projections, [slot], poles)
    cell = corr[mode][str(slot)][metric]
    if cell["n"] < 3:
        return None
    return float(cell["rho"]), int(cell["n"])


def collect_legacy_data(
    legacy_axis_dirs: List[Path],
    slot: int,
    price_basis: str = DEFAULT_PRICE_BASIS,
) -> Dict[str, List[dict]]:
    """
    Load gpt-4.1-mini and claude-sonnet-4 from the non-tier-eval axis dirs.
    These predate usage.json; costs are estimated at the real-time price
    (``list``, also used for ``recorded``: they were judged in real time) or
    at the model's own batch rate (``batch``).
    """
    per_model: Dict[str, List[dict]] = defaultdict(list)

    legacy_cells = [
        # Labelled "(May)": the May 2026 cells of axis_judge_correlation.py (T=0, max_tokens
        # 1024), kept apart from any tier-eval cell of the same model (gpt-4.1-mini rejudged at
        # T=0 and max_tokens 200 on 2026-11-01 sits under <axis>_tier_eval/gpt-4.1-mini/).
        ("gpt-4.1-mini (May)", "gpt", GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT),
        ("claude-sonnet-4 (May)", "sonnet", SONNET_RATE_IN, SONNET_RATE_OUT),
    ]

    for axis_dir in legacy_axis_dirs:
        axis_name = axis_dir.name
        for model, subdir, rin, rout in legacy_cells:
            corr = axis_dir / subdir / "correlations.json"
            result = load_rho(corr, slot)
            if result is None:
                result = rho_from_stored_scores(axis_dir / subdir, axis_name, slot)
                if result is not None:
                    print(f"  NOTE: {model} / {axis_name}: correlations.json has no descriptions "
                          f"result; rho recomputed from scores_descriptions.json and "
                          f"projections.json (rho={result[0]:.3f}, n={result[1]})", file=sys.stderr)
            if result is None:
                continue
            rho, n = result
            cost = estimate_cost(n, rin, rout) * price_factor(model.split(" (")[0], price_basis)
            per_model[model].append({
                "rho": rho, "n": n, "cost": cost,
                "cost_is_measured": False,
                "axis_name": axis_name,
                "max_tokens": cell_max_tokens(axis_dir / subdir, legacy=True),
            })

    return per_model


def rho_from_robustness(per_model: Dict[str, List[dict]], robustness_json: Path,
                        variant: str = "s6_l25_raw",
                        *, inputs: Optional[list] = None, policy: str = "warn") -> Dict[str, List[dict]]:
    """Replace every entry's rho and n with judge_tier_robustness.py's for ``variant``.

    The robustness tool recomputes rho from the stored scores and drops the nine
    trait/role collision names on bare-keyed (v1) axes, where the tier-eval cells
    judged a collision name's role description against the trait's cached
    projection; its s6_l25_raw values are therefore the corrected version of the
    cells' correlations.json.  An entry the JSON lacks is dropped with a note."""
    # load_and_register unwraps the envelope, checks its recorded inputs under ``policy``, and
    # records the file as one of the figure's inputs.
    r, _spec, _check = load_and_register(robustness_json, dep_key="robustness_json", inputs=inputs, policy=policy)
    r = r.get("result", r)       # a JSON with no provenance block keeps its "result" wrapper
    rho, n = r["rho"], r.get("n", {})
    out: Dict[str, List[dict]] = defaultdict(list)
    for model, entries in per_model.items():
        for e in entries:
            value = (rho.get(model, {}).get(e["axis_name"]) or {}).get(variant)
            if value is None:
                print(f"  NOTE: {model} / {e['axis_name']}: not in {robustness_json} for {variant}; dropped",
                      file=sys.stderr)
                continue
            out[model].append({**e, "rho": float(value), "n": int(n.get(model, {}).get(e["axis_name"], e["n"]))})
    return out


def aggregate(per_model: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Collapse per-axis entries into mean rho and mean cost per model."""
    agg: Dict[str, dict] = {}
    for model, entries in per_model.items():
        if not entries:
            continue
        mean_rho = float(np.mean([e["rho"] for e in entries]))
        mean_cost = float(np.mean([e["cost"] for e in entries]))
        n_axes = len(entries)
        measured = all(e["cost_is_measured"] for e in entries)
        agg[model] = {
            "mean_rho": mean_rho,
            "mean_cost": mean_cost,
            "n_axes": n_axes,
            "cost_is_measured": measured,
            "provider": provider_of(model),
            "max_tokens": {e.get("max_tokens") for e in entries},
        }
    return agg


def split_by_coverage(agg: Dict[str, dict], min_axes: int) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Split aggregated models into (plotted, excluded) by axis coverage.

    A model enters the plot only if it was judged on at least ``min_axes``
    axes.  The excluded models are returned so the plot can say they exist.
    """
    kept = {m: d for m, d in agg.items() if d["n_axes"] >= min_axes}
    excluded = {m: d for m, d in agg.items() if d["n_axes"] < min_axes}
    return kept, excluded


#: What a cell directory's label suffix (judge_tier_cost_eval.py --cell_label) means, for plot labels.
CELL_LABEL_TEXT = {"@none": "no reasoning", "@t0": "T=0", "@nothink": "thinking off",
                   "@low": "reasoning low", "@med": "reasoning medium"}
#: A note after a model's name in plot labels (OpenAI deprecations page, read 2026-11-01:
#: gpt-4.1-mini leaves the API on 2027-01-14, replacement gpt-6-luna).
MODEL_NOTES = {"gpt-4.1-mini": "retires 2027-01-14"}
#: The setting of cells without a label, where it matters for reading the plot: the 2026-10-26 runs
#: (Opus 5 and Sonnet 5 at output_config effort "low", GPT-5.6 at the default reasoning effort) and
#: claude-haiku-4-5 at its default temperature (superseded by claude-haiku-4-5@t0).
UNLABELED_SETTINGS = {"claude-opus-5": "effort low", "claude-sonnet-5": "effort low", "claude-haiku-4-5": "T=1",
                      "gpt-5.6-luna": "default reasoning", "gpt-5.6-terra": "default reasoning",
                      "gpt-5.6-sol": "default reasoning"}


#: Further label tokens (labels stack: ``gpt-5.6-luna@none@c2``): rubric variants of
#: judge_tier_rubrics.py and repeat runs.
LABEL_TOKEN_TEXT = {"v3fix": "rubric v3fix", "rep2": "repeat run"}


#: Rubric and batching tokens are shown as they are (``c8``, ``b10x2``), with their meaning in a
#: key under the plot title (:func:`label_key`): spelled out they made labels too long to follow
#: to their points (Roger, 2026-11-01).
TOKEN_KEY = {
    r"c(\d+)": "c{0}: single-entity rubric c{0}",
    r"b(\d+)n(\d+)x(\d+)": "b{0}n{1}x{2}: batched rubric b{0}, {1} entities per call, {2} shuffles averaged",
    r"b(\d+)n(\d+)s(\d+)": "b{0}n{1}s{2}: batched rubric b{0}, {1} entities per call, shuffle {2}",
    r"b(\d+)x(\d+)": "b{0}x{1}: batched rubric b{0}, 10 entities per call, {1} shuffles averaged",
    r"b(\d+)s(\d+)": "b{0}s{1}: batched rubric b{0}, shuffle {1}",
    r"b(\d+)g2": "b{0}g2: batched rubric b{0}, rows and columns averaged",
    r"b(\d+)g([rc])": "b{0}g{1}: batched rubric b{0}, grid pass {1}",
}


def _label_text(token: str) -> str:
    if f"@{token}" in CELL_LABEL_TEXT:
        return CELL_LABEL_TEXT[f"@{token}"]
    if token in LABEL_TOKEN_TEXT:
        return LABEL_TOKEN_TEXT[token]
    if any(re.fullmatch(pat, token) for pat in TOKEN_KEY):
        return token
    return f"@{token}"


def label_key(cells: Sequence[str]) -> str:
    """One line explaining the rubric and batching tokens the given cells use, for a plot title."""
    seen: List[str] = []
    for cell in cells:
        for token in cell.split("@")[1:]:
            for pat, text in TOKEN_KEY.items():
                m = re.fullmatch(pat, token)
                if m:
                    t = text.format(*m.groups())
                    if t not in seen:
                        seen.append(t)
    return "; ".join(seen)


def display_name(cell: str, short: bool = False) -> str:
    """A cell directory name as plots show it: ``gpt-5.6-luna@none`` ->
    ``gpt-5.6-luna (no reasoning)``, ``gpt-5.6-luna@none@c2`` -> ``gpt-5.6-luna
    (no reasoning, c2)``, ``gpt-4.1-mini`` -> ``gpt-4.1-mini (retires
    2027-01-14)``; other names unchanged.  ``short`` drops the model notes
    (for small panels)."""
    model, *labels = cell.split("@")
    if labels:
        notes = [_label_text(t) for t in labels]
    else:
        notes = [UNLABELED_SETTINGS[model]] if model in UNLABELED_SETTINGS else []
    if model in MODEL_NOTES and not short:
        notes.append(MODEL_NOTES[model])
    return f"{model} ({', '.join(notes)})" if notes else model


def keep_models(per_model: Dict[str, Any], names: Sequence[str]) -> List[str]:
    """Keep only the cells named in ``names`` (--include_models); returns the names removed."""
    if not names:
        return []
    for m in names:
        if m not in per_model:
            print(f"  NOTE: --include_models {m}: no such cell", file=sys.stderr)
    dropped = [m for m in per_model if m not in names]
    for m in dropped:
        del per_model[m]
    return dropped


def drop_models(per_model: Dict[str, Any], names: Sequence[str]) -> List[str]:
    """Remove the cells named in ``names`` (--exclude_models) from ``per_model``
    in place; returns the names actually removed."""
    dropped = [m for m in names if m in per_model]
    for m in dropped:
        del per_model[m]
    for m in names:
        if m not in dropped:
            print(f"  NOTE: --exclude_models {m}: no such cell", file=sys.stderr)
    return dropped


# Label positions tried around a point, in points: (dx, dy, ha).  The text
# sits with va="center" at (dx, dy) from the point; the later entries are
# farther out and get a leader line.
_LABEL_CANDIDATES = [(7, 0, "left"), (-7, 0, "right"), (0, 9, "center"), (0, -9, "center"),
                     (6, 9, "left"), (-6, 9, "right"), (6, -9, "left"), (-6, -9, "right"),
                     (16, 16, "left"), (-16, 16, "right"), (16, -16, "left"), (-16, -16, "right"),
                     (0, 20, "center"), (0, -20, "center"), (24, 0, "left"), (-24, 0, "right"),
                     (30, 26, "left"), (-30, 26, "right"), (30, -26, "left"), (-30, -26, "right")]
_NEAR_CANDIDATES = 8


def place_labels(ax, points: Sequence[Tuple[float, float]], labels: Sequence[str],
                 bold: Sequence[bool], fontsize: float = 7.5, leaders: str = "far") -> None:
    """Greedy label placement: each label takes the first candidate position
    whose (estimated) box overlaps no marker, no earlier label, stays in the
    axes and lies nearer its own point than any other point (so a label is
    never read as another point's); failing that, the least bad one.  A label
    may have several lines.  Call after the axes limits, scale and position
    are final.  ``leaders``: ``"far"`` draws a leader line only for a label
    placed away from its point, ``"all"`` for every label.  A point whose label
    is empty gets none but still keeps the others off it."""
    k = ax.figure.dpi / 72.0
    disp = ax.transData.transform(np.asarray(points, dtype=float))
    ab = ax.get_window_extent()
    r = 5.5 * k
    obstacles = [(x - r, y - r, x + r, y + r) for x, y in disp]

    def overlap(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0

    def gap(box, p):
        dx = max(box[0] - p[0], 0.0, p[0] - box[2])
        dy = max(box[1] - p[1], 0.0, p[1] - box[3])
        return (dx * dx + dy * dy) ** 0.5

    placed: List[Tuple[float, float, float, float]] = []
    for i in sorted(range(len(labels)), key=lambda i: (not bold[i], -disp[i][1])):
        if not labels[i]:
            continue
        lines = labels[i].split("\n")
        w = (0.62 if bold[i] else 0.56) * fontsize * max(len(t) for t in lines) * k
        h = 1.1 * fontsize * k * len(lines)
        best, best_cost = None, float("inf")
        for ci, (dx, dy, ha) in enumerate(_LABEL_CANDIDATES):
            x0 = disp[i][0] + dx * k - (w if ha == "right" else w / 2 if ha == "center" else 0.0)
            y0 = disp[i][1] + dy * k - h / 2
            box = (x0, y0, x0 + w, y0 + h)
            cost = sum(overlap(box, o) for o in obstacles) + sum(overlap(box, p) for p in placed)
            out = (max(0.0, ab.x0 - box[0]) + max(0.0, box[2] - ab.x1)
                   + max(0.0, ab.y0 - box[1]) + max(0.0, box[3] - ab.y1))
            own = gap(box, disp[i])
            nearer = sum(1 for j, p in enumerate(disp) if j != i and gap(box, p) < own)
            cost += out * h * 4 + nearer * h * h + ci * 1e-3
            if cost < best_cost:
                best, best_cost = (ci, dx, dy, ha, box), cost
            if cost < 1e-2:
                break
        ci, dx, dy, ha, box = best
        placed.append(box)
        ax.annotate(labels[i], points[i], textcoords="offset points", xytext=(dx, dy), ha=ha, va="center",
                    fontsize=fontsize, fontweight="bold" if bold[i] else "normal", zorder=5,
                    arrowprops=(dict(arrowstyle="-", color="#888888", lw=0.6, shrinkA=0, shrinkB=4)
                                if (leaders == "all" or ci >= _NEAR_CANDIDATES) else None))


def cost_ticks(lo: float, hi: float) -> List[float]:
    cands = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    return [c for c in cands if lo <= c <= hi]


PRICE_BASIS_LABEL = {
    "list": "real-time list price",
    "batch": "real price: each judge's batch or Flex rate",
    "recorded": "as billed (mixed transports)",
}


def plot(agg: Dict[str, dict], slot: int, n_total_axes: int, output_path: Path,
         excluded: Optional[Dict[str, dict]] = None, price_basis: str = DEFAULT_PRICE_BASIS,
         rho_note: str = "", left_out: Sequence[str] = (), links: Sequence[Tuple[str, str]] = (),
         links_label: str = "same judge: old rubric → new rubric",
         inputs: Optional[list] = None) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    model_list = sorted(agg.items(), key=lambda kv: kv[1]["mean_cost"])
    costs = [d["mean_cost"] for _, d in model_list]
    qualities = [1.0 / (1.0 - d["mean_rho"]) for _, d in model_list]

    frontier_idx = pareto_frontier(list(zip(costs, qualities)))
    frontier_costs = [costs[i] for i in frontier_idx]
    frontier_qualities = [qualities[i] for i in frontier_idx]

    # Draw Pareto frontier line
    ax.plot(frontier_costs, frontier_qualities, "k--", linewidth=1.2, alpha=0.5, zorder=2,
            label="Pareto frontier")

    # Costs span two orders of magnitude (GPT-6 Luna about $0.03 per axis, Opus 5 about $3): a log
    # axis keeps the cheap judges apart.
    ax.set_xscale("log")
    x_lo, x_hi = min(costs) / 1.8, max(costs) * 1.8
    ax.set_xlim(x_lo, x_hi)
    q_lo, q_hi = min(qualities), max(qualities)
    pad = 0.12 * ((q_hi - q_lo) or 0.1)
    ax.set_ylim(q_lo - pad, q_hi + pad)
    ticks = cost_ticks(x_lo, x_hi)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"\\${t:g}" for t in ticks])
        ax.minorticks_off()

    pts, labels, bold = [], [], []
    for i, (model, d) in enumerate(model_list):
        quality = qualities[i]
        cost = costs[i]
        is_pareto = i in frontier_idx
        facecolor = PROVIDER_COLOR[d["provider"]] if d["cost_is_measured"] else "none"
        edgecolor = PROVIDER_COLOR[d["provider"]]
        marker = "D" if is_pareto else "o"
        size = 200 if is_pareto else 120
        ax.scatter(cost, quality, s=size, marker=marker,
                   facecolors=facecolor, edgecolors=edgecolor, linewidths=2.0, zorder=4)
        cost_note = "" if d["cost_is_measured"] else " (est.)"
        price = f"\\${cost:.2f}" if cost >= 0.1 else f"\\${cost:.3f}"
        pts.append((cost, quality))
        labels.append(f"{display_name(model)}\nρ={d['mean_rho']:.3f}  {price}/axis{cost_note}")
        bold.append(is_pareto)
    # Arrows between named cells (--links): a judge's old-rubric point to its new-rubric point.
    pos = {model: (costs[i], qualities[i]) for i, (model, _) in enumerate(model_list)}
    for a, b in links:
        if a in pos and b in pos:
            ax.annotate("", xy=pos[b], xytext=pos[a], zorder=3,
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1, alpha=0.75,
                                        shrinkA=9, shrinkB=9))
        else:
            print(f"  NOTE: --links {a}:{b}: not both plotted", file=sys.stderr)
    fig.tight_layout()
    place_labels(ax, pts, labels, bold, fontsize=9)

    ax.set_xlabel(f"Mean cost per axis (USD at {PRICE_BASIS_LABEL[price_basis]}, log scale; descriptions mode)")
    ax.set_ylabel("Mean quality = 1 / (1 − ρ̄)")
    title = (f"Judge-tier Pareto, DESCRIPTION judging: {n_total_axes} axes, slot {slot}  "
             f"[◆ = Pareto-optimal (bold label), ● = dominated]")
    if price_basis == "list":
        title += "\nCost at real-time list price; Flex and Batch are half price for every judge here"
    elif price_basis == "batch":
        title += "\nCost at each judge's batch or Flex price (half of list for every judge here)"
    if any(not d["cost_is_measured"] for d in agg.values()):
        title += "\nFilled = measured cost (usage.json)  Hollow = estimated (legacy tiers)"
    key = label_key(list(agg))
    if key:
        title += f"\nLabel key: {key}"
    if rho_note:
        title += f"\n{rho_note}"
    if excluded:
        names = ", ".join(f"{display_name(m)} ({d['n_axes']})"
                          for m, d in sorted(excluded.items(), key=lambda kv: -kv[1]["n_axes"]))
        title += f"\nNot shown, judged on fewer than {n_total_axes} axes: {names}"
    if left_out:
        title += "\nLeft out (--exclude_models): " + ", ".join(display_name(m) for m in left_out)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)

    provider_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PROVIDER_COLOR["anthropic"],
                   markeredgecolor=PROVIDER_COLOR["anthropic"],
                   markersize=10, label="Anthropic"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PROVIDER_COLOR["openai"],
                   markeredgecolor=PROVIDER_COLOR["openai"],
                   markersize=10, label="OpenAI"),
        plt.Line2D([0], [0], linestyle="--", color="black", alpha=0.5,
                   label="Pareto frontier"),
    ]
    if any(d["provider"] == "google" for d in agg.values()):
        provider_handles.insert(2, plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PROVIDER_COLOR["google"],
                                              markeredgecolor=PROVIDER_COLOR["google"], markersize=10, label="Google"))
    if any(d["provider"] == "blend" for d in agg.values()):
        provider_handles.insert(2, plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PROVIDER_COLOR["blend"],
                                              markeredgecolor=PROVIDER_COLOR["blend"], markersize=10, label="Blend of judges"))
    if links:
        provider_handles.append(plt.Line2D([0], [0], color="#555555", lw=1.1, alpha=0.75, label=links_label))
    # "best" looks at the point labels too: a fixed corner covered the label of whichever
    # judge sat there (claude-haiku-4-5 at the lower right, 2026-11-02).
    ax.legend(handles=provider_handles, loc="best", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                metadata=png_metadata(
                    title=f"Judge-tier Pareto, description judging ({n_total_axes} axes, slot {slot})",
                    inputs=inputs,
                ))
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiments_dir", type=str,
                   help="Root dir to glob for *_tier_eval subdirectories "
                        "(e.g. roger/axis_judge_experiments)")
    g.add_argument("--tier_eval_dirs", nargs="+", type=str,
                   help="Explicit list of tier_eval dirs")
    parser.add_argument("--legacy_dirs", nargs="*", type=str, default=None,
                        help="Optional: non-tier-eval axis dirs that have gpt/ and sonnet/ "
                             "subdirs for estimating legacy model cost+rho "
                             "(e.g. roger/axis_judge_experiments/angel_vs_demon). "
                             "If omitted with --experiments_dir, auto-detected from "
                             "*_tier_eval siblings that have a gpt/correlations.json.")
    parser.add_argument("--slot", type=int, default=6)
    parser.add_argument("--legacy", action="store_true",
                        help="Also plot the May 2026 cells, \"gpt-4.1-mini (May)\" and \"claude-sonnet-4 "
                             "(May)\" (axis_judge_correlation.py, T=0, max_tokens 1024, rubric v2/v3/unrecorded; "
                             "the Sonnet cells mix claude-sonnet-4-5 and the retired claude-sonnet-4-20250514).  "
                             "Off by default since 2026-11-01 (Roger): gpt-4.1-mini is rejudged under the "
                             "tier-eval settings at T=0 and the Sonnet point cannot be reproduced.")
    parser.add_argument("--no_legacy", action="store_true",
                        help="Deprecated: legacy cells are off unless --legacy is given.")
    parser.add_argument("--min_axes", type=int, default=None,
                        help="Plot only models judged on at least this many axes. Default: the "
                             "headline axis count (the most any model has), so every plotted "
                             "model covers the same axes; --min_axes 1 shows everything.")
    parser.add_argument("--price_basis", choices=PRICE_BASES, default=DEFAULT_PRICE_BASIS,
                        help="How a point's cost is priced.  'batch' (default since 2026-11-03, the real "
                             "price): every cell recomputed from its token counts at the model's own batch "
                             "or Flex rate, whatever transport judged it; 'list': the same at the real-time "
                             "list price (the default before); 'recorded': usage.json's total_cost_usd as "
                             "billed, which mixes transports.")
    parser.add_argument("--robustness_json", type=str, default=None,
                        help="judge_tier_robustness.json: take each cell's rho from it (variant "
                             "--robustness_variant) instead of the cell's correlations.json.  Its rho is "
                             "recomputed from the stored scores with the nine trait/role collision names "
                             "dropped on bare-keyed axes, which the cells' correlations.json mismatch.")
    parser.add_argument("--robustness_variant", type=str, default="s6_l25_raw",
                        help="Variant key in --robustness_json (default s6_l25_raw: slot 6, layer 25, raw, "
                             "the cells' own projection).")
    parser.add_argument("--include_models", nargs="+", default=[], metavar="CELL",
                        help="Plot only these cell directory names (an allow-list; applied before --exclude_models).")
    parser.add_argument("--exclude_models", nargs="+", default=[], metavar="CELL",
                        help="Cell directory names to leave out, e.g. a run superseded by one under "
                             "consistent settings (claude-haiku-4-5, T=1, beside claude-haiku-4-5@t0); named in "
                             "the title.")
    parser.add_argument("--links", nargs="+", default=[], metavar="OLD:NEW",
                        help="Pairs of cell names joined by an arrow, e.g. gpt-6-luna:gpt-6-luna@c5 (a judge's "
                             "old-rubric point to its new-rubric point).")
    parser.add_argument("--links_label", default="same judge: old rubric → new rubric",
                        help="What the arrows mean, for the legend.")
    parser.add_argument("--axes_file", type=str, default=None,
                        help="With --experiments_dir: only the axes this file lists "
                             "(scripts/tier_eval_axes_20.tsv, for the 20-axis figures).")
    parser.add_argument("--cache-policy", dest="cache_policy", choices=CACHE_POLICIES, default="warn",
                        help="What to do when --robustness_json's recorded inputs have changed: strict | warn "
                             "(default) | rebuild | off.")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    links = []
    for item in args.links:
        if item.count(":") != 1:
            parser.error(f"--links {item!r}: expected OLD:NEW")
        links.append(tuple(item.split(":")))

    if args.experiments_dir:
        exp_dir = Path(args.experiments_dir)
        tier_eval_dirs = sorted(exp_dir.glob("*_tier_eval"))
        if args.axes_file:
            listed = {line.split()[0] for line in Path(args.axes_file).read_text().splitlines()
                      if line.strip() and not line.startswith("#")}
            tier_eval_dirs = [d for d in tier_eval_dirs if d.name[: -len("_tier_eval")] in listed]
    else:
        tier_eval_dirs = [Path(p) for p in args.tier_eval_dirs]

    if not tier_eval_dirs:
        print("ERROR: no *_tier_eval directories found. Run judge_tier_cost_eval.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tier_eval_dirs)} tier_eval dirs", file=sys.stderr)

    inputs: List[InputSpec] = [current_file_input("axes_file", Path(args.axes_file))] if args.axes_file else []
    per_model = collect_tier_eval_data(tier_eval_dirs, args.slot, args.price_basis)

    if args.legacy and not args.no_legacy:
        if args.legacy_dirs is not None:
            legacy_dirs = [Path(p) for p in args.legacy_dirs]
        elif args.experiments_dir:
            # Auto-detect: for each tier_eval dir, check if the base axis dir
            # has a gpt/correlations.json (meaning it has legacy judge data).
            exp_dir = Path(args.experiments_dir)
            legacy_dirs = []
            for td in tier_eval_dirs:
                base = exp_dir / td.name.replace("_tier_eval", "")
                if (base / "gpt" / "correlations.json").exists():
                    legacy_dirs.append(base)
        else:
            legacy_dirs = []

        if legacy_dirs:
            print(f"Loading legacy tiers from {len(legacy_dirs)} axis dirs", file=sys.stderr)
            legacy_data = collect_legacy_data(legacy_dirs, args.slot, args.price_basis)
            for model, entries in legacy_data.items():
                per_model[model].extend(entries)

    keep_models(per_model, args.include_models)
    left_out = drop_models(per_model, args.exclude_models)
    # Coverage is judged on the cells themselves: the robustness JSON holds only the models it compared.
    n_axes_raw = {m: len(v) for m, v in per_model.items() if v}
    n_total = max(n_axes_raw.values(), default=0)
    min_axes = args.min_axes if args.min_axes is not None else n_total
    excluded_raw = {m: {"n_axes": k} for m, k in n_axes_raw.items() if k < min_axes}

    if args.robustness_json:
        if not args.robustness_variant.startswith(f"s{args.slot}_"):
            print(f"WARNING: --robustness_variant {args.robustness_variant} is not slot {args.slot}",
                  file=sys.stderr)
        per_model = rho_from_robustness({m: v for m, v in per_model.items() if m not in excluded_raw},
                                        Path(args.robustness_json), args.robustness_variant,
                                        inputs=inputs, policy=args.cache_policy)

    if not per_model:
        print("ERROR: no data found at all.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(per_model)

    print(f"\nAggregated results ({args.slot=}):", file=sys.stderr)
    for model, d in sorted(agg.items(), key=lambda kv: -kv[1]["mean_rho"]):
        measured = "measured" if d["cost_is_measured"] else "estimated"
        print(f"  {model:30s}  mean_rho={d['mean_rho']:.4f}  "
              f"mean_cost=${d['mean_cost']:.2f}  n_axes={d['n_axes']}  cost={measured}",
              file=sys.stderr)

    agg, excluded = split_by_coverage(agg, min_axes)
    excluded = {**excluded_raw, **excluded}
    if excluded:
        print(f"Not plotted (judged on fewer than {min_axes} axes): "
              + ", ".join(f"{m} (n_axes={d['n_axes']})" for m, d in sorted(excluded.items())),
              file=sys.stderr)
    if not agg:
        print(f"ERROR: no model has {min_axes} axes; lower --min_axes.", file=sys.stderr)
        sys.exit(1)
    rho_note = (f"rho from {Path(args.robustness_json).name}, variant {args.robustness_variant} "
                f"(collision names dropped on bare-keyed axes)") if args.robustness_json else ""
    inputs.append(current_files_input("cell_files", sorted(READ_FILES)))
    plot(agg, args.slot, n_total, Path(args.output), excluded=excluded, price_basis=args.price_basis,
         rho_note=rho_note, left_out=left_out, links=links, links_label=args.links_label, inputs=inputs)


if __name__ == "__main__":
    main()
