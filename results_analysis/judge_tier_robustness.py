#!/usr/bin/env python3
"""Does the choice of LLM judge depend on how the projections are taken?

The judge-tier Pareto plot (:mod:`results_analysis.judge_tier_cost_plot_pooled`)
ranks judges by the mean over axes of the Spearman rho between a judge's
-3..+3 axis scores of the entity descriptions and the entities' activation
projections onto the axis.  Those projections are fixed at one choice: token
slot 6, layer 25, raw (no whitening), read from each axis's cached
``<axis>/gpt/projections.json``.  This tool recomputes every judge's rho on
every axis under a grid of projection variants and reports whether the
ranking of the judges and the Pareto frontier stay the same.  If the
variants disagree on which judge to pick, the comparison may be fitting the
one projection choice rather than judge quality.

Variants (default 12): slot in ``--slots`` (0, 6) x layer in ``--layers``
(25, 49) x regime in ``--regimes``:

- ``raw``        no whitening (key ``s6_l25_raw``);
- ``soft_K``     soft-K whitening at K = ``DEFAULT_SOFT_K`` (key
                 ``s0_l49_softK2``), fitted per axis on the augmented pool
                 (every trait and role vector plus ``default.pt``, the
                 axis's two poles held out of both kinds);
- ``soft_shear`` soft shear at L = ``DEFAULT_SOFT_SHEAR_L`` (key
                 ``s6_l25_shear3``), fitted per (slot, layer) on the
                 combined goal / no-goal residual subspaces.

K and L come from :mod:`results_analysis.canonical_angles.whitening`.  Layer
49 is the secondary peak of the rho-by-layer curve at slot 6
(``roger/axis_judge_experiments/rho_by_layer_simple_slot6_raw.json``).

Machinery reused from :mod:`results_analysis.rho_by_layer`:
``_build_pool_cache`` and ``_load_full_tensor`` (the activation slices; the
entity vectors are centred on ``default.pt`` exactly as
``_build_entity_cache`` centres them, from the same slices, so every vector
file is read once), ``_bases_for_pair`` / ``_all_K_bases`` (soft K),
``_all_L_bases`` (soft shear), ``_axis_unit`` and ``_project``.  The rho of a
cell is :func:`results_analysis.axis_judge_correlation.compute_correlations`
itself: the axis's two poles are excluded by bare name, and only entities
with a finite projection count.  Costs, the frontier rule and the coverage
rule are those of :mod:`results_analysis.judge_tier_cost_plot_pooled`.

Two choices that differ from what one might assume
--------------------------------------------------

- **The soft-K pool includes ``default.pt``**, as
  :func:`canonical_angles.data.build_augmented_whitening_pool` and
  ``whitening_k_sweep.py`` intend.  ``rho_by_layer._build_pool_cache`` keys
  the default as ``("traits", "default")`` while the pool builder asks for
  ``("combinations", "default")``, so ``rho_by_layer``'s own soft-K pool
  silently leaves the default out; this tool adds the combinations key.
- **Bare-name score keys are lifted with**
  :func:`assistant_axis.judge_loaders.migrate_v1_static_scores`, the
  project's consumer convention: a bare name that belongs to one kind gets
  that kind; the nine names that are both a trait and a role (``patient``,
  ``stoic``, ...) are dropped, because the v1 caches do not say which kind
  was scored.  About half the legacy cells, and the tier cells of axes whose
  cached ``projections.json`` is bare-keyed, are bare-keyed, so on those
  axes every judge is compared on the same 562 entities instead of the 571
  of the cached ``correlations.json``.  (The cached v1 projections hold the
  *trait* vector under a collision name, while
  ``judge_tier_cost_eval.resolve_entities`` judges the *role* description
  under it; dropping the nine avoids pairing them.)

Inputs
------

- ``--data_dir``: the activation vectors (``traits/vectors/*.pt``,
  ``roles/vectors/*.pt``, each ``{"vector": (8 slots, 64 layers, 5120)}``,
  bf16, cast to float32), ``combinations/vectors/default.pt`` and the
  goal / no-goal residual marginals under
  ``combinations/vectors/derived/``.
- ``--experiments_dir``: legacy cells
  ``<axis>/gpt/scores_descriptions.json`` (labelled ``gpt-4.1-mini (May)``) and
  ``<axis>/sonnet/scores_descriptions.json`` (``claude-sonnet-4 (May)``), whose
  cost is the pooled plotter's estimate; axes are the ``<pos>_vs_<neg>``
  directories that have ``gpt/scores_descriptions.json``, with poles and
  pair type from ``gpt/config.json`` (the record of how the cached axis was
  built), or ``--axes_file`` (``axis_dir pos neg pair_type`` per line, ``#``
  comments; e.g. ``scripts/tier_eval_axes.tsv``).
- ``--tier_eval_dir`` (default ``--experiments_dir``): new-tier cells
  ``<axis>_tier_eval/<model>/scores_descriptions.json`` with measured cost
  in ``usage.json`` beside it (a cell without ``usage.json`` is skipped and
  listed, as the pooled plotter does).

Coverage: only models with a cell on at least ``--min_axes`` axes enter the
comparison (default: the most any model has); the others are listed.

Outputs (to ``--output_dir``, default ``--experiments_dir``)
------------------------------------------------------------

- ``judge_tier_robustness.json`` (provenance envelope): per-cell rho
  ``rho[model][axis][variant]`` and ``n``; per variant and model the mean
  rho, quality 1/(1 - mean rho), number of axes, mean cost and frontier
  membership; the summary statistics (Kendall's tau of each variant's
  ranking against the default variant's; the frontier set of each variant
  and how often each model is on it; the cheapest frontier model, the best
  model under ``--price_band`` per axis and the best overall, per variant;
  a paired bootstrap over axes, per variant, of the probability that the
  default variant's top model and runner-up keep their order); and a check
  of the ``s6_l25_raw`` rho against each cell's cached ``correlations.json``.
- ``judge_tier_robustness.png``: a heatmap of mean rho (models by variants,
  frontier members boxed) above a grid of the Pareto plots, one per
  variant, rows sharing their y limits.
- ``judge_tier_robustness_cache/slices_s<slot>_l<layer>.npz`` (or
  ``--cache_dir``): the float32 activation slices, the two defaults and the
  goal / no-goal subspaces per (slot, layer), keyed by a fingerprint of the
  vector files (names, sizes, mtimes) and rebuilt when it changes.  The first
  run reads every vector file once (about 3 GB); later runs read only the
  cache.  ``--no_cache`` skips it.

No API calls are made.

Usage
-----

::

    # All 61 axes, default 12 variants, outputs next to the experiments:
    uv run python results_analysis/judge_tier_robustness.py

    # Explicit inputs and a scratch output directory:
    uv run python results_analysis/judge_tier_robustness.py \\
        --data_dir 'runpod_workspace/qwen/qwen-3-32b Roger 8slot' \\
        --experiments_dir roger/axis_judge_experiments \\
        --axes_file scripts/tier_eval_axes.tsv \\
        --output_dir /tmp/robustness

    # Only the axes the new tiers were judged on, every model with 2 axes:
    uv run python results_analysis/judge_tier_robustness.py \\
        --axes angel_vs_demon decisive_vs_indecisive --min_axes 2

    # A smaller grid (variant keys follow the same pattern):
    uv run python results_analysis/judge_tier_robustness.py \\
        --slots 6 --layers 25 --regimes raw soft_shear
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
# The project's .venv puts another checkout on sys.path through a .pth file;
# make this tree's packages win, as judge_tier_cost_plot_pooled.py does.
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from assistant_axis import entity_id, json_metadata, png_metadata  # noqa: E402
from assistant_axis.entity_id import parse_entity_id  # noqa: E402
from assistant_axis.judge_loaders import migrate_v1_static_scores  # noqa: E402
from assistant_axis.judge_pricing import (  # noqa: E402
    GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT, SONNET_RATE_IN, SONNET_RATE_OUT,
)
from assistant_axis.provenance import (  # noqa: E402
    CACHE_POLICIES,
    InputSpec,
    current_data_subtree_input,
    current_file_input,
    load_and_register,
)
from results_analysis import rho_by_layer as rbl  # noqa: E402
from results_analysis.axis_judge_correlation import compute_correlations  # noqa: E402
from results_analysis.canonical_angles.data import (  # noqa: E402
    build_augmented_whitening_pool,
    build_goal_nogoal_subspaces,
)
from results_analysis.canonical_angles.whitening import (  # noqa: E402
    DEFAULT_SOFT_K,
    DEFAULT_SOFT_SHEAR_L,
    WhiteningBasis,
)
from results_analysis.judge_tier_cost_plot_pooled import (  # noqa: E402
    cost_ticks as _cost_ticks,
    display_name,
    label_key,
    place_labels as _place_labels,
    DEFAULT_PRICE_BASIS,
    PRICE_BASES,
    PRICE_BASIS_LABEL,
    PROVIDER_COLOR,
    cell_max_tokens,
    estimate_cost,
    load_measured_cost,
    pareto_frontier,
    price_factor,
    provider_of,
)

DEFAULT_DATA_DIR = _REPO_ROOT / "runpod_workspace/qwen/qwen-3-32b Roger 8slot"
DEFAULT_EXPERIMENTS_DIR = _REPO_ROOT / "roger/axis_judge_experiments"
DEFAULT_SLOTS = [0, 6]
DEFAULT_LAYERS = [25, 49]
REGIMES = ("raw", "soft_K", "soft_shear")
DEFAULT_VARIANT = "s6_l25_raw"   # Konstantinos' projections: slot 6, layer 25, raw
#: USD per axis at DEFAULT_PRICE_BASIS: $0.25 at list price until 2026-11-03, halved with the move to the
#: real price (every judge here half price), so that the band holds the same judges.
DEFAULT_PRICE_BAND = 0.125
DEFAULT_BOOTSTRAP_DRAWS = 2000
DEFAULT_BOOTSTRAP_SEED = 20261101
TIER_EVAL_SUFFIX = "_tier_eval"
OUTPUT_STEM = "judge_tier_robustness"
CACHE_VERSION = "v1"
# (label, subdir, real-time input rate, output rate): the two judges whose
# caches predate usage.json, as judge_tier_cost_plot_pooled.collect_legacy_data.
LEGACY_CELLS = (
    # "(May)": the May 2026 cells (T=0, max_tokens 1024), kept apart from a tier-eval cell of the
    # same model (gpt-4.1-mini at T=0 and max_tokens 200, <axis>_tier_eval/gpt-4.1-mini/).
    ("gpt-4.1-mini (May)", "gpt", GPT_MINI_RATE_IN, GPT_MINI_RATE_OUT),
    ("claude-sonnet-4 (May)", "sonnet", SONNET_RATE_IN, SONNET_RATE_OUT),
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    slot: int
    layer: int
    regime: str          # "raw" | "soft_K" | "soft_shear"
    K: int = DEFAULT_SOFT_K
    L: int = DEFAULT_SOFT_SHEAR_L

    @property
    def key(self) -> str:
        return variant_key(self.slot, self.layer, self.regime, K=self.K, L=self.L)


def variant_key(slot: int, layer: int, regime: str, *,
                K: int = DEFAULT_SOFT_K, L: int = DEFAULT_SOFT_SHEAR_L) -> str:
    """``s6_l25_raw``, ``s0_l49_softK2``, ``s6_l25_shear3``."""
    if regime == "raw":
        tail = "raw"
    elif regime == "soft_K":
        tail = f"softK{K}"
    elif regime == "soft_shear":
        tail = f"shear{L}"
    else:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    return f"s{slot}_l{layer}_{tail}"


def build_variants(slots: Sequence[int], layers: Sequence[int], regimes: Sequence[str], *,
                   K: int = DEFAULT_SOFT_K, L: int = DEFAULT_SOFT_SHEAR_L) -> List[Variant]:
    """Slot-major, then layer, then regime in the order given."""
    for r in regimes:
        if r not in REGIMES:
            raise ValueError(f"unknown regime {r!r}; expected one of {REGIMES}")
    return [Variant(s, l, r, K=K, L=L) for s in slots for l in layers for r in regimes]


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisDef:
    axis_dir: str
    pos: str
    neg: str
    pair_type: str   # "roles" | "traits"


def read_axes_tsv(path: Path) -> List[AxisDef]:
    """Rows ``axis_dir pos neg pair_type`` (tabs or spaces), ``#`` comments."""
    axes: List[AxisDef] = []
    seen = set()
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"{path}:{lineno}: expected 'axis_dir pos neg pair_type', got {line!r}")
        axis_dir, pos, neg, pair_type = parts
        if pair_type not in ("roles", "traits"):
            raise ValueError(f"{path}:{lineno}: pair_type {pair_type!r} is not 'roles' or 'traits'")
        if axis_dir in seen:
            raise ValueError(f"{path}:{lineno}: axis {axis_dir!r} listed twice")
        seen.add(axis_dir)
        axes.append(AxisDef(axis_dir, pos, neg, pair_type))
    return axes


def vector_kinds(data_dir: Path) -> Dict[str, set]:
    """``{bare name: {"traits", "roles"} subset}`` from the vector files, as
    ``rho_by_layer._build_entity_cache`` builds ``kinds_for_name``."""
    kinds: Dict[str, set] = {}
    for et in ("traits", "roles"):
        for fp in sorted((Path(data_dir) / et / "vectors").glob("*.pt")):
            if fp.stem != "default":
                kinds.setdefault(fp.stem, set()).add(et)
    return kinds


def _read_json_payload(path: Path):
    with open(path) as f:
        return _unwrap(json.load(f))


def _unwrap(obj):
    """``{"result": X, ...}`` (``_provenance`` and ``schema_version``
    optional) to X.  ``load_and_register`` unwraps only envelopes that carry
    ``_provenance``; some judge caches have a bare ``{"result": ...}``."""
    if isinstance(obj, dict) and "result" in obj and set(obj) <= {"result", "_provenance", "schema_version"}:
        return obj["result"]
    return obj


def discover_axes(experiments_dir: Path, kinds_for_name: Dict[str, set]) -> List[AxisDef]:
    """Every ``<pos>_vs_<neg>`` directory with ``gpt/scores_descriptions.json``.

    Poles and pair type come from ``gpt/config.json`` (``args.pair``,
    ``args.pair_type``: how the cached axis was built); without it, the
    poles are the name split on ``_vs_`` and the pair type the one kind
    both poles belong to.
    """
    axes: List[AxisDef] = []
    for d in sorted(Path(experiments_dir).iterdir()):
        if not (d / "gpt" / "scores_descriptions.json").exists() or "_vs_" not in d.name:
            continue
        pos, neg = d.name.split("_vs_", 1)
        pair_type = None
        cfg = d / "gpt" / "config.json"
        if cfg.exists():
            args = (_read_json_payload(cfg) or {}).get("args", {})
            pair = args.get("pair")
            if pair and len(pair) == 2:
                if f"{pair[0]}_vs_{pair[1]}" != d.name:
                    raise ValueError(f"{cfg}: pair {pair} does not match the directory name {d.name!r}")
                pos, neg = pair
            pair_type = args.get("pair_type")
        if pair_type is None:
            common = kinds_for_name.get(pos, set()) & kinds_for_name.get(neg, set())
            if len(common) != 1:
                raise ValueError(f"{d.name}: cannot tell the pair type (poles share kinds {sorted(common)}); "
                                 f"pass --axes_file")
            pair_type = next(iter(common))
        axes.append(AxisDef(d.name, pos, neg, pair_type))
    return axes


# ---------------------------------------------------------------------------
# Judge cells
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    model: str
    axis_dir: str
    scores: Dict[str, float]      # entity_id -> score, lifted (see lift_scores)
    n_scored: int                 # numeric, non-pole entries in the file (before lifting)
    cost: float
    cost_is_measured: bool
    scores_path: Path


def numeric_scores(raw: dict) -> Dict[str, float]:
    """Keep the numeric, finite values (gaps are ``None``)."""
    out: Dict[str, float] = {}
    for k, v in (raw or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if not math.isfinite(float(v)):
            continue
        out[k] = float(v)
    return out


def lift_scores(scores: Dict[str, float], kinds_for_name: Dict[str, set]) -> Dict[str, float]:
    """Bare names to ``name|R`` / ``name|T``; collisions and orphans dropped."""
    return migrate_v1_static_scores(scores, kinds_for_name)


def _bare(key: str) -> str:
    try:
        return parse_entity_id(key).name
    except ValueError:
        return key


def load_cells(axes: Sequence[AxisDef], experiments_dir: Path, tier_eval_dir: Path,
               kinds_for_name: Dict[str, set], price_basis: str = DEFAULT_PRICE_BASIS,
               inputs: Optional[list] = None,
               include_legacy: bool = True, policy: str = "warn") -> Tuple[Dict[str, Dict[str, Cell]], List[dict]]:
    """``({model: {axis_dir: Cell}}, skipped)``.  Tier cells first; a legacy
    cell fills an axis only where no tier cell of the same label exists."""
    cells: Dict[str, Dict[str, Cell]] = {}
    skipped: List[dict] = []

    def _load(path: Path, dep_key: str) -> dict:
        payload, _spec, _check = load_and_register(path, dep_key=dep_key, inputs=inputs, policy=policy)
        return _unwrap(payload)

    for ax in axes:
        poles = {ax.pos, ax.neg}
        tier_dir = Path(tier_eval_dir) / f"{ax.axis_dir}{TIER_EVAL_SUFFIX}"
        if tier_dir.is_dir():
            for model_dir in sorted(p for p in tier_dir.iterdir() if p.is_dir()):
                model = model_dir.name
                sp = model_dir / "scores_descriptions.json"
                up = model_dir / "usage.json"
                if not sp.exists():
                    continue
                if not up.exists():
                    skipped.append({"model": model, "axis": ax.axis_dir, "reason": "no usage.json"})
                    continue
                cost = load_measured_cost(up, price_basis)
                if cost is None:
                    skipped.append({"model": model, "axis": ax.axis_dir, "reason": "no cost in usage.json"})
                    continue
                if inputs is not None:
                    inputs.append(current_file_input(f"usage_{ax.axis_dir}_{model}", up))
                raw = numeric_scores(_load(sp, f"judge_{ax.axis_dir}_descriptions_{model}"))
                cells.setdefault(model, {})[ax.axis_dir] = Cell(
                    model, ax.axis_dir, lift_scores(raw, kinds_for_name),
                    sum(1 for k in raw if _bare(k) not in poles), float(cost), True, sp)
        for model, sub, rin, rout in (LEGACY_CELLS if include_legacy else ()):
            sp = Path(experiments_dir) / ax.axis_dir / sub / "scores_descriptions.json"
            if not sp.exists() or ax.axis_dir in cells.get(model, {}):
                continue
            raw = numeric_scores(_load(sp, f"judge_{ax.axis_dir}_descriptions_{sub}"))
            n_scored = sum(1 for k in raw if _bare(k) not in poles)
            cells.setdefault(model, {})[ax.axis_dir] = Cell(
                model, ax.axis_dir, lift_scores(raw, kinds_for_name), n_scored,
                estimate_cost(n_scored, rin, rout) * price_factor(model.split(" (")[0], price_basis), False, sp)
    return cells, skipped


def apply_coverage(cells: Dict[str, Dict[str, Cell]], min_axes: Optional[int] = None
                   ) -> Tuple[List[str], Dict[str, int], int]:
    """``(kept models, {excluded model: n_axes}, min_axes used)``.  Default
    bar: the most axes any model has (the pooled plotter's rule)."""
    counts = {m: len(v) for m, v in cells.items()}
    if not counts:
        return [], {}, 0
    bar = max(counts.values()) if min_axes is None else int(min_axes)
    kept = sorted(m for m, n in counts.items() if n >= bar)
    excluded = {m: n for m, n in sorted(counts.items()) if n < bar}
    return kept, excluded, bar


# ---------------------------------------------------------------------------
# Activation slices (cached per slot and layer)
# ---------------------------------------------------------------------------

@dataclass
class SliceSet:
    """Float32 activations at one (slot, layer), not default-centred."""
    slot: int
    layer: int
    keys: List[Tuple[str, str]]          # (etype, stem) per row of ``raw``
    raw: np.ndarray                      # (n, D)
    default: np.ndarray                  # (D,) traits/vectors/default.pt
    comb_default: np.ndarray             # (D,) combinations/vectors/default.pt
    A_goal: Optional[np.ndarray] = None  # (D, n_g) for soft shear
    A_nogoal: Optional[np.ndarray] = None

    def entity_vecs(self) -> Dict[str, np.ndarray]:
        """``{entity_id: (1, 1, D)}``, centred as ``_build_entity_cache`` does."""
        return {entity_id(stem, et): (row - self.default)[None, None, :]
                for (et, stem), row in zip(self.keys, self.raw)}

    def pool_cache(self) -> Dict[Tuple[str, str], np.ndarray]:
        """``_build_pool_cache``'s layout at this (slot, layer), plus the
        ``("combinations", "default")`` entry the augmented pool asks for."""
        cache = {k: row[None, None, :] for k, row in zip(self.keys, self.raw)}
        cache[("traits", "default")] = self.default[None, None, :]
        cache[("combinations", "default")] = self.comb_default[None, None, :]
        return cache


def _fingerprint_files(data_dir: Path) -> List[Path]:
    d = Path(data_dir)
    files = sorted((d / "traits" / "vectors").glob("*.pt")) + sorted((d / "roles" / "vectors").glob("*.pt"))
    files.append(d / "combinations" / "vectors" / "default.pt")
    derived = d / "combinations" / "vectors" / "derived"
    files += sorted(derived.glob("marginals/*/*.pt")) + sorted(derived.glob("axis/*.pt"))
    return files


def data_fingerprint(data_dir: Path) -> str:
    """SHA-256 over (relative path, size, whole-second mtime) of every file
    the slices are built from."""
    h = hashlib.sha256()
    for fp in _fingerprint_files(data_dir):
        if not fp.exists():
            h.update(f"{fp.relative_to(data_dir)}|missing\n".encode())
            continue
        st = fp.stat()
        h.update(f"{fp.relative_to(data_dir)}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return h.hexdigest()


@contextlib.contextmanager
def _rbl_slots(slots: Sequence[int]):
    """``rho_by_layer._load_full_tensor`` keeps the slots in its module-level
    ``SLOTS``; point it at the requested slots for the duration."""
    old = rbl.SLOTS
    rbl.SLOTS = list(slots)
    try:
        yield
    finally:
        rbl.SLOTS = old


def build_slices(data_dir: Path, slots: Sequence[int], layers: Sequence[int], *,
                 with_shear: bool = True, ca_kind: str = "combined") -> Dict[Tuple[int, int], SliceSet]:
    """Read every vector file once (``rho_by_layer._build_pool_cache``) and
    keep the requested (slot, layer) slices."""
    data_dir = Path(data_dir)
    slots = list(dict.fromkeys(slots))
    with _rbl_slots(slots):
        default_arr = rbl._load_full_tensor(data_dir / "traits" / "vectors" / "default.pt")
        pool = rbl._build_pool_cache(data_dir, default_arr)
        comb_path = data_dir / "combinations" / "vectors" / "default.pt"
        comb_arr = rbl._load_full_tensor(comb_path) if comb_path.exists() else default_arr
    keys = sorted(k for k in pool if k[1] != "default")
    out: Dict[Tuple[int, int], SliceSet] = {}
    for si, slot in enumerate(slots):
        for layer in layers:
            raw = np.stack([pool[k][si, layer] for k in keys], axis=0).astype(np.float32, copy=False)
            ss = SliceSet(slot, layer, keys, np.ascontiguousarray(raw),
                          np.array(default_arr[si, layer], dtype=np.float32),
                          np.array(comb_arr[si, layer], dtype=np.float32))
            if with_shear:
                ss.A_goal, ss.A_nogoal = build_goal_nogoal_subspaces(
                    data_dir, slot=slot, layer=layer, kind=ca_kind)
            out[(slot, layer)] = ss
    del pool
    return out


def _cache_path(cache_dir: Path, slot: int, layer: int) -> Path:
    return Path(cache_dir) / f"slices_s{slot}_l{layer}.npz"


def save_slices(cache_dir: Path, slices: Dict[Tuple[int, int], SliceSet], meta: dict) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for (slot, layer), ss in slices.items():
        m = dict(meta, slot=slot, layer=layer, has_shear=ss.A_goal is not None)
        arrays = dict(keys=np.array([f"{et}/{stem}" for et, stem in ss.keys]), raw=ss.raw,
                      default=ss.default, comb_default=ss.comb_default, meta=np.array(json.dumps(m)))
        if ss.A_goal is not None:
            arrays.update(A_goal=ss.A_goal, A_nogoal=ss.A_nogoal)
        path = _cache_path(cache_dir, slot, layer)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(path)


def load_slices(cache_dir: Path, slots: Sequence[int], layers: Sequence[int], meta: dict,
                need_shear: bool) -> Optional[Dict[Tuple[int, int], SliceSet]]:
    """The cached slices, or None if any is missing or was built from other files."""
    out: Dict[Tuple[int, int], SliceSet] = {}
    for slot in slots:
        for layer in layers:
            path = _cache_path(cache_dir, slot, layer)
            if not path.exists():
                return None
            with np.load(path, allow_pickle=False) as z:
                m = json.loads(str(z["meta"]))
                if any(m.get(k) != v for k, v in meta.items()) or (need_shear and not m.get("has_shear")):
                    return None
                keys = [tuple(s.split("/", 1)) for s in z["keys"].tolist()]
                ss = SliceSet(slot, layer, keys, z["raw"], z["default"], z["comb_default"])
                if m.get("has_shear"):
                    ss.A_goal, ss.A_nogoal = z["A_goal"], z["A_nogoal"]
            out[(slot, layer)] = ss
    return out


# ---------------------------------------------------------------------------
# Projections and rho
# ---------------------------------------------------------------------------

def project_variants(slices: Dict[Tuple[int, int], SliceSet], axes: Sequence[AxisDef],
                     variants: Sequence[Variant], names_by_axis: Dict[str, Iterable[str]],
                     data_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """``{axis_dir: {variant key: {entity_id: projection}}}``."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {ax.axis_dir: {} for ax in axes}
    pool_entries: Dict[str, list] = {}
    by_sl: Dict[Tuple[int, int], List[Variant]] = {}
    for v in variants:
        by_sl.setdefault((v.slot, v.layer), []).append(v)
    for (slot, layer), vs in by_sl.items():
        ss = slices[(slot, layer)]
        ev = ss.entity_vecs()
        pc = ss.pool_cache() if any(v.regime == "soft_K" for v in vs) else None
        shear: Dict[int, WhiteningBasis] = {}
        for v in vs:
            if v.regime == "soft_shear" and v.L not in shear:
                if ss.A_goal is None:
                    raise ValueError(f"slot {slot} layer {layer}: no goal/no-goal subspaces for soft shear")
                shear.update(rbl._all_L_bases(ss.A_goal, ss.A_nogoal, [v.L]))
        for ax in axes:
            pos_eid, neg_eid = entity_id(ax.pos, ax.pair_type), entity_id(ax.neg, ax.pair_type)
            for eid in (pos_eid, neg_eid):
                if eid not in ev:
                    raise KeyError(f"{ax.axis_dir}: no vector for pole {eid} under {data_dir}")
            unit = rbl._axis_unit(ev, pos_eid, neg_eid, 0, 0)
            names = sorted(set(names_by_axis.get(ax.axis_dir, ())))
            for v in vs:
                if v.regime == "raw":
                    basis = WhiteningBasis(method="raw")
                elif v.regime == "soft_K":
                    if ax.axis_dir not in pool_entries:
                        leave_out = {(et, n) for n in (ax.pos, ax.neg) for et in ("traits", "roles")}
                        pool_entries[ax.axis_dir] = build_augmented_whitening_pool(
                            Path(data_dir), leave_out=leave_out, scope="roles+traits")
                    entries = pool_entries[ax.axis_dir]
                    missing = [e for e in entries if e not in pc]
                    if missing:
                        raise ValueError(f"{ax.axis_dir}: {len(missing)} pool entries have no cached slice "
                                         f"(e.g. {missing[:3]}); delete the cache and rerun")
                    basis = rbl._bases_for_pair(pc, entries, 0, 0, [v.K])[v.K]
                else:
                    basis = shear[v.L]
                out[ax.axis_dir][v.key] = rbl._project(ev, basis, 0, 0, unit, names)
    return out


def cell_rho(scores: Dict[str, float], proj: Dict[str, float], poles: Iterable[str]) -> Tuple[float, int]:
    """Spearman rho and n via ``compute_correlations`` (poles excluded by bare
    name, non-finite projections skipped, NaN below 3 pairs)."""
    per_slot = {k: {"raw": p, "whitened": p} for k, p in proj.items()}
    corr = compute_correlations({"descriptions": scores}, {0: per_slot}, [0], set(poles))
    cell = corr["descriptions"]["0"]["raw"]
    return float(cell["rho"]), int(cell["n"])


def compute_rho_table(cells: Dict[str, Dict[str, Cell]], models: Sequence[str], axes: Sequence[AxisDef],
                      projections: Dict[str, Dict[str, Dict[str, float]]], variants: Sequence[Variant]
                      ) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, int]]]:
    """``(rho[model][axis][variant], n[model][axis])``; n is the same for
    every variant (projections are finite under all of them) and is taken
    from the first."""
    rho: Dict[str, Dict[str, Dict[str, float]]] = {}
    n: Dict[str, Dict[str, int]] = {}
    for m in models:
        rho[m], n[m] = {}, {}
        for ax in axes:
            c = cells[m].get(ax.axis_dir)
            if c is None:
                continue
            rho[m][ax.axis_dir] = {}
            for i, v in enumerate(variants):
                r, k = cell_rho(c.scores, projections[ax.axis_dir][v.key], (ax.pos, ax.neg))
                rho[m][ax.axis_dir][v.key] = r
                if i == 0:
                    n[m][ax.axis_dir] = k
    return rho, n


# ---------------------------------------------------------------------------
# Aggregation and summary statistics
# ---------------------------------------------------------------------------

def summarise_variant(rho: Dict[str, Dict[str, Dict[str, float]]], costs: Dict[str, float],
                      models: Sequence[str], vkey: str) -> Tuple[Dict[str, dict], List[str]]:
    """Per model: mean rho over its axes, quality 1/(1 - mean rho), n axes,
    mean cost, frontier membership; and the frontier (cheapest first)."""
    per: Dict[str, dict] = {}
    for m in models:
        vals = [r[vkey] for r in rho[m].values() if vkey in r and math.isfinite(r[vkey])]
        mean = float(np.mean(vals)) if vals else float("nan")
        per[m] = {"mean_rho": mean, "quality": 1.0 / (1.0 - mean) if mean < 1 else float("inf"),
                  "n_axes": len(vals), "mean_cost": costs[m]}
    ok = [m for m in models if math.isfinite(per[m]["mean_rho"])]
    idx = pareto_frontier([(per[m]["mean_cost"], per[m]["quality"]) for m in ok])
    frontier = [ok[i] for i in idx]
    for m in models:
        per[m]["on_frontier"] = m in frontier
    return per, frontier


def ranking(per: Dict[str, dict], models: Sequence[str]) -> List[str]:
    """Models by mean rho, best first (ties by name)."""
    return sorted(models, key=lambda m: (-per[m]["mean_rho"], m))


def kendall_tau_vs_default(per_variant: Dict[str, Dict[str, dict]], models: Sequence[str],
                           default_key: str) -> Dict[str, Optional[float]]:
    """Kendall's tau (tau-b) between each variant's mean rho over the models
    and the default variant's; None with fewer than two models."""
    out: Dict[str, Optional[float]] = {}
    base = [per_variant[default_key][m]["mean_rho"] for m in models]
    for vkey, per in per_variant.items():
        if len(models) < 2:
            out[vkey] = None
            continue
        tau = kendalltau(base, [per[m]["mean_rho"] for m in models]).statistic
        out[vkey] = None if tau is None or not math.isfinite(tau) else float(tau)
    return out


def frontier_counts(frontiers: Dict[str, List[str]], models: Sequence[str]) -> Dict[str, int]:
    return {m: sum(m in f for f in frontiers.values()) for m in models}


def price_bands(per: Dict[str, dict], frontier: Sequence[str], models: Sequence[str],
                band: float) -> Dict[str, Optional[str]]:
    """The cheapest frontier model, the best model costing less than ``band``
    per axis (None if none does), and the best overall."""
    def best(ms):
        ms = [m for m in ms if math.isfinite(per[m]["mean_rho"])]
        return max(ms, key=lambda m: (per[m]["mean_rho"], -per[m]["mean_cost"])) if ms else None
    return {"cheapest_frontier": frontier[0] if frontier else None,
            "best_under_band": best([m for m in models if per[m]["mean_cost"] < band]),
            "best_overall": best(models)}


def bootstrap_keep_order(rho: Dict[str, Dict[str, Dict[str, float]]], top: str, runner_up: str,
                         vkeys: Sequence[str], draws: int = DEFAULT_BOOTSTRAP_DRAWS,
                         seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    """Paired bootstrap over axes: resample the axes both models were judged
    on, with replacement (``draws`` times, one fixed set of draws shared by
    every variant), and give per variant the share of draws in which
    ``top``'s mean rho exceeds ``runner_up``'s (ties count half)."""
    axes = sorted(a for a in rho[top] if a in rho[runner_up]
                  and all(math.isfinite(rho[top][a].get(v, float("nan")))
                          and math.isfinite(rho[runner_up][a].get(v, float("nan"))) for v in vkeys))
    n = len(axes)
    res = {"top": top, "runner_up": runner_up, "n_axes": n, "draws": draws, "seed": seed,
           "p_keep_order": {}}
    if n == 0:
        res["p_keep_order"] = {v: None for v in vkeys}
        return res
    idx = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    for v in vkeys:
        a = np.array([rho[top][x][v] for x in axes])
        b = np.array([rho[runner_up][x][v] for x in axes])
        d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        res["p_keep_order"][v] = float(np.mean(d > 0) + 0.5 * np.mean(d == 0))
    return res


def check_vs_cached(cells: Dict[str, Dict[str, Cell]], models: Sequence[str],
                    rho: Dict[str, Dict[str, Dict[str, float]]], n: Dict[str, Dict[str, int]],
                    vkey: str = "s6_l25_raw", inputs: Optional[list] = None, policy: str = "warn") -> dict:
    """Compare the ``s6_l25_raw`` rho with each cell's cached
    ``correlations.json`` (slot 6, raw, layer 25 from projections.json)."""
    rows = []
    for m in models:
        for axis, c in sorted(cells[m].items()):
            if vkey not in rho[m].get(axis, {}):
                continue
            cp = c.scores_path.parent / "correlations.json"
            cached = None
            if cp.exists():
                payload, _s, _c = load_and_register(cp, dep_key=f"corr_{axis}_{m}", inputs=inputs, policy=policy)
                try:
                    cell = _unwrap(payload)["descriptions"]["6"]["raw"]
                    cached = (float(cell["rho"]), int(cell["n"]))
                except (KeyError, TypeError, ValueError):
                    cached = None
            row = {"model": m, "axis": axis, "rho": rho[m][axis][vkey], "n": n[m][axis],
                   "cached_rho": cached[0] if cached else None, "cached_n": cached[1] if cached else None}
            row["diff"] = (row["rho"] - row["cached_rho"]) if cached else None
            rows.append(row)
    same_n = [r for r in rows if r["cached_n"] is not None and r["cached_n"] == r["n"]]
    other_n = [r for r in rows if r["cached_n"] is not None and r["cached_n"] != r["n"]]
    return {
        "variant": vkey,
        "n_cells": len(rows),
        "max_abs_diff_same_n": max((abs(r["diff"]) for r in same_n), default=None),
        "n_cells_same_n": len(same_n),
        "max_abs_diff_other_n": max((abs(r["diff"]) for r in other_n), default=None),
        "cells_other_n": [{k: r[k] for k in ("model", "axis", "n", "cached_n", "diff")} for r in other_n],
        "cells_without_cached_rho": [{"model": r["model"], "axis": r["axis"]} for r in rows
                                     if r["cached_rho"] is None],
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _short(model: str) -> str:
    """The panel label: the short display name (tokens as they are, no model notes) without the
    provider prefix."""
    model = display_name(model, short=True)
    for p in ("claude-", "gpt-"):
        if model.startswith(p):
            return model[len(p):]
    return model


def _usd(x: float) -> str:
    """Dollar amount with the sign escaped from mathtext."""
    return f"\\${x:.2f}" if x >= 0.1 else f"\\${x:.3f}"


@dataclass(frozen=True)
class FigureOptions:
    """For figures that compare settings rather than judges (scripts/judge_tier_grid_12way.py)."""
    headline: Optional[str] = None                     # replaces the "Judge ranking ..." title
    cell_labels: Optional[Dict[str, str]] = None       # cell -> text, for the heatmap rows and the points
    point_labels: Optional[Dict[str, str]] = None      # cell -> shorter text for the points, where it differs
    series: Sequence[Sequence[str]] = ()               # arrows through each list of cells, in order
    label_frontier_only: bool = False                  # leave the other points unlabelled (crowded panels)
    own_y: bool = False                                # each panel its own y limits, not its row's
    x_log: Optional[bool] = None                       # None: log when the costs span more than 8x
    x_ticks: Optional[Sequence[float]] = None          # None: cost_ticks
    x_log_pad: float = 1.6                             # a log axis runs from the lowest cost over this to the highest times it
    panel_w: float = 4.2
    panel_h: float = 3.3


def render_figure(out_png: Path, models_by_rho: Sequence[str], variants: Sequence[Variant],
                  per_variant: Dict[str, Dict[str, dict]], frontiers: Dict[str, List[str]],
                  measured: Dict[str, bool], default_key: str, n_axes_total: int,
                  subtitle: str, inputs: Optional[list], options: FigureOptions = FigureOptions()) -> None:
    """The heatmap above the Pareto panels."""
    opts = options
    vkeys = [v.key for v in variants]
    n_m, n_v = len(models_by_rho), len(vkeys)
    sl_pairs = list(dict.fromkeys((v.slot, v.layer) for v in variants))
    regimes = list(dict.fromkeys(v.regime for v in variants))
    n_rows, n_cols = len(regimes), len(sl_pairs)

    # Layout in inches, top to bottom.
    n_title_lines = 2 + subtitle.count("\n")
    top_in, heat_h, mid_in, row_gap, bot_in = (0.45 + 0.25 * n_title_lines, 0.5 + 0.38 * n_m, 1.55, 0.75, 0.6)
    panel_h = opts.panel_h
    fig_w = max(16.0, opts.panel_w * n_cols + 1.5)
    fig_h = top_in + heat_h + mid_in + n_rows * panel_h + (n_rows - 1) * row_gap + bot_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    heat_top = 1 - top_in / fig_h
    heat_bot = heat_top - heat_h / fig_h
    pan_top = heat_bot - mid_in / fig_h
    gs_h = fig.add_gridspec(1, 1, top=heat_top, bottom=heat_bot, left=0.13, right=0.97)
    gs_p = fig.add_gridspec(n_rows, n_cols, top=pan_top, bottom=bot_in / fig_h, left=0.05, right=0.975,
                            hspace=row_gap / panel_h, wspace=0.2)

    # (a) heatmap
    ax_h = fig.add_subplot(gs_h[0, 0])
    mat = np.array([[per_variant[k][m]["mean_rho"] for k in vkeys] for m in models_by_rho])
    finite = mat[np.isfinite(mat)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if vmax - vmin < 1e-6:
        vmin, vmax = vmin - 0.01, vmax + 0.01
    im = ax_h.imshow(mat, aspect="auto", cmap="Blues", vmin=vmin - 0.15 * (vmax - vmin), vmax=vmax)
    for i, m in enumerate(models_by_rho):
        for j, k in enumerate(vkeys):
            val = mat[i, j]
            dark = math.isfinite(val) and (val - vmin) / (vmax - vmin) > 0.6
            ax_h.text(j, i, f"{val:.3f}" if math.isfinite(val) else "n/a", ha="center", va="center",
                      fontsize=9, color="white" if dark else "#222222")
            if m in frontiers[k]:
                ax_h.add_patch(Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, fill=False,
                                         edgecolor="#d1495b", linewidth=2.0))
    ax_h.set_xticks(range(n_v))
    ax_h.set_xticklabels(vkeys, rotation=25, ha="right", fontsize=9)
    for lbl, k in zip(ax_h.get_xticklabels(), vkeys):
        if k == default_key:
            lbl.set_fontweight("bold")
    cost0 = per_variant[vkeys[0]]
    ax_h.set_yticks(range(n_m))
    ax_h.set_yticklabels([f"{opts.cell_labels[m] if opts.cell_labels else display_name(m)}  "
                          f"({_usd(cost0[m]['mean_cost'])}/axis{'' if measured[m] else ' est.'})"
                          for m in models_by_rho], fontsize=9)
    for j in range(1, n_v):
        if variants[j].layer != variants[j - 1].layer or variants[j].slot != variants[j - 1].slot:
            ax_h.axvline(j - 0.5, color="white", linewidth=3)
    ax_h.set_title(f"(a) Mean Spearman rho over axes by projection variant; rows ordered by "
                   f"{default_key} (bold); red box = on that variant's Pareto frontier",
                   fontsize=10, loc="left")
    fig.colorbar(im, ax=ax_h, shrink=0.9, pad=0.01, fraction=0.03, label="mean rho")

    # (b) Pareto small multiples
    all_costs = [per_variant[vkeys[0]][m]["mean_cost"] for m in models_by_rho]
    logx = opts.x_log if opts.x_log is not None else (min(all_costs) > 0 and max(all_costs) / min(all_costs) > 8)
    if logx:
        x_lo, x_hi = min(all_costs) / opts.x_log_pad, max(all_costs) * opts.x_log_pad
    else:
        span = max(all_costs) - min(all_costs) or max(all_costs) or 1.0
        x_lo, x_hi = min(all_costs) - 0.15 * span, max(all_costs) + 0.15 * span
    by_key = {v.key: v for v in variants}

    def y_limits(keys: Sequence[str]) -> Tuple[float, float]:
        qs = [per_variant[k][m]["quality"] for k in keys if k in by_key for m in models_by_rho]
        qs = [q for q in qs if math.isfinite(q)]
        y_lo, y_hi = (min(qs), max(qs)) if qs else (1.0, 2.0)
        pad = 0.14 * (y_hi - y_lo or 0.1)
        return y_lo - pad, y_hi + pad

    for r, regime in enumerate(regimes):
        row_keys = [variant_key(s, l, regime, K=variants[0].K, L=variants[0].L) for s, l in sl_pairs]
        for c, k in enumerate(row_keys):
            ax = fig.add_subplot(gs_p[r, c])
            if k not in by_key:
                ax.axis("off")
                continue
            per = per_variant[k]
            fr = frontiers[k]
            if logx:
                ax.set_xscale("log")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(*y_limits([k] if opts.own_y else row_keys))
            ticks = [t for t in opts.x_ticks if x_lo <= t <= x_hi] if opts.x_ticks else _cost_ticks(x_lo, x_hi)
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"\\${t:g}" for t in ticks])
                ax.minorticks_off()
            ax.plot([per[m]["mean_cost"] for m in fr], [per[m]["quality"] for m in fr], "--",
                    color="#555555", linewidth=1.0, alpha=0.7, zorder=2)
            for run in opts.series:
                for a, b in zip(run, run[1:]):
                    ax.annotate("", xy=(per[b]["mean_cost"], per[b]["quality"]),
                                xytext=(per[a]["mean_cost"], per[a]["quality"]),
                                arrowprops=dict(arrowstyle="-|>", color=PROVIDER_COLOR[provider_of(a)], lw=0.9,
                                                alpha=0.6, shrinkA=3, shrinkB=3, mutation_scale=7), zorder=3)
            pts, labels, bold = [], [], []
            for m in models_by_rho:
                x, y = per[m]["mean_cost"], per[m]["quality"]
                if not math.isfinite(y):
                    continue
                col = PROVIDER_COLOR[provider_of(m)]
                on = m in fr
                ax.scatter(x, y, s=64 if on else 40, marker="D" if on else "o",
                           facecolors=col if measured[m] else "none", edgecolors=col,
                           linewidths=1.6, zorder=4)
                pts.append((x, y))
                if opts.label_frontier_only and not on:
                    labels.append("")
                else:
                    labels.append((opts.point_labels or {}).get(m) or (opts.cell_labels or {}).get(m) or _short(m))
                bold.append(on)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7.5)
            ax.set_title(f"{k}{'  (default)' if k == default_key else ''}", fontsize=9,
                         fontweight="bold" if k == default_key else "normal")
            if c == 0:
                ax.set_ylabel("1 / (1 - mean rho)", fontsize=8.5)
            if r == n_rows - 1:
                ax.set_xlabel("mean cost per axis (USD)", fontsize=8.5)
            _place_labels(ax, pts, labels, bold, leaders="all")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PROVIDER_COLOR["anthropic"],
                   markeredgecolor=PROVIDER_COLOR["anthropic"], markersize=8, label="Anthropic"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PROVIDER_COLOR["openai"],
                   markeredgecolor=PROVIDER_COLOR["openai"], markersize=8, label="OpenAI"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#888888", markersize=8,
                   label="on the frontier (bold label)"),
        plt.Line2D([0], [0], linestyle="--", color="#555555", label="Pareto frontier"),
    ]
    if not all(measured[m] for m in models_by_rho):
        handles.insert(3, plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#888888",
                                     markersize=8, label="hollow = estimated cost"))
    for provider, name in (("google", "Google"), ("typesafe", "TypeSafe (Jev)"), ("blend", "Blend of judges")):
        if any(provider_of(m) == provider for m in models_by_rho):
            handles.insert(2, plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PROVIDER_COLOR[provider],
                                         markeredgecolor=PROVIDER_COLOR[provider], markersize=8, label=name))
    title = opts.headline or (f"Judge ranking under {n_v} projection variants, DESCRIPTION judging: "
                              f"{n_axes_total} axes, {n_m} judges (default {default_key})")
    fig.suptitle(title + "\n" + subtitle, fontsize=12, y=1 - 0.12 / fig_h, va="top")
    y_b = pan_top + 0.42 / fig_h
    key = "" if opts.cell_labels else label_key(list(models_by_rho))
    # The caption and the legend share a line, and the label key gets its own below: on one
    # line a long key ran under the legend (2026-11-02).
    y_cap = y_b + (0.11 / fig_h if key else 0)
    fig.text(0.01, y_cap, f"(b) Pareto plots, one per variant ({'each its own y limits' if opts.own_y else 'rows share y limits'}; "
             "x shared)", fontsize=10, va="center")
    if key:
        fig.text(0.01, y_b - 0.11 / fig_h, key, fontsize=9, va="center")
    fig.legend(handles=handles, loc="center right", ncol=len(handles), fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.99, y_cap))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight",
                metadata=png_metadata(title=title, source_text=Path(__file__).read_text(), inputs=inputs))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _clean(obj):
    """NaN / inf to None so the JSON is standard."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _subtree_input(data_dir: Path, rel: str, dep_key: str) -> InputSpec:
    try:
        return current_data_subtree_input(data_dir, rel, dep_key=dep_key)
    except (FileNotFoundError, KeyError, ValueError):
        return current_file_input(dep_key, Path(data_dir) / rel)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                   help="Activation vectors (default: the May 2026 8-slot extraction).")
    p.add_argument("--experiments_dir", default=str(DEFAULT_EXPERIMENTS_DIR),
                   help="Per-axis legacy cells <axis>/{gpt,sonnet}/ (default: roger/axis_judge_experiments).")
    p.add_argument("--tier_eval_dir", default=None,
                   help="Where the <axis>_tier_eval/<model>/ cells are (default: --experiments_dir).")
    p.add_argument("--axes_file", default=None,
                   help="TSV 'axis_dir pos neg pair_type' (default: every axis directory with "
                        "gpt/scores_descriptions.json, poles and pair type from gpt/config.json).")
    p.add_argument("--axes", nargs="+", default=None, help="Keep only these axis directories.")
    p.add_argument("--slots", type=int, nargs="+", default=DEFAULT_SLOTS, help="Token slots (default: 0 6).")
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS, help="Layers (default: 25 49).")
    p.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES),
                   help="Projection regimes (default: all three).")
    p.add_argument("--soft_K", type=int, default=DEFAULT_SOFT_K,
                   help=f"K of the soft_K regime (default: DEFAULT_SOFT_K = {DEFAULT_SOFT_K}).")
    p.add_argument("--shear_L", type=int, default=DEFAULT_SOFT_SHEAR_L,
                   help=f"L of the soft_shear regime (default: DEFAULT_SOFT_SHEAR_L = {DEFAULT_SOFT_SHEAR_L}).")
    p.add_argument("--ca_kind", default="combined", choices=("combined", "r_only", "t_only"),
                   help="Goal/no-goal subspaces for the shear (default: combined).")
    p.add_argument("--default_variant", default=DEFAULT_VARIANT,
                   help=f"The variant the others are compared with (default: {DEFAULT_VARIANT}).")
    p.add_argument("--price_basis", choices=PRICE_BASES, default=DEFAULT_PRICE_BASIS,
                   help=f"Cost basis, as judge_tier_cost_plot_pooled.py (default: {DEFAULT_PRICE_BASIS}, the real "
                        f"price, since 2026-11-03; list before).")
    p.add_argument("--legacy", action="store_true",
                   help="Also compare the May 2026 cells, \"gpt-4.1-mini (May)\" and \"claude-sonnet-4 (May)\" "
                        "(T=0, max_tokens 1024, mixed rubric versions; the Sonnet cells mix two models).  Off "
                        "by default since 2026-11-01 (Roger).")
    p.add_argument("--min_axes", type=int, default=None,
                   help="Compare only models with at least this many axes (default: the most any model has).")
    p.add_argument("--include_models", nargs="+", default=[], metavar="CELL",
                   help="Compare only these cell directory names (an allow-list; applied before --exclude_models).")
    p.add_argument("--exclude_models", nargs="+", default=[], metavar="CELL",
                   help="Cell directory names to leave out (e.g. claude-haiku-4-5 at T=1, superseded by "
                        "claude-haiku-4-5@t0).")
    p.add_argument("--price_band", type=float, default=DEFAULT_PRICE_BAND,
                   help=f"Report the best model under this cost per axis (default: {DEFAULT_PRICE_BAND}).")
    p.add_argument("--bootstrap_draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    p.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--output_dir", default=None, help="Default: --experiments_dir.")
    p.add_argument("--cache_dir", default=None,
                   help=f"Slice cache (default: <output_dir>/{OUTPUT_STEM}_cache).")
    p.add_argument("--no_cache", action="store_true", help="Neither read nor write the slice cache.")
    p.add_argument("--cache-policy", dest="cache_policy", choices=CACHE_POLICIES, default="warn",
                   help="What to do when a judge cache's recorded inputs have changed: strict | warn (default) | "
                        "rebuild | off.  The slice cache has its own check: built from other files, it is rebuilt.")
    p.add_argument("--stem", default=OUTPUT_STEM,
                   help=f"Name of the output files, <stem>.json and <stem>.png (default {OUTPUT_STEM}).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    data_dir = Path(args.data_dir).resolve()
    experiments_dir = Path(args.experiments_dir).resolve()
    tier_eval_dir = Path(args.tier_eval_dir).resolve() if args.tier_eval_dir else experiments_dir
    output_dir = Path(args.output_dir).resolve() if args.output_dir else experiments_dir
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / f"{OUTPUT_STEM}_cache"
    log(f"[setup] rho_by_layer from {rbl.__file__}")

    variants = build_variants(args.slots, args.layers, args.regimes, K=args.soft_K, L=args.shear_L)
    vkeys = [v.key for v in variants]
    if args.default_variant not in vkeys:
        raise SystemExit(f"--default_variant {args.default_variant} is not among the variants {vkeys}")

    kinds_for_name = vector_kinds(data_dir)
    axes = read_axes_tsv(Path(args.axes_file)) if args.axes_file else discover_axes(experiments_dir, kinds_for_name)
    if args.axes:
        wanted = set(args.axes)
        unknown = wanted - {a.axis_dir for a in axes}
        if unknown:
            raise SystemExit(f"--axes not found: {sorted(unknown)}")
        axes = [a for a in axes if a.axis_dir in wanted]
    log(f"[setup] {len(axes)} axes, {len(variants)} variants: {', '.join(vkeys)}")

    inputs: List[InputSpec] = [
        _subtree_input(data_dir, "traits/vectors", "traits_vectors"),
        _subtree_input(data_dir, "roles/vectors", "roles_vectors"),
    ]
    if "soft_shear" in args.regimes:
        for sub in ("r_goal", "r_nogoal", "t_goal", "t_nogoal"):
            inputs.append(_subtree_input(data_dir, f"combinations/vectors/derived/marginals/{sub}",
                                         f"marginals_{sub}"))
        inputs.append(_subtree_input(data_dir, "combinations/vectors/derived/axis", "theatricality_axis"))
    if args.axes_file:
        inputs.append(current_file_input("axes_file", Path(args.axes_file)))

    cells, skipped = load_cells(axes, experiments_dir, tier_eval_dir, kinds_for_name,
                                args.price_basis, inputs, include_legacy=args.legacy, policy=args.cache_policy)
    if args.include_models:
        for m in args.include_models:
            if m not in cells:
                log(f"  NOTE: --include_models {m}: no such cell")
        for m in [m for m in cells if m not in args.include_models]:
            del cells[m]
    left_out = [m for m in args.exclude_models if cells.pop(m, None) is not None]
    for m in args.exclude_models:
        if m not in left_out:
            log(f"  NOTE: --exclude_models {m}: no such cell")
    models, excluded, min_axes = apply_coverage(cells, args.min_axes)
    for s in skipped:
        log(f"  NOTE: {s['model']} / {s['axis']}: {s['reason']} -- cell skipped")
    log(f"[cells] {sum(len(v) for v in cells.values())} cells; compared (>= {min_axes} axes): "
        + ", ".join(f"{m} ({len(cells[m])})" for m in models))
    if excluded:
        log("[cells] excluded (fewer axes): " + ", ".join(f"{m} ({k})" for m, k in excluded.items()))
    if not models:
        raise SystemExit(f"no model has {min_axes} axes; lower --min_axes")
    t_cells = time.time()

    # Activation slices
    need_shear = "soft_shear" in args.regimes
    meta = {"version": CACHE_VERSION, "fingerprint": data_fingerprint(data_dir), "ca_kind": args.ca_kind}
    slices = None if args.no_cache else load_slices(cache_dir, args.slots, args.layers, meta, need_shear)
    if slices is not None:
        log(f"[slices] loaded from {cache_dir}")
    else:
        log(f"[slices] building from {data_dir} (reads every vector file once)")
        slices = build_slices(data_dir, args.slots, args.layers, with_shear=need_shear, ca_kind=args.ca_kind)
        if not args.no_cache:
            save_slices(cache_dir, slices, meta)
            log(f"[slices] cached under {cache_dir}")
    t_slices = time.time()

    names_by_axis: Dict[str, set] = {}
    for m in models:
        for axis, c in cells[m].items():
            names_by_axis.setdefault(axis, set()).update(c.scores)
    axes_used = [a for a in axes if a.axis_dir in names_by_axis]
    projections = project_variants(slices, axes_used, variants, names_by_axis, data_dir)
    t_proj = time.time()
    rho, n = compute_rho_table(cells, models, axes_used, projections, variants)
    t_rho = time.time()

    # Aggregation
    costs = {m: float(np.mean([c.cost for c in cells[m].values()])) for m in models}
    measured = {m: all(c.cost_is_measured for c in cells[m].values()) for m in models}
    # Each judge's reply cap (max_tokens) over its cells: tier-eval cells record it or
    # used the script's default (200); legacy cells have it in config.json.
    caps = {m: {cell_max_tokens(c.scores_path.parent, legacy=not c.cost_is_measured)
                for c in cells[m].values()} for m in models}
    per_variant: Dict[str, Dict[str, dict]] = {}
    frontiers: Dict[str, List[str]] = {}
    for k in vkeys:
        per_variant[k], frontiers[k] = summarise_variant(rho, costs, models, k)
    default_rank = ranking(per_variant[args.default_variant], models)
    taus = kendall_tau_vs_default(per_variant, default_rank, args.default_variant)
    counts = frontier_counts(frontiers, models)
    bands = {k: price_bands(per_variant[k], frontiers[k], models, args.price_band) for k in vkeys}
    boot = (bootstrap_keep_order(rho, default_rank[0], default_rank[1], vkeys,
                                 args.bootstrap_draws, args.seed) if len(models) >= 2 else None)
    check = check_vs_cached(cells, models, rho, n, inputs=inputs, policy=args.cache_policy) if "s6_l25_raw" in vkeys else None
    t_stats = time.time()

    # Print
    w = max(len(m) for m in models)
    print(f"\nMean rho by variant ({len(axes_used)} axes); rows by {args.default_variant}; * = on the frontier")
    print(" " * (w + 12) + "  ".join(f"{k:>14s}" for k in vkeys))
    for m in default_rank:
        cells_txt = "  ".join(f"{per_variant[k][m]['mean_rho']:>13.4f}{'*' if m in frontiers[k] else ' '}"
                              for k in vkeys)
        print(f"{m:<{w}s} ${costs[m]:>8.3f}  {cells_txt}")
    print("\nKendall tau vs " + args.default_variant + ": "
          + ", ".join(f"{k} {'n/a' if taus[k] is None else f'{taus[k]:+.3f}'}" for k in vkeys))
    print("Frontier per variant:")
    for k in vkeys:
        print(f"  {k:>14s}: {', '.join(frontiers[k])}")
    print("On the frontier (of %d): " % len(vkeys) + ", ".join(f"{m} {counts[m]}" for m in default_rank))
    print(f"Price bands (cheapest frontier | best under ${args.price_band:.2f}/axis | best overall):")
    for k in vkeys:
        b = bands[k]
        print(f"  {k:>14s}: {b['cheapest_frontier']} | {b['best_under_band']} | {b['best_overall']}")
    if boot:
        print(f"Bootstrap ({boot['draws']} draws, seed {boot['seed']}, {boot['n_axes']} axes): "
              f"P({boot['top']} ahead of {boot['runner_up']}):")
        print("  " + ", ".join(f"{k} {boot['p_keep_order'][k]:.3f}" for k in vkeys))
    if check:
        print(f"Check vs cached correlations.json (s6_l25_raw): max |diff| {check['max_abs_diff_same_n']} over "
              f"{check['n_cells_same_n']} cells with the same n; {len(check['cells_other_n'])} cells with "
              f"another n (max |diff| {check['max_abs_diff_other_n']}); "
              f"{len(check['cells_without_cached_rho'])} without a cached rho")

    # JSON
    payload = {
        "config": {
            "variants": vkeys, "default_variant": args.default_variant,
            "slots": list(args.slots), "layers": list(args.layers), "regimes": list(args.regimes),
            "soft_K": args.soft_K, "shear_L": args.shear_L, "ca_kind": args.ca_kind,
            "price_basis": args.price_basis, "price_band_usd": args.price_band, "min_axes": min_axes,
            "exclude_models": left_out,
            "data_dir": str(data_dir), "experiments_dir": str(experiments_dir),
            "tier_eval_dir": str(tier_eval_dir), "axes_file": args.axes_file,
            "score_keys": "bare names lifted by migrate_v1_static_scores (collision names dropped)",
            "soft_K_pool": "build_augmented_whitening_pool, poles held out, combinations/default.pt included",
        },
        "axes": [vars(a) for a in axes_used],
        "models": {m: {"n_axes": len(cells[m]), "mean_cost": costs[m], "cost_is_measured": measured[m],
                       "provider": provider_of(m), "max_tokens": sorted(caps[m], key=lambda v: (v is None, v))}
                   for m in models},
        "excluded_models": excluded,
        "skipped_cells": skipped,
        "rho": rho,
        "n": n,
        "per_variant": {k: {"models": per_variant[k], "frontier": frontiers[k]} for k in vkeys},
        "summary": {
            "default_ranking": default_rank,
            "kendall_tau_vs_default": taus,
            "frontier_sets": frontiers,
            "frontier_counts": counts,
            "price_bands": bands,
            "bootstrap_keep_order": boot,
        },
        "check_vs_cached_correlations": check,
        "timings_s": {"cells": round(t_cells - t0, 2), "slices": round(t_slices - t_cells, 2),
                      "projections": round(t_proj - t_slices, 2), "rho": round(t_rho - t_proj, 2),
                      "stats": round(t_stats - t_rho, 2)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    title = f"{args.stem}: {len(axes_used)} axes, {len(models)} judges, {len(vkeys)} variants"
    json_path = output_dir / f"{args.stem}.json"
    tmp = json_path.with_name(json_path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(json_metadata(payload, title=title, inputs=inputs)), indent=2))
    tmp.replace(json_path)
    print(f"\nWrote {json_path}")

    subtitle = (f"soft_K: K={args.soft_K} (augmented pool, poles held out); soft_shear: L={args.shear_L} "
                f"({args.ca_kind} goal/no-goal); cost at {PRICE_BASIS_LABEL[args.price_basis]}")
    if excluded:
        subtitle += "\nnot compared (judged on fewer axes): " + ", ".join(f"{display_name(m)} ({k})"
                                                                           for m, k in excluded.items())
    if left_out:
        subtitle += "\nleft out (--exclude_models): " + ", ".join(display_name(m) for m in left_out)
    png_path = output_dir / f"{args.stem}.png"
    render_figure(png_path, default_rank, variants, per_variant, frontiers, measured,
                  args.default_variant, len(axes_used), subtitle, inputs)
    print(f"Wrote {png_path}")
    log(f"[time] cells {t_cells - t0:.1f}s, slices {t_slices - t_cells:.1f}s, projections "
        f"{t_proj - t_slices:.1f}s, rho {t_rho - t_proj:.1f}s, total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
