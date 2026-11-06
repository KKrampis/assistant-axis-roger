#!/usr/bin/env python3
"""Multi-GPU steering sweep dispatcher.

Reads a YAML/JSON experiment config, builds a work-item queue
``[baselines, (slot, layer, sign) x N]``, and spawns one worker process
per visible GPU.  Each worker loads the model once (using
``setup_model_tmpfs_cache`` to mirror weights into /dev/shm and avoid
NFS-backed mmap stalls), then pulls items off the shared queue and runs
either :func:`compute_baselines` or :func:`run_steering_cell`.

Baselines (strength=0) are computed once per (persona, question) and
shared across all cells via ``{output_root}/baselines/records.jsonl``.

Usage::

    uv run steering/run_sweep.py --config configs/angel_to_demon.yaml
    uv run steering/run_sweep.py --config configs/foo.yaml --gpus 4

Config schema (YAML or JSON, by file extension)::

    experiment_id: angel_to_demon_v1
    model_name: Qwen/Qwen3-32B
    output_dir: /workspace/outputs/qwen-3-32b/steering    # parent dir
    axis_source:
      type: role_transplant         # or "axis"
      # for "role_transplant":
      vectors_dir: /workspace/outputs/qwen-3-32b/roles/vectors
      role_from: angel
      role_to: demon
      # for "axis":
      # axis_path: /workspace/outputs/qwen-3-32b/roles/axis.pt
    persona:
      type: role                    # or "trait", "combination"
      role: historian
      traits: []                    # used for "combination" / "trait"
      prompt_index: 0               # which of the 5 instruction variants
    cells:
      - { slot: 0, layer: 26 }
      - { slot: 0, layer: 49 }
      - { slot: 3, layer: 25 }
      # optional per-cell overrides:
      # - { slot: 1, layer: 25, weakest_strength: 0.5 }
    sweep:
      weakest_strength: 1.0
      max_strength: 64.0
      multiplier: 1.189
      signs: [+1, -1]
    positions_mode: all             # or "prefill_only"
    batch_size: 8
    max_new_tokens: 512
    questions_file: data/steering/questions/angel_to_demon_v1.json

Phase-1 limitations (deferred to follow-up plans):
  - Judge integration is stubbed (NoOpJudgeDispatcher); records land
    with judges=null and never trigger early termination.
  - positions_mode is restricted to {"all", "prefill_only"}.
  - axis_source supports "axis" and "role_transplant"; "header_replacement"
    is deferred (it needs intervention_type=replacement, a larger change).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make assistant_axis imports work whether run from repo root or steering/.
# steering/run_sweep.py -> parents[1] is the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant_axis.atomic_io import (   # noqa: E402
    atomic_write_text, read_text_with_retry,
)

logger = logging.getLogger("steering_sweep")


# ---------------------------------------------------------------------------
# Sweep log file handler (per-experiment, attached on both parent + workers)
# ---------------------------------------------------------------------------

# Module-level map (path -> FileHandler) so:
#   1. re-imports inside spawned workers don't accidentally double-attach
#      for the same path (idempotent attach), and
#   2. callers can look up the handler by path to detach it cleanly when
#      transitioning between experiments in a multi-config queue run.
_SWEEP_LOG_HANDLERS: Dict[str, "logging.FileHandler"] = {}


def _attach_sweep_log_filehandler(output_root: Path) -> "logging.FileHandler":
    """Attach a FileHandler writing to ``output_root / "sweep.log"`` to
    the root logger and return the handler.

    Idempotent: if a handler for this exact path is already attached on
    this process, returns the existing handler without re-attaching.
    Safe to call from both the parent process and each spawned worker
    -- line-based logging keeps the appended writes ordered well enough
    on Linux (records are well under PIPE_BUF) for combined parent+worker
    output to be readable without explicit cross-process locking.

    Append mode so resumed runs accumulate history rather than
    truncating prior logs (matches the rest of run_sweep.py's
    resume-friendly atomic-write conventions).
    """
    target = (output_root / "sweep.log").resolve()
    key = str(target)
    if key in _SWEEP_LOG_HANDLERS:
        return _SWEEP_LOG_HANDLERS[key]
    target.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(target), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logging.getLogger().addHandler(fh)
    _SWEEP_LOG_HANDLERS[key] = fh
    return fh


def _detach_sweep_log_filehandler(output_root: Path) -> None:
    """Detach (and close) the FileHandler previously attached for this
    output_root.  No-op if no handler is registered for the path.

    Used by the multi-config queue runner to swap the sweep.log handler
    when transitioning between experiments so each experiment's log stays
    cleanly partitioned to its own file.
    """
    target = (output_root / "sweep.log").resolve()
    key = str(target)
    fh = _SWEEP_LOG_HANDLERS.pop(key, None)
    if fh is None:
        return
    try:
        logging.getLogger().removeHandler(fh)
        fh.close()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"failed to detach sweep.log FileHandler at {target}: {e}"
        )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> Dict[str, Any]:
    text = read_text_with_retry(path, logger_obj=logger)
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise SystemExit(
                "PyYAML required for .yaml configs; install or use .json"
            ) from e
        return yaml.safe_load(text)
    return json.loads(text)


def _load_questions(path: Path) -> List[str]:
    """Load a question list from a JSON file.

    Accepts either:
      - a flat list: ["q1", "q2", ...]
      - or {"questions": ["q1", "q2", ...]}
    """
    raw = json.loads(read_text_with_retry(path, logger_obj=logger))
    if isinstance(raw, list):
        questions = raw
    elif isinstance(raw, dict) and "questions" in raw:
        questions = raw["questions"]
    else:
        raise ValueError(
            f"{path}: expected a list or {{questions: [...]}}; got {type(raw).__name__}"
        )
    if not all(isinstance(q, str) for q in questions):
        raise ValueError(f"{path}: all entries must be strings")
    return list(questions)


def _build_persona_system_prompt(persona_cfg: Dict[str, Any],
                                 *,
                                 instructions_dir: Path) -> str:
    """Build the system prompt for the persona using the standard instruction sources.

    role:        data/roles/instructions/<role>.json instruction[prompt_index].pos
    trait:       data/traits/instructions/<trait>.json instruction[prompt_index].pos
    combination: role_inst + "\\n" + trait_inst[i] for each trait, joined by "\\n"
                 (matches RoleResponseGenerator.generate_combined_responses
                 in assistant_axis/generation.py:475)
    """
    ptype = persona_cfg.get("type", "role")
    prompt_idx = int(persona_cfg.get("prompt_index", 0))

    def _load_inst(kind: str, name: str) -> str:
        path = instructions_dir / kind / "instructions" / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"persona instruction file not found: {path}"
            )
        data = json.loads(read_text_with_retry(path, logger_obj=logger))
        try:
            return data["instruction"][prompt_idx]["pos"]
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"{path}: missing instruction[{prompt_idx}].pos"
            ) from e

    if ptype == "role":
        return _load_inst("roles", persona_cfg["role"])
    if ptype == "trait":
        return _load_inst("traits", persona_cfg["trait"])
    if ptype == "combination":
        role = persona_cfg["role"]
        traits = persona_cfg.get("traits", [])
        if not traits:
            raise ValueError("combination persona requires at least one trait")
        parts = [_load_inst("roles", role)]
        for trait in traits:
            parts.append(_load_inst("traits", trait))
        return "\n".join(parts)
    raise ValueError(f"unknown persona.type: {ptype!r}")


# ---------------------------------------------------------------------------
# Work-item construction
# ---------------------------------------------------------------------------

def _cell_dir_name(slot: int, layer: int, sign: int) -> str:
    sign_str = "+1" if sign > 0 else "-1"
    return f"s{slot}_l{layer}_{sign_str}"


def _build_work_items(
    config: Dict[str, Any],
    output_root: Path,
    persona_prompt: str,
    questions: List[str],
    *,
    judging: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Construct the list of work items to dispatch to GPU workers.

    Produces one ``baselines`` item, then one ``cell`` item per
    ``(slot, layer, sign)``.  Items are simple dicts (picklable) that
    workers consume to call either ``compute_baselines`` or
    ``run_steering_cell``.

    The ``judging`` dict (built by :func:`_build_judging_config` from
    CLI flags + the experiment config) is attached to each cell item so
    workers can construct a RealJudgeDispatcher locally; baseline items
    don't need it (they don't judge).
    """
    sweep_cfg = config.get("sweep", {})
    default_weakest = float(sweep_cfg.get("weakest_strength", 1.0))
    max_strength = float(sweep_cfg.get("max_strength", 64.0))
    multiplier = float(sweep_cfg.get("multiplier", 1.189))
    # Bidirectional-scan knobs (2026-05-14: new default scan_mode).
    # See assistant_axis.steering_runner.BidirectionalCursor and the
    # design plan for full semantics.
    scan_mode = str(sweep_cfg.get("scan_mode", "bidirectional"))
    if scan_mode not in ("bidirectional", "legacy_unidirectional"):
        raise ValueError(
            f"sweep.scan_mode must be 'bidirectional' or "
            f"'legacy_unidirectional'; got {scan_mode!r}"
        )
    eff_stop_threshold = float(sweep_cfg.get("eff_stop_threshold", 1.0 / 3.0))
    eff_stop_consecutive = int(sweep_cfg.get("eff_stop_consecutive", 2))
    min_strength = float(sweep_cfg.get("min_strength", 0.125))
    # ``start_strength_multiplier_steps``: per-(mode, slot, layer) lookup
    # via assistant_axis.sweep_start_heuristics.compute_start_steps unless
    # the YAML explicitly sets a sweep-wide override (which then wins for
    # every cell, ignoring the table).  ``None`` here means "use the
    # heuristic per cell"; an int means "this fixed value for every cell".
    if "start_strength_multiplier_steps" in sweep_cfg:
        start_strength_multiplier_steps: int | None = int(
            sweep_cfg["start_strength_multiplier_steps"]
        )
    else:
        start_strength_multiplier_steps = None
    signs = list(sweep_cfg.get("signs", [+1, -1]))
    if not all(s in (+1, -1) for s in signs):
        raise ValueError(f"sweep.signs must be subset of [+1, -1]; got {signs}")

    batch_size = int(config.get("batch_size", 8))
    max_new_tokens = int(config.get("max_new_tokens", 512))
    positions_mode = config.get("positions_mode", "all")

    items: List[Dict[str, Any]] = [{
        "kind": "baselines",
        "persona": persona_prompt,
        "questions": questions,
        "output_dir": str(output_root),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
    }]

    cells = config.get("cells", [])
    if not cells:
        raise ValueError("config.cells is empty; nothing to sweep")

    # Lazy import to keep run_sweep.py's import cost low; only needed
    # when actually building cell items.
    from assistant_axis.sweep_start_heuristics import compute_start_steps

    for cell_cfg in cells:
        slot = int(cell_cfg["slot"])
        layer = int(cell_cfg["layer"])
        weakest = float(cell_cfg.get("weakest_strength", default_weakest))
        # Resolve the per-cell start-strength steps.  YAML override (set
        # above) wins for every cell; otherwise consult the empirical
        # heuristic table keyed on (positions_mode, slot, layer).
        if start_strength_multiplier_steps is None:
            cell_start_steps = compute_start_steps(
                positions_mode=positions_mode, slot=slot, layer=layer,
            )
        else:
            cell_start_steps = start_strength_multiplier_steps
        for sign in signs:
            cell_dir = output_root / _cell_dir_name(slot, layer, int(sign))
            items.append({
                "kind": "cell",
                "slot": slot,
                "layer": layer,
                "sign": int(sign),
                "cell_dir": str(cell_dir),
                "axis_source": config["axis_source"],
                "weakest_strength": weakest,
                "max_strength": max_strength,
                "multiplier": multiplier,
                "scan_mode": scan_mode,
                "eff_stop_threshold": eff_stop_threshold,
                "eff_stop_consecutive": eff_stop_consecutive,
                "min_strength": min_strength,
                "start_strength_multiplier_steps": cell_start_steps,
                "persona": persona_prompt,
                "questions": questions,
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "positions_mode": positions_mode,
                "baselines_records_path": str(output_root / "baselines"
                                              / "records.jsonl"),
                "judging": judging or {},
            })
    return items


def _build_judging_config(
    *,
    config: Dict[str, Any],
    args,
    instructions_dir: Path,
) -> Dict[str, Any]:
    """Build a serializable judging config attached to each cell work item.

    Picks up CLI overrides for models / thresholds / mode and resolves
    persona + steering specs from the experiment config + data dir so
    workers don't need filesystem access to data/{roles,traits}/.

    Returns ``{"enabled": False}`` if --no-live-judging was passed; the
    worker uses NoOpJudgeDispatcher in that case (still records jsonl,
    no judge calls, sweep runs to max strength).
    """
    if getattr(args, "no_live_judging", False):
        return {"enabled": False, "reason": "--no-live-judging"}

    # Persona spec
    persona_cfg = config.get("persona") or {}
    persona = _resolve_persona_spec(persona_cfg, instructions_dir)

    # Steering spec from axis_source
    axis_cfg = config.get("axis_source") or {}
    steering = _resolve_steering_spec(axis_cfg, instructions_dir)

    cfg: Dict[str, Any] = {
        "enabled": True,
        "persona": persona,
        "steering": steering,
        "coherence_model": getattr(args, "coherence_model", None)
                          or "gpt-4.1-mini",
        "rp_model": getattr(args, "rp_model", None) or "gpt-4.1-mini",
        "effect_models": (
            [m.strip() for m in args.effect_models.split(",") if m.strip()]
            if getattr(args, "effect_models", None)
            else ["gpt-4.1-mini", "claude-haiku-4-5-20251001"]
        ),
        "effect_mode": getattr(args, "effect_mode", "bidirectional"),
        "target_batch_size": int(getattr(args, "target_batch_size", 7)),
        "skip_threshold": float(
            getattr(args, "skip_if_strength_mean_coh_above", 1.0)
        ),
        "coh_stop_threshold": float(
            getattr(args, "coh_stop_threshold", 1.5)
        ),
        "coh_stop_consecutive": int(
            getattr(args, "coh_stop_consecutive", 2)
        ),
    }
    return cfg


def _resolve_persona_spec(persona_cfg: Dict[str, Any],
                          instructions_dir: Path) -> Dict[str, Any]:
    """Read role/trait descriptions and return a serializable spec dict."""
    ptype = persona_cfg.get("type", "role")

    def _desc(kind: str, name: str) -> str:
        path = instructions_dir / kind / "instructions" / f"{name}.json"
        if not path.exists():
            return ""
        try:
            return str(json.loads(read_text_with_retry(path,
                                                       logger_obj=logger))
                      .get("description", ""))
        except (OSError, json.JSONDecodeError):
            return ""

    if ptype == "role":
        name = persona_cfg["role"]
        return {"role": name, "description": _desc("roles", name),
                "extra_traits": []}
    if ptype == "trait":
        name = persona_cfg["trait"]
        return {"role": name, "description": _desc("traits", name),
                "extra_traits": []}
    if ptype == "combination":
        role = persona_cfg["role"]
        traits = persona_cfg.get("traits") or []
        extra = [[t, _desc("traits", t)] for t in traits]
        return {"role": role, "description": _desc("roles", role),
                "extra_traits": extra}
    raise ValueError(f"unknown persona.type: {ptype!r}")


def _resolve_steering_spec(axis_cfg: Dict[str, Any],
                           instructions_dir: Path) -> Dict[str, Any]:
    """Build a SteeringSpec dict from the axis_source config."""
    src_type = axis_cfg.get("type", "axis")
    pos_label = axis_cfg.get("pos_label")
    pos_desc = axis_cfg.get("pos_description")
    neg_label = axis_cfg.get("neg_label")
    neg_desc = axis_cfg.get("neg_description")
    axis_name = axis_cfg.get("axis_name")

    def _try_desc(kind_name: str) -> str:
        for kind in ("traits", "roles"):
            path = (instructions_dir / kind / "instructions"
                    / f"{kind_name}.json")
            if path.exists():
                try:
                    return str(json.loads(read_text_with_retry(path,
                                                               logger_obj=logger))
                              .get("description", ""))
                except (OSError, json.JSONDecodeError):
                    continue
        return ""

    if src_type == "role_transplant":
        if not pos_label:
            pos_label = axis_cfg.get("role_to", "pos")
        if not neg_label:
            neg_label = axis_cfg.get("role_from", "neg")
        if not pos_desc:
            pos_desc = _try_desc(pos_label)
        if not neg_desc:
            neg_desc = _try_desc(neg_label)
        if not axis_name:
            axis_name = f"{neg_label}-{pos_label}"

    return {
        "axis_name": str(axis_name or "unnamed"),
        "pos_label": str(pos_label or "positive"),
        "pos_description": str(pos_desc or ""),
        "neg_label": str(neg_label or "negative"),
        "neg_description": str(neg_desc or ""),
    }


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _setup_worker_logger(worker_id: int, gpu_id: int) -> logging.Logger:
    log = logging.getLogger(f"steering_worker_{worker_id}")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        f"%(asctime)s - W{worker_id}[GPU {gpu_id}] - %(levelname)s - %(message)s"
    ))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    # propagate=False prevents the worker-prefixed messages from also
    # being printed by the root logger configured by logging.basicConfig
    # at the top of _worker_main (which uses a different format meant for
    # module loggers).  Without this, every worker message would appear
    # twice -- once with the W0[GPU 0] prefix here, once with the
    # %(name)s prefix from the root.
    log.propagate = False
    return log


def _build_cell_dispatcher(
    item: Dict[str, Any], *,
    slot: int, layer: int, sign: int,
    config: Dict[str, Any], judging_cfg: Dict[str, Any],
    log: logging.Logger,
):
    """Construct a per-cell judge dispatcher.

    Returns either a RealJudgeDispatcher (live coherence-blocking +
    async RP/effect) or a NoOpJudgeDispatcher when:
      - judging is explicitly disabled (--no-live-judging), or
      - both OPENAI_API_KEY and ANTHROPIC_API_KEY are absent (we can't
        actually call any judge).

    Each cell gets its own dispatcher instance because records_path is
    cell-local; the asyncio thread is cheap to start/stop.
    """
    from assistant_axis.steering_judges import (
        NoOpJudgeDispatcher, PersonaSpec, RealJudgeDispatcher, SteeringSpec,
        build_baseline_lookup,
    )

    if not judging_cfg or not judging_cfg.get("enabled"):
        log.info(f"[cell s{slot}_l{layer}_{sign:+d}] judging disabled "
                 f"({judging_cfg.get('reason', 'no judging config')}); "
                 f"using NoOpJudgeDispatcher")
        return NoOpJudgeDispatcher()

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not (has_openai or has_anthropic):
        log.warning(f"[cell s{slot}_l{layer}_{sign:+d}] no API keys present "
                    f"(OPENAI_API_KEY / ANTHROPIC_API_KEY); falling back to "
                    f"NoOpJudgeDispatcher -- live coherence early-term "
                    f"will not trigger")
        return NoOpJudgeDispatcher()

    persona_dict = judging_cfg["persona"]
    steering_dict = judging_cfg["steering"]
    persona = PersonaSpec(
        role=persona_dict["role"],
        description=persona_dict["description"],
        extra_traits=[(t[0], t[1]) for t in persona_dict.get("extra_traits", [])],
    )
    steering = SteeringSpec(
        axis_name=steering_dict["axis_name"],
        pos_label=steering_dict["pos_label"],
        pos_description=steering_dict["pos_description"],
        neg_label=steering_dict["neg_label"],
        neg_description=steering_dict["neg_description"],
    )
    baseline_lookup = build_baseline_lookup(
        Path(item["baselines_records_path"])
    )
    cell_dir = Path(item["cell_dir"])
    records_path = cell_dir / "records.jsonl"
    cell_dir.mkdir(parents=True, exist_ok=True)
    return RealJudgeDispatcher(
        records_path=records_path,
        persona=persona, steering=steering,
        baseline_lookup=baseline_lookup,
        coherence_model=judging_cfg["coherence_model"],
        rp_model=judging_cfg["rp_model"],
        effect_models=tuple(judging_cfg["effect_models"]),
        effect_mode=judging_cfg["effect_mode"],
        effect_target_batch_size=int(judging_cfg["target_batch_size"]),
        skip_threshold=float(judging_cfg["skip_threshold"]),
        coh_stop_threshold=float(judging_cfg["coh_stop_threshold"]),
    )


def _load_axis_for_cell(axis_source: Dict[str, Any],
                        slot: int, layer: int):
    """Load and extract a steering vector for one cell.

    Returns a 1-D torch.Tensor of shape (hidden,).  Workers are expected
    to move it to their model's device before passing to run_steering_cell.
    The axis-loaders themselves use atomic_io.torch_load_with_retry, so
    transient NFS short-reads are retried automatically.
    """
    from assistant_axis.axis import load_axis_with_metadata, load_role_vector

    src_type = axis_source.get("type", "axis")

    def _ensure_3d(t):
        return t.unsqueeze(0) if t.ndim == 2 else t

    if src_type == "axis":
        axis_path = axis_source["axis_path"]
        axis_raw, _ = load_axis_with_metadata(axis_path)
        axis_raw = _ensure_3d(axis_raw)
        return axis_raw[slot, layer].clone()

    if src_type == "role_transplant":
        vectors_dir = Path(axis_source["vectors_dir"])
        role_from = axis_source["role_from"]
        role_to = axis_source["role_to"]
        v_from, _ = load_role_vector(str(vectors_dir / f"{role_from}.pt"))
        v_to, _ = load_role_vector(str(vectors_dir / f"{role_to}.pt"))
        v_from = _ensure_3d(v_from)
        v_to = _ensure_3d(v_to)
        diff = v_to - v_from
        return diff[slot, layer].clone()

    if src_type == "header_replacement":
        raise NotImplementedError(
            "axis_source.type='header_replacement' is deferred to a later "
            "plan -- it needs intervention_type=replacement which the v1 "
            "runner does not expose."
        )

    raise ValueError(f"unknown axis_source.type: {src_type!r}")


def _worker_main(
    worker_id: int,
    gpu_id: int,
    queue,                 # multiprocessing.Queue
    config: Dict[str, Any],
):
    """Worker process body: load model on assigned GPU, drain work queue."""
    # Restrict CUDA to just our GPU before any torch import that could
    # initialise CUDA context.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Configure logging at the top so module-level loggers (e.g.
    # assistant_axis.tmpfs.logger) actually emit their INFO messages.
    # multiprocessing's spawn context starts a fresh Python interpreter
    # that doesn't inherit the parent's logging.basicConfig, so without
    # this every module-logger INFO call falls through to Python's
    # lastResort handler -- which only fires at WARNING.  That made the
    # tmpfs fast-path's "already mirrored" success log silently
    # disappear while the broken-cache warning came through.
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - W{worker_id} - %(name)s - %(levelname)s - %(message)s",
    )

    # Attach the same per-experiment FileHandler the parent did so
    # worker log lines land in output_root/sweep.log alongside the
    # parent's.  Spawn-context workers start with a fresh interpreter
    # so the parent's handler doesn't carry across -- we rebuild it
    # locally on the path the parent threaded into the work-item
    # dict.  Atomic appends on Linux up to PIPE_BUF guarantee no
    # mid-line interleaving for our record sizes.
    sweep_log_dir = config.get("sweep_log_dir")
    if sweep_log_dir:
        _attach_sweep_log_filehandler(Path(sweep_log_dir))

    # Set up tmpfs (TMPDIR + HF cache mirror) before model load.
    from assistant_axis.tmpfs import (
        DEFAULT_TMPDIR, setup_model_tmpfs_cache, setup_tmpdir,
    )
    setup_tmpdir(config.get("tmpdir") or DEFAULT_TMPDIR)
    new_hf_home = setup_model_tmpfs_cache(model_name=config["model_name"])
    if new_hf_home is not None:
        os.environ["HF_HOME"] = new_hf_home

    log = _setup_worker_logger(worker_id, gpu_id)
    log.info("starting; loading model...")

    # Heavy imports inside the worker so they happen post-fork.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from assistant_axis.steering_runner import (
        compute_baselines, directional_schedule, run_steering_cell,
    )

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        device_map="cuda:0",  # only this GPU is visible after CUDA_VISIBLE_DEVICES
        dtype=torch.bfloat16,
    )
    model.eval()
    log.info(f"model loaded in {time.time() - t0:.1f}s")

    # Track the most recently seen sweep_log_dir so we can detect when
    # the queue transitions from one experiment to the next and swap
    # the FileHandler (and thus the sweep.log file we're appending to).
    current_sweep_log_dir = sweep_log_dir
    while True:
        item = queue.get()
        if item is None:
            log.info("got sentinel; exiting")
            break

        # Swap FileHandler if this work item belongs to a different
        # experiment than the previously processed one.  Each item
        # carries its own sweep_log_dir (set by the parent in
        # main()); detach the old handler before attaching the new
        # so each experiment's sweep.log is cleanly partitioned to
        # its own file.
        item_log_dir = item.get("sweep_log_dir")
        if item_log_dir and item_log_dir != current_sweep_log_dir:
            if current_sweep_log_dir:
                _detach_sweep_log_filehandler(Path(current_sweep_log_dir))
            _attach_sweep_log_filehandler(Path(item_log_dir))
            current_sweep_log_dir = item_log_dir
            log.info(f"sweep log -> {Path(item_log_dir) / 'sweep.log'}")

        # Multi-GPU safety: if this is a cell item, wait until its
        # config's baselines sentinel exists before constructing the
        # dispatcher (which calls build_baseline_lookup on the records
        # file, and would otherwise see an empty/partial baselines and
        # silently corrupt judge prompts with blank baseline_responses).
        # The sentinel is touched by compute_baselines on success.
        if item["kind"] == "cell":
            baselines_dir = Path(item["baselines_records_path"]).parent
            sentinel = baselines_dir / ".complete"
            waited = 0
            poll_s = 2
            while not sentinel.exists():
                if waited == 0:
                    log.info(
                        f"waiting for baselines sentinel at {sentinel} "
                        f"(another worker producing them)"
                    )
                time.sleep(poll_s)
                waited += poll_s
                if waited > 1200:
                    raise RuntimeError(
                        f"baselines sentinel {sentinel} not produced "
                        f"after {waited}s; aborting cell"
                    )
            if waited > 0:
                log.info(f"baselines ready after {waited}s wait")

        try:
            if item["kind"] == "baselines":
                log.info("computing baselines")
                compute_baselines(
                    model, tokenizer,
                    persona_system_prompt=item["persona"],
                    questions=item["questions"],
                    output_dir=item["output_dir"],
                    batch_size=item["batch_size"],
                    max_new_tokens=item["max_new_tokens"],
                    model_name=config["model_name"],
                )

            elif item["kind"] == "cell":
                slot, layer, sign = item["slot"], item["layer"], item["sign"]
                cell_dir = item["cell_dir"]
                scan_mode = item.get("scan_mode", "bidirectional")
                log.info(
                    f"running cell s{slot}_l{layer}_sign={sign:+d}, "
                    f"scan_mode={scan_mode}, weakest={item['weakest_strength']}, "
                    f"max={item['max_strength']}, "
                    f"min={item.get('min_strength', 0.125)}"
                )
                axis_vector = _load_axis_for_cell(
                    item["axis_source"], slot, layer
                ).to(model.device, dtype=torch.bfloat16)
                # Legacy mode needs an explicit schedule; bidirectional
                # mode builds its cursor lazily inside run_steering_cell.
                if scan_mode == "legacy_unidirectional":
                    schedule = directional_schedule(
                        sign=sign,
                        weakest=item["weakest_strength"],
                        max_strength=item["max_strength"],
                        multiplier=item["multiplier"],
                    )
                    log.info(f"legacy schedule: {len(schedule)} strengths "
                             f"({schedule[0]} .. {schedule[-1]})")
                else:
                    schedule = None
                # Build a per-cell judge dispatcher.  Each cell gets its
                # own dispatcher because the records_path is cell-local;
                # the dispatcher's background event loop is cheap to
                # spin up and tear down (a daemon thread + asyncio loop).
                # If judging is disabled (--no-live-judging) or no API
                # keys are present, fall back to NoOp so the sweep still
                # runs to its configured max strength.
                judge_dispatcher = _build_cell_dispatcher(
                    item, slot=slot, layer=layer, sign=sign,
                    config=config,
                    judging_cfg=item.get("judging") or {},
                    log=log,
                )
                judging_cfg = item.get("judging") or {}
                try:
                    run_steering_cell(
                        model, tokenizer,
                        axis_vector=axis_vector,
                        slot=slot,
                        layer=layer,
                        sign=sign,
                        strengths=schedule,
                        persona_system_prompt=item["persona"],
                        questions=item["questions"],
                        output_dir=cell_dir,
                        batch_size=item["batch_size"],
                        max_new_tokens=item["max_new_tokens"],
                        positions_mode=item["positions_mode"],
                        model_name=config["model_name"],
                        judge_dispatcher=judge_dispatcher,
                        coh_stop_threshold=float(
                            judging_cfg.get("coh_stop_threshold", 1.5)
                        ),
                        coh_stop_consecutive=int(
                            judging_cfg.get("coh_stop_consecutive", 2)
                        ),
                        scan_mode=scan_mode,
                        eff_stop_threshold=float(
                            item.get("eff_stop_threshold", 1.0 / 3.0)
                        ),
                        eff_stop_consecutive=int(
                            item.get("eff_stop_consecutive", 2)
                        ),
                        weakest_strength=float(item["weakest_strength"]),
                        max_strength=float(item["max_strength"]),
                        multiplier=float(item["multiplier"]),
                        min_strength=float(item.get("min_strength", 0.125)),
                        start_strength_multiplier_steps=int(
                            item.get("start_strength_multiplier_steps", 2)
                        ),
                    )
                finally:
                    if hasattr(judge_dispatcher, "shutdown"):
                        judge_dispatcher.shutdown()
            else:
                log.error(f"unknown work item kind: {item.get('kind')!r}")

        except Exception as e:
            log.exception(f"item {item.get('kind')!r} failed: {e}")
            # Continue processing remaining items rather than crash the
            # worker -- one bad cell shouldn't sink the whole sweep.

    log.info("worker exiting cleanly")


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def _detect_gpu_count(requested: Optional[int]) -> int:
    """Return number of GPUs to use.

    If `requested` is None, use all visible GPUs (CUDA_VISIBLE_DEVICES
    or torch.cuda.device_count()).  Otherwise cap at min(requested, available).
    """
    import torch
    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError(
            "no CUDA GPUs visible; this script requires at least 1 GPU"
        )
    if requested is None:
        return available
    return max(1, min(int(requested), available))


# Common locations for a populated HF cache, in priority order.  Used by
# ``_resolve_hf_home`` when the env var ``HF_HOME`` is unset.  We prefer
# ``/workspace/.cache/huggingface`` first because that's what
# ``run_pipeline.sh`` uses on RunPod -- it's NFS-backed (big, persistent
# across pod recreates) -- and falls back to the OS default last because
# on RunPod that lives on a 5 GB container overlay disk that fills up
# the moment HuggingFace tries to download a 60+ GB model into it.
DEFAULT_HF_HOME_FALLBACKS = (
    "/workspace/.cache/huggingface",
    "/workspace/hf-cache",
)


def _resolve_hf_home(model_name: str) -> Optional[str]:
    """Pick an HF cache directory that already contains ``model_name``.

    Returns whichever of ``$HF_HOME`` or ``DEFAULT_HF_HOME_FALLBACKS``
    has the model's ``hub/models--*`` subtree, or None if no candidate
    matches.  Caller decides whether to fail loudly or let HF download.
    """
    model_subdir = "models--" + model_name.replace("/", "--")
    candidates = []
    if os.environ.get("HF_HOME"):
        candidates.append(os.environ["HF_HOME"])
    candidates.extend(DEFAULT_HF_HOME_FALLBACKS)

    for candidate in candidates:
        if (Path(candidate) / "hub" / model_subdir).is_dir():
            return candidate
    return None


def _run_judges_only(config: Dict[str, Any], args) -> None:
    """Delegate to post_judge.py without re-implementing the loop.

    Imported lazily because post_judge.py is under steering/ which isn't
    a package; we re-use the path-insert that the rest of run_sweep
    already does.
    """
    # post_judge lives next to this file; the path-insert at the top of
    # run_sweep.py already covers parents[1] (the repo root) for
    # assistant_axis imports, but the sibling steering/ dir is not on
    # sys.path -- we resolve post_judge's main() via importlib.
    import importlib.util as _iu
    pj_path = Path(__file__).resolve().parent / "post_judge.py"
    spec = _iu.spec_from_file_location("_post_judge_inproc", pj_path)
    pj = _iu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(pj)

    # Build the post_judge argv.  We reach into config to find the
    # experiment dir; the rest is forwarded from our own args.
    output_root = Path(config["output_dir"]) / config["experiment_id"]
    pj_argv = [
        "--experiment_dir", str(output_root),
        "--instructions_dir", str(args.instructions_dir),
        "--judges", args.judges,
        "--effect-mode", args.effect_mode,
        "--target-batch-size", str(args.target_batch_size),
        "--skip-if-strength-mean-coh-above",
        str(args.skip_if_strength_mean_coh_above),
        "--coh-stop-threshold", str(args.coh_stop_threshold),
    ]
    if args.coherence_model:
        pj_argv += ["--coherence-model", args.coherence_model]
    if args.rp_model:
        pj_argv += ["--rp-model", args.rp_model]
    if args.effect_models:
        pj_argv += ["--effect-models", args.effect_models]
    if args.rerun_coherence_with_model:
        pj_argv += ["--rerun-coherence-with-model",
                    args.rerun_coherence_with_model]
    if args.rerun:
        pj_argv += ["--rerun"]

    # Drive post_judge.main() by transient sys.argv replacement; avoids
    # having to refactor it to take an argv list.
    saved_argv = sys.argv
    sys.argv = ["steering/post_judge.py"] + pj_argv
    try:
        pj.main()
    finally:
        sys.argv = saved_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, action="append", default=[],
                        help="YAML or JSON experiment config file. May be "
                             "repeated to queue multiple experiments in one "
                             "process invocation (saves cold-import + model-"
                             "load overhead per extra config).  See also "
                             "--config-list.  Required unless --config-list "
                             "is given.")
    parser.add_argument("--config-list", type=Path, default=None,
                        help="File containing one config path per line "
                             "(comments starting with # are ignored).  Equivalent "
                             "to repeating --config for each entry; mixable with "
                             "explicit --config args.  All listed configs are "
                             "queued sequentially on the same warmed worker(s).")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all visible)")
    parser.add_argument("--instructions_dir", type=Path, default=Path("data"),
                        help="Root of data/{roles,traits}/instructions/<name>.json "
                             "(default: data)")
    # Phase-2 judging knobs.  These are NOT consumed during generation
    # (the runner uses NoOpJudgeDispatcher unless explicitly told
    # otherwise), but get forwarded to post_judge.py via --judges-only.
    # When set, the live sweep runs with NoOp coherence and async
    # judging fills in afterward.  The "live coherence + early-term"
    # mode for the sweep is configured here too.
    parser.add_argument("--judges-only", action="store_true",
                        help="Skip generation; run async RP+effect (and "
                             "optionally coherence rerun) on an existing "
                             "experiment dir.  Equivalent to invoking "
                             "steering/post_judge.py with the same flags. "
                             "Useful immediately after a fresh sweep.")
    parser.add_argument("--judges", default="persona,effect",
                        help="Comma list, default 'persona,effect'.  Used "
                             "by --judges-only.")
    parser.add_argument("--effect-mode", default="bidirectional",
                        choices=["bidirectional", "separate_poles", "both"])
    parser.add_argument("--target-batch-size", type=int, default=7,
                        help="Effect-judge target batch size (canonical B=7, "
                             "matching the May 2026 project-wide B=7 default "
                             "for response-mode judging).  The previous "
                             "value of 10 is OBSOLETE and was the source of "
                             "a several-hundred-dollar B=10 mis-judging "
                             "incident; do not revert without explicit "
                             "user approval.")
    parser.add_argument("--skip-if-strength-mean-coh-above", type=float,
                        default=1.5,
                        help="Skip RP/effect on a strength group whose mean "
                             "coherence is at or above this value (default 1.5, "
                             "matching coh_stop_threshold so judging and "
                             "stop-counting are mutually exclusive).  Display-"
                             "time filtering at a stricter threshold is done "
                             "via the plot's --coh-filter-threshold.")
    parser.add_argument("--coh-stop-threshold", type=float, default=1.5,
                        help="Stop sweep early once K consecutive strengths "
                             "have mean coherence at or above this value "
                             "(default 1.5).")
    parser.add_argument("--coh-stop-consecutive", type=int, default=2,
                        help="Number of consecutive strengths above the "
                             "coherence threshold required to trigger early "
                             "stop (default 2 = 'crossing + one confirmation'; "
                             "1 = stop on first crossing; 0 = never stop).")
    parser.add_argument("--coherence-model", default=None,
                        help="Coherence model (default gpt-4.1-mini).")
    parser.add_argument("--rp-model", default=None,
                        help="RP/persona judge model (default gpt-4.1-mini).")
    parser.add_argument("--effect-models", default=None,
                        help="Comma list of effect-judge ensemble models "
                             "(default 'gpt-4.1-mini,claude-haiku-4-5-20251001').")
    parser.add_argument("--rerun-coherence-with-model", default=None,
                        help="(--judges-only only) Rerun coherence with "
                             "the given model and store under "
                             "judges.coherence_alts[<model>].  For the "
                             "coherence-model shootout.")
    parser.add_argument("--rerun", action="store_true",
                        help="(--judges-only only) Force re-judge even if "
                             "the field is already populated.")
    parser.add_argument("--no-live-judging", action="store_true",
                        help="Skip live coherence + async RP/effect "
                             "judging during the sweep (workers use "
                             "NoOpJudgeDispatcher).  Sweep runs to its "
                             "configured max strength regardless of "
                             "coherence; use post_judge.py afterward.")
    parser.add_argument("--tmpdir", default=None,
                        help="Override TMPDIR for atomic-write staging "
                             "(default: /dev/shm).  Set explicitly to "
                             "preserve a pre-existing $TMPDIR.  Avoid "
                             "NFS-backed paths -- staging writes are "
                             "on the hot path for every records.jsonl "
                             "and summary.json flush.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Collect config paths from --config (repeatable) + --config-list.
    config_paths: List[Path] = list(args.config)
    if args.config_list is not None:
        for line in args.config_list.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            config_paths.append(Path(line))
    if not config_paths:
        raise SystemExit(
            "no configs specified -- use --config <path> (repeatable) "
            "and/or --config-list <file>"
        )

    if args.judges_only:
        if len(config_paths) != 1:
            raise SystemExit(
                "--judges-only currently supports exactly one --config "
                f"(got {len(config_paths)})"
            )
        config = _load_config(config_paths[0])
        _run_judges_only(config, args)
        return

    # Load + validate all configs up-front so we fail fast on a malformed
    # one rather than mid-queue after the first run has already started.
    loaded: List[Tuple[Path, Dict[str, Any]]] = []
    for p in config_paths:
        cfg = _load_config(p)
        for k in ("experiment_id", "model_name", "output_dir", "axis_source",
                  "persona", "cells", "questions_file"):
            if k not in cfg:
                raise SystemExit(
                    f"{p}: config missing required key: {k!r}"
                )
        loaded.append((p, cfg))

    # All configs must share model_name (a single worker can only host
    # one model in GPU memory).  Mixed-model queues require multiple
    # invocations.
    model_names = sorted({cfg["model_name"] for _, cfg in loaded})
    if len(model_names) > 1:
        raise SystemExit(
            f"all queued configs must share model_name; got {model_names}.  "
            f"Run them in separate invocations to avoid model-reload churn."
        )
    shared_model_name = model_names[0]

    logger.info(
        f"queueing {len(loaded)} experiment(s) on model {shared_model_name}"
    )

    # Per-config setup: freeze inputs to each experiment's output_root,
    # build judging_cfg + work_items, accumulate the master work-item
    # list.  Each work item carries its own sweep_log_dir so the worker
    # can swap its FileHandler when transitioning between experiments.
    all_work_items: List[Dict[str, Any]] = []
    for cfg_path, config in loaded:
        output_root = Path(config["output_dir"]) / config["experiment_id"]
        output_root.mkdir(parents=True, exist_ok=True)

        # Parent-side per-experiment FileHandler swap so parent log
        # lines from this config's setup land in the right sweep.log.
        if all_work_items:
            # Detach the previous experiment's handler before attaching
            # this one's.  Workers do the same per-item; parent does it
            # per-config setup.
            prev_root = Path(all_work_items[-1].get("sweep_log_dir", ""))
            if prev_root != output_root:
                _detach_sweep_log_filehandler(prev_root)
        _attach_sweep_log_filehandler(output_root)
        logger.info(f"sweep log -> {output_root / 'sweep.log'}")

        questions = _load_questions(Path(config["questions_file"]))
        persona_prompt = _build_persona_system_prompt(
            config["persona"], instructions_dir=args.instructions_dir,
        )
        atomic_write_text(json.dumps(config, indent=2) + "\n",
                          output_root / "config.json")
        atomic_write_text(json.dumps(questions, indent=2) + "\n",
                          output_root / "questions.json")
        atomic_write_text(persona_prompt + "\n",
                          output_root / "persona_system_prompt.txt")
        logger.info(f"experiment {config['experiment_id']} -> {output_root}")
        logger.info(f"persona system prompt: {persona_prompt!r}")
        logger.info(
            f"{len(questions)} questions loaded from {config['questions_file']}"
        )

        judging_cfg = _build_judging_config(
            config=config, args=args, instructions_dir=args.instructions_dir,
        )
        if judging_cfg.get("enabled"):
            logger.info(
                f"live judging enabled: coherence={judging_cfg['coherence_model']} "
                f"rp={judging_cfg['rp_model']} "
                f"effect={','.join(judging_cfg['effect_models'])} "
                f"mode={judging_cfg['effect_mode']} "
                f"target_batch_size={judging_cfg['target_batch_size']} "
                f"skip_threshold={judging_cfg['skip_threshold']} "
                f"coh_stop_threshold={judging_cfg['coh_stop_threshold']}"
            )
        else:
            logger.info(f"live judging disabled "
                        f"({judging_cfg.get('reason', 'no config')})")

        per_cfg_items = _build_work_items(
            config, output_root, persona_prompt, questions,
            judging=judging_cfg,
        )
        # Tag each work item with the experiment's sweep_log_dir so the
        # worker (which runs in a separate process and doesn't share the
        # parent's handler state) can swap its own FileHandler when
        # transitioning between experiments in the queue.
        for it in per_cfg_items:
            it["sweep_log_dir"] = str(output_root)
        n_cells_cfg = sum(1 for it in per_cfg_items if it["kind"] == "cell")
        logger.info(
            f"  built {len(per_cfg_items)} work items "
            f"(1 baselines + {n_cells_cfg} cells) for {config['experiment_id']}"
        )
        all_work_items.extend(per_cfg_items)

    # Sort so all baselines come before any cell.  With N_gpus workers
    # draining a single shared queue, this maximises baseline throughput
    # (all baselines start in parallel) and ensures cells only start
    # popping once every config's baselines is at least in-flight.  The
    # worker-side .complete-sentinel wait covers the remaining race
    # (a cell popped while its own config's baselines is still running
    # on another worker -- the cell-worker briefly idles).  Stable sort
    # so within-config order is preserved.
    all_work_items.sort(key=lambda it: 0 if it["kind"] == "baselines" else 1)

    n_total_cells = sum(1 for it in all_work_items if it["kind"] == "cell")
    n_total_baselines = sum(
        1 for it in all_work_items if it["kind"] == "baselines"
    )
    logger.info(
        f"total queue: {len(all_work_items)} work items "
        f"({n_total_baselines} baselines + {n_total_cells} cells) "
        f"across {len(loaded)} experiment(s); "
        f"baselines first so multi-GPU workers parallelize them"
    )

    # Resolve HF_HOME before spawning workers (model_name is shared
    # across all configs per the validation above).
    resolved_hf_home = _resolve_hf_home(shared_model_name)
    if resolved_hf_home is None:
        raise SystemExit(
            f"HF_HOME unset and {shared_model_name} not found in any of "
            f"{(os.environ.get('HF_HOME'),) + DEFAULT_HF_HOME_FALLBACKS}.  "
            f"Pre-populate the cache (e.g. via the pipeline) or set HF_HOME "
            f"explicitly before launching."
        )
    if os.environ.get("HF_HOME") != resolved_hf_home:
        logger.info(
            f"HF_HOME -> {resolved_hf_home} "
            f"(was {os.environ.get('HF_HOME')!r})"
        )
        os.environ["HF_HOME"] = resolved_hf_home

    # Mirror the model into /dev/shm ONCE in the parent before
    # spawning workers.  Before May 2026 each worker independently
    # called setup_model_tmpfs_cache, producing N parallel rsync
    # processes that contended for NFS bandwidth and (on RunPod's
    # MooseFS-backed /workspace) routinely deadlocked at 0 MB/s
    # collective progress -- workers wedged in D-state mmap faults
    # for hours.  Serializing the mirror in the parent reduces the
    # contention to a single rsync that completes in ~20 min on a
    # cold cache, then workers fast-path past it via the
    # ``.tmpfs_mirror_complete`` marker.
    #
    # Best-effort: if the parent-side mirror fails we don't bail --
    # workers retain their own setup_model_tmpfs_cache call as a
    # second-chance fallback, and if that also fails they load from
    # NFS directly (with the expected multi-hour mmap stall warned
    # about by setup_model_tmpfs_cache itself).
    from assistant_axis.tmpfs import (
        DEFAULT_TMPDIR, setup_model_tmpfs_cache, setup_tmpdir,
    )
    tmpdir_target = args.tmpdir if args.tmpdir is not None else DEFAULT_TMPDIR
    setup_tmpdir(tmpdir_target)
    logger.info(
        f"[tmpfs] pre-spawn: mirroring {shared_model_name} into /dev/shm "
        f"(serialized in parent so workers don't contend) ..."
    )
    t_mirror_start = time.time()
    parent_hf_home = setup_model_tmpfs_cache(model_name=shared_model_name)
    if parent_hf_home is not None:
        os.environ["HF_HOME"] = parent_hf_home
        logger.info(
            f"[tmpfs] pre-spawn mirror complete in "
            f"{time.time() - t_mirror_start:.1f}s; "
            f"HF_HOME={parent_hf_home}.  Workers will fast-path "
            f"via the mirror-complete marker."
        )
    else:
        logger.warning(
            f"[tmpfs] pre-spawn mirror returned None after "
            f"{time.time() - t_mirror_start:.1f}s; workers will "
            f"individually retry and may load from NFS directly "
            f"(expect multi-hour stalls if /workspace is slow)."
        )

    n_gpus = _detect_gpu_count(args.gpus)
    logger.info(f"using {n_gpus} GPU worker(s)")

    # Construct a minimal worker config dict.  Worker only needs
    # model_name (for model load) and a default sweep_log_dir for
    # any pre-first-item log lines.  Per-item sweep_log_dir takes
    # over once items start flowing.  ``tmpdir`` is threaded through
    # so the worker calls setup_tmpdir() with the same target the
    # parent did (matters because spawn-context workers start with a
    # fresh interpreter that doesn't share env mutations the parent
    # already made -- though TMPDIR-as-env-var IS inherited, the
    # worker still calls setup_tmpdir() to apply the same override
    # semantics in case the parent's env got reverted somehow).
    worker_config: Dict[str, Any] = {
        "model_name": shared_model_name,
        "sweep_log_dir": all_work_items[0].get("sweep_log_dir")
        if all_work_items else None,
        "tmpdir": tmpdir_target,
    }

    # Spawn workers.  CUDA requires the 'spawn' start method.
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for item in all_work_items:
        queue.put(item)
    for _ in range(n_gpus):
        queue.put(None)  # sentinel per worker

    procs = []
    for worker_id in range(n_gpus):
        gpu_id = worker_id  # 1:1 mapping when CUDA_VISIBLE_DEVICES is preserved
        p = ctx.Process(
            target=_worker_main,
            args=(worker_id, gpu_id, queue, worker_config),
        )
        p.start()
        procs.append(p)

    rc = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            logger.error(f"worker pid={p.pid} exited with code {p.exitcode}")
            rc = 1

    logger.info("all workers finished")
    sys.exit(rc)


if __name__ == "__main__":
    main()
