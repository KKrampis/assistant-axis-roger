#!/usr/bin/env python3
"""Compute PC directions, save per-entity projections, and launch
``axis_judge_correlation.py`` for every (PC × {glossary, inline} ×
{openai, anthropic}) cell.

This is the **expensive** stage of the PC round-trip pipeline: each cell
makes a few hundred judge calls.  It assumes ``spec.json`` already exists
in each cell directory (produced upstream by
``infer_axis_description.py`` + ``standardize_axis_spec.py`` -- which
itself burns one Claude Opus call per cell with a 10K thinking budget).

Pipeline overview
-----------------

Each (PC, style) cell ends up with a directory under ``--sweep_dir``::

    pcNNN_{glossary,inline}/
        axis_postshear.pt          (PC direction in post-shear space)
        post_shear_projection.json (entity projections onto that PC)
        prompt.txt + response.txt + thinking.txt + spec.json
            ^-- produced by infer_axis_description.py (NOT this script)
        gpt/                       (axis_judge_correlation.py output)
            scores_descriptions.json
            scores_instructions.json
            correlations.json
            ...
        sonnet/
            (same shape as gpt/)

Smoke testing
-------------

For a dry run that validates the planning logic without making any API
calls or running the heavy SVD setup, use ``--dry_run``: the script
will print exactly which (PC, style, provider) cells it *would* run and
the full command line for each, then exit.

To exercise a single cell end-to-end (e.g. before kicking off all 28),
restrict the PC and style:

::

    uv run python results_analysis/pc_round_trip/launch_judge_runs.py \\
        --pcs 8 --styles inline --providers anthropic --max_parallel 1
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from assistant_axis import entity_id
from results_analysis.axis_judge_correlation import _load_vector_file
from results_analysis.canonical_angles.data import (
    DEFAULT_DATA_DIR, build_goal_nogoal_subspaces,
)
from results_analysis.canonical_angles.whitening import fit_shear

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# May 2026 migration: pc_round_trip canonical = 8-slot, slot 7, layer 25,
# raw (L=0, K=0).  See AGENT_NOTES "PC round-trip migration to 8-slot,
# slot 7, raw" for context.
DEFAULT_SWEEP_DIR = "roger/pc_axis_describer_sweep"
DEFAULT_SLOT = 7
DEFAULT_LAYER = 25
DEFAULT_SHEAR_L = 0       # raw canonical, no soft-shear
DEFAULT_WHITEN_K = 0      # raw canonical, no soft-K whitening
DEFAULT_PCS: list[int] = [
    1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56,
    64, 80, 96, 128, 192, 256, 384, 512,
]
DEFAULT_STYLES: list[str] = ["glossary", "inline"]
DEFAULT_PROVIDERS: list[str] = ["openai", "anthropic"]

PROVIDER_MODEL = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-6",
}
PROVIDER_SUBDIR = {
    "openai": "gpt",
    "anthropic": "sonnet",
}


# ---------------------------------------------------------------------------
# PC direction setup
# ---------------------------------------------------------------------------

def setup_pcs(*, data_dir: Path, sweep_dir: Path, slot: int, layer: int,
              pcs: list[int], styles: list[str],
              shear_L: int = DEFAULT_SHEAR_L) -> dict[int, np.ndarray]:
    """Compute PC directions in post-shear space at (slot, layer); save them
    into each pcNNN_{style}/axis_postshear.pt cell file."""
    default_v = _load_vector_file(
        data_dir / "traits" / "vectors" / "default.pt"
    ).float().numpy()[slot, layer]

    rows = []
    for et in ("roles", "traits"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            v = _load_vector_file(fp).float().numpy()[slot, layer]
            rows.append(v - default_v)
    M_raw = np.stack(rows, axis=0).astype(np.float32)

    A_g, A_n = build_goal_nogoal_subspaces(data_dir, slot=slot, layer=layer,
                                            kind="combined")
    shear = fit_shear(A_g, A_n, L=shear_L)
    M_sheared = shear.apply(M_raw)

    centered = M_sheared - M_sheared.mean(axis=0, keepdims=True)
    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
    print(f"PCA: Vt shape {Vt.shape} (post-shear, slot={slot}, layer={layer}, "
          f"shear_L={shear_L})")

    pc_dirs: dict[int, np.ndarray] = {}
    for pc in pcs:
        if pc - 1 >= Vt.shape[0]:
            print(f"  PC {pc} out of range")
            continue
        d = Vt[pc - 1].astype(np.float32)
        pc_dirs[pc] = d
        for style in styles:
            cell = sweep_dir / f"pc{pc:03d}_{style}"
            cell.mkdir(parents=True, exist_ok=True)
            torch.save(torch.from_numpy(d), cell / "axis_postshear.pt")
        print(f"  PC{pc:>3d}: ||d||={np.linalg.norm(d):.3e}, saved")
    return pc_dirs


def projection_at_post_shear(*, data_dir: Path, slot: int, layer: int,
                              pc_dir: np.ndarray,
                              shear_L: int = DEFAULT_SHEAR_L) -> dict[str, float]:
    """Compute post-shear projections of all entities onto pc_dir; returns
    {entity_name: projection}."""
    default_v = _load_vector_file(
        data_dir / "traits" / "vectors" / "default.pt"
    ).float().numpy()[slot, layer]
    A_g, A_n = build_goal_nogoal_subspaces(data_dir, slot=slot, layer=layer,
                                            kind="combined")
    shear = fit_shear(A_g, A_n, L=shear_L)

    norm = np.linalg.norm(pc_dir)
    projections: dict[str, float] = {}
    for et in ("roles", "traits"):
        for fp in sorted((data_dir / et / "vectors").glob("*.pt")):
            if fp.stem == "default":
                continue
            v = _load_vector_file(fp).float().numpy()[slot, layer] - default_v
            v_sheared = shear.apply(v[None, :])[0]
            projections[entity_id(fp.stem, et)] = float(
                np.dot(v_sheared, pc_dir) / norm
            )
    return projections


# ---------------------------------------------------------------------------
# Per-cell command building
# ---------------------------------------------------------------------------

def build_command(*, data_dir: Path, sweep_dir: Path, pc: int, style: str,
                   provider: str, layer: int, spec: dict,
                   batch_size: int, rps: float, save_every: int,
                   max_tokens: int, temperature: float,
                   whiten_K: int) -> list[str]:
    cell = sweep_dir / f"pc{pc:03d}_{style}"
    out = cell / PROVIDER_SUBDIR[provider]
    pos_pole = spec["pos_pole_standardized"]
    neg_pole = spec["neg_pole_standardized"]
    pos_examples = ",".join(spec.get("pos_examples", []))
    neg_examples = ",".join(spec.get("neg_examples", []))
    axis_name = spec.get("axis_name", f"PC{pc} {style}")

    return [
        "uv", "run", "python",
        "results_analysis/axis_judge_correlation.py",
        "--axis_file", str(cell / "axis_postshear.pt"),
        "--axis_name", axis_name,
        "--pos_pole", pos_pole,
        "--neg_pole", neg_pole,
        "--pos_examples", pos_examples,
        "--neg_examples", neg_examples,
        "--data_dir", str(data_dir),
        "--instructions_dir", "data",
        "--layer", str(layer), "--slot", "all",
        "--whiten_K", str(whiten_K),
        "--provider", provider,
        "--judge_model", PROVIDER_MODEL[provider],
        "--score_descriptions", "--score_instructions",
        "--batch_size", str(batch_size),
        "--rps", str(rps),
        "--save_every", str(save_every),
        "--max_tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--output_dir", str(out),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                   help="Base data directory (must contain roles/vectors and "
                        "traits/vectors with the entity .pt files).  "
                        "Default = the project's 8-slot DEFAULT_DATA_DIR.")
    p.add_argument("--sweep_dir", type=str, default=DEFAULT_SWEEP_DIR,
                   help="Directory containing pcNNN_<style>/ cells (and where "
                        "this script writes axis_postshear.pt and "
                        "post_shear_projection.json).")
    p.add_argument("--slot", type=int, default=DEFAULT_SLOT,
                   help=f"Token slot for the PC subspace (default {DEFAULT_SLOT}).")
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                   help=f"Transformer layer for the PC subspace (default {DEFAULT_LAYER}).")
    p.add_argument("--shear_L", type=int, default=DEFAULT_SHEAR_L,
                   help=f"Soft-shear truncation depth for the canonical PCA "
                        f"(default {DEFAULT_SHEAR_L} = raw, no shear).")
    p.add_argument("--pcs", type=str,
                   default=",".join(str(pc) for pc in DEFAULT_PCS),
                   help=f"Comma-separated PC indices. Default: {DEFAULT_PCS}")
    p.add_argument("--styles", type=str,
                   default=",".join(DEFAULT_STYLES),
                   help=f"Comma-separated styles. Default: {DEFAULT_STYLES}")
    p.add_argument("--providers", type=str,
                   default=",".join(DEFAULT_PROVIDERS),
                   help=f"Comma-separated providers. Default: {DEFAULT_PROVIDERS}")
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--rps", type=float, default=10.0)
    p.add_argument("--save_every", type=int, default=40)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--whiten_K", type=int, default=DEFAULT_WHITEN_K,
                   help=f"K for the inner axis_judge_correlation projection "
                        f"geometry (default {DEFAULT_WHITEN_K} = raw, "
                        "matches the canonical raw setup).  Independent of "
                        "the K swept in the round-trip experiment.")
    p.add_argument("--max_parallel", type=int, default=2,
                   help="Concurrent subprocess cap (default 2 to keep the ML "
                        "footprint comfortable on a 48 GB box; bump up on a "
                        "cleaner machine).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the planned commands and exit. Skips the PC "
                        "setup so this is also useful as a quick smoke "
                        "check of the planning logic itself.")
    p.add_argument("--skip_setup", action="store_true",
                   help="Skip the PC-direction setup + projection-save steps "
                        "(useful when the cell files already exist and you "
                        "just want to re-launch the judge calls).")
    p.add_argument("--setup_only", action="store_true",
                   help="Run setup (axis_postshear.pt + post_shear_projection."
                        "json) and stop before the judge calls.  Used as the "
                        "Phase 3 step of the migration pipeline.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    sweep_dir = Path(args.sweep_dir)
    pcs = [int(x) for x in args.pcs.split(",")]
    styles = [s.strip() for s in args.styles.split(",")]
    providers = [p.strip() for p in args.providers.split(",")]
    for prov in providers:
        if prov not in PROVIDER_MODEL:
            raise SystemExit(f"Unknown provider: {prov!r} "
                             f"(supported: {list(PROVIDER_MODEL)})")

    # Step 1 + 2: compute PC directions and save projections.
    if not args.skip_setup and not args.dry_run:
        print("=== Step 1: Compute and save PC directions (post-shear) ===")
        pc_dirs = setup_pcs(
            data_dir=data_dir, sweep_dir=sweep_dir,
            slot=args.slot, layer=args.layer,
            pcs=pcs, styles=styles, shear_L=args.shear_L,
        )

        print("\n=== Step 2: Save original post-shear projections per PC ===")
        for pc, d in pc_dirs.items():
            proj = projection_at_post_shear(
                data_dir=data_dir, slot=args.slot, layer=args.layer,
                pc_dir=d, shear_L=args.shear_L,
            )
            for style in styles:
                cell = sweep_dir / f"pc{pc:03d}_{style}"
                (cell / "post_shear_projection.json").write_text(
                    json.dumps(proj, indent=2)
                )
            print(f"  PC{pc:>3d}: {len(proj)} entity projections saved")

    if args.setup_only:
        print("\n--setup_only: stopping after axis_postshear.pt + "
              "post_shear_projection.json are written; skipping "
              "describer / judges.")
        return 0

    # Step 3: build commands.
    print("\n=== Step 3: Build axis_judge_correlation commands ===")
    commands: list[tuple[int, str, str, list[str]]] = []
    for pc in pcs:
        for style in styles:
            cell = sweep_dir / f"pc{pc:03d}_{style}"
            spec_path = cell / "spec.json"
            if not spec_path.exists():
                print(f"  SKIP {cell}: no spec.json (run "
                      f"infer_axis_description.py + standardize_axis_spec.py first)")
                continue
            spec = json.loads(spec_path.read_text())
            if not isinstance(spec.get("pos_pole_standardized"), str):
                print(f"  SKIP {cell}: spec.json lacks pos_pole_standardized "
                      f"(run standardize_axis_spec.py)")
                continue
            for provider in providers:
                out = cell / PROVIDER_SUBDIR[provider]
                if (out / "correlations.json").exists():
                    print(f"  ALREADY DONE: {out}")
                    continue
                commands.append((pc, style, provider, build_command(
                    data_dir=data_dir, sweep_dir=sweep_dir,
                    pc=pc, style=style, provider=provider,
                    layer=args.layer, spec=spec,
                    batch_size=args.batch_size, rps=args.rps,
                    save_every=args.save_every, max_tokens=args.max_tokens,
                    temperature=args.temperature, whiten_K=args.whiten_K,
                )))

    print(f"\n{len(commands)} cells to run.")
    if args.dry_run:
        for pc, style, provider, cmd in commands:
            print(f"\n--- PC{pc:03d}/{style}/{provider} ---")
            print("  " + " ".join(shlex.quote(t) for t in cmd))
        print("\n(dry run -- no commands executed)")
        return 0
    if not commands:
        print("Nothing to do.")
        return 0

    # Step 4: run with bounded concurrency.
    import concurrent.futures
    print(f"\n=== Step 4: Run {len(commands)} cells, up to {args.max_parallel} "
          f"in parallel ===")

    def run_one(item):
        pc, style, provider, cmd = item
        cell_label = f"PC{pc:03d}/{style}/{provider}"
        out = sweep_dir / f"pc{pc:03d}_{style}" / PROVIDER_SUBDIR[provider]
        out.mkdir(parents=True, exist_ok=True)
        log = out / "run.log"
        t0 = time.time()
        with open(log, "wb") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                 check=False).returncode
        return cell_label, rc, time.time() - t0

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_parallel
    ) as pool:
        futs = [pool.submit(run_one, c) for c in commands]
        for f in concurrent.futures.as_completed(futs):
            label, rc, dt = f.result()
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(f"  [{dt:6.1f}s] {label}: {status}")

    print(f"\nDone in {(time.time() - started) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
