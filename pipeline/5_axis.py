#!/usr/bin/env python3
"""
Compute the assistant axis from per-role vectors.

Formula: axis = mean(default_vectors) - mean(pos_3_vectors across roles)

The axis points FROM role-playing TOWARD default assistant behavior.

Usage:
    uv run scripts/5_axis.py \
        --vectors_dir outputs/gemma-2-27b/vectors \
        --output outputs/gemma-2-27b/axis.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.atomic_io import (  # noqa: E402
    torch_load_with_retry,
    torch_save_with_retry,
)

logger = logging.getLogger("pipeline.5_axis")


def load_vector(vector_file: Path) -> dict:
    """Load vector data from .pt file with project-standard NFS retry
    (5 attempts, [5, 20, 60, 180] s backoff -- handles transient
    short-reads that surface as ``RuntimeError``).
    """
    return torch_load_with_retry(
        vector_file, map_location="cpu", weights_only=False,
        logger_obj=logger,
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Compute assistant axis from vectors")
    parser.add_argument("--vectors_dir", type=str, required=True, help="Directory with vector .pt files")
    parser.add_argument("--output", type=str, required=True, help="Output axis.pt file path")
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    output_path = Path(args.output)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all vectors
    vector_files = sorted(vectors_dir.glob("*.pt"))
    print(f"Found {len(vector_files)} vector files")

    # Separate default and role vectors
    default_vectors = []
    role_vectors = []
    metadata = {}

    for vec_file in tqdm(vector_files, desc="Loading vectors"):
        data = load_vector(vec_file)
        vector = data["vector"]
        vector_type = data.get("type", "unknown")
        role = data.get("role", vec_file.stem)

        if "metadata" in data:
            file_meta = data["metadata"]
            if not metadata:
                metadata = file_meta
            else:
                if file_meta.get("header_ids") != metadata.get("header_ids"):
                    print(f"Error: metadata mismatch in {vec_file.name}: "
                          f"header_ids {file_meta.get('header_ids')} != {metadata.get('header_ids')}")
                    sys.exit(1)
                if file_meta.get("model_name") != metadata.get("model_name"):
                    print(f"Error: metadata mismatch in {vec_file.name}: "
                          f"model_name {file_meta.get('model_name')!r} != {metadata.get('model_name')!r}")
                    sys.exit(1)

        if "default" in role or vector_type == "mean":
            default_vectors.append(vector)
            print(f"  {role}: default/mean vector")
        else:
            role_vectors.append(vector)

    print(f"\nLoaded {len(default_vectors)} default vectors, {len(role_vectors)} role vectors")

    if not default_vectors:
        print("Error: No default vectors found")
        sys.exit(1)

    if not role_vectors:
        print("Error: No role vectors found")
        sys.exit(1)

    # Verify all vectors have the same shape
    all_vectors = default_vectors + role_vectors
    ref_shape = all_vectors[0].shape
    mismatched = [v for v in all_vectors if v.shape != ref_shape]
    if mismatched:
        shapes = set(str(v.shape) for v in all_vectors)
        print(f"Error: mixed vector shapes: {shapes}")
        sys.exit(1)

    # Compute means -- works for both 2D (n_layers, hidden) and 3D (1+N, n_layers, hidden)
    default_stacked = torch.stack(default_vectors)
    role_stacked = torch.stack(role_vectors)

    default_mean = default_stacked.nanmean(dim=0)
    role_mean = role_stacked.nanmean(dim=0)

    # Compute axis: points from role-playing toward default
    axis = default_mean - role_mean

    print(f"\nAxis shape: {axis.shape}")

    # Per-layer norms; use dim=-1 to handle both 2D and 3D
    norms = axis.norm(dim=-1)
    if axis.ndim == 3:
        n_slots = axis.shape[0]
        print(f"Produced {n_slots} axes (slot 0 = body mean, slots 1..{n_slots-1} = header tokens)")
        # Print norms for body-mean axis (slot 0)
        body_norms = norms[0]
        print("Body-mean axis norms per layer (first 10):")
        for i, norm in enumerate(body_norms[:10]):
            print(f"  Layer {i}: {norm:.4f}")
        print("  ...")
        print(f"  Mean norm: {body_norms.mean():.4f}")
        print(f"  Max norm: {body_norms.max():.4f} (layer {body_norms.argmax().item()})")
    else:
        print("Axis norms per layer (first 10):")
        for i, norm in enumerate(norms[:10]):
            print(f"  Layer {i}: {norm:.4f}")
        print("  ...")
        print(f"  Mean norm: {norms.mean():.4f}")
        print(f"  Max norm: {norms.max():.4f} (layer {norms.argmax().item()})")

    # Save axis -- dict format for metadata, backward compat via load_axis().
    # torch_save_with_retry stages to TMPDIR, copies with retry/backoff,
    # atomically renames into place, and verifies post-copy size.  A
    # mid-write crash or silent NFS short-write never leaves a
    # half-written axis.pt at the destination.
    save_data = {"axis": axis}
    if metadata:
        save_data["metadata"] = metadata
    torch_save_with_retry(save_data, output_path, logger_obj=logger)
    print(f"\nSaved axis to {output_path}")


if __name__ == "__main__":
    main()
