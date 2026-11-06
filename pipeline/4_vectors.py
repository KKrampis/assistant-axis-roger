#!/usr/bin/env python3
"""
Compute per-role vectors from activations and scores.

For regular roles: computes the mean of activations where score=3 (fully playing role)
For default role: computes the mean of ALL activations (no score filtering)

Usage:
    uv run scripts/4_vectors.py \
        --activations_dir outputs/gemma-2-27b/activations \
        --scores_dir outputs/gemma-2-27b/scores \
        --output_dir outputs/gemma-2-27b/vectors \
        --min_count 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.atomic_io import (  # noqa: E402
    read_text_with_retry,
    torch_load_with_retry,
    torch_save_with_retry,
)

logger = logging.getLogger("pipeline.4_vectors")


def load_scores(scores_file: Path) -> dict:
    """Load scores from JSON file (with project-standard NFS retry)."""
    text = read_text_with_retry(scores_file, logger_obj=logger)
    return json.loads(text)


def load_activations(activations_file: Path) -> tuple[dict, dict]:
    """Load activations from .pt file, separating metadata.

    Retries on transient I/O errors using the project standard
    (5 attempts, [5, 20, 60, 180] s backoff -- see
    :data:`assistant_axis.atomic_io.DEFAULT_RETRY_DELAYS_S`).
    MooseFS / NFS can return truncated reads under chunkserver load;
    torch.load surfaces these as ``RuntimeError`` ("unexpected EOF,
    expected N more bytes" or "storage has wrong byte size of dtype")
    -- the consolidated retry helper catches both flavours.

    Returns:
        (activations_dict, metadata) where metadata is empty for old-format files.
    """
    data = torch_load_with_retry(
        activations_file, map_location="cpu", weights_only=False,
        logger_obj=logger,
    )
    metadata = data.pop("metadata", {})
    return data, metadata


def _question_index(key: str) -> int:
    """Extract q_idx from an activation key like 'pos_p3_q7'.

    NOTE: this is the 0-based index into the already-reduced question list
    from step 1, not into the original questions file. See the TODO in
    assistant_axis/generation.py load_questions for why.
    """
    return int(key.rsplit("_q", 1)[1])


def _keep_by_reduce(key: str, reduce_questions: int) -> bool:
    """Return True if this activation's question index survives an Nth-only reduction."""
    if reduce_questions <= 1:
        return True
    return _question_index(key) % reduce_questions == 0


def compute_pos_3_vector(activations: dict, scores: dict, min_count: int,
                         reduce_questions: int = 1) -> torch.Tensor:
    """
    Compute mean vector from activations where score=3.

    Handles both old 2D tensors (n_layers, hidden_dim) and new 3D tensors
    (1+N, n_layers, hidden_dim). Output shape matches input tensor shape.

    With reduce_questions=N (default 1 = no further reduction), takes only
    activations whose question index satisfies q_idx % N == 0.  This is
    applied on top of any reduction already done at step 1, so the effective
    reduction compounds (reduce=3 at step 1 + reduce=3 here = 9x overall).

    NOTE: Each entity is filtered independently — the set of score=3 questions
    for trait A may differ substantially from trait B.  When computing trait
    *directions* (A_vec - B_vec) for antonym pairs, this means the two means
    are averaged over different question distributions, so the difference
    captures question-mix effects on top of the actual trait contrast.

    A cleaner approach for paired directions would be question-matched
    filtering: intersect the score=3 sets for both sides, recompute means
    over only the shared questions, then difference.  This would require a
    step 4b that loads both activation files for a pair, intersects their
    score=3 keys, and saves a matched direction vector.  The activation files
    are too large for ad-hoc analysis scripts to reopen, so this must be
    precomputed in the pipeline.
    """
    filtered_acts = []
    for key, act in activations.items():
        if key in scores and scores[key] == 3 and _keep_by_reduce(key, reduce_questions):
            filtered_acts.append(act)

    if len(filtered_acts) < min_count:
        raise ValueError(f"Only {len(filtered_acts)} score=3 samples, need {min_count}")

    stacked = torch.stack(filtered_acts)
    return stacked.nanmean(dim=0)


def compute_mean_vector(activations: dict, reduce_questions: int = 1) -> torch.Tensor:
    """
    Compute mean vector from all activations (no score filtering).

    With reduce_questions=N, takes only activations whose question index
    satisfies q_idx % N == 0. See compute_pos_3_vector for caveats.

    Handles both old 2D tensors (n_layers, hidden_dim) and new 3D tensors
    (1+N, n_layers, hidden_dim). Output shape matches input tensor shape.
    """
    all_acts = [act for key, act in activations.items()
                if _keep_by_reduce(key, reduce_questions)]
    stacked = torch.stack(all_acts)
    return stacked.nanmean(dim=0)


def main():
    # Ensure pipeline log messages (including io_retry's WARNING/ERROR
    # output) are visible.  Format mirrors axis_judge_correlation.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Compute per-role vectors")
    parser.add_argument("--activations_dir", type=str, required=True, help="Directory with activation .pt files")
    parser.add_argument("--scores_dir", type=str, required=True, help="Directory with score JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for vector .pt files")
    parser.add_argument("--min_count", type=int, default=50, help="Minimum score=3 samples required")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--reduce_questions", type=int, default=1,
                        help="Take only activations with q_idx %% N == 0 (default 1 = no reduction). "
                             "Compounds with any reduction already done at step 1.")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activations_dir = Path(args.activations_dir)
    scores_dir = Path(args.scores_dir)

    # Get all activation files
    activation_files = sorted(activations_dir.glob("*.pt"))
    print(f"Found {len(activation_files)} activation files")

    successful = 0
    skipped = 0
    failed = 0

    for act_file in tqdm(activation_files, desc="Computing vectors"):
        role = act_file.stem
        output_file = output_dir / f"{role}.pt"

        # Skip if exists (unless --overwrite)
        if output_file.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            activations, act_metadata = load_activations(act_file)
        except (RuntimeError, OSError, EOFError) as e:
            # io_retry already logged an ERROR with full retry history.
            logger.error("%s: skipping due to unrecoverable load failure (%s)", role, e)
            failed += 1
            continue

        if not activations:
            logger.error("%s: activation file loaded but contains no entries; skipping", role)
            failed += 1
            continue

        try:
            if "default" in role:
                # Default roles: use all activations (no score filtering)
                vector = compute_mean_vector(activations, args.reduce_questions)
                vector_type = "mean"
            else:
                # Regular roles: filter by score=3
                scores_file = scores_dir / f"{role}.json"
                if not scores_file.exists():
                    logger.warning("%s: no scores file at %s; skipping", role, scores_file)
                    failed += 1
                    continue

                scores = load_scores(scores_file)
                vector = compute_pos_3_vector(activations, scores, args.min_count,
                                              args.reduce_questions)
                vector_type = "pos_3"

            save_data = {
                "vector": vector,
                "type": vector_type,
                "role": role,
            }
            if act_metadata:
                save_data["metadata"] = act_metadata

            # Atomic save with project-standard retry: TMPDIR staging
            # + retry/backoff copy to (possibly NFS) destination + atomic
            # rename + post-copy size-sanity check.  A mid-write crash or
            # silent short-write never leaves a half-written .pt at the
            # final path.
            try:
                torch_save_with_retry(save_data, output_file, logger_obj=logger)
            except (RuntimeError, OSError, EOFError) as e:
                logger.error("%s: skipping due to unrecoverable save failure (%s)", role, e)
                failed += 1
                continue
            successful += 1

        except ValueError as e:
            # min_count not met -- legitimate skip, not an error.
            logger.warning("%s: %s", role, e)
            failed += 1

    print(f"\nSummary: {successful} successful, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
