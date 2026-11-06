"""
Assistant axis computation and projection utilities.

The assistant axis is a direction in activation space that captures the difference
between role-playing and default assistant behavior in language models.

Formula (as pipeline/5_axis.py computes it):
    axis = mean(default vectors) - nanmean(entity vectors)

Where:
    - the default vector is the mean of every default-prompt activation
      (pipeline/4_vectors.py, type "mean");
    - each entity vector is the mean of that role's / trait's /
      combination's score=3 activations (type "pos_3"), so the second
      term is a mean of per-entity means: every entity weighs the same,
      however many score=3 responses it kept.

The axis points FROM role-playing TOWARD default assistant behavior.

Tensor layout (the convention every vector and axis file follows):
    - legacy, no header slots: (n_layers, hidden_dim)
    - since the header-token extraction: (n_slots, n_layers, hidden_dim),
      slot 0 = mean over the response's tokens ("body-mean"), slots
      1..N = the assistant header tokens listed in
      metadata["header_tokens"], in that order (see :func:`slot_labels`;
      the 8-slot Qwen files have 7: <|im_start|>, assistant, \\n,
      <think>, \\n\\n, </think>, \\n\\n).  A header slot is NaN where the
      header was not found, which is why the pipeline averages with
      nanmean.
    - layer index i = the output of decoder block i (a forward hook on
      ``model.layers[i]``), 0-based, no embedding layer -- provided step 2
      ran with its default ``--layers all``.  The files do not record the
      layer list, so a step 2 run with a ``--layers`` subset would shift
      every index silently.

The helpers below that take a ``layer`` index (``project``,
``project_batch``, ``cosine_similarity_per_layer``,
``axis_norm_per_layer``, ``compute_axis``) expect the 2-D layout: pass
``axis[slot]`` for a slotted file.

Example:
    from assistant_axis import load_axis, project

    axis = load_axis("outputs/gemma-2-27b/axis.pt")
    projection = project(activation, axis, layer=22)
"""

import torch
import numpy as np
from typing import List, Optional, Sequence, Union


def compute_axis(
    role_activations: torch.Tensor,
    default_activations: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the assistant axis from role and default activations.

    Formula: axis = mean(default) - mean(role)

    The axis points FROM role-playing TOWARD assistant behavior.

    Args:
        role_activations: Tensor of shape (n_role, n_layers, hidden_dim)
                         Activations from role-playing responses (score=3)
        default_activations: Tensor of shape (n_default, n_layers, hidden_dim)
                            Activations from default/neutral system prompts

    Returns:
        Axis tensor of shape (n_layers, hidden_dim)
    """
    # Compute means
    role_mean = role_activations.mean(dim=0)       # (n_layers, hidden_dim)
    default_mean = default_activations.mean(dim=0)  # (n_layers, hidden_dim)

    # axis points from role toward default
    axis = default_mean - role_mean

    return axis


def project(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> float:
    """
    Project activations onto the axis at a specific layer.

    Args:
        activations: Tensor of shape (n_layers, hidden_dim) or (hidden_dim,)
        axis: Tensor of shape (n_layers, hidden_dim); for a slotted
              (n_slots, n_layers, hidden_dim) axis pass ``axis[slot]``
              (indexing the 3-D tensor with ``layer`` would pick a slot)
        layer: Layer index to use for projection
        normalize: Whether to normalize the axis before projection

    Returns:
        Projection value (scalar). Higher values indicate more "assistant-like",
        lower values indicate more "role-playing".
    """
    # Get layer activations
    if activations.ndim == 2:
        act = activations[layer].float()
    else:
        act = activations.float()

    ax = axis[layer].float()

    if normalize:
        ax = ax / (ax.norm() + 1e-8)

    return float(act @ ax)


def project_batch(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Project a batch of activations onto the axis.

    Args:
        activations: Tensor of shape (batch, n_layers, hidden_dim)
        axis: Tensor of shape (n_layers, hidden_dim)
        layer: Layer index to use
        normalize: Whether to normalize the axis

    Returns:
        Projection values of shape (batch,)
    """
    # Get layer activations for all samples
    acts = activations[:, layer, :].float()  # (batch, hidden_dim)
    ax = axis[layer].float()  # (hidden_dim,)

    if normalize:
        ax = ax / (ax.norm() + 1e-8)

    return acts @ ax  # (batch,)


def cosine_similarity_per_layer(
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> np.ndarray:
    """
    Compute cosine similarity between two vectors at each layer.

    Args:
        v1: Tensor of shape (n_layers, hidden_dim)
        v2: Tensor of shape (n_layers, hidden_dim)

    Returns:
        Array of cosine similarities, one per layer
    """
    v1 = v1.float()
    v2 = v2.float()

    # Normalize both vectors
    v1_norm = v1 / (v1.norm(dim=1, keepdim=True) + 1e-8)
    v2_norm = v2 / (v2.norm(dim=1, keepdim=True) + 1e-8)

    # Compute dot product per layer
    similarities = (v1_norm * v2_norm).sum(dim=1)

    return similarities.numpy()


def axis_norm_per_layer(axis: torch.Tensor) -> np.ndarray:
    """
    Compute the L2 norm of the axis at each layer.

    Args:
        axis: Tensor of shape (n_layers, hidden_dim)

    Returns:
        Array of norms, one per layer
    """
    return axis.float().norm(dim=1).numpy()


def save_axis(
    axis: torch.Tensor,
    path: str,
    metadata: Optional[dict] = None,
):
    """
    Save axis to a .pt file.

    Written through :func:`assistant_axis.atomic_io.torch_save_with_retry`
    (TMPDIR staging, verified copy, atomic rename), like every other axis
    and vector write; a plain ``torch.save`` here could leave a truncated
    file on /workspace.

    Args:
        axis: Axis tensor of shape (n_layers, hidden_dim) or
              (n_slots, n_layers, hidden_dim)
        path: Path to save to
        metadata: Optional metadata dict to include
    """
    from .atomic_io import torch_save_with_retry
    save_dict = {"axis": axis}
    if metadata:
        save_dict["metadata"] = metadata
    torch_save_with_retry(save_dict, path)


def load_axis(path: str) -> torch.Tensor:
    """
    Load axis from a .pt file.

    Uses :func:`assistant_axis.atomic_io.torch_load_with_retry` so transient
    NFS hiccups on /workspace get the same 5-attempt backoff treatment as
    other reads in this codebase.

    Args:
        path: Path to load from

    Returns:
        Axis tensor of shape (n_layers, hidden_dim)
    """
    from .atomic_io import torch_load_with_retry
    data = torch_load_with_retry(path, map_location="cpu", weights_only=False)

    # Handle both formats: dict with 'axis' key or raw tensor
    if isinstance(data, dict):
        if "axis" in data:
            return data["axis"]
        else:
            raise ValueError("Expected 'axis' key in saved dict")
    else:
        return data


def load_axis_with_metadata(path: str) -> tuple[torch.Tensor, dict]:
    """
    Load axis and its metadata from a .pt file.

    Like load_axis() but also returns the metadata dict, which may contain
    header_tokens, header_ids, model_name, etc. Returns an empty dict for
    old-format files that have no metadata.  NFS-flake-retried via
    :func:`assistant_axis.atomic_io.torch_load_with_retry`.

    Args:
        path: Path to load from

    Returns:
        (axis_tensor, metadata_dict)
    """
    from .atomic_io import torch_load_with_retry
    data = torch_load_with_retry(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        if "axis" not in data:
            raise ValueError("Expected 'axis' key in saved dict")
        return data["axis"], data.get("metadata", {})
    else:
        return data, {}


def load_role_vector(path: str) -> tuple[torch.Tensor, dict]:
    """
    Load a role/trait vector from a .pt file.

    Handles both the old bare-tensor format and the new dict-wrapped format
    produced by pipeline step 4 ({"vector": tensor, "metadata": {...}, ...}).
    NFS-flake-retried via
    :func:`assistant_axis.atomic_io.torch_load_with_retry`.

    Args:
        path: Path to the .pt file

    Returns:
        (vector_tensor, metadata_dict)
    """
    from .atomic_io import torch_load_with_retry
    data = torch_load_with_retry(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        if "vector" in data:
            return data["vector"], data.get("metadata", {})
        raise ValueError(f"Expected 'vector' key in saved dict, got keys: {list(data.keys())}")
    else:
        return data, {}


def slot_labels(metadata: Optional[dict]) -> List[str]:
    """
    Return human-readable labels for each activation slot.

    Slot 0 is always the body-mean. Slots 1..N correspond to header tokens
    listed in metadata["header_tokens"].

    Args:
        metadata: Metadata dict from load_axis_with_metadata or load_role_vector.
                  May be None or empty.

    Returns:
        List of labels, e.g. ["body-mean", "<|im_start|>", "assistant", "\\n"]
    """
    if not metadata or "header_tokens" not in metadata:
        return ["body-mean"]
    return ["body-mean"] + list(metadata["header_tokens"])


class SlotLayoutError(ValueError):
    """A tree whose header slots are not the layout a script's slot numbers
    or slot names assume."""


SLOT_HINT = "pass --slot explicitly to use another layout"


def require_header_layout(
    metadata: Optional[dict],
    expected: Optional[Sequence[str]] = None,
    where: str = "",
    hint: str = SLOT_HINT,
) -> None:
    """Refuse a tree whose header tokens are not ``expected`` (default: the
    seven of Qwen3's non-thinking header, the 8slot layout,
    ``assistant_axis.model_families.QWEN3_NON_THINKING_HEADER_TOKENS``).

    For the scripts whose default slot (``DEFAULT_SLOT = 6``, the
    ``</think>`` position; ``7`` for the PC round trip) or whose slot names
    are Qwen3's: on a tree with another header (another model family, the
    multi-model plan of 2026-10-13) such a number points at another token,
    silently, or past the end.  A script calls this only when its default or
    its names are used, so an explicit ``--slot`` on another tree still works
    (and :func:`slot_labels` names the slot).  Raises
    :class:`SlotLayoutError` naming ``where``, the tree's header and the
    expected one, and ending with ``hint`` (a script whose slot names are
    used on every run, with no ``--slot``, passes its own)."""
    if expected is None:
        from .model_families import QWEN3_NON_THINKING_HEADER_TOKENS
        expected = QWEN3_NON_THINKING_HEADER_TOKENS
    got = list((metadata or {}).get("header_tokens") or [])
    if got != list(expected):
        raise SlotLayoutError(
            f"{where + ': ' if where else ''}this tree's header slots are {got!r} "
            f"({len(got)} tokens), not the {len(expected)} the script's slot numbers and names "
            f"assume ({list(expected)!r}, Qwen3's layout); {hint}")


def is_qwen3_header(metadata: Optional[dict]) -> bool:
    """Whether ``metadata["header_tokens"]`` are the seven of Qwen3's
    non-thinking header (the 8slot layout)."""
    from .model_families import QWEN3_NON_THINKING_HEADER_TOKENS
    return list((metadata or {}).get("header_tokens") or []) == list(
        QWEN3_NON_THINKING_HEADER_TOKENS)


TREE_VECTOR_SUBDIRS = ("traits/vectors", "roles/vectors", "combinations/vectors",
                       "default/vectors")


def tree_vector_metadata(data_dir,
                         subdirs: Sequence[str] = TREE_VECTOR_SUBDIRS) -> Optional[dict]:
    """The metadata of one vector of an extraction tree: the first non-empty
    ``*.pt`` (sorted) of the first of ``subdirs`` under ``data_dir`` that has
    one, or None when there is none (a zero-byte file is never a vector; it
    is skipped, not retried).  One file stands for the tree: step 5 refuses
    vectors whose header, model or family differ.  Loads one vector file
    (about 5-10 MB)."""
    from pathlib import Path
    root = Path(data_dir)
    for sub in subdirs:
        for f in sorted((root / sub).glob("*.pt")):
            if f.stat().st_size == 0:
                continue
            _, meta = load_role_vector(str(f))
            return dict(meta or {})
    return None


def require_tree_header_layout(
    data_dir,
    where: str = "",
    expected: Optional[Sequence[str]] = None,
    subdirs: Sequence[str] = TREE_VECTOR_SUBDIRS,
    hint: str = SLOT_HINT,
) -> dict:
    """:func:`require_header_layout` on the metadata of ``data_dir``'s
    vectors (:func:`tree_vector_metadata`); returns that metadata.  A tree
    with no vector to read is not checked (a WARNING; the script's own
    checks report a missing tree), and ``{}`` is returned."""
    meta = tree_vector_metadata(data_dir, subdirs)
    if meta is None:
        import logging
        logging.getLogger(__name__).warning(
            "%s: no vector under %s to read the header layout from; slot layout not checked",
            where or "require_tree_header_layout", data_dir)
        return {}
    require_header_layout(meta, expected, where, hint)
    return meta


def aggregate_role_vectors(
    vectors: dict,
    exclude_roles: Optional[list] = None,
) -> torch.Tensor:
    """
    Aggregate per-role vectors into a single mean vector.

    Args:
        vectors: Dict mapping role names to vectors (n_layers, hidden_dim)
        exclude_roles: List of role names to exclude (e.g., ["default"])

    Returns:
        Mean vector of shape (n_layers, hidden_dim)
    """
    exclude_roles = exclude_roles or []

    filtered = [v for name, v in vectors.items() if name not in exclude_roles]

    if not filtered:
        raise ValueError("No vectors remaining after exclusions")

    stacked = torch.stack(filtered)  # (n_roles, n_layers, hidden_dim)
    return stacked.mean(dim=0)  # (n_layers, hidden_dim)
