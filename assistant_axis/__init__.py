"""
Assistant Axis - Tools for computing and steering with the assistant axis.

The assistant axis is a direction in activation space that captures the difference
between role-playing and default assistant behavior in language models.

Example:
    from assistant_axis import get_config, load_axis, ActivationSteering
    from assistant_axis.internals import ProbingModel

    # Load model and axis
    pm = ProbingModel("google/gemma-2-27b-it")
    axis = load_axis("outputs/axis.pt")
    config = get_config("google/gemma-2-27b-it")

    # Steer model outputs
    with ActivationSteering(pm.model, steering_vectors=[axis[config["target_layer"]]],
                           coefficients=[1.0], layer_indices=[config["target_layer"]]):
        output = pm.model.generate(...)
"""

import importlib as _importlib

# Lazy loading (PEP 562; review P1 D5, walk-through P8.3, 2026-10-12).  Until
# then this file imported every submodule below, so ``import
# assistant_axis.entity_id`` (or any submodule) took ~1.3 s and ~2,100
# modules (torch, matplotlib.pyplot, sklearn, scipy, plotly), minutes on the
# pod's network volume; pipeline/3_judge.py loaded its modules through a
# stand-in package to avoid it.  Now each public name is imported from its
# submodule the first time it is looked up (``from assistant_axis import
# load_axis`` or ``assistant_axis.load_axis``), and the submodules this file
# used to import stay reachable as attributes (``assistant_axis.axis``).
# ``__all__`` and every public name are unchanged.
#
# The entity_id names are imported eagerly: entity_id.py uses only the
# standard library, and its function ``entity_id`` has the same name as its
# module.  Importing it here binds ``assistant_axis.entity_id`` to the
# function (as before); a lazy lookup would leave the submodule there after
# ``import assistant_axis.entity_id``, and ``from assistant_axis import
# entity_id`` would then give the module.
from .entity_id import (
    EntityId,
    KIND_R,
    KIND_T,
    KIND_LONG,
    entity_id,
    parse_entity_id,
    is_entity_id,
    display_label,
    kind_short,
    kind_long,
    normalize_to_file_name,
    display_form_name,
    corpus_display_name,
    judge_label,
    judge_negative_label,
    judge_form_of_label,
    judge_form_of_negative_label,
)

# public name -> the submodule it comes from (imported on first lookup)
_LAZY_NAMES = {
    **dict.fromkeys(("get_config", "MODEL_CONFIGS"), "models"),
    **dict.fromkeys((
        "compute_axis", "load_axis", "load_axis_with_metadata", "load_role_vector",
        "save_axis", "slot_labels", "project", "project_batch",
        "cosine_similarity_per_layer", "axis_norm_per_layer", "aggregate_role_vectors",
    ), "axis"),
    **dict.fromkeys((
        "generate_response", "format_conversation", "VLLMGenerator", "RoleResponseGenerator",
    ), "generation"),
    **dict.fromkeys((
        "ActivationSteering", "create_feature_ablation_steerer",
        "create_multi_feature_steerer", "create_mean_ablation_steerer",
        "load_capping_config", "build_capping_steerer",
    ), "steering"),
    **dict.fromkeys((
        "compute_pca", "plot_variance_explained", "MeanScaler", "L2MeanScaler",
    ), "pca"),
    **dict.fromkeys(("json_metadata", "png_metadata", "suptitle_with_specs"), "plot_metadata"),
    **dict.fromkeys((
        "slot_color", "slot_colors", "slot_colors_8", "kind_color", "kind_marker",
        "kind_text_color", "kind_text_style",
    ), "plot_palette"),
    **dict.fromkeys(("cohort_from_pairs", "pair_type_of"), "pair_list_cohort"),
    **dict.fromkeys(("RESPONSE_BATCH_SIZE", "response_subdir"), "judge_batch"),
}

# Submodules the eager __init__ imported, so ``assistant_axis.<name>`` worked
# after a bare ``import assistant_axis``; they are imported on first access.
_LAZY_SUBMODULES = frozenset({
    "models", "axis", "generation", "steering", "pca", "plot_metadata",
    "plot_palette", "pair_list_cohort", "judge_batch",
})


def __getattr__(name):
    module = _LAZY_NAMES.get(name)
    if module is not None:
        value = getattr(_importlib.import_module(f".{module}", __name__), name)
        globals()[name] = value      # later lookups skip __getattr__
        return value
    if name in _LAZY_SUBMODULES:
        # import_module binds the submodule as an attribute of this package
        return _importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_NAMES) | _LAZY_SUBMODULES)


__all__ = [
    # Models
    "get_config",
    "MODEL_CONFIGS",
    # Axis
    "compute_axis",
    "load_axis",
    "load_axis_with_metadata",
    "load_role_vector",
    "save_axis",
    "slot_labels",
    "project",
    "project_batch",
    "cosine_similarity_per_layer",
    "axis_norm_per_layer",
    "aggregate_role_vectors",
    # Generation
    "generate_response",
    "format_conversation",
    "VLLMGenerator",
    "RoleResponseGenerator",
    # Steering
    "ActivationSteering",
    "create_feature_ablation_steerer",
    "create_multi_feature_steerer",
    "create_mean_ablation_steerer",
    "load_capping_config",
    "build_capping_steerer",
    # PCA
    "compute_pca",
    "plot_variance_explained",
    "MeanScaler",
    "L2MeanScaler",
    # Plot + cache provenance helpers
    "json_metadata",
    "png_metadata",
    "suptitle_with_specs",
    "slot_color",
    "slot_colors",
    "slot_colors_8",
    "kind_color",
    "kind_marker",
    "kind_text_color",
    "kind_text_style",
    # Trait / role disambiguation
    "EntityId",
    "KIND_R",
    "KIND_T",
    "KIND_LONG",
    "entity_id",
    "parse_entity_id",
    "is_entity_id",
    "display_label",
    "kind_short",
    "kind_long",
    "normalize_to_file_name",
    "display_form_name",
    "corpus_display_name",
    "judge_label",
    "judge_negative_label",
    "judge_form_of_label",
    "judge_form_of_negative_label",
    "cohort_from_pairs",
    "pair_type_of",
    # Response-judging batch-size convention
    "RESPONSE_BATCH_SIZE",
    "response_subdir",
]
