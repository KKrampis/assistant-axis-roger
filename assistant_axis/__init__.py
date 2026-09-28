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

from .models import get_config, MODEL_CONFIGS
from .axis import (
    compute_axis,
    load_axis,
    load_axis_with_metadata,
    load_role_vector,
    save_axis,
    slot_labels,
    project,
    project_batch,
    cosine_similarity_per_layer,
    axis_norm_per_layer,
    aggregate_role_vectors,
)
from .generation import (
    generate_response,
    format_conversation,
    VLLMGenerator,
    RoleResponseGenerator,
)
from .steering import (
    ActivationSteering,
    create_feature_ablation_steerer,
    create_multi_feature_steerer,
    create_mean_ablation_steerer,
    load_capping_config,
    build_capping_steerer,
)
from .pca import (
    compute_pca,
    plot_variance_explained,
    MeanScaler,
    L2MeanScaler,
)
from .plot_metadata import json_metadata, png_metadata, suptitle_with_specs
from .plot_palette import (
    slot_color,
    slot_colors,
    slot_colors_8,
    kind_color,
    kind_marker,
    kind_text_color,
    kind_text_style,
)
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
)
from .pair_list_cohort import cohort_from_pairs, pair_type_of
from .judge_batch import RESPONSE_BATCH_SIZE, response_subdir

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
    "cohort_from_pairs",
    "pair_type_of",
    # Response-judging batch-size convention
    "RESPONSE_BATCH_SIZE",
    "response_subdir",
]
