"""Canonical angles analysis tool.

Public API surface:

- Spec dataclasses (frozen, hashable): :class:`CASpec`, :class:`AggSpec`,
  :class:`OriginSpec`, :class:`WhiteningSpec`.
- Compute entry points: :func:`compute_ca`, :func:`compute_ca_grid`.
- Compute environment / cache: :class:`ComputeEnv`.
- Subspace builders in :mod:`.data` (e.g. :func:`build_subspace`,
  :func:`build_whitening_pool`).
- Whitening basis utilities in :mod:`.whitening` (:func:`fit_whitening`,
  :func:`fit_shear`, :func:`parse_whitening_spec`).
- Goal/no-goal subspace builder for soft-shear in :mod:`.data`
  (:func:`build_goal_nogoal_subspaces`).
- Plot helpers in :mod:`.plot_helpers` (:func:`plot_per_slot_panels`,
  :func:`plot_overlay_curves`, :func:`plot_grid`).

The typical wrapper flow is:

    from results_analysis.canonical_angles import (
        CASpec, AggSpec, OriginSpec, WhiteningSpec,
        compute_ca_grid, ComputeEnv,
    )
    from results_analysis.canonical_angles.data import (
        build_subspace, build_whitening_pool, detect_n_slots,
    )
    from results_analysis.canonical_angles.plot_helpers import plot_per_slot_panels

    specs = [CASpec(...) for ... in ...]
    results = compute_ca_grid(specs, data_dir=...)
    fig = plot_per_slot_panels(results, ...)
    fig.savefig(...)

See :mod:`.plots.goal_vs_nogoal_mutual` for a worked example.
"""

from .core import (
    AggSpec,
    CASpec,
    ComputeEnv,
    OriginSpec,
    WhiteningSpec,
    canonical_angles_deg,
    compute_ca,
    compute_ca_grid,
)

__all__ = [
    "AggSpec",
    "CASpec",
    "ComputeEnv",
    "OriginSpec",
    "WhiteningSpec",
    "canonical_angles_deg",
    "compute_ca",
    "compute_ca_grid",
]
